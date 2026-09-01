#if defined(__linux__)

#include "storage/io_uring_aligned_file_reader.h"

#include "common/defaults.h"
#include "common/diskann_exception.h"
#include "common/logger.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <liburing.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace powerlaw_ann {
namespace {

struct uring_context_t {
  io_uring ring{};
  uint32_t pending_prefetches = 0;
  std::vector<uint64_t> ready_prefetch_tags;
};

inline constexpr uint64_t k_prefetch_tag_mask = uint64_t{1} << 63U;

uint64_t encode_prefetch_tag(uint64_t tag) {
  if ((tag & k_prefetch_tag_mask) != 0) {
    throw diskann_exception_t("io_uring prefetch tag exceeds 63 bits", -1);
  }
  return tag | k_prefetch_tag_mask;
}

bool is_prefetch_tag(uint64_t tag) noexcept { return (tag & k_prefetch_tag_mask) != 0; }

uint64_t decode_prefetch_tag(uint64_t tag) noexcept { return tag & ~k_prefetch_tag_mask; }

uintptr_t encode_context(uring_context_t* context) noexcept {
  return reinterpret_cast<uintptr_t>(context);
}

uring_context_t* decode_context(uintptr_t opaque) noexcept {
  return reinterpret_cast<uring_context_t*>(opaque);
}

std::string uring_error(const char* operation, int result) {
  std::ostringstream message;
  message << operation << " failed; returned " << result;
  if (result < 0) {
    message << ": " << std::strerror(-result);
  }
  return message.str();
}

} // namespace

io_uring_aligned_file_reader_t::io_uring_aligned_file_reader_t(uint32_t queue_depth)
    : queue_depth_(queue_depth) {
  if (queue_depth_ == 0 || queue_depth_ > 4096) {
    throw diskann_exception_t("io_uring queue depth must be within [1, 4096]", -1);
  }
}

io_uring_aligned_file_reader_t::~io_uring_aligned_file_reader_t() {
  deregister_all_threads();
  close();
}

aligned_io_context_t& io_uring_aligned_file_reader_t::get_ctx() {
  std::lock_guard lock(ctx_mutex_);
  const auto thread_id = std::this_thread::get_id();
  const auto context = ctx_map_.find(thread_id);
  if (context == ctx_map_.end()) {
    throw diskann_exception_t("Current thread is not registered with the io_uring reader", -1);
  }
  return ctx_map_.at(thread_id);
}

void io_uring_aligned_file_reader_t::register_thread() {
  std::lock_guard lock(ctx_mutex_);
  const auto thread_id = std::this_thread::get_id();
  if (ctx_map_.find(thread_id) != ctx_map_.end()) {
    return;
  }
  auto* context = new uring_context_t{};
  const int result = io_uring_queue_init(queue_depth_, &context->ring, 0);
  if (result != 0) {
    delete context;
    throw diskann_exception_t(uring_error("io_uring_queue_init()", result), -1);
  }
  ctx_map_.insert({thread_id, aligned_io_context_t{encode_context(context)}});
}

void io_uring_aligned_file_reader_t::deregister_thread() {
  uring_context_t* ring_context = nullptr;
  {
    std::lock_guard lock(ctx_mutex_);
    const auto thread_id = std::this_thread::get_id();
    const auto context = ctx_map_.find(thread_id);
    if (context == ctx_map_.end()) {
      throw diskann_exception_t("Current thread is not registered with the io_uring reader", -1);
    }
    ring_context = decode_context(context->second.opaque);
    ctx_map_.erase(thread_id);
  }
  if (ring_context->pending_prefetches != 0) {
    throw diskann_exception_t("Cannot deregister a ring with outstanding prefetches", -1);
  }
  io_uring_queue_exit(&ring_context->ring);
  delete ring_context;
}

void io_uring_aligned_file_reader_t::deregister_all_threads() {
  std::lock_guard lock(ctx_mutex_);
  for (const auto& context : ctx_map_) {
    auto* ring_context = decode_context(context.second.opaque);
    io_uring_queue_exit(&ring_context->ring);
    delete ring_context;
  }
  ctx_map_.clear();
}

void io_uring_aligned_file_reader_t::open(const std::string& file_name) {
  close();
  file_descriptor_ = ::open(file_name.c_str(), O_DIRECT | O_RDONLY | O_LARGEFILE);
  if (file_descriptor_ < 0) {
    throw diskann_exception_t("Failed to open io_uring direct-I/O artifact " + file_name + ": " +
                                  std::strerror(errno),
                              -1);
  }
  file_name_ = file_name;
  cout << "Opened io_uring direct-I/O file: " << file_name_ << std::endl;
}

void io_uring_aligned_file_reader_t::close() {
  if (file_descriptor_ >= 0) {
    ::close(file_descriptor_);
    file_descriptor_ = -1;
    file_name_.clear();
  }
}

void io_uring_aligned_file_reader_t::read(std::vector<aligned_read_t>& requests,
                                          aligned_io_context_t& context, bool async) {
  if (async) {
    throw diskann_exception_t("Demand-only io_uring reader does not expose asynchronous return",
                              -1);
  }
  if (file_descriptor_ < 0) {
    throw diskann_exception_t("io_uring reader has no open artifact", -1);
  }
  auto* ring_context = decode_context(context.opaque);
  auto* ring = &ring_context->ring;
  size_t batch_begin = 0;
  while (batch_begin < requests.size()) {
    while (ring_context->pending_prefetches >= queue_depth_) {
      io_uring_cqe* event = nullptr;
      int result = 0;
      do {
        result = io_uring_wait_cqe(ring, &event);
      } while (result == -EINTR);
      if (result != 0 || event == nullptr) {
        short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
        throw diskann_exception_t(uring_error("io_uring_wait_cqe(prefetch capacity)", result), -1);
      }
      const uint64_t tag = io_uring_cqe_get_data64(event);
      const int completion_result = event->res;
      io_uring_cqe_seen(ring, event);
      if (!is_prefetch_tag(tag) || completion_result != static_cast<int>(defaults::SECTOR_LEN)) {
        short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
        throw diskann_exception_t("io_uring capacity completion is invalid", -1);
      }
      --ring_context->pending_prefetches;
      ring_context->ready_prefetch_tags.push_back(decode_prefetch_tag(tag));
      completed_.fetch_add(1, std::memory_order_relaxed);
      prefetch_completed_.fetch_add(1, std::memory_order_relaxed);
    }
    const size_t available = queue_depth_ - ring_context->pending_prefetches;
    const size_t batch_size = std::min<size_t>(available, requests.size() - batch_begin);
    for (size_t operation = 0; operation < batch_size; ++operation) {
      io_uring_sqe* entry = io_uring_get_sqe(ring);
      if (entry == nullptr) {
        throw diskann_exception_t("io_uring submission queue exhausted before its configured depth",
                                  -1);
      }
      const auto& request = requests[batch_begin + operation];
      io_uring_prep_read(entry, file_descriptor_, request.buf, request.len, request.offset);
      io_uring_sqe_set_data64(entry, operation);
    }
    const int submitted = io_uring_submit(ring);
    if (submitted != static_cast<int>(batch_size)) {
      throw diskann_exception_t(uring_error("io_uring_submit()", submitted) + "; expected " +
                                    std::to_string(batch_size),
                                -1);
    }
    submitted_.fetch_add(batch_size, std::memory_order_relaxed);
    const auto wait_start = std::chrono::steady_clock::now();
    size_t demand_completions = 0;
    while (demand_completions < batch_size) {
      io_uring_cqe* event = nullptr;
      int result = 0;
      do {
        result = io_uring_wait_cqe(ring, &event);
      } while (result == -EINTR);
      if (result != 0 || event == nullptr) {
        short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
        throw diskann_exception_t(uring_error("io_uring_wait_cqe()", result), -1);
      }
      const uint64_t tag = io_uring_cqe_get_data64(event);
      const int completion_result = event->res;
      io_uring_cqe_seen(ring, event);
      if (is_prefetch_tag(tag)) {
        if (ring_context->pending_prefetches == 0 ||
            completion_result != static_cast<int>(defaults::SECTOR_LEN)) {
          short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
          throw diskann_exception_t("io_uring prefetch returned a short or failed completion", -1);
        }
        --ring_context->pending_prefetches;
        ring_context->ready_prefetch_tags.push_back(decode_prefetch_tag(tag));
        completed_.fetch_add(1, std::memory_order_relaxed);
        prefetch_completed_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      const uint64_t expected =
          tag < batch_size ? requests[batch_begin + tag].len : std::numeric_limits<uint64_t>::max();
      if (completion_result < 0 || static_cast<uint64_t>(completion_result) != expected) {
        short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
        throw diskann_exception_t("io_uring read returned a short or failed completion", -1);
      }
      completed_.fetch_add(1, std::memory_order_relaxed);
      ++demand_completions;
    }
    wait_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - wait_start)
                           .count(),
                       std::memory_order_relaxed);
    completion_batches_.fetch_add(1, std::memory_order_relaxed);
    batch_begin += batch_size;
  }
}

void io_uring_aligned_file_reader_t::submit_prefetch(const aligned_read_t& request,
                                                     aligned_io_context_t& context, uint64_t tag) {
  if (file_descriptor_ < 0) {
    throw diskann_exception_t("io_uring reader has no open artifact", -1);
  }
  auto* ring_context = decode_context(context.opaque);
  if (ring_context->pending_prefetches >= queue_depth_) {
    throw diskann_exception_t("io_uring prefetch queue reached its configured depth", -1);
  }
  io_uring_sqe* entry = io_uring_get_sqe(&ring_context->ring);
  if (entry == nullptr) {
    throw diskann_exception_t("io_uring prefetch submission queue is exhausted", -1);
  }
  io_uring_prep_read(entry, file_descriptor_, request.buf, request.len, request.offset);
  io_uring_sqe_set_data64(entry, encode_prefetch_tag(tag));
  const int submitted = io_uring_submit(&ring_context->ring);
  if (submitted != 1) {
    throw diskann_exception_t(uring_error("io_uring_submit(prefetch)", submitted), -1);
  }
  ++ring_context->pending_prefetches;
  submitted_.fetch_add(1, std::memory_order_relaxed);
  prefetch_submitted_.fetch_add(1, std::memory_order_relaxed);
}

bool io_uring_aligned_file_reader_t::has_prefetch_headroom(aligned_io_context_t& context,
                                                           uint32_t reserve_demand_slots) const {
  const auto* ring_context = decode_context(context.opaque);
  return reserve_demand_slots < queue_depth_ &&
         ring_context->pending_prefetches < queue_depth_ - reserve_demand_slots;
}

std::vector<uint64_t>
io_uring_aligned_file_reader_t::poll_prefetches(aligned_io_context_t& context) {
  auto* ring_context = decode_context(context.opaque);
  std::vector<uint64_t> tags = std::move(ring_context->ready_prefetch_tags);
  ring_context->ready_prefetch_tags.clear();
  while (ring_context->pending_prefetches != 0) {
    io_uring_cqe* event = nullptr;
    const int result = io_uring_peek_cqe(&ring_context->ring, &event);
    if (result == -EAGAIN || event == nullptr) {
      break;
    }
    if (result != 0 || event == nullptr) {
      short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
      throw diskann_exception_t(uring_error("io_uring_peek_cqe(prefetch)", result), -1);
    }
    const uint64_t tag = io_uring_cqe_get_data64(event);
    const int completion_result = event->res;
    io_uring_cqe_seen(&ring_context->ring, event);
    if (!is_prefetch_tag(tag) || completion_result != static_cast<int>(defaults::SECTOR_LEN)) {
      short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
      throw diskann_exception_t("io_uring prefetch returned a short or failed completion", -1);
    }
    --ring_context->pending_prefetches;
    tags.push_back(decode_prefetch_tag(tag));
    completed_.fetch_add(1, std::memory_order_relaxed);
    prefetch_completed_.fetch_add(1, std::memory_order_relaxed);
  }
  return tags;
}

std::vector<uint64_t>
io_uring_aligned_file_reader_t::complete_prefetches(aligned_io_context_t& context) {
  auto* ring_context = decode_context(context.opaque);
  std::vector<uint64_t> tags = std::move(ring_context->ready_prefetch_tags);
  ring_context->ready_prefetch_tags.clear();
  tags.reserve(tags.size() + ring_context->pending_prefetches);
  const auto wait_start = std::chrono::steady_clock::now();
  while (ring_context->pending_prefetches != 0) {
    io_uring_cqe* event = nullptr;
    int result = 0;
    do {
      result = io_uring_wait_cqe(&ring_context->ring, &event);
    } while (result == -EINTR);
    if (result != 0 || event == nullptr) {
      short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
      throw diskann_exception_t(uring_error("io_uring_wait_cqe(prefetch)", result), -1);
    }
    const uint64_t tag = io_uring_cqe_get_data64(event);
    const int completion_result = event->res;
    io_uring_cqe_seen(&ring_context->ring, event);
    if (!is_prefetch_tag(tag) || completion_result != static_cast<int>(defaults::SECTOR_LEN)) {
      short_or_error_completions_.fetch_add(1, std::memory_order_relaxed);
      throw diskann_exception_t("io_uring prefetch returned a short or failed completion", -1);
    }
    --ring_context->pending_prefetches;
    tags.push_back(decode_prefetch_tag(tag));
    completed_.fetch_add(1, std::memory_order_relaxed);
    prefetch_completed_.fetch_add(1, std::memory_order_relaxed);
  }
  if (!tags.empty()) {
    wait_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - wait_start)
                           .count(),
                       std::memory_order_relaxed);
    completion_batches_.fetch_add(1, std::memory_order_relaxed);
  }
  return tags;
}

void io_uring_aligned_file_reader_t::discard_prefetches(aligned_io_context_t& context) {
  const auto tags = complete_prefetches(context);
  prefetch_obsolete_.fetch_add(tags.size(), std::memory_order_relaxed);
}

io_uring_reader_stats_t io_uring_aligned_file_reader_t::stats() const noexcept {
  return {submitted_.load(std::memory_order_relaxed),
          completed_.load(std::memory_order_relaxed),
          completion_batches_.load(std::memory_order_relaxed),
          wait_ns_.load(std::memory_order_relaxed),
          short_or_error_completions_.load(std::memory_order_relaxed),
          prefetch_submitted_.load(std::memory_order_relaxed),
          prefetch_completed_.load(std::memory_order_relaxed),
          prefetch_obsolete_.load(std::memory_order_relaxed),
          queue_depth_};
}

} // namespace powerlaw_ann

#endif // defined(__linux__)
