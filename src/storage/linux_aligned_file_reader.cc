#if defined(__linux__)

#include "storage/linux_aligned_file_reader.h"

#include "common/diskann_exception.h"
#include "common/logger.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace powerlaw_ann {
namespace {

inline constexpr uint64_t k_max_events = 1024;

uintptr_t encode_io_context(io_context_t ctx) noexcept {
  static_assert(sizeof(io_context_t) <= sizeof(uintptr_t));
  return reinterpret_cast<uintptr_t>(ctx);
}

io_context_t decode_io_context(uintptr_t opaque) noexcept {
  return reinterpret_cast<io_context_t>(opaque);
}

std::string libaio_error(const char* operation, int64_t result) {
  std::ostringstream message;
  message << operation << " failed; returned " << result;
  if (result < 0) {
    message << ": " << std::strerror(static_cast<int>(-result));
  }
  return message.str();
}

void execute_io(io_context_t ctx, int file_descriptor, std::vector<aligned_read_t>& read_requests) {
#if !defined(NDEBUG)
  for (const auto& request : read_requests) {
    assert(IS_512_ALIGNED(request.len));
    assert(IS_512_ALIGNED(request.offset));
    assert(IS_512_ALIGNED(request.buf));
  }
#endif

  const uint64_t num_batches = DIV_ROUND_UP(read_requests.size(), k_max_events);
  for (uint64_t batch = 0; batch < num_batches; ++batch) {
    const uint64_t start = batch * k_max_events;
    const uint64_t num_operations = std::min<uint64_t>(read_requests.size() - start, k_max_events);
    std::vector<struct iocb> control_blocks(num_operations);
    std::vector<struct iocb*> control_block_ptrs(num_operations, nullptr);
    std::vector<struct io_event> events(num_operations);

    for (uint64_t operation = 0; operation < num_operations; ++operation) {
      const auto& request = read_requests[start + operation];
      io_prep_pread(control_blocks.data() + operation, file_descriptor, request.buf, request.len,
                    request.offset);
      control_block_ptrs[operation] = control_blocks.data() + operation;
    }

    const int64_t submitted =
        io_submit(ctx, static_cast<int64_t>(num_operations), control_block_ptrs.data());
    if (submitted != static_cast<int64_t>(num_operations)) {
      throw diskann_exception_t(libaio_error("io_submit()", submitted) + "; expected " +
                                    std::to_string(num_operations),
                                -1);
    }

    const int64_t completed =
        io_getevents(ctx, static_cast<int64_t>(num_operations),
                     static_cast<int64_t>(num_operations), events.data(), nullptr);
    if (completed != static_cast<int64_t>(num_operations)) {
      throw diskann_exception_t(libaio_error("io_getevents()", completed) + "; expected " +
                                    std::to_string(num_operations),
                                -1);
    }
  }
}

} // namespace

linux_aligned_file_reader_t::~linux_aligned_file_reader_t() { close(); }

aligned_io_context_t& linux_aligned_file_reader_t::get_ctx() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  const auto thread_id = std::this_thread::get_id();
  const auto iter = ctx_map_.find(thread_id);
  if (iter == ctx_map_.end()) {
    throw diskann_exception_t("Current thread is not registered with the libaio reader", -1);
  }
  return ctx_map_.at(thread_id);
}

void linux_aligned_file_reader_t::register_thread() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  const auto thread_id = std::this_thread::get_id();
  if (ctx_map_.find(thread_id) != ctx_map_.end()) {
    return;
  }

  io_context_t native_ctx = 0;
  const int result = io_setup(k_max_events, &native_ctx);
  if (result != 0) {
    std::string message = libaio_error("io_setup()", result);
    if (result == -EAGAIN) {
      message += "; consider increasing /proc/sys/fs/aio-max-nr";
    }
    throw diskann_exception_t(message, -1);
  }
  ctx_map_.insert({thread_id, aligned_io_context_t{encode_io_context(native_ctx)}});
}

void linux_aligned_file_reader_t::deregister_thread() {
  aligned_io_context_t ctx;
  {
    std::lock_guard<std::mutex> lock(ctx_mutex_);
    const auto thread_id = std::this_thread::get_id();
    auto iter = ctx_map_.find(thread_id);
    if (iter == ctx_map_.end()) {
      throw diskann_exception_t("Current thread is not registered with the libaio reader", -1);
    }
    ctx = iter.value();
    ctx_map_.erase(thread_id);
  }

  const int result = io_destroy(decode_io_context(ctx.opaque));
  if (result != 0) {
    throw diskann_exception_t(libaio_error("io_destroy()", result), -1);
  }
}

void linux_aligned_file_reader_t::deregister_all_threads() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  for (const auto& entry : ctx_map_) {
    const int result = io_destroy(decode_io_context(entry.second.opaque));
    if (result != 0) {
      cerr << libaio_error("io_destroy()", result) << std::endl;
    }
  }
  ctx_map_.clear();
}

void linux_aligned_file_reader_t::open(const std::string& file_name) {
  close();
  file_descriptor_ = ::open(file_name.c_str(), O_DIRECT | O_RDONLY | O_LARGEFILE);
  if (file_descriptor_ < 0) {
    throw diskann_exception_t(
        "Failed to open direct-I/O disk index " + file_name + ": " + std::strerror(errno), -1);
  }
  file_name_ = file_name;
  cout << "Opened direct-I/O file: " << file_name_ << std::endl;
}

void linux_aligned_file_reader_t::open_overlay(const std::string& file_name,
                                               uint64_t virtual_offset) {
  if (file_descriptor_ < 0 || overlay_file_descriptor_ >= 0 || !IS_512_ALIGNED(virtual_offset)) {
    throw diskann_exception_t("Invalid direct-I/O overlay configuration", -1);
  }
  overlay_file_descriptor_ = ::open(file_name.c_str(), O_DIRECT | O_RDONLY | O_LARGEFILE);
  if (overlay_file_descriptor_ < 0) {
    throw diskann_exception_t(
        "Failed to open direct-I/O overlay " + file_name + ": " + std::strerror(errno), -1);
  }
  overlay_virtual_offset_ = virtual_offset;
  overlay_file_name_ = file_name;
  cout << "Opened direct-I/O overlay: " << overlay_file_name_ << std::endl;
}

void linux_aligned_file_reader_t::close() {
  if (overlay_file_descriptor_ >= 0) {
    ::close(overlay_file_descriptor_);
    overlay_file_descriptor_ = -1;
    overlay_virtual_offset_ = 0;
    overlay_file_name_.clear();
  }
  if (file_descriptor_ >= 0) {
    ::close(file_descriptor_);
    file_descriptor_ = -1;
    file_name_.clear();
  }
}

void linux_aligned_file_reader_t::read(std::vector<aligned_read_t>& read_requests,
                                       aligned_io_context_t& ctx, bool async) {
  if (async) {
    cout << "Asynchronous return is unsupported by the pinned Linux reader; waiting for the "
            "batch."
         << std::endl;
  }
  if (file_descriptor_ < 0) {
    throw diskann_exception_t("Libaio reader has no open disk index", -1);
  }
  if (read_requests.empty()) {
    return;
  }
  if (overlay_file_descriptor_ < 0) {
    execute_io(decode_io_context(ctx.opaque), file_descriptor_, read_requests);
    return;
  }
  std::vector<aligned_read_t> base_requests;
  std::vector<aligned_read_t> overlay_requests;
  base_requests.reserve(read_requests.size());
  overlay_requests.reserve(read_requests.size());
  for (const auto& request : read_requests) {
    if (request.offset >= overlay_virtual_offset_) {
      overlay_requests.emplace_back(request.offset - overlay_virtual_offset_, request.len,
                                    request.buf);
    } else {
      if (request.len > overlay_virtual_offset_ - request.offset) {
        throw diskann_exception_t("Aligned read crosses the Base/overlay boundary", -1);
      }
      base_requests.push_back(request);
    }
  }
  execute_io(decode_io_context(ctx.opaque), file_descriptor_, base_requests);
  execute_io(decode_io_context(ctx.opaque), overlay_file_descriptor_, overlay_requests);
}

void linux_aligned_file_reader_t::submit_async(std::vector<aligned_read_t>& requests,
                                               aligned_io_context_t& ctx,
                                               libaio_async_batch_t& batch) {
  if (batch.submitted || file_descriptor_ < 0 || overlay_file_descriptor_ >= 0 ||
      requests.empty() || requests.size() > k_max_events) {
    throw diskann_exception_t("Invalid libaio asynchronous batch submission", -1);
  }
#if !defined(NDEBUG)
  for (const auto& request : requests) {
    assert(IS_512_ALIGNED(request.len));
    assert(IS_512_ALIGNED(request.offset));
    assert(IS_512_ALIGNED(request.buf));
  }
#endif
  batch.control_blocks.resize(requests.size());
  batch.control_block_ptrs.resize(requests.size());
  batch.events.resize(requests.size());
  for (size_t operation = 0; operation < requests.size(); ++operation) {
    const auto& request = requests[operation];
    io_prep_pread(batch.control_blocks.data() + operation, file_descriptor_, request.buf,
                  request.len, request.offset);
    batch.control_blocks[operation].data = reinterpret_cast<void*>(operation + 1U);
    batch.control_block_ptrs[operation] = batch.control_blocks.data() + operation;
  }
  const int64_t submitted = io_submit(decode_io_context(ctx.opaque), requests.size(),
                                      batch.control_block_ptrs.data());
  if (submitted != static_cast<int64_t>(requests.size())) {
    throw diskann_exception_t(libaio_error("io_submit(async)", submitted) + "; expected " +
                                  std::to_string(requests.size()),
                              -1);
  }
  batch.submitted = true;
}

void linux_aligned_file_reader_t::complete_async(aligned_io_context_t& ctx,
                                                 libaio_async_batch_t& batch) {
  if (!batch.submitted || batch.control_blocks.empty()) {
    throw diskann_exception_t("Invalid libaio asynchronous batch completion", -1);
  }
  const int64_t expected = static_cast<int64_t>(batch.control_blocks.size());
  const int64_t completed =
      io_getevents(decode_io_context(ctx.opaque), expected, expected, batch.events.data(), nullptr);
  if (completed != expected) {
    batch.submitted = false;
    throw diskann_exception_t(libaio_error("io_getevents(async)", completed) + "; expected " +
                                  std::to_string(expected),
                              -1);
  }
  for (const auto& event : batch.events) {
    const auto operation = reinterpret_cast<uintptr_t>(event.data);
    if (operation == 0 || operation > batch.control_blocks.size() || event.res2 != 0 ||
        event.res != static_cast<int64_t>(batch.control_blocks[operation - 1U].u.c.nbytes)) {
      batch.submitted = false;
      throw diskann_exception_t("libaio asynchronous refinement read was short or failed", -1);
    }
  }
  batch.submitted = false;
}

} // namespace powerlaw_ann

#endif // defined(__linux__)
