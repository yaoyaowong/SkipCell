#include "storage/macos_aligned_file_reader.h"

#include "common/diskann_exception.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <unistd.h>

namespace powerlaw_ann {

macos_aligned_file_reader_t::~macos_aligned_file_reader_t() { close(); }

aligned_io_context_t& macos_aligned_file_reader_t::get_ctx() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  const auto thread_id = std::this_thread::get_id();
  const auto iter = ctx_map_.find(thread_id);
  if (iter == ctx_map_.end()) {
    throw diskann_exception_t("Current thread is not registered with the aligned reader", -1);
  }
  return ctx_map_.at(thread_id);
}

void macos_aligned_file_reader_t::register_thread() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  ctx_map_.try_emplace(std::this_thread::get_id(), aligned_io_context_t{});
}

void macos_aligned_file_reader_t::deregister_thread() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  ctx_map_.erase(std::this_thread::get_id());
}

void macos_aligned_file_reader_t::deregister_all_threads() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  ctx_map_.clear();
}

void macos_aligned_file_reader_t::open(const std::string& file_name) {
  close();
  file_descriptor_ = ::open(file_name.c_str(), O_RDONLY);
  if (file_descriptor_ < 0) {
    throw diskann_exception_t(
        "Failed to open disk index " + file_name + ": " + std::strerror(errno), -1);
  }
  file_name_ = file_name;
}

void macos_aligned_file_reader_t::open_overlay(const std::string& file_name,
                                               uint64_t virtual_offset) {
  if (file_descriptor_ < 0 || overlay_file_descriptor_ >= 0 || !IS_512_ALIGNED(virtual_offset)) {
    throw diskann_exception_t("Invalid aligned-reader overlay configuration", -1);
  }
  overlay_file_descriptor_ = ::open(file_name.c_str(), O_RDONLY);
  if (overlay_file_descriptor_ < 0) {
    throw diskann_exception_t("Failed to open overlay " + file_name + ": " + std::strerror(errno),
                              -1);
  }
  overlay_virtual_offset_ = virtual_offset;
  overlay_file_name_ = file_name;
}

void macos_aligned_file_reader_t::close() {
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

void macos_aligned_file_reader_t::read(std::vector<aligned_read_t>& read_requests,
                                       aligned_io_context_t&, bool) {
  if (file_descriptor_ < 0) {
    throw diskann_exception_t("Aligned reader has no open disk index", -1);
  }

  for (const auto& request : read_requests) {
    const bool use_overlay =
        overlay_file_descriptor_ >= 0 && request.offset >= overlay_virtual_offset_;
    if (!use_overlay && overlay_file_descriptor_ >= 0 &&
        request.len > overlay_virtual_offset_ - request.offset) {
      throw diskann_exception_t("Aligned read crosses the Base/overlay boundary", -1);
    }
    const int descriptor = use_overlay ? overlay_file_descriptor_ : file_descriptor_;
    const uint64_t request_offset =
        use_overlay ? request.offset - overlay_virtual_offset_ : request.offset;
    const auto& active_name = use_overlay ? overlay_file_name_ : file_name_;
    size_t bytes_read = 0;
    while (bytes_read < request.len) {
      const auto result = ::pread(descriptor, static_cast<char*>(request.buf) + bytes_read,
                                  request.len - bytes_read, request_offset + bytes_read);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result < 0) {
        throw diskann_exception_t(
            "Failed to read disk index " + active_name + ": " + std::strerror(errno), -1);
      }
      if (result == 0) {
        std::ostringstream message;
        message << "Unexpected end of disk index " << active_name << " at offset "
                << request_offset + bytes_read;
        throw diskann_exception_t(message.str(), -1);
      }
      bytes_read += static_cast<size_t>(result);
    }
  }
}

} // namespace powerlaw_ann
