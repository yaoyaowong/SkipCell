#ifndef STORAGE_MACOS_ALIGNED_FILE_READER
#define STORAGE_MACOS_ALIGNED_FILE_READER

#include "storage/aligned_file_reader.h"

#include <string>

namespace powerlaw_ann {

/**
 * @brief Checked synchronous `pread()` backend for macOS correctness runs.
 *
 * One instance owns one file descriptor. Positional reads allow concurrent
 * callers without sharing a mutable file offset. This backend intentionally
 * does not use direct I/O; Ubuntu performance parity belongs to Step 9.
 */
class macos_aligned_file_reader_t final : public aligned_file_reader_t {
public:
  macos_aligned_file_reader_t() = default;
  ~macos_aligned_file_reader_t() override;

  aligned_io_context_t& get_ctx() override;
  void register_thread() override;
  void deregister_thread() override;
  void deregister_all_threads() override;
  void open(const std::string& file_name) override;
  void open_overlay(const std::string& file_name, uint64_t virtual_offset) override;
  void close() override;
  void read(std::vector<aligned_read_t>& read_requests, aligned_io_context_t& ctx,
            bool async = false) override;

private:
  int file_descriptor_ = -1;
  int overlay_file_descriptor_ = -1;
  uint64_t overlay_virtual_offset_ = 0;
  std::string file_name_;
  std::string overlay_file_name_;
};

} // namespace powerlaw_ann

#endif // STORAGE_MACOS_ALIGNED_FILE_READER
