#ifndef STORAGE_LINUX_ALIGNED_FILE_READER
#define STORAGE_LINUX_ALIGNED_FILE_READER

#if defined(__linux__)

#if !defined(POWERLAWANN_USE_LIBAIO)
#error "linux_aligned_file_reader_t requires the Linux libaio build backend"
#endif

#include "storage/aligned_file_reader.h"

#include <libaio.h>
#include <string>
#include <vector>

namespace powerlaw_ann {

/** Caller-owned state for one outstanding libaio batch on a registered search thread. */
struct libaio_async_batch_t {
  std::vector<struct iocb> control_blocks;
  std::vector<struct iocb*> control_block_ptrs;
  std::vector<struct io_event> events;
  bool submitted = false;
};

/**
 * @brief Pinned DiskANN libaio and direct-I/O reader for Ubuntu performance runs.
 *
 * One libaio context is created for each registered search worker. `read()` submits
 * aligned positional reads in batches and blocks until every request in the batch
 * completes. The file is opened with `O_DIRECT`, so callers must preserve the
 * alignment contract declared by `aligned_read_t`.
 */
class linux_aligned_file_reader_t final : public aligned_file_reader_t {
public:
  linux_aligned_file_reader_t() = default;
  ~linux_aligned_file_reader_t() override;

  aligned_io_context_t& get_ctx() override;
  void register_thread() override;
  void deregister_thread() override;
  void deregister_all_threads() override;
  void open(const std::string& file_name) override;
  void open_overlay(const std::string& file_name, uint64_t virtual_offset) override;
  void close() override;
  void read(std::vector<aligned_read_t>& read_requests, aligned_io_context_t& ctx,
            bool async = false) override;

  /** Submit one O_DIRECT batch and return before completion; only one caller-owned batch may run. */
  void submit_async(std::vector<aligned_read_t>& requests, aligned_io_context_t& ctx,
                    libaio_async_batch_t& batch);

  /** Wait for and validate every operation previously submitted in `batch`. */
  void complete_async(aligned_io_context_t& ctx, libaio_async_batch_t& batch);

private:
  int file_descriptor_ = -1;
  int overlay_file_descriptor_ = -1;
  uint64_t overlay_virtual_offset_ = 0;
  std::string file_name_;
  std::string overlay_file_name_;
};

} // namespace powerlaw_ann

#endif // defined(__linux__)

#endif // STORAGE_LINUX_ALIGNED_FILE_READER
