#ifndef STORAGE_IO_URING_ALIGNED_FILE_READER
#define STORAGE_IO_URING_ALIGNED_FILE_READER

#if defined(__linux__)

#include "storage/aligned_file_reader.h"

#include <atomic>
#include <string>
#include <vector>

namespace powerlaw_ann {

struct io_uring_reader_stats_t {
  uint64_t submitted = 0;
  uint64_t completed = 0;
  uint64_t completion_batches = 0;
  uint64_t wait_ns = 0;
  uint64_t short_or_error_completions = 0;
  uint64_t prefetch_submitted = 0;
  uint64_t prefetch_completed = 0;
  uint64_t prefetch_obsolete = 0;
  uint32_t queue_depth = 0;
};

/** Linux-only demand/prefetch reader with one io_uring per registered search thread. */
class io_uring_aligned_file_reader_t final : public aligned_file_reader_t {
public:
  explicit io_uring_aligned_file_reader_t(uint32_t queue_depth = k_max_io_depth);
  ~io_uring_aligned_file_reader_t() override;

  aligned_io_context_t& get_ctx() override;
  void register_thread() override;
  void deregister_thread() override;
  void deregister_all_threads() override;
  void open(const std::string& file_name) override;
  void close() override;
  void read(std::vector<aligned_read_t>& read_requests, aligned_io_context_t& ctx,
            bool async = false) override;

  /** Submit one tagged prefetch without waiting; the caller owns the buffer until completion. */
  void submit_prefetch(const aligned_read_t& request, aligned_io_context_t& ctx, uint64_t tag);

  /** True when a prefetch leaves the requested number of demand SQEs available. */
  bool has_prefetch_headroom(aligned_io_context_t& ctx, uint32_t reserve_demand_slots = 1) const;

  /** Return completed prefetch tags without waiting for an incomplete request. */
  std::vector<uint64_t> poll_prefetches(aligned_io_context_t& ctx);

  /** Wait for every prefetch currently outstanding on this thread's ring and return its tags. */
  std::vector<uint64_t> complete_prefetches(aligned_io_context_t& ctx);

  /** Harvest outstanding prefetches whose predictions became obsolete, without publishing them. */
  void discard_prefetches(aligned_io_context_t& ctx);

  io_uring_reader_stats_t stats() const noexcept;

private:
  uint32_t queue_depth_ = 0;
  int file_descriptor_ = -1;
  std::string file_name_;
  std::atomic<uint64_t> submitted_{0};
  std::atomic<uint64_t> completed_{0};
  std::atomic<uint64_t> completion_batches_{0};
  std::atomic<uint64_t> wait_ns_{0};
  std::atomic<uint64_t> short_or_error_completions_{0};
  std::atomic<uint64_t> prefetch_submitted_{0};
  std::atomic<uint64_t> prefetch_completed_{0};
  std::atomic<uint64_t> prefetch_obsolete_{0};
};

} // namespace powerlaw_ann

#endif // defined(__linux__)

#endif // STORAGE_IO_URING_ALIGNED_FILE_READER
