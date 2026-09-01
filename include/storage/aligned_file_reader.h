#ifndef STORAGE_ALIGNED_FILE_READER
#define STORAGE_ALIGNED_FILE_READER

#include "common/utils.h"
#include "third/tsl/robin_map.h"

#include <cassert>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace powerlaw_ann {

inline constexpr uint32_t k_max_io_depth = 128;

struct aligned_io_context_t {
  uintptr_t opaque = 0;
};

/**
 * @brief Describes one sector-aligned positional read.
 *
 * The disk-search algorithm owns the destination buffer. Offset, length, and
 * buffer address are all 512-byte aligned so the same request is valid for the
 * macOS correctness reader and the later Linux direct-I/O reader.
 */
struct aligned_read_t {
  uint64_t offset = 0;
  uint64_t len = 0;
  void* buf = nullptr;

  aligned_read_t() = default;

  aligned_read_t(uint64_t read_offset, uint64_t read_len, void* read_buf)
      : offset(read_offset), len(read_len), buf(read_buf) {
    assert(IS_512_ALIGNED(offset));
    assert(IS_512_ALIGNED(len));
    assert(IS_512_ALIGNED(buf));
  }
};

/**
 * @brief Platform-neutral batch reader used by DiskANN disk search.
 *
 * Implementations register one context for each search worker. `read()` is a
 * blocking batch operation for the macOS backend. The Linux backend submits
 * the same batch through libaio and waits for all completions without changing
 * the search loop.
 */
class aligned_file_reader_t {
public:
  virtual ~aligned_file_reader_t() = default;

  virtual aligned_io_context_t& get_ctx() = 0;
  virtual void register_thread() = 0;
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;
  virtual void open(const std::string& file_name) = 0;
  /**
   * Add one process-local overlay whose byte zero is exposed at `virtual_offset`.
   * Implementations may reject this optional operation when the backend cannot safely route it.
   */
  virtual void open_overlay(const std::string&, uint64_t) {
    throw std::runtime_error("Aligned reader backend does not support an overlay file");
  }
  virtual void close() = 0;
  virtual void read(std::vector<aligned_read_t>& read_requests, aligned_io_context_t& ctx,
                    bool async = false) = 0;

protected:
  tsl::robin_map<std::thread::id, aligned_io_context_t> ctx_map_;
  std::mutex ctx_mutex_;
};

} // namespace powerlaw_ann

#endif // STORAGE_ALIGNED_FILE_READER
