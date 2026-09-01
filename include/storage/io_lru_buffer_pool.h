#ifndef STORAGE_IO_LRU_BUFFER_POOL
#define STORAGE_IO_LRU_BUFFER_POOL

#include "storage/chunk_buf_pool.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace powerlaw_ann {

enum class io_cache_reservation_state_t : uint8_t {
  HIT,
  LOAD,
  BYPASS,
};

enum class io_cache_access_t : uint8_t {
  DEMAND,
  PREFETCH,
  GATEWAY_PREFETCH,
};

struct io_cache_reservation_t {
  io_cache_reservation_state_t state = io_cache_reservation_state_t::BYPASS;
  uint64_t page_id = 0;
  chunk_t* frame = nullptr;
  char* data = nullptr;
  bool coalesced = false;
  bool evicted = false;
  io_cache_access_t access = io_cache_access_t::DEMAND;
  bool prefetched_hit = false;
};

struct io_lru_buffer_pool_stats_t {
  uint64_t demand_hits = 0;
  uint64_t prefetch_hits = 0;
  uint64_t prefetch_misses = 0;
  uint64_t prefetch_used = 0;
  uint64_t prefetch_pollution_evictions = 0;
  uint64_t prefetch_admission_rejections = 0;
  uint64_t prefetch_demand_evictions = 0;
  uint64_t misses = 0;
  uint64_t coalesced_misses = 0;
  uint64_t evictions = 0;
  uint64_t pinned_victim_failures = 0;
  uint64_t failed_loads = 0;
};

/** Concurrent read-only 4-KiB LRU with miss coalescing and pinned-frame safety. */
class io_lru_buffer_pool_t {
public:
  explicit io_lru_buffer_pool_t(size_t capacity_pages,
                                chunk_type_t chunk_type = chunk_type_t::chunk_4kb);

  io_lru_buffer_pool_t(const io_lru_buffer_pool_t&) = delete;
  io_lru_buffer_pool_t& operator=(const io_lru_buffer_pool_t&) = delete;

  /** Find and pin a resident page, or reserve one exclusive frame for loading. */
  io_cache_reservation_t reserve(uint64_t page_id,
                                 io_cache_access_t access = io_cache_access_t::DEMAND,
                                 bool wait_for_inflight = true);

  /** Reserve a strictly increasing page batch while amortizing the replacement lock. */
  std::vector<io_cache_reservation_t>
  reserve_many(std::span<const uint64_t> page_ids,
               io_cache_access_t access = io_cache_access_t::DEMAND);

  /** Pin a resident page without allocating a frame or recording a miss. */
  io_cache_reservation_t try_reserve_hit(uint64_t page_id,
                                         io_cache_access_t access = io_cache_access_t::DEMAND);

  /** Publish a complete successful load, or roll its frame back without exposure. */
  void publish(const io_cache_reservation_t& reservation, bool success);

  /** Publish a batch under one replacement lock. */
  void publish_many(std::span<const io_cache_reservation_t> reservations, bool success);

  /** Release one pin acquired by a HIT or LOAD reservation. */
  void release(const io_cache_reservation_t& reservation);

  /** Release a batch under one replacement lock. */
  void release_many(std::span<const io_cache_reservation_t> reservations);

  /** Drop every unpinned page and reset counters before an isolated curve point. */
  void clear();

  /** Replace all frames with a new empty pool of the requested 4-KiB-page capacity. */
  void reset_capacity(size_t capacity_pages);

  size_t capacity_pages() const noexcept { return capacity_pages_; }
  uint64_t resident_bytes() const noexcept { return capacity_pages_ * chunk_bytes_; }
  chunk_type_t chunk_type() const noexcept { return chunk_type_; }
  io_lru_buffer_pool_stats_t stats() const;

private:
  struct resident_entry_t {
    chunk_t* frame = nullptr;
    std::list<uint64_t>::iterator lru_position;
    bool prefetched_unused = false;
  };

  size_t capacity_pages_ = 0;
  chunk_type_t chunk_type_ = chunk_type_t::chunk_4kb;
  uint64_t chunk_bytes_ = 4096;
  std::unique_ptr<chunk_buf_pool_t> frames_;
  std::vector<chunk_t*> free_frames_;
  std::unordered_map<uint64_t, resident_entry_t> resident_;
  std::unordered_map<uint64_t, chunk_t*> loading_;
  std::list<uint64_t> lru_;
  size_t prefetched_unused_count_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable completion_;
  bool prefetch_admission_credit_ = true;
  io_lru_buffer_pool_stats_t stats_;

  io_cache_reservation_t reserve_locked(uint64_t page_id, io_cache_access_t access,
                                        bool wait_for_inflight, std::unique_lock<std::mutex>& lock);
};

/** Owns independent LRU replacement domains for every configured Chunk size. */
class io_size_specific_lru_t {
public:
  explicit io_size_specific_lru_t(
      const std::array<size_t, disk_manager_t::k_chunk_levels>& capacities);

  io_lru_buffer_pool_t* pool(chunk_type_t type) noexcept;
  const io_lru_buffer_pool_t* pool(chunk_type_t type) const noexcept;
  uint64_t resident_bytes() const noexcept;

private:
  std::array<std::unique_ptr<io_lru_buffer_pool_t>, disk_manager_t::k_chunk_levels> pools_;
};

} // namespace powerlaw_ann

#endif // STORAGE_IO_LRU_BUFFER_POOL
