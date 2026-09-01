#ifndef INDEX_ADAPTIVE_CELL_RUNTIME
#define INDEX_ADAPTIVE_CELL_RUNTIME

#include "index/io_optimized_index.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace powerlaw_ann {

/** Immutable physical Cell descriptor published only after its payload is ready. */
struct adaptive_cell_view_t {
  uint64_t view_id = 0;
  uint64_t generation = 0;
  uint32_t capacity_class = 0;
  uint32_t node_count = 0;
  bool physical_payload_ready = false;
  std::vector<uint32_t> logical_cell_ids;
  std::vector<io_page_run_t> page_runs;
  std::vector<uint32_t> vector_nodes;
  std::vector<io_vector_location_t> vector_locations;
  uint64_t payload_checksum = 0;
  uint64_t directory_checksum = 0;
};

struct adaptive_cell_merge_candidate_t {
  uint32_t left_cell = 0;
  uint32_t right_cell = 0;
};

struct adaptive_cell_runtime_config_t {
  bool enabled = false;
  uint64_t minimum_joint_observations = 1000;
  double minimum_coaccess_ratio = 0.90;
  double maximum_page_increase = 0.10;
  double maximum_unseen_increase = 0.10;
  uint32_t maximum_successful_publishes = UINT32_MAX;
  bool background_publish = false;
  bool force_first_observed_modification = false;
};

/** Optional process-local delta storage used by the physical modification experiment. */
struct adaptive_cell_physical_config_t {
  std::shared_ptr<const io_optimized_index_t> base_index;
  std::filesystem::path overlay_path;
  uint32_t vector_bytes = 0;
  std::vector<std::vector<uint32_t>> cell_nodes;
};

struct adaptive_cell_merge_profile_t {
  uint64_t parent_observations = 0;
  uint64_t joint_observations = 0;
  uint64_t fixed_pages = 0;
  uint64_t merged_pages = 0;
  uint64_t fixed_unseen = 0;
  uint64_t merged_unseen = 0;
  double coaccess_ratio = 0.0;
  double page_increase = 0.0;
  double unseen_increase = 0.0;
  bool eligible = false;
};

struct adaptive_cell_runtime_metrics_t {
  uint64_t observations = 0;
  uint64_t publish_attempts = 0;
  uint64_t successful_publishes = 0;
  uint64_t publish_conflicts = 0;
  uint64_t retired_snapshots = 0;
  uint64_t physical_generations = 0;
  uint64_t physical_bytes_written = 0;
  uint64_t physical_write_time_us = 0;
  uint64_t checksum_failures = 0;
  uint64_t directory_resident_bytes = 0;
  uint64_t overlay_artifact_bytes = 0;
};

/** One immutable mapping-table generation held by readers through shared ownership. */
class adaptive_cell_snapshot_t {
public:
  ~adaptive_cell_snapshot_t();

  uint64_t generation() const noexcept { return generation_; }
  const adaptive_cell_view_t& cell(uint32_t logical_cell) const;
  io_vector_location_t vector_location(uint32_t node_id,
                                       io_vector_location_t fallback) const noexcept;
  uint64_t cell_count() const noexcept { return cells_.size(); }

private:
  friend class adaptive_cell_runtime_t;

  uint64_t generation_ = 0;
  std::vector<std::shared_ptr<const adaptive_cell_view_t>> cells_;
  std::vector<uint32_t> override_nodes_;
  std::vector<io_vector_location_t> override_locations_;
  std::array<uint64_t, 64> override_filter_{};
  std::shared_ptr<std::atomic<uint64_t>> retirement_counter_;
  bool published_ = false;
};

/**
 * Default-off Bw-Tree-style Cell mapping table. Query threads only take a shared snapshot and
 * update bounded atomic counters. A background builder publishes a fully materialized replacement
 * with one mapping-table CAS; shared ownership reclaims the old generation after its last reader.
 */
class adaptive_cell_runtime_t {
public:
  adaptive_cell_runtime_t(std::span<const io_cell_page_range_t> fixed_cells,
                          std::span<const adaptive_cell_merge_candidate_t> candidates,
                          adaptive_cell_runtime_config_t config);
  adaptive_cell_runtime_t(std::span<const io_cell_page_range_t> fixed_cells,
                          std::span<const adaptive_cell_merge_candidate_t> candidates,
                          adaptive_cell_runtime_config_t config,
                          adaptive_cell_physical_config_t physical_config);
  ~adaptive_cell_runtime_t();

  adaptive_cell_runtime_t(const adaptive_cell_runtime_t&) = delete;
  adaptive_cell_runtime_t& operator=(const adaptive_cell_runtime_t&) = delete;

  std::shared_ptr<const adaptive_cell_snapshot_t> acquire() const noexcept;
  const adaptive_cell_snapshot_t* final_snapshot() const noexcept {
    return final_snapshot_.load(std::memory_order_acquire);
  }

  void observe(uint32_t candidate, bool left_loaded, bool right_loaded, uint32_t fixed_pages,
               uint32_t merged_pages, uint32_t fixed_unseen, uint32_t merged_unseen) noexcept;
  void observe_query(std::span<const uint32_t> loaded_cells) noexcept;
  adaptive_cell_merge_profile_t profile(uint32_t candidate) const;
  uint32_t logical_cell(uint32_t node_id) const noexcept;
  const std::filesystem::path& overlay_path() const noexcept { return overlay_path_; }

  /** Publish a ready replacement iff the frozen profile gate passes and the snapshot CAS wins. */
  bool try_publish(uint32_t candidate, std::shared_ptr<const adaptive_cell_view_t> replacement,
                   bool force_experimental_operation = false);
  void drain_background();

  adaptive_cell_runtime_metrics_t metrics() const noexcept;
  bool enabled() const noexcept { return config_.enabled; }
  bool accepts_observations() const noexcept {
    return config_.enabled && successful_publishes_.load(std::memory_order_relaxed) <
                                  config_.maximum_successful_publishes;
  }

private:
  struct candidate_counters_t;

  void background_loop() noexcept;
  std::shared_ptr<const adaptive_cell_view_t> make_contiguous_replacement(uint32_t candidate) const;
  std::shared_ptr<const adaptive_cell_view_t> make_physical_replacement(uint32_t candidate);
  void initialize(std::span<const io_cell_page_range_t> fixed_cells);

  adaptive_cell_runtime_config_t config_;
  std::vector<adaptive_cell_merge_candidate_t> candidates_;
  std::vector<std::vector<uint32_t>> candidates_by_cell_;
  std::unique_ptr<candidate_counters_t[]> counters_;
  std::shared_ptr<const adaptive_cell_snapshot_t> snapshot_;
  std::atomic<const adaptive_cell_snapshot_t*> final_snapshot_{nullptr};
  std::shared_ptr<std::atomic<uint64_t>> retirement_counter_;
  std::atomic<uint64_t> observations_{0};
  std::atomic<uint64_t> publish_attempts_{0};
  std::atomic<uint64_t> successful_publishes_{0};
  std::atomic<uint64_t> publish_conflicts_{0};
  std::unique_ptr<std::atomic<bool>[]> pending_;
  std::atomic<bool> stop_background_{false};
  std::atomic<uint32_t> background_active_{0};
  std::atomic<bool> forced_modification_queued_{false};
  std::atomic<uint32_t> forced_candidate_{UINT32_MAX};
  std::shared_ptr<const io_optimized_index_t> base_index_;
  std::filesystem::path overlay_path_;
  std::filesystem::path generation_path_;
  uint32_t vector_bytes_ = 0;
  std::vector<std::vector<uint32_t>> cell_nodes_;
  std::vector<uint32_t> node_to_cell_;
  int base_file_descriptor_ = -1;
  int overlay_file_descriptor_ = -1;
  uint64_t base_page_count_ = 0;
  uint64_t next_overlay_page_ = 0;
  std::atomic<uint64_t> physical_generations_{0};
  std::atomic<uint64_t> physical_bytes_written_{0};
  std::atomic<uint64_t> physical_write_time_us_{0};
  std::atomic<uint64_t> checksum_failures_{0};
  std::mutex background_mutex_;
  std::condition_variable background_wakeup_;
  std::thread background_thread_;
};

} // namespace powerlaw_ann

#endif // INDEX_ADAPTIVE_CELL_RUNTIME
