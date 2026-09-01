#ifndef SEARCH_DISKANN_SEARCH
#define SEARCH_DISKANN_SEARCH

#include "index/io_optimized_index.h"
#include "search/community_polar_search.h"
#include "search/percentile_stats.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace powerlaw_ann {

/** @brief Typed configuration for the Step 8 static float32 L2 disk search. */
struct diskann_search_config_t {
  std::filesystem::path index_path_prefix;
  std::filesystem::path query_path;
  std::filesystem::path ground_truth_path;
  std::filesystem::path result_path_prefix;
  std::filesystem::path node_visit_output_path;
  std::filesystem::path cell_oracle_output_path;
  std::filesystem::path cell_value_output_path;
  std::filesystem::path gateway_value_output_path;
  std::filesystem::path base_trace_output_path;
  std::filesystem::path community_polar_path;
  std::filesystem::path io_topology_path;
  std::filesystem::path io_vector_path;
  std::filesystem::path io_cell_adjacency_path;
  std::filesystem::path resident_u8_refinement_path;
  uint32_t top_k = 10;
  uint32_t search_list_size = 100;
  uint32_t beam_width = 4;
  uint32_t num_threads = 0;
  uint32_t num_nodes_to_cache = 0;
  uint32_t io_limit = std::numeric_limits<uint32_t>::max();
  uint32_t query_limit = 0;
  uint32_t query_offset = 0;
  uint32_t warmup_query_count = 0;
  uint32_t warmup_query_offset = 0;
  bool use_reorder_data = false;
  bool phase_profile = false;
  std::string cache_state = "warm";
};

/** @brief Platform backends active for one search execution. */
struct diskann_search_backends_t {
  std::string io;
  std::string math;
  std::string allocator;
  std::string simd;
};

/** @brief Batched Top-K IDs, distances, counters, and baseline metrics. */
struct diskann_search_result_t {
  size_t num_queries = 0;
  size_t query_dimension = 0;
  uint32_t top_k = 0;
  uint32_t num_threads = 0;
  std::vector<uint64_t> ids;
  std::vector<float> distances;
  std::vector<query_stats_t> query_stats;
  double elapsed_seconds = 0.0;
  std::optional<double> recall_percent;
  uint64_t index_size_bytes = 0;
  uint64_t resident_sidecar_bytes = 0;
  uint64_t sidecar_artifact_bytes = 0;
  uint64_t cell_adjacency_resident_bytes = 0;
  uint64_t cell_adjacency_artifact_bytes = 0;
  uint64_t topology_resident_bytes = 0;
  uint64_t topology_artifact_bytes = 0;
  uint64_t topology_cache_resident_bytes = 0;
  uint64_t topology_cache_hits = 0;
  uint64_t topology_cache_misses = 0;
  uint64_t topology_cache_evictions = 0;
  uint64_t vector_artifact_bytes = 0;
  uint64_t io_cache_resident_bytes = 0;
  uint64_t io_cache_demand_hits = 0;
  uint64_t io_cache_prefetch_hits = 0;
  uint64_t io_cache_prefetch_misses = 0;
  uint64_t io_cache_prefetch_used = 0;
  uint64_t io_cache_prefetch_pollution_evictions = 0;
  uint64_t io_cache_prefetch_admission_rejections = 0;
  uint64_t io_cache_prefetch_demand_evictions = 0;
  uint64_t io_cache_misses = 0;
  uint64_t io_cache_coalesced_misses = 0;
  uint64_t io_cache_evictions = 0;
  uint64_t io_cache_pinned_victim_failures = 0;
  uint64_t io_uring_submitted = 0;
  uint64_t io_uring_completed = 0;
  uint64_t io_uring_completion_batches = 0;
  uint64_t io_uring_wait_ns = 0;
  uint64_t io_uring_short_or_error_completions = 0;
  uint64_t io_uring_prefetch_submitted = 0;
  uint64_t io_uring_prefetch_completed = 0;
  uint64_t io_uring_prefetch_obsolete = 0;
  uint32_t io_uring_queue_depth = 0;
  uint64_t memgraph_resident_bytes = 0;
  uint32_t memgraph_nodes = 0;
  uint64_t memgraph_build_time_us = 0;
  uint64_t memgraph_build_peak_bytes = 0;
  uint64_t decoded_pq_build_time_us = 0;
  uint64_t decoded_pq_representation_bytes = 0;
  uint64_t decoded_pq_additional_resident_bytes = 0;
  uint64_t resident_u8_refinement_artifact_bytes = 0;
  uint64_t resident_u8_refinement_bytes = 0;
  uint64_t resident_u8_refinement_checksum = 0;
  uint64_t online_cell_adaptation_observations = 0;
  uint64_t online_cell_adaptation_publish_attempts = 0;
  uint64_t online_cell_adaptation_successful_publishes = 0;
  uint64_t online_cell_adaptation_publish_conflicts = 0;
  uint64_t online_cell_adaptation_retired_snapshots = 0;
  uint64_t online_cell_adaptation_physical_generations = 0;
  uint64_t online_cell_adaptation_physical_bytes_written = 0;
  uint64_t online_cell_adaptation_physical_write_time_us = 0;
  uint64_t online_cell_adaptation_checksum_failures = 0;
  uint64_t online_cell_adaptation_directory_resident_bytes = 0;
  uint64_t online_cell_adaptation_overlay_artifact_bytes = 0;
  std::string cache_state;
  diskann_search_backends_t backends;

  double qps() const;
  double mean_latency_us() const;
  double p50_latency_us() const;
  double p95_latency_us() const;
  double p99_latency_us() const;
  double reads_per_query() const;
  double io_time_us_per_query() const;
  double cpu_time_us_per_query() const;
  double read_bytes_per_query() const;
  double four_kib_reads_per_query() const;
  double cache_hits_per_query() const;
  double distance_computations_per_query() const;
  double hops_per_query() const;
  double base_neighbors_per_query() const;
  double base_hops_per_query() const;
  double sequential_ios_per_query() const;
  double random_ios_per_query() const;
  double physical_read_requests_per_query() const;
  double cell_page_batch_requests_per_query() const;
  double cell_page_batch_pages_per_query() const;
  double demand_useful_bytes_per_query() const;
  double demand_overread_bytes_per_query() const;
  double memgraph_nodes_scored_per_query() const;
  double memgraph_nodes_inserted_per_query() const;
  double memgraph_nodes_later_expanded_per_query() const;
  double dynamic_width_mean() const;
  uint32_t dynamic_width_max() const;
  double dynamic_width_useful_ios_per_query() const;
  double dynamic_width_wasted_ios_per_query() const;
  double community_meta_expansions_per_query() const;
  double community_superedges_per_query() const;
  double gateway_nodes_scored_per_query() const;
  double gateway_nodes_competitive_per_query() const;
  double gateway_nodes_inserted_per_query() const;
  double gateway_nodes_later_expanded_per_query() const;
  double gateway_prefetch_predictions_per_query() const;
  double gateway_prefetch_pages_submitted_per_query() const;
  double gateway_prefetch_queue_rejections_per_query() const;
  double gateway_prefetch_timely_hits_per_query() const;
  double gateway_prefetch_late_hits_per_query() const;
  double gateway_prefetch_unused_per_query() const;
  double gateway_prefetch_mean_lead_us() const;
  double gateway_prefetch_mean_lead_hops() const;
  double cell_candidates_per_query() const;
  double cell_blocks_per_query() const;
  double cell_hint_rejections_per_query() const;
  double cell_nodes_scored_per_query() const;
  double cell_nodes_competitive_per_query() const;
  double cell_nodes_inserted_per_query() const;
  double cell_nodes_later_expanded_per_query() const;
  double cell_nodes_already_visited_per_query() const;
  double cell_blocks_without_unseen_nodes_per_query() const;
  double cell_blocks_low_yield_per_query() const;
  double cell_packet_bytes_per_query() const;
  double cell_batch_search_rounds_per_query() const;
  double cell_batch_cached_expansions_per_query() const;
  double cell_pq_traversal_expansions_per_query() const;
  double cell_pq_refined_candidates_per_query() const;
  double cell_hierarchy_nodes_scored_per_query() const;
  double cell_hierarchy_leaf_groups_probed_per_query() const;
  double cell_adj_neighbor_edges_considered_per_query() const;
  double cell_adj_cells_scored_per_query() const;
  double cell_stitch_candidates_considered_per_query() const;
  double cell_stitch_owner_cells_per_query() const;
  double cell_stitch_max_owner_support_per_query() const;
  double cell_stitch_multi_owner_candidates_per_query() const;
  double fallback_queries() const;
  double beam_phase_transition_rate_percent() const;
  double beam_phase_approach_hops_per_query() const;
  double beam_phase_convergence_hops_per_query() const;
  double beam_phase_approach_ios_per_query() const;
  double beam_phase_convergence_ios_per_query() const;
  double beam_phase_approach_time_us() const;
  double beam_phase_convergence_time_us() const;
  double beam_phase_setup_time_us() const;
  double beam_phase_post_search_time_us() const;
  double beam_phase_approach_community_expansions_per_query() const;
  double beam_phase_convergence_community_expansions_per_query() const;
  double beam_phase_approach_gateway_nodes_scored_per_query() const;
  double beam_phase_convergence_gateway_nodes_scored_per_query() const;
  double beam_phase_approach_cell_blocks_per_query() const;
  double beam_phase_convergence_cell_blocks_per_query() const;
  double beam_phase_approach_cell_nodes_scored_per_query() const;
  double beam_phase_convergence_cell_nodes_scored_per_query() const;
};

/**
 * @brief Owns DiskANN-compatible disk-index load, cache, and Top-K execution.
 *
 * macOS selects the checked synchronous reader for correctness. Linux selects
 * the pinned libaio and direct-I/O reader for performance experiments.
 */
class diskann_search_t {
public:
  diskann_search_result_t search(const diskann_search_config_t& config);
  diskann_search_result_t search(const diskann_search_config_t& config,
                                 const io_optimization_config_t& io_config);
  diskann_search_result_t search(const diskann_search_config_t& config,
                                 const community_polar_search_config_t& hybrid_config);
  diskann_search_result_t search(const diskann_search_config_t& config,
                                 const community_polar_search_config_t& hybrid_config,
                                 const io_optimization_config_t& io_config);
  diskann_search_result_t search_observe_only(const diskann_search_config_t& config,
                                              const community_polar_search_config_t& hybrid_config);

  /** Load once and measure isolated search-list points with per-point warmup. */
  std::vector<diskann_search_result_t> search_curve(const diskann_search_config_t& config,
                                                    std::span<const uint32_t> search_list_sizes);
  std::vector<diskann_search_result_t> search_curve(
      const diskann_search_config_t& config, const community_polar_search_config_t& hybrid_config,
      const io_optimization_config_t& io_config, std::span<const uint32_t> search_list_sizes);

  /** Load once and measure isolated query-thread counts with per-point warmup. */
  std::vector<diskann_search_result_t> search_thread_curve(const diskann_search_config_t& config,
                                                           std::span<const uint32_t> thread_counts);
  std::vector<diskann_search_result_t> search_thread_curve(
      const diskann_search_config_t& config, const community_polar_search_config_t& hybrid_config,
      const io_optimization_config_t& io_config, std::span<const uint32_t> thread_counts);

  /** Load auxiliary artifacts once and rebuild an empty LRU for each cache-size point. */
  std::vector<diskann_search_result_t> search_cache_curve(
      const diskann_search_config_t& config, const community_polar_search_config_t& hybrid_config,
      const io_optimization_config_t& io_config, std::span<const uint64_t> cache_page_counts);
};

} // namespace powerlaw_ann

#endif // SEARCH_DISKANN_SEARCH
