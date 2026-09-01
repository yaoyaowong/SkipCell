#ifndef SEARCH_PERCENTILE_STATS
#define SEARCH_PERCENTILE_STATS

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace powerlaw_ann {

inline constexpr size_t k_cell_value_max_blocks = 4;

/** @brief One diagnostic scored-Gateway event from an unchanged Base trajectory. */
struct gateway_value_record_t {
  uint32_t source_community = 0;
  uint32_t target_community = 0;
  uint32_t source_node = 0;
  uint32_t gateway_node = 0;
  uint32_t trigger_hop = 0;
  uint32_t trigger_base_reads = 0;
  uint32_t gateway_pq_rank = 0;
  uint32_t later_expanded = 0;
  uint32_t first_later_expansion_hop = 0;
  uint32_t first_later_expansion_read = 0;
  uint32_t future_expansions_covered = 0;
};

/** @brief One diagnostic Base-record expansion in query execution order. */
struct base_expansion_record_t {
  uint32_t node_id = 0;
  uint32_t community_id = std::numeric_limits<uint32_t>::max();
  uint32_t cell_id = std::numeric_limits<uint32_t>::max();
  uint32_t hop = 0;
  uint32_t base_reads = 0;
};

/** @brief Per-query counters produced by DiskANN cached beam search. */
struct query_stats_t {
  float total_us = 0;
  float io_us = 0;
  float cpu_us = 0;
  uint32_t n_4k = 0;
  uint32_t n_8k = 0;
  uint32_t n_12k = 0;
  uint32_t n_ios = 0;
  uint32_t read_size = 0;
  uint32_t n_cmps_saved = 0;
  uint32_t n_cmps = 0;
  uint32_t n_cache_hits = 0;
  uint32_t n_hops = 0;
  uint32_t n_base_neighbors_scanned = 0;
  uint32_t base_hops = 0;
  uint32_t sequential_ios = 0;
  uint32_t random_ios = 0;
  uint32_t physical_read_requests = 0;
  uint32_t cell_page_batch_requests = 0;
  uint32_t cell_page_batch_pages = 0;
  uint32_t cell_batch_search_rounds = 0;
  uint32_t cell_batch_cached_expansions = 0;
  uint32_t cell_pq_traversal_expansions = 0;
  uint32_t cell_pq_refined_candidates = 0;
  uint64_t demand_useful_bytes = 0;
  uint32_t memgraph_nodes_scored = 0;
  uint32_t memgraph_nodes_inserted = 0;
  uint32_t memgraph_nodes_later_expanded = 0;
  uint32_t dynamic_width_rounds = 0;
  uint32_t dynamic_width_sum = 0;
  uint32_t dynamic_width_max = 0;
  uint32_t dynamic_width_useful_ios = 0;
  uint32_t dynamic_width_wasted_ios = 0;
  bool base_trace_enabled = false;
  std::vector<base_expansion_record_t> base_expansion_records;
  uint32_t community_meta_expansions = 0;
  uint32_t community_superedges_considered = 0;
  uint32_t gateway_nodes_scored = 0;
  uint32_t gateway_nodes_competitive = 0;
  uint32_t gateway_nodes_inserted = 0;
  uint32_t gateway_nodes_later_expanded = 0;
  uint32_t gateway_landing_cells_routed = 0;
  uint32_t gateway_prefetch_predictions = 0;
  uint32_t gateway_prefetch_pages_submitted = 0;
  uint32_t gateway_prefetch_queue_rejections = 0;
  uint32_t gateway_prefetch_timely_hits = 0;
  uint32_t gateway_prefetch_late_hits = 0;
  uint32_t gateway_prefetch_unused = 0;
  uint64_t gateway_prefetch_lead_us = 0;
  uint32_t gateway_prefetch_lead_hops = 0;
  std::vector<gateway_value_record_t> gateway_value_records;
  uint32_t cell_candidates_created = 0;
  uint32_t cell_blocks_expanded = 0;
  uint32_t cell_blocks_hint_rejected = 0;
  uint32_t cell_nodes_scored = 0;
  uint32_t cell_nodes_competitive = 0;
  uint32_t cell_nodes_inserted = 0;
  uint32_t cell_nodes_later_expanded = 0;
  uint32_t cell_nodes_already_visited = 0;
  uint32_t cell_blocks_without_unseen_nodes = 0;
  uint32_t cell_blocks_low_yield = 0;
  uint64_t cell_packet_bytes = 0;
  uint32_t cell_terminal_candidates = 0;
  uint32_t cell_terminal_activated = 0;
  uint32_t cell_hierarchy_nodes_scored = 0;
  uint32_t cell_hierarchy_leaf_groups_probed = 0;
  uint32_t cell_adj_neighbor_edges_considered = 0;
  uint32_t cell_adj_cells_scored = 0;
  uint32_t cell_stitch_candidates_considered = 0;
  uint32_t cell_stitch_owner_cells = 0;
  uint32_t cell_stitch_max_owner_support = 0;
  uint32_t cell_stitch_multi_owner_candidates = 0;
  uint32_t cell_value_blocks_recorded = 0;
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_block_ids = {
      std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(),
      std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_cell_ids = {
      std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(),
      std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_block_support{};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_cell_support{};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_node_count{};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_already_visited{};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_nodes_scored{};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_nodes_competitive{};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_nodes_inserted{};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_future_expansions{};
  std::array<uint32_t, k_cell_value_max_blocks> cell_value_admitted{};
  std::array<uint64_t, k_cell_value_max_blocks> cell_value_packet_bytes{};
  std::array<float, k_cell_value_max_blocks> cell_value_evidence_threshold_ratio = {-1.0F, -1.0F,
                                                                                    -1.0F, -1.0F};
  std::array<float, k_cell_value_max_blocks> cell_value_lower_bound_threshold_ratio = {
      -1.0F, -1.0F, -1.0F, -1.0F};
  std::array<float, k_cell_value_max_blocks> cell_value_best_threshold_ratio = {-1.0F, -1.0F, -1.0F,
                                                                                -1.0F};
  uint32_t cell_oracle_triggered = 0;
  uint32_t cell_oracle_trigger_hop = 0;
  uint32_t cell_oracle_eligible_blocks = 0;
  uint32_t cell_oracle_eligible_cells = 0;
  uint32_t cell_oracle_future_expansions = 0;
  uint32_t cell_oracle_future_distinct_cells = 0;
  uint32_t cell_oracle_future_distinct_blocks = 0;
  uint32_t cell_oracle_selected_block_hits = 0;
  uint32_t cell_oracle_selected_cell_hits = 0;
  uint32_t cell_oracle_selected_unseen_nodes = 0;
  uint32_t cell_oracle_selected_competitive_nodes = 0;
  float cell_oracle_selected_evidence_threshold_ratio = -1.0F;
  float cell_oracle_selected_best_threshold_ratio = -1.0F;
  uint32_t cell_oracle_best_eligible_block_hits = 0;
  uint32_t cell_oracle_best_eligible_cell_hits = 0;
  uint32_t cell_oracle_polar_eligible_cell_hits = 0;
  uint32_t cell_oracle_support_eligible_cell_hits = 0;
  uint32_t cell_oracle_evidence_eligible_cell_hits = 0;
  uint32_t cell_oracle_last_source_cell_hits = 0;
  uint32_t cell_oracle_top1_cell_hits = 0;
  uint32_t cell_oracle_top2_cell_hits = 0;
  uint32_t cell_oracle_top4_cell_hits = 0;
  uint32_t cell_oracle_top4_eligible_cell_hits = 0;
  uint32_t cell_oracle_top4_support_cell_hits = 0;
  uint32_t cell_oracle_top4_evidence_cell_hits = 0;
  uint32_t cell_oracle_top4_polar_cell_hits = 0;
  uint32_t cell_oracle_top1_block_hits = 0;
  uint32_t cell_oracle_top2_block_hits = 0;
  uint32_t cell_oracle_top4_block_hits = 0;
  uint32_t cell_oracle_selected_block_id = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_best_eligible_block_id = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_best_eligible_cell_id = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_polar_eligible_cell_id = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_support_eligible_cell_id = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_evidence_eligible_cell_id = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_best_global_block_id = std::numeric_limits<uint32_t>::max();
  uint32_t macro_hint_bound_violations = 0;
  uint32_t fallback_queries = 0;
  uint32_t beam_phase_transition_found = 0;
  uint32_t beam_phase_entry_io_hop = 0;
  uint32_t beam_phase_entry_expansion_rank = 0;
  uint32_t beam_phase_total_search_rounds = 0;
  uint32_t beam_phase_approach_base_hops = 0;
  uint32_t beam_phase_convergence_base_hops = 0;
  uint32_t beam_phase_approach_ios = 0;
  uint32_t beam_phase_convergence_ios = 0;
  uint32_t beam_phase_approach_cmps = 0;
  uint32_t beam_phase_convergence_cmps = 0;
  uint32_t beam_phase_approach_community_expansions = 0;
  uint32_t beam_phase_convergence_community_expansions = 0;
  uint32_t beam_phase_approach_gateway_nodes_scored = 0;
  uint32_t beam_phase_convergence_gateway_nodes_scored = 0;
  uint32_t beam_phase_approach_cell_blocks = 0;
  uint32_t beam_phase_convergence_cell_blocks = 0;
  uint32_t beam_phase_approach_cell_nodes_scored = 0;
  uint32_t beam_phase_convergence_cell_nodes_scored = 0;
  float beam_phase_setup_us = 0;
  float beam_phase_approach_us = 0;
  float beam_phase_convergence_us = 0;
  float beam_phase_search_loop_us = 0;
  float beam_phase_post_search_us = 0;
};

template <typename value_t>
value_t get_percentile_stats(const query_stats_t* stats, uint64_t length, float percentile,
                             const std::function<value_t(const query_stats_t&)>& member_fn) {
  if (length == 0) {
    return value_t{};
  }
  std::vector<value_t> values(length);
  for (uint64_t i = 0; i < length; ++i) {
    values[i] = member_fn(stats[i]);
  }
  std::sort(values.begin(), values.end());
  const auto index = std::min<uint64_t>(static_cast<uint64_t>(percentile * length), length - 1);
  return values[index];
}

template <typename value_t>
double get_mean_stats(const query_stats_t* stats, uint64_t length,
                      const std::function<value_t(const query_stats_t&)>& member_fn) {
  if (length == 0) {
    return 0.0;
  }
  double average = 0.0;
  for (uint64_t i = 0; i < length; ++i) {
    average += static_cast<double>(member_fn(stats[i]));
  }
  return average / static_cast<double>(length);
}

} // namespace powerlaw_ann

#endif // SEARCH_PERCENTILE_STATS
