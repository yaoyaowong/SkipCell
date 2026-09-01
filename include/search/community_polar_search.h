#ifndef SEARCH_COMMUNITY_POLAR_SEARCH
#define SEARCH_COMMUNITY_POLAR_SEARCH

#include "index/community_polar_index.h"
#include "index/io_optimized_index.h"
#include "search/percentile_stats.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace powerlaw_ann {

/** @brief Selects the unchanged DiskANN path or the explicit Community-Polar hybrid path. */
enum class powerann_search_mode_t {
  BASELINE,
  HYBRID,
};

/** @brief Configures the bounded Community-Skip and Cell-Expansion query operators. */
struct community_polar_search_config_t {
  powerann_search_mode_t mode = powerann_search_mode_t::BASELINE;
  bool community_skip = false;
  /** Enables an ablation-only complete overlay among persisted Navigation Hubs. */
  bool community_hub_clique = false;
  /** Bounds topology/PQ-only Base hops in the Community approaching phase. */
  uint32_t community_pq_approach_hops = 0;
  bool cell_expansion = false;
  uint32_t community_expansion_budget = 4;
  uint32_t gateway_score_budget = 32;
  // Bounds one macro step so the total Gateway budget can be distributed across Communities.
  uint32_t gateway_score_batch_cap = std::numeric_limits<uint32_t>::max();
  uint32_t gateway_direct_insert_cap = std::numeric_limits<uint32_t>::max();
  uint32_t cell_block_budget = 4;
  uint32_t cell_support = 2;
  uint32_t cell_insert_cap = 32;
  uint32_t cell_activation_marker = 0;
  uint32_t cell_min_search_list_size = 20;
  uint32_t cell_min_competitive_count = 1;
  float cell_min_competitive_fraction = 0.0F;
  float cell_evidence_threshold_ratio = 0.8F;
  float cell_candidate_threshold_ratio = 1.0F;
  float cell_insert_fraction = 1.0F;
  float graph_guard_fraction = 0.50F;
  bool cell_require_full_frontier = false;
  bool cell_use_provisional_frontier = false;
  bool cell_whole_cell_scan = false;
  bool cell_centroid_routing = false;
  bool gateway_landing_cell_handoff = false;
  uint32_t cell_routing_community_budget = 1;
  uint32_t cell_routing_cell_budget = 2;
  bool cell_terminal_convergence = false;
  uint32_t cell_terminal_approach_hops = 4;
  bool cell_terminal_l_aware_approach = false;
  uint32_t cell_terminal_approach_base_hops = 2;
  uint32_t cell_terminal_approach_l_divisor = 5;
  uint32_t cell_terminal_low_l_stitch_max_l = 0;
  uint32_t cell_terminal_low_l_stitch_extra_hops = 0;
  uint32_t cell_terminal_refine_hops = 0;
  bool cell_terminal_l_aware_refine = false;
  uint32_t cell_terminal_refine_base_hops = 0;
  uint32_t cell_terminal_refine_l_divisor = 25;
  float cell_terminal_distance_scale = 1.0F;
  uint32_t cell_terminal_frontier_cells = 0;
  bool cell_terminal_stitch_promote_cells = false;
  uint32_t cell_terminal_stitch_cell_budget = 4;
  uint32_t cell_terminal_stitch_min_support = 2;
  bool cell_terminal_l_aware_forest = false;
  uint32_t cell_terminal_forest_base_cells = 1;
  uint32_t cell_terminal_forest_l_divisor = 10;
  uint32_t cell_terminal_low_l_forest_max_l = 0;
  uint32_t cell_terminal_low_l_forest_extra_cells = 0;
  uint32_t cell_terminal_high_l_forest_min_l = 0;
  uint32_t cell_terminal_high_l_forest_cell_cap = 0;
  uint32_t cell_terminal_high_l_forest_cap_l_divisor = 0;
  bool cell_terminal_global_forest_routing = false;
  std::filesystem::path cell_terminal_route_path;
  bool cell_hierarchical_routing = false;
  uint32_t cell_hierarchy_branching = 8;
  uint32_t cell_hierarchy_leaf_size = 16;
  uint32_t cell_hierarchy_probe_budget = 16;
  bool cell_hierarchy_exact_routing = true;
  bool cell_hierarchy_preserve_leaf_order = true;
  bool cell_hierarchy_require_persisted = false;
  bool cell_hierarchy_graph_guided_routing = false;
  bool cell_defer_hierarchy_routing = false;
  bool cell_value_telemetry = false;
  bool gateway_value_telemetry = false;
  bool fast_query_distance = false;
  bool source_cell_batching = false;

  bool is_hybrid() const noexcept { return mode == powerann_search_mode_t::HYBRID; }
};

/** @brief Identifies how one candidate first entered the shared DiskANN frontier. */
enum class community_polar_candidate_origin_t : uint8_t {
  BASE_EDGE = 0,
  COMMUNITY_GATEWAY = 1,
  POLAR_CELL = 2,
};

/** @brief Identifies one scored real Gateway and its directed superedge provenance. */
struct community_polar_gateway_proposal_t {
  uint64_t gateway_id = 0;
  uint32_t source_community = 0;
  uint32_t target_community = 0;
  uint32_t source_node = 0;
  uint32_t target_node = 0;
};

/** @brief One bounded metadata proposal whose real nodes still require the normal query PQ table.
 */
struct community_polar_candidate_batch_t {
  community_polar_candidate_origin_t origin = community_polar_candidate_origin_t::BASE_EDGE;
  uint32_t object_id = 0;
  uint32_t block_ordinal = std::numeric_limits<uint32_t>::max();
  uint32_t insert_cap = std::numeric_limits<uint32_t>::max();
  float lower_bound_squared = 0.0F;
  float evidence_distance_squared = std::numeric_limits<float>::infinity();
  uint64_t packet_bytes = 0;
  uint64_t packet_record_begin = std::numeric_limits<uint64_t>::max();
  uint64_t packet_record_count = 0;
  std::span<const uint32_t> node_ids;
  std::span<const community_polar_gateway_proposal_t> gateway_proposals;
};

/** @brief Per-query mutable state; one instance is owned by one search invocation. */
class community_polar_query_state_t {
public:
  community_polar_candidate_origin_t origin(uint32_t node_id) const noexcept;
  uint32_t cell_block_ordinal(uint32_t node_id) const noexcept;
  void mark_macro_inserted(uint32_t node_id, community_polar_candidate_origin_t origin,
                           uint32_t block_ordinal = std::numeric_limits<uint32_t>::max());
  void mark_base_reached(uint32_t node_id);

private:
  friend class community_polar_search_t;

  void reset_cell_scratch(size_t cell_count, size_t block_count);
  void touch_cell(uint32_t cell_id);
  void touch_block(uint32_t block_id);

  std::vector<float> query;
  std::vector<float> community_lower_bounds;
  std::vector<float> community_routing_scores;
  std::vector<float> community_query_radii;
  std::vector<double> sector_query_angles;
  std::vector<uint8_t> reachable_communities;
  std::vector<uint8_t> expanded_communities;
  std::vector<uint32_t> community_support_counts;
  std::vector<uint8_t> community_cell_candidate_created;
  std::vector<uint8_t> cell_candidate_created;
  std::vector<uint32_t> cell_support_counts;
  std::vector<uint32_t> block_support_counts;
  std::vector<float> cell_best_graph_distance;
  std::vector<float> block_best_graph_distance;
  std::vector<uint8_t> loaded_blocks;
  std::vector<uint32_t> cell_routing_ranks;
  std::vector<uint8_t> touched_cell_flags;
  std::vector<uint8_t> touched_block_flags;
  std::vector<uint32_t> touched_cells;
  std::vector<uint32_t> touched_blocks;
  std::vector<uint32_t> routed_cell_scratch;
  std::vector<uint32_t> hierarchy_community_scratch;
  std::vector<uint32_t> hierarchy_hint_path_scratch;
  std::unordered_set<uint32_t> observed_base_sources;
  std::unordered_set<uint32_t> observed_cell_evidence_sources;
  std::unordered_map<uint32_t, community_polar_candidate_origin_t> macro_origins;
  std::unordered_map<uint32_t, uint32_t> macro_block_ordinals;
  bool cell_value_telemetry_enabled = false;
  std::vector<uint64_t> edge_order_scratch;
  std::vector<uint32_t> proposal_node_scratch;
  std::vector<community_polar_gateway_proposal_t> gateway_proposal_scratch;
  std::vector<uint32_t> cell_evidence_scratch;
  std::vector<uint32_t> block_evidence_scratch;
  std::vector<uint32_t> supported_block_scratch;
  std::vector<uint8_t> cell_oracle_eligible_blocks;
  std::vector<uint8_t> cell_oracle_eligible_cells;
  std::vector<uint32_t> cell_oracle_block_hits;
  std::vector<uint32_t> cell_oracle_cell_hits;
  std::unordered_set<uint32_t> cell_oracle_previsited_nodes;
  bool cell_oracle_active = false;
  bool cell_runtime_enabled = false;
  uint32_t cell_oracle_selected_block = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_polar_eligible_cell = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_support_eligible_cell = std::numeric_limits<uint32_t>::max();
  uint32_t cell_oracle_evidence_eligible_cell = std::numeric_limits<uint32_t>::max();
  uint32_t last_base_cell = std::numeric_limits<uint32_t>::max();
  uint32_t best_base_cell = std::numeric_limits<uint32_t>::max();
  float best_base_cell_distance = std::numeric_limits<float>::infinity();
  std::vector<std::pair<float, uint32_t>> graph_guided_cell_scratch;
  std::array<uint32_t, k_cell_value_max_blocks> cell_oracle_selected_blocks = {
      std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max(),
      std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
  uint32_t community_expansions = 0;
  uint32_t gateway_scores = 0;
  uint32_t cell_blocks = 0;
  uint32_t next_routed_cell = 0;
  uint32_t terminal_cell_budget = 0;
  uint32_t cell_hierarchy_nodes_scored = 0;
  uint32_t cell_hierarchy_leaf_groups_probed = 0;
  bool cell_hierarchy_stats_recorded = false;
};

/**
 * @brief Owns one validated resident sidecar and produces graph-triggered macro proposals.
 *
 * The class never reads Base records and never owns a result queue. Callers PQ-score returned real
 * node IDs and deduplicate/insert them through the existing DiskANN query state.
 */
class community_polar_search_t {
public:
  explicit community_polar_search_t(std::shared_ptr<community_polar_index_t> index,
                                    const community_polar_search_config_t& config = {},
                                    uint64_t artifact_bytes = 0, uint64_t artifact_checksum = 0);

  /** @brief Loads, checksums, fingerprints, and retains one sidecar fully in memory. */
  static std::shared_ptr<community_polar_search_t>
  load(const std::filesystem::path& index_path_prefix,
       const community_polar_search_config_t& config,
       const std::filesystem::path& sidecar_path = {});

  community_polar_query_state_t begin_query(std::span<const float> query) const;

  /** @brief Resets caller-owned query scratch while retaining its allocated capacity. */
  void begin_query(std::span<const float> query, community_polar_query_state_t& state,
                   bool cell_runtime_enabled = true) const;

  /** @brief Adds one distinct expanded Base source to live Community and Cell evidence. */
  void observe_base_expansion(community_polar_query_state_t& state, uint32_t source_node,
                              query_stats_t* stats) const;

  /** Activate the expanded node's own Cell Block without per-neighbor evidence bookkeeping. */
  void observe_source_cell(community_polar_query_state_t& state, uint32_t source_node,
                           float source_distance_squared, query_stats_t* stats) const;

  /** @brief Adds distinct graph evidence from one expanded source to concrete Cells and Blocks. */
  void observe_base_candidates(community_polar_query_state_t& state, uint32_t source_node,
                               std::span<const uint32_t> candidate_nodes,
                               std::span<const float> candidate_distances,
                               query_stats_t* stats) const;

  /** @brief Expands one reachable Community metadata item without reading a Base record. */
  std::optional<community_polar_candidate_batch_t>
  expand_next_community(community_polar_query_state_t& state, query_stats_t* stats) const;

  /** @brief Returns the next graph-supported contiguous Cell Block packet. */
  std::optional<community_polar_candidate_batch_t>
  expand_next_cell(community_polar_query_state_t& state, uint64_t search_list_size,
                   query_stats_t* stats) const;

  /** @brief Returns the next mature Cell Block under the live Top-L threshold. */
  std::optional<community_polar_candidate_batch_t>
  expand_next_cell(community_polar_query_state_t& state, uint64_t search_list_size,
                   float candidate_threshold, query_stats_t* stats) const;

  bool has_community_work(const community_polar_query_state_t& state) const noexcept;
  bool has_cell_work(const community_polar_query_state_t& state) const noexcept;
  bool has_cell_work(const community_polar_query_state_t& state,
                     float candidate_threshold) const noexcept;
  bool accepts_cell_evidence(const community_polar_query_state_t& state) const noexcept;

  void record_gateway_scores(community_polar_query_state_t& state, uint32_t count) const;

  /** @brief Routes one scored Gateway into its Landing Cell for sequential convergence. */
  bool observe_gateway_landing_candidate(community_polar_query_state_t& state, uint32_t node_id,
                                         float distance_squared, query_stats_t* stats) const;

  /** @brief Mixes frontier-supported Cells ahead of centroid-routed convergence Cells. */
  void prepare_terminal_convergence(community_polar_query_state_t& state,
                                    uint64_t search_list_size = 0) const;

  /** @brief Starts a baseline-equivalent Cell concentration oracle at the live trigger. */
  bool begin_cell_oracle(community_polar_query_state_t& state, query_stats_t* stats,
                         float candidate_threshold = std::numeric_limits<float>::infinity()) const;

  /** @brief Excludes a node already discovered when the oracle trigger snapshot is taken. */
  void record_cell_oracle_previsited(community_polar_query_state_t& state, uint32_t node_id) const;

  /** @brief Records the online Block selected by the unchanged proposal policy. */
  void record_cell_oracle_selection(community_polar_query_state_t& state, uint32_t block_id,
                                    uint32_t block_ordinal) const;

  /** @brief Counts one later Base expansion without affecting visited or frontier state. */
  void observe_cell_oracle_expansion(community_polar_query_state_t& state, uint32_t node_id) const;

  /** @brief Reduces per-Block/Cell oracle counts into one query's diagnostic counters. */
  void finish_cell_oracle(community_polar_query_state_t& state, query_stats_t* stats) const;

  /** @brief Copies the sidecar's existing DiskANN PQ codes into a contiguous scoring buffer. */
  void copy_pq_codes(std::span<const uint32_t> node_ids, std::span<uint8_t> output) const;

  /** Copy a consecutive packet-order PQ range for one-time SIMD representation construction. */
  void copy_packet_pq_codes(uint64_t packet_begin, uint64_t packet_count,
                            std::span<uint8_t> output) const;

  /** Release packet PQ bytes after the forced-tree decoded representation has been built. */
  void compact_for_forced_tree();

  const community_polar_search_config_t& config() const noexcept { return config_; }
  /** Return the explicit terminal handoff budget for one search-list size. */
  uint32_t terminal_approach_hops(uint64_t search_list_size) const noexcept;
  /** Return the frozen L-aware terminal Cell-forest width. */
  uint32_t terminal_cell_budget(uint64_t search_list_size) const noexcept;
  /** Return the frozen L-aware Base stitching budget after terminal Cells. */
  uint32_t terminal_refine_hops(uint64_t search_list_size) const noexcept;
  uint64_t point_count() const noexcept { return index_->point_count; }
  uint32_t dimension() const noexcept { return index_->dimension; }
  uint32_t pq_code_width() const noexcept { return index_->pq_code_width; }
  uint64_t artifact_bytes() const noexcept { return artifact_bytes_; }
  io_artifact_fingerprint_t artifact_fingerprint() const noexcept {
    return {artifact_bytes_, artifact_checksum_};
  }
  uint64_t gateway_count() const noexcept { return index_->gateways.size(); }
  uint64_t resident_bytes() const noexcept;

  /** @brief Returns the primary build-time Community for one validated Base node. */
  uint32_t community_id(uint32_t node_id) const noexcept {
    return index_->node_to_community[node_id];
  }

  /** @brief Returns the primary build-time Cell for one validated Base node. */
  uint32_t cell_id(uint32_t node_id) const noexcept { return index_->node_to_cell[node_id]; }
  uint64_t cell_count() const noexcept { return index_->cells.size(); }

  /** Return the one-owner packet-order node IDs for a validated leaf Cell. */
  std::span<const uint32_t> cell_node_ids(uint32_t cell_id) const noexcept;

  /** Squared query distance to one persisted graph-local Cell centroid. */
  float cell_query_distance_squared(const community_polar_query_state_t& state,
                                    uint32_t cell_id) const noexcept;

  /** Deterministically ranks Navigation Hubs and directed Gateway endpoints for MemGraph. */
  std::vector<uint32_t> important_memgraph_nodes(uint32_t budget) const;

  /** @brief Returns the next query-routed Cell without advancing proposal state. */
  std::optional<uint32_t>
  next_routed_cell(const community_polar_query_state_t& state) const noexcept {
    if (state.next_routed_cell >= state.routed_cell_scratch.size()) {
      return std::nullopt;
    }
    return state.routed_cell_scratch[state.next_routed_cell];
  }

private:
  bool has_direct_cell_work(const community_polar_query_state_t& state) const noexcept;
  bool is_mature_cell_block(const community_polar_query_state_t& state, uint32_t block_id,
                            float candidate_threshold) const noexcept;
  float block_lower_bound_squared(community_polar_query_state_t& state, uint32_t block_id) const;
  float query_distance_squared(std::span<const float> left,
                               std::span<const float> right) const noexcept;
  void build_cell_hierarchy();
  void route_hierarchical_cells(std::span<const float> query, std::span<const uint32_t> communities,
                                community_polar_query_state_t& state) const;

  std::shared_ptr<community_polar_index_t> index_;
  community_polar_search_config_t config_;
  uint64_t artifact_bytes_ = 0;
  uint64_t artifact_checksum_ = 0;
  std::vector<uint32_t> cell_to_sector_;
  std::vector<uint32_t> node_to_block_;
  std::vector<uint32_t> packet_node_ids_;
  uint32_t terminal_route_bucket_count_ = 0;
  uint32_t terminal_route_cells_per_bucket_ = 0;
  std::vector<float> terminal_route_centroids_;
  std::vector<uint32_t> terminal_route_cells_;
  std::vector<community_polar_cell_hierarchy_node_t> cell_hierarchy_nodes_;
  std::vector<uint32_t> cell_hierarchy_children_;
  std::vector<float> cell_hierarchy_centroids_;
  std::vector<uint32_t> community_hierarchy_roots_;
  std::vector<uint32_t> hierarchy_root_communities_;
  std::vector<uint32_t> cell_hierarchy_parents_;
  std::vector<uint32_t> cell_hierarchy_leaf_parents_;
};

/** @brief Validates one mode/operator/budget combination before any sidecar access. */
void validate_community_polar_search_config(const community_polar_search_config_t& config);

/** @brief Returns whether one more unexpanded macro candidate fits behind the Base guard. */
bool graph_guard_allows_macro(uint64_t search_list_size, float graph_guard_fraction,
                              uint64_t unexpanded_macro_candidates);

} // namespace powerlaw_ann

#endif // SEARCH_COMMUNITY_POLAR_SEARCH
