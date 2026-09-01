#include "power_ann.h"

#include "common/ann_error.h"
#include "index/community_polar_index.h"
#include "index/rcni.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <span>
#include <stdexcept>

namespace powerlaw_ann {
namespace {

std::filesystem::path make_rcni_output_path(const std::filesystem::path& index_path_prefix) {
  std::filesystem::path output_path = index_path_prefix;
  output_path += ".rcni.csv";
  return output_path;
}

std::filesystem::path
make_candidate_hubs_output_path(const std::filesystem::path& index_path_prefix) {
  std::filesystem::path output_path = index_path_prefix;
  output_path += ".candidate_hubs.csv";
  return output_path;
}

void write_powerann_build_outputs(const std::filesystem::path& index_path_prefix,
                                  std::span<const rcni_node_importance_t> importance,
                                  const candidate_hub_selection_result_t& candidate_hubs) {
  write_rcni_importance_csv(make_rcni_output_path(index_path_prefix), importance);
  write_candidate_hubs_csv(make_candidate_hubs_output_path(index_path_prefix), candidate_hubs);
}

void validate_powerann_build_config(const power_ann_config_t& config) {
  try {
    validate_candidate_hub_selection_config(config.hub_selection);
    if (config.community_polar_build.is_enabled()) {
      validate_community_polar_build_config(config.community_polar_build);
    }
  } catch (const std::invalid_argument& error) {
    throw ann_exception_t(error.what());
  }
}

} // namespace

power_ann_t::power_ann_t(const power_ann_config_t& config) : config_(config) {}

bool power_ann_t::is_powerann_enabled() const noexcept { return config_.enable_powerann; }

void power_ann_t::build_diskann_memory_index(const diskann_memory_index_config_t& config) {
  diskann_index_t index;
  if (!config_.enable_powerann) {
    index.build_memory_index(config);
    return;
  }

  validate_powerann_build_config(config_);
  if (config_.community_polar_build.is_enabled()) {
    throw ann_exception_t(
        "Community-Polar construction is supported only by the one-shot disk build");
  }
  rcni_aggregator_t aggregator(config.alpha);
  index.build_memory_index(config, &aggregator);
  const auto importance = aggregator.compute_importance();
  const auto candidate_hubs = select_candidate_hubs(importance, config_.hub_selection);
  write_powerann_build_outputs(config.index_path_prefix, importance, candidate_hubs);
}

void power_ann_t::build_diskann_index(const diskann_disk_index_config_t& config) {
  diskann_index_t index;
  if (!config_.enable_powerann) {
    index.build_disk_index(config);
    return;
  }

  validate_powerann_build_config(config_);
  rcni_aggregator_t aggregator;
  if (!config_.community_polar_build.is_enabled()) {
    index.build_disk_index(config, &aggregator);
    const auto importance = aggregator.compute_importance();
    const auto candidate_hubs = select_candidate_hubs(importance, config_.hub_selection);
    write_powerann_build_outputs(config.index_path_prefix, importance, candidate_hubs);
    return;
  }

  auto community_config = config_.community_polar_build;
  if (community_config.partition_threads == 0 && config.num_threads != 0) {
    community_config.partition_threads = config.num_threads;
  }
  community_polar_collector_t collector(aggregator, community_config);
  index.build_disk_index(config, &aggregator, &collector);
  const auto importance = aggregator.compute_importance();
  const auto candidate_hubs = select_candidate_hubs(importance, config_.hub_selection);
  write_powerann_build_outputs(config.index_path_prefix, importance, candidate_hubs);
  try {
    collector.finalize(config.index_path_prefix);
  } catch (const std::exception& error) {
    throw ann_exception_t("Community-Polar finalize failed: " + std::string(error.what()));
  }
}

void power_ann_t::build_community_polar_sidecar(const std::filesystem::path& data_path,
                                                const std::filesystem::path& index_path_prefix,
                                                const std::filesystem::path& rcni_path,
                                                const std::filesystem::path& output_sidecar_path) {
  if (!config_.enable_powerann) {
    throw ann_exception_t("sidecar-only construction requires enable_powerann=true");
  }
  validate_powerann_build_config(config_);
  if (!config_.community_polar_build.is_enabled()) {
    throw ann_exception_t("sidecar-only construction requires a positive Community count");
  }
  try {
    build_community_polar_sidecar_from_disk(data_path, index_path_prefix, rcni_path,
                                            output_sidecar_path, config_.community_polar_build);
  } catch (const ann_exception_t&) {
    throw;
  } catch (const std::exception& error) {
    throw ann_exception_t("Community-Polar sidecar-only build failed: " +
                          std::string(error.what()));
  }
}

std::vector<diskann_search_result_t> power_ann_t::search_diskann_index_impl(
    const diskann_search_config_t& config, std::span<const uint32_t> search_list_sizes,
    std::span<const uint32_t> thread_counts, std::span<const uint64_t> cache_page_counts) {
  if (search_list_sizes.empty()) {
    throw ann_exception_t("search curve requires at least one search-list size");
  }
  if ((!thread_counts.empty() || !cache_page_counts.empty()) && search_list_sizes.size() != 1) {
    throw ann_exception_t("thread/cache curve requires exactly one search-list size");
  }
  const bool thread_curve = !thread_counts.empty();
  const bool cache_curve = !cache_page_counts.empty();
  if (thread_curve && cache_curve) {
    throw ann_exception_t("thread and cache curves are mutually exclusive");
  }
  const bool curve = search_list_sizes.size() > 1 || thread_curve || cache_curve;
  diskann_search_t search;
  const bool io_enabled = config_.io_optimization.enable_topology_vector;
  if (config_.io_optimization.enable_paged_topology &&
      (!io_enabled || config_.io_optimization.topology_cache_pages == 0)) {
    throw ann_exception_t(
        "Paged topology requires topology/vector search and a positive topology cache");
  }
  if (!config_.io_optimization.enable_paged_topology &&
      config_.io_optimization.topology_cache_pages != 0) {
    throw ann_exception_t("Topology cache capacity requires paged topology");
  }
  if (config_.community_polar_search.community_pq_approach_hops != 0 && !io_enabled) {
    throw ann_exception_t(
        "Community PQ approach hops require the separated topology/vector layout");
  }
  if (config_.io_optimization.enable_cell_layout && !io_enabled) {
    throw ann_exception_t("Cell vector layout requires topology/vector search");
  }
  if (config_.io_optimization.enable_cell_u8_layout &&
      (!io_enabled || !config_.io_optimization.enable_cell_layout ||
       !config_.io_optimization.enable_cell_pq_traversal ||
       (!config_.io_optimization.enable_l_aware_search &&
        !config_.io_optimization.enable_forced_cell_tree_search) ||
       (config_.io_optimization.enable_cell_leaf_refinement &&
        !config_.io_optimization.enable_forced_cell_tree_search))) {
    throw ann_exception_t("Cell-u8 layout requires L-aware top-N or forced-tree refinement");
  }
  if (config_.io_optimization.enable_cell_page_batch &&
      (!io_enabled || !config_.io_optimization.enable_cell_layout ||
       !config_.io_optimization.enable_lru_cache)) {
    throw ann_exception_t(
        "Cell page batching requires fixed Cell-contiguous layout and the IO LRU");
  }
  if (config_.io_optimization.enable_cell_batch_search &&
      (!config_.enable_powerann ||
       config_.community_polar_search.mode != powerann_search_mode_t::HYBRID ||
       (!config_.community_polar_search.community_skip &&
        !config_.io_optimization.enable_forced_cell_tree_search &&
        !(config_.community_polar_search.cell_terminal_convergence &&
          config_.community_polar_search.cell_hierarchical_routing)) ||
       !config_.community_polar_search.cell_expansion ||
       !config_.io_optimization.enable_cell_layout ||
       !config_.io_optimization.enable_cell_page_batch ||
       !config_.io_optimization.enable_lru_cache)) {
    throw ann_exception_t(
        "Cell-batched search requires Hybrid Cell traversal, a Community, terminal hierarchy, "
        "or forced-tree approach, Cell-4K page batching, and the IO LRU");
  }
  if (config_.io_optimization.enable_interleaved_cell_batch_search &&
      (!config_.io_optimization.enable_cell_batch_search ||
       !config_.io_optimization.enable_cell_pq_traversal ||
       !config_.io_optimization.enable_forced_cell_tree_search ||
       config_.io_optimization.interleaved_cell_batch_graph_hops == 0)) {
    throw ann_exception_t(
        "Interleaved Cell-batched search requires forced-tree Cell-PQ batching and a positive "
        "graph-hop interval");
  }
  if (config_.io_optimization.cell_batch_preserve_graph &&
      (!config_.io_optimization.enable_cell_batch_search ||
       !config_.io_optimization.enable_cell_pq_traversal ||
       !config_.io_optimization.enable_forced_cell_tree_search ||
       config_.io_optimization.enable_interleaved_cell_batch_search)) {
    throw ann_exception_t(
        "Graph-preserving Cell batch requires non-interleaved forced-tree Cell-PQ search");
  }
  if (config_.io_optimization.cell_batch_graph_stitch_min_l != 0 &&
      !config_.io_optimization.cell_batch_preserve_graph) {
    throw ann_exception_t("L-aware Cell graph stitching requires graph-preserving Cell batching");
  }
  if (config_.io_optimization.enable_ranked_cell_page_refinement &&
      (!config_.io_optimization.enable_cell_batch_search ||
       !config_.io_optimization.cell_batch_preserve_graph ||
       !config_.io_optimization.enable_cell_pq_traversal ||
       !config_.io_optimization.enable_forced_cell_tree_search ||
       config_.io_optimization.enable_cell_leaf_refinement ||
       config_.io_optimization.ranked_cell_page_l_divisor == 0 ||
       config_.io_optimization.ranked_cell_page_max_pages == 0 ||
       config_.io_optimization.ranked_cell_page_base_pages >
           config_.io_optimization.ranked_cell_page_max_pages ||
       ((config_.io_optimization.ranked_cell_page_growth_start_l == 0) !=
        (config_.io_optimization.ranked_cell_page_growth_divisor == 0)) ||
       config_.io_optimization.ranked_cell_page_max_l == 0)) {
    throw ann_exception_t(
        "Ranked Cell-page refinement requires graph-preserving forced-tree Cell batching and a "
        "positive bounded page budget");
  }
  if (config_.io_optimization.enable_cell_pq_traversal &&
      (!config_.io_optimization.enable_topology_vector ||
       !config_.io_optimization.enable_cell_layout)) {
    throw ann_exception_t("Resident PQ traversal requires topology/vector search and Cell-4K");
  }
  if (config_.io_optimization.enable_cell_adj_correction &&
      (!config_.io_optimization.enable_forced_cell_tree_search ||
       !config_.io_optimization.enable_cell_pq_traversal ||
       config_.io_optimization.cell_adj_candidate_cells == 0 ||
       config_.io_optimization.cell_adj_expansion_cells == 0 ||
       config_.io_optimization.cell_adj_min_search_list_size == 0 ||
       config_.io_optimization.cell_adj_max_search_list_size == 0 ||
       config_.io_optimization.cell_adj_min_search_list_size >
           config_.io_optimization.cell_adj_max_search_list_size ||
       config_.io_optimization.cell_adj_support_shortlist <
           config_.io_optimization.cell_adj_candidate_cells)) {
    throw ann_exception_t(
        "Cell-Adj correction requires forced Cell-tree PQ traversal and a support shortlist "
        "covering its positive candidate budget");
  }
  if (config_.io_optimization.cell_pq_refine_candidates != 0 &&
      !config_.io_optimization.enable_cell_pq_traversal) {
    throw ann_exception_t("Cell-PQ refinement requires Cell-PQ traversal");
  }
  if (config_.io_optimization.cell_pq_refine_min_l == 0 ||
      (config_.io_optimization.cell_pq_refine_candidates == 0 &&
       config_.io_optimization.cell_pq_refine_min_l != 1)) {
    throw ann_exception_t("Cell-PQ refinement minimum L requires enabled candidate refinement");
  }
  if (config_.io_optimization.cell_pq_refine_prefetch_hop != 0 &&
      config_.io_optimization.cell_pq_refine_candidates == 0) {
    throw ann_exception_t("Cell-PQ refinement prefetch requires refinement candidates");
  }
  if (config_.io_optimization.cell_pq_refine_prefetch_hop != 0 &&
      config_.io_optimization.enable_io_uring) {
    throw ann_exception_t("Cell-PQ refinement prefetch is the isolated libaio path");
  }
  if (config_.io_optimization.enable_l_aware_refine_budget &&
      (!config_.io_optimization.enable_cell_pq_traversal ||
       config_.io_optimization.cell_pq_refine_candidates == 0 ||
       config_.io_optimization.l_aware_refine_base == 0 ||
       config_.io_optimization.l_aware_refine_base >
           config_.io_optimization.cell_pq_refine_candidates ||
       config_.io_optimization.l_aware_refine_divisor == 0)) {
    throw ann_exception_t(
        "L-aware refinement requires Cell-PQ traversal and a valid bounded budget");
  }
  if (config_.io_optimization.enable_cell_leaf_refinement &&
      (!config_.io_optimization.enable_forced_cell_tree_search ||
       config_.io_optimization.cell_pq_refine_prefetch_hop != 0)) {
    throw ann_exception_t(
        "Cell-leaf refinement requires forced-tree search without refinement prefetch");
  }
  if (config_.community_polar_search.cell_terminal_stitch_promote_cells &&
      (!config_.io_optimization.enable_cell_leaf_refinement ||
       config_.io_optimization.cell_pq_refine_candidates == 0)) {
    throw ann_exception_t(
        "Terminal graph stitching promotion requires leaf refinement and graph candidates");
  }
  if (config_.io_optimization.enable_resident_u8_refinement &&
      (!config_.io_optimization.enable_forced_cell_tree_search ||
       !config_.io_optimization.enable_compressed_pq_traversal ||
       config_.io_optimization.cell_pq_refine_candidates == 0 ||
       config_.io_optimization.cell_pq_refine_prefetch_hop != 0 ||
       config_.io_optimization.enable_cell_leaf_refinement)) {
    throw ann_exception_t(
        "Resident uint8 refinement requires isolated compressed forced-tree top-N refinement");
  }
  if (config_.io_optimization.enable_resident_u8_traversal &&
      !config_.io_optimization.enable_resident_u8_refinement) {
    throw ann_exception_t(
        "Resident uint8 traversal requires the isolated resident refinement artifact");
  }
  if ((config_.io_optimization.enable_cell_pq_dense_visited ||
       config_.io_optimization.enable_cell_pq_filter_visited ||
       config_.io_optimization.enable_cell_pq_decoded_u8 ||
       config_.io_optimization.enable_l_aware_search) &&
      !config_.io_optimization.enable_cell_pq_traversal) {
    throw ann_exception_t("L-aware PQ options require resident PQ traversal");
  }
  if (config_.io_optimization.enable_cell_pq_filter_visited &&
      !config_.io_optimization.enable_cell_pq_dense_visited) {
    throw ann_exception_t("Cell-PQ visited filtering requires the dense visited bitmap");
  }
  if (config_.io_optimization.enable_cell_pq_decoded_u8 &&
      config_.community_polar_search.is_hybrid() &&
      !config_.io_optimization.enable_forced_cell_tree_search) {
    throw ann_exception_t("Decoded PQ traversal is an isolated non-Hybrid low-L mode");
  }
  if (config_.io_optimization.enable_cell_pq_decoded_u8 &&
      config_.io_optimization.memgraph_mode != io_memgraph_mode_t::NONE) {
    throw ann_exception_t("Decoded PQ traversal and MemGraph are isolated stages");
  }
  if (config_.io_optimization.enable_cell_pq_decoded_u8 &&
      (!config_.io_optimization.enable_cell_pq_dense_visited ||
       !config_.io_optimization.enable_cell_pq_filter_visited)) {
    throw ann_exception_t("Decoded PQ traversal requires dense visited-neighbor filtering");
  }
  if (!std::isfinite(config_.io_optimization.cell_pq_decoded_scale) ||
      config_.io_optimization.cell_pq_decoded_scale <= 0.0F ||
      (config_.io_optimization.cell_pq_decoded_scale != 1.0F &&
       (!config_.io_optimization.enable_forced_cell_tree_search ||
        config_.io_optimization.enable_compressed_pq_traversal))) {
    throw ann_exception_t("Decoded PQ scale requires decoded forced Cell-tree traversal");
  }
  if (config_.io_optimization.enable_compressed_pq_traversal &&
      (!config_.io_optimization.enable_cell_pq_traversal ||
       (!config_.io_optimization.enable_l_aware_search &&
        !config_.io_optimization.enable_forced_cell_tree_search))) {
    throw ann_exception_t("Compressed PQ traversal requires an L-aware or forced Cell-tree search");
  }
  if (config_.io_optimization.enable_compressed_pq_traversal &&
      config_.io_optimization.enable_cell_pq_decoded_u8) {
    throw ann_exception_t("Compressed and decoded PQ traversal are mutually exclusive");
  }
  if (config_.io_optimization.pq_score_chunks != 0 &&
      (!config_.io_optimization.enable_compressed_pq_traversal ||
       !config_.io_optimization.enable_forced_cell_tree_search)) {
    throw ann_exception_t("PQ chunk pruning requires compressed forced Cell-tree traversal");
  }
  if (config_.io_optimization.pq_score_quantization_bits != 0 &&
      (config_.io_optimization.pq_score_quantization_bits != 8 &&
       config_.io_optimization.pq_score_quantization_bits != 16)) {
    throw ann_exception_t("PQ score quantization supports only 8-bit or 16-bit tables");
  }
  if (config_.io_optimization.pq_score_quantization_bits != 0 &&
      (!config_.io_optimization.enable_compressed_pq_traversal ||
       !config_.io_optimization.enable_forced_cell_tree_search)) {
    throw ann_exception_t("PQ score quantization requires compressed forced Cell-tree traversal");
  }
  if (config_.io_optimization.pq_score_chunks != 0 &&
      config_.io_optimization.pq_score_quantization_bits != 0) {
    throw ann_exception_t("PQ chunk pruning and PQ score quantization are isolated policies");
  }
  if (config_.io_optimization.graph_neighbor_score_limit != 0 &&
      (!config_.io_optimization.enable_compressed_pq_traversal ||
       !config_.io_optimization.enable_forced_cell_tree_search)) {
    throw ann_exception_t(
        "Graph-neighbor scoring limit requires compressed forced Cell-tree traversal");
  }
  if (config_.io_optimization.enable_l_aware_search &&
      (config_.io_optimization.l_aware_decoded_max_l == 0 ||
       config_.io_optimization.memgraph_mode != io_memgraph_mode_t::NONE)) {
    throw ann_exception_t("L-aware search requires a positive L threshold and excludes MemGraph");
  }
  if (config_.community_polar_search.cell_terminal_l_aware_approach &&
      !config_.io_optimization.enable_forced_cell_tree_search) {
    throw ann_exception_t(
        "L-aware terminal approach is an isolated forced Cell-tree search policy");
  }
  if (config_.io_optimization.enable_forced_cell_tree_search &&
      (config_.io_optimization.enable_l_aware_search || !config_.enable_powerann ||
       !config_.community_polar_search.is_hybrid() ||
       config_.community_polar_search.gateway_landing_cell_handoff ||
       !config_.community_polar_search.cell_expansion ||
       !config_.community_polar_search.cell_terminal_convergence ||
       !config_.community_polar_search.cell_hierarchical_routing ||
       !config_.community_polar_search.cell_hierarchy_require_persisted ||
       !config_.io_optimization.enable_cell_pq_traversal ||
       config_.io_optimization.memgraph_mode != io_memgraph_mode_t::NONE)) {
    throw ann_exception_t(
        "Forced Cell-tree search requires a graph/optional-Community approach and Hybrid terminal "
        "traversal of a persisted hierarchy");
  }
  if (config_.io_optimization.enable_cell_pq_traversal &&
      config_.io_optimization.enable_online_cell_adaptation) {
    throw ann_exception_t("Cell-PQ traversal and online Cell adaptation are isolated stages");
  }
  if (config_.io_optimization.enable_online_cell_adaptation &&
      (!config_.enable_powerann || !io_enabled || !config_.io_optimization.enable_cell_layout ||
       !config_.io_optimization.enable_lru_cache ||
       config_.io_optimization.enable_cell_page_batch || config_.io_optimization.enable_io_uring ||
       config_.io_optimization.enable_node_prefetch ||
       config_.io_optimization.enable_cell_prefetch ||
       config_.io_optimization.enable_gateway_prefetch)) {
    throw ann_exception_t(
        "Online Cell adaptation requires isolated Cell-4K plus LRU/libaio execution");
  }
  if (config_.io_optimization.enable_chunk_layout &&
      (!io_enabled || config_.io_optimization.enable_cell_layout ||
       config_.io_optimization.enable_weighted_reorder)) {
    throw ann_exception_t(
        "Chunk vector layout requires topology/vector search and excludes fixed Cell layout");
  }
  if (config_.io_optimization.enable_weighted_reorder &&
      (!io_enabled || config_.io_optimization.enable_cell_layout)) {
    throw ann_exception_t(
        "Weighted reorder requires topology/vector search and excludes plain Cell layout");
  }
  if (config_.io_optimization.enable_lru_cache &&
      (!io_enabled || config_.io_optimization.lru_cache_pages == 0)) {
    throw ann_exception_t("IO LRU requires topology/vector search and a positive page capacity");
  }
  if (!config_.io_optimization.enable_lru_cache && config_.io_optimization.lru_cache_pages != 0) {
    throw ann_exception_t("IO LRU page capacity requires the explicit cache switch");
  }
  if (config_.io_optimization.reset_lru_after_warmup && !config_.io_optimization.enable_lru_cache) {
    throw ann_exception_t("IO LRU warmup reset requires the explicit cache switch");
  }
  if (config_.io_optimization.enable_io_uring && !io_enabled) {
    throw ann_exception_t("io_uring requires topology/vector search");
  }
  if (config_.io_optimization.enable_io_uring &&
      config_.io_optimization.io_uring_queue_depth == 0) {
    throw ann_exception_t("io_uring queue depth must be positive");
  }
  if ((config_.io_optimization.enable_node_prefetch ||
       config_.io_optimization.enable_cell_prefetch ||
       config_.io_optimization.enable_gateway_prefetch) &&
      (!config_.io_optimization.enable_io_uring || !config_.io_optimization.enable_lru_cache)) {
    throw ann_exception_t("IO prefetch requires independent io_uring and LRU switches");
  }
  if (config_.io_optimization.enable_cell_prefetch && !config_.io_optimization.enable_cell_layout &&
      !config_.io_optimization.enable_chunk_layout &&
      !config_.io_optimization.enable_weighted_reorder) {
    throw ann_exception_t("Cell prefetch requires the Cell-contiguous layout");
  }
  if (config_.io_optimization.enable_gateway_prefetch &&
      (!config_.enable_powerann || !config_.community_polar_search.community_skip ||
       !config_.community_polar_search.cell_expansion ||
       !config_.community_polar_search.gateway_landing_cell_handoff ||
       (!config_.io_optimization.enable_cell_layout &&
        !config_.io_optimization.enable_chunk_layout &&
        !config_.io_optimization.enable_weighted_reorder))) {
    throw ann_exception_t(
        "Gateway prefetch requires Hybrid Community+Cell search and a Cell-contiguous layout");
  }
  if (config_.io_optimization.enable_gateway_prefetch &&
      (config_.io_optimization.gateway_prefetch_max_outstanding == 0 ||
       config_.io_optimization.gateway_prefetch_max_outstanding > 2)) {
    throw ann_exception_t("Gateway prefetch outstanding limit must be one or two");
  }
  if (config_.io_optimization.memgraph_mode != io_memgraph_mode_t::NONE &&
      (!io_enabled || config_.io_optimization.memgraph_nodes == 0 ||
       config_.io_optimization.memgraph_entry_candidates == 0 ||
       config_.io_optimization.memgraph_entry_candidates >
           config_.io_optimization.memgraph_nodes)) {
    throw ann_exception_t(
        "MemGraph requires topology/vector search and 0 < entries <= node budget");
  }
  if (config_.io_optimization.memgraph_mode == io_memgraph_mode_t::NONE &&
      config_.io_optimization.memgraph_nodes != 0) {
    throw ann_exception_t("MemGraph node budget requires an explicit MemGraph mode");
  }
  if (config_.io_optimization.enable_dynamic_width &&
      (!config_.io_optimization.enable_io_uring ||
       config_.io_optimization.dynamic_width_initial == 0 ||
       config_.io_optimization.dynamic_width_waste_threshold < 0.0F ||
       config_.io_optimization.dynamic_width_waste_threshold > 1.0F)) {
    throw ann_exception_t(
        "Dynamic Width requires io_uring, positive initial width, and threshold in [0,1]");
  }
  try {
    validate_community_polar_search_config(config_.community_polar_search);
  } catch (const std::invalid_argument& error) {
    throw ann_exception_t(error.what());
  }
  if (!config_.enable_powerann && config_.community_polar_search.is_hybrid()) {
    throw ann_exception_t("Hybrid search requires enable_powerann=true");
  }
  if (!config_.enable_powerann && io_enabled) {
    throw ann_exception_t("IO-optimized search requires enable_powerann=true");
  }
  if (!io_enabled && (!config.io_topology_path.empty() || !config.io_vector_path.empty())) {
    throw ann_exception_t("IO artifact path overrides require IO-optimized search");
  }
  if (!config_.enable_powerann || !config_.community_polar_search.is_hybrid()) {
    if (!config.community_polar_path.empty()) {
      throw ann_exception_t("Community-Polar path override requires enabled hybrid search");
    }
    if (!config.cell_oracle_output_path.empty() || !config.cell_value_output_path.empty() ||
        !config.gateway_value_output_path.empty()) {
      throw ann_exception_t("Community-Polar telemetry requires enabled hybrid search");
    }
    if (curve && io_enabled) {
      throw ann_exception_t("multi-L non-Hybrid IO search is unsupported");
    }
    if (curve) {
      if (thread_curve) {
        return search.search_thread_curve(config, thread_counts);
      }
      return search.search_curve(config, search_list_sizes);
    }
    return {io_enabled ? search.search(config, config_.io_optimization) : search.search(config)};
  }
  if ((!config.cell_oracle_output_path.empty() || !config.cell_value_output_path.empty()) &&
      !config_.community_polar_search.cell_expansion) {
    throw ann_exception_t("Cell telemetry requires Cell Expansion");
  }
  if (!config.gateway_value_output_path.empty() && !config_.community_polar_search.community_skip) {
    throw ann_exception_t("Gateway telemetry requires Community Skip");
  }
  if (!config.cell_oracle_output_path.empty() || !config.gateway_value_output_path.empty()) {
    if (curve) {
      throw ann_exception_t("multi-L observe-only search is unsupported");
    }
    return {search.search_observe_only(config, config_.community_polar_search)};
  }
  if (curve && !io_enabled) {
    throw ann_exception_t("multi-L Hybrid search requires topology/vector mode");
  }
  if (curve) {
    if (thread_curve) {
      return search.search_thread_curve(config, config_.community_polar_search,
                                        config_.io_optimization, thread_counts);
    }
    if (cache_curve) {
      return search.search_cache_curve(config, config_.community_polar_search,
                                       config_.io_optimization, cache_page_counts);
    }
    return search.search_curve(config, config_.community_polar_search, config_.io_optimization,
                               search_list_sizes);
  }
  return {io_enabled
              ? search.search(config, config_.community_polar_search, config_.io_optimization)
              : search.search(config, config_.community_polar_search)};
}

diskann_search_result_t power_ann_t::search_diskann_index(const diskann_search_config_t& config) {
  const std::array<uint32_t, 1> search_list_sizes{config.search_list_size};
  auto results = search_diskann_index_impl(config, search_list_sizes, {});
  return std::move(results.front());
}

std::vector<diskann_search_result_t>
power_ann_t::search_diskann_index_curve(const diskann_search_config_t& config,
                                        std::span<const uint32_t> search_list_sizes) {
  return search_diskann_index_impl(config, search_list_sizes, {});
}

std::vector<diskann_search_result_t>
power_ann_t::search_diskann_index_thread_curve(const diskann_search_config_t& config,
                                               std::span<const uint32_t> thread_counts) {
  const std::array<uint32_t, 1> search_list_sizes{config.search_list_size};
  return search_diskann_index_impl(config, search_list_sizes, thread_counts);
}

std::vector<diskann_search_result_t>
power_ann_t::search_diskann_index_cache_curve(const diskann_search_config_t& config,
                                              std::span<const uint64_t> cache_page_counts) {
  const std::array<uint32_t, 1> search_list_sizes{config.search_list_size};
  return search_diskann_index_impl(config, search_list_sizes, {}, cache_page_counts);
}

} // namespace powerlaw_ann
