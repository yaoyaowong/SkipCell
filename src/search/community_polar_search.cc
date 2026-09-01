#include "search/community_polar_search.h"

#include "common/ann_error.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace powerlaw_ann {
namespace {

constexpr uint32_t k_pole_sector_id = std::numeric_limits<uint32_t>::max();
constexpr uint64_t k_fnv_offset_basis = 14695981039346656037ULL;
constexpr uint64_t k_fnv_prime = 1099511628211ULL;

uint64_t file_checksum(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to fingerprint Community-Polar sidecar: " + path.string());
  }
  uint64_t checksum = k_fnv_offset_basis;
  std::array<char, 1U << 20U> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    const auto count = input.gcount();
    for (std::streamsize offset = 0; offset < count; ++offset) {
      checksum ^= static_cast<uint8_t>(buffer[static_cast<size_t>(offset)]);
      checksum *= k_fnv_prime;
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while fingerprinting Community-Polar sidecar: " +
                             path.string());
  }
  return checksum;
}

float squared_l2(std::span<const float> left, std::span<const float> right) {
  double total = 0.0;
  for (size_t dimension = 0; dimension < left.size(); ++dimension) {
    const double delta = static_cast<double>(left[dimension]) - right[dimension];
    total += delta * delta;
  }
  return static_cast<float>(total);
}

float squared_l2_fast(std::span<const float> left, std::span<const float> right) noexcept {
  size_t dimension = 0;
  float total = 0.0F;
#if defined(__AVX2__)
  __m256 vector_total = _mm256_setzero_ps();
  for (; dimension + 8U <= left.size(); dimension += 8U) {
    const __m256 lhs = _mm256_loadu_ps(left.data() + dimension);
    const __m256 rhs = _mm256_loadu_ps(right.data() + dimension);
    const __m256 delta = _mm256_sub_ps(lhs, rhs);
#if defined(__FMA__)
    vector_total = _mm256_fmadd_ps(delta, delta, vector_total);
#else
    vector_total = _mm256_add_ps(vector_total, _mm256_mul_ps(delta, delta));
#endif
  }
  alignas(32) std::array<float, 8> lanes{};
  _mm256_store_ps(lanes.data(), vector_total);
  for (const float lane : lanes) {
    total += lane;
  }
#endif
  for (; dimension < left.size(); ++dimension) {
    const float delta = left[dimension] - right[dimension];
    total += delta * delta;
  }
  return total;
}

uint64_t vector_bytes(size_t count, size_t element_size) {
  if (count > std::numeric_limits<uint64_t>::max() / element_size) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(count) * element_size;
}

void add_top_hit(std::array<uint32_t, 4>& top_hits, uint32_t value) {
  for (size_t position = 0; position < top_hits.size(); ++position) {
    if (value <= top_hits[position]) {
      continue;
    }
    for (size_t shift = top_hits.size() - 1; shift > position; --shift) {
      top_hits[shift] = top_hits[shift - 1];
    }
    top_hits[position] = value;
    return;
  }
}

uint32_t top_hit_sum(const std::array<uint32_t, 4>& top_hits, size_t count) {
  uint32_t total = 0;
  for (size_t position = 0; position < std::min(count, top_hits.size()); ++position) {
    total += top_hits[position];
  }
  return total;
}

} // namespace

void validate_community_polar_search_config(const community_polar_search_config_t& config) {
  if (!std::isfinite(config.graph_guard_fraction) || config.graph_guard_fraction < 0.0F ||
      config.graph_guard_fraction > 1.0F) {
    throw std::invalid_argument("Graph guard fraction must be finite and within [0, 1]");
  }
  if (!std::isfinite(config.cell_min_competitive_fraction) ||
      config.cell_min_competitive_fraction < 0.0F || config.cell_min_competitive_fraction > 1.0F) {
    throw std::invalid_argument("Cell minimum competitive fraction must be finite and in [0, 1]");
  }
  if (!std::isfinite(config.cell_candidate_threshold_ratio) ||
      config.cell_candidate_threshold_ratio <= 0.0F ||
      config.cell_candidate_threshold_ratio > 1.0F) {
    throw std::invalid_argument("Cell candidate threshold ratio must be finite and in (0, 1]");
  }
  if (!std::isfinite(config.cell_evidence_threshold_ratio) ||
      config.cell_evidence_threshold_ratio <= 0.0F || config.cell_evidence_threshold_ratio > 1.0F) {
    throw std::invalid_argument("Cell evidence threshold ratio must be finite and in (0, 1]");
  }
  if (!std::isfinite(config.cell_insert_fraction) || config.cell_insert_fraction <= 0.0F ||
      config.cell_insert_fraction > 1.0F) {
    throw std::invalid_argument("Cell insert fraction must be finite and in (0, 1]");
  }
  if (!config.is_hybrid()) {
    if (config.community_skip || config.community_hub_clique || config.cell_expansion) {
      throw std::invalid_argument("Community and Cell operators require hybrid mode");
    }
    return;
  }
  if (!config.community_skip && !config.cell_expansion) {
    throw std::invalid_argument("Hybrid mode requires Community Skip or Cell Expansion");
  }
  if (config.community_skip &&
      (config.community_expansion_budget == 0 || config.gateway_score_budget == 0 ||
       config.gateway_score_batch_cap == 0)) {
    throw std::invalid_argument("Community Skip requires positive Community and Gateway budgets");
  }
  if (config.community_hub_clique && !config.community_skip) {
    throw std::invalid_argument("Community Hub Clique requires Community Skip");
  }
  if (config.community_pq_approach_hops != 0 && !config.community_skip) {
    throw std::invalid_argument("Community PQ approach hops require Community Skip");
  }
  if (config.cell_expansion &&
      (config.cell_block_budget == 0 || config.cell_support == 0 || config.cell_insert_cap == 0 ||
       config.cell_min_competitive_count == 0)) {
    throw std::invalid_argument("Cell Expansion requires positive Block, support, and insert caps");
  }
  if (config.cell_centroid_routing &&
      (!config.cell_expansion || !config.cell_whole_cell_scan ||
       config.cell_routing_community_budget == 0 || config.cell_routing_cell_budget == 0 ||
       config.cell_routing_cell_budget > config.cell_block_budget)) {
    throw std::invalid_argument(
        "Cell centroid routing requires whole-Cell expansion and compatible positive budgets");
  }
  if (config.gateway_landing_cell_handoff &&
      (!config.community_skip || !config.cell_expansion || !config.cell_whole_cell_scan ||
       config.cell_routing_cell_budget == 0 ||
       config.cell_routing_cell_budget > config.cell_block_budget)) {
    throw std::invalid_argument(
        "Gateway Landing-Cell handoff requires Community Skip and compatible whole-Cell budgets");
  }
  if (config.cell_terminal_convergence &&
      (!config.cell_expansion || !config.cell_whole_cell_scan || !config.cell_centroid_routing)) {
    throw std::invalid_argument(
        "Terminal Cell convergence requires centroid-routed whole-Cell expansion");
  }
  if (config.cell_terminal_l_aware_approach &&
      (!config.cell_terminal_convergence || config.cell_terminal_approach_hops == 0 ||
       config.cell_terminal_approach_l_divisor == 0 ||
       config.cell_terminal_approach_base_hops > config.cell_terminal_approach_hops)) {
    throw std::invalid_argument(
        "L-aware terminal approach requires terminal convergence, a positive hop cap/divisor, "
        "and base hops no larger than the cap");
  }
  if ((config.cell_terminal_low_l_stitch_max_l == 0) !=
      (config.cell_terminal_low_l_stitch_extra_hops == 0)) {
    throw std::invalid_argument(
        "Low-L terminal stitching requires both a positive L threshold and extra-hop budget");
  }
  if (config.cell_terminal_l_aware_forest &&
      (!config.cell_terminal_convergence || config.cell_terminal_forest_base_cells == 0 ||
       config.cell_terminal_forest_l_divisor == 0 ||
       config.cell_terminal_forest_base_cells > config.cell_routing_cell_budget)) {
    throw std::invalid_argument(
        "L-aware terminal Cell forest requires terminal convergence, positive base/divisor, "
        "and base Cells no larger than the routed Cell cap");
  }
  if ((config.cell_terminal_low_l_forest_max_l == 0) !=
          (config.cell_terminal_low_l_forest_extra_cells == 0) ||
      (config.cell_terminal_low_l_forest_extra_cells != 0 &&
       !config.cell_terminal_l_aware_forest)) {
    throw std::invalid_argument(
        "Low-L terminal forest expansion requires an L-aware forest, threshold, and extra Cells");
  }
  if ((config.cell_terminal_high_l_forest_min_l == 0) !=
          (config.cell_terminal_high_l_forest_cell_cap == 0) ||
      (config.cell_terminal_high_l_forest_cell_cap != 0 &&
       (!config.cell_terminal_l_aware_forest ||
        config.cell_terminal_high_l_forest_cell_cap > config.cell_routing_cell_budget)) ||
      (config.cell_terminal_high_l_forest_cap_l_divisor != 0 &&
       config.cell_terminal_high_l_forest_min_l == 0)) {
    throw std::invalid_argument(
        "High-L terminal forest cap requires an L-aware forest, threshold, and valid Cell cap");
  }
  if (config.cell_terminal_l_aware_refine &&
      (!config.cell_terminal_convergence || config.cell_terminal_refine_l_divisor == 0 ||
       config.cell_terminal_refine_base_hops > config.cell_terminal_refine_hops)) {
    throw std::invalid_argument(
        "L-aware terminal refinement requires terminal convergence, a positive divisor, and "
        "base hops no larger than the refinement cap");
  }
  if (config.cell_terminal_stitch_promote_cells &&
      (!config.cell_terminal_convergence || config.cell_terminal_stitch_cell_budget == 0 ||
       config.cell_terminal_stitch_min_support == 0)) {
    throw std::invalid_argument(
        "Terminal graph stitching promotion requires terminal convergence and positive Cell "
        "budget/support");
  }
  if (!std::isfinite(config.cell_terminal_distance_scale) ||
      config.cell_terminal_distance_scale <= 0.0F) {
    throw std::invalid_argument("Terminal Cell distance scale must be finite and positive");
  }
  if (!config.cell_terminal_route_path.empty() && !config.cell_terminal_convergence) {
    throw std::invalid_argument("Terminal Cell route profile requires terminal convergence");
  }
  if (config.cell_terminal_global_forest_routing &&
      (!config.cell_terminal_convergence || !config.cell_hierarchical_routing ||
       !config.cell_defer_hierarchy_routing)) {
    throw std::invalid_argument(
        "Global terminal forest routing requires deferred hierarchical terminal convergence");
  }
  if (config.cell_hierarchical_routing &&
      (!config.cell_centroid_routing || config.cell_hierarchy_branching < 2 ||
       config.cell_hierarchy_leaf_size == 0 || config.cell_hierarchy_probe_budget == 0)) {
    throw std::invalid_argument(
        "Hierarchical Cell routing requires centroid routing, branching >= 2, and positive "
        "leaf/probe budgets");
  }
  if (config.cell_hierarchical_routing) {
    const uint64_t minimum_probe_budget = (static_cast<uint64_t>(config.cell_routing_cell_budget) +
                                           config.cell_hierarchy_leaf_size - 1) /
                                          config.cell_hierarchy_leaf_size;
    if (config.cell_hierarchy_probe_budget < minimum_probe_budget) {
      throw std::invalid_argument(
          "Hierarchical Cell probe budget cannot cover the requested routed Cell budget");
    }
  }
  if (config.cell_hierarchy_require_persisted && !config.cell_hierarchical_routing) {
    throw std::invalid_argument(
        "Persisted Cell hierarchy requirement requires hierarchical routing");
  }
  if (config.cell_hierarchy_graph_guided_routing && !config.cell_hierarchical_routing) {
    throw std::invalid_argument("Graph-guided Cell routing requires hierarchical Cell routing");
  }
}

bool graph_guard_allows_macro(uint64_t search_list_size, float graph_guard_fraction,
                              uint64_t unexpanded_macro_candidates) {
  if (search_list_size == 0) {
    return false;
  }
  const uint64_t base_guard = static_cast<uint64_t>(
      std::ceil(static_cast<double>(graph_guard_fraction) * search_list_size));
  const uint64_t macro_capacity = search_list_size - std::min(search_list_size, base_guard);
  return unexpanded_macro_candidates < macro_capacity;
}

community_polar_candidate_origin_t
community_polar_query_state_t::origin(uint32_t node_id) const noexcept {
  const auto found = macro_origins.find(node_id);
  return found == macro_origins.end() ? community_polar_candidate_origin_t::BASE_EDGE
                                      : found->second;
}

uint32_t community_polar_query_state_t::cell_block_ordinal(uint32_t node_id) const noexcept {
  const auto found = macro_block_ordinals.find(node_id);
  return found == macro_block_ordinals.end() ? std::numeric_limits<uint32_t>::max() : found->second;
}

void community_polar_query_state_t::mark_macro_inserted(
    uint32_t node_id, community_polar_candidate_origin_t candidate_origin, uint32_t block_ordinal) {
  macro_origins.insert_or_assign(node_id, candidate_origin);
  if (cell_value_telemetry_enabled &&
      candidate_origin == community_polar_candidate_origin_t::POLAR_CELL &&
      block_ordinal < k_cell_value_max_blocks) {
    macro_block_ordinals.insert_or_assign(node_id, block_ordinal);
  } else if (cell_value_telemetry_enabled) {
    macro_block_ordinals.erase(node_id);
  }
}

void community_polar_query_state_t::mark_base_reached(uint32_t node_id) {
  macro_origins.erase(node_id);
  if (cell_value_telemetry_enabled) {
    macro_block_ordinals.erase(node_id);
  }
}

float community_polar_search_t::query_distance_squared(
    std::span<const float> left, std::span<const float> right) const noexcept {
  return config_.fast_query_distance ? squared_l2_fast(left, right) : squared_l2(left, right);
}

uint32_t
community_polar_search_t::terminal_approach_hops(uint64_t search_list_size) const noexcept {
  if (!config_.cell_terminal_l_aware_approach) {
    return config_.cell_terminal_approach_hops;
  }
  const uint64_t scaled_hops =
      search_list_size == 0 ? 0
                            : 1 + (search_list_size - 1) / config_.cell_terminal_approach_l_divisor;
  const uint32_t remaining_hops =
      config_.cell_terminal_approach_hops - config_.cell_terminal_approach_base_hops;
  const uint32_t base_hops =
      scaled_hops >= remaining_hops
          ? config_.cell_terminal_approach_hops
          : config_.cell_terminal_approach_base_hops + static_cast<uint32_t>(scaled_hops);
  const uint32_t extra_hops =
      config_.cell_terminal_low_l_stitch_max_l != 0 &&
              search_list_size <= config_.cell_terminal_low_l_stitch_max_l
          ? config_.cell_terminal_low_l_stitch_extra_hops
          : 0U;
  return std::min<uint32_t>(config_.cell_terminal_approach_hops, base_hops + extra_hops);
}

uint32_t community_polar_search_t::terminal_cell_budget(uint64_t search_list_size) const noexcept {
  if (!config_.cell_terminal_l_aware_forest) {
    return config_.cell_routing_cell_budget;
  }
  const uint64_t scaled_cells =
      search_list_size == 0
          ? 0
          : 1 + (search_list_size - 1) / config_.cell_terminal_forest_l_divisor;
  uint64_t budget =
      std::max<uint64_t>(config_.cell_terminal_forest_base_cells, scaled_cells);
  if (config_.cell_terminal_low_l_forest_max_l != 0 &&
      search_list_size <= config_.cell_terminal_low_l_forest_max_l) {
    budget += config_.cell_terminal_low_l_forest_extra_cells;
  }
  if (config_.cell_terminal_high_l_forest_min_l != 0 &&
      search_list_size >= config_.cell_terminal_high_l_forest_min_l) {
    uint64_t high_l_cap = config_.cell_terminal_high_l_forest_cell_cap;
    if (config_.cell_terminal_high_l_forest_cap_l_divisor != 0) {
      const uint64_t reduction =
          (search_list_size - config_.cell_terminal_high_l_forest_min_l) /
          config_.cell_terminal_high_l_forest_cap_l_divisor;
      high_l_cap = reduction >= high_l_cap ? 1 : high_l_cap - reduction;
    }
    budget = std::min<uint64_t>(budget, high_l_cap);
  }
  return static_cast<uint32_t>(
      std::min<uint64_t>(config_.cell_routing_cell_budget, budget));
}

uint32_t community_polar_search_t::terminal_refine_hops(uint64_t search_list_size) const noexcept {
  if (!config_.cell_terminal_l_aware_refine) {
    return config_.cell_terminal_refine_hops;
  }
  const uint64_t scaled_hops = search_list_size / config_.cell_terminal_refine_l_divisor;
  const uint64_t extra_hops =
      config_.cell_terminal_low_l_stitch_max_l != 0 &&
              search_list_size <= config_.cell_terminal_low_l_stitch_max_l
          ? config_.cell_terminal_low_l_stitch_extra_hops
          : 0U;
  return static_cast<uint32_t>(std::min<uint64_t>(
      config_.cell_terminal_refine_hops,
      static_cast<uint64_t>(config_.cell_terminal_refine_base_hops) + scaled_hops + extra_hops));
}

void community_polar_query_state_t::reset_cell_scratch(size_t cell_count, size_t block_count) {
  if (cell_candidate_created.size() != cell_count || touched_cell_flags.size() != cell_count) {
    cell_candidate_created.assign(cell_count, 0);
    cell_support_counts.assign(cell_count, 0);
    cell_best_graph_distance.assign(cell_count, std::numeric_limits<float>::infinity());
    cell_routing_ranks.assign(cell_count, std::numeric_limits<uint32_t>::max());
    touched_cell_flags.assign(cell_count, 0);
  } else {
    for (const uint32_t cell : touched_cells) {
      cell_candidate_created[cell] = 0;
      cell_support_counts[cell] = 0;
      cell_best_graph_distance[cell] = std::numeric_limits<float>::infinity();
      cell_routing_ranks[cell] = std::numeric_limits<uint32_t>::max();
      touched_cell_flags[cell] = 0;
    }
  }
  if (block_support_counts.size() != block_count || touched_block_flags.size() != block_count) {
    block_support_counts.assign(block_count, 0);
    block_best_graph_distance.assign(block_count, std::numeric_limits<float>::infinity());
    loaded_blocks.assign(block_count, 0);
    touched_block_flags.assign(block_count, 0);
  } else {
    for (const uint32_t block : touched_blocks) {
      block_support_counts[block] = 0;
      block_best_graph_distance[block] = std::numeric_limits<float>::infinity();
      loaded_blocks[block] = 0;
      touched_block_flags[block] = 0;
    }
  }
  touched_cells.clear();
  touched_blocks.clear();
}

void community_polar_query_state_t::touch_cell(uint32_t cell_id) {
  if (touched_cell_flags[cell_id] == 0) {
    touched_cell_flags[cell_id] = 1;
    touched_cells.push_back(cell_id);
  }
}

void community_polar_query_state_t::touch_block(uint32_t block_id) {
  if (touched_block_flags[block_id] == 0) {
    touched_block_flags[block_id] = 1;
    touched_blocks.push_back(block_id);
  }
}

community_polar_search_t::community_polar_search_t(std::shared_ptr<community_polar_index_t> index,
                                                   const community_polar_search_config_t& config,
                                                   uint64_t artifact_bytes,
                                                   uint64_t artifact_checksum)
    : index_(std::move(index)), config_(config), artifact_bytes_(artifact_bytes),
      artifact_checksum_(artifact_checksum) {
  validate_community_polar_search_config(config_);
  if (index_ == nullptr || index_->point_count == 0 || index_->dimension == 0 ||
      index_->pq_code_width == 0) {
    throw std::invalid_argument("Community-Polar search requires a nonempty validated index");
  }
  if ((config_.cell_centroid_routing || config_.gateway_landing_cell_handoff) &&
      (index_->config.cell_partition == community_cell_partition_t::POLAR_RADIAL ||
       index_->direction_axes.size() != index_->cells.size() * index_->dimension)) {
    throw std::invalid_argument("Cell routing requires graph-local Cell centroids");
  }
  if (config_.cell_hierarchical_routing) {
    if (config_.cell_hierarchy_require_persisted && index_->cell_hierarchy_nodes.empty()) {
      throw std::invalid_argument("Hierarchical Cell routing requires a persisted hierarchy");
    }
    build_cell_hierarchy();
    const auto& roots = index_->cell_hierarchy_nodes.empty() ? community_hierarchy_roots_
                                                             : index_->cell_hierarchy_roots;
    for (uint32_t community = 0; community < roots.size(); ++community) {
      if (roots[community] != std::numeric_limits<uint32_t>::max()) {
        hierarchy_root_communities_.push_back(community);
      }
    }
    const auto& nodes =
        index_->cell_hierarchy_nodes.empty() ? cell_hierarchy_nodes_ : index_->cell_hierarchy_nodes;
    const auto& children = index_->cell_hierarchy_nodes.empty() ? cell_hierarchy_children_
                                                                : index_->cell_hierarchy_children;
    cell_hierarchy_parents_.assign(nodes.size(), std::numeric_limits<uint32_t>::max());
    cell_hierarchy_leaf_parents_.assign(index_->cells.size(), std::numeric_limits<uint32_t>::max());
    for (uint32_t parent = 0; parent < nodes.size(); ++parent) {
      const auto& node = nodes[parent];
      for (uint32_t offset = 0; offset < node.child_count; ++offset) {
        const uint32_t child = children[node.child_begin + offset];
        if (node.children_are_cells) {
          cell_hierarchy_leaf_parents_[child] = parent;
        } else {
          cell_hierarchy_parents_[child] = parent;
        }
      }
    }
  }
  if (!config_.cell_terminal_route_path.empty()) {
    std::ifstream input(config_.cell_terminal_route_path);
    uint32_t version = 0;
    uint32_t dimension = 0;
    if (!(input >> version >> dimension >> terminal_route_bucket_count_ >>
          terminal_route_cells_per_bucket_) ||
        version != 1 || dimension != index_->dimension || terminal_route_bucket_count_ == 0 ||
        terminal_route_cells_per_bucket_ == 0) {
      throw std::invalid_argument("Terminal Cell route profile header is invalid");
    }
    terminal_route_centroids_.resize(static_cast<size_t>(terminal_route_bucket_count_) * dimension);
    terminal_route_cells_.resize(static_cast<size_t>(terminal_route_bucket_count_) *
                                 terminal_route_cells_per_bucket_);
    for (uint32_t bucket = 0; bucket < terminal_route_bucket_count_; ++bucket) {
      for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
        if (!(input >>
              terminal_route_centroids_[static_cast<size_t>(bucket) * dimension + coordinate])) {
          throw std::invalid_argument("Terminal Cell route centroid payload is truncated");
        }
      }
      for (uint32_t rank = 0; rank < terminal_route_cells_per_bucket_; ++rank) {
        int64_t cell = -1;
        if (!(input >> cell) || cell < -1 || cell >= static_cast<int64_t>(index_->cells.size())) {
          throw std::invalid_argument("Terminal Cell route Cell payload is invalid");
        }
        terminal_route_cells_[static_cast<size_t>(bucket) * terminal_route_cells_per_bucket_ +
                              rank] =
            cell < 0 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(cell);
      }
    }
  }
  cell_to_sector_.assign(index_->cells.size(), std::numeric_limits<uint32_t>::max());
  for (uint32_t sector_id = 0; sector_id < index_->sectors.size(); ++sector_id) {
    const auto& sector = index_->sectors[sector_id];
    for (uint64_t cell = sector.cell_begin; cell < sector.cell_begin + sector.cell_count; ++cell) {
      if (cell >= cell_to_sector_.size()) {
        throw std::invalid_argument("Community-Polar Sector Cell range is invalid");
      }
      cell_to_sector_[cell] = sector_id;
    }
  }
  if (std::find(cell_to_sector_.begin(), cell_to_sector_.end(),
                std::numeric_limits<uint32_t>::max()) != cell_to_sector_.end()) {
    throw std::invalid_argument("Community-Polar Cell has no owning Sector");
  }

  node_to_block_.assign(index_->point_count, std::numeric_limits<uint32_t>::max());
  packet_node_ids_.assign(index_->point_count, std::numeric_limits<uint32_t>::max());
  const uint64_t record_bytes = sizeof(uint32_t) + index_->pq_code_width;
  for (uint32_t block_id = 0; block_id < index_->blocks.size(); ++block_id) {
    const auto& block = index_->blocks[block_id];
    for (uint32_t offset = 0; offset < block.node_count; ++offset) {
      const uint64_t packet = block.packet_record_begin + offset;
      const size_t byte_offset = static_cast<size_t>(packet * record_bytes);
      uint32_t node_id = 0;
      std::memcpy(&node_id, index_->packet_payload.data() + byte_offset, sizeof(node_id));
      if (node_id >= node_to_block_.size() ||
          node_to_block_[node_id] != std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Community-Polar Block packet membership is invalid");
      }
      node_to_block_[node_id] = block_id;
      packet_node_ids_[packet] = node_id;
    }
  }
  if (std::find(node_to_block_.begin(), node_to_block_.end(),
                std::numeric_limits<uint32_t>::max()) != node_to_block_.end()) {
    throw std::invalid_argument("Community-Polar node has no owning Block");
  }
}

void community_polar_search_t::build_cell_hierarchy() {
  cell_hierarchy_nodes_.clear();
  cell_hierarchy_children_.clear();
  cell_hierarchy_centroids_.clear();
  community_hierarchy_roots_.assign(index_->communities.size(),
                                    std::numeric_limits<uint32_t>::max());
  if (!index_->cell_hierarchy_nodes.empty()) {
    return;
  }

  const uint32_t dimension = index_->dimension;
  const auto cell_centroid = [&](uint32_t cell) {
    return std::span<const float>(index_->direction_axes)
        .subspan(static_cast<size_t>(cell) * dimension, dimension);
  };
  const auto centroid_distance = [&](uint32_t cell, std::span<const float> centroid) {
    return squared_l2(cell_centroid(cell), centroid);
  };

  std::function<uint32_t(std::vector<uint32_t>)> append_node;
  append_node = [&](std::vector<uint32_t> cells) -> uint32_t {
    if (cells.empty()) {
      throw std::logic_error("Hierarchical Cell node cannot be empty");
    }
    std::sort(cells.begin(), cells.end());
    const uint32_t node_id = static_cast<uint32_t>(cell_hierarchy_nodes_.size());
    cell_hierarchy_nodes_.push_back({});
    cell_hierarchy_nodes_.back().node_id = node_id;
    const size_t centroid_begin = cell_hierarchy_centroids_.size();
    cell_hierarchy_centroids_.resize(centroid_begin + dimension, 0.0F);
    double total_weight = 0.0;
    for (const uint32_t cell : cells) {
      const double weight = static_cast<double>(index_->cells[cell].node_count);
      total_weight += weight;
      const auto centroid = cell_centroid(cell);
      for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
        cell_hierarchy_centroids_[centroid_begin + coordinate] +=
            static_cast<float>(weight * centroid[coordinate]);
      }
    }
    for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
      cell_hierarchy_centroids_[centroid_begin + coordinate] =
          static_cast<float>(cell_hierarchy_centroids_[centroid_begin + coordinate] / total_weight);
    }
    const auto completed_centroid =
        std::span<const float>(cell_hierarchy_centroids_).subspan(centroid_begin, dimension);
    float radius = 0.0F;
    for (const uint32_t cell : cells) {
      radius = std::max(radius, std::sqrt(centroid_distance(cell, completed_centroid)));
    }
    cell_hierarchy_nodes_[node_id].radius =
        std::nextafter(radius, std::numeric_limits<float>::infinity());

    if (cells.size() <= config_.cell_hierarchy_leaf_size) {
      auto& node = cell_hierarchy_nodes_[node_id];
      node.child_begin = static_cast<uint32_t>(cell_hierarchy_children_.size());
      node.child_count = static_cast<uint32_t>(cells.size());
      node.children_are_cells = true;
      cell_hierarchy_children_.insert(cell_hierarchy_children_.end(), cells.begin(), cells.end());
      return node_id;
    }

    const uint32_t cluster_count =
        static_cast<uint32_t>(std::min<size_t>(config_.cell_hierarchy_branching, cells.size()));
    if (config_.cell_hierarchy_preserve_leaf_order) {
      std::vector<uint32_t> child_nodes;
      child_nodes.reserve(cluster_count);
      size_t begin = 0;
      for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
        const size_t remaining = cells.size() - begin;
        const size_t remaining_clusters = cluster_count - cluster;
        const size_t count = (remaining + remaining_clusters - 1) / remaining_clusters;
        child_nodes.push_back(append_node(
            std::vector<uint32_t>(cells.begin() + static_cast<std::ptrdiff_t>(begin),
                                  cells.begin() + static_cast<std::ptrdiff_t>(begin + count))));
        begin += count;
      }
      auto& node = cell_hierarchy_nodes_[node_id];
      node.child_begin = static_cast<uint32_t>(cell_hierarchy_children_.size());
      node.child_count = static_cast<uint32_t>(child_nodes.size());
      node.children_are_cells = false;
      cell_hierarchy_children_.insert(cell_hierarchy_children_.end(), child_nodes.begin(),
                                      child_nodes.end());
      return node_id;
    }
    std::vector<uint32_t> seeds;
    seeds.reserve(cluster_count);
    const auto node_centroid =
        std::span<const float>(cell_hierarchy_centroids_).subspan(centroid_begin, dimension);
    uint32_t first_seed = cells.front();
    float first_distance = centroid_distance(first_seed, node_centroid);
    for (const uint32_t cell : cells) {
      const float distance = centroid_distance(cell, node_centroid);
      if (std::tie(distance, cell) < std::tie(first_distance, first_seed)) {
        first_seed = cell;
        first_distance = distance;
      }
    }
    seeds.push_back(first_seed);
    while (seeds.size() < cluster_count) {
      uint32_t next_seed = std::numeric_limits<uint32_t>::max();
      float farthest_distance = -1.0F;
      for (const uint32_t cell : cells) {
        if (std::find(seeds.begin(), seeds.end(), cell) != seeds.end()) {
          continue;
        }
        float nearest_distance = std::numeric_limits<float>::infinity();
        for (const uint32_t seed : seeds) {
          nearest_distance =
              std::min(nearest_distance, squared_l2(cell_centroid(cell), cell_centroid(seed)));
        }
        if (nearest_distance > farthest_distance ||
            (nearest_distance == farthest_distance && cell < next_seed)) {
          next_seed = cell;
          farthest_distance = nearest_distance;
        }
      }
      seeds.push_back(next_seed);
    }

    std::vector<float> cluster_centroids(static_cast<size_t>(cluster_count) * dimension);
    for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
      std::copy(cell_centroid(seeds[cluster]).begin(), cell_centroid(seeds[cluster]).end(),
                cluster_centroids.begin() + static_cast<size_t>(cluster) * dimension);
    }
    std::vector<std::vector<uint32_t>> clusters(cluster_count);
    for (uint32_t iteration = 0; iteration < 4; ++iteration) {
      for (auto& cluster : clusters) {
        cluster.clear();
      }
      std::vector<uint32_t> capacity(cluster_count,
                                     static_cast<uint32_t>(cells.size() / cluster_count));
      for (uint32_t cluster = 0; cluster < cells.size() % cluster_count; ++cluster) {
        ++capacity[cluster];
      }
      std::vector<uint8_t> seeded(index_->cells.size(), 0);
      for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
        clusters[cluster].push_back(seeds[cluster]);
        seeded[seeds[cluster]] = 1;
        --capacity[cluster];
      }
      struct assignment_t {
        float advantage = 0.0F;
        uint32_t cell = 0;
      };
      std::vector<assignment_t> assignments;
      assignments.reserve(cells.size() - cluster_count);
      for (const uint32_t cell : cells) {
        if (seeded[cell] != 0) {
          continue;
        }
        float best = std::numeric_limits<float>::infinity();
        float second = std::numeric_limits<float>::infinity();
        for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
          const auto centroid = std::span<const float>(cluster_centroids)
                                    .subspan(static_cast<size_t>(cluster) * dimension, dimension);
          const float distance = centroid_distance(cell, centroid);
          if (distance < best) {
            second = best;
            best = distance;
          } else if (distance < second) {
            second = distance;
          }
        }
        assignments.push_back({second - best, cell});
      }
      std::sort(assignments.begin(), assignments.end(), [](const auto& left, const auto& right) {
        if (left.advantage != right.advantage) {
          return left.advantage > right.advantage;
        }
        return left.cell < right.cell;
      });
      for (const auto& assignment : assignments) {
        uint32_t best_cluster = std::numeric_limits<uint32_t>::max();
        float best_distance = std::numeric_limits<float>::infinity();
        for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
          if (capacity[cluster] == 0) {
            continue;
          }
          const auto centroid = std::span<const float>(cluster_centroids)
                                    .subspan(static_cast<size_t>(cluster) * dimension, dimension);
          const float distance = centroid_distance(assignment.cell, centroid);
          if (std::tie(distance, cluster) < std::tie(best_distance, best_cluster)) {
            best_distance = distance;
            best_cluster = cluster;
          }
        }
        clusters[best_cluster].push_back(assignment.cell);
        --capacity[best_cluster];
      }
      std::fill(cluster_centroids.begin(), cluster_centroids.end(), 0.0F);
      for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
        double cluster_weight = 0.0;
        for (const uint32_t cell : clusters[cluster]) {
          const double weight = static_cast<double>(index_->cells[cell].node_count);
          cluster_weight += weight;
          const auto centroid = cell_centroid(cell);
          for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
            cluster_centroids[static_cast<size_t>(cluster) * dimension + coordinate] +=
                static_cast<float>(weight * centroid[coordinate]);
          }
        }
        for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
          cluster_centroids[static_cast<size_t>(cluster) * dimension + coordinate] =
              static_cast<float>(
                  cluster_centroids[static_cast<size_t>(cluster) * dimension + coordinate] /
                  cluster_weight);
        }
        uint32_t best_seed = clusters[cluster].front();
        float best_seed_distance = std::numeric_limits<float>::infinity();
        const auto centroid = std::span<const float>(cluster_centroids)
                                  .subspan(static_cast<size_t>(cluster) * dimension, dimension);
        for (const uint32_t cell : clusters[cluster]) {
          const float distance = centroid_distance(cell, centroid);
          if (std::tie(distance, cell) < std::tie(best_seed_distance, best_seed)) {
            best_seed = cell;
            best_seed_distance = distance;
          }
        }
        seeds[cluster] = best_seed;
      }
    }

    std::vector<uint32_t> child_nodes;
    child_nodes.reserve(cluster_count);
    for (auto& cluster : clusters) {
      child_nodes.push_back(append_node(std::move(cluster)));
    }
    auto& node = cell_hierarchy_nodes_[node_id];
    node.child_begin = static_cast<uint32_t>(cell_hierarchy_children_.size());
    node.child_count = static_cast<uint32_t>(child_nodes.size());
    node.children_are_cells = false;
    cell_hierarchy_children_.insert(cell_hierarchy_children_.end(), child_nodes.begin(),
                                    child_nodes.end());
    return node_id;
  };

  for (uint32_t community = 0; community < index_->communities.size(); ++community) {
    const auto& record = index_->communities[community];
    if (record.cell_count == 0) {
      continue;
    }
    std::vector<uint32_t> cells(record.cell_count);
    std::iota(cells.begin(), cells.end(), static_cast<uint32_t>(record.cell_begin));
    community_hierarchy_roots_[community] = append_node(std::move(cells));
  }
}

void community_polar_search_t::route_hierarchical_cells(
    std::span<const float> query, std::span<const uint32_t> communities,
    community_polar_query_state_t& state) const {
  const uint32_t routed_cell_budget =
      state.terminal_cell_budget == 0 ? config_.cell_routing_cell_budget
                                      : state.terminal_cell_budget;
  const uint32_t hierarchy_probe_budget =
      config_.cell_terminal_l_aware_forest
          ? std::min(config_.cell_hierarchy_probe_budget, routed_cell_budget)
          : config_.cell_hierarchy_probe_budget;
  const auto& hierarchy_nodes =
      index_->cell_hierarchy_nodes.empty() ? cell_hierarchy_nodes_ : index_->cell_hierarchy_nodes;
  const auto& hierarchy_children = index_->cell_hierarchy_nodes.empty()
                                       ? cell_hierarchy_children_
                                       : index_->cell_hierarchy_children;
  const auto& hierarchy_centroids = index_->cell_hierarchy_nodes.empty()
                                        ? cell_hierarchy_centroids_
                                        : index_->cell_hierarchy_centroids;
  const auto& hierarchy_roots = index_->cell_hierarchy_nodes.empty() ? community_hierarchy_roots_
                                                                     : index_->cell_hierarchy_roots;
  const auto node_lower_bound = [&](uint32_t node, float distance_squared) {
    const float lower = std::max(0.0F, std::sqrt(distance_squared) - hierarchy_nodes[node].radius);
    return lower * lower;
  };
  using scored_node_t = std::pair<float, uint32_t>;
  if (config_.cell_hierarchy_graph_guided_routing) {
    std::sort(state.graph_guided_cell_scratch.begin(), state.graph_guided_cell_scratch.end(),
              [&](const auto& left, const auto& right) {
                return std::make_tuple(std::numeric_limits<uint32_t>::max() -
                                           state.cell_support_counts[left.second],
                                       left.first, left.second) <
                       std::make_tuple(std::numeric_limits<uint32_t>::max() -
                                           state.cell_support_counts[right.second],
                                       right.first, right.second);
              });
    const size_t hint_count =
        std::min<size_t>(routed_cell_budget, state.graph_guided_cell_scratch.size());
    for (size_t hint = 0; hint < hint_count; ++hint) {
      const uint32_t hinted_cell = state.graph_guided_cell_scratch[hint].second;
      if (hinted_cell >= cell_hierarchy_leaf_parents_.size()) {
        continue;
      }
      uint32_t node = cell_hierarchy_leaf_parents_[hinted_cell];
      while (node != std::numeric_limits<uint32_t>::max()) {
        state.hierarchy_hint_path_scratch.push_back(node);
        node = cell_hierarchy_parents_[node];
      }
    }
    std::sort(state.hierarchy_hint_path_scratch.begin(), state.hierarchy_hint_path_scratch.end());
    state.hierarchy_hint_path_scratch.erase(std::unique(state.hierarchy_hint_path_scratch.begin(),
                                                        state.hierarchy_hint_path_scratch.end()),
                                            state.hierarchy_hint_path_scratch.end());
  }
  const auto is_hinted_node = [&](uint32_t node) {
    return std::binary_search(state.hierarchy_hint_path_scratch.begin(),
                              state.hierarchy_hint_path_scratch.end(), node);
  };
  const auto route_score = [&](uint32_t node, float geometric_score) {
    return is_hinted_node(node) ? -std::numeric_limits<float>::infinity() : geometric_score;
  };
  const auto hinted_cell_score = [&](uint32_t cell) {
    const size_t hint_count =
        std::min<size_t>(routed_cell_budget, state.graph_guided_cell_scratch.size());
    for (size_t rank = 0; rank < hint_count; ++rank) {
      if (state.graph_guided_cell_scratch[rank].second == cell) {
        return -static_cast<float>(hint_count - rank);
      }
    }
    return std::numeric_limits<float>::infinity();
  };
  std::priority_queue<scored_node_t, std::vector<scored_node_t>, std::greater<>> frontier;
  for (const uint32_t community : communities) {
    const uint32_t root = hierarchy_roots[community];
    if (root == std::numeric_limits<uint32_t>::max()) {
      continue;
    }
    const auto centroid =
        std::span<const float>(hierarchy_centroids)
            .subspan(static_cast<size_t>(root) * index_->dimension, index_->dimension);
    const float distance = query_distance_squared(query, centroid);
    const float geometric_score =
        config_.cell_hierarchy_exact_routing ? node_lower_bound(root, distance) : distance;
    frontier.emplace(route_score(root, geometric_score), root);
    ++state.cell_hierarchy_nodes_scored;
  }

  std::priority_queue<std::pair<float, uint32_t>> routed_cells;
  while (!frontier.empty()) {
    if (state.cell_hierarchy_leaf_groups_probed >= hierarchy_probe_budget &&
        routed_cells.size() >= routed_cell_budget) {
      if (!config_.cell_hierarchy_exact_routing ||
          frontier.top().first > routed_cells.top().first) {
        break;
      }
    }
    const uint32_t node_id = frontier.top().second;
    frontier.pop();
    const auto& node = hierarchy_nodes[node_id];
    if (node.children_are_cells) {
      ++state.cell_hierarchy_leaf_groups_probed;
      for (uint32_t offset = 0; offset < node.child_count; ++offset) {
        const uint32_t cell = hierarchy_children[node.child_begin + offset];
        const auto centroid =
            std::span<const float>(index_->direction_axes)
                .subspan(static_cast<size_t>(cell) * index_->dimension, index_->dimension);
        const float hint_score = hinted_cell_score(cell);
        const float score =
            std::isfinite(hint_score) ? hint_score : query_distance_squared(query, centroid);
        const std::pair<float, uint32_t> candidate{score, cell};
        if (routed_cells.size() < routed_cell_budget) {
          routed_cells.push(candidate);
        } else if (candidate < routed_cells.top()) {
          routed_cells.pop();
          routed_cells.push(candidate);
        }
      }
      continue;
    }
    for (uint32_t offset = 0; offset < node.child_count; ++offset) {
      const uint32_t child = hierarchy_children[node.child_begin + offset];
      const auto centroid =
          std::span<const float>(hierarchy_centroids)
              .subspan(static_cast<size_t>(child) * index_->dimension, index_->dimension);
      const float distance = query_distance_squared(query, centroid);
      const float geometric_score =
          config_.cell_hierarchy_exact_routing ? node_lower_bound(child, distance) : distance;
      frontier.emplace(route_score(child, geometric_score), child);
      ++state.cell_hierarchy_nodes_scored;
    }
  }
  std::vector<std::pair<float, uint32_t>> ordered_cells;
  ordered_cells.reserve(routed_cells.size());
  while (!routed_cells.empty()) {
    ordered_cells.push_back(routed_cells.top());
    routed_cells.pop();
  }
  std::sort(ordered_cells.begin(), ordered_cells.end());
  for (uint32_t rank = 0; rank < ordered_cells.size(); ++rank) {
    const uint32_t cell = ordered_cells[rank].second;
    state.touch_cell(cell);
    state.routed_cell_scratch.push_back(cell);
    state.cell_routing_ranks[cell] = rank;
    state.cell_candidate_created[cell] = 1;
    state.cell_support_counts[cell] =
        std::max(state.cell_support_counts[cell], config_.cell_support);
    const auto& record = index_->cells[cell];
    for (uint64_t block = record.block_begin; block < record.block_begin + record.block_count;
         ++block) {
      state.touch_block(static_cast<uint32_t>(block));
      state.block_support_counts[block] = config_.cell_support;
      state.block_best_graph_distance[block] = 0.0F;
    }
  }
}

std::shared_ptr<community_polar_search_t>
community_polar_search_t::load(const std::filesystem::path& index_path_prefix,
                               const community_polar_search_config_t& config,
                               const std::filesystem::path& requested_sidecar_path) {
  validate_community_polar_search_config(config);
  const auto sidecar_path = requested_sidecar_path.empty()
                                ? make_community_polar_index_path(index_path_prefix)
                                : requested_sidecar_path;
  const uint64_t artifact_bytes = std::filesystem::file_size(sidecar_path);
  const uint64_t artifact_checksum = file_checksum(sidecar_path);
  auto index = std::make_shared<community_polar_index_t>(load_community_polar_index(sidecar_path));
  validate_community_polar_artifacts(*index, index_path_prefix);
  return std::make_shared<community_polar_search_t>(std::move(index), config, artifact_bytes,
                                                    artifact_checksum);
}

community_polar_query_state_t
community_polar_search_t::begin_query(std::span<const float> query) const {
  community_polar_query_state_t state;
  begin_query(query, state);
  return state;
}

void community_polar_search_t::begin_query(std::span<const float> query,
                                           community_polar_query_state_t& state,
                                           bool cell_runtime_enabled) const {
  if (query.size() != index_->dimension) {
    throw std::invalid_argument("Community-Polar query dimension is incompatible");
  }
  const bool needs_community_state =
      config_.community_skip || config_.gateway_landing_cell_handoff ||
      !config_.cell_defer_hierarchy_routing || config_.cell_terminal_frontier_cells != 0;
  if (needs_community_state) {
    state.reachable_communities.assign(index_->communities.size(), 0);
    state.expanded_communities.assign(index_->communities.size(), 0);
    state.community_support_counts.assign(index_->communities.size(), 0);
    state.community_cell_candidate_created.assign(index_->communities.size(), 0);
  } else {
    state.reachable_communities.clear();
    state.expanded_communities.clear();
    state.community_support_counts.clear();
    state.community_cell_candidate_created.clear();
  }
  const bool cell_enabled = config_.cell_expansion && cell_runtime_enabled;
  if (config_.community_skip ||
      (config_.cell_centroid_routing && !config_.cell_defer_hierarchy_routing)) {
    state.community_lower_bounds.resize(index_->communities.size());
    state.community_routing_scores.resize(index_->communities.size());
    state.community_query_radii.resize(index_->communities.size());
    for (uint32_t community = 0; community < index_->communities.size(); ++community) {
      const auto pole =
          std::span<const float>(index_->community_poles)
              .subspan(static_cast<size_t>(community) * index_->dimension, index_->dimension);
      const float routing_score = query_distance_squared(query, pole);
      const float query_radius = std::sqrt(routing_score);
      const float lower = std::max(0.0F, query_radius - index_->communities[community].radius);
      state.community_routing_scores[community] = routing_score;
      state.community_query_radii[community] = query_radius;
      state.community_lower_bounds[community] = lower * lower;
    }
  } else {
    state.community_lower_bounds.clear();
    state.community_routing_scores.clear();
    if (cell_enabled && !config_.cell_defer_hierarchy_routing) {
      state.community_query_radii.resize(index_->communities.size());
      for (uint32_t community = 0; community < index_->communities.size(); ++community) {
        const auto pole =
            std::span<const float>(index_->community_poles)
                .subspan(static_cast<size_t>(community) * index_->dimension, index_->dimension);
        state.community_query_radii[community] = std::sqrt(query_distance_squared(query, pole));
      }
    } else {
      state.community_query_radii.clear();
    }
  }
  state.cell_runtime_enabled = cell_enabled;
  if (state.cell_runtime_enabled) {
    state.query.assign(query.begin(), query.end());
    if (config_.cell_defer_hierarchy_routing) {
      state.sector_query_angles.clear();
    } else {
      state.sector_query_angles.assign(index_->sectors.size(),
                                       std::numeric_limits<double>::quiet_NaN());
    }
    state.reset_cell_scratch(index_->cells.size(), index_->blocks.size());
  } else {
    state.query.clear();
    state.sector_query_angles.clear();
    state.cell_candidate_created.clear();
    state.cell_support_counts.clear();
    state.block_support_counts.clear();
    state.cell_best_graph_distance.clear();
    state.block_best_graph_distance.clear();
    state.loaded_blocks.clear();
    state.cell_routing_ranks.clear();
    state.touched_cell_flags.clear();
    state.touched_block_flags.clear();
    state.touched_cells.clear();
    state.touched_blocks.clear();
  }
  state.observed_base_sources.clear();
  state.observed_cell_evidence_sources.clear();
  state.macro_origins.clear();
  state.macro_block_ordinals.clear();
  state.cell_value_telemetry_enabled = config_.cell_value_telemetry;
  state.edge_order_scratch.clear();
  state.proposal_node_scratch.clear();
  state.gateway_proposal_scratch.clear();
  state.cell_evidence_scratch.clear();
  state.block_evidence_scratch.clear();
  state.supported_block_scratch.clear();
  state.routed_cell_scratch.clear();
  state.hierarchy_community_scratch.clear();
  state.hierarchy_hint_path_scratch.clear();
  state.graph_guided_cell_scratch.clear();
  state.cell_oracle_eligible_blocks.clear();
  state.cell_oracle_eligible_cells.clear();
  state.cell_oracle_block_hits.clear();
  state.cell_oracle_cell_hits.clear();
  state.cell_oracle_previsited_nodes.clear();
  state.cell_oracle_active = false;
  state.cell_oracle_selected_block = std::numeric_limits<uint32_t>::max();
  state.cell_oracle_polar_eligible_cell = std::numeric_limits<uint32_t>::max();
  state.cell_oracle_support_eligible_cell = std::numeric_limits<uint32_t>::max();
  state.cell_oracle_evidence_eligible_cell = std::numeric_limits<uint32_t>::max();
  state.last_base_cell = std::numeric_limits<uint32_t>::max();
  state.best_base_cell = std::numeric_limits<uint32_t>::max();
  state.best_base_cell_distance = std::numeric_limits<float>::infinity();
  state.cell_oracle_selected_blocks.fill(std::numeric_limits<uint32_t>::max());
  state.community_expansions = 0;
  state.gateway_scores = 0;
  state.cell_blocks = 0;
  state.next_routed_cell = 0;
  state.terminal_cell_budget = 0;
  state.cell_hierarchy_nodes_scored = 0;
  state.cell_hierarchy_leaf_groups_probed = 0;
  state.cell_hierarchy_stats_recorded = false;
  if (config_.gateway_landing_cell_handoff) {
    state.reachable_communities[index_->node_to_community[index_->entry_point_id]] = 1;
  }
  if (state.cell_runtime_enabled && !terminal_route_centroids_.empty()) {
    uint32_t best_bucket = 0;
    float best_distance = std::numeric_limits<float>::infinity();
    for (uint32_t bucket = 0; bucket < terminal_route_bucket_count_; ++bucket) {
      const auto centroid =
          std::span<const float>(terminal_route_centroids_)
              .subspan(static_cast<size_t>(bucket) * index_->dimension, index_->dimension);
      const float distance = query_distance_squared(query, centroid);
      if (std::tie(distance, bucket) < std::tie(best_distance, best_bucket)) {
        best_distance = distance;
        best_bucket = bucket;
      }
    }
    for (uint32_t rank = 0; rank < terminal_route_cells_per_bucket_ &&
                            state.routed_cell_scratch.size() < config_.cell_routing_cell_budget;
         ++rank) {
      const uint32_t cell = terminal_route_cells_[static_cast<size_t>(best_bucket) *
                                                      terminal_route_cells_per_bucket_ +
                                                  rank];
      if (cell == std::numeric_limits<uint32_t>::max()) {
        continue;
      }
      state.graph_guided_cell_scratch.emplace_back(static_cast<float>(rank), cell);
      state.touch_cell(cell);
      state.cell_support_counts[cell] =
          std::max(state.cell_support_counts[cell], config_.cell_support);
      const uint32_t community = index_->cells[cell].community_id;
      if (std::find(state.hierarchy_community_scratch.begin(),
                    state.hierarchy_community_scratch.end(),
                    community) == state.hierarchy_community_scratch.end()) {
        state.hierarchy_community_scratch.push_back(community);
      }
    }
    if (config_.cell_hierarchical_routing && !config_.cell_defer_hierarchy_routing) {
      route_hierarchical_cells(query, state.hierarchy_community_scratch, state);
    } else if (!config_.cell_hierarchical_routing) {
      for (const auto& [rank, cell] : state.graph_guided_cell_scratch) {
        static_cast<void>(rank);
        state.touch_cell(cell);
        state.routed_cell_scratch.push_back(cell);
        state.cell_routing_ranks[cell] =
            static_cast<uint32_t>(state.routed_cell_scratch.size() - 1U);
        state.cell_candidate_created[cell] = 1;
        state.cell_support_counts[cell] = config_.cell_support;
        const auto& record = index_->cells[cell];
        for (uint64_t block = record.block_begin; block < record.block_begin + record.block_count;
             ++block) {
          state.touch_block(static_cast<uint32_t>(block));
          state.block_support_counts[block] = config_.cell_support;
        }
      }
    }
  } else if (state.cell_runtime_enabled && config_.cell_centroid_routing &&
             !config_.cell_defer_hierarchy_routing) {
    std::vector<uint32_t> communities(index_->communities.size());
    std::iota(communities.begin(), communities.end(), 0);
    const size_t community_count =
        std::min<size_t>(config_.cell_routing_community_budget, communities.size());
    std::partial_sort(communities.begin(), communities.begin() + community_count, communities.end(),
                      [&](uint32_t left, uint32_t right) {
                        return std::tie(state.community_routing_scores[left], left) <
                               std::tie(state.community_routing_scores[right], right);
                      });
    if (config_.cell_hierarchical_routing) {
      route_hierarchical_cells(query, std::span<const uint32_t>(communities).first(community_count),
                               state);
    } else {
      std::vector<std::pair<float, uint32_t>> routed_cells;
      for (size_t rank = 0; rank < community_count; ++rank) {
        const auto& community = index_->communities[communities[rank]];
        for (uint64_t cell = community.cell_begin;
             cell < community.cell_begin + community.cell_count; ++cell) {
          const auto centroid =
              std::span<const float>(index_->direction_axes)
                  .subspan(static_cast<size_t>(cell) * index_->dimension, index_->dimension);
          routed_cells.emplace_back(query_distance_squared(query, centroid),
                                    static_cast<uint32_t>(cell));
        }
      }
      const size_t cell_count =
          std::min<size_t>(config_.cell_routing_cell_budget, routed_cells.size());
      std::partial_sort(routed_cells.begin(), routed_cells.begin() + cell_count,
                        routed_cells.end());
      for (uint32_t rank = 0; rank < cell_count; ++rank) {
        const uint32_t cell = routed_cells[rank].second;
        state.touch_cell(cell);
        state.routed_cell_scratch.push_back(cell);
        state.cell_routing_ranks[cell] = rank;
        state.cell_candidate_created[cell] = 1;
        state.cell_support_counts[cell] = config_.cell_support;
        state.cell_best_graph_distance[cell] = 0.0F;
        const auto& record = index_->cells[cell];
        for (uint64_t block = record.block_begin; block < record.block_begin + record.block_count;
             ++block) {
          state.touch_block(static_cast<uint32_t>(block));
          state.block_support_counts[block] = config_.cell_support;
          state.block_best_graph_distance[block] = 0.0F;
        }
      }
    }
  }
  if (state.observed_base_sources.bucket_count() < 128) {
    state.observed_base_sources.reserve(128);
  }
  if (state.observed_cell_evidence_sources.bucket_count() < 128) {
    state.observed_cell_evidence_sources.reserve(128);
  }
  if (state.cell_oracle_previsited_nodes.bucket_count() < 4096) {
    state.cell_oracle_previsited_nodes.reserve(4096);
  }
  const size_t origin_reserve =
      config_.gateway_score_budget + config_.cell_block_budget * config_.cell_insert_cap;
  if (state.macro_origins.bucket_count() < origin_reserve) {
    state.macro_origins.reserve(origin_reserve);
  }
  if (state.cell_value_telemetry_enabled &&
      state.macro_block_ordinals.bucket_count() < origin_reserve) {
    state.macro_block_ordinals.reserve(origin_reserve);
  }
  if (state.proposal_node_scratch.capacity() <
      std::max({config_.gateway_score_budget, config_.cell_insert_cap,
                config_.cell_whole_cell_scan ? index_->config.cell_target_size : 0U})) {
    state.proposal_node_scratch.reserve(
        std::max({config_.gateway_score_budget, config_.cell_insert_cap,
                  config_.cell_whole_cell_scan ? index_->config.cell_target_size : 0U}));
  }
  if (state.gateway_proposal_scratch.capacity() < config_.gateway_score_budget) {
    state.gateway_proposal_scratch.reserve(config_.gateway_score_budget);
  }
}

void community_polar_search_t::record_cell_oracle_previsited(community_polar_query_state_t& state,
                                                             uint32_t node_id) const {
  if (state.cell_oracle_active && node_id < index_->point_count) {
    state.cell_oracle_previsited_nodes.insert(node_id);
  }
}

void community_polar_search_t::observe_base_expansion(community_polar_query_state_t& state,
                                                      uint32_t source_node,
                                                      query_stats_t* stats) const {
  if (config_.cell_defer_hierarchy_routing && !config_.community_skip &&
      !config_.gateway_landing_cell_handoff && config_.cell_terminal_frontier_cells == 0) {
    return;
  }
  if (source_node >= index_->point_count ||
      !state.observed_base_sources.insert(source_node).second) {
    return;
  }
  const uint32_t community = index_->node_to_community[source_node];
  state.last_base_cell = index_->node_to_cell[source_node];
  state.reachable_communities[community] = 1;

  if (!accepts_cell_evidence(state) || index_->communities[community].graph_only) {
    return;
  }
  ++state.community_support_counts[community];
  if (state.community_support_counts[community] == config_.cell_support) {
    state.community_cell_candidate_created[community] = 1;
  }
}

void community_polar_search_t::observe_source_cell(community_polar_query_state_t& state,
                                                   uint32_t source_node,
                                                   float source_distance_squared,
                                                   query_stats_t* stats) const {
  if ((config_.cell_defer_hierarchy_routing && config_.cell_terminal_convergence &&
       config_.cell_terminal_frontier_cells == 0 && !config_.cell_hierarchy_graph_guided_routing) ||
      !config_.source_cell_batching || !accepts_cell_evidence(state) ||
      source_node >= index_->point_count || !std::isfinite(source_distance_squared) ||
      !state.observed_cell_evidence_sources.insert(source_node).second) {
    return;
  }
  const uint32_t cell = index_->node_to_cell[source_node];
  if (std::tie(source_distance_squared, cell) <
      std::tie(state.best_base_cell_distance, state.best_base_cell)) {
    state.best_base_cell = cell;
    state.best_base_cell_distance = source_distance_squared;
  }
  if (config_.cell_hierarchy_graph_guided_routing) {
    auto existing =
        std::find_if(state.graph_guided_cell_scratch.begin(), state.graph_guided_cell_scratch.end(),
                     [cell](const auto& candidate) { return candidate.second == cell; });
    if (existing == state.graph_guided_cell_scratch.end()) {
      state.graph_guided_cell_scratch.emplace_back(source_distance_squared, cell);
    } else if (source_distance_squared < existing->first) {
      existing->first = source_distance_squared;
    }
  }
  const uint32_t community = index_->cells[cell].community_id;
  if (index_->communities[community].graph_only) {
    return;
  }
  const uint32_t block = node_to_block_[source_node];
  state.touch_cell(cell);
  state.touch_block(block);
  state.cell_best_graph_distance[cell] =
      std::min(state.cell_best_graph_distance[cell], source_distance_squared);
  state.block_best_graph_distance[block] =
      std::min(state.block_best_graph_distance[block], source_distance_squared);
  if (state.cell_candidate_created[cell] == 0) {
    state.cell_candidate_created[cell] = 1;
    state.cell_support_counts[cell] = config_.cell_support;
    if (stats != nullptr) {
      ++stats->cell_candidates_created;
    }
  } else if (config_.cell_terminal_convergence &&
             config_.cell_hierarchy_graph_guided_routing &&
             state.cell_support_counts[cell] != std::numeric_limits<uint32_t>::max()) {
    ++state.cell_support_counts[cell];
  }
  if (state.block_support_counts[block] == 0) {
    state.supported_block_scratch.push_back(block);
  }
  state.block_support_counts[block] = config_.cell_support;
}

void community_polar_search_t::observe_base_candidates(community_polar_query_state_t& state,
                                                       uint32_t source_node,
                                                       std::span<const uint32_t> candidate_nodes,
                                                       std::span<const float> candidate_distances,
                                                       query_stats_t* stats) const {
  if (!accepts_cell_evidence(state) || source_node >= index_->point_count ||
      candidate_nodes.size() != candidate_distances.size() ||
      !state.observed_cell_evidence_sources.insert(source_node).second) {
    return;
  }
  const uint32_t source_community = index_->node_to_community[source_node];
  if (index_->communities[source_community].graph_only) {
    return;
  }

  auto& cells = state.cell_evidence_scratch;
  auto& blocks = state.block_evidence_scratch;
  cells.clear();
  blocks.clear();
  const uint32_t source_cell = index_->node_to_cell[source_node];
  const uint32_t source_block = node_to_block_[source_node];
  cells.push_back(source_cell);
  blocks.push_back(source_block);

  for (size_t position = 0; position < candidate_nodes.size(); ++position) {
    const uint32_t node = candidate_nodes[position];
    if (node >= index_->point_count || !std::isfinite(candidate_distances[position])) {
      continue;
    }
    const uint32_t cell = index_->node_to_cell[node];
    const uint32_t community = index_->cells[cell].community_id;
    if (index_->communities[community].graph_only) {
      continue;
    }
    const uint32_t block = node_to_block_[node];
    state.touch_cell(cell);
    state.touch_block(block);
    state.cell_best_graph_distance[cell] =
        std::min(state.cell_best_graph_distance[cell], candidate_distances[position]);
    state.block_best_graph_distance[block] =
        std::min(state.block_best_graph_distance[block], candidate_distances[position]);
    cells.push_back(cell);
    blocks.push_back(block);
  }

  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  for (const uint32_t cell : cells) {
    state.touch_cell(cell);
    ++state.cell_support_counts[cell];
    if (state.cell_support_counts[cell] == config_.cell_support) {
      state.cell_candidate_created[cell] = 1;
      if (stats != nullptr) {
        ++stats->cell_candidates_created;
      }
    }
  }
  std::sort(blocks.begin(), blocks.end());
  blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
  for (const uint32_t block : blocks) {
    state.touch_block(block);
    if (state.block_support_counts[block] == 0) {
      state.supported_block_scratch.push_back(block);
    }
    ++state.block_support_counts[block];
  }
}

bool community_polar_search_t::has_community_work(
    const community_polar_query_state_t& state) const noexcept {
  if (!config_.community_skip || state.community_expansions >= config_.community_expansion_budget ||
      state.gateway_scores >= config_.gateway_score_budget) {
    return false;
  }
  for (uint32_t community = 0; community < index_->communities.size(); ++community) {
    if (state.reachable_communities[community] && !state.expanded_communities[community]) {
      return true;
    }
  }
  return false;
}

std::optional<community_polar_candidate_batch_t>
community_polar_search_t::expand_next_community(community_polar_query_state_t& state,
                                                query_stats_t* stats) const {
  if (!has_community_work(state)) {
    return std::nullopt;
  }
  uint32_t source = std::numeric_limits<uint32_t>::max();
  for (uint32_t community = 0; community < index_->communities.size(); ++community) {
    if (!state.reachable_communities[community] || state.expanded_communities[community]) {
      continue;
    }
    if (source == std::numeric_limits<uint32_t>::max() ||
        std::tie(state.community_routing_scores[community], community) <
            std::tie(state.community_routing_scores[source], source)) {
      source = community;
    }
  }
  state.expanded_communities[source] = 1;
  ++state.community_expansions;
  if (stats != nullptr) {
    ++stats->community_meta_expansions;
  }

  auto& edges = state.edge_order_scratch;
  edges.clear();
  if (config_.community_hub_clique) {
    for (uint32_t target = 0; target < index_->communities.size(); ++target) {
      if (target != source) {
        edges.push_back(target);
      }
    }
    std::sort(edges.begin(), edges.end(), [&](uint64_t left, uint64_t right) {
      const uint32_t left_target = static_cast<uint32_t>(left);
      const uint32_t right_target = static_cast<uint32_t>(right);
      return std::tie(state.community_routing_scores[left_target], left_target) <
             std::tie(state.community_routing_scores[right_target], right_target);
    });

    community_polar_candidate_batch_t batch;
    batch.origin = community_polar_candidate_origin_t::COMMUNITY_GATEWAY;
    batch.object_id = source;
    batch.insert_cap = config_.gateway_direct_insert_cap;
    batch.lower_bound_squared = std::numeric_limits<float>::infinity();
    const uint32_t remaining =
        std::min(config_.gateway_score_budget - state.gateway_scores,
                 config_.gateway_score_batch_cap);
    auto& nodes = state.proposal_node_scratch;
    auto& proposals = state.gateway_proposal_scratch;
    nodes.clear();
    proposals.clear();
    nodes.reserve(remaining);
    for (const uint64_t encoded_target : edges) {
      const uint32_t target = static_cast<uint32_t>(encoded_target);
      state.reachable_communities[target] = 1;
      batch.lower_bound_squared =
          std::min(batch.lower_bound_squared, state.community_lower_bounds[target]);
      if (stats != nullptr) {
        ++stats->community_superedges_considered;
      }
      nodes.push_back(index_->communities[target].navigation_hub_id);
      if (nodes.size() == remaining) {
        break;
      }
    }
    if (nodes.empty()) {
      return std::nullopt;
    }
    batch.node_ids = nodes;
    batch.gateway_proposals = proposals;
    return batch;
  }
  for (uint64_t edge = index_->community_edge_offsets[source];
       edge < index_->community_edge_offsets[source + 1]; ++edge) {
    edges.push_back(edge);
  }
  std::sort(edges.begin(), edges.end(), [&](uint64_t left, uint64_t right) {
    const uint32_t left_target = index_->community_edges[left].target_community;
    const uint32_t right_target = index_->community_edges[right].target_community;
    return std::tie(state.community_routing_scores[left_target], left_target, left) <
           std::tie(state.community_routing_scores[right_target], right_target, right);
  });

  community_polar_candidate_batch_t batch;
  batch.origin = community_polar_candidate_origin_t::COMMUNITY_GATEWAY;
  batch.object_id = source;
  batch.insert_cap = config_.gateway_direct_insert_cap;
  batch.lower_bound_squared = std::numeric_limits<float>::infinity();
  const uint32_t remaining =
      std::min(config_.gateway_score_budget - state.gateway_scores,
               config_.gateway_score_batch_cap);
  auto& nodes = state.proposal_node_scratch;
  auto& proposals = state.gateway_proposal_scratch;
  nodes.clear();
  proposals.clear();
  nodes.reserve(remaining);
  for (const uint64_t edge_id : edges) {
    const auto& edge = index_->community_edges[edge_id];
    state.reachable_communities[edge.target_community] = 1;
    batch.lower_bound_squared =
        std::min(batch.lower_bound_squared, state.community_lower_bounds[edge.target_community]);
    if (stats != nullptr) {
      ++stats->community_superedges_considered;
    }
    for (uint64_t gateway = edge.gateway_begin;
         gateway < edge.gateway_begin + edge.gateway_count && nodes.size() < remaining; ++gateway) {
      const auto& record = index_->gateways[gateway];
      nodes.push_back(record.target_node);
      proposals.push_back({gateway, edge.source_community, edge.target_community,
                           record.source_node, record.target_node});
    }
    if (nodes.size() == remaining) {
      break;
    }
  }
  if (nodes.empty()) {
    return std::nullopt;
  }
  batch.node_ids = nodes;
  batch.gateway_proposals = proposals;
  return batch;
}

float community_polar_search_t::cell_query_distance_squared(
    const community_polar_query_state_t& state, uint32_t cell_id) const noexcept {
  if (cell_id >= index_->cells.size() || state.query.size() != index_->dimension ||
      index_->direction_axes.size() != index_->cells.size() * index_->dimension) {
    return std::numeric_limits<float>::infinity();
  }
  const auto centroid =
      std::span<const float>(index_->direction_axes)
          .subspan(static_cast<size_t>(cell_id) * index_->dimension, index_->dimension);
  return query_distance_squared(state.query, centroid);
}

void community_polar_search_t::record_gateway_scores(community_polar_query_state_t& state,
                                                     uint32_t count) const {
  state.gateway_scores = std::min(config_.gateway_score_budget, state.gateway_scores + count);
}

bool community_polar_search_t::observe_gateway_landing_candidate(
    community_polar_query_state_t& state, uint32_t node_id, float distance_squared,
    query_stats_t* stats) const {
  if (!config_.gateway_landing_cell_handoff || !state.cell_runtime_enabled ||
      node_id >= index_->point_count || !std::isfinite(distance_squared) ||
      state.routed_cell_scratch.size() >= config_.cell_routing_cell_budget) {
    return false;
  }
  const uint32_t community = index_->node_to_community[node_id];
  const auto& community_record = index_->communities[community];
  uint32_t cell = std::numeric_limits<uint32_t>::max();
  float best_centroid_distance = std::numeric_limits<float>::infinity();
  for (uint64_t candidate = community_record.cell_begin;
       candidate < community_record.cell_begin + community_record.cell_count; ++candidate) {
    if (state.cell_candidate_created[candidate] != 0) {
      continue;
    }
    const auto centroid =
        std::span<const float>(index_->direction_axes)
            .subspan(static_cast<size_t>(candidate) * index_->dimension, index_->dimension);
    const float centroid_distance = query_distance_squared(state.query, centroid);
    if (std::tie(centroid_distance, candidate) < std::tie(best_centroid_distance, cell)) {
      cell = static_cast<uint32_t>(candidate);
      best_centroid_distance = centroid_distance;
    }
  }
  if (cell == std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  if (state.cell_candidate_created[cell] != 0) {
    state.touch_cell(cell);
    state.cell_best_graph_distance[cell] =
        std::min(state.cell_best_graph_distance[cell], distance_squared);
    return false;
  }
  const uint32_t routing_rank = static_cast<uint32_t>(state.routed_cell_scratch.size());
  state.touch_cell(cell);
  state.routed_cell_scratch.push_back(cell);
  state.cell_routing_ranks[cell] = routing_rank;
  state.cell_candidate_created[cell] = 1;
  state.cell_support_counts[cell] = config_.cell_support;
  state.cell_best_graph_distance[cell] = distance_squared;
  const auto& record = index_->cells[cell];
  for (uint64_t block = record.block_begin; block < record.block_begin + record.block_count;
       ++block) {
    state.touch_block(static_cast<uint32_t>(block));
    state.block_support_counts[block] = config_.cell_support;
    state.block_best_graph_distance[block] = distance_squared;
  }
  if (stats != nullptr) {
    ++stats->gateway_landing_cells_routed;
  }
  return true;
}

void community_polar_search_t::prepare_terminal_convergence(
    community_polar_query_state_t& state, uint64_t search_list_size) const {
  if (!config_.cell_terminal_convergence) {
    return;
  }
  state.terminal_cell_budget = terminal_cell_budget(search_list_size);
  std::vector<uint32_t> frontier_cells;
  frontier_cells.reserve(state.supported_block_scratch.size());
  for (const uint32_t block : state.supported_block_scratch) {
    const uint32_t cell = index_->blocks[block].cell_id;
    if (state.cell_candidate_created[cell] != 0 &&
        std::isfinite(state.cell_best_graph_distance[cell])) {
      frontier_cells.push_back(cell);
    }
  }
  std::sort(frontier_cells.begin(), frontier_cells.end(), [&](uint32_t left, uint32_t right) {
    return std::make_tuple(std::numeric_limits<uint32_t>::max() -
                               state.cell_support_counts[left],
                           state.cell_best_graph_distance[left], left) <
           std::make_tuple(std::numeric_limits<uint32_t>::max() -
                               state.cell_support_counts[right],
                           state.cell_best_graph_distance[right], right);
  });
  frontier_cells.erase(std::unique(frontier_cells.begin(), frontier_cells.end()),
                       frontier_cells.end());

  if (config_.cell_defer_hierarchy_routing) {
    auto root_communities = std::span<const uint32_t>(hierarchy_root_communities_);
    if (config_.cell_terminal_global_forest_routing) {
      // Community Skip accelerates the approach phase but must not become a hard search boundary.
      // The global forest remains metadata-only; raw vectors are read only for admitted leaves.
    } else if (!state.hierarchy_community_scratch.empty()) {
      root_communities = state.hierarchy_community_scratch;
    } else if (config_.community_skip) {
      for (const uint32_t community : hierarchy_root_communities_) {
        if (state.reachable_communities[community] != 0) {
          state.hierarchy_community_scratch.push_back(community);
        }
      }
      if (!state.hierarchy_community_scratch.empty()) {
        root_communities = state.hierarchy_community_scratch;
      }
    }
    route_hierarchical_cells(state.query, root_communities, state);
  }
  const auto centroid_cells = state.routed_cell_scratch;
  state.routed_cell_scratch.clear();
  const size_t frontier_count =
      std::min<size_t>({config_.cell_terminal_frontier_cells, frontier_cells.size(),
                        state.terminal_cell_budget});
  state.routed_cell_scratch.insert(state.routed_cell_scratch.end(), frontier_cells.begin(),
                                   frontier_cells.begin() + frontier_count);
  for (const uint32_t cell : centroid_cells) {
    if (state.routed_cell_scratch.size() >= state.terminal_cell_budget) {
      break;
    }
    if (std::find(state.routed_cell_scratch.begin(), state.routed_cell_scratch.end(), cell) ==
        state.routed_cell_scratch.end()) {
      state.routed_cell_scratch.push_back(cell);
    }
  }
  // reset_cell_scratch() already clears every previously touched rank. The hierarchy router
  // touches every newly routed Cell before assigning its rank, so a full O(total Cells) reset here
  // is redundant. At SIFT100M that fill wrote 18.6 MiB per query even when one leaf was probed.
  for (uint32_t rank = 0; rank < state.routed_cell_scratch.size(); ++rank) {
    const uint32_t cell = state.routed_cell_scratch[rank];
    state.touch_cell(cell);
    state.cell_routing_ranks[cell] = rank;
    state.cell_candidate_created[cell] = 1;
    state.cell_support_counts[cell] =
        std::max(state.cell_support_counts[cell], config_.cell_support);
    const auto& record = index_->cells[cell];
    for (uint64_t block = record.block_begin; block < record.block_begin + record.block_count;
         ++block) {
      state.touch_block(static_cast<uint32_t>(block));
      state.block_support_counts[block] =
          std::max(state.block_support_counts[block], config_.cell_support);
    }
  }
  state.next_routed_cell = 0;
}

bool community_polar_search_t::begin_cell_oracle(community_polar_query_state_t& state,
                                                 query_stats_t* stats,
                                                 float candidate_threshold) const {
  if (state.cell_oracle_active || !state.cell_runtime_enabled) {
    return false;
  }
  state.cell_oracle_eligible_blocks.assign(index_->blocks.size(), 0);
  state.cell_oracle_eligible_cells.assign(index_->cells.size(), 0);
  state.cell_oracle_block_hits.assign(index_->blocks.size(), 0);
  state.cell_oracle_cell_hits.assign(index_->cells.size(), 0);
  uint32_t eligible_blocks = 0;
  uint32_t eligible_cells = 0;
  const bool direct_work = has_direct_cell_work(state);
  if (direct_work) {
    for (const uint32_t block : state.supported_block_scratch) {
      const uint32_t cell = index_->blocks[block].cell_id;
      if (!is_mature_cell_block(state, block, candidate_threshold)) {
        continue;
      }
      state.cell_oracle_eligible_blocks[block] = 1;
      if (state.cell_oracle_eligible_cells[cell] == 0) {
        state.cell_oracle_eligible_cells[cell] = 1;
        ++eligible_cells;
      }
      ++eligible_blocks;
    }
  }
  for (uint32_t cell = 0; cell < index_->cells.size(); ++cell) {
    const auto& cell_record = index_->cells[cell];
    const bool eligible =
        direct_work ? state.cell_candidate_created[cell] != 0
                    : state.community_cell_candidate_created[cell_record.community_id] != 0;
    if (direct_work || !eligible || index_->communities[cell_record.community_id].graph_only) {
      continue;
    }
    for (uint64_t block = cell_record.block_begin;
         block < cell_record.block_begin + cell_record.block_count; ++block) {
      if (direct_work && state.block_support_counts[block] == 0) {
        continue;
      }
      if (state.cell_oracle_eligible_blocks[block] == 0) {
        state.cell_oracle_eligible_blocks[block] = 1;
        ++eligible_blocks;
      }
    }
  }
  state.cell_oracle_active = true;
  float best_cell_bound = std::numeric_limits<float>::infinity();
  uint32_t best_cell_support = 0;
  float best_support_evidence = std::numeric_limits<float>::infinity();
  float best_cell_evidence = std::numeric_limits<float>::infinity();
  for (uint32_t cell = 0; cell < index_->cells.size(); ++cell) {
    if (state.cell_oracle_eligible_cells[cell] == 0) {
      continue;
    }
    const auto& record = index_->cells[cell];
    float cell_bound = std::numeric_limits<float>::infinity();
    for (uint64_t block = record.block_begin; block < record.block_begin + record.block_count;
         ++block) {
      cell_bound = std::min(cell_bound, block_lower_bound_squared(state, block));
    }
    if (std::tie(cell_bound, cell) <
        std::tie(best_cell_bound, state.cell_oracle_polar_eligible_cell)) {
      best_cell_bound = cell_bound;
      state.cell_oracle_polar_eligible_cell = cell;
    }
    if (state.cell_support_counts[cell] > best_cell_support ||
        (state.cell_support_counts[cell] == best_cell_support &&
         std::tie(state.cell_best_graph_distance[cell], cell) <
             std::tie(best_support_evidence, state.cell_oracle_support_eligible_cell))) {
      best_cell_support = state.cell_support_counts[cell];
      best_support_evidence = state.cell_best_graph_distance[cell];
      state.cell_oracle_support_eligible_cell = cell;
    }
    if (std::tie(state.cell_best_graph_distance[cell], cell) <
        std::tie(best_cell_evidence, state.cell_oracle_evidence_eligible_cell)) {
      best_cell_evidence = state.cell_best_graph_distance[cell];
      state.cell_oracle_evidence_eligible_cell = cell;
    }
  }
  if (stats != nullptr) {
    stats->cell_oracle_triggered = 1;
    stats->cell_oracle_trigger_hop = stats->base_hops;
    stats->cell_oracle_eligible_blocks = eligible_blocks;
    stats->cell_oracle_eligible_cells = eligible_cells;
  }
  return true;
}

void community_polar_search_t::record_cell_oracle_selection(community_polar_query_state_t& state,
                                                            uint32_t block_id,
                                                            uint32_t block_ordinal) const {
  if (!state.cell_oracle_active || block_id >= index_->blocks.size()) {
    return;
  }
  state.cell_oracle_selected_block = block_id;
  if (block_ordinal < state.cell_oracle_selected_blocks.size()) {
    state.cell_oracle_selected_blocks[block_ordinal] = block_id;
  }
}

void community_polar_search_t::observe_cell_oracle_expansion(community_polar_query_state_t& state,
                                                             uint32_t node_id) const {
  if (!state.cell_oracle_active || node_id >= node_to_block_.size()) {
    return;
  }
  if (state.cell_oracle_previsited_nodes.contains(node_id)) {
    return;
  }
  const uint32_t block = node_to_block_[node_id];
  const uint32_t cell = index_->blocks[block].cell_id;
  ++state.cell_oracle_block_hits[block];
  ++state.cell_oracle_cell_hits[cell];
}

void community_polar_search_t::finish_cell_oracle(community_polar_query_state_t& state,
                                                  query_stats_t* stats) const {
  if (!state.cell_oracle_active || stats == nullptr) {
    return;
  }

  std::array<uint32_t, 4> top_block_hits{};
  std::array<uint32_t, 4> top_cell_hits{};
  std::array<uint32_t, 4> top_eligible_cell_hits{};
  uint32_t future_expansions = 0;
  uint32_t distinct_blocks = 0;
  uint32_t distinct_cells = 0;
  uint32_t best_eligible_hits = 0;
  uint32_t best_eligible_block = std::numeric_limits<uint32_t>::max();
  uint32_t best_global_hits = 0;
  uint32_t best_global_block = std::numeric_limits<uint32_t>::max();
  uint32_t best_eligible_cell_hits = 0;
  uint32_t best_eligible_cell = std::numeric_limits<uint32_t>::max();
  for (uint32_t block = 0; block < state.cell_oracle_block_hits.size(); ++block) {
    const uint32_t hits = state.cell_oracle_block_hits[block];
    future_expansions += hits;
    if (hits != 0) {
      ++distinct_blocks;
      add_top_hit(top_block_hits, hits);
    }
    if (hits > best_global_hits ||
        (hits == best_global_hits && hits != 0 && block < best_global_block)) {
      best_global_hits = hits;
      best_global_block = block;
    }
    if (state.cell_oracle_eligible_blocks[block] != 0 &&
        (hits > best_eligible_hits ||
         (hits == best_eligible_hits && hits != 0 && block < best_eligible_block))) {
      best_eligible_hits = hits;
      best_eligible_block = block;
    }
  }
  for (uint32_t cell = 0; cell < state.cell_oracle_cell_hits.size(); ++cell) {
    const uint32_t hits = state.cell_oracle_cell_hits[cell];
    if (hits == 0) {
      continue;
    }
    ++distinct_cells;
    add_top_hit(top_cell_hits, hits);
    if (state.cell_oracle_eligible_cells[cell] != 0) {
      add_top_hit(top_eligible_cell_hits, hits);
    }
    if (state.cell_oracle_eligible_cells[cell] != 0 &&
        (hits > best_eligible_cell_hits ||
         (hits == best_eligible_cell_hits && cell < best_eligible_cell))) {
      best_eligible_cell_hits = hits;
      best_eligible_cell = cell;
    }
  }

  stats->cell_oracle_future_expansions = future_expansions;
  stats->cell_oracle_future_distinct_cells = distinct_cells;
  stats->cell_oracle_future_distinct_blocks = distinct_blocks;
  stats->cell_oracle_selected_block_id = state.cell_oracle_selected_block;
  if (state.cell_oracle_selected_block < state.cell_oracle_block_hits.size()) {
    stats->cell_oracle_selected_block_hits =
        state.cell_oracle_block_hits[state.cell_oracle_selected_block];
    stats->cell_oracle_selected_cell_hits =
        state.cell_oracle_cell_hits[index_->blocks[state.cell_oracle_selected_block].cell_id];
  }
  for (size_t ordinal = 0; ordinal < state.cell_oracle_selected_blocks.size(); ++ordinal) {
    const uint32_t block = state.cell_oracle_selected_blocks[ordinal];
    if (block < state.cell_oracle_block_hits.size()) {
      stats->cell_value_future_expansions[ordinal] = state.cell_oracle_block_hits[block];
    }
  }
  stats->cell_oracle_best_eligible_block_hits = best_eligible_hits;
  stats->cell_oracle_best_eligible_block_id = best_eligible_block;
  stats->cell_oracle_best_eligible_cell_hits = best_eligible_cell_hits;
  stats->cell_oracle_best_eligible_cell_id = best_eligible_cell;
  stats->cell_oracle_polar_eligible_cell_id = state.cell_oracle_polar_eligible_cell;
  if (state.cell_oracle_polar_eligible_cell < state.cell_oracle_cell_hits.size()) {
    stats->cell_oracle_polar_eligible_cell_hits =
        state.cell_oracle_cell_hits[state.cell_oracle_polar_eligible_cell];
  }
  stats->cell_oracle_support_eligible_cell_id = state.cell_oracle_support_eligible_cell;
  if (state.cell_oracle_support_eligible_cell < state.cell_oracle_cell_hits.size()) {
    stats->cell_oracle_support_eligible_cell_hits =
        state.cell_oracle_cell_hits[state.cell_oracle_support_eligible_cell];
  }
  stats->cell_oracle_evidence_eligible_cell_id = state.cell_oracle_evidence_eligible_cell;
  if (state.cell_oracle_evidence_eligible_cell < state.cell_oracle_cell_hits.size()) {
    stats->cell_oracle_evidence_eligible_cell_hits =
        state.cell_oracle_cell_hits[state.cell_oracle_evidence_eligible_cell];
  }
  if (state.last_base_cell < state.cell_oracle_cell_hits.size()) {
    stats->cell_oracle_last_source_cell_hits = state.cell_oracle_cell_hits[state.last_base_cell];
  }
  stats->cell_oracle_best_global_block_id = best_global_block;
  stats->cell_oracle_top1_cell_hits = top_hit_sum(top_cell_hits, 1);
  stats->cell_oracle_top2_cell_hits = top_hit_sum(top_cell_hits, 2);
  stats->cell_oracle_top4_cell_hits = top_hit_sum(top_cell_hits, 4);
  stats->cell_oracle_top4_eligible_cell_hits = top_hit_sum(top_eligible_cell_hits, 4);

  std::vector<uint32_t> ranked_cells;
  ranked_cells.reserve(stats->cell_oracle_eligible_cells);
  for (uint32_t cell = 0; cell < state.cell_oracle_eligible_cells.size(); ++cell) {
    if (state.cell_oracle_eligible_cells[cell] != 0) {
      ranked_cells.push_back(cell);
    }
  }
  const auto sum_ranked_hits = [&](const auto& less) {
    std::sort(ranked_cells.begin(), ranked_cells.end(), less);
    uint32_t hits = 0;
    for (size_t rank = 0; rank < std::min<size_t>(4, ranked_cells.size()); ++rank) {
      hits += state.cell_oracle_cell_hits[ranked_cells[rank]];
    }
    return hits;
  };
  stats->cell_oracle_top4_support_cell_hits = sum_ranked_hits([&](uint32_t left, uint32_t right) {
    return std::make_tuple(std::numeric_limits<uint32_t>::max() - state.cell_support_counts[left],
                           state.cell_best_graph_distance[left], left) <
           std::make_tuple(std::numeric_limits<uint32_t>::max() - state.cell_support_counts[right],
                           state.cell_best_graph_distance[right], right);
  });
  stats->cell_oracle_top4_evidence_cell_hits = sum_ranked_hits([&](uint32_t left, uint32_t right) {
    return std::tie(state.cell_best_graph_distance[left], left) <
           std::tie(state.cell_best_graph_distance[right], right);
  });
  std::vector<float> cell_lower_bounds(index_->cells.size(),
                                       std::numeric_limits<float>::infinity());
  for (const uint32_t cell : ranked_cells) {
    const auto& record = index_->cells[cell];
    for (uint64_t block = record.block_begin; block < record.block_begin + record.block_count;
         ++block) {
      cell_lower_bounds[cell] =
          std::min(cell_lower_bounds[cell], block_lower_bound_squared(state, block));
    }
  }
  stats->cell_oracle_top4_polar_cell_hits = sum_ranked_hits([&](uint32_t left, uint32_t right) {
    return std::tie(cell_lower_bounds[left], left) < std::tie(cell_lower_bounds[right], right);
  });
  stats->cell_oracle_top1_block_hits = top_hit_sum(top_block_hits, 1);
  stats->cell_oracle_top2_block_hits = top_hit_sum(top_block_hits, 2);
  stats->cell_oracle_top4_block_hits = top_hit_sum(top_block_hits, 4);
}

bool community_polar_search_t::has_cell_work(
    const community_polar_query_state_t& state) const noexcept {
  return has_cell_work(state, std::numeric_limits<float>::infinity());
}

bool community_polar_search_t::has_cell_work(const community_polar_query_state_t& state,
                                             float candidate_threshold) const noexcept {
  if (!state.cell_runtime_enabled || state.cell_blocks >= config_.cell_block_budget) {
    return false;
  }
  if (config_.cell_require_full_frontier && !std::isfinite(candidate_threshold)) {
    return false;
  }
  if (config_.cell_whole_cell_scan &&
      (config_.cell_centroid_routing || config_.gateway_landing_cell_handoff)) {
    for (size_t rank = state.next_routed_cell; rank < state.routed_cell_scratch.size(); ++rank) {
      const auto& cell = index_->cells[state.routed_cell_scratch[rank]];
      if (cell.block_count != 0 && state.loaded_blocks[cell.block_begin] == 0) {
        return true;
      }
    }
    if (config_.cell_terminal_convergence) {
      return false;
    }
  }
  const bool direct_work = has_direct_cell_work(state);
  if (!direct_work) {
    return false;
  }
  for (const uint32_t block : state.supported_block_scratch) {
    if (is_mature_cell_block(state, block, candidate_threshold)) {
      return true;
    }
  }
  return false;
}

bool community_polar_search_t::accepts_cell_evidence(
    const community_polar_query_state_t& state) const noexcept {
  return state.cell_runtime_enabled && state.cell_blocks < config_.cell_block_budget;
}

std::optional<community_polar_candidate_batch_t>
community_polar_search_t::expand_next_cell(community_polar_query_state_t& state,
                                           uint64_t search_list_size, query_stats_t* stats) const {
  return expand_next_cell(state, search_list_size, std::numeric_limits<float>::infinity(), stats);
}

std::optional<community_polar_candidate_batch_t>
community_polar_search_t::expand_next_cell(community_polar_query_state_t& state,
                                           uint64_t search_list_size, float candidate_threshold,
                                           query_stats_t* stats) const {
  if (!has_cell_work(state, candidate_threshold)) {
    return std::nullopt;
  }
  if (stats != nullptr && !state.cell_hierarchy_stats_recorded) {
    stats->cell_hierarchy_nodes_scored = state.cell_hierarchy_nodes_scored;
    stats->cell_hierarchy_leaf_groups_probed = state.cell_hierarchy_leaf_groups_probed;
    state.cell_hierarchy_stats_recorded = true;
  }
  uint32_t selected_block = std::numeric_limits<uint32_t>::max();
  uint32_t selected_block_support = 0;
  uint32_t selected_cell_support = 0;
  uint32_t selected_cell = std::numeric_limits<uint32_t>::max();
  uint32_t selected_routing_rank = std::numeric_limits<uint32_t>::max();
  float selected_graph_distance = std::numeric_limits<float>::infinity();
  float selected_lower_bound = std::numeric_limits<float>::infinity();
  if (config_.cell_whole_cell_scan &&
      (config_.cell_centroid_routing || config_.gateway_landing_cell_handoff)) {
    while (state.next_routed_cell < state.routed_cell_scratch.size()) {
      const uint32_t routing_rank = state.next_routed_cell++;
      const uint32_t cell = state.routed_cell_scratch[routing_rank];
      const auto& cell_record = index_->cells[cell];
      if (cell_record.block_count == 0 || state.loaded_blocks[cell_record.block_begin] != 0) {
        continue;
      }
      selected_cell = cell;
      selected_routing_rank = routing_rank;
      selected_block = static_cast<uint32_t>(cell_record.block_begin);
      selected_block_support = state.block_support_counts[selected_block];
      selected_cell_support = state.cell_support_counts[cell];
      selected_graph_distance = state.cell_best_graph_distance[cell];
      if (config_.cell_defer_hierarchy_routing) {
        selected_lower_bound = 0.0F;
      } else {
        for (uint64_t block = cell_record.block_begin;
             block < cell_record.block_begin + cell_record.block_count; ++block) {
          selected_lower_bound = std::min(
              selected_lower_bound, block_lower_bound_squared(state, static_cast<uint32_t>(block)));
        }
      }
      break;
    }
  }
  const auto consider_block = [&](uint32_t block) {
    if (!is_mature_cell_block(state, block, candidate_threshold)) {
      return;
    }
    const uint32_t cell = index_->blocks[block].cell_id;
    const uint32_t block_support = state.block_support_counts[block];
    const uint32_t cell_support = state.cell_support_counts[cell];
    const float graph_distance = state.block_best_graph_distance[block];
    const float lower_bound = block_lower_bound_squared(state, block);
    if (selected_block == std::numeric_limits<uint32_t>::max() ||
        std::make_tuple(block_support == 0, std::numeric_limits<uint32_t>::max() - block_support,
                        std::numeric_limits<uint32_t>::max() - cell_support, lower_bound,
                        graph_distance, block) <
            std::make_tuple(selected_block_support == 0,
                            std::numeric_limits<uint32_t>::max() - selected_block_support,
                            std::numeric_limits<uint32_t>::max() - selected_cell_support,
                            selected_lower_bound, selected_graph_distance, selected_block)) {
      selected_block = block;
      selected_block_support = block_support;
      selected_cell_support = cell_support;
      selected_graph_distance = graph_distance;
      selected_lower_bound = lower_bound;
    }
  };
  const bool routed_cell_selected = selected_block != std::numeric_limits<uint32_t>::max();
  for (const uint32_t block : state.supported_block_scratch) {
    if (routed_cell_selected) {
      break;
    }
    if (!config_.cell_whole_cell_scan) {
      consider_block(block);
      continue;
    }
    if (!is_mature_cell_block(state, block, candidate_threshold)) {
      continue;
    }
    const uint32_t cell = index_->blocks[block].cell_id;
    const uint32_t routing_rank = state.cell_routing_ranks[cell];
    const uint32_t cell_support = state.cell_support_counts[cell];
    const float graph_distance = state.cell_best_graph_distance[cell];
    float lower_bound =
        config_.cell_defer_hierarchy_routing ? 0.0F : std::numeric_limits<float>::infinity();
    const auto& cell_record = index_->cells[cell];
    if (!config_.cell_defer_hierarchy_routing) {
      for (uint64_t candidate_block = cell_record.block_begin;
           candidate_block < cell_record.block_begin + cell_record.block_count; ++candidate_block) {
        lower_bound = std::min(lower_bound, block_lower_bound_squared(state, candidate_block));
      }
    }
    if (selected_cell == std::numeric_limits<uint32_t>::max() ||
        std::make_tuple(routing_rank, std::numeric_limits<uint32_t>::max() - cell_support,
                        graph_distance, lower_bound, cell) <
            std::make_tuple(selected_routing_rank,
                            std::numeric_limits<uint32_t>::max() - selected_cell_support,
                            selected_graph_distance, selected_lower_bound, selected_cell)) {
      selected_cell = cell;
      selected_routing_rank = routing_rank;
      selected_block = block;
      selected_block_support = state.block_support_counts[block];
      selected_cell_support = cell_support;
      selected_graph_distance = graph_distance;
      selected_lower_bound = lower_bound;
    }
  }

  const uint32_t block_ordinal = state.cell_blocks;
  ++state.cell_blocks;
  const auto& block = index_->blocks[selected_block];
  const auto& selected_record = index_->cells[block.cell_id];
  const uint64_t first_block =
      config_.cell_whole_cell_scan ? selected_record.block_begin : selected_block;
  const uint64_t block_end = config_.cell_whole_cell_scan
                                 ? selected_record.block_begin + selected_record.block_count
                                 : static_cast<uint64_t>(selected_block) + 1;
  uint64_t selected_node_count = 0;
  for (uint64_t candidate_block = first_block; candidate_block < block_end; ++candidate_block) {
    state.touch_block(static_cast<uint32_t>(candidate_block));
    state.loaded_blocks[candidate_block] = 1;
    selected_node_count += index_->blocks[candidate_block].node_count;
  }
  community_polar_candidate_batch_t batch;
  batch.origin = community_polar_candidate_origin_t::POLAR_CELL;
  batch.object_id = selected_block;
  batch.block_ordinal = block_ordinal;
  batch.lower_bound_squared = selected_lower_bound;
  batch.evidence_distance_squared = selected_graph_distance;
  const uint64_t record_bytes = sizeof(uint32_t) + index_->pq_code_width;
  batch.packet_bytes = selected_node_count * record_bytes;
  batch.packet_record_begin = index_->blocks[first_block].packet_record_begin;
  batch.packet_record_count = selected_node_count;
  batch.insert_cap = static_cast<uint32_t>(
      std::min<uint64_t>({selected_node_count, config_.cell_insert_cap, search_list_size}));
  auto& nodes = state.proposal_node_scratch;
  nodes.clear();
  nodes.reserve(selected_node_count);
  for (uint64_t candidate_block = first_block; candidate_block < block_end; ++candidate_block) {
    const auto& packet_block = index_->blocks[candidate_block];
    for (uint32_t offset = 0; offset < packet_block.node_count; ++offset) {
      const uint64_t packet = packet_block.packet_record_begin + offset;
      nodes.push_back(packet_node_ids_[packet]);
    }
  }
  batch.node_ids = nodes;
  if (stats != nullptr) {
    ++stats->cell_blocks_expanded;
    stats->cell_packet_bytes += batch.packet_bytes;
    if (config_.cell_value_telemetry && block_ordinal < k_cell_value_max_blocks) {
      stats->cell_value_blocks_recorded =
          std::max(stats->cell_value_blocks_recorded, block_ordinal + 1);
      stats->cell_value_block_ids[block_ordinal] = selected_block;
      stats->cell_value_cell_ids[block_ordinal] = block.cell_id;
      stats->cell_value_block_support[block_ordinal] = selected_block_support;
      stats->cell_value_cell_support[block_ordinal] = selected_cell_support;
      stats->cell_value_node_count[block_ordinal] = static_cast<uint32_t>(selected_node_count);
      stats->cell_value_packet_bytes[block_ordinal] = batch.packet_bytes;
      if (std::isfinite(candidate_threshold) && candidate_threshold > 0.0F) {
        stats->cell_value_evidence_threshold_ratio[block_ordinal] =
            selected_graph_distance / candidate_threshold;
        stats->cell_value_lower_bound_threshold_ratio[block_ordinal] =
            selected_lower_bound / candidate_threshold;
      }
    }
  }
  return batch;
}

bool community_polar_search_t::has_direct_cell_work(
    const community_polar_query_state_t& state) const noexcept {
  for (const uint32_t block : state.supported_block_scratch) {
    const uint32_t cell = index_->blocks[block].cell_id;
    if (state.loaded_blocks[block] == 0 && state.cell_candidate_created[cell] != 0) {
      return true;
    }
  }
  return false;
}

bool community_polar_search_t::is_mature_cell_block(const community_polar_query_state_t& state,
                                                    uint32_t block_id,
                                                    float candidate_threshold) const noexcept {
  if (block_id >= index_->blocks.size() || state.loaded_blocks[block_id] != 0 ||
      state.block_support_counts[block_id] < config_.cell_support) {
    return false;
  }
  const uint32_t cell = index_->blocks[block_id].cell_id;
  if (state.cell_candidate_created[cell] == 0) {
    return false;
  }
  const float graph_distance = state.block_best_graph_distance[block_id];
  if (!std::isfinite(graph_distance)) {
    return false;
  }
  return !std::isfinite(candidate_threshold) ||
         graph_distance <= candidate_threshold * config_.cell_evidence_threshold_ratio;
}

void community_polar_search_t::copy_pq_codes(std::span<const uint32_t> node_ids,
                                             std::span<uint8_t> output) const {
  if (index_->packet_payload.empty()) {
    throw std::logic_error("Community-Polar packet PQ payload has been released");
  }
  const size_t required = node_ids.size() * index_->pq_code_width;
  if (output.size() < required) {
    throw std::invalid_argument("Community-Polar PQ output buffer is too small");
  }
  const uint64_t record_bytes = sizeof(uint32_t) + index_->pq_code_width;
  for (size_t position = 0; position < node_ids.size(); ++position) {
    const uint32_t node = node_ids[position];
    if (node >= index_->point_count) {
      throw std::invalid_argument("Community-Polar candidate node is out of range");
    }
    const uint64_t packet = index_->node_to_packet_record[node];
    const size_t source = static_cast<size_t>(packet * record_bytes + sizeof(uint32_t));
    std::memcpy(output.data() + position * index_->pq_code_width,
                index_->packet_payload.data() + source, index_->pq_code_width);
  }
}

void community_polar_search_t::copy_packet_pq_codes(uint64_t packet_begin, uint64_t packet_count,
                                                    std::span<uint8_t> output) const {
  if (index_->packet_payload.empty()) {
    throw std::logic_error("Community-Polar packet PQ payload has been released");
  }
  if (packet_begin > index_->point_count || packet_count > index_->point_count - packet_begin ||
      output.size() < packet_count * index_->pq_code_width) {
    throw std::invalid_argument("Community-Polar packet PQ range is invalid");
  }
  const uint64_t record_bytes = sizeof(uint32_t) + index_->pq_code_width;
  for (uint64_t offset = 0; offset < packet_count; ++offset) {
    const size_t source =
        static_cast<size_t>((packet_begin + offset) * record_bytes + sizeof(uint32_t));
    std::memcpy(output.data() + offset * index_->pq_code_width,
                index_->packet_payload.data() + source, index_->pq_code_width);
  }
}

std::span<const uint32_t>
community_polar_search_t::cell_node_ids(uint32_t cell_id) const noexcept {
  if (cell_id >= index_->cells.size()) {
    return {};
  }
  const auto& cell = index_->cells[cell_id];
  if (cell.block_count == 0 || cell.block_begin >= index_->blocks.size()) {
    return {};
  }
  const uint64_t packet_begin = index_->blocks[cell.block_begin].packet_record_begin;
  if (packet_begin > packet_node_ids_.size() ||
      cell.node_count > packet_node_ids_.size() - packet_begin) {
    return {};
  }
  return std::span<const uint32_t>(packet_node_ids_).subspan(packet_begin, cell.node_count);
}

void community_polar_search_t::compact_for_forced_tree() {
  if (!config_.cell_defer_hierarchy_routing) {
    throw std::logic_error("Forced Cell-tree compaction requires deferred hierarchy routing");
  }
  index_->packet_payload.clear();
  index_->packet_payload.shrink_to_fit();
  index_->node_to_packet_record.clear();
  index_->node_to_packet_record.shrink_to_fit();
}

float community_polar_search_t::block_lower_bound_squared(community_polar_query_state_t& state,
                                                          uint32_t block_id) const {
  const auto& block = index_->blocks[block_id];
  const auto& cell = index_->cells[block.cell_id];
  const uint32_t sector_id = cell_to_sector_[block.cell_id];
  const auto& sector = index_->sectors[sector_id];
  const auto pole =
      std::span<const float>(index_->community_poles)
          .subspan(static_cast<size_t>(cell.community_id) * index_->dimension, index_->dimension);
  const float query_radius = state.community_query_radii[cell.community_id];
  if (sector.sector_id == k_pole_sector_id || query_radius == 0.0F) {
    const float radial =
        query_radius < block.minimum_radius
            ? block.minimum_radius - query_radius
            : query_radius > block.maximum_radius ? query_radius - block.maximum_radius : 0.0F;
    return radial * radial;
  }

  double query_angle = state.sector_query_angles[sector_id];
  if (!std::isfinite(query_angle)) {
    const auto axis =
        std::span<const float>(index_->direction_axes)
            .subspan(static_cast<size_t>(sector.axis_index) * index_->dimension, index_->dimension);
    double dot = 0.0;
    for (uint32_t dimension = 0; dimension < index_->dimension; ++dimension) {
      dot += (static_cast<double>(state.query[dimension]) - pole[dimension]) * axis[dimension];
    }
    const double cosine = std::clamp(dot / query_radius, -1.0, 1.0);
    query_angle = std::acos(cosine);
    state.sector_query_angles[sector_id] = query_angle;
  }
  const double minimum_angle = std::max(0.0, query_angle - block.maximum_angle);
  const double projected_radius = query_radius * std::cos(minimum_angle);
  const double best_radius = std::clamp(projected_radius, static_cast<double>(block.minimum_radius),
                                        static_cast<double>(block.maximum_radius));
  const double lower_squared = query_radius * query_radius + best_radius * best_radius -
                               2.0 * query_radius * best_radius * std::cos(minimum_angle);
  return static_cast<float>(std::max(0.0, lower_squared));
}

uint64_t community_polar_search_t::resident_bytes() const noexcept {
  uint64_t bytes = sizeof(*this) + sizeof(*index_);
  bytes += vector_bytes(index_->node_to_community.size(), sizeof(uint32_t));
  bytes += vector_bytes(index_->node_to_cell.size(), sizeof(uint32_t));
  bytes += vector_bytes(index_->node_to_packet_record.size(), sizeof(uint64_t));
  bytes += vector_bytes(index_->communities.size(), sizeof(community_polar_community_t));
  bytes += vector_bytes(index_->community_poles.size(), sizeof(float));
  bytes += vector_bytes(index_->community_edge_offsets.size(), sizeof(uint64_t));
  bytes += vector_bytes(index_->community_edges.size(), sizeof(community_polar_edge_t));
  bytes += vector_bytes(index_->gateways.size(), sizeof(community_polar_gateway_t));
  bytes += vector_bytes(index_->direction_axes.size(), sizeof(float));
  bytes += vector_bytes(index_->sectors.size(), sizeof(community_polar_sector_t));
  bytes += vector_bytes(index_->cells.size(), sizeof(community_polar_cell_t));
  bytes += vector_bytes(index_->blocks.size(), sizeof(community_polar_block_t));
  bytes += index_->packet_payload.size();
  bytes += vector_bytes(cell_to_sector_.size(), sizeof(uint32_t));
  bytes += vector_bytes(node_to_block_.size(), sizeof(uint32_t));
  bytes += vector_bytes(packet_node_ids_.size(), sizeof(uint32_t));
  bytes += vector_bytes(index_->cell_hierarchy_nodes.size(),
                        sizeof(community_polar_cell_hierarchy_node_t));
  bytes += vector_bytes(index_->cell_hierarchy_roots.size(), sizeof(uint32_t));
  bytes += vector_bytes(index_->cell_hierarchy_children.size(), sizeof(uint32_t));
  bytes += vector_bytes(index_->cell_hierarchy_centroids.size(), sizeof(float));
  bytes +=
      vector_bytes(cell_hierarchy_nodes_.size(), sizeof(community_polar_cell_hierarchy_node_t));
  bytes += vector_bytes(cell_hierarchy_children_.size(), sizeof(uint32_t));
  bytes += vector_bytes(cell_hierarchy_centroids_.size(), sizeof(float));
  bytes += vector_bytes(community_hierarchy_roots_.size(), sizeof(uint32_t));
  bytes += vector_bytes(cell_hierarchy_parents_.size(), sizeof(uint32_t));
  bytes += vector_bytes(cell_hierarchy_leaf_parents_.size(), sizeof(uint32_t));
  bytes += vector_bytes(terminal_route_centroids_.size(), sizeof(float));
  bytes += vector_bytes(terminal_route_cells_.size(), sizeof(uint32_t));
  return bytes;
}

std::vector<uint32_t> community_polar_search_t::important_memgraph_nodes(uint32_t budget) const {
  std::vector<uint32_t> nodes;
  nodes.reserve(budget);
  std::unordered_set<uint32_t> seen;
  auto append = [&](uint32_t node) {
    if (nodes.size() < budget && seen.insert(node).second) {
      nodes.push_back(node);
    }
  };
  for (const auto& community : index_->communities) {
    append(community.navigation_hub_id);
  }
  for (const auto& gateway : index_->gateways) {
    append(gateway.target_node);
  }
  for (const auto& gateway : index_->gateways) {
    append(gateway.source_node);
  }
  for (uint32_t node = 0; node < index_->point_count && nodes.size() < budget; ++node) {
    append(node);
  }
  return nodes;
}

} // namespace powerlaw_ann
