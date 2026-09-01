#include "pq/product_quantizer.h"
#include "search/community_polar_search.h"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace powerlaw_ann {
namespace {

std::shared_ptr<community_polar_index_t> make_index() {
  auto index = std::make_shared<community_polar_index_t>();
  index->config.community_count = 2;
  index->config.gateway_count = 2;
  index->config.gateway_shortlist = 2;
  index->config.polar_direction_count = 1;
  index->config.cell_target_size = 3;
  index->config.block_target_size = 3;
  index->config.spherical_kmeans_iterations = 1;
  index->point_count = 6;
  index->dimension = 2;
  index->pq_code_width = 2;
  index->entry_point_id = 0;
  index->node_to_community = {0, 0, 0, 1, 1, 1};
  index->node_to_cell = {0, 0, 1, 2, 2, 2};
  index->node_to_packet_record = {0, 1, 2, 3, 4, 5};

  community_polar_community_t first;
  first.community_id = 0;
  first.node_count = 3;
  first.navigation_hub_id = 0;
  first.radius = 2.0F;
  first.sector_begin = 0;
  first.sector_count = 1;
  first.cell_begin = 0;
  first.cell_count = 2;
  community_polar_community_t second = first;
  second.community_id = 1;
  second.navigation_hub_id = 3;
  second.sector_begin = 1;
  second.cell_begin = 2;
  second.cell_count = 1;
  index->communities = {first, second};
  index->community_poles = {0.0F, 0.0F, 10.0F, 0.0F};

  index->community_edge_offsets = {0, 1, 1};
  community_polar_edge_t edge;
  edge.source_community = 0;
  edge.target_community = 1;
  edge.directed_cross_edge_count = 2;
  edge.gateway_begin = 0;
  edge.gateway_count = 2;
  index->community_edges = {edge};
  index->gateways = {{1, 3, 3}, {2, 4, 4}};

  community_polar_sector_t first_sector;
  first_sector.community_id = 0;
  first_sector.sector_id = 0;
  first_sector.node_count = 3;
  first_sector.axis_index = 0;
  first_sector.minimum_radius = 0.0F;
  first_sector.maximum_radius = 2.0F;
  first_sector.maximum_angle = 0.25F;
  first_sector.cell_begin = 0;
  first_sector.cell_count = 2;
  community_polar_sector_t second_sector = first_sector;
  second_sector.community_id = 1;
  second_sector.axis_index = 1;
  second_sector.cell_begin = 2;
  second_sector.cell_count = 1;
  index->sectors = {first_sector, second_sector};
  index->direction_axes = {1.0F, 0.0F, 1.0F, 0.0F};

  community_polar_cell_t first_cell;
  first_cell.cell_id = 0;
  first_cell.community_id = 0;
  first_cell.sector_id = 0;
  first_cell.node_count = 2;
  first_cell.minimum_radius = 0.0F;
  first_cell.maximum_radius = 2.0F;
  first_cell.maximum_angle = 0.25F;
  first_cell.block_begin = 0;
  first_cell.block_count = 1;
  community_polar_cell_t radial_cell = first_cell;
  radial_cell.cell_id = 1;
  radial_cell.node_count = 1;
  radial_cell.minimum_radius = 3.0F;
  radial_cell.maximum_radius = 4.0F;
  radial_cell.block_begin = 1;
  community_polar_cell_t second_cell = first_cell;
  second_cell.cell_id = 2;
  second_cell.community_id = 1;
  second_cell.node_count = 3;
  second_cell.block_begin = 2;
  index->cells = {first_cell, radial_cell, second_cell};

  community_polar_block_t first_block;
  first_block.block_id = 0;
  first_block.cell_id = 0;
  first_block.node_count = 2;
  first_block.packet_record_begin = 0;
  first_block.minimum_radius = 0.0F;
  first_block.maximum_radius = 2.0F;
  first_block.maximum_angle = 0.25F;
  community_polar_block_t radial_block = first_block;
  radial_block.block_id = 1;
  radial_block.cell_id = 1;
  radial_block.node_count = 1;
  radial_block.packet_record_begin = 2;
  radial_block.minimum_radius = 3.0F;
  radial_block.maximum_radius = 4.0F;
  community_polar_block_t second_block = first_block;
  second_block.block_id = 2;
  second_block.cell_id = 2;
  second_block.node_count = 3;
  second_block.packet_record_begin = 3;
  index->blocks = {first_block, radial_block, second_block};

  const uint64_t record_bytes = sizeof(uint32_t) + index->pq_code_width;
  index->packet_payload.resize(index->point_count * record_bytes);
  for (uint32_t node = 0; node < index->point_count; ++node) {
    const size_t offset = static_cast<size_t>(node * record_bytes);
    std::memcpy(index->packet_payload.data() + offset, &node, sizeof(node));
    index->packet_payload[offset + sizeof(node)] = static_cast<uint8_t>(node);
    index->packet_payload[offset + sizeof(node) + 1] = static_cast<uint8_t>(node + 10);
  }
  return index;
}

community_polar_search_config_t hybrid_config() {
  community_polar_search_config_t config;
  config.mode = powerann_search_mode_t::HYBRID;
  config.community_skip = true;
  config.cell_expansion = true;
  config.community_expansion_budget = 2;
  config.gateway_score_budget = 4;
  config.cell_block_budget = 2;
  config.cell_support = 2;
  config.cell_insert_cap = 2;
  return config;
}

TEST(CommunityPolarSearchTest, ValidatesModeAndGraphGuardBoundary) {
  community_polar_search_config_t invalid;
  invalid.community_skip = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.community_skip = false;
  invalid.cell_expansion = false;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.gateway_score_batch_cap = 0;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.community_skip = false;
  invalid.community_hub_clique = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.community_skip = false;
  invalid.community_pq_approach_hops = 2;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_min_competitive_fraction = 1.01F;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_candidate_threshold_ratio = 0.0F;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_evidence_threshold_ratio = 0.0F;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_insert_fraction = 1.01F;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_centroid_routing = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.gateway_landing_cell_handoff = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_terminal_convergence = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_terminal_l_aware_approach = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_hierarchical_routing = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_hierarchy_require_persisted = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_hierarchy_graph_guided_routing = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  invalid = hybrid_config();
  invalid.cell_terminal_global_forest_routing = true;
  EXPECT_THROW(validate_community_polar_search_config(invalid), std::invalid_argument);

  EXPECT_TRUE(graph_guard_allows_macro(10, 0.5F, 4));
  EXPECT_FALSE(graph_guard_allows_macro(10, 0.5F, 5));
  EXPECT_FALSE(graph_guard_allows_macro(10, 1.0F, 0));
}

TEST(CommunityPolarSearchTest, TerminalApproachCanScaleDeterministicallyWithSearchListSize) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.community_skip = false;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_terminal_convergence = true;
  config.cell_terminal_approach_hops = 14;
  config.cell_terminal_l_aware_approach = true;
  config.cell_terminal_approach_base_hops = 2;
  config.cell_terminal_approach_l_divisor = 5;
  community_polar_search_t search(index, config);

  EXPECT_EQ(search.terminal_approach_hops(10), 4U);
  EXPECT_EQ(search.terminal_approach_hops(20), 6U);
  EXPECT_EQ(search.terminal_approach_hops(30), 8U);
  EXPECT_EQ(search.terminal_approach_hops(40), 10U);
  EXPECT_EQ(search.terminal_approach_hops(50), 12U);
  EXPECT_EQ(search.terminal_approach_hops(75), 14U);
  EXPECT_EQ(search.terminal_approach_hops(100), 14U);

  config.cell_terminal_low_l_stitch_max_l = 20;
  config.cell_terminal_low_l_stitch_extra_hops = 2;
  community_polar_search_t low_l_search(index, config);
  EXPECT_EQ(low_l_search.terminal_approach_hops(10), 6U);
  EXPECT_EQ(low_l_search.terminal_approach_hops(20), 8U);
  EXPECT_EQ(low_l_search.terminal_approach_hops(30), 8U);

  config.cell_terminal_low_l_stitch_extra_hops = 0;
  EXPECT_THROW(community_polar_search_t(index, config), std::invalid_argument);
  config.cell_terminal_low_l_stitch_max_l = 0;

  config.cell_terminal_l_aware_approach = false;
  community_polar_search_t fixed_search(index, config);
  EXPECT_EQ(fixed_search.terminal_approach_hops(10), 14U);
  EXPECT_EQ(fixed_search.terminal_approach_hops(100), 14U);

  config.cell_terminal_l_aware_approach = true;
  config.cell_terminal_approach_l_divisor = 0;
  EXPECT_THROW(community_polar_search_t(index, config), std::invalid_argument);
}

TEST(CommunityPolarSearchTest, TerminalForestAndStitchingCanScaleWithSearchListSize) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.community_skip = false;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_terminal_convergence = true;
  config.cell_block_budget = 16;
  config.cell_routing_cell_budget = 16;
  config.cell_terminal_l_aware_forest = true;
  config.cell_terminal_forest_base_cells = 1;
  config.cell_terminal_forest_l_divisor = 10;
  config.cell_terminal_refine_hops = 8;
  config.cell_terminal_l_aware_refine = true;
  config.cell_terminal_refine_base_hops = 1;
  config.cell_terminal_refine_l_divisor = 25;
  community_polar_search_t search(index, config);

  EXPECT_EQ(search.terminal_cell_budget(10), 1U);
  EXPECT_EQ(search.terminal_cell_budget(20), 2U);
  EXPECT_EQ(search.terminal_cell_budget(50), 5U);
  EXPECT_EQ(search.terminal_cell_budget(125), 13U);
  EXPECT_EQ(search.terminal_refine_hops(10), 1U);
  EXPECT_EQ(search.terminal_refine_hops(25), 2U);
  EXPECT_EQ(search.terminal_refine_hops(125), 6U);

  config.cell_terminal_low_l_forest_max_l = 20;
  config.cell_terminal_low_l_forest_extra_cells = 1;
  community_polar_search_t low_l_forest_search(index, config);
  EXPECT_EQ(low_l_forest_search.terminal_cell_budget(10), 2U);
  EXPECT_EQ(low_l_forest_search.terminal_cell_budget(20), 3U);
  EXPECT_EQ(low_l_forest_search.terminal_cell_budget(50), 5U);
  config.cell_terminal_low_l_forest_extra_cells = 0;
  EXPECT_THROW(community_polar_search_t(index, config), std::invalid_argument);
  config.cell_terminal_low_l_forest_max_l = 0;

  config.cell_terminal_high_l_forest_min_l = 100;
  config.cell_terminal_high_l_forest_cell_cap = 9;
  config.cell_terminal_high_l_forest_cap_l_divisor = 25;
  community_polar_search_t high_l_forest_search(index, config);
  EXPECT_EQ(high_l_forest_search.terminal_cell_budget(75), 8U);
  EXPECT_EQ(high_l_forest_search.terminal_cell_budget(100), 9U);
  EXPECT_EQ(high_l_forest_search.terminal_cell_budget(125), 8U);
  config.cell_terminal_high_l_forest_cell_cap = 0;
  EXPECT_THROW(community_polar_search_t(index, config), std::invalid_argument);
  config.cell_terminal_high_l_forest_min_l = 0;
  config.cell_terminal_high_l_forest_cap_l_divisor = 0;

  config.cell_terminal_low_l_stitch_max_l = 20;
  config.cell_terminal_low_l_stitch_extra_hops = 2;
  community_polar_search_t low_l_search(index, config);
  EXPECT_EQ(low_l_search.terminal_refine_hops(10), 3U);
  EXPECT_EQ(low_l_search.terminal_refine_hops(20), 3U);
  EXPECT_EQ(low_l_search.terminal_refine_hops(25), 2U);
  config.cell_terminal_low_l_stitch_max_l = 0;
  config.cell_terminal_low_l_stitch_extra_hops = 0;

  config.cell_terminal_forest_l_divisor = 0;
  EXPECT_THROW(community_polar_search_t(index, config), std::invalid_argument);
  config.cell_terminal_forest_l_divisor = 10;
  config.cell_terminal_refine_l_divisor = 0;
  EXPECT_THROW(community_polar_search_t(index, config), std::invalid_argument);
}

TEST(CommunityPolarSearchTest, TerminalForestDoesNotConsumeCellsOutsideFrozenBudget) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.community_skip = false;
  config.cell_block_budget = 3;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_routing_community_budget = 2;
  config.cell_routing_cell_budget = 3;
  config.cell_terminal_convergence = true;
  config.cell_terminal_l_aware_forest = true;
  config.cell_terminal_forest_base_cells = 1;
  config.cell_terminal_forest_l_divisor = 100;
  config.cell_hierarchical_routing = true;
  config.cell_hierarchy_branching = 2;
  config.cell_hierarchy_leaf_size = 1;
  config.cell_hierarchy_probe_budget = 3;
  config.cell_hierarchy_exact_routing = false;
  config.cell_hierarchy_graph_guided_routing = true;
  config.cell_defer_hierarchy_routing = true;
  config.source_cell_batching = true;
  community_polar_search_t search(index, config);

  auto state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  search.observe_source_cell(state, 0, 2.0F, nullptr);
  search.observe_source_cell(state, 2, 1.0F, nullptr);
  search.prepare_terminal_convergence(state, 10);
  ASSERT_TRUE(search.expand_next_cell(state, 10, nullptr).has_value());
  EXPECT_FALSE(search.has_cell_work(state));
  EXPECT_FALSE(search.expand_next_cell(state, 10, nullptr).has_value());
}

TEST(CommunityPolarSearchTest, CentroidRoutingCreatesWholeCellWorkWithoutGraphEvidence) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.community_skip = false;
  config.cell_block_budget = 1;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_routing_community_budget = 1;
  config.cell_routing_cell_budget = 1;
  community_polar_search_t search(index, config);

  auto state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  ASSERT_TRUE(search.has_cell_work(state));
  const auto batch = search.expand_next_cell(state, 50, nullptr);
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->origin, community_polar_candidate_origin_t::POLAR_CELL);
  EXPECT_EQ(std::vector<uint32_t>(batch->node_ids.begin(), batch->node_ids.end()),
            (std::vector<uint32_t>{2}));
  EXPECT_FALSE(search.has_cell_work(state));
}

TEST(CommunityPolarSearchTest, CentroidRoutingConsumesPreorderedCellsOnce) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.community_skip = false;
  config.cell_block_budget = 2;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_routing_community_budget = 1;
  config.cell_routing_cell_budget = 2;
  community_polar_search_t search(index, config);

  auto state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  ASSERT_TRUE(search.has_cell_work(state));
  const auto first = search.expand_next_cell(state, 50, nullptr);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(std::vector<uint32_t>(first->node_ids.begin(), first->node_ids.end()),
            (std::vector<uint32_t>{2}));
  ASSERT_TRUE(search.has_cell_work(state));
  const auto second = search.expand_next_cell(state, 50, nullptr);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(std::vector<uint32_t>(second->node_ids.begin(), second->node_ids.end()),
            (std::vector<uint32_t>{0, 1}));
  EXPECT_FALSE(search.has_cell_work(state));
}

TEST(CommunityPolarSearchTest, HierarchicalRoutingPreservesNearestLeafOrder) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.community_skip = false;
  config.cell_block_budget = 2;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_routing_community_budget = 1;
  config.cell_routing_cell_budget = 2;
  config.cell_hierarchical_routing = true;
  config.cell_hierarchy_branching = 2;
  config.cell_hierarchy_leaf_size = 1;
  config.cell_hierarchy_probe_budget = 2;
  community_polar_search_t search(index, config);

  auto state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  query_stats_t stats;
  const auto first = search.expand_next_cell(state, 50, &stats);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(std::vector<uint32_t>(first->node_ids.begin(), first->node_ids.end()),
            (std::vector<uint32_t>{2}));
  const auto second = search.expand_next_cell(state, 50, &stats);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(std::vector<uint32_t>(second->node_ids.begin(), second->node_ids.end()),
            (std::vector<uint32_t>{0, 1}));
  EXPECT_GT(stats.cell_hierarchy_nodes_scored, 0U);
  EXPECT_EQ(stats.cell_hierarchy_leaf_groups_probed, 2U);
}

TEST(CommunityPolarSearchTest, GraphGuidedRoutingDescendsTreeToApproachCell) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.cell_block_budget = 1;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_routing_community_budget = 1;
  config.cell_routing_cell_budget = 1;
  config.cell_terminal_convergence = true;
  config.cell_terminal_frontier_cells = 0;
  config.cell_hierarchical_routing = true;
  config.cell_hierarchy_branching = 2;
  config.cell_hierarchy_leaf_size = 1;
  config.cell_hierarchy_probe_budget = 1;
  config.cell_hierarchy_exact_routing = false;
  config.cell_hierarchy_graph_guided_routing = true;
  config.cell_defer_hierarchy_routing = true;
  config.source_cell_batching = true;
  community_polar_search_t search(index, config);

  auto state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  query_stats_t stats;
  search.observe_base_expansion(state, 2, &stats);
  search.observe_source_cell(state, 2, 1.0F, &stats);
  search.observe_base_expansion(state, 0, &stats);
  search.observe_source_cell(state, 0, 2.0F, &stats);
  search.prepare_terminal_convergence(state);
  const auto selected = search.expand_next_cell(state, 50, &stats);
  ASSERT_TRUE(selected.has_value());
  EXPECT_EQ(std::vector<uint32_t>(selected->node_ids.begin(), selected->node_ids.end()),
            (std::vector<uint32_t>{2}));
  EXPECT_GT(stats.cell_hierarchy_nodes_scored, 0U);
  EXPECT_EQ(stats.cell_hierarchy_leaf_groups_probed, 1U);
}

TEST(CommunityPolarSearchTest, GraphGuidedRoutingPrefersRepeatedCellSupport) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.cell_block_budget = 1;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_routing_community_budget = 1;
  config.cell_routing_cell_budget = 1;
  config.cell_terminal_convergence = true;
  config.cell_terminal_frontier_cells = 0;
  config.cell_hierarchical_routing = true;
  config.cell_hierarchy_branching = 2;
  config.cell_hierarchy_leaf_size = 1;
  config.cell_hierarchy_probe_budget = 1;
  config.cell_hierarchy_exact_routing = false;
  config.cell_hierarchy_graph_guided_routing = true;
  config.cell_defer_hierarchy_routing = true;
  config.source_cell_batching = true;
  community_polar_search_t search(index, config);

  auto state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  query_stats_t stats;
  search.observe_base_expansion(state, 2, &stats);
  search.observe_source_cell(state, 2, 0.1F, &stats);
  search.observe_base_expansion(state, 0, &stats);
  search.observe_source_cell(state, 0, 2.0F, &stats);
  search.observe_base_expansion(state, 1, &stats);
  search.observe_source_cell(state, 1, 2.5F, &stats);
  search.prepare_terminal_convergence(state);

  const auto selected = search.expand_next_cell(state, 50, &stats);
  ASSERT_TRUE(selected.has_value());
  EXPECT_EQ(std::vector<uint32_t>(selected->node_ids.begin(), selected->node_ids.end()),
            (std::vector<uint32_t>{0, 1}));
  EXPECT_EQ(stats.cell_hierarchy_leaf_groups_probed, 1U);
}

TEST(CommunityPolarSearchTest, GraphGuidedRoutingCanBatchTwoApproachCellsThroughTree) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.cell_block_budget = 2;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_routing_community_budget = 1;
  config.cell_routing_cell_budget = 2;
  config.cell_terminal_convergence = true;
  config.cell_terminal_frontier_cells = 0;
  config.cell_hierarchical_routing = true;
  config.cell_hierarchy_branching = 2;
  config.cell_hierarchy_leaf_size = 1;
  config.cell_hierarchy_probe_budget = 2;
  config.cell_hierarchy_exact_routing = false;
  config.cell_hierarchy_graph_guided_routing = true;
  config.cell_defer_hierarchy_routing = true;
  community_polar_search_t search(index, config);

  auto state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  query_stats_t stats;
  search.observe_base_expansion(state, 2, &stats);
  search.observe_source_cell(state, 2, 1.0F, &stats);
  search.observe_base_expansion(state, 0, &stats);
  search.observe_source_cell(state, 0, 2.0F, &stats);
  search.prepare_terminal_convergence(state);
  const auto first = search.expand_next_cell(state, 50, &stats);
  ASSERT_TRUE(first.has_value());
  const auto first_nodes = std::vector<uint32_t>(first->node_ids.begin(), first->node_ids.end());
  const auto second = search.expand_next_cell(state, 50, &stats);
  ASSERT_TRUE(second.has_value());
  const auto second_nodes = std::vector<uint32_t>(second->node_ids.begin(), second->node_ids.end());
  EXPECT_TRUE(
      (first_nodes == std::vector<uint32_t>{2} && second_nodes == std::vector<uint32_t>{0, 1}) ||
      (first_nodes == std::vector<uint32_t>{0, 1} && second_nodes == std::vector<uint32_t>{2}));
  EXPECT_GT(stats.cell_hierarchy_nodes_scored, 0U);
  EXPECT_EQ(stats.cell_hierarchy_leaf_groups_probed, 2U);
}

TEST(CommunityPolarSearchTest, GlobalTerminalForestDoesNotTreatCommunitySkipAsSearchBoundary) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.cell_block_budget = 1;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_routing_community_budget = 2;
  config.cell_routing_cell_budget = 1;
  config.cell_terminal_convergence = true;
  config.cell_terminal_frontier_cells = 0;
  config.cell_hierarchical_routing = true;
  config.cell_hierarchy_branching = 2;
  config.cell_hierarchy_leaf_size = 1;
  config.cell_hierarchy_probe_budget = 1;
  config.cell_hierarchy_exact_routing = false;
  config.cell_defer_hierarchy_routing = true;

  community_polar_search_t restricted(index, config);
  auto restricted_state = restricted.begin_query(std::vector<float>{10.9F, 0.0F});
  restricted.observe_base_expansion(restricted_state, 0, nullptr);
  restricted.prepare_terminal_convergence(restricted_state, 10);
  const auto restricted_cell = restricted.expand_next_cell(restricted_state, 10, nullptr);
  ASSERT_TRUE(restricted_cell.has_value());
  EXPECT_NE(std::vector<uint32_t>(restricted_cell->node_ids.begin(),
                                  restricted_cell->node_ids.end()),
            (std::vector<uint32_t>{3, 4, 5}));

  config.cell_terminal_global_forest_routing = true;
  community_polar_search_t global(index, config);
  auto global_state = global.begin_query(std::vector<float>{10.9F, 0.0F});
  global.observe_base_expansion(global_state, 0, nullptr);
  global.prepare_terminal_convergence(global_state, 10);
  const auto global_cell = global.expand_next_cell(global_state, 10, nullptr);
  ASSERT_TRUE(global_cell.has_value());
  EXPECT_EQ(std::vector<uint32_t>(global_cell->node_ids.begin(), global_cell->node_ids.end()),
            (std::vector<uint32_t>{3, 4, 5}));
}

TEST(CommunityPolarSearchTest, CommunityExpansionReturnsOnlyStoredRealGateways) {
  community_polar_search_t search(make_index(), hybrid_config());
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;
  search.observe_base_expansion(state, 0, &stats);
  const auto batch = search.expand_next_community(state, &stats);
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->origin, community_polar_candidate_origin_t::COMMUNITY_GATEWAY);
  EXPECT_EQ(std::vector<uint32_t>(batch->node_ids.begin(), batch->node_ids.end()),
            (std::vector<uint32_t>{3, 4}));
  EXPECT_EQ(stats.community_meta_expansions, 1U);
  EXPECT_EQ(stats.community_superedges_considered, 1U);
  EXPECT_TRUE(search.has_community_work(state));
}

TEST(CommunityPolarSearchTest, CommunityHubCliqueUsesPersistedHubsWithoutChangingBaseEdges) {
  auto index = make_index();
  const auto original_edges = index->community_edges;
  const auto original_gateways = index->gateways;
  auto config = hybrid_config();
  config.community_hub_clique = true;
  community_polar_search_t search(index, config);
  auto state = search.begin_query(std::vector<float>{10.0F, 0.0F});
  query_stats_t stats;
  search.observe_base_expansion(state, 0, &stats);

  const auto batch = search.expand_next_community(state, &stats);

  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(std::vector<uint32_t>(batch->node_ids.begin(), batch->node_ids.end()),
            (std::vector<uint32_t>{3}));
  EXPECT_TRUE(batch->gateway_proposals.empty());
  EXPECT_EQ(index->community_edges.size(), original_edges.size());
  EXPECT_EQ(index->gateways.size(), original_gateways.size());
  EXPECT_EQ(index->community_edges.front().target_community,
            original_edges.front().target_community);
  EXPECT_EQ(index->gateways.front().target_node, original_gateways.front().target_node);
}

TEST(CommunityPolarSearchTest, CommunityExpansionHonorsPerExpansionGatewayBatchCap) {
  auto config = hybrid_config();
  config.gateway_score_batch_cap = 1;
  community_polar_search_t search(make_index(), config);
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;
  search.observe_base_expansion(state, 0, &stats);

  const auto batch = search.expand_next_community(state, &stats);

  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(std::vector<uint32_t>(batch->node_ids.begin(), batch->node_ids.end()),
            (std::vector<uint32_t>{3}));
  EXPECT_EQ(stats.community_meta_expansions, 1U);
}

TEST(CommunityPolarSearchTest, GatewayHandoffStartsAtEntryCommunityAndRoutesLandingCells) {
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  auto config = hybrid_config();
  config.cell_whole_cell_scan = true;
  config.gateway_landing_cell_handoff = true;
  config.cell_block_budget = 2;
  config.cell_routing_cell_budget = 2;
  config.gateway_direct_insert_cap = 1;
  community_polar_search_t search(index, config);
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;

  ASSERT_TRUE(search.has_community_work(state));
  const auto gateways = search.expand_next_community(state, &stats);
  ASSERT_TRUE(gateways.has_value());
  EXPECT_EQ(gateways->insert_cap, 1U);
  EXPECT_TRUE(search.observe_gateway_landing_candidate(state, 3, 1.0F, &stats));
  EXPECT_FALSE(search.observe_gateway_landing_candidate(state, 4, 2.0F, &stats));
  EXPECT_EQ(stats.gateway_landing_cells_routed, 1U);

  ASSERT_TRUE(search.has_cell_work(state));
  const auto landing = search.expand_next_cell(state, 20, nullptr);
  ASSERT_TRUE(landing.has_value());
  EXPECT_EQ(std::vector<uint32_t>(landing->node_ids.begin(), landing->node_ids.end()),
            (std::vector<uint32_t>{3, 4, 5}));
}

TEST(CommunityPolarSearchTest, CellSupportCountsDistinctExpandedSourcesOnce) {
  auto config = hybrid_config();
  config.cell_value_telemetry = true;
  community_polar_search_t search(make_index(), config);
  auto state = search.begin_query(std::vector<float>{3.5F, 0.0F});
  query_stats_t stats;
  search.observe_base_expansion(state, 0, &stats);
  const std::vector<uint32_t> first_evidence = {1};
  const std::vector<float> first_distances = {4.0F};
  search.observe_base_candidates(state, 0, first_evidence, first_distances, &stats);
  search.observe_base_expansion(state, 0, &stats);
  search.observe_base_candidates(state, 0, first_evidence, first_distances, &stats);
  EXPECT_FALSE(search.has_cell_work(state));
  EXPECT_EQ(stats.cell_candidates_created, 0U);

  search.observe_base_expansion(state, 1, &stats);
  const std::vector<uint32_t> second_evidence = {0};
  const std::vector<float> second_distances = {3.0F};
  search.observe_base_candidates(state, 1, second_evidence, second_distances, &stats);
  ASSERT_TRUE(search.has_cell_work(state));
  const auto batch = search.expand_next_cell(state, 50, &stats);
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->origin, community_polar_candidate_origin_t::POLAR_CELL);
  EXPECT_EQ(batch->block_ordinal, 0U);
  EXPECT_EQ(std::vector<uint32_t>(batch->node_ids.begin(), batch->node_ids.end()),
            (std::vector<uint32_t>{0, 1}));
  EXPECT_EQ(batch->packet_bytes, 12U);
  EXPECT_EQ(stats.cell_candidates_created, 1U);
  EXPECT_EQ(stats.cell_blocks_expanded, 1U);
  EXPECT_EQ(stats.cell_value_blocks_recorded, 1U);
  EXPECT_EQ(stats.cell_value_block_ids[0], batch->object_id);
  EXPECT_EQ(stats.cell_value_node_count[0], batch->node_ids.size());
  EXPECT_EQ(stats.cell_value_block_support[0], 2U);
}

TEST(CommunityPolarSearchTest, SourceCellBatchingActivatesOneOwnedBlockInConstantWork) {
  auto config = hybrid_config();
  config.source_cell_batching = true;
  community_polar_search_t search(make_index(), config);
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;

  search.observe_source_cell(state, 0, 1.0F, &stats);
  search.observe_source_cell(state, 0, 0.5F, &stats);
  ASSERT_TRUE(search.has_cell_work(state, 10.0F));
  const auto batch = search.expand_next_cell(state, 50, 10.0F, &stats);
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->object_id, 0U);
  EXPECT_EQ(std::vector<uint32_t>(batch->node_ids.begin(), batch->node_ids.end()),
            (std::vector<uint32_t>{0, 1}));
  EXPECT_EQ(stats.cell_candidates_created, 1U);
  EXPECT_EQ(stats.cell_blocks_expanded, 1U);
}

TEST(CommunityPolarSearchTest, CellBlockWaitsForDistinctEvidenceWithoutSpendingBudget) {
  community_polar_search_t search(make_index(), hybrid_config());
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;
  const std::vector<uint32_t> first_evidence = {0};
  const std::vector<float> first_distances = {6.0F};
  search.observe_base_candidates(state, 0, first_evidence, first_distances, &stats);
  EXPECT_FALSE(search.has_cell_work(state, 10.0F));
  EXPECT_FALSE(search.expand_next_cell(state, 50, 10.0F, &stats).has_value());
  EXPECT_EQ(stats.cell_blocks_expanded, 0U);

  const std::vector<uint32_t> second_evidence = {1};
  const std::vector<float> second_distances = {5.0F};
  search.observe_base_candidates(state, 1, second_evidence, second_distances, &stats);
  EXPECT_TRUE(search.has_cell_work(state, 10.0F));
  const auto batch = search.expand_next_cell(state, 50, 10.0F, &stats);
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(batch->object_id, 0U);
  EXPECT_EQ(stats.cell_blocks_expanded, 1U);
}

TEST(CommunityPolarSearchTest, CellBlockCanMatureAfterCloserGraphEvidence) {
  community_polar_search_t search(make_index(), hybrid_config());
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;
  const std::vector<uint32_t> distant_evidence = {0};
  const std::vector<float> distant_distances = {9.0F};
  search.observe_base_candidates(state, 0, distant_evidence, distant_distances, &stats);
  search.observe_base_candidates(state, 1, distant_evidence, distant_distances, &stats);
  EXPECT_FALSE(search.has_cell_work(state, 10.0F));
  EXPECT_FALSE(search.expand_next_cell(state, 50, 10.0F, &stats).has_value());
  EXPECT_EQ(stats.cell_blocks_expanded, 0U);

  const std::vector<uint32_t> closer_evidence = {1};
  const std::vector<float> closer_distances = {7.0F};
  search.observe_base_candidates(state, 2, closer_evidence, closer_distances, &stats);
  EXPECT_TRUE(search.has_cell_work(state, 10.0F));
  EXPECT_TRUE(search.expand_next_cell(state, 50, 10.0F, &stats).has_value());
  EXPECT_EQ(stats.cell_blocks_expanded, 1U);
}

TEST(CommunityPolarSearchTest, CellBlockCanWaitForFiniteTopLThreshold) {
  auto config = hybrid_config();
  config.cell_require_full_frontier = true;
  community_polar_search_t search(make_index(), config);
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;
  const std::vector<uint32_t> evidence_nodes = {0};
  const std::vector<float> evidence_distances = {5.0F};
  search.observe_base_candidates(state, 0, evidence_nodes, evidence_distances, &stats);
  search.observe_base_candidates(state, 1, evidence_nodes, evidence_distances, &stats);

  EXPECT_FALSE(search.has_cell_work(state, std::numeric_limits<float>::infinity()));
  EXPECT_FALSE(search.expand_next_cell(state, 50, std::numeric_limits<float>::infinity(), &stats)
                   .has_value());
  EXPECT_EQ(stats.cell_blocks_expanded, 0U);

  EXPECT_TRUE(search.has_cell_work(state, 10.0F));
  EXPECT_TRUE(search.expand_next_cell(state, 50, 10.0F, &stats).has_value());
  EXPECT_EQ(stats.cell_blocks_expanded, 1U);
}

TEST(CommunityPolarSearchTest, SourceOnlySupportCannotMatureCellBlock) {
  community_polar_search_t search(make_index(), hybrid_config());
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;
  search.observe_base_candidates(state, 0, {}, {}, &stats);
  search.observe_base_candidates(state, 1, {}, {}, &stats);
  EXPECT_FALSE(search.has_cell_work(state));
  EXPECT_FALSE(search.expand_next_cell(state, 50, &stats).has_value());
  EXPECT_EQ(stats.cell_blocks_expanded, 0U);
}

TEST(CommunityPolarSearchTest, PacketBatchCodesMatchScalarLookup) {
  community_polar_search_t search(make_index(), hybrid_config());
  const std::vector<uint32_t> nodes = {1, 4};
  std::vector<uint8_t> codes(nodes.size() * search.pq_code_width());
  search.copy_pq_codes(nodes, codes);
  EXPECT_EQ(codes, (std::vector<uint8_t>{1, 11, 4, 14}));

  std::vector<float> table(2 * 256, 0.0F);
  for (uint32_t value = 0; value < 256; ++value) {
    table[value] = static_cast<float>(value);
    table[256 + value] = static_cast<float>(2 * value);
  }
  std::vector<float> batch_distances(nodes.size());
  pq_dist_lookup(codes.data(), nodes.size(), search.pq_code_width(), table.data(),
                 batch_distances.data());
  EXPECT_FLOAT_EQ(batch_distances[0], table[1] + table[256 + 11]);
  EXPECT_FLOAT_EQ(batch_distances[1], table[4] + table[256 + 14]);
}

TEST(CommunityPolarSearchTest, ForcedTreeCompactionRetainsPacketOrderAndReleasesPqPayload) {
  auto config = hybrid_config();
  config.community_skip = false;
  config.gateway_landing_cell_handoff = false;
  config.cell_whole_cell_scan = true;
  config.cell_centroid_routing = true;
  config.cell_hierarchical_routing = true;
  config.cell_defer_hierarchy_routing = true;
  config.cell_terminal_convergence = true;
  config.cell_terminal_frontier_cells = 0;
  config.cell_routing_cell_budget = 1;
  config.cell_hierarchy_probe_budget = 1;
  config.cell_block_budget = 1;
  auto index = make_index();
  index->config.cell_partition = community_cell_partition_t::GRAPH_LOCAL;
  index->direction_axes = {0.5F, 0.0F, 3.0F, 0.0F, 11.0F, 0.0F};
  community_polar_search_t search(index, config);

  auto before_state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  search.prepare_terminal_convergence(before_state);
  const auto before = search.expand_next_cell(before_state, 10, nullptr);
  ASSERT_TRUE(before.has_value());
  const std::vector<uint32_t> expected(before->node_ids.begin(), before->node_ids.end());
  const uint64_t resident_before = search.resident_bytes();

  search.compact_for_forced_tree();
  EXPECT_LT(search.resident_bytes(), resident_before);
  std::vector<uint8_t> released_codes(search.pq_code_width());
  EXPECT_THROW(search.copy_packet_pq_codes(0, 1, released_codes), std::logic_error);

  auto after_state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  search.prepare_terminal_convergence(after_state);
  const auto after = search.expand_next_cell(after_state, 10, nullptr);
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(std::vector<uint32_t>(after->node_ids.begin(), after->node_ids.end()), expected);
}

TEST(CommunityPolarSearchTest, ForcedTreeCompactionAllowsCommunitySkipMetadata) {
  auto config = hybrid_config();
  config.community_skip = true;
  config.cell_defer_hierarchy_routing = true;
  community_polar_search_t search(make_index(), config);
  const uint64_t resident_before = search.resident_bytes();

  EXPECT_NO_THROW(search.compact_for_forced_tree());
  EXPECT_LT(search.resident_bytes(), resident_before);
  EXPECT_EQ(search.point_count(), 6U);
  EXPECT_FALSE(search.cell_node_ids(0).empty());
  auto state = search.begin_query(std::vector<float>{3.1F, 0.0F});
  query_stats_t stats;
  EXPECT_NO_THROW(search.observe_source_cell(state, 0, 1.0F, &stats));
}

TEST(CommunityPolarSearchTest, CellOracleSeparatesSelectionFromGeometricOpportunity) {
  community_polar_search_t search(make_index(), hybrid_config());
  auto state = search.begin_query(std::vector<float>{3.5F, 0.0F});
  query_stats_t stats;
  const std::vector<uint32_t> evidence_nodes = {0, 2};
  const std::vector<float> evidence_distances = {2.0F, 1.0F};
  search.observe_base_expansion(state, 0, &stats);
  search.observe_base_candidates(state, 0, evidence_nodes, evidence_distances, &stats);
  search.observe_base_expansion(state, 2, &stats);
  search.observe_base_candidates(state, 2, evidence_nodes, evidence_distances, &stats);
  ASSERT_TRUE(search.has_cell_work(state));
  search.begin_cell_oracle(state, &stats);
  const auto batch = search.expand_next_cell(state, 50, &stats);
  ASSERT_TRUE(batch.has_value());
  search.record_cell_oracle_selection(state, batch->object_id, batch->block_ordinal);

  for (const uint32_t node : {0U, 1U, 2U, 3U, 4U}) {
    search.observe_cell_oracle_expansion(state, node);
  }
  search.finish_cell_oracle(state, &stats);

  EXPECT_EQ(stats.cell_oracle_triggered, 1U);
  EXPECT_EQ(stats.cell_oracle_eligible_blocks, 2U);
  EXPECT_EQ(stats.cell_oracle_future_expansions, 5U);
  EXPECT_EQ(stats.cell_oracle_future_distinct_cells, 3U);
  EXPECT_EQ(stats.cell_oracle_future_distinct_blocks, 3U);
  EXPECT_EQ(stats.cell_oracle_selected_block_id, 1U);
  EXPECT_EQ(stats.cell_oracle_selected_block_hits, 1U);
  EXPECT_EQ(stats.cell_value_future_expansions[0], 1U);
  EXPECT_EQ(stats.cell_oracle_best_eligible_block_id, 0U);
  EXPECT_EQ(stats.cell_oracle_best_eligible_block_hits, 2U);
  EXPECT_EQ(stats.cell_oracle_best_global_block_id, 0U);
  EXPECT_EQ(stats.cell_oracle_top1_cell_hits, 2U);
  EXPECT_EQ(stats.cell_oracle_top2_cell_hits, 4U);
  EXPECT_EQ(stats.cell_oracle_top4_cell_hits, 5U);
  EXPECT_EQ(stats.cell_oracle_top1_block_hits, 2U);
  EXPECT_EQ(stats.cell_oracle_top2_block_hits, 4U);
  EXPECT_EQ(stats.cell_oracle_top4_block_hits, 5U);
}

TEST(CommunityPolarSearchTest, BaseRediscoveryTransitionsMacroOrigin) {
  community_polar_search_t search(make_index(), hybrid_config());
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  state.mark_macro_inserted(3, community_polar_candidate_origin_t::COMMUNITY_GATEWAY);
  EXPECT_EQ(state.origin(3), community_polar_candidate_origin_t::COMMUNITY_GATEWAY);
  state.mark_base_reached(3);
  EXPECT_EQ(state.origin(3), community_polar_candidate_origin_t::BASE_EDGE);
}

TEST(CommunityPolarSearchTest, BudgetsAndGraphOnlyCellsFallBackWithoutMoreWork) {
  auto index = make_index();
  index->communities[0].graph_only = true;
  auto config = hybrid_config();
  config.community_expansion_budget = 1;
  config.cell_block_budget = 1;
  community_polar_search_t search(index, config);
  auto state = search.begin_query(std::vector<float>{0.0F, 0.0F});
  query_stats_t stats;
  search.observe_base_expansion(state, 0, &stats);
  search.observe_base_candidates(state, 0, {}, {}, &stats);
  search.observe_base_expansion(state, 1, &stats);
  search.observe_base_candidates(state, 1, {}, {}, &stats);
  EXPECT_FALSE(search.has_cell_work(state));
  ASSERT_TRUE(search.expand_next_community(state, &stats).has_value());
  EXPECT_FALSE(search.has_community_work(state));
  EXPECT_EQ(stats.community_meta_expansions, 1U);
}

TEST(CommunityPolarSearchTest, QueryStatesAreIndependent) {
  community_polar_search_t search(make_index(), hybrid_config());
  auto first = search.begin_query(std::vector<float>{0.0F, 0.0F});
  auto second = search.begin_query(std::vector<float>{10.0F, 0.0F});
  query_stats_t first_stats;
  query_stats_t second_stats;
  const std::vector<uint32_t> first_evidence = {0};
  const std::vector<float> first_distances = {1.0F};
  search.observe_base_expansion(first, 0, &first_stats);
  search.observe_base_candidates(first, 0, first_evidence, first_distances, &first_stats);
  search.observe_base_expansion(first, 1, &first_stats);
  search.observe_base_candidates(first, 1, first_evidence, first_distances, &first_stats);
  search.observe_base_expansion(second, 3, &second_stats);
  const std::vector<uint32_t> second_evidence = {3};
  const std::vector<float> second_distances = {1.0F};
  search.observe_base_candidates(second, 3, second_evidence, second_distances, &second_stats);
  EXPECT_TRUE(search.has_cell_work(first));
  EXPECT_FALSE(search.has_cell_work(second));
  EXPECT_EQ(first_stats.cell_candidates_created, 1U);
  EXPECT_EQ(second_stats.cell_candidates_created, 0U);
}

} // namespace
} // namespace powerlaw_ann
