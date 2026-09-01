#include "common/ann_error.h"
#include "common/defaults.h"
#include "common/utils.h"
#include "index/community_polar_index.h"
#include "power_ann.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

class temp_dir_t {
public:
  temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_disk_index_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

void write_dataset(const std::filesystem::path& path) {
  constexpr uint32_t num_points = 300;
  constexpr uint32_t dimensions = 4;
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
  output.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
  for (uint32_t point = 0; point < num_points; ++point) {
    const float vector[dimensions] = {
        static_cast<float>(point),
        static_cast<float>((point * 7) % 31),
        std::sin(static_cast<float>(point) * 0.1F),
        std::cos(static_cast<float>(point) * 0.1F),
    };
    output.write(reinterpret_cast<const char*>(vector), sizeof(vector));
  }
}

void write_queries(const std::filesystem::path& path) {
  constexpr uint32_t num_queries = 4;
  constexpr uint32_t dimensions = 4;
  constexpr uint32_t points[num_queries] = {7, 83, 161, 247};
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&num_queries), sizeof(num_queries));
  output.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
  for (const uint32_t point : points) {
    const float vector[dimensions] = {
        static_cast<float>(point),
        static_cast<float>((point * 7) % 31),
        std::sin(static_cast<float>(point) * 0.1F),
        std::cos(static_cast<float>(point) * 0.1F),
    };
    output.write(reinterpret_cast<const char*>(vector), sizeof(vector));
  }
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

powerlaw_ann::diskann_disk_index_config_t
make_disk_config(const std::filesystem::path& data_path,
                 const std::filesystem::path& index_prefix) {
  powerlaw_ann::diskann_disk_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_prefix;
  config.search_dram_budget_gb = 0.01;
  config.build_dram_budget_gb = 0.1;
  config.max_degree = 8;
  config.build_list_size = 16;
  config.num_threads = 1;
  config.quantized_dimension = 1;
  return config;
}

TEST(DiskannDiskIndexTest, EnabledPowerannBuildPreservesDiskannArtifactMetadata) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  write_dataset(data_path);

  powerlaw_ann::diskann_disk_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_prefix;
  config.search_dram_budget_gb = 0.01;
  config.build_dram_budget_gb = 0.1;
  config.max_degree = 8;
  config.build_list_size = 16;
  config.num_threads = 1;
  config.quantized_dimension = 1;

  powerlaw_ann::power_ann_config_t power_ann_config;
  power_ann_config.enable_powerann = true;
  power_ann_config.hub_selection.mode =
      powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  power_ann_config.hub_selection.importance_mass = 0.8;
  powerlaw_ann::power_ann_t power_ann(power_ann_config);
  ASSERT_NO_THROW(power_ann.build_diskann_index(config));

  const auto disk_index = index_prefix.string() + "_disk.index";
  const auto compressed_vectors = index_prefix.string() + "_pq_compressed.bin";
  const auto pq_pivots = index_prefix.string() + "_pq_pivots.bin";
  const auto sample_data = index_prefix.string() + "_sample_data.bin";
  const auto rcni_output = index_prefix.string() + ".rcni.csv";
  const auto candidate_hubs_output = index_prefix.string() + ".candidate_hubs.csv";
  ASSERT_TRUE(std::filesystem::is_regular_file(disk_index));
  ASSERT_TRUE(std::filesystem::is_regular_file(compressed_vectors));
  ASSERT_TRUE(std::filesystem::is_regular_file(pq_pivots));
  ASSERT_TRUE(std::filesystem::is_regular_file(sample_data));
  ASSERT_TRUE(std::filesystem::is_regular_file(rcni_output));
  ASSERT_TRUE(std::filesystem::is_regular_file(candidate_hubs_output));

  size_t compressed_points = 0;
  size_t compressed_dimensions = 0;
  powerlaw_ann::get_bin_metadata(compressed_vectors, compressed_points, compressed_dimensions);
  EXPECT_EQ(compressed_points, 300U);
  EXPECT_EQ(compressed_dimensions, 1U);

  std::unique_ptr<uint64_t[]> disk_metadata;
  size_t metadata_rows = 0;
  size_t metadata_columns = 0;
  powerlaw_ann::load_bin<uint64_t>(disk_index, disk_metadata, metadata_rows, metadata_columns);
  ASSERT_EQ(metadata_rows, 9U);
  ASSERT_EQ(metadata_columns, 1U);
  EXPECT_EQ(disk_metadata[0], 300U);
  EXPECT_EQ(disk_metadata[1], 4U);
  EXPECT_EQ(disk_metadata[7], 0U);
  EXPECT_EQ(disk_metadata[8], std::filesystem::file_size(disk_index));
  EXPECT_EQ(std::filesystem::file_size(disk_index) % powerlaw_ann::defaults::SECTOR_LEN, 0U);
}

TEST(DiskannDiskIndexTest, CommunityPolarBuildPublishesCompleteValidatedSidecar) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_prefix = temp_dir.path() / "community";
  write_dataset(data_path);

  powerlaw_ann::power_ann_config_t power_ann_config;
  power_ann_config.enable_powerann = true;
  power_ann_config.hub_selection.mode =
      powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  power_ann_config.hub_selection.importance_mass = 0.8;
  power_ann_config.community_polar_build.community_count = 4;
  power_ann_config.community_polar_build.partition_threads = 1;
  power_ann_config.community_polar_build.gateway_count = 4;
  power_ann_config.community_polar_build.gateway_shortlist = 8;
  power_ann_config.community_polar_build.polar_direction_count = 16;
  power_ann_config.community_polar_build.cell_target_size = 16;
  power_ann_config.community_polar_build.block_target_size = 8;
  powerlaw_ann::power_ann_t power_ann(power_ann_config);
  power_ann.build_diskann_index(make_disk_config(data_path, index_prefix));

  EXPECT_TRUE(std::filesystem::is_regular_file(index_prefix.string() + ".rcni.csv"));
  EXPECT_TRUE(std::filesystem::is_regular_file(index_prefix.string() + ".candidate_hubs.csv"));

  const auto sidecar = powerlaw_ann::load_community_polar_index(
      powerlaw_ann::make_community_polar_index_path(index_prefix));
  EXPECT_NO_THROW(powerlaw_ann::validate_community_polar_artifacts(sidecar, index_prefix));
  ASSERT_EQ(sidecar.point_count, 300U);
  ASSERT_EQ(sidecar.dimension, 4U);
  ASSERT_EQ(sidecar.communities.size(), 4U);
  ASSERT_EQ(sidecar.node_to_community.size(), 300U);
  ASSERT_EQ(sidecar.node_to_cell.size(), 300U);
  ASSERT_EQ(sidecar.node_to_packet_record.size(), 300U);
  ASSERT_EQ(sidecar.packet_payload.size(), 300U * (sizeof(uint32_t) + sidecar.pq_code_width));
  EXPECT_EQ(sidecar.base_index_fingerprint.size_bytes,
            std::filesystem::file_size(index_prefix.string() + "_disk.index"));
  EXPECT_EQ(sidecar.pq_fingerprint.size_bytes,
            std::filesystem::file_size(index_prefix.string() + "_pq_compressed.bin"));

  std::unique_ptr<uint8_t[]> pq_codes;
  size_t pq_points = 0;
  size_t pq_width = 0;
  powerlaw_ann::load_bin<uint8_t>(index_prefix.string() + "_pq_compressed.bin", pq_codes, pq_points,
                                  pq_width);
  ASSERT_EQ(pq_points, sidecar.point_count);
  ASSERT_EQ(pq_width, sidecar.pq_code_width);
  const size_t packet_width = sizeof(uint32_t) + pq_width;
  for (size_t packet = 0; packet < pq_points; ++packet) {
    const size_t packet_offset = packet * packet_width;
    uint32_t node = 0;
    for (uint32_t byte = 0; byte < sizeof(uint32_t); ++byte) {
      node |= static_cast<uint32_t>(sidecar.packet_payload[packet_offset + byte]) << (byte * 8U);
    }
    ASSERT_LT(node, pq_points);
    EXPECT_TRUE(std::equal(sidecar.packet_payload.begin() + packet_offset + sizeof(uint32_t),
                           sidecar.packet_payload.begin() + packet_offset + packet_width,
                           pq_codes.get() + static_cast<size_t>(node) * pq_width));
  }

  std::vector<bool> seen_packet(300, false);
  for (uint32_t node = 0; node < 300; ++node) {
    EXPECT_LT(sidecar.node_to_community[node], sidecar.communities.size());
    EXPECT_LT(sidecar.node_to_cell[node], sidecar.cells.size());
    ASSERT_LT(sidecar.node_to_packet_record[node], 300U);
    EXPECT_FALSE(seen_packet[sidecar.node_to_packet_record[node]]);
    seen_packet[sidecar.node_to_packet_record[node]] = true;
  }
  for (const auto& community : sidecar.communities) {
    EXPECT_GT(community.node_count, 0U);
    EXPECT_LT(community.navigation_hub_id, sidecar.point_count);
  }
  for (const auto& edge : sidecar.community_edges) {
    EXPECT_NE(edge.source_community, edge.target_community);
    EXPECT_GT(edge.directed_cross_edge_count, 0U);
    EXPECT_GT(edge.gateway_count, 0U);
  }
  for (const auto& gateway : sidecar.gateways) {
    EXPECT_LT(gateway.source_node, sidecar.point_count);
    EXPECT_LT(gateway.target_node, sidecar.point_count);
    EXPECT_EQ(sidecar.node_to_packet_record[gateway.target_node], gateway.packet_record_index);
  }

  const auto gateway_selection_path = temp_dir.path() / "gateway_selection.csv";
  const auto rewritten_sidecar_path = temp_dir.path() / "community.gateway_profile.bin";
  const auto& profiled_edge = sidecar.community_edges.front();
  const auto selected_gateway =
      sidecar.gateways[profiled_edge.gateway_begin + profiled_edge.gateway_count - 1];
  {
    std::ofstream output(gateway_selection_path);
    output << "source_community,target_community,gateway_node,rank,utility,rcni\n"
           << profiled_edge.source_community << ',' << profiled_edge.target_community << ','
           << selected_gateway.target_node << ",0,1.0,0.5\n";
  }
  auto rewrite_config = power_ann_config.community_polar_build;
  rewrite_config.gateway_count = 2;
  rewrite_config.gateway_selection_path = gateway_selection_path;
  rewrite_config.gateway_selection_source_path =
      powerlaw_ann::make_community_polar_index_path(index_prefix);
  powerlaw_ann::build_community_polar_sidecar_from_disk(data_path, index_prefix,
                                                        index_prefix.string() + ".rcni.csv",
                                                        rewritten_sidecar_path, rewrite_config);
  const auto rewritten = powerlaw_ann::load_community_polar_index(rewritten_sidecar_path);
  EXPECT_EQ(rewritten.node_to_community, sidecar.node_to_community);
  EXPECT_EQ(rewritten.node_to_cell, sidecar.node_to_cell);
  EXPECT_EQ(rewritten.packet_payload, sidecar.packet_payload);
  ASSERT_EQ(rewritten.community_edges.front().gateway_count, 2U);
  EXPECT_EQ(rewritten.gateways.front().target_node, selected_gateway.target_node);

  const auto query_path = temp_dir.path() / "queries.bin";
  write_queries(query_path);
  powerlaw_ann::diskann_search_config_t search_config;
  search_config.index_path_prefix = index_prefix;
  search_config.query_path = query_path;
  search_config.top_k = 5;
  search_config.search_list_size = 50;
  search_config.beam_width = 2;
  search_config.num_threads = 1;
  search_config.num_nodes_to_cache = 0;
  powerlaw_ann::diskann_search_t search;
  const auto baseline_result = search.search(search_config);

  powerlaw_ann::community_polar_search_config_t cost_gated_config;
  cost_gated_config.mode = powerlaw_ann::powerann_search_mode_t::HYBRID;
  cost_gated_config.cell_expansion = true;
  cost_gated_config.cell_min_search_list_size = search_config.search_list_size + 1;
  const auto cost_gated_result = search.search(search_config, cost_gated_config);
  EXPECT_EQ(cost_gated_result.ids, baseline_result.ids);
  EXPECT_EQ(cost_gated_result.distances, baseline_result.distances);
  EXPECT_EQ(cost_gated_result.resident_sidecar_bytes, 0U);
  for (const auto& stats : cost_gated_result.query_stats) {
    EXPECT_EQ(stats.cell_blocks_expanded, 0U);
    EXPECT_EQ(stats.cell_nodes_scored, 0U);
  }

  cost_gated_config.community_skip = true;
  cost_gated_config.cell_whole_cell_scan = true;
  cost_gated_config.cell_centroid_routing = true;
  cost_gated_config.cell_routing_cell_budget = 2;
  EXPECT_NO_THROW({
    const auto combined_cost_gated = search.search(search_config, cost_gated_config);
    for (const auto& stats : combined_cost_gated.query_stats) {
      EXPECT_EQ(stats.cell_blocks_expanded, 0U);
      EXPECT_EQ(stats.cell_nodes_scored, 0U);
    }
  });

  powerlaw_ann::community_polar_search_config_t shadow_config;
  shadow_config.mode = powerlaw_ann::powerann_search_mode_t::HYBRID;
  shadow_config.community_skip = true;
  shadow_config.cell_expansion = true;
  shadow_config.community_expansion_budget = 4;
  shadow_config.gateway_score_budget = 32;
  shadow_config.cell_block_budget = 4;
  shadow_config.cell_support = 1;
  shadow_config.cell_insert_cap = 8;
  const auto shadow_result = search.search_observe_only(search_config, shadow_config);
  EXPECT_EQ(shadow_result.ids, baseline_result.ids);
  EXPECT_EQ(shadow_result.distances, baseline_result.distances);
  ASSERT_EQ(shadow_result.query_stats.size(), baseline_result.query_stats.size());
  uint64_t gateway_opportunities = 0;
  uint64_t cell_opportunities = 0;
  uint64_t cell_packet_bytes = 0;
  for (size_t query = 0; query < shadow_result.query_stats.size(); ++query) {
    const auto& baseline_stats = baseline_result.query_stats[query];
    const auto& shadow_stats = shadow_result.query_stats[query];
    EXPECT_EQ(shadow_stats.n_ios, baseline_stats.n_ios);
    EXPECT_EQ(shadow_stats.n_hops, baseline_stats.n_hops);
    EXPECT_EQ(shadow_stats.n_base_neighbors_scanned, baseline_stats.n_base_neighbors_scanned);
    gateway_opportunities += shadow_stats.gateway_nodes_competitive;
    cell_opportunities += shadow_stats.cell_nodes_competitive;
    cell_packet_bytes += shadow_stats.cell_packet_bytes;
  }
  EXPECT_GT(gateway_opportunities, 0U);
  EXPECT_GT(cell_opportunities, 0U);
  EXPECT_GT(cell_packet_bytes, 0U);
  EXPECT_GT(shadow_result.resident_sidecar_bytes, 0U);
  EXPECT_EQ(
      shadow_result.sidecar_artifact_bytes,
      std::filesystem::file_size(powerlaw_ann::make_community_polar_index_path(index_prefix)));

  const auto cell_oracle_path = temp_dir.path() / "cell_oracle.csv";
  const auto cell_value_path = temp_dir.path() / "cell_value.csv";
  search_config.cell_oracle_output_path = cell_oracle_path;
  search_config.cell_value_output_path = cell_value_path;
  const auto oracle_result = search.search_observe_only(search_config, shadow_config);
  EXPECT_EQ(oracle_result.ids, baseline_result.ids);
  EXPECT_EQ(oracle_result.distances, baseline_result.distances);
  EXPECT_TRUE(std::filesystem::is_regular_file(cell_oracle_path));
  EXPECT_GT(std::filesystem::file_size(cell_oracle_path), 0U);
  EXPECT_TRUE(std::filesystem::is_regular_file(cell_value_path));
  const auto value_bytes = read_file(cell_value_path);
  const std::string value_csv(value_bytes.begin(), value_bytes.end());
  EXPECT_NE(value_csv.find("query_id,block_ordinal,block_id"), std::string::npos);
  EXPECT_NE(value_csv.find("future_expansions"), std::string::npos);

  const auto gateway_value_path = temp_dir.path() / "gateway_value.csv";
  search_config.gateway_value_output_path = gateway_value_path;
  const auto gateway_value_result = search.search_observe_only(search_config, shadow_config);
  EXPECT_EQ(gateway_value_result.ids, baseline_result.ids);
  EXPECT_EQ(gateway_value_result.distances, baseline_result.distances);
  EXPECT_TRUE(std::filesystem::is_regular_file(gateway_value_path));
  const auto gateway_value_bytes = read_file(gateway_value_path);
  const std::string gateway_value_csv(gateway_value_bytes.begin(), gateway_value_bytes.end());
  EXPECT_NE(gateway_value_csv.find("query_id,search_l,source_community,target_community"),
            std::string::npos);
  EXPECT_NE(gateway_value_csv.find("future_expansions_covered"), std::string::npos);
  EXPECT_TRUE(std::any_of(gateway_value_result.query_stats.begin(),
                          gateway_value_result.query_stats.end(),
                          [](const auto& stats) { return !stats.gateway_value_records.empty(); }));
  search_config.gateway_value_output_path.clear();

  auto provisional_config = shadow_config;
  provisional_config.cell_use_provisional_frontier = true;
  const auto provisional_result = search.search_observe_only(search_config, provisional_config);
  EXPECT_EQ(provisional_result.ids, baseline_result.ids);
  EXPECT_EQ(provisional_result.distances, baseline_result.distances);
  uint64_t finite_provisional_ratios = 0;
  for (const auto& stats : provisional_result.query_stats) {
    for (size_t ordinal = 0; ordinal < std::min<size_t>(stats.cell_value_blocks_recorded,
                                                        powerlaw_ann::k_cell_value_max_blocks);
         ++ordinal) {
      if (stats.cell_value_evidence_threshold_ratio[ordinal] >= 0.0F &&
          stats.cell_value_best_threshold_ratio[ordinal] >= 0.0F) {
        ++finite_provisional_ratios;
      }
    }
  }
  EXPECT_GT(finite_provisional_ratios, 0U);
  search_config.cell_oracle_output_path.clear();
  search_config.cell_value_output_path.clear();

  auto rejected_cell_config = shadow_config;
  rejected_cell_config.community_skip = false;
  rejected_cell_config.cell_block_budget = 1;
  rejected_cell_config.cell_activation_marker = 0;
  rejected_cell_config.cell_min_competitive_count = std::numeric_limits<uint32_t>::max();
  const auto rejected_cell_result = search.search(search_config, rejected_cell_config);
  EXPECT_EQ(rejected_cell_result.ids, baseline_result.ids);
  EXPECT_EQ(rejected_cell_result.distances, baseline_result.distances);
  uint64_t low_yield_blocks = 0;
  uint64_t inserted_cell_nodes = 0;
  for (const auto& stats : rejected_cell_result.query_stats) {
    low_yield_blocks += stats.cell_blocks_low_yield;
    inserted_cell_nodes += stats.cell_nodes_inserted;
  }
  EXPECT_GT(low_yield_blocks, 0U);
  EXPECT_EQ(inserted_cell_nodes, 0U);

  search_config.phase_profile = true;
  const auto profiled_baseline = search.search(search_config);
  EXPECT_EQ(profiled_baseline.ids, baseline_result.ids);
  EXPECT_EQ(profiled_baseline.distances, baseline_result.distances);
  size_t baseline_transitions = 0;
  for (const auto& stats : profiled_baseline.query_stats) {
    if (stats.beam_phase_transition_found == 0) {
      continue;
    }
    ++baseline_transitions;
    EXPECT_EQ(stats.beam_phase_approach_base_hops + stats.beam_phase_convergence_base_hops,
              stats.base_hops);
    EXPECT_EQ(stats.beam_phase_approach_ios + stats.beam_phase_convergence_ios, stats.n_ios);
    EXPECT_NEAR(stats.beam_phase_setup_us + stats.beam_phase_approach_us +
                    stats.beam_phase_convergence_us + stats.beam_phase_post_search_us,
                stats.total_us, 1.0F);
  }
  EXPECT_GT(baseline_transitions, 0U);

  search_config.phase_profile = false;
  const auto hybrid_result = search.search(search_config, shadow_config);
  search_config.phase_profile = true;
  const auto profiled_hybrid = search.search(search_config, shadow_config);
  EXPECT_EQ(profiled_hybrid.ids, hybrid_result.ids);
  EXPECT_EQ(profiled_hybrid.distances, hybrid_result.distances);
  EXPECT_GT(profiled_hybrid.beam_phase_transition_rate_percent(), 0.0);

  search_config.num_threads = 2;
  search_config.phase_profile = false;
  const auto parallel_shadow = search.search_observe_only(search_config, shadow_config);
  EXPECT_EQ(parallel_shadow.ids, baseline_result.ids);
  EXPECT_EQ(parallel_shadow.distances, baseline_result.distances);
  ASSERT_EQ(parallel_shadow.query_stats.size(), shadow_result.query_stats.size());
  for (size_t query = 0; query < parallel_shadow.query_stats.size(); ++query) {
    EXPECT_EQ(parallel_shadow.query_stats[query].n_ios, shadow_result.query_stats[query].n_ios);
    EXPECT_EQ(parallel_shadow.query_stats[query].n_hops, shadow_result.query_stats[query].n_hops);
    EXPECT_EQ(parallel_shadow.query_stats[query].gateway_nodes_scored,
              shadow_result.query_stats[query].gateway_nodes_scored);
    EXPECT_EQ(parallel_shadow.query_stats[query].cell_nodes_scored,
              shadow_result.query_stats[query].cell_nodes_scored);
  }

  auto corrupt_bytes = read_file(powerlaw_ann::make_community_polar_index_path(index_prefix));
  ASSERT_GT(corrupt_bytes.size(), 512U);
  corrupt_bytes.back() ^= 0x1;
  const auto corrupt_path = temp_dir.path() / "corrupt.community_polar.bin";
  std::ofstream corrupt_output(corrupt_path, std::ios::binary | std::ios::trunc);
  corrupt_output.write(corrupt_bytes.data(), static_cast<std::streamsize>(corrupt_bytes.size()));
  corrupt_output.close();
  EXPECT_THROW(static_cast<void>(powerlaw_ann::load_community_polar_index(corrupt_path)),
               std::runtime_error);

  const auto disk_path = std::filesystem::path(index_prefix.string() + "_disk.index");
  std::fstream disk_file(disk_path, std::ios::binary | std::ios::in | std::ios::out);
  disk_file.seekg(-1, std::ios::end);
  char last_byte = 0;
  disk_file.read(&last_byte, 1);
  last_byte ^= 0x1;
  disk_file.seekp(-1, std::ios::end);
  disk_file.write(&last_byte, 1);
  disk_file.close();
  EXPECT_THROW(powerlaw_ann::validate_community_polar_artifacts(sidecar, index_prefix),
               std::runtime_error);
}

TEST(DiskannDiskIndexTest, RejectsReorderDataWithoutDiskPq) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  write_dataset(data_path);

  powerlaw_ann::diskann_disk_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = temp_dir.path() / "diskann";
  config.append_reorder_data = true;

  powerlaw_ann::power_ann_t power_ann;
  EXPECT_THROW(power_ann.build_diskann_index(config), powerlaw_ann::ann_exception_t);
}

TEST(DiskannDiskIndexTest, EnabledRcniRejectsShardBuildBeforePqArtifacts) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  write_dataset(data_path);

  powerlaw_ann::diskann_disk_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_prefix;
  config.search_dram_budget_gb = 0.01;
  config.build_dram_budget_gb = 1.0e-9;
  config.max_degree = 8;
  config.build_list_size = 16;
  config.num_threads = 1;
  config.quantized_dimension = 1;

  powerlaw_ann::power_ann_config_t power_ann_config;
  power_ann_config.enable_powerann = true;
  powerlaw_ann::power_ann_t power_ann(power_ann_config);
  EXPECT_THROW(power_ann.build_diskann_index(config), powerlaw_ann::ann_exception_t);
  EXPECT_FALSE(std::filesystem::exists(index_prefix.string() + "_pq_pivots.bin"));
  EXPECT_FALSE(std::filesystem::exists(index_prefix.string() + ".rcni.csv"));
  EXPECT_FALSE(std::filesystem::exists(index_prefix.string() + ".candidate_hubs.csv"));
}

TEST(DiskannDiskIndexTest, RejectsDirectoryPrefixWithoutRemovingIt) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_directory = temp_dir.path() / "index_directory";
  write_dataset(data_path);
  std::filesystem::create_directory(index_directory);

  powerlaw_ann::diskann_disk_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_directory;

  powerlaw_ann::power_ann_config_t power_ann_config;
  power_ann_config.enable_powerann = true;
  powerlaw_ann::power_ann_t power_ann(power_ann_config);
  EXPECT_THROW(power_ann.build_diskann_index(config), powerlaw_ann::ann_exception_t);
  EXPECT_TRUE(std::filesystem::is_directory(index_directory));
  EXPECT_TRUE(std::filesystem::is_empty(index_directory));
}

} // namespace
