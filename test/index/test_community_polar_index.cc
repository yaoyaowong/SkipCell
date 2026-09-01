#include "index/community_polar_index.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class temp_dir_t {
public:
  temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_community_polar_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

class fixed_graph_t final : public powerlaw_ann::post_link_graph_view_t {
public:
  fixed_graph_t() : adjacency_(34), vectors_(34 * 2) {
    for (uint32_t group = 0; group < 2; ++group) {
      const uint32_t first = group * 17;
      const float center = static_cast<float>(group * 100);
      vectors_[static_cast<size_t>(first) * 2] = center;
      vectors_[static_cast<size_t>(first) * 2 + 1] = center;
      for (uint32_t position = 0; position < 16; ++position) {
        const uint32_t node = first + position + 1;
        const float angle = 2.0F * std::numbers::pi_v<float> * position / 16.0F;
        vectors_[static_cast<size_t>(node) * 2] = center + 10.0F * std::cos(angle);
        vectors_[static_cast<size_t>(node) * 2 + 1] = center + 10.0F * std::sin(angle);
      }
      for (uint32_t offset = 0; offset < 17; ++offset) {
        const uint32_t node = first + offset;
        adjacency_[node].push_back(first + (offset + 16) % 17);
        adjacency_[node].push_back(first + (offset + 1) % 17);
      }
    }
    adjacency_[1].push_back(18);
    adjacency_[2].push_back(19);
  }

  uint64_t point_count() const noexcept override { return adjacency_.size(); }

  uint32_t dimension() const noexcept override { return 2; }

  uint32_t entry_point_id() const noexcept override { return 0; }

  std::span<const uint32_t> neighbors(uint32_t node_id) const override {
    if (node_id >= adjacency_.size()) {
      throw std::out_of_range("node ID is outside fixed graph");
    }
    return adjacency_[node_id];
  }

  void copy_vector(uint32_t node_id, std::span<float> destination) const override {
    if (node_id >= adjacency_.size() || destination.size() != dimension()) {
      throw std::invalid_argument("invalid fixed graph vector request");
    }
    std::copy_n(vectors_.begin() + static_cast<size_t>(node_id) * dimension(), dimension(),
                destination.begin());
  }

  float squared_distance(uint32_t left, uint32_t right) const override {
    if (left >= adjacency_.size() || right >= adjacency_.size()) {
      throw std::out_of_range("node ID is outside fixed graph");
    }
    float distance = 0.0F;
    for (uint32_t dimension_id = 0; dimension_id < dimension(); ++dimension_id) {
      const float difference = vectors_[static_cast<size_t>(left) * dimension() + dimension_id] -
                               vectors_[static_cast<size_t>(right) * dimension() + dimension_id];
      distance += difference * difference;
    }
    return distance;
  }

  std::vector<std::vector<uint32_t>>& adjacency() { return adjacency_; }

private:
  std::vector<std::vector<uint32_t>> adjacency_;
  std::vector<float> vectors_;
};

class dense_graph_t final : public powerlaw_ann::post_link_graph_view_t {
public:
  explicit dense_graph_t(uint32_t point_count)
      : adjacency_(point_count), vectors_(point_count * 2) {
    for (uint32_t node = 0; node < point_count; ++node) {
      vectors_[static_cast<size_t>(node) * 2] = static_cast<float>(node);
      vectors_[static_cast<size_t>(node) * 2 + 1] = static_cast<float>((node * 7) % point_count);
      for (uint32_t target = 0; target < point_count; ++target) {
        if (target != node) {
          adjacency_[node].push_back(target);
        }
      }
    }
  }

  uint64_t point_count() const noexcept override { return adjacency_.size(); }
  uint32_t dimension() const noexcept override { return 2; }
  uint32_t entry_point_id() const noexcept override { return 0; }

  std::span<const uint32_t> neighbors(uint32_t node_id) const override {
    return adjacency_.at(node_id);
  }

  void copy_vector(uint32_t node_id, std::span<float> destination) const override {
    if (destination.size() != dimension()) {
      throw std::invalid_argument("invalid dense graph vector request");
    }
    std::copy_n(vectors_.begin() + static_cast<size_t>(node_id) * dimension(), dimension(),
                destination.begin());
  }

  float squared_distance(uint32_t left, uint32_t right) const override {
    float distance = 0.0F;
    for (uint32_t dimension_id = 0; dimension_id < dimension(); ++dimension_id) {
      const float difference = vectors_.at(static_cast<size_t>(left) * dimension() + dimension_id) -
                               vectors_.at(static_cast<size_t>(right) * dimension() + dimension_id);
      distance += difference * difference;
    }
    return distance;
  }

private:
  std::vector<std::vector<uint32_t>> adjacency_;
  std::vector<float> vectors_;
};

std::vector<powerlaw_ann::rcni_node_importance_t> make_importance(uint32_t point_count) {
  std::vector<powerlaw_ann::rcni_node_importance_t> importance(point_count);
  for (uint32_t node = 0; node < point_count; ++node) {
    importance[node].node_id = node;
    importance[node].normalized_importance = static_cast<double>(node + 1) / point_count;
  }
  return importance;
}

powerlaw_ann::community_polar_build_config_t make_config() {
  powerlaw_ann::community_polar_build_config_t config;
  config.community_count = 2;
  config.partition_threads = 1;
  config.partition_seed = 0;
  config.gateway_count = 2;
  config.gateway_shortlist = 4;
  config.polar_direction_count = 16;
  config.cell_target_size = 4;
  config.block_target_size = 2;
  return config;
}

void append_u32_le(std::vector<uint8_t>& bytes, uint32_t value) {
  for (uint32_t shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
  }
}

void append_u64_le(std::vector<uint8_t>& bytes, uint64_t value) {
  for (uint32_t shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
  }
}

void append_float_le(std::vector<uint8_t>& bytes, float value) {
  append_u32_le(bytes, std::bit_cast<uint32_t>(value));
}

uint64_t fnv1a(std::span<const uint8_t> bytes) {
  uint64_t hash = 14695981039346656037ULL;
  for (const uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint32_t crc32_bytes(std::span<const uint8_t> bytes) {
  uint32_t checksum = 0xFFFFFFFFU;
  for (const uint8_t byte : bytes) {
    checksum ^= byte;
    for (uint32_t bit = 0; bit < 8; ++bit) {
      checksum = (checksum >> 1U) ^ ((checksum & 1U) != 0 ? 0xEDB88320U : 0U);
    }
  }
  return checksum ^ 0xFFFFFFFFU;
}

void write_pq_locality_profile(const std::filesystem::path& path,
                               std::span<const uint32_t> nodes, uint32_t cell_size,
                               uint32_t dimension) {
  const uint64_t cell_count = (nodes.size() + cell_size - 1) / cell_size;
  std::vector<uint8_t> payload;
  for (uint64_t cell = 0; cell <= cell_count; ++cell) {
    append_u64_le(payload, std::min<uint64_t>(nodes.size(), cell * cell_size));
  }
  for (const uint32_t node : nodes) {
    append_u32_le(payload, node);
  }
  for (uint64_t cell = 0; cell < cell_count; ++cell) {
    for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
      append_float_le(payload, static_cast<float>(cell * 10 + coordinate));
    }
  }
  std::vector<uint8_t> file = {'P', 'L', 'C', 'P', 'Q', 'L', 'O', 'C'};
  append_u32_le(file, 1);
  append_u32_le(file, 52);
  append_u32_le(file, 0x01020304U);
  append_u32_le(file, cell_size);
  append_u32_le(file, dimension);
  append_u64_le(file, nodes.size());
  append_u64_le(file, cell_count);
  append_u64_le(file, crc32_bytes(payload));
  file.insert(file.end(), payload.begin(), payload.end());
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(file.data()),
               static_cast<std::streamsize>(file.size()));
}

void write_flat_global_hierarchy(const std::filesystem::path& path, uint32_t dimension,
                                 uint32_t community_count, uint32_t cell_count,
                                 uint64_t point_count) {
  std::vector<uint8_t> payload;
  append_u32_le(payload, 0);
  for (uint32_t community = 1; community < community_count; ++community) {
    append_u32_le(payload, std::numeric_limits<uint32_t>::max());
  }
  append_u32_le(payload, 0);
  append_u32_le(payload, 0);
  append_u32_le(payload, cell_count);
  append_u64_le(payload, point_count);
  append_float_le(payload, 1000000.0F);
  append_u32_le(payload, 1);
  for (uint32_t cell = 0; cell < cell_count; ++cell) {
    append_u32_le(payload, cell);
  }
  for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
    append_float_le(payload, 0.0F);
  }

  std::vector<uint8_t> file;
  append_u64_le(file, 0x0052454948434C50ULL);
  append_u32_le(file, 1);
  append_u32_le(file, dimension);
  append_u32_le(file, cell_count);
  append_u32_le(file, 1);
  append_u32_le(file, community_count);
  append_u32_le(file, cell_count);
  append_u64_le(file, fnv1a(payload));
  file.insert(file.end(), payload.begin(), payload.end());
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(file.data()),
               static_cast<std::streamsize>(file.size()));
}

void complete_for_serialization(powerlaw_ann::community_polar_build_result_t& result) {
  result.index.pq_code_width = 2;
  result.index.base_index_fingerprint = {1234, 5678};
  result.index.pq_fingerprint = {345, 678};
  for (const uint32_t node : result.packet_node_ids) {
    append_u32_le(result.index.packet_payload, node);
    result.index.packet_payload.push_back(static_cast<uint8_t>(node));
    result.index.packet_payload.push_back(static_cast<uint8_t>(node ^ 0xA5U));
  }
  for (auto& community : result.index.communities) {
    community.packet_bytes = community.node_count * 6;
  }
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

uint32_t read_u32_at(const std::filesystem::path& path, std::streamoff offset) {
  std::ifstream input(path, std::ios::binary);
  input.seekg(offset);
  std::array<uint8_t, 4> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) {
    throw std::runtime_error("failed to read test sidecar version");
  }
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
}

TEST(CommunityPolarIndexTest, BuildsDirectedCommunitiesPolarCellsAndDeterministicBytes) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto first = powerlaw_ann::build_community_polar_index(graph, importance, make_config());
  auto second = powerlaw_ann::build_community_polar_index(graph, importance, make_config());

  ASSERT_EQ(first.index.communities.size(), 2U);
  EXPECT_EQ(first.index.directed_base_edge_count, 70U);
  EXPECT_EQ(first.index.undirected_partition_edge_count, 36U);
  EXPECT_EQ(first.index.weighted_edge_cut, 2U);
  EXPECT_EQ(first.index.node_to_community.front(), 0U);
  EXPECT_EQ(first.index.node_to_community[16], 0U);
  EXPECT_EQ(first.index.node_to_community[17], 1U);
  EXPECT_EQ(first.index.node_to_community.back(), 1U);

  const auto& left = first.index.communities[0];
  const auto& right = first.index.communities[1];
  EXPECT_EQ(left.node_count, 17U);
  EXPECT_EQ(right.node_count, 17U);
  EXPECT_EQ(left.minimum_node_id, 0U);
  EXPECT_EQ(right.minimum_node_id, 17U);
  EXPECT_EQ(left.navigation_hub_id, 16U);
  EXPECT_EQ(right.navigation_hub_id, 33U);
  EXPECT_EQ(left.internal_directed_edge_count, 34U);
  EXPECT_EQ(left.outgoing_directed_edge_count, 2U);
  EXPECT_EQ(right.incoming_directed_edge_count, 2U);
  EXPECT_FALSE(left.graph_only);
  EXPECT_FALSE(right.graph_only);
  EXPECT_GT(left.cell_count, 0U);
  EXPECT_GT(right.cell_count, 0U);
  EXPECT_NEAR(first.index.community_poles[0], 0.0F, 1.0e-5F);
  EXPECT_NEAR(first.index.community_poles[1], 0.0F, 1.0e-5F);
  EXPECT_NEAR(first.index.community_poles[2], 100.0F, 1.0e-5F);
  EXPECT_NEAR(first.index.community_poles[3], 100.0F, 1.0e-5F);
  EXPECT_NEAR(left.radius, 10.0F, 1.0e-4F);
  EXPECT_NEAR(right.radius, 10.0F, 1.0e-4F);

  ASSERT_EQ(first.index.community_edges.size(), 1U);
  const auto& edge = first.index.community_edges.front();
  EXPECT_EQ(edge.source_community, 0U);
  EXPECT_EQ(edge.target_community, 1U);
  EXPECT_EQ(edge.directed_cross_edge_count, 2U);
  ASSERT_EQ(edge.gateway_count, 2U);
  ASSERT_EQ(first.index.gateways.size(), 2U);
  EXPECT_EQ(first.index.gateways[0].source_node, 2U);
  EXPECT_EQ(first.index.gateways[0].target_node, 19U);
  EXPECT_EQ(first.index.gateways[1].source_node, 1U);
  EXPECT_EQ(first.index.gateways[1].target_node, 18U);

  EXPECT_EQ(first.packet_node_ids.size(), graph.point_count());
  EXPECT_TRUE(
      std::all_of(first.index.blocks.begin(), first.index.blocks.end(),
                  [](const auto& block) { return block.node_count > 0 && block.node_count <= 2; }));
  EXPECT_TRUE(
      std::any_of(first.index.sectors.begin(), first.index.sectors.end(), [](const auto& sector) {
        return sector.sector_id == std::numeric_limits<uint32_t>::max();
      }));

  complete_for_serialization(first);
  complete_for_serialization(second);
  temp_dir_t temp_dir;
  const auto first_path = temp_dir.path() / "first.bin";
  const auto second_path = temp_dir.path() / "second.bin";
  powerlaw_ann::write_community_polar_index(first_path, first.index);
  powerlaw_ann::write_community_polar_index(second_path, second.index);
  EXPECT_EQ(read_u32_at(first_path, 8), 5U);
  EXPECT_EQ(read_file(first_path), read_file(second_path));
  const auto loaded = powerlaw_ann::load_community_polar_index(first_path);
  EXPECT_EQ(loaded.node_to_community, first.index.node_to_community);
  EXPECT_EQ(loaded.node_to_cell, first.index.node_to_cell);
  EXPECT_EQ(loaded.packet_payload, first.index.packet_payload);
}

TEST(CommunityPolarIndexTest, RejectsInvalidConfigurationAndMalformedBaseEdges) {
  auto config = make_config();
  config.gateway_shortlist = 1;
  EXPECT_THROW(powerlaw_ann::validate_community_polar_build_config(config), std::invalid_argument);
  config = make_config();
  config.partition_projection = static_cast<powerlaw_ann::community_partition_projection_t>(999);
  EXPECT_THROW(powerlaw_ann::validate_community_polar_build_config(config), std::invalid_argument);

  fixed_graph_t graph;
  graph.adjacency()[0].push_back(0);
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  EXPECT_THROW(static_cast<void>(
                   powerlaw_ann::build_community_polar_index(graph, importance, make_config())),
               std::runtime_error);

  fixed_graph_t duplicate_graph;
  duplicate_graph.adjacency()[0].push_back(1);
  EXPECT_THROW(static_cast<void>(powerlaw_ann::build_community_polar_index(
                   duplicate_graph, importance, make_config())),
               std::runtime_error);
}

TEST(CommunityPolarIndexTest, AppliesAValidatedStaticGatewaySelectionProfile) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  temp_dir_t temp_dir;
  const auto profile_path = temp_dir.path() / "gateways.csv";
  {
    std::ofstream output(profile_path);
    output << "source_community,target_community,gateway_node,rank,utility,rcni\n"
              "0,1,18,0,7.0,0.5\n";
  }
  auto config = make_config();
  config.gateway_selection_path = profile_path;
  const auto result = powerlaw_ann::build_community_polar_index(graph, importance, config);

  ASSERT_EQ(result.index.gateways.size(), 2U);
  EXPECT_EQ(result.index.gateways[0].target_node, 18U);
  EXPECT_EQ(result.index.gateways[1].target_node, 19U);

  {
    std::ofstream output(profile_path, std::ios::trunc);
    output << "source_community,target_community,gateway_node,rank,utility,rcni\n"
              "0,1,17,0,7.0,0.5\n";
  }
  EXPECT_THROW(powerlaw_ann::build_community_polar_index(graph, importance, config),
               std::invalid_argument);
}

TEST(CommunityPolarIndexTest, SidecarOnlyBuildRejectsOverwritingAnInputArtifact) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.fbin";
  std::ofstream(data_path, std::ios::binary).put('\0');

  EXPECT_THROW(powerlaw_ann::build_community_polar_sidecar_from_disk(
                   data_path, temp_dir.path() / "index", temp_dir.path() / "index.rcni.csv",
                   data_path, make_config()),
               std::invalid_argument);
}

TEST(CommunityPolarIndexTest, FastGraphLocalRepackIsDeterministicAndPreservesOwnership) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto source_config = make_config();
  source_config.cell_partition = powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;
  source_config.cell_target_size = 2;
  source_config.block_target_size = 2;
  source_config.build_contiguous_cell_hierarchy = true;
  source_config.cell_hierarchy_branching = 2;
  source_config.cell_hierarchy_leaf_size = 2;
  auto first = powerlaw_ann::build_community_polar_index(graph, importance, source_config);
  complete_for_serialization(first);
  auto second = first.index;
  auto packed_config = source_config;
  packed_config.cell_target_size = 8;
  powerlaw_ann::repack_graph_local_cells(first.index, packed_config);
  powerlaw_ann::repack_graph_local_cells(second, packed_config);

  EXPECT_EQ(first.index.node_to_cell, second.node_to_cell);
  EXPECT_EQ(first.index.cells.size(), second.cells.size());
  EXPECT_EQ(first.index.blocks.size(), second.blocks.size());
  EXPECT_EQ(first.index.cell_hierarchy_children, second.cell_hierarchy_children);
  EXPECT_LT(first.index.cells.size(), first.packet_node_ids.size());
  EXPECT_TRUE(std::all_of(first.index.cells.begin(), first.index.cells.end(), [](const auto& cell) {
    return cell.node_count != 0 && cell.node_count <= 8 && cell.capacity_class == 8;
  }));
  EXPECT_TRUE(std::all_of(first.index.blocks.begin(), first.index.blocks.end(), [](const auto& block) {
    return block.node_count != 0 && block.node_count <= 2;
  }));
  EXPECT_EQ(first.index.packet_payload, second.packet_payload);
  temp_dir_t temp_dir;
  const auto first_path = temp_dir.path() / "packed-a.bin";
  const auto second_path = temp_dir.path() / "packed-b.bin";
  powerlaw_ann::write_community_polar_index(first_path, first.index);
  powerlaw_ann::write_community_polar_index(second_path, second);
  EXPECT_EQ(read_file(first_path), read_file(second_path));
}

TEST(CommunityPolarIndexTest, PqLocalityRepackIsDeterministicAndPreservesPackets) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto source_config = make_config();
  source_config.cell_partition = powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;
  auto first = powerlaw_ann::build_community_polar_index(graph, importance, source_config);
  complete_for_serialization(first);
  auto second = first.index;
  auto adaptive = first.index;
  const auto original_payload = first.index.packet_payload;
  temp_dir_t temp_dir;
  const auto profile = temp_dir.path() / "cells.pqloc";
  const auto hierarchy = temp_dir.path() / "cells.hierarchy.bin";
  std::vector<uint32_t> order(graph.point_count());
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  write_pq_locality_profile(profile, order, 8, graph.dimension());
  write_flat_global_hierarchy(hierarchy, graph.dimension(), source_config.community_count,
                              (graph.point_count() + 7) / 8, graph.point_count());
  auto config = source_config;
  config.cell_partition = powerlaw_ann::community_cell_partition_t::GLOBAL_GEOMETRIC;
  config.cell_target_size = 8;
  config.block_target_size = 3;
  config.convergence_cell_hierarchy_path = hierarchy;

  powerlaw_ann::repack_pq_locality_cells(first.index, profile, config);
  powerlaw_ann::repack_pq_locality_cells(second, profile, config);
  EXPECT_EQ(first.index.node_to_cell, second.node_to_cell);
  EXPECT_EQ(first.index.node_to_packet_record, second.node_to_packet_record);
  EXPECT_EQ(first.index.packet_payload, second.packet_payload);
  EXPECT_EQ(first.index.cells.size(), (graph.point_count() + 7) / 8);
  EXPECT_TRUE(std::all_of(first.index.cells.begin(), first.index.cells.end(), [](const auto& cell) {
    return cell.node_count != 0 && cell.node_count <= 8;
  }));
  EXPECT_TRUE(std::all_of(first.index.blocks.begin(), first.index.blocks.end(), [](const auto& block) {
    return block.node_count != 0 && block.node_count <= 3;
  }));
  EXPECT_EQ(first.index.packet_payload.size(), original_payload.size());
  for (uint32_t node = 0; node < graph.point_count(); ++node) {
    const size_t source = static_cast<size_t>(node) * 6;
    const size_t destination = static_cast<size_t>(first.index.node_to_packet_record[node]) * 6;
    EXPECT_TRUE(std::equal(original_payload.begin() + source, original_payload.begin() + source + 6,
                           first.index.packet_payload.begin() + destination));
  }
  const auto corrupt = temp_dir.path() / "corrupt.pqloc";
  auto bytes = read_file(profile);
  bytes.back() ^= 1;
  std::ofstream corrupt_output(corrupt, std::ios::binary);
  corrupt_output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  corrupt_output.close();
  EXPECT_THROW(powerlaw_ann::repack_pq_locality_cells(second, corrupt, config),
               std::invalid_argument);

  auto adaptive_config = source_config;
  adaptive_config.cell_partition = powerlaw_ann::community_cell_partition_t::GLOBAL_GEOMETRIC;
  adaptive_config.cell_target_size = 128;
  adaptive_config.block_target_size = 16;
  adaptive_config.adaptive_multi_capacity = true;
  adaptive_config.adaptive_policy_version = 1;
  adaptive_config.adaptive_vector_bytes = graph.dimension() * sizeof(float);
  adaptive_config.convergence_cell_hierarchy_path = hierarchy;
  adaptive_config.convergence_cell_capacity_path = temp_dir.path() / "capacity.csv";
  EXPECT_NO_THROW(powerlaw_ann::repack_pq_locality_cells(adaptive, profile, adaptive_config));
  ASSERT_EQ(adaptive.cells.size(), (graph.point_count() + 7) / 8);
  EXPECT_TRUE(std::all_of(adaptive.cells.begin(), adaptive.cells.end(),
                          [](const auto& cell) { return cell.capacity_class == 128U; }));
}

TEST(CommunityPolarIndexTest, MaterializesEveryRequestedCommunityAtMaximumK) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto config = make_config();
  config.community_count = static_cast<uint32_t>(graph.point_count());
  config.partition_threads = 2;

  const auto result = powerlaw_ann::build_community_polar_index(graph, importance, config);

  ASSERT_EQ(result.index.communities.size(), graph.point_count());
  EXPECT_EQ(result.index.node_to_community.size(), graph.point_count());
  for (const auto& community : result.index.communities) {
    EXPECT_EQ(community.node_count, 1U);
  }
}

TEST(CommunityPolarIndexTest, LocalityProjectionKeepsFullBaseGraphForGateways) {
  dense_graph_t graph(20);
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto full_config = make_config();
  full_config.block_target_size = 4;
  auto k8_config = full_config;
  k8_config.partition_projection = powerlaw_ann::community_partition_projection_t::LOCAL_VAMANA_K8;
  auto k16_config = full_config;
  k16_config.partition_projection =
      powerlaw_ann::community_partition_projection_t::LOCAL_VAMANA_K16;

  const auto full = powerlaw_ann::build_community_polar_index(graph, importance, full_config);
  auto k8 = powerlaw_ann::build_community_polar_index(graph, importance, k8_config);
  const auto k16 = powerlaw_ann::build_community_polar_index(graph, importance, k16_config);

  EXPECT_EQ(full.index.directed_base_edge_count, 380U);
  EXPECT_EQ(k8.index.directed_base_edge_count, full.index.directed_base_edge_count);
  EXPECT_EQ(k16.index.directed_base_edge_count, full.index.directed_base_edge_count);
  EXPECT_EQ(full.index.undirected_partition_edge_count, 190U);
  EXPECT_LT(k8.index.undirected_partition_edge_count, k16.index.undirected_partition_edge_count);
  EXPECT_LT(k16.index.undirected_partition_edge_count, full.index.undirected_partition_edge_count);
  EXPECT_EQ(k8.index.community_edges.size(), full.index.community_edges.size());

  complete_for_serialization(k8);
  temp_dir_t temp_dir;
  const auto sidecar_path = temp_dir.path() / "local-k8.bin";
  powerlaw_ann::write_community_polar_index(sidecar_path, k8.index);
  const auto loaded = powerlaw_ann::load_community_polar_index(sidecar_path);
  EXPECT_EQ(loaded.config.partition_projection,
            powerlaw_ann::community_partition_projection_t::LOCAL_VAMANA_K8);
}

TEST(CommunityPolarIndexTest, GraphLocalCellsPersistOneCentroidPerCellDeterministically) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto config = make_config();
  config.cell_partition = powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;

  auto first = powerlaw_ann::build_community_polar_index(graph, importance, config);
  const auto second = powerlaw_ann::build_community_polar_index(graph, importance, config);

  ASSERT_FALSE(first.index.cells.empty());
  EXPECT_EQ(first.index.direction_axes.size(), first.index.cells.size() * graph.dimension());
  EXPECT_EQ(first.index.node_to_cell, second.index.node_to_cell);
  EXPECT_EQ(first.packet_node_ids, second.packet_node_ids);
  EXPECT_EQ(first.index.direction_axes, second.index.direction_axes);
  for (const auto& community : first.index.communities) {
    EXPECT_FALSE(community.graph_only);
    EXPECT_EQ(community.sector_count, 1U);
  }

  complete_for_serialization(first);
  temp_dir_t temp_dir;
  const auto sidecar_path = temp_dir.path() / "graph-local.bin";
  powerlaw_ann::write_community_polar_index(sidecar_path, first.index);
  const auto loaded = powerlaw_ann::load_community_polar_index(sidecar_path);
  EXPECT_EQ(loaded.config.cell_partition, powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL);
  EXPECT_EQ(loaded.direction_axes, first.index.direction_axes);
}

TEST(CommunityPolarIndexTest, GlobalGraphLocalCellsAreDeterministicAndHaveOneOwner) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto config = make_config();
  config.cell_partition = powerlaw_ann::community_cell_partition_t::GLOBAL_GRAPH_LOCAL;
  config.build_contiguous_cell_hierarchy = true;
  config.cell_hierarchy_branching = 2;
  config.cell_hierarchy_leaf_size = 2;

  auto first = powerlaw_ann::build_community_polar_index(graph, importance, config);
  const auto second = powerlaw_ann::build_community_polar_index(graph, importance, config);
  EXPECT_EQ(first.index.node_to_cell, second.index.node_to_cell);
  EXPECT_EQ(first.packet_node_ids, second.packet_node_ids);
  EXPECT_EQ(first.index.cell_hierarchy_children, second.index.cell_hierarchy_children);
  ASSERT_EQ(first.index.communities.front().cell_count, first.index.cells.size());
  EXPECT_EQ(first.index.cell_hierarchy_nodes[first.index.cell_hierarchy_roots.front()]
                .descendant_node_count,
            graph.point_count());
  for (size_t community = 1; community < first.index.communities.size(); ++community) {
    EXPECT_EQ(first.index.communities[community].cell_count, 0U);
    EXPECT_EQ(first.index.cell_hierarchy_roots[community],
              std::numeric_limits<uint32_t>::max());
  }
  std::vector<uint32_t> ownership(graph.point_count(), 0);
  for (const uint32_t node : first.packet_node_ids) {
    ASSERT_LT(node, ownership.size());
    ++ownership[node];
  }
  EXPECT_TRUE(std::all_of(ownership.begin(), ownership.end(),
                          [](uint32_t count) { return count == 1; }));
  EXPECT_TRUE(std::all_of(first.index.cells.begin(), first.index.cells.end(),
                          [&](const auto& cell) {
                            return cell.community_id == 0 && cell.node_count > 0 &&
                                   cell.node_count <= config.cell_target_size;
                          }));

  complete_for_serialization(first);
  temp_dir_t temp_dir;
  const auto sidecar_path = temp_dir.path() / "global-graph-local.bin";
  powerlaw_ann::write_community_polar_index(sidecar_path, first.index);
  EXPECT_EQ(read_u32_at(sidecar_path, 8), 7U);
  const auto loaded = powerlaw_ann::load_community_polar_index(sidecar_path);
  EXPECT_EQ(loaded.config.cell_partition,
            powerlaw_ann::community_cell_partition_t::GLOBAL_GRAPH_LOCAL);
  EXPECT_EQ(loaded.node_to_cell, first.index.node_to_cell);
  EXPECT_EQ(loaded.cell_hierarchy_children, first.index.cell_hierarchy_children);
}

TEST(CommunityPolarIndexTest, BuildsDeterministicPersistedHierarchyOverGraphLocalCells) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto config = make_config();
  config.cell_partition = powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;
  config.build_contiguous_cell_hierarchy = true;
  config.cell_hierarchy_branching = 2;
  config.cell_hierarchy_leaf_size = 2;

  auto first = powerlaw_ann::build_community_polar_index(graph, importance, config);
  auto second = powerlaw_ann::build_community_polar_index(graph, importance, config);
  ASSERT_EQ(first.index.cell_hierarchy_roots.size(), first.index.communities.size());
  ASSERT_FALSE(first.index.cell_hierarchy_nodes.empty());
  EXPECT_EQ(first.index.cell_hierarchy_roots, second.index.cell_hierarchy_roots);
  ASSERT_EQ(first.index.cell_hierarchy_nodes.size(), second.index.cell_hierarchy_nodes.size());
  for (uint32_t node = 0; node < first.index.cell_hierarchy_nodes.size(); ++node) {
    const auto& left = first.index.cell_hierarchy_nodes[node];
    const auto& right = second.index.cell_hierarchy_nodes[node];
    EXPECT_EQ(left.node_id, right.node_id);
    EXPECT_EQ(left.child_begin, right.child_begin);
    EXPECT_EQ(left.child_count, right.child_count);
    EXPECT_EQ(left.descendant_node_count, right.descendant_node_count);
    EXPECT_EQ(left.radius, right.radius);
    EXPECT_EQ(left.children_are_cells, right.children_are_cells);
  }
  EXPECT_EQ(first.index.cell_hierarchy_children, second.index.cell_hierarchy_children);
  EXPECT_EQ(first.index.cell_hierarchy_centroids, second.index.cell_hierarchy_centroids);
  for (uint32_t community = 0; community < first.index.communities.size(); ++community) {
    const auto& owner = first.index.communities[community];
    const uint32_t root = first.index.cell_hierarchy_roots[community];
    ASSERT_LT(root, first.index.cell_hierarchy_nodes.size());
    EXPECT_EQ(first.index.cell_hierarchy_nodes[root].descendant_node_count, owner.node_count);
  }

  complete_for_serialization(first);
  temp_dir_t temp_dir;
  const auto sidecar_path = temp_dir.path() / "graph-local-hierarchy.bin";
  powerlaw_ann::write_community_polar_index(sidecar_path, first.index);
  const auto loaded = powerlaw_ann::load_community_polar_index(sidecar_path);
  EXPECT_EQ(loaded.cell_hierarchy_roots, first.index.cell_hierarchy_roots);
  EXPECT_EQ(loaded.cell_hierarchy_children, first.index.cell_hierarchy_children);
  EXPECT_EQ(loaded.cell_hierarchy_centroids, first.index.cell_hierarchy_centroids);

  config.cell_hierarchy_branching = 1;
  EXPECT_THROW(powerlaw_ann::validate_community_polar_build_config(config), std::invalid_argument);
  config.cell_hierarchy_branching = 2;
  config.cell_partition = powerlaw_ann::community_cell_partition_t::POLAR_RADIAL;
  EXPECT_THROW(powerlaw_ann::validate_community_polar_build_config(config), std::invalid_argument);
}

TEST(CommunityPolarIndexTest, PersistsAndValidatesTrainedCellHierarchy) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto config = make_config();
  config.cell_partition = powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;
  auto result = powerlaw_ann::build_community_polar_index(graph, importance, config);
  result.index.config.cell_partition = powerlaw_ann::community_cell_partition_t::GLOBAL_GEOMETRIC;
  result.index.cell_hierarchy_roots.resize(result.index.communities.size());
  for (uint32_t community = 0; community < result.index.communities.size(); ++community) {
    const auto& owner = result.index.communities[community];
    result.index.cell_hierarchy_roots[community] = community;
    result.index.cell_hierarchy_nodes.push_back(
        {community, static_cast<uint32_t>(result.index.cell_hierarchy_children.size()),
         owner.cell_count, owner.node_count, 1000.0F, true});
    for (uint64_t cell = owner.cell_begin; cell < owner.cell_begin + owner.cell_count; ++cell) {
      result.index.cell_hierarchy_children.push_back(static_cast<uint32_t>(cell));
    }
    const auto pole_begin = static_cast<size_t>(community) * graph.dimension();
    result.index.cell_hierarchy_centroids.insert(
        result.index.cell_hierarchy_centroids.end(),
        result.index.community_poles.begin() + static_cast<std::ptrdiff_t>(pole_begin),
        result.index.community_poles.begin() +
            static_cast<std::ptrdiff_t>(pole_begin + graph.dimension()));
  }
  complete_for_serialization(result);

  temp_dir_t temp_dir;
  const auto sidecar_path = temp_dir.path() / "hierarchy.bin";
  powerlaw_ann::write_community_polar_index(sidecar_path, result.index);
  const auto loaded = powerlaw_ann::load_community_polar_index(sidecar_path);
  EXPECT_EQ(loaded.cell_hierarchy_nodes.size(), result.index.communities.size());
  EXPECT_EQ(loaded.cell_hierarchy_roots, result.index.cell_hierarchy_roots);
  EXPECT_EQ(loaded.cell_hierarchy_children, result.index.cell_hierarchy_children);
  EXPECT_EQ(loaded.cell_hierarchy_centroids, result.index.cell_hierarchy_centroids);

  result.index.cell_hierarchy_nodes.front().radius = 0.0F;
  EXPECT_THROW(powerlaw_ann::write_community_polar_index(sidecar_path, result.index),
               std::runtime_error);
}

TEST(CommunityPolarIndexTest, LoadsDeterministicAdaptiveAssignmentHierarchyAndCapacityArtifacts) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  temp_dir_t temp_dir;
  const auto assignment_path = temp_dir.path() / "assignment.csv";
  const auto capacity_path = temp_dir.path() / "capacity.csv";
  const auto hierarchy_path = temp_dir.path() / "hierarchy.bin";
  {
    std::ofstream output(assignment_path);
    output << "node_id,community_id,cell_id,order\n";
    constexpr std::array<uint32_t, 5> boundaries = {0, 8, 16, 24, 34};
    for (uint32_t cell = 0; cell < 4; ++cell) {
      for (uint32_t node = boundaries[cell]; node < boundaries[cell + 1]; ++node) {
        output << node << ",0," << cell << ',' << node - boundaries[cell] << '\n';
      }
    }
  }
  {
    std::ofstream output(capacity_path);
    output << "cell_id,capacity_class,actual_population,tree_node_id,tree_depth\n"
              "0,16,8,10,3\n"
              "1,32,8,11,2\n"
              "2,64,8,12,1\n"
              "3,128,10,13,0\n";
  }
  write_flat_global_hierarchy(hierarchy_path, graph.dimension(), 2, 4, graph.point_count());

  auto config = make_config();
  config.cell_partition = powerlaw_ann::community_cell_partition_t::GLOBAL_GEOMETRIC;
  config.cell_target_size = 128;
  config.adaptive_multi_capacity = true;
  config.adaptive_policy_version = 1;
  config.adaptive_vector_bytes = graph.dimension() * sizeof(float);
  config.convergence_cell_profile_path = assignment_path;
  config.convergence_cell_hierarchy_path = hierarchy_path;
  config.convergence_cell_capacity_path = capacity_path;
  const auto first = powerlaw_ann::build_community_polar_index(graph, importance, config);
  const auto second = powerlaw_ann::build_community_polar_index(graph, importance, config);
  ASSERT_EQ(first.index.cells.size(), 4U);
  EXPECT_EQ(first.index.node_to_cell, second.index.node_to_cell);
  EXPECT_EQ(first.index.cell_hierarchy_roots, second.index.cell_hierarchy_roots);
  EXPECT_EQ(first.index.cell_hierarchy_children, second.index.cell_hierarchy_children);
  EXPECT_EQ(first.index.cell_hierarchy_centroids, second.index.cell_hierarchy_centroids);
  for (uint32_t cell = 0; cell < first.index.cells.size(); ++cell) {
    EXPECT_EQ(first.index.cells[cell].capacity_class, 16U << cell);
    EXPECT_LE(first.index.cells[cell].node_count, first.index.cells[cell].capacity_class);
  }

  {
    std::ofstream output(capacity_path, std::ios::trunc);
    output << "cell_id,capacity_class,actual_population,tree_node_id,tree_depth\n"
              "0,17,8,10,3\n"
              "1,32,8,11,2\n"
              "2,64,8,12,1\n"
              "3,128,10,13,0\n";
  }
  EXPECT_THROW(powerlaw_ann::build_community_polar_index(graph, importance, config),
               std::invalid_argument);
}

TEST(CommunityPolarIndexTest, PersistsAdaptiveCapacityClassesOnlyInVersionSix) {
  fixed_graph_t graph;
  const auto importance = make_importance(static_cast<uint32_t>(graph.point_count()));
  auto config = make_config();
  config.cell_partition = powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;
  auto result = powerlaw_ann::build_community_polar_index(graph, importance, config);
  result.index.config.cell_partition = powerlaw_ann::community_cell_partition_t::GLOBAL_GEOMETRIC;
  result.index.config.cell_target_size = 128;
  result.index.config.adaptive_multi_capacity = true;
  result.index.config.adaptive_policy_version = 1;
  result.index.config.adaptive_vector_bytes = graph.dimension() * sizeof(float);
  result.index.cell_hierarchy_roots.resize(result.index.communities.size());
  for (uint32_t community = 0; community < result.index.communities.size(); ++community) {
    const auto& owner = result.index.communities[community];
    result.index.cell_hierarchy_roots[community] = community;
    result.index.cell_hierarchy_nodes.push_back(
        {community, static_cast<uint32_t>(result.index.cell_hierarchy_children.size()),
         owner.cell_count, owner.node_count, 1000.0F, true});
    for (uint64_t cell = owner.cell_begin; cell < owner.cell_begin + owner.cell_count; ++cell) {
      result.index.cell_hierarchy_children.push_back(static_cast<uint32_t>(cell));
    }
    const auto pole_begin = static_cast<size_t>(community) * graph.dimension();
    result.index.cell_hierarchy_centroids.insert(
        result.index.cell_hierarchy_centroids.end(),
        result.index.community_poles.begin() + static_cast<std::ptrdiff_t>(pole_begin),
        result.index.community_poles.begin() +
            static_cast<std::ptrdiff_t>(pole_begin + graph.dimension()));
  }
  for (auto& cell : result.index.cells) {
    cell.capacity_class = 16;
  }
  complete_for_serialization(result);

  temp_dir_t temp_dir;
  const auto sidecar_path = temp_dir.path() / "adaptive.bin";
  powerlaw_ann::write_community_polar_index(sidecar_path, result.index);
  EXPECT_EQ(read_u32_at(sidecar_path, 8), 6U);
  const auto loaded = powerlaw_ann::load_community_polar_index(sidecar_path);
  EXPECT_TRUE(loaded.config.adaptive_multi_capacity);
  EXPECT_EQ(loaded.config.adaptive_policy_version, 1U);
  EXPECT_EQ(loaded.config.adaptive_vector_bytes, graph.dimension() * sizeof(float));
  EXPECT_TRUE(std::all_of(loaded.cells.begin(), loaded.cells.end(),
                          [](const auto& cell) { return cell.capacity_class == 16; }));

  result.index.cells.front().capacity_class = 17;
  EXPECT_THROW(powerlaw_ann::write_community_polar_index(sidecar_path, result.index),
               std::runtime_error);
}

TEST(CommunityPolarIndexTest, RejectsMalformedAdaptivePolicy) {
  auto config = make_config();
  config.cell_partition = powerlaw_ann::community_cell_partition_t::GLOBAL_GEOMETRIC;
  config.cell_target_size = 128;
  config.adaptive_multi_capacity = true;
  config.adaptive_policy_version = 1;
  config.adaptive_vector_bytes = 512;
  EXPECT_NO_THROW(powerlaw_ann::validate_community_polar_build_config(config));

  config.adaptive_minimum_train_cross_child_coaccess = 1.1;
  EXPECT_THROW(powerlaw_ann::validate_community_polar_build_config(config), std::invalid_argument);
  config.adaptive_minimum_train_cross_child_coaccess = 0.5;
  config.adaptive_policy_version = 2;
  EXPECT_THROW(powerlaw_ann::validate_community_polar_build_config(config), std::invalid_argument);
}

} // namespace
