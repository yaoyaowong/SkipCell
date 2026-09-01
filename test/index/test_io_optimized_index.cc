#include "common/ann_error.h"
#include "common/diskann_exception.h"
#include "common/distance.h"
#include "index/io_optimized_index.h"
#include "power_ann.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

namespace {

class temp_dir_t {
public:
  temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_io_optimized_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

constexpr uint32_t k_point_count = 300;
constexpr uint32_t k_dimension = 8;

TEST(IoOptimizedIndexTest, GroupsDeterministicMaximalPageRuns) {
  const std::vector<uint64_t> pages = {9, 3, 4, 4, 8, 20, 6};
  const std::vector<powerlaw_ann::io_page_run_t> expected = {{3, 2}, {6, 1}, {8, 2}, {20, 1}};
  EXPECT_EQ(powerlaw_ann::make_io_page_runs(pages), expected);
  EXPECT_TRUE(powerlaw_ann::make_io_page_runs({}).empty());
}

std::vector<float> make_dataset() {
  std::vector<float> data(k_point_count * k_dimension);
  for (uint32_t point = 0; point < k_point_count; ++point) {
    for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
      data[point * k_dimension + dimension] =
          static_cast<float>((point * 13U + dimension * 37U + (point / 251U) * 19U) % 256U);
    }
  }
  return data;
}

void write_dataset(const std::filesystem::path& path, const std::vector<float>& data) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&k_point_count), sizeof(k_point_count));
  output.write(reinterpret_cast<const char*>(&k_dimension), sizeof(k_dimension));
  output.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
  ASSERT_TRUE(output.good());
}

void write_u8_dataset(const std::filesystem::path& path, const std::vector<float>& data) {
  std::vector<uint8_t> converted(data.size());
  std::transform(data.begin(), data.end(), converted.begin(), [](float value) {
    EXPECT_GE(value, 0.0F);
    EXPECT_LE(value, 255.0F);
    EXPECT_EQ(value, std::nearbyint(value));
    return static_cast<uint8_t>(value);
  });
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&k_point_count), sizeof(k_point_count));
  output.write(reinterpret_cast<const char*>(&k_dimension), sizeof(k_dimension));
  output.write(reinterpret_cast<const char*>(converted.data()),
               static_cast<std::streamsize>(converted.size()));
  ASSERT_TRUE(output.good());
}

void write_test_rcni(const std::filesystem::path& path) {
  std::ofstream output(path);
  output << "node_id,raw_importance,normalized_importance,importance_percentile,source_support,"
            "witness_event_count\n";
  for (uint32_t node = 0; node < k_point_count; ++node) {
    const double score = static_cast<double>(node) / k_point_count;
    output << node << ',' << score << ',' << score << ',' << score << ",1,1\n";
  }
  ASSERT_TRUE(output.good());
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void flip_byte(const std::filesystem::path& path, uint64_t offset) {
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  file.seekg(static_cast<std::streamoff>(offset));
  char value = 0;
  file.read(&value, 1);
  value ^= static_cast<char>(0x5A);
  file.seekp(static_cast<std::streamoff>(offset));
  file.write(&value, 1);
  ASSERT_TRUE(file.good());
}

powerlaw_ann::diskann_disk_index_config_t
make_build_config(const std::filesystem::path& data_path,
                  const std::filesystem::path& index_prefix) {
  powerlaw_ann::diskann_disk_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_prefix;
  config.search_dram_budget_gb = 0.01;
  config.build_dram_budget_gb = 0.1;
  config.max_degree = 12;
  config.build_list_size = 32;
  config.num_threads = 1;
  config.quantized_dimension = 1;
  return config;
}

struct disk_metadata_t {
  uint64_t max_node_len = 0;
  uint64_t nodes_per_sector = 0;
};

disk_metadata_t read_disk_metadata(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  uint32_t rows = 0;
  uint32_t columns = 0;
  uint64_t ignored = 0;
  disk_metadata_t metadata;
  input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
  input.read(reinterpret_cast<char*>(&ignored), sizeof(ignored));
  input.read(reinterpret_cast<char*>(&ignored), sizeof(ignored));
  input.read(reinterpret_cast<char*>(&ignored), sizeof(ignored));
  input.read(reinterpret_cast<char*>(&metadata.max_node_len), sizeof(metadata.max_node_len));
  input.read(reinterpret_cast<char*>(&metadata.nodes_per_sector),
             sizeof(metadata.nodes_per_sector));
  EXPECT_EQ(rows, 9U);
  EXPECT_EQ(columns, 1U);
  EXPECT_GT(metadata.nodes_per_sector, 0U);
  return metadata;
}

std::vector<uint8_t> read_base_record(const std::filesystem::path& disk_path,
                                      const disk_metadata_t& metadata, uint32_t node_id) {
  const uint64_t sector = 1U + node_id / metadata.nodes_per_sector;
  const uint64_t slot = node_id % metadata.nodes_per_sector;
  std::vector<uint8_t> record(metadata.max_node_len);
  std::ifstream input(disk_path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(sector * powerlaw_ann::k_io_vector_page_size +
                                          slot * metadata.max_node_len));
  input.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(record.size()));
  EXPECT_TRUE(input.good());
  return record;
}

TEST(IoOptimizedIndexTest, BuildsDeterministicValidatedTopologyAndVectorArtifacts) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  const auto topology_a = temp_dir.path() / "a.topology";
  const auto vectors_a = temp_dir.path() / "a.vectors";
  const auto topology_b = temp_dir.path() / "b.topology";
  const auto vectors_b = temp_dir.path() / "b.vectors";
  const auto topology_without_hints = temp_dir.path() / "no-hints.topology";
  const auto vectors_without_hints = temp_dir.path() / "no-hints.vectors";
  const auto data = make_dataset();
  write_dataset(data_path, data);

  powerlaw_ann::power_ann_t power_ann;
  power_ann.build_diskann_index(make_build_config(data_path, index_prefix));
  const auto first = powerlaw_ann::io_optimized_index_t::build(index_prefix, topology_a, vectors_a);
  const auto second =
      powerlaw_ann::io_optimized_index_t::build(index_prefix, topology_b, vectors_b);
  powerlaw_ann::io_optimized_index_t::build(
      index_prefix, topology_without_hints, vectors_without_hints,
      powerlaw_ann::io_adjacency_encoding_t::RAW_U32, powerlaw_ann::io_vector_layout_t::ORIGINAL_ID,
      {}, false);
  EXPECT_EQ(first.point_count, k_point_count);
  EXPECT_EQ(first.point_count, second.point_count);
  EXPECT_EQ(first.edge_count, second.edge_count);
  EXPECT_EQ(first.topology_bytes, std::filesystem::file_size(topology_a));
  EXPECT_EQ(first.vector_bytes, std::filesystem::file_size(vectors_a));
  EXPECT_EQ(first.vector_bytes % powerlaw_ann::k_io_vector_page_size, 0U);
  EXPECT_EQ(read_bytes(topology_a), read_bytes(topology_b));
  EXPECT_EQ(read_bytes(vectors_a), read_bytes(vectors_b));

  const auto index = powerlaw_ann::io_optimized_index_t::load(index_prefix, topology_a, vectors_a);
  const auto paged =
      powerlaw_ann::io_optimized_index_t::load(index_prefix, topology_a, vectors_a, false, true, 2);
  ASSERT_EQ(index->point_count(), k_point_count);
  ASSERT_EQ(index->dimension(), k_dimension);
  ASSERT_EQ(index->edge_count(), first.edge_count);
  EXPECT_EQ(index->artifact_bytes(), first.topology_bytes + first.vector_bytes);
  EXPECT_GT(index->resident_bytes(), 0U);
  EXPECT_TRUE(paged->paged_topology());
  EXPECT_EQ(paged->edge_count(), index->edge_count());
  EXPECT_EQ(paged->topology_cache_resident_bytes(), 2U * 4096U);
  EXPECT_LT(paged->resident_bytes(), index->resident_bytes());
  const auto index_without_hints = powerlaw_ann::io_optimized_index_t::load(
      index_prefix, topology_without_hints, vectors_without_hints);
  for (uint32_t node = 0; node < k_point_count; ++node) {
    EXPECT_EQ(index_without_hints->node_prefetch_page(node), powerlaw_ann::k_no_io_prefetch_page);
  }
  EXPECT_FALSE(index_without_hints->has_gateway_prefetch_hints());

  const auto disk_path = std::filesystem::path(index_prefix.string() + "_disk.index");
  const auto metadata = read_disk_metadata(disk_path);
  std::ifstream vector_input(vectors_a, std::ios::binary);
  for (uint32_t node = 0; node < k_point_count; ++node) {
    const auto record = read_base_record(disk_path, metadata, node);
    const auto* degree_address = record.data() + k_dimension * sizeof(float);
    uint32_t degree = 0;
    std::memcpy(&degree, degree_address, sizeof(degree));
    const auto neighbors = index->neighbors(node);
    const auto paged_neighbors = paged->neighbors(node);
    ASSERT_EQ(neighbors.size(), degree);
    ASSERT_EQ(paged_neighbors.size(), neighbors.size());
    for (uint32_t neighbor = 0; neighbor < degree; ++neighbor) {
      uint32_t expected = 0;
      std::memcpy(&expected, degree_address + sizeof(uint32_t) * (neighbor + 1U), sizeof(expected));
      EXPECT_EQ(neighbors[neighbor], expected);
      EXPECT_EQ(paged_neighbors[neighbor], expected);
    }

    const auto location = index->vector_location(node);
    const auto prefetch_page = index->node_prefetch_page(node);
    EXPECT_TRUE(prefetch_page == powerlaw_ann::k_no_io_prefetch_page ||
                prefetch_page < index->vector_page_count());
    std::vector<float> vector(k_dimension);
    vector_input.seekg(static_cast<std::streamoff>(
        location.page_id * powerlaw_ann::k_io_vector_page_size + location.page_offset));
    vector_input.read(reinterpret_cast<char*>(vector.data()),
                      static_cast<std::streamsize>(vector.size() * sizeof(float)));
    ASSERT_TRUE(vector_input.good());
    for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
      EXPECT_FLOAT_EQ(vector[dimension], data[node * k_dimension + dimension]);
    }
  }
  const auto paged_stats = paged->topology_cache_stats();
  EXPECT_GT(paged_stats.misses, 0U);
  EXPECT_GT(paged_stats.demand_hits, 0U);
}

TEST(IoOptimizedIndexTest, RejectsCorruptTruncatedVersionAndFingerprintArtifacts) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  const auto topology = temp_dir.path() / "topology";
  const auto vectors = temp_dir.path() / "vectors";
  write_dataset(data_path, make_dataset());
  powerlaw_ann::power_ann_t().build_diskann_index(make_build_config(data_path, index_prefix));
  powerlaw_ann::io_optimized_index_t::build(index_prefix, topology, vectors);

  const auto truncated = temp_dir.path() / "truncated";
  std::filesystem::copy_file(topology, truncated);
  std::filesystem::resize_file(truncated, 64);
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::load(index_prefix, truncated, vectors),
               std::runtime_error);

  const auto corrupt = temp_dir.path() / "corrupt";
  std::filesystem::copy_file(topology, corrupt);
  flip_byte(corrupt, 280);
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::load(index_prefix, corrupt, vectors),
               std::runtime_error);

  const auto wrong_version = temp_dir.path() / "wrong-version";
  std::filesystem::copy_file(topology, wrong_version);
  flip_byte(wrong_version, sizeof(uint64_t));
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::load(index_prefix, wrong_version, vectors),
               std::runtime_error);

  const auto wrong_prefix = temp_dir.path() / "wrong";
  const auto wrong_disk = std::filesystem::path(wrong_prefix.string() + "_disk.index");
  std::filesystem::copy_file(index_prefix.string() + "_disk.index", wrong_disk);
  flip_byte(wrong_disk, std::filesystem::file_size(wrong_disk) - 1U);
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::load(wrong_prefix, topology, vectors),
               std::runtime_error);

  const auto corrupt_vectors = temp_dir.path() / "corrupt-vectors";
  std::filesystem::copy_file(vectors, corrupt_vectors);
  flip_byte(corrupt_vectors, 0);
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::load(index_prefix, topology, corrupt_vectors),
               std::runtime_error);
}

TEST(IoOptimizedIndexTest, PForDeltaRoundTripsNeighborOrderAndRejectsUnsafeAliases) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  write_dataset(data_path, make_dataset());
  powerlaw_ann::power_ann_t().build_diskann_index(make_build_config(data_path, index_prefix));
  const auto raw_topology = temp_dir.path() / "raw.topology";
  const auto raw_vectors = temp_dir.path() / "raw.vectors";
  const auto pfor_topology = temp_dir.path() / "pfor.topology";
  const auto pfor_vectors = temp_dir.path() / "pfor.vectors";
  powerlaw_ann::io_optimized_index_t::build(index_prefix, raw_topology, raw_vectors);
  const auto pfor_result = powerlaw_ann::io_optimized_index_t::build(
      index_prefix, pfor_topology, pfor_vectors, powerlaw_ann::io_adjacency_encoding_t::PFOR_DELTA);
  EXPECT_GT(pfor_result.adjacency_bytes, 0U);
  const auto raw =
      powerlaw_ann::io_optimized_index_t::load(index_prefix, raw_topology, raw_vectors);
  const auto pfor =
      powerlaw_ann::io_optimized_index_t::load(index_prefix, pfor_topology, pfor_vectors);
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::load(index_prefix, pfor_topology, pfor_vectors,
                                                        false, true, 2),
               std::runtime_error);
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::load(index_prefix, raw_topology, raw_vectors,
                                                        false, false, 2),
               std::runtime_error);
  for (uint32_t node = 0; node < k_point_count; ++node) {
    EXPECT_TRUE(std::equal(raw->neighbors(node).begin(), raw->neighbors(node).end(),
                           pfor->neighbors(node).begin(), pfor->neighbors(node).end()));
  }
  const auto raw_cells = powerlaw_ann::build_io_global_graph_cell_assignment(*raw, 4);
  const auto pfor_cells = powerlaw_ann::build_io_global_graph_cell_assignment(*pfor, 4);
  const auto mapped_cells =
      powerlaw_ann::build_io_global_graph_cell_assignment_from_artifact(raw_topology, 4);
  EXPECT_EQ(raw_cells.node_to_cell, pfor_cells.node_to_cell);
  EXPECT_EQ(raw_cells.node_to_cell, mapped_cells.node_to_cell);
  EXPECT_EQ(raw_cells.node_to_cell.size(), k_point_count);
  EXPECT_GT(raw_cells.cell_count, 0U);
  std::vector<uint32_t> populations(raw_cells.cell_count, 0);
  for (const uint32_t cell : raw_cells.node_to_cell) {
    ASSERT_LT(cell, populations.size());
    ++populations[cell];
  }
  EXPECT_TRUE(std::all_of(populations.begin(), populations.end(),
                          [](uint32_t population) { return population >= 1 && population <= 4; }));

  const auto disk_path = std::filesystem::path(index_prefix.string() + "_disk.index");
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::build(index_prefix, disk_path, {}),
               std::runtime_error);
}

TEST(IoOptimizedIndexTest, BuildsDeterministicBoundedCellAdjacencyArtifact) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  const auto topology_path = temp_dir.path() / "cell.topology";
  const auto vector_path = temp_dir.path() / "cell.vectors";
  const auto adjacency_a = temp_dir.path() / "a.cell-adj";
  const auto adjacency_b = temp_dir.path() / "b.cell-adj";
  write_dataset(data_path, make_dataset());

  powerlaw_ann::power_ann_config_t facade_config;
  facade_config.enable_powerann = true;
  facade_config.community_polar_build.community_count = 4;
  facade_config.community_polar_build.partition_threads = 1;
  facade_config.community_polar_build.gateway_count = 2;
  facade_config.community_polar_build.gateway_shortlist = 4;
  facade_config.community_polar_build.polar_direction_count = 16;
  facade_config.community_polar_build.cell_partition =
      powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;
  facade_config.community_polar_build.cell_target_size = 16;
  facade_config.community_polar_build.block_target_size = 8;
  powerlaw_ann::power_ann_t(facade_config)
      .build_diskann_index(make_build_config(data_path, index_prefix));
  const auto sidecar_path = powerlaw_ann::make_community_polar_index_path(index_prefix);
  powerlaw_ann::io_optimized_index_t::build(
      index_prefix, topology_path, vector_path, powerlaw_ann::io_adjacency_encoding_t::RAW_U32,
      powerlaw_ann::io_vector_layout_t::CELL_4K, sidecar_path);
  const auto topology =
      powerlaw_ann::io_optimized_index_t::load(index_prefix, topology_path, vector_path);
  const auto cells = powerlaw_ann::load_community_polar_index(sidecar_path);

  const auto first = powerlaw_ann::io_cell_adjacency_index_t::build(
      adjacency_a, *topology, cells, topology->community_polar_fingerprint(), 4, 8, 1);
  const auto second = powerlaw_ann::io_cell_adjacency_index_t::build(
      adjacency_b, *topology, cells, topology->community_polar_fingerprint(), 4, 8, 4);
  EXPECT_EQ(read_bytes(adjacency_a), read_bytes(adjacency_b));
  EXPECT_EQ(first.edge_count, second.edge_count);
  EXPECT_EQ(first.build_threads, 1U);
  EXPECT_EQ(second.build_threads, 4U);
  const auto adjacency = powerlaw_ann::io_cell_adjacency_index_t::load(
      adjacency_a, cells.point_count, cells.cells.size(), topology->community_polar_fingerprint());
  EXPECT_EQ(adjacency->artifact_bytes(), first.artifact_bytes);
  EXPECT_EQ(adjacency->sampled_nodes_per_cell(), 4U);
  EXPECT_EQ(adjacency->maximum_degree(), 8U);
  for (uint32_t cell = 0; cell < cells.cells.size(); ++cell) {
    const auto neighbors = adjacency->neighbors(cell);
    EXPECT_LE(neighbors.size(), 8U);
    EXPECT_EQ(std::find(neighbors.begin(), neighbors.end(), cell), neighbors.end());
    auto sorted = std::vector<uint32_t>(neighbors.begin(), neighbors.end());
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::unique(sorted.begin(), sorted.end()), sorted.end());
  }

  const auto corrupt = temp_dir.path() / "corrupt.cell-adj";
  std::filesystem::copy_file(adjacency_a, corrupt);
  flip_byte(corrupt, 128);
  EXPECT_THROW(
      powerlaw_ann::io_cell_adjacency_index_t::load(corrupt, cells.point_count, cells.cells.size(),
                                                    topology->community_polar_fingerprint()),
      std::runtime_error);
  auto wrong_fingerprint = topology->community_polar_fingerprint();
  ++wrong_fingerprint.checksum;
  EXPECT_THROW(powerlaw_ann::io_cell_adjacency_index_t::load(adjacency_a, cells.point_count,
                                                             cells.cells.size(), wrong_fingerprint),
               std::runtime_error);
}

TEST(IoOptimizedIndexTest,
     BuildsDeterministicMixedCapacityCellPagesWithoutChangingLogicalIdsOrBase) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  const auto topology_a = temp_dir.path() / "cell-a.topology";
  const auto vectors_a = temp_dir.path() / "cell-a.vectors";
  const auto topology_b = temp_dir.path() / "cell-b.topology";
  const auto vectors_b = temp_dir.path() / "cell-b.vectors";
  const auto chunk_topology = temp_dir.path() / "chunks.topology";
  const auto chunk_vectors = temp_dir.path() / "chunks.vectors";
  const auto weighted_topology = temp_dir.path() / "weighted.topology";
  const auto weighted_vectors = temp_dir.path() / "weighted.vectors";
  const auto u8_topology = temp_dir.path() / "cell-u8.topology";
  const auto u8_vectors = temp_dir.path() / "cell-u8.vectors";
  const auto resident_u8 = temp_dir.path() / "base.u8bin";
  const auto rcni_path = temp_dir.path() / "rcni.csv";
  const auto trimmed_topology = temp_dir.path() / "trimmed.topology";
  const auto trimmed_vectors = temp_dir.path() / "trimmed.vectors";
  const auto data = make_dataset();
  write_dataset(data_path, data);
  write_u8_dataset(resident_u8, data);
  write_test_rcni(rcni_path);

  powerlaw_ann::power_ann_config_t facade_config;
  facade_config.enable_powerann = true;
  facade_config.community_polar_build.community_count = 4;
  facade_config.community_polar_build.partition_threads = 1;
  facade_config.community_polar_build.gateway_count = 2;
  facade_config.community_polar_build.gateway_shortlist = 4;
  facade_config.community_polar_build.polar_direction_count = 16;
  facade_config.community_polar_build.cell_partition =
      powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;
  facade_config.community_polar_build.cell_target_size = 16;
  facade_config.community_polar_build.block_target_size = 8;
  powerlaw_ann::power_ann_t(facade_config)
      .build_diskann_index(make_build_config(data_path, index_prefix));
  const auto sidecar_path = powerlaw_ann::make_community_polar_index_path(index_prefix);
  auto adaptive = powerlaw_ann::load_community_polar_index(sidecar_path);
  ASSERT_GE(adaptive.cells.size(), 4U);
  adaptive.config.cell_partition = powerlaw_ann::community_cell_partition_t::GLOBAL_GEOMETRIC;
  adaptive.config.cell_target_size = 128;
  adaptive.config.adaptive_multi_capacity = true;
  adaptive.config.adaptive_policy_version = 1;
  adaptive.config.adaptive_vector_bytes = k_dimension * sizeof(float);
  adaptive.cell_hierarchy_roots.resize(adaptive.communities.size());
  for (uint32_t community = 0; community < adaptive.communities.size(); ++community) {
    const auto& owner = adaptive.communities[community];
    adaptive.cell_hierarchy_roots[community] = community;
    adaptive.cell_hierarchy_nodes.push_back(
        {community, static_cast<uint32_t>(adaptive.cell_hierarchy_children.size()),
         owner.cell_count, owner.node_count, 1000000.0F, true});
    for (uint64_t cell = owner.cell_begin; cell < owner.cell_begin + owner.cell_count; ++cell) {
      adaptive.cell_hierarchy_children.push_back(static_cast<uint32_t>(cell));
    }
    adaptive.cell_hierarchy_centroids.insert(
        adaptive.cell_hierarchy_centroids.end(),
        adaptive.community_poles.begin() + static_cast<std::ptrdiff_t>(community * k_dimension),
        adaptive.community_poles.begin() +
            static_cast<std::ptrdiff_t>((community + 1U) * k_dimension));
  }
  constexpr std::array<uint32_t, 4> capacities = {16, 32, 64, 128};
  for (size_t cell = 0; cell < adaptive.cells.size(); ++cell) {
    adaptive.cells[cell].capacity_class = capacities[cell % capacities.size()];
    ASSERT_LE(adaptive.cells[cell].node_count, adaptive.cells[cell].capacity_class);
  }
  powerlaw_ann::write_community_polar_index(sidecar_path, adaptive);
  const auto disk_path = std::filesystem::path(index_prefix.string() + "_disk.index");
  const auto base_disk_before = read_bytes(disk_path);
  const auto first = powerlaw_ann::io_optimized_index_t::build(
      index_prefix, topology_a, vectors_a, powerlaw_ann::io_adjacency_encoding_t::RAW_U32,
      powerlaw_ann::io_vector_layout_t::CELL_4K, sidecar_path);
  const auto second = powerlaw_ann::io_optimized_index_t::build(
      index_prefix, topology_b, vectors_b, powerlaw_ann::io_adjacency_encoding_t::RAW_U32,
      powerlaw_ann::io_vector_layout_t::CELL_4K, sidecar_path);
  EXPECT_EQ(read_bytes(topology_a), read_bytes(topology_b));
  EXPECT_EQ(read_bytes(vectors_a), read_bytes(vectors_b));
  EXPECT_EQ(first.vector_bytes, second.vector_bytes);
  EXPECT_GT(first.cell_count, 0U);
  EXPECT_EQ(first.nonempty_vector_bytes, data.size() * sizeof(float));
  EXPECT_GE(first.vector_bytes, first.nonempty_vector_bytes);
  const auto chunks = powerlaw_ann::io_optimized_index_t::build(
      index_prefix, chunk_topology, chunk_vectors, powerlaw_ann::io_adjacency_encoding_t::RAW_U32,
      powerlaw_ann::io_vector_layout_t::CELL_CHUNKS, sidecar_path);
  EXPECT_EQ(chunks.chunk_counts[0], chunks.cell_count);
  EXPECT_EQ(chunks.chunk_counts[1], 0U);
  const auto weighted = powerlaw_ann::io_optimized_index_t::build(
      index_prefix, weighted_topology, weighted_vectors,
      powerlaw_ann::io_adjacency_encoding_t::RAW_U32,
      powerlaw_ann::io_vector_layout_t::CELL_WEIGHTED_4K, sidecar_path);
  EXPECT_EQ(weighted.vector_bytes, first.vector_bytes);
  const auto u8 = powerlaw_ann::io_optimized_index_t::build(
      index_prefix, u8_topology, u8_vectors, powerlaw_ann::io_adjacency_encoding_t::RAW_U32,
      powerlaw_ann::io_vector_layout_t::CELL_U8_4K, sidecar_path);
  EXPECT_EQ(u8.nonempty_vector_bytes, data.size());
  EXPECT_LE(u8.vector_bytes, first.vector_bytes);
  const auto u8_index =
      powerlaw_ann::io_optimized_index_t::load(index_prefix, u8_topology, u8_vectors);
  EXPECT_EQ(u8_index->vector_layout(), powerlaw_ann::io_vector_layout_t::CELL_U8_4K);
  std::ifstream u8_input(u8_vectors, std::ios::binary);
  for (uint32_t node = 0; node < k_point_count; ++node) {
    const auto location = u8_index->vector_location(node);
    std::array<uint8_t, k_dimension> vector{};
    u8_input.seekg(static_cast<std::streamoff>(
        location.page_id * powerlaw_ann::k_io_vector_page_size + location.page_offset));
    u8_input.read(reinterpret_cast<char*>(vector.data()), vector.size());
    ASSERT_TRUE(u8_input.good());
    for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
      EXPECT_EQ(vector[dimension], static_cast<uint8_t>(data[node * k_dimension + dimension]));
    }
  }
  EXPECT_EQ(read_bytes(disk_path), base_disk_before);

  const auto trimmed = powerlaw_ann::io_optimized_index_t::build(
      index_prefix, trimmed_topology, trimmed_vectors,
      powerlaw_ann::io_adjacency_encoding_t::RAW_U32, powerlaw_ann::io_vector_layout_t::CELL_4K,
      sidecar_path, true, rcni_path);
  uint64_t expected_trimmed_nodes = 0;
  for (const auto& community : adaptive.communities) {
    expected_trimmed_nodes += community.node_count / 2U;
  }
  EXPECT_EQ(trimmed.low_rcni_trimmed_nodes, expected_trimmed_nodes);
  EXPECT_EQ(trimmed.edge_count + expected_trimmed_nodes, first.edge_count);
  EXPECT_EQ(read_bytes(disk_path), base_disk_before);
  const auto original_index =
      powerlaw_ann::io_optimized_index_t::load(index_prefix, topology_a, vectors_a);
  const auto trimmed_index =
      powerlaw_ann::io_optimized_index_t::load(index_prefix, trimmed_topology, trimmed_vectors);
  uint64_t observed_trimmed_nodes = 0;
  for (uint32_t node = 0; node < k_point_count; ++node) {
    const auto original_degree = original_index->neighbors(node).size();
    const auto trimmed_degree = trimmed_index->neighbors(node).size();
    ASSERT_TRUE(trimmed_degree == original_degree || trimmed_degree + 1U == original_degree);
    observed_trimmed_nodes += trimmed_degree + 1U == original_degree;
  }
  EXPECT_EQ(observed_trimmed_nodes, expected_trimmed_nodes);

  const auto community = powerlaw_ann::load_community_polar_index(sidecar_path);
  EXPECT_TRUE(community.config.adaptive_multi_capacity);
  for (const uint32_t capacity : capacities) {
    EXPECT_TRUE(
        std::any_of(community.cells.begin(), community.cells.end(),
                    [capacity](const auto& cell) { return cell.capacity_class == capacity; }));
  }
  const auto index = powerlaw_ann::io_optimized_index_t::load(index_prefix, topology_a, vectors_a);
  ASSERT_EQ(index->vector_layout(), powerlaw_ann::io_vector_layout_t::CELL_4K);
  ASSERT_EQ(index->cell_page_ranges().size(), community.cells.size());
  ASSERT_EQ(first.gateway_prefetch_hint_count, community.gateways.size());
  ASSERT_EQ(first.gateway_prefetch_hint_bytes, community.gateways.size() * 64U);
  ASSERT_EQ(index->gateway_prefetch_hint_count(), community.gateways.size());
  EXPECT_EQ(index->community_polar_fingerprint().size, std::filesystem::file_size(sidecar_path));
  const auto cell_adj_a = temp_dir.path() / "a.cell-adj";
  const auto cell_adj_b = temp_dir.path() / "b.cell-adj";
  const auto cell_adj_result_a = powerlaw_ann::io_cell_adjacency_index_t::build(
      cell_adj_a, *index, community, index->community_polar_fingerprint(), 4, 8);
  const auto cell_adj_result_b = powerlaw_ann::io_cell_adjacency_index_t::build(
      cell_adj_b, *index, community, index->community_polar_fingerprint(), 4, 8);
  EXPECT_EQ(read_bytes(cell_adj_a), read_bytes(cell_adj_b));
  EXPECT_EQ(cell_adj_result_a.edge_count, cell_adj_result_b.edge_count);
  const auto cell_adj = powerlaw_ann::io_cell_adjacency_index_t::load(
      cell_adj_a, community.point_count, community.cells.size(),
      index->community_polar_fingerprint());
  EXPECT_EQ(cell_adj->artifact_bytes(), cell_adj_result_a.artifact_bytes);
  EXPECT_EQ(cell_adj->sampled_nodes_per_cell(), 4U);
  EXPECT_EQ(cell_adj->maximum_degree(), 8U);
  for (uint32_t cell = 0; cell < community.cells.size(); ++cell) {
    const auto neighbors = cell_adj->neighbors(cell);
    EXPECT_LE(neighbors.size(), 8U);
    EXPECT_EQ(std::find(neighbors.begin(), neighbors.end(), cell), neighbors.end());
    auto sorted = std::vector<uint32_t>(neighbors.begin(), neighbors.end());
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::unique(sorted.begin(), sorted.end()), sorted.end());
  }
  const auto corrupt_cell_adj = temp_dir.path() / "corrupt.cell-adj";
  std::filesystem::copy_file(cell_adj_a, corrupt_cell_adj);
  flip_byte(corrupt_cell_adj, 128);
  EXPECT_THROW(powerlaw_ann::io_cell_adjacency_index_t::load(
                   corrupt_cell_adj, community.point_count, community.cells.size(),
                   index->community_polar_fingerprint()),
               std::runtime_error);
  const auto rebound_sidecar = temp_dir.path() / "rebound.community_polar.bin";
  const auto rebound_topology = temp_dir.path() / "rebound.topology";
  auto rebound_community = community;
  ++rebound_community.pq_fingerprint.fnv1a_hash;
  powerlaw_ann::write_community_polar_index(rebound_sidecar, rebound_community);
  powerlaw_ann::rebind_io_topology_community_polar(topology_a, sidecar_path, rebound_sidecar,
                                                   rebound_topology);
  const auto rebound_index =
      powerlaw_ann::io_optimized_index_t::load(index_prefix, rebound_topology, vectors_a);
  EXPECT_EQ(rebound_index->community_polar_fingerprint().size,
            std::filesystem::file_size(rebound_sidecar));
  ASSERT_EQ(rebound_index->cell_page_ranges().size(), index->cell_page_ranges().size());
  EXPECT_TRUE(std::equal(rebound_index->cell_page_ranges().begin(),
                         rebound_index->cell_page_ranges().end(),
                         index->cell_page_ranges().begin()));
  EXPECT_EQ(rebound_index->vector_path(), index->vector_path());
  uint64_t expected_first_page = 0;
  for (size_t cell = 0; cell < community.cells.size(); ++cell) {
    const auto& range = index->cell_page_ranges()[cell];
    EXPECT_EQ(range.first_page, expected_first_page);
    EXPECT_EQ(range.node_count, community.cells[cell].node_count);
    expected_first_page = range.first_page + range.page_count;
    uint64_t block_nodes = 0;
    for (uint64_t block = community.cells[cell].block_begin;
         block < community.cells[cell].block_begin + community.cells[cell].block_count; ++block) {
      EXPECT_EQ(community.blocks[block].cell_id, cell);
      EXPECT_LE(community.blocks[block].node_count, community.config.block_target_size);
      block_nodes += community.blocks[block].node_count;
    }
    EXPECT_EQ(block_nodes, community.cells[cell].node_count);
  }
  EXPECT_EQ(expected_first_page, index->vector_page_count());
  for (size_t gateway = 0; gateway < community.gateways.size(); ++gateway) {
    const auto& hint = index->gateway_prefetch_hint(gateway);
    EXPECT_EQ(hint.target_node, community.gateways[gateway].target_node);
    EXPECT_EQ(hint.landing_cell, community.node_to_cell[hint.target_node]);
    EXPECT_EQ(hint.landing_node_page, index->vector_location(hint.target_node).page_id);
    EXPECT_EQ(hint.landing_cell_first_page,
              index->cell_page_ranges()[hint.landing_cell].first_page);
    EXPECT_GE(hint.outgoing_transition_count, hint.transition_counts[0]);
    EXPECT_GE(hint.outgoing_transition_count, hint.transition_counts[1]);
  }
  std::ifstream vector_input(vectors_a, std::ios::binary);
  for (uint32_t node = 0; node < k_point_count; ++node) {
    const auto location = index->vector_location(node);
    const auto range = index->cell_page_ranges()[community.node_to_cell[node]];
    EXPECT_GE(location.page_id, range.first_page);
    EXPECT_LT(location.page_id, range.first_page + range.page_count);
    std::vector<float> vector(k_dimension);
    vector_input.seekg(static_cast<std::streamoff>(
        location.page_id * powerlaw_ann::k_io_vector_page_size + location.page_offset));
    vector_input.read(reinterpret_cast<char*>(vector.data()),
                      static_cast<std::streamsize>(vector.size() * sizeof(float)));
    ASSERT_TRUE(vector_input.good());
    for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
      EXPECT_FLOAT_EQ(vector[dimension], data[node * k_dimension + dimension]);
    }
  }

  const auto query_path = temp_dir.path() / "query.bin";
  constexpr uint32_t query_count = 4;
  std::vector<float> queries(query_count * k_dimension);
  for (uint32_t query = 0; query < query_count; ++query) {
    std::copy_n(data.data() + query * 37U * k_dimension, k_dimension,
                queries.data() + query * k_dimension);
  }
  std::ofstream query_output(query_path, std::ios::binary);
  query_output.write(reinterpret_cast<const char*>(&query_count), sizeof(query_count));
  query_output.write(reinterpret_cast<const char*>(&k_dimension), sizeof(k_dimension));
  query_output.write(reinterpret_cast<const char*>(queries.data()),
                     static_cast<std::streamsize>(queries.size() * sizeof(float)));
  query_output.close();
  powerlaw_ann::diskann_search_config_t search_config;
  search_config.index_path_prefix = index_prefix;
  search_config.query_path = query_path;
  search_config.top_k = 5;
  search_config.search_list_size = 40;
  search_config.beam_width = 2;
  search_config.num_threads = 1;
  const auto baseline = powerlaw_ann::power_ann_t().search_diskann_index(search_config);
  powerlaw_ann::power_ann_config_t io_config;
  io_config.enable_powerann = true;
  io_config.io_optimization.enable_topology_vector = true;
  io_config.io_optimization.enable_cell_layout = true;
  search_config.io_topology_path = topology_a;
  search_config.io_vector_path = vectors_a;
  const auto cell_result = powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(cell_result.ids, baseline.ids);
  EXPECT_EQ(cell_result.distances, baseline.distances);
  ASSERT_EQ(cell_result.query_stats.size(), baseline.query_stats.size());
  for (size_t query = 0; query < baseline.query_stats.size(); ++query) {
    EXPECT_EQ(cell_result.query_stats[query].n_hops, baseline.query_stats[query].n_hops);
    EXPECT_EQ(cell_result.query_stats[query].n_cmps, baseline.query_stats[query].n_cmps);
  }
  EXPECT_NEAR(cell_result.sequential_ios_per_query() + cell_result.random_ios_per_query(),
              cell_result.reads_per_query(), 1e-9);
  EXPECT_GT(cell_result.demand_useful_bytes_per_query(), 0.0);
  EXPECT_GE(cell_result.demand_overread_bytes_per_query(), 0.0);
  io_config.io_optimization.enable_lru_cache = true;
  io_config.io_optimization.lru_cache_pages = 16;
  const auto cached_cell_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(cached_cell_result.ids, baseline.ids);
  EXPECT_EQ(cached_cell_result.distances, baseline.distances);
  EXPECT_EQ(cached_cell_result.io_cache_resident_bytes, 16U * 4096U);
  EXPECT_GT(cached_cell_result.io_cache_demand_hits, 0U);
  EXPECT_GT(cached_cell_result.io_cache_misses, 0U);
  EXPECT_LE(cached_cell_result.reads_per_query(), cell_result.reads_per_query());

  auto reset_after_warmup_config = io_config;
  reset_after_warmup_config.io_optimization.reset_lru_after_warmup = true;
  search_config.warmup_query_count = query_count;
  const auto reset_after_warmup_result =
      powerlaw_ann::power_ann_t(reset_after_warmup_config).search_diskann_index(search_config);
  search_config.warmup_query_count = 0;
  EXPECT_EQ(reset_after_warmup_result.ids, baseline.ids);
  EXPECT_GT(reset_after_warmup_result.io_cache_misses, 0U);

  auto invalid_warmup_reset_config = io_config;
  invalid_warmup_reset_config.io_optimization.enable_lru_cache = false;
  invalid_warmup_reset_config.io_optimization.lru_cache_pages = 0;
  invalid_warmup_reset_config.io_optimization.reset_lru_after_warmup = true;
  EXPECT_THROW(
      powerlaw_ann::power_ann_t(invalid_warmup_reset_config).search_diskann_index(search_config),
      powerlaw_ann::ann_exception_t);

  io_config.io_optimization.enable_cell_page_batch = true;
  const auto batched_cell_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(batched_cell_result.ids, baseline.ids);
  EXPECT_EQ(batched_cell_result.distances, baseline.distances);
  EXPECT_LE(batched_cell_result.physical_read_requests_per_query(),
            batched_cell_result.reads_per_query());
  EXPECT_NEAR(batched_cell_result.sequential_ios_per_query() +
                  batched_cell_result.random_ios_per_query(),
              batched_cell_result.reads_per_query(), 1e-9);

  auto pq_cell_config = io_config;
  pq_cell_config.community_polar_search.mode = powerlaw_ann::powerann_search_mode_t::HYBRID;
  pq_cell_config.community_polar_search.community_skip = true;
  pq_cell_config.community_polar_search.cell_expansion = true;
  pq_cell_config.community_polar_search.community_expansion_budget = 1;
  pq_cell_config.community_polar_search.gateway_score_budget = 4;
  pq_cell_config.community_polar_search.cell_block_budget = 2;
  pq_cell_config.community_polar_search.cell_support = 1;
  pq_cell_config.community_polar_search.cell_min_search_list_size = 1;
  pq_cell_config.io_optimization.enable_cell_pq_traversal = true;
  pq_cell_config.io_optimization.cell_pq_refine_candidates = 5;
  search_config.community_polar_path = sidecar_path;
  const auto pq_cell_result =
      powerlaw_ann::power_ann_t(pq_cell_config).search_diskann_index(search_config);
  const auto repeated_pq_cell_result =
      powerlaw_ann::power_ann_t(pq_cell_config).search_diskann_index(search_config);
  EXPECT_EQ(pq_cell_result.ids, repeated_pq_cell_result.ids);
  EXPECT_EQ(pq_cell_result.distances, repeated_pq_cell_result.distances);
  EXPECT_GT(pq_cell_result.cell_pq_traversal_expansions_per_query(), 0.0);
  EXPECT_DOUBLE_EQ(pq_cell_result.cell_pq_refined_candidates_per_query(), 5.0);
  EXPECT_GT(pq_cell_result.reads_per_query(), 0.0);
#if defined(__linux__)
  pq_cell_config.io_optimization.cell_pq_refine_prefetch_hop = 1;
  const auto prefetched_pq_cell_result =
      powerlaw_ann::power_ann_t(pq_cell_config).search_diskann_index(search_config);
  EXPECT_EQ(prefetched_pq_cell_result.ids, pq_cell_result.ids);
  EXPECT_EQ(prefetched_pq_cell_result.distances, pq_cell_result.distances);
#endif

  auto decoded_config = pq_cell_config;
  decoded_config.io_optimization.cell_pq_refine_candidates = 0;
  decoded_config.io_optimization.cell_pq_refine_prefetch_hop = 0;
  decoded_config.io_optimization.enable_l_aware_search = true;
  decoded_config.io_optimization.l_aware_decoded_max_l = search_config.search_list_size;
  const auto decoded_result =
      powerlaw_ann::power_ann_t(decoded_config).search_diskann_index(search_config);
  EXPECT_EQ(decoded_result.decoded_pq_representation_bytes,
            static_cast<uint64_t>(k_point_count) * k_dimension);
  EXPECT_EQ(decoded_result.decoded_pq_additional_resident_bytes,
            decoded_result.decoded_pq_representation_bytes);
  EXPECT_DOUBLE_EQ(decoded_result.reads_per_query(), 0.0);

  auto refined_l_aware_config = decoded_config;
  refined_l_aware_config.io_optimization.cell_pq_refine_candidates = 5;
  refined_l_aware_config.io_optimization.enable_lru_cache = true;
  refined_l_aware_config.io_optimization.lru_cache_pages = 8;
  const auto refined_l_aware_result =
      powerlaw_ann::power_ann_t(refined_l_aware_config).search_diskann_index(search_config);
  EXPECT_DOUBLE_EQ(refined_l_aware_result.cell_pq_refined_candidates_per_query(), 5.0);
  EXPECT_GT(refined_l_aware_result.reads_per_query(), 0.0);
  EXPECT_EQ(refined_l_aware_result.io_cache_resident_bytes, 8U * 4096U);

  auto budgeted_refine_config = refined_l_aware_config;
  budgeted_refine_config.io_optimization.cell_pq_refine_candidates = 12;
  budgeted_refine_config.io_optimization.enable_l_aware_refine_budget = true;
  budgeted_refine_config.io_optimization.l_aware_refine_base = 4;
  budgeted_refine_config.io_optimization.l_aware_refine_divisor = 10;
  const auto budgeted_refine_result =
      powerlaw_ann::power_ann_t(budgeted_refine_config).search_diskann_index(search_config);
  EXPECT_DOUBLE_EQ(budgeted_refine_result.cell_pq_refined_candidates_per_query(), 8.0);

  auto u8_l_aware_config = refined_l_aware_config;
  u8_l_aware_config.io_optimization.enable_cell_u8_layout = true;
  search_config.io_topology_path = u8_topology;
  search_config.io_vector_path = u8_vectors;
  const auto u8_l_aware_result =
      powerlaw_ann::power_ann_t(u8_l_aware_config).search_diskann_index(search_config);
  EXPECT_EQ(u8_l_aware_result.ids, refined_l_aware_result.ids);
  EXPECT_EQ(u8_l_aware_result.distances, refined_l_aware_result.distances);
  EXPECT_DOUBLE_EQ(u8_l_aware_result.cell_pq_refined_candidates_per_query(), 5.0);
  search_config.io_topology_path = topology_a;
  search_config.io_vector_path = vectors_a;

  auto forced_tree_config = pq_cell_config;
  forced_tree_config.community_polar_search.community_skip = false;
  forced_tree_config.community_polar_search.cell_terminal_convergence = true;
  forced_tree_config.community_polar_search.cell_terminal_approach_hops =
      search_config.search_list_size;
  forced_tree_config.community_polar_search.cell_whole_cell_scan = true;
  forced_tree_config.community_polar_search.cell_centroid_routing = true;
  forced_tree_config.community_polar_search.cell_routing_community_budget = 4;
  forced_tree_config.community_polar_search.cell_routing_cell_budget = 1;
  forced_tree_config.community_polar_search.cell_hierarchical_routing = true;
  forced_tree_config.community_polar_search.cell_hierarchy_probe_budget = 1;
  forced_tree_config.community_polar_search.cell_hierarchy_require_persisted = true;
  forced_tree_config.io_optimization.enable_forced_cell_tree_search = true;
  const auto forced_tree_result =
      powerlaw_ann::power_ann_t(forced_tree_config).search_diskann_index(search_config);
  EXPECT_EQ(forced_tree_result.decoded_pq_representation_bytes,
            static_cast<uint64_t>(k_point_count) * k_dimension);
  EXPECT_EQ(forced_tree_result.decoded_pq_additional_resident_bytes,
            2U * forced_tree_result.decoded_pq_representation_bytes);
  EXPECT_GT(forced_tree_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GT(forced_tree_result.cell_hierarchy_leaf_groups_probed_per_query(), 0.0);
  EXPECT_GT(forced_tree_result.reads_per_query(), 0.0);
  EXPECT_DOUBLE_EQ(forced_tree_result.cell_pq_refined_candidates_per_query(), 5.0);

  auto cacheless_forced_tree_config = forced_tree_config;
  cacheless_forced_tree_config.io_optimization.enable_cell_page_batch = false;
  cacheless_forced_tree_config.io_optimization.enable_lru_cache = false;
  cacheless_forced_tree_config.io_optimization.lru_cache_pages = 0;
  cacheless_forced_tree_config.io_optimization.cell_pq_refine_candidates = 0;
  cacheless_forced_tree_config.io_optimization.cell_pq_refine_prefetch_hop = 0;
  const auto saved_vector_path = search_config.io_vector_path;
  search_config.io_vector_path = temp_dir.path() / "must-not-be-opened.vectors";
  const auto cacheless_forced_tree_result =
      powerlaw_ann::power_ann_t(cacheless_forced_tree_config).search_diskann_index(search_config);
  search_config.io_vector_path = saved_vector_path;
  EXPECT_EQ(cacheless_forced_tree_result.io_cache_resident_bytes, 0U);
  EXPECT_DOUBLE_EQ(cacheless_forced_tree_result.reads_per_query(), 0.0);
  EXPECT_DOUBLE_EQ(cacheless_forced_tree_result.cell_pq_refined_candidates_per_query(), 0.0);
  EXPECT_GT(cacheless_forced_tree_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GT(cacheless_forced_tree_result.cell_hierarchy_leaf_groups_probed_per_query(), 0.0);

  auto decoded_community_tree_config = cacheless_forced_tree_config;
  decoded_community_tree_config.community_polar_search.community_skip = true;
  decoded_community_tree_config.community_polar_search.community_expansion_budget = 1;
  decoded_community_tree_config.community_polar_search.gateway_score_budget = 4;
  const auto decoded_community_tree_result =
      powerlaw_ann::power_ann_t(decoded_community_tree_config).search_diskann_index(search_config);
  EXPECT_DOUBLE_EQ(decoded_community_tree_result.reads_per_query(), 0.0);
  EXPECT_GT(decoded_community_tree_result.community_meta_expansions_per_query(), 0.0);
  EXPECT_GT(decoded_community_tree_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_EQ(decoded_community_tree_result.decoded_pq_additional_resident_bytes,
            2U * decoded_community_tree_result.decoded_pq_representation_bytes);

  auto pq_only_forest_config = cacheless_forced_tree_config;
  pq_only_forest_config.community_polar_search.community_skip = true;
  pq_only_forest_config.community_polar_search.community_expansion_budget = 1;
  pq_only_forest_config.community_polar_search.gateway_score_budget = 4;
  pq_only_forest_config.community_polar_search.cell_block_budget = 13;
  pq_only_forest_config.community_polar_search.cell_routing_cell_budget = 13;
  pq_only_forest_config.community_polar_search.cell_terminal_refine_hops = 27;
  pq_only_forest_config.community_polar_search.cell_terminal_l_aware_refine = true;
  pq_only_forest_config.community_polar_search.cell_terminal_refine_base_hops = 2;
  pq_only_forest_config.community_polar_search.cell_terminal_refine_l_divisor = 5;
  pq_only_forest_config.community_polar_search.cell_terminal_frontier_cells = 4;
  pq_only_forest_config.community_polar_search.cell_terminal_l_aware_forest = true;
  pq_only_forest_config.community_polar_search.cell_terminal_forest_base_cells = 1;
  pq_only_forest_config.community_polar_search.cell_terminal_forest_l_divisor = 10;
  pq_only_forest_config.community_polar_search.cell_hierarchy_probe_budget = 16;
  pq_only_forest_config.community_polar_search.cell_hierarchy_exact_routing = false;
  pq_only_forest_config.io_optimization.enable_compressed_pq_traversal = true;
  const auto pq_only_forest_result =
      powerlaw_ann::power_ann_t(pq_only_forest_config).search_diskann_index(search_config);
  EXPECT_DOUBLE_EQ(pq_only_forest_result.reads_per_query(), 0.0);
  EXPECT_DOUBLE_EQ(pq_only_forest_result.cell_pq_refined_candidates_per_query(), 0.0);
  EXPECT_GT(pq_only_forest_result.cell_hierarchy_nodes_scored_per_query(), 0.0);

  auto pq_only_cell_adj_config = pq_only_forest_config;
  pq_only_cell_adj_config.io_optimization.enable_cell_adj_correction = true;
  pq_only_cell_adj_config.io_optimization.cell_adj_candidate_cells = 2;
  pq_only_cell_adj_config.io_optimization.cell_adj_expansion_cells = 2;
  search_config.io_cell_adjacency_path = cell_adj_a;
  const auto pq_only_cell_adj_result =
      powerlaw_ann::power_ann_t(pq_only_cell_adj_config).search_diskann_index(search_config);
  EXPECT_DOUBLE_EQ(pq_only_cell_adj_result.reads_per_query(), 0.0);
  EXPECT_DOUBLE_EQ(pq_only_cell_adj_result.cell_pq_refined_candidates_per_query(), 0.0);
  EXPECT_GT(pq_only_cell_adj_result.cell_adj_neighbor_edges_considered_per_query(), 0.0);
  EXPECT_LE(pq_only_cell_adj_result.cell_adj_neighbor_edges_considered_per_query(), 16.0);
  EXPECT_GT(pq_only_cell_adj_result.cell_adj_cells_scored_per_query(), 0.0);
  EXPECT_EQ(pq_only_cell_adj_result.cell_adjacency_artifact_bytes,
            std::filesystem::file_size(cell_adj_a));
  EXPECT_GT(pq_only_cell_adj_result.cell_adjacency_resident_bytes, 0U);
  EXPECT_GT(pq_only_cell_adj_result.cell_nodes_scored_per_query(),
            pq_only_forest_result.cell_nodes_scored_per_query());
  auto bounded_cell_adj_config = pq_only_cell_adj_config;
  bounded_cell_adj_config.io_optimization.cell_adj_max_search_list_size = 1;
  const auto bounded_cell_adj_result =
      powerlaw_ann::power_ann_t(bounded_cell_adj_config).search_diskann_index(search_config);
  EXPECT_DOUBLE_EQ(bounded_cell_adj_result.cell_adj_neighbor_edges_considered_per_query(), 0.0);
  EXPECT_DOUBLE_EQ(bounded_cell_adj_result.cell_adj_cells_scored_per_query(), 0.0);
  auto invalid_cell_adj_config = pq_only_cell_adj_config;
  invalid_cell_adj_config.io_optimization.cell_adj_support_shortlist = 1;
  EXPECT_THROW(
      powerlaw_ann::power_ann_t(invalid_cell_adj_config).search_diskann_index(search_config),
      powerlaw_ann::ann_exception_t);
  invalid_cell_adj_config = pq_only_cell_adj_config;
  invalid_cell_adj_config.io_optimization.cell_adj_max_search_list_size = 0;
  EXPECT_THROW(
      powerlaw_ann::power_ann_t(invalid_cell_adj_config).search_diskann_index(search_config),
      powerlaw_ann::ann_exception_t);
  invalid_cell_adj_config = pq_only_cell_adj_config;
  invalid_cell_adj_config.io_optimization.cell_adj_min_search_list_size = 2;
  invalid_cell_adj_config.io_optimization.cell_adj_max_search_list_size = 1;
  EXPECT_THROW(
      powerlaw_ann::power_ann_t(invalid_cell_adj_config).search_diskann_index(search_config),
      powerlaw_ann::ann_exception_t);
  invalid_cell_adj_config = pq_only_cell_adj_config;
  invalid_cell_adj_config.io_optimization.cell_adj_expansion_cells = 0;
  EXPECT_THROW(
      powerlaw_ann::power_ann_t(invalid_cell_adj_config).search_diskann_index(search_config),
      powerlaw_ann::ann_exception_t);

  auto compressed_tree_config = forced_tree_config;
  compressed_tree_config.io_optimization.enable_compressed_pq_traversal = true;
  const auto compressed_tree_result =
      powerlaw_ann::power_ann_t(compressed_tree_config).search_diskann_index(search_config);
  EXPECT_EQ(compressed_tree_result.num_queries, forced_tree_result.num_queries);
  EXPECT_GT(compressed_tree_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GT(compressed_tree_result.cell_hierarchy_leaf_groups_probed_per_query(), 0.0);
  EXPECT_GT(compressed_tree_result.reads_per_query(), 0.0);
  EXPECT_DOUBLE_EQ(compressed_tree_result.cell_pq_refined_candidates_per_query(), 5.0);
  EXPECT_EQ(compressed_tree_result.decoded_pq_representation_bytes, 0U);
  EXPECT_EQ(compressed_tree_result.decoded_pq_additional_resident_bytes, 0U);

  auto exact_cell_batch_config = compressed_tree_config;
  exact_cell_batch_config.community_polar_search.community_skip = true;
  exact_cell_batch_config.io_optimization.enable_cell_batch_search = true;
  exact_cell_batch_config.io_optimization.cell_batch_max_cached_expansions = 1;
  exact_cell_batch_config.io_optimization.cell_pq_refine_candidates = 0;
  exact_cell_batch_config.io_optimization.cell_pq_refine_prefetch_hop = 0;
  exact_cell_batch_config.community_polar_search.cell_terminal_refine_hops = 0;
  auto exact_cell_batch_reference_config = exact_cell_batch_config;
  exact_cell_batch_reference_config.io_optimization.enable_cell_batch_search = false;
  const auto exact_cell_batch_reference =
      powerlaw_ann::power_ann_t(exact_cell_batch_reference_config)
          .search_diskann_index(search_config);
  const auto exact_cell_batch_result =
      powerlaw_ann::power_ann_t(exact_cell_batch_config).search_diskann_index(search_config);
  EXPECT_GT(exact_cell_batch_result.reads_per_query(), 0.0);
  EXPECT_GT(exact_cell_batch_result.cell_pq_refined_candidates_per_query(), 0.0);
  for (uint32_t query = 0; query < query_count; ++query) {
    for (uint32_t rank = 0; rank < search_config.top_k; ++rank) {
      const uint32_t node =
          static_cast<uint32_t>(exact_cell_batch_result.ids[query * search_config.top_k + rank]);
      float exact_distance = 0.0F;
      for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
        const float delta =
            queries[query * k_dimension + dimension] - data[node * k_dimension + dimension];
        exact_distance += delta * delta;
      }
      EXPECT_FLOAT_EQ(exact_cell_batch_result.distances[query * search_config.top_k + rank],
                      exact_distance);
    }
  }
  EXPECT_LE(exact_cell_batch_result.base_hops_per_query(),
            exact_cell_batch_reference.base_hops_per_query());

  auto preserving_cell_batch_config = exact_cell_batch_config;
  preserving_cell_batch_config.io_optimization.cell_batch_preserve_graph = true;
  const auto preserving_cell_batch_result =
      powerlaw_ann::power_ann_t(preserving_cell_batch_config).search_diskann_index(search_config);
  EXPECT_EQ(preserving_cell_batch_result.ids.size(),
            static_cast<size_t>(query_count) * search_config.top_k);
  EXPECT_GT(preserving_cell_batch_result.reads_per_query(), 0.0);
  EXPECT_GT(preserving_cell_batch_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GE(preserving_cell_batch_result.base_hops_per_query(),
            exact_cell_batch_result.base_hops_per_query());

  auto l_aware_stitch_config = preserving_cell_batch_config;
  l_aware_stitch_config.io_optimization.cell_batch_graph_stitch_min_l =
      search_config.search_list_size;
  const auto l_aware_stitch_result =
      powerlaw_ann::power_ann_t(l_aware_stitch_config).search_diskann_index(search_config);
  EXPECT_EQ(l_aware_stitch_result.ids.size(), preserving_cell_batch_result.ids.size());
  EXPECT_GT(l_aware_stitch_result.reads_per_query(), 0.0);
  auto invalid_l_aware_stitch_config = exact_cell_batch_config;
  invalid_l_aware_stitch_config.io_optimization.cell_batch_graph_stitch_min_l = 1;
  EXPECT_THROW(
      powerlaw_ann::power_ann_t(invalid_l_aware_stitch_config).search_diskann_index(search_config),
      powerlaw_ann::ann_exception_t);

  auto ranked_cell_page_config = preserving_cell_batch_config;
  ranked_cell_page_config.io_optimization.enable_ranked_cell_page_refinement = true;
  ranked_cell_page_config.io_optimization.ranked_cell_page_base_pages = 1;
  ranked_cell_page_config.io_optimization.ranked_cell_page_l_divisor = 2;
  ranked_cell_page_config.io_optimization.ranked_cell_page_growth_start_l = 2;
  ranked_cell_page_config.io_optimization.ranked_cell_page_growth_divisor = 2;
  ranked_cell_page_config.io_optimization.ranked_cell_page_max_pages = 2;
  const auto ranked_cell_page_result =
      powerlaw_ann::power_ann_t(ranked_cell_page_config).search_diskann_index(search_config);
  EXPECT_EQ(ranked_cell_page_result.ids.size(),
            static_cast<size_t>(query_count) * search_config.top_k);
  EXPECT_GT(ranked_cell_page_result.reads_per_query(), 0.0);
  EXPECT_LE(ranked_cell_page_result.reads_per_query(),
            preserving_cell_batch_result.reads_per_query());
  EXPECT_GT(ranked_cell_page_result.cell_pq_refined_candidates_per_query(), 0.0);
  EXPECT_GT(ranked_cell_page_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  auto invalid_ranked_cell_page_config = ranked_cell_page_config;
  invalid_ranked_cell_page_config.io_optimization.ranked_cell_page_l_divisor = 0;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_ranked_cell_page_config)
                   .search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
  invalid_ranked_cell_page_config = ranked_cell_page_config;
  invalid_ranked_cell_page_config.io_optimization.ranked_cell_page_base_pages = 3;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_ranked_cell_page_config)
                   .search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
  invalid_ranked_cell_page_config = ranked_cell_page_config;
  invalid_ranked_cell_page_config.io_optimization.ranked_cell_page_growth_divisor = 0;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_ranked_cell_page_config)
                   .search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
  auto gated_candidate_refine_config = ranked_cell_page_config;
  gated_candidate_refine_config.io_optimization.cell_pq_refine_candidates = 5;
  gated_candidate_refine_config.io_optimization.cell_pq_refine_min_l =
      search_config.search_list_size;
  const auto gated_candidate_refine_result =
      powerlaw_ann::power_ann_t(gated_candidate_refine_config).search_diskann_index(search_config);
  EXPECT_EQ(gated_candidate_refine_result.ids.size(), ranked_cell_page_result.ids.size());
  EXPECT_GT(gated_candidate_refine_result.reads_per_query(), 0.0);
  auto invalid_candidate_refine_config = preserving_cell_batch_config;
  invalid_candidate_refine_config.io_optimization.cell_pq_refine_min_l = 2;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_candidate_refine_config)
                   .search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);

  auto invalid_preserving_cell_batch_config = preserving_cell_batch_config;
  invalid_preserving_cell_batch_config.io_optimization.enable_interleaved_cell_batch_search = true;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_preserving_cell_batch_config)
                   .search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);

  auto graph_only_exact_cell_batch_config = exact_cell_batch_config;
  graph_only_exact_cell_batch_config.community_polar_search.community_skip = false;
  const auto graph_only_exact_cell_batch_result =
      powerlaw_ann::power_ann_t(graph_only_exact_cell_batch_config)
          .search_diskann_index(search_config);
  EXPECT_GT(graph_only_exact_cell_batch_result.reads_per_query(), 0.0);
  EXPECT_GT(graph_only_exact_cell_batch_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GT(graph_only_exact_cell_batch_result.cell_hierarchy_leaf_groups_probed_per_query(), 0.0);
  EXPECT_GT(graph_only_exact_cell_batch_result.cell_pq_refined_candidates_per_query(), 0.0);

  auto shared_frontier_exact_cell_batch_config = exact_cell_batch_config;
  shared_frontier_exact_cell_batch_config.io_optimization.enable_forced_cell_tree_search = false;
  shared_frontier_exact_cell_batch_config.io_optimization.enable_compressed_pq_traversal = false;
  const auto shared_frontier_exact_cell_batch_result =
      powerlaw_ann::power_ann_t(shared_frontier_exact_cell_batch_config)
          .search_diskann_index(search_config);
  EXPECT_EQ(shared_frontier_exact_cell_batch_result.ids.size(),
            static_cast<size_t>(query_count) * search_config.top_k);
  EXPECT_GT(shared_frontier_exact_cell_batch_result.base_hops_per_query(), 0.0);
  EXPECT_GT(shared_frontier_exact_cell_batch_result.reads_per_query(), 0.0);
  EXPECT_GT(shared_frontier_exact_cell_batch_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GT(shared_frontier_exact_cell_batch_result.cell_hierarchy_leaf_groups_probed_per_query(),
            0.0);
  EXPECT_GT(shared_frontier_exact_cell_batch_result.cell_pq_refined_candidates_per_query(), 0.0);

  auto graph_approach_exact_cell_batch_config = shared_frontier_exact_cell_batch_config;
  graph_approach_exact_cell_batch_config.community_polar_search.community_skip = false;
  const auto graph_approach_exact_cell_batch_result =
      powerlaw_ann::power_ann_t(graph_approach_exact_cell_batch_config)
          .search_diskann_index(search_config);
  EXPECT_EQ(graph_approach_exact_cell_batch_result.ids.size(),
            static_cast<size_t>(query_count) * search_config.top_k);
  EXPECT_GT(graph_approach_exact_cell_batch_result.base_hops_per_query(), 0.0);
  EXPECT_GT(graph_approach_exact_cell_batch_result.reads_per_query(), 0.0);
  EXPECT_GT(graph_approach_exact_cell_batch_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GT(graph_approach_exact_cell_batch_result.cell_hierarchy_leaf_groups_probed_per_query(),
            0.0);

  auto u8_exact_cell_batch_config = exact_cell_batch_config;
  u8_exact_cell_batch_config.io_optimization.enable_cell_u8_layout = true;
  search_config.io_topology_path = u8_topology;
  search_config.io_vector_path = u8_vectors;
  const auto u8_exact_cell_batch_result =
      powerlaw_ann::power_ann_t(u8_exact_cell_batch_config).search_diskann_index(search_config);
  EXPECT_EQ(u8_exact_cell_batch_result.ids, exact_cell_batch_result.ids);
  EXPECT_EQ(u8_exact_cell_batch_result.distances, exact_cell_batch_result.distances);
  EXPECT_GT(u8_exact_cell_batch_result.reads_per_query(), 0.0);
  EXPECT_GT(u8_exact_cell_batch_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GT(u8_exact_cell_batch_result.cell_hierarchy_leaf_groups_probed_per_query(), 0.0);
  search_config.io_topology_path = topology_a;
  search_config.io_vector_path = vectors_a;

  auto stitched_cell_batch_config = exact_cell_batch_config;
  stitched_cell_batch_config.io_optimization.cell_pq_refine_candidates = 5;
  const auto stitched_cell_batch_result =
      powerlaw_ann::power_ann_t(stitched_cell_batch_config).search_diskann_index(search_config);
  EXPECT_GT(stitched_cell_batch_result.reads_per_query(), 0.0);
  EXPECT_GT(stitched_cell_batch_result.cell_pq_refined_candidates_per_query(), 5.0);
  for (uint32_t query = 0; query < query_count; ++query) {
    for (uint32_t rank = 0; rank < search_config.top_k; ++rank) {
      const uint32_t node =
          static_cast<uint32_t>(stitched_cell_batch_result.ids[query * search_config.top_k + rank]);
      float exact_distance = 0.0F;
      for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
        const float delta =
            queries[query * k_dimension + dimension] - data[node * k_dimension + dimension];
        exact_distance += delta * delta;
      }
      EXPECT_FLOAT_EQ(stitched_cell_batch_result.distances[query * search_config.top_k + rank],
                      exact_distance);
    }
  }

  auto interleaved_cell_batch_config = exact_cell_batch_config;
  interleaved_cell_batch_config.io_optimization.enable_interleaved_cell_batch_search = true;
  interleaved_cell_batch_config.io_optimization.interleaved_cell_batch_graph_hops = 1;
  interleaved_cell_batch_config.io_optimization.cell_batch_max_cached_expansions = 3;
  const auto interleaved_cell_batch_result =
      powerlaw_ann::power_ann_t(interleaved_cell_batch_config).search_diskann_index(search_config);
  EXPECT_GT(interleaved_cell_batch_result.reads_per_query(), 0.0);
  EXPECT_GE(interleaved_cell_batch_result.cell_pq_refined_candidates_per_query(),
            exact_cell_batch_result.cell_pq_refined_candidates_per_query());
  for (uint32_t query = 0; query < query_count; ++query) {
    for (uint32_t rank = 0; rank < search_config.top_k; ++rank) {
      const uint32_t node = static_cast<uint32_t>(
          interleaved_cell_batch_result.ids[query * search_config.top_k + rank]);
      float exact_distance = 0.0F;
      for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
        const float delta =
            queries[query * k_dimension + dimension] - data[node * k_dimension + dimension];
        exact_distance += delta * delta;
      }
      EXPECT_FLOAT_EQ(interleaved_cell_batch_result.distances[query * search_config.top_k + rank],
                      exact_distance);
    }
  }

  auto cell_adj_config = exact_cell_batch_config;
  cell_adj_config.io_optimization.enable_cell_adj_correction = true;
  cell_adj_config.io_optimization.cell_adj_candidate_cells = 2;
  cell_adj_config.io_optimization.cell_batch_max_cached_expansions = 3;
  const auto cell_adj_result =
      powerlaw_ann::power_ann_t(cell_adj_config).search_diskann_index(search_config);
  EXPECT_GT(cell_adj_result.cell_adj_neighbor_edges_considered_per_query(), 0.0);
  EXPECT_GT(cell_adj_result.cell_adj_cells_scored_per_query(), 0.0);
  EXPECT_GT(cell_adj_result.cell_pq_refined_candidates_per_query(),
            exact_cell_batch_result.cell_pq_refined_candidates_per_query());
  EXPECT_EQ(cell_adj_result.resident_u8_refinement_bytes, 0U);

  auto leaf_cell_adj_config = compressed_tree_config;
  leaf_cell_adj_config.io_optimization.cell_pq_refine_candidates = 0;
  leaf_cell_adj_config.io_optimization.cell_pq_refine_prefetch_hop = 0;
  leaf_cell_adj_config.io_optimization.enable_cell_leaf_refinement = true;
  leaf_cell_adj_config.io_optimization.enable_cell_adj_correction = true;
  leaf_cell_adj_config.io_optimization.cell_adj_candidate_cells = 2;
  const auto leaf_cell_adj_result =
      powerlaw_ann::power_ann_t(leaf_cell_adj_config).search_diskann_index(search_config);
  EXPECT_GT(leaf_cell_adj_result.cell_adj_neighbor_edges_considered_per_query(), 0.0);
  EXPECT_GT(leaf_cell_adj_result.cell_adj_cells_scored_per_query(), 0.0);
  EXPECT_GT(leaf_cell_adj_result.cell_pq_refined_candidates_per_query(),
            compressed_tree_result.cell_pq_refined_candidates_per_query());
  EXPECT_EQ(leaf_cell_adj_result.resident_u8_refinement_bytes, 0U);

  auto resident_u8_config = compressed_tree_config;
  resident_u8_config.io_optimization.enable_resident_u8_refinement = true;
  resident_u8_config.io_optimization.cell_pq_refine_prefetch_hop = 0;
  search_config.resident_u8_refinement_path = resident_u8;
  const auto resident_u8_result =
      powerlaw_ann::power_ann_t(resident_u8_config).search_diskann_index(search_config);
  EXPECT_EQ(resident_u8_result.ids, compressed_tree_result.ids);
  EXPECT_EQ(resident_u8_result.distances, compressed_tree_result.distances);
  EXPECT_DOUBLE_EQ(resident_u8_result.reads_per_query(), 0.0);
  EXPECT_EQ(resident_u8_result.resident_u8_refinement_artifact_bytes,
            std::filesystem::file_size(resident_u8));
  EXPECT_EQ(resident_u8_result.resident_u8_refinement_bytes,
            static_cast<uint64_t>(k_point_count) * k_dimension);
  EXPECT_NE(resident_u8_result.resident_u8_refinement_checksum, 0U);

  resident_u8_config.io_optimization.enable_resident_u8_traversal = true;
  const auto resident_u8_traversal_result =
      powerlaw_ann::power_ann_t(resident_u8_config).search_diskann_index(search_config);
  EXPECT_DOUBLE_EQ(resident_u8_traversal_result.reads_per_query(), 0.0);
  EXPECT_GE(resident_u8_traversal_result.recall_percent.value_or(0.0),
            resident_u8_result.recall_percent.value_or(0.0));

  auto skip_tree_config = compressed_tree_config;
  skip_tree_config.community_polar_search.community_skip = true;
  skip_tree_config.community_polar_search.community_expansion_budget = 1;
  skip_tree_config.community_polar_search.gateway_score_budget = 8;
  const auto skip_tree_result =
      powerlaw_ann::power_ann_t(skip_tree_config).search_diskann_index(search_config);
  EXPECT_GT(skip_tree_result.cell_hierarchy_nodes_scored_per_query(), 0.0);
  EXPECT_GT(skip_tree_result.community_meta_expansions_per_query(), 0.0);

  const auto malformed_u8 = temp_dir.path() / "malformed.u8bin";
  std::filesystem::copy_file(resident_u8, malformed_u8);
  flip_byte(malformed_u8, 0);
  search_config.resident_u8_refinement_path = malformed_u8;
  EXPECT_THROW(powerlaw_ann::power_ann_t(resident_u8_config).search_diskann_index(search_config),
               powerlaw_ann::diskann_exception_t);
  search_config.resident_u8_refinement_path.clear();

  auto leaf_refine_config = compressed_tree_config;
  leaf_refine_config.io_optimization.cell_pq_refine_candidates = 0;
  leaf_refine_config.io_optimization.cell_pq_refine_prefetch_hop = 0;
  leaf_refine_config.io_optimization.enable_cell_leaf_refinement = true;
  const auto leaf_refine_result =
      powerlaw_ann::power_ann_t(leaf_refine_config).search_diskann_index(search_config);
  EXPECT_EQ(leaf_refine_result.num_queries, compressed_tree_result.num_queries);
  EXPECT_GT(leaf_refine_result.reads_per_query(), 0.0);
  EXPECT_GE(leaf_refine_result.cell_pq_refined_candidates_per_query(),
            static_cast<double>(search_config.top_k));
  auto stitched_leaf_refine_config = leaf_refine_config;
  stitched_leaf_refine_config.io_optimization.cell_pq_refine_candidates = 5;
  stitched_leaf_refine_config.community_polar_search.cell_terminal_stitch_promote_cells = true;
  stitched_leaf_refine_config.community_polar_search.cell_terminal_stitch_cell_budget = 2;
  stitched_leaf_refine_config.community_polar_search.cell_terminal_stitch_min_support = 1;
  const auto stitched_leaf_refine_result =
      powerlaw_ann::power_ann_t(stitched_leaf_refine_config).search_diskann_index(search_config);
  EXPECT_GT(stitched_leaf_refine_result.cell_pq_refined_candidates_per_query(),
            leaf_refine_result.cell_pq_refined_candidates_per_query());
  search_config.community_polar_path.clear();

  auto online_config = io_config;
  online_config.io_optimization.enable_cell_page_batch = false;
  online_config.community_polar_search.mode = powerlaw_ann::powerann_search_mode_t::HYBRID;
  online_config.community_polar_search.cell_expansion = true;
  search_config.community_polar_path = sidecar_path;
  const auto online_control =
      powerlaw_ann::power_ann_t(online_config).search_diskann_index(search_config);
  online_config.io_optimization.enable_online_cell_adaptation = true;
  const auto online_result =
      powerlaw_ann::power_ann_t(online_config).search_diskann_index(search_config);
  EXPECT_EQ(online_result.ids, online_control.ids);
  EXPECT_EQ(online_result.distances, online_control.distances);
  EXPECT_GT(online_result.online_cell_adaptation_observations, 0U);
  EXPECT_EQ(online_result.online_cell_adaptation_successful_publishes, 1U);
  EXPECT_EQ(online_result.online_cell_adaptation_physical_generations, 1U);
  EXPECT_GT(online_result.online_cell_adaptation_physical_bytes_written, 0U);
  EXPECT_GT(online_result.online_cell_adaptation_physical_write_time_us, 0U);
  EXPECT_EQ(online_result.online_cell_adaptation_checksum_failures, 0U);
  EXPECT_GT(online_result.online_cell_adaptation_directory_resident_bytes, 0U);
  EXPECT_GT(online_result.online_cell_adaptation_overlay_artifact_bytes, 0U);
  search_config.community_polar_path.clear();

  io_config.io_optimization.enable_cell_page_batch = false;
  io_config.io_optimization.enable_cell_layout = false;
  io_config.io_optimization.enable_chunk_layout = true;
  io_config.io_optimization.enable_lru_cache = false;
  io_config.io_optimization.lru_cache_pages = 0;
  search_config.io_topology_path = chunk_topology;
  search_config.io_vector_path = chunk_vectors;
  const auto chunk_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(chunk_result.ids, baseline.ids);
  EXPECT_EQ(chunk_result.distances, baseline.distances);

  io_config.io_optimization.enable_chunk_layout = false;
  io_config.io_optimization.enable_weighted_reorder = true;
  search_config.io_topology_path = weighted_topology;
  search_config.io_vector_path = weighted_vectors;
  const auto weighted_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(weighted_result.ids, baseline.ids);
  EXPECT_EQ(weighted_result.distances, baseline.distances);

  io_config.io_optimization.memgraph_mode = powerlaw_ann::io_memgraph_mode_t::RANDOM;
  io_config.io_optimization.memgraph_nodes = 32;
  io_config.io_optimization.memgraph_entry_candidates = 4;
  io_config.io_optimization.memgraph_seed = 7;
  const auto random_memgraph_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  const auto repeated_random_memgraph_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(random_memgraph_result.ids, repeated_random_memgraph_result.ids);
  EXPECT_EQ(random_memgraph_result.distances, repeated_random_memgraph_result.distances);
  EXPECT_EQ(random_memgraph_result.memgraph_nodes, 32U);
  EXPECT_GT(random_memgraph_result.memgraph_resident_bytes, 0U);
  EXPECT_DOUBLE_EQ(random_memgraph_result.memgraph_nodes_scored_per_query(), 32.0);
  EXPECT_LE(random_memgraph_result.memgraph_nodes_inserted_per_query(), 4.0);
  EXPECT_LE(random_memgraph_result.memgraph_nodes_later_expanded_per_query(),
            random_memgraph_result.memgraph_nodes_inserted_per_query());

  io_config.community_polar_search.mode = powerlaw_ann::powerann_search_mode_t::HYBRID;
  io_config.community_polar_search.community_skip = true;
  io_config.io_optimization.memgraph_mode = powerlaw_ann::io_memgraph_mode_t::COMMUNITY_IMPORTANT;
  search_config.community_polar_path = sidecar_path;
  const auto important_memgraph_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(important_memgraph_result.memgraph_nodes, 32U);
  EXPECT_GT(important_memgraph_result.memgraph_resident_bytes, 0U);
  EXPECT_DOUBLE_EQ(important_memgraph_result.memgraph_nodes_scored_per_query(), 32.0);

  io_config.community_polar_search = {};
  search_config.community_polar_path.clear();
  io_config.io_optimization.memgraph_mode = powerlaw_ann::io_memgraph_mode_t::HIGH_DEGREE;
  const auto high_degree_memgraph_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(high_degree_memgraph_result.memgraph_nodes, 32U);
  EXPECT_DOUBLE_EQ(high_degree_memgraph_result.memgraph_nodes_scored_per_query(), 32.0);
  io_config.io_optimization.memgraph_mode = powerlaw_ann::io_memgraph_mode_t::RCNI_ONLY;
  const auto rcni_memgraph_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(rcni_memgraph_result.memgraph_nodes, 32U);
  EXPECT_DOUBLE_EQ(rcni_memgraph_result.memgraph_nodes_scored_per_query(), 32.0);

#if defined(__linux__)
  io_config.io_optimization.memgraph_mode = powerlaw_ann::io_memgraph_mode_t::NONE;
  io_config.io_optimization.memgraph_nodes = 0;
  io_config.io_optimization.enable_io_uring = true;
  io_config.io_optimization.io_uring_queue_depth = 8;
  io_config.io_optimization.enable_dynamic_width = true;
  io_config.io_optimization.dynamic_width_initial = 1;
  io_config.io_optimization.dynamic_width_marker = 0;
  io_config.io_optimization.dynamic_width_waste_threshold = 1.0F;
  search_config.community_polar_path.clear();
  const auto dynamic_width_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(dynamic_width_result.ids, baseline.ids);
  EXPECT_EQ(dynamic_width_result.distances, baseline.distances);
  EXPECT_GT(dynamic_width_result.dynamic_width_mean(), 0.0);
  EXPECT_GE(dynamic_width_result.dynamic_width_max(), 1U);
  EXPECT_GT(dynamic_width_result.io_uring_submitted, 0U);

  io_config.io_optimization.enable_dynamic_width = false;
  io_config.io_optimization.enable_weighted_reorder = false;
  io_config.io_optimization.enable_cell_layout = true;
  io_config.io_optimization.enable_lru_cache = true;
  io_config.io_optimization.lru_cache_pages = 16;
  io_config.io_optimization.enable_node_prefetch = true;
  search_config.io_topology_path = topology_a;
  search_config.io_vector_path = vectors_a;
  const auto node_prefetch_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(node_prefetch_result.ids, baseline.ids);
  EXPECT_EQ(node_prefetch_result.distances, baseline.distances);
  EXPECT_GT(node_prefetch_result.io_uring_prefetch_submitted, 0U);

  io_config.io_optimization.enable_node_prefetch = false;
  io_config.community_polar_search.mode = powerlaw_ann::powerann_search_mode_t::HYBRID;
  io_config.community_polar_search.community_skip = true;
  io_config.community_polar_search.cell_expansion = true;
  io_config.community_polar_search.cell_block_budget = 4;
  io_config.community_polar_search.cell_support = 1;
  io_config.community_polar_search.cell_insert_cap = 8;
  io_config.community_polar_search.cell_min_search_list_size = 1;
  io_config.community_polar_search.cell_whole_cell_scan = true;
  io_config.community_polar_search.cell_centroid_routing = true;
  io_config.community_polar_search.gateway_landing_cell_handoff = true;
  search_config.community_polar_path = sidecar_path;
  const auto hybrid_demand_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  io_config.io_optimization.enable_cell_prefetch = true;
  const auto cell_prefetch_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(cell_prefetch_result.ids, hybrid_demand_result.ids);
  EXPECT_EQ(cell_prefetch_result.distances, hybrid_demand_result.distances);
  EXPECT_GT(cell_prefetch_result.io_uring_prefetch_submitted, 0U);

  io_config.io_optimization.enable_cell_prefetch = false;
  io_config.io_optimization.enable_gateway_prefetch = true;
  io_config.io_optimization.gateway_prefetch_max_outstanding = 2;
  const auto gateway_prefetch_result =
      powerlaw_ann::power_ann_t(io_config).search_diskann_index(search_config);
  EXPECT_EQ(gateway_prefetch_result.ids, hybrid_demand_result.ids);
  EXPECT_EQ(gateway_prefetch_result.distances, hybrid_demand_result.distances);
  EXPECT_GT(gateway_prefetch_result.gateway_prefetch_predictions_per_query(), 0.0);
  EXPECT_GT(gateway_prefetch_result.gateway_prefetch_pages_submitted_per_query(), 0.0);
  EXPECT_EQ(gateway_prefetch_result.io_uring_prefetch_submitted,
            gateway_prefetch_result.io_uring_prefetch_completed);
#endif

  const auto corrupt = temp_dir.path() / "cell-corrupt.topology";
  std::filesystem::copy_file(topology_a, corrupt);
  flip_byte(corrupt, std::filesystem::file_size(corrupt) - 1U);
  EXPECT_THROW(powerlaw_ann::io_optimized_index_t::load(index_prefix, corrupt, vectors_a),
               std::runtime_error);
}

} // namespace

TEST(IoOptimizedIndexTest, ScoresUnalignedNonMultipleOfEightFloatVectors) {
  alignas(32) std::array<float, 204> storage{};
  float* left = storage.data() + 1;
  float* right = storage.data() + 103;
  float expected = 0.0F;
  for (uint32_t dimension = 0; dimension < 100; ++dimension) {
    left[dimension] = static_cast<float>(static_cast<int32_t>(dimension % 17) - 8);
    right[dimension] = static_cast<float>(static_cast<int32_t>(dimension % 13) - 6);
    const float delta = left[dimension] - right[dimension];
    expected += delta * delta;
  }
  EXPECT_FLOAT_EQ(powerlaw_ann::l2_distance_float_t().compare(left, right, 100), expected);
}
