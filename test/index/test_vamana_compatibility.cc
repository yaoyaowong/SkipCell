#if defined(__APPLE__)
// The Homebrew Boost version used by the macOS correctness host still derives
// hash helpers from std::unary_function.
#define _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "common/utils.h"
#include "index/abstract_index.h"
#include "index/index_config.h"
#include "index/index_factory.h"
#include "power_ann.h"

#if defined(__APPLE__)
#pragma clang diagnostic pop
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kNumPoints = 64;
constexpr uint32_t kDimensions = 128;
constexpr uint32_t kMaxDegree = 8;
constexpr uint32_t kBuildListSize = 16;
constexpr uint32_t kSearchListSize = 64;
constexpr uint32_t kResultCount = 10;

class temp_dir_t {
public:
  temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_vamana_compatibility_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

template <typename value_t>
struct bin_matrix_t {
  uint32_t rows = 0;
  uint32_t columns = 0;
  std::vector<value_t> values;
};

struct vamana_graph_t {
  uint64_t file_size = 0;
  uint32_t max_degree = 0;
  uint32_t medoid = 0;
  uint64_t num_frozen_points = 0;
  uint64_t edge_count = 0;
  std::vector<std::vector<uint32_t>> adjacency;
};

struct search_result_t {
  std::vector<uint32_t> ids;
  std::vector<float> distances;
};

template <typename value_t>
value_t read_value(std::ifstream& input) {
  value_t value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) {
    throw std::runtime_error("unexpected end of binary file");
  }
  return value;
}

std::pair<uint32_t, uint32_t> read_bin_metadata(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open bin file: " + path.string());
  }
  const auto rows = read_value<int32_t>(input);
  const auto columns = read_value<int32_t>(input);
  if (rows <= 0 || columns <= 0) {
    throw std::runtime_error("bin metadata must be positive: " + path.string());
  }
  return {static_cast<uint32_t>(rows), static_cast<uint32_t>(columns)};
}

template <typename value_t>
bin_matrix_t<value_t> read_bin(const std::filesystem::path& path) {
  const auto [rows, columns] = read_bin_metadata(path);
  bin_matrix_t<value_t> matrix;
  matrix.rows = rows;
  matrix.columns = columns;
  matrix.values.resize(static_cast<size_t>(rows) * columns);

  std::ifstream input(path, std::ios::binary);
  input.seekg(2 * sizeof(int32_t), std::ios::beg);
  input.read(reinterpret_cast<char*>(matrix.values.data()),
             static_cast<std::streamsize>(matrix.values.size() * sizeof(value_t)));
  if (!input) {
    throw std::runtime_error("truncated bin payload: " + path.string());
  }
  return matrix;
}

std::vector<float> make_vector(uint32_t point) {
  std::vector<float> vector(kDimensions);
  for (uint32_t dimension = 0; dimension < kDimensions; ++dimension) {
    const uint32_t mixed = (point * 67U + dimension * 31U + point * dimension * 7U) % 251U;
    vector[dimension] = static_cast<float>(mixed) / 251.0F + static_cast<float>(point) * 0.01F;
  }
  return vector;
}

void write_dataset(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
  const int32_t rows = static_cast<int32_t>(kNumPoints);
  const int32_t columns = static_cast<int32_t>(kDimensions);
  output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  output.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
  for (uint32_t point = 0; point < kNumPoints; ++point) {
    const auto vector = make_vector(point);
    output.write(reinterpret_cast<const char*>(vector.data()),
                 static_cast<std::streamsize>(vector.size() * sizeof(float)));
  }
  if (!output) {
    throw std::runtime_error("failed to write generated dataset: " + path.string());
  }
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open artifact: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

vamana_graph_t read_graph(const std::filesystem::path& path, uint32_t expected_nodes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open Vamana graph: " + path.string());
  }

  vamana_graph_t graph;
  graph.file_size = read_value<uint64_t>(input);
  graph.max_degree = read_value<uint32_t>(input);
  graph.medoid = read_value<uint32_t>(input);
  graph.num_frozen_points = read_value<uint64_t>(input);

  constexpr uint64_t header_size = sizeof(uint64_t) + 2 * sizeof(uint32_t) + sizeof(uint64_t);
  if (graph.file_size != std::filesystem::file_size(path) || graph.file_size < header_size) {
    throw std::runtime_error("Vamana graph file-size metadata is invalid: " + path.string());
  }

  uint64_t bytes_read = header_size;
  uint32_t observed_max_degree = 0;
  while (bytes_read < graph.file_size) {
    const uint32_t degree = read_value<uint32_t>(input);
    bytes_read += sizeof(uint32_t);
    const uint64_t neighbor_bytes = static_cast<uint64_t>(degree) * sizeof(uint32_t);
    if (bytes_read + neighbor_bytes > graph.file_size) {
      throw std::runtime_error("Vamana adjacency exceeds declared file size: " + path.string());
    }

    std::vector<uint32_t> neighbors(degree);
    input.read(reinterpret_cast<char*>(neighbors.data()),
               static_cast<std::streamsize>(neighbor_bytes));
    if (!input) {
      throw std::runtime_error("truncated Vamana adjacency: " + path.string());
    }
    for (const uint32_t neighbor : neighbors) {
      if (neighbor >= expected_nodes) {
        throw std::runtime_error("Vamana neighbor id is outside the data range: " +
                                 std::to_string(neighbor));
      }
    }

    bytes_read += neighbor_bytes;
    graph.edge_count += degree;
    observed_max_degree = std::max(observed_max_degree, degree);
    graph.adjacency.push_back(std::move(neighbors));
  }

  if (bytes_read != graph.file_size || graph.adjacency.size() != expected_nodes) {
    throw std::runtime_error("Vamana node count does not match the data artifact: " +
                             path.string());
  }
  if (observed_max_degree != graph.max_degree) {
    throw std::runtime_error("Vamana width does not match the observed maximum degree: " +
                             path.string());
  }
  return graph;
}

powerlaw_ann::diskann_memory_index_config_t
make_build_config(const std::filesystem::path& data_path,
                  const std::filesystem::path& index_prefix) {
  powerlaw_ann::diskann_memory_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_prefix;
  config.max_degree = kMaxDegree;
  config.build_list_size = kBuildListSize;
  config.alpha = 1.2F;
  config.num_threads = 1;
  return config;
}

std::unique_ptr<powerlaw_ann::abstract_index_t>
load_memory_index(const std::filesystem::path& index_prefix, uint32_t dimensions,
                  uint32_t search_list_size) {
  const size_t num_frozen_points = powerlaw_ann::get_graph_num_frozen_points(index_prefix.string());
  auto config = powerlaw_ann::index_config_builder_t()
                    .with_metric(powerlaw_ann::metric_t::L2)
                    .with_dimension(dimensions)
                    .with_max_points(0)
                    .with_data_load_store_strategy(powerlaw_ann::data_store_strategy_t::MEMORY)
                    .with_graph_load_store_strategy(powerlaw_ann::graph_store_strategy_t::MEMORY)
                    .with_data_type("float")
                    .with_label_type("uint32")
                    .with_tag_type("uint32")
                    .is_dynamic_index(false)
                    .is_enable_tags(false)
                    .is_concurrent_consolidate(false)
                    .is_pq_dist_build(false)
                    .is_use_opq(false)
                    .with_num_pq_chunks(0)
                    .with_num_frozen_pts(num_frozen_points)
                    .build();

  powerlaw_ann::index_factory_t factory(config);
  auto index = factory.create_instance();
  index->load(index_prefix.c_str(), 1, search_list_size);
  return index;
}

search_result_t search(powerlaw_ann::abstract_index_t& index, const float* query,
                       uint32_t result_count, uint32_t search_list_size) {
  search_result_t result;
  result.ids.resize(result_count);
  result.distances.resize(result_count);
  index.search(query, result_count, search_list_size, result.ids.data(), result.distances.data());
  return result;
}

uint32_t external_search_list_size(uint32_t result_count) {
  const char* value = std::getenv("POWERLAWANN_DISKANN_SEARCH_LIST_SIZE");
  if (value == nullptr) {
    return std::max(100U, result_count);
  }
  const auto parsed = static_cast<uint32_t>(std::stoul(value));
  if (parsed < result_count) {
    throw std::runtime_error("POWERLAWANN_DISKANN_SEARCH_LIST_SIZE must be at least result count");
  }
  return parsed;
}

TEST(VamanaCompatibilityTest, GeneratedFixtureHasDeterministicFormatAndSearchResults) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "sift-shape-base.fbin";
  const auto first_prefix = temp_dir.path() / "first.index";
  const auto second_prefix = temp_dir.path() / "second.index";
  write_dataset(data_path);

  powerlaw_ann::power_ann_t power_ann;
  power_ann.build_diskann_memory_index(make_build_config(data_path, first_prefix));
  power_ann.build_diskann_memory_index(make_build_config(data_path, second_prefix));

  const auto first_graph = read_graph(first_prefix, kNumPoints);
  const auto second_graph = read_graph(second_prefix, kNumPoints);
  EXPECT_EQ(first_graph.file_size, second_graph.file_size);
  EXPECT_EQ(first_graph.max_degree, second_graph.max_degree);
  EXPECT_EQ(first_graph.medoid, second_graph.medoid);
  EXPECT_EQ(first_graph.num_frozen_points, 0U);
  EXPECT_EQ(first_graph.num_frozen_points, second_graph.num_frozen_points);
  EXPECT_EQ(first_graph.edge_count, second_graph.edge_count);
  EXPECT_EQ(first_graph.adjacency, second_graph.adjacency);
  EXPECT_LT(first_graph.medoid, kNumPoints);
  EXPECT_LE(first_graph.max_degree, kMaxDegree);
  EXPECT_GT(first_graph.edge_count, 0U);
  EXPECT_EQ(read_file(first_prefix), read_file(second_prefix));
  EXPECT_EQ(read_file(first_prefix.string() + ".data"),
            read_file(second_prefix.string() + ".data"));

  auto first_index = load_memory_index(first_prefix, kDimensions, kSearchListSize);
  auto second_index = load_memory_index(second_prefix, kDimensions, kSearchListSize);
  for (const uint32_t query_id : {0U, 17U, 63U}) {
    const auto query = make_vector(query_id);
    const auto first_result = search(*first_index, query.data(), kResultCount, kSearchListSize);
    const auto second_result = search(*second_index, query.data(), kResultCount, kSearchListSize);
    EXPECT_EQ(first_result.ids, second_result.ids);
    EXPECT_EQ(first_result.distances, second_result.distances);
    ASSERT_FALSE(first_result.ids.empty());
    EXPECT_EQ(first_result.ids.front(), query_id);
    EXPECT_FLOAT_EQ(first_result.distances.front(), 0.0F);
  }
}

TEST(VamanaCompatibilityTest, ExternalDiskannArtifactMatchesPinnedLoaderResults) {
  const char* index_prefix = std::getenv("POWERLAWANN_DISKANN_INDEX_PREFIX");
  const char* query_file = std::getenv("POWERLAWANN_DISKANN_QUERY_FILE");
  const char* result_file = std::getenv("POWERLAWANN_DISKANN_RESULT_FILE");
  if (index_prefix == nullptr && query_file == nullptr && result_file == nullptr) {
    GTEST_SKIP() << "set the POWERLAWANN_DISKANN_* variables on the Ubuntu "
                    "baseline host";
  }

  ASSERT_NE(index_prefix, nullptr);
  ASSERT_NE(query_file, nullptr);
  ASSERT_NE(result_file, nullptr);
  const std::filesystem::path prefix(index_prefix);
  ASSERT_TRUE(std::filesystem::is_regular_file(prefix));
  ASSERT_TRUE(std::filesystem::is_regular_file(prefix.string() + ".data"));

  const auto queries = read_bin<float>(query_file);
  const auto expected = read_bin<uint32_t>(result_file);
  const auto [data_rows, data_dimensions] = read_bin_metadata(prefix.string() + ".data");
  EXPECT_GT(data_rows, 0U);
  ASSERT_EQ(queries.columns, data_dimensions);
  ASSERT_EQ(queries.rows, expected.rows);
  ASSERT_GT(expected.columns, 0U);

  const uint32_t search_list_size = external_search_list_size(expected.columns);
  auto index = load_memory_index(prefix, data_dimensions, search_list_size);
  for (uint32_t query_id = 0; query_id < queries.rows; ++query_id) {
    const float* query = queries.values.data() + static_cast<size_t>(query_id) * queries.columns;
    const auto actual = search(*index, query, expected.columns, search_list_size);
    for (uint32_t rank = 0; rank < expected.columns; ++rank) {
      EXPECT_EQ(actual.ids[rank],
                expected.values[static_cast<size_t>(query_id) * expected.columns + rank])
          << "query=" << query_id << " rank=" << rank;
    }
  }
}

} // namespace
