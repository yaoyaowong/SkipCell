#include "common/ann_error.h"
#include "index/neighbor.h"
#include "power_ann.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

TEST(DiskannTopkSearchTest, ExactCandidateScoreReopensAnExistingFrontierNode) {
  powerlaw_ann::neighbor_priority_queue_t queue(4);
  queue.insert({7, 10.0F});
  queue.insert({8, 20.0F});

  queue.insert_or_update({7, 1.0F});
  ASSERT_TRUE(queue.has_unexpanded_node());
  const auto corrected = queue.closest_unexpanded();
  EXPECT_EQ(corrected.id, 7U);
  EXPECT_FLOAT_EQ(corrected.distance, 1.0F);
}

class temp_dir_t {
public:
  temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_topk_search_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

constexpr uint32_t k_num_points = 300;
constexpr uint32_t k_dimension = 8;

std::vector<float> make_dataset() {
  std::vector<float> data(k_num_points * k_dimension);
  for (uint32_t point = 0; point < k_num_points; ++point) {
    for (uint32_t dim = 0; dim < k_dimension; ++dim) {
      data[point * k_dimension + dim] =
          static_cast<float>(point) * (0.01F + static_cast<float>(dim) * 0.002F) +
          std::sin(static_cast<float>(point + dim * 17) * 0.07F);
    }
  }
  return data;
}

void write_bin(const std::filesystem::path& path, const float* data, uint32_t rows,
               uint32_t dimensions) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  output.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
  output.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(rows) * dimensions * sizeof(float));
  ASSERT_TRUE(output.good());
}

std::vector<std::pair<float, uint32_t>> exact_top_k(const std::vector<float>& data,
                                                    const float* query, size_t top_k) {
  std::vector<std::pair<float, uint32_t>> distances;
  distances.reserve(k_num_points);
  for (uint32_t point = 0; point < k_num_points; ++point) {
    float distance = 0.0F;
    for (uint32_t dim = 0; dim < k_dimension; ++dim) {
      const float delta = query[dim] - data[point * k_dimension + dim];
      distance += delta * delta;
    }
    distances.emplace_back(distance, point);
  }
  std::sort(distances.begin(), distances.end(), [](const auto& left, const auto& right) {
    return left.first < right.first || (left.first == right.first && left.second < right.second);
  });
  distances.resize(top_k);
  return distances;
}

void write_truthset(const std::filesystem::path& path, const std::vector<float>& data,
                    const std::vector<float>& queries, uint32_t top_k) {
  const uint32_t num_queries = static_cast<uint32_t>(queries.size() / k_dimension);
  std::vector<uint32_t> ids(num_queries * top_k);
  std::vector<float> distances(num_queries * top_k);
  for (uint32_t query = 0; query < num_queries; ++query) {
    const auto exact = exact_top_k(data, queries.data() + query * k_dimension, top_k);
    for (uint32_t rank = 0; rank < top_k; ++rank) {
      ids[query * top_k + rank] = exact[rank].second;
      distances[query * top_k + rank] = exact[rank].first;
    }
  }

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&num_queries), sizeof(num_queries));
  output.write(reinterpret_cast<const char*>(&top_k), sizeof(top_k));
  output.write(reinterpret_cast<const char*>(ids.data()),
               static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
  output.write(reinterpret_cast<const char*>(distances.data()),
               static_cast<std::streamsize>(distances.size() * sizeof(float)));
  ASSERT_TRUE(output.good());
}

TEST(DiskannTopkSearchTest, ReturnsExactSelfMatchesThroughFacade) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto query_path = temp_dir.path() / "query.bin";
  const auto ground_truth_path = temp_dir.path() / "ground_truth.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  const auto result_prefix = temp_dir.path() / "result";
  const auto visit_path = temp_dir.path() / "visits.csv";

  const auto data = make_dataset();
  write_bin(data_path, data.data(), k_num_points, k_dimension);

  const std::vector<uint32_t> expected_ids = {0, 37, 91, 173, 299};
  std::vector<float> queries(expected_ids.size() * k_dimension);
  for (size_t query = 0; query < expected_ids.size(); ++query) {
    std::copy_n(data.data() + expected_ids[query] * k_dimension, k_dimension,
                queries.data() + query * k_dimension);
  }
  write_bin(query_path, queries.data(), static_cast<uint32_t>(expected_ids.size()), k_dimension);
  write_truthset(ground_truth_path, data, queries, 5);

  powerlaw_ann::power_ann_t power_ann;
  powerlaw_ann::diskann_disk_index_config_t build_config;
  build_config.data_path = data_path;
  build_config.index_path_prefix = index_prefix;
  build_config.search_dram_budget_gb = 0.01;
  build_config.build_dram_budget_gb = 0.1;
  build_config.max_degree = 12;
  build_config.build_list_size = 32;
  build_config.num_threads = 1;
  build_config.quantized_dimension = 1;
  ASSERT_NO_THROW(power_ann.build_diskann_index(build_config));

  powerlaw_ann::diskann_search_config_t search_config;
  search_config.index_path_prefix = index_prefix;
  search_config.query_path = query_path;
  search_config.ground_truth_path = ground_truth_path;
  search_config.result_path_prefix = result_prefix;
  search_config.node_visit_output_path = visit_path;
  search_config.top_k = 5;
  search_config.search_list_size = 64;
  search_config.beam_width = 4;
  search_config.num_threads = 2;

  const auto result = power_ann.search_diskann_index(search_config);
  ASSERT_EQ(result.num_queries, expected_ids.size());
  ASSERT_EQ(result.query_dimension, k_dimension);
  EXPECT_EQ(result.num_threads, 2U);
  ASSERT_EQ(result.ids.size(), expected_ids.size() * search_config.top_k);
  ASSERT_EQ(result.distances.size(), result.ids.size());
  ASSERT_EQ(result.query_stats.size(), expected_ids.size());
  EXPECT_GT(result.qps(), 0.0);
  ASSERT_TRUE(result.recall_percent.has_value());
  EXPECT_DOUBLE_EQ(*result.recall_percent, 100.0);
  EXPECT_GT(result.mean_latency_us(), 0.0);
  EXPECT_GT(result.p50_latency_us(), 0.0);
  EXPECT_GE(result.p95_latency_us(), result.p50_latency_us());
  EXPECT_GE(result.p99_latency_us(), result.p95_latency_us());
  EXPECT_GT(result.reads_per_query(), 0.0);
  EXPECT_GT(result.io_time_us_per_query(), 0.0);
  EXPECT_GT(result.cpu_time_us_per_query(), 0.0);
  EXPECT_GT(result.read_bytes_per_query(), 0.0);
  EXPECT_GT(result.four_kib_reads_per_query(), 0.0);
  EXPECT_EQ(result.cache_hits_per_query(), 0.0);
  EXPECT_GT(result.distance_computations_per_query(), 0.0);
  EXPECT_GT(result.index_size_bytes, 0U);
  EXPECT_EQ(result.cache_state, "warm");
  EXPECT_FALSE(result.backends.io.empty());
  EXPECT_FALSE(result.backends.math.empty());
  EXPECT_FALSE(result.backends.allocator.empty());
  EXPECT_FALSE(result.backends.simd.empty());
  EXPECT_EQ(result.topology_resident_bytes, 0U);
  EXPECT_EQ(result.topology_artifact_bytes, 0U);
  EXPECT_EQ(result.vector_artifact_bytes, 0U);

  for (size_t query = 0; query < expected_ids.size(); ++query) {
    const auto exact = exact_top_k(data, queries.data() + query * k_dimension, search_config.top_k);
    for (size_t rank = 0; rank < search_config.top_k; ++rank) {
      EXPECT_EQ(result.ids[query * search_config.top_k + rank], exact[rank].second);
      EXPECT_NEAR(result.distances[query * search_config.top_k + rank], exact[rank].first, 1e-4F);
    }
    EXPECT_EQ(result.ids[query * search_config.top_k], expected_ids[query]);
    for (size_t rank = 1; rank < search_config.top_k; ++rank) {
      EXPECT_LE(result.distances[query * search_config.top_k + rank - 1],
                result.distances[query * search_config.top_k + rank]);
    }
    EXPECT_GT(result.query_stats[query].n_ios, 0U);
  }

  const std::string output_base =
      result_prefix.string() + "_" + std::to_string(search_config.search_list_size);
  EXPECT_TRUE(std::filesystem::is_regular_file(output_base + "_idx_uint32.bin"));
  EXPECT_TRUE(std::filesystem::is_regular_file(output_base + "_dists_float.bin"));
  EXPECT_TRUE(std::filesystem::is_regular_file(visit_path));

  search_config.result_path_prefix.clear();
  search_config.node_visit_output_path.clear();
  search_config.num_nodes_to_cache = 20;
  const auto cached_result = power_ann.search_diskann_index(search_config);
  EXPECT_EQ(cached_result.ids, result.ids);
  EXPECT_EQ(cached_result.distances, result.distances);
  ASSERT_TRUE(cached_result.recall_percent.has_value());
  EXPECT_DOUBLE_EQ(*cached_result.recall_percent, 100.0);

  search_config.query_limit = 2;
  const auto limited_result = power_ann.search_diskann_index(search_config);
  EXPECT_EQ(limited_result.num_queries, 2U);
  EXPECT_EQ(limited_result.ids.size(), 2U * search_config.top_k);
  ASSERT_TRUE(limited_result.recall_percent.has_value());
  EXPECT_DOUBLE_EQ(*limited_result.recall_percent, 100.0);
  search_config.query_offset = 2;
  const auto offset_result = power_ann.search_diskann_index(search_config);
  EXPECT_EQ(offset_result.num_queries, 2U);
  EXPECT_EQ(offset_result.ids, std::vector<uint64_t>(result.ids.begin() + 2 * search_config.top_k,
                                                     result.ids.begin() + 4 * search_config.top_k));
  ASSERT_TRUE(offset_result.recall_percent.has_value());
  EXPECT_DOUBLE_EQ(*offset_result.recall_percent, 100.0);
  search_config.query_offset = 0;
  search_config.query_limit = 0;

  search_config.warmup_query_count = 2;
  search_config.warmup_query_offset = 2;
  const auto warmed_result = power_ann.search_diskann_index(search_config);
  EXPECT_EQ(warmed_result.ids, result.ids);
  EXPECT_EQ(warmed_result.distances, result.distances);
  EXPECT_EQ(warmed_result.query_stats.size(), result.query_stats.size());
  search_config.warmup_query_offset = 4;
  EXPECT_THROW(power_ann.search_diskann_index(search_config), powerlaw_ann::ann_exception_t);
  search_config.warmup_query_count = 0;
  search_config.warmup_query_offset = 0;

  const std::array<uint32_t, 2> curve_search_lists{16, 32};
  const auto curve_results =
      power_ann.search_diskann_index_curve(search_config, curve_search_lists);
  ASSERT_EQ(curve_results.size(), curve_search_lists.size());
  auto point_config = search_config;
  point_config.search_list_size = curve_search_lists.front();
  const auto point_result = power_ann.search_diskann_index(point_config);
  EXPECT_EQ(curve_results[0].ids, point_result.ids);
  EXPECT_EQ(curve_results[0].distances, point_result.distances);
  EXPECT_EQ(curve_results[1].num_queries, result.num_queries);
  EXPECT_GT(curve_results[1].distance_computations_per_query(),
            curve_results[0].distance_computations_per_query());
  const std::array<uint32_t, 2> duplicate_search_lists{16, 16};
  EXPECT_THROW(power_ann.search_diskann_index_curve(search_config, duplicate_search_lists),
               powerlaw_ann::ann_exception_t);

  const uint32_t fixed_search_list_size = search_config.search_list_size;
  search_config.search_list_size = 32;
  const std::array<uint32_t, 2> curve_thread_counts{1, 2};
  const auto thread_curve_results =
      power_ann.search_diskann_index_thread_curve(search_config, curve_thread_counts);
  ASSERT_EQ(thread_curve_results.size(), curve_thread_counts.size());
  EXPECT_EQ(thread_curve_results[0].num_threads, 1U);
  EXPECT_EQ(thread_curve_results[1].num_threads, 2U);
  EXPECT_EQ(thread_curve_results[0].ids, thread_curve_results[1].ids);
  EXPECT_EQ(thread_curve_results[0].distances, thread_curve_results[1].distances);
  const std::array<uint32_t, 2> duplicate_thread_counts{2, 2};
  EXPECT_THROW(power_ann.search_diskann_index_thread_curve(search_config, duplicate_thread_counts),
               powerlaw_ann::ann_exception_t);
  const std::array<uint32_t, 2> zero_thread_counts{1, 0};
  EXPECT_THROW(power_ann.search_diskann_index_thread_curve(search_config, zero_thread_counts),
               powerlaw_ann::ann_exception_t);
  search_config.search_list_size = fixed_search_list_size;

  const auto io_build = powerlaw_ann::io_optimized_index_t::build(index_prefix);
  powerlaw_ann::power_ann_config_t io_facade_config;
  io_facade_config.enable_powerann = true;
  io_facade_config.io_optimization.enable_topology_vector = true;
  powerlaw_ann::power_ann_t io_power_ann(io_facade_config);
  search_config.num_nodes_to_cache = 0;
  const auto io_result = io_power_ann.search_diskann_index(search_config);
  EXPECT_EQ(io_result.ids, result.ids);
  EXPECT_EQ(io_result.distances, result.distances);
  ASSERT_TRUE(io_result.recall_percent.has_value());
  EXPECT_DOUBLE_EQ(*io_result.recall_percent, *result.recall_percent);
  ASSERT_EQ(io_result.query_stats.size(), result.query_stats.size());
  for (size_t query = 0; query < result.query_stats.size(); ++query) {
    EXPECT_EQ(io_result.query_stats[query].n_hops, result.query_stats[query].n_hops);
    EXPECT_EQ(io_result.query_stats[query].n_cmps, result.query_stats[query].n_cmps);
  }
  EXPECT_GT(io_result.topology_resident_bytes, 0U);
  EXPECT_EQ(io_result.topology_artifact_bytes, io_build.topology_bytes);
  EXPECT_EQ(io_result.vector_artifact_bytes, io_build.vector_bytes);
  EXPECT_LE(io_result.reads_per_query(), result.reads_per_query());

  powerlaw_ann::power_ann_config_t disabled_io_config;
  disabled_io_config.io_optimization.enable_topology_vector = true;
  EXPECT_THROW(powerlaw_ann::power_ann_t(disabled_io_config).search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);

  auto invalid_io_config = io_facade_config;
  invalid_io_config.io_optimization.enable_paged_topology = true;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_io_config).search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
  invalid_io_config = io_facade_config;
  invalid_io_config.io_optimization.topology_cache_pages = 2;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_io_config).search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
  invalid_io_config = io_facade_config;
  invalid_io_config.io_optimization.memgraph_nodes = 16;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_io_config).search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
  invalid_io_config = io_facade_config;
  invalid_io_config.io_optimization.enable_dynamic_width = true;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_io_config).search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
  invalid_io_config = io_facade_config;
  invalid_io_config.io_optimization.enable_online_cell_adaptation = true;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_io_config).search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
  invalid_io_config = io_facade_config;
  invalid_io_config.io_optimization.enable_cell_adj_correction = true;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_io_config).search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);

  invalid_io_config = io_facade_config;
  invalid_io_config.community_polar_search.mode = powerlaw_ann::powerann_search_mode_t::HYBRID;
  invalid_io_config.community_polar_search.community_skip = true;
  invalid_io_config.community_polar_search.cell_expansion = true;
  invalid_io_config.community_polar_search.cell_terminal_convergence = true;
  invalid_io_config.community_polar_search.cell_hierarchical_routing = true;
  invalid_io_config.community_polar_search.cell_hierarchy_require_persisted = true;
  invalid_io_config.io_optimization.enable_cell_layout = true;
  invalid_io_config.io_optimization.enable_lru_cache = true;
  invalid_io_config.io_optimization.enable_cell_pq_traversal = true;
  invalid_io_config.io_optimization.enable_forced_cell_tree_search = true;
  EXPECT_THROW(powerlaw_ann::power_ann_t(invalid_io_config).search_diskann_index(search_config),
               powerlaw_ann::ann_exception_t);
}

} // namespace
