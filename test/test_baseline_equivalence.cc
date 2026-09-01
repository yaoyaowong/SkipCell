#include "common/ann_error.h"
#include "power_ann.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t k_num_points = 300;
constexpr uint32_t k_dimension = 8;
constexpr uint32_t k_top_k = 5;

class temp_dir_t {
public:
  temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_baseline_equivalence_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

std::vector<float> make_dataset() {
  std::vector<float> data(k_num_points * k_dimension);
  for (uint32_t point = 0; point < k_num_points; ++point) {
    for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
      data[point * k_dimension + dimension] =
          static_cast<float>(point) * (0.01F + static_cast<float>(dimension) * 0.002F) +
          std::sin(static_cast<float>(point + dimension * 17) * 0.07F);
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
                                                    const float* query) {
  std::vector<std::pair<float, uint32_t>> distances;
  distances.reserve(k_num_points);
  for (uint32_t point = 0; point < k_num_points; ++point) {
    float distance = 0.0F;
    for (uint32_t dimension = 0; dimension < k_dimension; ++dimension) {
      const float delta = query[dimension] - data[point * k_dimension + dimension];
      distance += delta * delta;
    }
    distances.emplace_back(distance, point);
  }
  std::sort(distances.begin(), distances.end(), [](const auto& left, const auto& right) {
    return left.first < right.first || (left.first == right.first && left.second < right.second);
  });
  distances.resize(k_top_k);
  return distances;
}

void write_truthset(const std::filesystem::path& path, const std::vector<float>& data,
                    const std::vector<float>& queries) {
  const uint32_t num_queries = static_cast<uint32_t>(queries.size() / k_dimension);
  std::vector<uint32_t> ids(num_queries * k_top_k);
  std::vector<float> distances(num_queries * k_top_k);
  for (uint32_t query = 0; query < num_queries; ++query) {
    const auto exact = exact_top_k(data, queries.data() + query * k_dimension);
    for (uint32_t rank = 0; rank < k_top_k; ++rank) {
      ids[query * k_top_k + rank] = exact[rank].second;
      distances[query * k_top_k + rank] = exact[rank].first;
    }
  }

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&num_queries), sizeof(num_queries));
  output.write(reinterpret_cast<const char*>(&k_top_k), sizeof(k_top_k));
  output.write(reinterpret_cast<const char*>(ids.data()),
               static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
  output.write(reinterpret_cast<const char*>(distances.data()),
               static_cast<std::streamsize>(distances.size() * sizeof(float)));
  ASSERT_TRUE(output.good());
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open artifact: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

powerlaw_ann::diskann_memory_index_config_t
make_memory_config(const std::filesystem::path& data_path,
                   const std::filesystem::path& index_prefix) {
  powerlaw_ann::diskann_memory_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_prefix;
  config.max_degree = 12;
  config.build_list_size = 32;
  config.num_threads = 1;
  return config;
}

powerlaw_ann::diskann_disk_index_config_t
make_disk_config(const std::filesystem::path& data_path,
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

void expect_counter_equivalence(const std::vector<powerlaw_ann::query_stats_t>& baseline,
                                const std::vector<powerlaw_ann::query_stats_t>& disabled) {
  ASSERT_EQ(baseline.size(), disabled.size());
  for (size_t query = 0; query < baseline.size(); ++query) {
    EXPECT_EQ(baseline[query].n_4k, disabled[query].n_4k);
    EXPECT_EQ(baseline[query].n_8k, disabled[query].n_8k);
    EXPECT_EQ(baseline[query].n_12k, disabled[query].n_12k);
    EXPECT_EQ(baseline[query].n_ios, disabled[query].n_ios);
    EXPECT_EQ(baseline[query].read_size, disabled[query].read_size);
    EXPECT_EQ(baseline[query].n_cmps_saved, disabled[query].n_cmps_saved);
    EXPECT_EQ(baseline[query].n_cmps, disabled[query].n_cmps);
    EXPECT_EQ(baseline[query].n_cache_hits, disabled[query].n_cache_hits);
    EXPECT_EQ(baseline[query].n_hops, disabled[query].n_hops);
    EXPECT_EQ(baseline[query].n_base_neighbors_scanned, disabled[query].n_base_neighbors_scanned);
  }
}

TEST(BaselineEquivalenceTest, DisableMatchesDefaultAndEnabledRcniPreservesMemoryArtifacts) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto baseline_prefix = temp_dir.path() / "baseline.index";
  const auto disabled_prefix = temp_dir.path() / "disabled.index";
  const auto enabled_prefix = temp_dir.path() / "enabled.index";
  const auto repeated_prefix = temp_dir.path() / "repeated.index";
  const auto data = make_dataset();
  write_bin(data_path, data.data(), k_num_points, k_dimension);

  powerlaw_ann::power_ann_t baseline;
  powerlaw_ann::power_ann_config_t facade_config;
  facade_config.enable_powerann = false;
  powerlaw_ann::power_ann_t disabled(facade_config);
  facade_config.enable_powerann = true;
  facade_config.hub_selection.mode = powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  facade_config.hub_selection.importance_mass = 0.8;
  powerlaw_ann::power_ann_t enabled(facade_config);
  ASSERT_FALSE(baseline.is_powerann_enabled());
  ASSERT_FALSE(disabled.is_powerann_enabled());
  ASSERT_TRUE(enabled.is_powerann_enabled());

  baseline.build_diskann_memory_index(make_memory_config(data_path, baseline_prefix));
  disabled.build_diskann_memory_index(make_memory_config(data_path, disabled_prefix));
  enabled.build_diskann_memory_index(make_memory_config(data_path, enabled_prefix));
  enabled.build_diskann_memory_index(make_memory_config(data_path, repeated_prefix));

  EXPECT_EQ(read_file(baseline_prefix), read_file(disabled_prefix));
  EXPECT_EQ(read_file(baseline_prefix.string() + ".data"),
            read_file(disabled_prefix.string() + ".data"));
  EXPECT_FALSE(std::filesystem::exists(disabled_prefix.string() + ".rcni.csv"));
  EXPECT_FALSE(std::filesystem::exists(disabled_prefix.string() + ".candidate_hubs.csv"));
  EXPECT_EQ(read_file(baseline_prefix), read_file(enabled_prefix));
  EXPECT_EQ(read_file(baseline_prefix.string() + ".data"),
            read_file(enabled_prefix.string() + ".data"));
  const auto rcni_output = enabled_prefix.string() + ".rcni.csv";
  const auto repeated_rcni_output = repeated_prefix.string() + ".rcni.csv";
  const auto candidate_hubs_output = enabled_prefix.string() + ".candidate_hubs.csv";
  const auto repeated_candidate_hubs_output = repeated_prefix.string() + ".candidate_hubs.csv";
  ASSERT_TRUE(std::filesystem::is_regular_file(rcni_output));
  ASSERT_TRUE(std::filesystem::is_regular_file(repeated_rcni_output));
  ASSERT_TRUE(std::filesystem::is_regular_file(candidate_hubs_output));
  ASSERT_TRUE(std::filesystem::is_regular_file(repeated_candidate_hubs_output));
  EXPECT_EQ(read_file(rcni_output), read_file(repeated_rcni_output));
  EXPECT_EQ(read_file(candidate_hubs_output), read_file(repeated_candidate_hubs_output));
  const auto rcni_bytes = read_file(rcni_output);
  const std::string rcni_text(rcni_bytes.begin(), rcni_bytes.end());
  EXPECT_TRUE(rcni_text.starts_with(
      "node_id,raw_importance,normalized_importance,importance_percentile,source_support,"
      "witness_event_count\n"));
  const auto candidate_hub_bytes = read_file(candidate_hubs_output);
  const std::string candidate_hub_text(candidate_hub_bytes.begin(), candidate_hub_bytes.end());
  EXPECT_TRUE(candidate_hub_text.starts_with(
      "selection_mode,selection_value,total_importance,selected_importance,"
      "selected_importance_mass,rank,node_id,raw_importance,normalized_importance,"
      "cumulative_importance_mass\n"));
  EXPECT_NE(candidate_hub_text.find("importance_mass,"), std::string::npos);
}

TEST(BaselineEquivalenceTest, ExplicitDisableMatchesDefaultDiskSearch) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto query_path = temp_dir.path() / "query.bin";
  const auto ground_truth_path = temp_dir.path() / "ground_truth.bin";
  const auto index_prefix = temp_dir.path() / "diskann";
  const auto data = make_dataset();
  write_bin(data_path, data.data(), k_num_points, k_dimension);

  const std::vector<uint32_t> query_ids = {0, 37, 91, 173, 299};
  std::vector<float> queries(query_ids.size() * k_dimension);
  for (size_t query = 0; query < query_ids.size(); ++query) {
    std::copy_n(data.data() + query_ids[query] * k_dimension, k_dimension,
                queries.data() + query * k_dimension);
  }
  write_bin(query_path, queries.data(), static_cast<uint32_t>(query_ids.size()), k_dimension);
  write_truthset(ground_truth_path, data, queries);

  powerlaw_ann::power_ann_config_t facade_config;
  facade_config.enable_powerann = false;
  powerlaw_ann::power_ann_t disabled(facade_config);
  disabled.build_diskann_index(make_disk_config(data_path, index_prefix));

  powerlaw_ann::diskann_search_config_t baseline_search;
  baseline_search.index_path_prefix = index_prefix;
  baseline_search.query_path = query_path;
  baseline_search.ground_truth_path = ground_truth_path;
  baseline_search.result_path_prefix = temp_dir.path() / "baseline_result";
  baseline_search.node_visit_output_path = temp_dir.path() / "baseline_visits.csv";
  baseline_search.top_k = k_top_k;
  baseline_search.search_list_size = 64;
  baseline_search.beam_width = 4;
  baseline_search.num_threads = 1;

  auto disabled_search = baseline_search;
  disabled_search.result_path_prefix = temp_dir.path() / "disabled_result";
  disabled_search.node_visit_output_path = temp_dir.path() / "disabled_visits.csv";

  powerlaw_ann::power_ann_t baseline;
  const auto baseline_result = baseline.search_diskann_index(baseline_search);
  const auto disabled_result = disabled.search_diskann_index(disabled_search);
  powerlaw_ann::power_ann_config_t enabled_config;
  enabled_config.enable_powerann = true;
  powerlaw_ann::power_ann_t enabled(enabled_config);
  auto enabled_search = baseline_search;
  enabled_search.result_path_prefix.clear();
  enabled_search.node_visit_output_path.clear();
  const auto enabled_result = enabled.search_diskann_index(enabled_search);

  EXPECT_EQ(baseline_result.ids, disabled_result.ids);
  EXPECT_EQ(baseline_result.distances, disabled_result.distances);
  EXPECT_EQ(baseline_result.recall_percent, disabled_result.recall_percent);
  EXPECT_EQ(baseline_result.index_size_bytes, disabled_result.index_size_bytes);
  EXPECT_EQ(baseline_result.cache_state, disabled_result.cache_state);
  EXPECT_EQ(baseline_result.backends.io, disabled_result.backends.io);
  EXPECT_EQ(baseline_result.backends.math, disabled_result.backends.math);
  EXPECT_EQ(baseline_result.backends.allocator, disabled_result.backends.allocator);
  EXPECT_EQ(baseline_result.backends.simd, disabled_result.backends.simd);
  expect_counter_equivalence(baseline_result.query_stats, disabled_result.query_stats);
  EXPECT_EQ(baseline_result.ids, enabled_result.ids);
  EXPECT_EQ(baseline_result.distances, enabled_result.distances);
  EXPECT_EQ(baseline_result.recall_percent, enabled_result.recall_percent);
  expect_counter_equivalence(baseline_result.query_stats, enabled_result.query_stats);

  EXPECT_EQ(read_file(baseline_search.node_visit_output_path),
            read_file(disabled_search.node_visit_output_path));
  const std::string baseline_result_base = baseline_search.result_path_prefix.string() + "_64";
  const std::string disabled_result_base = disabled_search.result_path_prefix.string() + "_64";
  EXPECT_EQ(read_file(baseline_result_base + "_idx_uint32.bin"),
            read_file(disabled_result_base + "_idx_uint32.bin"));
  EXPECT_EQ(read_file(baseline_result_base + "_dists_float.bin"),
            read_file(disabled_result_base + "_dists_float.bin"));
}

TEST(BaselineEquivalenceTest, EnabledSearchUsesBaselineAndInvalidBuildFailsBeforeIo) {
  temp_dir_t temp_dir;
  powerlaw_ann::power_ann_config_t facade_config;
  facade_config.enable_powerann = true;
  powerlaw_ann::power_ann_t enabled(facade_config);
  ASSERT_TRUE(enabled.is_powerann_enabled());

  auto memory_config =
      make_memory_config(temp_dir.path() / "missing.bin", temp_dir.path() / "memory.index");
  EXPECT_THROW(enabled.build_diskann_memory_index(memory_config), powerlaw_ann::ann_exception_t);
  EXPECT_FALSE(std::filesystem::exists(memory_config.index_path_prefix));
  EXPECT_FALSE(std::filesystem::exists(memory_config.index_path_prefix.string() + ".rcni.csv"));
  EXPECT_FALSE(
      std::filesystem::exists(memory_config.index_path_prefix.string() + ".candidate_hubs.csv"));

  auto disk_config = make_disk_config(temp_dir.path() / "missing.bin", temp_dir.path() / "diskann");
  EXPECT_THROW(enabled.build_diskann_index(disk_config), powerlaw_ann::ann_exception_t);
  EXPECT_FALSE(std::filesystem::exists(disk_config.index_path_prefix.string() + "_disk.index"));
  EXPECT_FALSE(std::filesystem::exists(disk_config.index_path_prefix.string() + ".rcni.csv"));
  EXPECT_FALSE(
      std::filesystem::exists(disk_config.index_path_prefix.string() + ".candidate_hubs.csv"));

  powerlaw_ann::diskann_search_config_t search_config;
  search_config.index_path_prefix = temp_dir.path() / "missing_index";
  search_config.query_path = temp_dir.path() / "missing_query.bin";
  search_config.result_path_prefix = temp_dir.path() / "result";
  try {
    static_cast<void>(enabled.search_diskann_index(search_config));
    FAIL() << "missing baseline query must fail";
  } catch (const powerlaw_ann::ann_exception_t& error) {
    EXPECT_EQ(std::string(error.what()),
              "DiskANN query file does not exist: " + search_config.query_path.string());
  }
  EXPECT_FALSE(std::filesystem::exists(search_config.result_path_prefix));
}

TEST(BaselineEquivalenceTest, EnabledModeRejectsInvalidHubSelectionBeforeBuildIo) {
  temp_dir_t temp_dir;
  powerlaw_ann::power_ann_config_t facade_config;
  facade_config.enable_powerann = true;
  facade_config.hub_selection.top_ratio = 1.01;
  powerlaw_ann::power_ann_t enabled(facade_config);
  const auto index_prefix = temp_dir.path() / "invalid_selection.index";
  const auto memory_config = make_memory_config(temp_dir.path() / "missing.bin", index_prefix);

  try {
    enabled.build_diskann_memory_index(memory_config);
    FAIL() << "invalid candidate-Hub selection must fail";
  } catch (const powerlaw_ann::ann_exception_t& error) {
    EXPECT_EQ(std::string(error.what()),
              "candidate-Hub selection value must be finite and within [0, 1]");
  }
  EXPECT_FALSE(std::filesystem::exists(index_prefix));
  EXPECT_FALSE(std::filesystem::exists(index_prefix.string() + ".rcni.csv"));
  EXPECT_FALSE(std::filesystem::exists(index_prefix.string() + ".candidate_hubs.csv"));
}

} // namespace
