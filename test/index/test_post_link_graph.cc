#include "index/diskann_index.h"
#include "index/index_config.h"
#include "index/index_factory.h"
#include "index/post_link_graph.h"
#include "index/vamana_index.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
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
            ("powerlawann_post_link_graph_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

void write_dataset(const std::filesystem::path& path) {
  constexpr int32_t k_point_count = 32;
  constexpr int32_t k_dimension = 4;
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&k_point_count), sizeof(k_point_count));
  output.write(reinterpret_cast<const char*>(&k_dimension), sizeof(k_dimension));
  for (int32_t point = 0; point < k_point_count; ++point) {
    const float vector[k_dimension] = {
        static_cast<float>(point),
        static_cast<float>((point * 7) % 13),
        static_cast<float>((point * point) % 17),
        static_cast<float>((point * 11) % 19),
    };
    output.write(reinterpret_cast<const char*>(vector), sizeof(vector));
  }
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

powerlaw_ann::diskann_memory_index_config_t
make_config(const std::filesystem::path& data_path,
            const std::filesystem::path& index_path_prefix) {
  powerlaw_ann::diskann_memory_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_path_prefix;
  config.max_degree = 8;
  config.build_list_size = 16;
  config.alpha = 1.2F;
  config.num_threads = 1;
  return config;
}

std::unique_ptr<powerlaw_ann::abstract_index_t> make_vamana_index(powerlaw_ann::metric_t metric) {
  auto write_params = powerlaw_ann::index_write_parameters_builder_t(16, 8)
                          .with_filter_list_size(0)
                          .with_alpha(1.2F)
                          .with_saturate_graph(false)
                          .with_num_threads(1)
                          .build();
  auto index_config =
      powerlaw_ann::index_config_builder_t()
          .with_metric(metric)
          .with_dimension(4)
          .with_max_points(32)
          .with_data_load_store_strategy(powerlaw_ann::data_store_strategy_t::MEMORY)
          .with_graph_load_store_strategy(powerlaw_ann::graph_store_strategy_t::MEMORY)
          .with_data_type("float")
          .with_label_type("uint32")
          .with_tag_type("uint32")
          .is_dynamic_index(false)
          .with_index_write_params(write_params)
          .is_enable_tags(false)
          .is_use_opq(false)
          .is_pq_dist_build(false)
          .with_num_pq_chunks(0)
          .build();
  powerlaw_ann::index_factory_t factory(index_config);
  return factory.create_instance();
}

powerlaw_ann::vamana_index_t<float, uint32_t, uint32_t>&
get_vamana_index(powerlaw_ann::abstract_index_t& index) {
  auto* vamana = dynamic_cast<powerlaw_ann::vamana_index_t<float, uint32_t, uint32_t>*>(&index);
  if (vamana == nullptr) {
    throw std::runtime_error("factory did not create the expected Vamana index type");
  }
  return *vamana;
}

powerlaw_ann::index_filter_params_t
make_filter_config(const std::filesystem::path& index_path_prefix) {
  return powerlaw_ann::index_filter_params_builder_t()
      .with_universal_label("")
      .with_label_file("")
      .with_save_path_prefix(index_path_prefix.string())
      .build();
}

class recording_observer_t final : public powerlaw_ann::post_link_graph_observer_t {
public:
  void observe(const powerlaw_ann::post_link_graph_view_t& graph) override {
    ++callback_count;
    point_count = graph.point_count();
    dimension = graph.dimension();
    entry_point_id = graph.entry_point_id();

    const auto neighbors = graph.neighbors(1);
    node_one_neighbors.assign(neighbors.begin(), neighbors.end());
    node_one_vector.resize(dimension);
    graph.copy_vector(1, node_one_vector);
    node_zero_to_one_squared_distance = graph.squared_distance(0, 1);

    EXPECT_THROW(static_cast<void>(graph.neighbors(static_cast<uint32_t>(point_count))),
                 std::out_of_range);
    EXPECT_THROW(graph.copy_vector(static_cast<uint32_t>(point_count), node_one_vector),
                 std::invalid_argument);
    std::vector<float> wrong_dimension(static_cast<size_t>(dimension) + 1);
    EXPECT_THROW(graph.copy_vector(0, wrong_dimension), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(graph.squared_distance(0, static_cast<uint32_t>(point_count))),
                 std::out_of_range);
  }

  uint32_t callback_count = 0;
  uint64_t point_count = 0;
  uint32_t dimension = 0;
  uint32_t entry_point_id = 0;
  std::vector<uint32_t> node_one_neighbors;
  std::vector<float> node_one_vector;
  float node_zero_to_one_squared_distance = 0.0F;
};

TEST(PostLinkGraphTest, ObservesReadOnlyBaseOnceWithoutChangingArtifacts) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto baseline_prefix = temp_dir.path() / "baseline.index";
  const auto observed_prefix = temp_dir.path() / "observed.index";
  write_dataset(data_path);

  powerlaw_ann::diskann_index_t baseline_index;
  baseline_index.build_memory_index(make_config(data_path, baseline_prefix));

  recording_observer_t observer;
  powerlaw_ann::diskann_index_t observed_index;
  observed_index.build_memory_index(make_config(data_path, observed_prefix), nullptr, &observer);

  EXPECT_EQ(observer.callback_count, 1U);
  EXPECT_EQ(observer.point_count, 32U);
  EXPECT_EQ(observer.dimension, 4U);
  EXPECT_LT(observer.entry_point_id, observer.point_count);
  ASSERT_EQ(observer.node_one_vector.size(), 4U);
  EXPECT_EQ(observer.node_one_vector, (std::vector<float>{1.0F, 7.0F, 1.0F, 11.0F}));
  EXPECT_FLOAT_EQ(observer.node_zero_to_one_squared_distance, 172.0F);
  EXPECT_FALSE(observer.node_one_neighbors.empty());
  for (const uint32_t neighbor_id : observer.node_one_neighbors) {
    EXPECT_LT(neighbor_id, observer.point_count);
  }

  EXPECT_EQ(read_file(baseline_prefix), read_file(observed_prefix));
  EXPECT_EQ(read_file(baseline_prefix.string() + ".data"),
            read_file(observed_prefix.string() + ".data"));
}

TEST(PostLinkGraphTest, ObserverIsConsumedByOneBuild) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  write_dataset(data_path);

  auto index = make_vamana_index(powerlaw_ann::metric_t::L2);
  auto& vamana = get_vamana_index(*index);
  recording_observer_t observer;
  vamana.set_post_link_graph_observer(&observer);

  auto first_filter = make_filter_config(temp_dir.path() / "first.index");
  index->build(data_path.string(), 32, first_filter);
  ASSERT_EQ(observer.callback_count, 1U);

  auto second_filter = make_filter_config(temp_dir.path() / "second.index");
  index->build(data_path.string(), 32, second_filter);
  EXPECT_EQ(observer.callback_count, 1U);
}

TEST(PostLinkGraphTest, RejectsNonL2IndexesBeforeBuild) {
  recording_observer_t observer;
  for (const auto metric :
       {powerlaw_ann::metric_t::COSINE, powerlaw_ann::metric_t::INNER_PRODUCT}) {
    auto index = make_vamana_index(metric);
    auto& vamana = get_vamana_index(*index);
    EXPECT_THROW(vamana.set_post_link_graph_observer(&observer), std::invalid_argument);
  }
}

} // namespace
