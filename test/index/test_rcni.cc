#if defined(__APPLE__)
#define _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "common/parameters.h"
#include "index/index_build_params.h"
#include "index/index_config.h"
#include "index/index_factory.h"
#include "index/rcni.h"
#include "index/vamana_index.h"

#if defined(__APPLE__)
#pragma clang diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using powerlaw_ann::rcni_distance_semantics_t;
using powerlaw_ann::rcni_event_input_t;
using powerlaw_ann::rcni_neighbor_distance_t;
using powerlaw_ann::rcni_node_importance_t;
using powerlaw_ann::rcni_prune_event_t;

constexpr uint32_t k_rcni_test_points = 48;
constexpr uint32_t k_rcni_test_dimensions = 8;
constexpr uint32_t k_rcni_test_degree = 8;
constexpr uint32_t k_rcni_test_build_list_size = 24;

class rcni_temp_dir_t {
public:
  rcni_temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_rcni_observation_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~rcni_temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

struct rcni_source_observation_t {
  uint32_t source_id = 0;
  std::vector<rcni_prune_event_t> events;
  std::vector<uint32_t> selected_neighbors;
};

class collecting_rcni_observer_t final : public powerlaw_ann::rcni_prune_observer_t {
public:
  void observe_source(uint32_t source_id, std::span<const rcni_prune_event_t> events,
                      std::span<const uint32_t> selected_neighbors) override {
    std::lock_guard<std::mutex> guard(mutex_);
    observations_.push_back(rcni_source_observation_t{
        source_id, std::vector<rcni_prune_event_t>(events.begin(), events.end()),
        std::vector<uint32_t>(selected_neighbors.begin(), selected_neighbors.end())});
  }

  std::vector<rcni_source_observation_t> observations() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return observations_;
  }

private:
  mutable std::mutex mutex_;
  std::vector<rcni_source_observation_t> observations_;
};

std::vector<float> make_rcni_test_data() {
  std::vector<float> data(k_rcni_test_points * k_rcni_test_dimensions, 0.0F);
  for (uint32_t point = 0; point < k_rcni_test_points; ++point) {
    data[point * k_rcni_test_dimensions] = static_cast<float>(point);
    data[point * k_rcni_test_dimensions + 1] = static_cast<float>((point * 7U) % 11U) * 0.01F;
    for (uint32_t dimension = 2; dimension < k_rcni_test_dimensions; ++dimension) {
      data[point * k_rcni_test_dimensions + dimension] =
          static_cast<float>((point * (dimension + 3U)) % 17U) * 0.001F;
    }
  }
  return data;
}

void write_rcni_test_data(const std::filesystem::path& path, const std::vector<float>& data) {
  std::ofstream output(path, std::ios::binary);
  const int32_t points = static_cast<int32_t>(k_rcni_test_points);
  const int32_t dimensions = static_cast<int32_t>(k_rcni_test_dimensions);
  output.write(reinterpret_cast<const char*>(&points), sizeof(points));
  output.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
  output.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!output) {
    throw std::runtime_error("failed to write RCNI test data");
  }
}

std::vector<char> read_rcni_artifact(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open RCNI test artifact: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

float squared_l2(const std::vector<float>& data, uint32_t left, uint32_t right) {
  float distance = 0.0F;
  for (uint32_t dimension = 0; dimension < k_rcni_test_dimensions; ++dimension) {
    const float delta = data[left * k_rcni_test_dimensions + dimension] -
                        data[right * k_rcni_test_dimensions + dimension];
    distance += delta * delta;
  }
  return distance;
}

void build_rcni_test_index(const std::filesystem::path& data_path,
                           const std::filesystem::path& index_prefix,
                           powerlaw_ann::rcni_prune_observer_t* observer) {
  auto write_params = powerlaw_ann::index_write_parameters_builder_t(k_rcni_test_build_list_size,
                                                                     k_rcni_test_degree)
                          .with_filter_list_size(0)
                          .with_alpha(1.2F)
                          .with_saturate_graph(false)
                          .with_num_threads(1)
                          .build();
  auto filter_params = powerlaw_ann::index_filter_params_builder_t()
                           .with_universal_label("")
                           .with_label_file("")
                           .with_save_path_prefix(index_prefix.string())
                           .build();
  auto index_config =
      powerlaw_ann::index_config_builder_t()
          .with_metric(powerlaw_ann::metric_t::L2)
          .with_dimension(k_rcni_test_dimensions)
          .with_max_points(k_rcni_test_points)
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
  auto index = factory.create_instance();
  auto* vamana =
      dynamic_cast<powerlaw_ann::vamana_index_t<float, uint32_t, uint32_t>*>(index.get());
  if (vamana == nullptr) {
    throw std::runtime_error("factory did not create the expected Vamana index type");
  }
  vamana->set_rcni_prune_observer(observer);
  index->build(data_path.string(), k_rcni_test_points, filter_params);
  index->save(index_prefix.c_str());
}

rcni_event_input_t
make_event(float source_distance, float witness_distance,
           std::span<const rcni_neighbor_distance_t> selected_neighbors = {},
           rcni_distance_semantics_t semantics = rcni_distance_semantics_t::TRUE_L2) {
  rcni_event_input_t input;
  input.witness_id = 7;
  input.source_victim_distance = source_distance;
  input.witness_victim_distance = witness_distance;
  input.selected_neighbors = selected_neighbors;
  input.alpha = 1.2F;
  input.distance_semantics = semantics;
  return input;
}

rcni_prune_event_t make_prune_event(uint32_t source_id, uint32_t witness_id, uint32_t victim_id,
                                    float source_distance, float witness_distance,
                                    std::optional<rcni_neighbor_distance_t> alternative) {
  rcni_prune_event_t event;
  event.source_id = source_id;
  event.witness_id = witness_id;
  event.victim_id = victim_id;
  event.source_victim_distance = source_distance;
  event.witness_victim_distance = witness_distance;
  event.alternative = alternative;
  return event;
}

rcni_node_importance_t make_importance(uint32_t node_id, double normalized_importance) {
  rcni_node_importance_t importance;
  importance.node_id = node_id;
  importance.raw_importance = normalized_importance * 10.0;
  importance.normalized_importance = normalized_importance;
  return importance;
}

std::vector<rcni_node_importance_t> make_candidate_hub_importance() {
  return {
      make_importance(2, 0.5), make_importance(4, 0.0), make_importance(3, 0.9),
      make_importance(0, 0.1), make_importance(1, 0.5),
  };
}

void expect_importance_equal(const std::vector<rcni_node_importance_t>& left,
                             const std::vector<rcni_node_importance_t>& right) {
  ASSERT_EQ(left.size(), right.size());
  for (size_t node = 0; node < left.size(); ++node) {
    EXPECT_EQ(left[node].node_id, right[node].node_id);
    EXPECT_DOUBLE_EQ(left[node].raw_importance, right[node].raw_importance);
    EXPECT_DOUBLE_EQ(left[node].normalized_importance, right[node].normalized_importance);
    EXPECT_DOUBLE_EQ(left[node].importance_percentile, right[node].importance_percentile);
    EXPECT_EQ(left[node].source_support, right[node].source_support);
    EXPECT_EQ(left[node].witness_event_count, right[node].witness_event_count);
  }
}

TEST(RcniTest, CertifiedProgressIsZeroWhenContractionConditionFails) {
  const auto credit = powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 9.0F));

  EXPECT_FLOAT_EQ(credit.certified_progress, 0.0F);
  EXPECT_FLOAT_EQ(credit.event_credit, 0.0F);
}

TEST(RcniTest, UniqueStrongWitnessReceivesExpectedCredit) {
  const auto credit = powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 2.0F));

  EXPECT_NEAR(credit.witness_ratio, 0.2F, 1.0e-6F);
  EXPECT_FLOAT_EQ(credit.counterfactual_ratio, 1.0F);
  EXPECT_NEAR(credit.certified_progress, 0.76F, 1.0e-6F);
  EXPECT_NEAR(credit.counterfactual_irreplaceability, 0.8F, 1.0e-6F);
  EXPECT_NEAR(credit.event_credit, 0.608F, 1.0e-6F);
  EXPECT_FALSE(credit.alternative_id.has_value());
}

TEST(RcniTest, EquivalentWitnessesReceiveNoIrreplaceabilityCredit) {
  const std::array<rcni_neighbor_distance_t, 2> neighbors{{{7, 2.0F}, {9, 2.0F}}};
  const auto credit = powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 2.0F, neighbors));

  ASSERT_TRUE(credit.alternative_id.has_value());
  EXPECT_EQ(*credit.alternative_id, 9U);
  EXPECT_FLOAT_EQ(credit.counterfactual_irreplaceability, 0.0F);
  EXPECT_FLOAT_EQ(credit.event_credit, 0.0F);
}

TEST(RcniTest, CounterfactualSelectionUsesNodeIdForDistanceTies) {
  const std::array<rcni_neighbor_distance_t, 4> first_order{
      {{7, 2.0F}, {11, 4.0F}, {3, 4.0F}, {5, 6.0F}}};
  const std::array<rcni_neighbor_distance_t, 4> second_order{
      {{5, 6.0F}, {3, 4.0F}, {7, 2.0F}, {11, 4.0F}}};

  const auto first = powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 2.0F, first_order));
  const auto second =
      powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 2.0F, second_order));

  ASSERT_TRUE(first.alternative_id.has_value());
  ASSERT_TRUE(second.alternative_id.has_value());
  EXPECT_EQ(*first.alternative_id, 3U);
  EXPECT_EQ(first.alternative_id, second.alternative_id);
  EXPECT_FLOAT_EQ(first.counterfactual_ratio, second.counterfactual_ratio);
  EXPECT_FLOAT_EQ(first.event_credit, second.event_credit);
}

TEST(RcniTest, CausalWitnessUsesStrongestFactorAndNodeIdTieBreak) {
  powerlaw_ann::rcni_causal_witness_t state;

  powerlaw_ann::update_rcni_causal_witness(state, 11, 4.0F, 2.0F);
  powerlaw_ann::update_rcni_causal_witness(state, 13, 3.0F, 3.0F);
  powerlaw_ann::update_rcni_causal_witness(state, 7, 3.0F, 3.0F);
  powerlaw_ann::update_rcni_causal_witness(state, 5, 5.0F, 1.0F);

  ASSERT_TRUE(state.witness_id.has_value());
  EXPECT_EQ(*state.witness_id, 7U);
  EXPECT_FLOAT_EQ(state.witness_victim_distance, 3.0F);
  EXPECT_FLOAT_EQ(state.occlusion_factor, 3.0F);
}

TEST(RcniTest, CounterfactualFallsBackToSourceAndCapsDistantAlternative) {
  const std::array<rcni_neighbor_distance_t, 2> neighbors{{{7, 2.0F}, {9, 20.0F}}};
  const auto credit = powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 2.0F, neighbors));

  ASSERT_TRUE(credit.alternative_id.has_value());
  EXPECT_EQ(*credit.alternative_id, 9U);
  EXPECT_FLOAT_EQ(credit.counterfactual_ratio, 1.0F);
  EXPECT_NEAR(credit.counterfactual_irreplaceability, 0.8F, 1.0e-6F);
}

TEST(RcniTest, BetterAlternativeIsClampedToWitnessRatio) {
  const std::array<rcni_neighbor_distance_t, 2> neighbors{{{7, 2.0F}, {9, 1.0F}}};
  const auto credit = powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 2.0F, neighbors));

  EXPECT_NEAR(credit.counterfactual_ratio, credit.witness_ratio, 1.0e-6F);
  EXPECT_FLOAT_EQ(credit.counterfactual_irreplaceability, 0.0F);
  EXPECT_FLOAT_EQ(credit.event_credit, 0.0F);
}

TEST(RcniTest, SquaredAndTrueL2ProduceConsistentCredit) {
  const std::array<rcni_neighbor_distance_t, 2> true_neighbors{{{7, 2.0F}, {9, 4.0F}}};
  const std::array<rcni_neighbor_distance_t, 2> squared_neighbors{{{7, 4.0F}, {9, 16.0F}}};

  const auto true_credit = powerlaw_ann::compute_rcni_event_credit(
      make_event(10.0F, 2.0F, true_neighbors, rcni_distance_semantics_t::TRUE_L2));
  const auto squared_credit = powerlaw_ann::compute_rcni_event_credit(
      make_event(100.0F, 4.0F, squared_neighbors, rcni_distance_semantics_t::SQUARED_L2));

  EXPECT_NEAR(true_credit.witness_ratio, squared_credit.witness_ratio, 1.0e-6F);
  EXPECT_NEAR(true_credit.counterfactual_ratio, squared_credit.counterfactual_ratio, 1.0e-6F);
  EXPECT_NEAR(true_credit.certified_progress, squared_credit.certified_progress, 1.0e-6F);
  EXPECT_NEAR(true_credit.counterfactual_irreplaceability,
              squared_credit.counterfactual_irreplaceability, 1.0e-6F);
  EXPECT_NEAR(true_credit.event_credit, squared_credit.event_credit, 1.0e-6F);
}

TEST(RcniTest, ZeroSourceDistanceProducesZeroCredit) {
  const std::array<rcni_neighbor_distance_t, 1> neighbors{{{9, 0.0F}}};
  const auto credit = powerlaw_ann::compute_rcni_event_credit(make_event(0.0F, 0.0F, neighbors));

  EXPECT_FLOAT_EQ(credit.witness_ratio, 0.0F);
  EXPECT_FLOAT_EQ(credit.certified_progress, 0.0F);
  EXPECT_FLOAT_EQ(credit.counterfactual_irreplaceability, 0.0F);
  EXPECT_FLOAT_EQ(credit.event_credit, 0.0F);
}

TEST(RcniTest, DuplicateWitnessIsStrongOnlyWhenNoEquivalentAlternativeExists) {
  const auto unique_credit = powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 0.0F));
  const std::array<rcni_neighbor_distance_t, 1> neighbors{{{9, 0.0F}}};
  const auto equivalent_credit =
      powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 0.0F, neighbors));

  EXPECT_FLOAT_EQ(unique_credit.certified_progress, 1.0F);
  EXPECT_FLOAT_EQ(unique_credit.counterfactual_irreplaceability, 1.0F);
  EXPECT_FLOAT_EQ(unique_credit.event_credit, 1.0F);
  EXPECT_FLOAT_EQ(equivalent_credit.counterfactual_irreplaceability, 0.0F);
  EXPECT_FLOAT_EQ(equivalent_credit.event_credit, 0.0F);
}

TEST(RcniTest, VerySmallDistancesRemainConsistentAcrossDistanceSemantics) {
  const auto true_credit = powerlaw_ann::compute_rcni_event_credit(
      make_event(1.0e-10F, 2.0e-11F, {}, rcni_distance_semantics_t::TRUE_L2));
  const auto squared_credit = powerlaw_ann::compute_rcni_event_credit(
      make_event(1.0e-20F, 4.0e-22F, {}, rcni_distance_semantics_t::SQUARED_L2));

  EXPECT_NEAR(true_credit.event_credit, squared_credit.event_credit, 1.0e-5F);
  EXPECT_GT(true_credit.event_credit, 0.0F);
}

TEST(RcniTest, SourceSaturationIsBoundedAndOrderIndependent) {
  const std::array<float, 3> first_order{{0.25F, 0.5F, 0.1F}};
  const std::array<float, 3> second_order{{0.1F, 0.25F, 0.5F}};

  const float first = powerlaw_ann::compute_rcni_source_saturation(first_order);
  const float second = powerlaw_ann::compute_rcni_source_saturation(second_order);

  EXPECT_FLOAT_EQ(first, second);
  EXPECT_NEAR(first, 0.6625F, 1.0e-6F);
  EXPECT_GE(first, 0.0F);
  EXPECT_LE(first, 1.0F);
}

TEST(RcniTest, SourceSaturationHandlesEmptyAndCertainEvents) {
  const std::array<float, 0> empty{};
  const std::array<float, 3> certain{{0.2F, 1.0F, 0.4F}};

  EXPECT_FLOAT_EQ(powerlaw_ann::compute_rcni_source_saturation(empty), 0.0F);
  EXPECT_FLOAT_EQ(powerlaw_ann::compute_rcni_source_saturation(certain), 1.0F);
}

TEST(RcniTest, RejectsInvalidEventInputs) {
  auto input = make_event(10.0F, 2.0F);
  input.source_victim_distance = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(powerlaw_ann::compute_rcni_event_credit(input), std::invalid_argument);

  input = make_event(10.0F, std::numeric_limits<float>::infinity());
  EXPECT_THROW(powerlaw_ann::compute_rcni_event_credit(input), std::invalid_argument);

  input = make_event(-1.0F, 0.0F);
  EXPECT_THROW(powerlaw_ann::compute_rcni_event_credit(input), std::invalid_argument);

  input = make_event(10.0F, 2.0F);
  input.alpha = 0.9F;
  EXPECT_THROW(powerlaw_ann::compute_rcni_event_credit(input), std::invalid_argument);

  input.alpha = 1.2F;
  input.epsilon = 0.0F;
  EXPECT_THROW(powerlaw_ann::compute_rcni_event_credit(input), std::invalid_argument);
}

TEST(RcniTest, RejectsInvalidAlternativeAndSaturationInputs) {
  const std::array<rcni_neighbor_distance_t, 1> invalid_neighbors{
      {{9, std::numeric_limits<float>::infinity()}}};
  EXPECT_THROW(powerlaw_ann::compute_rcni_event_credit(make_event(10.0F, 2.0F, invalid_neighbors)),
               std::invalid_argument);

  const std::array<float, 1> nan_credit{{std::numeric_limits<float>::quiet_NaN()}};
  const std::array<float, 1> negative_credit{{-0.1F}};
  const std::array<float, 1> oversized_credit{{1.1F}};
  EXPECT_THROW(powerlaw_ann::compute_rcni_source_saturation(nan_credit), std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::compute_rcni_source_saturation(negative_credit),
               std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::compute_rcni_source_saturation(oversized_credit),
               std::invalid_argument);
}

TEST(CandidateHubSelectionTest, TopRatioUsesCeilingAndNodeIdTieBreak) {
  powerlaw_ann::candidate_hub_selection_config_t config;
  config.mode = powerlaw_ann::candidate_hub_selection_mode_t::TOP_RATIO;
  config.top_ratio = 0.5;

  const auto result = powerlaw_ann::select_candidate_hubs(make_candidate_hub_importance(), config);

  ASSERT_EQ(result.candidate_hubs.size(), 3U);
  EXPECT_EQ(result.candidate_hubs[0].node_id, 3U);
  EXPECT_EQ(result.candidate_hubs[1].node_id, 1U);
  EXPECT_EQ(result.candidate_hubs[2].node_id, 2U);
  EXPECT_DOUBLE_EQ(result.total_importance, 2.0);
  EXPECT_DOUBLE_EQ(result.selected_importance, 1.9);
  EXPECT_DOUBLE_EQ(result.selected_importance_mass, 0.95);
  EXPECT_DOUBLE_EQ(result.candidate_hubs.back().cumulative_importance_mass, 0.95);
}

TEST(CandidateHubSelectionTest, ImportanceMassUsesSmallestDeterministicPrefix) {
  powerlaw_ann::candidate_hub_selection_config_t config;
  config.mode = powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  config.importance_mass = 0.7;
  const auto importance = make_candidate_hub_importance();
  auto reversed_importance = importance;
  std::reverse(reversed_importance.begin(), reversed_importance.end());

  const auto first = powerlaw_ann::select_candidate_hubs(importance, config);
  const auto second = powerlaw_ann::select_candidate_hubs(reversed_importance, config);

  ASSERT_EQ(first.candidate_hubs.size(), 2U);
  ASSERT_EQ(second.candidate_hubs.size(), first.candidate_hubs.size());
  EXPECT_EQ(first.candidate_hubs[0].node_id, 3U);
  EXPECT_EQ(first.candidate_hubs[1].node_id, 1U);
  EXPECT_DOUBLE_EQ(first.selected_importance, 1.4);
  EXPECT_DOUBLE_EQ(first.selected_importance_mass, 0.7);
  for (size_t rank = 0; rank < first.candidate_hubs.size(); ++rank) {
    EXPECT_EQ(first.candidate_hubs[rank].node_id, second.candidate_hubs[rank].node_id);
    EXPECT_DOUBLE_EQ(first.candidate_hubs[rank].cumulative_importance_mass,
                     second.candidate_hubs[rank].cumulative_importance_mass);
  }
}

TEST(CandidateHubSelectionTest, HandlesEmptyZeroAndFullBoundaries) {
  powerlaw_ann::candidate_hub_selection_config_t config;
  EXPECT_TRUE(powerlaw_ann::select_candidate_hubs({}, config).candidate_hubs.empty());
  EXPECT_TRUE(powerlaw_ann::select_candidate_hubs(make_candidate_hub_importance(), config)
                  .candidate_hubs.empty());

  const std::array<rcni_node_importance_t, 3> zero_importance{
      make_importance(0, 0.0), make_importance(1, 0.0), make_importance(2, 0.0)};
  config.top_ratio = 1.0;
  const auto all_zero = powerlaw_ann::select_candidate_hubs(zero_importance, config);
  EXPECT_TRUE(all_zero.candidate_hubs.empty());
  EXPECT_DOUBLE_EQ(all_zero.total_importance, 0.0);

  config.top_ratio = 0.01;
  const auto minimum_positive =
      powerlaw_ann::select_candidate_hubs(make_candidate_hub_importance(), config);
  ASSERT_EQ(minimum_positive.candidate_hubs.size(), 1U);
  EXPECT_EQ(minimum_positive.candidate_hubs.front().node_id, 3U);

  config.top_ratio = 1.0;
  const auto full = powerlaw_ann::select_candidate_hubs(make_candidate_hub_importance(), config);
  ASSERT_EQ(full.candidate_hubs.size(), 5U);
  EXPECT_DOUBLE_EQ(full.selected_importance_mass, 1.0);

  config.mode = powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  config.importance_mass = 0.8;
  EXPECT_TRUE(powerlaw_ann::select_candidate_hubs(zero_importance, config).candidate_hubs.empty());
  config.importance_mass = 1.0;
  const auto full_mass =
      powerlaw_ann::select_candidate_hubs(make_candidate_hub_importance(), config);
  ASSERT_EQ(full_mass.candidate_hubs.size(), 4U);
  EXPECT_DOUBLE_EQ(full_mass.selected_importance_mass, 1.0);

  config.importance_mass = 0.0;
  EXPECT_TRUE(powerlaw_ann::select_candidate_hubs(make_candidate_hub_importance(), config)
                  .candidate_hubs.empty());
}

TEST(CandidateHubSelectionTest, RejectsInvalidConfigurationAndImportance) {
  powerlaw_ann::candidate_hub_selection_config_t config;
  config.top_ratio = 1.01;
  EXPECT_THROW(powerlaw_ann::select_candidate_hubs({}, config), std::invalid_argument);
  config.top_ratio = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(powerlaw_ann::select_candidate_hubs({}, config), std::invalid_argument);
  config.mode = powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  config.importance_mass = -0.01;
  EXPECT_THROW(powerlaw_ann::select_candidate_hubs({}, config), std::invalid_argument);
  config.mode = static_cast<powerlaw_ann::candidate_hub_selection_mode_t>(99);
  EXPECT_THROW(powerlaw_ann::select_candidate_hubs({}, config), std::invalid_argument);

  config = {};
  config.top_ratio = 0.5;
  auto invalid_importance = make_candidate_hub_importance();
  invalid_importance[0].normalized_importance = std::numeric_limits<double>::infinity();
  EXPECT_THROW(powerlaw_ann::select_candidate_hubs(invalid_importance, config),
               std::invalid_argument);

  invalid_importance = make_candidate_hub_importance();
  invalid_importance[0].node_id = invalid_importance[1].node_id;
  EXPECT_THROW(powerlaw_ann::select_candidate_hubs(invalid_importance, config),
               std::invalid_argument);
}

TEST(CandidateHubSelectionTest, CsvOutputIsByteStableAndReportsMass) {
  rcni_temp_dir_t temp_dir;
  const auto first_path = temp_dir.path() / "first.candidate_hubs.csv";
  const auto second_path = temp_dir.path() / "second.candidate_hubs.csv";
  powerlaw_ann::candidate_hub_selection_config_t config;
  config.mode = powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  config.importance_mass = 0.7;
  const auto result = powerlaw_ann::select_candidate_hubs(make_candidate_hub_importance(), config);

  powerlaw_ann::write_candidate_hubs_csv(first_path, result);
  powerlaw_ann::write_candidate_hubs_csv(second_path, result);

  const auto first = read_rcni_artifact(first_path);
  EXPECT_EQ(first, read_rcni_artifact(second_path));
  const std::string text(first.begin(), first.end());
  EXPECT_TRUE(
      text.starts_with("selection_mode,selection_value,total_importance,selected_importance,"
                       "selected_importance_mass,rank,node_id,raw_importance,normalized_importance,"
                       "cumulative_importance_mass\n"));
  EXPECT_NE(text.find("importance_mass,"), std::string::npos);
}

TEST(RcniAggregationTest, ConcurrentCallbackOrderProducesIdenticalImportance) {
  constexpr size_t node_count = 8;
  std::array<std::vector<rcni_prune_event_t>, 4> source_events;
  source_events[0] = {make_prune_event(0, 2, 5, 100.0F, 4.0F, rcni_neighbor_distance_t{3, 16.0F}),
                      make_prune_event(0, 2, 6, 100.0F, 9.0F, rcni_neighbor_distance_t{4, 25.0F})};
  source_events[1] = {make_prune_event(1, 2, 7, 100.0F, 4.0F, rcni_neighbor_distance_t{3, 16.0F})};
  source_events[2] = {make_prune_event(2, 4, 7, 100.0F, 1.0F, std::nullopt)};

  powerlaw_ann::rcni_aggregator_t serial;
  serial.prepare(node_count);
  for (uint32_t source = 0; source < source_events.size(); ++source) {
    serial.observe_source(source, source_events[source], {});
  }

  powerlaw_ann::rcni_aggregator_t concurrent;
  concurrent.prepare(node_count);
  std::vector<std::thread> workers;
  workers.reserve(source_events.size());
  for (uint32_t source = 0; source < source_events.size(); ++source) {
    const uint32_t reversed_source = static_cast<uint32_t>(source_events.size() - source - 1);
    workers.emplace_back([&, reversed_source]() {
      concurrent.observe_source(reversed_source, source_events[reversed_source], {});
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }

  const auto serial_importance = serial.compute_importance();
  const auto concurrent_importance = concurrent.compute_importance();
  expect_importance_equal(serial_importance, concurrent_importance);

  const std::array<rcni_neighbor_distance_t, 1> first_alternative{{{3, 16.0F}}};
  const std::array<rcni_neighbor_distance_t, 1> second_alternative{{{4, 25.0F}}};
  const float first_credit =
      powerlaw_ann::compute_rcni_event_credit(
          make_event(100.0F, 4.0F, first_alternative, rcni_distance_semantics_t::SQUARED_L2))
          .event_credit;
  const float second_credit =
      powerlaw_ann::compute_rcni_event_credit(
          make_event(100.0F, 9.0F, second_alternative, rcni_distance_semantics_t::SQUARED_L2))
          .event_credit;
  const std::array<float, 2> source_zero_credits{{first_credit, second_credit}};
  const double expected_witness_two =
      static_cast<double>(powerlaw_ann::compute_rcni_source_saturation(source_zero_credits)) +
      static_cast<double>(first_credit);

  ASSERT_EQ(serial_importance.size(), node_count);
  EXPECT_DOUBLE_EQ(serial_importance[2].raw_importance, expected_witness_two);
  EXPECT_DOUBLE_EQ(serial_importance[2].normalized_importance,
                   expected_witness_two /
                       (expected_witness_two + powerlaw_ann::k_rcni_default_epsilon));
  EXPECT_GT(serial_importance[2].raw_importance, serial_importance[4].raw_importance);
  EXPECT_EQ(serial_importance[2].source_support, 2U);
  EXPECT_EQ(serial_importance[2].witness_event_count, 3U);
  EXPECT_EQ(serial_importance[4].source_support, 1U);
  EXPECT_EQ(serial_importance[4].witness_event_count, 1U);
  EXPECT_GT(serial_importance[2].importance_percentile, serial_importance[4].importance_percentile);
}

TEST(RcniAggregationTest, EmptyObservationsProduceStableZeroOutput) {
  powerlaw_ann::rcni_aggregator_t aggregator;
  aggregator.prepare(3);
  for (uint32_t source = 0; source < 3; ++source) {
    aggregator.observe_source(source, {}, {});
  }

  const auto importance = aggregator.compute_importance();
  ASSERT_EQ(importance.size(), 3U);
  for (uint32_t node = 0; node < importance.size(); ++node) {
    EXPECT_EQ(importance[node].node_id, node);
    EXPECT_DOUBLE_EQ(importance[node].raw_importance, 0.0);
    EXPECT_DOUBLE_EQ(importance[node].normalized_importance, 0.0);
    EXPECT_DOUBLE_EQ(importance[node].importance_percentile, 0.0);
    EXPECT_EQ(importance[node].source_support, 0U);
    EXPECT_EQ(importance[node].witness_event_count, 0U);
  }
}

TEST(RcniAggregationTest, CsvOutputIsByteStable) {
  rcni_temp_dir_t temp_dir;
  const auto first_path = temp_dir.path() / "first.rcni.csv";
  const auto second_path = temp_dir.path() / "second.rcni.csv";

  powerlaw_ann::rcni_aggregator_t aggregator;
  aggregator.prepare(3);
  const std::array<rcni_prune_event_t, 1> events{
      {make_prune_event(0, 1, 2, 100.0F, 4.0F, rcni_neighbor_distance_t{2, 16.0F})}};
  aggregator.observe_source(0, events, {});
  aggregator.observe_source(1, {}, {});
  aggregator.observe_source(2, {}, {});
  const auto importance = aggregator.compute_importance();

  powerlaw_ann::write_rcni_importance_csv(first_path, importance);
  powerlaw_ann::write_rcni_importance_csv(second_path, importance);
  const auto first = read_rcni_artifact(first_path);
  EXPECT_EQ(first, read_rcni_artifact(second_path));
  const std::string text(first.begin(), first.end());
  EXPECT_TRUE(text.starts_with(
      "node_id,raw_importance,normalized_importance,importance_percentile,source_support,"
      "witness_event_count\n"));
}

TEST(RcniObservationTest, PrimaryObservationPreservesGraphBytesAndExcludesAuxiliaryPaths) {
  rcni_temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.fbin";
  const auto baseline_prefix = temp_dir.path() / "baseline.index";
  const auto observed_prefix = temp_dir.path() / "observed.index";
  const auto data = make_rcni_test_data();
  write_rcni_test_data(data_path, data);

  collecting_rcni_observer_t observer;
  build_rcni_test_index(data_path, baseline_prefix, nullptr);
  build_rcni_test_index(data_path, observed_prefix, &observer);

  EXPECT_EQ(read_rcni_artifact(baseline_prefix), read_rcni_artifact(observed_prefix));
  EXPECT_EQ(read_rcni_artifact(baseline_prefix.string() + ".data"),
            read_rcni_artifact(observed_prefix.string() + ".data"));

  const auto observations = observer.observations();
  ASSERT_EQ(observations.size(), k_rcni_test_points);
  std::vector<uint32_t> source_counts(k_rcni_test_points, 0);
  size_t event_count = 0;
  for (const auto& observation : observations) {
    ASSERT_LT(observation.source_id, k_rcni_test_points);
    ++source_counts[observation.source_id];
    EXPECT_FALSE(observation.selected_neighbors.empty());

    for (const auto& event : observation.events) {
      ++event_count;
      EXPECT_EQ(event.source_id, observation.source_id);
      ASSERT_LT(event.witness_id, k_rcni_test_points);
      ASSERT_LT(event.victim_id, k_rcni_test_points);
      EXPECT_NE(event.witness_id, event.victim_id);
      EXPECT_NE(std::find(observation.selected_neighbors.begin(),
                          observation.selected_neighbors.end(), event.witness_id),
                observation.selected_neighbors.end());
      EXPECT_EQ(std::find(observation.selected_neighbors.begin(),
                          observation.selected_neighbors.end(), event.victim_id),
                observation.selected_neighbors.end());

      const float expected_source_distance = squared_l2(data, event.source_id, event.victim_id);
      const float expected_witness_distance = squared_l2(data, event.witness_id, event.victim_id);
      EXPECT_NEAR(event.source_victim_distance, expected_source_distance, 1.0e-4F);
      EXPECT_NEAR(event.witness_victim_distance, expected_witness_distance, 1.0e-4F);
      if (event.witness_victim_distance > 0.0F) {
        EXPECT_GT(event.source_victim_distance / event.witness_victim_distance, 1.2F);
      }

      std::optional<rcni_neighbor_distance_t> expected_alternative;
      for (const uint32_t neighbor_id : observation.selected_neighbors) {
        if (neighbor_id == event.witness_id) {
          continue;
        }
        const float distance = squared_l2(data, event.victim_id, neighbor_id);
        if (!expected_alternative.has_value() || distance < expected_alternative->victim_distance ||
            (distance == expected_alternative->victim_distance &&
             neighbor_id < expected_alternative->node_id)) {
          expected_alternative = rcni_neighbor_distance_t{neighbor_id, distance};
        }
      }
      ASSERT_EQ(event.alternative.has_value(), expected_alternative.has_value());
      if (expected_alternative.has_value()) {
        EXPECT_EQ(event.alternative->node_id, expected_alternative->node_id);
        EXPECT_NEAR(event.alternative->victim_distance, expected_alternative->victim_distance,
                    1.0e-4F);
      }
    }
  }

  for (const uint32_t source_count : source_counts) {
    EXPECT_EQ(source_count, 1U);
  }
  EXPECT_GT(event_count, 0U);
}

} // namespace
