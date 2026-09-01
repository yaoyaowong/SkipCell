#include "index/adaptive_cell_runtime.h"

#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

namespace {

TEST(AdaptiveCellRuntimeTest, ProfilesPublishesOneSnapshotAndReclaimsAfterLastReader) {
  const std::vector<powerlaw_ann::io_cell_page_range_t> ranges = {
      {0, 4, 30, 4096},
      {4, 4, 31, 4096},
  };
  const std::vector<powerlaw_ann::adaptive_cell_merge_candidate_t> candidates = {{0, 1}};
  powerlaw_ann::adaptive_cell_runtime_config_t config;
  config.enabled = true;
  config.minimum_joint_observations = 100;
  config.minimum_coaccess_ratio = 0.90;
  powerlaw_ann::adaptive_cell_runtime_t runtime(ranges, candidates, config);

  std::vector<std::thread> workers;
  for (uint32_t worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&runtime]() {
      for (uint32_t observation = 0; observation < 25; ++observation) {
        runtime.observe(0, true, true, 8, 8, 20, 20);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  const auto profile = runtime.profile(0);
  EXPECT_EQ(profile.parent_observations, 200U);
  EXPECT_EQ(profile.joint_observations, 200U);
  EXPECT_DOUBLE_EQ(profile.coaccess_ratio, 1.0);
  EXPECT_TRUE(profile.eligible);

  auto old_reader = runtime.acquire();
  ASSERT_EQ(old_reader->generation(), 0U);
  auto replacement = std::make_shared<powerlaw_ann::adaptive_cell_view_t>();
  replacement->view_id = 100;
  replacement->generation = 1;
  replacement->capacity_class = 64;
  replacement->node_count = 61;
  replacement->physical_payload_ready = true;
  replacement->logical_cell_ids = {0, 1};
  replacement->page_runs = {{100, 8}};
  bool published = false;
  std::thread background_builder(
      [&runtime, &replacement, &published]() { published = runtime.try_publish(0, replacement); });
  background_builder.join();
  EXPECT_TRUE(published);

  const auto new_reader = runtime.acquire();
  EXPECT_EQ(new_reader->generation(), 1U);
  EXPECT_EQ(new_reader->cell(0).view_id, 100U);
  EXPECT_EQ(new_reader->cell(1).view_id, 100U);
  EXPECT_EQ(runtime.metrics().retired_snapshots, 0U);
  old_reader.reset();
  EXPECT_EQ(runtime.metrics().retired_snapshots, 1U);

  auto second_replacement = std::make_shared<powerlaw_ann::adaptive_cell_view_t>(*replacement);
  second_replacement->view_id = 101;
  second_replacement->generation = 2;
  EXPECT_FALSE(runtime.try_publish(0, second_replacement));
  EXPECT_EQ(runtime.metrics().successful_publishes, 1U);
}

TEST(AdaptiveCellRuntimeTest, DisabledModeDoesNotProfileOrPublish) {
  const std::vector<powerlaw_ann::io_cell_page_range_t> ranges = {
      {0, 1, 8, 4096},
      {1, 1, 8, 4096},
  };
  const std::vector<powerlaw_ann::adaptive_cell_merge_candidate_t> candidates = {{0, 1}};
  powerlaw_ann::adaptive_cell_runtime_config_t config;
  powerlaw_ann::adaptive_cell_runtime_t runtime(ranges, candidates, config);
  runtime.observe(0, true, true, 2, 1, 2, 2);
  EXPECT_EQ(runtime.profile(0).parent_observations, 0U);
  EXPECT_EQ(runtime.metrics().observations, 0U);

  auto replacement = std::make_shared<powerlaw_ann::adaptive_cell_view_t>();
  replacement->view_id = 2;
  replacement->generation = 1;
  replacement->capacity_class = 16;
  replacement->node_count = 16;
  replacement->physical_payload_ready = true;
  replacement->logical_cell_ids = {0, 1};
  replacement->page_runs = {{2, 2}};
  EXPECT_FALSE(runtime.try_publish(0, replacement));
  EXPECT_EQ(runtime.acquire()->generation(), 0U);
}

TEST(AdaptiveCellRuntimeTest, RejectsReplacementWithoutReadyContiguousPayload) {
  const std::vector<powerlaw_ann::io_cell_page_range_t> ranges = {
      {0, 1, 8, 4096},
      {1, 1, 8, 4096},
  };
  const std::vector<powerlaw_ann::adaptive_cell_merge_candidate_t> candidates = {{0, 1}};
  powerlaw_ann::adaptive_cell_runtime_config_t config;
  config.enabled = true;
  config.minimum_joint_observations = 1;
  config.minimum_coaccess_ratio = 1.0;
  powerlaw_ann::adaptive_cell_runtime_t runtime(ranges, candidates, config);
  runtime.observe(0, true, true, 2, 1, 4, 4);

  auto replacement = std::make_shared<powerlaw_ann::adaptive_cell_view_t>();
  replacement->view_id = 2;
  replacement->generation = 1;
  replacement->capacity_class = 16;
  replacement->node_count = 16;
  replacement->logical_cell_ids = {0, 1};
  replacement->page_runs = {{2, 1}};
  EXPECT_FALSE(runtime.try_publish(0, replacement));
  replacement->physical_payload_ready = true;
  replacement->page_runs.push_back({3, 1});
  EXPECT_FALSE(runtime.try_publish(0, replacement));
}

TEST(AdaptiveCellRuntimeTest, ChainedMergeAtomicallyReplacesEveryLogicalAlias) {
  const std::vector<powerlaw_ann::io_cell_page_range_t> ranges = {
      {0, 1, 8, 4096},
      {1, 1, 8, 4096},
      {2, 1, 8, 4096},
  };
  const std::vector<powerlaw_ann::adaptive_cell_merge_candidate_t> candidates = {
      {0, 1},
      {1, 2},
  };
  powerlaw_ann::adaptive_cell_runtime_config_t config;
  config.enabled = true;
  config.minimum_joint_observations = 1;
  config.minimum_coaccess_ratio = 1.0;
  powerlaw_ann::adaptive_cell_runtime_t runtime(ranges, candidates, config);
  runtime.observe(0, true, true, 2, 1, 4, 4);
  auto first = std::make_shared<powerlaw_ann::adaptive_cell_view_t>();
  first->view_id = 10;
  first->generation = 1;
  first->capacity_class = 16;
  first->node_count = 16;
  first->physical_payload_ready = true;
  first->logical_cell_ids = {0, 1};
  first->page_runs = {{10, 1}};
  ASSERT_TRUE(runtime.try_publish(0, first));

  runtime.observe(1, true, true, 2, 1, 4, 4);
  auto second = std::make_shared<powerlaw_ann::adaptive_cell_view_t>();
  second->view_id = 11;
  second->generation = 2;
  second->capacity_class = 32;
  second->node_count = 24;
  second->physical_payload_ready = true;
  second->logical_cell_ids = {0, 1, 2};
  second->page_runs = {{11, 1}};
  ASSERT_TRUE(runtime.try_publish(1, second));

  const auto snapshot = runtime.acquire();
  EXPECT_EQ(snapshot->generation(), 2U);
  EXPECT_EQ(snapshot->cell(0).view_id, 11U);
  EXPECT_EQ(snapshot->cell(1).view_id, 11U);
  EXPECT_EQ(snapshot->cell(2).view_id, 11U);
}

TEST(AdaptiveCellRuntimeTest, OneQueryObservationTriggersBackgroundContiguousMerge) {
  const std::vector<powerlaw_ann::io_cell_page_range_t> ranges = {
      {10, 1, 8, 4096},
      {11, 1, 8, 4096},
  };
  const std::vector<powerlaw_ann::adaptive_cell_merge_candidate_t> candidates = {{0, 1}};
  powerlaw_ann::adaptive_cell_runtime_config_t config;
  config.enabled = true;
  config.background_publish = true;
  config.minimum_joint_observations = 1;
  config.minimum_coaccess_ratio = 1.0;
  powerlaw_ann::adaptive_cell_runtime_t runtime(ranges, candidates, config);
  const std::array<uint32_t, 2> loaded = {0, 1};
  runtime.observe_query(loaded);
  runtime.drain_background();
  const auto snapshot = runtime.acquire();
  EXPECT_EQ(snapshot->generation(), 1U);
  EXPECT_EQ(snapshot->cell(0).view_id, snapshot->cell(1).view_id);
  EXPECT_EQ(snapshot->cell(0).page_runs, std::vector<powerlaw_ann::io_page_run_t>({{10, 2}}));
}

TEST(AdaptiveCellRuntimeTest, PublishBudgetStopsProfilingAfterTheRequestedOperation) {
  const std::vector<powerlaw_ann::io_cell_page_range_t> ranges = {
      {0, 1, 8, 4096},
      {1, 1, 8, 4096},
  };
  const std::vector<powerlaw_ann::adaptive_cell_merge_candidate_t> candidates = {{0, 1}};
  powerlaw_ann::adaptive_cell_runtime_config_t config;
  config.enabled = true;
  config.minimum_joint_observations = 1;
  config.minimum_coaccess_ratio = 1.0;
  config.maximum_successful_publishes = 1;
  powerlaw_ann::adaptive_cell_runtime_t runtime(ranges, candidates, config);
  runtime.observe(0, true, true, 2, 2, 4, 4);

  auto replacement = std::make_shared<powerlaw_ann::adaptive_cell_view_t>();
  replacement->view_id = 2;
  replacement->generation = 1;
  replacement->capacity_class = 16;
  replacement->node_count = 16;
  replacement->physical_payload_ready = true;
  replacement->logical_cell_ids = {0, 1};
  replacement->page_runs = {{2, 2}};
  ASSERT_TRUE(runtime.try_publish(0, replacement));
  ASSERT_FALSE(runtime.accepts_observations());
  ASSERT_EQ(runtime.final_snapshot(), runtime.acquire().get());

  const auto observations = runtime.metrics().observations;
  const std::array<uint32_t, 2> loaded = {0, 1};
  runtime.observe_query(loaded);
  EXPECT_EQ(runtime.metrics().observations, observations);

  auto extra = std::make_shared<powerlaw_ann::adaptive_cell_view_t>(*replacement);
  extra->view_id = 3;
  extra->generation = 2;
  EXPECT_FALSE(runtime.try_publish(0, extra, true));
  EXPECT_EQ(runtime.acquire()->generation(), 1U);
}

} // namespace
