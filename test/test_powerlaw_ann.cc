#include "index/delta_store.h"
#include "index/disk_index.h"
#include "index/hnsw.h"
#include "index/powerlaw_ann.h"
#include "index/vamana.h"

#include <filesystem>
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <unistd.h>

using namespace powerlaw_ann;

static std::vector<float> make_flat(int dim, int n, uint32_t seed_offset = 0) {
  std::vector<float> data(n * dim);
  std::mt19937 rng(seed_offset);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& value : data) {
    value = dist(rng);
  }
  return data;
}

static std::string tmp_ws() { return "/tmp/powerlaw_ann_test_" + std::to_string(::getpid()); }

TEST(hnsw_index_t_test, insert_and_search) {
  const int dim = 32;
  const int n = 200;
  const int k = 5;

  hnsw_index_t idx(dim, 512, 8, 100);
  const auto flat = make_flat(dim, n);

  for (int i = 0; i < n; ++i) {
    idx.insert(flat.data() + i * dim, static_cast<uint64_t>(i));
  }

  ASSERT_EQ(idx.size(), static_cast<size_t>(n));

  const auto res = idx.search(flat.data(), k);
  ASSERT_FALSE(res.empty());
  EXPECT_EQ(res[0].second, 0u);
  EXPECT_NEAR(res[0].first, 0.0f, 1e-4f);
}

TEST(hnsw_index_t_test, save_and_load) {
  const int dim = 16;
  const int n = 100;

  hnsw_index_t orig(dim, 256, 8, 50);
  const auto flat = make_flat(dim, n, 42);
  for (int i = 0; i < n; ++i) {
    orig.insert(flat.data() + i * dim, static_cast<uint64_t>(i));
  }

  std::ostringstream out_bin(std::ios::binary);
  orig.save(out_bin);

  hnsw_index_t loaded(dim, 256, 8, 50);
  std::istringstream in_bin(out_bin.str(), std::ios::binary);
  loaded.load(in_bin);

  ASSERT_EQ(loaded.size(), orig.size());

  const auto query = make_flat(dim, 1, 999);
  const auto r1 = orig.search(query.data(), 5);
  const auto r2 = loaded.search(query.data(), 5);

  ASSERT_EQ(r1.size(), r2.size());
  for (size_t i = 0; i < r1.size(); ++i) {
    EXPECT_EQ(r1[i].second, r2[i].second);
  }
}

TEST(delta_store_t_test, add_and_apply) {
  const int dim = 8;

  hnsw_index_t idx(dim, 256, 4, 50);
  delta_store_t ds(dim, 16);
  const auto flat = make_flat(dim, 5);

  for (int i = 0; i < 5; ++i) {
    ds.add(flat.data() + i * dim, static_cast<uint64_t>(i));
  }

  EXPECT_EQ(ds.size(), 5u);
  ds.apply_and_clear(idx);
  EXPECT_EQ(idx.size(), 5u);
  EXPECT_TRUE(ds.empty());
}

TEST(vamana_test, build_and_round_trip_graph) {
  const int dim = 4;
  const int n = 8;
  const auto flat = make_flat(dim, n, 7000);

  params_t params;
  params.R = 3;
  params.L = 5;
  params.C = 16;
  params.num_threads = 1;

  const auto graph =
      build(flat.data(), static_cast<uint32_t>(n), static_cast<uint32_t>(dim), params);

  EXPECT_EQ(graph.n, static_cast<uint32_t>(n));
  EXPECT_EQ(graph.dim, static_cast<uint32_t>(dim));
  EXPECT_LT(graph.entry_point, static_cast<uint32_t>(n));
  ASSERT_EQ(graph.adj.size(), static_cast<size_t>(n));
  for (uint32_t node = 0; node < graph.n; ++node) {
    EXPECT_LE(graph.adj[node].size(), static_cast<size_t>(params.R));
    for (uint32_t neighbor : graph.adj[node]) {
      EXPECT_LT(neighbor, graph.n);
      EXPECT_NE(neighbor, node);
    }
  }

  const std::string path = tmp_ws() + "_vamana_graph.bin";
  save_graph(graph, path);
  const auto loaded = load_graph(path);

  EXPECT_EQ(loaded.n, graph.n);
  EXPECT_EQ(loaded.dim, graph.dim);
  EXPECT_EQ(loaded.entry_point, graph.entry_point);
  EXPECT_EQ(loaded.adj, graph.adj);

  std::filesystem::remove(path);
}

TEST(powerlaw_ann_t_test, build_and_search) {
  powerlaw_ann_config_t cfg;
  cfg.dim = 16;
  cfg.m = 8;
  cfg.ef_construction = 50;
  cfg.ef_search = 20;
  cfg.max_elements = 512;
  cfg.workspace = tmp_ws();
  cfg.dataset_name = "build_test";

  powerlaw_ann_t bv(cfg);

  const int n = 150;
  const auto flat = make_flat(cfg.dim, n, 1000);
  std::vector<uint64_t> labels(n);
  for (int i = 0; i < n; ++i) {
    labels[i] = static_cast<uint64_t>(i);
  }

  bv.build(flat.data(), labels.data(), static_cast<size_t>(n));
  EXPECT_EQ(bv.size(), static_cast<size_t>(n));

  const auto res = bv.search(flat.data(), 1);
  ASSERT_FALSE(res.empty());
  EXPECT_EQ(res[0].second, 0u);
  EXPECT_NEAR(res[0].first, 0.0f, 1e-4f);

  std::filesystem::remove_all(cfg.workspace);
}

TEST(powerlaw_ann_t_test, save_and_load) {
  powerlaw_ann_config_t cfg;
  cfg.dim = 16;
  cfg.m = 8;
  cfg.ef_construction = 50;
  cfg.ef_search = 20;
  cfg.max_elements = 512;
  cfg.workspace = tmp_ws();
  cfg.dataset_name = "save_load_test";

  const int n = 80;
  const auto flat = make_flat(cfg.dim, n, 2000);
  std::vector<uint64_t> labels(n);
  for (int i = 0; i < n; ++i) {
    labels[i] = static_cast<uint64_t>(i + 2000);
  }

  {
    powerlaw_ann_t bv(cfg);
    bv.build(flat.data(), labels.data(), static_cast<size_t>(n));
    bv.save();
  }

  {
    powerlaw_ann_t bv2(cfg);
    bv2.load();
    EXPECT_EQ(bv2.size(), static_cast<size_t>(n));

    const auto res = bv2.search(flat.data(), 1);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res[0].second, labels[0]);
    EXPECT_NEAR(res[0].first, 0.0f, 1e-4f);
  }

  std::filesystem::remove_all(cfg.workspace);
}

TEST(powerlaw_ann_t_test, insert_via_buffer) {
  powerlaw_ann_config_t cfg;
  cfg.dim = 8;
  cfg.m = 4;
  cfg.ef_construction = 30;
  cfg.ef_search = 10;
  cfg.max_elements = 256;
  cfg.delta_buffer = 5;
  cfg.workspace = tmp_ws();
  cfg.dataset_name = "insert_test";

  powerlaw_ann_t bv(cfg);
  const auto flat = make_flat(cfg.dim, 12, 3000);

  for (int i = 0; i < 12; ++i) {
    bv.insert(flat.data() + i * cfg.dim, static_cast<uint64_t>(i + 3000));
  }

  bv.consolidate();
  EXPECT_GE(bv.size(), 12u);

  std::filesystem::remove_all(cfg.workspace);
}
