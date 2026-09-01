#include "index/louvain.h"

#include <gtest/gtest.h>
#include <vector>

using powerlaw_ann::third::louvain_edge_t;
using powerlaw_ann::third::louvain_options_t;
using powerlaw_ann::third::run_louvain_undirected;

TEST(louvain_third_test, detects_two_disconnected_communities) {
  const std::vector<louvain_edge_t> edges = {
      {1, 2, 1.0f},
      {3, 4, 1.0f},
  };

  louvain_options_t options;
  options.repeat = 1;
  options.num_threads = 1;

  const auto result = run_louvain_undirected(4, edges, options);

  ASSERT_EQ(result.membership.size(), 5u);
  EXPECT_EQ(result.membership[1], result.membership[2]);
  EXPECT_EQ(result.membership[3], result.membership[4]);
  EXPECT_NE(result.membership[1], result.membership[3]);
  EXPECT_GT(result.iterations, 0);
  EXPECT_GT(result.passes, 0);
  EXPECT_GT(result.modularity, 0.0);
}

TEST(louvain_third_test, rejects_edges_outside_the_vertex_range) {
  const std::vector<louvain_edge_t> edges = {{1, 3, 1.0f}};

  EXPECT_THROW(run_louvain_undirected(2, edges), std::out_of_range);
}

TEST(louvain_third_test, symmetric_csr_matches_edge_adapter_partition) {
  const std::vector<louvain_edge_t> edges = {
      {1, 2, 4.0F}, {2, 3, 4.0F}, {1, 3, 4.0F},
      {4, 5, 4.0F}, {5, 6, 4.0F}, {4, 6, 4.0F},
  };
  louvain_options_t options;
  options.repeat = 1;
  options.num_threads = 1;
  const auto edge_result = run_louvain_undirected(6, edges, options);
  const std::vector<size_t> offsets = {0, 2, 4, 6, 8, 10, 12};
  const std::vector<uint32_t> neighbors = {1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4};
  const std::vector<float> weights(neighbors.size(), 4.0F);
  const auto csr_result =
      powerlaw_ann::third::run_louvain_symmetric_csr(offsets, neighbors, weights, options);
  ASSERT_EQ(csr_result.membership.size(), 6U);
  EXPECT_EQ(csr_result.membership[0], csr_result.membership[1]);
  EXPECT_EQ(csr_result.membership[1], csr_result.membership[2]);
  EXPECT_EQ(csr_result.membership[3], csr_result.membership[4]);
  EXPECT_EQ(csr_result.membership[4], csr_result.membership[5]);
  EXPECT_NE(csr_result.membership[0], csr_result.membership[3]);
  EXPECT_EQ(edge_result.membership[1] == edge_result.membership[2],
            csr_result.membership[0] == csr_result.membership[1]);
}
