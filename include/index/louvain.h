#ifndef INDEX_LOUVAIN
#define INDEX_LOUVAIN

#ifndef OPENMP
#error "index/louvain.h requires OpenMP support."
#endif

#include "louvain.hxx"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace powerlaw_ann::third {

struct louvain_edge_t {
  uint32_t src = 0;
  uint32_t dst = 0;
  float weight = 1.0f;
};

struct louvain_options_t {
  int repeat = 1;
  int num_threads = 1;
};

struct louvain_result_t {
  std::vector<uint32_t> membership;
  int iterations = 0;
  int passes = 0;
  double modularity = 0.0;
};

inline louvain_result_t run_louvain_undirected(uint32_t vertex_count,
                                               const std::vector<louvain_edge_t>& edges,
                                               louvain_options_t options = {}) {
  if (vertex_count == 0) {
    throw std::invalid_argument("louvain graph must contain at least one vertex");
  }

  if (options.repeat <= 0) {
    throw std::invalid_argument("louvain repeat count must be positive");
  }

  if (options.num_threads <= 0) {
    throw std::invalid_argument("louvain thread count must be positive");
  }

  omp_set_num_threads(options.num_threads);

  DiGraph<uint32_t, None, float> graph;
  graph.reserve(static_cast<size_t>(vertex_count) + 1, edges.empty() ? 0 : 2);
  for (uint32_t id = 1; id <= vertex_count; ++id) {
    graph.addVertex(id);
  }

  for (const auto& edge : edges) {
    if (edge.src == 0 || edge.dst == 0 || edge.src > vertex_count || edge.dst > vertex_count) {
      throw std::out_of_range("louvain edge endpoint is outside the 1-based vertex range");
    }
    graph.addEdge(edge.src, edge.dst, edge.weight);
    if (edge.src != edge.dst) {
      graph.addEdge(edge.dst, edge.src, edge.weight);
    }
  }
  graph.update();

  const auto raw = louvainStaticOmp(graph, LouvainOptions(options.repeat));
  const double total_weight = edgeWeightOmp(graph) / 2;
  auto community_of = [&](auto vertex) { return raw.membership[vertex]; };

  return {raw.membership, raw.iterations, raw.passes,
          modularityBy(graph, community_of, total_weight, 1.0)};
}

/** @brief Runs Louvain directly on a zero-based symmetric CSR without duplicating edge objects. */
inline louvain_result_t run_louvain_symmetric_csr(
    std::vector<size_t> offsets, std::vector<uint32_t> neighbors,
    std::vector<float> weights, louvain_options_t options = {}) {
  if (offsets.size() < 2 || offsets.front() != 0 || offsets.back() != neighbors.size() ||
      neighbors.size() != weights.size()) {
    throw std::invalid_argument("louvain CSR shape is invalid");
  }
  const size_t vertex_count = offsets.size() - 1;
  if (vertex_count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("louvain CSR has too many vertices");
  }
  if (options.repeat <= 0 || options.num_threads <= 0) {
    throw std::invalid_argument("louvain options must be positive");
  }
  DiGraphCsr<uint32_t, None, float> graph(vertex_count, neighbors.size());
  graph.offsets = std::move(offsets);
  graph.edgeKeys = std::move(neighbors);
  graph.edgeValues = std::move(weights);
  for (uint32_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (graph.offsets[vertex] > graph.offsets[vertex + 1]) {
      throw std::invalid_argument("louvain CSR offsets are not monotonic");
    }
    graph.degrees[vertex] =
        static_cast<uint32_t>(graph.offsets[vertex + 1] - graph.offsets[vertex]);
    for (size_t edge = graph.offsets[vertex]; edge < graph.offsets[vertex + 1]; ++edge) {
      if (graph.edgeKeys[edge] >= vertex_count || !std::isfinite(graph.edgeValues[edge]) ||
          graph.edgeValues[edge] < 0.0F) {
        throw std::invalid_argument("louvain CSR edge is invalid");
      }
    }
  }
  omp_set_num_threads(options.num_threads);
  const auto raw = louvainStaticOmp(graph, LouvainOptions(options.repeat));
  const double total_weight = edgeWeightOmp(graph) / 2;
  auto community_of = [&](auto vertex) { return raw.membership[vertex]; };
  return {raw.membership, raw.iterations, raw.passes,
          modularityBy(graph, community_of, total_weight, 1.0)};
}

} // namespace powerlaw_ann::third

#endif // INDEX_LOUVAIN
