#ifndef INDEX_VAMANA
#define INDEX_VAMANA

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief Vamana graph builder parameters.
 */
struct params_t {

  /**
   * @brief Max out-degree per node.
   */
  uint32_t R = 64;

  /**
   * @brief Build-time beam width.
   */
  uint32_t L = 100;

  /**
   * @brief Max candidates considered during pruning.
   */
  uint32_t C = 750;

  /**
   * @brief Robust-prune occlusion threshold.
   */
  float alpha = 1.2f;

  /**
   * @brief Build thread count, or 0 for all available OpenMP threads.
   */
  uint32_t num_threads = 0;
};

/**
 * @brief The built graph structure.
 */
struct graph_t {
  /**
   * @brief Number of graph nodes.
   */
  uint32_t n;

  /**
   * @brief Vector dimension.
   */
  uint32_t dim;

  /**
   * @brief Greedy-search entry point.
   */
  uint32_t entry_point;

  /**
   * @brief Out-neighbor list for each node.
   */
  std::vector<std::vector<uint32_t>> adj;
};

/**
 * @brief Build a Vamana graph from row-major vectors.
 * @param data Pointer to row-major vector data.
 * @param n Number of vectors.
 * @param dim Vector dimension.
 * @param params Builder parameters.
 * @return Built graph.
 */
graph_t build(const float* data, uint32_t n, uint32_t dim, const params_t& params = {});

/**
 * @brief Save graph adjacency to a binary file.
 * @param g Source graph.
 * @param path Output file path.
 */
void save_graph(const graph_t& g, const std::string& path);

/**
 * @brief Load graph adjacency from a binary file.
 * @param path Input file path.
 * @return Loaded graph.
 */
graph_t load_graph(const std::string& path);

/**
 * @brief Save row-major vectors to a binary file.
 * @param data Pointer to row-major vector data.
 * @param n Number of vectors.
 * @param dim Vector dimension.
 * @param path Output file path.
 */
void save_vectors(const float* data, uint32_t n, uint32_t dim, const std::string& path);

#endif // INDEX_VAMANA
