#ifndef COMMON_PARAMETERS
#define COMMON_PARAMETERS

#include "common/defaults.h"
#include "omp.h"

#include <cstdint>

namespace powerlaw_ann {

class index_write_parameters_t

{
public:
  const uint32_t search_list_size; // L
  const uint32_t max_degree;       // R
  const bool saturate_graph;
  const uint32_t max_occlusion_size; // C
  const float alpha;
  const uint32_t num_threads;
  const uint32_t filter_list_size; // Lf

  index_write_parameters_t(const uint32_t search_list_size, const uint32_t max_degree,
                           const bool saturate_graph, const uint32_t max_occlusion_size,
                           const float alpha, const uint32_t num_threads,
                           const uint32_t filter_list_size)
      : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
        max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads),
        filter_list_size(filter_list_size) {}

  friend class index_write_parameters_builder_t;
};

class index_search_params_t {
public:
  index_search_params_t(const uint32_t initial_search_list_size, const uint32_t num_search_threads)
      : initial_search_list_size(initial_search_list_size), num_search_threads(num_search_threads) {
  }
  const uint32_t initial_search_list_size; // search L
  const uint32_t num_search_threads;       // search threads
};

class index_write_parameters_builder_t {
  /**
   * Fluent builder pattern to keep track of the 7 non-default properties
   * and their order. The basic ctor was getting unwieldy.
   */
public:
  index_write_parameters_builder_t(const uint32_t search_list_size, // L
                                   const uint32_t max_degree        // R
                                   )
      : search_list_size_(search_list_size), max_degree_(max_degree) {}

  index_write_parameters_builder_t& with_max_occlusion_size(const uint32_t max_occlusion_size) {
    max_occlusion_size_ = max_occlusion_size;
    return *this;
  }

  index_write_parameters_builder_t& with_saturate_graph(const bool saturate_graph) {
    saturate_graph_ = saturate_graph;
    return *this;
  }

  index_write_parameters_builder_t& with_alpha(const float alpha) {
    alpha_ = alpha;
    return *this;
  }

  index_write_parameters_builder_t& with_num_threads(const uint32_t num_threads) {
    num_threads_ = num_threads == 0 ? omp_get_num_procs() : num_threads;
    return *this;
  }

  index_write_parameters_builder_t& with_filter_list_size(const uint32_t filter_list_size) {
    filter_list_size_ = filter_list_size == 0 ? search_list_size_ : filter_list_size;
    return *this;
  }

  index_write_parameters_t build() const {
    return index_write_parameters_t(search_list_size_, max_degree_, saturate_graph_,
                                    max_occlusion_size_, alpha_, num_threads_, filter_list_size_);
  }

  index_write_parameters_builder_t(const index_write_parameters_t& wp)
      : search_list_size_(wp.search_list_size), max_degree_(wp.max_degree),
        max_occlusion_size_(wp.max_occlusion_size), saturate_graph_(wp.saturate_graph),
        alpha_(wp.alpha), filter_list_size_(wp.filter_list_size) {}
  index_write_parameters_builder_t(const index_write_parameters_builder_t&) = delete;
  index_write_parameters_builder_t& operator=(const index_write_parameters_builder_t&) = delete;

private:
  uint32_t search_list_size_{};
  uint32_t max_degree_{};
  uint32_t max_occlusion_size_{defaults::MAX_OCCLUSION_SIZE};
  bool saturate_graph_{defaults::SATURATE_GRAPH};
  float alpha_{defaults::ALPHA};
  uint32_t num_threads_{defaults::NUM_THREADS};
  uint32_t filter_list_size_{defaults::FILTER_LIST_SIZE};
};

} // namespace powerlaw_ann

#endif // COMMON_PARAMETERS
