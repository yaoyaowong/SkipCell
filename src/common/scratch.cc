#include "common/scratch.h"

#include "common/defaults.h"
#include "pq/pq_common.h"
#include "pq/pq_scratch.h"

#include <boost/dynamic_bitset.hpp>
#include <vector>

namespace powerlaw_ann {
//
// Functions to manage scratch space for in-memory index based search
//
template <typename T>
in_mem_query_scratch_t<T>::in_mem_query_scratch_t(uint32_t search_l, uint32_t indexing_l,
                                                  uint32_t r, uint32_t maxc, size_t dim,
                                                  size_t aligned_dim, size_t alignment_factor,
                                                  bool init_pq_scratch)
    : l_(0), r_(r), maxc_(maxc) {
  if (search_l == 0 || indexing_l == 0 || r == 0 || dim == 0) {
    std::stringstream ss;
    ss << "In in_mem_query_scratch_t, one of search_l = " << search_l
       << ", indexing_l = " << indexing_l << ", dim = " << dim << " or r = " << r << " is zero."
       << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1);
  }

  alloc_aligned(((void**) &this->aligned_query_), aligned_dim * sizeof(T),
                alignment_factor * sizeof(T));
  memset(this->aligned_query_, 0, aligned_dim * sizeof(T));

  if (init_pq_scratch)
    this->pq_scratch_ = new pq_scratch_t<T>(defaults::MAX_GRAPH_DEGREE, aligned_dim);
  else
    this->pq_scratch_ = nullptr;

  occlude_factor_.reserve(maxc);
  inserted_into_pool_bs_ = new boost::dynamic_bitset<>();
  id_scratch_.reserve((size_t) std::ceil(1.5 * defaults::GRAPH_SLACK_FACTOR * r_));
  dist_scratch_.reserve((size_t) std::ceil(1.5 * defaults::GRAPH_SLACK_FACTOR * r_));

  resize_for_new_l(std::max(search_l, indexing_l));
}

template <typename T>
void in_mem_query_scratch_t<T>::clear() {
  pool_.clear();
  best_l_nodes_.clear();
  occlude_factor_.clear();

  inserted_into_pool_rs_.clear();
  inserted_into_pool_bs_->reset();

  id_scratch_.clear();
  dist_scratch_.clear();

  expanded_nodes_set_.clear();
  expanded_nghrs_vec_.clear();
  occlude_list_output_.clear();
}

template <typename T>
void in_mem_query_scratch_t<T>::resize_for_new_l(uint32_t new_l) {
  if (new_l > l_) {
    l_ = new_l;
    pool_.reserve(3 * l_ + r_);
    best_l_nodes_.reserve(l_);

    inserted_into_pool_rs_.reserve(20 * l_);
  }
}

template <typename T>
in_mem_query_scratch_t<T>::~in_mem_query_scratch_t() {
  if (this->aligned_query_ != nullptr) {
    aligned_free(this->aligned_query_);
    this->aligned_query_ = nullptr;
  }

  delete this->pq_scratch_;
  delete inserted_into_pool_bs_;
}

//
// Functions to manage scratch space for SSD based search
//
template <typename T>
void ssd_query_scratch_t<T>::reset() {
  sector_idx = 0;
  visited.clear();
  retset.clear();
  full_retset.clear();
}

template <typename T>
ssd_query_scratch_t<T>::ssd_query_scratch_t(size_t aligned_dim, size_t visited_reserve,
                                            size_t candidate_capacity) {
  size_t coord_alloc_size = ROUND_UP(sizeof(T) * aligned_dim, 256);

  powerlaw_ann::alloc_aligned((void**) &coord_scratch, coord_alloc_size, 256);
  powerlaw_ann::alloc_aligned((void**) &sector_scratch,
                              defaults::MAX_N_SECTOR_READS * defaults::SECTOR_LEN,
                              defaults::SECTOR_LEN);
  powerlaw_ann::alloc_aligned((void**) &this->aligned_query_, aligned_dim * sizeof(T),
                              8 * sizeof(T));

  this->pq_scratch_ =
      new pq_scratch_t<T>(defaults::MAX_GRAPH_DEGREE, aligned_dim, candidate_capacity);

  memset(coord_scratch, 0, coord_alloc_size);
  memset(this->aligned_query_, 0, aligned_dim * sizeof(T));

  visited.reserve(visited_reserve);
  full_retset.reserve(visited_reserve);
}

template <typename T>
ssd_query_scratch_t<T>::~ssd_query_scratch_t() {
  powerlaw_ann::aligned_free((void*) coord_scratch);
  powerlaw_ann::aligned_free((void*) sector_scratch);
  powerlaw_ann::aligned_free((void*) this->aligned_query_);

  delete this->pq_scratch_;
}

template <typename T>
ssd_thread_data_t<T>::ssd_thread_data_t(size_t aligned_dim, size_t visited_reserve,
                                        size_t candidate_capacity)
    : scratch(aligned_dim, visited_reserve, candidate_capacity) {}

template <typename T>
void ssd_thread_data_t<T>::clear() {
  scratch.reset();
}

template <typename T>
pq_scratch_t<T>::pq_scratch_t(size_t graph_degree, size_t aligned_dim,
                              size_t candidate_capacity) {
  const size_t scratch_candidates =
      candidate_capacity == 0 ? graph_degree : std::max(graph_degree, candidate_capacity);
  powerlaw_ann::alloc_aligned((void**) &aligned_pq_coord_scratch,
                              scratch_candidates * (size_t) MAX_PQ_CHUNKS * sizeof(uint8_t),
                              256);
  powerlaw_ann::alloc_aligned((void**) &aligned_pqtable_dist_scratch,
                              256 * (size_t) MAX_PQ_CHUNKS * sizeof(float), 256);
  powerlaw_ann::alloc_aligned((void**) &aligned_dist_scratch,
                              scratch_candidates * sizeof(float),
                              256);
  powerlaw_ann::alloc_aligned((void**) &aligned_query_float, aligned_dim * sizeof(float),
                              8 * sizeof(float));
  powerlaw_ann::alloc_aligned((void**) &rotated_query, aligned_dim * sizeof(float),
                              8 * sizeof(float));

  memset(aligned_query_float, 0, aligned_dim * sizeof(float));
  memset(rotated_query, 0, aligned_dim * sizeof(float));
}

template <typename T>
pq_scratch_t<T>::~pq_scratch_t() {
  powerlaw_ann::aligned_free((void*) aligned_pq_coord_scratch);
  powerlaw_ann::aligned_free((void*) aligned_pqtable_dist_scratch);
  powerlaw_ann::aligned_free((void*) aligned_dist_scratch);
  powerlaw_ann::aligned_free((void*) aligned_query_float);
  powerlaw_ann::aligned_free((void*) rotated_query);
}

template <typename T>
void pq_scratch_t<T>::initialize(size_t dim, const T* query, const float norm) {
  for (size_t d = 0; d < dim; ++d) {
    if (norm != 1.0f)
      rotated_query[d] = aligned_query_float[d] = static_cast<float>(query[d]) / norm;
    else
      rotated_query[d] = aligned_query_float[d] = static_cast<float>(query[d]);
  }
}

template POWERLAWANN_DLLEXPORT class in_mem_query_scratch_t<int8_t>;
template POWERLAWANN_DLLEXPORT class in_mem_query_scratch_t<uint8_t>;
template POWERLAWANN_DLLEXPORT class in_mem_query_scratch_t<float>;

template POWERLAWANN_DLLEXPORT class ssd_query_scratch_t<int8_t>;
template POWERLAWANN_DLLEXPORT class ssd_query_scratch_t<uint8_t>;
template POWERLAWANN_DLLEXPORT class ssd_query_scratch_t<float>;

template POWERLAWANN_DLLEXPORT class pq_scratch_t<int8_t>;
template POWERLAWANN_DLLEXPORT class pq_scratch_t<uint8_t>;
template POWERLAWANN_DLLEXPORT class pq_scratch_t<float>;

template POWERLAWANN_DLLEXPORT class ssd_thread_data_t<int8_t>;
template POWERLAWANN_DLLEXPORT class ssd_thread_data_t<uint8_t>;
template POWERLAWANN_DLLEXPORT class ssd_thread_data_t<float>;

} // namespace powerlaw_ann
