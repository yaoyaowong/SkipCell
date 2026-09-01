#ifndef PQ_QUANTIZED_DISTANCE
#define PQ_QUANTIZED_DISTANCE

#include "common/abstract_scratch.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace powerlaw_ann {
template <typename data_t>
class pq_scratch_t;

template <typename data_t>
class quantized_distance_t {
public:
  quantized_distance_t() = default;
  quantized_distance_t(const quantized_distance_t&) = delete;
  quantized_distance_t& operator=(const quantized_distance_t&) = delete;
  virtual ~quantized_distance_t() = default;

  virtual bool is_opq() const = 0;
  virtual std::string get_quantized_vectors_filename(const std::string& prefix) const = 0;
  virtual std::string get_pivot_data_filename(const std::string& prefix) const = 0;
  virtual std::string get_rotation_matrix_suffix(const std::string& pq_pivots_filename) const = 0;

  // Loading the PQ centroid table need not be part of the abstract class.
  // However, we want to indicate that this function will change once we have a
  // file reader hierarchy, so leave it here as-is.
#ifdef EXEC_ENV_OLS
  virtual void load_pivot_data(memory_mapped_files_t& files, const std::string& pq_table_file,
                               size_t num_chunks) = 0;
#else
  virtual void load_pivot_data(const std::string& pq_table_file, size_t num_chunks) = 0;
#endif

  // Number of chunks in the PQ table. Depends on the compression level used.
  // Has to be < ndim
  virtual uint32_t get_num_chunks() const = 0;

  // Preprocess the query by computing chunk distances from the query vector to
  // various centroids. Since we don't want this class to do scratch management,
  // we will take a pq_scratch_t object which can come either from vamana_index_t class or
  // PQFlashIndex class.
  virtual void preprocess_query(const data_t* query_vec, uint32_t query_dim,
                                pq_scratch_t<data_t>& pq_scratch) = 0;

  // Workhorse
  // This function must be called after preprocess_query
  virtual void preprocessed_distance(pq_scratch_t<data_t>& pq_scratch, const uint32_t id_count,
                                     float* dists_out) = 0;

  // Same as above, but convenience function for index.cpp.
  virtual void preprocessed_distance(pq_scratch_t<data_t>& pq_scratch, const uint32_t n_ids,
                                     std::vector<float>& dists_out) = 0;

  // Currently this function is required for DiskPQ. However, it too can be subsumed
  // under preprocessed_distance if we add the appropriate scratch variables to
  // pq_scratch_t and initialize them in pq_flash_index.cpp::disk_iterate_to_fixed_point()
  virtual float brute_force_distance(const float* query_vec, uint8_t* base_vec) = 0;
};
} // namespace powerlaw_ann

#endif // PQ_QUANTIZED_DISTANCE
