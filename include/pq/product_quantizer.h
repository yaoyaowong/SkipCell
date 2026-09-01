#ifndef PQ_PRODUCT_QUANTIZER
#define PQ_PRODUCT_QUANTIZER

#include "common/distance.h"
#include "common/platform_compat.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace powerlaw_ann {
class fixed_chunk_pq_table_t {
  float* tables_ = nullptr;     // pq_tables = float array of size [256 * num_dimensions_]
  uint64_t num_dimensions_ = 0; // num_dimensions_ = true dimension of vectors
  uint64_t num_chunks_ = 0;
  bool use_rotation_ = false;
  uint32_t* chunk_offsets_ = nullptr;
  float* centroid_ = nullptr;
  float* transposed_tables_ = nullptr; // same as pq_tables, but col-major
  float* transposed_rotation_matrix_ = nullptr;

public:
  fixed_chunk_pq_table_t();

  virtual ~fixed_chunk_pq_table_t();

#ifdef EXEC_ENV_OLS
  void load_pq_centroid_bin(memory_mapped_files_t& files, const char* pq_table_file,
                            size_t num_chunks);
#else
  void load_pq_centroid_bin(const char* pq_table_file, size_t num_chunks);
#endif

  uint32_t get_num_chunks();

  void preprocess_query(float* query_vec);

  // assumes pre-processed query
  void populate_chunk_distances(const float* query_vec, float* dist_vec);

  float l2_distance(const float* query_vec, uint8_t* base_vec);

  float inner_product(const float* query_vec, uint8_t* base_vec);

  // assumes no rotation is involved
  void inflate_vector(uint8_t* base_vec, float* out_vec);

  void populate_chunk_inner_products(const float* query_vec, float* dist_vec);
};

void aggregate_coords(const std::vector<uint32_t>& ids, const uint8_t* all_coords,
                      size_t num_dimensions, uint8_t* out);

void pq_dist_lookup(const uint8_t* pq_ids, const size_t n_pts, const size_t pq_nchunks,
                    const float* pq_dists, std::vector<float>& dists_out);

// Need to replace calls to these with calls to vector& based functions above
void aggregate_coords(const uint32_t* ids, size_t num_ids, const uint8_t* all_coords,
                      size_t num_dimensions, uint8_t* out);

void pq_dist_lookup(const uint8_t* pq_ids, const size_t n_pts, const size_t pq_nchunks,
                    const float* pq_dists, float* dists_out);

/** Score decoded uint8 reconstructed vectors with squared L2 using AVX2 when available. */
void decoded_u8_l2_lookup(const uint8_t* vectors, size_t n_pts, size_t dimensions,
                          const uint8_t* query, float* dists_out);

/** Score original-ID uint8 vectors without gathering their payloads. */
void original_u8_l2_lookup(const uint8_t* vectors, const uint32_t* ids, size_t n_pts,
                           size_t dimensions, const uint8_t* query, float* dists_out);

POWERLAWANN_DLLEXPORT int generate_pq_pivots(const float* const train_data, size_t num_train,
                                             unsigned dim, unsigned num_centers,
                                             unsigned num_pq_chunks, unsigned max_k_means_reps,
                                             std::string pq_pivots_path,
                                             bool make_zero_mean = false);

POWERLAWANN_DLLEXPORT int generate_opq_pivots(const float* train_data, size_t num_train,
                                              unsigned dim, unsigned num_centers,
                                              unsigned num_pq_chunks, std::string opq_pivots_path,
                                              bool make_zero_mean = false);

POWERLAWANN_DLLEXPORT int generate_pq_pivots_simplified(const float* train_data, size_t num_train,
                                                        size_t dim, size_t num_pq_chunks,
                                                        std::vector<float>& pivot_data_vector);

template <typename T>
int generate_pq_data_from_pivots(const std::string& data_file, unsigned num_centers,
                                 unsigned num_pq_chunks, const std::string& pq_pivots_path,
                                 const std::string& pq_compressed_vectors_path,
                                 bool use_opq = false);

POWERLAWANN_DLLEXPORT int generate_pq_data_from_pivots_simplified(
    const float* data, const size_t num, const float* pivot_data, const size_t pivots_num,
    const size_t dim, const size_t num_pq_chunks, std::vector<uint8_t>& pq);

template <typename T>
void generate_disk_quantized_data(const std::string& data_file_to_use,
                                  const std::string& disk_pq_pivots_path,
                                  const std::string& disk_pq_compressed_vectors_path,
                                  const powerlaw_ann::metric_t compare_metric, const double p_val,
                                  size_t& disk_pq_dims);

template <typename T>
void generate_quantized_data(const std::string& data_file_to_use, const std::string& pq_pivots_path,
                             const std::string& pq_compressed_vectors_path,
                             const powerlaw_ann::metric_t compare_metric, const double p_val,
                             const size_t num_pq_chunks, const bool use_opq,
                             const std::string& codebook_prefix = "");
} // namespace powerlaw_ann

#endif // PQ_PRODUCT_QUANTIZER
