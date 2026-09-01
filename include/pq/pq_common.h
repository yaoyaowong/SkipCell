#ifndef PQ_PQ_COMMON
#define PQ_PQ_COMMON

#include <string>

#define NUM_PQ_BITS 8
#define NUM_PQ_CENTROIDS (1 << NUM_PQ_BITS)
#define MAX_OPQ_ITERS 20
#define NUM_KMEANS_REPS_PQ 12
#define MAX_PQ_TRAINING_SET_SIZE 256000
#define MAX_PQ_CHUNKS 512

namespace powerlaw_ann {
inline std::string get_quantized_vectors_filename(const std::string& prefix, bool use_opq,
                                                  uint32_t num_chunks) {
  return prefix + (use_opq ? "opq_" : "pq") + std::to_string(num_chunks) + "compressed_.bin";
}

inline std::string get_pivot_data_filename(const std::string& prefix, bool use_opq,
                                           uint32_t num_chunks) {
  return prefix + (use_opq ? "opq_" : "pq") + std::to_string(num_chunks) + "pivots_.bin";
}

inline std::string get_rotation_matrix_suffix(const std::string& pivot_data_filename) {
  return pivot_data_filename + "rotation_matrix_.bin";
}

} // namespace powerlaw_ann

#endif // PQ_PQ_COMMON
