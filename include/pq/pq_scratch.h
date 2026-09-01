#ifndef PQ_PQ_SCRATCH
#define PQ_PQ_SCRATCH

#include <cstddef>
#include <cstdint>

namespace powerlaw_ann {

template <typename T>
class pq_scratch_t {
public:
  float* aligned_pqtable_dist_scratch = nullptr; // MUST BE AT LEAST [256 * NCHUNKS]
  float* aligned_dist_scratch = nullptr;         // MUST BE AT LEAST diskann MAX_DEGREE
  uint8_t* aligned_pq_coord_scratch = nullptr;   // AT LEAST  [N_CHUNKS * MAX_DEGREE]
  float* rotated_query = nullptr;
  float* aligned_query_float = nullptr;

  pq_scratch_t(size_t graph_degree, size_t aligned_dim,
               size_t candidate_capacity = 0);
  void initialize(size_t dim, const T* query, const float norm = 1.0f);
  virtual ~pq_scratch_t();
};

} // namespace powerlaw_ann

#endif // PQ_PQ_SCRATCH
