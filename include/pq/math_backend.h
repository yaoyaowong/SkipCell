#ifndef PQ_MATH_BACKEND
#define PQ_MATH_BACKEND

#include "common/diskann_exception.h"

#include <cstdint>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <mkl.h>
#endif

namespace powerlaw_ann::math_backend {

#if defined(__APPLE__)
using math_int_t = __LAPACK_int;

inline void set_num_threads(uint32_t) {
  // Accelerate manages its own worker pool. OpenMP still controls the outer
  // parallel loops used by the index builder.
}

inline int compute_svd(math_int_t, float*, float*, float*, float*) {
  throw diskann_exception_t(
      "OPQ training is not enabled by the macOS correctness backend. Use the Ubuntu oneMKL "
      "backend for OPQ experiments.",
      -1);
}
#else
using math_int_t = MKL_INT;

inline void set_num_threads(uint32_t num_threads) {
  mkl_set_num_threads(static_cast<int>(num_threads));
}

inline int compute_svd(math_int_t dimension, float* matrix, float* singular_values,
                       float* left_vectors, float* right_vectors_transposed) {
  return LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', dimension, dimension, matrix, dimension,
                        singular_values, left_vectors, dimension, right_vectors_transposed,
                        dimension);
}
#endif

} // namespace powerlaw_ann::math_backend

#endif // PQ_MATH_BACKEND
