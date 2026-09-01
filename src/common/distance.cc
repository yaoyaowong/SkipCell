#include "common/arch_compat.h"

#if POWERLAWANN_ARCH_X86 && defined(_WINDOWS)
#include <immintrin.h>
#include <intrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#elif POWERLAWANN_ARCH_X86
#include <immintrin.h>
#endif

#include "common/cosine_similarity.h"
#include "common/diskann_exception.h"
#include "common/distance.h"
#include "common/logger.h"
#include "common/simd_utils.h"
#include "common/utils.h"

#include <iostream>

namespace powerlaw_ann {

//
// Base Class Implementatons
//
template <typename T>
float distance_t<T>::compare(const T* a, const T* b, const float norm_a, const float norm_b,
                             uint32_t length) const {
  throw std::logic_error("This function is not implemented.");
}

template <typename T>
uint32_t distance_t<T>::post_normalization_dimension(uint32_t orig_dimension) const {
  return orig_dimension;
}

template <typename T>
powerlaw_ann::metric_t distance_t<T>::get_metric() const {
  return distance_metric_;
}

template <typename T>
bool distance_t<T>::preprocessing_required() const {
  return false;
}

template <typename T>
void distance_t<T>::preprocess_base_points(T* original_data, const size_t orig_dim,
                                           const size_t num_points) {}

template <typename T>
void distance_t<T>::preprocess_query(const T* query_vec, const size_t query_dim, T* scratch_query) {
  std::memcpy(scratch_query, query_vec, query_dim * sizeof(T));
}

template <typename T>
size_t distance_t<T>::get_required_alignment() const {
  return alignment_factor_;
}

//
// Cosine distance functions.
//

float cosine_distance_int8_t::compare(const int8_t* a, const int8_t* b, uint32_t length) const {
#if defined(_WINDOWS) && POWERLAWANN_ARCH_X86
  return powerlaw_ann::cosine_similarity_2<int8_t>(a, b, length);
#else
  int mag_a = 0, mag_b = 0, scalar_product = 0;
  for (uint32_t i = 0; i < length; i++) {
    mag_a += ((int32_t) a[i]) * ((int32_t) a[i]);
    mag_b += ((int32_t) b[i]) * ((int32_t) b[i]);
    scalar_product += ((int32_t) a[i]) * ((int32_t) b[i]);
  }
  // similarity == 1-cosine distance
  return 1.0f - (float) (scalar_product / (sqrt(mag_a) * sqrt(mag_b)));
#endif
}

float cosine_distance_float_t::compare(const float* a, const float* b, uint32_t length) const {
#if defined(_WINDOWS) && POWERLAWANN_ARCH_X86
  return powerlaw_ann::cosine_similarity_2<float>(a, b, length);
#else
  float mag_a = 0, mag_b = 0, scalar_product = 0;
  for (uint32_t i = 0; i < length; i++) {
    mag_a += (a[i]) * (a[i]);
    mag_b += (b[i]) * (b[i]);
    scalar_product += (a[i]) * (b[i]);
  }
  // similarity == 1-cosine distance
  return 1.0f - (scalar_product / (sqrt(mag_a) * sqrt(mag_b)));
#endif
}

float slow_cosine_distance_uint8_t::compare(const uint8_t* a, const uint8_t* b,
                                            uint32_t length) const {
  int mag_a = 0, mag_b = 0, scalar_product = 0;
  for (uint32_t i = 0; i < length; i++) {
    mag_a += ((uint32_t) a[i]) * ((uint32_t) a[i]);
    mag_b += ((uint32_t) b[i]) * ((uint32_t) b[i]);
    scalar_product += ((uint32_t) a[i]) * ((uint32_t) b[i]);
  }
  // similarity == 1-cosine distance
  return 1.0f - (float) (scalar_product / (sqrt(mag_a) * sqrt(mag_b)));
}

//
// L2 distance functions.
//

float l2_distance_int8_t::compare(const int8_t* a, const int8_t* b, uint32_t size) const {
#ifdef _WINDOWS
#ifdef USE_AVX2
  __m256 r = _mm256_setzero_ps();
  char *x = (char*) a, *y = (char*) b;
  while (size >= 32) {
    __m256i r1 =
        _mm256_subs_epi8(_mm256_loadu_si256((__m256i*) x), _mm256_loadu_si256((__m256i*) y));
    r = _mm256_add_ps(r, _mm256_mul_epi8(r1, r1));
    x += 32;
    y += 32;
    size -= 32;
  }
  while (size > 0) {
    __m128i r2 = _mm_subs_epi8(_mm_loadu_si128((__m128i*) x), _mm_loadu_si128((__m128i*) y));
    r = _mm256_add_ps(r, _mm256_mul32_pi8(r2, r2));
    x += 4;
    y += 4;
    size -= 4;
  }
  r = _mm256_hadd_ps(_mm256_hadd_ps(r, r), r);
  return r.m256_f32[0] + r.m256_f32[4];
#else
  int32_t result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
  for (int32_t i = 0; i < (int32_t) size; i++) {
    result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
              ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
  }
  return (float) result;
#endif
#else
  int32_t result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
  for (int32_t i = 0; i < (int32_t) size; i++) {
    result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
              ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
  }
  return (float) result;
#endif
}

float l2_distance_uint8_t::compare(const uint8_t* a, const uint8_t* b, uint32_t size) const {
  uint32_t result = 0;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
#endif
  for (int32_t i = 0; i < (int32_t) size; i++) {
    result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) *
              ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
  }
  return (float) result;
}

float l2_distance_float_t::compare(const float* a, const float* b, uint32_t size) const {
  float result = 0;
#ifdef USE_AVX2
  const uint32_t niters = size / 8;
  __m256 sum = _mm256_setzero_ps();
  for (uint32_t j = 0; j < niters; ++j) {
    // scope is a[8j:8j+7], b[8j:8j+7]
    // load a_vec
    if (j < (niters - 1)) {
      _mm_prefetch((char*) (a + 8 * (j + 1)), _MM_HINT_T0);
      _mm_prefetch((char*) (b + 8 * (j + 1)), _MM_HINT_T0);
    }
    __m256 a_vec = _mm256_loadu_ps(a + 8 * j);
    // load b_vec
    __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
    // a_vec - b_vec
    __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);

    sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
  }

  // horizontal add sum
  result = _mm256_reduce_add_ps(sum);
  for (uint32_t dimension = niters * 8; dimension < size; ++dimension) {
    const float delta = a[dimension] - b[dimension];
    result += delta * delta;
  }
#else
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result)
#endif
  for (int32_t i = 0; i < (int32_t) size; i++) {
    result += (a[i] - b[i]) * (a[i] - b[i]);
  }
#endif
  return result;
}

template <typename T>
float slow_l2_distance_t<T>::compare(const T* a, const T* b, uint32_t length) const {
  float result = 0.0f;
  for (uint32_t i = 0; i < length; i++) {
    result += ((float) (a[i] - b[i])) * (a[i] - b[i]);
  }
  return result;
}

#ifdef _WINDOWS
float avx_l2_distance_int8_t::compare(const int8_t* a, const int8_t* b, uint32_t length) const {
  __m128 r = _mm_setzero_ps();
  __m128i r1;
  while (length >= 16) {
    r1 = _mm_subs_epi8(_mm_load_si128((__m128i*) a), _mm_load_si128((__m128i*) b));
    r = _mm_add_ps(r, _mm_mul_epi8(r1));
    a += 16;
    b += 16;
    length -= 16;
  }
  r = _mm_hadd_ps(_mm_hadd_ps(r, r), r);
  float res = r.m128_f32[0];

  if (length >= 8) {
    __m128 r2 = _mm_setzero_ps();
    __m128i r3 =
        _mm_subs_epi8(_mm_load_si128((__m128i*) (a - 8)), _mm_load_si128((__m128i*) (b - 8)));
    r2 = _mm_add_ps(r2, _mm_mulhi_epi8(r3));
    a += 8;
    b += 8;
    length -= 8;
    r2 = _mm_hadd_ps(_mm_hadd_ps(r2, r2), r2);
    res += r2.m128_f32[0];
  }

  if (length >= 4) {
    __m128 r2 = _mm_setzero_ps();
    __m128i r3 =
        _mm_subs_epi8(_mm_load_si128((__m128i*) (a - 12)), _mm_load_si128((__m128i*) (b - 12)));
    r2 = _mm_add_ps(r2, _mm_mulhi_epi8_shift32(r3));
    res += r2.m128_f32[0] + r2.m128_f32[1];
  }

  return res;
}

float avx_l2_distance_float_t::compare(const float* a, const float* b, uint32_t length) const {
  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (length >= 4) {
    v1 = _mm_loadu_ps(a);
    a += 4;
    v2 = _mm_loadu_ps(b);
    b += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    length -= 4;
  }

  return sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2] + sum.m128_f32[3];
}
#else
float avx_l2_distance_int8_t::compare(const int8_t*, const int8_t*, uint32_t) const { return 0; }
float avx_l2_distance_float_t::compare(const float*, const float*, uint32_t) const { return 0; }
#endif

template <typename T>
float inner_product_distance_t<T>::inner_product(const T* a, const T* b, uint32_t size) const {
  if (!std::is_floating_point<T>::value) {
    powerlaw_ann::cerr << "ERROR: Inner Product only defined for float currently." << std::endl;
    throw powerlaw_ann::diskann_exception_t(
        "ERROR: Inner Product only defined for float currently.", -1, __FUNCSIG__, __FILE__,
        __LINE__);
  }

  float result = 0;

#ifdef __GNUC__
#ifdef USE_AVX2
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2)                                                    \
  tmp1 = _mm256_loadu_ps(addr1);                                                                   \
  tmp2 = _mm256_loadu_ps(addr2);                                                                   \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);                                                                \
  dest = _mm256_add_ps(dest, tmp1);

  __m256 sum;
  __m256 l0, l1;
  __m256 r0, r1;
  uint32_t D = (size + 7) & ~7U;
  uint32_t DR = D % 16;
  uint32_t DD = D - DR;
  const float* l = (float*) a;
  const float* r = (float*) b;
  const float* e_l = l + DD;
  const float* e_r = r + DD;
  float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

  sum = _mm256_loadu_ps(unpack);
  if (DR) {
    AVX_DOT(e_l, e_r, sum, l0, r0);
  }

  for (uint32_t i = 0; i < DD; i += 16, l += 16, r += 16) {
    AVX_DOT(l, r, sum, l0, r0);
    AVX_DOT(l + 8, r + 8, sum, l1, r1);
  }
  _mm256_storeu_ps(unpack, sum);
  result =
      unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2)                                                    \
  tmp1 = _mm_loadu_ps(addr1);                                                                      \
  tmp2 = _mm_loadu_ps(addr2);                                                                      \
  tmp1 = _mm_mul_ps(tmp1, tmp2);                                                                   \
  dest = _mm_add_ps(dest, tmp1);
  __m128 sum;
  __m128 l0, l1, l2, l3;
  __m128 r0, r1, r2, r3;
  uint32_t D = (size + 3) & ~3U;
  uint32_t DR = D % 16;
  uint32_t DD = D - DR;
  const float* l = a;
  const float* r = b;
  const float* e_l = l + DD;
  const float* e_r = r + DD;
  float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

  sum = _mm_load_ps(unpack);
  switch (DR) {
  case 12:
    SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
    [[fallthrough]];
  case 8:
    SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
    [[fallthrough]];
  case 4:
    SSE_DOT(e_l, e_r, sum, l0, r0);
  default:
    break;
  }
  for (uint32_t i = 0; i < DD; i += 16, l += 16, r += 16) {
    SSE_DOT(l, r, sum, l0, r0);
    SSE_DOT(l + 4, r + 4, sum, l1, r1);
    SSE_DOT(l + 8, r + 8, sum, l2, r2);
    SSE_DOT(l + 12, r + 12, sum, l3, r3);
  }
  _mm_storeu_ps(unpack, sum);
  result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else

  float dot0, dot1, dot2, dot3;
  const float* last = a + size;
  const float* unroll_group = last - 3;

  /* Process 4 items with each loop for efficiency. */
  while (a < unroll_group) {
    dot0 = a[0] * b[0];
    dot1 = a[1] * b[1];
    dot2 = a[2] * b[2];
    dot3 = a[3] * b[3];
    result += dot0 + dot1 + dot2 + dot3;
    a += 4;
    b += 4;
  }
  /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
  while (a < last) {
    result += *a++ * *b++;
  }
#endif
#endif
#endif
  return result;
}

template <typename T>
float fast_l2_distance_t<T>::compare(const T* a, const T* b, float norm, uint32_t size) const {
  float result = -2 * inner_product_distance_t<T>::inner_product(a, b, size);
  result += norm;
  return result;
}

template <typename T>
float fast_l2_distance_t<T>::norm(const T* a, uint32_t size) const {
  if (!std::is_floating_point<T>::value) {
    powerlaw_ann::cerr << "ERROR: fast_l2 only defined for float currently." << std::endl;
    throw powerlaw_ann::diskann_exception_t("ERROR: fast_l2 only defined for float currently.", -1,
                                            __FUNCSIG__, __FILE__, __LINE__);
  }
  float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_L2NORM(addr, dest, tmp)                                                                \
  tmp = _mm256_loadu_ps(addr);                                                                     \
  tmp = _mm256_mul_ps(tmp, tmp);                                                                   \
  dest = _mm256_add_ps(dest, tmp);

  __m256 sum;
  __m256 l0, l1;
  uint32_t D = (size + 7) & ~7U;
  uint32_t DR = D % 16;
  uint32_t DD = D - DR;
  const float* l = (float*) a;
  const float* e_l = l + DD;
  float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

  sum = _mm256_loadu_ps(unpack);
  if (DR) {
    AVX_L2NORM(e_l, sum, l0);
  }
  for (uint32_t i = 0; i < DD; i += 16, l += 16) {
    AVX_L2NORM(l, sum, l0);
    AVX_L2NORM(l + 8, sum, l1);
  }
  _mm256_storeu_ps(unpack, sum);
  result =
      unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
#define SSE_L2NORM(addr, dest, tmp)                                                                \
  tmp = _mm_loadu_ps(addr);                                                                        \
  tmp = _mm_mul_ps(tmp, tmp);                                                                      \
  dest = _mm_add_ps(dest, tmp);

  __m128 sum;
  __m128 l0, l1, l2, l3;
  uint32_t D = (size + 3) & ~3U;
  uint32_t DR = D % 16;
  uint32_t DD = D - DR;
  const float* l = a;
  const float* e_l = l + DD;
  float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

  sum = _mm_load_ps(unpack);
  switch (DR) {
  case 12:
    SSE_L2NORM(e_l + 8, sum, l2);
    [[fallthrough]];
  case 8:
    SSE_L2NORM(e_l + 4, sum, l1);
    [[fallthrough]];
  case 4:
    SSE_L2NORM(e_l, sum, l0);
  default:
    break;
  }
  for (uint32_t i = 0; i < DD; i += 16, l += 16) {
    SSE_L2NORM(l, sum, l0);
    SSE_L2NORM(l + 4, sum, l1);
    SSE_L2NORM(l + 8, sum, l2);
    SSE_L2NORM(l + 12, sum, l3);
  }
  _mm_storeu_ps(unpack, sum);
  result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
  float dot0, dot1, dot2, dot3;
  const float* last = a + size;
  const float* unroll_group = last - 3;

  /* Process 4 items with each loop for efficiency. */
  while (a < unroll_group) {
    dot0 = a[0] * a[0];
    dot1 = a[1] * a[1];
    dot2 = a[2] * a[2];
    dot3 = a[3] * a[3];
    result += dot0 + dot1 + dot2 + dot3;
    a += 4;
  }
  /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
  while (a < last) {
    result += (*a) * (*a);
    a++;
  }
#endif
#endif
#endif
  return result;
}

float avx_inner_product_distance_float_t::compare(const float* a, const float* b,
                                                  uint32_t size) const {
#if POWERLAWANN_ARCH_X86 && defined(USE_AVX2)
  float result = 0.0f;
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2)                                                    \
  tmp1 = _mm256_loadu_ps(addr1);                                                                   \
  tmp2 = _mm256_loadu_ps(addr2);                                                                   \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);                                                                \
  dest = _mm256_add_ps(dest, tmp1);

  __m256 sum;
  __m256 l0, l1;
  __m256 r0, r1;
  uint32_t D = (size + 7) & ~7U;
  uint32_t DR = D % 16;
  uint32_t DD = D - DR;
  const float* l = (float*) a;
  const float* r = (float*) b;
  const float* e_l = l + DD;
  const float* e_r = r + DD;
#ifndef _WINDOWS
  float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};
#else
  __declspec(align(32)) float unpack[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#endif

  sum = _mm256_loadu_ps(unpack);
  if (DR) {
    AVX_DOT(e_l, e_r, sum, l0, r0);
  }

  for (uint32_t i = 0; i < DD; i += 16, l += 16, r += 16) {
    AVX_DOT(l, r, sum, l0, r0);
    AVX_DOT(l + 8, r + 8, sum, l1, r1);
  }
  _mm256_storeu_ps(unpack, sum);
  result =
      unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

  return -result;
#else
  float result = 0.0F;
  for (uint32_t i = 0; i < size; ++i) {
    result += a[i] * b[i];
  }
  return -result;
#endif
}

uint32_t avx_normalized_cosine_distance_float_t::post_normalization_dimension(
    uint32_t orig_dimension) const {
  return orig_dimension;
}
bool avx_normalized_cosine_distance_float_t::preprocessing_required() const { return true; }
void avx_normalized_cosine_distance_float_t::preprocess_base_points(float* original_data,
                                                                    const size_t orig_dim,
                                                                    const size_t num_points) {
  for (uint32_t i = 0; i < num_points; i++) {
    normalize((float*) (original_data + i * orig_dim), orig_dim);
  }
}

void avx_normalized_cosine_distance_float_t::preprocess_query(const float* query_vec,
                                                              const size_t query_dim,
                                                              float* query_scratch) {
  normalize_and_copy(query_vec, (uint32_t) query_dim, query_scratch);
}

void avx_normalized_cosine_distance_float_t::normalize_and_copy(const float* query_vec,
                                                                const uint32_t query_dim,
                                                                float* query_target) const {
  float norm = get_norm(query_vec, query_dim);

  for (uint32_t i = 0; i < query_dim; i++) {
    query_target[i] = query_vec[i] / norm;
  }
}

// Get the right distance function for the given metric.
template <>
powerlaw_ann::distance_t<float>* get_distance_function(powerlaw_ann::metric_t m) {
  if (m == powerlaw_ann::metric_t::L2) {
    if (avx2_supported_cpu) {
      powerlaw_ann::cout << "L2: Using AVX2 distance computation l2_distance_float_t" << std::endl;
      return new powerlaw_ann::l2_distance_float_t();
    } else if (avx_supported_cpu) {
      powerlaw_ann::cout << "L2: AVX2 not supported. Using AVX distance computation" << std::endl;
      return new powerlaw_ann::avx_l2_distance_float_t();
    } else {
      powerlaw_ann::cout << "L2: Older CPU. Using slow distance computation" << std::endl;
      return new powerlaw_ann::slow_l2_distance_t<float>();
    }
  } else if (m == powerlaw_ann::metric_t::COSINE) {
    powerlaw_ann::cout << "Cosine: Using either AVX or AVX2 implementation" << std::endl;
    return new powerlaw_ann::cosine_distance_float_t();
  } else if (m == powerlaw_ann::metric_t::INNER_PRODUCT) {
    powerlaw_ann::cout << "Inner product: Using AVX2 implementation "
                          "avx_inner_product_distance_float_t"
                       << std::endl;
    return new powerlaw_ann::avx_inner_product_distance_float_t();
  } else if (m == powerlaw_ann::metric_t::FAST_L2) {
    powerlaw_ann::cout << "Fast_L2: Using AVX2 implementation with norm "
                          "memoization fast_l2_distance_t<float>"
                       << std::endl;
    return new powerlaw_ann::fast_l2_distance_t<float>();
  } else {
    std::stringstream stream;
    stream << "Only L2, cosine, and inner product supported for floating "
              "point vectors as of now."
           << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
}

template <>
powerlaw_ann::distance_t<int8_t>* get_distance_function(powerlaw_ann::metric_t m) {
  if (m == powerlaw_ann::metric_t::L2) {
    if (avx2_supported_cpu) {
      powerlaw_ann::cout << "Using AVX2 distance computation l2_distance_int8_t." << std::endl;
      return new powerlaw_ann::l2_distance_int8_t();
    } else if (avx_supported_cpu) {
      powerlaw_ann::cout << "AVX2 not supported. Using AVX distance computation" << std::endl;
      return new powerlaw_ann::avx_l2_distance_int8_t();
    } else {
      powerlaw_ann::cout << "Older CPU. Using slow distance computation "
                            "slow_l2_distance_t<int8_t>."
                         << std::endl;
      return new powerlaw_ann::slow_l2_distance_t<int8_t>();
    }
  } else if (m == powerlaw_ann::metric_t::COSINE) {
    powerlaw_ann::cout << "Using either AVX or AVX2 for Cosine similarity "
                          "cosine_distance_int8_t."
                       << std::endl;
    return new powerlaw_ann::cosine_distance_int8_t();
  } else {
    std::stringstream stream;
    stream << "Only L2 and cosine supported for signed byte vectors." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
}

template <>
powerlaw_ann::distance_t<uint8_t>* get_distance_function(powerlaw_ann::metric_t m) {
  if (m == powerlaw_ann::metric_t::L2) {
#ifdef _WINDOWS
    powerlaw_ann::cout << "WARNING: AVX/AVX2 distance function not defined for Uint8. "
                          "Using "
                          "slow version. "
                          "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
                       << std::endl;
#endif
    return new powerlaw_ann::l2_distance_uint8_t();
  } else if (m == powerlaw_ann::metric_t::COSINE) {
    powerlaw_ann::cout << "AVX/AVX2 distance function not defined for Uint8. Using "
                          "slow version slow_cosine_distance_uint8_t() "
                          "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
                       << std::endl;
    return new powerlaw_ann::slow_cosine_distance_uint8_t();
  } else {
    std::stringstream stream;
    stream << "Only L2 and cosine supported for uint32_t byte vectors." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
}

template POWERLAWANN_DLLEXPORT class inner_product_distance_t<float>;
#if 0
template POWERLAWANN_DLLEXPORT class inner_product_distance_t<int8_t>;
template POWERLAWANN_DLLEXPORT class inner_product_distance_t<uint8_t>;
#endif

template POWERLAWANN_DLLEXPORT class fast_l2_distance_t<float>;
#if 0
template POWERLAWANN_DLLEXPORT class fast_l2_distance_t<int8_t>;
template POWERLAWANN_DLLEXPORT class fast_l2_distance_t<uint8_t>;
#endif

template POWERLAWANN_DLLEXPORT class slow_l2_distance_t<float>;
template POWERLAWANN_DLLEXPORT class slow_l2_distance_t<int8_t>;
template POWERLAWANN_DLLEXPORT class slow_l2_distance_t<uint8_t>;

} // namespace powerlaw_ann
