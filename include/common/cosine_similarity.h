#ifndef COMMON_COSINE_SIMILARITY
#define COMMON_COSINE_SIMILARITY

#include "common/arch_compat.h"

#if defined(_WINDOWS) && POWERLAWANN_ARCH_X86
#include "common/simd_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <smmintrin.h>
#include <stdexcept>
#include <tmmintrin.h>

extern bool avx2_supported_cpu;

// SIMD implementation of Cosine similarity. Taken from hnsw library.

/**
 * Non-metric Space Library
 *
 * Authors: Bilegsaikhan Naidan (https://github.com/bileg), Leonid Boytsov
 * (http://boytsov.info). With contributions from Lawrence Cayton
 * (http://lcayton.com/) and others.
 *
 * For the complete list of contributors and further details see:
 * https://github.com/searchivarius/NonMetricSpaceLib
 *
 * Copyright (c) 2014
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

namespace powerlaw_ann {

using namespace std;

#define PORTABLE_ALIGN16 __declspec(align(16))

static float norm_scalar_product_simd_2(const int8_t* vector_1, const int8_t* vector_2,
                                        uint32_t qty) {
  if (avx2_supported_cpu) {
    __m256 cos, p1Len, p2Len;
    cos = p1Len = p2Len = _mm256_setzero_ps();
    while (qty >= 32) {
      __m256i rx = _mm256_load_si256((__m256i*) vector_1),
              ry = _mm256_load_si256((__m256i*) vector_2);
      cos = _mm256_add_ps(cos, _mm256_mul_epi8(rx, ry));
      p1Len = _mm256_add_ps(p1Len, _mm256_mul_epi8(rx, rx));
      p2Len = _mm256_add_ps(p2Len, _mm256_mul_epi8(ry, ry));
      vector_1 += 32;
      vector_2 += 32;
      qty -= 32;
    }
    while (qty > 0) {
      __m128i rx = _mm_load_si128((__m128i*) vector_1), ry = _mm_load_si128((__m128i*) vector_2);
      cos = _mm256_add_ps(cos, _mm256_mul32_pi8(rx, ry));
      p1Len = _mm256_add_ps(p1Len, _mm256_mul32_pi8(rx, rx));
      p2Len = _mm256_add_ps(p2Len, _mm256_mul32_pi8(ry, ry));
      vector_1 += 4;
      vector_2 += 4;
      qty -= 4;
    }
    cos = _mm256_hadd_ps(_mm256_hadd_ps(cos, cos), cos);
    p1Len = _mm256_hadd_ps(_mm256_hadd_ps(p1Len, p1Len), p1Len);
    p2Len = _mm256_hadd_ps(_mm256_hadd_ps(p2Len, p2Len), p2Len);
    float denominator =
        max(numeric_limits<float>::min() * 2, sqrt(p1Len.m256_f32[0] + p1Len.m256_f32[4]) *
                                                  sqrt(p2Len.m256_f32[0] + p2Len.m256_f32[4]));
    float cosine = (cos.m256_f32[0] + cos.m256_f32[4]) / denominator;

    return max(float(-1), min(float(1), cosine));
  }

  __m128 cos, p1Len, p2Len;
  cos = p1Len = p2Len = _mm_setzero_ps();
  __m128i rx, ry;
  while (qty >= 16) {
    rx = _mm_load_si128((__m128i*) vector_1);
    ry = _mm_load_si128((__m128i*) vector_2);
    cos = _mm_add_ps(cos, _mm_mul_epi8(rx, ry));
    p1Len = _mm_add_ps(p1Len, _mm_mul_epi8(rx, rx));
    p2Len = _mm_add_ps(p2Len, _mm_mul_epi8(ry, ry));
    vector_1 += 16;
    vector_2 += 16;
    qty -= 16;
  }
  while (qty > 0) {
    rx = _mm_load_si128((__m128i*) vector_1);
    ry = _mm_load_si128((__m128i*) vector_2);
    cos = _mm_add_ps(cos, _mm_mul32_pi8(rx, ry));
    p1Len = _mm_add_ps(p1Len, _mm_mul32_pi8(rx, rx));
    p2Len = _mm_add_ps(p2Len, _mm_mul32_pi8(ry, ry));
    vector_1 += 4;
    vector_2 += 4;
    qty -= 4;
  }
  cos = _mm_hadd_ps(_mm_hadd_ps(cos, cos), cos);
  p1Len = _mm_hadd_ps(_mm_hadd_ps(p1Len, p1Len), p1Len);
  p2Len = _mm_hadd_ps(_mm_hadd_ps(p2Len, p2Len), p2Len);
  float norm1 = p1Len.m128_f32[0];
  float norm2 = p2Len.m128_f32[0];

  static const float eps = numeric_limits<float>::min() * 2;

  if (norm1 < eps) { /*
                      * This shouldn't normally happen for this space, but
                      * if it does, we don't want to get NANs
                      */
    if (norm2 < eps) {
      return 1;
    }
    return 0;
  }
  /*
   * Sometimes due to rounding errors, we get values > 1 or < -1.
   * This throws off other functions that use scalar product, e.g., acos
   */
  return max(float(-1), min(float(1), cos.m128_f32[0] / sqrt(norm1) / sqrt(norm2)));
}

static float norm_scalar_product_simd(const float* vector_1, const float* vector_2, uint32_t qty) {
  // Didn't get significant performance gain compared with 128bit version.
  static const float eps = numeric_limits<float>::min() * 2;

  if (avx2_supported_cpu) {
    uint32_t qty8 = qty / 8;

    const float* end_1 = vector_1 + 8 * qty8;
    const float* end_2 = vector_1 + qty;

    __m256 v1, v2;
    __m256 sum_prod = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);
    __m256 sum_square1 = sum_prod;
    __m256 sum_square2 = sum_prod;

    while (vector_1 < end_1) {
      v1 = _mm256_loadu_ps(vector_1);
      vector_1 += 8;
      v2 = _mm256_loadu_ps(vector_2);
      vector_2 += 8;
      sum_prod = _mm256_add_ps(sum_prod, _mm256_mul_ps(v1, v2));
      sum_square1 = _mm256_add_ps(sum_square1, _mm256_mul_ps(v1, v1));
      sum_square2 = _mm256_add_ps(sum_square2, _mm256_mul_ps(v2, v2));
    }

    float PORTABLE_ALIGN16 tmp_result_product[8];
    float PORTABLE_ALIGN16 tmp_result_square_1[8];
    float PORTABLE_ALIGN16 tmp_result_square_2[8];

    _mm256_store_ps(tmp_result_product, sum_prod);
    _mm256_store_ps(tmp_result_square_1, sum_square1);
    _mm256_store_ps(tmp_result_square_2, sum_square2);

    float sum = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    for (uint32_t i = 0; i < 8; ++i) {
      sum += tmp_result_product[i];
      norm1 += tmp_result_square_1[i];
      norm2 += tmp_result_square_2[i];
    }

    while (vector_1 < end_2) {
      sum += (*vector_1) * (*vector_2);
      norm1 += (*vector_1) * (*vector_1);
      norm2 += (*vector_2) * (*vector_2);

      ++vector_1;
      ++vector_2;
    }

    if (norm1 < eps) {
      return norm2 < eps ? 1.0f : 0.0f;
    }

    return max(float(-1), min(float(1), sum / sqrt(norm1) / sqrt(norm2)));
  }

  __m128 v1, v2;
  __m128 sum_prod = _mm_set1_ps(0);
  __m128 sum_square1 = sum_prod;
  __m128 sum_square2 = sum_prod;

  while (qty >= 4) {
    v1 = _mm_loadu_ps(vector_1);
    vector_1 += 4;
    v2 = _mm_loadu_ps(vector_2);
    vector_2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    sum_square1 = _mm_add_ps(sum_square1, _mm_mul_ps(v1, v1));
    sum_square2 = _mm_add_ps(sum_square2, _mm_mul_ps(v2, v2));

    qty -= 4;
  }

  float sum =
      sum_prod.m128_f32[0] + sum_prod.m128_f32[1] + sum_prod.m128_f32[2] + sum_prod.m128_f32[3];
  float norm1 = sum_square1.m128_f32[0] + sum_square1.m128_f32[1] + sum_square1.m128_f32[2] +
                sum_square1.m128_f32[3];
  float norm2 = sum_square2.m128_f32[0] + sum_square2.m128_f32[1] + sum_square2.m128_f32[2] +
                sum_square2.m128_f32[3];

  if (norm1 < eps) {
    return norm2 < eps ? 1.0f : 0.0f;
  }

  return max(float(-1), min(float(1), sum / sqrt(norm1) / sqrt(norm2)));
}

static float norm_scalar_product_simd_2(const float* vector_1, const float* vector_2,
                                        uint32_t qty) {
  return norm_scalar_product_simd(vector_1, vector_2, qty);
}

template <class T>
static float cosine_similarity_2(const T* p1, const T* p2, uint32_t qty) {
  return std::max(0.0f, 1.0f - norm_scalar_product_simd_2(p1, p2, qty));
}

// static template float cosine_similarity_2<__int8>(const __int8* vector_1,
//                                         const __int8* vector_2, size_t qty);

// static template float cosine_similarity_2<float>(const float* vector_1,
//                                        const float* vector_2, size_t qty);

template <class T>
static void normalize_cosine_similarity(T* vector, uint32_t qty) {
  T sum = 0;
  for (uint32_t i = 0; i < qty; ++i) {
    sum += vector[i] * vector[i];
  }
  sum = 1 / sqrt(sum);
  if (sum == 0) {
    sum = numeric_limits<T>::min();
  }
  for (uint32_t i = 0; i < qty; ++i) {
    vector[i] *= sum;
  }
}

// template static void normalize_cosine_similarity<float>(float* vector,
//                                                      size_t qty);
// template static void normalize_cosine_similarity<double>(double* vector,
//                                                       size_t  qty);

template <>
void normalize_cosine_similarity(__int8* /*vector*/, uint32_t /*qty*/) {
  throw std::runtime_error("For int8 type vector, you can not use cosine distance!");
}

template <>
void normalize_cosine_similarity(__int16* /*vector*/, uint32_t /*qty*/) {
  throw std::runtime_error("For int16 type vector, you can not use cosine distance!");
}

template <>
void normalize_cosine_similarity(int* /*vector*/, uint32_t /*qty*/) {
  throw std::runtime_error("For int type vector, you can not use cosine distance!");
}
} // namespace powerlaw_ann

#endif // defined(_WINDOWS) && POWERLAWANN_ARCH_X86

#endif // COMMON_COSINE_SIMILARITY
