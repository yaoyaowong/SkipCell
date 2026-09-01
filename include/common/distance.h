#ifndef COMMON_DISTANCE
#define COMMON_DISTANCE

#include "common/platform_compat.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace powerlaw_ann {
enum metric_t { L2 = 0, INNER_PRODUCT = 1, COSINE = 2, FAST_L2 = 3 };

template <typename T>
class distance_t {
public:
  POWERLAWANN_DLLEXPORT distance_t(powerlaw_ann::metric_t dist_metric)
      : distance_metric_(dist_metric) {}

  // distance comparison function
  POWERLAWANN_DLLEXPORT virtual float compare(const T* a, const T* b, uint32_t length) const = 0;

  // Needed only for COSINE-BYTE and INNER_PRODUCT-BYTE
  POWERLAWANN_DLLEXPORT virtual float compare(const T* a, const T* b, const float norm_a,
                                              const float norm_b, uint32_t length) const;

  // For MIPS, normalization adds an extra dimension to the vectors.
  // This function lets callers know if the normalization process
  // changes the dimension.
  POWERLAWANN_DLLEXPORT virtual uint32_t
  post_normalization_dimension(uint32_t orig_dimension) const;

  POWERLAWANN_DLLEXPORT virtual powerlaw_ann::metric_t get_metric() const;

  // This is for efficiency. If no normalization is required, the callers
  // can simply ignore the normalize_data_for_build() function.
  POWERLAWANN_DLLEXPORT virtual bool preprocessing_required() const;

  // Check the preprocessing_required() function before calling this.
  // Clients can call the function like this:
  //
  //  if (metric->preprocessing_required()){
  //     T* normalized_data_batch;
  //      Split data into batches of batch_size and for each, call:
  //       metric->preprocess_base_points(data_batch, batch_size);
  //
  //  TODO: This does not take into account the case for SSD inner product
  //  where the dimensions change after normalization.
  POWERLAWANN_DLLEXPORT virtual void preprocess_base_points(T* original_data, const size_t orig_dim,
                                                            const size_t num_points);

  // Invokes normalization for a single vector during search. The scratch space
  // has to be created by the caller keeping track of the fact that
  // normalization might change the dimension of the query vector.
  POWERLAWANN_DLLEXPORT virtual void preprocess_query(const T* query_vec, const size_t query_dim,
                                                      T* scratch_query);

  // If an algorithm has a requirement that some data be aligned to a certain
  // boundary it can use this function to indicate that requirement. Currently,
  // we are setting it to 8 because that works well for AVX2. If we have AVX512
  // implementations of distance algos, they might have to set this to 16
  // (depending on how they are implemented)
  POWERLAWANN_DLLEXPORT virtual size_t get_required_alignment() const;

  // Providing a default implementation for the virtual destructor because we
  // don't expect most metric implementations to need it.
  POWERLAWANN_DLLEXPORT virtual ~distance_t() = default;

protected:
  powerlaw_ann::metric_t distance_metric_;
  size_t alignment_factor_ = 8;
};

class cosine_distance_int8_t : public distance_t<int8_t> {
public:
  cosine_distance_int8_t() : distance_t<int8_t>(powerlaw_ann::metric_t::COSINE) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const int8_t* a, const int8_t* b,
                                              uint32_t length) const;
};

class l2_distance_int8_t : public distance_t<int8_t> {
public:
  l2_distance_int8_t() : distance_t<int8_t>(powerlaw_ann::metric_t::L2) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const int8_t* a, const int8_t* b,
                                              uint32_t size) const;
};

// AVX implementations. Borrowed from HNSW code.
class avx_l2_distance_int8_t : public distance_t<int8_t> {
public:
  avx_l2_distance_int8_t() : distance_t<int8_t>(powerlaw_ann::metric_t::L2) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const int8_t* a, const int8_t* b,
                                              uint32_t length) const;
};

class cosine_distance_float_t : public distance_t<float> {
public:
  cosine_distance_float_t() : distance_t<float>(powerlaw_ann::metric_t::COSINE) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const float* a, const float* b,
                                              uint32_t length) const;
};

class l2_distance_float_t : public distance_t<float> {
public:
  l2_distance_float_t() : distance_t<float>(powerlaw_ann::metric_t::L2) {}

#ifdef _WINDOWS
  POWERLAWANN_DLLEXPORT virtual float compare(const float* a, const float* b, uint32_t size) const;
#else
  POWERLAWANN_DLLEXPORT virtual float compare(const float* a, const float* b, uint32_t size) const
      __attribute__((hot));
#endif
};

class avx_l2_distance_float_t : public distance_t<float> {
public:
  avx_l2_distance_float_t() : distance_t<float>(powerlaw_ann::metric_t::L2) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const float* a, const float* b,
                                              uint32_t length) const;
};

template <typename T>
class slow_l2_distance_t : public distance_t<T> {
public:
  slow_l2_distance_t() : distance_t<T>(powerlaw_ann::metric_t::L2) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const T* a, const T* b, uint32_t length) const;
};

class slow_cosine_distance_uint8_t : public distance_t<uint8_t> {
public:
  slow_cosine_distance_uint8_t() : distance_t<uint8_t>(powerlaw_ann::metric_t::COSINE) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const uint8_t* a, const uint8_t* b,
                                              uint32_t length) const;
};

class l2_distance_uint8_t : public distance_t<uint8_t> {
public:
  l2_distance_uint8_t() : distance_t<uint8_t>(powerlaw_ann::metric_t::L2) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const uint8_t* a, const uint8_t* b,
                                              uint32_t size) const;
};

template <typename T>
class inner_product_distance_t : public distance_t<T> {
public:
  inner_product_distance_t() : distance_t<T>(powerlaw_ann::metric_t::INNER_PRODUCT) {}

  inner_product_distance_t(powerlaw_ann::metric_t metric) : distance_t<T>(metric) {}
  float inner_product(const T* a, const T* b, unsigned size) const;

  float compare(const T* a, const T* b, unsigned size) const override {
    float result = inner_product(a, b, size);
    //      if (result < 0)
    //      return std::numeric_limits<float>::max();
    //      else
    return -result;
  }
};

template <typename T>
class fast_l2_distance_t : public inner_product_distance_t<T> {
  // currently defined only for float.
  // templated for future use.
public:
  fast_l2_distance_t() : inner_product_distance_t<T>(powerlaw_ann::metric_t::FAST_L2) {}
  using inner_product_distance_t<T>::compare;
  float norm(const T* a, unsigned size) const;
  float compare(const T* a, const T* b, float norm, unsigned size) const;
};

class avx_inner_product_distance_float_t : public distance_t<float> {
public:
  avx_inner_product_distance_float_t() : distance_t<float>(powerlaw_ann::metric_t::INNER_PRODUCT) {}
  POWERLAWANN_DLLEXPORT virtual float compare(const float* a, const float* b,
                                              uint32_t length) const;
};

class avx_normalized_cosine_distance_float_t : public distance_t<float> {
private:
  avx_inner_product_distance_float_t inner_product_;

protected:
  void normalize_and_copy(const float* a, uint32_t length, float* a_norm) const;

public:
  avx_normalized_cosine_distance_float_t() : distance_t<float>(powerlaw_ann::metric_t::COSINE) {}
  POWERLAWANN_DLLEXPORT float compare(const float* a, const float* b,
                                      uint32_t length) const override {
    // Inner product returns negative values to indicate distance.
    // This will ensure that cosine is between -1 and 1.
    return 1.0f + inner_product_.compare(a, b, length);
  }
  POWERLAWANN_DLLEXPORT uint32_t
  post_normalization_dimension(uint32_t orig_dimension) const override;

  POWERLAWANN_DLLEXPORT bool preprocessing_required() const override;

  POWERLAWANN_DLLEXPORT void preprocess_base_points(float* original_data, const size_t orig_dim,
                                                    const size_t num_points) override;

  POWERLAWANN_DLLEXPORT void preprocess_query(const float* query_vec, const size_t query_dim,
                                              float* scratch_query_vector) override;
};

template <typename T>
distance_t<T>* get_distance_function(metric_t m);

} // namespace powerlaw_ann

#endif // COMMON_DISTANCE
