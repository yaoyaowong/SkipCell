#include "index/abstract_data_store.h"

namespace powerlaw_ann {

template <typename data_t>
abstract_data_store_t<data_t>::abstract_data_store_t(const location_t capacity, const size_t dim)
    : capacity_(capacity), dim_(dim) {}

template <typename data_t>
location_t abstract_data_store_t<data_t>::capacity() const {
  return capacity_;
}

template <typename data_t>
size_t abstract_data_store_t<data_t>::get_dims() const {
  return dim_;
}

template <typename data_t>
location_t abstract_data_store_t<data_t>::resize(const location_t new_num_points) {
  if (new_num_points > capacity_) {
    return expand(new_num_points);
  } else if (new_num_points < capacity_) {
    return shrink(new_num_points);
  } else {
    return capacity_;
  }
}

template POWERLAWANN_DLLEXPORT class abstract_data_store_t<float>;
template POWERLAWANN_DLLEXPORT class abstract_data_store_t<int8_t>;
template POWERLAWANN_DLLEXPORT class abstract_data_store_t<uint8_t>;
} // namespace powerlaw_ann
