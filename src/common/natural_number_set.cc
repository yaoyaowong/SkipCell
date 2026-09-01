#include "common/natural_number_set.h"

#include "common/diskann_exception.h"

#include <boost/dynamic_bitset.hpp>

namespace powerlaw_ann {
template <typename T>
natural_number_set_t<T>::natural_number_set_t()
    : values_bitset_(std::make_unique<boost::dynamic_bitset<>>()) {}

template <typename T>
bool natural_number_set_t<T>::is_empty() const {
  return values_vector_.empty();
}

template <typename T>
void natural_number_set_t<T>::reserve(size_t count) {
  values_vector_.reserve(count);
  values_bitset_->reserve(count);
}

template <typename T>
void natural_number_set_t<T>::insert(T id) {
  values_vector_.emplace_back(id);

  if (id >= values_bitset_->size())
    values_bitset_->resize(static_cast<size_t>(id) + 1);

  values_bitset_->set(id, true);
}

template <typename T>
T natural_number_set_t<T>::pop_any() {
  if (values_vector_.empty()) {
    throw powerlaw_ann::diskann_exception_t("No values available", -1, __FUNCSIG__, __FILE__,
                                            __LINE__);
  }

  const T id = values_vector_.back();
  values_vector_.pop_back();

  values_bitset_->set(id, false);

  return id;
}

template <typename T>
void natural_number_set_t<T>::clear() {
  values_vector_.clear();
  values_bitset_->clear();
}

template <typename T>
size_t natural_number_set_t<T>::size() const {
  return values_vector_.size();
}

template <typename T>
bool natural_number_set_t<T>::is_in_set(T id) const {
  return values_bitset_->test(id);
}

// Instantiate used templates.
template class natural_number_set_t<unsigned>;
} // namespace powerlaw_ann
