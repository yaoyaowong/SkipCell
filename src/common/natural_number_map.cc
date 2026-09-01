#include "common/natural_number_map.h"

#include "common/tag_uint128.h"

#include <assert.h>
#include <boost/dynamic_bitset.hpp>

namespace powerlaw_ann {
static constexpr auto invalid_position = boost::dynamic_bitset<>::npos;

template <typename key_t, typename value_t>
natural_number_map_t<key_t, value_t>::natural_number_map_t()
    : size_(0), values_bitset_(std::make_unique<boost::dynamic_bitset<>>()) {}

template <typename key_t, typename value_t>
natural_number_map_t<key_t, value_t>::~natural_number_map_t() = default;

template <typename key_t, typename value_t>
void natural_number_map_t<key_t, value_t>::reserve(size_t count) {
  values_vector_.reserve(count);
  values_bitset_->reserve(count);
}

template <typename key_t, typename value_t>
size_t natural_number_map_t<key_t, value_t>::size() const {
  return size_;
}

template <typename key_t, typename value_t>
void natural_number_map_t<key_t, value_t>::set(key_t key, value_t value) {
  if (key >= values_bitset_->size()) {
    values_bitset_->resize(static_cast<size_t>(key) + 1);
    values_vector_.resize(values_bitset_->size());
  }

  values_vector_[key] = value;
  const bool was_present = values_bitset_->test_set(key, true);

  if (!was_present) {
    ++size_;
  }
}

template <typename key_t, typename value_t>
void natural_number_map_t<key_t, value_t>::erase(key_t key) {
  if (key < values_bitset_->size()) {
    const bool was_present = values_bitset_->test_set(key, false);

    if (was_present) {
      --size_;
    }
  }
}

template <typename key_t, typename value_t>
bool natural_number_map_t<key_t, value_t>::contains(key_t key) const {
  return key < values_bitset_->size() && values_bitset_->test(key);
}

template <typename key_t, typename value_t>
bool natural_number_map_t<key_t, value_t>::try_get(key_t key, value_t& value) const {
  if (!contains(key)) {
    return false;
  }

  value = values_vector_[key];
  return true;
}

template <typename key_t, typename value_t>
typename natural_number_map_t<key_t, value_t>::position_t
natural_number_map_t<key_t, value_t>::find_first() const {
  return position_t{size_ > 0 ? values_bitset_->find_first() : invalid_position, 0};
}

template <typename key_t, typename value_t>
typename natural_number_map_t<key_t, value_t>::position_t
natural_number_map_t<key_t, value_t>::find_next(const position_t& after_position) const {
  return position_t{after_position.keys_already_enumerated_ < size_
                        ? values_bitset_->find_next(after_position.key_)
                        : invalid_position,
                    after_position.keys_already_enumerated_ + 1};
}

template <typename key_t, typename value_t>
bool natural_number_map_t<key_t, value_t>::position_t::is_valid() const {
  return key_ != invalid_position;
}

template <typename key_t, typename value_t>
value_t natural_number_map_t<key_t, value_t>::get(const position_t& pos) const {
  assert(pos.is_valid());
  return values_vector_[pos.key_];
}

template <typename key_t, typename value_t>
void natural_number_map_t<key_t, value_t>::clear() {
  size_ = 0;
  values_vector_.clear();
  values_bitset_->clear();
}

// Instantiate used templates.
template class natural_number_map_t<uint32_t, int32_t>;
template class natural_number_map_t<uint32_t, uint32_t>;
template class natural_number_map_t<uint32_t, int64_t>;
template class natural_number_map_t<uint32_t, uint64_t>;
template class natural_number_map_t<uint32_t, tag_uint128_t>;
} // namespace powerlaw_ann
