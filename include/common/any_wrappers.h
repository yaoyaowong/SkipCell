#ifndef COMMON_ANY_WRAPPER
#define COMMON_ANY_WRAPPER

#include "third/tsl/robin_set.h"

#include <any>
#include <cstddef>
#include <vector>

namespace any_wrapper {

/**
 * @brief Holds a non-owning, type-erased reference.
 *
 * The caller must keep the referenced object alive for the lifetime of this object.
 */
struct any_reference_t {
  template <typename value_t>
  any_reference_t(value_t& reference) : data_(&reference) {}

  template <typename value_t>
  value_t& get() {
    auto ptr = std::any_cast<value_t*>(data_);
    return *ptr;
  }

private:
  std::any data_;
};

struct any_robin_set_t : public any_reference_t {
  template <typename T>
  any_robin_set_t(const tsl::robin_set<T>& robin_set) : any_reference_t(robin_set) {}
  template <typename T>
  any_robin_set_t(tsl::robin_set<T>& robin_set) : any_reference_t(robin_set) {}
};

struct any_vector_t : public any_reference_t {
  template <typename T>
  any_vector_t(const std::vector<T>& vector) : any_reference_t(vector) {}
  template <typename T>
  any_vector_t(std::vector<T>& vector) : any_reference_t(vector) {}
};
} // namespace any_wrapper

#endif // COMMON_ANY_WRAPPER
