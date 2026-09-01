#ifndef COMMON_NATURAL_NUMBER_SET
#define COMMON_NATURAL_NUMBER_SET

#include <boost/dynamic_bitset_fwd.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace powerlaw_ann {
// A set of natural numbers (from 0 onwards). Made for scenario where the
// pool of numbers is consecutive from zero to some max value and very
// efficient methods for "add to set", "get any value from set", "is in set"
// are needed. The memory usage of the set is determined by the largest
// number of inserted entries (uses a vector as a backing store) as well as
// the largest value to be placed in it (uses bitset as well).
//
// Thread-safety: this class is not thread-safe in general.
// Exception: multiple read-only operations (e.g. is_in_set, empty, size) are
// safe on the object only if there are no writers to it in parallel.
template <typename T>
class natural_number_set_t {
public:
  static_assert(std::is_trivial<T>::value, "Identifier must be a trivial type");

  natural_number_set_t();

  bool is_empty() const;
  void reserve(size_t count);
  void insert(T id);
  T pop_any();
  void clear();
  size_t size() const;
  bool is_in_set(T id) const;

private:
  // Values that are currently in set.
  std::vector<T> values_vector_;

  // Values that are in the set have the corresponding bit index set
  // to 1.
  //
  // Use a pointer here to allow for forward declaration of dynamic_bitset
  // in public headers to avoid making boost a dependency for clients
  // of DiskANN.
  std::unique_ptr<boost::dynamic_bitset<>> values_bitset_;
};
} // namespace powerlaw_ann

#endif // COMMON_NATURAL_NUMBER_SET
