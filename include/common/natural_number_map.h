#ifndef COMMON_NATURAL_NUMBER_MAP
#define COMMON_NATURAL_NUMBER_MAP

#include <boost/dynamic_bitset_fwd.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace powerlaw_ann {
// A map whose key is a natural number (from 0 onwards) and maps to a value.
// Made as both memory and performance efficient map for scenario such as
// DiskANN location-to-tag map. There, the pool of numbers is consecutive from
// zero to some max value, and it's expected that most if not all keys from 0
// up to some current maximum will be present in the map. The memory usage of
// the map is determined by the largest inserted key since it uses vector as a
// backing store and bitset for presence indication.
//
// Thread-safety: this class is not thread-safe in general.
// Exception: multiple read-only operations are safe on the object only if
// there are no writers to it in parallel.
template <typename key_t, typename value_t>
class natural_number_map_t {
public:
  static_assert(std::is_trivial<key_t>::value, "key_t must be a trivial type");

  // Represents a reference to a element in the map. Used while iterating
  // over map entries.
  struct position_t {
    size_t key_;
    // The number of keys that were enumerated when iterating through the
    // map so far. Used to early-terminate enumeration when there are no
    // more entries in the map.
    size_t keys_already_enumerated_;

    // Returns whether it is valid to access the element at this position in
    // the map.
    bool is_valid() const;
  };

  natural_number_map_t();
  ~natural_number_map_t();

  void reserve(size_t count);
  size_t size() const;

  void set(key_t key, value_t value);
  void erase(key_t key);

  bool contains(key_t key) const;
  bool try_get(key_t key, value_t& value) const;

  // Returns the value at the specified position. Prerequisite: the position is
  // valid.
  value_t get(const position_t& pos) const;

  // Finds the first element in the map, if any. Invalidated by changes in the
  // map.
  position_t find_first() const;

  // Finds the next element in the map after the specified position.
  // Invalidated by changes in the map.
  position_t find_next(const position_t& after_position) const;

  void clear();

private:
  // Number of entries in the map. Not the same as size() of the
  // values_vector_ below.
  size_t size_;

  // Array of values. The key is the index of the value.
  std::vector<value_t> values_vector_;

  // Values that are in the set have the corresponding bit index set
  // to 1.
  //
  // Use a pointer here to allow for forward declaration of dynamic_bitset
  // in public headers to avoid making boost a dependency for clients
  // of DiskANN.
  std::unique_ptr<boost::dynamic_bitset<>> values_bitset_;
};
} // namespace powerlaw_ann

#endif // COMMON_NATURAL_NUMBER_MAP
