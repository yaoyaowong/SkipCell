#ifndef INDEX_NEIGHBOR
#define INDEX_NEIGHBOR

#include <cstddef>
#include <cstring>
#include <vector>

namespace powerlaw_ann {

struct neighbor_t {
  unsigned id;
  float distance;
  bool expanded;

  neighbor_t() = default;

  neighbor_t(unsigned id, float distance) : id{id}, distance{distance}, expanded(false) {}

  inline bool operator<(const neighbor_t& other) const {
    return distance < other.distance || (distance == other.distance && id < other.id);
  }

  inline bool operator==(const neighbor_t& other) const { return (id == other.id); }
};

// Invariant: after every `insert` and `closest_unexpanded()`, `cur_` points to
//            the first neighbor_t which is unexpanded.
class neighbor_priority_queue_t {
public:
  neighbor_priority_queue_t() : size_(0), capacity_(0), cur_(0) {}

  explicit neighbor_priority_queue_t(size_t capacity)
      : size_(0), capacity_(capacity), cur_(0), data_(capacity + 1) {}

  // Inserts the item ordered into the set up to the sets capacity.
  // The item will be dropped if it is the same id as an exiting
  // set item or it has a greated distance than the final
  // item in the set. The set cursor that is used to pop() the
  // next item will be set to the lowest index of an uncheck item
  void insert(const neighbor_t& nbr) {
    if (size_ == capacity_ && data_[size_ - 1] < nbr) {
      return;
    }

    size_t lo = 0, hi = size_;
    while (lo < hi) {
      size_t mid = (lo + hi) >> 1;
      if (nbr < data_[mid]) {
        hi = mid;
        // Make sure the same id isn't inserted into the set
      } else if (data_[mid].id == nbr.id) {
        return;
      } else {
        lo = mid + 1;
      }
    }

    if (lo < capacity_) {
      std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(neighbor_t));
    }
    data_[lo] = {nbr.id, nbr.distance};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
  }

  // Replaces an approximate score for an existing ID and makes it eligible for expansion again.
  // If the ID is absent, this has the same bounded behavior as insert().
  void insert_or_update(const neighbor_t& nbr) {
    for (size_t position = 0; position < size_; ++position) {
      if (data_[position].id != nbr.id) {
        continue;
      }
      if (position + 1 < size_) {
        std::memmove(&data_[position], &data_[position + 1],
                     (size_ - position - 1) * sizeof(neighbor_t));
      }
      --size_;
      if (position < cur_) {
        --cur_;
      }
      break;
    }
    insert(nbr);
  }

  neighbor_t closest_unexpanded() {
    data_[cur_].expanded = true;
    size_t pre = cur_;
    while (cur_ < size_ && data_[cur_].expanded) {
      cur_++;
    }
    return data_[pre];
  }

  bool has_unexpanded_node() const { return cur_ < size_; }

  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

  void reserve(size_t capacity) {
    if (capacity + 1 > data_.size()) {
      data_.resize(capacity + 1);
    }
    capacity_ = capacity;
  }

  neighbor_t& operator[](size_t i) { return data_[i]; }

  neighbor_t operator[](size_t i) const { return data_[i]; }

  void clear() {
    size_ = 0;
    cur_ = 0;
  }

private:
  size_t size_, capacity_, cur_;
  std::vector<neighbor_t> data_;
};

} // namespace powerlaw_ann

#endif // INDEX_NEIGHBOR
