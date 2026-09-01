#ifndef INDEX_ABSTRACT_GRAPH_STORE
#define INDEX_ABSTRACT_GRAPH_STORE

#include "common/types.h"

#include <string>
#include <vector>

namespace powerlaw_ann {

class abstract_graph_store_t {
public:
  abstract_graph_store_t(const size_t total_pts, const size_t reserve_graph_degree)
      : capacity_(total_pts), reserve_graph_degree_(reserve_graph_degree) {}

  virtual ~abstract_graph_store_t() = default;

  // returns tuple of <nodes_read, start, num_frozen_points>
  virtual std::tuple<uint32_t, uint32_t, size_t> load(const std::string& index_path_prefix,
                                                      const size_t num_points) = 0;
  virtual int store(const std::string& index_path_prefix, const size_t num_points,
                    const size_t num_fz_points, const uint32_t start) = 0;

  // not synchronised, user should use lock when necvessary.
  virtual const std::vector<location_t>& get_neighbours(const location_t i) const = 0;
  virtual void add_neighbour(const location_t i, location_t neighbour_id) = 0;
  virtual void clear_neighbours(const location_t i) = 0;
  virtual void swap_neighbours(const location_t a, location_t b) = 0;

  virtual void set_neighbours(const location_t i, std::vector<location_t>& neighbours) = 0;

  virtual size_t resize_graph(const size_t new_size) = 0;
  virtual void clear_graph() = 0;

  virtual uint32_t get_max_observed_degree() = 0;

  // set during load
  virtual size_t get_max_range_of_graph() = 0;

  // Total internal points max_points_ + num_frozen_points_
  size_t get_total_points() { return capacity_; }

protected:
  // Internal function, changes total points when resize_graph is called.
  void set_total_points(size_t new_capacity) { capacity_ = new_capacity; }

  size_t get_reserve_graph_degree() { return reserve_graph_degree_; }

private:
  size_t capacity_;
  size_t reserve_graph_degree_;
};

} // namespace powerlaw_ann

#endif // INDEX_ABSTRACT_GRAPH_STORE
