#ifndef INDEX_IN_MEM_GRAPH_STORE
#define INDEX_IN_MEM_GRAPH_STORE

#include "index/abstract_graph_store.h"

namespace powerlaw_ann {

class in_mem_graph_store_t : public abstract_graph_store_t {
public:
  in_mem_graph_store_t(const size_t total_pts, const size_t reserve_graph_degree);

  // returns tuple of <nodes_read, start, num_frozen_points>
  virtual std::tuple<uint32_t, uint32_t, size_t> load(const std::string& index_path_prefix,
                                                      const size_t num_points) override;
  virtual int store(const std::string& index_path_prefix, const size_t num_points,
                    const size_t num_frozen_points, const uint32_t start) override;

  virtual const std::vector<location_t>& get_neighbours(const location_t i) const override;
  virtual void add_neighbour(const location_t i, location_t neighbour_id) override;
  virtual void clear_neighbours(const location_t i) override;
  virtual void swap_neighbours(const location_t a, location_t b) override;

  virtual void set_neighbours(const location_t i, std::vector<location_t>& neighbors) override;

  virtual size_t resize_graph(const size_t new_size) override;
  virtual void clear_graph() override;

  virtual size_t get_max_range_of_graph() override;
  virtual uint32_t get_max_observed_degree() override;

protected:
  virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string& filename,
                                                           size_t expected_num_points);
#ifdef EXEC_ENV_OLS
  virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(aligned_file_reader_t& reader,
                                                           size_t expected_num_points);
#endif

  int save_graph(const std::string& index_path_prefix, const size_t active_points,
                 const size_t num_frozen_points, const uint32_t start);

private:
  size_t max_range_of_graph_ = 0;
  uint32_t max_observed_degree_ = 0;

  std::vector<std::vector<uint32_t>> graph_;
};

} // namespace powerlaw_ann

#endif // INDEX_IN_MEM_GRAPH_STORE
