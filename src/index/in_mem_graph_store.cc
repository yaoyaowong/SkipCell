#include "index/in_mem_graph_store.h"

#include "common/utils.h"

namespace powerlaw_ann {
in_mem_graph_store_t::in_mem_graph_store_t(const size_t total_pts,
                                           const size_t reserve_graph_degree)
    : abstract_graph_store_t(total_pts, reserve_graph_degree) {
  this->resize_graph(total_pts);
  for (size_t i = 0; i < total_pts; i++) {
    graph_[i].reserve(reserve_graph_degree);
  }
}

std::tuple<uint32_t, uint32_t, size_t>
in_mem_graph_store_t::load(const std::string& index_path_prefix, const size_t num_points) {
  return load_impl(index_path_prefix, num_points);
}
int in_mem_graph_store_t::store(const std::string& index_path_prefix, const size_t num_points,
                                const size_t num_frozen_points, const uint32_t start) {
  return save_graph(index_path_prefix, num_points, num_frozen_points, start);
}
const std::vector<location_t>& in_mem_graph_store_t::get_neighbours(const location_t i) const {
  return graph_.at(i);
}

void in_mem_graph_store_t::add_neighbour(const location_t i, location_t neighbour_id) {
  graph_[i].emplace_back(neighbour_id);
  if (max_observed_degree_ < graph_[i].size()) {
    max_observed_degree_ = (uint32_t) (graph_[i].size());
  }
}

void in_mem_graph_store_t::clear_neighbours(const location_t i) { graph_[i].clear(); };
void in_mem_graph_store_t::swap_neighbours(const location_t a, location_t b) {
  graph_[a].swap(graph_[b]);
};

void in_mem_graph_store_t::set_neighbours(const location_t i, std::vector<location_t>& neighbours) {
  graph_[i].assign(neighbours.begin(), neighbours.end());
  if (max_observed_degree_ < neighbours.size()) {
    max_observed_degree_ = (uint32_t) (neighbours.size());
  }
}

size_t in_mem_graph_store_t::resize_graph(const size_t new_size) {
  graph_.resize(new_size);
  set_total_points(new_size);
  return graph_.size();
}

void in_mem_graph_store_t::clear_graph() { graph_.clear(); }

#ifdef EXEC_ENV_OLS
std::tuple<uint32_t, uint32_t, size_t>
in_mem_graph_store_t::load_impl(aligned_file_reader_t& reader, size_t expected_num_points) {
  size_t expected_file_size;
  size_t file_frozen_pts;
  uint32_t start;

  auto max_points = get_max_points();
  int header_size = 2 * sizeof(size_t) + 2 * sizeof(uint32_t);
  std::unique_ptr<char[]> header = std::make_unique<char[]>(header_size);
  read_array(reader, header.get(), header_size);

  expected_file_size = *((size_t*) header.get());
  max_observed_degree_ = *((uint32_t*) (header.get() + sizeof(size_t)));
  start = *((uint32_t*) (header.get() + sizeof(size_t) + sizeof(uint32_t)));
  file_frozen_pts =
      *((size_t*) (header.get() + sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t)));

  powerlaw_ann::cout << "From graph header, expected_file_size: " << expected_file_size
                     << ", max_observed_degree_: " << max_observed_degree_ << ", start_: " << start
                     << ", file_frozen_pts: " << file_frozen_pts << std::endl;

  powerlaw_ann::cout << "Loading vamana graph from reader..." << std::flush;

  // If user provides more points than max_points
  // resize the graph_ to the larger size.
  if (get_total_points() < expected_num_points) {
    powerlaw_ann::cout << "resizing graph to " << expected_num_points << std::endl;
    this->resize_graph(expected_num_points);
  }

  uint32_t nodes_read = 0;
  size_t cc = 0;
  size_t graph_offset = header_size;
  while (nodes_read < expected_num_points) {
    uint32_t k;
    read_value(reader, k, graph_offset);
    graph_offset += sizeof(uint32_t);
    std::vector<uint32_t> tmp(k);
    tmp.reserve(k);
    read_array(reader, tmp.data(), k, graph_offset);
    graph_offset += k * sizeof(uint32_t);
    cc += k;
    graph_[nodes_read].swap(tmp);
    nodes_read++;
    if (nodes_read % 1000000 == 0) {
      powerlaw_ann::cout << "." << std::flush;
    }
    if (k > max_range_of_graph_) {
      max_range_of_graph_ = k;
    }
  }

  powerlaw_ann::cout << "done. vamana_index_t has " << nodes_read << " nodes and " << cc
                     << " out-edges, start_ is set to " << start << std::endl;
  return std::make_tuple(nodes_read, start, file_frozen_pts);
}
#endif

std::tuple<uint32_t, uint32_t, size_t> in_mem_graph_store_t::load_impl(const std::string& filename,
                                                                       size_t expected_num_points) {
  size_t expected_file_size;
  size_t file_frozen_pts;
  uint32_t start;
  size_t file_offset = 0; // will need this for single file format support

  std::ifstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);
  in.open(filename, std::ios::binary);
  in.seekg(file_offset, in.beg);
  in.read((char*) &expected_file_size, sizeof(size_t));
  in.read((char*) &max_observed_degree_, sizeof(uint32_t));
  in.read((char*) &start, sizeof(uint32_t));
  in.read((char*) &file_frozen_pts, sizeof(size_t));
  size_t vamana_metadata_size =
      sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

  powerlaw_ann::cout << "From graph header, expected_file_size: " << expected_file_size
                     << ", max_observed_degree_: " << max_observed_degree_ << ", start_: " << start
                     << ", file_frozen_pts: " << file_frozen_pts << std::endl;

  powerlaw_ann::cout << "Loading vamana graph " << filename << "..." << std::flush;

  // If user provides more points than max_points
  // resize the graph_ to the larger size.
  if (get_total_points() < expected_num_points) {
    powerlaw_ann::cout << "resizing graph to " << expected_num_points << std::endl;
    this->resize_graph(expected_num_points);
  }

  size_t bytes_read = vamana_metadata_size;
  size_t cc = 0;
  uint32_t nodes_read = 0;
  while (bytes_read != expected_file_size) {
    uint32_t k;
    in.read((char*) &k, sizeof(uint32_t));

    if (k == 0) {
      powerlaw_ann::cerr << "ERROR: Point found with no out-neighbours, point#" << nodes_read
                         << std::endl;
    }

    cc += k;
    ++nodes_read;
    std::vector<uint32_t> tmp(k);
    tmp.reserve(k);
    in.read((char*) tmp.data(), k * sizeof(uint32_t));
    graph_[nodes_read - 1].swap(tmp);
    bytes_read += sizeof(uint32_t) * ((size_t) k + 1);
    if (nodes_read % 10000000 == 0)
      powerlaw_ann::cout << "." << std::flush;
    if (k > max_range_of_graph_) {
      max_range_of_graph_ = k;
    }
  }

  powerlaw_ann::cout << "done. vamana_index_t has " << nodes_read << " nodes and " << cc
                     << " out-edges, start_ is set to " << start << std::endl;
  return std::make_tuple(nodes_read, start, file_frozen_pts);
}

int in_mem_graph_store_t::save_graph(const std::string& index_path_prefix, const size_t num_points,
                                     const size_t num_frozen_points, const uint32_t start) {
  std::ofstream out;
  open_file_to_write(out, index_path_prefix);

  size_t file_offset = 0;
  out.seekp(file_offset, out.beg);
  size_t index_size = 24;
  uint32_t max_degree = 0;
  out.write((char*) &index_size, sizeof(uint64_t));
  out.write((char*) &max_observed_degree_, sizeof(uint32_t));
  uint32_t ep_u32 = start;
  out.write((char*) &ep_u32, sizeof(uint32_t));
  out.write((char*) &num_frozen_points, sizeof(size_t));

  // Note: num_points = nd_ + num_frozen_points_
  for (uint32_t i = 0; i < num_points; i++) {
    uint32_t graph_k = (uint32_t) graph_[i].size();
    out.write((char*) &graph_k, sizeof(uint32_t));
    out.write((char*) graph_[i].data(), graph_k * sizeof(uint32_t));
    max_degree = graph_[i].size() > max_degree ? (uint32_t) graph_[i].size() : max_degree;
    index_size += (size_t) (sizeof(uint32_t) * (graph_k + 1));
  }
  out.seekp(file_offset, out.beg);
  out.write((char*) &index_size, sizeof(uint64_t));
  out.write((char*) &max_degree, sizeof(uint32_t));
  out.close();
  return (int) index_size;
}

size_t in_mem_graph_store_t::get_max_range_of_graph() { return max_range_of_graph_; }

uint32_t in_mem_graph_store_t::get_max_observed_degree() { return max_observed_degree_; }

} // namespace powerlaw_ann
