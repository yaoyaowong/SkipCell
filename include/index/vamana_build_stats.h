#ifndef INDEX_VAMANA_BUILD_STATS
#define INDEX_VAMANA_BUILD_STATS

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace powerlaw_ann {

class vamana_build_stats_t {
public:
  void reset(size_t node_count) {
    full_neighbors_.clear();
    full_neighbors_.resize(node_count);
    prune_counts_.assign(node_count, 0);
    locks_.clear();
    locks_.reserve(node_count);
    for (size_t i = 0; i < node_count; ++i) {
      locks_.emplace_back(std::make_unique<std::mutex>());
    }
  }

  bool active() const { return !full_neighbors_.empty(); }

  void record_candidate_pool(uint32_t node_id, const std::vector<uint32_t>& candidate_ids) {
    if (node_id >= full_neighbors_.size()) {
      return;
    }
    std::lock_guard<std::mutex> guard(*locks_[node_id]);
    for (uint32_t candidate_id : candidate_ids) {
      add_unique_neighbor(node_id, candidate_id);
    }
  }

  void record_reverse_insert(uint32_t target_id, uint32_t source_id) {
    if (target_id >= full_neighbors_.size()) {
      return;
    }
    std::lock_guard<std::mutex> guard(*locks_[target_id]);
    add_unique_neighbor(target_id, source_id);
  }

  void record_reverse_prune(uint32_t target_id) {
    if (target_id >= prune_counts_.size()) {
      return;
    }
    std::lock_guard<std::mutex> guard(*locks_[target_id]);
    ++prune_counts_[target_id];
  }

  std::vector<uint32_t> full_degrees() const {
    std::vector<uint32_t> degrees;
    degrees.reserve(full_neighbors_.size());
    for (const auto& neighbors : full_neighbors_) {
      degrees.push_back((uint32_t) neighbors.size());
    }
    return degrees;
  }

  const std::vector<uint64_t>& prune_counts() const { return prune_counts_; }

  void dump(const std::string& prefix, const std::vector<uint32_t>& final_degrees, uint32_t start,
            size_t num_frozen_points) const {
    if (final_degrees.size() > full_neighbors_.size()) {
      throw std::runtime_error("final degree count exceeds Vamana build stats node count");
    }

    dump_degree_csv(prefix + "degree_.csv", "degree", final_degrees);
    dump_prune_csv(prefix + "prune_counts_.csv", final_degrees.size());
    dump_full_degree_csv(prefix + "full_degree_.csv", final_degrees.size());
    dump_full_graph(prefix + "full_vamana_.index", final_degrees.size(), start, num_frozen_points);
    dump_summary(prefix + "summary_.json", final_degrees);
  }

private:
  void add_unique_neighbor(uint32_t node_id, uint32_t neighbor_id) {
    if (neighbor_id >= full_neighbors_.size() || neighbor_id == node_id) {
      return;
    }
    auto& neighbors = full_neighbors_[node_id];
    if (std::find(neighbors.begin(), neighbors.end(), neighbor_id) == neighbors.end()) {
      neighbors.push_back(neighbor_id);
    }
  }

  static void open_output(std::ofstream& output, const std::string& path,
                          std::ios_base::openmode mode = std::ios::out) {
    output.open(path, mode);
    if (!output.good()) {
      throw std::runtime_error("could not open Vamana build stats output: " + path);
    }
  }

  void dump_degree_csv(const std::string& path, const std::string& column_name,
                       const std::vector<uint32_t>& degrees) const {
    std::ofstream output;
    open_output(output, path);
    output << "node_id," << column_name << "\n";
    for (uint32_t node_id = 0; node_id < degrees.size(); ++node_id) {
      output << node_id << "," << degrees[node_id] << "\n";
    }
  }

  void dump_prune_csv(const std::string& path, size_t node_count) const {
    std::ofstream output;
    open_output(output, path);
    output << "node_id,prune_count\n";
    for (uint32_t node_id = 0; node_id < node_count; ++node_id) {
      output << node_id << "," << prune_counts_[node_id] << "\n";
    }
  }

  void dump_full_degree_csv(const std::string& path, size_t node_count) const {
    std::ofstream output;
    open_output(output, path);
    output << "node_id,full_degree\n";
    for (uint32_t node_id = 0; node_id < node_count; ++node_id) {
      output << node_id << "," << full_neighbors_[node_id].size() << "\n";
    }
  }

  void dump_full_graph(const std::string& path, size_t node_count, uint32_t start,
                       size_t num_frozen_points) const {
    std::ofstream output;
    open_output(output, path, std::ios::out | std::ios::binary);

    uint64_t index_size = 24;
    uint32_t max_degree = 0;
    for (size_t node_id = 0; node_id < node_count; ++node_id) {
      const auto degree = (uint32_t) full_neighbors_[node_id].size();
      max_degree = std::max(max_degree, degree);
      index_size += sizeof(uint32_t) * (uint64_t) (degree + 1);
    }

    output.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
    output.write(reinterpret_cast<const char*>(&max_degree), sizeof(max_degree));
    output.write(reinterpret_cast<const char*>(&start), sizeof(start));
    output.write(reinterpret_cast<const char*>(&num_frozen_points), sizeof(num_frozen_points));

    for (size_t node_id = 0; node_id < node_count; ++node_id) {
      const auto degree = (uint32_t) full_neighbors_[node_id].size();
      output.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
      output.write(reinterpret_cast<const char*>(full_neighbors_[node_id].data()),
                   degree * sizeof(uint32_t));
    }
  }

  void dump_summary(const std::string& path, const std::vector<uint32_t>& final_degrees) const {
    std::vector<uint32_t> full_degrees_snapshot;
    full_degrees_snapshot.reserve(final_degrees.size());
    for (size_t node_id = 0; node_id < final_degrees.size(); ++node_id) {
      full_degrees_snapshot.push_back((uint32_t) full_neighbors_[node_id].size());
    }
    const uint64_t total_edges =
        std::accumulate(final_degrees.begin(), final_degrees.end(), uint64_t{0});
    const uint64_t total_full_edges =
        std::accumulate(full_degrees_snapshot.begin(), full_degrees_snapshot.end(), uint64_t{0});
    const uint64_t total_prunes = std::accumulate(
        prune_counts_.begin(), prune_counts_.begin() + final_degrees.size(), uint64_t{0});
    const uint32_t max_degree =
        final_degrees.empty() ? 0 : *std::max_element(final_degrees.begin(), final_degrees.end());
    const uint32_t max_full_degree =
        full_degrees_snapshot.empty()
            ? 0
            : *std::max_element(full_degrees_snapshot.begin(), full_degrees_snapshot.end());
    const uint64_t max_prune_count =
        final_degrees.empty() ? 0
                              : *std::max_element(prune_counts_.begin(),
                                                  prune_counts_.begin() + final_degrees.size());

    std::ofstream output;
    open_output(output, path);
    output << "{\n";
    output << "  \"num_nodes\": " << final_degrees.size() << ",\n";
    output << "  \"total_edges\": " << total_edges << ",\n";
    output << "  \"total_full_edges\": " << total_full_edges << ",\n";
    output << "  \"total_reverse_prunes\": " << total_prunes << ",\n";
    output << "  \"max_degree\": " << max_degree << ",\n";
    output << "  \"max_full_degree\": " << max_full_degree << ",\n";
    output << "  \"max_prune_count\": " << max_prune_count << ",\n";
    output << "  \"full_edge_expansion_ratio\": "
           << (total_edges == 0 ? 0.0 : (double) total_full_edges / (double) total_edges) << "\n";
    output << "}\n";
  }

  std::vector<std::vector<uint32_t>> full_neighbors_;
  std::vector<uint64_t> prune_counts_;
  std::vector<std::unique_ptr<std::mutex>> locks_;
};

} // namespace powerlaw_ann

#endif // INDEX_VAMANA_BUILD_STATS
