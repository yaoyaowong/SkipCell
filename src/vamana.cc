#include "index/vamana.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr float graph_slack_factor = 1.3f;

struct neighbor_t {
  uint32_t id = 0;
  float dist = 0.0f;
  bool expanded = false;

  neighbor_t() = default;
  neighbor_t(uint32_t id, float dist) : id(id), dist(dist) {}

  bool operator<(const neighbor_t& other) const {
    return dist < other.dist || (dist == other.dist && id < other.id);
  }
};

class neighbor_queue_t {
public:
  void reserve(size_t cap) {
    data_.reserve(cap + 1);
    cap_ = cap;
  }

  void insert(const neighbor_t& neighbor) {
    if (contains(neighbor.id)) {
      return;
    }
    if (data_.size() == cap_ && !(neighbor < data_.back())) {
      return;
    }

    const auto pos = std::lower_bound(data_.begin(), data_.end(), neighbor);
    const size_t insert_pos = static_cast<size_t>(pos - data_.begin());
    data_.insert(pos, neighbor);
    if (data_.size() > cap_) {
      data_.pop_back();
    }
    if (cur_ >= data_.size() || insert_pos < cur_) {
      cur_ = insert_pos;
    }
  }

  neighbor_t closest_unexpanded() {
    data_[cur_].expanded = true;
    const neighbor_t result = data_[cur_];
    while (cur_ < data_.size() && data_[cur_].expanded) {
      ++cur_;
    }
    return result;
  }

  bool has_unexpanded() const { return cur_ < data_.size(); }

  void clear() {
    data_.clear();
    cur_ = 0;
  }

private:
  bool contains(uint32_t id) const {
    return std::any_of(data_.begin(), data_.end(),
                       [id](const neighbor_t& neighbor) { return neighbor.id == id; });
  }

  size_t cap_ = 0;
  size_t cur_ = 0;
  std::vector<neighbor_t> data_;
};

struct scratch_t {
  neighbor_queue_t best;
  std::vector<neighbor_t> pool;
  std::vector<float> occlude_factor;
  std::vector<uint32_t> id_scratch;
  std::vector<float> query;
  std::vector<uint8_t> visited_flags;
  std::vector<uint32_t> visited_list;

  void clear_search() {
    best.clear();
    pool.clear();
    id_scratch.clear();
    for (uint32_t id : visited_list) {
      visited_flags[id] = 0;
    }
    visited_list.clear();
  }
};

class lock_array_t {
public:
  explicit lock_array_t(size_t n) : locks_(n) {
    for (auto& lock : locks_) {
      lock = std::make_unique<std::mutex>();
    }
  }

  std::mutex& operator[](size_t i) { return *locks_[i]; }

private:
  std::vector<std::unique_ptr<std::mutex>> locks_;
};

float l2_squared(const float* a, const float* b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

class vamana_builder_t {
public:
  vamana_builder_t(const float* data, uint32_t n, uint32_t dim, const params_t& params)
      : data_(data), n_(n), dim_(dim), params_(params), adj_(n), locks_(n) {
    if (params_.L < params_.R) {
      params_.L = params_.R;
    }
  }

  graph_t run() {
    entry_point_ = find_medoid();
    std::cout << "Vamana entry point: " << entry_point_ << "\n";

    const int num_threads = resolve_threads();
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    auto scratch = make_scratch(static_cast<size_t>(num_threads) + 4);
    const auto start = std::chrono::steady_clock::now();

    build_links(scratch);
    prune_overfull_nodes(scratch);

    const auto end = std::chrono::steady_clock::now();
    std::cout << "Vamana build time: " << std::chrono::duration<double>(end - start).count()
              << "s\n";

    graph_t graph;
    graph.n = n_;
    graph.dim = dim_;
    graph.entry_point = entry_point_;
    graph.adj = std::move(adj_);
    return graph;
  }

private:
  int resolve_threads() const {
#ifdef _OPENMP
    return params_.num_threads == 0 ? omp_get_max_threads() : static_cast<int>(params_.num_threads);
#else
    return 1;
#endif
  }

  std::vector<scratch_t> make_scratch(size_t count) const {
    std::vector<scratch_t> scratch(count);
    for (auto& item : scratch) {
      item.query.resize(dim_);
      item.best.reserve(params_.L);
      item.pool.reserve(params_.L + params_.R + 16);
      item.id_scratch.reserve(params_.R + 16);
      item.visited_flags.assign(n_, 0);
      item.visited_list.reserve(params_.L + params_.R + 16);
    }
    return scratch;
  }

  uint32_t find_medoid() const {
    std::vector<double> centroid(dim_, 0.0);
    for (uint32_t i = 0; i < n_; ++i) {
      for (uint32_t d = 0; d < dim_; ++d) {
        centroid[d] += data_[static_cast<size_t>(i) * dim_ + d];
      }
    }
    for (double& value : centroid) {
      value /= n_;
    }

    uint32_t best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < n_; ++i) {
      float dist = 0.0f;
      for (uint32_t d = 0; d < dim_; ++d) {
        const float diff =
            data_[static_cast<size_t>(i) * dim_ + d] - static_cast<float>(centroid[d]);
        dist += diff * diff;
      }
      if (dist < best_dist) {
        best_dist = dist;
        best = i;
      }
    }
    return best;
  }

  void build_links(std::vector<scratch_t>& scratch) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 2048)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n_); ++i) {
      const uint32_t node = static_cast<uint32_t>(i);
      scratch_t& item = thread_scratch(scratch);
      item.clear_search();

      const float* vector = data_ + static_cast<size_t>(node) * dim_;
      std::copy(vector, vector + dim_, item.query.data());

      std::vector<uint32_t> pruned;
      search_and_prune(node, item, pruned);

      {
        std::lock_guard<std::mutex> guard(locks_[node]);
        adj_[node] = pruned;
      }

      inter_insert(node, pruned, item);
    }
  }

  void prune_overfull_nodes(std::vector<scratch_t>& scratch) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 2048)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n_); ++i) {
      const uint32_t node = static_cast<uint32_t>(i);
      if (adj_[node].size() <= params_.R) {
        continue;
      }

      scratch_t& item = thread_scratch(scratch);
      std::vector<neighbor_t> pool = make_pool(node, adj_[node]);
      std::vector<uint32_t> pruned;
      prune(node, pool, item, pruned);

      std::lock_guard<std::mutex> guard(locks_[node]);
      adj_[node] = std::move(pruned);
    }
  }

  scratch_t& thread_scratch(std::vector<scratch_t>& scratch) const {
#ifdef _OPENMP
    return scratch[static_cast<size_t>(omp_get_thread_num())];
#else
    return scratch[0];
#endif
  }

  void search_and_prune(uint32_t node, scratch_t& scratch, std::vector<uint32_t>& pruned) {
    beam_search(scratch);

    auto& pool = scratch.pool;
    pool.erase(std::remove_if(pool.begin(), pool.end(),
                              [node](const neighbor_t& neighbor) { return neighbor.id == node; }),
               pool.end());

    prune(node, pool, scratch, pruned);
  }

  void beam_search(scratch_t& scratch) {
    auto mark_visited = [&scratch](uint32_t id) {
      if (scratch.visited_flags[id]) {
        return false;
      }
      scratch.visited_flags[id] = 1;
      scratch.visited_list.push_back(id);
      return true;
    };

    if (mark_visited(entry_point_)) {
      const float dist =
          l2_squared(scratch.query.data(), data_ + static_cast<size_t>(entry_point_) * dim_, dim_);
      scratch.best.insert(neighbor_t(entry_point_, dist));
    }

    while (scratch.best.has_unexpanded()) {
      const neighbor_t current = scratch.best.closest_unexpanded();
      scratch.pool.emplace_back(current);

      {
        std::lock_guard<std::mutex> guard(locks_[current.id]);
        scratch.id_scratch = adj_[current.id];
      }

      for (uint32_t neighbor : scratch.id_scratch) {
        if (mark_visited(neighbor)) {
          const float dist =
              l2_squared(scratch.query.data(), data_ + static_cast<size_t>(neighbor) * dim_, dim_);
          scratch.best.insert(neighbor_t(neighbor, dist));
        }
      }
    }
  }

  void prune(uint32_t node, std::vector<neighbor_t>& pool, scratch_t& scratch,
             std::vector<uint32_t>& result) {
    if (pool.empty()) {
      result.clear();
      return;
    }

    std::sort(pool.begin(), pool.end());
    result.clear();
    result.reserve(params_.R);
    occlude_list(node, pool, scratch, result);
  }

  void occlude_list(uint32_t node, std::vector<neighbor_t>& pool, scratch_t& scratch,
                    std::vector<uint32_t>& result) {
    assert(std::is_sorted(pool.begin(), pool.end()));
    if (pool.size() > params_.C) {
      pool.resize(params_.C);
    }

    auto& occlude_factor = scratch.occlude_factor;
    occlude_factor.assign(pool.size(), 0.0f);

    float cur_alpha = 1.0f;
    while (cur_alpha <= params_.alpha && result.size() < params_.R) {
      for (size_t i = 0; i < pool.size() && result.size() < params_.R; ++i) {
        if (occlude_factor[i] > cur_alpha) {
          continue;
        }

        occlude_factor[i] = std::numeric_limits<float>::max();
        if (pool[i].id != node) {
          result.push_back(pool[i].id);
        }

        // Update occlusion using the already selected candidate.
        for (size_t j = i + 1; j < pool.size(); ++j) {
          if (occlude_factor[j] > params_.alpha) {
            continue;
          }
          const float dist_between =
              l2_squared(data_ + static_cast<size_t>(pool[j].id) * dim_,
                         data_ + static_cast<size_t>(pool[i].id) * dim_, dim_);
          occlude_factor[j] = dist_between == 0.0f
                                  ? std::numeric_limits<float>::max()
                                  : std::max(occlude_factor[j], pool[j].dist / dist_between);
        }
      }
      cur_alpha *= 1.2f;
    }
  }

  void inter_insert(uint32_t node, const std::vector<uint32_t>& pruned, scratch_t& scratch) {
    const size_t slack = static_cast<size_t>(graph_slack_factor * params_.R);

    for (uint32_t neighbor : pruned) {
      bool prune_needed = false;
      std::vector<uint32_t> copied_neighbors;

      {
        std::lock_guard<std::mutex> guard(locks_[neighbor]);
        auto& neighbors = adj_[neighbor];
        if (std::find(neighbors.begin(), neighbors.end(), node) == neighbors.end()) {
          if (neighbors.size() < slack) {
            neighbors.push_back(node);
          } else {
            copied_neighbors = neighbors;
            copied_neighbors.push_back(node);
            prune_needed = true;
          }
        }
      }

      if (prune_needed) {
        std::vector<neighbor_t> pool = make_pool(neighbor, copied_neighbors);
        std::vector<uint32_t> new_neighbors;
        prune(neighbor, pool, scratch, new_neighbors);

        std::lock_guard<std::mutex> guard(locks_[neighbor]);
        adj_[neighbor] = std::move(new_neighbors);
      }
    }
  }

  std::vector<neighbor_t> make_pool(uint32_t node, const std::vector<uint32_t>& neighbors) const {
    std::unordered_set<uint32_t> seen;
    std::vector<neighbor_t> pool;
    pool.reserve(neighbors.size());

    for (uint32_t neighbor : neighbors) {
      if (neighbor == node || !seen.insert(neighbor).second) {
        continue;
      }
      const float dist = l2_squared(data_ + static_cast<size_t>(node) * dim_,
                                    data_ + static_cast<size_t>(neighbor) * dim_, dim_);
      pool.emplace_back(neighbor, dist);
    }

    return pool;
  }

  const float* data_;
  uint32_t n_;
  uint32_t dim_;
  params_t params_;
  std::vector<std::vector<uint32_t>> adj_;
  lock_array_t locks_;
  uint32_t entry_point_ = 0;
};

void write_u32(std::ofstream& out, uint32_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

uint32_t read_u32(std::ifstream& in) {
  uint32_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

} // namespace

// Build a Vamana graph from row-major vectors.
graph_t build(const float* data, uint32_t n, uint32_t dim, const params_t& params) {
  if (data == nullptr || n == 0 || dim == 0) {
    throw std::invalid_argument("build: data, n, and dim must be non-empty");
  }
  if (params.R == 0 || params.L == 0 || params.C == 0 || params.alpha < 1.0f) {
    throw std::invalid_argument("build: invalid Vamana parameters");
  }

  std::cout << "Vamana build: n=" << n << " dim=" << dim << " R=" << params.R << " L=" << params.L
            << " alpha=" << params.alpha << "\n";

  vamana_builder_t builder(data, n, dim, params);
  return builder.run();
}

// Save graph adjacency to a binary file.
void save_graph(const graph_t& graph, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("save_graph: cannot open " + path);
  }

  write_u32(out, graph.n);
  write_u32(out, graph.dim);
  write_u32(out, graph.entry_point);
  for (uint32_t i = 0; i < graph.n; ++i) {
    const uint32_t degree = static_cast<uint32_t>(graph.adj[i].size());
    write_u32(out, degree);
    out.write(reinterpret_cast<const char*>(graph.adj[i].data()), degree * sizeof(uint32_t));
  }
}

// Load graph adjacency from a binary file.
graph_t load_graph(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("load_graph: cannot open " + path);
  }

  graph_t graph;
  graph.n = read_u32(in);
  graph.dim = read_u32(in);
  graph.entry_point = read_u32(in);
  graph.adj.resize(graph.n);

  for (uint32_t i = 0; i < graph.n; ++i) {
    const uint32_t degree = read_u32(in);
    graph.adj[i].resize(degree);
    in.read(reinterpret_cast<char*>(graph.adj[i].data()), degree * sizeof(uint32_t));
  }

  if (!in) {
    throw std::runtime_error("load_graph: truncated graph file " + path);
  }

  return graph;
}

// Save row-major vectors to a binary file.
void save_vectors(const float* data, uint32_t n, uint32_t dim, const std::string& path) {
  if (data == nullptr || n == 0 || dim == 0) {
    throw std::invalid_argument("save_vectors: data, n, and dim must be non-empty");
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("save_vectors: cannot open " + path);
  }

  write_u32(out, n);
  write_u32(out, dim);
  out.write(reinterpret_cast<const char*>(data), static_cast<size_t>(n) * dim * sizeof(float));
}
