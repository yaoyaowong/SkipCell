#include "index/hnsw.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <ostream>
#include <random>
#include <stdexcept>

namespace powerlaw_ann {

static thread_local std::mt19937 rng_local(std::random_device{}());

visited_table_t::visited_table_t(size_t n) : marks_(n, 0) {}

bool visited_table_t::visit(uint32_t v_id) {
  if (marks_[v_id] == epoch_) {
    return true;
  }
  marks_[v_id] = epoch_;
  return false;
}

void visited_table_t::reset() {
  ++epoch_;
  if (epoch_ == 0) {
    std::fill(marks_.begin(), marks_.end(), 0);
    epoch_ = 1;
  }
}

hnsw_index_t::hnsw_index_t(int dim, size_t max_elements, int m, int ef_construction)
    : dim_(dim), max_m_(m), max_m0_(m * 2), ef_construction_(ef_construction),
      mult_(1.0 / std::log(static_cast<double>(m))), max_elements_(max_elements) {
  size_links0_ = sizeof(uint16_t) + sizeof(uint32_t) * max_m0_;
  size_data_per_elem_ = size_links0_ + sizeof(float) * dim_ + sizeof(uint64_t);
  offset_data_ = size_links0_;
  offset_label_ = size_links0_ + sizeof(float) * dim_;
  link_list_size_ = sizeof(uint16_t) + sizeof(uint32_t) * max_m_;

  data_level0_.resize(max_elements_ * size_data_per_elem_, 0);
  link_lists_.assign(max_elements_, nullptr);
  element_level_.assign(max_elements_, 0);
  node_locks_ = std::vector<std::mutex>(max_elements_);
}

hnsw_index_t::~hnsw_index_t() {
  for (size_t i = 0; i < cur_elements_; ++i) {
    if (link_lists_[i]) {
      free(link_lists_[i]);
      link_lists_[i] = nullptr;
    }
  }
}

char* hnsw_index_t::link_lists_ptr(uint32_t v_id, int level) const {
  return link_lists_[v_id] + (level - 1) * static_cast<ptrdiff_t>(link_list_size_);
}

uint16_t* hnsw_index_t::link_list_count(uint32_t v_id, int level) {
  return reinterpret_cast<uint16_t*>(link_lists_ptr(v_id, level));
}

uint32_t* hnsw_index_t::link_list_data(uint32_t v_id, int level) {
  return reinterpret_cast<uint32_t*>(link_lists_ptr(v_id, level) + sizeof(uint16_t));
}

uint16_t hnsw_index_t::link_list_count_r(uint32_t v_id, int level) const {
  return *reinterpret_cast<const uint16_t*>(link_lists_ptr(v_id, level));
}

const uint32_t* hnsw_index_t::link_list_data_r(uint32_t v_id, int level) const {
  return reinterpret_cast<const uint32_t*>(link_lists_ptr(v_id, level) + sizeof(uint16_t));
}

int hnsw_index_t::random_level() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return static_cast<int>(-std::log(dist(rng_local)) * mult_);
}

float hnsw_index_t::l2_dist(const float* a, const float* b) const {
  float sum = 0.0f;
  for (int i = 0; i < dim_; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

std::priority_queue<std::pair<float, uint32_t>>
hnsw_index_t::search_layer(const float* query, uint32_t ep, int ef, int layer,
                           visited_table_t& vt) const {
  std::priority_queue<std::pair<float, uint32_t>> top_heap;
  std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>,
                      std::greater<>>
      candidates;

  const float dist_ep = l2_dist(query, vec_ptr(ep));
  top_heap.emplace(dist_ep, ep);
  candidates.emplace(dist_ep, ep);
  vt.visit(ep);

  while (!candidates.empty()) {
    const auto [dist_cand, cand_v_id] = candidates.top();
    candidates.pop();

    if (dist_cand > top_heap.top().first && static_cast<int>(top_heap.size()) >= ef) {
      break;
    }

    const uint32_t* nbrs = nullptr;
    int cnt = 0;
    if (layer == 0) {
      cnt = links0_count_r(cand_v_id);
      nbrs = links0_data_r(cand_v_id);
    } else {
      cnt = link_list_count_r(cand_v_id, layer);
      nbrs = link_list_data_r(cand_v_id, layer);
    }

    for (int i = 0; i < cnt; ++i) {
      const uint32_t nb_v_id = nbrs[i];
      if (vt.visit(nb_v_id)) {
        continue;
      }
      const float dist_nb = l2_dist(query, vec_ptr(nb_v_id));
      if (static_cast<int>(top_heap.size()) < ef || dist_nb < top_heap.top().first) {
        top_heap.emplace(dist_nb, nb_v_id);
        candidates.emplace(dist_nb, nb_v_id);
        if (static_cast<int>(top_heap.size()) > ef) {
          top_heap.pop();
        }
      }
    }
  }

  return top_heap;
}

uint32_t hnsw_index_t::greedy_lower(const float* query, uint32_t ep, int layer) const {
  uint32_t cur_v_id = ep;
  float best_dist = l2_dist(query, vec_ptr(cur_v_id));
  bool changed = true;

  while (changed) {
    changed = false;

    const uint32_t* nbrs = nullptr;
    int cnt = 0;
    if (layer == 0) {
      cnt = links0_count_r(cur_v_id);
      nbrs = links0_data_r(cur_v_id);
    } else {
      cnt = link_list_count_r(cur_v_id, layer);
      nbrs = link_list_data_r(cur_v_id, layer);
    }

    for (int i = 0; i < cnt; ++i) {
      const float dist = l2_dist(query, vec_ptr(nbrs[i]));
      if (dist < best_dist) {
        best_dist = dist;
        cur_v_id = nbrs[i];
        changed = true;
      }
    }
  }

  return cur_v_id;
}

void hnsw_index_t::select_neighbours(const float* query, std::vector<neighbor_t>& cands,
                                     int m_limit) {
  (void) query;

  std::sort(cands.begin(), cands.end(),
            [](const neighbor_t& lhs, const neighbor_t& rhs) { return lhs.dist < rhs.dist; });

  std::vector<neighbor_t> result;
  result.reserve(m_limit);

  for (const auto& cand : cands) {
    if (static_cast<int>(result.size()) >= m_limit) {
      break;
    }

    bool is_good = true;
    for (const auto& kept : result) {
      if (l2_dist(vec_ptr(cand.v_id), vec_ptr(kept.v_id)) < cand.dist) {
        is_good = false;
        break;
      }
    }

    if (is_good) {
      result.push_back(cand);
    }
  }

  if (static_cast<int>(result.size()) < m_limit) {
    for (const auto& cand : cands) {
      if (static_cast<int>(result.size()) >= m_limit) {
        break;
      }

      bool duplicated = false;
      for (const auto& kept : result) {
        if (kept.v_id == cand.v_id) {
          duplicated = true;
          break;
        }
      }

      if (!duplicated) {
        result.push_back(cand);
      }
    }
  }

  cands = std::move(result);
}

void hnsw_index_t::connect(uint32_t new_v_id, const std::vector<neighbor_t>& chosen, int layer,
                           int m_limit) {
  if (layer == 0) {
    uint16_t* count = links0_count(new_v_id);
    uint32_t* data = links0_data(new_v_id);
    *count = static_cast<uint16_t>(chosen.size());
    for (size_t i = 0; i < chosen.size(); ++i) {
      data[i] = chosen[i].v_id;
    }
  } else {
    uint16_t* count = link_list_count(new_v_id, layer);
    uint32_t* data = link_list_data(new_v_id, layer);
    *count = static_cast<uint16_t>(chosen.size());
    for (size_t i = 0; i < chosen.size(); ++i) {
      data[i] = chosen[i].v_id;
    }
  }

  for (const auto& nb : chosen) {
    std::lock_guard<std::mutex> lk(node_locks_[nb.v_id]);

    uint16_t* count = nullptr;
    uint32_t* data = nullptr;
    if (layer == 0) {
      count = links0_count(nb.v_id);
      data = links0_data(nb.v_id);
    } else {
      count = link_list_count(nb.v_id, layer);
      data = link_list_data(nb.v_id, layer);
    }

    const int cur_cnt = *count;
    if (cur_cnt < m_limit) {
      data[cur_cnt] = new_v_id;
      *count = static_cast<uint16_t>(cur_cnt + 1);
      continue;
    }

    const float new_dist = l2_dist(vec_ptr(nb.v_id), vec_ptr(new_v_id));
    int worst_idx = 0;
    float worst_dist = l2_dist(vec_ptr(nb.v_id), vec_ptr(data[0]));

    for (int i = 1; i < cur_cnt; ++i) {
      const float dist = l2_dist(vec_ptr(nb.v_id), vec_ptr(data[i]));
      if (dist > worst_dist) {
        worst_dist = dist;
        worst_idx = i;
      }
    }

    if (new_dist < worst_dist) {
      data[worst_idx] = new_v_id;
    }
  }
}

void hnsw_index_t::insert(const float* vec, uint64_t label) {
  uint32_t new_v_id;
  {
    std::lock_guard<std::mutex> lk(label_lock_);
    const auto it = label_lookup_.find(label);
    if (it != label_lookup_.end()) {
      const uint32_t old_v_id = it->second;
      std::memcpy(vec_ptr(old_v_id), vec, sizeof(float) * dim_);
      return;
    }

    if (cur_elements_ >= max_elements_) {
      throw std::runtime_error("hnsw_index_t: capacity exceeded");
    }

    new_v_id = static_cast<uint32_t>(cur_elements_++);
    label_lookup_[label] = new_v_id;
  }

  std::memcpy(vec_ptr(new_v_id), vec, sizeof(float) * dim_);
  label_ref(new_v_id) = label;
  *links0_count(new_v_id) = 0;

  const int new_level = random_level();
  element_level_[new_v_id] = new_level;

  if (new_level > 0) {
    link_lists_[new_v_id] = static_cast<char*>(calloc(new_level * link_list_size_, 1));
    if (!link_lists_[new_v_id]) {
      throw std::bad_alloc();
    }
  }

  if (new_v_id == 0) {
    std::lock_guard<std::mutex> lk(global_lock_);
    entry_point_ = 0;
    max_level_ = new_level;
    return;
  }

  uint32_t ep;
  int top_level;
  {
    std::lock_guard<std::mutex> lk(global_lock_);
    ep = entry_point_;
    top_level = max_level_;
  }

  visited_table_t vt(max_elements_);

  for (int level = top_level; level > new_level; --level) {
    ep = greedy_lower(vec, ep, level);
  }

  for (int level = std::min(new_level, top_level); level >= 0; --level) {
    const int m_limit = (level == 0) ? max_m0_ : max_m_;

    vt.reset();
    auto top_heap = search_layer(vec, ep, ef_construction_, level, vt);

    std::vector<neighbor_t> cands;
    cands.reserve(top_heap.size());
    while (!top_heap.empty()) {
      cands.push_back({top_heap.top().second, top_heap.top().first});
      top_heap.pop();
    }

    select_neighbours(vec, cands, m_limit);
    connect(new_v_id, cands, level, m_limit);

    if (!cands.empty()) {
      ep = cands.front().v_id;
    }
  }

  if (new_level > top_level) {
    std::lock_guard<std::mutex> lk(global_lock_);
    if (new_level > max_level_) {
      entry_point_ = new_v_id;
      max_level_ = new_level;
    }
  }
}

std::vector<std::pair<float, uint64_t>> hnsw_index_t::search(const float* query, int k,
                                                             int ef) const {
  if (cur_elements_ == 0) {
    return {};
  }

  ef = std::max(ef, k);

  uint32_t ep = entry_point_;
  const int top_level = max_level_;

  for (int level = top_level; level > 0; --level) {
    ep = greedy_lower(query, ep, level);
  }

  visited_table_t vt(max_elements_);
  auto top_heap = search_layer(query, ep, ef, 0, vt);

  std::vector<std::pair<float, uint64_t>> result;
  while (!top_heap.empty()) {
    result.emplace_back(top_heap.top().first, label_of(top_heap.top().second));
    top_heap.pop();
  }

  std::sort(result.begin(), result.end());
  if (static_cast<int>(result.size()) > k) {
    result.resize(k);
  }
  return result;
}

static void write_pod(std::ostream& out, const void* buf, size_t size) {
  out.write(reinterpret_cast<const char*>(buf), static_cast<std::streamsize>(size));
}

static void read_pod(std::istream& in, void* buf, size_t size) {
  in.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(size));
}

template <typename t_scalar>
static void write_scalar(std::ostream& out, t_scalar value) {
  write_pod(out, &value, sizeof(value));
}

template <typename t_scalar>
static void read_scalar(std::istream& in, t_scalar& value) {
  read_pod(in, &value, sizeof(value));
}

void hnsw_index_t::save(std::ostream& out) const {
  write_scalar(out, dim_);
  write_scalar(out, max_m_);
  write_scalar(out, max_m0_);
  write_scalar(out, ef_construction_);
  write_scalar(out, mult_);
  write_scalar(out, max_elements_);
  write_scalar(out, cur_elements_);
  write_scalar(out, size_data_per_elem_);
  write_scalar(out, size_links0_);
  write_scalar(out, offset_data_);
  write_scalar(out, offset_label_);
  write_scalar(out, link_list_size_);
  write_scalar(out, max_level_);
  write_scalar(out, entry_point_);

  write_pod(out, data_level0_.data(), cur_elements_ * size_data_per_elem_);

  for (size_t i = 0; i < cur_elements_; ++i) {
    const int level = element_level_[i];
    write_scalar(out, level);
    if (level > 0) {
      write_pod(out, link_lists_[i], level * link_list_size_);
    }
  }
}

void hnsw_index_t::load(std::istream& in) {
  for (size_t i = 0; i < cur_elements_; ++i) {
    if (link_lists_[i]) {
      free(link_lists_[i]);
      link_lists_[i] = nullptr;
    }
  }

  int read_dim;
  int read_max_m;
  int read_max_m0;
  int read_ef_construction;
  double read_mult;
  size_t read_max_elements;
  size_t read_cur_elements;
  size_t read_size_data_per_elem;
  size_t read_size_links0;
  size_t read_offset_data;
  size_t read_offset_label;
  size_t read_link_list_size;
  int read_max_level;
  uint32_t read_entry_point;

  read_scalar(in, read_dim);
  read_scalar(in, read_max_m);
  read_scalar(in, read_max_m0);
  read_scalar(in, read_ef_construction);
  read_scalar(in, read_mult);
  read_scalar(in, read_max_elements);
  read_scalar(in, read_cur_elements);
  read_scalar(in, read_size_data_per_elem);
  read_scalar(in, read_size_links0);
  read_scalar(in, read_offset_data);
  read_scalar(in, read_offset_label);
  read_scalar(in, read_link_list_size);
  read_scalar(in, read_max_level);
  read_scalar(in, read_entry_point);

  dim_ = read_dim;
  max_m_ = read_max_m;
  max_m0_ = read_max_m0;
  ef_construction_ = read_ef_construction;
  mult_ = read_mult;
  max_elements_ = read_max_elements;
  cur_elements_ = read_cur_elements;
  size_data_per_elem_ = read_size_data_per_elem;
  size_links0_ = read_size_links0;
  offset_data_ = read_offset_data;
  offset_label_ = read_offset_label;
  link_list_size_ = read_link_list_size;
  max_level_ = read_max_level;
  entry_point_ = read_entry_point;

  data_level0_.resize(max_elements_ * size_data_per_elem_, 0);
  link_lists_.assign(max_elements_, nullptr);
  element_level_.assign(max_elements_, 0);
  node_locks_ = std::vector<std::mutex>(max_elements_);
  label_lookup_.clear();

  read_pod(in, data_level0_.data(), cur_elements_ * size_data_per_elem_);

  for (size_t i = 0; i < cur_elements_; ++i) {
    int level;
    read_scalar(in, level);
    element_level_[i] = level;

    if (level > 0) {
      link_lists_[i] = static_cast<char*>(malloc(level * link_list_size_));
      if (!link_lists_[i]) {
        throw std::bad_alloc();
      }
      read_pod(in, link_lists_[i], level * link_list_size_);
    }

    const uint32_t v_id = static_cast<uint32_t>(i);
    label_lookup_[label_of(v_id)] = v_id;
  }
}

} // namespace powerlaw_ann
