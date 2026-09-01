#ifndef INDEX_HNSW
#define INDEX_HNSW

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

namespace powerlaw_ann {

/**
 * @brief Per-search visited marker table.
 */
class visited_table_t {
public:
  /**
   * @brief Construct visited table.
   * @param n Node capacity.
   */
  explicit visited_table_t(size_t n);

  /**
   * @brief Mark node visited in current epoch.
   * @param v_id Internal vertex id.
   * @return True if already visited in current epoch.
   */
  bool visit(uint32_t v_id);

  /**
   * @brief Advance epoch and clear table on wraparound.
   */
  void reset();

private:
  std::vector<uint16_t> marks_;
  uint16_t epoch_{1};
};

/**
 * @brief In-memory HNSW index.
 */
class hnsw_index_t {
public:
  /**
   * @brief neighbor_t candidate.
   */
  struct neighbor_t {
    uint32_t v_id;
    float dist;
  };

  /**
   * @brief Construct HNSW index.
   * @param dim Vector dimension.
   * @param max_elements Max number of vectors.
   * @param m Max out-degree at upper levels.
   * @param ef_construction Construction beam width.
   */
  hnsw_index_t(int dim, size_t max_elements, int m = 16, int ef_construction = 200);

  /**
   * @brief Destroy index and release dynamic link lists.
   */
  ~hnsw_index_t();

  hnsw_index_t(const hnsw_index_t&) = delete;
  hnsw_index_t& operator=(const hnsw_index_t&) = delete;

  /**
   * @brief Insert or update one vector.
   * @param vec Pointer to vector data.
   * @param label External label.
   */
  void insert(const float* vec, uint64_t label);

  /**
   * @brief Search nearest neighbors.
   * @param query Pointer to query vector.
   * @param k Number of results.
   * @param ef Search beam width.
   * @return Result list sorted by ascending distance.
   */
  std::vector<std::pair<float, uint64_t>> search(const float* query, int k, int ef = 50) const;

  size_t size() const { return cur_elements_; }
  int dim() const { return dim_; }
  int max_m() const { return max_m_; }
  int max_m0() const { return max_m0_; }
  size_t max_elements() const { return max_elements_; }
  int ef_construction() const { return ef_construction_; }
  int max_level() const { return max_level_; }
  uint32_t entry_point() const { return entry_point_; }
  double mult() const { return mult_; }

  /**
   * @brief Serialize index state.
   * @param out Output stream.
   */
  void save(std::ostream& out) const;

  /**
   * @brief Load index state.
   * @param in Input stream.
   */
  void load(std::istream& in);

private:
  int dim_;
  int max_m_;
  int max_m0_;
  int ef_construction_;
  double mult_;

  size_t max_elements_;
  size_t cur_elements_{0};

  size_t size_links0_;
  size_t size_data_per_elem_;
  size_t offset_data_;
  size_t offset_label_;
  size_t link_list_size_;

  std::vector<char> data_level0_;
  std::vector<char*> link_lists_;
  std::vector<int> element_level_;

  std::unordered_map<uint64_t, uint32_t> label_lookup_;

  int max_level_{-1};
  uint32_t entry_point_{0};

  mutable std::vector<std::mutex> node_locks_;
  mutable std::mutex global_lock_;
  mutable std::mutex label_lock_;

  /**
   * @brief Sample random layer level.
   * @return Random level.
   */
  int random_level();

  /**
   * @brief Compute squared L2 distance.
   * @param a Left vector.
   * @param b Right vector.
   * @return Squared L2 distance.
   */
  float l2_dist(const float* a, const float* b) const;

  char* elem_ptr(uint32_t v_id) { return data_level0_.data() + v_id * size_data_per_elem_; }
  const char* elem_ptr(uint32_t v_id) const {
    return data_level0_.data() + v_id * size_data_per_elem_;
  }
  float* vec_ptr(uint32_t v_id) { return reinterpret_cast<float*>(elem_ptr(v_id) + offset_data_); }
  const float* vec_ptr(uint32_t v_id) const {
    return reinterpret_cast<const float*>(elem_ptr(v_id) + offset_data_);
  }
  uint64_t& label_ref(uint32_t v_id) {
    return *reinterpret_cast<uint64_t*>(elem_ptr(v_id) + offset_label_);
  }
  uint64_t label_of(uint32_t v_id) const {
    return *reinterpret_cast<const uint64_t*>(elem_ptr(v_id) + offset_label_);
  }

  uint16_t* links0_count(uint32_t v_id) { return reinterpret_cast<uint16_t*>(elem_ptr(v_id)); }
  uint32_t* links0_data(uint32_t v_id) {
    return reinterpret_cast<uint32_t*>(elem_ptr(v_id) + sizeof(uint16_t));
  }
  uint16_t links0_count_r(uint32_t v_id) const {
    return *reinterpret_cast<const uint16_t*>(elem_ptr(v_id));
  }
  const uint32_t* links0_data_r(uint32_t v_id) const {
    return reinterpret_cast<const uint32_t*>(elem_ptr(v_id) + sizeof(uint16_t));
  }

  /**
   * @brief Get upper-layer link-list pointer.
   * @param v_id Internal vertex id.
   * @param level Link-list level (>0).
   * @return Raw pointer to link-list block.
   */
  char* link_lists_ptr(uint32_t v_id, int level) const;

  /**
   * @brief Get mutable neighbor count at level.
   * @param v_id Internal vertex id.
   * @param level Link-list level (>0).
   * @return Pointer to neighbor count field.
   */
  uint16_t* link_list_count(uint32_t v_id, int level);

  /**
   * @brief Get mutable neighbor array at level.
   * @param v_id Internal vertex id.
   * @param level Link-list level (>0).
   * @return Pointer to neighbor id array.
   */
  uint32_t* link_list_data(uint32_t v_id, int level);

  /**
   * @brief Get neighbor count at level.
   * @param v_id Internal vertex id.
   * @param level Link-list level (>0).
   * @return neighbor_t count.
   */
  uint16_t link_list_count_r(uint32_t v_id, int level) const;

  /**
   * @brief Get neighbor array at level.
   * @param v_id Internal vertex id.
   * @param level Link-list level (>0).
   * @return Read-only neighbor id array pointer.
   */
  const uint32_t* link_list_data_r(uint32_t v_id, int level) const;

  /**
   * @brief Beam search inside one level.
   * @param query Pointer to query vector.
   * @param ep Entry point.
   * @param ef Beam width.
   * @param layer Target level.
   * @param vt Visited marker table.
   * @return Max-heap of current best candidates.
   */
  std::priority_queue<std::pair<float, uint32_t>>
  search_layer(const float* query, uint32_t ep, int ef, int layer, visited_table_t& vt) const;

  /**
   * @brief Greedy descent within one level.
   * @param query Pointer to query vector.
   * @param ep Entry point.
   * @param layer Target level.
   * @return Best local vertex id.
   */
  uint32_t greedy_lower(const float* query, uint32_t ep, int layer) const;

  /**
   * @brief Select neighbor subset with diversity heuristic.
   * @param query Pointer to query vector.
   * @param cands Candidate list.
   * @param m_limit Degree limit.
   */
  void select_neighbours(const float* query, std::vector<neighbor_t>& cands, int m_limit);

  /**
   * @brief Connect new node with selected neighbors.
   * @param new_v_id New node id.
   * @param chosen Selected neighbors.
   * @param layer Target level.
   * @param m_limit Degree limit.
   */
  void connect(uint32_t new_v_id, const std::vector<neighbor_t>& chosen, int layer, int m_limit);
};

} // namespace powerlaw_ann

#endif // INDEX_HNSW
