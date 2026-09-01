#ifndef COMMON_SCRATCH
#define COMMON_SCRATCH

#include "common/abstract_scratch.h"
#include "common/concurrent_queue.h"
#include "index/neighbor.h"
#include "storage/aligned_file_reader.h"
#include "third/tsl/robin_set.h"

#include <boost/dynamic_bitset_fwd.hpp>
#include <cstdint>
#include <vector>

namespace powerlaw_ann {
template <typename T>
class pq_scratch_t;

//
// abstract_scratch_t space for in-memory index based search
//
template <typename T>
class in_mem_query_scratch_t : public abstract_scratch_t<T> {
public:
  ~in_mem_query_scratch_t();
  in_mem_query_scratch_t(uint32_t search_l, uint32_t indexing_l, uint32_t r, uint32_t maxc,
                         size_t dim, size_t aligned_dim, size_t alignment_factor,
                         bool init_pq_scratch = false);
  void resize_for_new_l(uint32_t new_search_l);
  void clear();

  inline uint32_t get_l() { return l_; }
  inline uint32_t get_r() { return r_; }
  inline uint32_t get_maxc() { return maxc_; }
  inline T* aligned_query() { return this->aligned_query_; }
  inline pq_scratch_t<T>* pq_scratch() { return this->pq_scratch_; }
  inline std::vector<neighbor_t>& pool() { return pool_; }
  inline neighbor_priority_queue_t& best_l_nodes() { return best_l_nodes_; }
  inline std::vector<float>& occlude_factor() { return occlude_factor_; }
  inline tsl::robin_set<uint32_t>& inserted_into_pool_rs() { return inserted_into_pool_rs_; }
  inline boost::dynamic_bitset<>& inserted_into_pool_bs() { return *inserted_into_pool_bs_; }
  inline std::vector<uint32_t>& id_scratch() { return id_scratch_; }
  inline std::vector<float>& dist_scratch() { return dist_scratch_; }
  inline tsl::robin_set<uint32_t>& expanded_nodes_set() { return expanded_nodes_set_; }
  inline std::vector<neighbor_t>& expanded_nodes_vec() { return expanded_nghrs_vec_; }
  inline std::vector<uint32_t>& occlude_list_output() { return occlude_list_output_; }

private:
  uint32_t l_;
  uint32_t r_;
  uint32_t maxc_;

  // pool_ stores all neighbors explored from best_l_nodes.
  // Usually around L+R, but could be higher.
  // Initialized to 3L+R for some slack, expands as needed.
  std::vector<neighbor_t> pool_;

  // best_l_nodes_ is reserved for storing best L entries
  // Underlying storage is L+1 to support inserts
  neighbor_priority_queue_t best_l_nodes_;

  // occlude_factor_.size() >= pool.size() in occlude_list function
  // pool_ is clipped to maxc in occlude_list before affecting occlude_factor_
  // occlude_factor_ is initialized to maxc size
  std::vector<float> occlude_factor_;

  // Capacity initialized to 20L
  tsl::robin_set<uint32_t> inserted_into_pool_rs_;

  // Use a pointer here to allow for forward declaration of dynamic_bitset
  // in public headers to avoid making boost a dependency for clients
  // of DiskANN.
  boost::dynamic_bitset<>* inserted_into_pool_bs_;

  // id_scratch_.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
  std::vector<uint32_t> id_scratch_;

  // dist_scratch_ must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
  // dist_scratch_ should be at least the size of id_scratch
  std::vector<float> dist_scratch_;

  //  Buffers used in process delete, capacity increases as needed
  tsl::robin_set<uint32_t> expanded_nodes_set_;
  std::vector<neighbor_t> expanded_nghrs_vec_;
  std::vector<uint32_t> occlude_list_output_;
};

//
// abstract_scratch_t space for SSD index based search
//

template <typename T>
class ssd_query_scratch_t : public abstract_scratch_t<T> {
public:
  T* coord_scratch = nullptr; // MUST BE AT LEAST [sizeof(T) * data_dim]

  char* sector_scratch = nullptr; // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
  size_t sector_idx = 0;          // index of next [SECTOR_LEN] scratch to use

  tsl::robin_set<size_t> visited;
  neighbor_priority_queue_t retset;
  std::vector<neighbor_t> full_retset;

  ssd_query_scratch_t(size_t aligned_dim, size_t visited_reserve,
                      size_t candidate_capacity = 0);
  ~ssd_query_scratch_t();

  void reset();
};

template <typename T>
class ssd_thread_data_t {
public:
  ssd_query_scratch_t<T> scratch;
  aligned_io_context_t ctx;

  ssd_thread_data_t(size_t aligned_dim, size_t visited_reserve,
                    size_t candidate_capacity = 0);
  void clear();
};

//
// Class to avoid the hassle of pushing and popping the query scratch.
//
template <typename T>
class scratch_store_manager_t {
public:
  scratch_store_manager_t(concurrent_queue_t<T*>& query_scratch) : scratch_pool_(query_scratch) {
    scratch_ = query_scratch.pop();
    while (scratch_ == nullptr) {
      query_scratch.wait_for_push_notify();
      scratch_ = query_scratch.pop();
    }
  }
  T* scratch_space() { return scratch_; }

  ~scratch_store_manager_t() {
    scratch_->clear();
    scratch_pool_.push(scratch_);
    scratch_pool_.push_notify_all();
  }

  void destroy() {
    while (!scratch_pool_.empty()) {
      auto scratch = scratch_pool_.pop();
      while (scratch == nullptr) {
        scratch_pool_.wait_for_push_notify();
        scratch = scratch_pool_.pop();
      }
      delete scratch;
    }
  }

private:
  T* scratch_;
  concurrent_queue_t<T*>& scratch_pool_;
  scratch_store_manager_t(const scratch_store_manager_t<T>&);
  scratch_store_manager_t& operator=(const scratch_store_manager_t<T>&);
};
} // namespace powerlaw_ann

#endif // COMMON_SCRATCH
