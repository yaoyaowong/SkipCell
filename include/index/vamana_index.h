#ifndef INDEX_VAMANA_INDEX
#define INDEX_VAMANA_INDEX

#ifdef EXEC_ENV_OLS
#include "storage/aligned_file_reader.h"
#endif

#include "common/distance.h"
#include "common/locking.h"
#include "common/natural_number_map.h"
#include "common/natural_number_set.h"
#include "common/parameters.h"
#include "common/platform_compat.h"
#include "common/scratch.h"
#include "index/abstract_data_store.h"
#include "index/abstract_graph_store.h"
#include "index/abstract_index.h"
#include "index/index_config.h"
#include "index/neighbor.h"
#include "index/post_link_graph.h"
#include "index/rcni.h"
#include "index/vamana_build_stats.h"
#include "third/tsl/sparse_map.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define OVERHEAD_FACTOR 1.1
#define EXPAND_IF_FULL 0
#define DEFAULT_MAXC 750

namespace powerlaw_ann {

inline double estimate_ram_usage(size_t size, uint32_t dim, uint32_t datasize, uint32_t degree) {
  constexpr uint64_t alignment = 8;
  const uint64_t aligned_dim =
      ((static_cast<uint64_t>(dim) + alignment - 1) / alignment) * alignment;
  const double point_count = static_cast<double>(size);
  const double size_of_data =
      point_count * static_cast<double>(aligned_dim) * static_cast<double>(datasize);
  const double size_of_graph =
      point_count * static_cast<double>(degree) * sizeof(uint32_t) * defaults::GRAPH_SLACK_FACTOR;
  const double size_of_locks = point_count * sizeof(non_recursive_mutex_t);
  const double size_of_outer_vector = point_count * sizeof(std::ptrdiff_t);

  return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks + size_of_outer_vector);
}

template <typename T, typename tag_t = uint32_t, typename label_t = uint32_t>
class vamana_index_t : public abstract_index_t {
  /**************************************************************************
   *
   * Public functions acquire one or more of update_lock_, consolidate_lock_,
   * tag_lock_, delete_lock_ before calling protected functions which DO NOT
   * acquire these locks. They might acquire locks on locks_[i]
   *
   **************************************************************************/

public:
  // Constructor for Bulk operations and for creating the index object solely
  // for loading a pre-existing index.
  POWERLAWANN_DLLEXPORT
  vamana_index_t(const index_config_t& index_config,
                 std::shared_ptr<abstract_data_store_t<T>> data_store,
                 std::unique_ptr<abstract_graph_store_t> graph_store,
                 std::shared_ptr<abstract_data_store_t<T>> pq_data_store = nullptr);

  // Constructor for incremental index
  POWERLAWANN_DLLEXPORT
  vamana_index_t(metric_t m, const size_t dim, const size_t max_points,
                 const std::shared_ptr<index_write_parameters_t> index_parameters,
                 const std::shared_ptr<index_search_params_t> index_search_params,
                 const size_t num_frozen_pts = 0, const bool dynamic_index = false,
                 const bool enable_tags = false, const bool concurrent_consolidate = false,
                 const bool pq_dist_build = false, const size_t num_pq_chunks = 0,
                 const bool use_opq = false, const bool filtered_index = false);

  POWERLAWANN_DLLEXPORT ~vamana_index_t() override;

  // Saves graph, data, metadata and associated tags.
  POWERLAWANN_DLLEXPORT void save(const char* filename, bool compact_before_save = false) override;

  // Load functions
#ifdef EXEC_ENV_OLS
  POWERLAWANN_DLLEXPORT void load(aligned_file_reader_t& reader, uint32_t num_threads,
                                  uint32_t search_l) override;
#else
  // Reads the number of frozen points from graph's metadata file section.
  POWERLAWANN_DLLEXPORT static size_t get_graph_num_frozen_points(const std::string& graph_file);

  POWERLAWANN_DLLEXPORT void load(const char* index_file, uint32_t num_threads,
                                  uint32_t search_l) override;
#endif

  // get some private variables
  POWERLAWANN_DLLEXPORT size_t get_num_points();
  POWERLAWANN_DLLEXPORT size_t get_max_points();

  POWERLAWANN_DLLEXPORT bool detect_common_filters(uint32_t point_id, bool search_invocation,
                                                   const std::vector<label_t>& incoming_labels);

  // Batch build from a file. Optionally pass tags vector.
  POWERLAWANN_DLLEXPORT void build(const char* filename, const size_t num_points_to_load,
                                   const std::vector<tag_t>& tags = std::vector<tag_t>());

  // Batch build from a file. Optionally pass tags file.
  POWERLAWANN_DLLEXPORT void build(const char* filename, const size_t num_points_to_load,
                                   const char* tag_filename);

  // Batch build from a data array, which must pad vectors to aligned_dim
  POWERLAWANN_DLLEXPORT void build(const T* data, const size_t num_points_to_load,
                                   const std::vector<tag_t>& tags);

  // Based on filter params builds a filtered or unfiltered index
  POWERLAWANN_DLLEXPORT void build(const std::string& data_file, const size_t num_points_to_load,
                                   index_filter_params_t& filter_params) override;

  // Filtered Support
  POWERLAWANN_DLLEXPORT void
  build_filtered_index(const char* filename, const std::string& label_file,
                       const size_t num_points_to_load,
                       const std::vector<tag_t>& tags = std::vector<tag_t>());

  POWERLAWANN_DLLEXPORT void set_universal_label(const label_t& label);

  // Get converted integer label from string to int map (label_map_)
  POWERLAWANN_DLLEXPORT label_t get_converted_label(const std::string& raw_label);

  // Set starting point of an index before inserting any points incrementally.
  // The data count should be equal to num_frozen_pts_ * aligned_dim_.
  POWERLAWANN_DLLEXPORT void set_start_points(const T* data, size_t data_count);
  // Set starting points to random points on a sphere of certain radius.
  // A fixed random seed can be specified for scenarios where it's important
  // to have higher consistency between index builds.
  POWERLAWANN_DLLEXPORT void set_start_points_at_random(T radius, uint32_t random_seed = 0);

  // For fast_l2 search on a static index, we interleave the data with graph
  POWERLAWANN_DLLEXPORT void optimize_index_layout() override;

  POWERLAWANN_DLLEXPORT void enable_vamana_build_stats();
  POWERLAWANN_DLLEXPORT void dump_vamana_build_stats(const std::string& prefix) const;

  /**
   * @brief Installs a non-owning observer for primary-build RCNI prune events.
   *
   * Set the observer before calling `build()`. The observer must outlive the build and must be
   * safe for concurrent callbacks when the build uses multiple threads. Passing `nullptr`
   * selects the allocation-free DiskANN baseline path.
   *
   * @param observer Observer to install, or `nullptr` to disable RCNI observation.
   * @throws std::exception If a non-null observer cannot prepare its build state.
   */
  POWERLAWANN_DLLEXPORT void set_rcni_prune_observer(rcni_prune_observer_t* observer);

  /**
   * @brief Installs a non-owning observer for the completed Base graph.
   *
   * The observer runs once immediately after `link()` while the immutable Base adjacency and
   * original float32 vectors remain resident. It must outlive the next `build()` call and is
   * cleared when that call returns or throws. Passing `nullptr` cancels a pending observer and
   * preserves the allocation-free baseline path.
   *
   * @param observer Observer to install, or `nullptr` to disable post-link observation.
   * @throws std::invalid_argument If a non-null observer is installed on a non-float32, non-L2,
   * dynamic, or filtered index.
   */
  POWERLAWANN_DLLEXPORT void set_post_link_graph_observer(post_link_graph_observer_t* observer);

  // For fast_l2 search on optimized layout
  POWERLAWANN_DLLEXPORT void search_with_optimized_layout(const T* query, size_t K, size_t L,
                                                          uint32_t* indices);

  // Added search overload that takes L as parameter, so that we
  // can customize L on a per-query basis without tampering with "Parameters"
  template <typename IDType>
  POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> search(const T* query, const size_t K,
                                                             const uint32_t L, IDType* indices,
                                                             float* distances = nullptr);

  // Initialize space for res_vectors before calling.
  POWERLAWANN_DLLEXPORT size_t search_with_tags(const T* query, const uint64_t K, const uint32_t L,
                                                tag_t* tags, float* distances,
                                                std::vector<T*>& res_vectors,
                                                bool use_filters = false,
                                                const std::string filter_label = "");

  // Filter support search
  template <typename index_t>
  POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
  search_with_filters(const T* query, const label_t& filter_label, const size_t K, const uint32_t L,
                      index_t* indices, float* distances);

  // Will fail if tag already in the index or if tag=0.
  POWERLAWANN_DLLEXPORT int insert_point(const T* point, const tag_t tag);

  // Will fail if tag already in the index or if tag=0.
  POWERLAWANN_DLLEXPORT int insert_point(const T* point, const tag_t tag,
                                         const std::vector<label_t>& label);

  // call this before issuing deletions to sets relevant flags
  POWERLAWANN_DLLEXPORT int enable_delete();

  // Record deleted point now and restructure graph later. Return -1 if tag
  // not found, 0 if OK.
  POWERLAWANN_DLLEXPORT int lazy_delete(const tag_t& tag);

  // Record deleted points now and restructure graph later. Add to failed_tags
  // if tag not found.
  POWERLAWANN_DLLEXPORT void lazy_delete(const std::vector<tag_t>& tags,
                                         std::vector<tag_t>& failed_tags);

  // Call after a series of lazy deletions
  // Returns number of live points left after consolidation
  // If conc_consolidates_ is set in the ctor, then this call can be invoked
  // alongside inserts and lazy deletes, else it acquires update_lock_
  POWERLAWANN_DLLEXPORT consolidation_report_t
  consolidate_deletes(const index_write_parameters_t& parameters) override;

  POWERLAWANN_DLLEXPORT void prune_all_neighbors(const uint32_t max_degree,
                                                 const uint32_t max_occlusion, const float alpha);

  POWERLAWANN_DLLEXPORT bool is_index_saved();

  // repositions frozen points to the end of data_ - if they have been moved
  // during deletion
  POWERLAWANN_DLLEXPORT void reposition_frozen_point_to_end();
  POWERLAWANN_DLLEXPORT void reposition_points(uint32_t old_location_start,
                                               uint32_t new_location_start, uint32_t num_locations);

  // POWERLAWANN_DLLEXPORT void save_index_as_one_file(bool flag);

  POWERLAWANN_DLLEXPORT void get_active_tags(tsl::robin_set<tag_t>& active_tags);

  // memory should be allocated for vec before calling this function
  POWERLAWANN_DLLEXPORT int get_vector_by_tag(tag_t& tag, T* vec);

  POWERLAWANN_DLLEXPORT void print_status();

  POWERLAWANN_DLLEXPORT void count_nodes_at_bfs_levels();

  // This variable MUST be updated if the number of entries in the metadata
  // change.
  POWERLAWANN_DLLEXPORT static const int METADATA_ROWS = 5;

  // ********************************
  //
  // Internals of the library
  //
  // ********************************

protected:
  // overload of abstract index virtual methods
  virtual void build_impl(const data_type_t& data, const size_t num_points_to_load,
                          tag_vector_t& tags) override;

  virtual std::pair<uint32_t, uint32_t> search_impl(const data_type_t& query, const size_t K,
                                                    const uint32_t L, std::any& indices,
                                                    float* distances = nullptr) override;
  virtual std::pair<uint32_t, uint32_t>
  search_with_filters_impl(const data_type_t& query, const std::string& filter_label_raw,
                           const size_t K, const uint32_t L, std::any& indices,
                           float* distances) override;

  virtual int insert_point_impl(const data_type_t& data_point, const tag_type_t tag) override;
  virtual int insert_point_impl(const data_type_t& data_point, const tag_type_t tag,
                                label_vector_t& labels) override;

  virtual int lazy_delete_impl(const tag_type_t& tag) override;

  virtual void lazy_delete_impl(tag_vector_t& tags, tag_vector_t& failed_tags) override;

  virtual void get_active_tags_impl(tag_robin_set_t& active_tags) override;

  virtual void set_start_points_at_random_impl(data_type_t radius,
                                               uint32_t random_seed = 0) override;

  virtual int get_vector_by_tag_impl(tag_type_t& tag, data_type_t& vec) override;

  virtual void search_with_optimized_layout_impl(const data_type_t& query, size_t K, size_t L,
                                                 uint32_t* indices) override;

  virtual size_t search_with_tags_impl(const data_type_t& query, const uint64_t K, const uint32_t L,
                                       const tag_type_t& tags, float* distances,
                                       data_vector_t& res_vectors, bool use_filters = false,
                                       const std::string filter_label = "") override;

  virtual void set_universal_label_impl(const label_type_t universal_label) override;

  // No copy/assign.
  vamana_index_t(const vamana_index_t<T, tag_t, label_t>&) = delete;
  vamana_index_t<T, tag_t, label_t>& operator=(const vamana_index_t<T, tag_t, label_t>&) = delete;

  // Use after data_ and nd_ have been populated
  // Acquire exclusive update_lock_ before calling
  void build_with_data_populated(const std::vector<tag_t>& tags);

  // generates 1 frozen point that will never be deleted from the graph
  // This is not visible to the user
  void generate_frozen_point();

  // determines navigating node of the graph by calculating medoid of datafopt
  uint32_t calculate_entry_point();

  void parse_label_file(const std::string& label_file, size_t& num_pts_labels);

  std::unordered_map<std::string, label_t> load_label_map(const std::string& map_file);

  // Returns the locations of start point and frozen points suitable for use
  // with iterate_to_fixed_point.
  std::vector<uint32_t> get_init_ids();

  // The query to use is placed in scratch->aligned_query
  std::pair<uint32_t, uint32_t>
  iterate_to_fixed_point(in_mem_query_scratch_t<T>* scratch, const uint32_t l_index,
                         const std::vector<uint32_t>& init_ids, bool use_filter,
                         const std::vector<label_t>& filters, bool search_invocation);

  void search_for_point_and_prune(int location, uint32_t l_index,
                                  std::vector<uint32_t>& pruned_list,
                                  in_mem_query_scratch_t<T>* scratch, bool use_filter = false,
                                  uint32_t filtered_l_index = 0);

  template <bool observe_rcni>
  void search_for_point_and_prune_impl(int location, uint32_t l_index,
                                       std::vector<uint32_t>& pruned_list,
                                       in_mem_query_scratch_t<T>* scratch, bool use_filter,
                                       uint32_t filtered_l_index);

  void prune_neighbors(const uint32_t location, std::vector<neighbor_t>& pool,
                       std::vector<uint32_t>& pruned_list, in_mem_query_scratch_t<T>* scratch);

  void prune_neighbors(const uint32_t location, std::vector<neighbor_t>& pool, const uint32_t range,
                       const uint32_t max_candidate_size, const float alpha,
                       std::vector<uint32_t>& pruned_list, in_mem_query_scratch_t<T>* scratch);

  template <bool observe_rcni>
  void prune_neighbors_impl(const uint32_t location, std::vector<neighbor_t>& pool,
                            const uint32_t range, const uint32_t max_candidate_size,
                            const float alpha, std::vector<uint32_t>& pruned_list,
                            in_mem_query_scratch_t<T>* scratch);

  // Prunes candidates in @pool to a shorter list @result
  // @pool must be sorted before calling
  void occlude_list(const uint32_t location, std::vector<neighbor_t>& pool, const float alpha,
                    const uint32_t degree, const uint32_t maxc, std::vector<uint32_t>& result,
                    in_mem_query_scratch_t<T>* scratch,
                    const tsl::robin_set<uint32_t>* const delete_set_ptr = nullptr);

  template <bool observe_rcni>
  void occlude_list_impl(const uint32_t location, std::vector<neighbor_t>& pool, const float alpha,
                         const uint32_t degree, const uint32_t maxc, std::vector<uint32_t>& result,
                         in_mem_query_scratch_t<T>* scratch,
                         std::vector<rcni_prune_event_t>* events,
                         const tsl::robin_set<uint32_t>* delete_set_ptr);

  // add reverse links from all the visited nodes to node n.
  void inter_insert(uint32_t n, std::vector<uint32_t>& pruned_list, const uint32_t range,
                    in_mem_query_scratch_t<T>* scratch);

  void inter_insert(uint32_t n, std::vector<uint32_t>& pruned_list,
                    in_mem_query_scratch_t<T>* scratch);

  // Acquire exclusive update_lock_ before calling
  void link();

  template <bool observe_rcni>
  void link_impl();

  void run_post_link_graph_observer();

  // Acquire exclusive tag_lock_ and delete_lock_ before calling
  int reserve_location();

  // Acquire exclusive tag_lock_ before calling
  size_t release_location(int location);
  size_t release_locations(const tsl::robin_set<uint32_t>& locations);

  // Resize the index when no slots are left for insertion.
  // Acquire exclusive update_lock_ and tag_lock_ before calling.
  void resize(size_t new_max_points);

  // Acquire unique lock on update_lock_, consolidate_lock_, tag_lock_
  // and delete_lock_ before calling these functions.
  // Renumber nodes, update tag and location maps and compact the
  // graph, mode = consolidated_order_ in case of lazy deletion and
  // compacted_order_ in case of eager deletion
  POWERLAWANN_DLLEXPORT void compact_data();
  POWERLAWANN_DLLEXPORT void compact_frozen_point();

  // Remove deleted nodes from adjacency list of node loc
  // Replace removed neighbors with second order neighbors.
  // Also acquires locks_[i] for i = loc and out-neighbors of loc.
  void process_delete(const tsl::robin_set<uint32_t>& old_delete_set, size_t loc,
                      const uint32_t range, const uint32_t maxc, const float alpha,
                      in_mem_query_scratch_t<T>* scratch);

  void initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l,
                                uint32_t r, uint32_t maxc, size_t dim);

  // Do not call without acquiring appropriate locks
  // call public member functions save and load to invoke these.
  POWERLAWANN_DLLEXPORT size_t save_graph(std::string filename);
  POWERLAWANN_DLLEXPORT size_t save_data(std::string filename);
  POWERLAWANN_DLLEXPORT size_t save_tags(std::string filename);
  POWERLAWANN_DLLEXPORT size_t save_delete_list(const std::string& filename);
#ifdef EXEC_ENV_OLS
  POWERLAWANN_DLLEXPORT size_t load_graph(aligned_file_reader_t& reader,
                                          size_t expected_num_points);
  POWERLAWANN_DLLEXPORT size_t load_data(aligned_file_reader_t& reader);
  POWERLAWANN_DLLEXPORT size_t load_tags(aligned_file_reader_t& reader);
  POWERLAWANN_DLLEXPORT size_t load_delete_set(aligned_file_reader_t& reader);
#else
  POWERLAWANN_DLLEXPORT size_t load_graph(const std::string filename, size_t expected_num_points);
  POWERLAWANN_DLLEXPORT size_t load_data(std::string filename0);
  POWERLAWANN_DLLEXPORT size_t load_tags(const std::string tag_file_name);
  POWERLAWANN_DLLEXPORT size_t load_delete_set(const std::string& filename);
#endif

private:
  // distance_t functions
  metric_t dist_metric_ = powerlaw_ann::L2;

  // Data
  std::shared_ptr<abstract_data_store_t<T>> data_store_;

  // Graph related data structures
  std::unique_ptr<abstract_graph_store_t> graph_store_;
  std::unique_ptr<vamana_build_stats_t> vamana_build_stats_;
  rcni_prune_observer_t* rcni_prune_observer_ = nullptr;
  post_link_graph_observer_t* post_link_graph_observer_ = nullptr;

  char* opt_graph_ = nullptr;

  // Dimensions
  size_t dim_ = 0;
  size_t nd_ = 0;         // number of active points i.e. existing in the graph
  size_t max_points_ = 0; // total number of points in given data set

  // num_frozen_pts_ is the number of points which are used as initial
  // candidates when iterating to closest point(s). These are not visible
  // externally and won't be returned by search. At least 1 frozen point is
  // needed for a dynamic index. The frozen points have consecutive locations.
  // See also start_ below.
  size_t num_frozen_pts_ = 0;
  size_t frozen_pts_used_ = 0;
  size_t node_size_;
  size_t data_len_;
  size_t neighbor_len_;

  //  Start point of the search. When num_frozen_pts_ is greater than zero,
  //  this is the location of the first frozen point. Otherwise, this is a
  //  location of one of the points in index.
  uint32_t start_ = 0;

  bool has_built_ = false;
  bool saturate_graph_ = false;
  bool save_as_one_file_ = false; // plan to support in next version
  bool dynamic_index_ = false;
  bool enable_tags_ = false;
  bool normalize_vecs = false; // Using normalied L2 for cosine.
  bool deletes_enabled_ = false;

  // Filter Support

  bool filtered_index_ = false;
  // Location to label is only updated during insert_point(), all other reads are protected by
  // default as a location can only be released at end of consolidate deletes
  std::vector<std::vector<label_t>> location_to_labels_;
  tsl::robin_set<label_t> labels_;
  std::string labels_file_;
  std::unordered_map<label_t, uint32_t> label_to_start_id_;
  std::unordered_map<uint32_t, uint32_t> medoid_counts_;

  bool use_universal_label_ = false;
  label_t universal_label_ = 0;
  uint32_t filter_indexing_queue_size_;
  std::unordered_map<std::string, label_t> label_map_;

  // Indexing parameters
  uint32_t indexing_queue_size_;
  uint32_t indexing_range_;
  uint32_t indexing_max_c_;
  float indexing_alpha_;
  uint32_t indexing_threads_;

  // Query scratch data structures
  concurrent_queue_t<in_mem_query_scratch_t<T>*> query_scratch_;

  // Flags for PQ based distance calculation
  bool pq_dist_ = false;
  bool use_opq_ = false;
  size_t num_pq_chunks_ = 0;
  // REFACTOR
  // uint8_t *pq_data_ = nullptr;
  std::shared_ptr<abstract_data_store_t<T>> pq_data_store_ = nullptr;
  bool pq_generated_ = false;

  //
  // Data structures, locks and flags for dynamic indexing and tags
  //

  // lazy_delete removes entry from location_to_tag_ and tag_to_location_. If
  // location_to_tag_ does not resolve a location, infer that it was deleted.
  tsl::sparse_map<tag_t, uint32_t> tag_to_location_;
  natural_number_map_t<uint32_t, tag_t> location_to_tag_;

  // empty_slots_ has unallocated slots and those freed by consolidate_delete.
  // delete_set_ has locations marked deleted by lazy_delete. Will not be
  // immediately available for insert. consolidate_delete will release these
  // slots to empty_slots_.
  natural_number_set_t<uint32_t> empty_slots_;
  std::unique_ptr<tsl::robin_set<uint32_t>> delete_set_;

  bool data_compacted_ = true;    // true if data has been compacted
  bool is_saved_ = false;         // Checking if the index is already saved.
  bool conc_consolidate_ = false; // use lock_ while searching

  // Acquire locks in the order below when acquiring multiple locks
  std::shared_timed_mutex // RW mutex between save/load (exclusive lock) and
      update_lock_;       // search/inserts/deletes/consolidate (shared lock)
  std::shared_timed_mutex // Ensure only one consolidate or compact_data is
      consolidate_lock_;  // ever active
  std::shared_timed_mutex // RW lock for tag_to_location_,
      tag_lock_;          // location_to_tag_, empty_slots_, nd_, max_points_, label_to_start_id_
  std::shared_timed_mutex // RW Lock on delete_set_ and data_compacted_
      delete_lock_;       // variable

  // Per node lock, cardinality=max_points_ + num_frozen_points_
  std::vector<non_recursive_mutex_t> locks_;

  static const float INDEX_GROWTH_FACTOR;
};
} // namespace powerlaw_ann

#endif // INDEX_VAMANA_INDEX
