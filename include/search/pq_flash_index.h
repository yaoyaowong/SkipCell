#ifndef SEARCH_PQ_FLASH_INDEX
#define SEARCH_PQ_FLASH_INDEX

#include "common/concurrent_queue.h"
#include "common/distance.h"
#include "common/scratch.h"
#include "index/adaptive_cell_runtime.h"
#include "index/io_optimized_index.h"
#include "pq/product_quantizer.h"
#include "search/community_polar_search.h"
#include "search/node_visit_tracker.h"
#include "search/percentile_stats.h"
#include "storage/aligned_file_reader.h"
#include "storage/io_lru_buffer_pool.h"
#include "third/tsl/robin_map.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace powerlaw_ann {

/**
 * @brief DiskANN-compatible PQ disk index and cached beam search.
 *
 * This Step 8 surface implements the approved static, unfiltered float32 L2
 * baseline. It owns the in-memory PQ codes, disk-layout metadata, optional BFS
 * cache, and per-thread query scratch. The aligned reader is selected before
 * loading, so the candidate-expansion loop contains no platform branch.
 */
template <typename data_t>
class pq_flash_index_t {
public:
  explicit pq_flash_index_t(std::shared_ptr<aligned_file_reader_t> reader,
                            metric_t metric = metric_t::L2);
  ~pq_flash_index_t();

  pq_flash_index_t(const pq_flash_index_t&) = delete;
  pq_flash_index_t& operator=(const pq_flash_index_t&) = delete;

  /** @brief Loads PQ data and opens the DiskANN disk-index artifact. */
  int load(uint32_t num_threads, const char* index_prefix,
           std::shared_ptr<const io_optimized_index_t> io_index = nullptr,
           std::shared_ptr<io_lru_buffer_pool_t> io_cache = nullptr,
           bool enable_cell_page_batch = false, bool enable_cell_batch_search = false,
           uint32_t cell_batch_max_cached_expansions = 16, bool enable_cell_pq_traversal = false,
           uint32_t cell_pq_refine_candidates = 0, uint32_t cell_pq_refine_prefetch_hop = 0,
           bool enable_cell_leaf_refinement = false, bool enable_cell_pq_dense_visited = false,
           bool enable_cell_pq_filter_visited = false, bool enable_cell_pq_decoded_u8 = false,
           bool enable_node_prefetch = false, bool enable_cell_prefetch = false,
           bool enable_gateway_prefetch = false, uint32_t gateway_prefetch_max_outstanding = 2,
           bool enable_dynamic_width = false, uint32_t dynamic_width_initial = 4,
           uint32_t dynamic_width_marker = 5, float dynamic_width_waste_threshold = 0.1F,
           float cell_pq_decoded_scale = 1.0F);

  /** Configure deterministic resident MemGraph entry nodes before queries begin. */
  void configure_memgraph(std::vector<uint32_t> nodes, uint32_t entry_candidates);
  /** Select a bounded exact-refinement prefix from L without changing traversal termination. */
  void configure_l_aware_refinement(uint32_t base_candidates, uint32_t l_divisor);
  /** Defer raw-vector candidate refinement until the requested search-list size. */
  void configure_cell_pq_refine_min_l(uint32_t minimum_search_list_size);
  /** Build the packet-order decoded representation used only by forced Cell-tree terminal scans. */
  void configure_forced_cell_tree(
      std::shared_ptr<const community_polar_search_t> community_polar_search);
  /** Enable query-local cross-Community Cell adjacency correction from resident Base topology. */
  void configure_cell_adj_correction(
      uint32_t candidate_cells, uint32_t support_shortlist,
      std::shared_ptr<const io_cell_adjacency_index_t> cell_adjacency = nullptr,
      uint32_t expansion_cells = 1,
      uint32_t min_search_list_size = 1,
      uint32_t max_search_list_size = std::numeric_limits<uint32_t>::max());
  /** Alternate bounded topology/PQ graph probes with one whole-Cell read at a time. */
  void configure_interleaved_cell_batch_search(uint32_t graph_hops);
  /** Keep a terminal Cell batch out of the already-converged Base frontier. */
  void configure_cell_batch_preserve_graph();
  /** Reopen graph stitching from exact Cell nodes only at or above the specified L. */
  void configure_cell_batch_graph_stitch(uint32_t minimum_search_list_size);
  /** Read an L-aware budget of the highest-yield physical pages selected by the Cell forest. */
  void configure_ranked_cell_page_refinement(uint32_t base_pages, uint32_t l_divisor,
                                             uint32_t growth_start_l,
                                             uint32_t growth_divisor, uint32_t max_pages,
                                             uint32_t max_search_list_size);
  /** Score graph candidates with an evenly spaced subset of the loaded PQ chunks. */
  void configure_pq_score_chunks(uint32_t chunk_count);
  /** Quantize the complete query-local PQ lookup table without changing the PQ codes. */
  void configure_pq_score_quantization(uint32_t bits);
  /** Bound scored neighbors per Base expansion while preserving stored adjacency. */
  void configure_graph_neighbor_score_limit(uint32_t limit);
  /** Load and validate an original-ID uint8 matrix used only for exact final refinement. */
  void configure_resident_u8_refinement(const std::filesystem::path& path,
                                        bool enable_traversal = false);
  void configure_adaptive_cell_runtime(
      std::shared_ptr<adaptive_cell_runtime_t> adaptive_cell_runtime) noexcept {
    adaptive_cell_runtime_ = std::move(adaptive_cell_runtime);
  }
  uint64_t memgraph_resident_bytes() const noexcept {
    return memgraph_nodes_.capacity() * sizeof(uint32_t) +
           memgraph_adjacency_.capacity() * sizeof(uint32_t);
  }
  uint32_t memgraph_node_count() const noexcept {
    return static_cast<uint32_t>(memgraph_nodes_.size());
  }
  uint64_t memgraph_build_time_us() const noexcept { return memgraph_build_time_us_; }
  uint64_t memgraph_build_peak_bytes() const noexcept { return memgraph_build_peak_bytes_; }
  uint64_t decoded_pq_build_time_us() const noexcept { return decoded_pq_build_time_us_; }
  uint64_t decoded_pq_representation_bytes() const noexcept {
    return enable_cell_pq_decoded_u8_ ? num_points_ * data_dim_ : 0;
  }
  uint64_t decoded_pq_additional_resident_bytes() const noexcept {
    return (decoded_node_pq_codes_.capacity() + decoded_cell_pq_codes_.capacity()) *
           sizeof(uint8_t);
  }
  uint64_t resident_u8_refinement_artifact_bytes() const noexcept {
    return resident_u8_refinement_artifact_bytes_;
  }
  uint64_t resident_u8_refinement_bytes() const noexcept {
    return resident_u8_refinement_.capacity() * sizeof(uint8_t);
  }
  uint64_t resident_u8_refinement_checksum() const noexcept {
    return resident_u8_refinement_checksum_;
  }

  /** @brief Selects up to `num_nodes_to_cache` nodes using deterministic BFS. */
  void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t>& node_list);

  /** @brief Reads the selected nodes into the neighborhood and coordinate caches. */
  void load_cache_list(const std::vector<uint32_t>& node_list);

  /**
   * @brief Runs the pinned DiskANN cached beam-search algorithm.
   * @param query Query vector with the index's original dimension.
   * @param top_k Number of sorted results to return.
   * @param search_list_size Candidate-list capacity; must be at least `top_k`.
   * @param indices Caller-owned output array of `top_k` IDs.
   * @param distances Optional caller-owned output array of `top_k` L2 distances.
   * @param beam_width Maximum number of uncached frontier nodes read per hop.
   * @param io_limit Maximum disk reads for this query.
   * @param use_reorder_data Re-rank with stored full-precision vectors.
   * @param stats Optional per-query counters.
   */
  void cached_beam_search(const data_t* query, uint64_t top_k, uint64_t search_list_size,
                          uint64_t* indices, float* distances, uint64_t beam_width,
                          uint32_t io_limit, bool use_reorder_data = false,
                          query_stats_t* stats = nullptr);

  /** Run unchanged Base semantics against the explicit IO-1 topology/vector layout. */
  void cached_beam_search_io_optimized(const data_t* query, uint64_t top_k,
                                       uint64_t search_list_size, uint64_t* indices,
                                       float* distances, uint64_t beam_width, uint32_t io_limit,
                                       query_stats_t* stats = nullptr);

  void cached_beam_search(const data_t* query, uint64_t top_k, uint64_t search_list_size,
                          uint64_t* indices, float* distances, uint64_t beam_width,
                          bool use_reorder_data = false, query_stats_t* stats = nullptr);

  /** @brief Runs the explicit shared-frontier Community-Polar hybrid specialization. */
  void cached_beam_search_hybrid(const data_t* query, uint64_t top_k, uint64_t search_list_size,
                                 uint64_t* indices, float* distances, uint64_t beam_width,
                                 uint32_t io_limit, const community_polar_search_t& hybrid_search,
                                 bool use_reorder_data = false, query_stats_t* stats = nullptr);

  /** Run explicit Hybrid semantics against the explicit IO-1 topology/vector layout. */
  void cached_beam_search_hybrid_io_optimized(const data_t* query, uint64_t top_k,
                                              uint64_t search_list_size, uint64_t* indices,
                                              float* distances, uint64_t beam_width,
                                              uint32_t io_limit,
                                              const community_polar_search_t& hybrid_search,
                                              query_stats_t* stats = nullptr);

  /** @brief Scores identical macro proposals without mutating the Base frontier or results. */
  void cached_beam_search_observe_only(const data_t* query, uint64_t top_k,
                                       uint64_t search_list_size, uint64_t* indices,
                                       float* distances, uint64_t beam_width, uint32_t io_limit,
                                       const community_polar_search_t& hybrid_search,
                                       bool use_reorder_data = false,
                                       query_stats_t* stats = nullptr);

  /** @brief Runs the diagnostic PipeANN-proxy phase profiler on unchanged Base search. */
  void cached_beam_search_phase_profiled(const data_t* query, uint64_t top_k,
                                         uint64_t search_list_size, uint64_t* indices,
                                         float* distances, uint64_t beam_width, uint32_t io_limit,
                                         bool use_reorder_data = false,
                                         query_stats_t* stats = nullptr);

  /** Run the diagnostic phase profiler on explicit IO-1 Base execution. */
  void cached_beam_search_io_optimized_phase_profiled(const data_t* query, uint64_t top_k,
                                                      uint64_t search_list_size, uint64_t* indices,
                                                      float* distances, uint64_t beam_width,
                                                      uint32_t io_limit,
                                                      query_stats_t* stats = nullptr);

  /** @brief Runs the same diagnostic phase profiler around explicit hybrid search. */
  void cached_beam_search_hybrid_phase_profiled(const data_t* query, uint64_t top_k,
                                                uint64_t search_list_size, uint64_t* indices,
                                                float* distances, uint64_t beam_width,
                                                uint32_t io_limit,
                                                const community_polar_search_t& hybrid_search,
                                                bool use_reorder_data = false,
                                                query_stats_t* stats = nullptr);

  /** Run the diagnostic phase profiler on explicit IO-1 Hybrid execution. */
  void cached_beam_search_hybrid_io_optimized_phase_profiled(
      const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
      float* distances, uint64_t beam_width, uint32_t io_limit,
      const community_polar_search_t& hybrid_search, query_stats_t* stats = nullptr);

  uint64_t get_data_dim() const { return data_dim_; }
  uint64_t get_num_points() const { return num_points_; }
  uint64_t get_max_degree() const { return max_degree_; }

  void start_node_visit_tracking();
  void stop_node_visit_tracking();
  void dump_node_visit_counts(const std::string& output_path) const;

private:
  template <bool use_hybrid, bool insert_macro, bool track_beam_phase, bool use_io_layout,
            bool forced_terminal_tree>
  void cached_beam_search_impl(const data_t* query, uint64_t top_k, uint64_t search_list_size,
                               uint64_t* indices, float* distances, uint64_t beam_width,
                               uint32_t io_limit, bool use_reorder_data, query_stats_t* stats,
                               const community_polar_search_t* hybrid_search);

  void setup_thread_data(uint64_t num_threads, uint64_t visited_reserve = 4096,
                         uint64_t candidate_capacity = 0);
  void use_medoids_data_as_centroids();
  std::vector<bool> read_nodes(const std::vector<uint32_t>& node_ids,
                               std::vector<data_t*>& coord_buffers,
                               std::vector<std::pair<uint32_t, uint32_t*>>& neighbor_buffers);

  uint64_t get_node_sector(uint64_t node_id) const;
  char* offset_to_node(char* sector_buffer, uint64_t node_id) const;
  uint32_t* offset_to_node_neighborhood(char* node_buffer) const;
  data_t* offset_to_node_coords(char* node_buffer) const;

  std::shared_ptr<aligned_file_reader_t> reader_;
  std::shared_ptr<const io_optimized_index_t> io_index_;
  std::shared_ptr<io_lru_buffer_pool_t> io_cache_;
  std::shared_ptr<adaptive_cell_runtime_t> adaptive_cell_runtime_;
  bool enable_cell_page_batch_ = false;
  bool enable_cell_batch_search_ = false;
  uint32_t cell_batch_max_cached_expansions_ = 16;
  bool cell_batch_preserve_graph_ = false;
  uint32_t cell_batch_graph_stitch_min_l_ = 0;
  bool enable_ranked_cell_page_refinement_ = false;
  uint32_t ranked_cell_page_base_pages_ = 0;
  uint32_t ranked_cell_page_l_divisor_ = 1;
  uint32_t ranked_cell_page_growth_start_l_ = 0;
  uint32_t ranked_cell_page_growth_divisor_ = 0;
  uint32_t ranked_cell_page_max_pages_ = 1;
  uint32_t ranked_cell_page_max_l_ = std::numeric_limits<uint32_t>::max();
  bool enable_interleaved_cell_batch_search_ = false;
  uint32_t interleaved_cell_batch_graph_hops_ = 1;
  bool enable_cell_pq_traversal_ = false;
  std::vector<uint32_t> pq_score_chunk_indices_;
  uint32_t pq_score_quantization_bits_ = 0;
  uint32_t graph_neighbor_score_limit_ = 0;
  uint32_t cell_pq_refine_candidates_ = 0;
  uint32_t cell_pq_refine_min_l_ = 1;
  uint32_t cell_pq_refine_prefetch_hop_ = 0;
  bool enable_l_aware_refine_budget_ = false;
  uint32_t l_aware_refine_base_ = 0;
  uint32_t l_aware_refine_divisor_ = 1;
  bool enable_cell_leaf_refinement_ = false;
  bool enable_cell_pq_dense_visited_ = false;
  bool enable_cell_pq_filter_visited_ = false;
  bool enable_cell_pq_decoded_u8_ = false;
  float cell_pq_decoded_scale_ = 1.0F;
  uint64_t decoded_pq_build_time_us_ = 0;
  std::vector<uint8_t> decoded_node_pq_codes_;
  std::vector<uint8_t> decoded_cell_pq_codes_;
  std::vector<uint8_t> resident_u8_refinement_;
  uint64_t resident_u8_refinement_artifact_bytes_ = 0;
  uint64_t resident_u8_refinement_checksum_ = 0;
  bool enable_resident_u8_traversal_ = false;
  bool forced_cell_tree_configured_ = false;
  bool enable_cell_adj_correction_ = false;
  uint32_t cell_adj_candidate_cells_ = 0;
  uint32_t cell_adj_support_shortlist_ = 0;
  uint32_t cell_adj_expansion_cells_ = 1;
  uint32_t cell_adj_min_search_list_size_ = 1;
  uint32_t cell_adj_max_search_list_size_ = std::numeric_limits<uint32_t>::max();
  std::shared_ptr<const io_cell_adjacency_index_t> cell_adjacency_index_;
  bool enable_node_prefetch_ = false;
  bool enable_cell_prefetch_ = false;
  bool enable_gateway_prefetch_ = false;
  uint32_t gateway_prefetch_max_outstanding_ = 2;
  std::vector<uint32_t> memgraph_nodes_;
  std::vector<uint32_t> memgraph_adjacency_;
  uint32_t memgraph_degree_ = 0;
  uint32_t memgraph_entry_candidates_ = 0;
  uint64_t memgraph_build_time_us_ = 0;
  uint64_t memgraph_build_peak_bytes_ = 0;
  bool enable_dynamic_width_ = false;
  uint32_t dynamic_width_initial_ = 4;
  uint32_t dynamic_width_marker_ = 5;
  float dynamic_width_waste_threshold_ = 0.1F;
  metric_t metric_ = metric_t::L2;
  std::shared_ptr<distance_t<data_t>> distance_comparator_;
  std::shared_ptr<distance_t<float>> float_distance_comparator_;

  uint64_t max_node_len_ = 0;
  uint64_t nodes_per_sector_ = 0;
  uint64_t max_degree_ = 0;
  uint64_t reorder_dimensions_ = 0;
  uint64_t reorder_data_start_sector_ = 0;
  uint64_t vectors_per_sector_ = 0;
  uint64_t num_points_ = 0;
  uint64_t num_frozen_points_ = 0;
  uint64_t frozen_location_ = 0;
  uint64_t data_dim_ = 0;
  uint64_t aligned_dim_ = 0;
  uint64_t disk_bytes_per_point_ = 0;
  std::string disk_index_file_;

  uint8_t* data_ = nullptr;
  uint64_t num_chunks_ = 0;
  fixed_chunk_pq_table_t pq_table_;
  bool use_disk_index_pq_ = false;
  uint64_t disk_pq_num_chunks_ = 0;
  fixed_chunk_pq_table_t disk_pq_table_;

  uint32_t* medoids_ = nullptr;
  size_t num_medoids_ = 0;
  float* centroid_data_ = nullptr;

  uint32_t* neighborhood_cache_buffer_ = nullptr;
  tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t*>> neighborhood_cache_;
  data_t* coordinate_cache_buffer_ = nullptr;
  tsl::robin_map<uint32_t, data_t*> coordinate_cache_;

  concurrent_queue_t<ssd_thread_data_t<data_t>*> thread_data_{nullptr};
  bool is_loaded_ = false;
  bool reorder_data_exists_ = false;
  node_visit_tracker_t node_visit_tracker_;
};

extern template class pq_flash_index_t<float>;

} // namespace powerlaw_ann

#endif // SEARCH_PQ_FLASH_INDEX
