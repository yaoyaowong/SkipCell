#ifndef INDEX_IO_OPTIMIZED_INDEX
#define INDEX_IO_OPTIMIZED_INDEX

#include "storage/io_lru_buffer_pool.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace powerlaw_ann {

class io_optimized_index_t;
struct community_polar_index_t;

inline constexpr uint32_t k_io_vector_page_size = 4096;
inline constexpr uint64_t k_no_io_prefetch_page = UINT64_MAX;

enum class io_adjacency_encoding_t : uint32_t {
  RAW_U32 = 0,
  PFOR_DELTA = 1,
};

enum class io_vector_layout_t : uint32_t {
  ORIGINAL_ID = 0,
  CELL_4K = 1,
  CELL_CHUNKS = 2,
  CELL_WEIGHTED_4K = 3,
  CELL_U8_4K = 4,
};

enum class io_memgraph_mode_t : uint32_t {
  NONE = 0,
  RANDOM = 1,
  COMMUNITY_IMPORTANT = 2,
  HIGH_DEGREE = 3,
  RCNI_ONLY = 4,
};

struct io_vector_location_t {
  uint64_t page_id = 0;
  uint32_t page_offset = 0;

  bool operator==(const io_vector_location_t&) const = default;
};

struct io_optimized_build_result_t {
  uint64_t point_count = 0;
  uint64_t edge_count = 0;
  uint64_t topology_bytes = 0;
  uint64_t adjacency_bytes = 0;
  uint64_t vector_bytes = 0;
  uint64_t vector_padding_bytes = 0;
  uint64_t build_peak_payload_bytes = 0;
  uint64_t cell_count = 0;
  uint64_t nonempty_vector_bytes = 0;
  uint64_t cell_page_count = 0;
  uint64_t gateway_prefetch_hint_count = 0;
  uint64_t gateway_prefetch_hint_bytes = 0;
  uint64_t low_rcni_trimmed_nodes = 0;
  std::array<uint64_t, 4> chunk_counts{};
};

struct io_optimization_config_t {
  bool enable_topology_vector = false;
  bool enable_paged_topology = false;
  uint64_t topology_cache_pages = 0;
  bool enable_cell_layout = false;
  bool enable_cell_u8_layout = false;
  bool enable_cell_page_batch = false;
  bool enable_cell_batch_search = false;
  uint32_t cell_batch_max_cached_expansions = 16;
  bool cell_batch_preserve_graph = false;
  uint32_t cell_batch_graph_stitch_min_l = 0;
  bool enable_ranked_cell_page_refinement = false;
  uint32_t ranked_cell_page_base_pages = 0;
  uint32_t ranked_cell_page_l_divisor = 50;
  uint32_t ranked_cell_page_growth_start_l = 0;
  uint32_t ranked_cell_page_growth_divisor = 0;
  uint32_t ranked_cell_page_max_pages = 4;
  uint32_t ranked_cell_page_max_l = std::numeric_limits<uint32_t>::max();
  bool enable_interleaved_cell_batch_search = false;
  uint32_t interleaved_cell_batch_graph_hops = 1;
  bool enable_cell_pq_traversal = false;
  uint32_t cell_pq_refine_candidates = 0;
  uint32_t cell_pq_refine_min_l = 1;
  uint32_t cell_pq_refine_prefetch_hop = 0;
  bool enable_cell_leaf_refinement = false;
  bool enable_resident_u8_refinement = false;
  bool enable_resident_u8_traversal = false;
  bool enable_cell_pq_dense_visited = false;
  bool enable_cell_pq_filter_visited = false;
  bool enable_cell_pq_decoded_u8 = false;
  float cell_pq_decoded_scale = 1.0F;
  bool enable_compressed_pq_traversal = false;
  uint32_t pq_score_chunks = 0;
  uint32_t pq_score_quantization_bits = 0;
  uint32_t graph_neighbor_score_limit = 0;
  bool enable_l_aware_search = false;
  uint32_t l_aware_decoded_max_l = 100;
  bool enable_l_aware_refine_budget = false;
  uint32_t l_aware_refine_base = 8;
  uint32_t l_aware_refine_divisor = 8;
  bool enable_forced_cell_tree_search = false;
  bool enable_cell_adj_correction = false;
  uint32_t cell_adj_candidate_cells = 4;
  uint32_t cell_adj_support_shortlist = 64;
  uint32_t cell_adj_expansion_cells = 1;
  uint32_t cell_adj_min_search_list_size = 1;
  uint32_t cell_adj_max_search_list_size = std::numeric_limits<uint32_t>::max();
  bool enable_online_cell_adaptation = false;
  bool enable_chunk_layout = false;
  bool enable_weighted_reorder = false;
  bool enable_lru_cache = false;
  uint64_t lru_cache_pages = 0;
  bool reset_lru_after_warmup = false;
  bool enable_io_uring = false;
  uint32_t io_uring_queue_depth = 128;
  bool enable_node_prefetch = false;
  bool enable_cell_prefetch = false;
  bool enable_gateway_prefetch = false;
  uint32_t gateway_prefetch_max_outstanding = 2;
  io_memgraph_mode_t memgraph_mode = io_memgraph_mode_t::NONE;
  uint32_t memgraph_nodes = 0;
  uint32_t memgraph_entry_candidates = 8;
  uint64_t memgraph_seed = 0;
  bool enable_dynamic_width = false;
  uint32_t dynamic_width_initial = 4;
  uint32_t dynamic_width_marker = 5;
  float dynamic_width_waste_threshold = 0.1F;
};

struct io_artifact_fingerprint_t {
  uint64_t size = 0;
  uint64_t checksum = 0;

  bool operator==(const io_artifact_fingerprint_t&) const = default;
};

/** Persisted physical-page prediction for one directed Community Gateway edge. */
struct io_gateway_prefetch_hint_t {
  uint32_t target_node = 0;
  uint32_t landing_cell = 0;
  std::array<uint32_t, 2> successor_cells{UINT32_MAX, UINT32_MAX};
  uint64_t landing_node_page = k_no_io_prefetch_page;
  uint64_t landing_cell_first_page = k_no_io_prefetch_page;
  std::array<uint64_t, 2> successor_first_pages{k_no_io_prefetch_page, k_no_io_prefetch_page};
  std::array<uint32_t, 2> transition_counts{};
  uint32_t outgoing_transition_count = 0;
  uint32_t reserved = 0;

  bool operator==(const io_gateway_prefetch_hint_t&) const = default;
};

struct io_cell_page_range_t {
  uint64_t first_page = 0;
  uint32_t page_count = 0;
  uint32_t node_count = 0;
  uint32_t chunk_size_bytes = k_io_vector_page_size;

  bool operator==(const io_cell_page_range_t&) const = default;
};

struct io_page_run_t {
  uint64_t first_page = 0;
  uint32_t page_count = 0;

  bool operator==(const io_page_run_t&) const = default;
};

struct io_global_graph_cell_assignment_t {
  std::vector<uint32_t> node_to_cell;
  uint32_t cell_count = 0;
};

struct io_cell_adjacency_build_result_t {
  uint64_t cell_count = 0;
  uint64_t edge_count = 0;
  uint64_t artifact_bytes = 0;
  uint32_t maximum_degree = 0;
  uint32_t sampled_nodes_per_cell = 0;
  uint32_t build_threads = 0;
};

/** Sort, deduplicate, and group physical page IDs into maximal consecutive runs. */
std::vector<io_page_run_t> make_io_page_runs(std::span<const uint64_t> page_ids);

/** Deterministically form bounded global Cells by BFS over the immutable resident topology. */
io_global_graph_cell_assignment_t
build_io_global_graph_cell_assignment(const io_optimized_index_t& index, uint32_t target_size);

/** Build the same diagnostic assignment from a raw topology artifact without resident copying. */
io_global_graph_cell_assignment_t
build_io_global_graph_cell_assignment_from_artifact(const std::filesystem::path& topology_path,
                                                    uint32_t target_size);

/** Resident immutable Vamana topology and logical-node vector-page directory. */
class io_optimized_index_t {
public:
  ~io_optimized_index_t();

  /** Build the IO-1 raw-adjacency/original-ID vector layout from an existing Base index. */
  static io_optimized_build_result_t
  build(const std::filesystem::path& index_path_prefix,
        const std::filesystem::path& topology_path = {},
        const std::filesystem::path& vector_path = {},
        io_adjacency_encoding_t encoding = io_adjacency_encoding_t::RAW_U32,
        io_vector_layout_t layout = io_vector_layout_t::ORIGINAL_ID,
        const std::filesystem::path& community_polar_path = {}, bool build_prefetch_hints = true,
        const std::filesystem::path& hub_clique_low_rcni_path = {});

  /**
   * Load the topology and validate its checksums and Base fingerprint.
   *
   * By default the vector artifact is also fingerprinted and exposed to the data reader. Set
   * topology_only when the caller can prove that the query path consumes only resident topology
   * and PQ records; in that mode vector metadata is validated but the vector artifact is neither
   * opened nor read.
   */
  static std::shared_ptr<const io_optimized_index_t>
  load(const std::filesystem::path& index_path_prefix,
       const std::filesystem::path& topology_path = {},
       const std::filesystem::path& vector_path = {}, bool topology_only = false,
       bool paged_topology = false, uint64_t topology_cache_pages = 0);

  std::span<const uint32_t> neighbors(uint32_t node_id) const;
  io_vector_location_t vector_location(uint32_t node_id) const;
  uint64_t node_prefetch_page(uint32_t node_id) const;
  const io_gateway_prefetch_hint_t& gateway_prefetch_hint(uint64_t gateway_id) const;

  uint64_t point_count() const noexcept { return point_count_; }
  uint64_t dimension() const noexcept { return dimension_; }
  uint32_t max_degree() const noexcept { return max_degree_; }
  uint64_t edge_count() const noexcept { return edge_count_; }
  uint64_t vector_page_count() const noexcept { return vector_page_count_; }
  io_vector_layout_t vector_layout() const noexcept { return vector_layout_; }
  std::span<const io_cell_page_range_t> cell_page_ranges() const noexcept {
    return cell_page_ranges_;
  }
  uint64_t resident_bytes() const noexcept;
  uint64_t artifact_bytes() const noexcept { return topology_bytes_ + vector_bytes_; }
  uint64_t topology_bytes() const noexcept { return topology_bytes_; }
  uint64_t vector_bytes() const noexcept { return vector_bytes_; }
  bool has_gateway_prefetch_hints() const noexcept { return !gateway_prefetch_hints_.empty(); }
  uint64_t gateway_prefetch_hint_count() const noexcept { return gateway_prefetch_hints_.size(); }
  io_artifact_fingerprint_t community_polar_fingerprint() const noexcept {
    return community_polar_fingerprint_;
  }
  const std::filesystem::path& vector_path() const noexcept { return vector_path_; }
  bool paged_topology() const noexcept { return paged_topology_; }
  uint64_t topology_cache_resident_bytes() const noexcept;
  io_lru_buffer_pool_stats_t topology_cache_stats() const;

private:
  uint64_t point_count_ = 0;
  uint64_t dimension_ = 0;
  uint32_t max_degree_ = 0;
  uint64_t edge_count_ = 0;
  uint64_t vector_page_count_ = 0;
  io_vector_layout_t vector_layout_ = io_vector_layout_t::ORIGINAL_ID;
  uint64_t topology_bytes_ = 0;
  uint64_t vector_bytes_ = 0;
  std::filesystem::path vector_path_;
  std::vector<uint64_t> neighbor_offsets_;
  std::vector<uint32_t> adjacency_;
  std::vector<io_vector_location_t> vector_locations_;
  std::vector<io_cell_page_range_t> cell_page_ranges_;
  std::vector<uint64_t> node_prefetch_pages_;
  io_artifact_fingerprint_t community_polar_fingerprint_;
  std::vector<io_gateway_prefetch_hint_t> gateway_prefetch_hints_;
  bool paged_topology_ = false;
  uint64_t adjacency_file_offset_ = 0;
  int topology_file_descriptor_ = -1;
  std::shared_ptr<io_lru_buffer_pool_t> topology_cache_;
};

/** Immutable bounded Cell graph derived from sampled Base edges, without vector duplication. */
class io_cell_adjacency_index_t {
public:
  static io_cell_adjacency_build_result_t
  build(const std::filesystem::path& path, const io_optimized_index_t& topology,
        const community_polar_index_t& cells, io_artifact_fingerprint_t community_polar_fingerprint,
        uint32_t sampled_nodes_per_cell, uint32_t maximum_degree, uint32_t num_threads = 0);

  static std::shared_ptr<const io_cell_adjacency_index_t>
  load(const std::filesystem::path& path, uint64_t expected_point_count,
       uint64_t expected_cell_count, io_artifact_fingerprint_t community_polar_fingerprint);

  std::span<const uint32_t> neighbors(uint32_t cell_id) const;
  uint64_t point_count() const noexcept { return point_count_; }
  uint64_t cell_count() const noexcept { return cell_count_; }
  uint64_t edge_count() const noexcept { return adjacency_.size(); }
  uint32_t maximum_degree() const noexcept { return maximum_degree_; }
  uint32_t sampled_nodes_per_cell() const noexcept { return sampled_nodes_per_cell_; }
  uint64_t resident_bytes() const noexcept;
  uint64_t artifact_bytes() const noexcept { return artifact_bytes_; }

private:
  uint64_t point_count_ = 0;
  uint64_t cell_count_ = 0;
  uint32_t maximum_degree_ = 0;
  uint32_t sampled_nodes_per_cell_ = 0;
  uint64_t artifact_bytes_ = 0;
  std::vector<uint64_t> offsets_;
  std::vector<uint32_t> adjacency_;
};

std::filesystem::path make_io_topology_path(const std::filesystem::path& index_path_prefix);
std::filesystem::path make_io_vector_path(const std::filesystem::path& index_path_prefix);
std::filesystem::path make_io_cell_adjacency_path(const std::filesystem::path& index_path_prefix);

/** Rebind a version-5 IO topology to a structurally identical Community-Polar sidecar. */
void rebind_io_topology_community_polar(
    const std::filesystem::path& source_topology_path,
    const std::filesystem::path& source_community_polar_path,
    const std::filesystem::path& replacement_community_polar_path,
    const std::filesystem::path& output_topology_path);

} // namespace powerlaw_ann

#endif // INDEX_IO_OPTIMIZED_INDEX
