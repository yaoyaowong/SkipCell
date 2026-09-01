#ifndef INDEX_COMMUNITY_POLAR_INDEX
#define INDEX_COMMUNITY_POLAR_INDEX

#include "index/post_link_graph.h"
#include "index/rcni.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace powerlaw_ann {

/** @brief Selects the KaMinPar construction context for one Community partition. */
enum class community_partition_quality_t {
  FAST,
  QUALITY,
};

/** @brief Selects the immutable final-Vamana projection used only for Community partitioning. */
enum class community_partition_projection_t {
  FULL_RECIPROCAL_VAMANA,
  LOCAL_VAMANA_K8,
  LOCAL_VAMANA_K16,
};

/** @brief Selects the Cell construction used by the sidecar. */
enum class community_cell_partition_t {
  POLAR_RADIAL,
  GRAPH_LOCAL,
  CONVERGENCE_COACCESS,
  GLOBAL_GEOMETRIC,
  GLOBAL_GRAPH_LOCAL,
};

/** @brief Configures the immutable Community-Polar sidecar built after Vamana linking. */
struct community_polar_build_config_t {
  uint32_t community_count = 0;
  double community_imbalance = 0.05;
  community_partition_quality_t partition_quality = community_partition_quality_t::FAST;
  community_partition_projection_t partition_projection =
      community_partition_projection_t::FULL_RECIPROCAL_VAMANA;
  uint32_t partition_threads = 0;
  int32_t partition_seed = 0;
  uint32_t gateway_count = 8;
  uint32_t gateway_shortlist = 32;
  uint32_t polar_direction_count = 16;
  uint32_t cell_target_size = 256;
  uint32_t block_target_size = 64;
  uint32_t spherical_kmeans_iterations = 20;
  community_cell_partition_t cell_partition = community_cell_partition_t::POLAR_RADIAL;
  /** Build-only policy that persists a deterministic hierarchy over the final Cell order. */
  bool build_contiguous_cell_hierarchy = false;
  uint32_t cell_hierarchy_branching = 4;
  uint32_t cell_hierarchy_leaf_size = 16;
  bool adaptive_multi_capacity = false;
  uint32_t adaptive_policy_version = 0;
  uint32_t adaptive_page_bytes = 4096;
  uint32_t adaptive_vector_bytes = 0;
  uint64_t adaptive_lru_pages = 0;
  double adaptive_maximum_normalized_rms_radius = 1.25;
  double adaptive_maximum_normalized_p95_radius = 1.25;
  double adaptive_maximum_split_gain = 0.15;
  double adaptive_minimum_centroid_radius_overlap = 0.50;
  double adaptive_minimum_train_cross_child_coaccess = 0.50;
  double adaptive_maximum_unseen_pq_increase = 0.10;
  std::filesystem::path convergence_cell_profile_path;
  std::filesystem::path convergence_cell_hierarchy_path;
  std::filesystem::path convergence_cell_capacity_path;
  std::filesystem::path convergence_cell_source_path;
  std::filesystem::path gateway_selection_path;
  std::filesystem::path gateway_selection_source_path;

  /** @brief Reports whether Community-Polar construction was explicitly requested. */
  bool is_enabled() const noexcept { return community_count != 0; }
};

/** @brief Stores one file fingerprint used to bind the sidecar to Base and PQ artifacts. */
struct community_polar_file_fingerprint_t {
  uint64_t size_bytes = 0;
  uint64_t fnv1a_hash = 0;
};

/** @brief Stores one canonically numbered Community and its geometric/navigation metadata. */
struct community_polar_community_t {
  uint32_t community_id = 0;
  uint64_t node_count = 0;
  uint64_t packet_bytes = 0;
  uint64_t internal_directed_edge_count = 0;
  uint64_t incoming_directed_edge_count = 0;
  uint64_t outgoing_directed_edge_count = 0;
  uint32_t minimum_node_id = 0;
  uint32_t navigation_hub_id = 0;
  float radius = 0.0F;
  bool graph_only = false;
  uint64_t sector_begin = 0;
  uint32_t sector_count = 0;
  uint64_t cell_begin = 0;
  uint32_t cell_count = 0;
};

/** @brief Stores one directed Community superedge and its Gateway range. */
struct community_polar_edge_t {
  uint32_t source_community = 0;
  uint32_t target_community = 0;
  uint64_t directed_cross_edge_count = 0;
  uint64_t gateway_begin = 0;
  uint32_t gateway_count = 0;
};

/** @brief Stores one real cross-edge target selected as a query-time Gateway. */
struct community_polar_gateway_t {
  uint32_t source_node = 0;
  uint32_t target_node = 0;
  uint64_t packet_record_index = 0;
};

/** @brief Stores one nonempty direction Sector or the deterministic Pole Sector. */
struct community_polar_sector_t {
  uint32_t community_id = 0;
  uint32_t sector_id = 0;
  uint64_t node_count = 0;
  uint64_t axis_index = 0;
  float minimum_radius = 0.0F;
  float maximum_radius = 0.0F;
  float maximum_angle = 0.0F;
  uint64_t cell_begin = 0;
  uint32_t cell_count = 0;
};

/** @brief Stores one nonempty Community-local radial Cell. */
struct community_polar_cell_t {
  uint32_t cell_id = 0;
  uint32_t community_id = 0;
  uint32_t sector_id = 0;
  uint32_t radial_cell_id = 0;
  uint64_t node_count = 0;
  uint32_t capacity_class = 0;
  float minimum_radius = 0.0F;
  float maximum_radius = 0.0F;
  float maximum_angle = 0.0F;
  uint64_t block_begin = 0;
  uint32_t block_count = 0;
};

/** @brief Stores one bounded sequential packet Block and its conservative PQ error envelope. */
struct community_polar_block_t {
  uint32_t block_id = 0;
  uint32_t cell_id = 0;
  uint32_t node_count = 0;
  uint64_t packet_record_begin = 0;
  float minimum_radius = 0.0F;
  float maximum_radius = 0.0F;
  float maximum_angle = 0.0F;
  float maximum_pq_reconstruction_error = 0.0F;
};

/** @brief Stores one persisted coarse-to-fine Cell routing node. */
struct community_polar_cell_hierarchy_node_t {
  uint32_t node_id = 0;
  uint32_t child_begin = 0;
  uint32_t child_count = 0;
  uint64_t descendant_node_count = 0;
  float radius = 0.0F;
  bool children_are_cells = false;
};

/**
 * @brief Owns the immutable, validated Community-Polar index loaded by future query execution.
 *
 * `packet_payload` stores fixed-width records in Block order. Each record contains one
 * little-endian uint32 node ID followed by `pq_code_width` bytes copied from the DiskANN PQ
 * artifact.
 */
struct community_polar_index_t {
  community_polar_build_config_t config;
  uint64_t point_count = 0;
  uint32_t dimension = 0;
  uint32_t entry_point_id = 0;
  uint32_t pq_code_width = 0;
  uint64_t directed_base_edge_count = 0;
  uint64_t undirected_partition_edge_count = 0;
  uint64_t weighted_edge_cut = 0;
  community_polar_file_fingerprint_t base_index_fingerprint;
  community_polar_file_fingerprint_t pq_fingerprint;
  std::vector<uint32_t> node_to_community;
  std::vector<uint32_t> node_to_cell;
  std::vector<uint64_t> node_to_packet_record;
  std::vector<community_polar_community_t> communities;
  std::vector<float> community_poles;
  std::vector<uint64_t> community_edge_offsets;
  std::vector<community_polar_edge_t> community_edges;
  std::vector<community_polar_gateway_t> gateways;
  std::vector<float> direction_axes;
  std::vector<community_polar_sector_t> sectors;
  std::vector<community_polar_cell_t> cells;
  std::vector<community_polar_block_t> blocks;
  std::vector<community_polar_cell_hierarchy_node_t> cell_hierarchy_nodes;
  std::vector<uint32_t> cell_hierarchy_roots;
  std::vector<uint32_t> cell_hierarchy_children;
  std::vector<float> cell_hierarchy_centroids;
  std::vector<uint8_t> packet_payload;
};

/**
 * @brief Holds the offline index plus transient vectors and packet order needed for PQ finalize.
 *
 * The transient arrays are intentionally not serialized and may use substantial build memory.
 */
struct community_polar_build_result_t {
  community_polar_index_t index;
  std::vector<float> original_vectors;
  std::vector<uint32_t> packet_node_ids;
};

/** @brief Validates a requested Community-Polar build configuration. */
void validate_community_polar_build_config(const community_polar_build_config_t& config);

/**
 * @brief Builds Communities, directed Gateways, and Community-local Polar Cells from final Vamana.
 * @param graph Callback-scoped immutable final Base graph.
 * @param importance One RCNI record for every Base node.
 * @param config Valid enabled construction configuration.
 * @return Offline index state awaiting PQ packet finalization.
 */
community_polar_build_result_t
build_community_polar_index(const post_link_graph_view_t& graph,
                            std::span<const rcni_node_importance_t> importance,
                            const community_polar_build_config_t& config);

/** Deterministically pack adjacent graph-local Cells without reading Base vectors or adjacency. */
void repack_graph_local_cells(community_polar_index_t& index,
                              const community_polar_build_config_t& config);

/** Repack a source sidecar from one checksummed PQ-locality Cell order without reading Raw Vector. */
void repack_pq_locality_cells(community_polar_index_t& index,
                              const std::filesystem::path& profile_path,
                              const community_polar_build_config_t& config);

/**
 * @brief Finalizes PQ packets, computes Block error envelopes, fingerprints Base artifacts, and
 * atomically writes `<index-prefix>.community_polar.bin`.
 */
void finalize_community_polar_index(community_polar_build_result_t& result,
                                    const std::filesystem::path& index_path_prefix);

/**
 * @brief Finalizes against one immutable Base prefix and writes an explicit sidecar path.
 *
 * This overload supports construction ablations without replacing the canonical sidecar or any
 * Base DiskANN artifact.
 */
void finalize_community_polar_index(community_polar_build_result_t& result,
                                    const std::filesystem::path& index_path_prefix,
                                    const std::filesystem::path& output_sidecar_path);

/** Rebind immutable Cell packets to another PQ artifact without rebuilding topology or Cells. */
void rebind_community_polar_pq(const std::filesystem::path& source_sidecar_path,
                               const std::filesystem::path& index_path_prefix,
                               const std::filesystem::path& output_sidecar_path);

/**
 * @brief Rebuilds only a Community-Polar sidecar from existing Base, PQ, data, and RCNI artifacts.
 *
 * The function reads the immutable final graph from `<index-prefix>_disk.index`; it never invokes
 * Vamana construction and never writes any Base or PQ artifact.
 */
void build_community_polar_sidecar_from_disk(const std::filesystem::path& data_path,
                                             const std::filesystem::path& index_path_prefix,
                                             const std::filesystem::path& rcni_path,
                                             const std::filesystem::path& output_sidecar_path,
                                             const community_polar_build_config_t& config);

/** @brief Returns the sidecar path derived from one DiskANN index prefix. */
std::filesystem::path
make_community_polar_index_path(const std::filesystem::path& index_path_prefix);

/** @brief Writes a finalized versioned little-endian Community-Polar sidecar atomically. */
void write_community_polar_index(const std::filesystem::path& path,
                                 const community_polar_index_t& index);

/** @brief Loads and fully validates one Community-Polar sidecar. */
community_polar_index_t load_community_polar_index(const std::filesystem::path& path);

/** @brief Validates that a loaded sidecar still matches its Base disk and compressed-PQ files. */
void validate_community_polar_artifacts(const community_polar_index_t& index,
                                        const std::filesystem::path& index_path_prefix);

/**
 * @brief Collects one Community-Polar build from the post-link callback and finalizes it later.
 *
 * The RCNI aggregator must outlive this collector. `finalize()` is called only after DiskANN has
 * published the Base disk and PQ artifacts.
 */
class community_polar_collector_t final : public post_link_graph_observer_t {
public:
  community_polar_collector_t(const rcni_aggregator_t& aggregator,
                              const community_polar_build_config_t& config);

  void observe(const post_link_graph_view_t& graph) override;

  /** @brief Finalizes and publishes the observed sidecar. */
  void finalize(const std::filesystem::path& index_path_prefix);

  /** @brief Returns the finalized or in-progress index for audit and testing. */
  const community_polar_index_t& index() const;

private:
  const rcni_aggregator_t& aggregator_;
  community_polar_build_config_t config_;
  std::optional<community_polar_build_result_t> result_;
};

} // namespace powerlaw_ann

#endif // INDEX_COMMUNITY_POLAR_INDEX
