#ifndef INDEX_IO_ADJACENCY_STATISTICS
#define INDEX_IO_ADJACENCY_STATISTICS

#include <cstdint>
#include <filesystem>

namespace powerlaw_ann {

/** Byte accounting for one statistics-only compressed-adjacency artifact. */
struct io_adjacency_statistics_result_t {
  uint64_t point_count = 0;
  uint64_t edge_count = 0;
  uint32_t maximum_degree = 0;
  uint64_t raw_adjacency_bytes = 0;
  uint64_t compressed_offset_bytes = 0;
  uint64_t compressed_payload_bytes = 0;
  uint64_t permutation_bytes = 0;
  uint64_t artifact_bytes = 0;
  uint64_t build_peak_payload_bytes = 0;
  uint64_t artifact_checksum = 0;
  bool preserves_neighbor_order = false;
};

/**
 * Build a deterministic statistics-only Delta+GroupVarint adjacency sidecar.
 *
 * The input must be a checksummed raw-U32 IO topology artifact. The output has an independent
 * magic and therefore cannot be mistaken for a queryable IO topology. When neighbor order is not
 * preserved, the result is only a sorted-adjacency space lower bound. When it is preserved, a
 * bit-packed permutation permits exact reconstruction of the immutable Base neighbor order.
 */
io_adjacency_statistics_result_t
build_io_groupvarint_adjacency_statistics(const std::filesystem::path& raw_topology_path,
                                          const std::filesystem::path& statistics_path,
                                          bool preserve_neighbor_order);

/** Validate checksums and fully replay a statistics sidecar against its bound raw topology. */
io_adjacency_statistics_result_t
verify_io_groupvarint_adjacency_statistics(const std::filesystem::path& raw_topology_path,
                                           const std::filesystem::path& statistics_path);

} // namespace powerlaw_ann

#endif // INDEX_IO_ADJACENCY_STATISTICS
