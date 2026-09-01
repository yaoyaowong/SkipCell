#ifndef INDEX_INDEX_LAYOUT
#define INDEX_INDEX_LAYOUT

// index_layout_t<T>  —  analogous to PipeANN's SSDIndexMetadata<T>.
//
// Single source of truth for all slot-size and chunk-type calculations.
// Both the build pipeline and the search path share this struct.
//
// PipeANN pattern mirrored:
//   - Disk-resident fields are plain data (can be memcpy'd to/from a file).
//   - Derived fields are computed by init_derived_fields() (= PipeANN's
//     init_temporary_fields()), called after load or construction.
//   - All layout arithmetic goes through this struct, not scattered in callers.

#include "index/node_defs.h"
#include "storage/storage_types.h"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace powerlaw_ann::graph {

template <typename T>
struct index_layout_t {
  // -------------------------------------------------------------------------
  // Disk-resident fields  (written/read as a flat binary block, like PipeANN)
  // -------------------------------------------------------------------------
  static constexpr uint64_t k_magic = 0x54594C00'44574342ULL; // "BWCDLYT\0"
  static constexpr uint32_t k_version = 1;

  uint64_t n_nodes = 0;     // total node count
  uint64_t dim = 0;         // vector dimension
  uint64_t entry_point = 0; // greedy search start node

  // Max neighbor slots per class.  These drive all slot-size calculations.
  // Analogous to SSDIndexMetadata::range (which is the uniform max-degree).
  uint64_t hub_max_deg = 1024;
  uint64_t normal_max_deg = 64;
  uint64_t peripheral_max_deg = 16;

  // Classification thresholds (stored so the directory can be rebuilt).
  uint64_t hub_degree_threshold = 128;
  uint64_t peripheral_degree_threshold = 8;

  // -------------------------------------------------------------------------
  // Derived fields  (not written to disk; recomputed by init_derived_fields)
  // Analogous to SSDIndexMetadata::normal_node_len, range_dense, etc.
  // -------------------------------------------------------------------------
  uint32_t hub_slot_bytes = 0;
  uint32_t normal_slot_bytes = 0;
  uint32_t peripheral_slot_bytes = 0;

  // Maximum slot size across all classes — used to size scratch buffers.
  uint32_t max_slot_bytes = 0;

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------
  index_layout_t() = default;

  index_layout_t(uint64_t n_nodes, uint64_t dim, uint64_t entry_point, uint64_t hub_max_deg = 1024,
                 uint64_t normal_max_deg = 64, uint64_t peripheral_max_deg = 16)
      : n_nodes(n_nodes), dim(dim), entry_point(entry_point), hub_max_deg(hub_max_deg),
        normal_max_deg(normal_max_deg), peripheral_max_deg(peripheral_max_deg) {
    init_derived_fields();
  }

  // Called after any field change or after load_from_file().
  // Mirrors SSDIndexMetadata::init_temporary_fields().
  void init_derived_fields() {
    hub_slot_bytes = node_header_t::slot_bytes<T>(dim, hub_max_deg);
    normal_slot_bytes = node_header_t::slot_bytes<T>(dim, normal_max_deg);
    peripheral_slot_bytes = node_header_t::slot_bytes<T>(dim, peripheral_max_deg);
    max_slot_bytes = hub_slot_bytes; // hub is always the largest
  }

  // -------------------------------------------------------------------------
  // Per-class slot size  (hot path — keep inline)
  // Mirrors SSDIndexMetadata::io_size() / io_size_dense()
  // -------------------------------------------------------------------------
  uint32_t slot_bytes(node_class_t cls) const {
    switch (cls) {
    case node_class_t::hub:
      return hub_slot_bytes;
    case node_class_t::normal:
      return normal_slot_bytes;
    case node_class_t::peripheral:
      return peripheral_slot_bytes;
    }
    return normal_slot_bytes;
  }

  uint16_t max_degree(node_class_t cls) const {
    switch (cls) {
    case node_class_t::hub:
      return static_cast<uint16_t>(hub_max_deg);
    case node_class_t::normal:
      return static_cast<uint16_t>(normal_max_deg);
    case node_class_t::peripheral:
      return static_cast<uint16_t>(peripheral_max_deg);
    }
    return static_cast<uint16_t>(normal_max_deg);
  }

  // -------------------------------------------------------------------------
  // Classification  (mirrors SSDIndexMetadata's loc_sector_no — both map
  // a node property to a layout parameter)
  // -------------------------------------------------------------------------
  node_class_t classify(uint32_t degree) const {
    if (degree >= static_cast<uint32_t>(hub_degree_threshold))
      return node_class_t::hub;
    if (degree <= static_cast<uint32_t>(peripheral_degree_threshold))
      return node_class_t::peripheral;
    return node_class_t::normal;
  }

  // -------------------------------------------------------------------------
  // Chunk-type selection.
  //
  // Maps (node class, community size) → which chunk level to use.
  // Analogous to the sector arithmetic in SSDIndexMetadata, but for
  // our 4-level chunk system instead of uniform 4KB sectors.
  //
  // Rules (community_size == 0 means per-node default, no community context):
  //   HUB        → 4KB  (hot/cold extreme; hub is also in memory)
  //   PERIPHERAL → 4KB  (cold extreme; pack many per 4KB)
  //   NORMAL, large community  (>= 512KB data) → 2MB chunk
  //   NORMAL, medium community (>= 32KB  data) → 512KB chunk
  //   NORMAL, small community  (>= 4KB   data) → 64KB chunk
  //   NORMAL, tiny             (< 4KB    data) → 4KB chunk
  // -------------------------------------------------------------------------
  chunk_type_t chunk_type(node_class_t cls, uint32_t community_size = 0) const {
    if (cls == node_class_t::hub || cls == node_class_t::peripheral)
      return chunk_type_t::chunk_4kb;

    // For normal nodes, let community size drive chunk level selection.
    const uint64_t community_bytes = static_cast<uint64_t>(community_size) * normal_slot_bytes;

    if (community_bytes >= 512ULL * 1024)
      return chunk_type_t::chunk_2mb;
    if (community_bytes >= 32ULL * 1024)
      return chunk_type_t::chunk_512kb;
    if (community_bytes >= 4ULL * 1024)
      return chunk_type_t::chunk_64kb;
    return chunk_type_t::chunk_4kb;
  }

  // -------------------------------------------------------------------------
  // Disk node constructor helper.
  // Given a loaded chunk buffer and the byte offset of a node within it,
  // return a non-owning reference into the buffer.
  // Mirrors how PipeANN constructs DiskNode(page_buf, loc, meta).
  // -------------------------------------------------------------------------
  disk_node_ref_t<T> node_ref(char* chunk_buf, uint32_t offset_in_chunk) const {
    return disk_node_ref_t<T>(chunk_buf, offset_in_chunk, static_cast<uint32_t>(dim));
  }

  // -------------------------------------------------------------------------
  // Persistence  (analogous to SSDIndexMetadata::save/load_from_disk_index)
  // -------------------------------------------------------------------------
  void save_to_file(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
      throw std::runtime_error("index_layout: cannot write: " + path);

    const uint32_t version = k_version;
    f.write(reinterpret_cast<const char*>(&k_magic), sizeof(k_magic));
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write disk-resident fields as a flat block.
    f.write(reinterpret_cast<const char*>(&n_nodes), sizeof(n_nodes));
    f.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    f.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
    f.write(reinterpret_cast<const char*>(&hub_max_deg), sizeof(hub_max_deg));
    f.write(reinterpret_cast<const char*>(&normal_max_deg), sizeof(normal_max_deg));
    f.write(reinterpret_cast<const char*>(&peripheral_max_deg), sizeof(peripheral_max_deg));
    f.write(reinterpret_cast<const char*>(&hub_degree_threshold), sizeof(hub_degree_threshold));
    f.write(reinterpret_cast<const char*>(&peripheral_degree_threshold),
            sizeof(peripheral_degree_threshold));

    if (!f)
      throw std::runtime_error("index_layout: write failed: " + path);
  }

  void load_from_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("index_layout: cannot read: " + path);

    uint64_t magic = 0;
    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != k_magic)
      throw std::runtime_error("index_layout: bad magic");
    if (version != k_version)
      throw std::runtime_error("index_layout: unsupported version");

    f.read(reinterpret_cast<char*>(&n_nodes), sizeof(n_nodes));
    f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    f.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
    f.read(reinterpret_cast<char*>(&hub_max_deg), sizeof(hub_max_deg));
    f.read(reinterpret_cast<char*>(&normal_max_deg), sizeof(normal_max_deg));
    f.read(reinterpret_cast<char*>(&peripheral_max_deg), sizeof(peripheral_max_deg));
    f.read(reinterpret_cast<char*>(&hub_degree_threshold), sizeof(hub_degree_threshold));
    f.read(reinterpret_cast<char*>(&peripheral_degree_threshold),
           sizeof(peripheral_degree_threshold));

    if (!f)
      throw std::runtime_error("index_layout: read failed: " + path);
    init_derived_fields();
  }

  void print() const {
    // Intentionally kept simple — callers can format as needed.
    // (Mirrors SSDIndexMetadata::print() pattern but without glog dependency.)
    printf(
        "[index_layout] n=%llu dim=%llu ep=%llu hub_max=%llu normal_max=%llu peripheral_max=%llu\n",
        (unsigned long long) n_nodes, (unsigned long long) dim, (unsigned long long) entry_point,
        (unsigned long long) hub_max_deg, (unsigned long long) normal_max_deg,
        (unsigned long long) peripheral_max_deg);
    printf("[index_layout] slot_bytes: hub=%u normal=%u peripheral=%u max=%u\n", hub_slot_bytes,
           normal_slot_bytes, peripheral_slot_bytes, max_slot_bytes);
  }
};

} // namespace powerlaw_ann::graph

#endif // INDEX_INDEX_LAYOUT
