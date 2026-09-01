#ifndef INDEX_NODE_DIRECTORY
#define INDEX_NODE_DIRECTORY

#include "index/node_defs.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace powerlaw_ann::graph {

// ---------------------------------------------------------------------------
// Replaces PipeANN's formula-based offset computation.
//
// PipeANN computes disk location of node i as:
//   sector_no = 1 + i / nnodes_per_sector          (SSDIndexMetadata::loc_sector_no)
//   offset    = sector_no * SECTOR_LEN + (i % nnodes_per_sector) * max_node_len
//
// That formula only works for uniform slot sizes.  Once slot sizes vary
// (hub=1024 nbrs, normal=64, peripheral=16), offset is not computable from
// node_id alone — we need a lookup table.
//
// node_directory_t is that table: node_id → (chunk_id, offset, slot_len, class).
// Persisted alongside the chunk files so it can be loaded at search startup.
// Analogous role to PipeANN's sector addressing, but explicit rather than derived.
// ---------------------------------------------------------------------------
class node_directory_t {
public:
  static constexpr uint64_t k_magic = 0x52494400'44574342ULL; // "BWCDDIR\0"
  static constexpr uint32_t k_version = 1;

  node_directory_t() = default;

  explicit node_directory_t(uint32_t n_nodes) {
    addrs_.assign(n_nodes, node_addr_t::invalid());
    classes_.assign(n_nodes, node_class_t::normal);
  }

  // -------------------------------------------------------------------------
  // Build-time population
  // -------------------------------------------------------------------------

  void resize(uint32_t n_nodes) {
    addrs_.assign(n_nodes, node_addr_t::invalid());
    classes_.assign(n_nodes, node_class_t::normal);
  }

  void set(uint32_t node_id, node_addr_t addr, node_class_t cls) {
    assert(node_id < addrs_.size());
    addrs_[node_id] = addr;
    classes_[node_id] = cls;
  }

  // -------------------------------------------------------------------------
  // Search-time access (hot path — keep inline)
  // -------------------------------------------------------------------------

  node_addr_t addr(uint32_t node_id) const { return addrs_[node_id]; }
  node_class_t cls(uint32_t node_id) const { return classes_[node_id]; }
  uint32_t size() const { return static_cast<uint32_t>(addrs_.size()); }

  bool is_hub(uint32_t node_id) const { return classes_[node_id] == node_class_t::hub; }
  bool is_peripheral(uint32_t node_id) const {
    return classes_[node_id] == node_class_t::peripheral;
  }

  // -------------------------------------------------------------------------
  // Serialization  (binary, fixed-width records for O(1) random access)
  //
  // File layout:
  //   [Header: magic(8) + version(4) + n_nodes(4)]
  //   [node_addr_t * n_nodes]    -- 16 bytes each
  //   [uint8_t     * n_nodes]    -- node_class_t, 1 byte each, padded to 8B
  // -------------------------------------------------------------------------

  // Analogous to SSDIndexMetadata::save_to_disk_index / load_from_disk_index.
  void save_to_file(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
      throw std::runtime_error("node_directory: cannot open for write: " + path);

    const uint32_t n = static_cast<uint32_t>(addrs_.size());

    // Header
    f.write(reinterpret_cast<const char*>(&k_magic), sizeof(k_magic));
    f.write(reinterpret_cast<const char*>(&k_version), sizeof(k_version));
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    // Addresses
    f.write(reinterpret_cast<const char*>(addrs_.data()),
            static_cast<std::streamsize>(n * sizeof(node_addr_t)));

    // Classes (cast to uint8_t array)
    for (uint32_t i = 0; i < n; ++i) {
      uint8_t c = static_cast<uint8_t>(classes_[i]);
      f.write(reinterpret_cast<const char*>(&c), 1);
    }

    if (!f)
      throw std::runtime_error("node_directory: write failed: " + path);
  }

  void load_from_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("node_directory: cannot open for read: " + path);

    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t n = 0;

    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    f.read(reinterpret_cast<char*>(&n), sizeof(n));

    if (magic != k_magic)
      throw std::runtime_error("node_directory: bad magic in " + path);
    if (version != k_version)
      throw std::runtime_error("node_directory: unsupported version in " + path);

    addrs_.resize(n);
    classes_.resize(n);

    f.read(reinterpret_cast<char*>(addrs_.data()),
           static_cast<std::streamsize>(n * sizeof(node_addr_t)));

    for (uint32_t i = 0; i < n; ++i) {
      uint8_t c = 0;
      f.read(reinterpret_cast<char*>(&c), 1);
      classes_[i] = static_cast<node_class_t>(c);
    }

    if (!f)
      throw std::runtime_error("node_directory: read failed: " + path);
  }

  // -------------------------------------------------------------------------
  // Diagnostics
  // -------------------------------------------------------------------------

  struct stats_t {
    uint32_t n_hub;
    uint32_t n_normal;
    uint32_t n_peripheral;
    uint32_t n_unplaced; // addr.is_valid() == false
  };

  stats_t compute_stats() const {
    stats_t s{};
    for (uint32_t i = 0; i < addrs_.size(); ++i) {
      if (!addrs_[i].is_valid()) {
        ++s.n_unplaced;
        continue;
      }
      switch (classes_[i]) {
      case node_class_t::hub:
        ++s.n_hub;
        break;
      case node_class_t::normal:
        ++s.n_normal;
        break;
      case node_class_t::peripheral:
        ++s.n_peripheral;
        break;
      }
    }
    return s;
  }

private:
  std::vector<node_addr_t> addrs_;    // indexed by node_id
  std::vector<node_class_t> classes_; // indexed by node_id
};

} // namespace powerlaw_ann::graph

#endif // INDEX_NODE_DIRECTORY
