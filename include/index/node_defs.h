#ifndef INDEX_NODE_DEFS
#define INDEX_NODE_DEFS

#include "storage/storage_types.h"

#include <cassert>
#include <cstdint>

namespace powerlaw_ann::graph {

// ---------------------------------------------------------------------------
// Node importance classification.
// ---------------------------------------------------------------------------
enum class node_class_t : uint8_t {
  hub = 0,        // In-memory HNSW + large disk slot
  normal = 1,     // Standard slot
  peripheral = 2, // Minimal slot
};

// ---------------------------------------------------------------------------
// On-disk header at the start of every node's variable-length slot.
//
// Disk layout of one node slot  (mirrors PipeANN's DiskNode field order,
// but adds an explicit header so we can identify nodes without external
// metadata — critical for variable-slot recovery):
//
//   [node_header_t      :  16 bytes            ]
//   [T[dim]             :  dim * sizeof(T)      ]  raw vector (coords)
//   [uint32_t[max_deg]  :  max_deg * 4 bytes    ]  neighbor IDs
//   [padding to 16-byte boundary                ]
//
// Total slot size: node_header_t::slot_bytes<T>(dim, max_deg).
//
// Compare PipeANN DiskNode:
//   [T[dim] | uint16_t nnbrs | uint16_t n_dense | uint32_t[1+range] | attrs | dense_nbrs]
// Difference: we put the header first so a slot is self-describing without
// the global metadata struct.  The search path uses index_layout_t (analogous
// to SSDIndexMetadata) for all layout arithmetic, so hot-path cost is the same.
// ---------------------------------------------------------------------------
struct node_header_t {
  uint32_t node_id;
  uint16_t degree;     // current neighbor count (<= max_degree)
  uint16_t max_degree; // allocated slot capacity
  uint32_t community_id;
  uint8_t node_class; // stored as uint8_t; use get_class() for the enum
  uint8_t _pad[3];

  template <typename T>
  static constexpr uint32_t slot_bytes(uint32_t dim, uint16_t max_deg) {
    uint32_t raw = static_cast<uint32_t>(sizeof(node_header_t)) +
                   dim * static_cast<uint32_t>(sizeof(T)) +
                   max_deg * static_cast<uint32_t>(sizeof(uint32_t));
    return (raw + 15u) & ~15u; // round up to 16-byte alignment
  }

  node_class_t get_class() const { return static_cast<node_class_t>(node_class); }
};
static_assert(sizeof(node_header_t) == 16);

// ---------------------------------------------------------------------------
// Physical address of a node's slot inside a disk_manager_t chunk.
// ---------------------------------------------------------------------------
struct node_addr_t {
  chunk_id_t chunk_id; // encoded chunk ID (type << 62 | chunk_no)
  uint32_t offset;     // byte offset of node_header_t within the chunk
  uint32_t slot_len;   // total slot bytes (== node_header_t::slot_bytes<T>(...))

  static constexpr chunk_id_t k_invalid = ~chunk_id_t{0};

  bool is_valid() const { return chunk_id != k_invalid; }
  static node_addr_t invalid() { return {k_invalid, 0, 0}; }
};

// ---------------------------------------------------------------------------
// Non-owning reference into a loaded chunk buffer.
//
// Analogous to PipeANN's DiskNode<T>: constructed from a raw buffer pointer
// + byte offset, holds typed pointers/references into the buffer so callers
// can read or write node fields without copying.
//
//   PipeANN:  DiskNode(char *page_buf, uint32_t loc, SSDIndexMetadata &meta)
//   Ours:     disk_node_ref_t(char *chunk_buf, uint32_t offset, index_layout_t &layout)
//
// The key design constraint (same as PipeANN): all fields are aliases into
// the caller's buffer.  Mutating coords/nbrs/degree mutates the buffer
// in-place, which is correct for build-time writes and update paths.
// ---------------------------------------------------------------------------
template <typename T>
struct disk_node_ref_t {
  node_header_t* hdr;   // header (id, degree, class, community)
  T* coords;            // raw vector — dim elements
  uint16_t& degree;     // alias into hdr->degree (current neighbor count)
  uint16_t& max_degree; // alias into hdr->max_degree
  uint32_t* nbrs;       // neighbor ID array — max_degree slots

  // chunk_buf : start of the loaded chunk data
  // offset    : byte offset of node_header_t within chunk_buf
  // dim       : vector dimension (needed to locate nbrs after coords)
  disk_node_ref_t(char* chunk_buf, uint32_t offset, uint32_t dim)
      : hdr(reinterpret_cast<node_header_t*>(chunk_buf + offset)),
        coords(reinterpret_cast<T*>(chunk_buf + offset + sizeof(node_header_t))),
        degree(hdr->degree), max_degree(hdr->max_degree),
        nbrs(reinterpret_cast<uint32_t*>(chunk_buf + offset + sizeof(node_header_t) +
                                         dim * sizeof(T))) {}
};

} // namespace powerlaw_ann::graph

#endif // INDEX_NODE_DEFS
