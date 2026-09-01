#ifndef INDEX_SLOT_POLICY
#define INDEX_SLOT_POLICY

// slot_policy_t  —  classifies nodes into hub / normal / peripheral.
//
// Deliberately kept separate from index_layout_t so callers can experiment
// with different classification strategies (degree-based, rank-based,
// community-centrality-based) without touching the layout struct.
//
// Once classification is done, pass the result to index_layout_t for all
// slot-size and chunk-type decisions.

#include "index/node_defs.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace powerlaw_ann::graph {

// ---------------------------------------------------------------------------
// Tunable thresholds.  Stored separately from index_layout_t so they can be
// adjusted for a dataset without rebuilding the persisted layout.
// ---------------------------------------------------------------------------
struct slot_policy_params_t {
  uint32_t hub_degree_threshold = 128;      // >= this → hub
  uint32_t peripheral_degree_threshold = 8; // <= this → peripheral
  float hub_fraction = 0.0f;                // if > 0, top fraction forced hub
};

// ---------------------------------------------------------------------------
// Stateless classifier — no heap allocation.
// ---------------------------------------------------------------------------
struct slot_policy_t {
  slot_policy_params_t params;

  explicit slot_policy_t(const slot_policy_params_t& p = {}) : params(p) {}

  // Classify by absolute out-degree from initial Vamana build.
  node_class_t classify_by_degree(uint32_t degree) const {
    if (degree >= params.hub_degree_threshold)
      return node_class_t::hub;
    if (degree <= params.peripheral_degree_threshold)
      return node_class_t::peripheral;
    return node_class_t::normal;
  }

  // Classify all nodes in one pass, returning a class vector indexed by node_id.
  // degrees[i] = out-degree of node i from initial Vamana build.
  // If hub_fraction > 0, the top hub_fraction% by degree are forced to hub
  // regardless of the threshold.
  std::vector<node_class_t> classify_all(const std::vector<uint32_t>& degrees) const {
    const uint32_t n = static_cast<uint32_t>(degrees.size());
    std::vector<node_class_t> result(n, node_class_t::normal);

    // Threshold-based pass.
    for (uint32_t i = 0; i < n; ++i)
      result[i] = classify_by_degree(degrees[i]);

    // Fraction-based override: ensure at least hub_fraction*n nodes are hub.
    if (params.hub_fraction > 0.0f) {
      uint32_t target = static_cast<uint32_t>(params.hub_fraction * n);
      if (target > 0) {
        // Build sorted index by descending degree.
        std::vector<uint32_t> idx(n);
        for (uint32_t i = 0; i < n; ++i)
          idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + target, idx.end(),
                          [&](uint32_t a, uint32_t b) { return degrees[a] > degrees[b]; });
        for (uint32_t k = 0; k < target; ++k)
          result[idx[k]] = node_class_t::hub;
      }
    }

    return result;
  }
};

} // namespace powerlaw_ann::graph

#endif // INDEX_SLOT_POLICY
