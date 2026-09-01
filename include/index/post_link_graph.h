#ifndef INDEX_POST_LINK_GRAPH
#define INDEX_POST_LINK_GRAPH

#include <cstdint>
#include <span>

namespace powerlaw_ann {

/**
 * @brief Exposes a completed in-memory Vamana graph through a read-only callback view.
 *
 * The view and all spans returned from it remain valid only for the duration of the observer
 * callback. The interface does not permit graph or vector mutation.
 */
class post_link_graph_view_t {
public:
  virtual ~post_link_graph_view_t() = default;

  /** @brief Returns the number of Base vertices visible to the callback. */
  virtual uint64_t point_count() const noexcept = 0;

  /** @brief Returns the unaligned vector dimension. */
  virtual uint32_t dimension() const noexcept = 0;

  /** @brief Returns the Vamana entry-point vertex ID. */
  virtual uint32_t entry_point_id() const noexcept = 0;

  /**
   * @brief Borrows the final Base adjacency of one vertex.
   * @param node_id Base vertex ID in `[0, point_count())`.
   * @return A callback-scoped span of neighbor IDs.
   * @throws std::out_of_range If `node_id` is outside the Base graph.
   */
  virtual std::span<const uint32_t> neighbors(uint32_t node_id) const = 0;

  /**
   * @brief Copies one original float32 vector into caller-owned storage.
   * @param node_id Base vertex ID in `[0, point_count())`.
   * @param destination Buffer containing exactly `dimension()` float values.
   * @throws std::invalid_argument If the ID or destination size is invalid.
   */
  virtual void copy_vector(uint32_t node_id, std::span<float> destination) const = 0;

  /**
   * @brief Computes the Base data store's squared-L2 distance between two vertices.
   * @param left First Base vertex ID.
   * @param right Second Base vertex ID.
   * @return Squared-L2 distance in the original vector space.
   * @throws std::out_of_range If either vertex ID is outside the Base graph.
   */
  virtual float squared_distance(uint32_t left, uint32_t right) const = 0;
};

/** @brief Receives one serial read-only callback immediately after Vamana linking completes. */
class post_link_graph_observer_t {
public:
  virtual ~post_link_graph_observer_t() = default;

  /**
   * @brief Observes the completed Base graph without modifying its topology or vectors.
   * @param graph Callback-scoped read-only graph view.
   */
  virtual void observe(const post_link_graph_view_t& graph) = 0;
};

} // namespace powerlaw_ann

#endif // INDEX_POST_LINK_GRAPH
