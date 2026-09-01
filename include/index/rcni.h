#ifndef INDEX_RCNI
#define INDEX_RCNI

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace powerlaw_ann {

inline constexpr float k_rcni_default_epsilon = 1.0e-12F;

/**
 * @brief Describes how an RCNI input distance is encoded.
 */
enum class rcni_distance_semantics_t {
  TRUE_L2,
  SQUARED_L2,
};

/**
 * @brief Identifies one selected neighbor and its distance to the prune victim.
 *
 * The distance must use the semantics selected by `rcni_event_input_t`.
 */
struct rcni_neighbor_distance_t {
  uint32_t node_id = 0;
  float victim_distance = 0.0F;
};

/**
 * @brief Describes one causal prune event observed in the primary Vamana build path.
 *
 * Distances retain the encoding used by Vamana. The first PowerLawANN build path uses squared
 * L2, which is converted by `compute_rcni_event_credit()` rather than by the observer. The
 * optional alternative is the closest final selected neighbor other than the witness; it is
 * absent when the source fallback defines the counterfactual.
 */
struct rcni_prune_event_t {
  uint32_t source_id = 0;
  uint32_t witness_id = 0;
  uint32_t victim_id = 0;
  float source_victim_distance = 0.0F;
  float witness_victim_distance = 0.0F;
  std::optional<rcni_neighbor_distance_t> alternative;
};

/**
 * @brief Tracks the strongest witness update observed for one RobustPrune candidate.
 */
struct rcni_causal_witness_t {
  std::optional<uint32_t> witness_id;
  float witness_victim_distance = 0.0F;
  float occlusion_factor = 0.0F;
};

/**
 * @brief Updates one candidate's causal witness using strength and node-ID ordering.
 *
 * A larger occlusion factor wins. Equal factors use the smaller witness node ID, making the
 * result independent of equal-distance iteration order.
 *
 * @param state Causal witness state to update.
 * @param witness_id Candidate witness node ID.
 * @param witness_victim_distance Distance from the witness to the victim.
 * @param occlusion_factor RobustPrune occlusion factor produced by the witness.
 */
void update_rcni_causal_witness(rcni_causal_witness_t& state, uint32_t witness_id,
                                float witness_victim_distance, float occlusion_factor);

/**
 * @brief Receives source-scoped RCNI observations from the primary Vamana RobustPrune call.
 *
 * The callback may run concurrently for different source vertices. Both spans are borrowed
 * only for the duration of the callback. Implementations that retain observations must copy
 * them and provide their own synchronization.
 */
class rcni_prune_observer_t {
public:
  virtual ~rcni_prune_observer_t() = default;

  /**
   * @brief Prepares the observer for one static Vamana construction.
   *
   * The build calls this method once before any concurrent source callback. The default
   * implementation preserves compatibility with stateless and test-only observers.
   *
   * @param source_count Number of source vertices that may be observed.
   */
  virtual void prepare(size_t source_count) { static_cast<void>(source_count); }

  /**
   * @brief Observes all certified prune events and final selected neighbors for one source.
   *
   * @param source_id Source vertex whose outgoing adjacency is being constructed.
   * @param events Causal prune events certified by the primary RobustPrune call.
   * @param selected_neighbors Final outgoing neighbors after optional graph saturation.
   */
  virtual void observe_source(uint32_t source_id, std::span<const rcni_prune_event_t> events,
                              std::span<const uint32_t> selected_neighbors) = 0;
};

/**
 * @brief Contains the geometric inputs for one RCNI prune event.
 *
 * `selected_neighbors` is borrowed for the duration of `compute_rcni_event_credit()` and may
 * contain the witness itself. The implementation excludes every entry whose ID equals
 * `witness_id` when choosing the counterfactual alternative.
 */
struct rcni_event_input_t {
  uint32_t witness_id = 0;
  float source_victim_distance = 0.0F;
  float witness_victim_distance = 0.0F;
  std::span<const rcni_neighbor_distance_t> selected_neighbors{};
  float alpha = 1.2F;
  float epsilon = k_rcni_default_epsilon;
  rcni_distance_semantics_t distance_semantics = rcni_distance_semantics_t::SQUARED_L2;
};

/**
 * @brief Reports the components of the RCNI credit assigned to one prune event.
 */
struct rcni_event_credit_t {
  float witness_ratio = 0.0F;
  float counterfactual_ratio = 1.0F;
  float certified_progress = 0.0F;
  float counterfactual_irreplaceability = 0.0F;
  float event_credit = 0.0F;
  std::optional<uint32_t> alternative_id;
};

/**
 * @brief Computes Certified Progress, Counterfactual Irreplaceability, and event credit.
 *
 * The closest selected neighbor other than the witness is the counterfactual alternative.
 * Equal-distance alternatives are ordered by node ID. If none exists, remaining at the source
 * gives a counterfactual ratio of one. A source-to-victim distance no greater than `epsilon`
 * produces zero credit because the contraction ratio is undefined.
 *
 * @param input Geometric data and distance semantics for one prune event.
 * @return The bounded `P`, `Q`, and `E = P * Q` values and their effective ratios.
 * @throws std::invalid_argument If alpha, epsilon, or any supplied distance is invalid.
 */
rcni_event_credit_t compute_rcni_event_credit(const rcni_event_input_t& input);

/**
 * @brief Combines one witness's event credits within one source using deterministic Noisy-OR.
 *
 * The input is copied and sorted before accumulation, so the result is independent of event
 * order. The function is stateless and safe to call concurrently.
 *
 * @param event_credits Event credits in the closed interval `[0, 1]`.
 * @return `1 - product(1 - event_credit)`, bounded to `[0, 1]`.
 * @throws std::invalid_argument If an event credit is non-finite or outside `[0, 1]` beyond
 *         the RCNI numerical tolerance.
 */
float compute_rcni_source_saturation(std::span<const float> event_credits);

/**
 * @brief Contains the deterministic RCNI importance statistics for one node.
 */
struct rcni_node_importance_t {
  uint32_t node_id = 0;
  double raw_importance = 0.0;
  double normalized_importance = 0.0;
  double importance_percentile = 0.0;
  uint64_t source_support = 0;
  uint64_t witness_event_count = 0;
};

/**
 * @brief Selects how RCNI importance is converted into the candidate Hub set.
 */
enum class candidate_hub_selection_mode_t {
  TOP_RATIO,
  IMPORTANCE_MASS,
};

/**
 * @brief Configures deterministic candidate-Hub selection.
 *
 * Both values are fractions in `[0, 1]`. Only the value selected by `mode` controls the
 * result. A zero value selects no candidate Hubs, and no experimental ratio is hardcoded.
 */
struct candidate_hub_selection_config_t {
  candidate_hub_selection_mode_t mode = candidate_hub_selection_mode_t::TOP_RATIO;
  double top_ratio = 0.0;
  double importance_mass = 0.0;
};

/**
 * @brief Describes one candidate Hub in deterministic importance order.
 */
struct candidate_hub_t {
  uint32_t node_id = 0;
  double raw_importance = 0.0;
  double normalized_importance = 0.0;
  double cumulative_importance_mass = 0.0;
};

/**
 * @brief Contains the candidate Hub set and its achieved importance mass.
 */
struct candidate_hub_selection_result_t {
  candidate_hub_selection_mode_t mode = candidate_hub_selection_mode_t::TOP_RATIO;
  double selection_value = 0.0;
  double total_importance = 0.0;
  double selected_importance = 0.0;
  double selected_importance_mass = 0.0;
  std::vector<candidate_hub_t> candidate_hubs;
};

/**
 * @brief Validates candidate-Hub selection parameters without allocating selection state.
 * @param config Candidate-Hub selection configuration.
 * @throws std::invalid_argument If the mode or active value is invalid.
 */
void validate_candidate_hub_selection_config(const candidate_hub_selection_config_t& config);

/**
 * @brief Selects candidate Hubs from RCNI importance deterministically.
 *
 * Records are ordered by descending normalized importance, then ascending node ID. Top-ratio
 * selection uses `ceil(node_count * top_ratio)`. Importance-mass selection returns the smallest
 * prefix whose normalized-importance mass reaches the configured threshold. Empty input, a
 * zero active value, or all-zero normalized importance returns an empty set.
 *
 * @param records RCNI records in any input order.
 * @param config Selection mode and threshold.
 * @return Candidate Hubs in deterministic importance order with aggregate mass statistics.
 * @throws std::invalid_argument If the configuration, importance, or node IDs are invalid.
 */
candidate_hub_selection_result_t
select_candidate_hubs(std::span<const rcni_node_importance_t> records,
                      const candidate_hub_selection_config_t& config);

/**
 * @brief Aggregates concurrent source-scoped prune observations deterministically.
 *
 * Each callback immediately compresses raw events into source-witness saturation records.
 * Records are buffered in low-contention shards and merged in `(source_id, witness_id)` order
 * after construction, so thread scheduling cannot change floating-point accumulation order.
 * Call `compute_importance()` only after all source callbacks have completed.
 */
class rcni_aggregator_t final : public rcni_prune_observer_t {
public:
  explicit rcni_aggregator_t(float alpha = 1.2F, float epsilon = k_rcni_default_epsilon);
  ~rcni_aggregator_t() override;

  rcni_aggregator_t(const rcni_aggregator_t&) = delete;
  rcni_aggregator_t& operator=(const rcni_aggregator_t&) = delete;
  rcni_aggregator_t(rcni_aggregator_t&&) = delete;
  rcni_aggregator_t& operator=(rcni_aggregator_t&&) = delete;

  void prepare(size_t source_count) override;
  void observe_source(uint32_t source_id, std::span<const rcni_prune_event_t> events,
                      std::span<const uint32_t> selected_neighbors) override;

  /**
   * @brief Computes one stable importance record for every prepared node.
   * @return Records in ascending node-ID order.
   * @throws std::logic_error If the observer has not been prepared.
   */
  std::vector<rcni_node_importance_t> compute_importance() const;

private:
  class impl_t;
  std::unique_ptr<impl_t> impl_;
};

/**
 * @brief Writes deterministic RCNI importance records as CSV.
 *
 * The function writes a temporary sibling and renames it only after the complete stream has
 * been flushed successfully.
 *
 * @param path Final CSV path.
 * @param records Records to emit, normally in ascending node-ID order.
 * @throws std::runtime_error If the file cannot be written or published.
 */
void write_rcni_importance_csv(const std::filesystem::path& path,
                               std::span<const rcni_node_importance_t> records);

/**
 * @brief Writes one deterministic candidate-Hub selection as an audit CSV.
 *
 * The function writes a temporary sibling and atomically publishes it after a complete flush.
 * Empty selections produce a header-only file.
 *
 * @param path Final CSV path.
 * @param result Candidate-Hub selection and aggregate mass statistics.
 * @throws std::runtime_error If the file cannot be written or published.
 */
void write_candidate_hubs_csv(const std::filesystem::path& path,
                              const candidate_hub_selection_result_t& result);

} // namespace powerlaw_ann

#endif // INDEX_RCNI
