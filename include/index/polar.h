#ifndef INDEX_POLAR
#define INDEX_POLAR

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace powerlaw_ann {

inline constexpr float k_polar_default_epsilon = 1.0e-12F;

/**
 * @brief Contains a Hub-relative radius and direction-validity state.
 *
 * A point within `epsilon` of its Hub has radius zero and `has_direction == false`. This explicit
 * state prevents zero-distance points from being assigned to an arbitrary direction bin.
 */
struct polar_coordinate_t {
  float radius = 0.0F;
  bool has_direction = false;
};

/**
 * @brief Computes the true-L2 radius and unit direction of a point relative to one Hub.
 * @param hub Hub vector.
 * @param point Candidate, neighbor, or query vector in the same coordinate space.
 * @param unit_direction Caller-owned output buffer with the same dimension. Zero-distance points
 * receive an all-zero output and must not be direction-binned.
 * @param epsilon Positive zero-distance threshold.
 * @return The radius and whether `unit_direction` is valid.
 * @throws std::invalid_argument If dimensions, epsilon, or vector components are invalid.
 */
polar_coordinate_t compute_polar_coordinate(std::span<const float> hub,
                                            std::span<const float> point,
                                            std::span<float> unit_direction,
                                            float epsilon = k_polar_default_epsilon);

/**
 * @brief Stores a shared row-major unit-direction codebook.
 *
 * `centroids` contains `direction_bin_count * dimension` float32 values. Direction bins are
 * restricted to 16, 32, or 64 in the first implementation.
 */
struct polar_direction_codebook_t {
  uint32_t dimension = 0;
  uint32_t direction_bin_count = 0;
  std::vector<float> centroids;
};

/**
 * @brief Configures deterministic weighted Spherical K-means training.
 *
 * The caller explicitly supplies the codebook size, iteration cap, and seed. The seed chooses
 * the first positive-weight sample; remaining initial centroids use deterministic weighted
 * farthest-first selection.
 */
struct spherical_kmeans_config_t {
  uint32_t direction_bin_count = 0;
  uint32_t max_iterations = 0;
  uint64_t seed = 0;
};

/**
 * @brief Reports a trained codebook and deterministic training diagnostics.
 */
struct spherical_kmeans_result_t {
  polar_direction_codebook_t codebook;
  uint32_t iterations = 0;
  double weighted_mean_cosine = 0.0;
};

/**
 * @brief Validates codebook dimensions, supported bin count, finite values, and unit centroids.
 * @param codebook Shared direction codebook to validate.
 * @throws std::invalid_argument If any codebook invariant is violated.
 */
void validate_polar_direction_codebook(const polar_direction_codebook_t& codebook);

/**
 * @brief Assigns a unit direction to the maximum-dot-product codebook centroid.
 *
 * Equal dot products use the smaller direction-bin ID. The function performs no heap allocation.
 *
 * @param codebook Shared direction codebook previously validated after training or loading. The
 * assignment path checks its dimensions but does not rescan every centroid.
 * @param unit_direction Unit direction with `codebook.dimension` components.
 * @return Direction-bin ID in `[0, direction_bin_count)`.
 * @throws std::invalid_argument If the codebook shape or direction is invalid.
 */
uint32_t find_polar_direction_bin(const polar_direction_codebook_t& codebook,
                                  std::span<const float> unit_direction);

/**
 * @brief Trains a shared codebook with deterministic weighted Spherical K-means.
 *
 * `directions` is a row-major matrix whose row count equals `weights.size()`. Input rows may be
 * unnormalized but must be finite and non-zero. Weights must be finite and non-negative, with at
 * least `direction_bin_count` positive-weight samples. Empty clusters retain their previous unit
 * centroid, making every iteration deterministic and valid.
 *
 * @param directions Flattened row-major direction samples.
 * @param dimension Number of components per sample.
 * @param weights Non-negative sample weights, normally proportional to `A(v) / |C_v|`.
 * @param config Direction-bin count, iteration cap, and explicit seed.
 * @return Trained codebook, executed iteration count, and final weighted mean cosine.
 * @throws std::invalid_argument If samples, weights, dimensions, or configuration are invalid.
 */
spherical_kmeans_result_t train_weighted_spherical_kmeans(std::span<const float> directions,
                                                          uint32_t dimension,
                                                          std::span<const double> weights,
                                                          const spherical_kmeans_config_t& config);

/**
 * @brief Computes deterministic nearest-rank radial quantile boundaries for one Hub.
 *
 * For `radial_bin_count` bins, the result contains `radial_bin_count - 1` non-decreasing upper
 * bounds. Equal boundaries are retained so the configured Cell-ID space stays fixed.
 *
 * @param radii Candidate true-L2 radii for one Hub.
 * @param radial_bin_count Positive number of radial bins.
 * @return Non-decreasing inclusive upper bounds for all bins except the final `+inf` bin.
 * @throws std::invalid_argument If input is empty or contains invalid radii/bin counts.
 */
std::vector<float> compute_radial_quantile_boundaries(std::span<const float> radii,
                                                      uint32_t radial_bin_count);

/**
 * @brief Assigns a radius using inclusive quantile upper bounds.
 * @param radius Finite non-negative true-L2 radius.
 * @param boundaries Non-decreasing inclusive upper bounds.
 * @return The smallest bin whose boundary is at least `radius`, or the final radial bin.
 * @throws std::invalid_argument If the radius or boundaries are invalid.
 */
uint32_t find_polar_radial_bin(float radius, std::span<const float> boundaries);

/**
 * @brief Identifies one decoded Polar Cell.
 */
struct polar_cell_t {
  uint32_t direction_bin = 0;
  uint32_t radial_bin = 0;
};

/**
 * @brief Encodes one `(direction_bin, radial_bin)` pair as a compact row-major Cell ID.
 * @throws std::invalid_argument If counts or IDs are invalid or the Cell-ID space overflows.
 */
uint32_t encode_polar_cell_id(uint32_t direction_bin, uint32_t radial_bin,
                              uint32_t direction_bin_count, uint32_t radial_bin_count);

/**
 * @brief Decodes and validates one compact row-major Polar Cell ID.
 * @throws std::invalid_argument If counts, ID, or the Cell-ID space are invalid.
 */
polar_cell_t decode_polar_cell_id(uint32_t cell_id, uint32_t direction_bin_count,
                                  uint32_t radial_bin_count);

/**
 * @brief Writes a versioned, checksummed little-endian direction-codebook file atomically.
 *
 * Layout: magic u64, version u32, header size u32, dimension u32, direction-bin count u32,
 * centroid scalar count u64, payload FNV-1a hash u64, followed by row-major float32 centroids.
 * Integers and floats are little-endian with no implicit structure padding.
 *
 * @param path Final codebook path.
 * @param codebook Valid direction codebook.
 * @throws std::invalid_argument If the codebook is invalid.
 * @throws std::runtime_error If the file cannot be written or published.
 */
void write_polar_direction_codebook(const std::filesystem::path& path,
                                    const polar_direction_codebook_t& codebook);

/**
 * @brief Loads and validates a versioned little-endian direction-codebook file.
 * @param path Codebook path.
 * @return Validated shared direction codebook.
 * @throws std::runtime_error If the file is missing, truncated, corrupt, or incompatible.
 */
polar_direction_codebook_t load_polar_direction_codebook(const std::filesystem::path& path);

} // namespace powerlaw_ann

#endif // INDEX_POLAR
