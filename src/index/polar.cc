#include "index/polar.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace powerlaw_ann {
namespace {

constexpr uint64_t k_polar_codebook_magic = 0x0052414C4F504C50ULL; // "PLPOLAR\0"
constexpr uint32_t k_polar_codebook_version = 1;
constexpr uint32_t k_polar_codebook_header_size = 40;
constexpr double k_polar_unit_norm_tolerance = 1.0e-4;
constexpr uint64_t k_fnv_offset_basis = 14695981039346656037ULL;
constexpr uint64_t k_fnv_prime = 1099511628211ULL;

bool is_supported_direction_bin_count(uint32_t count) {
  return count == 16 || count == 32 || count == 64;
}

void validate_codebook_shape(const polar_direction_codebook_t& codebook) {
  if (codebook.dimension == 0) {
    throw std::invalid_argument("Polar codebook dimension must be greater than zero");
  }
  if (!is_supported_direction_bin_count(codebook.direction_bin_count)) {
    throw std::invalid_argument("Polar direction-bin count must be 16, 32, or 64");
  }

  const uint64_t expected_scalar_count = static_cast<uint64_t>(codebook.dimension) *
                                         static_cast<uint64_t>(codebook.direction_bin_count);
  if (expected_scalar_count != codebook.centroids.size()) {
    throw std::invalid_argument("Polar codebook centroid size does not match its dimensions");
  }
}

double squared_norm(std::span<const float> values) {
  double norm = 0.0;
  for (const float value : values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("Polar vectors must contain only finite values");
    }
    norm += static_cast<double>(value) * static_cast<double>(value);
    if (!std::isfinite(norm)) {
      throw std::invalid_argument("Polar vector norm overflowed");
    }
  }
  return norm;
}

void validate_unit_direction(std::span<const float> direction, uint32_t expected_dimension) {
  if (direction.size() != expected_dimension) {
    throw std::invalid_argument("Polar direction dimension does not match the codebook");
  }
  const double norm = std::sqrt(squared_norm(direction));
  if (std::abs(norm - 1.0) > k_polar_unit_norm_tolerance) {
    throw std::invalid_argument("Polar directions must have unit norm");
  }
}

double dot_product(std::span<const float> left, std::span<const float> right) {
  double dot = 0.0;
  for (size_t dimension = 0; dimension < left.size(); ++dimension) {
    dot += static_cast<double>(left[dimension]) * static_cast<double>(right[dimension]);
  }
  return dot;
}

std::span<const float> row(std::span<const float> matrix, size_t row_id, uint32_t dimension) {
  return matrix.subspan(row_id * dimension, dimension);
}

std::span<float> mutable_row(std::vector<float>& matrix, size_t row_id, uint32_t dimension) {
  return std::span<float>(matrix).subspan(row_id * dimension, dimension);
}

void copy_row(std::span<const float> source, std::span<float> destination) {
  std::copy(source.begin(), source.end(), destination.begin());
}

uint32_t find_direction_bin_unchecked(const polar_direction_codebook_t& codebook,
                                      std::span<const float> direction, double* best_dot) {
  uint32_t selected_bin = 0;
  double selected_dot = -std::numeric_limits<double>::infinity();
  const std::span<const float> centroids(codebook.centroids);
  for (uint32_t bin = 0; bin < codebook.direction_bin_count; ++bin) {
    const double current_dot = dot_product(direction, row(centroids, bin, codebook.dimension));
    if (current_dot > selected_dot) {
      selected_dot = current_dot;
      selected_bin = bin;
    }
  }
  if (best_dot != nullptr) {
    *best_dot = selected_dot;
  }
  return selected_bin;
}

void append_u32_le(std::vector<uint8_t>& bytes, uint32_t value) {
  for (uint32_t shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
  }
}

void append_u64_le(std::vector<uint8_t>& bytes, uint64_t value) {
  for (uint32_t shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
  }
}

void append_float_le(std::vector<uint8_t>& bytes, float value) {
  append_u32_le(bytes, std::bit_cast<uint32_t>(value));
}

uint32_t read_u32_le(std::span<const uint8_t> bytes, size_t& cursor) {
  if (bytes.size() - cursor < sizeof(uint32_t)) {
    throw std::runtime_error("Polar codebook header is truncated");
  }
  uint32_t value = 0;
  for (uint32_t shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(bytes[cursor++]) << shift;
  }
  return value;
}

uint64_t read_u64_le(std::span<const uint8_t> bytes, size_t& cursor) {
  if (bytes.size() - cursor < sizeof(uint64_t)) {
    throw std::runtime_error("Polar codebook header is truncated");
  }
  uint64_t value = 0;
  for (uint32_t shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(bytes[cursor++]) << shift;
  }
  return value;
}

float read_float_le(std::span<const uint8_t> bytes, size_t& cursor) {
  return std::bit_cast<float>(read_u32_le(bytes, cursor));
}

uint64_t hash_bytes(std::span<const uint8_t> bytes) {
  uint64_t hash = k_fnv_offset_basis;
  for (const uint8_t byte : bytes) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return hash;
}

uint64_t validate_cell_space(uint32_t direction_bin_count, uint32_t radial_bin_count) {
  if (!is_supported_direction_bin_count(direction_bin_count)) {
    throw std::invalid_argument("Polar direction-bin count must be 16, 32, or 64");
  }
  if (radial_bin_count == 0) {
    throw std::invalid_argument("Polar radial-bin count must be greater than zero");
  }

  const uint64_t cell_count =
      static_cast<uint64_t>(direction_bin_count) * static_cast<uint64_t>(radial_bin_count);
  const uint64_t maximum_cell_count =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL;
  if (cell_count > maximum_cell_count) {
    throw std::invalid_argument("Polar Cell-ID space exceeds the uint32 range");
  }
  return cell_count;
}

} // namespace

polar_coordinate_t compute_polar_coordinate(std::span<const float> hub,
                                            std::span<const float> point,
                                            std::span<float> unit_direction, float epsilon) {
  if (hub.empty() || hub.size() != point.size() || hub.size() != unit_direction.size()) {
    throw std::invalid_argument("Polar vectors must have the same non-zero dimension");
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
    throw std::invalid_argument("Polar epsilon must be finite and greater than zero");
  }

  double radius_squared = 0.0;
  for (size_t dimension = 0; dimension < hub.size(); ++dimension) {
    if (!std::isfinite(hub[dimension]) || !std::isfinite(point[dimension])) {
      throw std::invalid_argument("Polar vectors must contain only finite values");
    }
    const double difference =
        static_cast<double>(point[dimension]) - static_cast<double>(hub[dimension]);
    if (!std::isfinite(difference)) {
      throw std::invalid_argument("Polar coordinate difference overflowed");
    }
    radius_squared += difference * difference;
    if (!std::isfinite(radius_squared)) {
      throw std::invalid_argument("Polar radius overflowed");
    }
  }

  const double radius = std::sqrt(radius_squared);
  if (radius > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::invalid_argument("Polar radius exceeds the float32 range");
  }
  if (radius <= static_cast<double>(epsilon)) {
    std::fill(unit_direction.begin(), unit_direction.end(), 0.0F);
    return {};
  }

  for (size_t dimension = 0; dimension < hub.size(); ++dimension) {
    const double difference =
        static_cast<double>(point[dimension]) - static_cast<double>(hub[dimension]);
    unit_direction[dimension] = static_cast<float>(difference / radius);
  }
  return polar_coordinate_t{static_cast<float>(radius), true};
}

void validate_polar_direction_codebook(const polar_direction_codebook_t& codebook) {
  validate_codebook_shape(codebook);

  const std::span<const float> centroids(codebook.centroids);
  for (uint32_t bin = 0; bin < codebook.direction_bin_count; ++bin) {
    validate_unit_direction(row(centroids, bin, codebook.dimension), codebook.dimension);
  }
}

uint32_t find_polar_direction_bin(const polar_direction_codebook_t& codebook,
                                  std::span<const float> unit_direction) {
  // Codebooks are validated when trained or loaded. Keep candidate assignment O(BD), not O(2BD).
  validate_codebook_shape(codebook);
  validate_unit_direction(unit_direction, codebook.dimension);
  return find_direction_bin_unchecked(codebook, unit_direction, nullptr);
}

spherical_kmeans_result_t train_weighted_spherical_kmeans(std::span<const float> directions,
                                                          uint32_t dimension,
                                                          std::span<const double> weights,
                                                          const spherical_kmeans_config_t& config) {
  if (dimension == 0 || directions.empty() || directions.size() % dimension != 0) {
    throw std::invalid_argument("Spherical K-means directions must form a non-empty matrix");
  }
  if (!is_supported_direction_bin_count(config.direction_bin_count)) {
    throw std::invalid_argument("Spherical K-means direction bins must be 16, 32, or 64");
  }
  if (config.max_iterations == 0) {
    throw std::invalid_argument("Spherical K-means iteration cap must be greater than zero");
  }

  const size_t sample_count = directions.size() / dimension;
  if (weights.size() != sample_count || sample_count < config.direction_bin_count) {
    throw std::invalid_argument(
        "Spherical K-means requires one weight per sample and at least one sample per bin");
  }

  size_t positive_weight_count = 0;
  double total_weight = 0.0;
  for (const double weight : weights) {
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument("Spherical K-means weights must be finite and non-negative");
    }
    if (weight > 0.0) {
      ++positive_weight_count;
      total_weight += weight;
    }
  }
  if (positive_weight_count < config.direction_bin_count || !std::isfinite(total_weight) ||
      total_weight <= 0.0) {
    throw std::invalid_argument(
        "Spherical K-means requires at least one positive-weight sample per bin");
  }

  std::vector<double> normalized_weights(sample_count);
  for (size_t sample = 0; sample < sample_count; ++sample) {
    normalized_weights[sample] = weights[sample] / total_weight;
  }

  std::vector<float> normalized_directions(directions.size());
  for (size_t sample = 0; sample < sample_count; ++sample) {
    const auto input = row(directions, sample, dimension);
    const double norm = std::sqrt(squared_norm(input));
    if (norm <= static_cast<double>(k_polar_default_epsilon)) {
      throw std::invalid_argument("Spherical K-means directions must be non-zero");
    }
    auto output = mutable_row(normalized_directions, sample, dimension);
    for (uint32_t value = 0; value < dimension; ++value) {
      output[value] = static_cast<float>(static_cast<double>(input[value]) / norm);
    }
  }

  polar_direction_codebook_t codebook;
  codebook.dimension = dimension;
  codebook.direction_bin_count = config.direction_bin_count;
  codebook.centroids.resize(static_cast<size_t>(dimension) * config.direction_bin_count);

  std::vector<bool> selected_samples(sample_count, false);
  size_t first_sample = static_cast<size_t>(config.seed % static_cast<uint64_t>(sample_count));
  for (size_t offset = 0; offset < sample_count; ++offset) {
    const size_t candidate = (first_sample + offset) % sample_count;
    if (normalized_weights[candidate] > 0.0) {
      first_sample = candidate;
      break;
    }
  }
  copy_row(row(normalized_directions, first_sample, dimension),
           mutable_row(codebook.centroids, 0, dimension));
  selected_samples[first_sample] = true;

  std::vector<double> best_initial_dot(sample_count, -std::numeric_limits<double>::infinity());
  for (uint32_t bin = 1; bin < config.direction_bin_count; ++bin) {
    const auto newest_centroid = row(codebook.centroids, bin - 1, dimension);
    for (size_t sample = 0; sample < sample_count; ++sample) {
      best_initial_dot[sample] =
          std::max(best_initial_dot[sample],
                   dot_product(row(normalized_directions, sample, dimension), newest_centroid));
    }

    size_t selected_sample = sample_count;
    double selected_score = -1.0;
    for (size_t sample = 0; sample < sample_count; ++sample) {
      if (selected_samples[sample] || normalized_weights[sample] == 0.0) {
        continue;
      }
      const double angular_loss = 1.0 - std::clamp(best_initial_dot[sample], -1.0, 1.0);
      const double score = normalized_weights[sample] * angular_loss;
      if (score > selected_score || (score == selected_score && sample < selected_sample)) {
        selected_score = score;
        selected_sample = sample;
      }
    }
    if (selected_sample == sample_count) {
      throw std::logic_error("Spherical K-means could not initialize all direction bins");
    }
    copy_row(row(normalized_directions, selected_sample, dimension),
             mutable_row(codebook.centroids, bin, dimension));
    selected_samples[selected_sample] = true;
  }

  std::vector<uint32_t> assignments(sample_count, std::numeric_limits<uint32_t>::max());
  std::vector<double> sums(codebook.centroids.size());
  std::vector<double> cluster_weights(config.direction_bin_count);
  uint32_t executed_iterations = 0;
  for (uint32_t iteration = 0; iteration < config.max_iterations; ++iteration) {
    bool assignments_changed = false;
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(cluster_weights.begin(), cluster_weights.end(), 0.0);

    for (size_t sample = 0; sample < sample_count; ++sample) {
      const auto direction = row(normalized_directions, sample, dimension);
      const uint32_t bin = find_direction_bin_unchecked(codebook, direction, nullptr);
      assignments_changed = assignments_changed || assignments[sample] != bin;
      assignments[sample] = bin;

      const double weight = normalized_weights[sample];
      cluster_weights[bin] += weight;
      for (uint32_t value = 0; value < dimension; ++value) {
        sums[static_cast<size_t>(bin) * dimension + value] +=
            weight * static_cast<double>(direction[value]);
      }
    }

    for (uint32_t bin = 0; bin < config.direction_bin_count; ++bin) {
      if (cluster_weights[bin] == 0.0) {
        continue;
      }
      const auto sum =
          std::span<const double>(sums).subspan(static_cast<size_t>(bin) * dimension, dimension);
      double norm_squared = 0.0;
      for (const double value : sum) {
        norm_squared += value * value;
      }
      constexpr double k_min_norm_squared =
          static_cast<double>(k_polar_default_epsilon) * k_polar_default_epsilon;
      if (!std::isfinite(norm_squared) || norm_squared <= k_min_norm_squared) {
        continue;
      }
      const double norm = std::sqrt(norm_squared);
      auto centroid = mutable_row(codebook.centroids, bin, dimension);
      for (uint32_t value = 0; value < dimension; ++value) {
        centroid[value] = static_cast<float>(sum[value] / norm);
      }
    }

    executed_iterations = iteration + 1;
    if (!assignments_changed) {
      break;
    }
  }

  validate_polar_direction_codebook(codebook);
  double weighted_mean_cosine = 0.0;
  for (size_t sample = 0; sample < sample_count; ++sample) {
    double best_dot = 0.0;
    find_direction_bin_unchecked(codebook, row(normalized_directions, sample, dimension),
                                 &best_dot);
    weighted_mean_cosine += normalized_weights[sample] * std::clamp(best_dot, -1.0, 1.0);
  }

  return spherical_kmeans_result_t{std::move(codebook), executed_iterations, weighted_mean_cosine};
}

std::vector<float> compute_radial_quantile_boundaries(std::span<const float> radii,
                                                      uint32_t radial_bin_count) {
  if (radii.empty()) {
    throw std::invalid_argument("Polar radial quantiles require at least one radius");
  }
  if (radial_bin_count == 0) {
    throw std::invalid_argument("Polar radial-bin count must be greater than zero");
  }

  std::vector<float> sorted_radii(radii.begin(), radii.end());
  for (const float radius : sorted_radii) {
    if (!std::isfinite(radius) || radius < 0.0F) {
      throw std::invalid_argument("Polar radii must be finite and non-negative");
    }
  }
  std::sort(sorted_radii.begin(), sorted_radii.end());

  std::vector<float> boundaries;
  boundaries.reserve(static_cast<size_t>(radial_bin_count) - 1);
  const size_t base_rank = sorted_radii.size() / radial_bin_count;
  const uint64_t remainder = sorted_radii.size() % radial_bin_count;
  for (uint32_t boundary = 1; boundary < radial_bin_count; ++boundary) {
    const uint64_t remainder_rank =
        (remainder * boundary + radial_bin_count - 1ULL) / radial_bin_count;
    const size_t nearest_rank = base_rank * boundary + static_cast<size_t>(remainder_rank);
    boundaries.push_back(sorted_radii[nearest_rank - 1]);
  }
  return boundaries;
}

uint32_t find_polar_radial_bin(float radius, std::span<const float> boundaries) {
  if (!std::isfinite(radius) || radius < 0.0F) {
    throw std::invalid_argument("Polar radius must be finite and non-negative");
  }
  if (boundaries.size() >= std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("Polar radial-bin count exceeds the uint32 range");
  }

  float previous = 0.0F;
  for (size_t boundary = 0; boundary < boundaries.size(); ++boundary) {
    const float value = boundaries[boundary];
    if (!std::isfinite(value) || value < 0.0F || (boundary > 0 && value < previous)) {
      throw std::invalid_argument("Polar radial boundaries must be finite and non-decreasing");
    }
    previous = value;
  }

  return static_cast<uint32_t>(std::lower_bound(boundaries.begin(), boundaries.end(), radius) -
                               boundaries.begin());
}

uint32_t encode_polar_cell_id(uint32_t direction_bin, uint32_t radial_bin,
                              uint32_t direction_bin_count, uint32_t radial_bin_count) {
  validate_cell_space(direction_bin_count, radial_bin_count);
  if (direction_bin >= direction_bin_count || radial_bin >= radial_bin_count) {
    throw std::invalid_argument("Polar Cell bin is outside the configured range");
  }
  const uint64_t cell_id = static_cast<uint64_t>(direction_bin) * radial_bin_count + radial_bin;
  return static_cast<uint32_t>(cell_id);
}

polar_cell_t decode_polar_cell_id(uint32_t cell_id, uint32_t direction_bin_count,
                                  uint32_t radial_bin_count) {
  const uint64_t cell_count = validate_cell_space(direction_bin_count, radial_bin_count);
  if (cell_id >= cell_count) {
    throw std::invalid_argument("Polar Cell ID is outside the configured range");
  }
  return polar_cell_t{cell_id / radial_bin_count, cell_id % radial_bin_count};
}

void write_polar_direction_codebook(const std::filesystem::path& path,
                                    const polar_direction_codebook_t& codebook) {
  validate_polar_direction_codebook(codebook);

  std::vector<uint8_t> payload;
  payload.reserve(codebook.centroids.size() * sizeof(float));
  for (const float centroid : codebook.centroids) {
    append_float_le(payload, centroid);
  }

  std::vector<uint8_t> file_bytes;
  file_bytes.reserve(k_polar_codebook_header_size + payload.size());
  append_u64_le(file_bytes, k_polar_codebook_magic);
  append_u32_le(file_bytes, k_polar_codebook_version);
  append_u32_le(file_bytes, k_polar_codebook_header_size);
  append_u32_le(file_bytes, codebook.dimension);
  append_u32_le(file_bytes, codebook.direction_bin_count);
  append_u64_le(file_bytes, codebook.centroids.size());
  append_u64_le(file_bytes, hash_bytes(payload));
  file_bytes.insert(file_bytes.end(), payload.begin(), payload.end());

  std::filesystem::path temporary_path = path;
  temporary_path += ".tmp";
  std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open Polar codebook output: " + temporary_path.string());
  }
  output.write(reinterpret_cast<const char*>(file_bytes.data()),
               static_cast<std::streamsize>(file_bytes.size()));
  output.flush();
  if (!output) {
    output.close();
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::runtime_error("failed to write Polar codebook output: " + temporary_path.string());
  }
  output.close();

  std::error_code rename_error;
  std::filesystem::rename(temporary_path, path, rename_error);
  if (rename_error) {
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::runtime_error("failed to publish Polar codebook output: " + rename_error.message());
  }
}

polar_direction_codebook_t load_polar_direction_codebook(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("failed to open Polar codebook: " + path.string());
  }
  const std::streampos end = input.tellg();
  if (end < 0 || static_cast<uint64_t>(end) >
                     static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("Polar codebook size is invalid: " + path.string());
  }
  std::vector<uint8_t> file_bytes(static_cast<size_t>(end));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(file_bytes.data()), static_cast<std::streamsize>(end));
  if (!input) {
    throw std::runtime_error("failed to read Polar codebook: " + path.string());
  }

  const std::span<const uint8_t> bytes(file_bytes);
  size_t cursor = 0;
  const uint64_t magic = read_u64_le(bytes, cursor);
  const uint32_t version = read_u32_le(bytes, cursor);
  const uint32_t header_size = read_u32_le(bytes, cursor);
  const uint32_t dimension = read_u32_le(bytes, cursor);
  const uint32_t direction_bin_count = read_u32_le(bytes, cursor);
  const uint64_t scalar_count = read_u64_le(bytes, cursor);
  const uint64_t expected_hash = read_u64_le(bytes, cursor);

  if (magic != k_polar_codebook_magic) {
    throw std::runtime_error("Polar codebook magic does not match");
  }
  if (version != k_polar_codebook_version || header_size != k_polar_codebook_header_size) {
    throw std::runtime_error("Polar codebook version or header size is unsupported");
  }
  const uint64_t expected_scalar_count =
      static_cast<uint64_t>(dimension) * static_cast<uint64_t>(direction_bin_count);
  if (scalar_count != expected_scalar_count ||
      scalar_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("Polar codebook scalar count is invalid");
  }
  if (scalar_count > (std::numeric_limits<uint64_t>::max() - header_size) / sizeof(float)) {
    throw std::runtime_error("Polar codebook payload size overflows uint64");
  }
  const uint64_t expected_file_size =
      static_cast<uint64_t>(header_size) + scalar_count * sizeof(float);
  if (expected_file_size != bytes.size()) {
    throw std::runtime_error("Polar codebook payload size is invalid");
  }

  const auto payload = bytes.subspan(header_size);
  if (hash_bytes(payload) != expected_hash) {
    throw std::runtime_error("Polar codebook payload hash does not match");
  }

  polar_direction_codebook_t codebook;
  codebook.dimension = dimension;
  codebook.direction_bin_count = direction_bin_count;
  codebook.centroids.reserve(static_cast<size_t>(scalar_count));
  cursor = header_size;
  for (uint64_t scalar = 0; scalar < scalar_count; ++scalar) {
    codebook.centroids.push_back(read_float_le(bytes, cursor));
  }

  try {
    validate_polar_direction_codebook(codebook);
  } catch (const std::invalid_argument& error) {
    throw std::runtime_error(std::string("Polar codebook payload is invalid: ") + error.what());
  }
  return codebook;
}

} // namespace powerlaw_ann
