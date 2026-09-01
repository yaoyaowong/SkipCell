#include "index/polar.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double k_pi = 3.14159265358979323846;

class polar_temp_dir_t {
public:
  polar_temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_polar_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~polar_temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

powerlaw_ann::polar_direction_codebook_t make_circle_codebook() {
  powerlaw_ann::polar_direction_codebook_t codebook;
  codebook.dimension = 2;
  codebook.direction_bin_count = 16;
  codebook.centroids.reserve(32);
  for (uint32_t bin = 0; bin < codebook.direction_bin_count; ++bin) {
    const double angle =
        2.0 * k_pi * static_cast<double>(bin) / static_cast<double>(codebook.direction_bin_count);
    codebook.centroids.push_back(static_cast<float>(std::cos(angle)));
    codebook.centroids.push_back(static_cast<float>(std::sin(angle)));
  }
  return codebook;
}

std::vector<float> make_training_directions() {
  std::vector<float> directions;
  directions.reserve(128);
  for (uint32_t sample = 0; sample < 64; ++sample) {
    const double base_angle = 2.0 * k_pi * static_cast<double>(sample % 16U) / 16.0;
    const double offset = (static_cast<int>(sample / 16U) - 1.5) * 0.015;
    const double scale = 1.0 + static_cast<double>(sample % 3U);
    directions.push_back(static_cast<float>(scale * std::cos(base_angle + offset)));
    directions.push_back(static_cast<float>(scale * std::sin(base_angle + offset)));
  }
  return directions;
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open Polar test file: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("failed to write Polar test file: " + path.string());
  }
}

TEST(PolarCoordinateTest, ComputesTrueL2RadiusAndUnitDirection) {
  const std::array<float, 2> hub{{1.0F, 2.0F}};
  const std::array<float, 2> point{{4.0F, 6.0F}};
  std::array<float, 2> direction{};

  const auto coordinate = powerlaw_ann::compute_polar_coordinate(hub, point, direction);

  ASSERT_TRUE(coordinate.has_direction);
  EXPECT_FLOAT_EQ(coordinate.radius, 5.0F);
  EXPECT_NEAR(direction[0], 0.6F, 1.0e-6F);
  EXPECT_NEAR(direction[1], 0.8F, 1.0e-6F);
  const double norm = std::sqrt(static_cast<double>(direction[0]) * direction[0] +
                                static_cast<double>(direction[1]) * direction[1]);
  EXPECT_NEAR(norm, 1.0, 1.0e-6);
}

TEST(PolarCoordinateTest, RepresentsZeroDistanceWithoutAnArbitraryDirection) {
  const std::array<float, 3> hub{{1.0F, 2.0F, 3.0F}};
  const std::array<float, 3> same_point = hub;
  const std::array<float, 3> near_point{{1.0F + 1.0e-7F, 2.0F, 3.0F}};
  std::array<float, 3> direction{{1.0F, 1.0F, 1.0F}};

  const auto exact = powerlaw_ann::compute_polar_coordinate(hub, same_point, direction);

  EXPECT_FLOAT_EQ(exact.radius, 0.0F);
  EXPECT_FALSE(exact.has_direction);
  EXPECT_EQ(direction, (std::array<float, 3>{0.0F, 0.0F, 0.0F}));

  direction.fill(1.0F);
  const auto within_epsilon =
      powerlaw_ann::compute_polar_coordinate(hub, near_point, direction, 1.0e-6F);
  EXPECT_FLOAT_EQ(within_epsilon.radius, 0.0F);
  EXPECT_FALSE(within_epsilon.has_direction);
  EXPECT_EQ(direction, (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
}

TEST(PolarCoordinateTest, RejectsInvalidDimensionsAndNumerics) {
  const std::array<float, 2> hub{{0.0F, 0.0F}};
  const std::array<float, 1> short_point{{1.0F}};
  const std::array<float, 2> invalid_point{{1.0F, std::numeric_limits<float>::quiet_NaN()}};
  std::array<float, 2> direction{};

  EXPECT_THROW(powerlaw_ann::compute_polar_coordinate(hub, short_point, direction),
               std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::compute_polar_coordinate(hub, invalid_point, direction),
               std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::compute_polar_coordinate(hub, hub, direction, 0.0F),
               std::invalid_argument);

  std::array<float, 1> short_direction{};
  EXPECT_THROW(powerlaw_ann::compute_polar_coordinate(hub, hub, short_direction),
               std::invalid_argument);
}

TEST(PolarDirectionTest, ArgmaxUsesCosineAndSmallestBinForTies) {
  auto codebook = make_circle_codebook();
  codebook.centroids[2] = codebook.centroids[0];
  codebook.centroids[3] = codebook.centroids[1];
  const std::array<float, 2> direction{{1.0F, 0.0F}};

  EXPECT_NO_THROW(powerlaw_ann::validate_polar_direction_codebook(codebook));
  EXPECT_EQ(powerlaw_ann::find_polar_direction_bin(codebook, direction), 0U);

  const std::array<float, 2> non_unit{{2.0F, 0.0F}};
  EXPECT_THROW(powerlaw_ann::find_polar_direction_bin(codebook, non_unit), std::invalid_argument);
}

TEST(PolarDirectionTest, ValidatesSupportedCodebookShapeAndUnitCentroids) {
  auto codebook = make_circle_codebook();
  codebook.direction_bin_count = 8;
  EXPECT_THROW(powerlaw_ann::validate_polar_direction_codebook(codebook), std::invalid_argument);

  codebook = make_circle_codebook();
  codebook.centroids.pop_back();
  EXPECT_THROW(powerlaw_ann::validate_polar_direction_codebook(codebook), std::invalid_argument);

  codebook = make_circle_codebook();
  codebook.centroids[0] = 2.0F;
  EXPECT_THROW(powerlaw_ann::validate_polar_direction_codebook(codebook), std::invalid_argument);
}

TEST(SphericalKMeansTest, WeightedTrainingIsDeterministicAndProducesUnitCentroids) {
  const auto directions = make_training_directions();
  std::vector<double> weights(directions.size() / 2);
  for (size_t sample = 0; sample < weights.size(); ++sample) {
    weights[sample] = 1.0 + static_cast<double>(sample % 5U);
  }
  powerlaw_ann::spherical_kmeans_config_t config;
  config.direction_bin_count = 16;
  config.max_iterations = 30;
  config.seed = 17;

  const auto first = powerlaw_ann::train_weighted_spherical_kmeans(directions, 2, weights, config);
  const auto second = powerlaw_ann::train_weighted_spherical_kmeans(directions, 2, weights, config);

  EXPECT_EQ(first.iterations, second.iterations);
  EXPECT_EQ(first.codebook.centroids, second.codebook.centroids);
  EXPECT_DOUBLE_EQ(first.weighted_mean_cosine, second.weighted_mean_cosine);
  EXPECT_GT(first.iterations, 0U);
  EXPECT_LE(first.iterations, config.max_iterations);
  EXPECT_GT(first.weighted_mean_cosine, 0.99);
  EXPECT_LE(first.weighted_mean_cosine, 1.0);
  EXPECT_NO_THROW(powerlaw_ann::validate_polar_direction_codebook(first.codebook));
}

TEST(SphericalKMeansTest, SupportsEveryApprovedDirectionBinCount) {
  const auto directions = make_training_directions();
  const std::vector<double> weights(directions.size() / 2, 1.0);

  for (const uint32_t direction_bin_count : {16U, 32U, 64U}) {
    powerlaw_ann::spherical_kmeans_config_t config;
    config.direction_bin_count = direction_bin_count;
    config.max_iterations = 10;
    config.seed = 3;

    const auto result =
        powerlaw_ann::train_weighted_spherical_kmeans(directions, 2, weights, config);
    EXPECT_EQ(result.codebook.direction_bin_count, direction_bin_count);
    EXPECT_NO_THROW(powerlaw_ann::validate_polar_direction_codebook(result.codebook));
  }
}

TEST(SphericalKMeansTest, RejectsInvalidSamplesWeightsAndConfiguration) {
  const auto directions = make_training_directions();
  std::vector<double> weights(directions.size() / 2, 1.0);
  powerlaw_ann::spherical_kmeans_config_t config;
  config.direction_bin_count = 8;
  config.max_iterations = 10;
  EXPECT_THROW(powerlaw_ann::train_weighted_spherical_kmeans(directions, 2, weights, config),
               std::invalid_argument);

  config.direction_bin_count = 16;
  weights[0] = -1.0;
  EXPECT_THROW(powerlaw_ann::train_weighted_spherical_kmeans(directions, 2, weights, config),
               std::invalid_argument);

  weights.assign(directions.size() / 2, 1.0);
  auto zero_direction = directions;
  zero_direction[0] = 0.0F;
  zero_direction[1] = 0.0F;
  EXPECT_THROW(powerlaw_ann::train_weighted_spherical_kmeans(zero_direction, 2, weights, config),
               std::invalid_argument);

  weights.resize(15);
  EXPECT_THROW(powerlaw_ann::train_weighted_spherical_kmeans(directions, 2, weights, config),
               std::invalid_argument);
}

TEST(PolarRadialTest, NearestRankQuantilesAndInclusiveBoundariesAreDeterministic) {
  const std::array<float, 8> radii{{8.0F, 1.0F, 6.0F, 3.0F, 5.0F, 2.0F, 7.0F, 4.0F}};
  const auto boundaries = powerlaw_ann::compute_radial_quantile_boundaries(radii, 4);

  ASSERT_EQ(boundaries, (std::vector<float>{2.0F, 4.0F, 6.0F}));
  EXPECT_EQ(powerlaw_ann::find_polar_radial_bin(0.0F, boundaries), 0U);
  EXPECT_EQ(powerlaw_ann::find_polar_radial_bin(2.0F, boundaries), 0U);
  EXPECT_EQ(powerlaw_ann::find_polar_radial_bin(2.1F, boundaries), 1U);
  EXPECT_EQ(powerlaw_ann::find_polar_radial_bin(4.0F, boundaries), 1U);
  EXPECT_EQ(powerlaw_ann::find_polar_radial_bin(6.0F, boundaries), 2U);
  EXPECT_EQ(powerlaw_ann::find_polar_radial_bin(7.0F, boundaries), 3U);
}

TEST(PolarRadialTest, RetainsRepeatedQuantilesAndSupportsOneBin) {
  const std::array<float, 3> repeated{{5.0F, 5.0F, 5.0F}};
  const auto boundaries = powerlaw_ann::compute_radial_quantile_boundaries(repeated, 4);
  EXPECT_EQ(boundaries, (std::vector<float>{5.0F, 5.0F, 5.0F}));
  EXPECT_EQ(powerlaw_ann::find_polar_radial_bin(5.0F, boundaries), 0U);
  EXPECT_EQ(powerlaw_ann::find_polar_radial_bin(5.1F, boundaries), 3U);

  EXPECT_TRUE(powerlaw_ann::compute_radial_quantile_boundaries(repeated, 1).empty());
}

TEST(PolarRadialTest, RejectsInvalidRadiiAndBoundaries) {
  const std::array<float, 0> empty{};
  const std::array<float, 1> invalid_radius{{std::numeric_limits<float>::infinity()}};
  const std::array<float, 2> descending{{2.0F, 1.0F}};

  EXPECT_THROW(powerlaw_ann::compute_radial_quantile_boundaries(empty, 4), std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::compute_radial_quantile_boundaries(invalid_radius, 4),
               std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::compute_radial_quantile_boundaries(invalid_radius, 0),
               std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::find_polar_radial_bin(1.0F, descending), std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::find_polar_radial_bin(-1.0F, {}), std::invalid_argument);
}

TEST(PolarCellTest, EncodesAndDecodesValidIds) {
  const uint32_t cell_id = powerlaw_ann::encode_polar_cell_id(3, 2, 16, 4);
  EXPECT_EQ(cell_id, 14U);

  const auto cell = powerlaw_ann::decode_polar_cell_id(cell_id, 16, 4);
  EXPECT_EQ(cell.direction_bin, 3U);
  EXPECT_EQ(cell.radial_bin, 2U);
}

TEST(PolarCellTest, RejectsInvalidBinsIdsAndOverflow) {
  EXPECT_THROW(powerlaw_ann::encode_polar_cell_id(16, 0, 16, 4), std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::encode_polar_cell_id(0, 4, 16, 4), std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::decode_polar_cell_id(64, 16, 4), std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::decode_polar_cell_id(0, 8, 4), std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::decode_polar_cell_id(0, 64, std::numeric_limits<uint32_t>::max()),
               std::invalid_argument);
}

TEST(PolarCodebookIoTest, RoundTripIsByteStableAndPreservesAssignments) {
  polar_temp_dir_t temp_dir;
  const auto first_path = temp_dir.path() / "first.polar.bin";
  const auto second_path = temp_dir.path() / "second.polar.bin";
  const auto codebook = make_circle_codebook();

  powerlaw_ann::write_polar_direction_codebook(first_path, codebook);
  powerlaw_ann::write_polar_direction_codebook(second_path, codebook);
  const auto loaded = powerlaw_ann::load_polar_direction_codebook(first_path);

  EXPECT_EQ(read_bytes(first_path), read_bytes(second_path));
  EXPECT_EQ(loaded.dimension, codebook.dimension);
  EXPECT_EQ(loaded.direction_bin_count, codebook.direction_bin_count);
  EXPECT_EQ(loaded.centroids, codebook.centroids);
  const std::array<float, 2> direction{{1.0F, 0.0F}};
  EXPECT_EQ(powerlaw_ann::find_polar_direction_bin(loaded, direction), 0U);
}

TEST(PolarCodebookIoTest, RejectsMagicHashAndTruncationCorruption) {
  polar_temp_dir_t temp_dir;
  const auto valid_path = temp_dir.path() / "valid.polar.bin";
  const auto magic_path = temp_dir.path() / "bad_magic.polar.bin";
  const auto hash_path = temp_dir.path() / "bad_hash.polar.bin";
  const auto truncated_path = temp_dir.path() / "truncated.polar.bin";
  powerlaw_ann::write_polar_direction_codebook(valid_path, make_circle_codebook());
  const auto valid_bytes = read_bytes(valid_path);

  auto bad_magic = valid_bytes;
  bad_magic.front() ^= 0xFFU;
  write_bytes(magic_path, bad_magic);
  EXPECT_THROW(powerlaw_ann::load_polar_direction_codebook(magic_path), std::runtime_error);

  auto bad_hash = valid_bytes;
  bad_hash.back() ^= 0x01U;
  write_bytes(hash_path, bad_hash);
  EXPECT_THROW(powerlaw_ann::load_polar_direction_codebook(hash_path), std::runtime_error);

  auto truncated = valid_bytes;
  truncated.pop_back();
  write_bytes(truncated_path, truncated);
  EXPECT_THROW(powerlaw_ann::load_polar_direction_codebook(truncated_path), std::runtime_error);
}

} // namespace
