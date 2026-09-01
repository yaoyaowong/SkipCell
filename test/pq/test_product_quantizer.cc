#include "pq/product_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {

TEST(ProductQuantizerTest, GeneratesPivotsAndCompressedCodes) {
  constexpr size_t num_points = 256;
  constexpr size_t dimension = 4;
  constexpr size_t num_chunks = 2;

  std::vector<float> data(num_points * dimension);
  for (size_t point = 0; point < num_points; ++point) {
    data[point * dimension] = static_cast<float>(point);
    data[point * dimension + 1] = static_cast<float>((point * 7) % 31);
    data[point * dimension + 2] = std::sin(static_cast<float>(point) * 0.1F);
    data[point * dimension + 3] = std::cos(static_cast<float>(point) * 0.1F);
  }

  std::vector<float> pivots;
  ASSERT_EQ(powerlaw_ann::generate_pq_pivots_simplified(data.data(), num_points, dimension,
                                                        num_chunks, pivots),
            0);
  ASSERT_EQ(pivots.size(), 256U * dimension);
  for (const float pivot : pivots) {
    EXPECT_TRUE(std::isfinite(pivot));
  }

  std::vector<uint8_t> compressed;
  ASSERT_EQ(
      powerlaw_ann::generate_pq_data_from_pivots_simplified(
          data.data(), num_points, pivots.data(), pivots.size(), dimension, num_chunks, compressed),
      0);
  EXPECT_EQ(compressed.size(), num_points * num_chunks);
}

TEST(ProductQuantizerTest, RejectsInvalidChunkPartition) {
  const std::vector<float> data(256 * 4, 0.0F);
  std::vector<float> pivots;
  EXPECT_EQ(powerlaw_ann::generate_pq_pivots_simplified(data.data(), 256, 4, 3, pivots), -1);
}

TEST(ProductQuantizerTest, PqLookupMatchesScalarForVectorAndTailCandidates) {
  constexpr size_t points = 13;
  constexpr size_t chunks = 64;
  std::vector<uint8_t> codes(points * chunks);
  std::vector<float> tables(chunks * 256);
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    for (size_t center = 0; center < 256; ++center) {
      tables[chunk * 256 + center] =
          static_cast<float>((chunk * 17 + center * 3) % 101) * 0.25F;
    }
    for (size_t point = 0; point < points; ++point) {
      codes[point * chunks + chunk] = static_cast<uint8_t>((point * 29 + chunk * 7) % 256);
    }
  }

  std::vector<float> expected(points, 0.0F);
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    for (size_t point = 0; point < points; ++point) {
      expected[point] += tables[chunk * 256 + codes[point * chunks + chunk]];
    }
  }
  std::vector<float> actual(points);
  powerlaw_ann::pq_dist_lookup(codes.data(), points, chunks, tables.data(), actual.data());
  EXPECT_EQ(actual, expected);
}

TEST(ProductQuantizerTest, DecodedUint8LookupMatchesScalarSquaredL2) {
  constexpr size_t points = 3;
  constexpr size_t dimensions = 19;
  std::vector<uint8_t> vectors(points * dimensions);
  std::vector<uint8_t> query(dimensions);
  for (size_t dimension = 0; dimension < dimensions; ++dimension) {
    query[dimension] = static_cast<uint8_t>(dimension * 7);
    for (size_t point = 0; point < points; ++point) {
      vectors[point * dimensions + dimension] = static_cast<uint8_t>(dimension * 5 + point * 11);
    }
  }

  std::vector<float> actual(points);
  powerlaw_ann::decoded_u8_l2_lookup(vectors.data(), points, dimensions, query.data(),
                                     actual.data());
  for (size_t point = 0; point < points; ++point) {
    uint32_t expected = 0;
    for (size_t dimension = 0; dimension < dimensions; ++dimension) {
      const int32_t difference = static_cast<int32_t>(vectors[point * dimensions + dimension]) -
                                 static_cast<int32_t>(query[dimension]);
      expected += static_cast<uint32_t>(difference * difference);
    }
    EXPECT_FLOAT_EQ(actual[point], static_cast<float>(expected));
  }
}

TEST(ProductQuantizerTest, OriginalIdUint8LookupMatchesContiguousLookup) {
  constexpr size_t points = 5;
  constexpr size_t dimensions = 32;
  const std::vector<uint32_t> ids = {4, 1, 3};
  std::vector<uint8_t> vectors(points * dimensions);
  std::vector<uint8_t> gathered(ids.size() * dimensions);
  std::vector<uint8_t> query(dimensions);
  for (size_t dimension = 0; dimension < dimensions; ++dimension) {
    query[dimension] = static_cast<uint8_t>(dimension * 3);
    for (size_t point = 0; point < points; ++point) {
      vectors[point * dimensions + dimension] =
          static_cast<uint8_t>(point * 17 + dimension);
    }
  }
  for (size_t position = 0; position < ids.size(); ++position) {
    std::copy_n(vectors.data() + ids[position] * dimensions, dimensions,
                gathered.data() + position * dimensions);
  }
  std::vector<float> expected(ids.size());
  std::vector<float> actual(ids.size());
  powerlaw_ann::decoded_u8_l2_lookup(gathered.data(), ids.size(), dimensions, query.data(),
                                     expected.data());
  powerlaw_ann::original_u8_l2_lookup(vectors.data(), ids.data(), ids.size(), dimensions,
                                      query.data(), actual.data());
  EXPECT_EQ(actual, expected);
}

} // namespace
