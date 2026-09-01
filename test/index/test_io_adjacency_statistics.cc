#include "index/io_adjacency_statistics.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <random>
#include <span>
#include <vector>

namespace {

inline constexpr uint64_t k_topology_magic = 0x31504F544F494C50ULL;
inline constexpr uint64_t k_fnv_offset_basis = 14695981039346656037ULL;
inline constexpr uint64_t k_fnv_prime = 1099511628211ULL;

class temp_dir_t {
public:
  temp_dir_t() {
    std::mt19937_64 generator(1234567);
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann-groupvarint-test-" + std::to_string(generator()));
    std::filesystem::create_directories(path_);
  }
  ~temp_dir_t() { std::filesystem::remove_all(path_); }
  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

template <typename value_t> void append_little(std::vector<uint8_t>& output, value_t value) {
  for (size_t byte = 0; byte < sizeof(value_t); ++byte) {
    output.push_back(static_cast<uint8_t>(value >> (byte * 8U)));
  }
}

uint64_t checksum(std::span<const uint8_t> bytes) {
  uint64_t value = k_fnv_offset_basis;
  for (const uint8_t byte : bytes) {
    value ^= byte;
    value *= k_fnv_prime;
  }
  return value;
}

std::filesystem::path write_raw_topology(const std::filesystem::path& path) {
  std::vector<std::vector<uint32_t>> lists = {
      {3, 2, 1, 0, 2},
      {},
      {1},
      {},
  };
  for (uint32_t value = 0; value < 64; ++value) {
    lists[3].push_back((value * 37U + 11U) % 4U);
  }
  std::vector<uint8_t> offset_bytes;
  std::vector<uint8_t> adjacency_bytes;
  uint64_t edge_count = 0;
  append_little(offset_bytes, edge_count);
  for (const auto& list : lists) {
    for (const uint32_t target : list) {
      append_little(adjacency_bytes, target);
    }
    edge_count += list.size();
    append_little(offset_bytes, edge_count);
  }

  std::vector<uint8_t> header;
  append_little(header, k_topology_magic);
  append_little(header, uint32_t{6});
  append_little(header, uint32_t{0x01020304});
  append_little(header, uint32_t{320});
  append_little(header, uint32_t{4096});
  append_little(header, static_cast<uint64_t>(lists.size()));
  append_little(header, uint64_t{8});
  append_little(header, uint32_t{64});
  append_little(header, uint32_t{0});
  append_little(header, uint32_t{0});
  append_little(header, edge_count);
  append_little(header, uint64_t{1});
  append_little(header, uint64_t{0});
  append_little(header, uint64_t{0});
  append_little(header, uint64_t{0});
  append_little(header, uint64_t{0});
  append_little(header, uint64_t{0});
  append_little(header, uint64_t{320});
  append_little(header, static_cast<uint64_t>(offset_bytes.size()));
  append_little(header, checksum(offset_bytes));
  append_little(header, static_cast<uint64_t>(320 + offset_bytes.size()));
  append_little(header, static_cast<uint64_t>(adjacency_bytes.size()));
  append_little(header, checksum(adjacency_bytes));
  header.resize(320, 0);

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(header.data()), header.size());
  output.write(reinterpret_cast<const char*>(offset_bytes.data()), offset_bytes.size());
  output.write(reinterpret_cast<const char*>(adjacency_bytes.data()), adjacency_bytes.size());
  output.close();
  return path;
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void flip_last_byte(const std::filesystem::path& path) {
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  file.seekg(-1, std::ios::end);
  char value = 0;
  file.read(&value, 1);
  value ^= 1;
  file.seekp(-1, std::ios::end);
  file.write(&value, 1);
}

} // namespace

TEST(IoAdjacencyStatisticsTest, BuildsDeterministicDegree64GroupVarintArtifacts) {
  temp_dir_t temp;
  const auto raw = write_raw_topology(temp.path() / "raw.topology");
  const auto unordered_a = temp.path() / "unordered-a.stats";
  const auto unordered_b = temp.path() / "unordered-b.stats";
  const auto preserving_a = temp.path() / "preserving-a.stats";
  const auto preserving_b = temp.path() / "preserving-b.stats";

  const auto unordered =
      powerlaw_ann::build_io_groupvarint_adjacency_statistics(raw, unordered_a, false);
  powerlaw_ann::build_io_groupvarint_adjacency_statistics(raw, unordered_b, false);
  const auto preserving =
      powerlaw_ann::build_io_groupvarint_adjacency_statistics(raw, preserving_a, true);
  powerlaw_ann::build_io_groupvarint_adjacency_statistics(raw, preserving_b, true);

  EXPECT_EQ(read_file(unordered_a), read_file(unordered_b));
  EXPECT_EQ(read_file(preserving_a), read_file(preserving_b));
  EXPECT_FALSE(unordered.preserves_neighbor_order);
  EXPECT_TRUE(preserving.preserves_neighbor_order);
  EXPECT_EQ(unordered.maximum_degree, 64U);
  EXPECT_EQ(unordered.edge_count, 70U);
  EXPECT_EQ(unordered.permutation_bytes, 0U);
  EXPECT_EQ(preserving.permutation_bytes, 50U);
  EXPECT_LT(unordered.compressed_payload_bytes, unordered.raw_adjacency_bytes);
  EXPECT_GT(preserving.compressed_payload_bytes, unordered.compressed_payload_bytes);
  EXPECT_EQ(
      powerlaw_ann::verify_io_groupvarint_adjacency_statistics(raw, unordered_a).artifact_checksum,
      unordered.artifact_checksum);
  EXPECT_EQ(
      powerlaw_ann::verify_io_groupvarint_adjacency_statistics(raw, preserving_a).artifact_checksum,
      preserving.artifact_checksum);

  flip_last_byte(preserving_a);
  EXPECT_THROW(powerlaw_ann::verify_io_groupvarint_adjacency_statistics(raw, preserving_a),
               std::runtime_error);
}

TEST(IoAdjacencyStatisticsTest, RejectsOutputOverwriteAndNonTopologyInput) {
  temp_dir_t temp;
  const auto raw = write_raw_topology(temp.path() / "raw.topology");
  const auto output = temp.path() / "stats.bin";
  powerlaw_ann::build_io_groupvarint_adjacency_statistics(raw, output, false);
  EXPECT_THROW(powerlaw_ann::build_io_groupvarint_adjacency_statistics(raw, output, false),
               std::invalid_argument);
  EXPECT_THROW(powerlaw_ann::build_io_groupvarint_adjacency_statistics(
                   output, temp.path() / "bad.bin", false),
               std::runtime_error);
}
