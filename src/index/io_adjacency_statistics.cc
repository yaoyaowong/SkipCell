#include "index/io_adjacency_statistics.h"

#include "index/io_optimized_index.h"
#include "storage/memory_mapper.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace powerlaw_ann {
namespace {

inline constexpr uint64_t k_topology_magic = 0x31504F544F494C50ULL;
inline constexpr uint64_t k_statistics_magic = 0x5441545356474C50ULL; // "PLGVSTAT"
inline constexpr uint32_t k_statistics_version = 1;
inline constexpr uint32_t k_statistics_header_bytes = 192;
inline constexpr uint32_t k_little_endian_marker = 0x01020304U;
inline constexpr uint32_t k_groupvarint_delta_codec = 1;
inline constexpr uint32_t k_statistics_only_flag = 1U << 0U;
inline constexpr uint32_t k_sorted_adjacency_flag = 1U << 1U;
inline constexpr uint32_t k_preserves_neighbor_order_flag = 1U << 2U;
inline constexpr uint64_t k_fnv_offset_basis = 14695981039346656037ULL;
inline constexpr uint64_t k_fnv_prime = 1099511628211ULL;

struct raw_topology_view_t {
  std::span<const uint8_t> file;
  uint64_t point_count = 0;
  uint64_t edge_count = 0;
  uint32_t maximum_degree = 0;
  uint64_t offsets_offset = 0;
  uint64_t offsets_bytes = 0;
  uint64_t offsets_checksum = 0;
  uint64_t adjacency_offset = 0;
  uint64_t adjacency_bytes = 0;
  uint64_t adjacency_checksum = 0;
};

struct statistics_header_t {
  uint32_t flags = k_statistics_only_flag | k_sorted_adjacency_flag;
  uint64_t point_count = 0;
  uint64_t edge_count = 0;
  uint32_t maximum_degree = 0;
  uint64_t source_topology_bytes = 0;
  uint64_t source_topology_checksum = 0;
  uint64_t raw_adjacency_bytes = 0;
  uint64_t source_adjacency_checksum = 0;
  uint64_t offsets_offset = k_statistics_header_bytes;
  uint64_t offsets_bytes = 0;
  uint64_t offsets_checksum = 0;
  uint64_t payload_offset = 0;
  uint64_t payload_bytes = 0;
  uint64_t payload_checksum = 0;
  uint64_t permutation_bytes = 0;
};

uint64_t checked_add(uint64_t left, uint64_t right, const char* description) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    throw std::runtime_error(std::string(description) + " overflows uint64");
  }
  return left + right;
}

uint64_t checked_multiply(uint64_t left, uint64_t right, const char* description) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    throw std::runtime_error(std::string(description) + " overflows uint64");
  }
  return left * right;
}

void hash_bytes(uint64_t& checksum, std::span<const uint8_t> bytes) {
  for (const uint8_t value : bytes) {
    checksum ^= value;
    checksum *= k_fnv_prime;
  }
}

uint64_t checksum_bytes(std::span<const uint8_t> bytes) {
  uint64_t checksum = k_fnv_offset_basis;
  hash_bytes(checksum, bytes);
  return checksum;
}

template <typename value_t> void append_little(std::vector<uint8_t>& output, value_t value) {
  static_assert(std::is_integral_v<value_t>);
  using unsigned_t = std::make_unsigned_t<value_t>;
  const auto unsigned_value = static_cast<unsigned_t>(value);
  for (size_t byte = 0; byte < sizeof(value_t); ++byte) {
    output.push_back(static_cast<uint8_t>(unsigned_value >> (byte * 8U)));
  }
}

template <typename value_t>
value_t read_little(std::span<const uint8_t> input, size_t& cursor, const char* description) {
  static_assert(std::is_integral_v<value_t>);
  if (cursor > input.size() || input.size() - cursor < sizeof(value_t)) {
    throw std::runtime_error(std::string("truncated ") + description);
  }
  using unsigned_t = std::make_unsigned_t<value_t>;
  unsigned_t value = 0;
  for (size_t byte = 0; byte < sizeof(value_t); ++byte) {
    value |= static_cast<unsigned_t>(input[cursor++]) << (byte * 8U);
  }
  return static_cast<value_t>(value);
}

uint64_t raw_offset(const raw_topology_view_t& topology, uint64_t node) {
  size_t cursor = static_cast<size_t>(topology.offsets_offset + node * sizeof(uint64_t));
  return read_little<uint64_t>(topology.file, cursor, "raw topology neighbor offset");
}

uint32_t raw_neighbor(const raw_topology_view_t& topology, uint64_t edge) {
  size_t cursor = static_cast<size_t>(topology.adjacency_offset + edge * sizeof(uint32_t));
  return read_little<uint32_t>(topology.file, cursor, "raw topology adjacency target");
}

raw_topology_view_t parse_raw_topology(std::span<const uint8_t> file) {
  if (file.size() < 256) {
    throw std::runtime_error("truncated raw IO topology header");
  }
  size_t cursor = 0;
  if (read_little<uint64_t>(file, cursor, "raw topology magic") != k_topology_magic) {
    throw std::runtime_error("statistics input is not an IO topology artifact");
  }
  const uint32_t version = read_little<uint32_t>(file, cursor, "raw topology version");
  if (version < 4 || version > 6 ||
      read_little<uint32_t>(file, cursor, "raw topology byte-order marker") !=
          k_little_endian_marker) {
    throw std::runtime_error("unsupported raw IO topology version or byte order");
  }
  const uint32_t header_bytes = read_little<uint32_t>(file, cursor, "raw topology header size");
  if ((version == 4 && header_bytes != 256) || (version >= 5 && header_bytes != 320) ||
      file.size() < header_bytes ||
      read_little<uint32_t>(file, cursor, "raw topology page size") != k_io_vector_page_size) {
    throw std::runtime_error("invalid raw IO topology header");
  }

  raw_topology_view_t result;
  result.file = file;
  result.point_count = read_little<uint64_t>(file, cursor, "raw topology point count");
  static_cast<void>(read_little<uint64_t>(file, cursor, "raw topology dimension"));
  result.maximum_degree = read_little<uint32_t>(file, cursor, "raw topology maximum degree");
  const uint32_t encoding = read_little<uint32_t>(file, cursor, "raw topology encoding");
  static_cast<void>(read_little<uint32_t>(file, cursor, "raw topology vector layout"));
  result.edge_count = read_little<uint64_t>(file, cursor, "raw topology edge count");
  static_cast<void>(read_little<uint64_t>(file, cursor, "raw topology vector page count"));
  static_cast<void>(read_little<uint64_t>(file, cursor, "raw topology Cell count"));
  static_cast<void>(read_little<uint64_t>(file, cursor, "raw topology Base size"));
  static_cast<void>(read_little<uint64_t>(file, cursor, "raw topology Base checksum"));
  static_cast<void>(read_little<uint64_t>(file, cursor, "raw topology vector size"));
  static_cast<void>(read_little<uint64_t>(file, cursor, "raw topology vector checksum"));
  result.offsets_offset = read_little<uint64_t>(file, cursor, "raw topology offsets offset");
  result.offsets_bytes = read_little<uint64_t>(file, cursor, "raw topology offsets size");
  result.offsets_checksum = read_little<uint64_t>(file, cursor, "raw topology offsets checksum");
  result.adjacency_offset = read_little<uint64_t>(file, cursor, "raw topology adjacency offset");
  result.adjacency_bytes = read_little<uint64_t>(file, cursor, "raw topology adjacency size");
  result.adjacency_checksum =
      read_little<uint64_t>(file, cursor, "raw topology adjacency checksum");

  const uint64_t expected_offsets =
      checked_multiply(checked_add(result.point_count, 1, "raw topology offset count"),
                       sizeof(uint64_t), "raw topology offset bytes");
  const uint64_t expected_adjacency =
      checked_multiply(result.edge_count, sizeof(uint32_t), "raw topology adjacency bytes");
  if (result.point_count == 0 || result.maximum_degree == 0 || encoding != 0 ||
      result.offsets_offset != header_bytes || result.offsets_bytes != expected_offsets ||
      result.adjacency_offset != result.offsets_offset + result.offsets_bytes ||
      result.adjacency_bytes != expected_adjacency || result.adjacency_offset > file.size() ||
      result.adjacency_bytes > file.size() - result.adjacency_offset) {
    throw std::runtime_error("statistics input requires compatible raw-U32 topology");
  }
  const auto offsets = file.subspan(static_cast<size_t>(result.offsets_offset),
                                    static_cast<size_t>(result.offsets_bytes));
  const auto adjacency = file.subspan(static_cast<size_t>(result.adjacency_offset),
                                      static_cast<size_t>(result.adjacency_bytes));
  if (checksum_bytes(offsets) != result.offsets_checksum ||
      checksum_bytes(adjacency) != result.adjacency_checksum) {
    throw std::runtime_error("raw IO topology adjacency checksum mismatch");
  }
  if (raw_offset(result, 0) != 0 || raw_offset(result, result.point_count) != result.edge_count) {
    throw std::runtime_error("raw IO topology offsets do not cover adjacency");
  }
  uint64_t previous = 0;
  for (uint64_t node = 0; node < result.point_count; ++node) {
    const uint64_t next = raw_offset(result, node + 1);
    if (next < previous || next - previous > result.maximum_degree) {
      throw std::runtime_error("raw IO topology degree exceeds its declared bound");
    }
    previous = next;
  }
  return result;
}

uint8_t byte_width(uint32_t value) {
  if (value <= UINT8_MAX) {
    return 1;
  }
  if (value <= UINT16_MAX) {
    return 2;
  }
  if (value <= 0x00FFFFFFU) {
    return 3;
  }
  return 4;
}

void append_packed_permutation(std::vector<uint8_t>& output,
                               std::span<const std::pair<uint32_t, uint8_t>> ordered,
                               uint8_t bits) {
  if (bits == 0) {
    return;
  }
  const size_t bytes = (ordered.size() * bits + 7U) / 8U;
  const size_t begin = output.size();
  output.resize(begin + bytes, 0);
  for (size_t index = 0; index < ordered.size(); ++index) {
    const size_t bit = index * bits;
    const size_t byte = bit / 8U;
    const uint32_t shifted = static_cast<uint32_t>(ordered[index].second) << (bit % 8U);
    output[begin + byte] |= static_cast<uint8_t>(shifted);
    if (byte + 1U < bytes) {
      output[begin + byte + 1U] |= static_cast<uint8_t>(shifted >> 8U);
    }
  }
}

uint8_t read_packed_permutation(std::span<const uint8_t> bytes, size_t index, uint8_t bits) {
  if (bits == 0) {
    return 0;
  }
  const size_t bit = index * bits;
  const size_t byte = bit / 8U;
  uint16_t packed = bytes[byte];
  if (byte + 1U < bytes.size()) {
    packed |= static_cast<uint16_t>(bytes[byte + 1U]) << 8U;
  }
  return static_cast<uint8_t>((packed >> (bit % 8U)) & ((1U << bits) - 1U));
}

void append_groupvarint_deltas(std::vector<uint8_t>& output,
                               std::span<const std::pair<uint32_t, uint8_t>> ordered) {
  for (size_t group = 0; group < ordered.size(); group += 4U) {
    const size_t count = std::min<size_t>(4, ordered.size() - group);
    std::array<uint32_t, 4> deltas{};
    std::array<uint8_t, 4> widths{1, 1, 1, 1};
    uint8_t tag = 0;
    for (size_t index = 0; index < count; ++index) {
      const size_t position = group + index;
      deltas[index] = position == 0 ? ordered[position].first
                                    : ordered[position].first - ordered[position - 1U].first;
      widths[index] = byte_width(deltas[index]);
      tag |= static_cast<uint8_t>((widths[index] - 1U) << (index * 2U));
    }
    output.push_back(tag);
    for (size_t index = 0; index < count; ++index) {
      for (uint8_t byte = 0; byte < widths[index]; ++byte) {
        output.push_back(static_cast<uint8_t>(deltas[index] >> (byte * 8U)));
      }
    }
  }
}

std::vector<uint8_t> serialize_offsets(std::span<const uint64_t> offsets) {
  std::vector<uint8_t> bytes;
  bytes.reserve(offsets.size() * sizeof(uint64_t));
  for (const uint64_t value : offsets) {
    append_little(bytes, value);
  }
  return bytes;
}

std::vector<uint8_t> serialize_statistics_header(const statistics_header_t& header) {
  std::vector<uint8_t> bytes;
  bytes.reserve(k_statistics_header_bytes);
  append_little(bytes, k_statistics_magic);
  append_little(bytes, k_statistics_version);
  append_little(bytes, k_little_endian_marker);
  append_little(bytes, k_statistics_header_bytes);
  append_little(bytes, k_groupvarint_delta_codec);
  append_little(bytes, header.flags);
  append_little(bytes, uint32_t{0});
  append_little(bytes, header.point_count);
  append_little(bytes, header.edge_count);
  append_little(bytes, header.maximum_degree);
  append_little(bytes, uint32_t{0});
  append_little(bytes, header.source_topology_bytes);
  append_little(bytes, header.source_topology_checksum);
  append_little(bytes, header.raw_adjacency_bytes);
  append_little(bytes, header.source_adjacency_checksum);
  append_little(bytes, header.offsets_offset);
  append_little(bytes, header.offsets_bytes);
  append_little(bytes, header.offsets_checksum);
  append_little(bytes, header.payload_offset);
  append_little(bytes, header.payload_bytes);
  append_little(bytes, header.payload_checksum);
  append_little(bytes, header.permutation_bytes);
  bytes.resize(k_statistics_header_bytes, 0);
  return bytes;
}

statistics_header_t parse_statistics_header(std::span<const uint8_t> file) {
  if (file.size() < k_statistics_header_bytes) {
    throw std::runtime_error("truncated GroupVarint statistics header");
  }
  size_t cursor = 0;
  if (read_little<uint64_t>(file, cursor, "statistics magic") != k_statistics_magic ||
      read_little<uint32_t>(file, cursor, "statistics version") != k_statistics_version ||
      read_little<uint32_t>(file, cursor, "statistics byte-order marker") !=
          k_little_endian_marker ||
      read_little<uint32_t>(file, cursor, "statistics header size") != k_statistics_header_bytes ||
      read_little<uint32_t>(file, cursor, "statistics codec") != k_groupvarint_delta_codec) {
    throw std::runtime_error("invalid GroupVarint statistics artifact identity");
  }
  statistics_header_t header;
  header.flags = read_little<uint32_t>(file, cursor, "statistics flags");
  static_cast<void>(read_little<uint32_t>(file, cursor, "statistics reserved field"));
  header.point_count = read_little<uint64_t>(file, cursor, "statistics point count");
  header.edge_count = read_little<uint64_t>(file, cursor, "statistics edge count");
  header.maximum_degree = read_little<uint32_t>(file, cursor, "statistics maximum degree");
  static_cast<void>(read_little<uint32_t>(file, cursor, "statistics reserved field"));
  header.source_topology_bytes =
      read_little<uint64_t>(file, cursor, "statistics source topology size");
  header.source_topology_checksum =
      read_little<uint64_t>(file, cursor, "statistics source topology checksum");
  header.raw_adjacency_bytes = read_little<uint64_t>(file, cursor, "statistics raw adjacency size");
  header.source_adjacency_checksum =
      read_little<uint64_t>(file, cursor, "statistics source adjacency checksum");
  header.offsets_offset = read_little<uint64_t>(file, cursor, "statistics offsets offset");
  header.offsets_bytes = read_little<uint64_t>(file, cursor, "statistics offsets size");
  header.offsets_checksum = read_little<uint64_t>(file, cursor, "statistics offsets checksum");
  header.payload_offset = read_little<uint64_t>(file, cursor, "statistics payload offset");
  header.payload_bytes = read_little<uint64_t>(file, cursor, "statistics payload size");
  header.payload_checksum = read_little<uint64_t>(file, cursor, "statistics payload checksum");
  header.permutation_bytes = read_little<uint64_t>(file, cursor, "statistics permutation bytes");
  return header;
}

io_adjacency_statistics_result_t make_result(const statistics_header_t& header,
                                             uint64_t artifact_bytes, uint64_t artifact_checksum) {
  return {header.point_count,
          header.edge_count,
          header.maximum_degree,
          header.raw_adjacency_bytes,
          header.offsets_bytes,
          header.payload_bytes,
          header.permutation_bytes,
          artifact_bytes,
          header.offsets_bytes + header.payload_bytes,
          artifact_checksum,
          (header.flags & k_preserves_neighbor_order_flag) != 0};
}

} // namespace

io_adjacency_statistics_result_t
build_io_groupvarint_adjacency_statistics(const std::filesystem::path& raw_topology_path,
                                          const std::filesystem::path& statistics_path,
                                          bool preserve_neighbor_order) {
  if (raw_topology_path.empty() || statistics_path.empty() ||
      !std::filesystem::is_regular_file(raw_topology_path)) {
    throw std::invalid_argument("GroupVarint statistics build requires input and output paths");
  }
  if (std::filesystem::exists(statistics_path)) {
    throw std::invalid_argument("GroupVarint statistics output already exists");
  }
  memory_mapper_t raw_mapping(raw_topology_path.string());
  const auto raw_file = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(raw_mapping.get_buf()), raw_mapping.get_file_size());
  const auto raw = parse_raw_topology(raw_file);
  if (raw.maximum_degree > UINT8_MAX) {
    throw std::runtime_error("GroupVarint statistics format supports maximum degree 255");
  }

  std::vector<uint64_t> offsets(static_cast<size_t>(raw.point_count + 1U), 0);
  std::vector<uint8_t> payload;
  payload.reserve(static_cast<size_t>(raw.adjacency_bytes));
  std::vector<std::pair<uint32_t, uint8_t>> ordered;
  ordered.reserve(raw.maximum_degree);
  uint64_t permutation_bytes = 0;
  for (uint64_t node = 0; node < raw.point_count; ++node) {
    offsets[static_cast<size_t>(node)] = payload.size();
    const uint64_t begin = raw_offset(raw, node);
    const uint64_t end = raw_offset(raw, node + 1U);
    const size_t degree = static_cast<size_t>(end - begin);
    ordered.clear();
    for (size_t position = 0; position < degree; ++position) {
      const uint32_t target = raw_neighbor(raw, begin + position);
      if (target >= raw.point_count) {
        throw std::runtime_error("raw IO topology adjacency target " + std::to_string(target) +
                                 " at node " + std::to_string(node) + " edge " +
                                 std::to_string(begin + position) + " is out of range for " +
                                 std::to_string(raw.point_count) + " points");
      }
      ordered.emplace_back(target, static_cast<uint8_t>(position));
    }
    std::sort(ordered.begin(), ordered.end());
    payload.push_back(static_cast<uint8_t>(degree));
    if (preserve_neighbor_order) {
      const uint8_t bits = degree <= 1 ? 0 : static_cast<uint8_t>(std::bit_width(degree - 1U));
      const uint64_t bytes = (degree * bits + 7U) / 8U;
      append_packed_permutation(payload, ordered, bits);
      permutation_bytes += bytes;
    }
    append_groupvarint_deltas(payload, ordered);
  }
  offsets.back() = payload.size();
  const auto offset_bytes = serialize_offsets(offsets);

  statistics_header_t header;
  if (preserve_neighbor_order) {
    header.flags |= k_preserves_neighbor_order_flag;
  }
  header.point_count = raw.point_count;
  header.edge_count = raw.edge_count;
  header.maximum_degree = raw.maximum_degree;
  header.source_topology_bytes = raw_file.size();
  header.source_topology_checksum = checksum_bytes(raw_file);
  header.raw_adjacency_bytes = raw.adjacency_bytes;
  header.source_adjacency_checksum = raw.adjacency_checksum;
  header.offsets_bytes = offset_bytes.size();
  header.offsets_checksum = checksum_bytes(offset_bytes);
  header.payload_offset =
      checked_add(header.offsets_offset, header.offsets_bytes, "GroupVarint payload offset");
  header.payload_bytes = payload.size();
  header.payload_checksum = checksum_bytes(payload);
  header.permutation_bytes = permutation_bytes;
  const auto header_bytes = serialize_statistics_header(header);

  const auto temporary = std::filesystem::path(statistics_path.string() + ".tmp");
  std::error_code ignored;
  std::filesystem::remove(temporary, ignored);
  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());
    output.write(reinterpret_cast<const char*>(offset_bytes.data()), offset_bytes.size());
    output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    output.close();
    if (!output ||
        std::filesystem::file_size(temporary) != header.payload_offset + header.payload_bytes) {
      throw std::runtime_error("failed writing GroupVarint statistics artifact");
    }
    const auto verified = verify_io_groupvarint_adjacency_statistics(raw_topology_path, temporary);
    std::filesystem::rename(temporary, statistics_path);
    return verified;
  } catch (...) {
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

io_adjacency_statistics_result_t
verify_io_groupvarint_adjacency_statistics(const std::filesystem::path& raw_topology_path,
                                           const std::filesystem::path& statistics_path) {
  memory_mapper_t raw_mapping(raw_topology_path.string());
  memory_mapper_t statistics_mapping(statistics_path.string());
  const auto raw_file = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(raw_mapping.get_buf()), raw_mapping.get_file_size());
  const auto statistics_file =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(statistics_mapping.get_buf()),
                               statistics_mapping.get_file_size());
  const auto raw = parse_raw_topology(raw_file);
  const auto header = parse_statistics_header(statistics_file);
  const uint32_t required_flags = k_statistics_only_flag | k_sorted_adjacency_flag;
  const uint64_t expected_offsets =
      checked_multiply(checked_add(header.point_count, 1, "GroupVarint offset count"),
                       sizeof(uint64_t), "GroupVarint offset bytes");
  if ((header.flags & required_flags) != required_flags ||
      (header.flags & ~(required_flags | k_preserves_neighbor_order_flag)) != 0 ||
      header.point_count != raw.point_count || header.edge_count != raw.edge_count ||
      header.maximum_degree != raw.maximum_degree ||
      header.source_topology_bytes != raw_file.size() ||
      header.source_topology_checksum != checksum_bytes(raw_file) ||
      header.raw_adjacency_bytes != raw.adjacency_bytes ||
      header.source_adjacency_checksum != raw.adjacency_checksum ||
      header.offsets_offset != k_statistics_header_bytes ||
      header.offsets_bytes != expected_offsets ||
      header.payload_offset != header.offsets_offset + header.offsets_bytes ||
      header.payload_offset > statistics_file.size() ||
      header.payload_bytes > statistics_file.size() - header.payload_offset ||
      header.payload_offset + header.payload_bytes != statistics_file.size()) {
    throw std::runtime_error("GroupVarint statistics metadata does not match raw topology");
  }
  const auto offset_section = statistics_file.subspan(static_cast<size_t>(header.offsets_offset),
                                                      static_cast<size_t>(header.offsets_bytes));
  const auto payload = statistics_file.subspan(static_cast<size_t>(header.payload_offset),
                                               static_cast<size_t>(header.payload_bytes));
  if (checksum_bytes(offset_section) != header.offsets_checksum ||
      checksum_bytes(payload) != header.payload_checksum) {
    throw std::runtime_error("GroupVarint statistics section checksum mismatch");
  }

  const bool preserves_order = (header.flags & k_preserves_neighbor_order_flag) != 0;
  uint64_t counted_permutation_bytes = 0;
  std::vector<uint32_t> sorted;
  std::vector<uint32_t> expected;
  std::vector<uint32_t> restored;
  std::vector<uint8_t> seen;
  sorted.reserve(header.maximum_degree);
  expected.reserve(header.maximum_degree);
  restored.reserve(header.maximum_degree);
  seen.reserve(header.maximum_degree);
  uint64_t previous_record_end = 0;
  for (uint64_t node = 0; node < header.point_count; ++node) {
    size_t offset_cursor = static_cast<size_t>(node * sizeof(uint64_t));
    const uint64_t begin =
        read_little<uint64_t>(offset_section, offset_cursor, "GroupVarint record offset");
    const uint64_t end =
        read_little<uint64_t>(offset_section, offset_cursor, "GroupVarint record end");
    if (begin != previous_record_end || end < begin || end > payload.size()) {
      throw std::runtime_error("GroupVarint record offsets are invalid");
    }
    previous_record_end = end;
    const auto record =
        payload.subspan(static_cast<size_t>(begin), static_cast<size_t>(end - begin));
    size_t cursor = 0;
    const uint8_t degree = read_little<uint8_t>(record, cursor, "GroupVarint degree");
    const uint64_t raw_begin = raw_offset(raw, node);
    const uint64_t raw_end = raw_offset(raw, node + 1U);
    if (degree != raw_end - raw_begin) {
      throw std::runtime_error("GroupVarint degree does not match raw topology");
    }
    const uint8_t permutation_bits =
        degree <= 1 ? 0 : static_cast<uint8_t>(std::bit_width(static_cast<size_t>(degree) - 1U));
    const size_t permutation_size =
        preserves_order ? (static_cast<size_t>(degree) * permutation_bits + 7U) / 8U : 0;
    if (permutation_size > record.size() - cursor) {
      throw std::runtime_error("truncated GroupVarint permutation");
    }
    const auto permutation = record.subspan(cursor, permutation_size);
    cursor += permutation_size;
    counted_permutation_bytes += permutation_size;

    sorted.clear();
    uint32_t value = 0;
    for (size_t group = 0; group < degree; group += 4U) {
      const uint8_t tag = read_little<uint8_t>(record, cursor, "GroupVarint tag");
      const size_t count = std::min<size_t>(4, degree - group);
      for (size_t index = 0; index < count; ++index) {
        const uint8_t width = static_cast<uint8_t>(((tag >> (index * 2U)) & 3U) + 1U);
        uint32_t delta = 0;
        for (uint8_t byte = 0; byte < width; ++byte) {
          delta |= static_cast<uint32_t>(read_little<uint8_t>(record, cursor, "GroupVarint value"))
                   << (byte * 8U);
        }
        if (group + index != 0 && delta > UINT32_MAX - value) {
          throw std::runtime_error("GroupVarint delta reconstruction overflows");
        }
        value = group + index == 0 ? delta : value + delta;
        sorted.push_back(value);
      }
    }
    if (cursor != record.size()) {
      throw std::runtime_error("GroupVarint record has trailing bytes");
    }

    expected.clear();
    for (uint64_t edge = raw_begin; edge < raw_end; ++edge) {
      expected.push_back(raw_neighbor(raw, edge));
    }
    if (preserves_order) {
      restored.assign(degree, 0);
      seen.assign(degree, 0);
      for (size_t index = 0; index < degree; ++index) {
        const uint8_t position = read_packed_permutation(permutation, index, permutation_bits);
        if (position >= degree || seen[position] != 0) {
          throw std::runtime_error("GroupVarint permutation is invalid");
        }
        seen[position] = 1;
        restored[position] = sorted[index];
      }
      if (restored != expected) {
        throw std::runtime_error("GroupVarint order-preserving replay mismatch");
      }
    } else {
      std::sort(expected.begin(), expected.end());
      if (sorted != expected) {
        throw std::runtime_error("GroupVarint sorted-adjacency replay mismatch");
      }
    }
  }
  size_t last_cursor = static_cast<size_t>(header.point_count * sizeof(uint64_t));
  if (read_little<uint64_t>(offset_section, last_cursor, "GroupVarint final offset") !=
          payload.size() ||
      previous_record_end != payload.size() ||
      counted_permutation_bytes != header.permutation_bytes) {
    throw std::runtime_error("GroupVarint statistics coverage is incomplete");
  }
  return make_result(header, statistics_file.size(), checksum_bytes(statistics_file));
}

} // namespace powerlaw_ann
