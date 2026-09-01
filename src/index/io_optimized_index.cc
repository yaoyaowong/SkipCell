#include "index/io_optimized_index.h"

#include "common/defaults.h"
#include "index/community_polar_index.h"
#include "storage/memory_mapper.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(POWERLAWANN_USE_PFORDELTA)
#include <codecfactory.h>
#endif
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <omp.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace powerlaw_ann {
namespace {

inline constexpr uint64_t k_topology_magic = 0x31504F544F494C50ULL;
inline constexpr uint64_t k_cell_adjacency_magic = 0x314A4441434C50ULL; // "PLCADJ1"
inline constexpr uint32_t k_cell_adjacency_version = 1;
inline constexpr uint32_t k_cell_adjacency_header_bytes = 128;
inline constexpr uint32_t k_topology_version_v4 = 4;
inline constexpr uint32_t k_topology_version_v5 = 5;
inline constexpr uint32_t k_topology_version = 6;
inline constexpr uint32_t k_little_endian_marker = 0x01020304U;
inline constexpr uint32_t k_topology_v4_header_bytes = 256;
inline constexpr uint32_t k_topology_header_bytes = 320;
inline constexpr uint64_t k_fnv_offset_basis = 14695981039346656037ULL;
inline constexpr uint64_t k_fnv_prime = 1099511628211ULL;

struct fingerprint_t {
  uint64_t size = 0;
  uint64_t checksum = 0;
};

struct disk_layout_metadata_t {
  uint64_t point_count = 0;
  uint64_t dimension = 0;
  uint64_t max_node_len = 0;
  uint64_t nodes_per_sector = 0;
  uint32_t max_degree = 0;
};

std::vector<double> read_normalized_rcni(const std::filesystem::path& path, uint64_t point_count) {
  std::ifstream input(path);
  std::string line;
  if (!input || !std::getline(input, line) ||
      line != "node_id,raw_importance,normalized_importance,importance_percentile,source_support,"
              "witness_event_count") {
    throw std::runtime_error("low-RCNI topology trim input has an invalid header");
  }
  std::vector<double> result(point_count, std::numeric_limits<double>::quiet_NaN());
  std::vector<uint8_t> seen(point_count, 0);
  uint64_t records = 0;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    std::array<std::string, 6> fields;
    for (auto& field : fields) {
      if (!std::getline(row, field, ',')) {
        throw std::runtime_error("low-RCNI topology trim row is truncated");
      }
    }
    if (row.peek() != std::char_traits<char>::eof()) {
      throw std::runtime_error("low-RCNI topology trim row has trailing fields");
    }
    uint64_t node = 0;
    double normalized = 0.0;
    try {
      node = std::stoull(fields[0]);
      normalized = std::stod(fields[2]);
    } catch (const std::exception& error) {
      throw std::runtime_error("low-RCNI topology trim row is invalid: " +
                               std::string(error.what()));
    }
    if (node >= point_count || !std::isfinite(normalized) || seen[node] != 0) {
      throw std::runtime_error("low-RCNI topology trim node or score is invalid");
    }
    result[node] = normalized;
    seen[node] = 1;
    ++records;
  }
  if (records != point_count) {
    throw std::runtime_error("low-RCNI topology trim does not cover every Base node");
  }
  return result;
}

struct topology_header_t {
  uint32_t version = k_topology_version;
  uint32_t header_bytes = k_topology_header_bytes;
  uint64_t point_count = 0;
  uint64_t dimension = 0;
  uint32_t max_degree = 0;
  io_adjacency_encoding_t encoding = io_adjacency_encoding_t::RAW_U32;
  io_vector_layout_t layout = io_vector_layout_t::ORIGINAL_ID;
  uint64_t edge_count = 0;
  uint64_t vector_page_count = 0;
  uint64_t cell_count = 0;
  fingerprint_t base;
  fingerprint_t vectors;
  uint64_t offsets_offset = 0;
  uint64_t offsets_bytes = 0;
  uint64_t offsets_checksum = 0;
  uint64_t adjacency_offset = 0;
  uint64_t adjacency_bytes = 0;
  uint64_t adjacency_checksum = 0;
  uint64_t directory_offset = 0;
  uint64_t directory_bytes = 0;
  uint64_t directory_checksum = 0;
  uint64_t cell_directory_offset = 0;
  uint64_t cell_directory_bytes = 0;
  uint64_t cell_directory_checksum = 0;
  uint64_t prefetch_directory_offset = 0;
  uint64_t prefetch_directory_bytes = 0;
  uint64_t prefetch_directory_checksum = 0;
  fingerprint_t community_polar;
  uint64_t gateway_count = 0;
  uint64_t gateway_directory_offset = 0;
  uint64_t gateway_directory_bytes = 0;
  uint64_t gateway_directory_checksum = 0;
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

void hash_bytes(uint64_t& checksum, const uint8_t* data, size_t size) {
  for (size_t index = 0; index < size; ++index) {
    checksum ^= data[index];
    checksum *= k_fnv_prime;
  }
}

uint64_t checksum_bytes(std::span<const uint8_t> bytes) {
  uint64_t checksum = k_fnv_offset_basis;
  hash_bytes(checksum, bytes.data(), bytes.size());
  return checksum;
}

fingerprint_t fingerprint_file(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("required artifact is not a regular file: " + path.string());
  }
  fingerprint_t result;
  result.size = std::filesystem::file_size(path);
  result.checksum = k_fnv_offset_basis;
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open artifact for fingerprinting: " + path.string());
  }
  std::array<uint8_t, 1U << 20U> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const auto count = input.gcount();
    if (count > 0) {
      hash_bytes(result.checksum, buffer.data(), static_cast<size_t>(count));
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while fingerprinting artifact: " + path.string());
  }
  return result;
}

template <typename value_t>
void append_little(std::vector<uint8_t>& output, value_t value) {
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

std::vector<uint8_t> serialize_u64(std::span<const uint64_t> values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(uint64_t));
  for (const uint64_t value : values) {
    append_little(bytes, value);
  }
  return bytes;
}

std::vector<uint8_t> serialize_u32(std::span<const uint32_t> values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(uint32_t));
  for (const uint32_t value : values) {
    append_little(bytes, value);
  }
  return bytes;
}

#if defined(POWERLAWANN_USE_PFORDELTA)
std::pair<std::vector<uint64_t>, std::vector<uint8_t>>
compress_adjacency(std::span<const uint64_t> edge_offsets, std::span<const uint32_t> adjacency) {
  constexpr size_t k_simd_overread = 32;
  std::vector<uint64_t> byte_offsets(edge_offsets.size(), 0);
  std::vector<uint8_t> output;
  auto codec = SIMDCompressionLib::CODECFactory::getFromName("s4-bp128-d1");
  for (size_t node = 0; node + 1U < edge_offsets.size(); ++node) {
    byte_offsets[node] = output.size();
    const size_t degree = static_cast<size_t>(edge_offsets[node + 1U] - edge_offsets[node]);
    std::vector<std::pair<uint32_t, uint8_t>> ordered;
    ordered.reserve(degree);
    for (size_t position = 0; position < degree; ++position) {
      ordered.emplace_back(adjacency[edge_offsets[node] + position],
                           static_cast<uint8_t>(position));
    }
    std::sort(ordered.begin(), ordered.end());
    const size_t padded = degree == 0 ? 0 : ((degree + 127U) / 128U) * 128U;
    std::vector<uint32_t> input(padded + k_simd_overread, degree == 0 ? 0 : ordered.back().first);
    for (size_t position = 0; position < degree; ++position) {
      input[position] = ordered[position].first;
    }
    std::vector<uint32_t> compressed(padded * 2U + 1024U + k_simd_overread);
    size_t compressed_words = compressed.size();
    if (degree != 0) {
      codec->encodeArray(input.data(), padded, compressed.data(), compressed_words);
    } else {
      compressed_words = 0;
    }
    append_little(output, static_cast<uint32_t>(degree));
    append_little(output, static_cast<uint32_t>(compressed_words));
    for (const auto& entry : ordered) {
      output.push_back(entry.second);
    }
    while (output.size() % sizeof(uint32_t) != 0) {
      output.push_back(0);
    }
    for (size_t word = 0; word < compressed_words; ++word) {
      append_little(output, compressed[word]);
    }
  }
  byte_offsets.back() = output.size();
  return {std::move(byte_offsets), std::move(output)};
}

std::vector<uint32_t> decompress_adjacency(std::span<const uint64_t> byte_offsets,
                                           std::span<const uint8_t> bytes, uint64_t expected_edges,
                                           uint32_t max_degree,
                                           std::vector<uint64_t>& edge_offsets) {
  constexpr size_t k_simd_overread = 32;
  std::vector<uint32_t> output;
  output.reserve(expected_edges);
  edge_offsets.assign(byte_offsets.size(), 0);
  auto codec = SIMDCompressionLib::CODECFactory::getFromName("s4-bp128-d1");
  for (size_t node = 0; node + 1U < byte_offsets.size(); ++node) {
    edge_offsets[node] = output.size();
    if (byte_offsets[node] > byte_offsets[node + 1U] || byte_offsets[node + 1U] > bytes.size()) {
      throw std::runtime_error("PForDelta byte offsets are invalid");
    }
    const auto record =
        bytes.subspan(static_cast<size_t>(byte_offsets[node]),
                      static_cast<size_t>(byte_offsets[node + 1U] - byte_offsets[node]));
    size_t cursor = 0;
    const uint32_t degree = read_little<uint32_t>(record, cursor, "PForDelta degree");
    const uint32_t compressed_words = read_little<uint32_t>(record, cursor, "PForDelta word count");
    if (degree > max_degree || cursor + degree > record.size()) {
      throw std::runtime_error("PForDelta degree or permutation is invalid");
    }
    std::vector<uint8_t> permutation(degree);
    for (uint32_t position = 0; position < degree; ++position) {
      permutation[position] = record[cursor++];
      if (permutation[position] >= degree) {
        throw std::runtime_error("PForDelta permutation is out of range");
      }
    }
    cursor = (cursor + sizeof(uint32_t) - 1U) & ~(sizeof(uint32_t) - 1U);
    if (compressed_words > (record.size() - cursor) / sizeof(uint32_t) ||
        cursor + static_cast<size_t>(compressed_words) * sizeof(uint32_t) != record.size()) {
      throw std::runtime_error("PForDelta compressed payload size is invalid");
    }
    std::vector<uint32_t> compressed(compressed_words + k_simd_overread, 0);
    for (uint32_t word = 0; word < compressed_words; ++word) {
      compressed[word] = read_little<uint32_t>(record, cursor, "PForDelta word");
    }
    const size_t padded = degree == 0 ? 0 : ((degree + 127U) / 128U) * 128U;
    std::vector<uint32_t> sorted(padded + k_simd_overread, 0);
    if (degree != 0) {
      size_t recovered = padded;
      codec->decodeArray(compressed.data(), compressed_words, sorted.data(), recovered);
      if (recovered < degree) {
        throw std::runtime_error("PForDelta decode returned too few values");
      }
    }
    const size_t output_begin = output.size();
    output.resize(output_begin + degree);
    std::vector<uint8_t> seen(degree, 0);
    for (uint32_t position = 0; position < degree; ++position) {
      if (seen[permutation[position]] != 0) {
        throw std::runtime_error("PForDelta permutation contains duplicates");
      }
      seen[permutation[position]] = 1;
      output[output_begin + permutation[position]] = sorted[position];
    }
  }
  if (output.size() != expected_edges) {
    throw std::runtime_error("PForDelta edge count mismatch");
  }
  edge_offsets.back() = output.size();
  return output;
}
#endif

std::vector<uint8_t> serialize_locations(std::span<const io_vector_location_t> locations) {
  std::vector<uint8_t> bytes;
  bytes.reserve(locations.size() * (sizeof(uint64_t) + sizeof(uint32_t)));
  for (const auto& location : locations) {
    append_little(bytes, location.page_id);
    append_little(bytes, location.page_offset);
  }
  return bytes;
}

std::vector<uint8_t> serialize_cell_ranges(std::span<const io_cell_page_range_t> ranges) {
  std::vector<uint8_t> bytes;
  bytes.reserve(ranges.size() * (sizeof(uint64_t) + 3U * sizeof(uint32_t)));
  for (const auto& range : ranges) {
    append_little(bytes, range.first_page);
    append_little(bytes, range.page_count);
    append_little(bytes, range.node_count);
    append_little(bytes, range.chunk_size_bytes);
  }
  return bytes;
}

std::vector<uint8_t> serialize_gateway_hints(std::span<const io_gateway_prefetch_hint_t> hints) {
  std::vector<uint8_t> bytes;
  bytes.reserve(hints.size() * 64U);
  for (const auto& hint : hints) {
    append_little(bytes, hint.target_node);
    append_little(bytes, hint.landing_cell);
    append_little(bytes, hint.successor_cells[0]);
    append_little(bytes, hint.successor_cells[1]);
    append_little(bytes, hint.landing_node_page);
    append_little(bytes, hint.landing_cell_first_page);
    append_little(bytes, hint.successor_first_pages[0]);
    append_little(bytes, hint.successor_first_pages[1]);
    append_little(bytes, hint.transition_counts[0]);
    append_little(bytes, hint.transition_counts[1]);
    append_little(bytes, hint.outgoing_transition_count);
    append_little(bytes, hint.reserved);
  }
  return bytes;
}

std::vector<uint8_t> serialize_header(const topology_header_t& header) {
  std::vector<uint8_t> bytes;
  bytes.reserve(k_topology_header_bytes);
  append_little(bytes, k_topology_magic);
  append_little(bytes, header.version);
  append_little(bytes, k_little_endian_marker);
  append_little(bytes, k_topology_header_bytes);
  append_little(bytes, k_io_vector_page_size);
  append_little(bytes, header.point_count);
  append_little(bytes, header.dimension);
  append_little(bytes, header.max_degree);
  append_little(bytes, static_cast<uint32_t>(header.encoding));
  append_little(bytes, static_cast<uint32_t>(header.layout));
  append_little(bytes, header.edge_count);
  append_little(bytes, header.vector_page_count);
  append_little(bytes, header.cell_count);
  append_little(bytes, header.base.size);
  append_little(bytes, header.base.checksum);
  append_little(bytes, header.vectors.size);
  append_little(bytes, header.vectors.checksum);
  append_little(bytes, header.offsets_offset);
  append_little(bytes, header.offsets_bytes);
  append_little(bytes, header.offsets_checksum);
  append_little(bytes, header.adjacency_offset);
  append_little(bytes, header.adjacency_bytes);
  append_little(bytes, header.adjacency_checksum);
  append_little(bytes, header.directory_offset);
  append_little(bytes, header.directory_bytes);
  append_little(bytes, header.directory_checksum);
  append_little(bytes, header.cell_directory_offset);
  append_little(bytes, header.cell_directory_bytes);
  append_little(bytes, header.cell_directory_checksum);
  append_little(bytes, header.prefetch_directory_offset);
  append_little(bytes, header.prefetch_directory_bytes);
  append_little(bytes, header.prefetch_directory_checksum);
  append_little(bytes, header.community_polar.size);
  append_little(bytes, header.community_polar.checksum);
  append_little(bytes, header.gateway_count);
  append_little(bytes, header.gateway_directory_offset);
  append_little(bytes, header.gateway_directory_bytes);
  append_little(bytes, header.gateway_directory_checksum);
  bytes.resize(k_topology_header_bytes, 0);
  return bytes;
}

topology_header_t parse_header(std::span<const uint8_t> bytes) {
  if (bytes.size() < k_topology_v4_header_bytes) {
    throw std::runtime_error("truncated IO topology header");
  }
  size_t cursor = 0;
  if (read_little<uint64_t>(bytes, cursor, "topology magic") != k_topology_magic) {
    throw std::runtime_error("invalid IO topology magic");
  }
  const uint32_t version = read_little<uint32_t>(bytes, cursor, "topology version");
  if (version != k_topology_version_v4 && version != k_topology_version_v5 &&
      version != k_topology_version) {
    throw std::runtime_error("unsupported IO topology version");
  }
  if (read_little<uint32_t>(bytes, cursor, "byte-order marker") != k_little_endian_marker) {
    throw std::runtime_error("invalid IO topology byte-order marker");
  }
  const uint32_t header_bytes = read_little<uint32_t>(bytes, cursor, "header size");
  const uint32_t expected_header_bytes =
      version == k_topology_version_v4 ? k_topology_v4_header_bytes : k_topology_header_bytes;
  if (header_bytes != expected_header_bytes || bytes.size() < header_bytes ||
      read_little<uint32_t>(bytes, cursor, "vector page size") != k_io_vector_page_size) {
    throw std::runtime_error("invalid IO topology header or page size");
  }
  topology_header_t header;
  header.version = version;
  header.header_bytes = header_bytes;
  header.point_count = read_little<uint64_t>(bytes, cursor, "point count");
  header.dimension = read_little<uint64_t>(bytes, cursor, "dimension");
  header.max_degree = read_little<uint32_t>(bytes, cursor, "maximum degree");
  header.encoding = static_cast<io_adjacency_encoding_t>(
      read_little<uint32_t>(bytes, cursor, "adjacency encoding"));
  header.layout =
      static_cast<io_vector_layout_t>(read_little<uint32_t>(bytes, cursor, "vector layout"));
  header.edge_count = read_little<uint64_t>(bytes, cursor, "edge count");
  header.vector_page_count = read_little<uint64_t>(bytes, cursor, "vector page count");
  header.cell_count = read_little<uint64_t>(bytes, cursor, "Cell count");
  header.base.size = read_little<uint64_t>(bytes, cursor, "Base size");
  header.base.checksum = read_little<uint64_t>(bytes, cursor, "Base checksum");
  header.vectors.size = read_little<uint64_t>(bytes, cursor, "vector size");
  header.vectors.checksum = read_little<uint64_t>(bytes, cursor, "vector checksum");
  header.offsets_offset = read_little<uint64_t>(bytes, cursor, "offset section offset");
  header.offsets_bytes = read_little<uint64_t>(bytes, cursor, "offset section size");
  header.offsets_checksum = read_little<uint64_t>(bytes, cursor, "offset checksum");
  header.adjacency_offset = read_little<uint64_t>(bytes, cursor, "adjacency section offset");
  header.adjacency_bytes = read_little<uint64_t>(bytes, cursor, "adjacency section size");
  header.adjacency_checksum = read_little<uint64_t>(bytes, cursor, "adjacency checksum");
  header.directory_offset = read_little<uint64_t>(bytes, cursor, "directory section offset");
  header.directory_bytes = read_little<uint64_t>(bytes, cursor, "directory section size");
  header.directory_checksum = read_little<uint64_t>(bytes, cursor, "directory checksum");
  header.cell_directory_offset = read_little<uint64_t>(bytes, cursor, "Cell directory offset");
  header.cell_directory_bytes = read_little<uint64_t>(bytes, cursor, "Cell directory size");
  header.cell_directory_checksum = read_little<uint64_t>(bytes, cursor, "Cell directory checksum");
  header.prefetch_directory_offset =
      read_little<uint64_t>(bytes, cursor, "prefetch directory offset");
  header.prefetch_directory_bytes = read_little<uint64_t>(bytes, cursor, "prefetch directory size");
  header.prefetch_directory_checksum =
      read_little<uint64_t>(bytes, cursor, "prefetch directory checksum");
  if (version >= k_topology_version_v5) {
    header.community_polar.size = read_little<uint64_t>(bytes, cursor, "Community-Polar size");
    header.community_polar.checksum =
        read_little<uint64_t>(bytes, cursor, "Community-Polar checksum");
    header.gateway_count = read_little<uint64_t>(bytes, cursor, "Gateway hint count");
    header.gateway_directory_offset =
        read_little<uint64_t>(bytes, cursor, "Gateway directory offset");
    header.gateway_directory_bytes = read_little<uint64_t>(bytes, cursor, "Gateway directory size");
    header.gateway_directory_checksum =
        read_little<uint64_t>(bytes, cursor, "Gateway directory checksum");
  }
  if (std::any_of(bytes.begin() + static_cast<ptrdiff_t>(cursor),
                  bytes.begin() + static_cast<ptrdiff_t>(header_bytes),
                  [](uint8_t value) { return value != 0; })) {
    throw std::runtime_error("IO topology header reserved bytes are nonzero");
  }
  return header;
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("artifact is not a regular file: " + path.string());
  }
  const uint64_t file_size = std::filesystem::file_size(path);
  if (file_size > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("artifact is too large to load: " + path.string());
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("failed to read complete artifact: " + path.string());
  }
  return bytes;
}

std::vector<uint8_t> read_file_range(const std::filesystem::path& path, uint64_t offset,
                                     uint64_t size) {
  if (size > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("artifact section is too large to load: " + path.string());
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  std::ifstream input(path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(offset));
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    throw std::runtime_error("failed to read artifact section: " + path.string());
  }
  return bytes;
}

uint64_t validate_raw_adjacency_range(const std::filesystem::path& path, uint64_t offset,
                                      uint64_t edge_count, uint64_t point_count) {
  constexpr size_t k_chunk_bytes = 1U << 20U;
  std::array<uint8_t, k_chunk_bytes> buffer{};
  std::ifstream input(path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(offset));
  uint64_t checksum = k_fnv_offset_basis;
  uint64_t remaining = checked_multiply(edge_count, sizeof(uint32_t), "adjacency validation size");
  while (remaining != 0) {
    const size_t count = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
    input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(count));
    if (!input) {
      throw std::runtime_error("paged topology adjacency is truncated");
    }
    hash_bytes(checksum, buffer.data(), count);
    for (size_t cursor = 0; cursor < count; cursor += sizeof(uint32_t)) {
      const uint32_t target = static_cast<uint32_t>(buffer[cursor]) |
                              (static_cast<uint32_t>(buffer[cursor + 1U]) << 8U) |
                              (static_cast<uint32_t>(buffer[cursor + 2U]) << 16U) |
                              (static_cast<uint32_t>(buffer[cursor + 3U]) << 24U);
      if (target >= point_count) {
        throw std::runtime_error("paged topology adjacency target is out of range");
      }
    }
    remaining -= count;
  }
  return checksum;
}

disk_layout_metadata_t read_disk_layout_metadata(const std::filesystem::path& disk_path) {
  std::ifstream input(disk_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open Base disk index: " + disk_path.string());
  }
  auto read_native = [&input](auto& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
      throw std::runtime_error("truncated Base disk metadata");
    }
  };
  uint32_t rows = 0;
  uint32_t columns = 0;
  uint64_t medoid = 0;
  uint64_t frozen_points = 0;
  uint64_t frozen_location = 0;
  uint64_t reorder_exists = 0;
  uint64_t declared_file_size = 0;
  disk_layout_metadata_t metadata;
  read_native(rows);
  read_native(columns);
  read_native(metadata.point_count);
  read_native(metadata.dimension);
  read_native(medoid);
  read_native(metadata.max_node_len);
  read_native(metadata.nodes_per_sector);
  read_native(frozen_points);
  read_native(frozen_location);
  read_native(reorder_exists);
  if (rows < 9 || columns != 1 || metadata.point_count == 0 || metadata.dimension == 0 ||
      medoid >= metadata.point_count || reorder_exists != 0) {
    throw std::runtime_error("unsupported Base disk metadata for IO-1 layout");
  }
  if (rows == 9) {
    read_native(declared_file_size);
    if (declared_file_size != std::filesystem::file_size(disk_path)) {
      throw std::runtime_error("Base disk index size disagrees with metadata");
    }
  }
  const uint64_t vector_bytes = checked_multiply(metadata.dimension, sizeof(float), "vector size");
  if (metadata.max_node_len <= vector_bytes + sizeof(uint32_t) ||
      (metadata.max_node_len - vector_bytes) % sizeof(uint32_t) != 0) {
    throw std::runtime_error("invalid Base disk record width");
  }
  const uint64_t degree = (metadata.max_node_len - vector_bytes) / sizeof(uint32_t) - 1U;
  if (degree > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Base maximum degree exceeds uint32");
  }
  metadata.max_degree = static_cast<uint32_t>(degree);
  return metadata;
}

void publish_file(const std::filesystem::path& temporary,
                  const std::filesystem::path& destination) {
  std::error_code error;
  std::filesystem::remove(destination, error);
  error.clear();
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    throw std::runtime_error("failed to publish artifact " + destination.string() + ": " +
                             error.message());
  }
}

void validate_output_path(const std::filesystem::path& path) {
  if (path.empty() || path.filename().empty()) {
    throw std::runtime_error("IO artifact path must include a file name");
  }
  const auto parent = path.parent_path();
  if (!parent.empty() && !std::filesystem::is_directory(parent)) {
    throw std::runtime_error("IO artifact output directory does not exist: " + parent.string());
  }
}

struct cell_sidecar_descriptor_t {
  uint64_t point_count = 0;
  uint32_t dimension = 0;
  uint32_t pq_width = 0;
  uint64_t body_checksum = 0;
  uint64_t base_size = 0;
  uint64_t base_checksum = 0;
  uint32_t header_bytes = 0;
  std::vector<std::tuple<uint32_t, uint64_t, uint64_t, uint64_t>> sections;
};

cell_sidecar_descriptor_t read_cell_sidecar_descriptor(std::ifstream& input,
                                                       const std::filesystem::path& path) {
  constexpr uint64_t k_cell_magic = 0x0058444950434C50ULL;
  constexpr uint32_t k_fixed_bytes = 96;
  constexpr uint32_t k_entry_bytes = 32;
  constexpr uint32_t k_sections = 18;
  std::array<uint8_t, k_fixed_bytes> fixed{};
  input.read(reinterpret_cast<char*>(fixed.data()), fixed.size());
  if (!input) {
    throw std::runtime_error("truncated Cell sidecar header: " + path.string());
  }
  size_t cursor = 0;
  if (read_little<uint64_t>(fixed, cursor, "Cell magic") != k_cell_magic) {
    throw std::runtime_error("invalid Cell sidecar magic");
  }
  const uint32_t version = read_little<uint32_t>(fixed, cursor, "Cell version");
  cell_sidecar_descriptor_t descriptor;
  descriptor.header_bytes = read_little<uint32_t>(fixed, cursor, "Cell header size");
  if (read_little<uint32_t>(fixed, cursor, "Cell byte order") != k_little_endian_marker ||
      read_little<uint32_t>(fixed, cursor, "Cell section count") != k_sections || version < 5 ||
      version > 7 || descriptor.header_bytes != k_fixed_bytes + k_sections * k_entry_bytes) {
    throw std::runtime_error("unsupported Cell sidecar header");
  }
  descriptor.point_count = read_little<uint64_t>(fixed, cursor, "Cell point count");
  descriptor.dimension = read_little<uint32_t>(fixed, cursor, "Cell dimension");
  descriptor.pq_width = read_little<uint32_t>(fixed, cursor, "Cell PQ width");
  descriptor.body_checksum = read_little<uint64_t>(fixed, cursor, "Cell body checksum");
  descriptor.base_size = read_little<uint64_t>(fixed, cursor, "Cell Base size");
  descriptor.base_checksum = read_little<uint64_t>(fixed, cursor, "Cell Base checksum");
  static_cast<void>(read_little<uint64_t>(fixed, cursor, "Cell PQ size"));
  static_cast<void>(read_little<uint64_t>(fixed, cursor, "Cell PQ checksum"));
  const uint64_t body_bytes = read_little<uint64_t>(fixed, cursor, "Cell body size");
  std::vector<uint8_t> directory(descriptor.header_bytes - k_fixed_bytes);
  input.read(reinterpret_cast<char*>(directory.data()), directory.size());
  if (!input) {
    throw std::runtime_error("truncated Cell sidecar directory");
  }
  cursor = 0;
  uint64_t next_offset = descriptor.header_bytes;
  for (uint32_t section = 1; section <= k_sections; ++section) {
    const uint32_t id = read_little<uint32_t>(directory, cursor, "Cell section ID");
    static_cast<void>(read_little<uint32_t>(directory, cursor, "Cell section reserved"));
    const uint64_t offset = read_little<uint64_t>(directory, cursor, "Cell section offset");
    const uint64_t length = read_little<uint64_t>(directory, cursor, "Cell section length");
    const uint64_t count = read_little<uint64_t>(directory, cursor, "Cell section count");
    if (id != section || offset != next_offset) {
      throw std::runtime_error("noncanonical Cell sidecar directory");
    }
    descriptor.sections.emplace_back(id, offset, length, count);
    next_offset = checked_add(offset, length, "Cell sidecar bytes");
  }
  if (next_offset != descriptor.header_bytes + body_bytes ||
      next_offset != std::filesystem::file_size(path)) {
    throw std::runtime_error("Cell sidecar section coverage is invalid");
  }
  return descriptor;
}

void validate_rebound_cell_structure(const std::filesystem::path& source_path,
                                     const std::filesystem::path& replacement_path) {
  std::ifstream source(source_path, std::ios::binary);
  std::ifstream replacement(replacement_path, std::ios::binary);
  if (!source || !replacement) {
    throw std::runtime_error("failed opening Cell sidecars for structural comparison");
  }
  const auto left = read_cell_sidecar_descriptor(source, source_path);
  const auto right = read_cell_sidecar_descriptor(replacement, replacement_path);
  if (left.point_count != right.point_count || left.dimension != right.dimension ||
      left.base_size != right.base_size || left.base_checksum != right.base_checksum ||
      left.sections.size() != right.sections.size()) {
    throw std::runtime_error("replacement Cell sidecar changes Base structure");
  }
  uint64_t left_hash = k_fnv_offset_basis;
  uint64_t right_hash = k_fnv_offset_basis;
  const auto read_record = [](std::ifstream& input, std::span<uint8_t> bytes, uint64_t& checksum) {
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
      throw std::runtime_error("truncated Cell sidecar section during structural comparison");
    }
    hash_bytes(checksum, bytes.data(), bytes.size());
  };
  std::array<uint8_t, 1U << 20U> left_buffer{};
  std::array<uint8_t, 1U << 20U> right_buffer{};
  for (size_t index = 0; index < left.sections.size(); ++index) {
    const auto [id, left_offset, left_length, left_count] = left.sections[index];
    const auto [right_id, right_offset, right_length, right_count] = right.sections[index];
    if (id != right_id || left_count != right_count) {
      throw std::runtime_error("replacement Cell sidecar changes section record counts");
    }
    source.seekg(static_cast<std::streamoff>(left_offset));
    replacement.seekg(static_cast<std::streamoff>(right_offset));
    if (id == 5) {
      constexpr size_t k_record_bytes = 84;
      if (left_length != left_count * k_record_bytes || right_length != left_length) {
        throw std::runtime_error("Cell Community section width changed");
      }
      std::array<uint8_t, k_record_bytes> left_record{};
      std::array<uint8_t, k_record_bytes> right_record{};
      for (uint64_t record = 0; record < left_count; ++record) {
        read_record(source, left_record, left_hash);
        read_record(replacement, right_record, right_hash);
        std::fill_n(left_record.begin() + 12, 8, 0);
        std::fill_n(right_record.begin() + 12, 8, 0);
        if (left_record != right_record) {
          throw std::runtime_error("replacement Cell sidecar changes Community structure");
        }
      }
      continue;
    }
    if (id == 13) {
      constexpr size_t k_record_bytes = 36;
      if (left_length != left_count * k_record_bytes || right_length != left_length) {
        throw std::runtime_error("Cell Block section width changed");
      }
      std::array<uint8_t, k_record_bytes> left_record{};
      std::array<uint8_t, k_record_bytes> right_record{};
      for (uint64_t record = 0; record < left_count; ++record) {
        read_record(source, left_record, left_hash);
        read_record(replacement, right_record, right_hash);
        if (!std::equal(left_record.begin(), left_record.begin() + 32, right_record.begin())) {
          throw std::runtime_error("replacement Cell sidecar changes Block ownership");
        }
      }
      continue;
    }
    if (id == 14) {
      const uint64_t left_width = sizeof(uint32_t) + left.pq_width;
      const uint64_t right_width = sizeof(uint32_t) + right.pq_width;
      if (left_length != left_count * left_width || right_length != right_count * right_width) {
        throw std::runtime_error("Cell packet section width is invalid");
      }
      std::vector<uint8_t> left_record(left_width);
      std::vector<uint8_t> right_record(right_width);
      for (uint64_t record = 0; record < left_count; ++record) {
        read_record(source, left_record, left_hash);
        read_record(replacement, right_record, right_hash);
        if (!std::equal(left_record.begin(), left_record.begin() + sizeof(uint32_t),
                        right_record.begin())) {
          throw std::runtime_error("replacement Cell sidecar changes packet ownership");
        }
      }
      continue;
    }
    if (left_length != right_length) {
      throw std::runtime_error("replacement Cell sidecar changes immutable section length");
    }
    uint64_t remaining = left_length;
    while (remaining != 0) {
      const size_t count = static_cast<size_t>(std::min<uint64_t>(remaining, left_buffer.size()));
      read_record(source, std::span<uint8_t>(left_buffer).first(count), left_hash);
      read_record(replacement, std::span<uint8_t>(right_buffer).first(count), right_hash);
      if (!std::equal(left_buffer.begin(), left_buffer.begin() + count, right_buffer.begin())) {
        throw std::runtime_error("replacement Cell sidecar changes immutable structure");
      }
      remaining -= count;
    }
  }
  if (left_hash != left.body_checksum || right_hash != right.body_checksum) {
    throw std::runtime_error("Cell sidecar body checksum failed during structural comparison");
  }
}

} // namespace

std::vector<io_page_run_t> make_io_page_runs(std::span<const uint64_t> page_ids) {
  std::vector<uint64_t> ordered(page_ids.begin(), page_ids.end());
  std::sort(ordered.begin(), ordered.end());
  ordered.erase(std::unique(ordered.begin(), ordered.end()), ordered.end());
  std::vector<io_page_run_t> runs;
  for (const uint64_t page : ordered) {
    const bool can_extend =
        !runs.empty() && runs.back().page_count < std::numeric_limits<uint32_t>::max() &&
        runs.back().first_page <= std::numeric_limits<uint64_t>::max() - runs.back().page_count &&
        page == runs.back().first_page + runs.back().page_count;
    if (!can_extend) {
      runs.push_back({page, 1});
      continue;
    }
    ++runs.back().page_count;
  }
  return runs;
}

io_global_graph_cell_assignment_t
build_io_global_graph_cell_assignment(const io_optimized_index_t& index, uint32_t target_size) {
  if (target_size == 0 || index.point_count() == 0 ||
      index.point_count() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(
        "global graph Cell assignment requires representable points and a positive target");
  }
  io_global_graph_cell_assignment_t result;
  result.node_to_cell.assign(static_cast<size_t>(index.point_count()), UINT32_MAX);
  std::vector<uint8_t> state(static_cast<size_t>(index.point_count()), 0);
  std::vector<uint32_t> frontier;
  const uint64_t reserve = std::min<uint64_t>(
      index.point_count(), static_cast<uint64_t>(target_size) * index.max_degree());
  frontier.reserve(static_cast<size_t>(reserve));
  uint32_t population = 0;

  for (uint32_t seed = 0; seed < index.point_count(); ++seed) {
    if (state[seed] != 0) {
      continue;
    }
    if (result.cell_count == UINT32_MAX) {
      throw std::overflow_error("global graph Cell count exceeds uint32");
    }
    const uint32_t cell = result.cell_count++;
    frontier.clear();
    population = 0;
    const auto add_node = [&](uint32_t node) {
      state[node] = 1;
      result.node_to_cell[node] = cell;
      ++population;
      for (const uint32_t neighbor : index.neighbors(node)) {
        if (state[neighbor] == 0) {
          state[neighbor] = 2;
          frontier.push_back(neighbor);
        }
      }
    };
    add_node(seed);
    size_t cursor = 0;
    while (population < target_size && cursor < frontier.size()) {
      add_node(frontier[cursor++]);
    }
    for (; cursor < frontier.size(); ++cursor) {
      if (state[frontier[cursor]] == 2) {
        state[frontier[cursor]] = 0;
      }
    }
  }
  if (std::find(result.node_to_cell.begin(), result.node_to_cell.end(), UINT32_MAX) !=
      result.node_to_cell.end()) {
    throw std::logic_error("global graph Cell assignment does not cover every node");
  }
  return result;
}

io_global_graph_cell_assignment_t
build_io_global_graph_cell_assignment_from_artifact(const std::filesystem::path& topology_path,
                                                    uint32_t target_size) {
  if (topology_path.empty() || target_size == 0) {
    throw std::invalid_argument(
        "mapped global graph Cell assignment requires an artifact and target");
  }
  memory_mapper_t mapped(topology_path.string());
  const auto file = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(mapped.get_buf()),
                                             mapped.get_file_size());
  const auto header =
      parse_header(file.first(std::min(file.size(), static_cast<size_t>(k_topology_header_bytes))));
  if (header.encoding != io_adjacency_encoding_t::RAW_U32 || header.point_count == 0 ||
      header.point_count > std::numeric_limits<uint32_t>::max() ||
      header.offsets_bytes != (header.point_count + 1) * sizeof(uint64_t) ||
      header.adjacency_bytes != header.edge_count * sizeof(uint32_t) ||
      header.offsets_offset > file.size() ||
      header.offsets_bytes > file.size() - header.offsets_offset ||
      header.adjacency_offset > file.size() ||
      header.adjacency_bytes > file.size() - header.adjacency_offset ||
      header.offsets_offset % alignof(uint64_t) != 0 ||
      header.adjacency_offset % alignof(uint32_t) != 0) {
    throw std::invalid_argument(
        "mapped global graph Cell assignment requires compatible raw topology");
  }
  const auto offsets = std::span<const uint64_t>(
      reinterpret_cast<const uint64_t*>(file.data() + header.offsets_offset),
      static_cast<size_t>(header.point_count + 1));
  const auto adjacency = std::span<const uint32_t>(
      reinterpret_cast<const uint32_t*>(file.data() + header.adjacency_offset),
      static_cast<size_t>(header.edge_count));
  if (offsets.front() != 0 || offsets.back() != adjacency.size()) {
    throw std::invalid_argument("mapped raw topology offsets do not cover adjacency");
  }

  io_global_graph_cell_assignment_t result;
  result.node_to_cell.assign(static_cast<size_t>(header.point_count), UINT32_MAX);
  std::vector<uint8_t> state(static_cast<size_t>(header.point_count), 0);
  std::vector<uint32_t> frontier;
  const uint64_t reserve = std::min<uint64_t>(
      header.point_count, static_cast<uint64_t>(target_size) * header.max_degree);
  frontier.reserve(static_cast<size_t>(reserve));
  for (uint32_t seed = 0; seed < header.point_count; ++seed) {
    if (state[seed] != 0) {
      continue;
    }
    if (result.cell_count == UINT32_MAX) {
      throw std::overflow_error("mapped global graph Cell count exceeds uint32");
    }
    const uint32_t cell = result.cell_count++;
    frontier.clear();
    uint32_t population = 0;
    const auto add_node = [&](uint32_t node) {
      state[node] = 1;
      result.node_to_cell[node] = cell;
      ++population;
      if (offsets[node] > offsets[node + 1] || offsets[node + 1] > adjacency.size()) {
        throw std::invalid_argument("mapped raw topology contains an invalid neighbor range");
      }
      for (const uint32_t neighbor :
           adjacency.subspan(static_cast<size_t>(offsets[node]),
                             static_cast<size_t>(offsets[node + 1] - offsets[node]))) {
        if (neighbor >= header.point_count) {
          throw std::invalid_argument("mapped raw topology contains an invalid neighbor ID");
        }
        if (state[neighbor] == 0) {
          state[neighbor] = 2;
          frontier.push_back(neighbor);
        }
      }
    };
    add_node(seed);
    size_t cursor = 0;
    while (population < target_size && cursor < frontier.size()) {
      add_node(frontier[cursor++]);
    }
    for (; cursor < frontier.size(); ++cursor) {
      if (state[frontier[cursor]] == 2) {
        state[frontier[cursor]] = 0;
      }
    }
  }
  return result;
}

std::filesystem::path make_io_topology_path(const std::filesystem::path& index_path_prefix) {
  auto path = index_path_prefix;
  path += ".io_topology.bin";
  return path;
}

std::filesystem::path make_io_vector_path(const std::filesystem::path& index_path_prefix) {
  auto path = index_path_prefix;
  path += ".io_vectors.bin";
  return path;
}

std::filesystem::path make_io_cell_adjacency_path(const std::filesystem::path& index_path_prefix) {
  auto path = index_path_prefix;
  path += ".cell_adj.bin";
  return path;
}

io_optimized_build_result_t io_optimized_index_t::build(
    const std::filesystem::path& index_path_prefix,
    const std::filesystem::path& requested_topology_path,
    const std::filesystem::path& requested_vector_path, io_adjacency_encoding_t encoding,
    io_vector_layout_t layout, const std::filesystem::path& community_polar_path,
    bool build_prefetch_hints, const std::filesystem::path& hub_clique_low_rcni_path) {
  if (index_path_prefix.empty()) {
    throw std::runtime_error("IO-1 build requires an index prefix");
  }
  if (encoding != io_adjacency_encoding_t::RAW_U32 &&
      encoding != io_adjacency_encoding_t::PFOR_DELTA) {
    throw std::runtime_error("unsupported IO adjacency encoding");
  }
#if !defined(POWERLAWANN_USE_PFORDELTA)
  if (encoding == io_adjacency_encoding_t::PFOR_DELTA) {
    throw std::runtime_error("PForDelta adjacency is unavailable in this build");
  }
#endif
  if (layout != io_vector_layout_t::ORIGINAL_ID && layout != io_vector_layout_t::CELL_4K &&
      layout != io_vector_layout_t::CELL_CHUNKS && layout != io_vector_layout_t::CELL_WEIGHTED_4K &&
      layout != io_vector_layout_t::CELL_U8_4K) {
    throw std::runtime_error("unsupported IO vector layout");
  }
  if (!hub_clique_low_rcni_path.empty() && layout == io_vector_layout_t::ORIGINAL_ID) {
    throw std::runtime_error("low-RCNI Hub-clique trim requires a Cell layout and sidecar");
  }
  const auto disk_path = std::filesystem::path(index_path_prefix.string() + "_disk.index");
  if (std::filesystem::exists(disk_path.string() + "_pq_pivots.bin")) {
    throw std::runtime_error("IO-1 vector extraction does not support disk-PQ Base records");
  }
  const auto topology_path = requested_topology_path.empty()
                                 ? make_io_topology_path(index_path_prefix)
                                 : requested_topology_path;
  const auto vector_path = requested_vector_path.empty() ? make_io_vector_path(index_path_prefix)
                                                         : requested_vector_path;
  validate_output_path(topology_path);
  validate_output_path(vector_path);
  if (topology_path == vector_path || topology_path == disk_path || vector_path == disk_path) {
    throw std::runtime_error(
        "IO artifact paths must be distinct from each other and the Base index");
  }

  const auto metadata = read_disk_layout_metadata(disk_path);
  const bool cell_layout = layout != io_vector_layout_t::ORIGINAL_ID;
  const bool u8_layout = layout == io_vector_layout_t::CELL_U8_4K;
  const uint64_t source_vector_bytes =
      checked_multiply(metadata.dimension, sizeof(float), "source vector bytes");
  const uint64_t vector_bytes = u8_layout ? metadata.dimension : source_vector_bytes;
  if (vector_bytes == 0 || vector_bytes > k_io_vector_page_size) {
    throw std::runtime_error("IO-1 requires one complete vector to fit in a 4-KiB page");
  }
  const uint64_t raw_vector_spool_bytes =
      checked_multiply(metadata.point_count, vector_bytes, "raw-vector spool size");
  const uint64_t vectors_per_page = k_io_vector_page_size / vector_bytes;

  std::vector<uint32_t> placement_order;
  std::vector<io_cell_page_range_t> cell_ranges;
  std::optional<community_polar_index_t> community_index;
  std::vector<double> normalized_rcni;
  std::vector<uint8_t> trim_low_rcni;
  fingerprint_t base_fingerprint;
  fingerprint_t community_polar_fingerprint;
  if (cell_layout) {
    const auto sidecar_path = community_polar_path.empty()
                                  ? make_community_polar_index_path(index_path_prefix)
                                  : community_polar_path;
    community_index = load_community_polar_index(sidecar_path);
    community_polar_fingerprint = fingerprint_file(sidecar_path);
    validate_community_polar_artifacts(*community_index, index_path_prefix);
    base_fingerprint = {community_index->base_index_fingerprint.size_bytes,
                        community_index->base_index_fingerprint.fnv1a_hash};
    if (community_index->point_count != metadata.point_count ||
        community_index->dimension != metadata.dimension || community_index->cells.empty()) {
      throw std::runtime_error("Community-Polar sidecar shape is incompatible with IO-2 layout");
    }
    placement_order.assign(metadata.point_count, std::numeric_limits<uint32_t>::max());
    for (uint64_t node = 0; node < metadata.point_count; ++node) {
      const uint64_t packet = community_index->node_to_packet_record[node];
      if (packet >= metadata.point_count ||
          placement_order[packet] != std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Community-Polar packet order is not a complete permutation");
      }
      placement_order[packet] = static_cast<uint32_t>(node);
    }
    if (std::find(placement_order.begin(), placement_order.end(),
                  std::numeric_limits<uint32_t>::max()) != placement_order.end()) {
      throw std::runtime_error("Community-Polar packet order does not cover every Base node");
    }
    cell_ranges.resize(community_index->cells.size());
    if (!hub_clique_low_rcni_path.empty()) {
      normalized_rcni = read_normalized_rcni(hub_clique_low_rcni_path, metadata.point_count);
      trim_low_rcni.assign(metadata.point_count, 0);
      std::vector<std::vector<uint32_t>> members(community_index->communities.size());
      for (uint32_t node = 0; node < metadata.point_count; ++node) {
        members[community_index->node_to_community[node]].push_back(node);
      }
      for (auto& community : members) {
        std::sort(community.begin(), community.end(), [&](uint32_t left, uint32_t right) {
          return std::tie(normalized_rcni[left], left) < std::tie(normalized_rcni[right], right);
        });
        const size_t low_count = community.size() / 2U;
        for (size_t rank = 0; rank < low_count; ++rank) {
          trim_low_rcni[community[rank]] = 1;
        }
      }
    }
  } else {
    base_fingerprint = fingerprint_file(disk_path);
  }

  std::vector<uint64_t> offsets(metadata.point_count + 1U, 0);
  std::vector<uint32_t> adjacency;
  adjacency.reserve(static_cast<size_t>(metadata.point_count) * metadata.max_degree);
  std::vector<io_vector_location_t> locations(metadata.point_count);
  const auto temporary_vector = std::filesystem::path(vector_path.string() + ".tmp");
  const auto temporary_topology = std::filesystem::path(topology_path.string() + ".tmp");
  const auto temporary_raw_vectors =
      std::filesystem::path(vector_path.string() + ".raw_vectors.tmp");
  std::error_code ignored;
  uint64_t low_rcni_trimmed_nodes = 0;
  std::filesystem::remove(temporary_vector, ignored);
  std::filesystem::remove(temporary_topology, ignored);
  std::filesystem::remove(temporary_raw_vectors, ignored);
  try {
    std::ifstream disk(disk_path, std::ios::binary);
    std::ofstream vectors(temporary_vector, std::ios::binary | std::ios::trunc);
    std::ofstream raw_vector_spool;
    if (cell_layout) {
      raw_vector_spool.open(temporary_raw_vectors, std::ios::binary | std::ios::trunc);
    }
    if (!disk || !vectors || (cell_layout && !raw_vector_spool)) {
      throw std::runtime_error("failed to open IO-1 input or temporary vector artifact");
    }
    std::array<uint8_t, k_io_vector_page_size> output_page{};
    uint64_t output_slot = 0;
    uint64_t pages_written = 0;
    uint64_t vector_checksum = k_fnv_offset_basis;
    auto flush_page = [&]() {
      if (output_slot == 0) {
        return;
      }
      vectors.write(reinterpret_cast<const char*>(output_page.data()), output_page.size());
      if (!vectors) {
        throw std::runtime_error("failed writing IO vector artifact");
      }
      hash_bytes(vector_checksum, output_page.data(), output_page.size());
      output_page.fill(0);
      output_slot = 0;
      ++pages_written;
    };
    auto append_padding_page = [&]() {
      if (output_slot != 0) {
        throw std::runtime_error("cannot append Chunk padding into a partial vector page");
      }
      vectors.write(reinterpret_cast<const char*>(output_page.data()), output_page.size());
      if (!vectors) {
        throw std::runtime_error("failed writing IO Chunk padding");
      }
      hash_bytes(vector_checksum, output_page.data(), output_page.size());
      ++pages_written;
    };
    auto append_vector = [&](uint32_t node_id, const uint8_t* vector) {
      const uint64_t output_offset = output_slot * vector_bytes;
      locations[node_id] = {pages_written, static_cast<uint32_t>(output_offset)};
      std::memcpy(output_page.data() + output_offset, vector, static_cast<size_t>(vector_bytes));
      ++output_slot;
      if (output_slot == vectors_per_page) {
        flush_page();
      }
    };
    auto append_record = [&](uint64_t node_id, const uint8_t* record) {
      if (node_id >= metadata.point_count) {
        throw std::runtime_error("Base record node ID exceeds point count");
      }
      if (cell_layout) {
        std::array<uint8_t, k_io_vector_page_size> converted{};
        const uint8_t* vector = record;
        if (u8_layout) {
          const auto* source = reinterpret_cast<const float*>(record);
          for (uint64_t dimension = 0; dimension < metadata.dimension; ++dimension) {
            const float value = source[dimension];
            if (!std::isfinite(value) || value < 0.0F || value > 255.0F ||
                std::nearbyint(value) != value) {
              throw std::runtime_error(
                  "Cell-u8 layout requires integral Base coordinates in [0, 255]");
            }
            converted[dimension] = static_cast<uint8_t>(value);
          }
          vector = converted.data();
        }
        raw_vector_spool.write(reinterpret_cast<const char*>(vector),
                               static_cast<std::streamsize>(vector_bytes));
        if (!raw_vector_spool) {
          throw std::runtime_error("failed writing bounded raw-vector spool");
        }
      } else {
        append_vector(static_cast<uint32_t>(node_id), record);
      }

      const auto* neighborhood = reinterpret_cast<const uint32_t*>(record + source_vector_bytes);
      uint32_t degree = 0;
      std::memcpy(&degree, neighborhood, sizeof(degree));
      if (degree > metadata.max_degree) {
        throw std::runtime_error("Base record degree exceeds metadata");
      }
      offsets[node_id] = adjacency.size();
      uint32_t removed_neighbor = degree;
      if (!trim_low_rcni.empty() && trim_low_rcni[node_id] != 0 && degree != 0) {
        removed_neighbor = 0;
        uint32_t removed_target = 0;
        std::memcpy(&removed_target, neighborhood + 1U, sizeof(removed_target));
        for (uint32_t neighbor = 1; neighbor < degree; ++neighbor) {
          uint32_t target = 0;
          std::memcpy(&target, neighborhood + 1U + neighbor, sizeof(target));
          if (std::tie(normalized_rcni[target], target) <
              std::tie(normalized_rcni[removed_target], removed_target)) {
            removed_neighbor = neighbor;
            removed_target = target;
          }
        }
        ++low_rcni_trimmed_nodes;
      }
      for (uint32_t neighbor = 0; neighbor < degree; ++neighbor) {
        uint32_t target = 0;
        std::memcpy(&target, neighborhood + 1U + neighbor, sizeof(target));
        if (target >= metadata.point_count) {
          throw std::runtime_error("Base adjacency endpoint exceeds point count");
        }
        if (neighbor != removed_neighbor) {
          adjacency.push_back(target);
        }
      }
    };

    if (metadata.nodes_per_sector != 0) {
      if (metadata.max_node_len * metadata.nodes_per_sector > defaults::SECTOR_LEN) {
        throw std::runtime_error("Base nodes-per-sector metadata exceeds one sector");
      }
      std::array<uint8_t, defaults::SECTOR_LEN> sector{};
      uint64_t node_id = 0;
      const uint64_t sector_count =
          (metadata.point_count + metadata.nodes_per_sector - 1U) / metadata.nodes_per_sector;
      for (uint64_t sector_id = 0; sector_id < sector_count; ++sector_id) {
        disk.seekg(static_cast<std::streamoff>((sector_id + 1U) * defaults::SECTOR_LEN));
        disk.read(reinterpret_cast<char*>(sector.data()), sector.size());
        if (!disk) {
          throw std::runtime_error("truncated Base node sector");
        }
        for (uint64_t slot = 0; slot < metadata.nodes_per_sector && node_id < metadata.point_count;
             ++slot, ++node_id) {
          append_record(node_id, sector.data() + slot * metadata.max_node_len);
        }
      }
    } else {
      const uint64_t sectors_per_node =
          (metadata.max_node_len + defaults::SECTOR_LEN - 1U) / defaults::SECTOR_LEN;
      std::vector<uint8_t> record(sectors_per_node * defaults::SECTOR_LEN);
      for (uint64_t node_id = 0; node_id < metadata.point_count; ++node_id) {
        const uint64_t sector = 1U + node_id * sectors_per_node;
        disk.seekg(static_cast<std::streamoff>(sector * defaults::SECTOR_LEN));
        disk.read(reinterpret_cast<char*>(record.data()),
                  static_cast<std::streamsize>(record.size()));
        if (!disk) {
          throw std::runtime_error("truncated Base multi-sector node record");
        }
        append_record(node_id, record.data());
      }
    }
    offsets[metadata.point_count] = adjacency.size();
    if (cell_layout) {
      raw_vector_spool.close();
      if (!raw_vector_spool ||
          std::filesystem::file_size(temporary_raw_vectors) != raw_vector_spool_bytes) {
        throw std::runtime_error("raw-vector spool size accounting failed");
      }
    }
    if (layout == io_vector_layout_t::CELL_WEIGHTED_4K) {
      auto has_edge = [&](uint32_t source, uint32_t target) {
        return std::find(adjacency.begin() + static_cast<ptrdiff_t>(offsets[source]),
                         adjacency.begin() + static_cast<ptrdiff_t>(offsets[source + 1U]),
                         target) !=
               adjacency.begin() + static_cast<ptrdiff_t>(offsets[source + 1U]);
      };
      size_t cell_begin = 0;
      while (cell_begin < placement_order.size()) {
        const uint32_t cell = community_index->node_to_cell[placement_order[cell_begin]];
        size_t cell_end = cell_begin;
        while (cell_end < placement_order.size() &&
               community_index->node_to_cell[placement_order[cell_end]] == cell) {
          ++cell_end;
        }
        std::vector<uint32_t> remaining(placement_order.begin() +
                                            static_cast<ptrdiff_t>(cell_begin),
                                        placement_order.begin() + static_cast<ptrdiff_t>(cell_end));
        std::sort(remaining.begin(), remaining.end());
        std::vector<uint32_t> ordered;
        ordered.reserve(remaining.size());
        while (!remaining.empty()) {
          auto best = remaining.begin();
          uint32_t best_score = 0;
          for (auto candidate = remaining.begin(); candidate != remaining.end(); ++candidate) {
            uint32_t score = 0;
            const size_t context_begin =
                ordered.size() > vectors_per_page ? ordered.size() - vectors_per_page : 0;
            for (size_t context = context_begin; context < ordered.size(); ++context) {
              const bool forward = has_edge(ordered[context], *candidate);
              const bool reverse = has_edge(*candidate, ordered[context]);
              score += static_cast<uint32_t>(forward) + static_cast<uint32_t>(reverse) +
                       static_cast<uint32_t>(forward && reverse) * 2U;
            }
            if (score > best_score || (score == best_score && *candidate < *best)) {
              best = candidate;
              best_score = score;
            }
          }
          ordered.push_back(*best);
          remaining.erase(best);
        }
        std::copy(ordered.begin(), ordered.end(),
                  placement_order.begin() + static_cast<ptrdiff_t>(cell_begin));
        cell_begin = cell_end;
      }
    }
    std::array<uint64_t, 4> chunk_counts{};
    if (cell_layout) {
      std::ifstream raw_vector_input;
      std::array<uint8_t, k_io_vector_page_size> raw_vector{};
      raw_vector_input.open(temporary_raw_vectors, std::ios::binary);
      if (!raw_vector_input) {
        throw std::runtime_error("failed opening bounded raw-vector spool");
      }
      size_t placement_cursor = 0;
      for (size_t cell_id = 0; cell_id < cell_ranges.size(); ++cell_id) {
        auto& range = cell_ranges[cell_id];
        range.first_page = pages_written;
        const uint64_t cell_vector_bytes = checked_multiply(
            community_index->cells[cell_id].node_count, vector_bytes, "Cell vector bytes");
        uint64_t chunk_bytes = k_io_vector_page_size;
        size_t chunk_class = 0;
        if (layout == io_vector_layout_t::CELL_CHUNKS) {
          constexpr std::array<uint64_t, 4> classes = {4096, 65536, 524288, 2097152};
          while (chunk_class + 1U < classes.size() && cell_vector_bytes > classes[chunk_class]) {
            ++chunk_class;
          }
          chunk_bytes = classes[chunk_class];
        }
        range.chunk_size_bytes = static_cast<uint32_t>(chunk_bytes);
        while (placement_cursor < placement_order.size() &&
               community_index->node_to_cell[placement_order[placement_cursor]] == cell_id) {
          const uint32_t node = placement_order[placement_cursor++];
          const auto source_offset = static_cast<uint64_t>(node) * vector_bytes;
          const uint8_t* source = raw_vector.data();
          raw_vector_input.seekg(static_cast<std::streamoff>(source_offset));
          raw_vector_input.read(reinterpret_cast<char*>(raw_vector.data()),
                                static_cast<std::streamsize>(vector_bytes));
          if (!raw_vector_input) {
            throw std::runtime_error("failed reading bounded raw-vector spool");
          }
          append_vector(node, source);
          ++range.node_count;
        }
        flush_page();
        const uint64_t useful_pages = pages_written - range.first_page;
        uint64_t allocated_pages = useful_pages;
        if (layout == io_vector_layout_t::CELL_CHUNKS) {
          const uint64_t pages_per_chunk = chunk_bytes / k_io_vector_page_size;
          const uint64_t chunks = (useful_pages + pages_per_chunk - 1U) / pages_per_chunk;
          allocated_pages = checked_multiply(chunks, pages_per_chunk, "Cell Chunk pages");
          chunk_counts[chunk_class] += chunks;
          while (pages_written - range.first_page < allocated_pages) {
            append_padding_page();
          }
        } else {
          chunk_counts[0] += useful_pages;
        }
        const uint64_t page_count = pages_written - range.first_page;
        if (range.node_count != community_index->cells[cell_id].node_count || page_count == 0 ||
            page_count > std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error("IO-2 Cell placement does not cover its declared nodes");
        }
        range.page_count = static_cast<uint32_t>(page_count);
      }
      if (placement_cursor != placement_order.size()) {
        throw std::runtime_error("IO-2 Cell placement does not cover all nodes");
      }
      raw_vector_input.close();
      std::filesystem::remove(temporary_raw_vectors, ignored);
    } else {
      flush_page();
    }
    vectors.close();
    const uint64_t vector_page_count = pages_written;
    const uint64_t expected_vector_file_size =
        checked_multiply(vector_page_count, k_io_vector_page_size, "vector artifact size");
    if (std::filesystem::file_size(temporary_vector) != expected_vector_file_size) {
      throw std::runtime_error("IO-1 vector artifact size accounting failed");
    }

    std::vector<uint64_t> stored_offsets = offsets;
    std::vector<uint8_t> adjacency_bytes;
    if (encoding == io_adjacency_encoding_t::RAW_U32) {
      adjacency_bytes = serialize_u32(adjacency);
    } else {
#if defined(POWERLAWANN_USE_PFORDELTA)
      auto compressed = compress_adjacency(offsets, adjacency);
      stored_offsets = std::move(compressed.first);
      adjacency_bytes = std::move(compressed.second);
#endif
    }
    const auto offset_bytes = serialize_u64(stored_offsets);
    const auto directory_bytes = serialize_locations(locations);
    const auto cell_directory_bytes = serialize_cell_ranges(cell_ranges);
    std::vector<uint64_t> prefetch_pages(metadata.point_count, k_no_io_prefetch_page);
    if (build_prefetch_hints) {
      for (uint64_t node = 0; node < metadata.point_count; ++node) {
        std::vector<std::pair<uint64_t, uint32_t>> page_counts;
        const uint64_t direct_begin = offsets[node];
        const uint64_t direct_count = offsets[node + 1U] - direct_begin;
        const uint64_t direct_limit = std::min<uint64_t>(direct_count, 8U);
        for (uint64_t direct = 0; direct < direct_limit; ++direct) {
          const uint32_t middle = adjacency[direct_begin + direct];
          const uint64_t second_begin = offsets[middle];
          const uint64_t second_count = offsets[middle + 1U] - second_begin;
          const uint64_t second_limit = std::min<uint64_t>(second_count, 8U);
          for (uint64_t second = 0; second < second_limit; ++second) {
            const uint64_t page = locations[adjacency[second_begin + second]].page_id;
            if (page == locations[node].page_id) {
              continue;
            }
            const auto existing =
                std::find_if(page_counts.begin(), page_counts.end(),
                             [page](const auto& entry) { return entry.first == page; });
            if (existing == page_counts.end()) {
              page_counts.emplace_back(page, 1U);
            } else {
              ++existing->second;
            }
          }
        }
        if (!page_counts.empty()) {
          const auto best = std::max_element(
              page_counts.begin(), page_counts.end(), [](const auto& left, const auto& right) {
                return left.second < right.second ||
                       (left.second == right.second && left.first > right.first);
              });
          prefetch_pages[node] = best->first;
        }
      }
    }
    const auto prefetch_directory_bytes = serialize_u64(prefetch_pages);
    std::vector<io_gateway_prefetch_hint_t> gateway_hints;
    if (build_prefetch_hints && community_index.has_value()) {
      struct transition_summary_t {
        std::array<uint32_t, 2> cells{UINT32_MAX, UINT32_MAX};
        std::array<uint32_t, 2> counts{};
        uint32_t total = 0;
      };
      std::unordered_map<uint64_t, uint32_t> transition_counts;
      transition_counts.reserve(std::min<uint64_t>(adjacency.size(), cell_ranges.size() * 64U));
      for (uint32_t source = 0; source < metadata.point_count; ++source) {
        const uint32_t source_cell = community_index->node_to_cell[source];
        for (uint64_t edge = offsets[source]; edge < offsets[source + 1U]; ++edge) {
          const uint32_t target_cell = community_index->node_to_cell[adjacency[edge]];
          if (source_cell == target_cell) {
            continue;
          }
          const uint64_t key = (static_cast<uint64_t>(source_cell) << 32U) | target_cell;
          auto& count = transition_counts[key];
          if (count != std::numeric_limits<uint32_t>::max()) {
            ++count;
          }
        }
      }
      std::vector<transition_summary_t> summaries(cell_ranges.size());
      std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> ordered_transitions;
      ordered_transitions.reserve(transition_counts.size());
      for (const auto& [key, count] : transition_counts) {
        ordered_transitions.emplace_back(static_cast<uint32_t>(key >> 32U),
                                         static_cast<uint32_t>(key), count);
      }
      std::sort(ordered_transitions.begin(), ordered_transitions.end(),
                [](const auto& left, const auto& right) {
                  if (std::get<0>(left) != std::get<0>(right)) {
                    return std::get<0>(left) < std::get<0>(right);
                  }
                  if (std::get<2>(left) != std::get<2>(right)) {
                    return std::get<2>(left) > std::get<2>(right);
                  }
                  return std::get<1>(left) < std::get<1>(right);
                });
      for (const auto& [source_cell, target_cell, count] : ordered_transitions) {
        auto& summary = summaries[source_cell];
        summary.total = static_cast<uint32_t>(std::min<uint64_t>(
            std::numeric_limits<uint32_t>::max(), static_cast<uint64_t>(summary.total) + count));
        if (summary.cells[0] == UINT32_MAX) {
          summary.cells[0] = target_cell;
          summary.counts[0] = count;
        } else if (summary.cells[1] == UINT32_MAX) {
          summary.cells[1] = target_cell;
          summary.counts[1] = count;
        }
      }
      gateway_hints.reserve(community_index->gateways.size());
      for (const auto& gateway : community_index->gateways) {
        io_gateway_prefetch_hint_t hint;
        hint.target_node = gateway.target_node;
        hint.landing_cell = community_index->node_to_cell[gateway.target_node];
        hint.landing_node_page = locations[gateway.target_node].page_id;
        hint.landing_cell_first_page = cell_ranges[hint.landing_cell].first_page;
        const auto& summary = summaries[hint.landing_cell];
        hint.successor_cells = summary.cells;
        hint.transition_counts = summary.counts;
        hint.outgoing_transition_count = summary.total;
        for (size_t successor = 0; successor < hint.successor_cells.size(); ++successor) {
          if (hint.successor_cells[successor] != UINT32_MAX) {
            hint.successor_first_pages[successor] =
                cell_ranges[hint.successor_cells[successor]].first_page;
          }
        }
        gateway_hints.push_back(hint);
      }
    }
    const auto gateway_directory_bytes = serialize_gateway_hints(gateway_hints);
    topology_header_t header;
    header.version = u8_layout ? k_topology_version : k_topology_version_v5;
    header.point_count = metadata.point_count;
    header.dimension = metadata.dimension;
    header.max_degree = metadata.max_degree;
    header.encoding = encoding;
    header.layout = layout;
    header.edge_count = adjacency.size();
    header.vector_page_count = vector_page_count;
    header.cell_count = cell_ranges.size();
    header.base = base_fingerprint;
    header.vectors = {expected_vector_file_size, vector_checksum};
    header.offsets_offset = k_topology_header_bytes;
    header.offsets_bytes = offset_bytes.size();
    header.offsets_checksum = checksum_bytes(offset_bytes);
    header.adjacency_offset =
        checked_add(header.offsets_offset, header.offsets_bytes, "adjacency section offset");
    header.adjacency_bytes = adjacency_bytes.size();
    header.adjacency_checksum = checksum_bytes(adjacency_bytes);
    header.directory_offset =
        checked_add(header.adjacency_offset, header.adjacency_bytes, "directory section offset");
    header.directory_bytes = directory_bytes.size();
    header.directory_checksum = checksum_bytes(directory_bytes);
    header.cell_directory_offset = checked_add(header.directory_offset, header.directory_bytes,
                                               "Cell directory section offset");
    header.cell_directory_bytes = cell_directory_bytes.size();
    header.cell_directory_checksum = checksum_bytes(cell_directory_bytes);
    header.prefetch_directory_offset =
        checked_add(header.cell_directory_offset, header.cell_directory_bytes,
                    "prefetch directory section offset");
    header.prefetch_directory_bytes = prefetch_directory_bytes.size();
    header.prefetch_directory_checksum = checksum_bytes(prefetch_directory_bytes);
    header.community_polar = community_polar_fingerprint;
    header.gateway_count = gateway_hints.size();
    header.gateway_directory_offset =
        checked_add(header.prefetch_directory_offset, header.prefetch_directory_bytes,
                    "Gateway directory section offset");
    header.gateway_directory_bytes = gateway_directory_bytes.size();
    header.gateway_directory_checksum = checksum_bytes(gateway_directory_bytes);
    const auto header_bytes = serialize_header(header);

    std::ofstream topology(temporary_topology, std::ios::binary | std::ios::trunc);
    topology.write(reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());
    topology.write(reinterpret_cast<const char*>(offset_bytes.data()), offset_bytes.size());
    topology.write(reinterpret_cast<const char*>(adjacency_bytes.data()), adjacency_bytes.size());
    topology.write(reinterpret_cast<const char*>(directory_bytes.data()), directory_bytes.size());
    topology.write(reinterpret_cast<const char*>(cell_directory_bytes.data()),
                   cell_directory_bytes.size());
    topology.write(reinterpret_cast<const char*>(prefetch_directory_bytes.data()),
                   prefetch_directory_bytes.size());
    topology.write(reinterpret_cast<const char*>(gateway_directory_bytes.data()),
                   gateway_directory_bytes.size());
    topology.close();
    if (!topology) {
      throw std::runtime_error("failed writing IO-1 topology artifact");
    }
    const uint64_t expected_topology_size = checked_add(
        header.gateway_directory_offset, header.gateway_directory_bytes, "topology artifact size");
    if (std::filesystem::file_size(temporary_topology) != expected_topology_size) {
      throw std::runtime_error("IO-1 topology artifact size accounting failed");
    }
    publish_file(temporary_vector, vector_path);
    publish_file(temporary_topology, topology_path);

    io_optimized_build_result_t result;
    result.point_count = metadata.point_count;
    result.edge_count = adjacency.size();
    result.topology_bytes = expected_topology_size;
    result.adjacency_bytes = adjacency_bytes.size();
    result.vector_bytes = expected_vector_file_size;
    result.vector_padding_bytes = expected_vector_file_size - metadata.point_count * vector_bytes;
    result.build_peak_payload_bytes =
        offset_bytes.size() + adjacency.size() * sizeof(uint32_t) + adjacency_bytes.size() +
        directory_bytes.size() + cell_directory_bytes.size() + prefetch_directory_bytes.size() +
        gateway_directory_bytes.size() + k_io_vector_page_size;
    result.cell_count = cell_ranges.size();
    result.nonempty_vector_bytes = metadata.point_count * vector_bytes;
    result.cell_page_count = vector_page_count;
    result.gateway_prefetch_hint_count = gateway_hints.size();
    result.gateway_prefetch_hint_bytes = gateway_directory_bytes.size();
    result.low_rcni_trimmed_nodes = low_rcni_trimmed_nodes;
    result.chunk_counts = chunk_counts;
    return result;
  } catch (...) {
    std::filesystem::remove(temporary_vector, ignored);
    std::filesystem::remove(temporary_topology, ignored);
    std::filesystem::remove(temporary_raw_vectors, ignored);
    throw;
  }
}

void rebind_io_topology_community_polar(
    const std::filesystem::path& source_topology_path,
    const std::filesystem::path& source_community_polar_path,
    const std::filesystem::path& replacement_community_polar_path,
    const std::filesystem::path& output_topology_path) {
  if (source_topology_path.empty() || source_community_polar_path.empty() ||
      replacement_community_polar_path.empty() || output_topology_path.empty() ||
      std::filesystem::absolute(source_topology_path).lexically_normal() ==
          std::filesystem::absolute(output_topology_path).lexically_normal()) {
    throw std::invalid_argument("IO topology rebind requires distinct, nonempty paths");
  }
  validate_output_path(output_topology_path);

  std::array<uint8_t, k_topology_header_bytes> source_header_bytes{};
  std::ifstream source(source_topology_path, std::ios::binary);
  source.read(reinterpret_cast<char*>(source_header_bytes.data()), source_header_bytes.size());
  if (!source) {
    throw std::runtime_error("failed reading source IO topology header");
  }
  auto header = parse_header(source_header_bytes);
  if (header.version < k_topology_version_v5 || header.version > k_topology_version ||
      header.header_bytes != k_topology_header_bytes || header.cell_count == 0 ||
      header.community_polar.size == 0) {
    throw std::runtime_error("IO topology rebind requires a version-5/6 Cell topology");
  }
  const auto source_fingerprint = fingerprint_file(source_community_polar_path);
  if (header.community_polar.size != source_fingerprint.size ||
      header.community_polar.checksum != source_fingerprint.checksum) {
    throw std::runtime_error("source IO topology is not bound to the supplied Cell sidecar");
  }
  validate_rebound_cell_structure(source_community_polar_path, replacement_community_polar_path);

  header.community_polar = fingerprint_file(replacement_community_polar_path);
  const auto rebound_header_bytes = serialize_header(header);
  const auto temporary = std::filesystem::path(output_topology_path.string() + ".tmp");
  std::error_code ignored;
  std::filesystem::remove(temporary, ignored);
  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(rebound_header_bytes.data()),
                 rebound_header_bytes.size());
    std::array<char, 1U << 20U> buffer{};
    while (source) {
      source.read(buffer.data(), buffer.size());
      const auto count = source.gcount();
      if (count > 0) {
        output.write(buffer.data(), count);
      }
    }
    output.close();
    if (!source.eof() || !output ||
        std::filesystem::file_size(temporary) != std::filesystem::file_size(source_topology_path)) {
      throw std::runtime_error("failed streaming rebound IO topology");
    }
    publish_file(temporary, output_topology_path);
  } catch (...) {
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

std::shared_ptr<const io_optimized_index_t>
io_optimized_index_t::load(const std::filesystem::path& index_path_prefix,
                           const std::filesystem::path& requested_topology_path,
                           const std::filesystem::path& requested_vector_path, bool topology_only,
                           bool paged_topology, uint64_t topology_cache_pages) {
  const auto disk_path = std::filesystem::path(index_path_prefix.string() + "_disk.index");
  const auto topology_path = requested_topology_path.empty()
                                 ? make_io_topology_path(index_path_prefix)
                                 : requested_topology_path;
  const auto vector_path = requested_vector_path.empty() ? make_io_vector_path(index_path_prefix)
                                                         : requested_vector_path;
  const uint64_t topology_file_size = std::filesystem::file_size(topology_path);
  std::vector<uint8_t> bytes;
  topology_header_t header;
  if (paged_topology) {
    header = parse_header(read_file_range(topology_path, 0, k_topology_header_bytes));
  } else {
    bytes = read_file(topology_path);
    header = parse_header(bytes);
  }
  if (paged_topology &&
      (topology_cache_pages == 0 || header.encoding != io_adjacency_encoding_t::RAW_U32)) {
    throw std::runtime_error("paged topology requires a positive cache and raw uint32 adjacency");
  }
  if (!paged_topology && topology_cache_pages != 0) {
    throw std::runtime_error("topology cache capacity requires paged topology");
  }
  if (header.point_count == 0 || header.dimension == 0 ||
      (header.encoding != io_adjacency_encoding_t::RAW_U32 &&
       header.encoding != io_adjacency_encoding_t::PFOR_DELTA) ||
      (header.layout != io_vector_layout_t::ORIGINAL_ID &&
       header.layout != io_vector_layout_t::CELL_4K &&
       header.layout != io_vector_layout_t::CELL_CHUNKS &&
       header.layout != io_vector_layout_t::CELL_WEIGHTED_4K &&
       header.layout != io_vector_layout_t::CELL_U8_4K) ||
      (header.layout == io_vector_layout_t::CELL_U8_4K && header.version != k_topology_version) ||
      (header.layout == io_vector_layout_t::ORIGINAL_ID && header.cell_count != 0) ||
      ((header.layout == io_vector_layout_t::CELL_4K ||
        header.layout == io_vector_layout_t::CELL_CHUNKS ||
        header.layout == io_vector_layout_t::CELL_WEIGHTED_4K ||
        header.layout == io_vector_layout_t::CELL_U8_4K) &&
       header.cell_count == 0)) {
    throw std::runtime_error("unsupported or empty IO topology payload");
  }
  if (header.gateway_count != 0 && (header.version < k_topology_version_v5 ||
                                    header.cell_count == 0 || header.community_polar.size == 0)) {
    throw std::runtime_error("IO Gateway hints require a bound Cell sidecar");
  }
  const auto offsets_count = checked_add(header.point_count, 1U, "offset count");
  const auto expected_offsets_bytes =
      checked_multiply(offsets_count, sizeof(uint64_t), "offset section size");
  const auto expected_raw_adjacency_bytes =
      checked_multiply(header.edge_count, sizeof(uint32_t), "adjacency section size");
  const auto expected_directory_bytes = checked_multiply(
      header.point_count, sizeof(uint64_t) + sizeof(uint32_t), "directory section size");
  const auto expected_cell_directory_bytes = checked_multiply(
      header.cell_count, sizeof(uint64_t) + 3U * sizeof(uint32_t), "Cell directory section size");
  const auto expected_prefetch_directory_bytes =
      checked_multiply(header.point_count, sizeof(uint64_t), "prefetch directory section size");
  const auto expected_gateway_directory_bytes =
      checked_multiply(header.gateway_count, 64U, "Gateway directory section size");
  const auto expected_adjacency_offset =
      checked_add(header.offsets_offset, header.offsets_bytes, "adjacency section offset");
  const auto expected_directory_offset =
      checked_add(header.adjacency_offset, header.adjacency_bytes, "directory section offset");
  const auto expected_cell_directory_offset =
      checked_add(header.directory_offset, header.directory_bytes, "Cell directory offset");
  const auto expected_prefetch_directory_offset = checked_add(
      header.cell_directory_offset, header.cell_directory_bytes, "prefetch directory offset");
  const auto expected_gateway_directory_offset =
      checked_add(header.prefetch_directory_offset, header.prefetch_directory_bytes,
                  "Gateway directory offset");
  const uint64_t expected_file_size =
      header.version == k_topology_version_v4
          ? expected_gateway_directory_offset
          : checked_add(expected_gateway_directory_offset, header.gateway_directory_bytes,
                        "IO topology file size");
  if (header.offsets_offset != header.header_bytes ||
      header.offsets_bytes != expected_offsets_bytes ||
      header.adjacency_offset != expected_adjacency_offset ||
      (header.encoding == io_adjacency_encoding_t::RAW_U32 &&
       header.adjacency_bytes != expected_raw_adjacency_bytes) ||
      header.directory_offset != expected_directory_offset ||
      header.directory_bytes != expected_directory_bytes ||
      header.cell_directory_offset != expected_cell_directory_offset ||
      header.cell_directory_bytes != expected_cell_directory_bytes ||
      header.prefetch_directory_offset != expected_prefetch_directory_offset ||
      header.prefetch_directory_bytes != expected_prefetch_directory_bytes ||
      (header.version >= k_topology_version_v5 &&
       (header.gateway_directory_offset != expected_gateway_directory_offset ||
        header.gateway_directory_bytes != expected_gateway_directory_bytes)) ||
      expected_file_size != topology_file_size) {
    throw std::runtime_error("invalid IO topology section offsets or file size");
  }
  std::vector<uint8_t> offset_storage;
  std::vector<uint8_t> directory_storage;
  std::vector<uint8_t> cell_directory_storage;
  std::vector<uint8_t> prefetch_directory_storage;
  std::vector<uint8_t> gateway_directory_storage;
  auto section = [&](uint64_t offset, uint64_t size) -> std::span<const uint8_t> {
    return {bytes.data() + static_cast<size_t>(offset), static_cast<size_t>(size)};
  };
  if (paged_topology) {
    offset_storage = read_file_range(topology_path, header.offsets_offset, header.offsets_bytes);
    directory_storage =
        read_file_range(topology_path, header.directory_offset, header.directory_bytes);
    cell_directory_storage =
        read_file_range(topology_path, header.cell_directory_offset, header.cell_directory_bytes);
    prefetch_directory_storage = read_file_range(topology_path, header.prefetch_directory_offset,
                                                 header.prefetch_directory_bytes);
    if (header.version >= k_topology_version_v5) {
      gateway_directory_storage = read_file_range(topology_path, header.gateway_directory_offset,
                                                  header.gateway_directory_bytes);
    }
  }
  const std::span<const uint8_t> offset_section =
      paged_topology ? std::span<const uint8_t>(offset_storage)
                     : section(header.offsets_offset, header.offsets_bytes);
  const std::span<const uint8_t> adjacency_section =
      paged_topology ? std::span<const uint8_t>()
                     : section(header.adjacency_offset, header.adjacency_bytes);
  const std::span<const uint8_t> directory_section =
      paged_topology ? std::span<const uint8_t>(directory_storage)
                     : section(header.directory_offset, header.directory_bytes);
  const std::span<const uint8_t> cell_directory_section =
      paged_topology ? std::span<const uint8_t>(cell_directory_storage)
                     : section(header.cell_directory_offset, header.cell_directory_bytes);
  const std::span<const uint8_t> prefetch_directory_section =
      paged_topology ? std::span<const uint8_t>(prefetch_directory_storage)
                     : section(header.prefetch_directory_offset, header.prefetch_directory_bytes);
  const std::span<const uint8_t> gateway_directory_section =
      header.version < k_topology_version_v5
          ? std::span<const uint8_t>()
          : paged_topology
                ? std::span<const uint8_t>(gateway_directory_storage)
                : section(header.gateway_directory_offset, header.gateway_directory_bytes);
  const uint64_t adjacency_checksum =
      paged_topology ? validate_raw_adjacency_range(topology_path, header.adjacency_offset,
                                                    header.edge_count, header.point_count)
                     : checksum_bytes(adjacency_section);
  if (checksum_bytes(offset_section) != header.offsets_checksum ||
      adjacency_checksum != header.adjacency_checksum ||
      checksum_bytes(directory_section) != header.directory_checksum ||
      checksum_bytes(cell_directory_section) != header.cell_directory_checksum ||
      checksum_bytes(prefetch_directory_section) != header.prefetch_directory_checksum ||
      (header.version >= k_topology_version_v5 &&
       checksum_bytes(gateway_directory_section) != header.gateway_directory_checksum)) {
    throw std::runtime_error("IO topology section checksum mismatch");
  }
  const auto base_fingerprint = fingerprint_file(disk_path);
  if (base_fingerprint.size != header.base.size ||
      base_fingerprint.checksum != header.base.checksum) {
    throw std::runtime_error("IO topology Base fingerprint mismatch");
  }
  if (header.vectors.size !=
      checked_multiply(header.vector_page_count, k_io_vector_page_size, "vector artifact size")) {
    throw std::runtime_error("IO vector artifact fingerprint or size mismatch");
  }
  if (!topology_only) {
    const auto vector_fingerprint = fingerprint_file(vector_path);
    if (vector_fingerprint.size != header.vectors.size ||
        vector_fingerprint.checksum != header.vectors.checksum) {
      throw std::runtime_error("IO vector artifact fingerprint or size mismatch");
    }
  }

  auto index = std::shared_ptr<io_optimized_index_t>(new io_optimized_index_t());
  index->point_count_ = header.point_count;
  index->dimension_ = header.dimension;
  index->max_degree_ = header.max_degree;
  index->edge_count_ = header.edge_count;
  index->vector_page_count_ = header.vector_page_count;
  index->vector_layout_ = header.layout;
  index->topology_bytes_ = topology_file_size;
  index->vector_bytes_ = header.vectors.size;
  index->vector_path_ = topology_only ? std::filesystem::path() : vector_path;
  index->neighbor_offsets_.reserve(header.point_count + 1U);
  if (!paged_topology) {
    index->adjacency_.reserve(header.edge_count);
  }
  index->vector_locations_.reserve(header.point_count);
  index->cell_page_ranges_.reserve(header.cell_count);
  index->node_prefetch_pages_.reserve(header.point_count);
  index->community_polar_fingerprint_ = {header.community_polar.size,
                                         header.community_polar.checksum};
  index->gateway_prefetch_hints_.reserve(header.gateway_count);
  size_t cursor = 0;
  for (uint64_t node = 0; node <= header.point_count; ++node) {
    index->neighbor_offsets_.push_back(
        read_little<uint64_t>(offset_section, cursor, "neighbor offset"));
  }
  if (index->neighbor_offsets_.front() != 0) {
    throw std::runtime_error("IO topology neighbor offsets do not start at zero");
  }
  if (header.encoding == io_adjacency_encoding_t::RAW_U32) {
    if (index->neighbor_offsets_.back() != header.edge_count) {
      throw std::runtime_error("IO topology neighbor offsets do not cover adjacency");
    }
    for (uint64_t node = 0; node < header.point_count; ++node) {
      if (index->neighbor_offsets_[node] > index->neighbor_offsets_[node + 1U] ||
          index->neighbor_offsets_[node + 1U] - index->neighbor_offsets_[node] >
              header.max_degree) {
        throw std::runtime_error("IO topology neighbor offsets are invalid");
      }
    }
    if (!paged_topology) {
      cursor = 0;
      for (uint64_t edge = 0; edge < header.edge_count; ++edge) {
        const uint32_t target =
            read_little<uint32_t>(adjacency_section, cursor, "adjacency target");
        if (target >= header.point_count) {
          throw std::runtime_error("IO topology adjacency target is out of range");
        }
        index->adjacency_.push_back(target);
      }
    }
  } else {
#if defined(POWERLAWANN_USE_PFORDELTA)
    if (index->neighbor_offsets_.back() != header.adjacency_bytes) {
      throw std::runtime_error("PForDelta offsets do not cover the compressed payload");
    }
    std::vector<uint64_t> edge_offsets;
    index->adjacency_ = decompress_adjacency(index->neighbor_offsets_, adjacency_section,
                                             header.edge_count, header.max_degree, edge_offsets);
    index->neighbor_offsets_ = std::move(edge_offsets);
    if (std::any_of(index->adjacency_.begin(), index->adjacency_.end(),
                    [count = header.point_count](uint32_t target) { return target >= count; })) {
      throw std::runtime_error("PForDelta adjacency target is out of range");
    }
#else
    throw std::runtime_error("PForDelta topology is unavailable in this build");
#endif
  }
  if (paged_topology) {
    index->topology_file_descriptor_ = ::open(topology_path.c_str(), O_RDONLY);
    if (index->topology_file_descriptor_ < 0) {
      throw std::runtime_error("failed opening paged topology artifact: " +
                               std::string(std::strerror(errno)));
    }
#if defined(POSIX_FADV_RANDOM)
    static_cast<void>(::posix_fadvise(index->topology_file_descriptor_, 0, 0, POSIX_FADV_RANDOM));
#endif
    index->paged_topology_ = true;
    index->adjacency_file_offset_ = header.adjacency_offset;
    index->topology_cache_ = std::make_shared<io_lru_buffer_pool_t>(topology_cache_pages);
    std::vector<uint32_t>().swap(index->adjacency_);
  }
  const uint64_t vector_bytes =
      header.layout == io_vector_layout_t::CELL_U8_4K
          ? header.dimension
          : checked_multiply(header.dimension, sizeof(float), "vector size");
  cursor = 0;
  for (uint64_t node = 0; node < header.point_count; ++node) {
    io_vector_location_t location;
    location.page_id = read_little<uint64_t>(directory_section, cursor, "vector page ID");
    location.page_offset = read_little<uint32_t>(directory_section, cursor, "vector page offset");
    if (location.page_id >= header.vector_page_count ||
        location.page_offset % alignof(float) != 0 ||
        location.page_offset > k_io_vector_page_size ||
        vector_bytes > k_io_vector_page_size - location.page_offset) {
      throw std::runtime_error("IO vector directory entry is out of range");
    }
    index->vector_locations_.push_back(location);
  }
  cursor = 0;
  uint64_t expected_page = 0;
  uint64_t covered_nodes = 0;
  for (uint64_t cell = 0; cell < header.cell_count; ++cell) {
    io_cell_page_range_t range;
    range.first_page = read_little<uint64_t>(cell_directory_section, cursor, "Cell first page");
    range.page_count = read_little<uint32_t>(cell_directory_section, cursor, "Cell page count");
    range.node_count = read_little<uint32_t>(cell_directory_section, cursor, "Cell node count");
    range.chunk_size_bytes =
        read_little<uint32_t>(cell_directory_section, cursor, "Cell Chunk size");
    if (range.first_page != expected_page || range.first_page >= header.vector_page_count ||
        range.page_count == 0 || range.node_count == 0 ||
        range.page_count > header.vector_page_count - range.first_page ||
        (range.chunk_size_bytes != 4096 && range.chunk_size_bytes != 65536 &&
         range.chunk_size_bytes != 524288 && range.chunk_size_bytes != 2097152) ||
        (header.layout != io_vector_layout_t::CELL_CHUNKS &&
         range.chunk_size_bytes != k_io_vector_page_size) ||
        (header.layout == io_vector_layout_t::CELL_CHUNKS &&
         range.page_count % (range.chunk_size_bytes / k_io_vector_page_size) != 0)) {
      throw std::runtime_error("IO Cell page directory is invalid");
    }
    expected_page = checked_add(range.first_page, range.page_count, "Cell page range");
    covered_nodes = checked_add(covered_nodes, range.node_count, "Cell node coverage");
    index->cell_page_ranges_.push_back(range);
  }
  if ((header.layout == io_vector_layout_t::CELL_4K ||
       header.layout == io_vector_layout_t::CELL_CHUNKS ||
       header.layout == io_vector_layout_t::CELL_WEIGHTED_4K ||
       header.layout == io_vector_layout_t::CELL_U8_4K) &&
      (expected_page != header.vector_page_count || covered_nodes != header.point_count)) {
    throw std::runtime_error("IO Cell page directory does not cover the vector layout");
  }
  cursor = 0;
  for (uint64_t node = 0; node < header.point_count; ++node) {
    const uint64_t page =
        read_little<uint64_t>(prefetch_directory_section, cursor, "node prefetch page");
    if (page != k_no_io_prefetch_page && page >= header.vector_page_count) {
      throw std::runtime_error("IO node prefetch page is out of range");
    }
    index->node_prefetch_pages_.push_back(page);
  }
  cursor = 0;
  for (uint64_t gateway = 0; gateway < header.gateway_count; ++gateway) {
    io_gateway_prefetch_hint_t hint;
    hint.target_node =
        read_little<uint32_t>(gateway_directory_section, cursor, "Gateway target node");
    hint.landing_cell =
        read_little<uint32_t>(gateway_directory_section, cursor, "Gateway landing Cell");
    for (auto& cell : hint.successor_cells) {
      cell = read_little<uint32_t>(gateway_directory_section, cursor, "successor Cell");
    }
    hint.landing_node_page =
        read_little<uint64_t>(gateway_directory_section, cursor, "landing-node page");
    hint.landing_cell_first_page =
        read_little<uint64_t>(gateway_directory_section, cursor, "landing-Cell first page");
    for (auto& page : hint.successor_first_pages) {
      page = read_little<uint64_t>(gateway_directory_section, cursor, "successor first page");
    }
    for (auto& count : hint.transition_counts) {
      count = read_little<uint32_t>(gateway_directory_section, cursor, "transition count");
    }
    hint.outgoing_transition_count =
        read_little<uint32_t>(gateway_directory_section, cursor, "outgoing transition count");
    hint.reserved = read_little<uint32_t>(gateway_directory_section, cursor, "reserved field");
    if (hint.target_node >= header.point_count || hint.landing_cell >= header.cell_count ||
        hint.landing_node_page >= header.vector_page_count ||
        hint.landing_cell_first_page >= header.vector_page_count || hint.reserved != 0) {
      throw std::runtime_error("IO Gateway prefetch hint is out of range");
    }
    const auto& landing_range = index->cell_page_ranges_[hint.landing_cell];
    if (hint.landing_node_page != index->vector_locations_[hint.target_node].page_id ||
        hint.landing_cell_first_page != landing_range.first_page ||
        hint.landing_node_page < landing_range.first_page ||
        hint.landing_node_page >= landing_range.first_page + landing_range.page_count ||
        hint.outgoing_transition_count < hint.transition_counts[0] ||
        hint.outgoing_transition_count < hint.transition_counts[1]) {
      throw std::runtime_error("IO Gateway landing-page hint is inconsistent");
    }
    for (size_t successor = 0; successor < hint.successor_cells.size(); ++successor) {
      const bool absent = hint.successor_cells[successor] == UINT32_MAX;
      if (absent != (hint.successor_first_pages[successor] == k_no_io_prefetch_page) ||
          (!absent && (hint.successor_cells[successor] >= header.cell_count ||
                       hint.successor_first_pages[successor] >= header.vector_page_count ||
                       hint.successor_first_pages[successor] !=
                           index->cell_page_ranges_[hint.successor_cells[successor]].first_page ||
                       hint.transition_counts[successor] == 0))) {
        throw std::runtime_error("IO Gateway successor hint is invalid");
      }
    }
    index->gateway_prefetch_hints_.push_back(hint);
  }
  return index;
}

io_optimized_index_t::~io_optimized_index_t() {
  if (topology_file_descriptor_ >= 0) {
    ::close(topology_file_descriptor_);
  }
}

std::span<const uint32_t> io_optimized_index_t::neighbors(uint32_t node_id) const {
  if (node_id >= point_count_) {
    throw std::out_of_range("IO topology node ID is out of range");
  }
  const auto begin = neighbor_offsets_[node_id];
  const auto end = neighbor_offsets_[node_id + 1U];
  if (paged_topology_) {
    thread_local std::vector<uint32_t> scratch;
    const size_t degree = static_cast<size_t>(end - begin);
    scratch.resize(degree);
    if (degree == 0) {
      return {};
    }
    const uint64_t first_byte = adjacency_file_offset_ + begin * sizeof(uint32_t);
    const uint64_t byte_count = degree * sizeof(uint32_t);
    uint64_t copied = 0;
    while (copied < byte_count) {
      const uint64_t absolute = first_byte + copied;
      const uint64_t page_id = absolute / k_io_vector_page_size;
      const size_t page_offset = static_cast<size_t>(absolute % k_io_vector_page_size);
      const size_t count = static_cast<size_t>(
          std::min<uint64_t>(byte_count - copied, k_io_vector_page_size - page_offset));
      auto reservation = topology_cache_->reserve(page_id);
      if (reservation.state == io_cache_reservation_state_t::BYPASS) {
        throw std::runtime_error("paged topology cache has no available frame");
      }
      if (reservation.state == io_cache_reservation_state_t::LOAD) {
        size_t loaded = 0;
        while (loaded < k_io_vector_page_size) {
          const ssize_t result = ::pread(
              topology_file_descriptor_, reservation.data + loaded, k_io_vector_page_size - loaded,
              static_cast<off_t>(page_id * k_io_vector_page_size + loaded));
          if (result < 0 && errno == EINTR) {
            continue;
          }
          if (result <= 0) {
            topology_cache_->publish(reservation, false);
            throw std::runtime_error("paged topology read failed or was truncated");
          }
          loaded += static_cast<size_t>(result);
        }
        topology_cache_->publish(reservation, true);
      }
      std::memcpy(reinterpret_cast<uint8_t*>(scratch.data()) + copied,
                  reservation.data + page_offset, count);
      topology_cache_->release(reservation);
      copied += count;
    }
    return std::span<const uint32_t>(scratch.data(), scratch.size());
  }
  return std::span<const uint32_t>(adjacency_.data() + begin, static_cast<size_t>(end - begin));
}

io_vector_location_t io_optimized_index_t::vector_location(uint32_t node_id) const {
  if (node_id >= point_count_) {
    throw std::out_of_range("IO vector node ID is out of range");
  }
  return vector_locations_[node_id];
}

uint64_t io_optimized_index_t::node_prefetch_page(uint32_t node_id) const {
  if (node_id >= point_count_) {
    throw std::out_of_range("IO prefetch node ID is out of range");
  }
  return node_prefetch_pages_[node_id];
}

const io_gateway_prefetch_hint_t&
io_optimized_index_t::gateway_prefetch_hint(uint64_t gateway_id) const {
  if (gateway_id >= gateway_prefetch_hints_.size()) {
    throw std::out_of_range("IO Gateway prefetch ID is out of range");
  }
  return gateway_prefetch_hints_[gateway_id];
}

uint64_t io_optimized_index_t::resident_bytes() const noexcept {
  return neighbor_offsets_.capacity() * sizeof(uint64_t) +
         adjacency_.capacity() * sizeof(uint32_t) +
         vector_locations_.capacity() * sizeof(io_vector_location_t) +
         cell_page_ranges_.capacity() * sizeof(io_cell_page_range_t) +
         node_prefetch_pages_.capacity() * sizeof(uint64_t) +
         gateway_prefetch_hints_.capacity() * sizeof(io_gateway_prefetch_hint_t) +
         topology_cache_resident_bytes();
}

uint64_t io_optimized_index_t::topology_cache_resident_bytes() const noexcept {
  return topology_cache_ == nullptr ? 0 : topology_cache_->resident_bytes();
}

io_lru_buffer_pool_stats_t io_optimized_index_t::topology_cache_stats() const {
  return topology_cache_ == nullptr ? io_lru_buffer_pool_stats_t{} : topology_cache_->stats();
}

io_cell_adjacency_build_result_t io_cell_adjacency_index_t::build(
    const std::filesystem::path& path, const io_optimized_index_t& topology,
    const community_polar_index_t& cells, io_artifact_fingerprint_t community_polar_fingerprint,
    uint32_t sampled_nodes_per_cell, uint32_t maximum_degree, uint32_t num_threads) {
  validate_output_path(path);
  if (sampled_nodes_per_cell == 0 || maximum_degree == 0 || cells.point_count == 0 ||
      cells.cells.empty() || topology.point_count() != cells.point_count ||
      cells.node_to_cell.size() != cells.point_count ||
      cells.cells.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("Cell adjacency build requires compatible nonempty inputs");
  }

  const uint32_t build_threads =
      num_threads == 0 ? static_cast<uint32_t>(omp_get_max_threads()) : num_threads;
  if (build_threads == 0) {
    throw std::invalid_argument("Cell adjacency build requires positive thread count");
  }
  const uint64_t fixed_slots =
      checked_multiply(cells.cells.size(), maximum_degree, "Cell adjacency fixed-degree scratch");
  if (fixed_slots > std::numeric_limits<size_t>::max()) {
    throw std::overflow_error("Cell adjacency fixed-degree scratch exceeds size_t");
  }
  std::vector<uint32_t> fixed_adjacency(static_cast<size_t>(fixed_slots), UINT32_MAX);
  std::vector<uint32_t> degrees(cells.cells.size(), 0);
  std::atomic<bool> invalid_packet = false;
  const uint64_t packet_record_bytes = sizeof(uint32_t) + cells.pq_code_width;

#pragma omp parallel num_threads(build_threads)
  {
    std::unordered_map<uint32_t, uint32_t> support;
    std::vector<std::pair<uint32_t, uint32_t>> ranked;
    support.reserve(static_cast<size_t>(sampled_nodes_per_cell) * topology.max_degree());
    ranked.reserve(static_cast<size_t>(sampled_nodes_per_cell) * topology.max_degree());
#pragma omp for schedule(static)
    for (int64_t signed_cell_id = 0; signed_cell_id < static_cast<int64_t>(cells.cells.size());
         ++signed_cell_id) {
      const uint32_t cell_id = static_cast<uint32_t>(signed_cell_id);
      const auto& cell = cells.cells[cell_id];
      support.clear();
      ranked.clear();
      const uint32_t sample_count =
          static_cast<uint32_t>(std::min<uint64_t>(sampled_nodes_per_cell, cell.node_count));
      for (uint32_t sample = 0; sample < sample_count; ++sample) {
        uint64_t ordinal = static_cast<uint64_t>(sample) * cell.node_count / sample_count;
        uint64_t packet = std::numeric_limits<uint64_t>::max();
        for (uint64_t block_id = cell.block_begin; block_id < cell.block_begin + cell.block_count;
             ++block_id) {
          const auto& block = cells.blocks[block_id];
          if (ordinal < block.node_count) {
            packet = block.packet_record_begin + ordinal;
            break;
          }
          ordinal -= block.node_count;
        }
        if (packet == std::numeric_limits<uint64_t>::max() ||
            packet > std::numeric_limits<size_t>::max() / packet_record_bytes) {
          invalid_packet = true;
          continue;
        }
        const size_t byte_offset = static_cast<size_t>(packet * packet_record_bytes);
        if (byte_offset > cells.packet_payload.size() ||
            cells.packet_payload.size() - byte_offset < sizeof(uint32_t)) {
          invalid_packet = true;
          continue;
        }
        uint32_t source_node = 0;
        std::memcpy(&source_node, cells.packet_payload.data() + byte_offset, sizeof(source_node));
        if (source_node >= cells.point_count || cells.node_to_cell[source_node] != cell_id) {
          invalid_packet = true;
          continue;
        }
        for (const uint32_t neighbor : topology.neighbors(source_node)) {
          const uint32_t target_cell = cells.node_to_cell[neighbor];
          if (target_cell == cell_id) {
            continue;
          }
          auto& count = support[target_cell];
          if (count != std::numeric_limits<uint32_t>::max()) {
            ++count;
          }
        }
      }
      ranked.reserve(support.size());
      for (const auto& [target_cell, count] : support) {
        ranked.emplace_back(std::numeric_limits<uint32_t>::max() - count, target_cell);
      }
      std::sort(ranked.begin(), ranked.end());
      const uint32_t retained =
          static_cast<uint32_t>(std::min<size_t>(maximum_degree, ranked.size()));
      degrees[cell_id] = retained;
      const size_t destination = static_cast<size_t>(cell_id) * maximum_degree;
      for (uint32_t rank = 0; rank < retained; ++rank) {
        fixed_adjacency[destination + rank] = ranked[rank].second;
      }
    }
  }
  if (invalid_packet) {
    throw std::runtime_error("Cell adjacency sampled packet is invalid");
  }

  std::vector<uint64_t> offsets(cells.cells.size() + 1U, 0);
  for (size_t cell_id = 0; cell_id < cells.cells.size(); ++cell_id) {
    offsets[cell_id + 1U] = offsets[cell_id] + degrees[cell_id];
  }
  std::vector<uint32_t> adjacency(static_cast<size_t>(offsets.back()));
#pragma omp parallel for schedule(static) num_threads(build_threads)
  for (int64_t signed_cell_id = 0; signed_cell_id < static_cast<int64_t>(cells.cells.size());
       ++signed_cell_id) {
    const size_t cell_id = static_cast<size_t>(signed_cell_id);
    std::copy_n(fixed_adjacency.begin() + static_cast<std::ptrdiff_t>(cell_id * maximum_degree),
                degrees[cell_id],
                adjacency.begin() + static_cast<std::ptrdiff_t>(offsets[cell_id]));
  }

  const auto offset_bytes = serialize_u64(offsets);
  const auto adjacency_bytes = serialize_u32(adjacency);
  const uint64_t offsets_offset = k_cell_adjacency_header_bytes;
  const uint64_t adjacency_offset =
      checked_add(offsets_offset, offset_bytes.size(), "Cell adjacency section offset");
  std::vector<uint8_t> header;
  append_little(header, k_cell_adjacency_magic);
  append_little(header, k_cell_adjacency_version);
  append_little(header, k_cell_adjacency_header_bytes);
  append_little(header, k_little_endian_marker);
  append_little(header, uint32_t{0});
  append_little(header, cells.point_count);
  append_little(header, static_cast<uint64_t>(cells.cells.size()));
  append_little(header, static_cast<uint64_t>(adjacency.size()));
  append_little(header, maximum_degree);
  append_little(header, sampled_nodes_per_cell);
  append_little(header, community_polar_fingerprint.size);
  append_little(header, community_polar_fingerprint.checksum);
  append_little(header, offsets_offset);
  append_little(header, static_cast<uint64_t>(offset_bytes.size()));
  append_little(header, checksum_bytes(offset_bytes));
  append_little(header, adjacency_offset);
  append_little(header, static_cast<uint64_t>(adjacency_bytes.size()));
  append_little(header, checksum_bytes(adjacency_bytes));
  header.resize(k_cell_adjacency_header_bytes, 0);

  const auto temporary = std::filesystem::path(path.string() + ".tmp");
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(header.data()), header.size());
  output.write(reinterpret_cast<const char*>(offset_bytes.data()),
               static_cast<std::streamsize>(offset_bytes.size()));
  output.write(reinterpret_cast<const char*>(adjacency_bytes.data()),
               static_cast<std::streamsize>(adjacency_bytes.size()));
  output.close();
  if (!output) {
    throw std::runtime_error("failed writing Cell adjacency artifact");
  }
  publish_file(temporary, path);

  io_cell_adjacency_build_result_t result;
  result.cell_count = cells.cells.size();
  result.edge_count = adjacency.size();
  result.artifact_bytes = std::filesystem::file_size(path);
  result.maximum_degree = maximum_degree;
  result.sampled_nodes_per_cell = sampled_nodes_per_cell;
  result.build_threads = build_threads;
  return result;
}

std::shared_ptr<const io_cell_adjacency_index_t>
io_cell_adjacency_index_t::load(const std::filesystem::path& path, uint64_t expected_point_count,
                                uint64_t expected_cell_count,
                                io_artifact_fingerprint_t community_polar_fingerprint) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("Cell adjacency artifact is not a regular file: " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  std::vector<uint8_t> header(k_cell_adjacency_header_bytes);
  input.read(reinterpret_cast<char*>(header.data()), header.size());
  if (!input) {
    throw std::runtime_error("Cell adjacency header is truncated");
  }
  size_t cursor = 0;
  if (read_little<uint64_t>(header, cursor, "Cell adjacency magic") != k_cell_adjacency_magic ||
      read_little<uint32_t>(header, cursor, "Cell adjacency version") != k_cell_adjacency_version ||
      read_little<uint32_t>(header, cursor, "Cell adjacency header size") !=
          k_cell_adjacency_header_bytes ||
      read_little<uint32_t>(header, cursor, "Cell adjacency byte order") !=
          k_little_endian_marker) {
    throw std::runtime_error("Cell adjacency header is incompatible");
  }
  static_cast<void>(read_little<uint32_t>(header, cursor, "Cell adjacency reserved field"));
  const uint64_t point_count = read_little<uint64_t>(header, cursor, "Cell adjacency points");
  const uint64_t cell_count = read_little<uint64_t>(header, cursor, "Cell adjacency Cells");
  const uint64_t edge_count = read_little<uint64_t>(header, cursor, "Cell adjacency edges");
  const uint32_t maximum_degree =
      read_little<uint32_t>(header, cursor, "Cell adjacency maximum degree");
  const uint32_t sampled_nodes =
      read_little<uint32_t>(header, cursor, "Cell adjacency sampled nodes");
  io_artifact_fingerprint_t persisted_fingerprint;
  persisted_fingerprint.size = read_little<uint64_t>(header, cursor, "Cell adjacency sidecar size");
  persisted_fingerprint.checksum =
      read_little<uint64_t>(header, cursor, "Cell adjacency sidecar checksum");
  const uint64_t offsets_offset =
      read_little<uint64_t>(header, cursor, "Cell adjacency offsets offset");
  const uint64_t offsets_bytes =
      read_little<uint64_t>(header, cursor, "Cell adjacency offsets bytes");
  const uint64_t offsets_checksum =
      read_little<uint64_t>(header, cursor, "Cell adjacency offsets checksum");
  const uint64_t adjacency_offset =
      read_little<uint64_t>(header, cursor, "Cell adjacency payload offset");
  const uint64_t adjacency_bytes =
      read_little<uint64_t>(header, cursor, "Cell adjacency payload bytes");
  const uint64_t adjacency_checksum =
      read_little<uint64_t>(header, cursor, "Cell adjacency payload checksum");
  const uint64_t expected_offsets_bytes =
      checked_multiply(cell_count + 1U, sizeof(uint64_t), "Cell adjacency offsets bytes");
  const uint64_t expected_adjacency_bytes =
      checked_multiply(edge_count, sizeof(uint32_t), "Cell adjacency payload bytes");
  const uint64_t file_size = std::filesystem::file_size(path);
  const uint64_t expected_adjacency_offset =
      checked_add(offsets_offset, offsets_bytes, "Cell adjacency payload offset");
  const uint64_t expected_file_size =
      checked_add(adjacency_offset, adjacency_bytes, "Cell adjacency file size");
  if (point_count != expected_point_count || cell_count != expected_cell_count ||
      persisted_fingerprint != community_polar_fingerprint || maximum_degree == 0 ||
      sampled_nodes == 0 || offsets_offset != k_cell_adjacency_header_bytes ||
      offsets_bytes != expected_offsets_bytes || adjacency_offset != expected_adjacency_offset ||
      adjacency_bytes != expected_adjacency_bytes || expected_file_size != file_size ||
      cell_count > std::numeric_limits<size_t>::max() - 1U ||
      edge_count > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Cell adjacency metadata is incompatible");
  }
  std::vector<uint8_t> offset_section(static_cast<size_t>(offsets_bytes));
  std::vector<uint8_t> adjacency_section(static_cast<size_t>(adjacency_bytes));
  input.read(reinterpret_cast<char*>(offset_section.data()), offset_section.size());
  input.read(reinterpret_cast<char*>(adjacency_section.data()), adjacency_section.size());
  if (!input || checksum_bytes(offset_section) != offsets_checksum ||
      checksum_bytes(adjacency_section) != adjacency_checksum) {
    throw std::runtime_error("Cell adjacency section checksum is invalid");
  }

  auto index = std::shared_ptr<io_cell_adjacency_index_t>(new io_cell_adjacency_index_t());
  index->point_count_ = point_count;
  index->cell_count_ = cell_count;
  index->maximum_degree_ = maximum_degree;
  index->sampled_nodes_per_cell_ = sampled_nodes;
  index->artifact_bytes_ = file_size;
  index->offsets_.reserve(static_cast<size_t>(cell_count + 1U));
  cursor = 0;
  while (cursor < offset_section.size()) {
    index->offsets_.push_back(
        read_little<uint64_t>(offset_section, cursor, "Cell adjacency offset"));
  }
  index->adjacency_.reserve(static_cast<size_t>(edge_count));
  cursor = 0;
  while (cursor < adjacency_section.size()) {
    index->adjacency_.push_back(
        read_little<uint32_t>(adjacency_section, cursor, "Cell adjacency target"));
  }
  if (index->offsets_.front() != 0 || index->offsets_.back() != edge_count) {
    throw std::runtime_error("Cell adjacency offset endpoints are invalid");
  }
  for (uint64_t cell = 0; cell < cell_count; ++cell) {
    const uint64_t begin = index->offsets_[cell];
    const uint64_t end = index->offsets_[cell + 1U];
    if (begin > end || end - begin > maximum_degree) {
      throw std::runtime_error("Cell adjacency degree is invalid");
    }
    for (uint64_t edge = begin; edge < end; ++edge) {
      if (index->adjacency_[edge] >= cell_count || index->adjacency_[edge] == cell) {
        throw std::runtime_error("Cell adjacency target is invalid");
      }
      if (std::find(index->adjacency_.begin() + static_cast<std::ptrdiff_t>(begin),
                    index->adjacency_.begin() + static_cast<std::ptrdiff_t>(edge),
                    index->adjacency_[edge]) !=
          index->adjacency_.begin() + static_cast<std::ptrdiff_t>(edge)) {
        throw std::runtime_error("Cell adjacency target is duplicated");
      }
    }
  }
  return index;
}

std::span<const uint32_t> io_cell_adjacency_index_t::neighbors(uint32_t cell_id) const {
  if (cell_id >= cell_count_) {
    throw std::out_of_range("Cell adjacency Cell ID is out of range");
  }
  return std::span<const uint32_t>(adjacency_)
      .subspan(static_cast<size_t>(offsets_[cell_id]),
               static_cast<size_t>(offsets_[cell_id + 1U] - offsets_[cell_id]));
}

uint64_t io_cell_adjacency_index_t::resident_bytes() const noexcept {
  return offsets_.capacity() * sizeof(uint64_t) + adjacency_.capacity() * sizeof(uint32_t);
}

} // namespace powerlaw_ann
