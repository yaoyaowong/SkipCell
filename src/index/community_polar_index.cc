#include "index/community_polar_index.h"

#include "common/defaults.h"
#include "common/utils.h"
#include "index/polar.h"
#include "pq/product_quantizer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <istream>
#include <kaminpar.h>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <queue>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace powerlaw_ann {
namespace {

constexpr uint64_t k_community_polar_magic = 0x0058444950434C50ULL;        // "PLCPI DX\0"
constexpr uint64_t k_cell_hierarchy_profile_magic = 0x0052454948434C50ULL; // "PLCHIER\0"
constexpr uint32_t k_cell_hierarchy_profile_version = 2;
constexpr uint32_t k_cell_hierarchy_profile_minimum_version = 1;
constexpr uint32_t k_community_polar_version = 7;
constexpr uint32_t k_adaptive_capacity_sidecar_version = 6;
constexpr uint32_t k_fixed_capacity_sidecar_version = 5;
constexpr uint32_t k_community_polar_minimum_version = 1;
constexpr uint32_t k_endian_marker = 0x01020304U;
constexpr uint32_t k_legacy_section_count = 14;
constexpr uint32_t k_section_count = 18;
constexpr uint32_t k_fixed_header_size = 96;
constexpr uint32_t k_section_entry_size = 32;
constexpr uint32_t k_pole_sector_id = std::numeric_limits<uint32_t>::max();
constexpr uint64_t k_invalid_axis_index = std::numeric_limits<uint64_t>::max();
constexpr uint64_t k_fnv_offset_basis = 14695981039346656037ULL;
constexpr uint64_t k_fnv_prime = 1099511628211ULL;
constexpr std::array<uint8_t, 8> k_pq_locality_magic = {'P', 'L', 'C', 'P', 'Q', 'L', 'O', 'C'};
constexpr uint32_t k_pq_locality_version = 1;
constexpr uint32_t k_pq_locality_header_size = 52;

uint32_t crc32_bytes(std::span<const uint8_t> bytes) {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> values{};
    for (uint32_t index = 0; index < values.size(); ++index) {
      uint32_t value = index;
      for (uint32_t bit = 0; bit < 8; ++bit) {
        value = (value >> 1U) ^ ((value & 1U) != 0 ? 0xEDB88320U : 0U);
      }
      values[index] = value;
    }
    return values;
  }();
  uint32_t checksum = 0xFFFFFFFFU;
  for (const uint8_t value : bytes) {
    checksum = table[(checksum ^ value) & 0xFFU] ^ (checksum >> 8U);
  }
  return checksum ^ 0xFFFFFFFFU;
}

enum class section_id_t : uint32_t {
  CONFIG = 1,
  NODE_TO_COMMUNITY = 2,
  NODE_TO_CELL = 3,
  NODE_TO_PACKET = 4,
  COMMUNITIES = 5,
  COMMUNITY_POLES = 6,
  COMMUNITY_EDGE_OFFSETS = 7,
  COMMUNITY_EDGES = 8,
  GATEWAYS = 9,
  DIRECTION_AXES = 10,
  SECTORS = 11,
  CELLS = 12,
  BLOCKS = 13,
  PACKETS = 14,
  CELL_HIERARCHY_NODES = 15,
  CELL_HIERARCHY_ROOTS = 16,
  CELL_HIERARCHY_CHILDREN = 17,
  CELL_HIERARCHY_CENTROIDS = 18,
};

struct serialized_section_t {
  section_id_t id;
  uint64_t record_count = 0;
  std::vector<uint8_t> bytes;
};

struct section_directory_entry_t {
  section_id_t id;
  uint64_t offset = 0;
  uint64_t length = 0;
  uint64_t record_count = 0;
};

struct partition_graph_t {
  std::vector<kaminpar::shm::EdgeID> offsets;
  std::vector<kaminpar::shm::NodeID> neighbors;
  std::vector<kaminpar::shm::NodeWeight> node_weights;
  std::vector<kaminpar::shm::EdgeWeight> edge_weights;
  uint64_t directed_edge_count = 0;
  uint64_t undirected_edge_count = 0;
};

struct cross_arc_t {
  uint32_t source_community = 0;
  uint32_t target_community = 0;
  uint32_t source_node = 0;
  uint32_t target_node = 0;
};

struct gateway_candidate_t {
  uint32_t source_node = 0;
  uint32_t target_node = 0;
  double importance = 0.0;
  bool has_direction = false;
  std::vector<float> direction;
};

struct polar_node_t {
  uint32_t node_id = 0;
  float radius = 0.0F;
  float angle = 0.0F;
};

using gateway_selection_map_t = std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>;

struct convergence_cell_assignment_t {
  uint32_t community_id = 0;
  uint32_t cell_id = 0;
  uint32_t order = 0;
};

uint32_t read_u32_le(std::span<const uint8_t> bytes, size_t& cursor);
uint64_t read_u64_le(std::span<const uint8_t> bytes, size_t& cursor);
float read_float_le(std::span<const uint8_t> bytes, size_t& cursor);

bool is_adaptive_capacity_class(uint32_t capacity) {
  return capacity == 16 || capacity == 32 || capacity == 64 || capacity == 128;
}

size_t checked_size(uint64_t value, const char* context);

std::vector<std::string> split_csv_row(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

gateway_selection_map_t load_gateway_selection(const community_polar_build_config_t& config) {
  gateway_selection_map_t selections;
  if (config.gateway_selection_path.empty()) {
    return selections;
  }
  std::ifstream input(config.gateway_selection_path);
  if (!input) {
    throw std::invalid_argument("Failed to open Gateway selection CSV: " +
                                config.gateway_selection_path.string());
  }
  std::string line;
  if (!std::getline(input, line) ||
      line != "source_community,target_community,gateway_node,rank,utility,rcni") {
    throw std::invalid_argument("Gateway selection CSV has an unexpected header");
  }
  std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>> ranked;
  uint64_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    const auto fields = split_csv_row(line);
    if (fields.size() != 6) {
      throw std::invalid_argument("Gateway selection CSV row " + std::to_string(line_number) +
                                  " has an unexpected field count");
    }
    try {
      const uint32_t source = static_cast<uint32_t>(std::stoul(fields[0]));
      const uint32_t target = static_cast<uint32_t>(std::stoul(fields[1]));
      const uint32_t node = static_cast<uint32_t>(std::stoul(fields[2]));
      const uint32_t rank = static_cast<uint32_t>(std::stoul(fields[3]));
      ranked[{source, target}].emplace_back(rank, node);
    } catch (const std::exception& error) {
      throw std::invalid_argument("Gateway selection CSV row " + std::to_string(line_number) +
                                  " is invalid: " + error.what());
    }
  }
  for (auto& [edge, entries] : ranked) {
    std::sort(entries.begin(), entries.end());
    if (entries.size() > config.gateway_count) {
      throw std::invalid_argument("Gateway selection CSV exceeds the configured Gateway count");
    }
    auto& nodes = selections[edge];
    nodes.reserve(entries.size());
    for (size_t position = 0; position < entries.size(); ++position) {
      if (entries[position].first != position ||
          (position != 0 && entries[position - 1].second == entries[position].second)) {
        throw std::invalid_argument(
            "Gateway selection CSV ranks must be contiguous with unique nodes");
      }
      nodes.push_back(entries[position].second);
    }
  }
  return selections;
}

std::vector<convergence_cell_assignment_t>
load_convergence_cell_profile(const community_polar_build_config_t& config, uint64_t point_count) {
  if (config.cell_partition != community_cell_partition_t::CONVERGENCE_COACCESS &&
      config.cell_partition != community_cell_partition_t::GLOBAL_GEOMETRIC) {
    return {};
  }
  if (config.convergence_cell_profile_path.empty()) {
    throw std::invalid_argument("Convergence co-access Cells require a profile CSV");
  }
  std::ifstream input(config.convergence_cell_profile_path);
  if (!input) {
    throw std::invalid_argument("Failed to open convergence Cell profile CSV: " +
                                config.convergence_cell_profile_path.string());
  }
  std::string line;
  if (!std::getline(input, line)) {
    throw std::invalid_argument("Convergence Cell profile CSV is empty");
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  if (line != "node_id,community_id,cell_id,order") {
    throw std::invalid_argument("Convergence Cell profile CSV has an unexpected header");
  }
  std::vector<convergence_cell_assignment_t> assignments(
      checked_size(point_count, "Convergence Cell profile points"));
  std::vector<uint8_t> seen(assignments.size(), 0);
  uint64_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    if (line.back() == '\r') {
      line.pop_back();
    }
    const auto fields = split_csv_row(line);
    if (fields.size() != 4) {
      throw std::invalid_argument("Convergence Cell profile row " + std::to_string(line_number) +
                                  " has an unexpected field count");
    }
    try {
      const uint64_t parsed_node = std::stoull(fields[0]);
      if (parsed_node >= point_count || seen[parsed_node] != 0) {
        throw std::invalid_argument("node ID is out of range or repeated");
      }
      const uint32_t node = static_cast<uint32_t>(parsed_node);
      assignments[node] = {static_cast<uint32_t>(std::stoul(fields[1])),
                           static_cast<uint32_t>(std::stoul(fields[2])),
                           static_cast<uint32_t>(std::stoul(fields[3]))};
      seen[node] = 1;
    } catch (const std::exception& error) {
      throw std::invalid_argument("Convergence Cell profile row " + std::to_string(line_number) +
                                  " is invalid: " + error.what());
    }
  }
  if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
    throw std::invalid_argument("Convergence Cell profile does not cover every Base node");
  }
  return assignments;
}

void load_convergence_cell_capacities(const community_polar_build_config_t& config,
                                      community_polar_index_t& index) {
  if (!config.adaptive_multi_capacity) {
    return;
  }
  if (config.convergence_cell_capacity_path.empty()) {
    throw std::invalid_argument("Adaptive multi-capacity Cells require a capacity CSV");
  }
  std::ifstream input(config.convergence_cell_capacity_path);
  if (!input) {
    throw std::invalid_argument("Failed to open convergence Cell capacity CSV: " +
                                config.convergence_cell_capacity_path.string());
  }
  std::string line;
  if (!std::getline(input, line)) {
    throw std::invalid_argument("Convergence Cell capacity CSV is empty");
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  if (line != "cell_id,capacity_class,actual_population,tree_node_id,tree_depth") {
    throw std::invalid_argument("Convergence Cell capacity CSV has an unexpected header");
  }
  std::vector<uint8_t> seen(index.cells.size(), 0);
  std::vector<uint32_t> tree_nodes;
  uint64_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    if (line.back() == '\r') {
      line.pop_back();
    }
    const auto fields = split_csv_row(line);
    if (fields.size() != 5) {
      throw std::invalid_argument("Convergence Cell capacity row " + std::to_string(line_number) +
                                  " has an unexpected field count");
    }
    try {
      const uint64_t parsed_cell = std::stoull(fields[0]);
      const uint32_t capacity = static_cast<uint32_t>(std::stoul(fields[1]));
      const uint64_t population = std::stoull(fields[2]);
      const uint32_t tree_node = static_cast<uint32_t>(std::stoul(fields[3]));
      static_cast<void>(std::stoul(fields[4]));
      if (parsed_cell >= index.cells.size() || seen[parsed_cell] != 0 ||
          !is_adaptive_capacity_class(capacity) || population == 0 || population > capacity ||
          population != index.cells[parsed_cell].node_count) {
        throw std::invalid_argument("Cell ID, capacity class, or population is invalid");
      }
      index.cells[parsed_cell].capacity_class = capacity;
      seen[parsed_cell] = 1;
      tree_nodes.push_back(tree_node);
    } catch (const std::exception& error) {
      throw std::invalid_argument("Convergence Cell capacity row " + std::to_string(line_number) +
                                  " is invalid: " + error.what());
    }
  }
  std::sort(tree_nodes.begin(), tree_nodes.end());
  if (std::find(seen.begin(), seen.end(), 0) != seen.end() ||
      std::adjacent_find(tree_nodes.begin(), tree_nodes.end()) != tree_nodes.end()) {
    throw std::invalid_argument(
        "Convergence Cell capacity CSV must cover every Cell with unique tree nodes");
  }
}

uint64_t checked_add(uint64_t left, uint64_t right, const char* context) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    throw std::invalid_argument(std::string(context) + " overflows uint64");
  }
  return left + right;
}

uint64_t checked_multiply(uint64_t left, uint64_t right, const char* context) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    throw std::invalid_argument(std::string(context) + " overflows uint64");
  }
  return left * right;
}

size_t checked_size(uint64_t value, const char* context) {
  if (value > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(std::string(context) + " exceeds addressable memory");
  }
  return static_cast<size_t>(value);
}

uint64_t make_undirected_key(uint32_t left, uint32_t right) {
  const uint32_t minimum = std::min(left, right);
  const uint32_t maximum = std::max(left, right);
  return (static_cast<uint64_t>(minimum) << 32U) | maximum;
}

uint32_t key_left(uint64_t key) { return static_cast<uint32_t>(key >> 32U); }

uint32_t key_right(uint64_t key) { return static_cast<uint32_t>(key); }

uint32_t effective_thread_count(uint32_t requested) {
  if (requested != 0) {
    return requested;
  }
  return std::max(1U, std::thread::hardware_concurrency());
}

class disk_post_link_graph_view_t final : public post_link_graph_view_t {
public:
  disk_post_link_graph_view_t(const std::filesystem::path& data_path,
                              const std::filesystem::path& disk_index_path) {
    size_t point_count = 0;
    size_t dimension = 0;
    load_bin<float>(data_path.string(), vectors_, point_count, dimension);
    if (point_count == 0 || point_count > std::numeric_limits<uint32_t>::max() || dimension == 0 ||
        dimension > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("sidecar-only Base data shape is invalid");
    }
    point_count_ = static_cast<uint32_t>(point_count);
    dimension_ = static_cast<uint32_t>(dimension);
    load_graph(disk_index_path);
  }

  uint64_t point_count() const noexcept override { return point_count_; }

  uint32_t dimension() const noexcept override { return dimension_; }

  uint32_t entry_point_id() const noexcept override { return entry_point_id_; }

  std::span<const uint32_t> neighbors(uint32_t node_id) const override {
    if (node_id >= point_count_) {
      throw std::out_of_range("sidecar-only graph node ID is out of range");
    }
    return std::span<const uint32_t>(neighbors_)
        .subspan(
            checked_size(offsets_[node_id], "sidecar-only graph offset"),
            checked_size(offsets_[node_id + 1] - offsets_[node_id], "sidecar-only graph degree"));
  }

  void copy_vector(uint32_t node_id, std::span<float> destination) const override {
    if (node_id >= point_count_ || destination.size() != dimension_) {
      throw std::invalid_argument("sidecar-only graph vector request is invalid");
    }
    std::copy_n(vectors_.get() + static_cast<size_t>(node_id) * dimension_, dimension_,
                destination.begin());
  }

  float squared_distance(uint32_t left, uint32_t right) const override {
    if (left >= point_count_ || right >= point_count_) {
      throw std::out_of_range("sidecar-only graph distance node ID is out of range");
    }
    const float* left_vector = vectors_.get() + static_cast<size_t>(left) * dimension_;
    const float* right_vector = vectors_.get() + static_cast<size_t>(right) * dimension_;
    double distance = 0.0;
    for (uint32_t dimension = 0; dimension < dimension_; ++dimension) {
      const double difference =
          static_cast<double>(left_vector[dimension]) - right_vector[dimension];
      distance += difference * difference;
    }
    return static_cast<float>(distance);
  }

private:
  template <typename value_t>
  static void read_value(std::istream& input, value_t& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
      throw std::runtime_error("sidecar-only disk index metadata is truncated");
    }
  }

  void append_node(const char* record, uint64_t disk_bytes_per_point, uint64_t max_degree) {
    uint32_t degree = 0;
    std::memcpy(&degree, record + disk_bytes_per_point, sizeof(degree));
    if (degree == 0 || degree > max_degree) {
      throw std::runtime_error("sidecar-only disk index node degree is invalid");
    }
    const char* neighbor_bytes = record + disk_bytes_per_point + sizeof(uint32_t);
    for (uint32_t offset = 0; offset < degree; ++offset) {
      uint32_t neighbor = 0;
      std::memcpy(&neighbor, neighbor_bytes + static_cast<size_t>(offset) * sizeof(uint32_t),
                  sizeof(neighbor));
      if (neighbor >= point_count_) {
        throw std::runtime_error("sidecar-only disk index neighbor ID is invalid");
      }
      neighbors_.push_back(neighbor);
    }
    offsets_.push_back(neighbors_.size());
  }

  void load_graph(const std::filesystem::path& disk_index_path) {
    std::ifstream input(disk_index_path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("failed to open sidecar-only disk index: " +
                               disk_index_path.string());
    }
    uint32_t metadata_rows = 0;
    uint32_t metadata_columns = 0;
    uint64_t disk_point_count = 0;
    uint64_t disk_dimensions = 0;
    uint64_t entry_point = 0;
    uint64_t max_node_len = 0;
    uint64_t nodes_per_sector = 0;
    uint64_t frozen_points = 0;
    uint64_t frozen_location = 0;
    uint64_t reorder_exists = 0;
    read_value(input, metadata_rows);
    read_value(input, metadata_columns);
    read_value(input, disk_point_count);
    read_value(input, disk_dimensions);
    read_value(input, entry_point);
    read_value(input, max_node_len);
    read_value(input, nodes_per_sector);
    read_value(input, frozen_points);
    read_value(input, frozen_location);
    read_value(input, reorder_exists);
    static_cast<void>(frozen_location);
    static_cast<void>(reorder_exists);
    if (metadata_rows < 9 || metadata_columns != 1 || disk_point_count != point_count_ ||
        entry_point >= point_count_ || frozen_points > 1 || max_node_len == 0) {
      throw std::runtime_error("sidecar-only disk index metadata is incompatible");
    }
    entry_point_id_ = static_cast<uint32_t>(entry_point);

    const auto disk_pq_path = std::filesystem::path(disk_index_path.string() + "_pq_pivots.bin");
    const uint64_t disk_bytes_per_point =
        std::filesystem::is_regular_file(disk_pq_path)
            ? disk_dimensions
            : checked_multiply(disk_dimensions, sizeof(float), "disk vector bytes");
    if ((!std::filesystem::is_regular_file(disk_pq_path) && disk_dimensions != dimension_) ||
        max_node_len <= disk_bytes_per_point + sizeof(uint32_t) ||
        max_node_len > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("sidecar-only disk index record layout is incompatible");
    }
    const uint64_t max_degree =
        (max_node_len - disk_bytes_per_point - sizeof(uint32_t)) / sizeof(uint32_t);
    if (max_degree == 0 || max_degree > defaults::MAX_GRAPH_DEGREE ||
        (nodes_per_sector != 0 && checked_multiply(nodes_per_sector, max_node_len,
                                                   "nodes per sector") > defaults::SECTOR_LEN)) {
      throw std::runtime_error("sidecar-only disk index graph width is invalid");
    }

    offsets_.reserve(static_cast<size_t>(point_count_) + 1);
    offsets_.push_back(0);
    neighbors_.reserve(
        checked_size(checked_multiply(point_count_, max_degree, "sidecar-only graph reserve"),
                     "sidecar-only graph reserve"));
    input.seekg(defaults::SECTOR_LEN, std::ios::beg);
    if (!input) {
      throw std::runtime_error("sidecar-only disk index metadata sector is truncated");
    }
    uint32_t node = 0;
    if (nodes_per_sector != 0) {
      std::vector<char> sector(defaults::SECTOR_LEN);
      while (node < point_count_) {
        input.read(sector.data(), sector.size());
        if (!input) {
          throw std::runtime_error("sidecar-only disk index graph is truncated");
        }
        for (uint64_t slot = 0; slot < nodes_per_sector && node < point_count_; ++slot, ++node) {
          append_node(sector.data() + checked_size(slot * max_node_len, "disk node offset"),
                      disk_bytes_per_point, max_degree);
        }
      }
    } else {
      const uint64_t sectors_per_node =
          (max_node_len + defaults::SECTOR_LEN - 1) / defaults::SECTOR_LEN;
      std::vector<char> record(checked_size(
          checked_multiply(sectors_per_node, defaults::SECTOR_LEN, "disk node sectors"),
          "disk node sectors"));
      while (node < point_count_) {
        input.read(record.data(), record.size());
        if (!input) {
          throw std::runtime_error("sidecar-only disk index graph is truncated");
        }
        append_node(record.data(), disk_bytes_per_point, max_degree);
        ++node;
      }
    }
    if (offsets_.size() != static_cast<size_t>(point_count_) + 1) {
      throw std::logic_error("sidecar-only graph did not load every node");
    }
  }

  uint32_t point_count_ = 0;
  uint32_t dimension_ = 0;
  uint32_t entry_point_id_ = 0;
  std::unique_ptr<float[]> vectors_;
  std::vector<uint64_t> offsets_;
  std::vector<uint32_t> neighbors_;
};

std::vector<rcni_node_importance_t> read_rcni_importance_csv(const std::filesystem::path& path,
                                                             uint64_t point_count) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open sidecar-only RCNI input: " + path.string());
  }
  std::string line;
  if (!std::getline(input, line) ||
      line != "node_id,raw_importance,normalized_importance,importance_percentile,source_support,"
              "witness_event_count") {
    throw std::runtime_error("sidecar-only RCNI header is invalid");
  }
  std::vector<rcni_node_importance_t> importance;
  importance.reserve(checked_size(point_count, "sidecar-only RCNI records"));
  while (std::getline(input, line)) {
    std::istringstream row(line);
    std::array<std::string, 6> fields;
    for (size_t field = 0; field < fields.size(); ++field) {
      if (!std::getline(row, fields[field], ',')) {
        throw std::runtime_error("sidecar-only RCNI row is truncated");
      }
    }
    if (row.peek() != std::char_traits<char>::eof()) {
      throw std::runtime_error("sidecar-only RCNI row has trailing fields");
    }
    rcni_node_importance_t record;
    try {
      record.node_id = static_cast<uint32_t>(std::stoul(fields[0]));
      record.raw_importance = std::stod(fields[1]);
      record.normalized_importance = std::stod(fields[2]);
      record.importance_percentile = std::stod(fields[3]);
      record.source_support = std::stoull(fields[4]);
      record.witness_event_count = std::stoull(fields[5]);
    } catch (const std::exception& error) {
      throw std::runtime_error("sidecar-only RCNI row is invalid: " + std::string(error.what()));
    }
    importance.push_back(record);
  }
  if (importance.size() != point_count) {
    throw std::runtime_error("sidecar-only RCNI record count does not match Base points");
  }
  return importance;
}

std::vector<uint32_t> unique_neighbors(const post_link_graph_view_t& graph, uint32_t node) {
  const auto borrowed = graph.neighbors(node);
  std::vector<uint32_t> neighbors(borrowed.begin(), borrowed.end());
  std::sort(neighbors.begin(), neighbors.end());
  if (std::adjacent_find(neighbors.begin(), neighbors.end()) != neighbors.end()) {
    throw std::invalid_argument("Base Vamana contains a duplicate directed edge");
  }
  for (const uint32_t target : neighbors) {
    if (target >= graph.point_count() || target == node) {
      throw std::invalid_argument("Base Vamana contains an invalid directed edge");
    }
  }
  return neighbors;
}

uint32_t projection_degree(community_partition_projection_t projection) {
  switch (projection) {
  case community_partition_projection_t::FULL_RECIPROCAL_VAMANA:
    return std::numeric_limits<uint32_t>::max();
  case community_partition_projection_t::LOCAL_VAMANA_K8:
    return 8;
  case community_partition_projection_t::LOCAL_VAMANA_K16:
    return 16;
  }
  throw std::invalid_argument("Community partition projection is invalid");
}

std::vector<uint32_t> projected_neighbors(const post_link_graph_view_t& graph, uint32_t node,
                                          community_partition_projection_t projection) {
  auto neighbors = unique_neighbors(graph, node);
  const uint32_t degree = projection_degree(projection);
  if (neighbors.size() <= degree) {
    return neighbors;
  }
  std::vector<std::pair<float, uint32_t>> ranked;
  ranked.reserve(neighbors.size());
  for (const uint32_t target : neighbors) {
    const float distance = graph.squared_distance(node, target);
    if (!std::isfinite(distance) || distance < 0.0F) {
      throw std::invalid_argument("Base Vamana projection distance is invalid");
    }
    ranked.emplace_back(distance, target);
  }
  std::sort(ranked.begin(), ranked.end());
  neighbors.clear();
  neighbors.reserve(degree);
  for (uint32_t offset = 0; offset < degree; ++offset) {
    neighbors.push_back(ranked[offset].second);
  }
  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

partition_graph_t build_partition_graph(const post_link_graph_view_t& graph,
                                        community_partition_projection_t projection,
                                        uint32_t thread_count) {
  if (graph.point_count() == 0 ||
      graph.point_count() > std::numeric_limits<kaminpar::shm::NodeID>::max() ||
      graph.point_count() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("Community partition point count is invalid");
  }

  if (projection != community_partition_projection_t::FULL_RECIPROCAL_VAMANA) {
    const uint32_t degree_limit = projection_degree(projection);
    const size_t point_count = static_cast<size_t>(graph.point_count());
    const uint64_t target_count =
        checked_multiply(point_count, degree_limit, "bounded projection targets");
    std::vector<uint32_t> projected_targets(
        checked_size(target_count, "bounded projection targets"));
    std::vector<uint16_t> projected_counts(point_count, 0);
    std::vector<uint16_t> base_degrees(point_count, 0);
    std::vector<uint8_t> invalid_nodes(point_count, 0);

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t signed_node = 0; signed_node < static_cast<int64_t>(point_count);
         ++signed_node) {
      const uint32_t node = static_cast<uint32_t>(signed_node);
      const auto neighbors = graph.neighbors(node);
      if (neighbors.empty() || neighbors.size() > defaults::MAX_GRAPH_DEGREE) {
        invalid_nodes[node] = 1;
        continue;
      }
      std::array<uint32_t, defaults::MAX_GRAPH_DEGREE> ordered_neighbors{};
      std::copy(neighbors.begin(), neighbors.end(), ordered_neighbors.begin());
      std::sort(ordered_neighbors.begin(), ordered_neighbors.begin() + neighbors.size());
      bool valid = true;
      for (size_t offset = 0; offset < neighbors.size(); ++offset) {
        const uint32_t target = ordered_neighbors[offset];
        if (target >= graph.point_count() || target == node ||
            (offset != 0 && target == ordered_neighbors[offset - 1])) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        invalid_nodes[node] = 1;
        continue;
      }
      base_degrees[node] = static_cast<uint16_t>(neighbors.size());
      const uint32_t count = std::min<uint32_t>(degree_limit, neighbors.size());
      projected_counts[node] = static_cast<uint16_t>(count);
      std::array<std::pair<float, uint32_t>, defaults::MAX_GRAPH_DEGREE> ranked{};
      for (size_t offset = 0; offset < neighbors.size(); ++offset) {
        const uint32_t target = ordered_neighbors[offset];
        const float distance = graph.squared_distance(node, target);
        if (!std::isfinite(distance) || distance < 0.0F) {
          valid = false;
          break;
        }
        ranked[offset] = {distance, target};
      }
      if (!valid) {
        invalid_nodes[node] = 2;
        projected_counts[node] = 0;
        continue;
      }
      std::partial_sort(ranked.begin(), ranked.begin() + count,
                        ranked.begin() + neighbors.size());
      auto output = projected_targets.begin() + static_cast<size_t>(node) * degree_limit;
      for (uint32_t offset = 0; offset < count; ++offset) {
        output[offset] = ranked[offset].second;
      }
      std::sort(output, output + count);
    }
    const auto invalid = std::find_if(invalid_nodes.begin(), invalid_nodes.end(),
                                      [](uint8_t value) { return value != 0; });
    if (invalid != invalid_nodes.end()) {
      if (*invalid == 2) {
        throw std::invalid_argument("Base Vamana projection distance is invalid");
      }
      throw std::invalid_argument("Base Vamana contains an invalid or duplicate directed edge");
    }

    uint64_t directed_base_edge_count = 0;
    for (const uint16_t degree : base_degrees) {
      directed_base_edge_count =
          checked_add(directed_base_edge_count, degree, "directed Base edges");
    }

    struct weighted_pair_t {
      uint32_t left;
      uint32_t right;
      kaminpar::shm::EdgeWeight weight;
    };
    std::vector<uint16_t> pair_counts(point_count, 0);
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t signed_node = 0; signed_node < static_cast<int64_t>(point_count);
         ++signed_node) {
      const uint32_t node = static_cast<uint32_t>(signed_node);
      const auto row = projected_targets.begin() + static_cast<size_t>(node) * degree_limit;
      uint16_t count = 0;
      for (uint32_t offset = 0; offset < projected_counts[node]; ++offset) {
        const uint32_t target = row[offset];
        const auto reverse =
            projected_targets.begin() + static_cast<size_t>(target) * degree_limit;
        const bool reciprocal =
            std::binary_search(reverse, reverse + projected_counts[target], node);
        if (node < target || !reciprocal) {
          ++count;
        }
      }
      pair_counts[node] = count;
    }
    std::vector<uint64_t> pair_offsets(point_count + 1, 0);
    for (size_t node = 0; node < point_count; ++node) {
      pair_offsets[node + 1] =
          checked_add(pair_offsets[node], pair_counts[node], "bounded projection pairs");
    }
    std::vector<weighted_pair_t> pairs(
        checked_size(pair_offsets.back(), "bounded projection pairs"));
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t signed_node = 0; signed_node < static_cast<int64_t>(point_count);
         ++signed_node) {
      const uint32_t node = static_cast<uint32_t>(signed_node);
      const auto row = projected_targets.begin() + static_cast<size_t>(node) * degree_limit;
      uint64_t output = pair_offsets[node];
      for (uint32_t offset = 0; offset < projected_counts[node]; ++offset) {
        const uint32_t target = row[offset];
        const auto reverse =
            projected_targets.begin() + static_cast<size_t>(target) * degree_limit;
        const bool reciprocal =
            std::binary_search(reverse, reverse + projected_counts[target], node);
        if (node < target || !reciprocal) {
          pairs[output++] = {std::min(node, target), std::max(node, target),
                             static_cast<kaminpar::shm::EdgeWeight>(reciprocal ? 2 : 1)};
        }
      }
    }
    std::vector<uint32_t>().swap(projected_targets);
    std::vector<uint16_t>().swap(projected_counts);
    std::vector<uint16_t>().swap(base_degrees);
    std::vector<uint8_t>().swap(invalid_nodes);
    std::vector<uint16_t>().swap(pair_counts);
    std::vector<uint64_t>().swap(pair_offsets);

    std::vector<uint64_t> degrees(point_count, 0);
    for (const auto& pair : pairs) {
      ++degrees[pair.left];
      ++degrees[pair.right];
    }
    partition_graph_t result;
    result.directed_edge_count = directed_base_edge_count;
    result.undirected_edge_count = pairs.size();
    result.offsets.resize(point_count + 1);
    for (size_t node = 0; node < point_count; ++node) {
      const uint64_t next = checked_add(result.offsets[node], degrees[node], "partition adjacency");
      if (next > std::numeric_limits<kaminpar::shm::EdgeID>::max()) {
        throw std::invalid_argument("Community partition adjacency exceeds KaMinPar EdgeID");
      }
      result.offsets[node + 1] = static_cast<kaminpar::shm::EdgeID>(next);
    }
    result.neighbors.resize(result.offsets.back());
    result.edge_weights.resize(result.offsets.back());
    std::vector<kaminpar::shm::EdgeID> cursors = result.offsets;
    for (const auto& pair : pairs) {
      const auto left_offset = cursors[pair.left]++;
      result.neighbors[left_offset] = pair.right;
      result.edge_weights[left_offset] = pair.weight;
      const auto right_offset = cursors[pair.right]++;
      result.neighbors[right_offset] = pair.left;
      result.edge_weights[right_offset] = pair.weight;
    }
    std::vector<weighted_pair_t>().swap(pairs);
    std::vector<uint64_t>().swap(degrees);
    std::vector<kaminpar::shm::EdgeID>().swap(cursors);

#pragma omp parallel num_threads(thread_count)
    {
      std::vector<std::pair<kaminpar::shm::NodeID, kaminpar::shm::EdgeWeight>> neighborhood;
#pragma omp for schedule(static)
      for (int64_t signed_node = 0; signed_node < static_cast<int64_t>(point_count);
           ++signed_node) {
        const size_t node = static_cast<size_t>(signed_node);
        const size_t begin = result.offsets[node];
        const size_t end = result.offsets[node + 1];
        neighborhood.clear();
        neighborhood.reserve(end - begin);
        for (size_t edge = begin; edge < end; ++edge) {
          neighborhood.emplace_back(result.neighbors[edge], result.edge_weights[edge]);
        }
        std::sort(neighborhood.begin(), neighborhood.end());
        for (size_t edge = 0; edge < neighborhood.size(); ++edge) {
          result.neighbors[begin + edge] = neighborhood[edge].first;
          result.edge_weights[begin + edge] = neighborhood[edge].second;
        }
      }
    }
    result.node_weights.assign(point_count, 1);
    return result;
  }

  std::vector<uint64_t> undirected_keys;
  uint64_t directed_base_edge_count = 0;
  for (uint32_t node = 0; node < graph.point_count(); ++node) {
    const auto base_neighbors = unique_neighbors(graph, node);
    directed_base_edge_count =
        checked_add(directed_base_edge_count, base_neighbors.size(), "directed Base edges");
    const auto neighbors = projected_neighbors(graph, node, projection);
    for (const uint32_t target : neighbors) {
      undirected_keys.push_back(make_undirected_key(node, target));
    }
  }
  std::sort(undirected_keys.begin(), undirected_keys.end());

  struct weighted_pair_t {
    uint32_t left;
    uint32_t right;
    kaminpar::shm::EdgeWeight weight;
  };
  std::vector<weighted_pair_t> pairs;
  for (size_t begin = 0; begin < undirected_keys.size();) {
    size_t end = begin + 1;
    while (end < undirected_keys.size() && undirected_keys[end] == undirected_keys[begin]) {
      ++end;
    }
    const size_t direction_count = end - begin;
    if (direction_count == 0 || direction_count > 2) {
      throw std::logic_error("deduplicated Base pair has an invalid reciprocal multiplicity");
    }
    pairs.push_back({key_left(undirected_keys[begin]), key_right(undirected_keys[begin]),
                     static_cast<kaminpar::shm::EdgeWeight>(direction_count)});
    begin = end;
  }

  const size_t point_count = static_cast<size_t>(graph.point_count());
  std::vector<uint64_t> degrees(point_count, 0);
  for (const auto& pair : pairs) {
    ++degrees[pair.left];
    ++degrees[pair.right];
  }

  partition_graph_t result;
  result.directed_edge_count = directed_base_edge_count;
  result.undirected_edge_count = pairs.size();
  result.offsets.resize(point_count + 1);
  for (size_t node = 0; node < point_count; ++node) {
    const uint64_t next = checked_add(result.offsets[node], degrees[node], "partition adjacency");
    if (next > std::numeric_limits<kaminpar::shm::EdgeID>::max()) {
      throw std::invalid_argument("Community partition adjacency exceeds KaMinPar EdgeID");
    }
    result.offsets[node + 1] = static_cast<kaminpar::shm::EdgeID>(next);
  }
  result.neighbors.resize(result.offsets.back());
  result.edge_weights.resize(result.offsets.back());
  std::vector<kaminpar::shm::EdgeID> cursors = result.offsets;
  for (const auto& pair : pairs) {
    const auto left_offset = cursors[pair.left]++;
    result.neighbors[left_offset] = pair.right;
    result.edge_weights[left_offset] = pair.weight;
    const auto right_offset = cursors[pair.right]++;
    result.neighbors[right_offset] = pair.left;
    result.edge_weights[right_offset] = pair.weight;
  }

  for (size_t node = 0; node < point_count; ++node) {
    const size_t begin = result.offsets[node];
    const size_t end = result.offsets[node + 1];
    std::vector<std::pair<kaminpar::shm::NodeID, kaminpar::shm::EdgeWeight>> neighborhood;
    neighborhood.reserve(end - begin);
    for (size_t edge = begin; edge < end; ++edge) {
      neighborhood.emplace_back(result.neighbors[edge], result.edge_weights[edge]);
    }
    std::sort(neighborhood.begin(), neighborhood.end());
    for (size_t edge = 0; edge < neighborhood.size(); ++edge) {
      result.neighbors[begin + edge] = neighborhood[edge].first;
      result.edge_weights[begin + edge] = neighborhood[edge].second;
    }
  }

  // Every packet record has the same width, so a unit node weight is exactly equivalent to the
  // design's fixed `(node ID bytes + PQ code bytes)` payload weight.
  result.node_weights.assign(point_count, 1);
  return result;
}

std::vector<uint32_t> canonicalize_partition(std::span<const kaminpar::shm::BlockID> raw,
                                             uint32_t community_count) {
  std::vector<uint32_t> minimum_node(community_count, std::numeric_limits<uint32_t>::max());
  for (uint32_t node = 0; node < raw.size(); ++node) {
    if (raw[node] >= community_count) {
      throw std::runtime_error("KaMinPar returned an out-of-range Community ID");
    }
    minimum_node[raw[node]] = std::min(minimum_node[raw[node]], node);
  }
  for (const uint32_t minimum : minimum_node) {
    if (minimum == std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("KaMinPar returned an empty Community");
    }
  }

  std::vector<uint32_t> order(community_count);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](uint32_t left, uint32_t right) {
    return std::tie(minimum_node[left], left) < std::tie(minimum_node[right], right);
  });
  std::vector<uint32_t> remap(community_count);
  for (uint32_t canonical = 0; canonical < community_count; ++canonical) {
    remap[order[canonical]] = canonical;
  }
  std::vector<uint32_t> partition(raw.size());
  for (size_t node = 0; node < raw.size(); ++node) {
    partition[node] = remap[raw[node]];
  }
  return partition;
}

std::vector<double> importance_by_node(std::span<const rcni_node_importance_t> importance,
                                       uint64_t point_count) {
  if (importance.size() != point_count) {
    throw std::invalid_argument("Community construction requires one RCNI record per Base node");
  }
  std::vector<double> values(importance.size(), -1.0);
  for (const auto& record : importance) {
    if (record.node_id >= point_count || values[record.node_id] >= 0.0 ||
        !std::isfinite(record.normalized_importance) || record.normalized_importance < 0.0) {
      throw std::invalid_argument("Community construction received invalid RCNI records");
    }
    values[record.node_id] = record.normalized_importance;
  }
  if (std::find(values.begin(), values.end(), -1.0) != values.end()) {
    throw std::invalid_argument("Community construction RCNI records have missing node IDs");
  }
  return values;
}

std::span<const float> vector_row(const community_polar_build_result_t& result, uint32_t node) {
  return std::span<const float>(result.original_vectors)
      .subspan(static_cast<size_t>(node) * result.index.dimension, result.index.dimension);
}

std::span<const float> pole_row(const community_polar_index_t& index, uint32_t community) {
  return std::span<const float>(index.community_poles)
      .subspan(static_cast<size_t>(community) * index.dimension, index.dimension);
}

float compute_radius(std::span<const float> point, std::span<const float> pole) {
  double squared = 0.0;
  for (size_t dimension = 0; dimension < point.size(); ++dimension) {
    const double difference = static_cast<double>(point[dimension]) - pole[dimension];
    squared += difference * difference;
  }
  return static_cast<float>(std::sqrt(squared));
}

bool compute_unit_direction(std::span<const float> point, std::span<const float> pole,
                            std::span<float> direction, float* radius = nullptr) {
  const float computed_radius = compute_radius(point, pole);
  if (radius != nullptr) {
    *radius = computed_radius;
  }
  if (computed_radius <= k_polar_default_epsilon) {
    std::fill(direction.begin(), direction.end(), 0.0F);
    return false;
  }
  for (size_t dimension = 0; dimension < point.size(); ++dimension) {
    direction[dimension] = (point[dimension] - pole[dimension]) / computed_radius;
  }
  return true;
}

float angular_distance(std::span<const float> left, std::span<const float> right) {
  double dot = 0.0;
  for (size_t dimension = 0; dimension < left.size(); ++dimension) {
    dot += static_cast<double>(left[dimension]) * right[dimension];
  }
  return static_cast<float>(std::acos(std::clamp(dot, -1.0, 1.0)));
}

std::vector<community_polar_gateway_t> select_gateways(std::span<const cross_arc_t> arcs,
                                                       uint32_t target_community,
                                                       const std::vector<double>& importance,
                                                       const community_polar_build_result_t& result,
                                                       const community_polar_build_config_t& config,
                                                       std::span<const uint32_t> profiled_targets) {
  std::vector<gateway_candidate_t> candidates;
  for (size_t begin = 0; begin < arcs.size();) {
    size_t end = begin + 1;
    while (end < arcs.size() && arcs[end].target_node == arcs[begin].target_node) {
      ++end;
    }
    uint32_t source_node = arcs[begin].source_node;
    for (size_t position = begin + 1; position < end; ++position) {
      source_node = std::min(source_node, arcs[position].source_node);
    }
    gateway_candidate_t candidate;
    candidate.source_node = source_node;
    candidate.target_node = arcs[begin].target_node;
    candidate.importance = importance[candidate.target_node];
    candidate.direction.resize(result.index.dimension);
    candidate.has_direction =
        compute_unit_direction(vector_row(result, candidate.target_node),
                               pole_row(result.index, target_community), candidate.direction);
    candidates.push_back(std::move(candidate));
    begin = end;
  }

  std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
    if (left.importance != right.importance) {
      return left.importance > right.importance;
    }
    return left.target_node < right.target_node;
  });
  const size_t selected_count = std::min<size_t>(config.gateway_count, candidates.size());
  std::vector<size_t> selected;
  std::vector<bool> is_selected(candidates.size(), false);
  for (const uint32_t target : profiled_targets) {
    const auto found =
        std::find_if(candidates.begin(), candidates.end(),
                     [&](const auto& candidate) { return candidate.target_node == target; });
    if (found == candidates.end()) {
      throw std::invalid_argument("Gateway selection CSV names a node outside its superedge");
    }
    const size_t position = static_cast<size_t>(found - candidates.begin());
    if (is_selected[position]) {
      throw std::invalid_argument("Gateway selection CSV repeats a target node");
    }
    selected.push_back(position);
    is_selected[position] = true;
  }
  if (selected.empty() && selected_count != 0) {
    selected.push_back(0);
    is_selected[0] = true;
  }
  const size_t fallback_pool_size = std::min<size_t>(config.gateway_shortlist, candidates.size());
  while (selected.size() < selected_count) {
    size_t best = candidates.size();
    double best_separation = -1.0;
    for (size_t candidate = 0; candidate < fallback_pool_size; ++candidate) {
      if (is_selected[candidate]) {
        continue;
      }
      double maximum_dot = -1.0;
      bool compared = false;
      if (candidates[candidate].has_direction) {
        for (const size_t chosen : selected) {
          if (!candidates[chosen].has_direction) {
            continue;
          }
          double dot = 0.0;
          for (uint32_t dimension = 0; dimension < result.index.dimension; ++dimension) {
            dot += static_cast<double>(candidates[candidate].direction[dimension]) *
                   candidates[chosen].direction[dimension];
          }
          maximum_dot = std::max(maximum_dot, std::clamp(dot, -1.0, 1.0));
          compared = true;
        }
      }
      const double separation = compared ? 1.0 - maximum_dot : 0.0;
      if (separation > best_separation ||
          (separation == best_separation &&
           (best == candidates.size() ||
            candidates[candidate].importance > candidates[best].importance)) ||
          (separation == best_separation && best != candidates.size() &&
           candidates[candidate].importance == candidates[best].importance &&
           candidates[candidate].target_node < candidates[best].target_node)) {
        best = candidate;
        best_separation = separation;
      }
    }
    if (best == candidates.size()) {
      throw std::logic_error("Gateway selection could not fill its bounded shortlist");
    }
    selected.push_back(best);
    is_selected[best] = true;
  }

  std::vector<community_polar_gateway_t> gateways;
  gateways.reserve(selected.size());
  for (const size_t selected_candidate : selected) {
    gateways.push_back({candidates[selected_candidate].source_node,
                        candidates[selected_candidate].target_node, 0});
  }
  return gateways;
}

void append_cell_blocks(community_polar_build_result_t& result, uint32_t community_id,
                        uint32_t sector_id, std::vector<polar_node_t> nodes,
                        float sector_maximum_angle) {
  if (nodes.empty()) {
    return;
  }
  std::sort(nodes.begin(), nodes.end(), [](const polar_node_t& left, const polar_node_t& right) {
    return std::tie(left.radius, left.node_id) < std::tie(right.radius, right.node_id);
  });

  const uint32_t radial_bin_count = std::max<uint32_t>(
      1, static_cast<uint32_t>((nodes.size() + result.index.config.cell_target_size - 1) /
                               result.index.config.cell_target_size));
  std::vector<float> radii;
  radii.reserve(nodes.size());
  for (const auto& node : nodes) {
    radii.push_back(node.radius);
  }
  const auto boundaries = compute_radial_quantile_boundaries(radii, radial_bin_count);
  std::vector<std::vector<polar_node_t>> radial_nodes(radial_bin_count);
  for (const auto& node : nodes) {
    radial_nodes[find_polar_radial_bin(node.radius, boundaries)].push_back(node);
  }

  for (uint32_t radial_cell = 0; radial_cell < radial_bin_count; ++radial_cell) {
    auto& cell_nodes = radial_nodes[radial_cell];
    if (cell_nodes.empty()) {
      continue;
    }
    std::sort(cell_nodes.begin(), cell_nodes.end(), [](const auto& left, const auto& right) {
      return std::tie(left.radius, left.node_id) < std::tie(right.radius, right.node_id);
    });

    community_polar_cell_t cell;
    cell.cell_id = static_cast<uint32_t>(result.index.cells.size());
    cell.community_id = community_id;
    cell.sector_id = sector_id;
    cell.radial_cell_id = radial_cell;
    cell.node_count = cell_nodes.size();
    cell.capacity_class = result.index.config.cell_target_size;
    cell.minimum_radius = cell_nodes.front().radius;
    cell.maximum_radius = cell_nodes.back().radius;
    cell.maximum_angle = sector_maximum_angle;
    cell.block_begin = result.index.blocks.size();

    for (size_t begin = 0; begin < cell_nodes.size();
         begin += result.index.config.block_target_size) {
      const size_t end = std::min(cell_nodes.size(), begin + result.index.config.block_target_size);
      community_polar_block_t block;
      block.block_id = static_cast<uint32_t>(result.index.blocks.size());
      block.cell_id = cell.cell_id;
      block.node_count = static_cast<uint32_t>(end - begin);
      block.packet_record_begin = result.packet_node_ids.size();
      block.minimum_radius = cell_nodes[begin].radius;
      block.maximum_radius = cell_nodes[end - 1].radius;
      block.maximum_angle = 0.0F;
      for (size_t position = begin; position < end; ++position) {
        block.maximum_angle = std::max(block.maximum_angle, cell_nodes[position].angle);
        const uint64_t packet_index = result.packet_node_ids.size();
        result.packet_node_ids.push_back(cell_nodes[position].node_id);
        result.index.node_to_cell[cell_nodes[position].node_id] = cell.cell_id;
        result.index.node_to_packet_record[cell_nodes[position].node_id] = packet_index;
      }
      result.index.blocks.push_back(block);
    }
    cell.block_count = static_cast<uint32_t>(result.index.blocks.size() - cell.block_begin);
    result.index.cells.push_back(cell);
  }
}

void append_sector(community_polar_build_result_t& result, uint32_t community_id,
                   uint32_t sector_id, uint64_t axis_index, std::vector<polar_node_t> nodes,
                   float maximum_angle) {
  if (nodes.empty()) {
    return;
  }
  community_polar_sector_t sector;
  sector.community_id = community_id;
  sector.sector_id = sector_id;
  sector.node_count = nodes.size();
  sector.axis_index = axis_index;
  sector.minimum_radius = std::numeric_limits<float>::max();
  sector.maximum_radius = 0.0F;
  sector.maximum_angle = maximum_angle;
  sector.cell_begin = result.index.cells.size();
  for (const auto& node : nodes) {
    sector.minimum_radius = std::min(sector.minimum_radius, node.radius);
    sector.maximum_radius = std::max(sector.maximum_radius, node.radius);
  }
  append_cell_blocks(result, community_id, sector_id, std::move(nodes), maximum_angle);
  sector.cell_count = static_cast<uint32_t>(result.index.cells.size() - sector.cell_begin);
  result.index.sectors.push_back(sector);
}

void append_graph_local_cell(community_polar_build_result_t& result, uint32_t community_id,
                             uint32_t cell_ordinal, std::span<const uint32_t> nodes) {
  if (nodes.empty()) {
    return;
  }
  const auto pole = pole_row(result.index, community_id);
  community_polar_cell_t cell;
  cell.cell_id = static_cast<uint32_t>(result.index.cells.size());
  cell.community_id = community_id;
  cell.sector_id = k_pole_sector_id;
  cell.radial_cell_id = cell_ordinal;
  cell.node_count = nodes.size();
  cell.capacity_class = result.index.config.cell_target_size;
  cell.minimum_radius = std::numeric_limits<float>::max();
  cell.maximum_angle = std::numbers::pi_v<float>;
  cell.block_begin = result.index.blocks.size();
  const size_t centroid_begin = result.index.direction_axes.size();
  result.index.direction_axes.resize(centroid_begin + result.index.dimension, 0.0F);
  for (const uint32_t node : nodes) {
    const float radius = compute_radius(vector_row(result, node), pole);
    cell.minimum_radius = std::min(cell.minimum_radius, radius);
    cell.maximum_radius = std::max(cell.maximum_radius, radius);
    const auto vector = vector_row(result, node);
    for (uint32_t dimension = 0; dimension < result.index.dimension; ++dimension) {
      result.index.direction_axes[centroid_begin + dimension] += vector[dimension];
    }
  }
  for (uint32_t dimension = 0; dimension < result.index.dimension; ++dimension) {
    result.index.direction_axes[centroid_begin + dimension] /= static_cast<float>(nodes.size());
  }

  for (size_t begin = 0; begin < nodes.size(); begin += result.index.config.block_target_size) {
    const size_t end = std::min(nodes.size(), begin + result.index.config.block_target_size);
    community_polar_block_t block;
    block.block_id = static_cast<uint32_t>(result.index.blocks.size());
    block.cell_id = cell.cell_id;
    block.node_count = static_cast<uint32_t>(end - begin);
    block.packet_record_begin = result.packet_node_ids.size();
    block.minimum_radius = std::numeric_limits<float>::max();
    block.maximum_angle = std::numbers::pi_v<float>;
    for (size_t position = begin; position < end; ++position) {
      const uint32_t node = nodes[position];
      const float radius = compute_radius(vector_row(result, node), pole);
      block.minimum_radius = std::min(block.minimum_radius, radius);
      block.maximum_radius = std::max(block.maximum_radius, radius);
      const uint64_t packet_index = result.packet_node_ids.size();
      result.packet_node_ids.push_back(node);
      result.index.node_to_cell[node] = cell.cell_id;
      result.index.node_to_packet_record[node] = packet_index;
    }
    result.index.blocks.push_back(block);
  }
  cell.block_count = static_cast<uint32_t>(result.index.blocks.size() - cell.block_begin);
  result.index.cells.push_back(cell);
}

void build_graph_local_cells(const post_link_graph_view_t& graph,
                             community_polar_build_result_t& result,
                             const std::vector<std::vector<uint32_t>>& community_nodes,
                             uint32_t thread_count) {
  std::vector<uint32_t> local_position(checked_size(graph.point_count(), "Base point count"));
  for (const auto& nodes : community_nodes) {
    for (uint32_t position = 0; position < nodes.size(); ++position) {
      local_position[nodes[position]] = position;
    }
  }

  std::vector<std::vector<std::vector<uint32_t>>> clustered_communities(community_nodes.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
  for (int64_t signed_community = 0;
       signed_community < static_cast<int64_t>(community_nodes.size()); ++signed_community) {
    const uint32_t community_id = static_cast<uint32_t>(signed_community);
    const auto& nodes = community_nodes[community_id];

    std::vector<std::vector<uint32_t>> adjacency(nodes.size());
    for (uint32_t source = 0; source < nodes.size(); ++source) {
      for (const uint32_t target : graph.neighbors(nodes[source])) {
        if (result.index.node_to_community[target] != community_id) {
          continue;
        }
        const uint32_t target_position = local_position[target];
        adjacency[source].push_back(target_position);
        adjacency[target_position].push_back(source);
      }
    }
    for (auto& neighbors : adjacency) {
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    std::vector<uint32_t> seeds(nodes.size());
    std::iota(seeds.begin(), seeds.end(), 0);
    std::sort(seeds.begin(), seeds.end(), [&](uint32_t left, uint32_t right) {
      return std::make_tuple(std::numeric_limits<size_t>::max() - adjacency[left].size(),
                             nodes[left]) <
             std::make_tuple(std::numeric_limits<size_t>::max() - adjacency[right].size(),
                             nodes[right]);
    });
    std::vector<uint8_t> assigned(nodes.size(), 0);
    std::vector<uint32_t> affinity(nodes.size(), 0);
    std::vector<uint32_t> touched;
    std::vector<uint32_t> cluster;
    for (const uint32_t seed : seeds) {
      if (assigned[seed] != 0) {
        continue;
      }
      std::priority_queue<std::tuple<uint32_t, size_t, uint32_t>> frontier;
      touched.clear();
      cluster.clear();
      const auto add_node = [&](uint32_t node) {
        assigned[node] = 1;
        cluster.push_back(nodes[node]);
        for (const uint32_t neighbor : adjacency[node]) {
          if (assigned[neighbor] != 0) {
            continue;
          }
          if (affinity[neighbor] == 0) {
            touched.push_back(neighbor);
          }
          ++affinity[neighbor];
          frontier.emplace(affinity[neighbor], adjacency[neighbor].size(),
                           std::numeric_limits<uint32_t>::max() - nodes[neighbor]);
        }
      };
      add_node(seed);
      while (cluster.size() < result.index.config.cell_target_size && !frontier.empty()) {
        while (!frontier.empty()) {
          const auto [candidate_affinity, candidate_degree, reversed_node] = frontier.top();
          frontier.pop();
          const uint32_t node_id = std::numeric_limits<uint32_t>::max() - reversed_node;
          const uint32_t candidate = local_position[node_id];
          if (assigned[candidate] != 0 || affinity[candidate] != candidate_affinity ||
              adjacency[candidate].size() != candidate_degree) {
            continue;
          }
          add_node(candidate);
          break;
        }
      }
      clustered_communities[community_id].push_back(cluster);
      for (const uint32_t node : touched) {
        affinity[node] = 0;
      }
    }
  }

  for (uint32_t community_id = 0; community_id < community_nodes.size(); ++community_id) {
    const auto& nodes = community_nodes[community_id];
    auto& community = result.index.communities[community_id];
    community.sector_begin = result.index.sectors.size();
    community.cell_begin = result.index.cells.size();
    community.graph_only = false;
    for (uint32_t cell_ordinal = 0;
         cell_ordinal < clustered_communities[community_id].size(); ++cell_ordinal) {
      append_graph_local_cell(result, community_id, cell_ordinal,
                              clustered_communities[community_id][cell_ordinal]);
    }

    community_polar_sector_t sector;
    sector.community_id = community_id;
    sector.sector_id = k_pole_sector_id;
    sector.node_count = nodes.size();
    sector.axis_index = k_invalid_axis_index;
    sector.minimum_radius = 0.0F;
    sector.maximum_radius = community.radius;
    sector.maximum_angle = std::numbers::pi_v<float>;
    sector.cell_begin = community.cell_begin;
    sector.cell_count = static_cast<uint32_t>(result.index.cells.size() - community.cell_begin);
    result.index.sectors.push_back(sector);
    community.sector_count = 1;
    community.cell_count = sector.cell_count;
  }
}

void build_global_graph_local_cells(const post_link_graph_view_t& graph,
                                    community_polar_build_result_t& result) {
  const size_t point_count = checked_size(graph.point_count(), "Base point count");
  // 0 = unseen, 1 = assigned, 2 = queued by the current Cell. The queued state suppresses
  // duplicate edges without retaining a global edge set; any unconsumed frontier is reset when a
  // bounded Cell closes.
  std::vector<uint8_t> state(point_count, 0);
  std::vector<uint32_t> frontier;
  frontier.reserve(static_cast<size_t>(result.index.config.cell_target_size) * 64U);
  std::vector<uint32_t> cluster;
  cluster.reserve(result.index.config.cell_target_size);

  for (auto& community : result.index.communities) {
    community.sector_begin = 1;
    community.sector_count = 0;
    community.cell_begin = 0;
    community.cell_count = 0;
    community.graph_only = false;
  }
  auto& owner = result.index.communities.front();
  owner.sector_begin = 0;
  owner.cell_begin = 0;

  uint32_t cell_ordinal = 0;
  for (uint32_t seed = 0; seed < graph.point_count(); ++seed) {
    if (state[seed] != 0) {
      continue;
    }
    frontier.clear();
    cluster.clear();
    const auto add_node = [&](uint32_t node) {
      state[node] = 1;
      cluster.push_back(node);
      for (const uint32_t neighbor : graph.neighbors(node)) {
        if (state[neighbor] != 0) {
          continue;
        }
        state[neighbor] = 2;
        frontier.push_back(neighbor);
      }
    };
    add_node(seed);
    size_t frontier_cursor = 0;
    while (cluster.size() < result.index.config.cell_target_size &&
           frontier_cursor < frontier.size()) {
      add_node(frontier[frontier_cursor++]);
    }
    append_graph_local_cell(result, 0, cell_ordinal++, cluster);
    for (; frontier_cursor < frontier.size(); ++frontier_cursor) {
      state[frontier[frontier_cursor]] = 0;
    }
  }

  community_polar_sector_t sector;
  sector.community_id = 0;
  sector.sector_id = k_pole_sector_id;
  sector.node_count = result.index.point_count;
  sector.axis_index = k_invalid_axis_index;
  sector.minimum_radius = 0.0F;
  sector.maximum_radius = owner.radius;
  sector.maximum_angle = std::numbers::pi_v<float>;
  sector.cell_begin = 0;
  sector.cell_count = static_cast<uint32_t>(result.index.cells.size());
  result.index.sectors.push_back(sector);
  owner.sector_count = 1;
  owner.cell_count = sector.cell_count;
  for (size_t community = 1; community < result.index.communities.size(); ++community) {
    result.index.communities[community].cell_begin = result.index.cells.size();
  }
}

void build_convergence_coaccess_cells(community_polar_build_result_t& result,
                                      const std::vector<std::vector<uint32_t>>& community_nodes,
                                      std::span<const convergence_cell_assignment_t> assignments) {
  if (assignments.size() != result.index.point_count) {
    throw std::invalid_argument("Convergence Cell profile point count is incompatible");
  }
  for (uint32_t community_id = 0; community_id < community_nodes.size(); ++community_id) {
    auto& community = result.index.communities[community_id];
    community.sector_begin = result.index.sectors.size();
    community.cell_begin = result.index.cells.size();
    community.graph_only = false;
    std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> profiled_cells;
    for (const uint32_t node : community_nodes[community_id]) {
      const auto& assignment = assignments[node];
      if (assignment.community_id != community_id) {
        throw std::invalid_argument(
            "Convergence Cell profile disagrees with the rebuilt Community partition");
      }
      profiled_cells[assignment.cell_id].emplace_back(assignment.order, node);
    }
    uint32_t cell_ordinal = 0;
    for (auto& [profile_cell, ordered_nodes] : profiled_cells) {
      static_cast<void>(profile_cell);
      if (ordered_nodes.size() > result.index.config.cell_target_size) {
        throw std::invalid_argument("Convergence Cell profile exceeds the target Cell size");
      }
      std::sort(ordered_nodes.begin(), ordered_nodes.end());
      for (size_t position = 1; position < ordered_nodes.size(); ++position) {
        if (ordered_nodes[position - 1].first == ordered_nodes[position].first) {
          throw std::invalid_argument("Convergence Cell profile repeats an in-Cell order");
        }
      }
      std::vector<uint32_t> nodes;
      nodes.reserve(ordered_nodes.size());
      for (const auto& [order, node] : ordered_nodes) {
        static_cast<void>(order);
        nodes.push_back(node);
      }
      append_graph_local_cell(result, community_id, cell_ordinal++, nodes);
    }

    community_polar_sector_t sector;
    sector.community_id = community_id;
    sector.sector_id = k_pole_sector_id;
    sector.node_count = community_nodes[community_id].size();
    sector.axis_index = k_invalid_axis_index;
    sector.minimum_radius = 0.0F;
    sector.maximum_radius = community.radius;
    sector.maximum_angle = std::numbers::pi_v<float>;
    sector.cell_begin = community.cell_begin;
    sector.cell_count = static_cast<uint32_t>(result.index.cells.size() - community.cell_begin);
    result.index.sectors.push_back(sector);
    community.sector_count = 1;
    community.cell_count = sector.cell_count;
  }
}

void build_global_geometric_cells(community_polar_build_result_t& result,
                                  std::span<const convergence_cell_assignment_t> assignments) {
  if (assignments.size() != result.index.point_count) {
    throw std::invalid_argument("Global geometric Cell profile point count is incompatible");
  }
  std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> profiled_cells;
  for (uint32_t node = 0; node < assignments.size(); ++node) {
    profiled_cells[assignments[node].cell_id].emplace_back(assignments[node].order, node);
  }
  for (auto& community : result.index.communities) {
    community.sector_begin = result.index.sectors.size();
    community.sector_count = 0;
    community.cell_begin = result.index.cells.size();
    community.cell_count = 0;
  }
  auto& owner = result.index.communities.front();
  owner.sector_begin = 0;
  owner.cell_begin = 0;
  uint32_t cell_ordinal = 0;
  for (auto& [profile_cell, ordered_nodes] : profiled_cells) {
    static_cast<void>(profile_cell);
    if (ordered_nodes.size() > result.index.config.cell_target_size) {
      throw std::invalid_argument("Global geometric Cell profile exceeds the target Cell size");
    }
    std::sort(ordered_nodes.begin(), ordered_nodes.end());
    std::vector<uint32_t> nodes;
    nodes.reserve(ordered_nodes.size());
    for (const auto& [order, node] : ordered_nodes) {
      static_cast<void>(order);
      nodes.push_back(node);
    }
    append_graph_local_cell(result, 0, cell_ordinal++, nodes);
  }
  community_polar_sector_t sector;
  sector.community_id = 0;
  sector.sector_id = k_pole_sector_id;
  sector.node_count = result.index.point_count;
  sector.axis_index = k_invalid_axis_index;
  sector.minimum_radius = 0.0F;
  sector.maximum_radius = owner.radius;
  sector.maximum_angle = std::numbers::pi_v<float>;
  sector.cell_begin = 0;
  sector.cell_count = static_cast<uint32_t>(result.index.cells.size());
  result.index.sectors.push_back(sector);
  owner.sector_count = 1;
  owner.cell_count = sector.cell_count;
  for (size_t community = 1; community < result.index.communities.size(); ++community) {
    result.index.communities[community].sector_begin = 1;
    result.index.communities[community].cell_begin = result.index.cells.size();
  }
}

void build_contiguous_cell_hierarchy(community_polar_index_t& index) {
  index.cell_hierarchy_nodes.clear();
  index.cell_hierarchy_roots.assign(index.communities.size(),
                                    std::numeric_limits<uint32_t>::max());
  index.cell_hierarchy_children.clear();
  index.cell_hierarchy_centroids.clear();
  std::vector<std::vector<uint32_t>> node_children;

  const uint32_t dimension = index.dimension;
  const auto cell_centroid = [&](uint32_t cell) {
    return std::span<const float>(index.direction_axes)
        .subspan(static_cast<size_t>(cell) * dimension, dimension);
  };
  std::function<uint32_t(uint32_t, uint32_t)> append_node;
  append_node = [&](uint32_t begin, uint32_t end) -> uint32_t {
    if (begin >= end || end > index.cells.size()) {
      throw std::logic_error("Contiguous Cell hierarchy range is invalid");
    }
    const uint32_t node_id = static_cast<uint32_t>(index.cell_hierarchy_nodes.size());
    index.cell_hierarchy_nodes.push_back({});
    node_children.emplace_back();
    index.cell_hierarchy_nodes.back().node_id = node_id;
    const size_t centroid_begin = index.cell_hierarchy_centroids.size();
    index.cell_hierarchy_centroids.resize(centroid_begin + dimension, 0.0F);
    uint64_t descendant_count = 0;
    for (uint32_t cell = begin; cell < end; ++cell) {
      descendant_count = checked_add(descendant_count, index.cells[cell].node_count,
                                     "Contiguous Cell hierarchy descendants");
      const auto centroid = cell_centroid(cell);
      for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
        index.cell_hierarchy_centroids[centroid_begin + coordinate] +=
            static_cast<float>(static_cast<double>(index.cells[cell].node_count) *
                               centroid[coordinate]);
      }
    }
    for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
      index.cell_hierarchy_centroids[centroid_begin + coordinate] =
          static_cast<float>(index.cell_hierarchy_centroids[centroid_begin + coordinate] /
                             static_cast<double>(descendant_count));
    }
    const auto centroid =
        std::span<const float>(index.cell_hierarchy_centroids)
            .subspan(centroid_begin, dimension);
    float radius = 0.0F;
    for (uint32_t cell = begin; cell < end; ++cell) {
      double distance = 0.0;
      const auto leaf_centroid = cell_centroid(cell);
      for (uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
        const double difference = static_cast<double>(centroid[coordinate]) -
                                  static_cast<double>(leaf_centroid[coordinate]);
        distance += difference * difference;
      }
      radius = std::max(radius, static_cast<float>(std::sqrt(distance)));
    }
    auto& node = index.cell_hierarchy_nodes[node_id];
    node.descendant_node_count = descendant_count;
    node.radius = std::nextafter(radius, std::numeric_limits<float>::infinity());

    const uint32_t cell_count = end - begin;
    if (cell_count <= index.config.cell_hierarchy_leaf_size) {
      node.child_count = cell_count;
      node.children_are_cells = true;
      node_children[node_id].reserve(cell_count);
      for (uint32_t cell = begin; cell < end; ++cell) {
        node_children[node_id].push_back(cell);
      }
      return node_id;
    }

    const uint32_t child_count = std::min(index.config.cell_hierarchy_branching, cell_count);
    std::vector<uint32_t> children;
    children.reserve(child_count);
    uint32_t child_begin = begin;
    for (uint32_t child = 0; child < child_count; ++child) {
      const uint32_t remaining = end - child_begin;
      const uint32_t remaining_children = child_count - child;
      const uint32_t count = (remaining + remaining_children - 1) / remaining_children;
      children.push_back(append_node(child_begin, child_begin + count));
      child_begin += count;
    }
    auto& completed_node = index.cell_hierarchy_nodes[node_id];
    completed_node.child_count = child_count;
    completed_node.children_are_cells = false;
    node_children[node_id] = std::move(children);
    return node_id;
  };

  for (uint32_t community = 0; community < index.communities.size(); ++community) {
    const auto& record = index.communities[community];
    if (record.cell_count == 0) {
      continue;
    }
    const uint64_t end = record.cell_begin + record.cell_count;
    if (end > index.cells.size() || end > std::numeric_limits<uint32_t>::max()) {
      throw std::logic_error("Community Cell range cannot be represented in its hierarchy");
    }
    index.cell_hierarchy_roots[community] =
        append_node(static_cast<uint32_t>(record.cell_begin), static_cast<uint32_t>(end));
  }
  for (uint32_t node_id = 0; node_id < index.cell_hierarchy_nodes.size(); ++node_id) {
    auto& node = index.cell_hierarchy_nodes[node_id];
    node.child_begin = static_cast<uint32_t>(index.cell_hierarchy_children.size());
    index.cell_hierarchy_children.insert(index.cell_hierarchy_children.end(),
                                         node_children[node_id].begin(),
                                         node_children[node_id].end());
  }
}

void repack_adjacent_graph_local_cells(community_polar_index_t& index,
                                       const community_polar_build_config_t& config) {
  if (index.config.cell_partition != community_cell_partition_t::GRAPH_LOCAL ||
      config.cell_partition != community_cell_partition_t::GRAPH_LOCAL ||
      config.adaptive_multi_capacity || config.cell_target_size == 0 ||
      index.direction_axes.size() != index.cells.size() * index.dimension) {
    throw std::invalid_argument("Fast Cell repack requires fixed graph-local source Cells");
  }
  const auto source_cells = std::move(index.cells);
  const auto source_blocks = std::move(index.blocks);
  const auto source_centroids = std::move(index.direction_axes);
  index.cells.clear();
  index.blocks.clear();
  index.direction_axes.clear();
  index.node_to_cell.assign(index.point_count, std::numeric_limits<uint32_t>::max());

  for (auto& community : index.communities) {
    const uint64_t source_begin = community.cell_begin;
    const uint64_t source_end = source_begin + community.cell_count;
    community.cell_begin = index.cells.size();
    uint64_t cursor = source_begin;
    while (cursor < source_end) {
      const uint64_t group_begin = cursor;
      uint32_t population = 0;
      while (cursor < source_end &&
             source_cells[cursor].sector_id == source_cells[group_begin].sector_id &&
             source_cells[cursor].node_count <= config.cell_target_size - population) {
        population += source_cells[cursor].node_count;
        ++cursor;
      }
      if (cursor == group_begin) {
        throw std::invalid_argument("Source Cell exceeds the requested fixed capacity");
      }

      community_polar_cell_t cell;
      cell.cell_id = static_cast<uint32_t>(index.cells.size());
      cell.community_id = community.community_id;
      cell.sector_id = source_cells[group_begin].sector_id;
      cell.node_count = population;
      cell.capacity_class = config.cell_target_size;
      cell.minimum_radius = std::numeric_limits<float>::max();
      cell.block_begin = index.blocks.size();
      const size_t centroid_begin = index.direction_axes.size();
      index.direction_axes.resize(centroid_begin + index.dimension, 0.0F);
      for (uint64_t source_cell_id = group_begin; source_cell_id < cursor; ++source_cell_id) {
        const auto& source_cell = source_cells[source_cell_id];
        cell.minimum_radius = std::min(cell.minimum_radius, source_cell.minimum_radius);
        cell.maximum_radius = std::max(cell.maximum_radius, source_cell.maximum_radius);
        cell.maximum_angle = std::max(cell.maximum_angle, source_cell.maximum_angle);
        for (uint32_t dimension = 0; dimension < index.dimension; ++dimension) {
          index.direction_axes[centroid_begin + dimension] +=
              static_cast<float>(static_cast<double>(source_cell.node_count) *
                                 source_centroids[source_cell_id * index.dimension + dimension]);
        }
        for (uint64_t block_id = source_cell.block_begin;
             block_id < source_cell.block_begin + source_cell.block_count; ++block_id) {
          auto block = source_blocks[block_id];
          block.block_id = static_cast<uint32_t>(index.blocks.size());
          block.cell_id = cell.cell_id;
          index.blocks.push_back(block);
          const uint64_t record_bytes = sizeof(uint32_t) + index.pq_code_width;
          for (uint64_t packet = block.packet_record_begin;
               packet < block.packet_record_begin + block.node_count; ++packet) {
            const size_t packet_cursor = static_cast<size_t>(packet * record_bytes);
            uint32_t node = 0;
            std::memcpy(&node, index.packet_payload.data() + packet_cursor, sizeof(node));
            index.node_to_cell[node] = cell.cell_id;
          }
        }
      }
      for (uint32_t dimension = 0; dimension < index.dimension; ++dimension) {
        index.direction_axes[centroid_begin + dimension] /= static_cast<float>(population);
      }
      cell.block_count = static_cast<uint32_t>(index.blocks.size() - cell.block_begin);
      index.cells.push_back(cell);
    }
    community.cell_count = static_cast<uint32_t>(index.cells.size() - community.cell_begin);
  }

  uint64_t cell_cursor = 0;
  for (auto& sector : index.sectors) {
    sector.cell_begin = cell_cursor;
    sector.cell_count = 0;
    while (cell_cursor + sector.cell_count < index.cells.size() &&
           index.cells[cell_cursor + sector.cell_count].community_id == sector.community_id &&
           index.cells[cell_cursor + sector.cell_count].sector_id == sector.sector_id) {
      ++sector.cell_count;
    }
    cell_cursor += sector.cell_count;
  }
  if (cell_cursor != index.cells.size() ||
      std::find(index.node_to_cell.begin(), index.node_to_cell.end(),
                std::numeric_limits<uint32_t>::max()) != index.node_to_cell.end()) {
    throw std::logic_error("Fast Cell repack did not preserve complete one-owner coverage");
  }
  index.config = config;
  index.config.convergence_cell_source_path.clear();
  build_contiguous_cell_hierarchy(index);
  for (auto& gateway : index.gateways) {
    gateway.packet_record_index = index.node_to_packet_record[gateway.target_node];
  }
}

void repack_pq_locality_cells_impl(community_polar_index_t& index,
                                   const std::filesystem::path& profile_path,
                                   const community_polar_build_config_t& config) {
  if (config.cell_partition != community_cell_partition_t::GLOBAL_GEOMETRIC ||
      config.cell_target_size == 0 || config.block_target_size == 0 || profile_path.empty()) {
    throw std::invalid_argument("PQ-locality Cell repack configuration is invalid");
  }
  std::ifstream input(profile_path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("Failed to open PQ-locality Cell profile: " +
                                profile_path.string());
  }
  std::array<uint8_t, k_pq_locality_header_size> header{};
  input.read(reinterpret_cast<char*>(header.data()), header.size());
  if (!input || !std::equal(k_pq_locality_magic.begin(), k_pq_locality_magic.end(),
                            header.begin())) {
    throw std::invalid_argument("PQ-locality Cell profile header is truncated or invalid");
  }
  size_t cursor = k_pq_locality_magic.size();
  const uint32_t version = read_u32_le(header, cursor);
  const uint32_t header_size = read_u32_le(header, cursor);
  const uint32_t endian = read_u32_le(header, cursor);
  const uint32_t target_size = read_u32_le(header, cursor);
  const uint32_t dimension = read_u32_le(header, cursor);
  const uint64_t point_count = read_u64_le(header, cursor);
  const uint64_t cell_count = read_u64_le(header, cursor);
  const uint64_t checksum = read_u64_le(header, cursor);
  const uint64_t offset_bytes = checked_multiply(cell_count + 1, sizeof(uint64_t),
                                                 "PQ-locality Cell offsets");
  const uint64_t node_bytes = checked_multiply(point_count, sizeof(uint32_t),
                                               "PQ-locality Cell node order");
  const uint64_t centroid_bytes = checked_multiply(
      checked_multiply(cell_count, dimension, "PQ-locality centroids"), sizeof(float),
      "PQ-locality centroid bytes");
  const uint64_t payload_bytes = checked_add(
      checked_add(offset_bytes, node_bytes, "PQ-locality payload"), centroid_bytes,
      "PQ-locality payload");
  const bool target_is_compatible = config.adaptive_multi_capacity
                                        ? target_size <= config.cell_target_size
                                        : target_size == config.cell_target_size;
  if (version != k_pq_locality_version || header_size != k_pq_locality_header_size ||
      endian != k_endian_marker || !target_is_compatible || dimension != index.dimension ||
      point_count != index.point_count || cell_count == 0 ||
      cell_count > std::numeric_limits<uint32_t>::max() ||
      std::filesystem::file_size(profile_path) !=
          checked_add(header_size, payload_bytes, "PQ-locality file bytes")) {
    throw std::invalid_argument("PQ-locality Cell profile metadata is incompatible");
  }
  std::vector<uint8_t> payload(checked_size(payload_bytes, "PQ-locality payload"));
  input.read(reinterpret_cast<char*>(payload.data()), payload.size());
  if (!input || static_cast<uint64_t>(crc32_bytes(payload)) != checksum) {
    throw std::invalid_argument("PQ-locality Cell profile checksum is invalid");
  }

  cursor = 0;
  std::vector<uint64_t> cell_offsets(checked_size(cell_count + 1, "PQ-locality Cell offsets"));
  for (auto& value : cell_offsets) {
    value = read_u64_le(payload, cursor);
  }
  if (cell_offsets.front() != 0 || cell_offsets.back() != point_count ||
      !std::is_sorted(cell_offsets.begin(), cell_offsets.end()) ||
      std::adjacent_find(cell_offsets.begin(), cell_offsets.end()) != cell_offsets.end()) {
    throw std::invalid_argument("PQ-locality Cell offsets do not cover nonempty Cells");
  }
  std::vector<uint32_t> ordered_nodes(checked_size(point_count, "PQ-locality node order"));
  std::vector<uint8_t> seen(checked_size((point_count + 7) / 8, "PQ-locality ownership"), 0);
  for (auto& node : ordered_nodes) {
    node = read_u32_le(payload, cursor);
    if (node >= point_count || (seen[node >> 3U] & (1U << (node & 7U))) != 0) {
      throw std::invalid_argument("PQ-locality Cell profile violates one-owner coverage");
    }
    seen[node >> 3U] |= static_cast<uint8_t>(1U << (node & 7U));
  }
  std::vector<float> cell_centroids(
      checked_size(checked_multiply(cell_count, dimension, "PQ-locality centroids"),
                   "PQ-locality centroids"));
  for (auto& value : cell_centroids) {
    value = read_float_le(payload, cursor);
    if (!std::isfinite(value)) {
      throw std::invalid_argument("PQ-locality Cell profile contains a non-finite centroid");
    }
  }
  if (cursor != payload.size()) {
    throw std::invalid_argument("PQ-locality Cell profile contains trailing bytes");
  }

  const uint64_t record_bytes = sizeof(uint32_t) + index.pq_code_width;
  const auto source_payload = std::move(index.packet_payload);
  const auto source_node_to_packet = std::move(index.node_to_packet_record);
  if (source_payload.size() != checked_multiply(point_count, record_bytes,
                                                "source Community-Polar packets") ||
      source_node_to_packet.size() != point_count) {
    throw std::invalid_argument("PQ-locality source sidecar packet directory is invalid");
  }
  index.config = config;
  index.config.convergence_cell_profile_path.clear();
  index.config.convergence_cell_hierarchy_path.clear();
  index.config.convergence_cell_capacity_path.clear();
  index.config.convergence_cell_source_path.clear();
  index.node_to_cell.assign(checked_size(point_count, "PQ-locality node ownership"),
                            std::numeric_limits<uint32_t>::max());
  index.node_to_packet_record.assign(checked_size(point_count, "PQ-locality packet directory"),
                                     std::numeric_limits<uint64_t>::max());
  index.direction_axes = std::move(cell_centroids);
  index.sectors.clear();
  index.cells.clear();
  index.blocks.clear();
  index.cell_hierarchy_nodes.clear();
  index.cell_hierarchy_roots.clear();
  index.cell_hierarchy_children.clear();
  index.cell_hierarchy_centroids.clear();
  index.packet_payload.resize(checked_size(checked_multiply(point_count, record_bytes,
                                                            "PQ-locality packets"),
                                           "PQ-locality packets"));

  for (auto& community : index.communities) {
    community.sector_begin = 1;
    community.sector_count = 0;
    community.cell_begin = cell_count;
    community.cell_count = 0;
  }
  auto& owner = index.communities.front();
  owner.sector_begin = 0;
  owner.sector_count = 1;
  owner.cell_begin = 0;
  owner.cell_count = static_cast<uint32_t>(cell_count);
  uint64_t packet = 0;
  for (uint32_t cell_id = 0; cell_id < cell_count; ++cell_id) {
    const uint64_t begin = cell_offsets[cell_id];
    const uint64_t end = cell_offsets[cell_id + 1];
    if (end - begin > config.cell_target_size) {
      throw std::invalid_argument("PQ-locality Cell exceeds the requested capacity");
    }
    community_polar_cell_t cell;
    cell.cell_id = cell_id;
    cell.community_id = 0;
    cell.sector_id = k_pole_sector_id;
    cell.radial_cell_id = cell_id;
    cell.node_count = end - begin;
    cell.capacity_class = config.cell_target_size;
    cell.minimum_radius = 0.0F;
    cell.maximum_radius = owner.radius;
    cell.maximum_angle = std::numbers::pi_v<float>;
    cell.block_begin = index.blocks.size();
    for (uint64_t block_begin = begin; block_begin < end;
         block_begin += config.block_target_size) {
      const uint64_t block_end = std::min<uint64_t>(end, block_begin + config.block_target_size);
      community_polar_block_t block;
      block.block_id = static_cast<uint32_t>(index.blocks.size());
      block.cell_id = cell_id;
      block.node_count = static_cast<uint32_t>(block_end - block_begin);
      block.packet_record_begin = packet;
      block.minimum_radius = 0.0F;
      block.maximum_radius = owner.radius;
      block.maximum_angle = std::numbers::pi_v<float>;
      block.maximum_pq_reconstruction_error = std::numeric_limits<float>::max();
      for (uint64_t position = block_begin; position < block_end; ++position, ++packet) {
        const uint32_t node = ordered_nodes[position];
        const uint64_t source_packet = source_node_to_packet[node];
        if (source_packet >= point_count) {
          throw std::invalid_argument("PQ-locality source packet mapping is out of range");
        }
        std::copy_n(source_payload.begin() + static_cast<size_t>(source_packet * record_bytes),
                    record_bytes,
                    index.packet_payload.begin() + static_cast<size_t>(packet * record_bytes));
        index.node_to_cell[node] = cell_id;
        index.node_to_packet_record[node] = packet;
      }
      index.blocks.push_back(block);
    }
    cell.block_count = static_cast<uint32_t>(index.blocks.size() - cell.block_begin);
    index.cells.push_back(cell);
  }
  community_polar_sector_t sector;
  sector.community_id = 0;
  sector.sector_id = k_pole_sector_id;
  sector.node_count = point_count;
  sector.axis_index = k_invalid_axis_index;
  sector.minimum_radius = 0.0F;
  sector.maximum_radius = owner.radius;
  sector.maximum_angle = std::numbers::pi_v<float>;
  sector.cell_begin = 0;
  sector.cell_count = static_cast<uint32_t>(cell_count);
  index.sectors.push_back(sector);
  for (auto& gateway : index.gateways) {
    gateway.packet_record_index = index.node_to_packet_record[gateway.target_node];
  }
}

void build_graph_only_cells(community_polar_build_result_t& result, uint32_t community_id,
                            std::span<const uint32_t> nodes) {
  std::vector<polar_node_t> fallback;
  fallback.reserve(nodes.size());
  const auto pole = pole_row(result.index, community_id);
  for (const uint32_t node : nodes) {
    fallback.push_back(
        {node, compute_radius(vector_row(result, node), pole), std::numbers::pi_v<float>});
  }
  append_sector(result, community_id, k_pole_sector_id, k_invalid_axis_index, std::move(fallback),
                std::numbers::pi_v<float>);
}

void build_community_polar_cells(community_polar_build_result_t& result,
                                 const std::vector<std::vector<uint32_t>>& community_nodes) {
  for (uint32_t community_id = 0; community_id < community_nodes.size(); ++community_id) {
    auto& community = result.index.communities[community_id];
    community.sector_begin = result.index.sectors.size();
    community.cell_begin = result.index.cells.size();
    const auto pole = pole_row(result.index, community_id);

    std::vector<uint32_t> direction_nodes;
    std::vector<uint32_t> pole_nodes;
    std::vector<float> directions;
    directions.reserve(
        checked_size(checked_multiply(community_nodes[community_id].size(), result.index.dimension,
                                      "Community direction matrix"),
                     "Community direction matrix"));
    std::vector<float> direction(result.index.dimension);
    for (const uint32_t node : community_nodes[community_id]) {
      if (compute_unit_direction(vector_row(result, node), pole, direction)) {
        direction_nodes.push_back(node);
        directions.insert(directions.end(), direction.begin(), direction.end());
      } else {
        pole_nodes.push_back(node);
      }
    }

    if (direction_nodes.size() < result.index.config.polar_direction_count) {
      community.graph_only = true;
      build_graph_only_cells(result, community_id, community_nodes[community_id]);
    } else {
      std::vector<double> weights(direction_nodes.size(), 1.0);
      spherical_kmeans_config_t kmeans_config;
      kmeans_config.direction_bin_count = result.index.config.polar_direction_count;
      kmeans_config.max_iterations = result.index.config.spherical_kmeans_iterations;
      kmeans_config.seed = static_cast<uint64_t>(
          static_cast<uint32_t>(result.index.config.partition_seed) + community_id);
      const auto trained = train_weighted_spherical_kmeans(directions, result.index.dimension,
                                                           weights, kmeans_config);
      const uint64_t axis_begin = result.index.direction_axes.size() / result.index.dimension;
      result.index.direction_axes.insert(result.index.direction_axes.end(),
                                         trained.codebook.centroids.begin(),
                                         trained.codebook.centroids.end());

      std::vector<std::vector<polar_node_t>> sector_nodes(
          result.index.config.polar_direction_count);
      std::vector<float> sector_maximum_angle(result.index.config.polar_direction_count, 0.0F);
      for (size_t position = 0; position < direction_nodes.size(); ++position) {
        const auto unit_direction =
            std::span<const float>(directions)
                .subspan(position * result.index.dimension, result.index.dimension);
        const uint32_t sector_id = find_polar_direction_bin(trained.codebook, unit_direction);
        const auto axis = std::span<const float>(trained.codebook.centroids)
                              .subspan(static_cast<size_t>(sector_id) * result.index.dimension,
                                       result.index.dimension);
        const float angle = angular_distance(unit_direction, axis);
        sector_maximum_angle[sector_id] = std::max(sector_maximum_angle[sector_id], angle);
        sector_nodes[sector_id].push_back(
            {direction_nodes[position],
             compute_radius(vector_row(result, direction_nodes[position]), pole), angle});
      }
      for (uint32_t sector_id = 0; sector_id < sector_nodes.size(); ++sector_id) {
        append_sector(result, community_id, sector_id, axis_begin + sector_id,
                      std::move(sector_nodes[sector_id]), sector_maximum_angle[sector_id]);
      }
      if (!pole_nodes.empty()) {
        std::vector<polar_node_t> pole_cell_nodes;
        pole_cell_nodes.reserve(pole_nodes.size());
        for (const uint32_t node : pole_nodes) {
          pole_cell_nodes.push_back({node, 0.0F, 0.0F});
        }
        append_sector(result, community_id, k_pole_sector_id, k_invalid_axis_index,
                      std::move(pole_cell_nodes), 0.0F);
      }
    }
    community.sector_count =
        static_cast<uint32_t>(result.index.sectors.size() - community.sector_begin);
    community.cell_count = static_cast<uint32_t>(result.index.cells.size() - community.cell_begin);
  }
}

uint64_t fnv1a_update(uint64_t hash, std::span<const uint8_t> bytes) {
  for (const uint8_t byte : bytes) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return hash;
}

uint64_t fnv1a(std::span<const uint8_t> bytes) { return fnv1a_update(k_fnv_offset_basis, bytes); }

community_polar_file_fingerprint_t fingerprint_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open fingerprint input: " + path.string());
  }
  community_polar_file_fingerprint_t fingerprint;
  fingerprint.fnv1a_hash = k_fnv_offset_basis;
  std::vector<char> buffer(1U << 20U);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count < 0) {
      throw std::runtime_error("failed to read fingerprint input: " + path.string());
    }
    fingerprint.size_bytes =
        checked_add(fingerprint.size_bytes, static_cast<uint64_t>(count), "file fingerprint");
    for (std::streamsize index = 0; index < count; ++index) {
      fingerprint.fnv1a_hash ^= static_cast<uint8_t>(buffer[static_cast<size_t>(index)]);
      fingerprint.fnv1a_hash *= k_fnv_prime;
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed to complete fingerprint input: " + path.string());
  }
  return fingerprint;
}

void append_u32_le(std::vector<uint8_t>& bytes, uint32_t value) {
  for (uint32_t shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
  }
}

void append_i32_le(std::vector<uint8_t>& bytes, int32_t value) {
  append_u32_le(bytes, std::bit_cast<uint32_t>(value));
}

void append_u64_le(std::vector<uint8_t>& bytes, uint64_t value) {
  for (uint32_t shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xFFU));
  }
}

void append_float_le(std::vector<uint8_t>& bytes, float value) {
  append_u32_le(bytes, std::bit_cast<uint32_t>(value));
}

void append_double_le(std::vector<uint8_t>& bytes, double value) {
  append_u64_le(bytes, std::bit_cast<uint64_t>(value));
}

uint32_t read_u32_le(std::span<const uint8_t> bytes, size_t& cursor) {
  if (cursor > bytes.size() || bytes.size() - cursor < sizeof(uint32_t)) {
    throw std::runtime_error("Community-Polar sidecar is truncated");
  }
  uint32_t value = 0;
  for (uint32_t shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(bytes[cursor++]) << shift;
  }
  return value;
}

int32_t read_i32_le(std::span<const uint8_t> bytes, size_t& cursor) {
  return std::bit_cast<int32_t>(read_u32_le(bytes, cursor));
}

uint64_t read_u64_le(std::span<const uint8_t> bytes, size_t& cursor) {
  if (cursor > bytes.size() || bytes.size() - cursor < sizeof(uint64_t)) {
    throw std::runtime_error("Community-Polar sidecar is truncated");
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

double read_double_le(std::span<const uint8_t> bytes, size_t& cursor) {
  return std::bit_cast<double>(read_u64_le(bytes, cursor));
}

void load_convergence_cell_hierarchy(const community_polar_build_config_t& config,
                                     community_polar_index_t& index) {
  if (config.convergence_cell_hierarchy_path.empty()) {
    return;
  }
  std::ifstream input(config.convergence_cell_hierarchy_path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("Failed to open convergence Cell hierarchy: " +
                                config.convergence_cell_hierarchy_path.string());
  }
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0 || static_cast<uint64_t>(file_size) > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("Convergence Cell hierarchy size is invalid");
  }
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> file(static_cast<size_t>(file_size));
  input.read(reinterpret_cast<char*>(file.data()), file_size);
  if (!input) {
    throw std::invalid_argument("Failed to read convergence Cell hierarchy");
  }

  size_t cursor = 0;
  const uint64_t magic = read_u64_le(file, cursor);
  const uint32_t version = read_u32_le(file, cursor);
  const uint32_t dimension = read_u32_le(file, cursor);
  const uint32_t cell_count = read_u32_le(file, cursor);
  const uint32_t node_count = read_u32_le(file, cursor);
  const uint32_t community_count = read_u32_le(file, cursor);
  const uint32_t child_count = read_u32_le(file, cursor);
  const uint64_t payload_hash = read_u64_le(file, cursor);
  const auto payload = std::span<const uint8_t>(file).subspan(cursor);
  const bool valid_payload_checksum =
      version == 1 ? fnv1a(payload) == payload_hash
                   : static_cast<uint64_t>(crc32_bytes(payload)) == payload_hash;
  if (magic != k_cell_hierarchy_profile_magic ||
      version < k_cell_hierarchy_profile_minimum_version ||
      version > k_cell_hierarchy_profile_version || dimension != index.dimension ||
      cell_count != index.cells.size() || community_count != index.communities.size() ||
      node_count == 0 || child_count == 0 || !valid_payload_checksum) {
    throw std::invalid_argument("Convergence Cell hierarchy header is incompatible or corrupt");
  }

  index.cell_hierarchy_roots.resize(community_count);
  for (auto& root : index.cell_hierarchy_roots) {
    root = read_u32_le(file, cursor);
  }
  index.cell_hierarchy_nodes.resize(node_count);
  for (auto& node : index.cell_hierarchy_nodes) {
    node.node_id = read_u32_le(file, cursor);
    node.child_begin = read_u32_le(file, cursor);
    node.child_count = read_u32_le(file, cursor);
    node.descendant_node_count = read_u64_le(file, cursor);
    node.radius = read_float_le(file, cursor);
    const uint32_t children_are_cells = read_u32_le(file, cursor);
    if (children_are_cells > 1) {
      throw std::invalid_argument("Convergence Cell hierarchy node kind is invalid");
    }
    node.children_are_cells = children_are_cells != 0;
  }
  index.cell_hierarchy_children.resize(child_count);
  for (auto& child : index.cell_hierarchy_children) {
    child = read_u32_le(file, cursor);
  }
  const uint64_t centroid_count =
      checked_multiply(node_count, dimension, "Convergence Cell hierarchy centroids");
  index.cell_hierarchy_centroids.resize(
      checked_size(centroid_count, "Convergence Cell hierarchy centroids"));
  for (auto& value : index.cell_hierarchy_centroids) {
    value = read_float_le(file, cursor);
  }
  if (cursor != file.size()) {
    throw std::invalid_argument("Convergence Cell hierarchy has trailing bytes");
  }
}

void append_zeroes(std::vector<uint8_t>& bytes, size_t count) {
  bytes.insert(bytes.end(), count, 0);
}

serialized_section_t make_config_section(const community_polar_index_t& index, uint32_t version) {
  serialized_section_t section{section_id_t::CONFIG, 1, {}};
  auto& bytes = section.bytes;
  append_u32_le(bytes, index.config.community_count);
  append_double_le(bytes, index.config.community_imbalance);
  append_u32_le(bytes, static_cast<uint32_t>(index.config.partition_quality));
  append_u32_le(bytes, index.config.partition_threads);
  append_i32_le(bytes, index.config.partition_seed);
  append_u32_le(bytes, index.config.gateway_count);
  append_u32_le(bytes, index.config.gateway_shortlist);
  append_u32_le(bytes, index.config.polar_direction_count);
  append_u32_le(bytes, index.config.cell_target_size);
  append_u32_le(bytes, index.config.block_target_size);
  append_u32_le(bytes, index.config.spherical_kmeans_iterations);
  append_u32_le(bytes, index.entry_point_id);
  append_u64_le(bytes, index.directed_base_edge_count);
  append_u64_le(bytes, index.undirected_partition_edge_count);
  append_u64_le(bytes, index.weighted_edge_cut);
  append_u32_le(bytes, static_cast<uint32_t>(index.config.partition_projection));
  append_u32_le(bytes, static_cast<uint32_t>(index.config.cell_partition));
  if (version >= 6) {
    append_u32_le(bytes, index.config.adaptive_multi_capacity ? 1U : 0U);
    append_u32_le(bytes, index.config.adaptive_policy_version);
    append_u32_le(bytes, index.config.adaptive_page_bytes);
    append_u32_le(bytes, index.config.adaptive_vector_bytes);
    append_u64_le(bytes, index.config.adaptive_lru_pages);
    append_double_le(bytes, index.config.adaptive_maximum_normalized_rms_radius);
    append_double_le(bytes, index.config.adaptive_maximum_normalized_p95_radius);
    append_double_le(bytes, index.config.adaptive_maximum_split_gain);
    append_double_le(bytes, index.config.adaptive_minimum_centroid_radius_overlap);
    append_double_le(bytes, index.config.adaptive_minimum_train_cross_child_coaccess);
    append_double_le(bytes, index.config.adaptive_maximum_unseen_pq_increase);
  }
  return section;
}

template <typename value_t, typename append_fn_t>
serialized_section_t make_scalar_section(section_id_t id, std::span<const value_t> values,
                                         append_fn_t append) {
  serialized_section_t section{id, values.size(), {}};
  for (const auto value : values) {
    append(section.bytes, value);
  }
  return section;
}

serialized_section_t make_communities_section(const community_polar_index_t& index) {
  serialized_section_t section{section_id_t::COMMUNITIES, index.communities.size(), {}};
  for (const auto& community : index.communities) {
    append_u32_le(section.bytes, community.community_id);
    append_u64_le(section.bytes, community.node_count);
    append_u64_le(section.bytes, community.packet_bytes);
    append_u64_le(section.bytes, community.internal_directed_edge_count);
    append_u64_le(section.bytes, community.incoming_directed_edge_count);
    append_u64_le(section.bytes, community.outgoing_directed_edge_count);
    append_u32_le(section.bytes, community.minimum_node_id);
    append_u32_le(section.bytes, community.navigation_hub_id);
    append_float_le(section.bytes, community.radius);
    append_u32_le(section.bytes, community.graph_only ? 1U : 0U);
    append_u64_le(section.bytes, community.sector_begin);
    append_u32_le(section.bytes, community.sector_count);
    append_u64_le(section.bytes, community.cell_begin);
    append_u32_le(section.bytes, community.cell_count);
  }
  return section;
}

serialized_section_t make_edges_section(const community_polar_index_t& index) {
  serialized_section_t section{section_id_t::COMMUNITY_EDGES, index.community_edges.size(), {}};
  for (const auto& edge : index.community_edges) {
    append_u32_le(section.bytes, edge.source_community);
    append_u32_le(section.bytes, edge.target_community);
    append_u64_le(section.bytes, edge.directed_cross_edge_count);
    append_u64_le(section.bytes, edge.gateway_begin);
    append_u32_le(section.bytes, edge.gateway_count);
  }
  return section;
}

serialized_section_t make_gateways_section(const community_polar_index_t& index) {
  serialized_section_t section{section_id_t::GATEWAYS, index.gateways.size(), {}};
  for (const auto& gateway : index.gateways) {
    append_u32_le(section.bytes, gateway.source_node);
    append_u32_le(section.bytes, gateway.target_node);
    append_u64_le(section.bytes, gateway.packet_record_index);
  }
  return section;
}

serialized_section_t make_sectors_section(const community_polar_index_t& index) {
  serialized_section_t section{section_id_t::SECTORS, index.sectors.size(), {}};
  for (const auto& sector : index.sectors) {
    append_u32_le(section.bytes, sector.community_id);
    append_u32_le(section.bytes, sector.sector_id);
    append_u64_le(section.bytes, sector.node_count);
    append_u64_le(section.bytes, sector.axis_index);
    append_float_le(section.bytes, sector.minimum_radius);
    append_float_le(section.bytes, sector.maximum_radius);
    append_float_le(section.bytes, sector.maximum_angle);
    append_u64_le(section.bytes, sector.cell_begin);
    append_u32_le(section.bytes, sector.cell_count);
  }
  return section;
}

serialized_section_t make_cells_section(const community_polar_index_t& index, uint32_t version) {
  serialized_section_t section{section_id_t::CELLS, index.cells.size(), {}};
  for (const auto& cell : index.cells) {
    append_u32_le(section.bytes, cell.cell_id);
    append_u32_le(section.bytes, cell.community_id);
    append_u32_le(section.bytes, cell.sector_id);
    append_u32_le(section.bytes, cell.radial_cell_id);
    append_u64_le(section.bytes, cell.node_count);
    if (version >= 6) {
      append_u32_le(section.bytes, cell.capacity_class);
    }
    append_float_le(section.bytes, cell.minimum_radius);
    append_float_le(section.bytes, cell.maximum_radius);
    append_float_le(section.bytes, cell.maximum_angle);
    append_u64_le(section.bytes, cell.block_begin);
    append_u32_le(section.bytes, cell.block_count);
  }
  return section;
}

serialized_section_t make_blocks_section(const community_polar_index_t& index) {
  serialized_section_t section{section_id_t::BLOCKS, index.blocks.size(), {}};
  for (const auto& block : index.blocks) {
    append_u32_le(section.bytes, block.block_id);
    append_u32_le(section.bytes, block.cell_id);
    append_u32_le(section.bytes, block.node_count);
    append_u64_le(section.bytes, block.packet_record_begin);
    append_float_le(section.bytes, block.minimum_radius);
    append_float_le(section.bytes, block.maximum_radius);
    append_float_le(section.bytes, block.maximum_angle);
    append_float_le(section.bytes, block.maximum_pq_reconstruction_error);
  }
  return section;
}

serialized_section_t make_cell_hierarchy_nodes_section(const community_polar_index_t& index) {
  serialized_section_t section{
      section_id_t::CELL_HIERARCHY_NODES, index.cell_hierarchy_nodes.size(), {}};
  for (const auto& node : index.cell_hierarchy_nodes) {
    append_u32_le(section.bytes, node.node_id);
    append_u32_le(section.bytes, node.child_begin);
    append_u32_le(section.bytes, node.child_count);
    append_u64_le(section.bytes, node.descendant_node_count);
    append_float_le(section.bytes, node.radius);
    append_u32_le(section.bytes, node.children_are_cells ? 1U : 0U);
  }
  return section;
}

std::vector<serialized_section_t> make_sections(const community_polar_index_t& index,
                                                uint32_t version) {
  std::vector<serialized_section_t> sections;
  sections.reserve(k_section_count);
  sections.push_back(make_config_section(index, version));
  sections.push_back(make_scalar_section<uint32_t>(section_id_t::NODE_TO_COMMUNITY,
                                                   index.node_to_community, append_u32_le));
  sections.push_back(
      make_scalar_section<uint32_t>(section_id_t::NODE_TO_CELL, index.node_to_cell, append_u32_le));
  sections.push_back(make_scalar_section<uint64_t>(section_id_t::NODE_TO_PACKET,
                                                   index.node_to_packet_record, append_u64_le));
  sections.push_back(make_communities_section(index));
  sections.push_back(make_scalar_section<float>(section_id_t::COMMUNITY_POLES,
                                                index.community_poles, append_float_le));
  sections.push_back(make_scalar_section<uint64_t>(section_id_t::COMMUNITY_EDGE_OFFSETS,
                                                   index.community_edge_offsets, append_u64_le));
  sections.push_back(make_edges_section(index));
  sections.push_back(make_gateways_section(index));
  sections.push_back(make_scalar_section<float>(section_id_t::DIRECTION_AXES, index.direction_axes,
                                                append_float_le));
  sections.push_back(make_sectors_section(index));
  sections.push_back(make_cells_section(index, version));
  sections.push_back(make_blocks_section(index));
  sections.push_back(
      serialized_section_t{section_id_t::PACKETS, index.point_count, index.packet_payload});
  sections.push_back(make_cell_hierarchy_nodes_section(index));
  sections.push_back(make_scalar_section<uint32_t>(section_id_t::CELL_HIERARCHY_ROOTS,
                                                   index.cell_hierarchy_roots, append_u32_le));
  sections.push_back(make_scalar_section<uint32_t>(section_id_t::CELL_HIERARCHY_CHILDREN,
                                                   index.cell_hierarchy_children, append_u32_le));
  sections.push_back(make_scalar_section<float>(section_id_t::CELL_HIERARCHY_CENTROIDS,
                                                index.cell_hierarchy_centroids, append_float_le));
  return sections;
}

const section_directory_entry_t& find_section(std::span<const section_directory_entry_t> directory,
                                              section_id_t id) {
  const auto found = std::find_if(directory.begin(), directory.end(),
                                  [id](const auto& entry) { return entry.id == id; });
  if (found == directory.end()) {
    throw std::runtime_error("Community-Polar sidecar is missing a required section");
  }
  return *found;
}

std::span<const uint8_t> section_bytes(std::span<const uint8_t> file,
                                       const section_directory_entry_t& section) {
  if (section.offset > file.size() || section.length > file.size() - section.offset) {
    throw std::runtime_error("Community-Polar section lies outside the file");
  }
  return file.subspan(static_cast<size_t>(section.offset), static_cast<size_t>(section.length));
}

template <typename value_t, typename read_fn_t>
std::vector<value_t> read_scalar_section(std::span<const uint8_t> bytes, uint64_t count,
                                         size_t scalar_size, read_fn_t read) {
  if (checked_multiply(count, scalar_size, "Community-Polar scalar section") != bytes.size()) {
    throw std::runtime_error("Community-Polar scalar section has an invalid length");
  }
  std::vector<value_t> values(checked_size(count, "Community-Polar scalar count"));
  size_t cursor = 0;
  for (auto& value : values) {
    value = read(bytes, cursor);
  }
  return values;
}

void validate_cell_hierarchy(const community_polar_index_t& index) {
  const bool has_nodes = !index.cell_hierarchy_nodes.empty();
  const bool has_roots = !index.cell_hierarchy_roots.empty();
  const bool has_children = !index.cell_hierarchy_children.empty();
  const bool has_centroids = !index.cell_hierarchy_centroids.empty();
  if (!has_nodes && !has_roots && !has_children && !has_centroids) {
    return;
  }
  if (!has_nodes || !has_roots || !has_children || !has_centroids ||
      (index.config.cell_partition != community_cell_partition_t::GLOBAL_GEOMETRIC &&
       index.config.cell_partition != community_cell_partition_t::GLOBAL_GRAPH_LOCAL &&
       index.config.cell_partition != community_cell_partition_t::GRAPH_LOCAL &&
       index.config.cell_partition != community_cell_partition_t::CONVERGENCE_COACCESS) ||
      index.cell_hierarchy_roots.size() != index.communities.size() ||
      index.cell_hierarchy_centroids.size() != checked_multiply(index.cell_hierarchy_nodes.size(),
                                                                index.dimension,
                                                                "Cell hierarchy centroids")) {
    throw std::runtime_error("Community-Polar Cell hierarchy metadata is incomplete");
  }

  uint64_t expected_child = 0;
  for (uint32_t node_id = 0; node_id < index.cell_hierarchy_nodes.size(); ++node_id) {
    const auto& node = index.cell_hierarchy_nodes[node_id];
    if (node.node_id != node_id || node.child_count == 0 || node.child_begin != expected_child ||
        checked_add(node.child_begin, node.child_count, "Cell hierarchy child range") >
            index.cell_hierarchy_children.size() ||
        node.descendant_node_count == 0 || !std::isfinite(node.radius) || node.radius < 0.0F) {
      throw std::runtime_error("Community-Polar Cell hierarchy node is invalid");
    }
    expected_child = checked_add(expected_child, node.child_count, "Cell hierarchy children");
    const size_t centroid_begin = static_cast<size_t>(node_id) * index.dimension;
    if (std::any_of(index.cell_hierarchy_centroids.begin() + centroid_begin,
                    index.cell_hierarchy_centroids.begin() + centroid_begin + index.dimension,
                    [](float value) { return !std::isfinite(value); })) {
      throw std::runtime_error("Community-Polar Cell hierarchy centroid is invalid");
    }
  }
  if (expected_child != index.cell_hierarchy_children.size()) {
    throw std::runtime_error("Community-Polar Cell hierarchy child ranges are incomplete");
  }

  constexpr uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
  std::vector<uint8_t> node_state(index.cell_hierarchy_nodes.size(), 0);
  std::vector<uint32_t> parent_count(index.cell_hierarchy_nodes.size(), 0);
  std::vector<uint8_t> cell_seen(index.cells.size(), 0);
  auto centroid = [&](uint32_t node_id) {
    return std::span<const float>(index.cell_hierarchy_centroids)
        .subspan(static_cast<size_t>(node_id) * index.dimension, index.dimension);
  };
  auto cell_centroid = [&](uint32_t cell_id) {
    return std::span<const float>(index.direction_axes)
        .subspan(static_cast<size_t>(cell_id) * index.dimension, index.dimension);
  };
  auto visit = [&](auto&& self, uint32_t node_id,
                   uint32_t community_id) -> std::pair<uint64_t, std::vector<uint32_t>> {
    if (node_id >= index.cell_hierarchy_nodes.size() || node_state[node_id] != 0) {
      throw std::runtime_error("Community-Polar Cell hierarchy contains a cycle or duplicate node");
    }
    node_state[node_id] = 1;
    const auto& node = index.cell_hierarchy_nodes[node_id];
    uint64_t descendant_count = 0;
    std::vector<uint32_t> descendant_cells;
    descendant_cells.reserve(node.children_are_cells ? node.child_count : 1);
    for (uint64_t offset = node.child_begin; offset < node.child_begin + node.child_count;
         ++offset) {
      const uint32_t child = index.cell_hierarchy_children[offset];
      if (node.children_are_cells) {
        if (child >= index.cells.size() || cell_seen[child] != 0 ||
            index.cells[child].community_id != community_id) {
          throw std::runtime_error("Community-Polar Cell hierarchy leaf is invalid");
        }
        cell_seen[child] = 1;
        descendant_count = checked_add(descendant_count, index.cells[child].node_count,
                                       "Cell hierarchy descendant nodes");
        descendant_cells.push_back(child);
      } else {
        if (child >= index.cell_hierarchy_nodes.size() || ++parent_count[child] != 1) {
          throw std::runtime_error("Community-Polar Cell hierarchy internal edge is invalid");
        }
        auto [child_count, child_cells] = self(self, child, community_id);
        descendant_count =
            checked_add(descendant_count, child_count, "Cell hierarchy descendant nodes");
        descendant_cells.insert(descendant_cells.end(), child_cells.begin(), child_cells.end());
      }
    }
    if (descendant_count != node.descendant_node_count) {
      throw std::runtime_error("Community-Polar Cell hierarchy descendant count is invalid");
    }
    const auto center = centroid(node_id);
    for (const uint32_t cell_id : descendant_cells) {
      const auto leaf_center = cell_centroid(cell_id);
      double squared_distance = 0.0;
      for (uint32_t dimension = 0; dimension < index.dimension; ++dimension) {
        const double difference = static_cast<double>(center[dimension]) - leaf_center[dimension];
        squared_distance += difference * difference;
      }
      const double tolerance = std::max(1.0e-4, static_cast<double>(node.radius) * 1.0e-5);
      if (std::sqrt(squared_distance) > static_cast<double>(node.radius) + tolerance) {
        throw std::runtime_error("Community-Polar Cell hierarchy radius is not conservative");
      }
    }
    node_state[node_id] = 2;
    return {descendant_count, std::move(descendant_cells)};
  };

  for (uint32_t community = 0; community < index.communities.size(); ++community) {
    const uint32_t root = index.cell_hierarchy_roots[community];
    const bool owns_cells = index.communities[community].cell_count != 0;
    if (!owns_cells) {
      if (root != invalid_id) {
        throw std::runtime_error("Community-Polar empty Community has a Cell hierarchy root");
      }
      continue;
    }
    if (root >= index.cell_hierarchy_nodes.size() || ++parent_count[root] != 1) {
      throw std::runtime_error("Community-Polar Cell hierarchy root is invalid");
    }
    const auto [descendant_count, descendant_cells] = visit(visit, root, community);
    if (descendant_count != index.point_count &&
        descendant_count != index.communities[community].node_count) {
      throw std::runtime_error("Community-Polar Cell hierarchy root coverage is invalid");
    }
    static_cast<void>(descendant_cells);
  }
  if (std::find(node_state.begin(), node_state.end(), 0) != node_state.end() ||
      std::find(cell_seen.begin(), cell_seen.end(), 0) != cell_seen.end() ||
      std::find(parent_count.begin(), parent_count.end(), 0) != parent_count.end()) {
    throw std::runtime_error("Community-Polar Cell hierarchy does not cover its nodes and Cells");
  }
}

void validate_loaded_index(const community_polar_index_t& index) {
  validate_community_polar_build_config(index.config);
  if (index.config.adaptive_multi_capacity &&
      index.config.adaptive_vector_bytes !=
          checked_multiply(index.dimension, sizeof(float), "Adaptive vector bytes")) {
    throw std::runtime_error("Adaptive policy vector bytes disagree with the Base dimension");
  }
  if (index.point_count == 0 || index.dimension == 0 || index.pq_code_width == 0 ||
      index.entry_point_id >= index.point_count ||
      index.communities.size() != index.config.community_count ||
      index.node_to_community.size() != index.point_count ||
      index.node_to_cell.size() != index.point_count ||
      index.node_to_packet_record.size() != index.point_count ||
      index.community_poles.size() !=
          checked_multiply(index.communities.size(), index.dimension, "Community poles") ||
      index.community_edge_offsets.size() != index.communities.size() + 1) {
    throw std::runtime_error("Community-Polar sidecar has inconsistent global metadata");
  }
  if (index.community_edge_offsets.front() != 0 ||
      index.community_edge_offsets.back() != index.community_edges.size() ||
      !std::is_sorted(index.community_edge_offsets.begin(), index.community_edge_offsets.end())) {
    throw std::runtime_error("Community-Polar Community CSR offsets are invalid");
  }
  const uint64_t record_bytes = sizeof(uint32_t) + index.pq_code_width;
  if (checked_multiply(index.point_count, record_bytes, "Community-Polar packet bytes") !=
      index.packet_payload.size()) {
    throw std::runtime_error("Community-Polar packet payload has an invalid length");
  }
  if (index.direction_axes.size() % index.dimension != 0) {
    throw std::runtime_error("Community-Polar direction axes have an invalid length");
  }
  if (index.config.cell_partition != community_cell_partition_t::POLAR_RADIAL &&
      index.direction_axes.size() !=
          checked_multiply(index.cells.size(), index.dimension, "Cell centroids")) {
    throw std::runtime_error("Community-Polar Cell centroids are invalid");
  }

  std::vector<uint64_t> community_members(index.communities.size(), 0);
  std::vector<uint32_t> community_minimum(index.communities.size(),
                                          std::numeric_limits<uint32_t>::max());
  for (uint32_t node = 0; node < index.point_count; ++node) {
    if (index.node_to_community[node] >= index.communities.size() ||
        index.node_to_cell[node] >= index.cells.size()) {
      throw std::runtime_error("Community-Polar node membership is invalid");
    }
    ++community_members[index.node_to_community[node]];
    community_minimum[index.node_to_community[node]] =
        std::min(community_minimum[index.node_to_community[node]], node);
  }
  uint64_t expected_sector = 0;
  uint64_t expected_cell = 0;
  for (uint32_t community = 0; community < index.communities.size(); ++community) {
    const auto& record = index.communities[community];
    if (record.community_id != community || record.node_count == 0 ||
        community_members[community] != record.node_count ||
        record.minimum_node_id != community_minimum[community] ||
        record.navigation_hub_id >= index.point_count ||
        index.node_to_community[record.minimum_node_id] != community ||
        index.node_to_community[record.navigation_hub_id] != community ||
        record.packet_bytes !=
            checked_multiply(record.node_count, record_bytes, "Community packet bytes") ||
        record.sector_begin != expected_sector || record.cell_begin != expected_cell ||
        checked_add(record.sector_begin, record.sector_count, "Community Sector range") >
            index.sectors.size() ||
        checked_add(record.cell_begin, record.cell_count, "Community Cell range") >
            index.cells.size() ||
        !std::isfinite(record.radius) || record.radius < 0.0F) {
      throw std::runtime_error("Community-Polar Community record is invalid");
    }
    expected_sector = checked_add(expected_sector, record.sector_count, "Community Sectors");
    expected_cell = checked_add(expected_cell, record.cell_count, "Community Cells");
  }
  if (expected_sector != index.sectors.size() || expected_cell != index.cells.size()) {
    throw std::runtime_error("Community-Polar Community ranges are incomplete");
  }
  std::vector<uint32_t> sector_to_community(index.sectors.size(),
                                            std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> cell_to_community(index.cells.size(), std::numeric_limits<uint32_t>::max());
  for (uint32_t community = 0; community < index.communities.size(); ++community) {
    const auto& record = index.communities[community];
    std::fill_n(sector_to_community.begin() + static_cast<size_t>(record.sector_begin),
                record.sector_count, community);
    std::fill_n(cell_to_community.begin() + static_cast<size_t>(record.cell_begin),
                record.cell_count, community);
  }

  uint64_t expected_gateway = 0;
  for (uint32_t source = 0; source < index.communities.size(); ++source) {
    uint32_t previous_target = 0;
    bool has_previous_target = false;
    for (uint64_t edge_id = index.community_edge_offsets[source];
         edge_id < index.community_edge_offsets[source + 1]; ++edge_id) {
      const auto& record = index.community_edges[edge_id];
      if (record.source_community != source ||
          record.target_community >= index.communities.size() ||
          record.source_community == record.target_community ||
          record.directed_cross_edge_count == 0 || record.gateway_count == 0 ||
          record.gateway_count > index.config.gateway_count ||
          record.gateway_begin != expected_gateway ||
          checked_add(record.gateway_begin, record.gateway_count, "Gateway range") >
              index.gateways.size() ||
          (has_previous_target && record.target_community <= previous_target)) {
        throw std::runtime_error("Community-Polar Community edge is invalid");
      }
      std::vector<uint32_t> gateway_targets;
      gateway_targets.reserve(record.gateway_count);
      for (uint64_t gateway_id = record.gateway_begin;
           gateway_id < record.gateway_begin + record.gateway_count; ++gateway_id) {
        const auto& gateway = index.gateways[gateway_id];
        if (gateway.source_node >= index.point_count || gateway.target_node >= index.point_count ||
            index.node_to_community[gateway.source_node] != record.source_community ||
            index.node_to_community[gateway.target_node] != record.target_community ||
            gateway.packet_record_index != index.node_to_packet_record[gateway.target_node]) {
          throw std::runtime_error("Community-Polar Gateway record is invalid");
        }
        gateway_targets.push_back(gateway.target_node);
      }
      std::sort(gateway_targets.begin(), gateway_targets.end());
      if (std::adjacent_find(gateway_targets.begin(), gateway_targets.end()) !=
          gateway_targets.end()) {
        throw std::runtime_error("Community-Polar Gateway targets are not unique");
      }
      expected_gateway = checked_add(expected_gateway, record.gateway_count, "Gateways");
      previous_target = record.target_community;
      has_previous_target = true;
    }
  }
  if (expected_gateway != index.gateways.size()) {
    throw std::runtime_error("Community-Polar Gateway ranges are incomplete");
  }

  std::vector<uint32_t> cell_to_sector(index.cells.size(), std::numeric_limits<uint32_t>::max());
  uint64_t sector_cell_cursor = 0;
  for (uint32_t sector_id = 0; sector_id < index.sectors.size(); ++sector_id) {
    const auto& record = index.sectors[sector_id];
    const bool is_pole_sector = record.sector_id == k_pole_sector_id;
    if (record.community_id != sector_to_community[sector_id] || record.node_count == 0 ||
        record.cell_count == 0 || record.cell_begin != sector_cell_cursor ||
        checked_add(record.cell_begin, record.cell_count, "Sector Cell range") >
            index.cells.size() ||
        !std::isfinite(record.minimum_radius) || !std::isfinite(record.maximum_radius) ||
        !std::isfinite(record.maximum_angle) || record.minimum_radius < 0.0F ||
        record.minimum_radius > record.maximum_radius || record.maximum_angle < 0.0F ||
        record.maximum_angle > std::numbers::pi_v<float> + 1.0e-4F ||
        (is_pole_sector && record.axis_index != k_invalid_axis_index) ||
        (!is_pole_sector && (record.sector_id >= index.config.polar_direction_count ||
                             record.axis_index >= index.direction_axes.size() / index.dimension))) {
      throw std::runtime_error("Community-Polar Sector record is invalid");
    }
    for (uint64_t cell = record.cell_begin; cell < record.cell_begin + record.cell_count; ++cell) {
      cell_to_sector[cell] = sector_id;
    }
    sector_cell_cursor = checked_add(sector_cell_cursor, record.cell_count, "Sector Cells");
  }
  if (sector_cell_cursor != index.cells.size()) {
    throw std::runtime_error("Community-Polar Sector ranges are incomplete");
  }

  std::vector<uint32_t> block_to_cell(index.blocks.size(), std::numeric_limits<uint32_t>::max());
  std::vector<uint64_t> sector_members(index.sectors.size(), 0);
  uint64_t cell_block_cursor = 0;
  for (uint32_t cell = 0; cell < index.cells.size(); ++cell) {
    const auto& record = index.cells[cell];
    const auto& sector = index.sectors[cell_to_sector[cell]];
    const bool capacity_is_valid = index.config.adaptive_multi_capacity
                                       ? is_adaptive_capacity_class(record.capacity_class)
                                       : record.capacity_class == index.config.cell_target_size;
    if (record.cell_id != cell || record.community_id != cell_to_community[cell] ||
        record.community_id != sector.community_id || record.sector_id != sector.sector_id ||
        record.node_count == 0 || !capacity_is_valid || record.node_count > record.capacity_class ||
        record.block_count == 0 || record.block_begin != cell_block_cursor ||
        checked_add(record.block_begin, record.block_count, "Cell Block range") >
            index.blocks.size() ||
        !std::isfinite(record.minimum_radius) || !std::isfinite(record.maximum_radius) ||
        !std::isfinite(record.maximum_angle) || record.minimum_radius < 0.0F ||
        record.minimum_radius > record.maximum_radius || record.maximum_angle < 0.0F ||
        record.maximum_angle > sector.maximum_angle + 1.0e-4F) {
      throw std::runtime_error("Community-Polar Cell record is invalid");
    }
    for (uint64_t block = record.block_begin; block < record.block_begin + record.block_count;
         ++block) {
      block_to_cell[block] = cell;
    }
    sector_members[cell_to_sector[cell]] =
        checked_add(sector_members[cell_to_sector[cell]], record.node_count, "Sector members");
    cell_block_cursor = checked_add(cell_block_cursor, record.block_count, "Cell Blocks");
  }
  if (cell_block_cursor != index.blocks.size()) {
    throw std::runtime_error("Community-Polar Cell ranges are incomplete");
  }

  std::vector<uint32_t> packet_to_cell(checked_size(index.point_count, "Community-Polar packets"),
                                       std::numeric_limits<uint32_t>::max());
  std::vector<uint64_t> cell_members(index.cells.size(), 0);
  uint64_t packet_cursor = 0;
  for (uint32_t block = 0; block < index.blocks.size(); ++block) {
    const auto& record = index.blocks[block];
    const auto& cell = index.cells[block_to_cell[block]];
    if (record.block_id != block || record.cell_id != cell.cell_id || record.node_count == 0 ||
        record.node_count > index.config.block_target_size ||
        record.packet_record_begin != packet_cursor ||
        checked_add(record.packet_record_begin, record.node_count, "Block packet range") >
            index.point_count ||
        !std::isfinite(record.minimum_radius) || !std::isfinite(record.maximum_radius) ||
        !std::isfinite(record.maximum_angle) || record.minimum_radius < cell.minimum_radius ||
        record.minimum_radius > record.maximum_radius ||
        record.maximum_radius > cell.maximum_radius || record.maximum_angle < 0.0F ||
        record.maximum_angle > cell.maximum_angle + 1.0e-4F ||
        !std::isfinite(record.maximum_pq_reconstruction_error) ||
        record.maximum_pq_reconstruction_error < 0.0F) {
      throw std::runtime_error("Community-Polar Block record is invalid");
    }
    for (uint64_t packet = record.packet_record_begin;
         packet < record.packet_record_begin + record.node_count; ++packet) {
      packet_to_cell[packet] = record.cell_id;
    }
    cell_members[record.cell_id] =
        checked_add(cell_members[record.cell_id], record.node_count, "Cell members");
    packet_cursor = checked_add(packet_cursor, record.node_count, "Block packets");
  }
  if (packet_cursor != index.point_count) {
    throw std::runtime_error("Community-Polar Block ranges are incomplete");
  }
  for (uint32_t sector = 0; sector < index.sectors.size(); ++sector) {
    if (sector_members[sector] != index.sectors[sector].node_count) {
      throw std::runtime_error("Community-Polar Sector membership count is invalid");
    }
  }
  for (uint32_t cell = 0; cell < index.cells.size(); ++cell) {
    if (cell_members[cell] != index.cells[cell].node_count) {
      throw std::runtime_error("Community-Polar Cell membership count is invalid");
    }
  }

  if (index.config.adaptive_multi_capacity &&
      (index.cell_hierarchy_nodes.empty() || index.cell_hierarchy_roots.empty() ||
       index.cell_hierarchy_children.empty() || index.cell_hierarchy_centroids.empty())) {
    throw std::runtime_error("Adaptive multi-capacity Cells require a persisted hierarchy");
  }
  validate_cell_hierarchy(index);

  std::vector<bool> seen_nodes(checked_size(index.point_count, "Community-Polar points"), false);
  for (uint64_t packet = 0; packet < index.point_count; ++packet) {
    const uint64_t byte_offset = checked_multiply(packet, record_bytes, "packet offset");
    size_t cursor = static_cast<size_t>(byte_offset);
    const uint32_t node = read_u32_le(index.packet_payload, cursor);
    if (node >= index.point_count || seen_nodes[node] ||
        index.node_to_packet_record[node] != packet ||
        index.node_to_cell[node] != packet_to_cell[packet] ||
        (index.config.cell_partition != community_cell_partition_t::GLOBAL_GEOMETRIC &&
         index.config.cell_partition != community_cell_partition_t::GLOBAL_GRAPH_LOCAL &&
         index.cells[packet_to_cell[packet]].community_id != index.node_to_community[node])) {
      throw std::runtime_error("Community-Polar packet node mapping is invalid");
    }
    seen_nodes[node] = true;
  }
}

} // namespace

void repack_graph_local_cells(community_polar_index_t& index,
                              const community_polar_build_config_t& config) {
  repack_adjacent_graph_local_cells(index, config);
}

void repack_pq_locality_cells(community_polar_index_t& index,
                              const std::filesystem::path& profile_path,
                              const community_polar_build_config_t& config) {
  repack_pq_locality_cells_impl(index, profile_path, config);
  if (!config.convergence_cell_hierarchy_path.empty()) {
    load_convergence_cell_hierarchy(config, index);
  } else {
    build_contiguous_cell_hierarchy(index);
  }
  validate_loaded_index(index);
}

void validate_community_polar_build_config(const community_polar_build_config_t& config) {
  if (!config.is_enabled()) {
    throw std::invalid_argument("Community count must be greater than zero");
  }
  if (!std::isfinite(config.community_imbalance) || config.community_imbalance < 0.0 ||
      config.community_imbalance >= 1.0) {
    throw std::invalid_argument("Community imbalance must be finite and within [0, 1)");
  }
  if (config.partition_quality != community_partition_quality_t::FAST &&
      config.partition_quality != community_partition_quality_t::QUALITY) {
    throw std::invalid_argument("Community partition quality is invalid");
  }
  static_cast<void>(projection_degree(config.partition_projection));
  if (config.cell_partition != community_cell_partition_t::POLAR_RADIAL &&
      config.cell_partition != community_cell_partition_t::GRAPH_LOCAL &&
      config.cell_partition != community_cell_partition_t::CONVERGENCE_COACCESS &&
      config.cell_partition != community_cell_partition_t::GLOBAL_GEOMETRIC &&
      config.cell_partition != community_cell_partition_t::GLOBAL_GRAPH_LOCAL) {
    throw std::invalid_argument("Community Cell partition is invalid");
  }
  if (config.gateway_count == 0 || config.gateway_shortlist < config.gateway_count) {
    throw std::invalid_argument("Gateway shortlist must be at least the positive Gateway count");
  }
  if (config.polar_direction_count != 16 && config.polar_direction_count != 32 &&
      config.polar_direction_count != 64) {
    throw std::invalid_argument("Polar direction count must be 16, 32, or 64");
  }
  if (config.cell_target_size == 0 || config.block_target_size == 0 ||
      config.spherical_kmeans_iterations == 0) {
    throw std::invalid_argument("Cell, Block, and Spherical K-means sizes must be positive");
  }
  if (!config.gateway_selection_source_path.empty() && config.gateway_selection_path.empty()) {
    throw std::invalid_argument("Gateway selection source requires a selection CSV");
  }
  if (!config.convergence_cell_hierarchy_path.empty() &&
      (config.cell_partition != community_cell_partition_t::GLOBAL_GEOMETRIC ||
       config.convergence_cell_profile_path.empty())) {
    throw std::invalid_argument(
        "A convergence Cell hierarchy requires a global-geometric Cell profile");
  }
  if (config.cell_hierarchy_branching < 2 || config.cell_hierarchy_leaf_size == 0 ||
      (config.build_contiguous_cell_hierarchy &&
       (!config.convergence_cell_hierarchy_path.empty() ||
        config.cell_partition == community_cell_partition_t::POLAR_RADIAL))) {
    throw std::invalid_argument("Contiguous Cell hierarchy build policy is invalid");
  }
  if (!config.adaptive_multi_capacity) {
    if (config.adaptive_policy_version != 0 || !config.convergence_cell_capacity_path.empty()) {
      throw std::invalid_argument(
          "Adaptive capacity policy fields require adaptive multi-capacity Cells");
    }
    return;
  }
  const std::array<double, 6> policy_thresholds = {
      config.adaptive_maximum_normalized_rms_radius,
      config.adaptive_maximum_normalized_p95_radius,
      config.adaptive_maximum_split_gain,
      config.adaptive_minimum_centroid_radius_overlap,
      config.adaptive_minimum_train_cross_child_coaccess,
      config.adaptive_maximum_unseen_pq_increase};
  if (config.cell_partition != community_cell_partition_t::GLOBAL_GEOMETRIC ||
      config.cell_target_size != 128 || config.adaptive_policy_version != 1 ||
      config.adaptive_page_bytes != 4096 || config.adaptive_vector_bytes == 0 ||
      std::any_of(policy_thresholds.begin(), policy_thresholds.end(),
                  [](double value) { return !std::isfinite(value) || value < 0.0; }) ||
      config.adaptive_minimum_centroid_radius_overlap > 1.0 ||
      config.adaptive_minimum_train_cross_child_coaccess > 1.0) {
    throw std::invalid_argument("Adaptive multi-capacity Cell policy is invalid");
  }
}

community_polar_build_result_t
build_community_polar_index(const post_link_graph_view_t& graph,
                            std::span<const rcni_node_importance_t> importance,
                            const community_polar_build_config_t& config) {
  validate_community_polar_build_config(config);
  if (config.adaptive_multi_capacity && (config.convergence_cell_profile_path.empty() ||
                                         config.convergence_cell_hierarchy_path.empty() ||
                                         config.convergence_cell_capacity_path.empty())) {
    throw std::invalid_argument(
        "Adaptive multi-capacity construction requires assignment, hierarchy, and capacity "
        "artifacts");
  }
  if (config.community_count > graph.point_count()) {
    throw std::invalid_argument("Community count must not exceed the Base point count");
  }
  const auto node_importance = importance_by_node(importance, graph.point_count());
  const auto gateway_selection = load_gateway_selection(config);
  const auto convergence_assignments = load_convergence_cell_profile(config, graph.point_count());

  community_polar_build_result_t result;
  result.index.config = config;
  result.index.config.partition_threads = effective_thread_count(config.partition_threads);
  result.index.point_count = graph.point_count();
  result.index.dimension = graph.dimension();
  result.index.entry_point_id = graph.entry_point_id();
  result.index.node_to_community.resize(checked_size(graph.point_count(), "Base point count"));
  result.index.node_to_cell.assign(checked_size(graph.point_count(), "Base point count"),
                                   std::numeric_limits<uint32_t>::max());
  result.index.node_to_packet_record.assign(checked_size(graph.point_count(), "Base point count"),
                                            std::numeric_limits<uint64_t>::max());
  result.original_vectors.resize(
      checked_size(checked_multiply(graph.point_count(), graph.dimension(), "Base vector matrix"),
                   "Base vector matrix"));
  const uint32_t thread_count = result.index.config.partition_threads;
#pragma omp parallel for schedule(static) num_threads(thread_count)
  for (int64_t signed_node = 0; signed_node < static_cast<int64_t>(graph.point_count());
       ++signed_node) {
    const uint32_t node = static_cast<uint32_t>(signed_node);
    graph.copy_vector(
        node, std::span<float>(result.original_vectors)
                  .subspan(static_cast<size_t>(node) * graph.dimension(), graph.dimension()));
  }

  partition_graph_t partition_graph;
  try {
    partition_graph = build_partition_graph(graph, config.partition_projection,
                                            result.index.config.partition_threads);
  } catch (const std::exception& error) {
    throw std::runtime_error("Community partition projection failed: " + std::string(error.what()));
  }
  result.index.directed_base_edge_count = partition_graph.directed_edge_count;
  result.index.undirected_partition_edge_count = partition_graph.undirected_edge_count;
  kaminpar::KaMinPar::reseed(config.partition_seed);
  auto context =
      config.partition_quality == community_partition_quality_t::QUALITY
          ? kaminpar::shm::create_strong_context()
          : (result.index.config.partition_threads == 1 ? kaminpar::shm::create_default_context()
                                                        : kaminpar::shm::create_fast_context());
  kaminpar::KaMinPar partitioner(static_cast<int>(result.index.config.partition_threads),
                                 std::move(context));
  partitioner.set_output_level(kaminpar::OutputLevel::QUIET);
  std::vector<kaminpar::shm::BlockID> raw_partition(
      checked_size(graph.point_count(), "Community partition"));
  std::vector<kaminpar::shm::BlockWeight> minimum_block_weights(config.community_count, 1);
  kaminpar::shm::EdgeWeight edge_cut = 0;
  try {
    partitioner.borrow_and_mutate_graph(partition_graph.offsets, partition_graph.neighbors,
                                        partition_graph.node_weights, partition_graph.edge_weights);
    partitioner.set_k(config.community_count);
    partitioner.set_uniform_max_block_weights(config.community_imbalance);
    // A maximum-weight constraint alone permits empty blocks whenever k - 1 blocks have enough
    // aggregate capacity for the whole graph. The sidecar requires exactly k nonempty Communities,
    // so enforce one unit-weight node per block inside KaMinPar instead of repairing its output and
    // invalidating the reported cut or balance guarantees.
    partitioner.set_absolute_min_block_weights(minimum_block_weights);
    edge_cut = partitioner.compute_partition(raw_partition);
  } catch (const std::exception& error) {
    throw std::runtime_error("KaMinPar Community partition failed: " + std::string(error.what()));
  }
  if (edge_cut < 0) {
    throw std::runtime_error("KaMinPar returned a negative weighted edge cut");
  }
  result.index.weighted_edge_cut = static_cast<uint64_t>(edge_cut);
  result.index.node_to_community = canonicalize_partition(raw_partition, config.community_count);

  std::vector<std::vector<uint32_t>> community_nodes(config.community_count);
  for (uint32_t node = 0; node < graph.point_count(); ++node) {
    community_nodes[result.index.node_to_community[node]].push_back(node);
  }

  result.index.communities.resize(config.community_count);
  result.index.community_poles.resize(
      checked_size(checked_multiply(config.community_count, graph.dimension(), "Community poles"),
                   "Community poles"));
  for (uint32_t community_id = 0; community_id < config.community_count; ++community_id) {
    auto& community = result.index.communities[community_id];
    const auto& nodes = community_nodes[community_id];
    if (nodes.empty()) {
      throw std::runtime_error("canonical Community is unexpectedly empty");
    }
    community.community_id = community_id;
    community.node_count = nodes.size();
    community.minimum_node_id = nodes.front();
    community.navigation_hub_id = nodes.front();
    for (const uint32_t node : nodes) {
      if (node_importance[node] > node_importance[community.navigation_hub_id] ||
          (node_importance[node] == node_importance[community.navigation_hub_id] &&
           node < community.navigation_hub_id)) {
        community.navigation_hub_id = node;
      }
    }
    for (uint32_t dimension = 0; dimension < graph.dimension(); ++dimension) {
      double sum = 0.0;
      for (const uint32_t node : nodes) {
        sum += vector_row(result, node)[dimension];
      }
      result.index
          .community_poles[static_cast<size_t>(community_id) * graph.dimension() + dimension] =
          static_cast<float>(sum / nodes.size());
    }
    const auto pole = pole_row(result.index, community_id);
    for (const uint32_t node : nodes) {
      community.radius = std::max(community.radius, compute_radius(vector_row(result, node), pole));
    }
  }

  if (gateway_selection.empty()) {
    // Gateway selection only consumes the best `gateway_shortlist` unique target nodes for each
    // directed Community pair. Retaining every cross edge and globally sorting it costs O(E)
    // auxiliary memory, which is prohibitive at SIFT100M. Build a bounded shortlist per worker,
    // then merge those lists deterministically. A target absent from a worker's top-k cannot be in
    // the global top-k, so this produces the same Gateway candidates as the full sort.
    struct gateway_seed_t {
      uint32_t source_node = 0;
      uint32_t target_node = 0;
    };
    struct gateway_pair_accumulator_t {
      uint64_t edge_count = 0;
      std::vector<gateway_seed_t> shortlist;
    };
    const size_t pair_count = checked_size(
        checked_multiply(config.community_count, config.community_count, "Community pairs"),
        "Community pairs");
    std::vector<std::vector<gateway_pair_accumulator_t>> worker_pairs(
        thread_count, std::vector<gateway_pair_accumulator_t>(pair_count));
    std::vector<std::vector<uint64_t>> worker_internal(
        thread_count, std::vector<uint64_t>(config.community_count, 0));
    const auto better_gateway = [&](const gateway_seed_t& left, const gateway_seed_t& right) {
      if (node_importance[left.target_node] != node_importance[right.target_node]) {
        return node_importance[left.target_node] > node_importance[right.target_node];
      }
      return left.target_node < right.target_node;
    };
    const auto consider_gateway = [&](std::vector<gateway_seed_t>& shortlist,
                                      gateway_seed_t candidate) {
      if (shortlist.size() == config.gateway_shortlist &&
          better_gateway(shortlist.back(), candidate)) {
        return;
      }
      const auto position = std::lower_bound(shortlist.begin(), shortlist.end(), candidate,
                                             better_gateway);
      if (position != shortlist.end() && position->target_node == candidate.target_node) {
        position->source_node = std::min(position->source_node, candidate.source_node);
        return;
      }
      shortlist.insert(position, candidate);
      if (shortlist.size() > config.gateway_shortlist) {
        shortlist.pop_back();
      }
    };

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int64_t signed_worker = 0; signed_worker < static_cast<int64_t>(thread_count);
         ++signed_worker) {
      const uint32_t worker = static_cast<uint32_t>(signed_worker);
      auto& pairs = worker_pairs[worker];
      auto& internal = worker_internal[worker];
      for (uint64_t node = worker; node < graph.point_count(); node += thread_count) {
        const uint32_t source_node = static_cast<uint32_t>(node);
        const uint32_t source_community = result.index.node_to_community[source_node];
        for (const uint32_t target_node : graph.neighbors(source_node)) {
          const uint32_t target_community = result.index.node_to_community[target_node];
          if (source_community == target_community) {
            ++internal[source_community];
            continue;
          }
          auto& pair = pairs[static_cast<size_t>(source_community) * config.community_count +
                             target_community];
          ++pair.edge_count;
          consider_gateway(pair.shortlist, {source_node, target_node});
        }
      }
    }

    for (uint32_t community = 0; community < config.community_count; ++community) {
      for (uint32_t worker = 0; worker < thread_count; ++worker) {
        result.index.communities[community].internal_directed_edge_count = checked_add(
            result.index.communities[community].internal_directed_edge_count,
            worker_internal[worker][community], "internal directed Community edges");
      }
    }
    for (uint32_t source = 0; source < config.community_count; ++source) {
      for (uint32_t target = 0; target < config.community_count; ++target) {
        if (source == target) {
          continue;
        }
        uint64_t directed_cross_edge_count = 0;
        std::vector<gateway_seed_t> shortlist;
        shortlist.reserve(config.gateway_shortlist);
        const size_t pair = static_cast<size_t>(source) * config.community_count + target;
        for (uint32_t worker = 0; worker < thread_count; ++worker) {
          directed_cross_edge_count =
              checked_add(directed_cross_edge_count, worker_pairs[worker][pair].edge_count,
                          "directed cross-Community edges");
          for (const auto& candidate : worker_pairs[worker][pair].shortlist) {
            consider_gateway(shortlist, candidate);
          }
        }
        if (directed_cross_edge_count == 0) {
          continue;
        }
        result.index.communities[source].outgoing_directed_edge_count = checked_add(
            result.index.communities[source].outgoing_directed_edge_count,
            directed_cross_edge_count, "outgoing directed Community edges");
        result.index.communities[target].incoming_directed_edge_count = checked_add(
            result.index.communities[target].incoming_directed_edge_count,
            directed_cross_edge_count, "incoming directed Community edges");

        std::vector<cross_arc_t> shortlisted_arcs;
        shortlisted_arcs.reserve(shortlist.size());
        for (const auto& candidate : shortlist) {
          shortlisted_arcs.push_back({source, target, candidate.source_node,
                                      candidate.target_node});
        }
        std::sort(shortlisted_arcs.begin(), shortlisted_arcs.end(),
                  [](const auto& left, const auto& right) {
                    return std::tie(left.target_node, left.source_node) <
                           std::tie(right.target_node, right.source_node);
                  });
        community_polar_edge_t edge;
        edge.source_community = source;
        edge.target_community = target;
        edge.directed_cross_edge_count = directed_cross_edge_count;
        edge.gateway_begin = result.index.gateways.size();
        auto gateways = select_gateways(shortlisted_arcs, target, node_importance, result, config,
                                        std::span<const uint32_t>());
        edge.gateway_count = static_cast<uint32_t>(gateways.size());
        result.index.gateways.insert(result.index.gateways.end(), gateways.begin(), gateways.end());
        result.index.community_edges.push_back(edge);
      }
    }
  } else {
    // Explicit Gateway profiles must validate arbitrary named targets, so retain the original
    // complete-arc path for that uncommon offline construction mode.
    std::vector<cross_arc_t> cross_arcs;
    for (uint32_t node = 0; node < graph.point_count(); ++node) {
      const uint32_t source_community = result.index.node_to_community[node];
      for (const uint32_t target : graph.neighbors(node)) {
        const uint32_t target_community = result.index.node_to_community[target];
        if (source_community == target_community) {
          ++result.index.communities[source_community].internal_directed_edge_count;
        } else {
          ++result.index.communities[source_community].outgoing_directed_edge_count;
          ++result.index.communities[target_community].incoming_directed_edge_count;
          cross_arcs.push_back({source_community, target_community, node, target});
        }
      }
    }
    std::sort(cross_arcs.begin(), cross_arcs.end(), [](const auto& left, const auto& right) {
      return std::tie(left.source_community, left.target_community, left.target_node,
                      left.source_node) < std::tie(right.source_community, right.target_community,
                                                   right.target_node, right.source_node);
    });

    for (size_t begin = 0; begin < cross_arcs.size();) {
      size_t end = begin + 1;
      while (end < cross_arcs.size() &&
             cross_arcs[end].source_community == cross_arcs[begin].source_community &&
             cross_arcs[end].target_community == cross_arcs[begin].target_community) {
        ++end;
      }
      community_polar_edge_t edge;
      edge.source_community = cross_arcs[begin].source_community;
      edge.target_community = cross_arcs[begin].target_community;
      edge.directed_cross_edge_count = end - begin;
      edge.gateway_begin = result.index.gateways.size();
      const auto selection = gateway_selection.find({edge.source_community, edge.target_community});
      const std::span<const uint32_t> profiled_targets =
          selection == gateway_selection.end() ? std::span<const uint32_t>() : selection->second;
      auto gateways =
          select_gateways(std::span<const cross_arc_t>(cross_arcs).subspan(begin, end - begin),
                          edge.target_community, node_importance, result, config, profiled_targets);
      edge.gateway_count = static_cast<uint32_t>(gateways.size());
      result.index.gateways.insert(result.index.gateways.end(), gateways.begin(), gateways.end());
      result.index.community_edges.push_back(edge);
      begin = end;
    }
  }
  result.index.community_edge_offsets.assign(config.community_count + 1, 0);
  for (const auto& edge : result.index.community_edges) {
    ++result.index.community_edge_offsets[edge.source_community + 1];
  }
  std::partial_sum(result.index.community_edge_offsets.begin(),
                   result.index.community_edge_offsets.end(),
                   result.index.community_edge_offsets.begin());

  try {
    if (config.cell_partition == community_cell_partition_t::GRAPH_LOCAL) {
      build_graph_local_cells(graph, result, community_nodes,
                              result.index.config.partition_threads);
    } else if (config.cell_partition == community_cell_partition_t::CONVERGENCE_COACCESS) {
      build_convergence_coaccess_cells(result, community_nodes, convergence_assignments);
    } else if (config.cell_partition == community_cell_partition_t::GLOBAL_GEOMETRIC) {
      build_global_geometric_cells(result, convergence_assignments);
    } else if (config.cell_partition == community_cell_partition_t::GLOBAL_GRAPH_LOCAL) {
      build_global_graph_local_cells(graph, result);
    } else {
      build_community_polar_cells(result, community_nodes);
    }
  } catch (const std::exception& error) {
    throw std::runtime_error("Community-local Polar Cell construction failed: " +
                             std::string(error.what()));
  }
  if (config.build_contiguous_cell_hierarchy) {
    build_contiguous_cell_hierarchy(result.index);
  }
  load_convergence_cell_hierarchy(config, result.index);
  load_convergence_cell_capacities(config, result.index);
  if (result.packet_node_ids.size() != graph.point_count() ||
      std::find(result.index.node_to_cell.begin(), result.index.node_to_cell.end(),
                std::numeric_limits<uint32_t>::max()) != result.index.node_to_cell.end() ||
      std::find(result.index.node_to_packet_record.begin(),
                result.index.node_to_packet_record.end(),
                std::numeric_limits<uint64_t>::max()) != result.index.node_to_packet_record.end()) {
    throw std::logic_error("Community-Polar packing did not cover every Base node exactly once");
  }
  for (auto& gateway : result.index.gateways) {
    gateway.packet_record_index = result.index.node_to_packet_record[gateway.target_node];
  }
  return result;
}

void finalize_community_polar_index(community_polar_build_result_t& result,
                                    const std::filesystem::path& index_path_prefix) {
  finalize_community_polar_index(result, index_path_prefix,
                                 make_community_polar_index_path(index_path_prefix));
}

void finalize_community_polar_index(community_polar_build_result_t& result,
                                    const std::filesystem::path& index_path_prefix,
                                    const std::filesystem::path& output_sidecar_path) {
  const auto pq_path = std::filesystem::path(index_path_prefix.string() + "_pq_compressed.bin");
  const auto pivots_path = std::filesystem::path(index_path_prefix.string() + "_pq_pivots.bin");
  const auto disk_path = std::filesystem::path(index_path_prefix.string() + "_disk.index");
  if (!std::filesystem::is_regular_file(pq_path) ||
      !std::filesystem::is_regular_file(pivots_path) ||
      !std::filesystem::is_regular_file(disk_path)) {
    throw std::runtime_error("Community-Polar finalize requires completed DiskANN artifacts");
  }

  std::unique_ptr<uint8_t[]> pq_codes;
  size_t pq_points = 0;
  size_t pq_code_width = 0;
  load_bin<uint8_t>(pq_path.string(), pq_codes, pq_points, pq_code_width);
  if (pq_points != result.index.point_count || pq_code_width == 0 ||
      pq_code_width > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("DiskANN PQ codes do not match Community-Polar construction");
  }
  result.index.pq_code_width = static_cast<uint32_t>(pq_code_width);
  result.index.base_index_fingerprint = fingerprint_file(disk_path);
  result.index.pq_fingerprint = fingerprint_file(pq_path);

  fixed_chunk_pq_table_t pq_table;
  pq_table.load_pq_centroid_bin(pivots_path.c_str(), pq_code_width);
  if (pq_table.get_num_chunks() != pq_code_width) {
    throw std::runtime_error("DiskANN PQ pivot width does not match compressed codes");
  }

  const uint64_t record_bytes = sizeof(uint32_t) + pq_code_width;
  result.index.packet_payload.clear();
  result.index.packet_payload.reserve(checked_size(
      checked_multiply(result.index.point_count, record_bytes, "Community-Polar packets"),
      "Community-Polar packets"));
  std::vector<float> reconstructed(result.index.dimension);
  std::vector<float> reconstruction_error(result.packet_node_ids.size());
  for (const uint32_t node : result.packet_node_ids) {
    append_u32_le(result.index.packet_payload, node);
    const uint8_t* code = pq_codes.get() + static_cast<size_t>(node) * pq_code_width;
    result.index.packet_payload.insert(result.index.packet_payload.end(), code,
                                       code + pq_code_width);
    pq_table.inflate_vector(const_cast<uint8_t*>(code), reconstructed.data());
    double squared_error = 0.0;
    const auto original = vector_row(result, node);
    for (uint32_t dimension = 0; dimension < result.index.dimension; ++dimension) {
      const double difference = static_cast<double>(original[dimension]) - reconstructed[dimension];
      squared_error += difference * difference;
    }
    reconstruction_error[result.index.node_to_packet_record[node]] =
        static_cast<float>(std::sqrt(squared_error));
  }
  for (auto& block : result.index.blocks) {
    block.maximum_pq_reconstruction_error = 0.0F;
    for (uint64_t packet = block.packet_record_begin;
         packet < block.packet_record_begin + block.node_count; ++packet) {
      block.maximum_pq_reconstruction_error =
          std::max(block.maximum_pq_reconstruction_error, reconstruction_error[packet]);
    }
  }
  for (auto& community : result.index.communities) {
    community.packet_bytes =
        checked_multiply(community.node_count, record_bytes, "Community packet bytes");
  }

  validate_loaded_index(result.index);
  write_community_polar_index(output_sidecar_path, result.index);
  result.original_vectors.clear();
  result.original_vectors.shrink_to_fit();
  result.packet_node_ids.clear();
  result.packet_node_ids.shrink_to_fit();
}

void rebind_community_polar_pq(const std::filesystem::path& source_sidecar_path,
                               const std::filesystem::path& index_path_prefix,
                               const std::filesystem::path& output_sidecar_path) {
  if (source_sidecar_path.empty() || output_sidecar_path.empty() ||
      std::filesystem::absolute(source_sidecar_path).lexically_normal() ==
          std::filesystem::absolute(output_sidecar_path).lexically_normal()) {
    throw std::invalid_argument("PQ rebind requires distinct source and output sidecars");
  }
  const auto disk_path = std::filesystem::path(index_path_prefix.string() + "_disk.index");
  const auto pq_path = std::filesystem::path(index_path_prefix.string() + "_pq_compressed.bin");
  if (!std::filesystem::is_regular_file(disk_path) || !std::filesystem::is_regular_file(pq_path)) {
    throw std::runtime_error("PQ rebind requires completed Base and compressed-PQ artifacts");
  }
  std::ifstream source(source_sidecar_path, std::ios::binary);
  if (!source) {
    throw std::runtime_error("failed to open source Community-Polar sidecar");
  }
  std::array<uint8_t, k_fixed_header_size> fixed_header{};
  source.read(reinterpret_cast<char*>(fixed_header.data()), fixed_header.size());
  if (!source) {
    throw std::runtime_error("source Community-Polar header is truncated");
  }
  size_t cursor = 0;
  const uint64_t magic = read_u64_le(fixed_header, cursor);
  const uint32_t version = read_u32_le(fixed_header, cursor);
  const uint32_t header_size = read_u32_le(fixed_header, cursor);
  const uint32_t endian = read_u32_le(fixed_header, cursor);
  const uint32_t section_count = read_u32_le(fixed_header, cursor);
  const uint64_t point_count = read_u64_le(fixed_header, cursor);
  const uint32_t dimension = read_u32_le(fixed_header, cursor);
  const uint32_t old_pq_width = read_u32_le(fixed_header, cursor);
  const uint64_t old_body_hash = read_u64_le(fixed_header, cursor);
  community_polar_file_fingerprint_t base_fingerprint;
  base_fingerprint.size_bytes = read_u64_le(fixed_header, cursor);
  base_fingerprint.fnv1a_hash = read_u64_le(fixed_header, cursor);
  static_cast<void>(read_u64_le(fixed_header, cursor));
  static_cast<void>(read_u64_le(fixed_header, cursor));
  const uint64_t old_body_size = read_u64_le(fixed_header, cursor);
  if (magic != k_community_polar_magic || version < 5 || version > k_community_polar_version ||
      endian != k_endian_marker || section_count != k_section_count ||
      header_size != k_fixed_header_size + k_section_count * k_section_entry_size ||
      point_count == 0 || dimension == 0 || old_pq_width == 0 ||
      std::filesystem::file_size(source_sidecar_path) != header_size + old_body_size) {
    throw std::runtime_error("source Community-Polar header is incompatible with streaming rebind");
  }
  std::vector<uint8_t> directory_bytes(header_size - k_fixed_header_size);
  source.read(reinterpret_cast<char*>(directory_bytes.data()), directory_bytes.size());
  if (!source) {
    throw std::runtime_error("source Community-Polar directory is truncated");
  }
  cursor = 0;
  std::vector<section_directory_entry_t> source_directory;
  source_directory.reserve(section_count);
  uint64_t expected_offset = header_size;
  for (uint32_t section = 0; section < section_count; ++section) {
    const uint32_t raw_id = read_u32_le(directory_bytes, cursor);
    static_cast<void>(read_u32_le(directory_bytes, cursor));
    const uint64_t offset = read_u64_le(directory_bytes, cursor);
    const uint64_t length = read_u64_le(directory_bytes, cursor);
    const uint64_t count = read_u64_le(directory_bytes, cursor);
    if (raw_id != section + 1U || offset != expected_offset) {
      throw std::runtime_error("source Community-Polar directory is not canonical");
    }
    source_directory.push_back({static_cast<section_id_t>(raw_id), offset, length, count});
    expected_offset = checked_add(offset, length, "source sidecar section");
  }
  if (expected_offset != std::filesystem::file_size(source_sidecar_path)) {
    throw std::runtime_error("source Community-Polar section coverage is incomplete");
  }
  const auto base_actual = fingerprint_file(disk_path);
  if (base_actual.size_bytes != base_fingerprint.size_bytes ||
      base_actual.fnv1a_hash != base_fingerprint.fnv1a_hash) {
    throw std::runtime_error("PQ rebind Base fingerprint does not match the source sidecar");
  }

  std::unique_ptr<uint8_t[]> pq_codes;
  size_t pq_points = 0;
  size_t pq_code_width = 0;
  load_bin<uint8_t>(pq_path.string(), pq_codes, pq_points, pq_code_width);
  if (pq_points != point_count || pq_code_width == 0 ||
      pq_code_width > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("PQ rebind codes do not match the source sidecar");
  }
  const auto pq_fingerprint = fingerprint_file(pq_path);
  const uint64_t old_record_bytes = sizeof(uint32_t) + old_pq_width;
  const uint64_t new_record_bytes = sizeof(uint32_t) + pq_code_width;
  auto output_directory = source_directory;
  uint64_t next_offset = header_size;
  for (auto& section : output_directory) {
    section.offset = next_offset;
    if (section.id == section_id_t::PACKETS) {
      if (section.record_count != point_count ||
          section.length != checked_multiply(point_count, old_record_bytes, "old packets")) {
        throw std::runtime_error("source Community-Polar packet section is invalid");
      }
      section.length = checked_multiply(point_count, new_record_bytes, "new packets");
    }
    next_offset = checked_add(next_offset, section.length, "rebound sidecar bytes");
  }
  const uint64_t new_body_size = next_offset - header_size;

  const auto temporary_path = std::filesystem::path(output_sidecar_path.string() + ".tmp");
  std::error_code ignored;
  std::filesystem::remove(temporary_path, ignored);
  std::fstream output(temporary_path,
                      std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open temporary PQ-rebound sidecar");
  }
  std::vector<uint8_t> placeholder(header_size, 0);
  output.write(reinterpret_cast<const char*>(placeholder.data()), placeholder.size());
  uint64_t new_body_hash = k_fnv_offset_basis;
  uint64_t verified_old_body_hash = k_fnv_offset_basis;
  std::array<uint8_t, 1U << 20U> buffer{};
  const auto write_hashed = [&](std::span<const uint8_t> bytes) {
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    for (const uint8_t value : bytes) {
      new_body_hash ^= value;
      new_body_hash *= k_fnv_prime;
    }
  };
  std::vector<uint8_t> seen((point_count + 7U) / 8U, 0);
  for (size_t section_index = 0; section_index < source_directory.size(); ++section_index) {
    const auto& section = source_directory[section_index];
    source.seekg(static_cast<std::streamoff>(section.offset));
    uint64_t remaining = section.length;
    if (section.id == section_id_t::COMMUNITIES) {
      constexpr uint64_t record_bytes = 84;
      if (section.length != section.record_count * record_bytes) {
        throw std::runtime_error("Community section width is invalid during PQ rebind");
      }
      std::array<uint8_t, record_bytes> record{};
      for (uint64_t record_id = 0; record_id < section.record_count; ++record_id) {
        source.read(reinterpret_cast<char*>(record.data()), record.size());
        for (const uint8_t value : record) {
          verified_old_body_hash ^= value;
          verified_old_body_hash *= k_fnv_prime;
        }
        size_t record_cursor = 4;
        const uint64_t node_count = read_u64_le(record, record_cursor);
        std::vector<uint8_t> packet_bytes;
        append_u64_le(packet_bytes,
                      checked_multiply(node_count, new_record_bytes, "Community packet bytes"));
        std::copy(packet_bytes.begin(), packet_bytes.end(), record.begin() + 12);
        write_hashed(record);
      }
      continue;
    }
    if (section.id == section_id_t::BLOCKS) {
      constexpr uint64_t record_bytes = 36;
      if (section.length != section.record_count * record_bytes) {
        throw std::runtime_error("Block section width is invalid during PQ rebind");
      }
      std::array<uint8_t, record_bytes> record{};
      for (uint64_t record_id = 0; record_id < section.record_count; ++record_id) {
        source.read(reinterpret_cast<char*>(record.data()), record.size());
        for (const uint8_t value : record) {
          verified_old_body_hash ^= value;
          verified_old_body_hash *= k_fnv_prime;
        }
        std::vector<uint8_t> maximum_error;
        append_float_le(maximum_error, std::numeric_limits<float>::max());
        std::copy(maximum_error.begin(), maximum_error.end(), record.begin() + 32);
        write_hashed(record);
      }
      continue;
    }
    if (section.id == section_id_t::PACKETS) {
      std::vector<uint8_t> source_record(old_record_bytes);
      std::vector<uint8_t> output_record(new_record_bytes);
      for (uint64_t packet = 0; packet < point_count; ++packet) {
        source.read(reinterpret_cast<char*>(source_record.data()), source_record.size());
        for (const uint8_t value : source_record) {
          verified_old_body_hash ^= value;
          verified_old_body_hash *= k_fnv_prime;
        }
        size_t record_cursor = 0;
        const uint32_t node = read_u32_le(source_record, record_cursor);
        if (node >= point_count || (seen[node >> 3U] & (1U << (node & 7U))) != 0) {
          throw std::runtime_error("source packet ownership is not one-to-one");
        }
        seen[node >> 3U] |= static_cast<uint8_t>(1U << (node & 7U));
        std::copy_n(source_record.begin(), sizeof(uint32_t), output_record.begin());
        std::copy_n(pq_codes.get() + static_cast<size_t>(node) * pq_code_width, pq_code_width,
                    output_record.begin() + sizeof(uint32_t));
        write_hashed(output_record);
      }
      continue;
    }
    while (remaining != 0) {
      const size_t count = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
      source.read(reinterpret_cast<char*>(buffer.data()), count);
      if (!source) {
        throw std::runtime_error("source Community-Polar section is truncated");
      }
      for (size_t offset = 0; offset < count; ++offset) {
        verified_old_body_hash ^= buffer[offset];
        verified_old_body_hash *= k_fnv_prime;
      }
      write_hashed(std::span<const uint8_t>(buffer).first(count));
      remaining -= count;
    }
  }
  if (!source || !output || verified_old_body_hash != old_body_hash) {
    throw std::runtime_error("source checksum or rebound sidecar write failed");
  }

  std::vector<uint8_t> output_header;
  append_u64_le(output_header, k_community_polar_magic);
  append_u32_le(output_header, version);
  append_u32_le(output_header, header_size);
  append_u32_le(output_header, k_endian_marker);
  append_u32_le(output_header, k_section_count);
  append_u64_le(output_header, point_count);
  append_u32_le(output_header, dimension);
  append_u32_le(output_header, static_cast<uint32_t>(pq_code_width));
  append_u64_le(output_header, new_body_hash);
  append_u64_le(output_header, base_fingerprint.size_bytes);
  append_u64_le(output_header, base_fingerprint.fnv1a_hash);
  append_u64_le(output_header, pq_fingerprint.size_bytes);
  append_u64_le(output_header, pq_fingerprint.fnv1a_hash);
  append_u64_le(output_header, new_body_size);
  append_zeroes(output_header, k_fixed_header_size - output_header.size());
  for (const auto& section : output_directory) {
    append_u32_le(output_header, static_cast<uint32_t>(section.id));
    append_u32_le(output_header, 0);
    append_u64_le(output_header, section.offset);
    append_u64_le(output_header, section.length);
    append_u64_le(output_header, section.record_count);
  }
  output.seekp(0);
  output.write(reinterpret_cast<const char*>(output_header.data()), output_header.size());
  output.close();
  if (!output || std::filesystem::file_size(temporary_path) != next_offset) {
    std::filesystem::remove(temporary_path, ignored);
    throw std::runtime_error("PQ-rebound sidecar size accounting failed");
  }
  std::filesystem::rename(temporary_path, output_sidecar_path);
}

void build_community_polar_sidecar_from_disk(const std::filesystem::path& data_path,
                                             const std::filesystem::path& index_path_prefix,
                                             const std::filesystem::path& rcni_path,
                                             const std::filesystem::path& output_sidecar_path,
                                             const community_polar_build_config_t& config) {
  validate_community_polar_build_config(config);
  if (!std::filesystem::is_regular_file(data_path)) {
    throw std::invalid_argument("sidecar-only data path is not a regular file: " +
                                data_path.string());
  }
  if (output_sidecar_path.empty() || output_sidecar_path.filename().empty()) {
    throw std::invalid_argument("sidecar-only output path must include a file name");
  }
  const auto parent = output_sidecar_path.parent_path();
  if (!parent.empty() && !std::filesystem::is_directory(parent)) {
    throw std::invalid_argument("sidecar-only output directory does not exist: " + parent.string());
  }
  const auto disk_path = std::filesystem::path(index_path_prefix.string() + "_disk.index");
  const auto pq_path = std::filesystem::path(index_path_prefix.string() + "_pq_compressed.bin");
  const auto pivots_path = std::filesystem::path(index_path_prefix.string() + "_pq_pivots.bin");
  const auto output_absolute = std::filesystem::absolute(output_sidecar_path).lexically_normal();
  std::vector<std::filesystem::path> protected_paths = {data_path, disk_path, pq_path, pivots_path,
                                                        rcni_path};
  if (!config.gateway_selection_source_path.empty()) {
    protected_paths.push_back(config.gateway_selection_source_path);
  }
  if (!config.convergence_cell_profile_path.empty()) {
    protected_paths.push_back(config.convergence_cell_profile_path);
  }
  if (!config.convergence_cell_hierarchy_path.empty()) {
    protected_paths.push_back(config.convergence_cell_hierarchy_path);
  }
  if (!config.convergence_cell_capacity_path.empty()) {
    protected_paths.push_back(config.convergence_cell_capacity_path);
  }
  if (!config.convergence_cell_source_path.empty()) {
    protected_paths.push_back(config.convergence_cell_source_path);
  }
  for (const auto& protected_path : protected_paths) {
    const auto protected_absolute = std::filesystem::absolute(protected_path).lexically_normal();
    std::error_code equivalent_error;
    const bool equivalent =
        std::filesystem::exists(output_sidecar_path) && std::filesystem::exists(protected_path) &&
        std::filesystem::equivalent(output_sidecar_path, protected_path, equivalent_error);
    if (output_absolute == protected_absolute || (!equivalent_error && equivalent)) {
      throw std::invalid_argument("sidecar-only output must not overwrite an input artifact: " +
                                  protected_path.string());
    }
  }
  if (!config.gateway_selection_source_path.empty()) {
    auto index = load_community_polar_index(config.gateway_selection_source_path);
    validate_community_polar_artifacts(index, index_path_prefix);
    if (index.config.community_count != config.community_count ||
        index.config.community_imbalance != config.community_imbalance ||
        index.config.partition_quality != config.partition_quality ||
        index.config.partition_projection != config.partition_projection ||
        index.config.partition_threads != effective_thread_count(config.partition_threads) ||
        index.config.partition_seed != config.partition_seed ||
        index.config.gateway_shortlist != config.gateway_shortlist ||
        index.config.polar_direction_count != config.polar_direction_count ||
        index.config.cell_partition != config.cell_partition ||
        index.config.cell_target_size != config.cell_target_size ||
        index.config.block_target_size != config.block_target_size ||
        index.config.spherical_kmeans_iterations != config.spherical_kmeans_iterations) {
      throw std::invalid_argument(
          "Gateway selection source construction parameters do not match the requested profile");
    }
    const auto selections = load_gateway_selection(config);
    std::vector<community_polar_gateway_t> gateways;
    gateways.reserve(index.community_edges.size() * config.gateway_count);
    for (auto& edge : index.community_edges) {
      const uint64_t source_begin = edge.gateway_begin;
      const uint64_t source_end = edge.gateway_begin + edge.gateway_count;
      edge.gateway_begin = gateways.size();
      const auto selected = selections.find({edge.source_community, edge.target_community});
      if (selected != selections.end()) {
        for (const uint32_t target : selected->second) {
          const auto source_first =
              index.gateways.begin() + static_cast<std::ptrdiff_t>(source_begin);
          const auto source_last = index.gateways.begin() + static_cast<std::ptrdiff_t>(source_end);
          const auto found = std::find_if(source_first, source_last, [&](const auto& gateway) {
            return gateway.target_node == target;
          });
          if (found == source_last) {
            throw std::invalid_argument(
                "Gateway selection CSV target is absent from its profiled source sidecar");
          }
          gateways.push_back(*found);
        }
      }
      for (uint64_t gateway = source_begin;
           gateway < source_end && gateways.size() - edge.gateway_begin < config.gateway_count;
           ++gateway) {
        const auto& fallback = index.gateways[gateway];
        const bool duplicate = std::any_of(
            gateways.begin() + static_cast<std::ptrdiff_t>(edge.gateway_begin), gateways.end(),
            [&](const auto& chosen) { return chosen.target_node == fallback.target_node; });
        if (!duplicate) {
          gateways.push_back(fallback);
        }
      }
      edge.gateway_count = static_cast<uint32_t>(gateways.size() - edge.gateway_begin);
      if (edge.gateway_count == 0) {
        throw std::invalid_argument("Gateway selection produced an empty directed superedge");
      }
    }
    index.gateways = std::move(gateways);
    index.config.gateway_count = config.gateway_count;
    index.config.gateway_shortlist = config.gateway_shortlist;
    index.config.gateway_selection_path.clear();
    index.config.gateway_selection_source_path.clear();
    write_community_polar_index(output_sidecar_path, index);
    return;
  }
  if ((config.cell_partition == community_cell_partition_t::GRAPH_LOCAL ||
       config.cell_partition == community_cell_partition_t::CONVERGENCE_COACCESS ||
       config.cell_partition == community_cell_partition_t::GLOBAL_GEOMETRIC ||
       config.cell_partition == community_cell_partition_t::GLOBAL_GRAPH_LOCAL) &&
      !config.convergence_cell_source_path.empty()) {
    auto source_index = load_community_polar_index(config.convergence_cell_source_path);
    validate_community_polar_artifacts(source_index, index_path_prefix);
    if (source_index.config.community_count != config.community_count ||
        source_index.config.community_imbalance != config.community_imbalance ||
        source_index.config.partition_quality != config.partition_quality ||
        source_index.config.partition_projection != config.partition_projection ||
        source_index.config.partition_seed != config.partition_seed ||
        source_index.config.gateway_count != config.gateway_count ||
        source_index.config.gateway_shortlist != config.gateway_shortlist ||
        source_index.config.polar_direction_count != config.polar_direction_count) {
      throw std::invalid_argument(
          "Convergence Cell source construction parameters do not match the requested profile");
    }
    if (config.cell_partition == community_cell_partition_t::GRAPH_LOCAL &&
        config.convergence_cell_profile_path.empty()) {
      repack_graph_local_cells(source_index, config);
      write_community_polar_index(output_sidecar_path, source_index);
      return;
    }
    if (config.cell_partition == community_cell_partition_t::GLOBAL_GEOMETRIC &&
        config.convergence_cell_profile_path.extension() == ".pqloc") {
      repack_pq_locality_cells(source_index, config.convergence_cell_profile_path, config);
      load_convergence_cell_capacities(config, source_index);
      write_community_polar_index(output_sidecar_path, source_index);
      return;
    }
    disk_post_link_graph_view_t graph(data_path, disk_path);
    const auto assignments = load_convergence_cell_profile(config, graph.point_count());
    community_polar_build_result_t result;
    result.index = std::move(source_index);
    result.index.config = config;
    result.index.config.partition_threads = effective_thread_count(config.partition_threads);
    result.index.config.gateway_selection_path.clear();
    result.index.config.gateway_selection_source_path.clear();
    result.index.config.convergence_cell_profile_path.clear();
    result.index.config.convergence_cell_hierarchy_path.clear();
    result.index.config.convergence_cell_capacity_path.clear();
    result.index.config.convergence_cell_source_path.clear();
    result.index.node_to_cell.assign(checked_size(graph.point_count(), "Base point count"),
                                     std::numeric_limits<uint32_t>::max());
    result.index.node_to_packet_record.assign(checked_size(graph.point_count(), "Base point count"),
                                              std::numeric_limits<uint64_t>::max());
    result.index.direction_axes.clear();
    result.index.sectors.clear();
    result.index.cells.clear();
    result.index.blocks.clear();
    result.index.cell_hierarchy_nodes.clear();
    result.index.cell_hierarchy_roots.clear();
    result.index.cell_hierarchy_children.clear();
    result.index.cell_hierarchy_centroids.clear();
    result.index.packet_payload.clear();
    result.original_vectors.resize(
        checked_size(checked_multiply(graph.point_count(), graph.dimension(), "Base vector matrix"),
                     "Base vector matrix"));
    std::vector<std::vector<uint32_t>> community_nodes(config.community_count);
    for (uint32_t node = 0; node < graph.point_count(); ++node) {
      community_nodes[result.index.node_to_community[node]].push_back(node);
    }
#pragma omp parallel for schedule(static) num_threads(result.index.config.partition_threads)
    for (int64_t signed_node = 0; signed_node < static_cast<int64_t>(graph.point_count());
         ++signed_node) {
      const uint32_t node = static_cast<uint32_t>(signed_node);
      graph.copy_vector(
          node, std::span<float>(result.original_vectors)
                    .subspan(static_cast<size_t>(node) * graph.dimension(), graph.dimension()));
    }
    if (config.cell_partition == community_cell_partition_t::GLOBAL_GEOMETRIC) {
      build_global_geometric_cells(result, assignments);
    } else if (config.cell_partition == community_cell_partition_t::GLOBAL_GRAPH_LOCAL) {
      build_global_graph_local_cells(graph, result);
    } else if (config.cell_partition == community_cell_partition_t::GRAPH_LOCAL) {
      build_graph_local_cells(graph, result, community_nodes,
                              result.index.config.partition_threads);
    } else {
      build_convergence_coaccess_cells(result, community_nodes, assignments);
    }
    if (config.build_contiguous_cell_hierarchy) {
      build_contiguous_cell_hierarchy(result.index);
    }
    load_convergence_cell_hierarchy(config, result.index);
    load_convergence_cell_capacities(config, result.index);
    if (result.packet_node_ids.size() != graph.point_count()) {
      throw std::logic_error("Convergence Cell packing did not cover every Base node");
    }
    for (auto& gateway : result.index.gateways) {
      gateway.packet_record_index = result.index.node_to_packet_record[gateway.target_node];
    }
    finalize_community_polar_index(result, index_path_prefix, output_sidecar_path);
    return;
  }
  if (config.cell_partition == community_cell_partition_t::GLOBAL_GEOMETRIC &&
      config.convergence_cell_profile_path.extension() == ".pqloc") {
    if (config.community_count != 1) {
      throw std::invalid_argument(
          "Direct PQ-locality sidecar construction requires exactly one global Community");
    }
    auto read_scalar = [](std::istream& input, auto& value, const char* context) {
      input.read(reinterpret_cast<char*>(&value), sizeof(value));
      if (!input) {
        throw std::runtime_error(std::string(context) + " is truncated");
      }
    };
    std::ifstream data_input(data_path, std::ios::binary);
    uint32_t point_count = 0;
    uint32_t dimension = 0;
    read_scalar(data_input, point_count, "Direct PQ-locality Base header");
    read_scalar(data_input, dimension, "Direct PQ-locality Base header");
    const uint64_t expected_data_bytes =
        checked_add(sizeof(uint32_t) * 2,
                    checked_multiply(checked_multiply(point_count, dimension, "Base values"),
                                     sizeof(float), "Base bytes"),
                    "Base artifact bytes");
    if (point_count == 0 || dimension == 0 ||
        std::filesystem::file_size(data_path) != expected_data_bytes) {
      throw std::invalid_argument("Direct PQ-locality Base artifact is incompatible");
    }

    std::ifstream disk_input(disk_path, std::ios::binary);
    uint32_t metadata_rows = 0;
    uint32_t metadata_columns = 0;
    uint64_t disk_point_count = 0;
    uint64_t disk_dimension = 0;
    uint64_t entry_point = 0;
    read_scalar(disk_input, metadata_rows, "Direct PQ-locality disk metadata");
    read_scalar(disk_input, metadata_columns, "Direct PQ-locality disk metadata");
    read_scalar(disk_input, disk_point_count, "Direct PQ-locality disk metadata");
    read_scalar(disk_input, disk_dimension, "Direct PQ-locality disk metadata");
    read_scalar(disk_input, entry_point, "Direct PQ-locality disk metadata");
    static_cast<void>(disk_dimension);
    if (metadata_rows < 9 || metadata_columns != 1 || disk_point_count != point_count ||
        entry_point >= point_count) {
      throw std::invalid_argument("Direct PQ-locality disk metadata is incompatible");
    }

    std::ifstream pq_input(pq_path, std::ios::binary);
    uint32_t pq_points = 0;
    uint32_t pq_code_width = 0;
    read_scalar(pq_input, pq_points, "Direct PQ-locality compressed-PQ header");
    read_scalar(pq_input, pq_code_width, "Direct PQ-locality compressed-PQ header");
    const uint64_t expected_pq_bytes = checked_add(
        sizeof(uint32_t) * 2, checked_multiply(pq_points, pq_code_width, "compressed-PQ bytes"),
        "compressed-PQ artifact bytes");
    if (pq_points != point_count || pq_code_width == 0 ||
        std::filesystem::file_size(pq_path) != expected_pq_bytes) {
      throw std::invalid_argument("Direct PQ-locality compressed-PQ artifact is incompatible");
    }

    community_polar_index_t index;
    index.config = config;
    index.config.partition_threads = effective_thread_count(config.partition_threads);
    index.point_count = point_count;
    index.dimension = dimension;
    index.entry_point_id = static_cast<uint32_t>(entry_point);
    index.pq_code_width = pq_code_width;
    index.base_index_fingerprint = fingerprint_file(disk_path);
    index.pq_fingerprint = fingerprint_file(pq_path);
    index.node_to_community.assign(point_count, 0);
    index.node_to_cell.assign(point_count, 0);
    index.node_to_packet_record.resize(point_count);
    std::iota(index.node_to_packet_record.begin(), index.node_to_packet_record.end(), 0);
    community_polar_community_t community;
    community.node_count = point_count;
    community.packet_bytes = checked_multiply(point_count, sizeof(uint32_t) + pq_code_width,
                                              "global Community packet bytes");
    community.navigation_hub_id = index.entry_point_id;
    community.radius = std::numeric_limits<float>::max();
    community.graph_only = true;
    index.communities.push_back(community);
    index.community_poles.assign(dimension, 0.0F);
    index.community_edge_offsets = {0, 0};

    const uint64_t record_bytes = sizeof(uint32_t) + pq_code_width;
    index.packet_payload.resize(
        checked_size(checked_multiply(point_count, record_bytes, "Direct PQ-locality packets"),
                     "Direct PQ-locality packets"));
    constexpr uint32_t block_points = 65536;
    std::vector<uint8_t> codes(
        checked_size(checked_multiply(block_points, pq_code_width, "compressed-PQ block"),
                     "compressed-PQ block"));
    for (uint32_t begin = 0; begin < point_count; begin += block_points) {
      const uint32_t count = std::min(block_points, point_count - begin);
      const size_t code_bytes = checked_size(
          checked_multiply(count, pq_code_width, "compressed-PQ block"), "compressed-PQ block");
      pq_input.read(reinterpret_cast<char*>(codes.data()),
                    static_cast<std::streamsize>(code_bytes));
      if (!pq_input) {
        throw std::runtime_error("Direct PQ-locality compressed-PQ payload is truncated");
      }
      for (uint32_t offset = 0; offset < count; ++offset) {
        const uint32_t node = begin + offset;
        const size_t record_offset = static_cast<size_t>(node) * record_bytes;
        for (uint32_t byte = 0; byte < sizeof(uint32_t); ++byte) {
          index.packet_payload[record_offset + byte] = static_cast<uint8_t>(node >> (byte * 8U));
        }
        std::copy_n(codes.begin() + static_cast<size_t>(offset) * pq_code_width, pq_code_width,
                    index.packet_payload.begin() + record_offset + sizeof(uint32_t));
      }
    }
    repack_pq_locality_cells(index, config.convergence_cell_profile_path, config);
    load_convergence_cell_capacities(config, index);
    write_community_polar_index(output_sidecar_path, index);
    return;
  }
  disk_post_link_graph_view_t graph(data_path, disk_path);
  const auto importance = read_rcni_importance_csv(rcni_path, graph.point_count());
  auto result = build_community_polar_index(graph, importance, config);
  finalize_community_polar_index(result, index_path_prefix, output_sidecar_path);
}

std::filesystem::path
make_community_polar_index_path(const std::filesystem::path& index_path_prefix) {
  std::filesystem::path path = index_path_prefix;
  path += ".community_polar.bin";
  return path;
}

void write_community_polar_index(const std::filesystem::path& path,
                                 const community_polar_index_t& index) {
  validate_loaded_index(index);
  const uint32_t version =
      index.config.cell_partition == community_cell_partition_t::GLOBAL_GRAPH_LOCAL
          ? k_community_polar_version
          : (index.config.adaptive_multi_capacity ? k_adaptive_capacity_sidecar_version
                                                  : k_fixed_capacity_sidecar_version);
  auto sections = make_sections(index, version);
  if (sections.size() != k_section_count) {
    throw std::logic_error("Community-Polar serializer section count is invalid");
  }
  const uint32_t header_size = k_fixed_header_size + k_section_count * k_section_entry_size;
  std::vector<section_directory_entry_t> directory;
  directory.reserve(sections.size());
  uint64_t next_offset = header_size;
  uint64_t body_hash = k_fnv_offset_basis;
  for (const auto& section : sections) {
    directory.push_back({section.id, next_offset, section.bytes.size(), section.record_count});
    next_offset = checked_add(next_offset, section.bytes.size(), "Community-Polar file bytes");
    body_hash = fnv1a_update(body_hash, section.bytes);
  }
  const uint64_t body_size = next_offset - header_size;

  std::vector<uint8_t> header;
  header.reserve(header_size);
  append_u64_le(header, k_community_polar_magic);
  append_u32_le(header, version);
  append_u32_le(header, header_size);
  append_u32_le(header, k_endian_marker);
  append_u32_le(header, k_section_count);
  append_u64_le(header, index.point_count);
  append_u32_le(header, index.dimension);
  append_u32_le(header, index.pq_code_width);
  append_u64_le(header, body_hash);
  append_u64_le(header, index.base_index_fingerprint.size_bytes);
  append_u64_le(header, index.base_index_fingerprint.fnv1a_hash);
  append_u64_le(header, index.pq_fingerprint.size_bytes);
  append_u64_le(header, index.pq_fingerprint.fnv1a_hash);
  append_u64_le(header, body_size);
  append_zeroes(header, k_fixed_header_size - header.size());
  for (const auto& entry : directory) {
    append_u32_le(header, static_cast<uint32_t>(entry.id));
    append_u32_le(header, 0);
    append_u64_le(header, entry.offset);
    append_u64_le(header, entry.length);
    append_u64_le(header, entry.record_count);
  }
  if (header.size() != header_size) {
    throw std::logic_error("Community-Polar serialized header size is invalid");
  }

  std::filesystem::path temporary_path = path;
  temporary_path += ".tmp";
  std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open Community-Polar sidecar: " + temporary_path.string());
  }
  output.write(reinterpret_cast<const char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
  for (const auto& section : sections) {
    output.write(reinterpret_cast<const char*>(section.bytes.data()),
                 static_cast<std::streamsize>(section.bytes.size()));
  }
  output.flush();
  if (!output) {
    output.close();
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::runtime_error("failed to write Community-Polar sidecar: " + temporary_path.string());
  }
  output.close();
  std::error_code rename_error;
  std::filesystem::rename(temporary_path, path, rename_error);
  if (rename_error) {
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::runtime_error("failed to publish Community-Polar sidecar: " +
                             rename_error.message());
  }
}

community_polar_index_t load_community_polar_index(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open Community-Polar sidecar: " + path.string());
  }
  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0 || static_cast<uint64_t>(file_size) > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Community-Polar sidecar size is invalid");
  }
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> file(static_cast<size_t>(file_size));
  input.read(reinterpret_cast<char*>(file.data()), file_size);
  if (!input) {
    throw std::runtime_error("failed to read Community-Polar sidecar: " + path.string());
  }

  size_t cursor = 0;
  const uint64_t magic = read_u64_le(file, cursor);
  const uint32_t version = read_u32_le(file, cursor);
  const uint32_t header_size = read_u32_le(file, cursor);
  const uint32_t endian = read_u32_le(file, cursor);
  const uint32_t section_count = read_u32_le(file, cursor);
  community_polar_index_t index;
  index.point_count = read_u64_le(file, cursor);
  index.dimension = read_u32_le(file, cursor);
  index.pq_code_width = read_u32_le(file, cursor);
  const uint64_t body_hash = read_u64_le(file, cursor);
  index.base_index_fingerprint.size_bytes = read_u64_le(file, cursor);
  index.base_index_fingerprint.fnv1a_hash = read_u64_le(file, cursor);
  index.pq_fingerprint.size_bytes = read_u64_le(file, cursor);
  index.pq_fingerprint.fnv1a_hash = read_u64_le(file, cursor);
  const uint64_t body_size = read_u64_le(file, cursor);
  const uint32_t expected_section_count = version >= 5 ? k_section_count : k_legacy_section_count;
  if (magic != k_community_polar_magic || version < k_community_polar_minimum_version ||
      version > k_community_polar_version || endian != k_endian_marker ||
      section_count != expected_section_count ||
      header_size != k_fixed_header_size + expected_section_count * k_section_entry_size ||
      header_size > file.size() || body_size != file.size() - header_size) {
    throw std::runtime_error("Community-Polar sidecar header is incompatible or corrupt");
  }
  cursor = k_fixed_header_size;
  std::vector<section_directory_entry_t> directory;
  directory.reserve(section_count);
  std::vector<bool> section_seen(expected_section_count + 1, false);
  uint64_t expected_offset = header_size;
  for (uint32_t section = 0; section < section_count; ++section) {
    const uint32_t raw_id = read_u32_le(file, cursor);
    static_cast<void>(read_u32_le(file, cursor));
    const uint64_t offset = read_u64_le(file, cursor);
    const uint64_t length = read_u64_le(file, cursor);
    const uint64_t record_count = read_u64_le(file, cursor);
    if (raw_id == 0 || raw_id > expected_section_count || section_seen[raw_id] ||
        offset != expected_offset || offset > file.size() || length > file.size() - offset) {
      throw std::runtime_error("Community-Polar section directory is invalid");
    }
    section_seen[raw_id] = true;
    directory.push_back({static_cast<section_id_t>(raw_id), offset, length, record_count});
    expected_offset = checked_add(offset, length, "Community-Polar section directory");
  }
  if (cursor != header_size || expected_offset != file.size() ||
      fnv1a(std::span<const uint8_t>(file).subspan(header_size)) != body_hash) {
    throw std::runtime_error("Community-Polar sidecar checksum or section coverage is invalid");
  }

  const auto& config_entry = find_section(directory, section_id_t::CONFIG);
  auto config_bytes = section_bytes(file, config_entry);
  size_t config_cursor = 0;
  index.config.community_count = read_u32_le(config_bytes, config_cursor);
  index.config.community_imbalance = read_double_le(config_bytes, config_cursor);
  index.config.partition_quality =
      static_cast<community_partition_quality_t>(read_u32_le(config_bytes, config_cursor));
  index.config.partition_threads = read_u32_le(config_bytes, config_cursor);
  index.config.partition_seed = read_i32_le(config_bytes, config_cursor);
  index.config.gateway_count = read_u32_le(config_bytes, config_cursor);
  index.config.gateway_shortlist = read_u32_le(config_bytes, config_cursor);
  index.config.polar_direction_count = read_u32_le(config_bytes, config_cursor);
  index.config.cell_target_size = read_u32_le(config_bytes, config_cursor);
  index.config.block_target_size = read_u32_le(config_bytes, config_cursor);
  index.config.spherical_kmeans_iterations = read_u32_le(config_bytes, config_cursor);
  index.entry_point_id = read_u32_le(config_bytes, config_cursor);
  index.directed_base_edge_count = read_u64_le(config_bytes, config_cursor);
  index.undirected_partition_edge_count = read_u64_le(config_bytes, config_cursor);
  index.weighted_edge_cut = read_u64_le(config_bytes, config_cursor);
  if (version >= 2) {
    index.config.partition_projection =
        static_cast<community_partition_projection_t>(read_u32_le(config_bytes, config_cursor));
  }
  if (version >= 3) {
    index.config.cell_partition =
        static_cast<community_cell_partition_t>(read_u32_le(config_bytes, config_cursor));
  }
  if (version >= 6) {
    const uint32_t adaptive_multi_capacity = read_u32_le(config_bytes, config_cursor);
    if (adaptive_multi_capacity > 1) {
      throw std::runtime_error("Community-Polar adaptive capacity flag is invalid");
    }
    index.config.adaptive_multi_capacity = adaptive_multi_capacity != 0;
    index.config.adaptive_policy_version = read_u32_le(config_bytes, config_cursor);
    index.config.adaptive_page_bytes = read_u32_le(config_bytes, config_cursor);
    index.config.adaptive_vector_bytes = read_u32_le(config_bytes, config_cursor);
    index.config.adaptive_lru_pages = read_u64_le(config_bytes, config_cursor);
    index.config.adaptive_maximum_normalized_rms_radius =
        read_double_le(config_bytes, config_cursor);
    index.config.adaptive_maximum_normalized_p95_radius =
        read_double_le(config_bytes, config_cursor);
    index.config.adaptive_maximum_split_gain = read_double_le(config_bytes, config_cursor);
    index.config.adaptive_minimum_centroid_radius_overlap =
        read_double_le(config_bytes, config_cursor);
    index.config.adaptive_minimum_train_cross_child_coaccess =
        read_double_le(config_bytes, config_cursor);
    index.config.adaptive_maximum_unseen_pq_increase = read_double_le(config_bytes, config_cursor);
  }
  if (config_cursor != config_bytes.size() || config_entry.record_count != 1) {
    throw std::runtime_error("Community-Polar config section has an invalid length");
  }

  auto read_u32_section = [&](section_id_t id) {
    const auto& entry = find_section(directory, id);
    return read_scalar_section<uint32_t>(section_bytes(file, entry), entry.record_count,
                                         sizeof(uint32_t), read_u32_le);
  };
  auto read_u64_section = [&](section_id_t id) {
    const auto& entry = find_section(directory, id);
    return read_scalar_section<uint64_t>(section_bytes(file, entry), entry.record_count,
                                         sizeof(uint64_t), read_u64_le);
  };
  auto read_float_section = [&](section_id_t id) {
    const auto& entry = find_section(directory, id);
    return read_scalar_section<float>(section_bytes(file, entry), entry.record_count, sizeof(float),
                                      read_float_le);
  };
  index.node_to_community = read_u32_section(section_id_t::NODE_TO_COMMUNITY);
  index.node_to_cell = read_u32_section(section_id_t::NODE_TO_CELL);
  index.node_to_packet_record = read_u64_section(section_id_t::NODE_TO_PACKET);
  index.community_poles = read_float_section(section_id_t::COMMUNITY_POLES);
  index.community_edge_offsets = read_u64_section(section_id_t::COMMUNITY_EDGE_OFFSETS);
  index.direction_axes = read_float_section(section_id_t::DIRECTION_AXES);

  const auto& communities_entry = find_section(directory, section_id_t::COMMUNITIES);
  auto communities_bytes = section_bytes(file, communities_entry);
  size_t communities_cursor = 0;
  index.communities.resize(checked_size(communities_entry.record_count, "Community records"));
  for (auto& community : index.communities) {
    community.community_id = read_u32_le(communities_bytes, communities_cursor);
    community.node_count = read_u64_le(communities_bytes, communities_cursor);
    community.packet_bytes = read_u64_le(communities_bytes, communities_cursor);
    community.internal_directed_edge_count = read_u64_le(communities_bytes, communities_cursor);
    community.incoming_directed_edge_count = read_u64_le(communities_bytes, communities_cursor);
    community.outgoing_directed_edge_count = read_u64_le(communities_bytes, communities_cursor);
    community.minimum_node_id = read_u32_le(communities_bytes, communities_cursor);
    community.navigation_hub_id = read_u32_le(communities_bytes, communities_cursor);
    community.radius = read_float_le(communities_bytes, communities_cursor);
    community.graph_only = read_u32_le(communities_bytes, communities_cursor) != 0;
    community.sector_begin = read_u64_le(communities_bytes, communities_cursor);
    community.sector_count = read_u32_le(communities_bytes, communities_cursor);
    community.cell_begin = read_u64_le(communities_bytes, communities_cursor);
    community.cell_count = read_u32_le(communities_bytes, communities_cursor);
  }
  if (communities_cursor != communities_bytes.size()) {
    throw std::runtime_error("Community-Polar Community section has trailing bytes");
  }

  const auto& edges_entry = find_section(directory, section_id_t::COMMUNITY_EDGES);
  auto edges_bytes = section_bytes(file, edges_entry);
  size_t edges_cursor = 0;
  index.community_edges.resize(checked_size(edges_entry.record_count, "Community edges"));
  for (auto& edge : index.community_edges) {
    edge.source_community = read_u32_le(edges_bytes, edges_cursor);
    edge.target_community = read_u32_le(edges_bytes, edges_cursor);
    edge.directed_cross_edge_count = read_u64_le(edges_bytes, edges_cursor);
    edge.gateway_begin = read_u64_le(edges_bytes, edges_cursor);
    edge.gateway_count = read_u32_le(edges_bytes, edges_cursor);
  }
  if (edges_cursor != edges_bytes.size()) {
    throw std::runtime_error("Community-Polar edge section has trailing bytes");
  }

  const auto& gateways_entry = find_section(directory, section_id_t::GATEWAYS);
  auto gateways_bytes = section_bytes(file, gateways_entry);
  size_t gateways_cursor = 0;
  index.gateways.resize(checked_size(gateways_entry.record_count, "Gateway records"));
  for (auto& gateway : index.gateways) {
    gateway.source_node = read_u32_le(gateways_bytes, gateways_cursor);
    gateway.target_node = read_u32_le(gateways_bytes, gateways_cursor);
    gateway.packet_record_index = read_u64_le(gateways_bytes, gateways_cursor);
  }
  if (gateways_cursor != gateways_bytes.size()) {
    throw std::runtime_error("Community-Polar Gateway section has trailing bytes");
  }

  const auto& sectors_entry = find_section(directory, section_id_t::SECTORS);
  auto sectors_bytes = section_bytes(file, sectors_entry);
  size_t sectors_cursor = 0;
  index.sectors.resize(checked_size(sectors_entry.record_count, "Sector records"));
  for (auto& sector : index.sectors) {
    sector.community_id = read_u32_le(sectors_bytes, sectors_cursor);
    sector.sector_id = read_u32_le(sectors_bytes, sectors_cursor);
    sector.node_count = read_u64_le(sectors_bytes, sectors_cursor);
    sector.axis_index = read_u64_le(sectors_bytes, sectors_cursor);
    sector.minimum_radius = read_float_le(sectors_bytes, sectors_cursor);
    sector.maximum_radius = read_float_le(sectors_bytes, sectors_cursor);
    sector.maximum_angle = read_float_le(sectors_bytes, sectors_cursor);
    sector.cell_begin = read_u64_le(sectors_bytes, sectors_cursor);
    sector.cell_count = read_u32_le(sectors_bytes, sectors_cursor);
  }
  if (sectors_cursor != sectors_bytes.size()) {
    throw std::runtime_error("Community-Polar Sector section has trailing bytes");
  }

  const auto& cells_entry = find_section(directory, section_id_t::CELLS);
  auto cells_bytes = section_bytes(file, cells_entry);
  size_t cells_cursor = 0;
  index.cells.resize(checked_size(cells_entry.record_count, "Cell records"));
  for (auto& cell : index.cells) {
    cell.cell_id = read_u32_le(cells_bytes, cells_cursor);
    cell.community_id = read_u32_le(cells_bytes, cells_cursor);
    cell.sector_id = read_u32_le(cells_bytes, cells_cursor);
    cell.radial_cell_id = read_u32_le(cells_bytes, cells_cursor);
    cell.node_count = read_u64_le(cells_bytes, cells_cursor);
    cell.capacity_class =
        version >= 6 ? read_u32_le(cells_bytes, cells_cursor) : index.config.cell_target_size;
    cell.minimum_radius = read_float_le(cells_bytes, cells_cursor);
    cell.maximum_radius = read_float_le(cells_bytes, cells_cursor);
    cell.maximum_angle = read_float_le(cells_bytes, cells_cursor);
    cell.block_begin = read_u64_le(cells_bytes, cells_cursor);
    cell.block_count = read_u32_le(cells_bytes, cells_cursor);
  }
  if (cells_cursor != cells_bytes.size()) {
    throw std::runtime_error("Community-Polar Cell section has trailing bytes");
  }

  const auto& blocks_entry = find_section(directory, section_id_t::BLOCKS);
  auto blocks_bytes = section_bytes(file, blocks_entry);
  size_t blocks_cursor = 0;
  index.blocks.resize(checked_size(blocks_entry.record_count, "Block records"));
  for (auto& block : index.blocks) {
    block.block_id = read_u32_le(blocks_bytes, blocks_cursor);
    block.cell_id = read_u32_le(blocks_bytes, blocks_cursor);
    block.node_count = read_u32_le(blocks_bytes, blocks_cursor);
    block.packet_record_begin = read_u64_le(blocks_bytes, blocks_cursor);
    block.minimum_radius = read_float_le(blocks_bytes, blocks_cursor);
    block.maximum_radius = read_float_le(blocks_bytes, blocks_cursor);
    block.maximum_angle = read_float_le(blocks_bytes, blocks_cursor);
    block.maximum_pq_reconstruction_error = read_float_le(blocks_bytes, blocks_cursor);
  }
  if (blocks_cursor != blocks_bytes.size()) {
    throw std::runtime_error("Community-Polar Block section has trailing bytes");
  }

  const auto& packets_entry = find_section(directory, section_id_t::PACKETS);
  const auto packets = section_bytes(file, packets_entry);
  index.packet_payload.assign(packets.begin(), packets.end());
  if (packets_entry.record_count != index.point_count) {
    throw std::runtime_error("Community-Polar packet record count is invalid");
  }
  if (version >= 5) {
    const auto& hierarchy_nodes_entry = find_section(directory, section_id_t::CELL_HIERARCHY_NODES);
    const auto hierarchy_nodes_bytes = section_bytes(file, hierarchy_nodes_entry);
    size_t hierarchy_nodes_cursor = 0;
    index.cell_hierarchy_nodes.resize(
        checked_size(hierarchy_nodes_entry.record_count, "Cell hierarchy nodes"));
    for (auto& node : index.cell_hierarchy_nodes) {
      node.node_id = read_u32_le(hierarchy_nodes_bytes, hierarchy_nodes_cursor);
      node.child_begin = read_u32_le(hierarchy_nodes_bytes, hierarchy_nodes_cursor);
      node.child_count = read_u32_le(hierarchy_nodes_bytes, hierarchy_nodes_cursor);
      node.descendant_node_count = read_u64_le(hierarchy_nodes_bytes, hierarchy_nodes_cursor);
      node.radius = read_float_le(hierarchy_nodes_bytes, hierarchy_nodes_cursor);
      const uint32_t children_are_cells =
          read_u32_le(hierarchy_nodes_bytes, hierarchy_nodes_cursor);
      if (children_are_cells > 1) {
        throw std::runtime_error("Community-Polar Cell hierarchy node kind is invalid");
      }
      node.children_are_cells = children_are_cells != 0;
    }
    if (hierarchy_nodes_cursor != hierarchy_nodes_bytes.size()) {
      throw std::runtime_error("Community-Polar Cell hierarchy nodes have trailing bytes");
    }
    index.cell_hierarchy_roots = read_u32_section(section_id_t::CELL_HIERARCHY_ROOTS);
    index.cell_hierarchy_children = read_u32_section(section_id_t::CELL_HIERARCHY_CHILDREN);
    index.cell_hierarchy_centroids = read_float_section(section_id_t::CELL_HIERARCHY_CENTROIDS);
  }
  validate_loaded_index(index);
  return index;
}

void validate_community_polar_artifacts(const community_polar_index_t& index,
                                        const std::filesystem::path& index_path_prefix) {
  const auto disk_path = std::filesystem::path(index_path_prefix.string() + "_disk.index");
  const auto pq_path = std::filesystem::path(index_path_prefix.string() + "_pq_compressed.bin");
  const auto base_fingerprint = fingerprint_file(disk_path);
  const auto pq_fingerprint = fingerprint_file(pq_path);
  if (base_fingerprint.size_bytes != index.base_index_fingerprint.size_bytes ||
      base_fingerprint.fnv1a_hash != index.base_index_fingerprint.fnv1a_hash ||
      pq_fingerprint.size_bytes != index.pq_fingerprint.size_bytes ||
      pq_fingerprint.fnv1a_hash != index.pq_fingerprint.fnv1a_hash) {
    throw std::runtime_error("Community-Polar sidecar does not match its Base or PQ artifact");
  }
}

community_polar_collector_t::community_polar_collector_t(
    const rcni_aggregator_t& aggregator, const community_polar_build_config_t& config)
    : aggregator_(aggregator), config_(config) {
  validate_community_polar_build_config(config_);
}

void community_polar_collector_t::observe(const post_link_graph_view_t& graph) {
  if (result_.has_value()) {
    throw std::logic_error("Community-Polar collector may observe only one completed Base graph");
  }
  result_ = build_community_polar_index(graph, aggregator_.compute_importance(), config_);
}

void community_polar_collector_t::finalize(const std::filesystem::path& index_path_prefix) {
  if (!result_.has_value()) {
    throw std::logic_error("Community-Polar collector has no observed Base graph");
  }
  finalize_community_polar_index(*result_, index_path_prefix);
}

const community_polar_index_t& community_polar_collector_t::index() const {
  if (!result_.has_value()) {
    throw std::logic_error("Community-Polar collector has no observed Base graph");
  }
  return result_->index;
}

} // namespace powerlaw_ann
