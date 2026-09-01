#include "index/community_polar_index.h"
#include "index/io_optimized_index.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_value(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
}

void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " --index_path_prefix PREFIX [--topology_path FILE] [--vector_path FILE]"
               " [--vector_layout original_id|cell_4k|cell_chunks|cell_weighted_4k|cell_u8_4k]"
               " [--adjacency raw|pfor_delta]"
               " [--skipcell_sidecar_path FILE] [--prefetch_hints true|false]"
               " [--hub_clique_low_rcni_trim FILE]"
               " [--cell_adj_path FILE] [--cell_adj_sample_nodes N] [--cell_adj_degree N]"
               " [--cell_adj_threads N]"
               " [--global_graph_assignment_path FILE]"
               " [--global_graph_cell_target_size N]\n";
}

bool parse_bool(const std::string& value, const std::string& option) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }
  throw std::invalid_argument(option + " must be 'true' or 'false'");
}

void append_u32(std::vector<uint8_t>& bytes, uint32_t value) {
  for (uint32_t shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>(value >> shift));
  }
}

void append_u64(std::vector<uint8_t>& bytes, uint64_t value) {
  for (uint32_t shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<uint8_t>(value >> shift));
  }
}

uint64_t fnv1a(std::span<const std::byte> bytes) {
  uint64_t checksum = 14695981039346656037ULL;
  for (const std::byte value : bytes) {
    checksum ^= std::to_integer<uint8_t>(value);
    checksum *= 1099511628211ULL;
  }
  return checksum;
}

powerlaw_ann::io_artifact_fingerprint_t fingerprint_file(const std::filesystem::path& path) {
  powerlaw_ann::io_artifact_fingerprint_t result;
  result.size = std::filesystem::file_size(path);
  result.checksum = 14695981039346656037ULL;
  std::ifstream input(path, std::ios::binary);
  std::array<uint8_t, 1U << 20U> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const auto count = input.gcount();
    for (std::streamsize index = 0; index < count; ++index) {
      result.checksum ^= buffer[static_cast<size_t>(index)];
      result.checksum *= 1099511628211ULL;
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed fingerprinting Cell adjacency sidecar");
  }
  return result;
}

void write_global_graph_assignment(
    const std::filesystem::path& path,
    const powerlaw_ann::io_global_graph_cell_assignment_t& assignment, uint32_t target_size) {
  const auto node_bytes = std::as_bytes(std::span(assignment.node_to_cell));
  std::vector<uint8_t> header;
  header.insert(header.end(), {'P', 'L', 'C', 'G', 'C', 'E', 'L', 'L'});
  append_u32(header, 1);
  append_u32(header, 48);
  append_u64(header, assignment.node_to_cell.size());
  append_u32(header, assignment.cell_count);
  append_u32(header, target_size);
  append_u64(header, fnv1a(node_bytes));
  header.resize(48, 0);
  const auto temporary = std::filesystem::path(path.string() + ".tmp");
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(header.data()), header.size());
  output.write(reinterpret_cast<const char*>(node_bytes.data()),
               static_cast<std::streamsize>(node_bytes.size()));
  output.close();
  if (!output) {
    throw std::runtime_error("failed writing global graph Cell assignment");
  }
  std::filesystem::rename(temporary, path);
}

} // namespace

int main(int argc, char** argv) {
  std::filesystem::path index_prefix;
  std::filesystem::path topology_path;
  std::filesystem::path vector_path;
  std::filesystem::path community_polar_path;
  std::filesystem::path hub_clique_low_rcni_path;
  std::filesystem::path cell_adj_path;
  std::filesystem::path global_graph_assignment_path;
  uint32_t global_graph_cell_target_size = 128;
  uint32_t cell_adj_sample_nodes = 8;
  uint32_t cell_adj_degree = 64;
  uint32_t cell_adj_threads = 0;
  auto layout = powerlaw_ann::io_vector_layout_t::ORIGINAL_ID;
  auto encoding = powerlaw_ann::io_adjacency_encoding_t::RAW_U32;
  bool build_prefetch_hints = true;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--index_path_prefix") {
        index_prefix = read_value(argc, argv, index);
      } else if (option == "--topology_path") {
        topology_path = read_value(argc, argv, index);
      } else if (option == "--vector_path") {
        vector_path = read_value(argc, argv, index);
      } else if (option == "--community_polar_path" || option == "--skipcell_sidecar_path") {
        community_polar_path = read_value(argc, argv, index);
      } else if (option == "--hub_clique_low_rcni_trim") {
        hub_clique_low_rcni_path = read_value(argc, argv, index);
      } else if (option == "--cell_adj_path") {
        cell_adj_path = read_value(argc, argv, index);
      } else if (option == "--cell_adj_sample_nodes") {
        cell_adj_sample_nodes = static_cast<uint32_t>(std::stoul(read_value(argc, argv, index)));
      } else if (option == "--cell_adj_degree") {
        cell_adj_degree = static_cast<uint32_t>(std::stoul(read_value(argc, argv, index)));
      } else if (option == "--cell_adj_threads") {
        cell_adj_threads = static_cast<uint32_t>(std::stoul(read_value(argc, argv, index)));
      } else if (option == "--global_graph_assignment_path") {
        global_graph_assignment_path = read_value(argc, argv, index);
      } else if (option == "--global_graph_cell_target_size") {
        global_graph_cell_target_size =
            static_cast<uint32_t>(std::stoul(read_value(argc, argv, index)));
      } else if (option == "--vector_layout") {
        const auto value = read_value(argc, argv, index);
        if (value == "original_id") {
          layout = powerlaw_ann::io_vector_layout_t::ORIGINAL_ID;
        } else if (value == "cell_4k") {
          layout = powerlaw_ann::io_vector_layout_t::CELL_4K;
        } else if (value == "cell_chunks") {
          layout = powerlaw_ann::io_vector_layout_t::CELL_CHUNKS;
        } else if (value == "cell_weighted_4k") {
          layout = powerlaw_ann::io_vector_layout_t::CELL_WEIGHTED_4K;
        } else if (value == "cell_u8_4k") {
          layout = powerlaw_ann::io_vector_layout_t::CELL_U8_4K;
        } else {
          throw std::invalid_argument(
              "--vector_layout must be original_id, cell_4k, cell_chunks, cell_weighted_4k, or "
              "cell_u8_4k");
        }
      } else if (option == "--adjacency") {
        const auto value = read_value(argc, argv, index);
        if (value == "raw") {
          encoding = powerlaw_ann::io_adjacency_encoding_t::RAW_U32;
        } else if (value == "pfor_delta") {
          encoding = powerlaw_ann::io_adjacency_encoding_t::PFOR_DELTA;
        } else {
          throw std::invalid_argument("--adjacency must be raw or pfor_delta");
        }
      } else if (option == "--prefetch_hints") {
        build_prefetch_hints = parse_bool(read_value(argc, argv, index), option);
      } else if (option == "--help" || option == "-h") {
        print_usage(argv[0]);
        return 0;
      } else {
        throw std::invalid_argument("unknown option: " + option);
      }
    }
    if (index_prefix.empty()) {
      throw std::invalid_argument("--index_path_prefix is required");
    }
    if (!cell_adj_path.empty()) {
      if (topology_path.empty() || community_polar_path.empty() || cell_adj_sample_nodes == 0 ||
          cell_adj_degree == 0) {
        throw std::invalid_argument(
            "Cell adjacency build requires topology, Community-Polar, sample, and degree inputs");
      }
      const auto topology =
          powerlaw_ann::io_optimized_index_t::load(index_prefix, topology_path, vector_path, true);
      const auto cells = powerlaw_ann::load_community_polar_index(community_polar_path);
      powerlaw_ann::validate_community_polar_artifacts(cells, index_prefix);
      const auto fingerprint = fingerprint_file(community_polar_path);
      if (topology->community_polar_fingerprint() != fingerprint) {
        throw std::invalid_argument(
            "Cell adjacency inputs do not share the same Community-Polar fingerprint");
      }
      const auto result = powerlaw_ann::io_cell_adjacency_index_t::build(
          cell_adj_path, *topology, cells, fingerprint, cell_adj_sample_nodes, cell_adj_degree,
          cell_adj_threads);
      std::cout << "cells=" << result.cell_count << " cell_edges=" << result.edge_count
                << " artifact_bytes=" << result.artifact_bytes
                << " maximum_degree=" << result.maximum_degree
                << " sampled_nodes_per_cell=" << result.sampled_nodes_per_cell
                << " build_threads=" << result.build_threads << '\n';
      return 0;
    }
    if (!global_graph_assignment_path.empty()) {
      if (topology_path.empty() || global_graph_cell_target_size == 0) {
        throw std::invalid_argument(
            "global graph assignment requires topology path and a positive target size");
      }
      const auto assignment = powerlaw_ann::build_io_global_graph_cell_assignment_from_artifact(
          topology_path, global_graph_cell_target_size);
      write_global_graph_assignment(global_graph_assignment_path, assignment,
                                    global_graph_cell_target_size);
      std::cout << "points=" << assignment.node_to_cell.size()
                << " global_graph_cells=" << assignment.cell_count
                << " target_size=" << global_graph_cell_target_size << '\n';
      return 0;
    }
    const auto result = powerlaw_ann::io_optimized_index_t::build(
        index_prefix, topology_path, vector_path, encoding, layout, community_polar_path,
        build_prefetch_hints, hub_clique_low_rcni_path);
    std::cout << "points=" << result.point_count << " edges=" << result.edge_count
              << " topology_bytes=" << result.topology_bytes
              << " adjacency_bytes=" << result.adjacency_bytes
              << " vector_bytes=" << result.vector_bytes
              << " vector_padding_bytes=" << result.vector_padding_bytes
              << " build_peak_payload_bytes=" << result.build_peak_payload_bytes
              << " cell_count=" << result.cell_count
              << " nonempty_vector_bytes=" << result.nonempty_vector_bytes
              << " cell_page_count=" << result.cell_page_count
              << " gateway_prefetch_hint_count=" << result.gateway_prefetch_hint_count
              << " gateway_prefetch_hint_bytes=" << result.gateway_prefetch_hint_bytes
              << " low_rcni_trimmed_nodes=" << result.low_rcni_trimmed_nodes
              << " chunks_4k=" << result.chunk_counts[0] << " chunks_64k=" << result.chunk_counts[1]
              << " chunks_512k=" << result.chunk_counts[2]
              << " chunks_2m=" << result.chunk_counts[3] << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "build_io_optimized_index: " << error.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }
}
