#include "search/diskann_search.h"

#include "common/ann_error.h"
#include "common/utils.h"
#include "search/pq_flash_index.h"
#if defined(__APPLE__)
#include "storage/macos_aligned_file_reader.h"
#elif defined(__linux__)
#include "storage/io_uring_aligned_file_reader.h"
#include "storage/linux_aligned_file_reader.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <omp.h>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unistd.h>

namespace powerlaw_ann {
namespace {

uint64_t mix_memgraph_id(uint64_t value) noexcept {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

std::vector<uint32_t> read_rcni_memgraph_nodes(const std::filesystem::path& path,
                                               uint32_t point_count, uint32_t budget) {
  std::ifstream input(path);
  if (!input) {
    throw ann_exception_t("RCNI-only MemGraph input does not exist: " + path.string());
  }
  std::string line;
  if (!std::getline(input, line) ||
      line != "node_id,raw_importance,normalized_importance,importance_percentile,source_support,"
              "witness_event_count") {
    throw ann_exception_t("RCNI-only MemGraph input has an invalid header");
  }
  std::vector<std::pair<double, uint32_t>> ranked;
  ranked.reserve(point_count);
  std::vector<uint8_t> seen(point_count, 0);
  while (std::getline(input, line)) {
    std::istringstream row(line);
    std::string node_text;
    std::string importance_text;
    if (!std::getline(row, node_text, ',') || !std::getline(row, importance_text, ',')) {
      throw ann_exception_t("RCNI-only MemGraph input has a malformed row");
    }
    const auto node = static_cast<uint32_t>(std::stoul(node_text));
    const double importance = std::stod(importance_text);
    if (node >= point_count || seen[node] != 0 || !std::isfinite(importance)) {
      throw ann_exception_t("RCNI-only MemGraph input has invalid node data");
    }
    seen[node] = 1;
    ranked.emplace_back(importance, node);
  }
  if (ranked.size() != point_count) {
    throw ann_exception_t("RCNI-only MemGraph input does not cover every Base node");
  }
  std::partial_sort(ranked.begin(), ranked.begin() + budget, ranked.end(),
                    [](const auto& left, const auto& right) {
                      return left.first > right.first ||
                             (left.first == right.first && left.second < right.second);
                    });
  std::vector<uint32_t> nodes;
  nodes.reserve(budget);
  for (uint32_t position = 0; position < budget; ++position) {
    nodes.push_back(ranked[position].second);
  }
  return nodes;
}

void validate_config(const diskann_search_config_t& config) {
  if (config.index_path_prefix.empty()) {
    throw ann_exception_t("DiskANN search index prefix must not be empty");
  }
  if (config.query_path.empty() || !std::filesystem::is_regular_file(config.query_path)) {
    throw ann_exception_t("DiskANN query file does not exist: " + config.query_path.string());
  }
  if (!config.ground_truth_path.empty() &&
      !std::filesystem::is_regular_file(config.ground_truth_path)) {
    throw ann_exception_t("DiskANN ground-truth file does not exist: " +
                          config.ground_truth_path.string());
  }
  if (config.top_k == 0 || config.search_list_size < config.top_k) {
    throw ann_exception_t("DiskANN search requires 0 < top_k <= search_list_size");
  }
  if (config.beam_width == 0) {
    throw ann_exception_t("DiskANN search beam width must be positive");
  }
  if (config.cache_state != "warm" && config.cache_state != "cold") {
    throw ann_exception_t("DiskANN search cache state must be 'warm' or 'cold'");
  }
}

uint64_t calculate_index_size(const std::filesystem::path& index_path_prefix) {
  const auto parent = index_path_prefix.has_parent_path() ? index_path_prefix.parent_path()
                                                          : std::filesystem::current_path();
  const std::string file_prefix = index_path_prefix.filename().string() + "_";
  uint64_t size_bytes = 0;
  for (const auto& entry : std::filesystem::directory_iterator(parent)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (!filename.starts_with(file_prefix)) {
      continue;
    }
    const uint64_t file_size = entry.file_size();
    if (file_size > std::numeric_limits<uint64_t>::max() - size_bytes) {
      throw ann_exception_t("DiskANN index artifact size overflow");
    }
    size_bytes += file_size;
  }
  return size_bytes;
}

diskann_search_backends_t active_backends() {
#if defined(__APPLE__)
  return {"macos_synchronous", "accelerate", "system", "portable"};
#elif defined(__linux__)
#if defined(USE_AVX2)
  constexpr const char* simd = "avx2_fma";
#else
  constexpr const char* simd = "portable";
#endif
  return {"libaio_o_direct", "onemkl_ilp64", "tcmalloc", simd};
#else
  return {"unsupported", "unsupported", "unsupported", "unsupported"};
#endif
}

void calculate_result_recall(const diskann_search_config_t& config,
                             diskann_search_result_t& result) {
  if (config.ground_truth_path.empty()) {
    return;
  }

  uint32_t* ground_truth_ids = nullptr;
  float* ground_truth_distances = nullptr;
  size_t ground_truth_queries = 0;
  size_t ground_truth_dimension = 0;
  load_truthset(config.ground_truth_path.string(), ground_truth_ids, ground_truth_distances,
                ground_truth_queries, ground_truth_dimension);
  std::unique_ptr<uint32_t[]> id_owner(ground_truth_ids);
  std::unique_ptr<float[]> distance_owner(ground_truth_distances);
  if (config.query_offset > ground_truth_queries ||
      result.num_queries > ground_truth_queries - config.query_offset ||
      (config.query_limit == 0 &&
       ground_truth_queries - config.query_offset != result.num_queries)) {
    throw ann_exception_t("Ground-truth query count does not cover the searched queries");
  }
  if (ground_truth_dimension < result.top_k) {
    throw ann_exception_t("Ground-truth neighbor count is smaller than recall_at");
  }

  std::vector<uint32_t> result_ids(result.ids.size());
  for (size_t i = 0; i < result.ids.size(); ++i) {
    if (result.ids[i] > std::numeric_limits<uint32_t>::max()) {
      throw ann_exception_t("DiskANN result ID exceeds uint32 compatibility range");
    }
    result_ids[i] = static_cast<uint32_t>(result.ids[i]);
  }
  result.recall_percent = calculate_recall(
      static_cast<uint32_t>(result.num_queries),
      ground_truth_ids + config.query_offset * ground_truth_dimension,
      ground_truth_distances == nullptr
          ? nullptr
          : ground_truth_distances + config.query_offset * ground_truth_dimension,
      static_cast<uint32_t>(ground_truth_dimension), result_ids.data(), result.top_k, result.top_k);
}

void save_results(const diskann_search_config_t& config, diskann_search_result_t& result) {
  if (config.result_path_prefix.empty()) {
    return;
  }
  const auto parent = config.result_path_prefix.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }

  std::vector<uint32_t> ids(result.ids.size());
  for (size_t i = 0; i < result.ids.size(); ++i) {
    ids[i] = static_cast<uint32_t>(result.ids[i]);
  }
  const std::string base =
      config.result_path_prefix.string() + "_" + std::to_string(config.search_list_size);
  save_bin<uint32_t>(base + "_idx_uint32.bin", ids.data(), result.num_queries, result.top_k);
  save_bin<float>(base + "_dists_float.bin", result.distances.data(), result.num_queries,
                  result.top_k);
}

void save_cell_oracle(const diskann_search_config_t& config,
                      const diskann_search_result_t& result) {
  if (config.cell_oracle_output_path.empty()) {
    return;
  }
  const auto parent = config.cell_oracle_output_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  auto temporary = config.cell_oracle_output_path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::trunc);
  if (!output) {
    throw ann_exception_t("Failed to open Cell oracle output: " + temporary.string());
  }
  output << "query_id,triggered,trigger_hop,eligible_blocks,eligible_cells,future_expansions,"
            "future_distinct_"
            "cells,"
            "future_distinct_blocks,selected_block_id,selected_block_hits,"
            "selected_unseen_nodes,selected_competitive_nodes,selected_best_threshold_ratio,"
            "selected_evidence_threshold_ratio,"
            "selected_cell_hits,best_eligible_block_id,best_eligible_block_hits,"
            "best_eligible_cell_id,best_eligible_cell_hits,polar_eligible_cell_id,"
            "polar_eligible_cell_hits,support_eligible_cell_id,support_eligible_cell_hits,"
            "evidence_eligible_cell_id,evidence_eligible_cell_hits,last_source_cell_hits,"
            "best_global_block_id,"
            "top1_cell_hits,top2_cell_hits,top4_cell_hits,top4_eligible_cell_hits,"
            "top4_support_cell_hits,top4_evidence_cell_hits,top4_polar_cell_hits,"
            "top1_block_hits,top2_block_hits,"
            "top4_block_hits\n";
  for (size_t query_id = 0; query_id < result.query_stats.size(); ++query_id) {
    const auto& stats = result.query_stats[query_id];
    const auto write_id = [&](uint32_t value) {
      if (value == std::numeric_limits<uint32_t>::max()) {
        output << -1;
      } else {
        output << value;
      }
    };
    output << query_id << ',' << stats.cell_oracle_triggered << ',' << stats.cell_oracle_trigger_hop
           << ',' << stats.cell_oracle_eligible_blocks << ',' << stats.cell_oracle_eligible_cells
           << ',' << stats.cell_oracle_future_expansions << ','
           << stats.cell_oracle_future_distinct_cells << ','
           << stats.cell_oracle_future_distinct_blocks << ',';
    write_id(stats.cell_oracle_selected_block_id);
    output << ',' << stats.cell_oracle_selected_block_hits << ','
           << stats.cell_oracle_selected_unseen_nodes << ','
           << stats.cell_oracle_selected_competitive_nodes << ','
           << stats.cell_oracle_selected_best_threshold_ratio << ','
           << stats.cell_oracle_selected_evidence_threshold_ratio << ','
           << stats.cell_oracle_selected_cell_hits << ',';
    write_id(stats.cell_oracle_best_eligible_block_id);
    output << ',' << stats.cell_oracle_best_eligible_block_hits << ',';
    write_id(stats.cell_oracle_best_eligible_cell_id);
    output << ',' << stats.cell_oracle_best_eligible_cell_hits << ',';
    write_id(stats.cell_oracle_polar_eligible_cell_id);
    output << ',' << stats.cell_oracle_polar_eligible_cell_hits << ',';
    write_id(stats.cell_oracle_support_eligible_cell_id);
    output << ',' << stats.cell_oracle_support_eligible_cell_hits << ',';
    write_id(stats.cell_oracle_evidence_eligible_cell_id);
    output << ',' << stats.cell_oracle_evidence_eligible_cell_hits << ','
           << stats.cell_oracle_last_source_cell_hits << ',';
    write_id(stats.cell_oracle_best_global_block_id);
    output << ',' << stats.cell_oracle_top1_cell_hits << ',' << stats.cell_oracle_top2_cell_hits
           << ',' << stats.cell_oracle_top4_cell_hits << ','
           << stats.cell_oracle_top4_eligible_cell_hits << ','
           << stats.cell_oracle_top4_support_cell_hits << ','
           << stats.cell_oracle_top4_evidence_cell_hits << ','
           << stats.cell_oracle_top4_polar_cell_hits << ',' << stats.cell_oracle_top1_block_hits
           << ',' << stats.cell_oracle_top2_block_hits << ',' << stats.cell_oracle_top4_block_hits
           << '\n';
  }
  output.close();
  if (!output) {
    throw ann_exception_t("Failed to write Cell oracle output: " + temporary.string());
  }
  std::error_code error;
  std::filesystem::remove(config.cell_oracle_output_path, error);
  std::filesystem::rename(temporary, config.cell_oracle_output_path);
}

void save_cell_value_telemetry(const diskann_search_config_t& config,
                               const diskann_search_result_t& result) {
  if (config.cell_value_output_path.empty()) {
    return;
  }
  const auto parent = config.cell_value_output_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  auto temporary = config.cell_value_output_path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::trunc);
  if (!output) {
    throw ann_exception_t("Failed to open Cell value telemetry output: " + temporary.string());
  }
  output << "query_id,block_ordinal,block_id,cell_id,block_support,cell_support,node_count,"
            "packet_bytes,already_visited,nodes_scored,nodes_competitive,nodes_inserted,"
            "future_expansions,admitted,evidence_threshold_ratio,lower_bound_threshold_ratio,"
            "best_threshold_ratio\n";
  for (size_t query_id = 0; query_id < result.query_stats.size(); ++query_id) {
    const auto& stats = result.query_stats[query_id];
    const size_t block_count =
        std::min<size_t>(stats.cell_value_blocks_recorded, k_cell_value_max_blocks);
    for (size_t ordinal = 0; ordinal < block_count; ++ordinal) {
      output << query_id << ',' << ordinal << ',' << stats.cell_value_block_ids[ordinal] << ','
             << stats.cell_value_cell_ids[ordinal] << ',' << stats.cell_value_block_support[ordinal]
             << ',' << stats.cell_value_cell_support[ordinal] << ','
             << stats.cell_value_node_count[ordinal] << ','
             << stats.cell_value_packet_bytes[ordinal] << ','
             << stats.cell_value_already_visited[ordinal] << ','
             << stats.cell_value_nodes_scored[ordinal] << ','
             << stats.cell_value_nodes_competitive[ordinal] << ','
             << stats.cell_value_nodes_inserted[ordinal] << ','
             << stats.cell_value_future_expansions[ordinal] << ','
             << stats.cell_value_admitted[ordinal] << ','
             << stats.cell_value_evidence_threshold_ratio[ordinal] << ','
             << stats.cell_value_lower_bound_threshold_ratio[ordinal] << ','
             << stats.cell_value_best_threshold_ratio[ordinal] << '\n';
    }
  }
  output.close();
  if (!output) {
    throw ann_exception_t("Failed to write Cell value telemetry output: " + temporary.string());
  }
  std::error_code error;
  std::filesystem::remove(config.cell_value_output_path, error);
  std::filesystem::rename(temporary, config.cell_value_output_path);
}

void save_gateway_value_telemetry(const diskann_search_config_t& config,
                                  const diskann_search_result_t& result) {
  if (config.gateway_value_output_path.empty()) {
    return;
  }
  const auto parent = config.gateway_value_output_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  auto temporary = config.gateway_value_output_path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::trunc);
  if (!output) {
    throw ann_exception_t("Failed to open Gateway value telemetry output: " + temporary.string());
  }
  output << "query_id,search_l,source_community,target_community,source_node,gateway_node,"
            "trigger_hop,gateway_pq_rank,later_expanded,first_later_expansion_hop,"
            "future_expansions_covered,estimated_hops_bypassed,estimated_reads_bypassed\n";
  for (size_t query_id = 0; query_id < result.query_stats.size(); ++query_id) {
    for (const auto& record : result.query_stats[query_id].gateway_value_records) {
      const uint32_t hops_bypassed =
          record.later_expanded != 0 && record.first_later_expansion_hop >= record.trigger_hop
              ? record.first_later_expansion_hop - record.trigger_hop
              : 0;
      const uint32_t reads_bypassed =
          record.later_expanded != 0 &&
                  record.first_later_expansion_read >= record.trigger_base_reads
              ? record.first_later_expansion_read - record.trigger_base_reads
              : 0;
      output << query_id << ',' << config.search_list_size << ',' << record.source_community << ','
             << record.target_community << ',' << record.source_node << ',' << record.gateway_node
             << ',' << record.trigger_hop << ',' << record.gateway_pq_rank << ','
             << record.later_expanded << ',' << record.first_later_expansion_hop << ','
             << record.future_expansions_covered << ',' << hops_bypassed << ',' << reads_bypassed
             << '\n';
    }
  }
  output.close();
  if (!output) {
    throw ann_exception_t("Failed to write Gateway value telemetry output: " + temporary.string());
  }
  std::error_code error;
  std::filesystem::remove(config.gateway_value_output_path, error);
  std::filesystem::rename(temporary, config.gateway_value_output_path);
}

void save_base_trace(const diskann_search_config_t& config, const diskann_search_result_t& result) {
  if (config.base_trace_output_path.empty()) {
    return;
  }
  const auto parent = config.base_trace_output_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  auto temporary = config.base_trace_output_path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::trunc);
  if (!output) {
    throw ann_exception_t("Failed to open Base expansion trace: " + temporary.string());
  }
  output << "query_id,search_l,expansion_ordinal,node_id,community_id,cell_id,hop,base_reads\n";
  for (size_t query_id = 0; query_id < result.query_stats.size(); ++query_id) {
    const auto& records = result.query_stats[query_id].base_expansion_records;
    for (size_t ordinal = 0; ordinal < records.size(); ++ordinal) {
      const auto& record = records[ordinal];
      output << query_id << ',' << config.search_list_size << ',' << ordinal << ','
             << record.node_id << ',' << record.community_id << ',' << record.cell_id << ','
             << record.hop << ',' << record.base_reads << '\n';
    }
  }
  output.close();
  if (!output) {
    throw ann_exception_t("Failed to write Base expansion trace: " + temporary.string());
  }
  std::error_code error;
  std::filesystem::remove(config.base_trace_output_path, error);
  std::filesystem::rename(temporary, config.base_trace_output_path);
}

} // namespace

double diskann_search_result_t::qps() const {
  return elapsed_seconds > 0.0 ? static_cast<double>(num_queries) / elapsed_seconds : 0.0;
}

double diskann_search_result_t::mean_latency_us() const {
  return get_mean_stats<float>(query_stats.data(), query_stats.size(),
                               [](const query_stats_t& stats) { return stats.total_us; });
}

double diskann_search_result_t::p50_latency_us() const {
  return get_percentile_stats<float>(query_stats.data(), query_stats.size(), 0.50F,
                                     [](const query_stats_t& stats) { return stats.total_us; });
}

double diskann_search_result_t::p95_latency_us() const {
  return get_percentile_stats<float>(query_stats.data(), query_stats.size(), 0.95F,
                                     [](const query_stats_t& stats) { return stats.total_us; });
}

double diskann_search_result_t::p99_latency_us() const {
  return get_percentile_stats<float>(query_stats.data(), query_stats.size(), 0.99F,
                                     [](const query_stats_t& stats) { return stats.total_us; });
}

double diskann_search_result_t::reads_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.n_ios; });
}

double diskann_search_result_t::io_time_us_per_query() const {
  return get_mean_stats<float>(query_stats.data(), query_stats.size(),
                               [](const query_stats_t& stats) { return stats.io_us; });
}

double diskann_search_result_t::cpu_time_us_per_query() const {
  return get_mean_stats<float>(query_stats.data(), query_stats.size(),
                               [](const query_stats_t& stats) { return stats.cpu_us; });
}

double diskann_search_result_t::read_bytes_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.read_size; });
}

double diskann_search_result_t::four_kib_reads_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.n_4k; });
}

double diskann_search_result_t::cache_hits_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.n_cache_hits; });
}

double diskann_search_result_t::distance_computations_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.n_cmps; });
}

double diskann_search_result_t::hops_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.n_hops; });
}

double diskann_search_result_t::base_neighbors_per_query() const {
  return get_mean_stats<uint32_t>(
      query_stats.data(), query_stats.size(),
      [](const query_stats_t& stats) { return stats.n_base_neighbors_scanned; });
}

double diskann_search_result_t::base_hops_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.base_hops; });
}

double diskann_search_result_t::sequential_ios_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.sequential_ios; });
}

double diskann_search_result_t::random_ios_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const query_stats_t& stats) { return stats.random_ios; });
}

double diskann_search_result_t::physical_read_requests_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.physical_read_requests; });
}

double diskann_search_result_t::cell_page_batch_requests_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_page_batch_requests; });
}

double diskann_search_result_t::cell_page_batch_pages_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_page_batch_pages; });
}

double diskann_search_result_t::demand_useful_bytes_per_query() const {
  return get_mean_stats<uint64_t>(
      query_stats.data(), query_stats.size(),
      [](const query_stats_t& stats) { return stats.demand_useful_bytes; });
}

double diskann_search_result_t::demand_overread_bytes_per_query() const {
  return std::max(0.0, read_bytes_per_query() - demand_useful_bytes_per_query());
}

double diskann_search_result_t::memgraph_nodes_scored_per_query() const {
  return get_mean_stats<uint32_t>(
      query_stats.data(), query_stats.size(),
      [](const query_stats_t& stats) { return stats.memgraph_nodes_scored; });
}

double diskann_search_result_t::memgraph_nodes_inserted_per_query() const {
  return get_mean_stats<uint32_t>(
      query_stats.data(), query_stats.size(),
      [](const query_stats_t& stats) { return stats.memgraph_nodes_inserted; });
}

double diskann_search_result_t::memgraph_nodes_later_expanded_per_query() const {
  return get_mean_stats<uint32_t>(
      query_stats.data(), query_stats.size(),
      [](const query_stats_t& stats) { return stats.memgraph_nodes_later_expanded; });
}

double diskann_search_result_t::dynamic_width_mean() const {
  uint64_t rounds = 0;
  uint64_t widths = 0;
  for (const auto& stats : query_stats) {
    rounds += stats.dynamic_width_rounds;
    widths += stats.dynamic_width_sum;
  }
  return rounds == 0 ? 0.0 : static_cast<double>(widths) / static_cast<double>(rounds);
}

uint32_t diskann_search_result_t::dynamic_width_max() const {
  uint32_t width = 0;
  for (const auto& stats : query_stats) {
    width = std::max(width, stats.dynamic_width_max);
  }
  return width;
}

double diskann_search_result_t::dynamic_width_useful_ios_per_query() const {
  return get_mean_stats<uint32_t>(
      query_stats.data(), query_stats.size(),
      [](const query_stats_t& stats) { return stats.dynamic_width_useful_ios; });
}

double diskann_search_result_t::dynamic_width_wasted_ios_per_query() const {
  return get_mean_stats<uint32_t>(
      query_stats.data(), query_stats.size(),
      [](const query_stats_t& stats) { return stats.dynamic_width_wasted_ios; });
}

double diskann_search_result_t::community_meta_expansions_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.community_meta_expansions;
  });
}

double diskann_search_result_t::community_superedges_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.community_superedges_considered;
  });
}

double diskann_search_result_t::gateway_nodes_scored_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.gateway_nodes_scored; });
}

double diskann_search_result_t::gateway_nodes_competitive_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.gateway_nodes_competitive;
  });
}

double diskann_search_result_t::gateway_nodes_inserted_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.gateway_nodes_inserted; });
}

double diskann_search_result_t::gateway_nodes_later_expanded_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.gateway_nodes_later_expanded;
  });
}

double diskann_search_result_t::gateway_prefetch_predictions_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.gateway_prefetch_predictions;
  });
}

double diskann_search_result_t::gateway_prefetch_pages_submitted_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.gateway_prefetch_pages_submitted;
  });
}

double diskann_search_result_t::gateway_prefetch_queue_rejections_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.gateway_prefetch_queue_rejections;
  });
}

double diskann_search_result_t::gateway_prefetch_timely_hits_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.gateway_prefetch_timely_hits;
  });
}

double diskann_search_result_t::gateway_prefetch_late_hits_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.gateway_prefetch_late_hits;
  });
}

double diskann_search_result_t::gateway_prefetch_unused_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.gateway_prefetch_unused; });
}

double diskann_search_result_t::gateway_prefetch_mean_lead_us() const {
  uint64_t hits = 0;
  uint64_t lead = 0;
  for (const auto& stats : query_stats) {
    hits += stats.gateway_prefetch_timely_hits;
    lead += stats.gateway_prefetch_lead_us;
  }
  return hits == 0 ? 0.0 : static_cast<double>(lead) / static_cast<double>(hits);
}

double diskann_search_result_t::gateway_prefetch_mean_lead_hops() const {
  uint64_t hits = 0;
  uint64_t lead = 0;
  for (const auto& stats : query_stats) {
    hits += stats.gateway_prefetch_timely_hits;
    lead += stats.gateway_prefetch_lead_hops;
  }
  return hits == 0 ? 0.0 : static_cast<double>(lead) / static_cast<double>(hits);
}

double diskann_search_result_t::cell_candidates_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_candidates_created; });
}

double diskann_search_result_t::cell_blocks_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_blocks_expanded; });
}

double diskann_search_result_t::cell_hint_rejections_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_blocks_hint_rejected;
  });
}

double diskann_search_result_t::cell_nodes_scored_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_nodes_scored; });
}

double diskann_search_result_t::cell_nodes_competitive_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_nodes_competitive; });
}

double diskann_search_result_t::cell_nodes_inserted_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_nodes_inserted; });
}

double diskann_search_result_t::cell_nodes_later_expanded_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_nodes_later_expanded;
  });
}

double diskann_search_result_t::cell_nodes_already_visited_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_nodes_already_visited;
  });
}

double diskann_search_result_t::cell_blocks_without_unseen_nodes_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_blocks_without_unseen_nodes;
  });
}

double diskann_search_result_t::cell_blocks_low_yield_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_blocks_low_yield; });
}

double diskann_search_result_t::cell_packet_bytes_per_query() const {
  return get_mean_stats<uint64_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_packet_bytes; });
}

double diskann_search_result_t::cell_batch_search_rounds_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_batch_search_rounds; });
}

double diskann_search_result_t::cell_batch_cached_expansions_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_batch_cached_expansions;
  });
}

double diskann_search_result_t::cell_pq_traversal_expansions_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_pq_traversal_expansions;
  });
}

double diskann_search_result_t::cell_pq_refined_candidates_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_pq_refined_candidates;
  });
}

double diskann_search_result_t::cell_hierarchy_nodes_scored_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_hierarchy_nodes_scored;
  });
}

double diskann_search_result_t::cell_hierarchy_leaf_groups_probed_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_hierarchy_leaf_groups_probed;
  });
}

double diskann_search_result_t::cell_adj_neighbor_edges_considered_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_adj_neighbor_edges_considered;
  });
}

double diskann_search_result_t::cell_adj_cells_scored_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_adj_cells_scored; });
}

double diskann_search_result_t::cell_stitch_candidates_considered_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_stitch_candidates_considered;
  });
}

double diskann_search_result_t::cell_stitch_owner_cells_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(),
                                  [](const auto& stats) { return stats.cell_stitch_owner_cells; });
}

double diskann_search_result_t::cell_stitch_max_owner_support_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_stitch_max_owner_support;
  });
}

double diskann_search_result_t::cell_stitch_multi_owner_candidates_per_query() const {
  return get_mean_stats<uint32_t>(query_stats.data(), query_stats.size(), [](const auto& stats) {
    return stats.cell_stitch_multi_owner_candidates;
  });
}

double diskann_search_result_t::fallback_queries() const {
  double total = 0.0;
  for (const auto& stats : query_stats) {
    total += stats.fallback_queries;
  }
  return total;
}

namespace {

template <typename accessor_t>
double transitioned_mean(const std::vector<query_stats_t>& query_stats, accessor_t accessor) {
  double total = 0.0;
  size_t count = 0;
  for (const auto& stats : query_stats) {
    if (stats.beam_phase_transition_found == 0) {
      continue;
    }
    total += static_cast<double>(accessor(stats));
    ++count;
  }
  return count == 0 ? 0.0 : total / static_cast<double>(count);
}

} // namespace

double diskann_search_result_t::beam_phase_transition_rate_percent() const {
  if (query_stats.empty()) {
    return 0.0;
  }
  size_t transitioned = 0;
  for (const auto& stats : query_stats) {
    transitioned += stats.beam_phase_transition_found != 0 ? 1U : 0U;
  }
  return 100.0 * static_cast<double>(transitioned) / static_cast<double>(query_stats.size());
}

double diskann_search_result_t::beam_phase_approach_hops_per_query() const {
  return transitioned_mean(query_stats,
                           [](const auto& stats) { return stats.beam_phase_approach_base_hops; });
}

double diskann_search_result_t::beam_phase_convergence_hops_per_query() const {
  return transitioned_mean(
      query_stats, [](const auto& stats) { return stats.beam_phase_convergence_base_hops; });
}

double diskann_search_result_t::beam_phase_approach_ios_per_query() const {
  return transitioned_mean(query_stats,
                           [](const auto& stats) { return stats.beam_phase_approach_ios; });
}

double diskann_search_result_t::beam_phase_convergence_ios_per_query() const {
  return transitioned_mean(query_stats,
                           [](const auto& stats) { return stats.beam_phase_convergence_ios; });
}

double diskann_search_result_t::beam_phase_approach_time_us() const {
  return transitioned_mean(query_stats, [](const auto& stats) {
    return stats.beam_phase_setup_us + stats.beam_phase_approach_us;
  });
}

double diskann_search_result_t::beam_phase_convergence_time_us() const {
  return transitioned_mean(query_stats, [](const auto& stats) {
    return stats.beam_phase_convergence_us + stats.beam_phase_post_search_us;
  });
}

double diskann_search_result_t::beam_phase_setup_time_us() const {
  return transitioned_mean(query_stats,
                           [](const auto& stats) { return stats.beam_phase_setup_us; });
}

double diskann_search_result_t::beam_phase_post_search_time_us() const {
  return transitioned_mean(query_stats,
                           [](const auto& stats) { return stats.beam_phase_post_search_us; });
}

double diskann_search_result_t::beam_phase_approach_community_expansions_per_query() const {
  return transitioned_mean(query_stats, [](const auto& stats) {
    return stats.beam_phase_approach_community_expansions;
  });
}

double diskann_search_result_t::beam_phase_convergence_community_expansions_per_query() const {
  return transitioned_mean(query_stats, [](const auto& stats) {
    return stats.beam_phase_convergence_community_expansions;
  });
}

double diskann_search_result_t::beam_phase_approach_gateway_nodes_scored_per_query() const {
  return transitioned_mean(query_stats, [](const auto& stats) {
    return stats.beam_phase_approach_gateway_nodes_scored;
  });
}

double diskann_search_result_t::beam_phase_convergence_gateway_nodes_scored_per_query() const {
  return transitioned_mean(query_stats, [](const auto& stats) {
    return stats.beam_phase_convergence_gateway_nodes_scored;
  });
}

double diskann_search_result_t::beam_phase_approach_cell_blocks_per_query() const {
  return transitioned_mean(query_stats,
                           [](const auto& stats) { return stats.beam_phase_approach_cell_blocks; });
}

double diskann_search_result_t::beam_phase_convergence_cell_blocks_per_query() const {
  return transitioned_mean(
      query_stats, [](const auto& stats) { return stats.beam_phase_convergence_cell_blocks; });
}

double diskann_search_result_t::beam_phase_approach_cell_nodes_scored_per_query() const {
  return transitioned_mean(
      query_stats, [](const auto& stats) { return stats.beam_phase_approach_cell_nodes_scored; });
}

double diskann_search_result_t::beam_phase_convergence_cell_nodes_scored_per_query() const {
  return transitioned_mean(query_stats, [](const auto& stats) {
    return stats.beam_phase_convergence_cell_nodes_scored;
  });
}

namespace {

std::vector<diskann_search_result_t>
execute_search_curve(const diskann_search_config_t& input_config,
                     const community_polar_search_config_t* hybrid_config,
                     const io_optimization_config_t* io_config, bool observe_only,
                     std::span<const uint32_t> search_list_sizes,
                     std::span<const uint32_t> thread_counts = {},
                     std::span<const uint64_t> cache_page_counts = {}) {
  if (search_list_sizes.empty()) {
    throw ann_exception_t("DiskANN search curve requires at least one search-list size");
  }
  const bool thread_curve = !thread_counts.empty();
  const bool cache_curve = !cache_page_counts.empty();
  if (thread_curve && cache_curve) {
    throw ann_exception_t("DiskANN thread and cache curves are mutually exclusive");
  }
  if (thread_curve && search_list_sizes.size() != 1) {
    throw ann_exception_t("DiskANN thread curve requires exactly one search-list size");
  }
  if (cache_curve && search_list_sizes.size() != 1) {
    throw ann_exception_t("DiskANN cache curve requires exactly one search-list size");
  }
  std::vector<uint32_t> unique_search_list_sizes(search_list_sizes.begin(),
                                                 search_list_sizes.end());
  std::sort(unique_search_list_sizes.begin(), unique_search_list_sizes.end());
  if (std::adjacent_find(unique_search_list_sizes.begin(), unique_search_list_sizes.end()) !=
      unique_search_list_sizes.end()) {
    throw ann_exception_t("DiskANN search curve requires unique search-list sizes");
  }
  std::vector<uint32_t> unique_thread_counts(thread_counts.begin(), thread_counts.end());
  std::sort(unique_thread_counts.begin(), unique_thread_counts.end());
  if (std::adjacent_find(unique_thread_counts.begin(), unique_thread_counts.end()) !=
      unique_thread_counts.end()) {
    throw ann_exception_t("DiskANN thread curve requires unique thread counts");
  }
  if (std::any_of(thread_counts.begin(), thread_counts.end(),
                  [](uint32_t value) { return value == 0; })) {
    throw ann_exception_t("DiskANN thread curve requires positive thread counts");
  }
  std::vector<uint64_t> unique_cache_page_counts(cache_page_counts.begin(),
                                                 cache_page_counts.end());
  std::sort(unique_cache_page_counts.begin(), unique_cache_page_counts.end());
  if (std::adjacent_find(unique_cache_page_counts.begin(), unique_cache_page_counts.end()) !=
      unique_cache_page_counts.end()) {
    throw ann_exception_t("DiskANN cache curve requires unique cache capacities");
  }
  if (std::any_of(cache_page_counts.begin(), cache_page_counts.end(),
                  [](uint64_t value) { return value == 0; })) {
    throw ann_exception_t("DiskANN cache curve requires positive page capacities");
  }
  if (cache_curve && (io_config == nullptr || !io_config->enable_topology_vector ||
                      !io_config->enable_lru_cache)) {
    throw ann_exception_t("DiskANN cache curve requires topology/vector search with LRU enabled");
  }
  const size_t point_count =
      thread_curve ? thread_counts.size()
                   : cache_curve ? cache_page_counts.size() : search_list_sizes.size();
  for (size_t point = 0; point < point_count; ++point) {
    auto point_config = input_config;
    point_config.search_list_size =
        (thread_curve || cache_curve) ? search_list_sizes.front() : search_list_sizes[point];
    if (thread_curve) {
      point_config.num_threads = thread_counts[point];
    }
    validate_config(point_config);
  }
  if (point_count > 1 && (!input_config.node_visit_output_path.empty() ||
                          !input_config.cell_oracle_output_path.empty() ||
                          !input_config.cell_value_output_path.empty() ||
                          !input_config.gateway_value_output_path.empty() ||
                          !input_config.base_trace_output_path.empty())) {
    throw ann_exception_t("Multi-point search excludes per-query telemetry outputs");
  }
  if (point_count > 1 && io_config != nullptr && io_config->enable_online_cell_adaptation) {
    throw ann_exception_t("Multi-point search excludes online Cell adaptation");
  }
  diskann_search_config_t config = input_config;
  config.search_list_size = search_list_sizes.front();
  validate_config(config);
  if (config.warmup_query_count != 0 && io_config != nullptr &&
      io_config->enable_online_cell_adaptation) {
    throw ann_exception_t("In-process warmup cannot be combined with online Cell adaptation");
  }
  std::optional<community_polar_search_config_t> cost_gated_config;
  std::optional<io_optimization_config_t> l_aware_io_config;
  if (io_config != nullptr && io_config->enable_l_aware_search) {
    const bool first_uses_resident_policy =
        search_list_sizes.front() <= io_config->l_aware_decoded_max_l;
    if (std::any_of(search_list_sizes.begin(), search_list_sizes.end(), [&](uint32_t value) {
          return (value <= io_config->l_aware_decoded_max_l) != first_uses_resident_policy;
        })) {
      throw ann_exception_t("Multi-L search cannot cross the L-aware policy threshold");
    }
  }
  if (hybrid_config != nullptr && hybrid_config->cell_expansion) {
    const bool first_uses_cells =
        search_list_sizes.front() >= hybrid_config->cell_min_search_list_size;
    const bool crosses_cell_threshold =
        std::any_of(search_list_sizes.begin(), search_list_sizes.end(), [&](uint32_t value) {
          return (value >= hybrid_config->cell_min_search_list_size) != first_uses_cells;
        });
    if (crosses_cell_threshold && !hybrid_config->community_skip) {
      throw ann_exception_t("Multi-L Cell-only search cannot cross the Cell activation threshold");
    }
  }
  if (!observe_only && io_config != nullptr && io_config->enable_l_aware_search &&
      config.search_list_size <= io_config->l_aware_decoded_max_l) {
    l_aware_io_config = *io_config;
    l_aware_io_config->enable_cell_pq_decoded_u8 =
        !l_aware_io_config->enable_compressed_pq_traversal;
    l_aware_io_config->enable_cell_pq_dense_visited = true;
    l_aware_io_config->enable_cell_pq_filter_visited = true;
    l_aware_io_config->enable_cell_batch_search = false;
    if (l_aware_io_config->cell_pq_refine_candidates == 0) {
      l_aware_io_config->cell_pq_refine_prefetch_hop = 0;
      l_aware_io_config->enable_cell_page_batch = false;
      l_aware_io_config->enable_lru_cache = false;
      l_aware_io_config->lru_cache_pages = 0;
    }
    io_config = &*l_aware_io_config;
    hybrid_config = nullptr;
  }
  std::optional<io_optimization_config_t> forced_tree_io_config;
  if (!observe_only && io_config != nullptr && io_config->enable_forced_cell_tree_search) {
    forced_tree_io_config = *io_config;
    forced_tree_io_config->enable_cell_pq_decoded_u8 =
        !forced_tree_io_config->enable_compressed_pq_traversal;
    forced_tree_io_config->enable_cell_pq_dense_visited = true;
    forced_tree_io_config->enable_cell_pq_filter_visited = true;
    io_config = &*forced_tree_io_config;
    if (hybrid_config != nullptr) {
      cost_gated_config = *hybrid_config;
      cost_gated_config->fast_query_distance = true;
      cost_gated_config->source_cell_batching = true;
      cost_gated_config->cell_defer_hierarchy_routing = true;
      hybrid_config = &*cost_gated_config;
    }
  }
  if (io_config != nullptr && io_config->enable_resident_u8_refinement &&
      config.resident_u8_refinement_path.empty()) {
    throw ann_exception_t("Resident uint8 refinement requires an explicit artifact path");
  }
  if (hybrid_config != nullptr && !config.cell_value_output_path.empty()) {
    cost_gated_config = *hybrid_config;
    cost_gated_config->cell_value_telemetry = true;
    hybrid_config = &*cost_gated_config;
  }
  if (hybrid_config != nullptr && !config.gateway_value_output_path.empty()) {
    if (!cost_gated_config.has_value()) {
      cost_gated_config = *hybrid_config;
    }
    cost_gated_config->gateway_value_telemetry = true;
    hybrid_config = &*cost_gated_config;
  }
  if (!observe_only && hybrid_config != nullptr && io_config != nullptr &&
      io_config->enable_cell_pq_traversal) {
    if (!cost_gated_config.has_value()) {
      cost_gated_config = *hybrid_config;
    }
    cost_gated_config->fast_query_distance = true;
    cost_gated_config->source_cell_batching = true;
    hybrid_config = &*cost_gated_config;
  }
  if (search_list_sizes.size() == 1 && !observe_only && hybrid_config != nullptr &&
      hybrid_config->cell_expansion &&
      config.search_list_size < hybrid_config->cell_min_search_list_size) {
    if (!hybrid_config->community_skip) {
      hybrid_config = nullptr;
    } else {
      if (!cost_gated_config.has_value()) {
        cost_gated_config = *hybrid_config;
      }
      cost_gated_config->cell_expansion = false;
      cost_gated_config->cell_whole_cell_scan = false;
      cost_gated_config->cell_centroid_routing = false;
      hybrid_config = &*cost_gated_config;
    }
  }

  std::shared_ptr<aligned_file_reader_t> reader;
#if defined(__linux__)
  std::shared_ptr<io_uring_aligned_file_reader_t> io_uring_reader;
#endif
#if defined(__APPLE__)
  reader = std::make_shared<macos_aligned_file_reader_t>();
#elif defined(__linux__)
  if (io_config != nullptr && io_config->enable_io_uring) {
    io_uring_reader =
        std::make_shared<io_uring_aligned_file_reader_t>(io_config->io_uring_queue_depth);
    reader = io_uring_reader;
  } else {
    reader = std::make_shared<linux_aligned_file_reader_t>();
  }
#else
  throw ann_exception_t("DiskANN search is supported only on macOS and Linux");
#endif

  const uint32_t load_thread_count =
      thread_curve ? *std::max_element(thread_counts.begin(), thread_counts.end())
                   : (config.num_threads == 0 ? static_cast<uint32_t>(omp_get_max_threads())
                                              : config.num_threads);
  std::shared_ptr<const io_optimized_index_t> io_index;
  std::shared_ptr<io_lru_buffer_pool_t> io_cache;
  if (io_config != nullptr && io_config->enable_topology_vector) {
    if (observe_only) {
      throw ann_exception_t("IO-optimized layout is unsupported in observe-only search");
    }
    try {
      const bool topology_only =
          io_config->enable_cell_pq_traversal && !io_config->enable_lru_cache &&
          io_config->cell_pq_refine_candidates == 0 &&
          io_config->cell_pq_refine_prefetch_hop == 0 && !io_config->enable_cell_leaf_refinement &&
          !io_config->enable_node_prefetch && !io_config->enable_cell_prefetch &&
          !io_config->enable_gateway_prefetch;
      io_index = io_optimized_index_t::load(
          config.index_path_prefix, config.io_topology_path, config.io_vector_path, topology_only,
          io_config->enable_paged_topology, io_config->topology_cache_pages);
      const auto expected_layout =
          io_config->enable_cell_u8_layout
              ? io_vector_layout_t::CELL_U8_4K
              : io_config->enable_chunk_layout
                    ? io_vector_layout_t::CELL_CHUNKS
                    : (io_config->enable_weighted_reorder
                           ? io_vector_layout_t::CELL_WEIGHTED_4K
                           : (io_config->enable_cell_layout ? io_vector_layout_t::CELL_4K
                                                            : io_vector_layout_t::ORIGINAL_ID));
      if (io_index->vector_layout() != expected_layout) {
        throw std::runtime_error("loaded IO vector layout disagrees with the requested switch");
      }
      if (io_config->enable_lru_cache) {
        const uint64_t initial_cache_pages =
            cache_curve ? cache_page_counts.front() : io_config->lru_cache_pages;
        io_cache = std::make_shared<io_lru_buffer_pool_t>(initial_cache_pages);
      }
    } catch (const std::exception& error) {
      throw ann_exception_t("IO-optimized index load failed: " + std::string(error.what()));
    }
  }
  pq_flash_index_t<float> index(reader, metric_t::L2);
  if (index.load(load_thread_count, config.index_path_prefix.c_str(), io_index, io_cache,
                 io_config != nullptr && io_config->enable_cell_page_batch,
                 io_config != nullptr && io_config->enable_cell_batch_search,
                 io_config != nullptr ? io_config->cell_batch_max_cached_expansions : 16,
                 io_config != nullptr && io_config->enable_cell_pq_traversal,
                 io_config != nullptr ? io_config->cell_pq_refine_candidates : 0,
                 io_config != nullptr ? io_config->cell_pq_refine_prefetch_hop : 0,
                 io_config != nullptr && io_config->enable_cell_leaf_refinement,
                 io_config != nullptr && io_config->enable_cell_pq_dense_visited,
                 io_config != nullptr && io_config->enable_cell_pq_filter_visited,
                 io_config != nullptr && io_config->enable_cell_pq_decoded_u8,
                 io_config != nullptr && io_config->enable_node_prefetch,
                 io_config != nullptr && io_config->enable_cell_prefetch,
                 io_config != nullptr && io_config->enable_gateway_prefetch,
                 io_config != nullptr ? io_config->gateway_prefetch_max_outstanding : 2,
                 io_config != nullptr && io_config->enable_dynamic_width,
                 io_config != nullptr ? io_config->dynamic_width_initial : 4,
                 io_config != nullptr ? io_config->dynamic_width_marker : 5,
                 io_config != nullptr ? io_config->dynamic_width_waste_threshold : 0.1F,
                 io_config != nullptr ? io_config->cell_pq_decoded_scale : 1.0F) != 0) {
    throw ann_exception_t("Failed to load DiskANN disk index");
  }
  if (io_config != nullptr && io_config->enable_resident_u8_refinement) {
    index.configure_resident_u8_refinement(config.resident_u8_refinement_path,
                                           io_config->enable_resident_u8_traversal);
  }
  if (io_config != nullptr && io_config->enable_l_aware_refine_budget) {
    index.configure_l_aware_refinement(io_config->l_aware_refine_base,
                                       io_config->l_aware_refine_divisor);
  }
  if (io_config != nullptr && io_config->cell_pq_refine_min_l != 1) {
    index.configure_cell_pq_refine_min_l(io_config->cell_pq_refine_min_l);
  }
  if (io_config != nullptr && io_config->pq_score_chunks != 0) {
    index.configure_pq_score_chunks(io_config->pq_score_chunks);
  }
  if (io_config != nullptr && io_config->pq_score_quantization_bits != 0) {
    index.configure_pq_score_quantization(io_config->pq_score_quantization_bits);
  }
  if (io_config != nullptr && io_config->graph_neighbor_score_limit != 0) {
    index.configure_graph_neighbor_score_limit(io_config->graph_neighbor_score_limit);
  }

  std::shared_ptr<community_polar_search_t> hybrid_search;
  std::shared_ptr<const io_cell_adjacency_index_t> cell_adjacency;
  std::shared_ptr<adaptive_cell_runtime_t> adaptive_cell_runtime;
  if (hybrid_config != nullptr) {
    try {
      hybrid_search = community_polar_search_t::load(config.index_path_prefix, *hybrid_config,
                                                     config.community_polar_path);
    } catch (const std::exception& error) {
      throw ann_exception_t("Community-Polar search load failed: " + std::string(error.what()));
    }
    if (hybrid_search->point_count() != index.get_num_points() ||
        hybrid_search->dimension() != index.get_data_dim()) {
      throw ann_exception_t("Community-Polar sidecar shape disagrees with the Base index");
    }
    if (io_config != nullptr &&
        (io_config->enable_cell_batch_search || io_config->enable_cell_pq_traversal) &&
        (io_index == nullptr ||
         io_index->community_polar_fingerprint() != hybrid_search->artifact_fingerprint())) {
      throw ann_exception_t(
          "Cell-batched/PQ search requires a Cell-4K artifact built from the loaded sidecar");
    }
    if (io_config != nullptr && io_config->enable_forced_cell_tree_search) {
      index.configure_forced_cell_tree(hybrid_search);
      hybrid_search->compact_for_forced_tree();
    }
    if (io_config != nullptr && io_config->enable_cell_adj_correction) {
      if (!config.io_cell_adjacency_path.empty()) {
        try {
          cell_adjacency = io_cell_adjacency_index_t::load(
              config.io_cell_adjacency_path, hybrid_search->point_count(),
              hybrid_search->cell_count(), hybrid_search->artifact_fingerprint());
        } catch (const std::exception& error) {
          throw ann_exception_t("Cell adjacency load failed: " + std::string(error.what()));
        }
      }
      index.configure_cell_adj_correction(
          io_config->cell_adj_candidate_cells, io_config->cell_adj_support_shortlist,
          cell_adjacency, io_config->cell_adj_expansion_cells,
          io_config->cell_adj_min_search_list_size, io_config->cell_adj_max_search_list_size);
    }
    if (io_config != nullptr && io_config->enable_interleaved_cell_batch_search) {
      index.configure_interleaved_cell_batch_search(io_config->interleaved_cell_batch_graph_hops);
    }
    if (io_config != nullptr && io_config->cell_batch_preserve_graph) {
      index.configure_cell_batch_preserve_graph();
    }
    if (io_config != nullptr && io_config->cell_batch_graph_stitch_min_l != 0) {
      index.configure_cell_batch_graph_stitch(io_config->cell_batch_graph_stitch_min_l);
    }
    if (io_config != nullptr && io_config->enable_ranked_cell_page_refinement) {
      index.configure_ranked_cell_page_refinement(
          io_config->ranked_cell_page_base_pages, io_config->ranked_cell_page_l_divisor,
          io_config->ranked_cell_page_growth_start_l, io_config->ranked_cell_page_growth_divisor,
          io_config->ranked_cell_page_max_pages, io_config->ranked_cell_page_max_l);
    }
    if (io_config != nullptr && io_config->enable_gateway_prefetch) {
      if (io_index == nullptr || !io_index->has_gateway_prefetch_hints() ||
          io_index->gateway_prefetch_hint_count() != hybrid_search->gateway_count() ||
          io_index->community_polar_fingerprint() != hybrid_search->artifact_fingerprint()) {
        throw ann_exception_t(
            "Gateway prefetch metadata does not match the loaded Community-Polar sidecar");
      }
    }
  }

  if (io_config != nullptr && io_config->enable_online_cell_adaptation) {
    if (io_index == nullptr || io_index->cell_page_ranges().size() < 2) {
      throw ann_exception_t("Online Cell adaptation requires a nonempty Cell-4K directory");
    }
    const auto ranges = io_index->cell_page_ranges();
    std::vector<uint32_t> page_to_cell(io_index->vector_page_count(), UINT32_MAX);
    std::vector<std::vector<uint32_t>> cell_nodes(ranges.size());
    for (uint32_t cell = 0; cell < ranges.size(); ++cell) {
      const auto& range = ranges[cell];
      for (uint64_t page = range.first_page; page < range.first_page + range.page_count; ++page) {
        if (page >= page_to_cell.size() || page_to_cell[page] != UINT32_MAX) {
          throw ann_exception_t("Online Cell adaptation found overlapping physical Cell pages");
        }
        page_to_cell[page] = cell;
      }
      cell_nodes[cell].reserve(range.node_count);
    }
    for (uint32_t node = 0; node < io_index->point_count(); ++node) {
      const auto location = io_index->vector_location(node);
      if (location.page_id >= page_to_cell.size() || page_to_cell[location.page_id] == UINT32_MAX) {
        throw ann_exception_t("Online Cell adaptation cannot resolve a node's physical Cell");
      }
      cell_nodes[page_to_cell[location.page_id]].push_back(node);
    }
    std::vector<adaptive_cell_merge_candidate_t> candidates;
    candidates.reserve(ranges.size());
    const uint32_t stride = ranges.size() > 2 ? 2U : 1U;
    for (uint32_t cell = 0; cell + stride < ranges.size(); ++cell) {
      if (ranges[cell].node_count + ranges[cell + stride].node_count <= 512U) {
        candidates.push_back({cell, cell + stride});
      }
    }
    if (candidates.empty()) {
      throw ann_exception_t("Online Cell adaptation has no bounded modification candidate");
    }
    adaptive_cell_runtime_config_t adaptive_config;
    adaptive_config.enabled = true;
    adaptive_config.minimum_joint_observations = 32;
    adaptive_config.maximum_successful_publishes = 1;
    adaptive_config.background_publish = true;
    adaptive_config.force_first_observed_modification = true;
    auto overlay_path = io_index->vector_path();
    overlay_path += ".online." + std::to_string(static_cast<uint64_t>(::getpid())) + ".delta";
    adaptive_cell_physical_config_t physical_config;
    physical_config.base_index = io_index;
    physical_config.overlay_path = overlay_path;
    physical_config.vector_bytes = static_cast<uint32_t>(index.get_data_dim() * sizeof(float));
    physical_config.cell_nodes = std::move(cell_nodes);
    adaptive_cell_runtime = std::make_shared<adaptive_cell_runtime_t>(
        ranges, candidates, adaptive_config, std::move(physical_config));
    reader->open_overlay(overlay_path.string(),
                         io_index->vector_page_count() * k_io_vector_page_size);
    index.configure_adaptive_cell_runtime(adaptive_cell_runtime);
  }

  if (io_config != nullptr && io_config->memgraph_mode != io_memgraph_mode_t::NONE) {
    if (io_config->memgraph_nodes > index.get_num_points()) {
      throw ann_exception_t("MemGraph node budget exceeds the Base point count");
    }
    std::vector<uint32_t> nodes;
    if (io_config->memgraph_mode == io_memgraph_mode_t::RANDOM) {
      using ranked_node_t = std::pair<uint64_t, uint32_t>;
      std::priority_queue<ranked_node_t> selected;
      for (uint32_t node = 0; node < index.get_num_points(); ++node) {
        const ranked_node_t candidate{mix_memgraph_id(io_config->memgraph_seed ^ node), node};
        if (selected.size() < io_config->memgraph_nodes) {
          selected.push(candidate);
        } else if (candidate < selected.top()) {
          selected.pop();
          selected.push(candidate);
        }
      }
      std::vector<ranked_node_t> ordered;
      while (!selected.empty()) {
        ordered.push_back(selected.top());
        selected.pop();
      }
      std::sort(ordered.begin(), ordered.end());
      nodes.reserve(ordered.size());
      for (const auto& candidate : ordered) {
        nodes.push_back(candidate.second);
      }
    } else if (io_config->memgraph_mode == io_memgraph_mode_t::COMMUNITY_IMPORTANT) {
      if (hybrid_search == nullptr) {
        throw ann_exception_t("Community-important MemGraph requires Hybrid sidecar load");
      }
      nodes = hybrid_search->important_memgraph_nodes(io_config->memgraph_nodes);
    } else if (io_config->memgraph_mode == io_memgraph_mode_t::HIGH_DEGREE) {
      std::vector<std::pair<uint32_t, uint32_t>> ranked;
      ranked.reserve(index.get_num_points());
      for (uint32_t node = 0; node < index.get_num_points(); ++node) {
        ranked.emplace_back(static_cast<uint32_t>(io_index->neighbors(node).size()), node);
      }
      std::partial_sort(ranked.begin(), ranked.begin() + io_config->memgraph_nodes, ranked.end(),
                        [](const auto& left, const auto& right) {
                          return left.first > right.first ||
                                 (left.first == right.first && left.second < right.second);
                        });
      nodes.reserve(io_config->memgraph_nodes);
      for (uint32_t position = 0; position < io_config->memgraph_nodes; ++position) {
        nodes.push_back(ranked[position].second);
      }
    } else {
      auto rcni_path = config.index_path_prefix;
      rcni_path += ".rcni.csv";
      nodes = read_rcni_memgraph_nodes(rcni_path, static_cast<uint32_t>(index.get_num_points()),
                                       io_config->memgraph_nodes);
    }
    index.configure_memgraph(std::move(nodes), io_config->memgraph_entry_candidates);
  }

  std::vector<uint32_t> cache_nodes;
  index.cache_bfs_levels(config.num_nodes_to_cache, cache_nodes);
  index.load_cache_list(cache_nodes);

  float* queries = nullptr;
  size_t num_queries = 0;
  size_t query_dimension = 0;
  size_t aligned_query_dimension = 0;
  load_aligned_bin<float>(config.query_path.string(), queries, num_queries, query_dimension,
                          aligned_query_dimension);
  std::unique_ptr<float, decltype(&aligned_free)> query_owner(queries, &aligned_free);
  const size_t available_queries = num_queries;
  if (config.query_offset > available_queries) {
    throw ann_exception_t("Measured query offset exceeds the query artifact");
  }
  num_queries -= config.query_offset;
  if (config.query_limit != 0) {
    num_queries = std::min<size_t>(num_queries, config.query_limit);
  }
  if (query_dimension != index.get_data_dim()) {
    throw ann_exception_t("Query dimension does not match the DiskANN index");
  }

  std::vector<diskann_search_result_t> curve_results;
  curve_results.reserve(point_count);
  for (size_t point = 0; point < point_count; ++point) {
    config.search_list_size =
        (thread_curve || cache_curve) ? search_list_sizes.front() : search_list_sizes[point];
    const uint32_t point_thread_count = thread_curve ? thread_counts[point] : load_thread_count;
    config.num_threads = point_thread_count;
    if (io_cache != nullptr) {
      if (cache_curve) {
        io_cache->reset_capacity(cache_page_counts[point]);
      } else {
        io_cache->clear();
      }
    }
    diskann_search_result_t result;
    result.num_queries = num_queries;
    result.query_dimension = query_dimension;
    result.top_k = config.top_k;
    result.num_threads = point_thread_count;
    result.memgraph_resident_bytes = index.memgraph_resident_bytes();
    result.memgraph_nodes = index.memgraph_node_count();
    result.memgraph_build_time_us = index.memgraph_build_time_us();
    result.memgraph_build_peak_bytes = index.memgraph_build_peak_bytes();
    result.decoded_pq_build_time_us = index.decoded_pq_build_time_us();
    result.decoded_pq_representation_bytes = index.decoded_pq_representation_bytes();
    result.decoded_pq_additional_resident_bytes = index.decoded_pq_additional_resident_bytes();
    result.resident_u8_refinement_artifact_bytes = index.resident_u8_refinement_artifact_bytes();
    result.resident_u8_refinement_bytes = index.resident_u8_refinement_bytes();
    result.resident_u8_refinement_checksum = index.resident_u8_refinement_checksum();
    result.index_size_bytes = calculate_index_size(config.index_path_prefix);
    if (hybrid_search != nullptr) {
      result.resident_sidecar_bytes = hybrid_search->resident_bytes();
      result.sidecar_artifact_bytes = hybrid_search->artifact_bytes();
    }
    if (cell_adjacency != nullptr) {
      result.cell_adjacency_resident_bytes = cell_adjacency->resident_bytes();
      result.cell_adjacency_artifact_bytes = cell_adjacency->artifact_bytes();
    }
    if (io_index != nullptr) {
      result.topology_resident_bytes = io_index->resident_bytes();
      result.topology_artifact_bytes = io_index->topology_bytes();
      result.vector_artifact_bytes = io_index->vector_bytes();
      result.topology_cache_resident_bytes = io_index->topology_cache_resident_bytes();
    }
    if (io_cache != nullptr) {
      result.io_cache_resident_bytes = io_cache->resident_bytes();
    }
    result.cache_state = config.cache_state;
    result.backends = active_backends();
#if defined(__linux__)
    if (io_uring_reader != nullptr) {
      result.backends.io = "io_uring_o_direct_demand";
    }
#endif
    result.ids.resize(num_queries * config.top_k);
    result.distances.resize(num_queries * config.top_k);
    result.query_stats.resize(num_queries);
    if (!config.base_trace_output_path.empty()) {
      for (auto& stats : result.query_stats) {
        stats.base_trace_enabled = true;
        stats.base_expansion_records.reserve(config.search_list_size + config.beam_width);
      }
    }

    omp_set_num_threads(static_cast<int>(point_thread_count));
    const auto search_one = [&](size_t query_id, uint64_t* output_ids, float* output_distances,
                                query_stats_t* output_stats) {
      if (hybrid_search == nullptr) {
        if (io_index != nullptr && config.phase_profile) {
          index.cached_beam_search_io_optimized_phase_profiled(
              queries + query_id * aligned_query_dimension, config.top_k, config.search_list_size,
              output_ids, output_distances, config.beam_width, config.io_limit, output_stats);
        } else if (io_index != nullptr) {
          index.cached_beam_search_io_optimized(
              queries + query_id * aligned_query_dimension, config.top_k, config.search_list_size,
              output_ids, output_distances, config.beam_width, config.io_limit, output_stats);
        } else if (config.phase_profile) {
          index.cached_beam_search_phase_profiled(
              queries + query_id * aligned_query_dimension, config.top_k, config.search_list_size,
              output_ids, output_distances, config.beam_width, config.io_limit,
              config.use_reorder_data, output_stats);
        } else {
          index.cached_beam_search(queries + query_id * aligned_query_dimension, config.top_k,
                                   config.search_list_size, output_ids, output_distances,
                                   config.beam_width, config.io_limit, config.use_reorder_data,
                                   output_stats);
        }
      } else if (observe_only) {
        index.cached_beam_search_observe_only(
            queries + query_id * aligned_query_dimension, config.top_k, config.search_list_size,
            output_ids, output_distances, config.beam_width, config.io_limit, *hybrid_search,
            config.use_reorder_data, output_stats);
      } else {
        if (io_index != nullptr && config.phase_profile) {
          index.cached_beam_search_hybrid_io_optimized_phase_profiled(
              queries + query_id * aligned_query_dimension, config.top_k, config.search_list_size,
              output_ids, output_distances, config.beam_width, config.io_limit, *hybrid_search,
              output_stats);
        } else if (io_index != nullptr) {
          index.cached_beam_search_hybrid_io_optimized(
              queries + query_id * aligned_query_dimension, config.top_k, config.search_list_size,
              output_ids, output_distances, config.beam_width, config.io_limit, *hybrid_search,
              output_stats);
        } else if (config.phase_profile) {
          index.cached_beam_search_hybrid_phase_profiled(
              queries + query_id * aligned_query_dimension, config.top_k, config.search_list_size,
              output_ids, output_distances, config.beam_width, config.io_limit, *hybrid_search,
              config.use_reorder_data, output_stats);
        } else {
          index.cached_beam_search_hybrid(queries + query_id * aligned_query_dimension,
                                          config.top_k, config.search_list_size, output_ids,
                                          output_distances, config.beam_width, config.io_limit,
                                          *hybrid_search, config.use_reorder_data, output_stats);
        }
      }
    };

    if (config.warmup_query_offset > available_queries ||
        config.warmup_query_count > available_queries - config.warmup_query_offset) {
      throw ann_exception_t("Warmup query range exceeds the query artifact");
    }
    const size_t warmup_queries = config.warmup_query_count;
    const size_t warmup_query_offset = config.warmup_query_offset;
    if (warmup_queries != 0) {
      std::vector<uint64_t> warmup_ids(warmup_queries * config.top_k);
      std::vector<float> warmup_distances(warmup_queries * config.top_k);
      std::vector<query_stats_t> warmup_stats(warmup_queries);
      if (io_config != nullptr && io_config->enable_forced_cell_tree_search) {
#pragma omp parallel for schedule(static)
        for (int64_t query_id = 0; query_id < static_cast<int64_t>(warmup_queries); ++query_id) {
          search_one(warmup_query_offset + static_cast<size_t>(query_id),
                     warmup_ids.data() + query_id * config.top_k,
                     warmup_distances.data() + query_id * config.top_k,
                     warmup_stats.data() + query_id);
        }
      } else {
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t query_id = 0; query_id < static_cast<int64_t>(warmup_queries); ++query_id) {
          search_one(warmup_query_offset + static_cast<size_t>(query_id),
                     warmup_ids.data() + query_id * config.top_k,
                     warmup_distances.data() + query_id * config.top_k,
                     warmup_stats.data() + query_id);
        }
      }
    }
    if (io_cache != nullptr && io_config != nullptr && io_config->reset_lru_after_warmup) {
      io_cache->clear();
    }

    const auto cache_stats_before_measurement =
        io_cache == nullptr ? io_lru_buffer_pool_stats_t{} : io_cache->stats();
#if defined(__linux__)
    const auto io_uring_stats_before_measurement =
        io_uring_reader == nullptr ? io_uring_reader_stats_t{} : io_uring_reader->stats();
#endif
    if (!config.node_visit_output_path.empty()) {
      index.start_node_visit_tracking();
    }

    const auto start = std::chrono::steady_clock::now();
    if (io_config != nullptr && io_config->enable_forced_cell_tree_search) {
#pragma omp parallel for schedule(static)
      for (int64_t query_id = 0; query_id < static_cast<int64_t>(num_queries); ++query_id) {
        search_one(config.query_offset + static_cast<size_t>(query_id),
                   result.ids.data() + query_id * config.top_k,
                   result.distances.data() + query_id * config.top_k,
                   result.query_stats.data() + query_id);
      }
    } else {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t query_id = 0; query_id < static_cast<int64_t>(num_queries); ++query_id) {
        search_one(config.query_offset + static_cast<size_t>(query_id),
                   result.ids.data() + query_id * config.top_k,
                   result.distances.data() + query_id * config.top_k,
                   result.query_stats.data() + query_id);
      }
    }
    if (adaptive_cell_runtime != nullptr) {
      adaptive_cell_runtime->drain_background();
      const auto adaptive_metrics = adaptive_cell_runtime->metrics();
      result.online_cell_adaptation_observations = adaptive_metrics.observations;
      result.online_cell_adaptation_publish_attempts = adaptive_metrics.publish_attempts;
      result.online_cell_adaptation_successful_publishes = adaptive_metrics.successful_publishes;
      result.online_cell_adaptation_publish_conflicts = adaptive_metrics.publish_conflicts;
      result.online_cell_adaptation_retired_snapshots = adaptive_metrics.retired_snapshots;
      result.online_cell_adaptation_physical_generations = adaptive_metrics.physical_generations;
      result.online_cell_adaptation_physical_bytes_written =
          adaptive_metrics.physical_bytes_written;
      result.online_cell_adaptation_physical_write_time_us =
          adaptive_metrics.physical_write_time_us;
      result.online_cell_adaptation_checksum_failures = adaptive_metrics.checksum_failures;
      result.online_cell_adaptation_directory_resident_bytes =
          adaptive_metrics.directory_resident_bytes;
      result.online_cell_adaptation_overlay_artifact_bytes =
          adaptive_metrics.overlay_artifact_bytes;
    }
    if (io_index != nullptr) {
      const auto topology_cache_stats = io_index->topology_cache_stats();
      result.topology_cache_hits = topology_cache_stats.demand_hits;
      result.topology_cache_misses = topology_cache_stats.misses;
      result.topology_cache_evictions = topology_cache_stats.evictions;
    }
    result.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (io_cache != nullptr) {
      const auto cache_stats = io_cache->stats();
      result.io_cache_demand_hits =
          cache_stats.demand_hits - cache_stats_before_measurement.demand_hits;
      result.io_cache_prefetch_hits =
          cache_stats.prefetch_hits - cache_stats_before_measurement.prefetch_hits;
      result.io_cache_prefetch_misses =
          cache_stats.prefetch_misses - cache_stats_before_measurement.prefetch_misses;
      result.io_cache_prefetch_used =
          cache_stats.prefetch_used - cache_stats_before_measurement.prefetch_used;
      result.io_cache_prefetch_pollution_evictions =
          cache_stats.prefetch_pollution_evictions -
          cache_stats_before_measurement.prefetch_pollution_evictions;
      result.io_cache_prefetch_admission_rejections =
          cache_stats.prefetch_admission_rejections -
          cache_stats_before_measurement.prefetch_admission_rejections;
      result.io_cache_prefetch_demand_evictions =
          cache_stats.prefetch_demand_evictions -
          cache_stats_before_measurement.prefetch_demand_evictions;
      result.io_cache_misses = cache_stats.misses - cache_stats_before_measurement.misses;
      result.io_cache_coalesced_misses =
          cache_stats.coalesced_misses - cache_stats_before_measurement.coalesced_misses;
      result.io_cache_evictions = cache_stats.evictions - cache_stats_before_measurement.evictions;
      result.io_cache_pinned_victim_failures =
          cache_stats.pinned_victim_failures -
          cache_stats_before_measurement.pinned_victim_failures;
    }
#if defined(__linux__)
    if (io_uring_reader != nullptr) {
      const auto reader_stats = io_uring_reader->stats();
      result.io_uring_submitted =
          reader_stats.submitted - io_uring_stats_before_measurement.submitted;
      result.io_uring_completed =
          reader_stats.completed - io_uring_stats_before_measurement.completed;
      result.io_uring_completion_batches =
          reader_stats.completion_batches - io_uring_stats_before_measurement.completion_batches;
      result.io_uring_wait_ns = reader_stats.wait_ns - io_uring_stats_before_measurement.wait_ns;
      result.io_uring_short_or_error_completions =
          reader_stats.short_or_error_completions -
          io_uring_stats_before_measurement.short_or_error_completions;
      result.io_uring_prefetch_submitted =
          reader_stats.prefetch_submitted - io_uring_stats_before_measurement.prefetch_submitted;
      result.io_uring_prefetch_completed =
          reader_stats.prefetch_completed - io_uring_stats_before_measurement.prefetch_completed;
      result.io_uring_prefetch_obsolete =
          reader_stats.prefetch_obsolete - io_uring_stats_before_measurement.prefetch_obsolete;
      result.io_uring_queue_depth = reader_stats.queue_depth;
    }
#endif
    calculate_result_recall(config, result);

    if (!config.node_visit_output_path.empty()) {
      index.stop_node_visit_tracking();
      index.dump_node_visit_counts(config.node_visit_output_path.string());
    }
    save_results(config, result);
    save_cell_oracle(config, result);
    save_cell_value_telemetry(config, result);
    save_gateway_value_telemetry(config, result);
    save_base_trace(config, result);
    curve_results.push_back(std::move(result));
  }
  return curve_results;
}

diskann_search_result_t execute_search(const diskann_search_config_t& config,
                                       const community_polar_search_config_t* hybrid_config,
                                       const io_optimization_config_t* io_config,
                                       bool observe_only) {
  const std::array<uint32_t, 1> search_list_sizes{config.search_list_size};
  auto results =
      execute_search_curve(config, hybrid_config, io_config, observe_only, search_list_sizes);
  return std::move(results.front());
}

} // namespace

diskann_search_result_t diskann_search_t::search(const diskann_search_config_t& config) {
  return execute_search(config, nullptr, nullptr, false);
}

diskann_search_result_t diskann_search_t::search(const diskann_search_config_t& config,
                                                 const io_optimization_config_t& io_config) {
  return execute_search(config, nullptr, &io_config, false);
}

diskann_search_result_t
diskann_search_t::search(const diskann_search_config_t& config,
                         const community_polar_search_config_t& hybrid_config) {
  try {
    validate_community_polar_search_config(hybrid_config);
  } catch (const std::invalid_argument& error) {
    throw ann_exception_t(error.what());
  }
  if (!hybrid_config.is_hybrid()) {
    throw ann_exception_t("Community-Polar search overload requires hybrid mode");
  }
  return execute_search(config, &hybrid_config, nullptr, false);
}

diskann_search_result_t
diskann_search_t::search(const diskann_search_config_t& config,
                         const community_polar_search_config_t& hybrid_config,
                         const io_optimization_config_t& io_config) {
  try {
    validate_community_polar_search_config(hybrid_config);
  } catch (const std::invalid_argument& error) {
    throw ann_exception_t(error.what());
  }
  if (!hybrid_config.is_hybrid()) {
    throw ann_exception_t("Community-Polar search overload requires hybrid mode");
  }
  return execute_search(config, &hybrid_config, &io_config, false);
}

diskann_search_result_t
diskann_search_t::search_observe_only(const diskann_search_config_t& config,
                                      const community_polar_search_config_t& hybrid_config) {
  try {
    validate_community_polar_search_config(hybrid_config);
  } catch (const std::invalid_argument& error) {
    throw ann_exception_t(error.what());
  }
  if (!hybrid_config.is_hybrid()) {
    throw ann_exception_t("Community-Polar observe-only search requires hybrid mode");
  }
  return execute_search(config, &hybrid_config, nullptr, true);
}

std::vector<diskann_search_result_t>
diskann_search_t::search_curve(const diskann_search_config_t& config,
                               std::span<const uint32_t> search_list_sizes) {
  return execute_search_curve(config, nullptr, nullptr, false, search_list_sizes);
}

std::vector<diskann_search_result_t> diskann_search_t::search_curve(
    const diskann_search_config_t& config, const community_polar_search_config_t& hybrid_config,
    const io_optimization_config_t& io_config, std::span<const uint32_t> search_list_sizes) {
  try {
    validate_community_polar_search_config(hybrid_config);
  } catch (const std::invalid_argument& error) {
    throw ann_exception_t(error.what());
  }
  if (!hybrid_config.is_hybrid()) {
    throw ann_exception_t("Community-Polar search curve requires hybrid mode");
  }
  return execute_search_curve(config, &hybrid_config, &io_config, false, search_list_sizes);
}

std::vector<diskann_search_result_t>
diskann_search_t::search_thread_curve(const diskann_search_config_t& config,
                                      std::span<const uint32_t> thread_counts) {
  const std::array<uint32_t, 1> search_list_sizes{config.search_list_size};
  return execute_search_curve(config, nullptr, nullptr, false, search_list_sizes, thread_counts);
}

std::vector<diskann_search_result_t> diskann_search_t::search_thread_curve(
    const diskann_search_config_t& config, const community_polar_search_config_t& hybrid_config,
    const io_optimization_config_t& io_config, std::span<const uint32_t> thread_counts) {
  try {
    validate_community_polar_search_config(hybrid_config);
  } catch (const std::invalid_argument& error) {
    throw ann_exception_t(error.what());
  }
  if (!hybrid_config.is_hybrid()) {
    throw ann_exception_t("Community-Polar thread curve requires hybrid mode");
  }
  const std::array<uint32_t, 1> search_list_sizes{config.search_list_size};
  return execute_search_curve(config, &hybrid_config, &io_config, false, search_list_sizes,
                              thread_counts);
}

std::vector<diskann_search_result_t> diskann_search_t::search_cache_curve(
    const diskann_search_config_t& config, const community_polar_search_config_t& hybrid_config,
    const io_optimization_config_t& io_config, std::span<const uint64_t> cache_page_counts) {
  try {
    validate_community_polar_search_config(hybrid_config);
  } catch (const std::invalid_argument& error) {
    throw ann_exception_t(error.what());
  }
  if (!hybrid_config.is_hybrid()) {
    throw ann_exception_t("Community-Polar cache curve requires hybrid mode");
  }
  const std::array<uint32_t, 1> search_list_sizes{config.search_list_size};
  return execute_search_curve(config, &hybrid_config, &io_config, false, search_list_sizes, {},
                              cache_page_counts);
}

} // namespace powerlaw_ann
