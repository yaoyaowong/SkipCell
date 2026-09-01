#include "power_ann.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct command_config_t {
  powerlaw_ann::power_ann_config_t power_ann;
  powerlaw_ann::diskann_search_config_t search;
  std::vector<uint32_t> search_list_series;
  std::vector<uint32_t> thread_series;
  std::vector<uint64_t> cache_page_series;
  bool compact_output = false;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program
            << " --index_path_prefix PREFIX --query_file QUERY.bin"
               " [--gt_file GT.bin] [--result_path PREFIX] [--recall_at K] [--search_list L]"
               " [--search_list_series L1,L2,...]"
               " [--beamwidth W] [--num_threads T] [--thread_series T1,T2,...]"
               " [--cache_page_series P1,P2,...]"
               " [--num_nodes_to_cache N]"
               " [--search_io_limit N] [--use_reorder_data]"
               " [--query_limit N] [--query_offset N]"
               " [--warmup_queries N] [--warmup_query_offset N]"
               " [--phase_profile true|false]"
               " [--output_style full|compact]"
               " [--cache_state warm|cold] [--node_visit_output FILE]"
               " [--cell_oracle_output FILE] [--cell_value_output FILE]"
               " [--gateway_value_output FILE]"
               " [--base_trace_output FILE]"
               " [--skipcell_sidecar_path FILE]"
               " [--io_topology_vector true|false] [--io_topology_path FILE]"
               " [--io_paged_topology true|false] [--io_topology_cache_pages N]"
               " [--io_vector_path FILE] [--io_cell_layout true|false]"
               " [--io_cell_u8_layout true|false]"
               " [--io_cell_page_batch true|false]"
               " [--io_cell_batch_search true|false]"
               " [--io_cell_batch_max_cached_expansions N]"
               " [--io_interleaved_cell_batch_search true|false]"
               " [--io_interleaved_cell_batch_graph_hops N]"
               " [--io_cell_batch_preserve_graph true|false]"
               " [--io_cell_batch_graph_stitch_min_l N]"
               " [--io_ranked_cell_page_refinement true|false]"
               " [--io_ranked_cell_page_base_pages N]"
               " [--io_ranked_cell_page_l_divisor N]"
               " [--io_ranked_cell_page_growth_start_l N]"
               " [--io_ranked_cell_page_growth_divisor N]"
               " [--io_ranked_cell_page_max_pages N]"
               " [--io_ranked_cell_page_max_l N]"
               " [--io_cell_pq_traversal true|false] [--io_cell_pq_refine_candidates N]"
               " [--io_cell_pq_refine_min_l N]"
               " [--io_cell_pq_refine_prefetch_hop N]"
               " [--io_cell_leaf_refinement true|false]"
               " [--io_resident_u8_refinement true|false]"
               " [--io_resident_u8_traversal true|false]"
               " [--io_resident_u8_refinement_path FILE]"
               " [--io_cell_pq_dense_visited true|false]"
               " [--io_cell_pq_filter_visited true|false]"
               " [--io_cell_pq_decoded_u8 true|false] [--io_cell_pq_decoded_scale F]"
               " [--io_compressed_pq_traversal true|false] [--io_pq_score_chunks N]"
               " [--io_pq_score_quantization_bits 0|8|16]"
               " [--io_graph_neighbor_score_limit N]"
               " [--io_l_aware_search true|false] [--io_l_aware_decoded_max_l N]"
               " [--io_l_aware_refine_budget true|false]"
               " [--io_l_aware_refine_base N] [--io_l_aware_refine_divisor N]"
               " [--io_forced_cell_tree_search true|false]"
               " [--io_cell_adj_correction true|false] [--io_cell_adj_candidate_cells N]"
               " [--io_cell_adj_support_shortlist N]"
               " [--io_cell_adj_expansion_cells N]"
               " [--io_cell_adj_min_search_list_size N]"
               " [--io_cell_adj_max_search_list_size N]"
               " [--io_cell_adj_path FILE]"
               " [--online_cell_adaptation true|false]"
               " [--io_chunk_layout true|false]"
               " [--io_weighted_reorder true|false]"
               " [--io_lru_cache true|false] [--io_lru_cache_pages N]"
               " [--io_lru_reset_after_warmup true|false]"
               " [--io_uring true|false] [--io_uring_queue_depth N]"
               " [--io_node_prefetch true|false] [--io_cell_prefetch true|false]"
               " [--io_gateway_prefetch true|false]"
               " [--io_gateway_prefetch_max_outstanding 1|2]"
               " [--io_memgraph none|random|important|high_degree|rcni]"
               " [--io_memgraph_nodes N]"
               " [--io_memgraph_entry_candidates N] [--io_memgraph_seed N]"
               " [--io_dynamic_width true|false] [--io_dynamic_initial N]"
               " [--io_dynamic_marker N] [--io_dynamic_waste_threshold F]"
               " [--enable_powerann true|false] [--powerann_mode baseline|hybrid]"
               " [--community_skip true|false] [--community_hub_clique true|false]"
               " [--community_pq_approach_hops N]"
               " [--cell_expansion true|false]"
               " [--community_expansion_budget N] [--gateway_score_budget N]"
               " [--gateway_score_batch_cap N]"
               " [--gateway_direct_insert_cap N] [--gateway_landing_cell_handoff true|false]"
               " [--cell_block_budget N] [--cell_support N] [--cell_insert_cap N]"
               " [--cell_activation_marker N] [--cell_min_search_list_size N]"
               " [--cell_min_competitive_count N] [--cell_min_competitive_fraction F]"
               " [--cell_evidence_threshold_ratio F]"
               " [--cell_candidate_threshold_ratio F] [--cell_insert_fraction F]"
               " [--cell_require_full_frontier true|false]"
               " [--cell_use_provisional_frontier true|false]"
               " [--cell_whole_cell_scan true|false]"
               " [--cell_centroid_routing true|false]"
               " [--cell_routing_community_budget N] [--cell_routing_cell_budget N]"
               " [--cell_terminal_convergence true|false]"
               " [--cell_terminal_approach_hops N]"
               " [--cell_terminal_l_aware_approach true|false]"
               " [--cell_terminal_approach_base_hops N]"
               " [--cell_terminal_approach_l_divisor N]"
               " [--cell_terminal_low_l_stitch_max_l N]"
               " [--cell_terminal_low_l_stitch_extra_hops N]"
               " [--cell_terminal_refine_hops N]"
               " [--cell_terminal_l_aware_refine true|false]"
               " [--cell_terminal_refine_base_hops N]"
               " [--cell_terminal_refine_l_divisor N]"
               " [--cell_terminal_distance_scale F]"
               " [--cell_terminal_frontier_cells N]"
               " [--cell_terminal_stitch_promote_cells true|false]"
               " [--cell_terminal_stitch_cell_budget N]"
               " [--cell_terminal_stitch_min_support N]"
               " [--cell_terminal_l_aware_forest true|false]"
               " [--cell_terminal_forest_base_cells N]"
               " [--cell_terminal_forest_l_divisor N]"
               " [--cell_terminal_low_l_forest_max_l N]"
               " [--cell_terminal_low_l_forest_extra_cells N]"
               " [--cell_terminal_high_l_forest_min_l N]"
               " [--cell_terminal_high_l_forest_cell_cap N]"
               " [--cell_terminal_high_l_forest_cap_l_divisor N]"
               " [--cell_terminal_global_forest_routing true|false]"
               " [--cell_terminal_route_path FILE]"
               " [--cell_hierarchical_routing true|false]"
               " [--cell_hierarchy_branching N] [--cell_hierarchy_leaf_size N]"
               " [--cell_hierarchy_probe_budget N]"
               " [--cell_hierarchy_exact_routing true|false]"
               " [--cell_hierarchy_preserve_leaf_order true|false]"
               " [--cell_hierarchy_require_persisted true|false]"
               " [--cell_hierarchy_graph_guided_routing true|false]"
               " [--cell_defer_hierarchy_routing true|false]"
               " [--graph_guard_fraction F]\n";
}

std::string read_value(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
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

powerlaw_ann::powerann_search_mode_t parse_powerann_mode(const std::string& value) {
  if (value == "baseline") {
    return powerlaw_ann::powerann_search_mode_t::BASELINE;
  }
  if (value == "hybrid") {
    return powerlaw_ann::powerann_search_mode_t::HYBRID;
  }
  throw std::invalid_argument("--powerann_mode must be 'baseline' or 'hybrid'");
}

command_config_t parse_config(int argc, char** argv) {
  command_config_t config;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--index_path_prefix") {
      config.search.index_path_prefix = read_value(argc, argv, i);
    } else if (option == "--query_file") {
      config.search.query_path = read_value(argc, argv, i);
    } else if (option == "--gt_file") {
      config.search.ground_truth_path = read_value(argc, argv, i);
    } else if (option == "--result_path") {
      config.search.result_path_prefix = read_value(argc, argv, i);
    } else if (option == "--recall_at" || option == "-K") {
      config.search.top_k = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--search_list" || option == "-L") {
      config.search.search_list_size = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--search_list_series") {
      const std::string values = read_value(argc, argv, i);
      size_t begin = 0;
      while (begin < values.size()) {
        const size_t end = values.find(',', begin);
        const std::string value = values.substr(begin, end - begin);
        if (value.empty()) {
          throw std::invalid_argument("--search_list_series contains an empty value");
        }
        config.search_list_series.push_back(static_cast<uint32_t>(std::stoul(value)));
        if (end == std::string::npos) {
          break;
        }
        begin = end + 1;
      }
    } else if (option == "--beamwidth" || option == "-W") {
      config.search.beam_width = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--num_threads" || option == "-T") {
      config.search.num_threads = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--thread_series") {
      const std::string values = read_value(argc, argv, i);
      size_t begin = 0;
      while (begin < values.size()) {
        const size_t end = values.find(',', begin);
        const std::string value = values.substr(begin, end - begin);
        if (value.empty()) {
          throw std::invalid_argument("--thread_series contains an empty value");
        }
        config.thread_series.push_back(static_cast<uint32_t>(std::stoul(value)));
        if (end == std::string::npos) {
          break;
        }
        begin = end + 1;
      }
    } else if (option == "--cache_page_series") {
      const std::string values = read_value(argc, argv, i);
      size_t begin = 0;
      while (begin < values.size()) {
        const size_t end = values.find(',', begin);
        const std::string value = values.substr(begin, end - begin);
        if (value.empty()) {
          throw std::invalid_argument("--cache_page_series contains an empty value");
        }
        config.cache_page_series.push_back(std::stoull(value));
        if (end == std::string::npos) {
          break;
        }
        begin = end + 1;
      }
    } else if (option == "--num_nodes_to_cache") {
      config.search.num_nodes_to_cache =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--search_io_limit") {
      config.search.io_limit = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--query_limit") {
      config.search.query_limit = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--query_offset") {
      config.search.query_offset = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--warmup_queries") {
      config.search.warmup_query_count =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--warmup_query_offset") {
      config.search.warmup_query_offset =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--use_reorder_data") {
      config.search.use_reorder_data = true;
    } else if (option == "--phase_profile") {
      config.search.phase_profile = parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--output_style") {
      const auto value = read_value(argc, argv, i);
      if (value != "full" && value != "compact") {
        throw std::invalid_argument("--output_style must be full or compact");
      }
      config.compact_output = value == "compact";
    } else if (option == "--cache_state") {
      config.search.cache_state = read_value(argc, argv, i);
    } else if (option == "--node_visit_output") {
      config.search.node_visit_output_path = read_value(argc, argv, i);
    } else if (option == "--cell_oracle_output") {
      config.search.cell_oracle_output_path = read_value(argc, argv, i);
    } else if (option == "--cell_value_output") {
      config.search.cell_value_output_path = read_value(argc, argv, i);
    } else if (option == "--gateway_value_output") {
      config.search.gateway_value_output_path = read_value(argc, argv, i);
    } else if (option == "--base_trace_output") {
      config.search.base_trace_output_path = read_value(argc, argv, i);
    } else if (option == "--community_polar_path" || option == "--skipcell_sidecar_path") {
      config.search.community_polar_path = read_value(argc, argv, i);
    } else if (option == "--io_topology_vector") {
      config.power_ann.io_optimization.enable_topology_vector =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_paged_topology") {
      config.power_ann.io_optimization.enable_paged_topology =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_topology_cache_pages") {
      config.power_ann.io_optimization.topology_cache_pages =
          std::stoull(read_value(argc, argv, i));
    } else if (option == "--io_topology_path") {
      config.search.io_topology_path = read_value(argc, argv, i);
    } else if (option == "--io_vector_path") {
      config.search.io_vector_path = read_value(argc, argv, i);
    } else if (option == "--io_cell_adj_path") {
      config.search.io_cell_adjacency_path = read_value(argc, argv, i);
    } else if (option == "--io_cell_layout") {
      config.power_ann.io_optimization.enable_cell_layout =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_u8_layout") {
      config.power_ann.io_optimization.enable_cell_u8_layout =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_page_batch") {
      config.power_ann.io_optimization.enable_cell_page_batch =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_batch_search") {
      config.power_ann.io_optimization.enable_cell_batch_search =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_batch_max_cached_expansions") {
      config.power_ann.io_optimization.cell_batch_max_cached_expansions =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_interleaved_cell_batch_search") {
      config.power_ann.io_optimization.enable_interleaved_cell_batch_search =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_interleaved_cell_batch_graph_hops") {
      config.power_ann.io_optimization.interleaved_cell_batch_graph_hops =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_batch_preserve_graph") {
      config.power_ann.io_optimization.cell_batch_preserve_graph =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_batch_graph_stitch_min_l") {
      config.power_ann.io_optimization.cell_batch_graph_stitch_min_l =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_ranked_cell_page_refinement") {
      config.power_ann.io_optimization.enable_ranked_cell_page_refinement =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_ranked_cell_page_base_pages") {
      config.power_ann.io_optimization.ranked_cell_page_base_pages =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_ranked_cell_page_l_divisor") {
      config.power_ann.io_optimization.ranked_cell_page_l_divisor =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_ranked_cell_page_growth_start_l") {
      config.power_ann.io_optimization.ranked_cell_page_growth_start_l =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_ranked_cell_page_growth_divisor") {
      config.power_ann.io_optimization.ranked_cell_page_growth_divisor =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_ranked_cell_page_max_pages") {
      config.power_ann.io_optimization.ranked_cell_page_max_pages =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_ranked_cell_page_max_l") {
      config.power_ann.io_optimization.ranked_cell_page_max_l =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_pq_traversal") {
      config.power_ann.io_optimization.enable_cell_pq_traversal =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_pq_refine_candidates") {
      config.power_ann.io_optimization.cell_pq_refine_candidates =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_pq_refine_min_l") {
      config.power_ann.io_optimization.cell_pq_refine_min_l =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_pq_refine_prefetch_hop") {
      config.power_ann.io_optimization.cell_pq_refine_prefetch_hop =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_leaf_refinement") {
      config.power_ann.io_optimization.enable_cell_leaf_refinement =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_resident_u8_refinement") {
      config.power_ann.io_optimization.enable_resident_u8_refinement =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_resident_u8_traversal") {
      config.power_ann.io_optimization.enable_resident_u8_traversal =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_resident_u8_refinement_path") {
      config.search.resident_u8_refinement_path = read_value(argc, argv, i);
    } else if (option == "--io_cell_pq_dense_visited") {
      config.power_ann.io_optimization.enable_cell_pq_dense_visited =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_pq_filter_visited") {
      config.power_ann.io_optimization.enable_cell_pq_filter_visited =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_pq_decoded_u8") {
      config.power_ann.io_optimization.enable_cell_pq_decoded_u8 =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_pq_decoded_scale") {
      config.power_ann.io_optimization.cell_pq_decoded_scale = std::stof(read_value(argc, argv, i));
    } else if (option == "--io_compressed_pq_traversal") {
      config.power_ann.io_optimization.enable_compressed_pq_traversal =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_pq_score_chunks") {
      config.power_ann.io_optimization.pq_score_chunks =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_pq_score_quantization_bits") {
      config.power_ann.io_optimization.pq_score_quantization_bits =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_graph_neighbor_score_limit") {
      config.power_ann.io_optimization.graph_neighbor_score_limit =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_l_aware_search") {
      config.power_ann.io_optimization.enable_l_aware_search =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_l_aware_decoded_max_l") {
      config.power_ann.io_optimization.l_aware_decoded_max_l =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_l_aware_refine_budget") {
      config.power_ann.io_optimization.enable_l_aware_refine_budget =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_l_aware_refine_base") {
      config.power_ann.io_optimization.l_aware_refine_base =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_l_aware_refine_divisor") {
      config.power_ann.io_optimization.l_aware_refine_divisor =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_forced_cell_tree_search") {
      config.power_ann.io_optimization.enable_forced_cell_tree_search =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_adj_correction") {
      config.power_ann.io_optimization.enable_cell_adj_correction =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_adj_candidate_cells") {
      config.power_ann.io_optimization.cell_adj_candidate_cells =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_adj_support_shortlist") {
      config.power_ann.io_optimization.cell_adj_support_shortlist =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_adj_expansion_cells") {
      config.power_ann.io_optimization.cell_adj_expansion_cells =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_adj_min_search_list_size") {
      config.power_ann.io_optimization.cell_adj_min_search_list_size =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_cell_adj_max_search_list_size") {
      config.power_ann.io_optimization.cell_adj_max_search_list_size =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--online_cell_adaptation") {
      config.power_ann.io_optimization.enable_online_cell_adaptation =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_chunk_layout") {
      config.power_ann.io_optimization.enable_chunk_layout =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_weighted_reorder") {
      config.power_ann.io_optimization.enable_weighted_reorder =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_lru_cache") {
      config.power_ann.io_optimization.enable_lru_cache =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_lru_cache_pages") {
      config.power_ann.io_optimization.lru_cache_pages = std::stoull(read_value(argc, argv, i));
    } else if (option == "--io_lru_reset_after_warmup") {
      config.power_ann.io_optimization.reset_lru_after_warmup =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_uring") {
      config.power_ann.io_optimization.enable_io_uring =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_uring_queue_depth") {
      config.power_ann.io_optimization.io_uring_queue_depth =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_node_prefetch") {
      config.power_ann.io_optimization.enable_node_prefetch =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_cell_prefetch") {
      config.power_ann.io_optimization.enable_cell_prefetch =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_gateway_prefetch") {
      config.power_ann.io_optimization.enable_gateway_prefetch =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_gateway_prefetch_max_outstanding") {
      config.power_ann.io_optimization.gateway_prefetch_max_outstanding =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_memgraph") {
      const auto value = read_value(argc, argv, i);
      if (value == "none") {
        config.power_ann.io_optimization.memgraph_mode = powerlaw_ann::io_memgraph_mode_t::NONE;
      } else if (value == "random") {
        config.power_ann.io_optimization.memgraph_mode = powerlaw_ann::io_memgraph_mode_t::RANDOM;
      } else if (value == "important") {
        config.power_ann.io_optimization.memgraph_mode =
            powerlaw_ann::io_memgraph_mode_t::COMMUNITY_IMPORTANT;
      } else if (value == "high_degree") {
        config.power_ann.io_optimization.memgraph_mode =
            powerlaw_ann::io_memgraph_mode_t::HIGH_DEGREE;
      } else if (value == "rcni") {
        config.power_ann.io_optimization.memgraph_mode =
            powerlaw_ann::io_memgraph_mode_t::RCNI_ONLY;
      } else {
        throw std::invalid_argument(
            "--io_memgraph must be none, random, important, high_degree, or rcni");
      }
    } else if (option == "--io_memgraph_nodes") {
      config.power_ann.io_optimization.memgraph_nodes =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_memgraph_entry_candidates") {
      config.power_ann.io_optimization.memgraph_entry_candidates =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_memgraph_seed") {
      config.power_ann.io_optimization.memgraph_seed = std::stoull(read_value(argc, argv, i));
    } else if (option == "--io_dynamic_width") {
      config.power_ann.io_optimization.enable_dynamic_width =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--io_dynamic_initial") {
      config.power_ann.io_optimization.dynamic_width_initial =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_dynamic_marker") {
      config.power_ann.io_optimization.dynamic_width_marker =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--io_dynamic_waste_threshold") {
      config.power_ann.io_optimization.dynamic_width_waste_threshold =
          std::stof(read_value(argc, argv, i));
    } else if (option == "--enable_powerann") {
      config.power_ann.enable_powerann = parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--powerann_mode") {
      config.power_ann.community_polar_search.mode = parse_powerann_mode(read_value(argc, argv, i));
    } else if (option == "--community_skip") {
      config.power_ann.community_polar_search.community_skip =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--community_hub_clique") {
      config.power_ann.community_polar_search.community_hub_clique =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--community_pq_approach_hops") {
      config.power_ann.community_polar_search.community_pq_approach_hops =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_expansion") {
      config.power_ann.community_polar_search.cell_expansion =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--community_expansion_budget") {
      config.power_ann.community_polar_search.community_expansion_budget =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--gateway_score_budget") {
      config.power_ann.community_polar_search.gateway_score_budget =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--gateway_score_batch_cap") {
      config.power_ann.community_polar_search.gateway_score_batch_cap =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--gateway_direct_insert_cap") {
      config.power_ann.community_polar_search.gateway_direct_insert_cap =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--gateway_landing_cell_handoff") {
      config.power_ann.community_polar_search.gateway_landing_cell_handoff =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_block_budget") {
      config.power_ann.community_polar_search.cell_block_budget =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_support") {
      config.power_ann.community_polar_search.cell_support =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_insert_cap") {
      config.power_ann.community_polar_search.cell_insert_cap =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_activation_marker") {
      config.power_ann.community_polar_search.cell_activation_marker =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_min_search_list_size") {
      config.power_ann.community_polar_search.cell_min_search_list_size =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_min_competitive_count") {
      config.power_ann.community_polar_search.cell_min_competitive_count =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_min_competitive_fraction") {
      config.power_ann.community_polar_search.cell_min_competitive_fraction =
          std::stof(read_value(argc, argv, i));
    } else if (option == "--cell_evidence_threshold_ratio") {
      config.power_ann.community_polar_search.cell_evidence_threshold_ratio =
          std::stof(read_value(argc, argv, i));
    } else if (option == "--cell_candidate_threshold_ratio") {
      config.power_ann.community_polar_search.cell_candidate_threshold_ratio =
          std::stof(read_value(argc, argv, i));
    } else if (option == "--cell_insert_fraction") {
      config.power_ann.community_polar_search.cell_insert_fraction =
          std::stof(read_value(argc, argv, i));
    } else if (option == "--cell_require_full_frontier") {
      config.power_ann.community_polar_search.cell_require_full_frontier =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_use_provisional_frontier") {
      config.power_ann.community_polar_search.cell_use_provisional_frontier =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_whole_cell_scan") {
      config.power_ann.community_polar_search.cell_whole_cell_scan =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_centroid_routing") {
      config.power_ann.community_polar_search.cell_centroid_routing =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_routing_community_budget") {
      config.power_ann.community_polar_search.cell_routing_community_budget =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_routing_cell_budget") {
      config.power_ann.community_polar_search.cell_routing_cell_budget =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_convergence") {
      config.power_ann.community_polar_search.cell_terminal_convergence =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_terminal_approach_hops") {
      config.power_ann.community_polar_search.cell_terminal_approach_hops =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_l_aware_approach") {
      config.power_ann.community_polar_search.cell_terminal_l_aware_approach =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_terminal_approach_base_hops") {
      config.power_ann.community_polar_search.cell_terminal_approach_base_hops =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_approach_l_divisor") {
      config.power_ann.community_polar_search.cell_terminal_approach_l_divisor =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_low_l_stitch_max_l") {
      config.power_ann.community_polar_search.cell_terminal_low_l_stitch_max_l =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_low_l_stitch_extra_hops") {
      config.power_ann.community_polar_search.cell_terminal_low_l_stitch_extra_hops =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_refine_hops") {
      config.power_ann.community_polar_search.cell_terminal_refine_hops =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_l_aware_refine") {
      config.power_ann.community_polar_search.cell_terminal_l_aware_refine =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_terminal_refine_base_hops") {
      config.power_ann.community_polar_search.cell_terminal_refine_base_hops =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_refine_l_divisor") {
      config.power_ann.community_polar_search.cell_terminal_refine_l_divisor =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_distance_scale") {
      config.power_ann.community_polar_search.cell_terminal_distance_scale =
          std::stof(read_value(argc, argv, i));
    } else if (option == "--cell_terminal_frontier_cells") {
      config.power_ann.community_polar_search.cell_terminal_frontier_cells =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_stitch_promote_cells") {
      config.power_ann.community_polar_search.cell_terminal_stitch_promote_cells =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_terminal_stitch_cell_budget") {
      config.power_ann.community_polar_search.cell_terminal_stitch_cell_budget =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_stitch_min_support") {
      config.power_ann.community_polar_search.cell_terminal_stitch_min_support =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_l_aware_forest") {
      config.power_ann.community_polar_search.cell_terminal_l_aware_forest =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_terminal_forest_base_cells") {
      config.power_ann.community_polar_search.cell_terminal_forest_base_cells =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_forest_l_divisor") {
      config.power_ann.community_polar_search.cell_terminal_forest_l_divisor =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_low_l_forest_max_l") {
      config.power_ann.community_polar_search.cell_terminal_low_l_forest_max_l =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_low_l_forest_extra_cells") {
      config.power_ann.community_polar_search.cell_terminal_low_l_forest_extra_cells =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_high_l_forest_min_l") {
      config.power_ann.community_polar_search.cell_terminal_high_l_forest_min_l =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_high_l_forest_cell_cap") {
      config.power_ann.community_polar_search.cell_terminal_high_l_forest_cell_cap =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_high_l_forest_cap_l_divisor") {
      config.power_ann.community_polar_search.cell_terminal_high_l_forest_cap_l_divisor =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_terminal_global_forest_routing") {
      config.power_ann.community_polar_search.cell_terminal_global_forest_routing =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_terminal_route_path") {
      config.power_ann.community_polar_search.cell_terminal_route_path = read_value(argc, argv, i);
    } else if (option == "--cell_hierarchical_routing") {
      config.power_ann.community_polar_search.cell_hierarchical_routing =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_hierarchy_branching") {
      config.power_ann.community_polar_search.cell_hierarchy_branching =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_hierarchy_leaf_size") {
      config.power_ann.community_polar_search.cell_hierarchy_leaf_size =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_hierarchy_probe_budget") {
      config.power_ann.community_polar_search.cell_hierarchy_probe_budget =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--cell_hierarchy_exact_routing") {
      config.power_ann.community_polar_search.cell_hierarchy_exact_routing =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_hierarchy_preserve_leaf_order") {
      config.power_ann.community_polar_search.cell_hierarchy_preserve_leaf_order =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_hierarchy_require_persisted") {
      config.power_ann.community_polar_search.cell_hierarchy_require_persisted =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_hierarchy_graph_guided_routing") {
      config.power_ann.community_polar_search.cell_hierarchy_graph_guided_routing =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--cell_defer_hierarchy_routing") {
      config.power_ann.community_polar_search.cell_defer_hierarchy_routing =
          parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--graph_guard_fraction") {
      config.power_ann.community_polar_search.graph_guard_fraction =
          std::stof(read_value(argc, argv, i));
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  return config;
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    return argc == 1 ? 1 : 0;
  }

  try {
    const auto config = parse_config(argc, argv);
    const uint32_t series_count = static_cast<uint32_t>(!config.search_list_series.empty()) +
                                  static_cast<uint32_t>(!config.thread_series.empty()) +
                                  static_cast<uint32_t>(!config.cache_page_series.empty());
    if (series_count > 1) {
      throw std::invalid_argument("search-list, thread, and cache series are mutually exclusive");
    }
    powerlaw_ann::power_ann_t power_ann(config.power_ann);
    const auto results =
        !config.thread_series.empty()
            ? power_ann.search_diskann_index_thread_curve(config.search, config.thread_series)
            : !config.cache_page_series.empty()
                  ? power_ann.search_diskann_index_cache_curve(config.search,
                                                               config.cache_page_series)
                  : config.search_list_series.empty()
                        ? std::vector{power_ann.search_diskann_index(config.search)}
                        : power_ann.search_diskann_index_curve(config.search,
                                                               config.search_list_series);
    if (config.compact_output) {
      std::cout << "[Search] System       L    Beam  Threads  Recall(%)    QPS        P99(us)   "
                   "4K-reads/q\n";
    }
    for (size_t point = 0; point < results.size(); ++point) {
      const auto& result = results[point];
      const uint32_t search_list_size = config.search_list_series.empty()
                                            ? config.search.search_list_size
                                            : config.search_list_series[point];
      if (config.compact_output) {
        const char* system = config.power_ann.enable_powerann ? "SkipCell" : "DiskANN";
        std::cout << std::fixed << std::setprecision(3) << "[Search] " << std::left << std::setw(12)
                  << system << std::right << std::setw(5) << search_list_size << std::setw(6)
                  << config.search.beam_width << std::setw(9) << result.num_threads
                  << std::setw(13);
        if (result.recall_percent.has_value()) {
          std::cout << *result.recall_percent;
        } else {
          std::cout << "n/a";
        }
        std::cout << std::setw(11) << result.qps() << std::setw(11) << result.p99_latency_us()
                  << std::setw(13) << result.four_kib_reads_per_query() << '\n';
        continue;
      }
      std::cout << std::fixed << std::setprecision(4) << "queries=" << result.num_queries
                << " top_k=" << result.top_k << " search_l=" << search_list_size
                << " beam_width=" << config.search.beam_width
                << " num_threads=" << result.num_threads
                << " warmup_queries=" << config.search.warmup_query_count
                << " query_offset=" << config.search.query_offset
                << " warmup_query_offset=" << config.search.warmup_query_offset
                << " num_nodes_to_cache=" << config.search.num_nodes_to_cache
                << " io_limit=" << config.search.io_limit
                << " reorder=" << (config.search.use_reorder_data ? "true" : "false")
                << " qps=" << result.qps() << " mean_latency_us=" << result.mean_latency_us()
                << " p50_latency_us=" << result.p50_latency_us()
                << " p95_latency_us=" << result.p95_latency_us()
                << " p99_latency_us=" << result.p99_latency_us()
                << " reads_per_query=" << result.reads_per_query()
                << " io_time_us_per_query=" << result.io_time_us_per_query()
                << " cpu_time_us_per_query=" << result.cpu_time_us_per_query()
                << " read_bytes_per_query=" << result.read_bytes_per_query()
                << " four_kib_reads_per_query=" << result.four_kib_reads_per_query()
                << " cache_hits_per_query=" << result.cache_hits_per_query()
                << " distance_computations_per_query=" << result.distance_computations_per_query()
                << " hops_per_query=" << result.hops_per_query()
                << " base_neighbors_per_query=" << result.base_neighbors_per_query();
      std::cout
          << " base_hops_per_query=" << result.base_hops_per_query()
          << " sequential_ios_per_query=" << result.sequential_ios_per_query()
          << " random_ios_per_query=" << result.random_ios_per_query()
          << " physical_read_requests_per_query=" << result.physical_read_requests_per_query()
          << " cell_page_batch_requests_per_query=" << result.cell_page_batch_requests_per_query()
          << " cell_page_batch_pages_per_query=" << result.cell_page_batch_pages_per_query()
          << " demand_useful_bytes_per_query=" << result.demand_useful_bytes_per_query()
          << " demand_overread_bytes_per_query=" << result.demand_overread_bytes_per_query()
          << " community_meta_expansions_per_query=" << result.community_meta_expansions_per_query()
          << " community_superedges_per_query=" << result.community_superedges_per_query()
          << " gateway_nodes_scored_per_query=" << result.gateway_nodes_scored_per_query()
          << " gateway_nodes_competitive_per_query=" << result.gateway_nodes_competitive_per_query()
          << " gateway_nodes_inserted_per_query=" << result.gateway_nodes_inserted_per_query()
          << " gateway_nodes_later_expanded_per_query="
          << result.gateway_nodes_later_expanded_per_query()
          << " gateway_prefetch_predictions_per_query="
          << result.gateway_prefetch_predictions_per_query()
          << " gateway_prefetch_pages_submitted_per_query="
          << result.gateway_prefetch_pages_submitted_per_query()
          << " gateway_prefetch_queue_rejections_per_query="
          << result.gateway_prefetch_queue_rejections_per_query()
          << " gateway_prefetch_timely_hits_per_query="
          << result.gateway_prefetch_timely_hits_per_query()
          << " gateway_prefetch_late_hits_per_query="
          << result.gateway_prefetch_late_hits_per_query()
          << " gateway_prefetch_unused_per_query=" << result.gateway_prefetch_unused_per_query()
          << " gateway_prefetch_mean_lead_us=" << result.gateway_prefetch_mean_lead_us()
          << " gateway_prefetch_mean_lead_hops=" << result.gateway_prefetch_mean_lead_hops()
          << " cell_candidates_per_query=" << result.cell_candidates_per_query()
          << " cell_blocks_per_query=" << result.cell_blocks_per_query()
          << " cell_hint_rejections_per_query=" << result.cell_hint_rejections_per_query()
          << " cell_nodes_scored_per_query=" << result.cell_nodes_scored_per_query()
          << " cell_nodes_competitive_per_query=" << result.cell_nodes_competitive_per_query()
          << " cell_nodes_inserted_per_query=" << result.cell_nodes_inserted_per_query()
          << " cell_nodes_later_expanded_per_query=" << result.cell_nodes_later_expanded_per_query()
          << " cell_nodes_already_visited_per_query="
          << result.cell_nodes_already_visited_per_query()
          << " cell_blocks_without_unseen_nodes_per_query="
          << result.cell_blocks_without_unseen_nodes_per_query()
          << " cell_blocks_low_yield_per_query=" << result.cell_blocks_low_yield_per_query()
          << " cell_packet_bytes_per_query=" << result.cell_packet_bytes_per_query()
          << " cell_batch_search_rounds_per_query=" << result.cell_batch_search_rounds_per_query()
          << " cell_batch_cached_expansions_per_query="
          << result.cell_batch_cached_expansions_per_query()
          << " cell_pq_traversal_expansions_per_query="
          << result.cell_pq_traversal_expansions_per_query()
          << " cell_pq_refined_candidates_per_query="
          << result.cell_pq_refined_candidates_per_query()
          << " cell_hierarchy_nodes_scored_per_query="
          << result.cell_hierarchy_nodes_scored_per_query()
          << " cell_hierarchy_leaf_groups_probed_per_query="
          << result.cell_hierarchy_leaf_groups_probed_per_query()
          << " cell_adj_neighbor_edges_considered_per_query="
          << result.cell_adj_neighbor_edges_considered_per_query()
          << " cell_adj_cells_scored_per_query=" << result.cell_adj_cells_scored_per_query()
          << " cell_stitch_candidates_considered_per_query="
          << result.cell_stitch_candidates_considered_per_query()
          << " cell_stitch_owner_cells_per_query=" << result.cell_stitch_owner_cells_per_query()
          << " cell_stitch_max_owner_support_per_query="
          << result.cell_stitch_max_owner_support_per_query()
          << " cell_stitch_multi_owner_candidates_per_query="
          << result.cell_stitch_multi_owner_candidates_per_query()
          << " fallback_queries=" << result.fallback_queries()
          << " phase_transition_rate_percent=" << result.beam_phase_transition_rate_percent()
          << " phase_approach_hops_per_query=" << result.beam_phase_approach_hops_per_query()
          << " phase_convergence_hops_per_query=" << result.beam_phase_convergence_hops_per_query()
          << " phase_approach_ios_per_query=" << result.beam_phase_approach_ios_per_query()
          << " phase_convergence_ios_per_query=" << result.beam_phase_convergence_ios_per_query()
          << " phase_approach_time_us=" << result.beam_phase_approach_time_us()
          << " phase_convergence_time_us=" << result.beam_phase_convergence_time_us()
          << " phase_setup_time_us=" << result.beam_phase_setup_time_us()
          << " phase_post_search_time_us=" << result.beam_phase_post_search_time_us()
          << " phase_approach_community_expansions_per_query="
          << result.beam_phase_approach_community_expansions_per_query()
          << " phase_convergence_community_expansions_per_query="
          << result.beam_phase_convergence_community_expansions_per_query()
          << " phase_approach_gateway_nodes_scored_per_query="
          << result.beam_phase_approach_gateway_nodes_scored_per_query()
          << " phase_convergence_gateway_nodes_scored_per_query="
          << result.beam_phase_convergence_gateway_nodes_scored_per_query()
          << " phase_approach_cell_blocks_per_query="
          << result.beam_phase_approach_cell_blocks_per_query()
          << " phase_convergence_cell_blocks_per_query="
          << result.beam_phase_convergence_cell_blocks_per_query()
          << " phase_approach_cell_nodes_scored_per_query="
          << result.beam_phase_approach_cell_nodes_scored_per_query()
          << " phase_convergence_cell_nodes_scored_per_query="
          << result.beam_phase_convergence_cell_nodes_scored_per_query()
          << " resident_sidecar_bytes=" << result.resident_sidecar_bytes
          << " sidecar_artifact_bytes=" << result.sidecar_artifact_bytes
          << " cell_adjacency_resident_bytes=" << result.cell_adjacency_resident_bytes
          << " cell_adjacency_artifact_bytes=" << result.cell_adjacency_artifact_bytes
          << " topology_resident_bytes=" << result.topology_resident_bytes
          << " topology_artifact_bytes=" << result.topology_artifact_bytes
          << " topology_cache_resident_bytes=" << result.topology_cache_resident_bytes
          << " topology_cache_hits=" << result.topology_cache_hits
          << " topology_cache_misses=" << result.topology_cache_misses
          << " topology_cache_evictions=" << result.topology_cache_evictions
          << " vector_artifact_bytes=" << result.vector_artifact_bytes
          << " io_cache_resident_bytes=" << result.io_cache_resident_bytes
          << " io_cache_demand_hits=" << result.io_cache_demand_hits
          << " io_cache_prefetch_hits=" << result.io_cache_prefetch_hits
          << " io_cache_prefetch_misses=" << result.io_cache_prefetch_misses
          << " io_cache_prefetch_used=" << result.io_cache_prefetch_used
          << " io_cache_prefetch_pollution_evictions="
          << result.io_cache_prefetch_pollution_evictions
          << " io_cache_prefetch_admission_rejections="
          << result.io_cache_prefetch_admission_rejections
          << " io_cache_prefetch_demand_evictions=" << result.io_cache_prefetch_demand_evictions
          << " io_cache_misses=" << result.io_cache_misses
          << " io_cache_coalesced_misses=" << result.io_cache_coalesced_misses
          << " io_cache_evictions=" << result.io_cache_evictions
          << " io_cache_pinned_victim_failures=" << result.io_cache_pinned_victim_failures
          << " io_uring_submitted=" << result.io_uring_submitted
          << " io_uring_completed=" << result.io_uring_completed
          << " io_uring_completion_batches=" << result.io_uring_completion_batches
          << " io_uring_wait_ns=" << result.io_uring_wait_ns
          << " io_uring_short_or_error_completions=" << result.io_uring_short_or_error_completions
          << " io_uring_prefetch_submitted=" << result.io_uring_prefetch_submitted
          << " io_uring_prefetch_completed=" << result.io_uring_prefetch_completed
          << " io_uring_prefetch_obsolete=" << result.io_uring_prefetch_obsolete
          << " io_uring_queue_depth=" << result.io_uring_queue_depth
          << " memgraph_resident_bytes=" << result.memgraph_resident_bytes
          << " memgraph_nodes=" << result.memgraph_nodes
          << " memgraph_build_time_us=" << result.memgraph_build_time_us
          << " memgraph_build_peak_bytes=" << result.memgraph_build_peak_bytes
          << " decoded_pq_build_time_us=" << result.decoded_pq_build_time_us
          << " decoded_pq_representation_bytes=" << result.decoded_pq_representation_bytes
          << " decoded_pq_additional_resident_bytes=" << result.decoded_pq_additional_resident_bytes
          << " resident_u8_refinement_artifact_bytes="
          << result.resident_u8_refinement_artifact_bytes
          << " resident_u8_refinement_bytes=" << result.resident_u8_refinement_bytes
          << " resident_u8_refinement_checksum=" << result.resident_u8_refinement_checksum
          << " memgraph_nodes_scored_per_query=" << result.memgraph_nodes_scored_per_query()
          << " memgraph_nodes_inserted_per_query=" << result.memgraph_nodes_inserted_per_query()
          << " memgraph_nodes_later_expanded_per_query="
          << result.memgraph_nodes_later_expanded_per_query()
          << " dynamic_width_mean=" << result.dynamic_width_mean()
          << " dynamic_width_max=" << result.dynamic_width_max()
          << " dynamic_width_useful_ios_per_query=" << result.dynamic_width_useful_ios_per_query()
          << " dynamic_width_wasted_ios_per_query=" << result.dynamic_width_wasted_ios_per_query()
          << " online_cell_adaptation_observations=" << result.online_cell_adaptation_observations
          << " online_cell_adaptation_publish_attempts="
          << result.online_cell_adaptation_publish_attempts
          << " online_cell_adaptation_successful_publishes="
          << result.online_cell_adaptation_successful_publishes
          << " online_cell_adaptation_publish_conflicts="
          << result.online_cell_adaptation_publish_conflicts
          << " online_cell_adaptation_retired_snapshots="
          << result.online_cell_adaptation_retired_snapshots
          << " online_cell_adaptation_physical_generations="
          << result.online_cell_adaptation_physical_generations
          << " online_cell_adaptation_physical_bytes_written="
          << result.online_cell_adaptation_physical_bytes_written
          << " online_cell_adaptation_physical_write_time_us="
          << result.online_cell_adaptation_physical_write_time_us
          << " online_cell_adaptation_checksum_failures="
          << result.online_cell_adaptation_checksum_failures
          << " online_cell_adaptation_directory_resident_bytes="
          << result.online_cell_adaptation_directory_resident_bytes
          << " online_cell_adaptation_overlay_artifact_bytes="
          << result.online_cell_adaptation_overlay_artifact_bytes;
      if (result.recall_percent.has_value()) {
        std::cout << " recall_percent=" << *result.recall_percent;
      } else {
        std::cout << " recall_percent=unavailable";
      }
      std::cout << " index_size_bytes=" << result.index_size_bytes << '\n'
                << "cache_state=" << result.cache_state << " io_backend=" << result.backends.io
                << " math_backend=" << result.backends.math
                << " allocator_backend=" << result.backends.allocator
                << " simd_backend=" << result.backends.simd << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "search_diskann_index: " << error.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }
}
