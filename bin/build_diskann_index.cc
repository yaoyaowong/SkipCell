#include "power_ann.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct command_config_t {
  powerlaw_ann::power_ann_config_t power_ann;
  powerlaw_ann::diskann_disk_index_config_t index;
  bool community_sidecar_only = false;
  std::filesystem::path rcni_path;
  std::filesystem::path community_sidecar_path;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program
            << " --data_path DATA.bin --index_path_prefix PREFIX"
               " --search_DRAM_budget GB --build_DRAM_budget GB"
               " [--max_degree R] [--Lbuild L] [--num_threads T]"
               " [--PQ_disk_bytes N] [--append_reorder_data]"
               " [--build_PQ_bytes N] [--QD N] [--enable_powerann true|false]"
               " [--hub_selection top_ratio|importance_mass]"
               " [--hub_top_ratio VALUE | --hub_importance_mass VALUE]"
               " [--community_count N] [--community_imbalance VALUE]"
               " [--partition_quality fast|quality] [--partition_threads T]"
               " [--community_projection full_reciprocal_vamana|local_vamana_k8|"
               "local_vamana_k16]"
               " [--partition_seed N] [--gateway_count N] [--gateway_shortlist N]"
               " [--gateway_selection_path FILE] [--gateway_selection_source_path FILE]"
               " [--direction_count 16|32|64] [--cell_target_size N]"
               " [--cell_partition "
               "polar_radial|graph_local|convergence_coaccess|global_geometric|global_graph_local]"
               " [--build_contiguous_cell_hierarchy true|false]"
               " [--cell_hierarchy_branching N] [--cell_hierarchy_leaf_size N]"
               " [--cell_profile_path FILE]"
               " [--cell_hierarchy_path FILE]"
               " [--adaptive_multi_capacity true|false]"
               " [--convergence_cell_capacity_path FILE]"
               " [--adaptive_policy_version N] [--adaptive_page_bytes N]"
               " [--adaptive_vector_bytes N] [--adaptive_lru_pages N]"
               " [--adaptive_maximum_normalized_rms_radius VALUE]"
               " [--adaptive_maximum_normalized_p95_radius VALUE]"
               " [--adaptive_maximum_split_gain VALUE]"
               " [--adaptive_minimum_centroid_radius_overlap VALUE]"
               " [--adaptive_minimum_train_cross_child_coaccess VALUE]"
               " [--adaptive_maximum_unseen_pq_increase VALUE]"
               " [--convergence_cell_source_path FILE]"
               " [--block_target_size N] [--community_sidecar_only true|false]"
               " [--rcni_path FILE] [--skipcell_sidecar_path FILE]\n";
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

powerlaw_ann::candidate_hub_selection_mode_t parse_hub_selection_mode(const std::string& value,
                                                                      const std::string& option) {
  if (value == "top_ratio") {
    return powerlaw_ann::candidate_hub_selection_mode_t::TOP_RATIO;
  }
  if (value == "importance_mass") {
    return powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  }
  throw std::invalid_argument(option + " must be 'top_ratio' or 'importance_mass'");
}

powerlaw_ann::community_partition_quality_t parse_partition_quality(const std::string& value,
                                                                    const std::string& option) {
  if (value == "fast") {
    return powerlaw_ann::community_partition_quality_t::FAST;
  }
  if (value == "quality") {
    return powerlaw_ann::community_partition_quality_t::QUALITY;
  }
  throw std::invalid_argument(option + " must be 'fast' or 'quality'");
}

powerlaw_ann::community_partition_projection_t
parse_partition_projection(const std::string& value, const std::string& option) {
  if (value == "full_reciprocal_vamana") {
    return powerlaw_ann::community_partition_projection_t::FULL_RECIPROCAL_VAMANA;
  }
  if (value == "local_vamana_k8") {
    return powerlaw_ann::community_partition_projection_t::LOCAL_VAMANA_K8;
  }
  if (value == "local_vamana_k16") {
    return powerlaw_ann::community_partition_projection_t::LOCAL_VAMANA_K16;
  }
  throw std::invalid_argument(option + " has an unknown Community projection");
}

powerlaw_ann::community_cell_partition_t parse_cell_partition(const std::string& value,
                                                              const std::string& option) {
  if (value == "polar_radial") {
    return powerlaw_ann::community_cell_partition_t::POLAR_RADIAL;
  }
  if (value == "graph_local") {
    return powerlaw_ann::community_cell_partition_t::GRAPH_LOCAL;
  }
  if (value == "convergence_coaccess") {
    return powerlaw_ann::community_cell_partition_t::CONVERGENCE_COACCESS;
  }
  if (value == "global_geometric") {
    return powerlaw_ann::community_cell_partition_t::GLOBAL_GEOMETRIC;
  }
  if (value == "global_graph_local") {
    return powerlaw_ann::community_cell_partition_t::GLOBAL_GRAPH_LOCAL;
  }
  throw std::invalid_argument(option + " has an unknown Cell partition");
}

command_config_t parse_config(int argc, char** argv) {
  command_config_t config;
  bool selection_mode_explicit = false;
  bool top_ratio_explicit = false;
  bool importance_mass_explicit = false;
  bool community_option_explicit = false;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--data_path") {
      config.index.data_path = read_value(argc, argv, i);
    } else if (option == "--index_path_prefix") {
      config.index.index_path_prefix = read_value(argc, argv, i);
    } else if (option == "--search_DRAM_budget" || option == "-B") {
      config.index.search_dram_budget_gb = std::stod(read_value(argc, argv, i));
    } else if (option == "--build_DRAM_budget" || option == "-M") {
      config.index.build_dram_budget_gb = std::stod(read_value(argc, argv, i));
    } else if (option == "--max_degree" || option == "-R") {
      config.index.max_degree = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--Lbuild" || option == "-L") {
      config.index.build_list_size = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--num_threads" || option == "-T") {
      config.index.num_threads = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--PQ_disk_bytes") {
      config.index.disk_pq_bytes = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--append_reorder_data") {
      config.index.append_reorder_data = true;
    } else if (option == "--build_PQ_bytes") {
      config.index.build_pq_bytes = static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--QD") {
      config.index.quantized_dimension =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--enable_powerann") {
      config.power_ann.enable_powerann = parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--hub_selection") {
      config.power_ann.hub_selection.mode =
          parse_hub_selection_mode(read_value(argc, argv, i), option);
      selection_mode_explicit = true;
    } else if (option == "--hub_top_ratio") {
      config.power_ann.hub_selection.top_ratio = std::stod(read_value(argc, argv, i));
      top_ratio_explicit = true;
    } else if (option == "--hub_importance_mass") {
      config.power_ann.hub_selection.importance_mass = std::stod(read_value(argc, argv, i));
      importance_mass_explicit = true;
    } else if (option == "--community_count") {
      config.power_ann.community_polar_build.community_count =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
    } else if (option == "--community_imbalance") {
      config.power_ann.community_polar_build.community_imbalance =
          std::stod(read_value(argc, argv, i));
      community_option_explicit = true;
    } else if (option == "--partition_quality") {
      config.power_ann.community_polar_build.partition_quality =
          parse_partition_quality(read_value(argc, argv, i), option);
      community_option_explicit = true;
    } else if (option == "--community_projection") {
      config.power_ann.community_polar_build.partition_projection =
          parse_partition_projection(read_value(argc, argv, i), option);
      community_option_explicit = true;
    } else if (option == "--partition_threads") {
      config.power_ann.community_polar_build.partition_threads =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--partition_seed") {
      config.power_ann.community_polar_build.partition_seed =
          static_cast<int32_t>(std::stol(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--gateway_count") {
      config.power_ann.community_polar_build.gateway_count =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--gateway_shortlist") {
      config.power_ann.community_polar_build.gateway_shortlist =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--gateway_selection_path") {
      config.power_ann.community_polar_build.gateway_selection_path = read_value(argc, argv, i);
      community_option_explicit = true;
    } else if (option == "--gateway_selection_source_path") {
      config.power_ann.community_polar_build.gateway_selection_source_path =
          read_value(argc, argv, i);
      community_option_explicit = true;
    } else if (option == "--polar_direction_count" || option == "--direction_count") {
      config.power_ann.community_polar_build.polar_direction_count =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--cell_target_size") {
      config.power_ann.community_polar_build.cell_target_size =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--cell_partition") {
      config.power_ann.community_polar_build.cell_partition =
          parse_cell_partition(read_value(argc, argv, i), option);
      community_option_explicit = true;
    } else if (option == "--build_contiguous_cell_hierarchy") {
      config.power_ann.community_polar_build.build_contiguous_cell_hierarchy =
          parse_bool(read_value(argc, argv, i), option);
      community_option_explicit = true;
    } else if (option == "--cell_hierarchy_branching") {
      config.power_ann.community_polar_build.cell_hierarchy_branching =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--cell_hierarchy_leaf_size") {
      config.power_ann.community_polar_build.cell_hierarchy_leaf_size =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--convergence_cell_profile_path" || option == "--cell_profile_path") {
      config.power_ann.community_polar_build.convergence_cell_profile_path =
          read_value(argc, argv, i);
      community_option_explicit = true;
    } else if (option == "--convergence_cell_hierarchy_path" || option == "--cell_hierarchy_path") {
      config.power_ann.community_polar_build.convergence_cell_hierarchy_path =
          read_value(argc, argv, i);
      community_option_explicit = true;
    } else if (option == "--adaptive_multi_capacity") {
      config.power_ann.community_polar_build.adaptive_multi_capacity =
          parse_bool(read_value(argc, argv, i), option);
      community_option_explicit = true;
    } else if (option == "--convergence_cell_capacity_path") {
      config.power_ann.community_polar_build.convergence_cell_capacity_path =
          read_value(argc, argv, i);
      community_option_explicit = true;
    } else if (option == "--adaptive_policy_version") {
      config.power_ann.community_polar_build.adaptive_policy_version =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--adaptive_page_bytes") {
      config.power_ann.community_polar_build.adaptive_page_bytes =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--adaptive_vector_bytes") {
      config.power_ann.community_polar_build.adaptive_vector_bytes =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--adaptive_lru_pages") {
      config.power_ann.community_polar_build.adaptive_lru_pages =
          std::stoull(read_value(argc, argv, i));
      community_option_explicit = true;
    } else if (option == "--adaptive_maximum_normalized_rms_radius") {
      config.power_ann.community_polar_build.adaptive_maximum_normalized_rms_radius =
          std::stod(read_value(argc, argv, i));
      community_option_explicit = true;
    } else if (option == "--adaptive_maximum_normalized_p95_radius") {
      config.power_ann.community_polar_build.adaptive_maximum_normalized_p95_radius =
          std::stod(read_value(argc, argv, i));
      community_option_explicit = true;
    } else if (option == "--adaptive_maximum_split_gain") {
      config.power_ann.community_polar_build.adaptive_maximum_split_gain =
          std::stod(read_value(argc, argv, i));
      community_option_explicit = true;
    } else if (option == "--adaptive_minimum_centroid_radius_overlap") {
      config.power_ann.community_polar_build.adaptive_minimum_centroid_radius_overlap =
          std::stod(read_value(argc, argv, i));
      community_option_explicit = true;
    } else if (option == "--adaptive_minimum_train_cross_child_coaccess") {
      config.power_ann.community_polar_build.adaptive_minimum_train_cross_child_coaccess =
          std::stod(read_value(argc, argv, i));
      community_option_explicit = true;
    } else if (option == "--adaptive_maximum_unseen_pq_increase") {
      config.power_ann.community_polar_build.adaptive_maximum_unseen_pq_increase =
          std::stod(read_value(argc, argv, i));
      community_option_explicit = true;
    } else if (option == "--convergence_cell_source_path") {
      config.power_ann.community_polar_build.convergence_cell_source_path =
          read_value(argc, argv, i);
      community_option_explicit = true;
    } else if (option == "--block_target_size") {
      config.power_ann.community_polar_build.block_target_size =
          static_cast<uint32_t>(std::stoul(read_value(argc, argv, i)));
      community_option_explicit = true;
    } else if (option == "--community_sidecar_only") {
      config.community_sidecar_only = parse_bool(read_value(argc, argv, i), option);
    } else if (option == "--rcni_path") {
      config.rcni_path = read_value(argc, argv, i);
    } else if (option == "--community_sidecar_path" || option == "--skipcell_sidecar_path") {
      config.community_sidecar_path = read_value(argc, argv, i);
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }

  if (top_ratio_explicit && importance_mass_explicit) {
    throw std::invalid_argument("--hub_top_ratio and --hub_importance_mass are mutually exclusive");
  }
  if (top_ratio_explicit) {
    if (selection_mode_explicit && config.power_ann.hub_selection.mode !=
                                       powerlaw_ann::candidate_hub_selection_mode_t::TOP_RATIO) {
      throw std::invalid_argument("--hub_top_ratio conflicts with --hub_selection");
    }
    config.power_ann.hub_selection.mode = powerlaw_ann::candidate_hub_selection_mode_t::TOP_RATIO;
  }
  if (importance_mass_explicit) {
    if (selection_mode_explicit &&
        config.power_ann.hub_selection.mode !=
            powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS) {
      throw std::invalid_argument("--hub_importance_mass conflicts with --hub_selection");
    }
    config.power_ann.hub_selection.mode =
        powerlaw_ann::candidate_hub_selection_mode_t::IMPORTANCE_MASS;
  }
  if (community_option_explicit && !config.power_ann.community_polar_build.is_enabled()) {
    throw std::invalid_argument("Community-Polar options require --community_count");
  }
  if (config.community_sidecar_only) {
    if (!config.power_ann.enable_powerann || !config.power_ann.community_polar_build.is_enabled()) {
      throw std::invalid_argument(
          "--community_sidecar_only requires --enable_powerann true and --community_count");
    }
    if (config.index.index_path_prefix.empty()) {
      throw std::invalid_argument("--community_sidecar_only requires --index_path_prefix");
    }
    if (config.rcni_path.empty()) {
      config.rcni_path = config.index.index_path_prefix;
      config.rcni_path += ".rcni.csv";
    }
    if (config.community_sidecar_path.empty()) {
      throw std::invalid_argument("--community_sidecar_only requires --community_sidecar_path");
    }
  } else if (!config.rcni_path.empty() || !config.community_sidecar_path.empty()) {
    throw std::invalid_argument(
        "--rcni_path and --community_sidecar_path require --community_sidecar_only true");
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
    powerlaw_ann::power_ann_t power_ann(config.power_ann);
    if (config.community_sidecar_only) {
      power_ann.build_community_polar_sidecar(config.index.data_path,
                                              config.index.index_path_prefix, config.rcni_path,
                                              config.community_sidecar_path);
    } else {
      power_ann.build_diskann_index(config.index);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "build_diskann_index: " << error.what() << '\n';
    print_usage(argv[0]);
    return 1;
  }
}
