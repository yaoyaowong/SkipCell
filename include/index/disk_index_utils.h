#ifndef INDEX_DISK_INDEX_UTILS
#define INDEX_DISK_INDEX_UTILS

#include "common/distance.h"
#include "common/platform_compat.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace powerlaw_ann {
class post_link_graph_observer_t;
class rcni_prune_observer_t;

const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
const double PQ_TRAINING_SET_FRACTION = 0.1;
const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
const uint32_t NUM_NODES_TO_CACHE = 250000;
const uint32_t WARMUP_L = 20;
const uint32_t NUM_KMEANS_REPS = 12;

POWERLAWANN_DLLEXPORT double get_memory_budget(const std::string& mem_budget_str);
POWERLAWANN_DLLEXPORT double get_memory_budget(double search_ram_budget_in_gb);
POWERLAWANN_DLLEXPORT void add_new_file_to_single_index(std::string index_file,
                                                        std::string new_file);

POWERLAWANN_DLLEXPORT size_t calculate_num_pq_chunks(double final_index_ram_limit,
                                                     size_t points_num, uint32_t dim);

POWERLAWANN_DLLEXPORT void read_idmap(const std::string& fname, std::vector<uint32_t>& ivecs);

POWERLAWANN_DLLEXPORT int merge_shards(const std::string& vamana_prefix,
                                       const std::string& vamana_suffix,
                                       const std::string& idmaps_prefix,
                                       const std::string& idmaps_suffix, const uint64_t nshards,
                                       uint32_t max_degree, const std::string& output_vamana,
                                       const std::string& medoids_file, bool use_filters = false,
                                       const std::string& labels_to_medoids_file = std::string(""));

POWERLAWANN_DLLEXPORT void extract_shard_labels(const std::string& in_label_file,
                                                const std::string& shard_ids_bin,
                                                const std::string& shard_label_file);

template <typename T>
POWERLAWANN_DLLEXPORT std::string preprocess_base_file(const std::string& infile,
                                                       const std::string& index_prefix,
                                                       powerlaw_ann::metric_t& distance_metric);

template <typename T, typename label_t = uint32_t>
POWERLAWANN_DLLEXPORT int build_merged_vamana_index(
    std::string base_file, powerlaw_ann::metric_t compare_metric, uint32_t l, uint32_t r,
    double sampling_rate, double ram_budget, std::string mem_index_path, std::string medoids_file,
    std::string centroids_file, size_t build_pq_bytes, bool use_opq, uint32_t num_threads,
    bool use_filters = false, const std::string& label_file = std::string(""),
    const std::string& labels_to_medoids_file = std::string(""),
    const std::string& universal_label = "", const uint32_t filtered_l = 0,
    const std::string& vamana_build_stats_prefix = std::string(""),
    rcni_prune_observer_t* rcni_observer = nullptr,
    post_link_graph_observer_t* post_link_observer = nullptr);

template <typename T, typename label_t = uint32_t>
POWERLAWANN_DLLEXPORT int build_disk_index(
    const char* data_file_path, const char* index_file_path, const char* index_build_parameters,
    powerlaw_ann::metric_t compare_metric, bool use_opq = false,
    const std::string& codebook_prefix = "", // default is empty for no codebook pass in
    bool use_filters = false,
    const std::string& label_file = std::string(""), // default is empty string for no label_file
    const std::string& universal_label = "", const uint32_t filter_threshold = 0,
    const uint32_t filtered_l = 0,
    const std::string& vamana_build_stats_prefix =
        std::string(""), // default is empty string for no universal label
    rcni_prune_observer_t* rcni_observer = nullptr,
    post_link_graph_observer_t* post_link_observer = nullptr);

template <typename T>
POWERLAWANN_DLLEXPORT void
create_disk_layout(const std::string base_file, const std::string mem_index_file,
                   const std::string output_file,
                   const std::string reorder_data_file = std::string(""));

} // namespace powerlaw_ann

#endif // INDEX_DISK_INDEX_UTILS
