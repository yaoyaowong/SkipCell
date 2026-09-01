#ifndef INDEX_DISKANN_INDEX
#define INDEX_DISKANN_INDEX

#include <cstdint>
#include <filesystem>

namespace powerlaw_ann {

class post_link_graph_observer_t;
class rcni_prune_observer_t;

struct diskann_memory_index_config_t {
  std::filesystem::path data_path;
  std::filesystem::path index_path_prefix;
  uint32_t max_degree = 64;
  uint32_t build_list_size = 100;
  float alpha = 1.2F;
  uint32_t num_threads = 0;
};

struct diskann_disk_index_config_t {
  std::filesystem::path data_path;
  std::filesystem::path index_path_prefix;
  double search_dram_budget_gb = 1.0;
  double build_dram_budget_gb = 4.0;
  uint32_t max_degree = 64;
  uint32_t build_list_size = 100;
  uint32_t num_threads = 0;
  uint32_t disk_pq_bytes = 0;
  bool append_reorder_data = false;
  uint32_t build_pq_bytes = 0;
  uint32_t quantized_dimension = 0;
};

class diskann_index_t {
public:
  void build_memory_index(const diskann_memory_index_config_t& config,
                          rcni_prune_observer_t* rcni_observer = nullptr,
                          post_link_graph_observer_t* post_link_observer = nullptr);
  void build_disk_index(const diskann_disk_index_config_t& config,
                        rcni_prune_observer_t* rcni_observer = nullptr,
                        post_link_graph_observer_t* post_link_observer = nullptr);
};

} // namespace powerlaw_ann

#endif // INDEX_DISKANN_INDEX
