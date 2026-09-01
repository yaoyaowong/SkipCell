#ifndef POWER_ANN
#define POWER_ANN

#include "index/community_polar_index.h"
#include "index/diskann_index.h"
#include "index/io_optimized_index.h"
#include "index/rcni.h"
#include "search/community_polar_search.h"
#include "search/diskann_search.h"

namespace powerlaw_ann {

/**
 * @brief Global feature configuration for the PowerLawANN facade.
 *
 * The default keeps the imported DiskANN baseline active. Enabled build calls collect and
 * persist RCNI importance and deterministically select candidate Hubs. Candidate-Hub selection
 * defaults to a zero top ratio, so experiments must opt into a ratio or importance-mass target.
 * Query execution remains the unchanged DiskANN baseline until a later explicitly selected search
 * method is integrated.
 */
struct power_ann_config_t {
  bool enable_powerann = false;
  candidate_hub_selection_config_t hub_selection;
  community_polar_build_config_t community_polar_build;
  community_polar_search_config_t community_polar_search;
  io_optimization_config_t io_optimization;
};

class power_ann_t {
public:
  /**
   * @brief Construct the facade without allocating build or search state.
   * @param config Global feature configuration copied by the facade.
   */
  explicit power_ann_t(const power_ann_config_t& config = {});

  /**
   * @brief Report whether the caller requested an experimental PowerANN path.
   * @return True only when `enable_powerann` was explicitly enabled.
   */
  bool is_powerann_enabled() const noexcept;

  void build_diskann_memory_index(const diskann_memory_index_config_t& config);
  void build_diskann_index(const diskann_disk_index_config_t& config);
  void build_community_polar_sidecar(const std::filesystem::path& data_path,
                                     const std::filesystem::path& index_path_prefix,
                                     const std::filesystem::path& rcni_path,
                                     const std::filesystem::path& output_sidecar_path);
  diskann_search_result_t search_diskann_index(const diskann_search_config_t& config);
  std::vector<diskann_search_result_t>
  search_diskann_index_curve(const diskann_search_config_t& config,
                             std::span<const uint32_t> search_list_sizes);
  std::vector<diskann_search_result_t>
  search_diskann_index_thread_curve(const diskann_search_config_t& config,
                                    std::span<const uint32_t> thread_counts);
  std::vector<diskann_search_result_t>
  search_diskann_index_cache_curve(const diskann_search_config_t& config,
                                   std::span<const uint64_t> cache_page_counts);

private:
  std::vector<diskann_search_result_t>
  search_diskann_index_impl(const diskann_search_config_t& config,
                            std::span<const uint32_t> search_list_sizes,
                            std::span<const uint32_t> thread_counts,
                            std::span<const uint64_t> cache_page_counts = {});
  power_ann_config_t config_;
};

} // namespace powerlaw_ann

#endif // POWER_ANN
