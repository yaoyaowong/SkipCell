#include "index/diskann_index.h"

#include "common/ann_error.h"
#include "common/logger.h"
#include "common/parameters.h"
#include "common/utils.h"
#include "index/disk_index_utils.h"
#include "index/index_build_params.h"
#include "index/index_config.h"
#include "index/index_factory.h"
#include "index/vamana_index.h"
#include "pq/pq_common.h"

#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

namespace powerlaw_ann {
namespace {

void validate_index_output_prefix(const std::filesystem::path& index_path_prefix) {
  if (index_path_prefix.empty()) {
    throw ann_exception_t("index_path_prefix must not be empty");
  }
  if (index_path_prefix.filename().empty()) {
    throw ann_exception_t(
        "index_path_prefix must include a file name and must not end with a path separator: " +
        index_path_prefix.string());
  }
  if (std::filesystem::is_directory(index_path_prefix)) {
    throw ann_exception_t("index_path_prefix must be a file prefix, not a directory: " +
                          index_path_prefix.string());
  }

  const auto parent = index_path_prefix.parent_path();
  if (!parent.empty() && !std::filesystem::is_directory(parent)) {
    throw ann_exception_t("index output directory does not exist: " + parent.string());
  }
}

void validate_config(const diskann_memory_index_config_t& config) {
  if (config.data_path.empty()) {
    throw ann_exception_t("data_path must not be empty");
  }
  if (!std::filesystem::is_regular_file(config.data_path)) {
    throw ann_exception_t("data_path is not a regular file: " + config.data_path.string());
  }
  validate_index_output_prefix(config.index_path_prefix);
  if (config.max_degree == 0) {
    throw ann_exception_t("max_degree must be greater than zero");
  }
  if (config.build_list_size < config.max_degree) {
    throw ann_exception_t("build_list_size must be greater than or equal to max_degree");
  }
  if (!std::isfinite(config.alpha) || config.alpha < 1.0F) {
    throw ann_exception_t("alpha must be finite and greater than or equal to 1.0");
  }
}

void validate_config(const diskann_disk_index_config_t& config) {
  if (config.data_path.empty()) {
    throw ann_exception_t("data_path must not be empty");
  }
  if (!std::filesystem::is_regular_file(config.data_path)) {
    throw ann_exception_t("data_path is not a regular file: " + config.data_path.string());
  }
  validate_index_output_prefix(config.index_path_prefix);
  if (!std::isfinite(config.search_dram_budget_gb) || config.search_dram_budget_gb <= 0.0) {
    throw ann_exception_t("search_dram_budget_gb must be finite and greater than zero");
  }
  if (!std::isfinite(config.build_dram_budget_gb) || config.build_dram_budget_gb <= 0.0) {
    throw ann_exception_t("build_dram_budget_gb must be finite and greater than zero");
  }
  if (config.max_degree == 0) {
    throw ann_exception_t("max_degree must be greater than zero");
  }
  if (config.build_list_size < config.max_degree) {
    throw ann_exception_t("build_list_size must be greater than or equal to max_degree");
  }
  if (config.append_reorder_data && config.disk_pq_bytes == 0) {
    throw ann_exception_t("append_reorder_data requires disk_pq_bytes greater than zero");
  }

  size_t num_points = 0;
  size_t dimension = 0;
  get_bin_metadata(config.data_path.string(), num_points, dimension);
  if (num_points == 0 || dimension == 0) {
    throw ann_exception_t("input bin metadata must contain non-zero point and dimension counts");
  }
  if (config.build_pq_bytes > dimension) {
    throw ann_exception_t("build_pq_bytes must not exceed the vector dimension");
  }
  if (config.quantized_dimension > dimension || config.quantized_dimension > MAX_PQ_CHUNKS) {
    throw ann_exception_t(
        "quantized_dimension must be zero or no greater than the vector dimension and 512");
  }
}

} // namespace

void diskann_index_t::build_memory_index(const diskann_memory_index_config_t& config,
                                         rcni_prune_observer_t* rcni_observer,
                                         post_link_graph_observer_t* post_link_observer) {
  validate_config(config);

  try {
    size_t data_num = 0;
    size_t data_dim = 0;
    get_bin_metadata(config.data_path.string(), data_num, data_dim);
    if (data_num == 0 || data_dim == 0) {
      throw ann_exception_t("input bin metadata must contain non-zero point "
                            "and dimension counts");
    }

    cout << "Starting PowerLawANN baseline build with R: " << config.max_degree
         << "  Lbuild: " << config.build_list_size << "  alpha: " << config.alpha
         << "  #threads: " << config.num_threads << std::endl;

    auto write_params = index_write_parameters_builder_t(config.build_list_size, config.max_degree)
                            .with_filter_list_size(0)
                            .with_alpha(config.alpha)
                            .with_saturate_graph(false)
                            .with_num_threads(config.num_threads)
                            .build();

    auto filter_params = index_filter_params_builder_t()
                             .with_universal_label("")
                             .with_label_file("")
                             .with_save_path_prefix(config.index_path_prefix.string())
                             .build();

    auto index_config = index_config_builder_t()
                            .with_metric(metric_t::L2)
                            .with_dimension(data_dim)
                            .with_max_points(data_num)
                            .with_data_load_store_strategy(data_store_strategy_t::MEMORY)
                            .with_graph_load_store_strategy(graph_store_strategy_t::MEMORY)
                            .with_data_type("float")
                            .with_label_type("uint32")
                            .with_tag_type("uint32")
                            .is_dynamic_index(false)
                            .with_index_write_params(write_params)
                            .is_enable_tags(false)
                            .is_use_opq(false)
                            .is_pq_dist_build(false)
                            .with_num_pq_chunks(0)
                            .build();

    index_factory_t factory(index_config);
    std::unique_ptr<abstract_index_t> index = factory.create_instance();
    if (rcni_observer != nullptr) {
      auto* vamana = dynamic_cast<vamana_index_t<float, uint32_t, uint32_t>*>(index.get());
      if (vamana == nullptr) {
        throw ann_exception_t("RCNI requires the static float32 L2 Vamana build path");
      }
      vamana->set_rcni_prune_observer(rcni_observer);
    }
    if (post_link_observer != nullptr) {
      auto* vamana = dynamic_cast<vamana_index_t<float, uint32_t, uint32_t>*>(index.get());
      if (vamana == nullptr) {
        throw ann_exception_t(
            "Post-link graph observation requires the static float32 L2 Vamana build path");
      }
      vamana->set_post_link_graph_observer(post_link_observer);
    }
    index->build(config.data_path.string(), data_num, filter_params);
    index->save(config.index_path_prefix.c_str());
  } catch (const ann_exception_t&) {
    throw;
  } catch (const std::exception& error) {
    throw ann_exception_t("DiskANN-compatible Vamana build failed: " + std::string(error.what()));
  }
}

void diskann_index_t::build_disk_index(const diskann_disk_index_config_t& config,
                                       rcni_prune_observer_t* rcni_observer,
                                       post_link_graph_observer_t* post_link_observer) {
  validate_config(config);

  std::ostringstream parameters;
  parameters << config.max_degree << ' ' << config.build_list_size << ' '
             << config.search_dram_budget_gb << ' ' << config.build_dram_budget_gb << ' '
             << config.num_threads << ' ' << config.disk_pq_bytes << ' '
             << static_cast<uint32_t>(config.append_reorder_data) << ' ' << config.build_pq_bytes
             << ' ' << config.quantized_dimension;

  try {
    const int status = powerlaw_ann::build_disk_index<float, uint32_t>(
        config.data_path.c_str(), config.index_path_prefix.c_str(), parameters.str().c_str(),
        metric_t::L2, false, "", false, "", "", 0, 0, "", rcni_observer, post_link_observer);
    if (status != 0) {
      throw ann_exception_t("DiskANN-compatible disk index build returned status " +
                            std::to_string(status));
    }
  } catch (const ann_exception_t&) {
    throw;
  } catch (const std::exception& error) {
    throw ann_exception_t("DiskANN-compatible disk index build failed: " +
                          std::string(error.what()));
  }
}

} // namespace powerlaw_ann
