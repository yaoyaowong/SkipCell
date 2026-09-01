#ifndef INDEX_INDEX_CONFIG
#define INDEX_INDEX_CONFIG

#include "common/diskann_exception.h"
#include "common/distance.h"
#include "common/logger.h"
#include "common/parameters.h"

#include <cstddef>
#include <memory>
#include <string>

namespace powerlaw_ann {
enum class data_store_strategy_t { MEMORY };

enum class graph_store_strategy_t { MEMORY };

struct index_config_t {
  data_store_strategy_t data_strategy;
  graph_store_strategy_t graph_strategy;

  metric_t metric;
  size_t dimension;
  size_t max_points;

  bool dynamic_index;
  bool enable_tags;
  bool pq_dist_build;
  bool concurrent_consolidate;
  bool use_opq;
  bool filtered_index;

  size_t num_pq_chunks;
  size_t num_frozen_pts;

  std::string label_type;
  std::string tag_type;
  std::string data_type;

  // Params for building index
  std::shared_ptr<index_write_parameters_t> index_write_params;
  // Params for searching index
  std::shared_ptr<index_search_params_t> index_search_params;

private:
  index_config_t(data_store_strategy_t data_strategy, graph_store_strategy_t graph_strategy,
                 metric_t metric, size_t dimension, size_t max_points, size_t num_pq_chunks,
                 size_t num_frozen_points, bool dynamic_index, bool enable_tags, bool pq_dist_build,
                 bool concurrent_consolidate, bool use_opq, bool filtered_index,
                 std::string& data_type, const std::string& tag_type, const std::string& label_type,
                 std::shared_ptr<index_write_parameters_t> index_write_params,
                 std::shared_ptr<index_search_params_t> index_search_params)
      : data_strategy(data_strategy), graph_strategy(graph_strategy), metric(metric),
        dimension(dimension), max_points(max_points), dynamic_index(dynamic_index),
        enable_tags(enable_tags), pq_dist_build(pq_dist_build),
        concurrent_consolidate(concurrent_consolidate), use_opq(use_opq),
        filtered_index(filtered_index), num_pq_chunks(num_pq_chunks),
        num_frozen_pts(num_frozen_points), label_type(label_type), tag_type(tag_type),
        data_type(data_type), index_write_params(index_write_params),
        index_search_params(index_search_params) {}

  friend class index_config_builder_t;
};

class index_config_builder_t {
public:
  index_config_builder_t() = default;

  index_config_builder_t& with_metric(metric_t m) {
    this->metric_ = m;
    return *this;
  }

  index_config_builder_t& with_graph_load_store_strategy(graph_store_strategy_t graph_strategy) {
    this->graph_strategy_ = graph_strategy;
    return *this;
  }

  index_config_builder_t& with_data_load_store_strategy(data_store_strategy_t data_strategy) {
    this->data_strategy_ = data_strategy;
    return *this;
  }

  index_config_builder_t& with_dimension(size_t dimension) {
    this->dimension_ = dimension;
    return *this;
  }

  index_config_builder_t& with_max_points(size_t max_points) {
    this->max_points_ = max_points;
    return *this;
  }

  index_config_builder_t& is_dynamic_index(bool dynamic_index) {
    this->dynamic_index_ = dynamic_index;
    return *this;
  }

  index_config_builder_t& is_enable_tags(bool enable_tags) {
    this->enable_tags_ = enable_tags;
    return *this;
  }

  index_config_builder_t& is_pq_dist_build(bool pq_dist_build) {
    this->pq_dist_build_ = pq_dist_build;
    return *this;
  }

  index_config_builder_t& is_concurrent_consolidate(bool concurrent_consolidate) {
    this->concurrent_consolidate_ = concurrent_consolidate;
    return *this;
  }

  index_config_builder_t& is_use_opq(bool use_opq) {
    this->use_opq_ = use_opq;
    return *this;
  }

  index_config_builder_t& is_filtered(bool is_filtered) {
    this->filtered_index_ = is_filtered;
    return *this;
  }

  index_config_builder_t& with_num_pq_chunks(size_t num_pq_chunks) {
    this->num_pq_chunks_ = num_pq_chunks;
    return *this;
  }

  index_config_builder_t& with_num_frozen_pts(size_t num_frozen_pts) {
    this->num_frozen_pts_ = num_frozen_pts;
    return *this;
  }

  index_config_builder_t& with_label_type(const std::string& label_type) {
    this->label_type_ = label_type;
    return *this;
  }

  index_config_builder_t& with_tag_type(const std::string& tag_type) {
    this->tag_type_ = tag_type;
    return *this;
  }

  index_config_builder_t& with_data_type(const std::string& data_type) {
    this->data_type_ = data_type;
    return *this;
  }

  index_config_builder_t& with_index_write_params(index_write_parameters_t& index_write_params) {
    this->index_write_params_ = std::make_shared<index_write_parameters_t>(index_write_params);
    return *this;
  }

  index_config_builder_t&
  with_index_write_params(std::shared_ptr<index_write_parameters_t> index_write_params_ptr) {
    if (index_write_params_ptr == nullptr) {
      powerlaw_ann::cout << "Passed, empty build_params while creating index config" << std::endl;
      return *this;
    }
    this->index_write_params_ = index_write_params_ptr;
    return *this;
  }

  index_config_builder_t& with_index_search_params(index_search_params_t& search_params) {
    this->index_search_params_ = std::make_shared<index_search_params_t>(search_params);
    return *this;
  }

  index_config_builder_t&
  with_index_search_params(std::shared_ptr<index_search_params_t> search_params_ptr) {
    if (search_params_ptr == nullptr) {
      powerlaw_ann::cout << "Passed, empty search_params while creating index config" << std::endl;
      return *this;
    }
    this->index_search_params_ = search_params_ptr;
    return *this;
  }

  index_config_t build() {
    if (data_type_ == "" || data_type_.empty())
      throw diskann_exception_t("Error: data_type can not be empty", -1);

    if (dynamic_index_ && num_frozen_pts_ == 0) {
      num_frozen_pts_ = 1;
    }

    if (dynamic_index_) {
      if (index_search_params_ != nullptr && index_search_params_->initial_search_list_size == 0)
        throw diskann_exception_t(
            "Error: please pass initial_search_list_size for building dynamic index.", -1);
    }

    // sanity check
    if (dynamic_index_ && num_frozen_pts_ == 0) {
      powerlaw_ann::cout
          << "num_frozen_pts_ passed as 0 for dynamic_index. Setting it to 1 for safety."
          << std::endl;
      num_frozen_pts_ = 1;
    }

    return index_config_t(data_strategy_, graph_strategy_, metric_, dimension_, max_points_,
                          num_pq_chunks_, num_frozen_pts_, dynamic_index_, enable_tags_,
                          pq_dist_build_, concurrent_consolidate_, use_opq_, filtered_index_,
                          data_type_, tag_type_, label_type_, index_write_params_,
                          index_search_params_);
  }

  index_config_builder_t(const index_config_builder_t&) = delete;
  index_config_builder_t& operator=(const index_config_builder_t&) = delete;

private:
  data_store_strategy_t data_strategy_;
  graph_store_strategy_t graph_strategy_;

  metric_t metric_;
  size_t dimension_;
  size_t max_points_;

  bool dynamic_index_ = false;
  bool enable_tags_ = false;
  bool pq_dist_build_ = false;
  bool concurrent_consolidate_ = false;
  bool use_opq_ = false;
  bool filtered_index_{defaults::HAS_LABELS};

  size_t num_pq_chunks_ = 0;
  size_t num_frozen_pts_{defaults::NUM_FROZEN_POINTS_STATIC};

  std::string label_type_{"uint32"};
  std::string tag_type_{"uint32"};
  std::string data_type_;

  std::shared_ptr<index_write_parameters_t> index_write_params_;
  std::shared_ptr<index_search_params_t> index_search_params_;
};
} // namespace powerlaw_ann

#endif // INDEX_INDEX_CONFIG
