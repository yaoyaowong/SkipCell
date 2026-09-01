#include "index/index_factory.h"

#include "index/in_mem_data_store.h"
#include "index/in_mem_graph_store.h"
#include "index/vamana_index.h"
#include "pq/pq_data_store.h"
#include "pq/pq_l2_distance.h"

#include <type_traits>

namespace powerlaw_ann {

index_factory_t::index_factory_t(const index_config_t& config)
    : config_(std::make_unique<index_config_t>(config)) {
  check_config();
}

std::unique_ptr<abstract_index_t> index_factory_t::create_instance() {
  return create_instance<float, uint32_t, uint32_t>();
}

void index_factory_t::check_config() {
  if (config_->dynamic_index && !config_->enable_tags) {
    throw diskann_exception_t("ERROR: Dynamic Indexing must have tags enabled.", -1, __FUNCSIG__,
                              __FILE__, __LINE__);
  }

  if (config_->data_type != "float") {
    throw diskann_exception_t("PowerLawANN Step 5 supports only float data.", -1);
  }

  if (config_->tag_type != "uint32" || config_->label_type != "uint32") {
    throw diskann_exception_t("PowerLawANN Step 5 supports only uint32 tags and labels.", -1);
  }
}

template <typename T>
distance_t<T>* index_factory_t::construct_inmem_distance_fn(metric_t metric) {
  if (metric == powerlaw_ann::metric_t::COSINE && std::is_same<T, float>::value) {
    return (distance_t<T>*) new avx_normalized_cosine_distance_float_t();
  } else {
    return (distance_t<T>*) get_distance_function<T>(metric);
  }
}

template <typename T>
std::shared_ptr<abstract_data_store_t<T>>
index_factory_t::construct_datastore(data_store_strategy_t strategy, size_t total_internal_points,
                                     size_t dimension, metric_t metric) {
  std::unique_ptr<distance_t<T>> distance;
  switch (strategy) {
  case data_store_strategy_t::MEMORY:
    distance.reset(construct_inmem_distance_fn<T>(metric));
    return std::make_shared<powerlaw_ann::in_mem_data_store_t<T>>(
        (location_t) total_internal_points, dimension, std::move(distance));
  default:
    break;
  }
  return nullptr;
}

std::unique_ptr<abstract_graph_store_t>
index_factory_t::construct_graphstore(const graph_store_strategy_t strategy, const size_t size,
                                      const size_t reserve_graph_degree) {
  switch (strategy) {
  case graph_store_strategy_t::MEMORY:
    return std::make_unique<in_mem_graph_store_t>(size, reserve_graph_degree);
  default:
    throw diskann_exception_t("Error : Current graph_store_strategy is not supported.", -1);
  }
}

template <typename T>
std::shared_ptr<abstract_data_store_t<T>>
index_factory_t::construct_pq_datastore(data_store_strategy_t strategy, size_t num_points,
                                        size_t dimension, metric_t metric, size_t num_pq_chunks,
                                        bool use_opq) {
  if (strategy != data_store_strategy_t::MEMORY) {
    throw diskann_exception_t(
        "PQ-distance construction currently requires an in-memory data store.", -1, __FUNCSIG__,
        __FILE__, __LINE__);
  }

  std::unique_ptr<distance_t<T>> distance_fn(construct_inmem_distance_fn<T>(metric));
  auto pq_distance_fn =
      std::make_unique<pq_l2_distance_t<T>>(static_cast<uint32_t>(num_pq_chunks), use_opq);
  return std::make_shared<pq_data_store_t<T>>(dimension, static_cast<location_t>(num_points),
                                              num_pq_chunks, std::move(distance_fn),
                                              std::move(pq_distance_fn));
}

template <typename data_type, typename tag_type, typename label_type>
std::unique_ptr<abstract_index_t> index_factory_t::create_instance() {
  size_t num_points = config_->max_points + config_->num_frozen_pts;
  size_t dim = config_->dimension;
  // auto graph_store = construct_graphstore(config_->graph_strategy, num_points);
  auto data_store =
      construct_datastore<data_type>(config_->data_strategy, num_points, dim, config_->metric);
  std::shared_ptr<abstract_data_store_t<data_type>> pq_data_store = nullptr;

  if (config_->data_strategy == data_store_strategy_t::MEMORY && config_->pq_dist_build) {
    pq_data_store =
        construct_pq_datastore<data_type>(config_->data_strategy, num_points, dim, config_->metric,
                                          config_->num_pq_chunks, config_->use_opq);
  } else {
    pq_data_store = data_store;
  }
  size_t max_reserve_degree =
      (size_t) (defaults::GRAPH_SLACK_FACTOR * 1.05 *
                (config_->index_write_params == nullptr ? 0
                                                        : config_->index_write_params->max_degree));
  std::unique_ptr<abstract_graph_store_t> graph_store =
      construct_graphstore(config_->graph_strategy, num_points, max_reserve_degree);

  // REFACTOR TODO: Must construct in-memory PQDatastore if strategy == ONDISK and must construct
  // in-mem and on-disk pq_data_store_t if strategy == ONDISK and disk_pq is required.
  return std::make_unique<powerlaw_ann::vamana_index_t<data_type, tag_type, label_type>>(
      *config_, data_store, std::move(graph_store), pq_data_store);
}

#if 0
std::unique_ptr<abstract_index_t> index_factory_t::create_instance(const std::string &data_type, const std::string &tag_type,
                                                             const std::string &label_type)
{
    if (data_type == std::string("float"))
    {
        return create_instance<float>(tag_type, label_type);
    }
    else if (data_type == std::string("uint8"))
    {
        return create_instance<uint8_t>(tag_type, label_type);
    }
    else if (data_type == std::string("int8"))
    {
        return create_instance<int8_t>(tag_type, label_type);
    }
    else
        throw diskann_exception_t("Error: unsupported data_type please choose from [float/int8/uint8]", -1);
}

template <typename data_type>
std::unique_ptr<abstract_index_t> index_factory_t::create_instance(const std::string &tag_type, const std::string &label_type)
{
    if (tag_type == std::string("int32"))
    {
        return create_instance<data_type, int32_t>(label_type);
    }
    else if (tag_type == std::string("uint32"))
    {
        return create_instance<data_type, uint32_t>(label_type);
    }
    else if (tag_type == std::string("int64"))
    {
        return create_instance<data_type, int64_t>(label_type);
    }
    else if (tag_type == std::string("uint64"))
    {
        return create_instance<data_type, uint64_t>(label_type);
    }
    else
        throw diskann_exception_t("Error: unsupported tag_type please choose from [int32/uint32/int64/uint64]", -1);
}

template <typename data_type, typename tag_type>
std::unique_ptr<abstract_index_t> index_factory_t::create_instance(const std::string &label_type)
{
    if (label_type == std::string("uint16") || label_type == std::string("ushort"))
    {
        return create_instance<data_type, tag_type, uint16_t>();
    }
    else if (label_type == std::string("uint32") || label_type == std::string("uint"))
    {
        return create_instance<data_type, tag_type, uint32_t>();
    }
    else
        throw diskann_exception_t("Error: unsupported label_type please choose from [uint/ushort]", -1);
}
#endif

// template POWERLAWANN_DLLEXPORT std::shared_ptr<abstract_data_store_t<uint8_t>>
// index_factory_t::construct_datastore(
//     data_store_strategy_t stratagy, size_t num_points, size_t dimension, metric_t m);
// template POWERLAWANN_DLLEXPORT std::shared_ptr<abstract_data_store_t<int8_t>>
// index_factory_t::construct_datastore(
//     data_store_strategy_t stratagy, size_t num_points, size_t dimension, metric_t m);
// template POWERLAWANN_DLLEXPORT std::shared_ptr<abstract_data_store_t<float>>
// index_factory_t::construct_datastore(
//     data_store_strategy_t stratagy, size_t num_points, size_t dimension, metric_t m);

template POWERLAWANN_DLLEXPORT std::shared_ptr<abstract_data_store_t<int8_t>>
index_factory_t::construct_pq_datastore(data_store_strategy_t strategy, size_t num_points,
                                        size_t dimension, metric_t metric, size_t num_pq_chunks,
                                        bool use_opq);
template POWERLAWANN_DLLEXPORT std::shared_ptr<abstract_data_store_t<uint8_t>>
index_factory_t::construct_pq_datastore(data_store_strategy_t strategy, size_t num_points,
                                        size_t dimension, metric_t metric, size_t num_pq_chunks,
                                        bool use_opq);
template POWERLAWANN_DLLEXPORT std::shared_ptr<abstract_data_store_t<float>>
index_factory_t::construct_pq_datastore(data_store_strategy_t strategy, size_t num_points,
                                        size_t dimension, metric_t metric, size_t num_pq_chunks,
                                        bool use_opq);

} // namespace powerlaw_ann
