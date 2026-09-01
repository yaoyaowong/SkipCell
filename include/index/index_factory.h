#ifndef INDEX_INDEX_FACTORY
#define INDEX_INDEX_FACTORY

#include "common/distance.h"
#include "index/abstract_data_store.h"
#include "index/abstract_graph_store.h"
#include "index/abstract_index.h"
#include "index/index_config.h"

#include <cstddef>
#include <memory>
#include <string>

namespace powerlaw_ann {
class index_factory_t {
public:
  POWERLAWANN_DLLEXPORT explicit index_factory_t(const index_config_t& config);
  POWERLAWANN_DLLEXPORT std::unique_ptr<abstract_index_t> create_instance();

  POWERLAWANN_DLLEXPORT static std::unique_ptr<abstract_graph_store_t>
  construct_graphstore(const graph_store_strategy_t stratagy, const size_t size,
                       const size_t reserve_graph_degree);

  template <typename T>
  POWERLAWANN_DLLEXPORT static std::shared_ptr<abstract_data_store_t<T>>
  construct_datastore(data_store_strategy_t stratagy, size_t num_points, size_t dimension,
                      metric_t m);
  template <typename T>
  POWERLAWANN_DLLEXPORT static std::shared_ptr<abstract_data_store_t<T>>
  construct_pq_datastore(data_store_strategy_t strategy, size_t num_points, size_t dimension,
                         metric_t m, size_t num_pq_chunks, bool use_opq);
  template <typename T>
  static distance_t<T>* construct_inmem_distance_fn(metric_t m);

private:
  void check_config();

  template <typename data_type, typename tag_type, typename label_type>
  std::unique_ptr<abstract_index_t> create_instance();

  std::unique_ptr<abstract_index_t> create_instance(const std::string& data_type,
                                                    const std::string& tag_type,
                                                    const std::string& label_type);

  template <typename data_type>
  std::unique_ptr<abstract_index_t> create_instance(const std::string& tag_type,
                                                    const std::string& label_type);

  template <typename data_type, typename tag_type>
  std::unique_ptr<abstract_index_t> create_instance(const std::string& label_type);

  std::unique_ptr<index_config_t> config_;
};

} // namespace powerlaw_ann

#endif // INDEX_INDEX_FACTORY
