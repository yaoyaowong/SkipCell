#include "index/abstract_index.h"

#include "common/platform_compat.h"

namespace powerlaw_ann {

template <typename data_type, typename tag_type>
void abstract_index_t::build(const data_type* data, const size_t num_points_to_load,
                             const std::vector<tag_type>& tags) {
  auto any_data = std::any(data);
  auto any_tags_vec = tag_vector_t(tags);
  this->build_impl(any_data, num_points_to_load, any_tags_vec);
}

template <typename data_type, typename IDType>
std::pair<uint32_t, uint32_t> abstract_index_t::search(const data_type* query, const size_t K,
                                                       const uint32_t L, IDType* indices,
                                                       float* distances) {
  auto any_indices = std::any(indices);
  auto any_query = std::any(query);
  return search_impl(any_query, K, L, any_indices, distances);
}

template <typename data_type, typename tag_type>
size_t abstract_index_t::search_with_tags(const data_type* query, const uint64_t K,
                                          const uint32_t L, tag_type* tags, float* distances,
                                          std::vector<data_type*>& res_vectors, bool use_filters,
                                          const std::string filter_label) {
  auto any_query = std::any(query);
  auto any_tags = std::any(tags);
  auto any_res_vectors = data_vector_t(res_vectors);
  return this->search_with_tags_impl(any_query, K, L, any_tags, distances, any_res_vectors,
                                     use_filters, filter_label);
}

template <typename index_t>
std::pair<uint32_t, uint32_t>
abstract_index_t::search_with_filters(const data_type_t& query, const std::string& raw_label,
                                      const size_t K, const uint32_t L, index_t* indices,
                                      float* distances) {
  auto any_indices = std::any(indices);
  return search_with_filters_impl(query, raw_label, K, L, any_indices, distances);
}

template <typename data_type>
void abstract_index_t::search_with_optimized_layout(const data_type* query, size_t K, size_t L,
                                                    uint32_t* indices) {
  auto any_query = std::any(query);
  this->search_with_optimized_layout_impl(any_query, K, L, indices);
}

template <typename data_type, typename tag_type>
int abstract_index_t::insert_point(const data_type* point, const tag_type tag) {
  auto any_point = std::any(point);
  auto any_tag = std::any(tag);
  return this->insert_point_impl(any_point, any_tag);
}

template <typename data_type, typename tag_type, typename label_type>
int abstract_index_t::insert_point(const data_type* point, const tag_type tag,
                                   const std::vector<label_type>& labels) {
  auto any_point = std::any(point);
  auto any_tag = std::any(tag);
  auto any_labels = label_vector_t(labels);
  return this->insert_point_impl(any_point, any_tag, any_labels);
}

template <typename tag_type>
int abstract_index_t::lazy_delete(const tag_type& tag) {
  auto any_tag = std::any(tag);
  return this->lazy_delete_impl(any_tag);
}

template <typename tag_type>
void abstract_index_t::lazy_delete(const std::vector<tag_type>& tags,
                                   std::vector<tag_type>& failed_tags) {
  auto any_tags = tag_vector_t(tags);
  auto any_failed_tags = tag_vector_t(failed_tags);
  this->lazy_delete_impl(any_tags, any_failed_tags);
}

template <typename tag_type>
void abstract_index_t::get_active_tags(tsl::robin_set<tag_type>& active_tags) {
  auto any_active_tags = tag_robin_set_t(active_tags);
  this->get_active_tags_impl(any_active_tags);
}

template <typename data_type>
void abstract_index_t::set_start_points_at_random(data_type radius, uint32_t random_seed) {
  auto any_radius = std::any(radius);
  this->set_start_points_at_random_impl(any_radius, random_seed);
}

template <typename tag_type, typename data_type>
int abstract_index_t::get_vector_by_tag(tag_type& tag, data_type* vec) {
  auto any_tag = std::any(tag);
  auto any_data_ptr = std::any(vec);
  return this->get_vector_by_tag_impl(any_tag, any_data_ptr);
}

template <typename label_type>
void abstract_index_t::set_universal_label(const label_type universal_label) {
  auto any_label = std::any(universal_label);
  this->set_universal_label_impl(any_label);
}

// exports
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<float, int32_t>(const float* data, const size_t num_points_to_load,
                                        const std::vector<int32_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<int8_t, int32_t>(const int8_t* data, const size_t num_points_to_load,
                                         const std::vector<int32_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<uint8_t, int32_t>(const uint8_t* data, const size_t num_points_to_load,
                                          const std::vector<int32_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<float, uint32_t>(const float* data, const size_t num_points_to_load,
                                         const std::vector<uint32_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<int8_t, uint32_t>(const int8_t* data, const size_t num_points_to_load,
                                          const std::vector<uint32_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<uint8_t, uint32_t>(const uint8_t* data, const size_t num_points_to_load,
                                           const std::vector<uint32_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<float, int64_t>(const float* data, const size_t num_points_to_load,
                                        const std::vector<int64_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<int8_t, int64_t>(const int8_t* data, const size_t num_points_to_load,
                                         const std::vector<int64_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<uint8_t, int64_t>(const uint8_t* data, const size_t num_points_to_load,
                                          const std::vector<int64_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<float, uint64_t>(const float* data, const size_t num_points_to_load,
                                         const std::vector<uint64_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<int8_t, uint64_t>(const int8_t* data, const size_t num_points_to_load,
                                          const std::vector<uint64_t>& tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::build<uint8_t, uint64_t>(const uint8_t* data, const size_t num_points_to_load,
                                           const std::vector<uint64_t>& tags);

template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
abstract_index_t::search<float, uint32_t>(const float* query, const size_t K, const uint32_t L,
                                          uint32_t* indices, float* distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
abstract_index_t::search<uint8_t, uint32_t>(const uint8_t* query, const size_t K, const uint32_t L,
                                            uint32_t* indices, float* distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
abstract_index_t::search<int8_t, uint32_t>(const int8_t* query, const size_t K, const uint32_t L,
                                           uint32_t* indices, float* distances);

template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
abstract_index_t::search<float, uint64_t>(const float* query, const size_t K, const uint32_t L,
                                          uint64_t* indices, float* distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
abstract_index_t::search<uint8_t, uint64_t>(const uint8_t* query, const size_t K, const uint32_t L,
                                            uint64_t* indices, float* distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
abstract_index_t::search<int8_t, uint64_t>(const int8_t* query, const size_t K, const uint32_t L,
                                           uint64_t* indices, float* distances);

template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
abstract_index_t::search_with_filters<uint32_t>(const data_type_t& query,
                                                const std::string& raw_label, const size_t K,
                                                const uint32_t L, uint32_t* indices,
                                                float* distances);

template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t>
abstract_index_t::search_with_filters<uint64_t>(const data_type_t& query,
                                                const std::string& raw_label, const size_t K,
                                                const uint32_t L, uint64_t* indices,
                                                float* distances);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<float, int32_t>(
    const float* query, const uint64_t K, const uint32_t L, int32_t* tags, float* distances,
    std::vector<float*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<uint8_t, int32_t>(
    const uint8_t* query, const uint64_t K, const uint32_t L, int32_t* tags, float* distances,
    std::vector<uint8_t*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<int8_t, int32_t>(
    const int8_t* query, const uint64_t K, const uint32_t L, int32_t* tags, float* distances,
    std::vector<int8_t*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<float, uint32_t>(
    const float* query, const uint64_t K, const uint32_t L, uint32_t* tags, float* distances,
    std::vector<float*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<uint8_t, uint32_t>(
    const uint8_t* query, const uint64_t K, const uint32_t L, uint32_t* tags, float* distances,
    std::vector<uint8_t*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<int8_t, uint32_t>(
    const int8_t* query, const uint64_t K, const uint32_t L, uint32_t* tags, float* distances,
    std::vector<int8_t*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<float, int64_t>(
    const float* query, const uint64_t K, const uint32_t L, int64_t* tags, float* distances,
    std::vector<float*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<uint8_t, int64_t>(
    const uint8_t* query, const uint64_t K, const uint32_t L, int64_t* tags, float* distances,
    std::vector<uint8_t*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<int8_t, int64_t>(
    const int8_t* query, const uint64_t K, const uint32_t L, int64_t* tags, float* distances,
    std::vector<int8_t*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<float, uint64_t>(
    const float* query, const uint64_t K, const uint32_t L, uint64_t* tags, float* distances,
    std::vector<float*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<uint8_t, uint64_t>(
    const uint8_t* query, const uint64_t K, const uint32_t L, uint64_t* tags, float* distances,
    std::vector<uint8_t*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT size_t abstract_index_t::search_with_tags<int8_t, uint64_t>(
    const int8_t* query, const uint64_t K, const uint32_t L, uint64_t* tags, float* distances,
    std::vector<int8_t*>& res_vectors, bool use_filters, const std::string filter_label);

template POWERLAWANN_DLLEXPORT void
abstract_index_t::search_with_optimized_layout<float>(const float* query, size_t K, size_t L,
                                                      uint32_t* indices);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::search_with_optimized_layout<uint8_t>(const uint8_t* query, size_t K, size_t L,
                                                        uint32_t* indices);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::search_with_optimized_layout<int8_t>(const int8_t* query, size_t K, size_t L,
                                                       uint32_t* indices);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, int32_t>(const float* point, const int32_t tag);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<uint8_t, int32_t>(const uint8_t* point, const int32_t tag);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, int32_t>(const int8_t* point, const int32_t tag);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, uint32_t>(const float* point, const uint32_t tag);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<uint8_t, uint32_t>(const uint8_t* point, const uint32_t tag);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, uint32_t>(const int8_t* point, const uint32_t tag);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, int64_t>(const float* point, const int64_t tag);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<uint8_t, int64_t>(const uint8_t* point, const int64_t tag);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, int64_t>(const int8_t* point, const int64_t tag);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, uint64_t>(const float* point, const uint64_t tag);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<uint8_t, uint64_t>(const uint8_t* point, const uint64_t tag);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, uint64_t>(const int8_t* point, const uint64_t tag);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, int32_t, uint16_t>(const float* point, const int32_t tag,
                                                         const std::vector<uint16_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<uint8_t, int32_t, uint16_t>(const uint8_t* point, const int32_t tag,
                                                           const std::vector<uint16_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, int32_t, uint16_t>(const int8_t* point, const int32_t tag,
                                                          const std::vector<uint16_t>& labels);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, uint32_t, uint16_t>(const float* point, const uint32_t tag,
                                                          const std::vector<uint16_t>& labels);
template POWERLAWANN_DLLEXPORT int abstract_index_t::insert_point<uint8_t, uint32_t, uint16_t>(
    const uint8_t* point, const uint32_t tag, const std::vector<uint16_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, uint32_t, uint16_t>(const int8_t* point, const uint32_t tag,
                                                           const std::vector<uint16_t>& labels);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, int64_t, uint16_t>(const float* point, const int64_t tag,
                                                         const std::vector<uint16_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<uint8_t, int64_t, uint16_t>(const uint8_t* point, const int64_t tag,
                                                           const std::vector<uint16_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, int64_t, uint16_t>(const int8_t* point, const int64_t tag,
                                                          const std::vector<uint16_t>& labels);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, uint64_t, uint16_t>(const float* point, const uint64_t tag,
                                                          const std::vector<uint16_t>& labels);
template POWERLAWANN_DLLEXPORT int abstract_index_t::insert_point<uint8_t, uint64_t, uint16_t>(
    const uint8_t* point, const uint64_t tag, const std::vector<uint16_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, uint64_t, uint16_t>(const int8_t* point, const uint64_t tag,
                                                           const std::vector<uint16_t>& labels);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, int32_t, uint32_t>(const float* point, const int32_t tag,
                                                         const std::vector<uint32_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<uint8_t, int32_t, uint32_t>(const uint8_t* point, const int32_t tag,
                                                           const std::vector<uint32_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, int32_t, uint32_t>(const int8_t* point, const int32_t tag,
                                                          const std::vector<uint32_t>& labels);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, uint32_t, uint32_t>(const float* point, const uint32_t tag,
                                                          const std::vector<uint32_t>& labels);
template POWERLAWANN_DLLEXPORT int abstract_index_t::insert_point<uint8_t, uint32_t, uint32_t>(
    const uint8_t* point, const uint32_t tag, const std::vector<uint32_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, uint32_t, uint32_t>(const int8_t* point, const uint32_t tag,
                                                           const std::vector<uint32_t>& labels);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, int64_t, uint32_t>(const float* point, const int64_t tag,
                                                         const std::vector<uint32_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<uint8_t, int64_t, uint32_t>(const uint8_t* point, const int64_t tag,
                                                           const std::vector<uint32_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, int64_t, uint32_t>(const int8_t* point, const int64_t tag,
                                                          const std::vector<uint32_t>& labels);

template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<float, uint64_t, uint32_t>(const float* point, const uint64_t tag,
                                                          const std::vector<uint32_t>& labels);
template POWERLAWANN_DLLEXPORT int abstract_index_t::insert_point<uint8_t, uint64_t, uint32_t>(
    const uint8_t* point, const uint64_t tag, const std::vector<uint32_t>& labels);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::insert_point<int8_t, uint64_t, uint32_t>(const int8_t* point, const uint64_t tag,
                                                           const std::vector<uint32_t>& labels);

template POWERLAWANN_DLLEXPORT int abstract_index_t::lazy_delete<int32_t>(const int32_t& tag);
template POWERLAWANN_DLLEXPORT int abstract_index_t::lazy_delete<uint32_t>(const uint32_t& tag);
template POWERLAWANN_DLLEXPORT int abstract_index_t::lazy_delete<int64_t>(const int64_t& tag);
template POWERLAWANN_DLLEXPORT int abstract_index_t::lazy_delete<uint64_t>(const uint64_t& tag);

template POWERLAWANN_DLLEXPORT void
abstract_index_t::lazy_delete<int32_t>(const std::vector<int32_t>& tags,
                                       std::vector<int32_t>& failed_tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::lazy_delete<uint32_t>(const std::vector<uint32_t>& tags,
                                        std::vector<uint32_t>& failed_tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::lazy_delete<int64_t>(const std::vector<int64_t>& tags,
                                       std::vector<int64_t>& failed_tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::lazy_delete<uint64_t>(const std::vector<uint64_t>& tags,
                                        std::vector<uint64_t>& failed_tags);

template POWERLAWANN_DLLEXPORT void
abstract_index_t::get_active_tags<int32_t>(tsl::robin_set<int32_t>& active_tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::get_active_tags<uint32_t>(tsl::robin_set<uint32_t>& active_tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::get_active_tags<int64_t>(tsl::robin_set<int64_t>& active_tags);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::get_active_tags<uint64_t>(tsl::robin_set<uint64_t>& active_tags);

template POWERLAWANN_DLLEXPORT void
abstract_index_t::set_start_points_at_random<float>(float radius, uint32_t random_seed);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::set_start_points_at_random<uint8_t>(uint8_t radius, uint32_t random_seed);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::set_start_points_at_random<int8_t>(int8_t radius, uint32_t random_seed);

template POWERLAWANN_DLLEXPORT int abstract_index_t::get_vector_by_tag<int32_t, float>(int32_t& tag,
                                                                                       float* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<int32_t, uint8_t>(int32_t& tag, uint8_t* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<int32_t, int8_t>(int32_t& tag, int8_t* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<uint32_t, float>(uint32_t& tag, float* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<uint32_t, uint8_t>(uint32_t& tag, uint8_t* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<uint32_t, int8_t>(uint32_t& tag, int8_t* vec);

template POWERLAWANN_DLLEXPORT int abstract_index_t::get_vector_by_tag<int64_t, float>(int64_t& tag,
                                                                                       float* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<int64_t, uint8_t>(int64_t& tag, uint8_t* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<int64_t, int8_t>(int64_t& tag, int8_t* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<uint64_t, float>(uint64_t& tag, float* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<uint64_t, uint8_t>(uint64_t& tag, uint8_t* vec);
template POWERLAWANN_DLLEXPORT int
abstract_index_t::get_vector_by_tag<uint64_t, int8_t>(uint64_t& tag, int8_t* vec);

template POWERLAWANN_DLLEXPORT void
abstract_index_t::set_universal_label<uint16_t>(const uint16_t label);
template POWERLAWANN_DLLEXPORT void
abstract_index_t::set_universal_label<uint32_t>(const uint32_t label);

} // namespace powerlaw_ann
