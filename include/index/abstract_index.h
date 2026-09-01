#ifndef INDEX_ABSTRACT_INDEX
#define INDEX_ABSTRACT_INDEX

#include "common/parameters.h"
#include "common/types.h"
#include "index/index_build_params.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace powerlaw_ann {
struct consolidation_report_t {
  enum status_code_t { SUCCESS = 0, FAIL = 1, LOCK_FAIL = 2, INCONSISTENT_COUNT_ERROR = 3 };
  status_code_t status_;
  size_t active_points_, max_points_, empty_slots_, slots_released_, delete_set_size_,
      num_calls_to_process_delete_;
  double time_;

  consolidation_report_t(status_code_t status, size_t active_points, size_t max_points,
                         size_t empty_slots, size_t slots_released, size_t delete_set_size,
                         size_t num_calls_to_process_delete, double time_secs)
      : status_(status), active_points_(active_points), max_points_(max_points),
        empty_slots_(empty_slots), slots_released_(slots_released),
        delete_set_size_(delete_set_size),
        num_calls_to_process_delete_(num_calls_to_process_delete), time_(time_secs) {}
};

/*
 * A type-erased interface for interacting with vamana_index_t implementations through
 * std::any-backed data, tag, and label values.
 */
class abstract_index_t {
public:
  abstract_index_t() = default;
  virtual ~abstract_index_t() = default;

  virtual void build(const std::string& data_file, const size_t num_points_to_load,
                     index_filter_params_t& build_params) = 0;

  template <typename data_type, typename tag_type>
  void build(const data_type* data, const size_t num_points_to_load,
             const std::vector<tag_type>& tags);

  virtual void save(const char* filename, bool compact_before_save = false) = 0;

#ifdef EXEC_ENV_OLS
  virtual void load(aligned_file_reader_t& reader, uint32_t num_threads, uint32_t search_l) = 0;
#else
  virtual void load(const char* index_file, uint32_t num_threads, uint32_t search_l) = 0;
#endif

  // For fast_l2 search on optimized layout
  template <typename data_type>
  void search_with_optimized_layout(const data_type* query, size_t K, size_t L, uint32_t* indices);

  // Initialize space for res_vectors before calling.
  template <typename data_type, typename tag_type>
  size_t search_with_tags(const data_type* query, const uint64_t K, const uint32_t L,
                          tag_type* tags, float* distances, std::vector<data_type*>& res_vectors,
                          bool use_filters = false, const std::string filter_label = "");

  // Added search overload that takes L as parameter, so that we
  // can customize L on a per-query basis without tampering with "Parameters"
  // IDtype is either uint32_t or uint64_t
  template <typename data_type, typename IDType>
  std::pair<uint32_t, uint32_t> search(const data_type* query, const size_t K, const uint32_t L,
                                       IDType* indices, float* distances = nullptr);

  // Filter support search
  // index_t is either uint32_t or uint64_t
  template <typename index_t>
  std::pair<uint32_t, uint32_t>
  search_with_filters(const data_type_t& query, const std::string& raw_label, const size_t K,
                      const uint32_t L, index_t* indices, float* distances);

  // insert points with labels, labels should be present for filtered index
  template <typename data_type, typename tag_type, typename label_type>
  int insert_point(const data_type* point, const tag_type tag,
                   const std::vector<label_type>& labels);

  // insert point for unfiltered index build. do not use with filtered index
  template <typename data_type, typename tag_type>
  int insert_point(const data_type* point, const tag_type tag);

  // Delete a point by tag, or return -1 if the point cannot be deleted.
  template <typename tag_type>
  int lazy_delete(const tag_type& tag);

  // Delete a batch of tags and populate failed_tags for tags that cannot be deleted.
  template <typename tag_type>
  void lazy_delete(const std::vector<tag_type>& tags, std::vector<tag_type>& failed_tags);

  template <typename tag_type>
  void get_active_tags(tsl::robin_set<tag_type>& active_tags);

  template <typename data_type>
  void set_start_points_at_random(data_type radius, uint32_t random_seed = 0);

  virtual consolidation_report_t
  consolidate_deletes(const index_write_parameters_t& parameters) = 0;

  virtual void optimize_index_layout() = 0;

  // memory should be allocated for vec before calling this function
  template <typename tag_type, typename data_type>
  int get_vector_by_tag(tag_type& tag, data_type* vec);

  template <typename label_type>
  void set_universal_label(const label_type universal_label);

private:
  virtual void build_impl(const data_type_t& data, const size_t num_points_to_load,
                          tag_vector_t& tags) = 0;
  virtual std::pair<uint32_t, uint32_t> search_impl(const data_type_t& query, const size_t K,
                                                    const uint32_t L, std::any& indices,
                                                    float* distances = nullptr) = 0;
  virtual std::pair<uint32_t, uint32_t> search_with_filters_impl(const data_type_t& query,
                                                                 const std::string& filter_label,
                                                                 const size_t K, const uint32_t L,
                                                                 std::any& indices,
                                                                 float* distances) = 0;
  virtual int insert_point_impl(const data_type_t& data_point, const tag_type_t tag,
                                label_vector_t& labels) = 0;
  virtual int insert_point_impl(const data_type_t& data_point, const tag_type_t tag) = 0;
  virtual int lazy_delete_impl(const tag_type_t& tag) = 0;
  virtual void lazy_delete_impl(tag_vector_t& tags, tag_vector_t& failed_tags) = 0;
  virtual void get_active_tags_impl(tag_robin_set_t& active_tags) = 0;
  virtual void set_start_points_at_random_impl(data_type_t radius, uint32_t random_seed = 0) = 0;
  virtual int get_vector_by_tag_impl(tag_type_t& tag, data_type_t& vec) = 0;
  virtual size_t search_with_tags_impl(const data_type_t& query, const uint64_t K, const uint32_t L,
                                       const tag_type_t& tags, float* distances,
                                       data_vector_t& res_vectors, bool use_filters = false,
                                       const std::string filter_label = "") = 0;
  virtual void search_with_optimized_layout_impl(const data_type_t& query, size_t K, size_t L,
                                                 uint32_t* indices) = 0;
  virtual void set_universal_label_impl(const label_type_t universal_label) = 0;
};
} // namespace powerlaw_ann

#endif // INDEX_ABSTRACT_INDEX
