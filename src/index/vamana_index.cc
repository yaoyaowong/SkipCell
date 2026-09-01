#include "common/arch_compat.h"
#include "common/platform_compat.h"
#include "common/tag_uint128.h"
#include "common/timer.h"
#include "index/index_factory.h"
#include "third/tsl/robin_set.h"

#include <algorithm>
#include <any>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <omp.h>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#include "index/vamana_index.h"

#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace powerlaw_ann {
namespace {

class post_link_observer_reset_guard_t {
public:
  explicit post_link_observer_reset_guard_t(post_link_graph_observer_t*& observer)
      : observer_(observer) {}

  ~post_link_observer_reset_guard_t() { observer_ = nullptr; }

  post_link_observer_reset_guard_t(const post_link_observer_reset_guard_t&) = delete;
  post_link_observer_reset_guard_t& operator=(const post_link_observer_reset_guard_t&) = delete;

private:
  post_link_graph_observer_t*& observer_;
};

} // namespace

// Initialize an index with metric m, load the data of type T with filename
// (bin), and initialize max_points
template <typename T, typename tag_t, typename label_t>
vamana_index_t<T, tag_t, label_t>::vamana_index_t(
    const index_config_t& index_config, std::shared_ptr<abstract_data_store_t<T>> data_store,
    std::unique_ptr<abstract_graph_store_t> graph_store,
    std::shared_ptr<abstract_data_store_t<T>> pq_data_store)
    : dist_metric_(index_config.metric), dim_(index_config.dimension),
      max_points_(index_config.max_points), num_frozen_pts_(index_config.num_frozen_pts),
      dynamic_index_(index_config.dynamic_index), enable_tags_(index_config.enable_tags),
      filtered_index_(index_config.filtered_index), indexing_max_c_(DEFAULT_MAXC),
      query_scratch_(nullptr), pq_dist_(index_config.pq_dist_build), use_opq_(index_config.use_opq),
      num_pq_chunks_(index_config.num_pq_chunks), delete_set_(new tsl::robin_set<uint32_t>),
      conc_consolidate_(index_config.concurrent_consolidate) {
  if (dynamic_index_ && !enable_tags_) {
    throw diskann_exception_t("ERROR: Dynamic Indexing must have tags enabled.", -1, __FUNCSIG__,
                              __FILE__, __LINE__);
  }

  if (pq_dist_) {
    if (dynamic_index_)
      throw diskann_exception_t("ERROR: Dynamic Indexing not supported with PQ distance based "
                                "index construction",
                                -1, __FUNCSIG__, __FILE__, __LINE__);
    if (dist_metric_ == powerlaw_ann::metric_t::INNER_PRODUCT)
      throw diskann_exception_t("ERROR: Inner product metrics not yet supported "
                                "with PQ distance "
                                "base index",
                                -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (dynamic_index_ && num_frozen_pts_ == 0) {
    num_frozen_pts_ = 1;
  }
  // Sanity check. While logically it is correct, max_points = 0 causes
  // downstream problems.
  if (max_points_ == 0) {
    max_points_ = 1;
  }
  const size_t total_internal_points = max_points_ + num_frozen_pts_;

  start_ = (uint32_t) max_points_;

  data_store_ = data_store;
  pq_data_store_ = pq_data_store;
  graph_store_ = std::move(graph_store);

  locks_ = std::vector<non_recursive_mutex_t>(total_internal_points);
  if (enable_tags_) {
    location_to_tag_.reserve(total_internal_points);
    tag_to_location_.reserve(total_internal_points);
  }

  if (dynamic_index_) {
    this->enable_delete(); // enable delete by default for dynamic index
    if (filtered_index_) {
      location_to_labels_.resize(total_internal_points);
    }
  }

  if (index_config.index_write_params != nullptr) {
    indexing_queue_size_ = index_config.index_write_params->search_list_size;
    indexing_range_ = index_config.index_write_params->max_degree;
    indexing_max_c_ = index_config.index_write_params->max_occlusion_size;
    indexing_alpha_ = index_config.index_write_params->alpha;
    filter_indexing_queue_size_ = index_config.index_write_params->filter_list_size;
    indexing_threads_ = index_config.index_write_params->num_threads;
    saturate_graph_ = index_config.index_write_params->saturate_graph;

    if (index_config.index_search_params != nullptr) {
      uint32_t num_scratch_spaces =
          index_config.index_search_params->num_search_threads + indexing_threads_;
      initialize_query_scratch(
          num_scratch_spaces, index_config.index_search_params->initial_search_list_size,
          indexing_queue_size_, indexing_range_, indexing_max_c_, data_store_->get_dims());
    }
  }
}

template <typename T, typename tag_t, typename label_t>
vamana_index_t<T, tag_t, label_t>::vamana_index_t(
    metric_t m, const size_t dim, const size_t max_points,
    const std::shared_ptr<index_write_parameters_t> index_parameters,
    const std::shared_ptr<index_search_params_t> index_search_params, const size_t num_frozen_pts,
    const bool dynamic_index, const bool enable_tags, const bool concurrent_consolidate,
    const bool pq_dist_build, const size_t num_pq_chunks, const bool use_opq,
    const bool filtered_index)
    : vamana_index_t(
          index_config_builder_t()
              .with_metric(m)
              .with_dimension(dim)
              .with_max_points(max_points)
              .with_index_write_params(index_parameters)
              .with_index_search_params(index_search_params)
              .with_num_frozen_pts(num_frozen_pts)
              .is_dynamic_index(dynamic_index)
              .is_enable_tags(enable_tags)
              .is_concurrent_consolidate(concurrent_consolidate)
              .is_pq_dist_build(pq_dist_build)
              .with_num_pq_chunks(num_pq_chunks)
              .is_use_opq(use_opq)
              .is_filtered(filtered_index)
              .with_data_type(diskann_type_to_name<T>())
              .build(),
          index_factory_t::construct_datastore<T>(
              data_store_strategy_t::MEMORY,
              (max_points == 0 ? (size_t) 1 : max_points) +
                  (dynamic_index && num_frozen_pts == 0 ? (size_t) 1 : num_frozen_pts),
              dim, m),
          index_factory_t::construct_graphstore(
              graph_store_strategy_t::MEMORY,
              (max_points == 0 ? (size_t) 1 : max_points) +
                  (dynamic_index && num_frozen_pts == 0 ? (size_t) 1 : num_frozen_pts),
              (size_t) ((index_parameters == nullptr ? 0 : index_parameters->max_degree) *
                        defaults::GRAPH_SLACK_FACTOR * 1.05))) {
  if (pq_dist_) {
    pq_data_store_ = index_factory_t::construct_pq_datastore<T>(
        data_store_strategy_t::MEMORY, max_points + num_frozen_pts, dim, m, num_pq_chunks, use_opq);
  } else {
    pq_data_store_ = data_store_;
  }
}

template <typename T, typename tag_t, typename label_t>
vamana_index_t<T, tag_t, label_t>::~vamana_index_t() {
  // Ensure that no other activity is happening before dtor()
  std::unique_lock<std::shared_timed_mutex> ul(update_lock_);
  std::unique_lock<std::shared_timed_mutex> cl(consolidate_lock_);
  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  std::unique_lock<std::shared_timed_mutex> dl(delete_lock_);

  for (auto& lock : locks_) {
    lock_guard_t lg(lock);
  }

  if (opt_graph_ != nullptr) {
    delete[] opt_graph_;
  }

  if (!query_scratch_.empty()) {
    scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
    manager.destroy();
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::enable_vamana_build_stats() {
  vamana_build_stats_ = std::make_unique<vamana_build_stats_t>();
  vamana_build_stats_->reset(max_points_ + num_frozen_pts_);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::set_rcni_prune_observer(rcni_prune_observer_t* observer) {
  rcni_prune_observer_ = observer;
  if (rcni_prune_observer_ != nullptr) {
    rcni_prune_observer_->prepare(max_points_);
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::set_post_link_graph_observer(
    post_link_graph_observer_t* observer) {
  if (observer != nullptr &&
      (!std::is_same_v<T, float> || dist_metric_ != powerlaw_ann::metric_t::L2 || dynamic_index_ ||
       filtered_index_)) {
    throw std::invalid_argument(
        "Post-link graph observation requires a static unfiltered float32 L2 Vamana index");
  }
  post_link_graph_observer_ = observer;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::dump_vamana_build_stats(const std::string& prefix) const {
  if (!vamana_build_stats_ || !vamana_build_stats_->active()) {
    throw powerlaw_ann::diskann_exception_t("Vamana build stats are not enabled", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
  }

  std::vector<uint32_t> final_degrees;
  final_degrees.reserve(nd_ + num_frozen_pts_);
  for (uint32_t node_id = 0; node_id < (uint32_t) (nd_ + num_frozen_pts_); ++node_id) {
    final_degrees.push_back((uint32_t) graph_store_->get_neighbours((location_t) node_id).size());
  }
  vamana_build_stats_->dump(prefix, final_degrees, start_, num_frozen_pts_);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::initialize_query_scratch(uint32_t num_threads,
                                                                 uint32_t search_l,
                                                                 uint32_t indexing_l, uint32_t r,
                                                                 uint32_t maxc, size_t dim) {
  for (uint32_t i = 0; i < num_threads; i++) {
    auto scratch = new in_mem_query_scratch_t<T>(search_l, indexing_l, r, maxc, dim,
                                                 data_store_->get_aligned_dim(),
                                                 data_store_->get_alignment_factor(), pq_dist_);
    query_scratch_.push(scratch);
  }
}

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::save_tags(std::string tags_file) {
  if (!enable_tags_) {
    powerlaw_ann::cout << "Not saving tags as they are not enabled." << std::endl;
    return 0;
  }

  size_t tag_bytes_written;
  tag_t* tag_data = new tag_t[nd_ + num_frozen_pts_];
  for (uint32_t i = 0; i < nd_; i++) {
    tag_t tag;
    if (location_to_tag_.try_get(i, tag)) {
      tag_data[i] = tag;
    } else {
      // catering to future when tag_t can be any type.
      std::memset((char*) &tag_data[i], 0, sizeof(tag_t));
    }
  }
  if (num_frozen_pts_ > 0) {
    std::memset((char*) &tag_data[start_], 0, sizeof(tag_t) * num_frozen_pts_);
  }
  try {
    tag_bytes_written = save_bin<tag_t>(tags_file, tag_data, nd_ + num_frozen_pts_, 1);
  } catch (std::system_error& e) {
    throw file_exception_t(tags_file, e, __FUNCSIG__, __FILE__, __LINE__);
  }
  delete[] tag_data;
  return tag_bytes_written;
}

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::save_data(std::string data_file) {
  // Note: at this point, either nd_ == max_points_ or any frozen points have
  // been temporarily moved to nd_, so nd_ + num_frozen_pts_ is the valid
  // location limit.
  return data_store_->save(data_file, (location_t) (nd_ + num_frozen_pts_));
}

// save the graph index on a file as an adjacency list. For each point,
// first store the number of neighbors, and then the neighbor list (each as
// 4 byte uint32_t)
template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::save_graph(std::string graph_file) {
  return graph_store_->store(graph_file, nd_ + num_frozen_pts_, num_frozen_pts_, start_);
}

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::save_delete_list(const std::string& filename) {
  if (delete_set_->size() == 0) {
    return 0;
  }
  std::unique_ptr<uint32_t[]> delete_list = std::make_unique<uint32_t[]>(delete_set_->size());
  uint32_t i = 0;
  for (auto& del : *delete_set_) {
    delete_list[i++] = del;
  }
  return save_bin<uint32_t>(filename, delete_list.get(), delete_set_->size(), 1);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::save(const char* filename, bool compact_before_save) {
  powerlaw_ann::timer_t timer;

  std::unique_lock<std::shared_timed_mutex> ul(update_lock_);
  std::unique_lock<std::shared_timed_mutex> cl(consolidate_lock_);
  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  std::unique_lock<std::shared_timed_mutex> dl(delete_lock_);

  if (compact_before_save) {
    compact_data();
    compact_frozen_point();
  } else {
    if (!data_compacted_) {
      throw diskann_exception_t(
          "vamana_index_t save for non-compacted index is not yet implemented", -1, __FUNCSIG__,
          __FILE__, __LINE__);
    }
  }

  if (!save_as_one_file_) {
    if (filtered_index_) {
      if (label_to_start_id_.size() > 0) {
        std::ofstream medoid_writer(std::string(filename) + "labels_to_medoids_.txt");
        if (medoid_writer.fail()) {
          throw powerlaw_ann::diskann_exception_t(std::string("Failed to open file ") + filename,
                                                  -1);
        }
        for (auto iter : label_to_start_id_) {
          medoid_writer << iter.first << ", " << iter.second << std::endl;
        }
        medoid_writer.close();
      }

      if (use_universal_label_) {
        std::ofstream universal_label_writer(std::string(filename) + "universal_label_.txt");
        assert(universal_label_writer.is_open());
        universal_label_writer << universal_label_ << std::endl;
        universal_label_writer.close();
      }

      if (location_to_labels_.size() > 0) {
        std::ofstream label_writer(std::string(filename) + "labels_.txt");
        assert(label_writer.is_open());
        for (uint32_t i = 0; i < nd_ + num_frozen_pts_; i++) {
          for (uint32_t j = 0; j + 1 < location_to_labels_[i].size(); j++) {
            label_writer << location_to_labels_[i][j] << ",";
          }
          if (location_to_labels_[i].size() != 0)
            label_writer << location_to_labels_[i][location_to_labels_[i].size() - 1];

          label_writer << std::endl;
        }
        label_writer.close();

        // write compacted raw_labels if data hence location_to_labels_ was also compacted
        if (compact_before_save && dynamic_index_) {
          label_map_ = load_label_map(std::string(filename) + "labels_map_.txt");
          std::unordered_map<label_t, std::string> mapped_to_raw_labels;
          // invert label map
          for (const auto& [key, value] : label_map_) {
            mapped_to_raw_labels.insert({value, key});
          }

          // write updated labels
          std::ofstream raw_label_writer(std::string(filename) + "raw_labels_.txt");
          assert(raw_label_writer.is_open());
          for (uint32_t i = 0; i < nd_ + num_frozen_pts_; i++) {
            for (uint32_t j = 0; j + 1 < location_to_labels_[i].size(); j++) {
              raw_label_writer << mapped_to_raw_labels[location_to_labels_[i][j]] << ",";
            }
            if (location_to_labels_[i].size() != 0)
              raw_label_writer << mapped_to_raw_labels
                      [location_to_labels_[i][location_to_labels_[i].size() - 1]];

            raw_label_writer << std::endl;
          }
          raw_label_writer.close();
        }
      }
    }

    std::string graph_file = std::string(filename);
    std::string tags_file = std::string(filename) + ".tags";
    std::string data_file = std::string(filename) + ".data";
    std::string delete_list_file = std::string(filename) + ".del";

    // Because the save_* functions use append mode, ensure that
    // the files are deleted before save. Ideally, we should check
    // the error code for delete_file, but will ignore now because
    // delete should succeed if save will succeed.
    delete_file(graph_file);
    save_graph(graph_file);
    delete_file(data_file);
    save_data(data_file);
    delete_file(tags_file);
    save_tags(tags_file);
    delete_file(delete_list_file);
    save_delete_list(delete_list_file);
  } else {
    powerlaw_ann::cout << "Save index in a single file currently not supported. "
                          "Not saving the index."
                       << std::endl;
  }

  // If frozen points were temporarily compacted to nd_, move back to
  // max_points_.
  reposition_frozen_point_to_end();

  powerlaw_ann::cout << "Time taken for save: " << timer.elapsed() / 1000000.0 << "s." << std::endl;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::load_tags(aligned_file_reader_t& reader) {
#else
template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::load_tags(const std::string tag_filename) {
  if (enable_tags_ && !file_exists(tag_filename)) {
    powerlaw_ann::cerr << "Tag file " << tag_filename << " does not exist!" << std::endl;
    throw powerlaw_ann::diskann_exception_t("Tag file " + tag_filename + " does not exist!", -1,
                                            __FUNCSIG__, __FILE__, __LINE__);
  }
#endif
  if (!enable_tags_) {
    powerlaw_ann::cout << "Tags not loaded as tags not enabled." << std::endl;
    return 0;
  }

  size_t file_dim, file_num_points;
  tag_t* tag_data;
#ifdef EXEC_ENV_OLS
  load_bin<tag_t>(reader, tag_data, file_num_points, file_dim);
#else
  load_bin<tag_t>(std::string(tag_filename), tag_data, file_num_points, file_dim);
#endif

  if (file_dim != 1) {
    std::stringstream stream;
    stream << "ERROR: Found " << file_dim << " dimensions for tags,"
           << "but tag file must have 1 dimension." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    delete[] tag_data;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  const size_t num_data_points = file_num_points - num_frozen_pts_;
  location_to_tag_.reserve(num_data_points);
  tag_to_location_.reserve(num_data_points);
  for (uint32_t i = 0; i < (uint32_t) num_data_points; i++) {
    tag_t tag = *(tag_data + i);
    if (delete_set_->find(i) == delete_set_->end()) {
      location_to_tag_.set(i, tag);
      tag_to_location_[tag] = i;
    }
  }
  powerlaw_ann::cout << "Tags loaded." << std::endl;
  delete[] tag_data;
  return file_num_points;
}

template <typename T, typename tag_t, typename label_t>
#ifdef EXEC_ENV_OLS
size_t vamana_index_t<T, tag_t, label_t>::load_data(aligned_file_reader_t& reader) {
#else
size_t vamana_index_t<T, tag_t, label_t>::load_data(std::string filename) {
#endif
  size_t file_dim, file_num_points;
#ifdef EXEC_ENV_OLS
  powerlaw_ann::get_bin_metadata(reader, file_num_points, file_dim);
#else
  if (!file_exists(filename)) {
    std::stringstream stream;
    stream << "ERROR: data file " << filename << " does not exist." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
  powerlaw_ann::get_bin_metadata(filename, file_num_points, file_dim);
#endif

  // since we are loading a new dataset, empty_slots_ must be cleared
  empty_slots_.clear();

  if (file_dim != dim_) {
    std::stringstream stream;
    stream << "ERROR: Driver requests loading " << dim_ << " dimension," << "but file has "
           << file_dim << " dimension." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (file_num_points > max_points_ + num_frozen_pts_) {
    // update and tag lock acquired in load() before calling load_data
    resize(file_num_points - num_frozen_pts_);
  }

#ifdef EXEC_ENV_OLS
  // REFACTOR TODO: Must figure out how to support aligned reader in a clean
  // manner.
  copy_aligned_data_from_file<T>(reader, data_, file_num_points, file_dim,
                                 data_store_->get_aligned_dim());
#else
  data_store_->load(filename); // offset == 0.
#endif
  return file_num_points;
}

#ifdef EXEC_ENV_OLS
template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::load_delete_set(aligned_file_reader_t& reader) {
#else
template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::load_delete_set(const std::string& filename) {
#endif
  std::unique_ptr<uint32_t[]> delete_list;
  size_t npts, ndim;

#ifdef EXEC_ENV_OLS
  powerlaw_ann::load_bin<uint32_t>(reader, delete_list, npts, ndim);
#else
  powerlaw_ann::load_bin<uint32_t>(filename, delete_list, npts, ndim);
#endif
  assert(ndim == 1);
  for (uint32_t i = 0; i < npts; i++) {
    delete_set_->insert(delete_list[i]);
  }
  return npts;
}

// load the index from file and update the max_degree, cur (navigating
// node loc), and final_graph_ (adjacency list)
template <typename T, typename tag_t, typename label_t>
#ifdef EXEC_ENV_OLS
void vamana_index_t<T, tag_t, label_t>::load(aligned_file_reader_t& reader, uint32_t num_threads,
                                             uint32_t search_l) {
#else
void vamana_index_t<T, tag_t, label_t>::load(const char* filename, uint32_t num_threads,
                                             uint32_t search_l) {
#endif
  std::unique_lock<std::shared_timed_mutex> ul(update_lock_);
  std::unique_lock<std::shared_timed_mutex> cl(consolidate_lock_);
  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  std::unique_lock<std::shared_timed_mutex> dl(delete_lock_);

  has_built_ = true;

  size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0, label_num_pts = 0;

  std::string mem_index_file(filename);
  std::string labels_file = mem_index_file + "labels_.txt";
  std::string labels_to_medoids = mem_index_file + "labels_to_medoids_.txt";
  std::string labels_map_file = mem_index_file + "labels_map_.txt";

  if (!save_as_one_file_) {
    // For DLVS Store, we will not support saving the index in multiple
    // files.
#ifndef EXEC_ENV_OLS
    std::string data_file = std::string(filename) + ".data";
    std::string tags_file = std::string(filename) + ".tags";
    std::string delete_set_file = std::string(filename) + ".del";
    std::string graph_file = std::string(filename);
    data_file_num_pts = load_data(data_file);
    if (file_exists(delete_set_file)) {
      load_delete_set(delete_set_file);
    }
    if (enable_tags_) {
      tags_file_num_pts = load_tags(tags_file);
    }
    graph_num_pts = load_graph(graph_file, data_file_num_pts);
#endif
  } else {
    powerlaw_ann::cout << "Single index file saving/loading support not yet "
                          "enabled. Not loading the index."
                       << std::endl;
    return;
  }

  if (data_file_num_pts != graph_num_pts ||
      (data_file_num_pts != tags_file_num_pts && enable_tags_)) {
    std::stringstream stream;
    stream << "ERROR: When loading index, loaded " << data_file_num_pts << " points from datafile, "
           << graph_num_pts << " from graph, and " << tags_file_num_pts
           << " tags, with num_frozen_pts being set to " << num_frozen_pts_ << " in constructor."
           << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (file_exists(labels_file)) {
    label_map_ = load_label_map(labels_map_file);
    parse_label_file(labels_file, label_num_pts);
    assert(label_num_pts == data_file_num_pts - num_frozen_pts_);
    if (file_exists(labels_to_medoids)) {
      std::ifstream medoid_stream(labels_to_medoids);
      std::string line, token;
      label_to_start_id_.clear();

      while (std::getline(medoid_stream, line)) {
        std::istringstream iss(line);
        uint32_t cnt = 0;
        uint32_t medoid = 0;
        label_t label;
        while (std::getline(iss, token, ',')) {
          token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
          token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
          label_t token_as_num = (label_t) std::stoul(token);
          if (cnt == 0)
            label = token_as_num;
          else
            medoid = token_as_num;
          cnt++;
        }
        label_to_start_id_[label] = medoid;
      }
    }

    std::string universal_label_file(filename);
    universal_label_file += "universal_label_.txt";
    if (file_exists(universal_label_file)) {
      std::ifstream universal_label_reader(universal_label_file);
      universal_label_reader >> universal_label_;
      use_universal_label_ = true;
      universal_label_reader.close();
    }
  }

  nd_ = data_file_num_pts - num_frozen_pts_;
  empty_slots_.clear();
  empty_slots_.reserve(max_points_);
  for (auto i = nd_; i < max_points_; i++) {
    empty_slots_.insert((uint32_t) i);
  }

  reposition_frozen_point_to_end();
  powerlaw_ann::cout << "Num frozen points:" << num_frozen_pts_ << " nd_: " << nd_
                     << " start_: " << start_
                     << " size(location_to_tag_): " << location_to_tag_.size()
                     << " size(tag_to_location_):" << tag_to_location_.size()
                     << " Max points: " << max_points_ << std::endl;

  // For incremental index, query_scratch_ is initialized in the constructor.
  // For the bulk index, the params required to initialize query_scratch_
  // are known only at load time, hence this check and the call to
  // initialize_q_s().
  if (query_scratch_.size() == 0) {
    initialize_query_scratch(num_threads, search_l, search_l,
                             (uint32_t) graph_store_->get_max_range_of_graph(), indexing_max_c_,
                             dim_);
  }
}

#ifndef EXEC_ENV_OLS
template <typename T, typename tag_t, typename label_t>
size_t
vamana_index_t<T, tag_t, label_t>::get_graph_num_frozen_points(const std::string& graph_file) {
  size_t expected_file_size;
  uint32_t max_observed_degree, start;
  size_t file_frozen_pts;

  std::ifstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);

  in.open(graph_file, std::ios::binary);
  in.read((char*) &expected_file_size, sizeof(size_t));
  in.read((char*) &max_observed_degree, sizeof(uint32_t));
  in.read((char*) &start, sizeof(uint32_t));
  in.read((char*) &file_frozen_pts, sizeof(size_t));

  return file_frozen_pts;
}
#endif

#ifdef EXEC_ENV_OLS
template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::load_graph(aligned_file_reader_t& reader,
                                                     size_t expected_num_points) {
#else

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::load_graph(std::string filename,
                                                     size_t expected_num_points) {
#endif
  auto res = graph_store_->load(filename, expected_num_points);
  start_ = std::get<1>(res);
  num_frozen_pts_ = std::get<2>(res);
  return std::get<0>(res);
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::get_vector_by_tag_impl(tag_type_t& tag, data_type_t& vec) {
  try {
    tag_t tag_val = std::any_cast<tag_t>(tag);
    T* vec_val = std::any_cast<T*>(vec);
    return this->get_vector_by_tag(tag_val, vec_val);
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t("Error: bad any cast while performing get_vector_by_tags_impl() " +
                                  std::string(e.what()),
                              -1);
  } catch (const std::exception& e) {
    throw diskann_exception_t("Error: " + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::get_vector_by_tag(tag_t& tag, T* vec) {
  std::shared_lock<std::shared_timed_mutex> lock(tag_lock_);
  if (tag_to_location_.find(tag) == tag_to_location_.end()) {
    powerlaw_ann::cout << "Tag " << get_tag_string(tag) << " does not exist" << std::endl;
    return -1;
  }

  location_t location = tag_to_location_[tag];
  data_store_->get_vector(location, vec);

  return 0;
}

template <typename T, typename tag_t, typename label_t>
uint32_t vamana_index_t<T, tag_t, label_t>::calculate_entry_point() {
  // REFACTOR TODO: This function does not support multi-threaded calculation of medoid.
  // Must revisit if perf is a concern.
  return data_store_->calculate_medoid();
}

template <typename T, typename tag_t, typename label_t>
std::vector<uint32_t> vamana_index_t<T, tag_t, label_t>::get_init_ids() {
  std::vector<uint32_t> init_ids;
  init_ids.reserve(1 + num_frozen_pts_);

  init_ids.emplace_back(start_);

  for (uint32_t frozen = (uint32_t) max_points_; frozen < max_points_ + num_frozen_pts_; frozen++) {
    if (frozen != start_) {
      init_ids.emplace_back(frozen);
    }
  }

  return init_ids;
}

// Find common filter between a node's labels and a given set of labels, while
// taking into account universal label.
// Note: incoming_labels should be sorted.
template <typename T, typename tag_t, typename label_t>
bool vamana_index_t<T, tag_t, label_t>::detect_common_filters(
    uint32_t point_id, bool search_invocation, const std::vector<label_t>& incoming_labels) {
  auto& curr_node_labels = location_to_labels_[point_id];
  // Check for intersection between incoming_labels and curr_node_labels
  // using two-pointer approach (both vectors are sorted)
  auto it_inc = incoming_labels.begin();
  auto it_curr = curr_node_labels.begin();

  while (it_inc != incoming_labels.end() && it_curr != curr_node_labels.end()) {
    if (*it_inc < *it_curr) {
      ++it_inc;
    } else if (*it_curr < *it_inc) {
      ++it_curr;
    } else {
      // common label found
      return true;
    }
  }
  // intersection empty; proceed to check the universal label logic

  if (use_universal_label_) {
    if (!search_invocation) {
      if (std::find(incoming_labels.begin(), incoming_labels.end(), universal_label_) !=
              incoming_labels.end() ||
          std::find(curr_node_labels.begin(), curr_node_labels.end(), universal_label_) !=
              curr_node_labels.end())
        return true;
    } else {
      if (std::find(curr_node_labels.begin(), curr_node_labels.end(), universal_label_) !=
          curr_node_labels.end())
        return true;
    }
  }
  return false;
}

template <typename T, typename tag_t, typename label_t>
std::pair<uint32_t, uint32_t> vamana_index_t<T, tag_t, label_t>::iterate_to_fixed_point(
    in_mem_query_scratch_t<T>* scratch, const uint32_t l_size,
    const std::vector<uint32_t>& init_ids, bool use_filter,
    const std::vector<label_t>& filter_labels, bool search_invocation) {
  std::vector<neighbor_t>& expanded_nodes = scratch->pool();
  neighbor_priority_queue_t& best_l_nodes = scratch->best_l_nodes();
  best_l_nodes.reserve(l_size);
  tsl::robin_set<uint32_t>& inserted_into_pool_rs = scratch->inserted_into_pool_rs();
  boost::dynamic_bitset<>& inserted_into_pool_bs = scratch->inserted_into_pool_bs();
  std::vector<uint32_t>& id_scratch = scratch->id_scratch();
  std::vector<float>& dist_scratch = scratch->dist_scratch();
  assert(id_scratch.size() == 0);

  T* aligned_query = scratch->aligned_query();

  pq_data_store_->preprocess_query(aligned_query, scratch);

  if (expanded_nodes.size() > 0 || id_scratch.size() > 0) {
    throw diskann_exception_t("ERROR: Clear scratch space before passing.", -1, __FUNCSIG__,
                              __FILE__, __LINE__);
  }

  // Decide whether to use bitset or robin set to mark visited nodes
  auto total_num_points = max_points_ + num_frozen_pts_;
  bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

  if (fast_iterate) {
    if (inserted_into_pool_bs.size() < total_num_points) {
      // hopefully using 2X will reduce the number of allocations.
      auto resize_size = 2 * total_num_points > MAX_POINTS_FOR_USING_BITSET
                             ? MAX_POINTS_FOR_USING_BITSET
                             : 2 * total_num_points;
      inserted_into_pool_bs.resize(resize_size);
    }
  }

  // Lambda to determine if a node has been visited
  auto is_not_visited = [fast_iterate, &inserted_into_pool_bs,
                         &inserted_into_pool_rs](const uint32_t id) {
    return fast_iterate ? inserted_into_pool_bs[id] == 0
                        : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
  };

  // Lambda to batch compute query<-> node distances in PQ space
  auto compute_dists = [this, scratch](const std::vector<uint32_t>& ids,
                                       std::vector<float>& dists_out) {
    pq_data_store_->get_distance(scratch->aligned_query(), ids, dists_out, scratch);
  };

  // Initialize the candidate pool with starting points
  for (auto id : init_ids) {
    if (id >= max_points_ + num_frozen_pts_) {
      powerlaw_ann::cerr << "Out of range loc found as an edge : " << id << std::endl;
      throw powerlaw_ann::diskann_exception_t(std::string("Wrong loc") + std::to_string(id), -1,
                                              __FUNCSIG__, __FILE__, __LINE__);
    }

    if (use_filter) {
      if (!detect_common_filters(id, search_invocation, filter_labels))
        continue;
    }

    if (is_not_visited(id)) {
      if (fast_iterate) {
        inserted_into_pool_bs[id] = 1;
      } else {
        inserted_into_pool_rs.insert(id);
      }

      float distance;
      uint32_t ids[] = {id};
      float distances[] = {std::numeric_limits<float>::max()};
      pq_data_store_->get_distance(aligned_query, ids, 1, distances, scratch);
      distance = distances[0];

      neighbor_t nn = neighbor_t(id, distance);
      best_l_nodes.insert(nn);
    }
  }

  uint32_t hops = 0;
  uint32_t cmps = 0;

  while (best_l_nodes.has_unexpanded_node()) {
    auto nbr = best_l_nodes.closest_unexpanded();
    auto n = nbr.id;

    // Add node to expanded nodes to create pool for prune later
    if (!search_invocation) {
      if (!use_filter) {
        expanded_nodes.emplace_back(nbr);
      } else { // in filter based indexing, the same point might invoke
        // multiple iterate_to_fixed_points, so need to be careful
        // not to add the same item to pool multiple times.
        if (std::find(expanded_nodes.begin(), expanded_nodes.end(), nbr) == expanded_nodes.end()) {
          expanded_nodes.emplace_back(nbr);
        }
      }
    }

    // Find which of the nodes in des have not been visited before
    id_scratch.clear();
    dist_scratch.clear();
    if (dynamic_index_) {
      lock_guard_t guard(locks_[n]);
      for (auto id : graph_store_->get_neighbours(n)) {
        assert(id < max_points_ + num_frozen_pts_);

        if (use_filter) {
          // NOTE: NEED TO CHECK IF THIS CORRECT WITH NEW LOCKS.
          if (!detect_common_filters(id, search_invocation, filter_labels))
            continue;
        }

        if (is_not_visited(id)) {
          id_scratch.push_back(id);
        }
      }
    } else {
      locks_[n].lock();
      auto nbrs = graph_store_->get_neighbours(n);
      locks_[n].unlock();
      for (auto id : nbrs) {
        assert(id < max_points_ + num_frozen_pts_);

        if (use_filter) {
          // NOTE: NEED TO CHECK IF THIS CORRECT WITH NEW LOCKS.
          if (!detect_common_filters(id, search_invocation, filter_labels))
            continue;
        }

        if (is_not_visited(id)) {
          id_scratch.push_back(id);
        }
      }
    }

    // Mark nodes visited
    for (auto id : id_scratch) {
      if (fast_iterate) {
        inserted_into_pool_bs[id] = 1;
      } else {
        inserted_into_pool_rs.insert(id);
      }
    }

    assert(dist_scratch.capacity() >= id_scratch.size());
    compute_dists(id_scratch, dist_scratch);
    cmps += (uint32_t) id_scratch.size();

    // Insert <id, dist> pairs into the pool of candidates
    for (size_t m = 0; m < id_scratch.size(); ++m) {
      best_l_nodes.insert(neighbor_t(id_scratch[m], dist_scratch[m]));
    }
  }
  return std::make_pair(hops, cmps);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::search_for_point_and_prune(
    int location, uint32_t l_index, std::vector<uint32_t>& pruned_list,
    in_mem_query_scratch_t<T>* scratch, bool use_filter, uint32_t filtered_l_index) {
  search_for_point_and_prune_impl<false>(location, l_index, pruned_list, scratch, use_filter,
                                         filtered_l_index);
}

template <typename T, typename tag_t, typename label_t>
template <bool observe_rcni>
void vamana_index_t<T, tag_t, label_t>::search_for_point_and_prune_impl(
    int location, uint32_t l_index, std::vector<uint32_t>& pruned_list,
    in_mem_query_scratch_t<T>* scratch, bool use_filter, uint32_t filtered_l_index) {
  const std::vector<uint32_t> init_ids = get_init_ids();
  const std::vector<label_t> unused_filter_label;

  if (!use_filter) {
    data_store_->get_vector(location, scratch->aligned_query());
    iterate_to_fixed_point(scratch, l_index, init_ids, false, unused_filter_label, false);
  } else {
    std::shared_lock<std::shared_timed_mutex> tl(tag_lock_, std::defer_lock);
    if (dynamic_index_)
      tl.lock();
    std::vector<uint32_t> filter_specific_start_nodes;
    for (auto& x : location_to_labels_[location])
      filter_specific_start_nodes.emplace_back(label_to_start_id_[x]);

    if (dynamic_index_)
      tl.unlock();

    data_store_->get_vector(location, scratch->aligned_query());
    iterate_to_fixed_point(scratch, filtered_l_index, filter_specific_start_nodes, true,
                           location_to_labels_[location], false);

    // combine candidate pools obtained with filter and unfiltered criteria.
    std::set<neighbor_t> best_candidate_pool;
    for (auto filtered_neighbor : scratch->pool()) {
      best_candidate_pool.insert(filtered_neighbor);
    }

    // clear scratch for finding unfiltered candidates
    scratch->clear();

    data_store_->get_vector(location, scratch->aligned_query());
    iterate_to_fixed_point(scratch, l_index, init_ids, false, unused_filter_label, false);

    for (auto unfiltered_neighbour : scratch->pool()) {
      // insert if this neighbour is not already in best_candidate_pool
      if (best_candidate_pool.find(unfiltered_neighbour) == best_candidate_pool.end()) {
        best_candidate_pool.insert(unfiltered_neighbour);
      }
    }

    scratch->pool().clear();
    std::copy(best_candidate_pool.begin(), best_candidate_pool.end(),
              std::back_inserter(scratch->pool()));
  }

  auto& pool = scratch->pool();

  for (uint32_t i = 0; i < pool.size(); i++) {
    if (pool[i].id == (uint32_t) location) {
      pool.erase(pool.begin() + i);
      i--;
    }
  }

  if (vamana_build_stats_ && vamana_build_stats_->active()) {
    std::vector<uint32_t> candidate_ids;
    candidate_ids.reserve(pool.size());
    for (const auto& candidate : pool) {
      candidate_ids.push_back(candidate.id);
    }
    vamana_build_stats_->record_candidate_pool((uint32_t) location, candidate_ids);
  }

  if (pruned_list.size() > 0) {
    throw powerlaw_ann::diskann_exception_t("ERROR: non-empty pruned_list passed", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
  }

  if constexpr (observe_rcni) {
    prune_neighbors_impl<true>(location, pool, indexing_range_, indexing_max_c_, indexing_alpha_,
                               pruned_list, scratch);
  } else {
    prune_neighbors(location, pool, pruned_list, scratch);
  }

  assert(!pruned_list.empty());
  assert(graph_store_->get_total_points() == max_points_ + num_frozen_pts_);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::occlude_list(
    const uint32_t location, std::vector<neighbor_t>& pool, const float alpha,
    const uint32_t degree, const uint32_t maxc, std::vector<uint32_t>& result,
    in_mem_query_scratch_t<T>* scratch, const tsl::robin_set<uint32_t>* const delete_set_ptr) {
  occlude_list_impl<false>(location, pool, alpha, degree, maxc, result, scratch, nullptr,
                           delete_set_ptr);
}

template <typename T, typename tag_t, typename label_t>
template <bool observe_rcni>
void vamana_index_t<T, tag_t, label_t>::occlude_list_impl(
    const uint32_t location, std::vector<neighbor_t>& pool, const float alpha,
    const uint32_t degree, const uint32_t maxc, std::vector<uint32_t>& result,
    in_mem_query_scratch_t<T>* scratch, std::vector<rcni_prune_event_t>* events,
    const tsl::robin_set<uint32_t>* delete_set_ptr) {
  if (pool.size() == 0)
    return;

  // Truncate pool at maxc and initialize scratch spaces
  assert(std::is_sorted(pool.begin(), pool.end()));
  assert(result.size() == 0);
  if (pool.size() > maxc)
    pool.resize(maxc);
  std::vector<float>& occlude_factor = scratch->occlude_factor();
  // occlude_list can be called with the same scratch more than once by
  // search_for_point_and_add_link through inter_insert.
  occlude_factor.clear();
  // Initialize occlude_factor to pool.size() many 0.0f values for correctness
  occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

  std::vector<rcni_causal_witness_t> causal_witnesses;
  if constexpr (observe_rcni) {
    assert(events != nullptr);
    causal_witnesses.resize(pool.size());
  }

  float cur_alpha = 1;
  while (cur_alpha <= alpha && result.size() < degree) {
    // used for MIPS, where we store a value of eps in cur_alpha to
    // denote pruned out entries which we can skip in later rounds.
    float eps = cur_alpha + 0.01f;

    for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter) {
      if (occlude_factor[iter - pool.begin()] > cur_alpha) {
        continue;
      }
      // Set the entry to float::max so that is not considered again
      occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
      // Add the entry to the result if its not been deleted, and doesn't
      // add a self loop
      if (delete_set_ptr == nullptr || delete_set_ptr->find(iter->id) == delete_set_ptr->end()) {
        if (iter->id != location) {
          result.push_back(iter->id);
        }
      }

      // Update occlude factor for points from iter+1 to pool.end()
      for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
        auto t = iter2 - pool.begin();
        if (occlude_factor[t] > alpha)
          continue;

        bool prune_allowed = true;
        if (filtered_index_) {
          uint32_t a = iter->id;
          uint32_t b = iter2->id;
          if (location_to_labels_.size() < b || location_to_labels_.size() < a)
            continue;
          for (auto& x : location_to_labels_[b]) {
            if (std::find(location_to_labels_[a].begin(), location_to_labels_[a].end(), x) ==
                location_to_labels_[a].end()) {
              prune_allowed = false;
            }
            if (!prune_allowed)
              break;
          }
        }
        if (!prune_allowed)
          continue;

        float djk = data_store_->get_distance(iter2->id, iter->id);
        if (dist_metric_ == powerlaw_ann::metric_t::L2 ||
            dist_metric_ == powerlaw_ann::metric_t::COSINE) {
          const float candidate_factor =
              (djk == 0) ? std::numeric_limits<float>::max() : iter2->distance / djk;
          if constexpr (observe_rcni) {
            update_rcni_causal_witness(causal_witnesses[t], iter->id, djk, candidate_factor);
          }
          occlude_factor[t] = std::max(occlude_factor[t], candidate_factor);
        } else if (dist_metric_ == powerlaw_ann::metric_t::INNER_PRODUCT) {
          // Improvization for flipping max and min dist for MIPS
          float x = -iter2->distance;
          float y = -djk;
          if (y > cur_alpha * x) {
            occlude_factor[t] = std::max(occlude_factor[t], eps);
          }
        }
      }
    }
    cur_alpha *= 1.2f;
  }

  if constexpr (observe_rcni) {
    for (size_t candidate = 0; candidate < pool.size(); ++candidate) {
      const auto& causal_witness = causal_witnesses[candidate];
      if (!causal_witness.witness_id.has_value() || causal_witness.occlusion_factor <= alpha ||
          std::find(result.begin(), result.end(), pool[candidate].id) != result.end()) {
        continue;
      }
      events->push_back(rcni_prune_event_t{
          static_cast<uint32_t>(location), *causal_witness.witness_id, pool[candidate].id,
          pool[candidate].distance, causal_witness.witness_victim_distance, std::nullopt});
    }
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::prune_neighbors(const uint32_t location,
                                                        std::vector<neighbor_t>& pool,
                                                        std::vector<uint32_t>& pruned_list,
                                                        in_mem_query_scratch_t<T>* scratch) {
  prune_neighbors(location, pool, indexing_range_, indexing_max_c_, indexing_alpha_, pruned_list,
                  scratch);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::prune_neighbors(
    const uint32_t location, std::vector<neighbor_t>& pool, const uint32_t range,
    const uint32_t max_candidate_size, const float alpha, std::vector<uint32_t>& pruned_list,
    in_mem_query_scratch_t<T>* scratch) {
  prune_neighbors_impl<false>(location, pool, range, max_candidate_size, alpha, pruned_list,
                              scratch);
}

template <typename T, typename tag_t, typename label_t>
template <bool observe_rcni>
void vamana_index_t<T, tag_t, label_t>::prune_neighbors_impl(
    const uint32_t location, std::vector<neighbor_t>& pool, const uint32_t range,
    const uint32_t max_candidate_size, const float alpha, std::vector<uint32_t>& pruned_list,
    in_mem_query_scratch_t<T>* scratch) {
  if (pool.size() == 0) {
    // if the pool is empty, behave like a noop
    pruned_list.clear();
    return;
  }

  // If using pq_build_, over-write the PQ distances with actual distances
  // REFACTOR PQ: TODO: How to get rid of this!?
  if (pq_dist_) {
    for (auto& ngh : pool)
      ngh.distance = data_store_->get_distance(ngh.id, location);
  }

  // sort the pool based on distance to query and prune it with occlude_list
  std::sort(pool.begin(), pool.end());
  pruned_list.clear();
  pruned_list.reserve(range);

  std::vector<rcni_prune_event_t> events;
  if constexpr (observe_rcni) {
    events.reserve(pool.size());
    occlude_list_impl<true>(location, pool, alpha, range, max_candidate_size, pruned_list, scratch,
                            &events, nullptr);
  } else {
    occlude_list(location, pool, alpha, range, max_candidate_size, pruned_list, scratch);
  }
  assert(pruned_list.size() <= range);

  if (saturate_graph_ && alpha > 1) {
    for (const auto& node : pool) {
      if (pruned_list.size() >= range)
        break;
      if ((std::find(pruned_list.begin(), pruned_list.end(), node.id) == pruned_list.end()) &&
          node.id != location)
        pruned_list.push_back(node.id);
    }
  }

  if constexpr (observe_rcni) {
    events.erase(std::remove_if(events.begin(), events.end(),
                                [&](const auto& event) {
                                  return std::find(pruned_list.begin(), pruned_list.end(),
                                                   event.victim_id) != pruned_list.end();
                                }),
                 events.end());

    for (auto& event : events) {
      for (const uint32_t neighbor_id : pruned_list) {
        if (neighbor_id == event.witness_id) {
          continue;
        }
        const float victim_distance = data_store_->get_distance(event.victim_id, neighbor_id);
        if (!event.alternative.has_value() ||
            victim_distance < event.alternative->victim_distance ||
            (victim_distance == event.alternative->victim_distance &&
             neighbor_id < event.alternative->node_id)) {
          event.alternative = rcni_neighbor_distance_t{neighbor_id, victim_distance};
        }
      }
    }

    assert(rcni_prune_observer_ != nullptr);
    rcni_prune_observer_->observe_source(location, events, pruned_list);
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::inter_insert(uint32_t n, std::vector<uint32_t>& pruned_list,
                                                     const uint32_t range,
                                                     in_mem_query_scratch_t<T>* scratch) {
  const auto& src_pool = pruned_list;

  assert(!src_pool.empty());

  for (auto des : src_pool) {
    // des.loc is the loc of the neighbors of n
    assert(des < max_points_ + num_frozen_pts_);
    // des_pool contains the neighbors of the neighbors of n
    std::vector<uint32_t> copy_of_neighbors;
    bool prune_needed = false;
    {
      lock_guard_t guard(locks_[des]);
      auto& des_pool = graph_store_->get_neighbours(des);
      if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
        if (vamana_build_stats_ && vamana_build_stats_->active()) {
          vamana_build_stats_->record_reverse_insert(des, n);
        }
        if (des_pool.size() < (uint64_t) (defaults::GRAPH_SLACK_FACTOR * range)) {
          // des_pool.emplace_back(n);
          graph_store_->add_neighbour(des, n);
          prune_needed = false;
        } else {
          copy_of_neighbors.reserve(des_pool.size() + 1);
          copy_of_neighbors = des_pool;
          copy_of_neighbors.push_back(n);
          prune_needed = true;
        }
      }
    } // des lock is released by this point

    if (prune_needed) {
      if (vamana_build_stats_ && vamana_build_stats_->active()) {
        vamana_build_stats_->record_reverse_prune(des);
      }
      tsl::robin_set<uint32_t> dummy_visited(0);
      std::vector<neighbor_t> dummy_pool(0);

      size_t reserve_size = (size_t) (std::ceil(1.05 * defaults::GRAPH_SLACK_FACTOR * range));
      dummy_visited.reserve(reserve_size);
      dummy_pool.reserve(reserve_size);

      for (auto cur_nbr : copy_of_neighbors) {
        if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des) {
          float dist = data_store_->get_distance(des, cur_nbr);
          dummy_pool.emplace_back(neighbor_t(cur_nbr, dist));
          dummy_visited.insert(cur_nbr);
        }
      }
      std::vector<uint32_t> new_out_neighbors;
      prune_neighbors(des, dummy_pool, new_out_neighbors, scratch);
      {
        lock_guard_t guard(locks_[des]);

        graph_store_->set_neighbours(des, new_out_neighbors);
      }
    }
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::inter_insert(uint32_t n, std::vector<uint32_t>& pruned_list,
                                                     in_mem_query_scratch_t<T>* scratch) {
  inter_insert(n, pruned_list, indexing_range_, scratch);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::link() {
  if (rcni_prune_observer_ == nullptr) {
    link_impl<false>();
  } else {
    link_impl<true>();
  }
}

template <typename T, typename tag_t, typename label_t>
template <bool observe_rcni>
void vamana_index_t<T, tag_t, label_t>::link_impl() {
  uint32_t num_threads = indexing_threads_;
  if (num_threads != 0)
    omp_set_num_threads(num_threads);

  /* visit_order is a vector that is initialized to the entire graph */
  std::vector<uint32_t> visit_order;
  std::vector<powerlaw_ann::neighbor_t> pool, tmp;
  tsl::robin_set<uint32_t> visited;
  visit_order.reserve(nd_ + num_frozen_pts_);
  for (uint32_t i = 0; i < (uint32_t) nd_; i++) {
    visit_order.emplace_back(i);
  }

  // If there are any frozen points, add them all.
  for (uint32_t frozen = (uint32_t) max_points_; frozen < max_points_ + num_frozen_pts_; frozen++) {
    visit_order.emplace_back(frozen);
  }

  // if there are frozen points, the first such one is set to be the start_
  if (num_frozen_pts_ > 0)
    start_ = (uint32_t) max_points_;
  else
    start_ = calculate_entry_point();

  powerlaw_ann::timer_t link_timer;

#pragma omp parallel for schedule(dynamic, 2048)
  for (int64_t node_ctr = 0; node_ctr < (int64_t) (visit_order.size()); node_ctr++) {
    auto node = visit_order[node_ctr];

    // Find and add appropriate graph edges
    scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
    auto scratch = manager.scratch_space();
    std::vector<uint32_t> pruned_list;
    if constexpr (observe_rcni) {
      search_for_point_and_prune_impl<true>(node, indexing_queue_size_, pruned_list, scratch,
                                            filtered_index_,
                                            filtered_index_ ? filter_indexing_queue_size_ : 0);
    } else if (filtered_index_) {
      search_for_point_and_prune(node, indexing_queue_size_, pruned_list, scratch, true,
                                 filter_indexing_queue_size_);
    } else {
      search_for_point_and_prune(node, indexing_queue_size_, pruned_list, scratch);
    }
    assert(pruned_list.size() > 0);

    {
      lock_guard_t guard(locks_[node]);

      graph_store_->set_neighbours(node, pruned_list);
      assert(graph_store_->get_neighbours((location_t) node).size() <= indexing_range_);
    }

    inter_insert(node, pruned_list, scratch);

    if (node_ctr % 100000 == 0) {
      powerlaw_ann::cout << "\r" << (100.0 * node_ctr) / (visit_order.size())
                         << "% of index build completed." << std::flush;
    }
  }

  if (nd_ > 0) {
    powerlaw_ann::cout << "Starting final cleanup.." << std::flush;
  }
#pragma omp parallel for schedule(dynamic, 2048)
  for (int64_t node_ctr = 0; node_ctr < (int64_t) (visit_order.size()); node_ctr++) {
    auto node = visit_order[node_ctr];
    if (graph_store_->get_neighbours((location_t) node).size() > indexing_range_) {
      scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
      auto scratch = manager.scratch_space();

      tsl::robin_set<uint32_t> dummy_visited(0);
      std::vector<neighbor_t> dummy_pool(0);
      std::vector<uint32_t> new_out_neighbors;

      for (auto cur_nbr : graph_store_->get_neighbours((location_t) node)) {
        if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node) {
          float dist = data_store_->get_distance(node, cur_nbr);
          dummy_pool.emplace_back(neighbor_t(cur_nbr, dist));
          dummy_visited.insert(cur_nbr);
        }
      }
      prune_neighbors(node, dummy_pool, new_out_neighbors, scratch);

      graph_store_->clear_neighbours((location_t) node);
      graph_store_->set_neighbours((location_t) node, new_out_neighbors);
    }
  }
  if (nd_ > 0) {
    powerlaw_ann::cout << "done. Link time: " << ((double) link_timer.elapsed() / (double) 1000000)
                       << "s" << std::endl;
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::prune_all_neighbors(const uint32_t max_degree,
                                                            const uint32_t max_occlusion_size,
                                                            const float alpha) {
  const uint32_t range = max_degree;
  const uint32_t maxc = max_occlusion_size;

  filtered_index_ = true;

  powerlaw_ann::timer_t timer;
#pragma omp parallel for
  for (int64_t node = 0; node < (int64_t) (max_points_ + num_frozen_pts_); node++) {
    if ((size_t) node < nd_ || (size_t) node >= max_points_) {
      if (graph_store_->get_neighbours((location_t) node).size() > range) {
        tsl::robin_set<uint32_t> dummy_visited(0);
        std::vector<neighbor_t> dummy_pool(0);
        std::vector<uint32_t> new_out_neighbors;

        scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
        auto scratch = manager.scratch_space();

        for (auto cur_nbr : graph_store_->get_neighbours((location_t) node)) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node) {
            float dist = data_store_->get_distance((location_t) node, (location_t) cur_nbr);
            dummy_pool.emplace_back(neighbor_t(cur_nbr, dist));
            dummy_visited.insert(cur_nbr);
          }
        }

        prune_neighbors((uint32_t) node, dummy_pool, range, maxc, alpha, new_out_neighbors,
                        scratch);
        graph_store_->clear_neighbours((location_t) node);
        graph_store_->set_neighbours((location_t) node, new_out_neighbors);
      }
    }
  }

  powerlaw_ann::cout << "Prune time : " << timer.elapsed() / 1000 << "ms" << std::endl;
  size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
  for (size_t i = 0; i < max_points_ + num_frozen_pts_; i++) {
    if (i < nd_ || i >= max_points_) {
      const std::vector<uint32_t>& pool = graph_store_->get_neighbours((location_t) i);
      max = (std::max) (max, pool.size());
      min = (std::min) (min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
  }
  if (min > max)
    min = max;
  if (nd_ > 0) {
    powerlaw_ann::cout << "vamana_index_t built with degree: max:" << max
                       << "  avg:" << (float) total / (float) (nd_ + num_frozen_pts_)
                       << "  min:" << min << "  count(deg<2):" << cnt << std::endl;
  }
}

// REFACTOR
template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::set_start_points(const T* data, size_t data_count) {
  std::unique_lock<std::shared_timed_mutex> ul(update_lock_);
  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  if (nd_ > 0)
    throw diskann_exception_t("Can not set starting point for a non-empty index", -1, __FUNCSIG__,
                              __FILE__, __LINE__);

  if (data_count != num_frozen_pts_ * dim_)
    throw diskann_exception_t("Invalid number of points", -1, __FUNCSIG__, __FILE__, __LINE__);

  //     memcpy(data_ + aligned_dim_ * max_points_, data, aligned_dim_ *
  //     sizeof(T) * num_frozen_pts_);
  for (location_t i = 0; i < num_frozen_pts_; i++) {
    data_store_->set_vector((location_t) (i + max_points_), data + i * dim_);
  }
  has_built_ = true;
  powerlaw_ann::cout << "vamana_index_t start points set: #" << num_frozen_pts_ << std::endl;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::set_start_points_at_random_impl(data_type_t radius,
                                                                        uint32_t random_seed) {
  try {
    T radius_to_use = std::any_cast<T>(radius);
    this->set_start_points_at_random(radius_to_use, random_seed);
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t(
        "Error: bad any cast while performing set_start_points_at_random_impl() " +
            std::string(e.what()),
        -1);
  } catch (const std::exception& e) {
    throw diskann_exception_t("Error: " + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::set_start_points_at_random(T radius, uint32_t random_seed) {
  std::mt19937 gen{random_seed};
  std::normal_distribution<> d{0.0, 1.0};

  std::vector<T> points_data;
  points_data.reserve(dim_ * num_frozen_pts_);
  std::vector<double> real_vec(dim_);

  for (size_t frozen_point = 0; frozen_point < num_frozen_pts_; frozen_point++) {
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim_; ++i) {
      auto r = d(gen);
      real_vec[i] = r;
      norm_sq += r * r;
    }

    const double norm = std::sqrt(norm_sq);
    for (auto iter : real_vec)
      points_data.push_back(static_cast<T>(iter * radius / norm));
  }

  set_start_points(points_data.data(), points_data.size());
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::build_with_data_populated(const std::vector<tag_t>& tags) {
  powerlaw_ann::cout << "Starting index build with " << nd_ << " points... " << std::endl;

  if (nd_ < 1)
    throw diskann_exception_t("Error: Trying to build an index with 0 points", -1, __FUNCSIG__,
                              __FILE__, __LINE__);

  if (enable_tags_ && tags.size() != nd_) {
    std::stringstream stream;
    stream << "ERROR: Driver requests loading " << nd_ << " points from file,"
           << "but tags vector is of size " << tags.size() << "." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
  if (enable_tags_) {
    for (size_t i = 0; i < tags.size(); ++i) {
      tag_to_location_[tags[i]] = (uint32_t) i;
      location_to_tag_.set(static_cast<uint32_t>(i), tags[i]);
    }
  }

  uint32_t index_r = indexing_range_;
  uint32_t num_threads_index = indexing_threads_;
  uint32_t index_l = indexing_queue_size_;
  uint32_t maxc = indexing_max_c_;

  if (query_scratch_.size() == 0) {
    initialize_query_scratch(5 + num_threads_index, index_l, index_l, index_r, maxc,
                             data_store_->get_aligned_dim());
  }

  generate_frozen_point();
  link();
  run_post_link_graph_observer();

  size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto& pool = graph_store_->get_neighbours((location_t) i);
    max = std::max(max, pool.size());
    min = std::min(min, pool.size());
    total += pool.size();
    if (pool.size() < 2)
      cnt++;
  }
  powerlaw_ann::cout << "vamana_index_t built with degree: max:" << max
                     << "  avg:" << (float) total / (float) (nd_ + num_frozen_pts_)
                     << "  min:" << min << "  count(deg<2):" << cnt << std::endl;

  has_built_ = true;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::run_post_link_graph_observer() {
  if (post_link_graph_observer_ == nullptr) {
    return;
  }
  if constexpr (!std::is_same_v<T, float>) {
    throw diskann_exception_t("Post-link graph observation requires float32 vectors", -1,
                              __FUNCSIG__, __FILE__, __LINE__);
  } else {
    if (dist_metric_ != powerlaw_ann::metric_t::L2 || filtered_index_ || dynamic_index_ ||
        dim_ == 0 || dim_ > std::numeric_limits<uint32_t>::max() || nd_ == 0 || start_ >= nd_ ||
        nd_ >= std::numeric_limits<uint32_t>::max()) {
      throw diskann_exception_t(
          "Post-link graph observation requires a completed static unfiltered float32 Vamana graph",
          -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    class graph_view_t final : public post_link_graph_view_t {
    public:
      graph_view_t(size_t point_count, size_t dimension, uint32_t entry_point_id,
                   const abstract_graph_store_t& graph_store,
                   const abstract_data_store_t<float>& data_store)
          : point_count_(point_count), dimension_(dimension), entry_point_id_(entry_point_id),
            graph_store_(graph_store), data_store_(data_store) {}

      uint64_t point_count() const noexcept override { return point_count_; }

      uint32_t dimension() const noexcept override { return static_cast<uint32_t>(dimension_); }

      uint32_t entry_point_id() const noexcept override { return entry_point_id_; }

      std::span<const uint32_t> neighbors(uint32_t node_id) const override {
        if (node_id >= point_count_) {
          throw std::out_of_range("Post-link neighbor source exceeds the Base graph");
        }
        const auto& neighbors = graph_store_.get_neighbours(node_id);
        static_assert(std::is_same_v<location_t, uint32_t>);
        return neighbors;
      }

      void copy_vector(uint32_t node_id, std::span<float> destination) const override {
        if (node_id >= point_count_ || destination.size() != dimension_) {
          throw std::invalid_argument("Post-link vector request is out of range");
        }
        data_store_.get_vector(node_id, destination.data());
      }

      float squared_distance(uint32_t left, uint32_t right) const override {
        if (left >= point_count_ || right >= point_count_) {
          throw std::out_of_range("Post-link distance endpoint exceeds the Base graph");
        }
        return data_store_.get_distance(left, right);
      }

    private:
      size_t point_count_;
      size_t dimension_;
      uint32_t entry_point_id_;
      const abstract_graph_store_t& graph_store_;
      const abstract_data_store_t<float>& data_store_;
    };

    const graph_view_t graph(nd_, dim_, start_, *graph_store_, *data_store_);
    post_link_graph_observer_->observe(graph);
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::build_impl(const data_type_t& data,
                                                   const size_t num_points_to_load,
                                                   tag_vector_t& tags) {
  post_link_observer_reset_guard_t reset_post_link_observer(post_link_graph_observer_);
  try {
    this->build(std::any_cast<const T*>(data), num_points_to_load,
                tags.get<const std::vector<tag_t>>());
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t(
        "Error: bad any cast in while building index. " + std::string(e.what()), -1);
  } catch (const std::exception& e) {
    throw diskann_exception_t("Error" + std::string(e.what()), -1);
  }
}
template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::build(const T* data, const size_t num_points_to_load,
                                              const std::vector<tag_t>& tags) {
  post_link_observer_reset_guard_t reset_post_link_observer(post_link_graph_observer_);
  if (num_points_to_load == 0) {
    throw diskann_exception_t("Do not call build with 0 points", -1, __FUNCSIG__, __FILE__,
                              __LINE__);
  }
  if (pq_dist_) {
    throw diskann_exception_t("ERROR: DO not use this build interface with PQ distance", -1,
                              __FUNCSIG__, __FILE__, __LINE__);
  }

  std::unique_lock<std::shared_timed_mutex> ul(update_lock_);

  {
    std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
    nd_ = num_points_to_load;

    data_store_->populate_data(data, (location_t) num_points_to_load);
  }

  build_with_data_populated(tags);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::build(const char* filename, const size_t num_points_to_load,
                                              const std::vector<tag_t>& tags) {
  post_link_observer_reset_guard_t reset_post_link_observer(post_link_graph_observer_);
  // idealy this should call build_filtered_index based on params passed

  std::unique_lock<std::shared_timed_mutex> ul(update_lock_);

  // error checks
  if (num_points_to_load == 0)
    throw diskann_exception_t("Do not call build with 0 points", -1, __FUNCSIG__, __FILE__,
                              __LINE__);

  if (!file_exists(filename)) {
    std::stringstream stream;
    stream << "ERROR: Data file " << filename << " does not exist." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  size_t file_num_points, file_dim;
  if (filename == nullptr) {
    throw powerlaw_ann::diskann_exception_t("Can not build with an empty file", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
  }

  powerlaw_ann::get_bin_metadata(filename, file_num_points, file_dim);
  if (file_num_points > max_points_) {
    std::stringstream stream;
    stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
           << file_num_points << " points, but " << "index can support only " << max_points_
           << " points as specified in constructor." << std::endl;

    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (num_points_to_load > file_num_points) {
    std::stringstream stream;
    stream << "ERROR: Driver requests loading " << num_points_to_load
           << " points and file has only " << file_num_points << " points." << std::endl;

    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (file_dim != dim_) {
    std::stringstream stream;
    stream << "ERROR: Driver requests loading " << dim_ << " dimension," << "but file has "
           << file_dim << " dimension." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;

    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  // REFACTOR PQ TODO: We can remove this if and add a check in the in_mem_data_store_t
  // to not populate_data if it has been called once.
  if (pq_dist_) {
#ifdef EXEC_ENV_OLS
    std::stringstream ss;
    ss << "PQ Build is not supported in DLVS environment (i.e. if EXEC_ENV_OLS is defined)"
       << std::endl;
    powerlaw_ann::cerr << ss.str() << std::endl;
    throw diskann_exception_t(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
#else
    // REFACTOR TODO: Both in the previous code and in the current pq_data_store_t,
    // we are writing the PQ files in the same path as the input file. Now we
    // may not have write permissions to that folder, but we will always have
    // write permissions to the output folder. So we should write the PQ files
    // there. The problem is that the vamana_index_t class gets the output folder prefix
    // only at the time of save(), by which time we are too late. So leaving it
    // as-is for now.
    pq_data_store_->populate_data(filename, 0U);
#endif
  }

  data_store_->populate_data(filename, 0U);
  powerlaw_ann::cout << "Using only first " << num_points_to_load << " from file.. " << std::endl;

  {
    std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
    nd_ = num_points_to_load;
  }
  build_with_data_populated(tags);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::build(const char* filename, const size_t num_points_to_load,
                                              const char* tag_filename) {
  post_link_observer_reset_guard_t reset_post_link_observer(post_link_graph_observer_);
  std::vector<tag_t> tags;

  if (enable_tags_) {
    std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
    if (tag_filename == nullptr) {
      throw diskann_exception_t("Tag filename is null, while enable_tags_ is set", -1, __FUNCSIG__,
                                __FILE__, __LINE__);
    } else {
      if (file_exists(tag_filename)) {
        powerlaw_ann::cout << "Loading tags from " << tag_filename << " for vamana index build"
                           << std::endl;
        tag_t* tag_data = nullptr;
        size_t npts, ndim;
        powerlaw_ann::load_bin(tag_filename, tag_data, npts, ndim);
        if (npts < num_points_to_load) {
          std::stringstream sstream;
          sstream << "Loaded " << npts << " tags, insufficient to populate tags for "
                  << num_points_to_load << "  points to load";
          throw powerlaw_ann::diskann_exception_t(sstream.str(), -1, __FUNCSIG__, __FILE__,
                                                  __LINE__);
        }
        for (size_t i = 0; i < num_points_to_load; i++) {
          tags.push_back(tag_data[i]);
        }
        delete[] tag_data;
      } else {
        throw powerlaw_ann::diskann_exception_t(std::string("Tag file") + tag_filename +
                                                    " does not exist",
                                                -1, __FUNCSIG__, __FILE__, __LINE__);
      }
    }
  }
  build(filename, num_points_to_load, tags);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::build(const std::string& data_file,
                                              const size_t num_points_to_load,
                                              index_filter_params_t& filter_params) {
  post_link_observer_reset_guard_t reset_post_link_observer(post_link_graph_observer_);
  size_t points_to_load = num_points_to_load == 0 ? max_points_ : num_points_to_load;

  auto s = std::chrono::high_resolution_clock::now();
  if (filter_params.label_file == "") {
    this->build(data_file.c_str(), points_to_load);
  } else {
    // TODO: this should ideally happen in save()
    std::string labels_file_to_use = filter_params.save_path_prefix + "label_formatted_.txt";
    std::string mem_labels_int_map_file = filter_params.save_path_prefix + "labels_map_.txt";
    convert_labels_string_to_int(filter_params.label_file, labels_file_to_use,
                                 mem_labels_int_map_file, filter_params.universal_label);
    if (filter_params.universal_label != "") {
      label_t unv_label_as_num = 0;
      this->set_universal_label(unv_label_as_num);
    }
    this->build_filtered_index(data_file.c_str(), labels_file_to_use, points_to_load);
  }
  std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
  std::cout << "Indexing time: " << diff.count() << "\n";
}

template <typename T, typename tag_t, typename label_t>
std::unordered_map<std::string, label_t>
vamana_index_t<T, tag_t, label_t>::load_label_map(const std::string& labels_map_file) {
  std::unordered_map<std::string, label_t> string_to_int_mp;
  std::ifstream map_reader(labels_map_file);
  std::string line, token;
  label_t token_as_num;
  std::string label_str;
  while (std::getline(map_reader, line)) {
    std::istringstream iss(line);
    getline(iss, token, '\t');
    label_str = token;
    getline(iss, token, '\t');
    token_as_num = (label_t) std::stoul(token);
    string_to_int_mp[label_str] = token_as_num;
  }
  return string_to_int_mp;
}

template <typename T, typename tag_t, typename label_t>
label_t vamana_index_t<T, tag_t, label_t>::get_converted_label(const std::string& raw_label) {
  if (label_map_.find(raw_label) != label_map_.end()) {
    return label_map_[raw_label];
  }
  if (use_universal_label_) {
    return universal_label_;
  }
  std::stringstream stream;
  stream << "Unable to find label in the Label Map";
  powerlaw_ann::cerr << stream.str() << std::endl;
  throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::parse_label_file(const std::string& label_file,
                                                         size_t& num_points) {
  // Format of Label txt file: filters with comma separators

  std::ifstream infile(label_file);
  if (infile.fail()) {
    throw powerlaw_ann::diskann_exception_t(std::string("Failed to open file ") + label_file, -1);
  }

  std::string line, token;
  uint32_t line_cnt = 0;

  while (std::getline(infile, line)) {
    line_cnt++;
  }
  location_to_labels_.resize(line_cnt, std::vector<label_t>());

  infile.clear();
  infile.seekg(0, std::ios::beg);
  line_cnt = 0;

  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::vector<label_t> lbls(0);
    getline(iss, token, '\t');
    std::istringstream new_iss(token);
    while (getline(new_iss, token, ',')) {
      token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
      token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
      label_t token_as_num = (label_t) std::stoul(token);
      lbls.push_back(token_as_num);
      labels_.insert(token_as_num);
    }

    std::sort(lbls.begin(), lbls.end());
    location_to_labels_[line_cnt] = lbls;
    line_cnt++;
  }
  num_points = (size_t) line_cnt;
  powerlaw_ann::cout << "Identified " << labels_.size() << " distinct label(s)" << std::endl;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::set_universal_label_impl(
    const label_type_t universal_label) {
  this->set_universal_label(std::any_cast<const label_t>(universal_label));
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::set_universal_label(const label_t& label) {
  use_universal_label_ = true;
  universal_label_ = label;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::build_filtered_index(const char* filename,
                                                             const std::string& label_file,
                                                             const size_t num_points_to_load,
                                                             const std::vector<tag_t>& tags) {
  post_link_observer_reset_guard_t reset_post_link_observer(post_link_graph_observer_);
  filtered_index_ = true;
  label_to_start_id_.clear();
  size_t num_points_labels = 0;

  parse_label_file(label_file,
                   num_points_labels); // determines medoid for each label and identifies
                                       // the points to label mapping

  std::unordered_map<label_t, std::vector<uint32_t>> label_to_points;

  for (uint32_t point_id = 0; point_id < num_points_to_load; point_id++) {
    for (auto label : location_to_labels_[point_id]) {
      if (label != universal_label_) {
        label_to_points[label].emplace_back(point_id);
      } else {
        for (typename tsl::robin_set<label_t>::size_type lbl = 0; lbl < labels_.size(); lbl++) {
          auto itr = labels_.begin();
          std::advance(itr, lbl);
          auto& x = *itr;
          label_to_points[x].emplace_back(point_id);
        }
      }
    }
  }

  uint32_t num_cands = 25;
  for (auto itr = labels_.begin(); itr != labels_.end(); itr++) {
    uint32_t best_medoid_count = std::numeric_limits<uint32_t>::max();
    auto& curr_label = *itr;
    uint32_t best_medoid;
    auto labeled_points = label_to_points[curr_label];
    for (uint32_t cnd = 0; cnd < num_cands; cnd++) {
      uint32_t cur_cnd = labeled_points[rand() % labeled_points.size()];
      uint32_t cur_cnt = std::numeric_limits<uint32_t>::max();
      if (medoid_counts_.find(cur_cnd) == medoid_counts_.end()) {
        medoid_counts_[cur_cnd] = 0;
        cur_cnt = 0;
      } else {
        cur_cnt = medoid_counts_[cur_cnd];
      }
      if (cur_cnt < best_medoid_count) {
        best_medoid_count = cur_cnt;
        best_medoid = cur_cnd;
      }
    }
    label_to_start_id_[curr_label] = best_medoid;
    medoid_counts_[best_medoid]++;
  }

  this->build(filename, num_points_to_load, tags);
}

template <typename T, typename tag_t, typename label_t>
std::pair<uint32_t, uint32_t>
vamana_index_t<T, tag_t, label_t>::search_impl(const data_type_t& query, const size_t K,
                                               const uint32_t L, std::any& indices,
                                               float* distances) {
  try {
    auto typed_query = std::any_cast<const T*>(query);
    if (typeid(uint32_t*) == indices.type()) {
      auto u32_ptr = std::any_cast<uint32_t*>(indices);
      return this->search(typed_query, K, L, u32_ptr, distances);
    } else if (typeid(uint64_t*) == indices.type()) {
      auto u64_ptr = std::any_cast<uint64_t*>(indices);
      return this->search(typed_query, K, L, u64_ptr, distances);
    } else {
      throw diskann_exception_t("Error: indices type can only be uint64_t or uint32_t.", -1);
    }
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t("Error: bad any cast while searching. " + std::string(e.what()), -1);
  } catch (const std::exception& e) {
    throw diskann_exception_t("Error: " + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
template <typename id_t>
std::pair<uint32_t, uint32_t>
vamana_index_t<T, tag_t, label_t>::search(const T* query, const size_t K, const uint32_t L,
                                          id_t* indices, float* distances) {
  if (K > (uint64_t) L) {
    throw diskann_exception_t("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__,
                              __LINE__);
  }

  scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
  auto scratch = manager.scratch_space();

  if (L > scratch->get_l()) {
    powerlaw_ann::cout << "Attempting to expand query scratch_space. Was created "
                       << "with l_size: " << scratch->get_l() << " but search L is: " << L
                       << std::endl;
    scratch->resize_for_new_l(L);
    powerlaw_ann::cout << "Resize completed. New scratch->L is " << scratch->get_l() << std::endl;
  }

  const std::vector<label_t> unused_filter_label;
  const std::vector<uint32_t> init_ids = get_init_ids();

  std::shared_lock<std::shared_timed_mutex> lock(update_lock_);

  data_store_->preprocess_query(query, scratch);

  auto retval = iterate_to_fixed_point(scratch, L, init_ids, false, unused_filter_label, true);

  neighbor_priority_queue_t& best_l_nodes = scratch->best_l_nodes();

  size_t pos = 0;
  for (size_t i = 0; i < best_l_nodes.size(); ++i) {
    if (best_l_nodes[i].id < max_points_) {
      // safe because vamana_index_t uses uint32_t ids internally
      // and IDType will be uint32_t or uint64_t
      indices[pos] = (id_t) best_l_nodes[i].id;
      if (distances != nullptr) {
#ifdef EXEC_ENV_OLS
        // DLVS expects negative distances
        distances[pos] = best_l_nodes[i].distance;
#else
        distances[pos] = dist_metric_ == powerlaw_ann::metric_t::INNER_PRODUCT
                             ? -1 * best_l_nodes[i].distance
                             : best_l_nodes[i].distance;
#endif
      }
      pos++;
    }
    if (pos == K)
      break;
  }
  if (pos < K) {
    powerlaw_ann::cerr << "Found pos: " << pos << "fewer than K elements " << K << " for query"
                       << std::endl;
  }

  return retval;
}

template <typename T, typename tag_t, typename label_t>
std::pair<uint32_t, uint32_t> vamana_index_t<T, tag_t, label_t>::search_with_filters_impl(
    const data_type_t& query, const std::string& raw_label, const size_t K, const uint32_t L,
    std::any& indices, float* distances) {
  auto converted_label = this->get_converted_label(raw_label);
  if (typeid(uint64_t*) == indices.type()) {
    auto ptr = std::any_cast<uint64_t*>(indices);
    return this->search_with_filters(std::any_cast<T*>(query), converted_label, K, L, ptr,
                                     distances);
  } else if (typeid(uint32_t*) == indices.type()) {
    auto ptr = std::any_cast<uint32_t*>(indices);
    return this->search_with_filters(std::any_cast<T*>(query), converted_label, K, L, ptr,
                                     distances);
  } else {
    throw diskann_exception_t("Error: Id type can only be uint64_t or uint32_t.", -1);
  }
}

template <typename T, typename tag_t, typename label_t>
template <typename id_t>
std::pair<uint32_t, uint32_t>
vamana_index_t<T, tag_t, label_t>::search_with_filters(const T* query, const label_t& filter_label,
                                                       const size_t K, const uint32_t L,
                                                       id_t* indices, float* distances) {
  if (K > (uint64_t) L) {
    throw diskann_exception_t("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__,
                              __LINE__);
  }

  scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
  auto scratch = manager.scratch_space();

  if (L > scratch->get_l()) {
    powerlaw_ann::cout << "Attempting to expand query scratch_space. Was created "
                       << "with l_size: " << scratch->get_l() << " but search L is: " << L
                       << std::endl;
    scratch->resize_for_new_l(L);
    powerlaw_ann::cout << "Resize completed. New scratch->L is " << scratch->get_l() << std::endl;
  }

  std::vector<label_t> filter_vec;
  std::vector<uint32_t> init_ids = get_init_ids();

  std::shared_lock<std::shared_timed_mutex> lock(update_lock_);
  std::shared_lock<std::shared_timed_mutex> tl(tag_lock_, std::defer_lock);
  if (dynamic_index_)
    tl.lock();

  if (label_to_start_id_.find(filter_label) != label_to_start_id_.end()) {
    init_ids.emplace_back(label_to_start_id_[filter_label]);
  } else {
    powerlaw_ann::cout << "No filtered medoid found. exitting "
                       << std::endl; // RKNOTE: If universal label found start there
    throw powerlaw_ann::diskann_exception_t("No filtered medoid found. exitting ", -1);
  }
  if (dynamic_index_)
    tl.unlock();

  filter_vec.emplace_back(filter_label);

  data_store_->preprocess_query(query, scratch);
  auto retval = iterate_to_fixed_point(scratch, L, init_ids, true, filter_vec, true);

  auto best_l_nodes = scratch->best_l_nodes();

  size_t pos = 0;
  for (size_t i = 0; i < best_l_nodes.size(); ++i) {
    if (best_l_nodes[i].id < max_points_) {
      indices[pos] = (id_t) best_l_nodes[i].id;

      if (distances != nullptr) {
#ifdef EXEC_ENV_OLS
        // DLVS expects negative distances
        distances[pos] = best_l_nodes[i].distance;
#else
        distances[pos] = dist_metric_ == powerlaw_ann::metric_t::INNER_PRODUCT
                             ? -1 * best_l_nodes[i].distance
                             : best_l_nodes[i].distance;
#endif
      }
      pos++;
    }
    if (pos == K)
      break;
  }
  if (pos < K) {
    powerlaw_ann::cerr << "Found fewer than K elements for query" << std::endl;
  }

  return retval;
}

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::search_with_tags_impl(
    const data_type_t& query, const uint64_t K, const uint32_t L, const tag_type_t& tags,
    float* distances, data_vector_t& res_vectors, bool use_filters,
    const std::string filter_label) {
  try {
    return this->search_with_tags(std::any_cast<const T*>(query), K, L, std::any_cast<tag_t*>(tags),
                                  distances, res_vectors.get<std::vector<T*>>(), use_filters,
                                  filter_label);
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t("Error: bad any cast while performing search_with_tags_impl() " +
                                  std::string(e.what()),
                              -1);
  } catch (const std::exception& e) {
    throw diskann_exception_t("Error: " + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::search_with_tags(
    const T* query, const uint64_t K, const uint32_t L, tag_t* tags, float* distances,
    std::vector<T*>& res_vectors, bool use_filters, const std::string filter_label) {
  if (K > (uint64_t) L) {
    throw diskann_exception_t("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__,
                              __LINE__);
  }
  scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
  auto scratch = manager.scratch_space();

  if (L > scratch->get_l()) {
    powerlaw_ann::cout << "Attempting to expand query scratch_space. Was created "
                       << "with l_size: " << scratch->get_l() << " but search L is: " << L
                       << std::endl;
    scratch->resize_for_new_l(L);
    powerlaw_ann::cout << "Resize completed. New scratch->L is " << scratch->get_l() << std::endl;
  }

  std::shared_lock<std::shared_timed_mutex> ul(update_lock_);

  const std::vector<uint32_t> init_ids = get_init_ids();

  // distance_->preprocess_query(query, data_store_->get_dims(),
  //  scratch->aligned_query());
  data_store_->preprocess_query(query, scratch);
  if (!use_filters) {
    const std::vector<label_t> unused_filter_label;
    iterate_to_fixed_point(scratch, L, init_ids, false, unused_filter_label, true);
  } else {
    std::vector<label_t> filter_vec;
    auto converted_label = this->get_converted_label(filter_label);
    filter_vec.push_back(converted_label);
    iterate_to_fixed_point(scratch, L, init_ids, true, filter_vec, true);
  }

  neighbor_priority_queue_t& best_l_nodes = scratch->best_l_nodes();
  assert(best_l_nodes.size() <= L);

  std::shared_lock<std::shared_timed_mutex> tl(tag_lock_);

  size_t pos = 0;
  for (size_t i = 0; i < best_l_nodes.size(); ++i) {
    auto node = best_l_nodes[i];

    tag_t tag;
    if (location_to_tag_.try_get(node.id, tag)) {
      tags[pos] = tag;

      if (res_vectors.size() > 0) {
        data_store_->get_vector(node.id, res_vectors[pos]);
      }

      if (distances != nullptr) {
#ifdef EXEC_ENV_OLS
        distances[pos] = node.distance; // DLVS expects negative distances
#else
        distances[pos] = dist_metric_ == INNER_PRODUCT ? -1 * node.distance : node.distance;
#endif
      }
      pos++;
      // If res_vectors.size() < k, clip at the value.
      if (pos == K || pos == res_vectors.size())
        break;
    }
  }

  return pos;
}

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::get_num_points() {
  std::shared_lock<std::shared_timed_mutex> tl(tag_lock_);
  return nd_;
}

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::get_max_points() {
  std::shared_lock<std::shared_timed_mutex> tl(tag_lock_);
  return max_points_;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::generate_frozen_point() {
  if (num_frozen_pts_ == 0)
    return;

  if (num_frozen_pts_ > 1) {
    throw diskann_exception_t("More than one frozen point not supported in generate_frozen_point",
                              -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (nd_ == 0) {
    throw diskann_exception_t("ERROR: Can not pick a frozen point since nd=0", -1, __FUNCSIG__,
                              __FILE__, __LINE__);
  }
  size_t res = calculate_entry_point();

  // REFACTOR PQ: Not sure if we should do this for both stores.
  if (pq_dist_) {
    // copy the PQ data corresponding to the point returned by
    // calculate_entry_point
    // memcpy(pq_data_ + max_points_ * num_pq_chunks_,
    //       pq_data_ + res * num_pq_chunks_,
    //       num_pq_chunks_ * DIV_ROUND_UP(NUM_PQ_BITS, 8));
    pq_data_store_->copy_vectors((location_t) res, (location_t) max_points_, 1);
  } else {
    data_store_->copy_vectors((location_t) res, (location_t) max_points_, 1);
  }
  frozen_pts_used_++;
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::enable_delete() {
  assert(enable_tags_);

  if (!enable_tags_) {
    powerlaw_ann::cerr << "Tags must be instantiated for deletions" << std::endl;
    return -2;
  }

  if (this->deletes_enabled_) {
    return 0;
  }

  std::unique_lock<std::shared_timed_mutex> ul(update_lock_);
  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  std::unique_lock<std::shared_timed_mutex> dl(delete_lock_);

  if (data_compacted_) {
    for (uint32_t slot = (uint32_t) nd_; slot < max_points_; ++slot) {
      empty_slots_.insert(slot);
    }
  }
  this->deletes_enabled_ = true;
  return 0;
}

template <typename T, typename tag_t, typename label_t>
inline void vamana_index_t<T, tag_t, label_t>::process_delete(
    const tsl::robin_set<uint32_t>& old_delete_set, size_t loc, const uint32_t range,
    const uint32_t maxc, const float alpha, in_mem_query_scratch_t<T>* scratch) {
  tsl::robin_set<uint32_t>& expanded_nodes_set = scratch->expanded_nodes_set();
  std::vector<neighbor_t>& expanded_nghrs_vec = scratch->expanded_nodes_vec();

  // If this condition were not true, deadlock could result
  assert(old_delete_set.find((uint32_t) loc) == old_delete_set.end());

  std::vector<uint32_t> adj_list;
  {
    // Acquire and release lock[loc] before acquiring locks for neighbors
    std::unique_lock<non_recursive_mutex_t> adj_list_lock;
    if (conc_consolidate_)
      adj_list_lock = std::unique_lock<non_recursive_mutex_t>(locks_[loc]);
    adj_list = graph_store_->get_neighbours((location_t) loc);
  }

  bool modify = false;
  for (auto ngh : adj_list) {
    if (old_delete_set.find(ngh) == old_delete_set.end()) {
      expanded_nodes_set.insert(ngh);
    } else {
      modify = true;

      std::unique_lock<non_recursive_mutex_t> ngh_lock;
      if (conc_consolidate_)
        ngh_lock = std::unique_lock<non_recursive_mutex_t>(locks_[ngh]);
      for (auto j : graph_store_->get_neighbours((location_t) ngh))
        if (j != loc && old_delete_set.find(j) == old_delete_set.end())
          expanded_nodes_set.insert(j);
    }
  }

  if (modify) {
    if (expanded_nodes_set.size() <= range) {
      std::unique_lock<non_recursive_mutex_t> adj_list_lock(locks_[loc]);
      graph_store_->clear_neighbours((location_t) loc);
      for (auto& ngh : expanded_nodes_set)
        graph_store_->add_neighbour((location_t) loc, ngh);
    } else {
      // Create a pool of neighbor_t candidates from the expanded_nodes_set
      expanded_nghrs_vec.reserve(expanded_nodes_set.size());
      for (auto& ngh : expanded_nodes_set) {
        expanded_nghrs_vec.emplace_back(
            ngh, data_store_->get_distance((location_t) loc, (location_t) ngh));
      }
      std::sort(expanded_nghrs_vec.begin(), expanded_nghrs_vec.end());
      std::vector<uint32_t>& occlude_list_output = scratch->occlude_list_output();
      occlude_list((uint32_t) loc, expanded_nghrs_vec, alpha, range, maxc, occlude_list_output,
                   scratch, &old_delete_set);
      std::unique_lock<non_recursive_mutex_t> adj_list_lock(locks_[loc]);
      graph_store_->set_neighbours((location_t) loc, occlude_list_output);
    }
  }
}

// Returns number of live points left after consolidation
template <typename T, typename tag_t, typename label_t>
consolidation_report_t
vamana_index_t<T, tag_t, label_t>::consolidate_deletes(const index_write_parameters_t& params) {
  if (!enable_tags_)
    throw powerlaw_ann::diskann_exception_t("Point tag array not instantiated", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);

  {
    std::shared_lock<std::shared_timed_mutex> ul(update_lock_);
    std::shared_lock<std::shared_timed_mutex> tl(tag_lock_);
    std::shared_lock<std::shared_timed_mutex> dl(delete_lock_);
    if (empty_slots_.size() + nd_ != max_points_) {
      std::string err = "#empty slots + nd != max points";
      powerlaw_ann::cerr << err << std::endl;
      throw diskann_exception_t(err, -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (location_to_tag_.size() + delete_set_->size() != nd_) {
      powerlaw_ann::cerr << "Error: location_to_tag_.size (" << location_to_tag_.size()
                         << ")  + delete_set_->size (" << delete_set_->size() << ") != nd_(" << nd_
                         << ") ";
      return consolidation_report_t(
          powerlaw_ann::consolidation_report_t::status_code_t::INCONSISTENT_COUNT_ERROR, 0, 0, 0, 0,
          0, 0, 0);
    }

    if (location_to_tag_.size() != tag_to_location_.size()) {
      throw powerlaw_ann::diskann_exception_t(
          "location_to_tag_ and tag_to_location_ not of same size", -1, __FUNCSIG__, __FILE__,
          __LINE__);
    }
  }

  std::unique_lock<std::shared_timed_mutex> update_lock(update_lock_, std::defer_lock);
  if (!conc_consolidate_)
    update_lock.lock();

  std::unique_lock<std::shared_timed_mutex> cl(consolidate_lock_, std::defer_lock);
  if (!cl.try_lock()) {
    powerlaw_ann::cerr << "Consildate delete function failed to acquire consolidate lock"
                       << std::endl;
    return consolidation_report_t(powerlaw_ann::consolidation_report_t::status_code_t::LOCK_FAIL, 0,
                                  0, 0, 0, 0, 0, 0);
  }

  powerlaw_ann::cout << "Starting consolidate_deletes... ";

  std::unique_ptr<tsl::robin_set<uint32_t>> old_delete_set(new tsl::robin_set<uint32_t>);
  {
    std::unique_lock<std::shared_timed_mutex> dl(delete_lock_);
    std::swap(delete_set_, old_delete_set);
  }

  if (old_delete_set->find(start_) != old_delete_set->end()) {
    throw powerlaw_ann::diskann_exception_t("ERROR: start node has been deleted", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
  }

  const uint32_t range = params.max_degree;
  const uint32_t maxc = params.max_occlusion_size;
  const float alpha = params.alpha;
  const uint32_t num_threads = params.num_threads == 0 ? omp_get_num_procs() : params.num_threads;

  uint32_t num_calls_to_process_delete = 0;
  powerlaw_ann::timer_t timer;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8192)                          \
    reduction(+ : num_calls_to_process_delete)
  for (int64_t loc = 0; loc < (int64_t) max_points_; loc++) {
    if (old_delete_set->find((uint32_t) loc) == old_delete_set->end() &&
        !empty_slots_.is_in_set((uint32_t) loc)) {
      scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
      auto scratch = manager.scratch_space();
      process_delete(*old_delete_set, loc, range, maxc, alpha, scratch);
      num_calls_to_process_delete += 1;
    }
  }
  for (int64_t loc = max_points_; loc < (int64_t) (max_points_ + num_frozen_pts_); loc++) {
    scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
    auto scratch = manager.scratch_space();
    process_delete(*old_delete_set, loc, range, maxc, alpha, scratch);
    num_calls_to_process_delete += 1;
  }

  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  size_t ret_nd = release_locations(*old_delete_set);
  size_t max_points = max_points_;
  size_t empty_slots_size = empty_slots_.size();

  std::shared_lock<std::shared_timed_mutex> dl(delete_lock_);
  size_t delete_set_size = delete_set_->size();
  size_t old_delete_set_size = old_delete_set->size();

  if (!conc_consolidate_) {
    update_lock.unlock();
  }

  double duration = timer.elapsed() / 1000000.0;
  powerlaw_ann::cout << " done in " << duration << " seconds." << std::endl;
  return consolidation_report_t(powerlaw_ann::consolidation_report_t::status_code_t::SUCCESS,
                                ret_nd, max_points, empty_slots_size, old_delete_set_size,
                                delete_set_size, num_calls_to_process_delete, duration);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::compact_frozen_point() {
  if (nd_ < max_points_ && num_frozen_pts_ > 0) {
    reposition_points((uint32_t) max_points_, (uint32_t) nd_, (uint32_t) num_frozen_pts_);
    start_ = (uint32_t) nd_;

    if (filtered_index_ && dynamic_index_) {
      //  update medoid id's as frozen points are treated as medoid
      for (auto& [label, medoid_id] : label_to_start_id_) {
        /*  if (label == universal_label_)
              continue;*/
        label_to_start_id_[label] = (uint32_t) nd_ + (medoid_id - (uint32_t) max_points_);
      }
    }
  }
}

// Should be called after acquiring update_lock_
template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::compact_data() {
  if (!dynamic_index_)
    throw diskann_exception_t("Can not compact a non-dynamic index", -1, __FUNCSIG__, __FILE__,
                              __LINE__);

  if (data_compacted_) {
    powerlaw_ann::cerr << "Warning! Calling compact_data() when data_compacted_ is true!"
                       << std::endl;
    return;
  }

  if (delete_set_->size() > 0) {
    throw diskann_exception_t("Can not compact data when index has non-empty delete_set_ of "
                              "size: " +
                                  std::to_string(delete_set_->size()),
                              -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  powerlaw_ann::timer_t timer;

  std::vector<uint32_t> new_location =
      std::vector<uint32_t>(max_points_ + num_frozen_pts_, UINT32_MAX);

  uint32_t new_counter = 0;
  std::set<uint32_t> empty_locations;
  for (uint32_t old_location = 0; old_location < max_points_; old_location++) {
    if (location_to_tag_.contains(old_location)) {
      new_location[old_location] = new_counter;
      new_counter++;
    } else {
      empty_locations.insert(old_location);
    }
  }
  for (uint32_t old_location = (uint32_t) max_points_; old_location < max_points_ + num_frozen_pts_;
       old_location++) {
    new_location[old_location] = old_location;
  }

  // If start node is removed, throw an exception
  if (start_ < max_points_ && !location_to_tag_.contains(start_)) {
    throw powerlaw_ann::diskann_exception_t("ERROR: Start node deleted.", -1, __FUNCSIG__, __FILE__,
                                            __LINE__);
  }

  size_t num_dangling = 0;
  for (uint32_t old = 0; old < max_points_ + num_frozen_pts_; ++old) {
    // compact final_graph_
    std::vector<uint32_t> new_adj_list;

    if ((new_location[old] < max_points_) // If point continues to exist
        || (old >= max_points_ && old < max_points_ + num_frozen_pts_)) {
      new_adj_list.reserve(graph_store_->get_neighbours((location_t) old).size());
      for (auto ngh_iter : graph_store_->get_neighbours((location_t) old)) {
        if (empty_locations.find(ngh_iter) != empty_locations.end()) {
          ++num_dangling;
          powerlaw_ann::cerr << "Error in compact_data(). final_graph_[" << old << "] has neighbor "
                             << ngh_iter << " which is a location not associated with any tag."
                             << std::endl;
        } else {
          new_adj_list.push_back(new_location[ngh_iter]);
        }
      }
      // graph_store_->get_neighbours((location_t)old).swap(new_adj_list);
      graph_store_->set_neighbours((location_t) old, new_adj_list);

      // Move the data and adj list to the correct position
      if (new_location[old] != old) {
        assert(new_location[old] < old);
        graph_store_->swap_neighbours(new_location[old], (location_t) old);

        if (filtered_index_) {
          location_to_labels_[new_location[old]].swap(location_to_labels_[old]);
        }

        data_store_->copy_vectors(old, new_location[old], 1);
      }
    } else {
      graph_store_->clear_neighbours((location_t) old);
    }
  }
  powerlaw_ann::cerr << "#dangling references after data compaction: " << num_dangling << std::endl;

  tag_to_location_.clear();
  for (auto pos = location_to_tag_.find_first(); pos.is_valid();
       pos = location_to_tag_.find_next(pos)) {
    const auto tag = location_to_tag_.get(pos);
    tag_to_location_[tag] = new_location[pos.key_];
  }
  location_to_tag_.clear();
  for (const auto& iter : tag_to_location_) {
    location_to_tag_.set(iter.second, iter.first);
  }
  // remove all cleared up old
  for (size_t old = nd_; old < max_points_; ++old) {
    graph_store_->clear_neighbours((location_t) old);
  }
  if (filtered_index_) {
    for (size_t old = nd_; old < max_points_; old++) {
      location_to_labels_[old].clear();
    }
  }

  empty_slots_.clear();
  // mark all slots after nd_ as empty
  for (auto i = nd_; i < max_points_; i++) {
    empty_slots_.insert((uint32_t) i);
  }
  data_compacted_ = true;
  powerlaw_ann::cout << "Time taken for compact_data: " << timer.elapsed() / 1000000. << "s."
                     << std::endl;
}

//
// Caller must hold unique tag_lock_ and delete_lock_ before calling this
//
template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::reserve_location() {
  if (nd_ >= max_points_) {
    return -1;
  }
  uint32_t location;
  if (data_compacted_ && empty_slots_.is_empty()) {
    // This code path is encountered when enable_delete hasn't been
    // called yet, so no points have been deleted and empty_slots_
    // hasn't been filled in. In that case, just keep assigning
    // consecutive locations.
    location = (uint32_t) nd_;
  } else {
    assert(empty_slots_.size() != 0);
    assert(empty_slots_.size() + nd_ == max_points_);

    location = empty_slots_.pop_any();
    delete_set_->erase(location);
  }
  ++nd_;
  return location;
}

template <typename T, typename tag_t, typename label_t>
size_t vamana_index_t<T, tag_t, label_t>::release_location(int location) {
  if (empty_slots_.is_in_set(location))
    throw diskann_exception_t("Trying to release location, but location already in empty slots", -1,
                              __FUNCSIG__, __FILE__, __LINE__);
  empty_slots_.insert(location);

  nd_--;
  return nd_;
}

template <typename T, typename tag_t, typename label_t>
size_t
vamana_index_t<T, tag_t, label_t>::release_locations(const tsl::robin_set<uint32_t>& locations) {
  for (auto location : locations) {
    if (empty_slots_.is_in_set(location))
      throw diskann_exception_t("Trying to release location, but location "
                                "already in empty slots",
                                -1, __FUNCSIG__, __FILE__, __LINE__);
    empty_slots_.insert(location);

    nd_--;
  }

  if (empty_slots_.size() + nd_ != max_points_)
    throw diskann_exception_t("#empty slots + nd != max points", -1, __FUNCSIG__, __FILE__,
                              __LINE__);

  return nd_;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::reposition_points(uint32_t old_location_start,
                                                          uint32_t new_location_start,
                                                          uint32_t num_locations) {
  if (num_locations == 0 || old_location_start == new_location_start) {
    return;
  }

  // Update pointers to the moved nodes. Note: the computation is correct even
  // when new_location_start < old_location_start given the C++ uint32_t
  // integer arithmetic rules.
  const uint32_t location_delta = new_location_start - old_location_start;

  std::vector<location_t> updated_neighbours_location;
  for (uint32_t i = 0; i < max_points_ + num_frozen_pts_; i++) {
    auto& i_neighbours = graph_store_->get_neighbours((location_t) i);
    std::vector<location_t> i_neighbours_copy(i_neighbours.begin(), i_neighbours.end());
    for (auto& loc : i_neighbours_copy) {
      if (loc >= old_location_start && loc < old_location_start + num_locations)
        loc += location_delta;
    }
    graph_store_->set_neighbours(i, i_neighbours_copy);
  }

  // The [start, end) interval which will contain obsolete points to be
  // cleared.
  uint32_t mem_clear_loc_start = old_location_start;
  uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

  // Move the adjacency lists. Make sure that overlapping ranges are handled
  // correctly.
  if (new_location_start < old_location_start) {
    // New location before the old location: copy the entries in order
    // to avoid modifying locations that are yet to be copied.
    for (uint32_t loc_offset = 0; loc_offset < num_locations; loc_offset++) {
      assert(graph_store_->get_neighbours(new_location_start + loc_offset).empty());
      graph_store_->swap_neighbours(new_location_start + loc_offset,
                                    old_location_start + loc_offset);
      if (dynamic_index_ && filtered_index_) {
        location_to_labels_[new_location_start + loc_offset].swap(
            location_to_labels_[old_location_start + loc_offset]);
      }
    }
    // If ranges are overlapping, make sure not to clear the newly copied
    // data.
    if (mem_clear_loc_start < new_location_start + num_locations) {
      // Clear only after the end of the new range.
      mem_clear_loc_start = new_location_start + num_locations;
    }
  } else {
    // Old location after the new location: copy from the end of the range
    // to avoid modifying locations that are yet to be copied.
    for (uint32_t loc_offset = num_locations; loc_offset > 0; loc_offset--) {
      assert(graph_store_->get_neighbours(new_location_start + loc_offset - 1u).empty());
      graph_store_->swap_neighbours(new_location_start + loc_offset - 1u,
                                    old_location_start + loc_offset - 1u);
      if (dynamic_index_ && filtered_index_) {
        location_to_labels_[new_location_start + loc_offset - 1u].swap(
            location_to_labels_[old_location_start + loc_offset - 1u]);
      }
    }

    // If ranges are overlapping, make sure not to clear the newly copied
    // data.
    if (mem_clear_loc_end_limit > new_location_start) {
      // Clear only up to the beginning of the new range.
      mem_clear_loc_end_limit = new_location_start;
    }
  }
  data_store_->move_vectors(old_location_start, new_location_start, num_locations);
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::reposition_frozen_point_to_end() {
  if (num_frozen_pts_ == 0)
    return;

  if (nd_ == max_points_) {
    powerlaw_ann::cout << "Not repositioning frozen point as it is already at the end."
                       << std::endl;
    return;
  }

  reposition_points((uint32_t) nd_, (uint32_t) max_points_, (uint32_t) num_frozen_pts_);
  start_ = (uint32_t) max_points_;

  // update medoid id's as frozen points are treated as medoid
  if (filtered_index_ && dynamic_index_) {
    for (auto& [label, medoid_id] : label_to_start_id_) {
      /*if (label == universal_label_)
          continue;*/
      label_to_start_id_[label] = (uint32_t) max_points_ + (medoid_id - (uint32_t) nd_);
    }
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::resize(size_t new_max_points) {
  const size_t new_internal_points = new_max_points + num_frozen_pts_;
  auto start = std::chrono::high_resolution_clock::now();
  assert(empty_slots_.size() == 0); // should not resize if there are empty slots.

  data_store_->resize((location_t) new_internal_points);
  graph_store_->resize_graph(new_internal_points);
  locks_ = std::vector<non_recursive_mutex_t>(new_internal_points);

  if (num_frozen_pts_ != 0) {
    reposition_points((uint32_t) max_points_, (uint32_t) new_max_points,
                      (uint32_t) num_frozen_pts_);
    start_ = (uint32_t) new_max_points;
  }

  max_points_ = new_max_points;
  empty_slots_.reserve(max_points_);
  for (auto i = nd_; i < max_points_; i++) {
    empty_slots_.insert((uint32_t) i);
  }

  auto stop = std::chrono::high_resolution_clock::now();
  powerlaw_ann::cout << "Resizing took: " << std::chrono::duration<double>(stop - start).count()
                     << "s" << std::endl;
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::insert_point_impl(const data_type_t& point,
                                                         const tag_type_t tag) {
  try {
    return this->insert_point(std::any_cast<const T*>(point), std::any_cast<const tag_t>(tag));
  } catch (const std::bad_any_cast& anycast_e) {
    throw new diskann_exception_t(
        "Error:Trying to insert invalid data type" + std::string(anycast_e.what()), -1);
  } catch (const std::exception& e) {
    throw new diskann_exception_t("Error:" + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::insert_point_impl(const data_type_t& point,
                                                         const tag_type_t tag,
                                                         label_vector_t& labels) {
  try {
    return this->insert_point(std::any_cast<const T*>(point), std::any_cast<const tag_t>(tag),
                              labels.get<const std::vector<label_t>>());
  } catch (const std::bad_any_cast& anycast_e) {
    throw new diskann_exception_t(
        "Error:Trying to insert invalid data type" + std::string(anycast_e.what()), -1);
  } catch (const std::exception& e) {
    throw new diskann_exception_t("Error:" + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::insert_point(const T* point, const tag_t tag) {
  std::vector<label_t> no_labels{0};
  return insert_point(point, tag, no_labels);
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::insert_point(const T* point, const tag_t tag,
                                                    const std::vector<label_t>& labels) {

  assert(has_built_);
  if (tag == 0) {
    throw powerlaw_ann::diskann_exception_t("Do not insert point with tag 0. That is "
                                            "reserved for points hidden "
                                            "from the user.",
                                            -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  std::shared_lock<std::shared_timed_mutex> shared_ul(update_lock_);
  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  std::unique_lock<std::shared_timed_mutex> dl(delete_lock_);

  auto location = reserve_location();
  if (filtered_index_) {
    if (labels.empty()) {
      release_location(location);
      std::cerr << "Error: Can't insert point with tag " + get_tag_string(tag) +
                       " . there are no labels for the point."
                << std::endl;
      return -1;
    }

    location_to_labels_[location] = labels;

    for (label_t label : labels) {
      if (labels_.find(label) == labels_.end()) {
        if (frozen_pts_used_ >= num_frozen_pts_) {
          throw diskann_exception_t("Error: For dynamic filtered index, the number of frozen "
                                    "points should be atleast equal "
                                    "to number of unique labels.",
                                    -1);
        }

        auto fz_location = (int) (max_points_) + frozen_pts_used_; // as first fz_point_
        labels_.insert(label);
        label_to_start_id_[label] = (uint32_t) fz_location;
        location_to_labels_[fz_location] = {label};
        data_store_->set_vector((location_t) fz_location, point);
        frozen_pts_used_++;
      }
    }
  }

  if (location == -1) {
#if EXPAND_IF_FULL
    dl.unlock();
    tl.unlock();
    shared_ul.unlock();

    {
      std::unique_lock<std::shared_timed_mutex> ul(update_lock_);
      tl.lock();
      dl.lock();

      if (nd_ >= max_points_) {
        auto new_max_points = (size_t) (max_points_ * INDEX_GROWTH_FACTOR);
        resize(new_max_points);
      }

      dl.unlock();
      tl.unlock();
      ul.unlock();
    }

    shared_ul.lock();
    tl.lock();
    dl.lock();

    location = reserve_location();
    if (location == -1) {
      throw powerlaw_ann::diskann_exception_t("Cannot reserve location even after "
                                              "expanding graph. Terminating.",
                                              -1, __FUNCSIG__, __FILE__, __LINE__);
    }
#else
    return -1;
#endif
  } // cant insert as active pts >= max_pts
  dl.unlock();

  // Insert tag and mapping to location
  if (enable_tags_) {
    // if tags are enabled and tag is already inserted. so we can't reuse that tag.
    if (tag_to_location_.find(tag) != tag_to_location_.end()) {
      release_location(location);
      return -1;
    }

    tag_to_location_[tag] = location;
    location_to_tag_.set(location, tag);
  }
  tl.unlock();

  data_store_->set_vector(location, point); // update datastore

  // Find and add appropriate graph edges
  scratch_store_manager_t<in_mem_query_scratch_t<T>> manager(query_scratch_);
  auto scratch = manager.scratch_space();
  std::vector<uint32_t> pruned_list; // it is the set best candidates to connect to this point
  if (filtered_index_) {
    // when filtered the best_candidates will share the same label ( label_present > distance)
    search_for_point_and_prune(location, indexing_queue_size_, pruned_list, scratch, true,
                               filter_indexing_queue_size_);
  } else {
    search_for_point_and_prune(location, indexing_queue_size_, pruned_list, scratch);
  }
  assert(pruned_list.size() >
         0); // should find atleast one neighbour (i.e frozen point acting as medoid)

  {
    std::shared_lock<std::shared_timed_mutex> tlock(tag_lock_, std::defer_lock);
    if (conc_consolidate_)
      tlock.lock();

    lock_guard_t guard(locks_[location]);
    graph_store_->clear_neighbours(location);

    std::vector<uint32_t> neighbor_links;
    for (auto link : pruned_list) {
      if (conc_consolidate_)
        if (!location_to_tag_.contains(link))
          continue;
      neighbor_links.emplace_back(link);
    }
    graph_store_->set_neighbours(location, neighbor_links);
    assert(graph_store_->get_neighbours(location).size() <= indexing_range_);

    if (conc_consolidate_)
      tlock.unlock();
  }

  inter_insert(location, pruned_list, scratch);

  return 0;
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::lazy_delete_impl(const tag_type_t& tag) {
  try {
    return lazy_delete(std::any_cast<const tag_t>(tag));
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t(std::string("Error: ") + e.what(), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::lazy_delete_impl(tag_vector_t& tags,
                                                         tag_vector_t& failed_tags) {
  try {
    this->lazy_delete(tags.get<const std::vector<tag_t>>(), failed_tags.get<std::vector<tag_t>>());
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t(
        "Error: bad any cast while performing lazy_delete_impl() " + std::string(e.what()), -1);
  } catch (const std::exception& e) {
    throw diskann_exception_t("Error: " + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
int vamana_index_t<T, tag_t, label_t>::lazy_delete(const tag_t& tag) {
  std::shared_lock<std::shared_timed_mutex> ul(update_lock_);
  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  std::unique_lock<std::shared_timed_mutex> dl(delete_lock_);
  data_compacted_ = false;

  if (tag_to_location_.find(tag) == tag_to_location_.end()) {
    powerlaw_ann::cerr << "Delete tag not found " << get_tag_string(tag) << std::endl;
    return -1;
  }
  assert(tag_to_location_[tag] < max_points_);

  const auto location = tag_to_location_[tag];
  delete_set_->insert(location);
  location_to_tag_.erase(location);
  tag_to_location_.erase(tag);
  return 0;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::lazy_delete(const std::vector<tag_t>& tags,
                                                    std::vector<tag_t>& failed_tags) {
  if (failed_tags.size() > 0) {
    throw diskann_exception_t("failed_tags should be passed as an empty list", -1, __FUNCSIG__,
                              __FILE__, __LINE__);
  }
  std::shared_lock<std::shared_timed_mutex> ul(update_lock_);
  std::unique_lock<std::shared_timed_mutex> tl(tag_lock_);
  std::unique_lock<std::shared_timed_mutex> dl(delete_lock_);
  data_compacted_ = false;

  for (auto tag : tags) {
    if (tag_to_location_.find(tag) == tag_to_location_.end()) {
      failed_tags.push_back(tag);
    } else {
      const auto location = tag_to_location_[tag];
      delete_set_->insert(location);
      location_to_tag_.erase(location);
      tag_to_location_.erase(tag);
    }
  }
}

template <typename T, typename tag_t, typename label_t>
bool vamana_index_t<T, tag_t, label_t>::is_index_saved() {
  return is_saved_;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::get_active_tags_impl(tag_robin_set_t& active_tags) {
  try {
    this->get_active_tags(active_tags.get<tsl::robin_set<tag_t>>());
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t(
        "Error: bad_any cast while performing get_active_tags_impl() " + std::string(e.what()), -1);
  } catch (const std::exception& e) {
    throw diskann_exception_t("Error :" + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::get_active_tags(tsl::robin_set<tag_t>& active_tags) {
  active_tags.clear();
  std::shared_lock<std::shared_timed_mutex> tl(tag_lock_);
  for (auto iter : tag_to_location_) {
    active_tags.insert(iter.first);
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::print_status() {
  std::shared_lock<std::shared_timed_mutex> ul(update_lock_);
  std::shared_lock<std::shared_timed_mutex> cl(consolidate_lock_);
  std::shared_lock<std::shared_timed_mutex> tl(tag_lock_);
  std::shared_lock<std::shared_timed_mutex> dl(delete_lock_);

  powerlaw_ann::cout << "------------------- vamana_index_t object: " << (uint64_t) this
                     << " -------------------" << std::endl;
  powerlaw_ann::cout << "Number of points: " << nd_ << std::endl;
  powerlaw_ann::cout << "Graph size: " << graph_store_->get_total_points() << std::endl;
  powerlaw_ann::cout << "Location to tag size: " << location_to_tag_.size() << std::endl;
  powerlaw_ann::cout << "Tag to location size: " << tag_to_location_.size() << std::endl;
  powerlaw_ann::cout << "Number of empty slots: " << empty_slots_.size() << std::endl;
  powerlaw_ann::cout << std::boolalpha << "Data compacted: " << this->data_compacted_ << std::endl;
  powerlaw_ann::cout << "---------------------------------------------------------"
                        "------------"
                     << std::endl;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::count_nodes_at_bfs_levels() {
  std::unique_lock<std::shared_timed_mutex> ul(update_lock_);

  boost::dynamic_bitset<> visited(max_points_ + num_frozen_pts_);

  size_t MAX_BFS_LEVELS = 32;
  auto bfs_sets = new tsl::robin_set<uint32_t>[MAX_BFS_LEVELS];

  bfs_sets[0].insert(start_);
  visited.set(start_);

  for (uint32_t i = (uint32_t) max_points_; i < max_points_ + num_frozen_pts_; ++i) {
    if (i != start_) {
      bfs_sets[0].insert(i);
      visited.set(i);
    }
  }

  for (size_t l = 0; l < MAX_BFS_LEVELS - 1; ++l) {
    powerlaw_ann::cout << "Number of nodes at BFS level " << l << " is " << bfs_sets[l].size()
                       << std::endl;
    if (bfs_sets[l].size() == 0)
      break;
    for (auto node : bfs_sets[l]) {
      for (auto nghbr : graph_store_->get_neighbours((location_t) node)) {
        if (!visited.test(nghbr)) {
          visited.set(nghbr);
          bfs_sets[l + 1].insert(nghbr);
        }
      }
    }
  }

  delete[] bfs_sets;
}

// REFACTOR: This should be an optimized_data_store_t class
template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::optimize_index_layout() { // use after build or load
  if (dynamic_index_) {
    throw powerlaw_ann::diskann_exception_t(
        "Optimize_index_layout not implemented for dyanmic indices", -1, __FUNCSIG__, __FILE__,
        __LINE__);
  }

  float* cur_vec = new float[data_store_->get_aligned_dim()];
  std::memset(cur_vec, 0, data_store_->get_aligned_dim() * sizeof(float));
  data_len_ = (data_store_->get_aligned_dim() + 1) * sizeof(float);
  neighbor_len_ = (graph_store_->get_max_observed_degree() + 1) * sizeof(uint32_t);
  node_size_ = data_len_ + neighbor_len_;
  opt_graph_ = new char[node_size_ * nd_];
  auto dist_fast = (fast_l2_distance_t<T>*) (data_store_->get_dist_fn());
  for (uint32_t i = 0; i < nd_; i++) {
    char* cur_node_offset = opt_graph_ + i * node_size_;
    data_store_->get_vector(i, (T*) cur_vec);
    float cur_norm = dist_fast->norm((T*) cur_vec, (uint32_t) data_store_->get_aligned_dim());
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), cur_vec, data_len_ - sizeof(float));

    cur_node_offset += data_len_;
    uint32_t k = (uint32_t) graph_store_->get_neighbours(i).size();
    std::memcpy(cur_node_offset, &k, sizeof(uint32_t));
    std::memcpy(cur_node_offset + sizeof(uint32_t), graph_store_->get_neighbours(i).data(),
                k * sizeof(uint32_t));
    // std::vector<uint32_t>().swap(graph_store_->get_neighbours(i));
    graph_store_->clear_neighbours(i);
  }
  graph_store_->clear_graph();
  graph_store_->resize_graph(0);
  delete[] cur_vec;
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::search_with_optimized_layout_impl(const data_type_t& query,
                                                                          size_t K, size_t L,
                                                                          uint32_t* indices) {
  try {
    return this->search_with_optimized_layout(std::any_cast<const T*>(query), K, L, indices);
  } catch (const std::bad_any_cast& e) {
    throw diskann_exception_t("Error: bad any cast while performing "
                              "search_with_optimized_layout_impl() " +
                                  std::string(e.what()),
                              -1);
  } catch (const std::exception& e) {
    throw diskann_exception_t("Error: " + std::string(e.what()), -1);
  }
}

template <typename T, typename tag_t, typename label_t>
void vamana_index_t<T, tag_t, label_t>::search_with_optimized_layout(const T* query, size_t K,
                                                                     size_t L, uint32_t* indices) {
  fast_l2_distance_t<T>* dist_fast = (fast_l2_distance_t<T>*) (data_store_->get_dist_fn());

  neighbor_priority_queue_t retset(L);
  std::vector<uint32_t> init_ids(L);

  boost::dynamic_bitset<> flags{nd_, 0};
  uint32_t tmp_l = 0;
  uint32_t* neighbors = (uint32_t*) (opt_graph_ + node_size_ * start_ + data_len_);
  uint32_t max_m_ep = *neighbors;
  neighbors++;

  for (; tmp_l < L && tmp_l < max_m_ep; tmp_l++) {
    init_ids[tmp_l] = neighbors[tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    uint32_t id = rand() % nd_;
    if (flags[id])
      continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (uint32_t i = 0; i < init_ids.size(); i++) {
    uint32_t id = init_ids[i];
    if (id >= nd_)
      continue;
    cpu_prefetch_t0(opt_graph_ + node_size_ * id);
  }
  L = 0;
  for (uint32_t i = 0; i < init_ids.size(); i++) {
    uint32_t id = init_ids[i];
    if (id >= nd_)
      continue;
    T* x = (T*) (opt_graph_ + node_size_ * id);
    float norm_x = *x;
    x++;
    float dist = dist_fast->compare(x, query, norm_x, (uint32_t) data_store_->get_aligned_dim());
    retset.insert(neighbor_t(id, dist));
    flags[id] = true;
    L++;
  }

  while (retset.has_unexpanded_node()) {
    auto nbr = retset.closest_unexpanded();
    auto n = nbr.id;
    cpu_prefetch_t0(opt_graph_ + node_size_ * n + data_len_);
    neighbors = (uint32_t*) (opt_graph_ + node_size_ * n + data_len_);
    uint32_t max_m = *neighbors;
    neighbors++;
    for (uint32_t m = 0; m < max_m; ++m)
      cpu_prefetch_t0(opt_graph_ + node_size_ * neighbors[m]);
    for (uint32_t m = 0; m < max_m; ++m) {
      uint32_t id = neighbors[m];
      if (flags[id])
        continue;
      flags[id] = 1;
      T* data = (T*) (opt_graph_ + node_size_ * id);
      float norm = *data;
      data++;
      float dist = dist_fast->compare(query, data, norm, (uint32_t) data_store_->get_aligned_dim());
      neighbor_t nn(id, dist);
      retset.insert(nn);
    }
  }

  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

/*  Internals of the library */
template <typename T, typename tag_t, typename label_t>
const float vamana_index_t<T, tag_t, label_t>::INDEX_GROWTH_FACTOR = 1.5f;

// EXPORTS
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, uint32_t, uint32_t>;
#if 0
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, int32_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, int32_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, int32_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, uint32_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, uint32_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, uint32_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, int64_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, int64_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, int64_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, uint64_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, uint64_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, uint64_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, tag_uint128_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, tag_uint128_t, uint32_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, tag_uint128_t, uint32_t>;
// Label with short int 2 byte
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, int32_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, int32_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, int32_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, uint32_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, uint32_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, uint32_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, int64_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, int64_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, int64_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, uint64_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, uint64_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, uint64_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<float, tag_uint128_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<int8_t, tag_uint128_t, uint16_t>;
template POWERLAWANN_DLLEXPORT class vamana_index_t<uint8_t, tag_uint128_t, uint16_t>;

template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint64_t, uint32_t>::search<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint64_t, uint32_t>::search<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint64_t, uint32_t>::search<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint64_t, uint32_t>::search<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint64_t, uint32_t>::search<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint64_t, uint32_t>::search<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
// tag_t==uint32_t
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint32_t, uint32_t>::search<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint32_t, uint32_t>::search<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint32_t, uint32_t>::search<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint32_t, uint32_t>::search<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint32_t, uint32_t>::search<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint32_t, uint32_t>::search<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);

template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint64_t, uint32_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint64_t, uint32_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
// tag_t==uint32_t
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const float *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint32_t, uint32_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint32_t, uint32_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint32_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);

template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint64_t, uint16_t>::search<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint64_t, uint16_t>::search<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint64_t, uint16_t>::search<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint64_t, uint16_t>::search<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint64_t, uint16_t>::search<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint64_t, uint16_t>::search<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
// tag_t==uint32_t
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint32_t, uint16_t>::search<uint64_t>(
    const float *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint32_t, uint16_t>::search<uint32_t>(
    const float *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint32_t, uint16_t>::search<uint64_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint32_t, uint16_t>::search<uint32_t>(
    const uint8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint32_t, uint16_t>::search<uint64_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint64_t *indices, float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint32_t, uint16_t>::search<uint32_t>(
    const int8_t *query, const size_t K, const uint32_t L, uint32_t *indices, float *distances);

template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint64_t, uint16_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint64_t, uint16_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
// tag_t==uint32_t
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<float, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const float *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<uint8_t, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const uint8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint32_t, uint16_t>::search_with_filters<
    uint64_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint64_t *indices,
              float *distances);
template POWERLAWANN_DLLEXPORT std::pair<uint32_t, uint32_t> vamana_index_t<int8_t, uint32_t, uint16_t>::search_with_filters<
    uint32_t>(const int8_t *query, const uint16_t &filter_label, const size_t K, const uint32_t L, uint32_t *indices,
              float *distances);
#endif

} // namespace powerlaw_ann
