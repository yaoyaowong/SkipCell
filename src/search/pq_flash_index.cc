#include "search/pq_flash_index.h"

#include "common/arch_compat.h"
#include "common/defaults.h"
#include "common/diskann_exception.h"
#include "common/timer.h"
#include "common/utils.h"
#include "pq/pq_common.h"
#include "pq/pq_scratch.h"
#if defined(__linux__)
#include "storage/io_uring_aligned_file_reader.h"
#include "storage/linux_aligned_file_reader.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace powerlaw_ann {
namespace {

inline constexpr uint64_t k_full_precision_reorder_multiplier = 3;
inline constexpr size_t k_pipeann_convergence_marker = 5;
inline constexpr double k_pipeann_waste_threshold = 0.1;

struct disabled_beam_phase_profile_t {};

} // namespace

template <typename data_t>
pq_flash_index_t<data_t>::pq_flash_index_t(std::shared_ptr<aligned_file_reader_t> reader,
                                           metric_t metric)
    : reader_(std::move(reader)), metric_(metric) {
  if (reader_ == nullptr) {
    throw diskann_exception_t("pq_flash_index_t requires an aligned file reader", -1);
  }
  if (metric_ != metric_t::L2) {
    throw diskann_exception_t("Step 8 supports only the static float32 L2 baseline", -1);
  }
  distance_comparator_.reset(get_distance_function<data_t>(metric_));
  float_distance_comparator_.reset(get_distance_function<float>(metric_));
}

template <typename data_t> pq_flash_index_t<data_t>::~pq_flash_index_t() {
  delete[] data_;
  delete[] neighborhood_cache_buffer_;
  delete[] medoids_;
  if (coordinate_cache_buffer_ != nullptr) {
    aligned_free(coordinate_cache_buffer_);
  }
  if (centroid_data_ != nullptr) {
    aligned_free(centroid_data_);
  }
  if (is_loaded_) {
    while (!thread_data_.empty()) {
      delete thread_data_.pop();
    }
    reader_->deregister_all_threads();
    reader_->close();
  }
}

template <typename data_t>
uint64_t pq_flash_index_t<data_t>::get_node_sector(uint64_t node_id) const {
  return 1 + (nodes_per_sector_ > 0 ? node_id / nodes_per_sector_
                                    : node_id * DIV_ROUND_UP(max_node_len_, defaults::SECTOR_LEN));
}

template <typename data_t>
char* pq_flash_index_t<data_t>::offset_to_node(char* sector_buffer, uint64_t node_id) const {
  return sector_buffer +
         (nodes_per_sector_ == 0 ? 0 : (node_id % nodes_per_sector_) * max_node_len_);
}

template <typename data_t>
uint32_t* pq_flash_index_t<data_t>::offset_to_node_neighborhood(char* node_buffer) const {
  return reinterpret_cast<uint32_t*>(node_buffer + disk_bytes_per_point_);
}

template <typename data_t>
data_t* pq_flash_index_t<data_t>::offset_to_node_coords(char* node_buffer) const {
  return reinterpret_cast<data_t*>(node_buffer);
}

template <typename data_t>
void pq_flash_index_t<data_t>::setup_thread_data(uint64_t num_threads, uint64_t visited_reserve,
                                                 uint64_t candidate_capacity) {
  cout << "Setting up thread-specific contexts for nthreads: " << num_threads << std::endl;
#pragma omp parallel for num_threads(static_cast <int>(num_threads))
  for (int64_t thread = 0; thread < static_cast<int64_t>(num_threads); ++thread) {
#pragma omp critical
    {
      auto* thread_data =
          new ssd_thread_data_t<data_t>(aligned_dim_, visited_reserve, candidate_capacity);
      reader_->register_thread();
      thread_data->ctx = reader_->get_ctx();
      thread_data_.push(thread_data);
    }
  }
  is_loaded_ = true;
}

template <typename data_t>
std::vector<bool> pq_flash_index_t<data_t>::read_nodes(
    const std::vector<uint32_t>& node_ids, std::vector<data_t*>& coord_buffers,
    std::vector<std::pair<uint32_t, uint32_t*>>& neighbor_buffers) {
  if (node_ids.size() != coord_buffers.size() || node_ids.size() != neighbor_buffers.size()) {
    throw diskann_exception_t("read_nodes buffer counts do not match node count", -1);
  }
  std::vector<bool> read_status(node_ids.size(), true);
  if (node_ids.empty()) {
    return read_status;
  }

  const uint64_t num_sectors =
      io_index_ != nullptr
          ? 1
          : (nodes_per_sector_ > 0 ? 1 : DIV_ROUND_UP(max_node_len_, defaults::SECTOR_LEN));
  char* buffer = nullptr;
  alloc_aligned(reinterpret_cast<void**>(&buffer),
                node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

  try {
    std::vector<aligned_read_t> read_requests;
    read_requests.reserve(node_ids.size());
    for (size_t i = 0; i < node_ids.size(); ++i) {
      const uint64_t sector = io_index_ != nullptr ? io_index_->vector_location(node_ids[i]).page_id
                                                   : get_node_sector(node_ids[i]);
      read_requests.emplace_back(sector * defaults::SECTOR_LEN, num_sectors * defaults::SECTOR_LEN,
                                 buffer + i * num_sectors * defaults::SECTOR_LEN);
    }

    scratch_store_manager_t<ssd_thread_data_t<data_t>> manager(thread_data_);
    auto* thread_data = manager.scratch_space();
    reader_->read(read_requests, thread_data->ctx);

    for (size_t i = 0; i < read_requests.size(); ++i) {
      char* node_buffer =
          io_index_ != nullptr
              ? static_cast<char*>(read_requests[i].buf) +
                    io_index_->vector_location(node_ids[i]).page_offset
              : offset_to_node(static_cast<char*>(read_requests[i].buf), node_ids[i]);
      if (coord_buffers[i] != nullptr) {
        const void* coordinate_data =
            io_index_ != nullptr ? static_cast<const void*>(node_buffer)
                                 : static_cast<const void*>(offset_to_node_coords(node_buffer));
        std::memcpy(coord_buffers[i], coordinate_data, disk_bytes_per_point_);
      }
      if (neighbor_buffers[i].second != nullptr) {
        const auto io_neighbors =
            io_index_ != nullptr ? io_index_->neighbors(node_ids[i]) : std::span<const uint32_t>();
        uint32_t* neighborhood =
            io_index_ != nullptr ? nullptr : offset_to_node_neighborhood(node_buffer);
        const uint32_t num_neighbors =
            io_index_ != nullptr ? static_cast<uint32_t>(io_neighbors.size()) : *neighborhood;
        if (num_neighbors > max_degree_) {
          throw diskann_exception_t("Disk node degree exceeds index metadata", -1);
        }
        neighbor_buffers[i].first = num_neighbors;
        std::memcpy(neighbor_buffers[i].second,
                    io_index_ != nullptr ? io_neighbors.data() : neighborhood + 1,
                    num_neighbors * sizeof(uint32_t));
      }
    }
  } catch (...) {
    aligned_free(buffer);
    throw;
  }

  aligned_free(buffer);
  return read_status;
}

template <typename data_t>
void pq_flash_index_t<data_t>::load_cache_list(const std::vector<uint32_t>& node_list) {
  delete[] neighborhood_cache_buffer_;
  neighborhood_cache_buffer_ = nullptr;
  if (coordinate_cache_buffer_ != nullptr) {
    aligned_free(coordinate_cache_buffer_);
    coordinate_cache_buffer_ = nullptr;
  }
  neighborhood_cache_.clear();
  coordinate_cache_.clear();

  if (node_list.empty()) {
    return;
  }

  cout << "Loading the cache list into memory..." << std::flush;
  neighborhood_cache_buffer_ = new uint32_t[node_list.size() * (max_degree_ + 1)]{};
  alloc_aligned(reinterpret_cast<void**>(&coordinate_cache_buffer_),
                node_list.size() * aligned_dim_ * sizeof(data_t), 8 * sizeof(data_t));
  std::memset(coordinate_cache_buffer_, 0, node_list.size() * aligned_dim_ * sizeof(data_t));

  constexpr size_t k_block_size = 8;
  for (size_t start = 0; start < node_list.size(); start += k_block_size) {
    const size_t end = std::min(node_list.size(), start + k_block_size);
    std::vector<uint32_t> nodes_to_read;
    std::vector<data_t*> coord_buffers;
    std::vector<std::pair<uint32_t, uint32_t*>> neighbor_buffers;
    for (size_t node_index = start; node_index < end; ++node_index) {
      nodes_to_read.push_back(node_list[node_index]);
      coord_buffers.push_back(coordinate_cache_buffer_ + node_index * aligned_dim_);
      neighbor_buffers.emplace_back(0, neighborhood_cache_buffer_ + node_index * (max_degree_ + 1));
    }

    const auto status = read_nodes(nodes_to_read, coord_buffers, neighbor_buffers);
    for (size_t i = 0; i < status.size(); ++i) {
      if (status[i]) {
        coordinate_cache_.insert({nodes_to_read[i], coord_buffers[i]});
        neighborhood_cache_.insert({nodes_to_read[i], neighbor_buffers[i]});
      }
    }
  }
  cout << "done." << std::endl;
}

template <typename data_t>
void pq_flash_index_t<data_t>::cache_bfs_levels(uint64_t num_nodes_to_cache,
                                                std::vector<uint32_t>& node_list) {
  node_list.clear();
  if (num_nodes_to_cache == 0 || num_points_ == 0) {
    return;
  }

  const uint64_t ten_percent = std::max<uint64_t>(1, std::llround(num_points_ * 0.1));
  num_nodes_to_cache = std::min({num_nodes_to_cache, ten_percent, num_points_});

  tsl::robin_set<uint32_t> cached_nodes;
  auto current_level = std::make_unique<tsl::robin_set<uint32_t>>();
  auto previous_level = std::make_unique<tsl::robin_set<uint32_t>>();
  for (size_t i = 0; i < num_medoids_ && current_level->size() < num_nodes_to_cache; ++i) {
    current_level->insert(medoids_[i]);
  }

  while (cached_nodes.size() + current_level->size() < num_nodes_to_cache &&
         !current_level->empty()) {
    current_level.swap(previous_level);
    current_level->clear();

    std::vector<uint32_t> nodes_to_expand;
    for (uint32_t node_id : *previous_level) {
      if (cached_nodes.find(node_id) == cached_nodes.end()) {
        cached_nodes.insert(node_id);
        nodes_to_expand.push_back(node_id);
      }
    }
    std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

    constexpr size_t k_block_size = 1024;
    bool cache_target_reached = false;
    for (size_t start = 0; start < nodes_to_expand.size() && !cache_target_reached;
         start += k_block_size) {
      const size_t end = std::min(nodes_to_expand.size(), start + k_block_size);
      std::vector<uint32_t> nodes(nodes_to_expand.begin() + start, nodes_to_expand.begin() + end);
      std::vector<data_t*> coord_buffers(nodes.size(), nullptr);
      std::vector<std::unique_ptr<uint32_t[]>> owned_buffers;
      std::vector<std::pair<uint32_t, uint32_t*>> neighbor_buffers;
      owned_buffers.reserve(nodes.size());
      neighbor_buffers.reserve(nodes.size());
      for (size_t i = 0; i < nodes.size(); ++i) {
        owned_buffers.push_back(std::make_unique<uint32_t[]>(max_degree_ + 1));
        neighbor_buffers.emplace_back(0, owned_buffers.back().get());
      }

      const auto status = read_nodes(nodes, coord_buffers, neighbor_buffers);
      for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) {
          continue;
        }
        for (uint32_t j = 0; j < neighbor_buffers[i].first && !cache_target_reached; ++j) {
          const uint32_t neighbor_id = neighbor_buffers[i].second[j];
          if (cached_nodes.find(neighbor_id) == cached_nodes.end()) {
            current_level->insert(neighbor_id);
          }
          if (cached_nodes.size() + current_level->size() >= num_nodes_to_cache) {
            cache_target_reached = true;
          }
        }
      }
    }
  }

  assert(cached_nodes.size() + current_level->size() == num_nodes_to_cache ||
         current_level->empty());
  node_list.reserve(cached_nodes.size() + current_level->size());
  for (uint32_t node_id : cached_nodes) {
    node_list.push_back(node_id);
  }
  for (uint32_t node_id : *current_level) {
    node_list.push_back(node_id);
  }
}

template <typename data_t> void pq_flash_index_t<data_t>::use_medoids_data_as_centroids() {
  if (centroid_data_ != nullptr) {
    aligned_free(centroid_data_);
  }
  alloc_aligned(reinterpret_cast<void**>(&centroid_data_),
                num_medoids_ * aligned_dim_ * sizeof(float), 32);
  std::memset(centroid_data_, 0, num_medoids_ * aligned_dim_ * sizeof(float));

  std::vector<uint32_t> nodes_to_read;
  std::vector<std::unique_ptr<data_t[]>> owned_buffers;
  std::vector<data_t*> medoid_buffers;
  std::vector<std::pair<uint32_t, uint32_t*>> neighbor_buffers;
  for (size_t i = 0; i < num_medoids_; ++i) {
    nodes_to_read.push_back(medoids_[i]);
    owned_buffers.push_back(std::make_unique<data_t[]>(aligned_dim_));
    medoid_buffers.push_back(owned_buffers.back().get());
    neighbor_buffers.emplace_back(0, nullptr);
  }

  const auto status = read_nodes(nodes_to_read, medoid_buffers, neighbor_buffers);
  for (size_t medoid = 0; medoid < num_medoids_; ++medoid) {
    if (!status[medoid]) {
      throw diskann_exception_t("Unable to read a medoid", -1);
    }
    if (use_disk_index_pq_) {
      disk_pq_table_.inflate_vector(reinterpret_cast<uint8_t*>(medoid_buffers[medoid]),
                                    centroid_data_ + medoid * aligned_dim_);
    } else {
      for (size_t dim = 0; dim < data_dim_; ++dim) {
        centroid_data_[medoid * aligned_dim_ + dim] =
            static_cast<float>(medoid_buffers[medoid][dim]);
      }
    }
  }
}

template <typename data_t>
int pq_flash_index_t<data_t>::load(
    uint32_t num_threads, const char* index_prefix,
    std::shared_ptr<const io_optimized_index_t> io_index,
    std::shared_ptr<io_lru_buffer_pool_t> io_cache, bool enable_cell_page_batch,
    bool enable_cell_batch_search, uint32_t cell_batch_max_cached_expansions,
    bool enable_cell_pq_traversal, uint32_t cell_pq_refine_candidates,
    uint32_t cell_pq_refine_prefetch_hop, bool enable_cell_leaf_refinement,
    bool enable_cell_pq_dense_visited, bool enable_cell_pq_filter_visited,
    bool enable_cell_pq_decoded_u8, bool enable_node_prefetch, bool enable_cell_prefetch,
    bool enable_gateway_prefetch, uint32_t gateway_prefetch_max_outstanding,
    bool enable_dynamic_width, uint32_t dynamic_width_initial, uint32_t dynamic_width_marker,
    float dynamic_width_waste_threshold, float cell_pq_decoded_scale) {
  if (is_loaded_) {
    throw diskann_exception_t("pq_flash_index_t cannot load twice", -1);
  }
  if (num_threads == 0) {
    throw diskann_exception_t("Search thread count must be positive", -1);
  }

  const std::string prefix(index_prefix);
  io_index_ = std::move(io_index);
  io_cache_ = std::move(io_cache);
  enable_cell_page_batch_ = enable_cell_page_batch;
  enable_cell_batch_search_ = enable_cell_batch_search;
  cell_batch_max_cached_expansions_ = cell_batch_max_cached_expansions;
  enable_cell_pq_traversal_ = enable_cell_pq_traversal;
  cell_pq_refine_candidates_ = cell_pq_refine_candidates;
  cell_pq_refine_prefetch_hop_ = cell_pq_refine_prefetch_hop;
  enable_cell_leaf_refinement_ = enable_cell_leaf_refinement;
  enable_cell_pq_dense_visited_ = enable_cell_pq_dense_visited;
  enable_cell_pq_filter_visited_ = enable_cell_pq_filter_visited;
  enable_cell_pq_decoded_u8_ = enable_cell_pq_decoded_u8;
  cell_pq_decoded_scale_ = cell_pq_decoded_scale;
  if (!std::isfinite(cell_pq_decoded_scale_) || cell_pq_decoded_scale_ <= 0.0F ||
      (!enable_cell_pq_decoded_u8_ && cell_pq_decoded_scale_ != 1.0F)) {
    throw diskann_exception_t("Decoded PQ scale requires decoded PQ traversal", -1);
  }
  enable_node_prefetch_ = enable_node_prefetch;
  enable_cell_prefetch_ = enable_cell_prefetch;
  enable_gateway_prefetch_ = enable_gateway_prefetch;
  gateway_prefetch_max_outstanding_ = gateway_prefetch_max_outstanding;
  enable_dynamic_width_ = enable_dynamic_width;
  dynamic_width_initial_ = dynamic_width_initial;
  dynamic_width_marker_ = dynamic_width_marker;
  dynamic_width_waste_threshold_ = dynamic_width_waste_threshold;
  if (io_cache_ != nullptr && io_index_ == nullptr) {
    throw diskann_exception_t("IO Buffer Pool requires an IO vector layout", -1);
  }
  if (enable_cell_page_batch_ && io_cache_ == nullptr) {
    throw diskann_exception_t("Cell page batching requires an IO Buffer Pool", -1);
  }
  if (enable_cell_batch_search_ && (io_cache_ == nullptr || !enable_cell_page_batch_)) {
    throw diskann_exception_t("Cell-batched search requires page batching and an IO Buffer Pool",
                              -1);
  }
  if (enable_cell_pq_traversal_ && io_index_ == nullptr) {
    throw diskann_exception_t("Cell-PQ traversal requires an IO topology", -1);
  }
  if (cell_pq_refine_candidates_ != 0 && !enable_cell_pq_traversal_) {
    throw diskann_exception_t("Cell-PQ refinement requires Cell-PQ traversal", -1);
  }
  if (cell_pq_refine_prefetch_hop_ != 0 && cell_pq_refine_candidates_ == 0) {
    throw diskann_exception_t("Cell-PQ prefetch requires refinement candidates", -1);
  }
  if (enable_cell_leaf_refinement_ &&
      (!enable_cell_pq_traversal_ || io_cache_ == nullptr ||
       cell_pq_refine_prefetch_hop_ != 0)) {
    throw diskann_exception_t(
        "Cell-leaf refinement requires Cell-PQ traversal, an IO Buffer Pool, and no prefetch",
        -1);
  }
  if ((enable_node_prefetch_ || enable_cell_prefetch_ || enable_gateway_prefetch_) &&
      io_cache_ == nullptr) {
    throw diskann_exception_t("IO prefetch requires an IO Buffer Pool", -1);
  }
  if (gateway_prefetch_max_outstanding_ == 0 || gateway_prefetch_max_outstanding_ > 2) {
    throw diskann_exception_t("Gateway prefetch outstanding limit must be one or two", -1);
  }
  if (enable_dynamic_width_ &&
      (dynamic_width_initial_ == 0 || dynamic_width_waste_threshold_ < 0.0F ||
       dynamic_width_waste_threshold_ > 1.0F)) {
    throw diskann_exception_t("Dynamic Width parameters are invalid", -1);
  }
  const std::string pq_table_file = prefix + "_pq_pivots.bin";
  const std::string compressed_file = prefix + "_pq_compressed.bin";
  disk_index_file_ = prefix + "_disk.index";

  size_t pq_file_dim = 0;
  size_t pq_file_num_centroids = 0;
  get_bin_metadata(pq_table_file, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);
  if (pq_file_num_centroids != NUM_PQ_CENTROIDS || pq_file_dim == 0) {
    throw diskann_exception_t("Invalid PQ pivot metadata", -1);
  }

  data_dim_ = pq_file_dim;
  disk_bytes_per_point_ = data_dim_ * sizeof(data_t);
  aligned_dim_ = ROUND_UP(data_dim_, 8);

  size_t num_points = 0;
  size_t num_chunks = 0;
  load_bin<uint8_t>(compressed_file, data_, num_points, num_chunks);
  num_points_ = num_points;
  num_chunks_ = num_chunks;
  if (num_chunks_ == 0 || num_chunks_ > MAX_PQ_CHUNKS) {
    throw diskann_exception_t("Invalid number of in-memory PQ chunks", -1);
  }
  pq_table_.load_pq_centroid_bin(pq_table_file.c_str(), num_chunks_);
  if (enable_cell_pq_decoded_u8_) {
    if (num_points_ > std::numeric_limits<size_t>::max() / data_dim_) {
      throw diskann_exception_t("Decoded uint8 PQ representation size overflows", -1);
    }
    timer_t decode_timer;
    if (num_chunks_ != data_dim_) {
      decoded_node_pq_codes_.resize(num_points_ * data_dim_);
    }
    std::vector<float> reconstructed(data_dim_);
    for (size_t point = 0; point < num_points_; ++point) {
      uint8_t* code = data_ + point * num_chunks_;
      pq_table_.inflate_vector(code, reconstructed.data());
      uint8_t* destination =
          num_chunks_ == data_dim_ ? code : decoded_node_pq_codes_.data() + point * data_dim_;
      for (size_t dimension = 0; dimension < data_dim_; ++dimension) {
        destination[dimension] = static_cast<uint8_t>(
            std::clamp(std::lround(reconstructed[dimension] * cell_pq_decoded_scale_), 0L, 255L));
      }
    }
    decoded_pq_build_time_us_ = static_cast<uint64_t>(decode_timer.elapsed());
  }

  const std::string disk_pq_file = disk_index_file_ + "_pq_pivots.bin";
  if (file_exists(disk_pq_file)) {
    use_disk_index_pq_ = true;
    disk_pq_table_.load_pq_centroid_bin(disk_pq_file.c_str(), 0);
    disk_pq_num_chunks_ = disk_pq_table_.get_num_chunks();
    disk_bytes_per_point_ = disk_pq_num_chunks_ * sizeof(uint8_t);
  }

  std::ifstream metadata(disk_index_file_, std::ios::binary);
  if (!metadata) {
    throw diskann_exception_t("Unable to open disk index metadata: " + disk_index_file_, -1);
  }
  auto read_metadata = [&metadata](auto& value) {
    metadata.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!metadata) {
      throw diskann_exception_t("Truncated disk index metadata", -1);
    }
  };

  uint32_t metadata_rows = 0;
  uint32_t metadata_columns = 0;
  uint64_t disk_num_points = 0;
  uint64_t disk_dimensions = 0;
  uint64_t medoid_on_file = 0;
  read_metadata(metadata_rows);
  read_metadata(metadata_columns);
  read_metadata(disk_num_points);
  read_metadata(disk_dimensions);
  read_metadata(medoid_on_file);
  read_metadata(max_node_len_);
  read_metadata(nodes_per_sector_);
  read_metadata(num_frozen_points_);
  uint64_t frozen_location = 0;
  read_metadata(frozen_location);
  uint64_t reorder_exists = 0;
  read_metadata(reorder_exists);

  if (metadata_columns != 1 || metadata_rows < 9 || disk_num_points != num_points_) {
    throw diskann_exception_t("Disk index metadata does not match compressed vectors", -1);
  }
  const uint64_t expected_disk_dimensions = use_disk_index_pq_ ? disk_pq_num_chunks_ : data_dim_;
  if (disk_dimensions != expected_disk_dimensions || num_points_ > UINT32_MAX ||
      medoid_on_file >= num_points_) {
    throw diskann_exception_t("Disk index dimensions or identifier range are invalid", -1);
  }
  if (max_node_len_ <= disk_bytes_per_point_ + sizeof(uint32_t)) {
    throw diskann_exception_t("Invalid disk node length", -1);
  }
  max_degree_ = ((max_node_len_ - disk_bytes_per_point_) / sizeof(uint32_t)) - 1;
  if (max_degree_ > defaults::MAX_GRAPH_DEGREE) {
    throw diskann_exception_t("Disk index degree exceeds supported maximum", -1);
  }
  if (io_index_ != nullptr &&
      (io_index_->point_count() != num_points_ || io_index_->dimension() != data_dim_ ||
       io_index_->max_degree() != max_degree_)) {
    throw diskann_exception_t("IO topology shape disagrees with the Base disk index", -1);
  }
  if (num_frozen_points_ == 1) {
    frozen_location_ = frozen_location;
  }
  reorder_data_exists_ = reorder_exists != 0;
  if (reorder_data_exists_) {
    if (!use_disk_index_pq_) {
      throw diskann_exception_t("Reorder data requires a disk-PQ index", -1);
    }
    read_metadata(reorder_data_start_sector_);
    read_metadata(reorder_dimensions_);
    read_metadata(vectors_per_sector_);
    if (reorder_dimensions_ != data_dim_ || vectors_per_sector_ == 0) {
      throw diskann_exception_t("Invalid reorder metadata", -1);
    }
  }
  metadata.close();

  reader_->open(io_index_ != nullptr && !io_index_->vector_path().empty()
                    ? io_index_->vector_path().string()
                    : disk_index_file_);
  setup_thread_data(num_threads, 4096,
                    enable_cell_batch_search_ ? defaults::MAX_GRAPH_DEGREE : 0);

  const std::string medoids_file = disk_index_file_ + "_medoids.bin";
  const std::string centroids_file = disk_index_file_ + "_centroids.bin";
  if (file_exists(medoids_file)) {
    size_t medoid_dimension = 0;
    load_bin<uint32_t>(medoids_file, medoids_, num_medoids_, medoid_dimension);
    if (medoid_dimension != 1 || num_medoids_ == 0) {
      throw diskann_exception_t("Invalid medoids file", -1);
    }
    if (file_exists(centroids_file)) {
      size_t num_centroids = 0;
      size_t centroid_dimension = 0;
      size_t aligned_centroid_dimension = 0;
      load_aligned_bin<float>(centroids_file, centroid_data_, num_centroids, centroid_dimension,
                              aligned_centroid_dimension);
      if (num_centroids != num_medoids_ || centroid_dimension != data_dim_ ||
          aligned_centroid_dimension != aligned_dim_) {
        throw diskann_exception_t("Centroid file does not match disk index", -1);
      }
    } else {
      use_medoids_data_as_centroids();
    }
  } else {
    num_medoids_ = 1;
    medoids_ = new uint32_t[1]{static_cast<uint32_t>(medoid_on_file)};
    use_medoids_data_as_centroids();
  }

  cout << "Loaded DiskANN PQ index. #points: " << num_points_ << " #dim: " << data_dim_
       << " #chunks: " << num_chunks_ << " #max_degree: " << max_degree_ << std::endl;
  return 0;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_l_aware_refinement(uint32_t base_candidates,
                                                             uint32_t l_divisor) {
  if (!enable_cell_pq_traversal_ || cell_pq_refine_candidates_ == 0 ||
      base_candidates == 0 || base_candidates > cell_pq_refine_candidates_ || l_divisor == 0) {
    throw diskann_exception_t(
        "L-aware refinement requires Cell-PQ traversal and a valid bounded budget", -1);
  }
  enable_l_aware_refine_budget_ = true;
  l_aware_refine_base_ = base_candidates;
  l_aware_refine_divisor_ = l_divisor;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_cell_pq_refine_min_l(
    uint32_t minimum_search_list_size) {
  if (!is_loaded_ || cell_pq_refine_candidates_ == 0 || minimum_search_list_size == 0) {
    throw diskann_exception_t(
        "Cell-PQ refinement minimum L requires loaded, enabled candidate refinement", -1);
  }
  cell_pq_refine_min_l_ = minimum_search_list_size;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_forced_cell_tree(
    std::shared_ptr<const community_polar_search_t> community_polar_search) {
  if (community_polar_search == nullptr || community_polar_search->point_count() != num_points_ ||
      community_polar_search->pq_code_width() != num_chunks_) {
    throw diskann_exception_t("Forced Cell-tree decoded PQ configuration is incompatible", -1);
  }
  if (!enable_cell_pq_decoded_u8_) {
    forced_cell_tree_configured_ = true;
    return;
  }
  timer_t decode_timer;
  if (num_points_ > std::numeric_limits<size_t>::max() / data_dim_) {
    throw diskann_exception_t("Forced Cell-tree decoded PQ representation size overflows", -1);
  }
  decoded_cell_pq_codes_.resize(num_points_ * data_dim_);
  constexpr uint64_t packet_batch = 4096;
  std::vector<uint8_t> source(packet_batch * num_chunks_);
  std::vector<float> reconstructed(data_dim_);
  for (uint64_t packet_begin = 0; packet_begin < num_points_; packet_begin += packet_batch) {
    const uint64_t packet_count = std::min<uint64_t>(packet_batch, num_points_ - packet_begin);
    community_polar_search->copy_packet_pq_codes(
        packet_begin, packet_count, std::span<uint8_t>(source).first(packet_count * num_chunks_));
    for (uint64_t offset = 0; offset < packet_count; ++offset) {
      pq_table_.inflate_vector(source.data() + offset * num_chunks_, reconstructed.data());
      uint8_t* destination = decoded_cell_pq_codes_.data() + (packet_begin + offset) * data_dim_;
      for (size_t dimension = 0; dimension < data_dim_; ++dimension) {
        destination[dimension] = static_cast<uint8_t>(
            std::clamp(std::lround(reconstructed[dimension] * cell_pq_decoded_scale_), 0L, 255L));
      }
    }
  }
  decoded_pq_build_time_us_ += static_cast<uint64_t>(decode_timer.elapsed());
  forced_cell_tree_configured_ = true;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_cell_adj_correction(
    uint32_t candidate_cells, uint32_t support_shortlist,
    std::shared_ptr<const io_cell_adjacency_index_t> cell_adjacency, uint32_t expansion_cells,
    uint32_t min_search_list_size, uint32_t max_search_list_size) {
  if (!forced_cell_tree_configured_ || io_index_ == nullptr || candidate_cells == 0 ||
      support_shortlist < candidate_cells || expansion_cells == 0 ||
      min_search_list_size == 0 || min_search_list_size > max_search_list_size ||
      (cell_adjacency == nullptr && expansion_cells != 1)) {
    throw diskann_exception_t(
        "Cell-Adj correction requires a loaded forced Cell tree and resident topology", -1);
  }
  enable_cell_adj_correction_ = true;
  cell_adj_candidate_cells_ = candidate_cells;
  cell_adj_support_shortlist_ = support_shortlist;
  cell_adj_expansion_cells_ = expansion_cells;
  cell_adj_min_search_list_size_ = min_search_list_size;
  cell_adj_max_search_list_size_ = max_search_list_size;
  cell_adjacency_index_ = std::move(cell_adjacency);
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_interleaved_cell_batch_search(uint32_t graph_hops) {
  if (!forced_cell_tree_configured_ || !enable_cell_batch_search_ ||
      !enable_cell_pq_traversal_ || graph_hops == 0) {
    throw diskann_exception_t(
        "Interleaved Cell batching requires a loaded forced Cell tree and positive graph hops",
        -1);
  }
  enable_interleaved_cell_batch_search_ = true;
  interleaved_cell_batch_graph_hops_ = graph_hops;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_cell_batch_preserve_graph() {
  if (!forced_cell_tree_configured_ || !enable_cell_batch_search_ || !enable_cell_pq_traversal_ ||
      enable_interleaved_cell_batch_search_) {
    throw diskann_exception_t(
        "Graph-preserving Cell batch requires non-interleaved forced-tree Cell-PQ search", -1);
  }
  cell_batch_preserve_graph_ = true;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_cell_batch_graph_stitch(
    uint32_t minimum_search_list_size) {
  if (!cell_batch_preserve_graph_ || minimum_search_list_size == 0) {
    throw diskann_exception_t(
        "L-aware Cell graph stitching requires graph-preserving Cell batching and positive L",
        -1);
  }
  cell_batch_graph_stitch_min_l_ = minimum_search_list_size;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_ranked_cell_page_refinement(
    uint32_t base_pages, uint32_t l_divisor, uint32_t growth_start_l,
    uint32_t growth_divisor, uint32_t max_pages, uint32_t max_search_list_size) {
  if (!forced_cell_tree_configured_ || !enable_cell_batch_search_ ||
      !cell_batch_preserve_graph_ || !enable_cell_pq_traversal_ ||
      enable_cell_leaf_refinement_ || l_divisor == 0 || max_pages == 0 ||
      max_search_list_size == 0 || base_pages > max_pages ||
      ((growth_start_l == 0) != (growth_divisor == 0)) ||
      max_pages > defaults::MAX_N_SECTOR_READS) {
    throw diskann_exception_t(
        "Ranked Cell-page refinement requires a graph-preserving forced Cell tree and a positive "
        "bounded page budget",
        -1);
  }
  enable_ranked_cell_page_refinement_ = true;
  ranked_cell_page_base_pages_ = base_pages;
  ranked_cell_page_l_divisor_ = l_divisor;
  ranked_cell_page_growth_start_l_ = growth_start_l;
  ranked_cell_page_growth_divisor_ = growth_divisor;
  ranked_cell_page_max_pages_ = max_pages;
  ranked_cell_page_max_l_ = max_search_list_size;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_pq_score_chunks(uint32_t chunk_count) {
  if (!is_loaded_ || chunk_count == 0 || chunk_count >= num_chunks_) {
    throw diskann_exception_t(
        "PQ chunk pruning requires a positive proper subset of loaded PQ chunks", -1);
  }
  pq_score_chunk_indices_.resize(chunk_count);
  if (chunk_count == 1) {
    pq_score_chunk_indices_.front() = static_cast<uint32_t>(num_chunks_ / 2);
    return;
  }
  for (uint32_t rank = 0; rank < chunk_count; ++rank) {
    pq_score_chunk_indices_[rank] = static_cast<uint32_t>(
        (static_cast<uint64_t>(rank) * (num_chunks_ - 1)) / (chunk_count - 1));
  }
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_pq_score_quantization(uint32_t bits) {
  if (!is_loaded_ || (bits != 8 && bits != 16)) {
    throw diskann_exception_t("PQ score quantization supports only loaded 8/16-bit policies", -1);
  }
  pq_score_quantization_bits_ = bits;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_graph_neighbor_score_limit(uint32_t limit) {
  if (!is_loaded_ || limit == 0 || limit >= max_degree_) {
    throw diskann_exception_t(
        "Graph-neighbor scoring limit must be a positive proper subset of max degree", -1);
  }
  graph_neighbor_score_limit_ = limit;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_resident_u8_refinement(const std::filesystem::path& path,
                                                                bool enable_traversal) {
  if (!is_loaded_ || !resident_u8_refinement_.empty() || path.empty() ||
      !std::filesystem::is_regular_file(path)) {
    throw diskann_exception_t("Resident uint8 refinement artifact is invalid", -1);
  }
  if (data_dim_ != 0 && num_points_ > std::numeric_limits<uint64_t>::max() / data_dim_) {
    throw diskann_exception_t("Resident uint8 refinement artifact size overflows", -1);
  }
  const uint64_t payload_bytes = num_points_ * data_dim_;
  if (payload_bytes > std::numeric_limits<size_t>::max() ||
      std::filesystem::file_size(path) != payload_bytes + 2U * sizeof(uint32_t)) {
    throw diskann_exception_t("Resident uint8 refinement artifact has an invalid size", -1);
  }
  std::ifstream input(path, std::ios::binary);
  std::array<uint8_t, 2U * sizeof(uint32_t)> header{};
  input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
  const auto read_u32 = [&](size_t offset) {
    return static_cast<uint32_t>(header[offset]) |
           (static_cast<uint32_t>(header[offset + 1]) << 8U) |
           (static_cast<uint32_t>(header[offset + 2]) << 16U) |
           (static_cast<uint32_t>(header[offset + 3]) << 24U);
  };
  if (!input || read_u32(0) != num_points_ || read_u32(sizeof(uint32_t)) != data_dim_) {
    throw diskann_exception_t("Resident uint8 refinement shape disagrees with the index", -1);
  }
  resident_u8_refinement_.resize(static_cast<size_t>(payload_bytes));
  constexpr size_t chunk_bytes = 64U * 1024U * 1024U;
  size_t cursor = 0;
  while (cursor < resident_u8_refinement_.size()) {
    const size_t count = std::min(chunk_bytes, resident_u8_refinement_.size() - cursor);
    input.read(reinterpret_cast<char*>(resident_u8_refinement_.data() + cursor),
               static_cast<std::streamsize>(count));
    if (!input) {
      resident_u8_refinement_.clear();
      throw diskann_exception_t("Resident uint8 refinement payload is truncated", -1);
    }
    cursor += count;
  }
  constexpr uint64_t fnv_offset = 14695981039346656037ULL;
  constexpr uint64_t fnv_prime = 1099511628211ULL;
  uint64_t checksum = fnv_offset;
  for (const uint8_t byte : header) {
    checksum = (checksum ^ byte) * fnv_prime;
  }
  for (const uint8_t byte : resident_u8_refinement_) {
    checksum = (checksum ^ byte) * fnv_prime;
  }
  resident_u8_refinement_artifact_bytes_ = payload_bytes + header.size();
  resident_u8_refinement_checksum_ = checksum;
  enable_resident_u8_traversal_ = enable_traversal;
}

template <typename data_t>
void pq_flash_index_t<data_t>::configure_memgraph(std::vector<uint32_t> nodes,
                                                  uint32_t entry_candidates) {
  if (!is_loaded_) {
    throw diskann_exception_t("MemGraph configuration requires a loaded PQ index", -1);
  }
  if ((!nodes.empty() && entry_candidates == 0) || entry_candidates > nodes.size()) {
    throw diskann_exception_t("MemGraph entry-candidate count is invalid", -1);
  }
  std::unordered_set<uint32_t> unique;
  unique.reserve(nodes.size());
  for (uint32_t node : nodes) {
    if (node >= num_points_ || !unique.insert(node).second) {
      throw diskann_exception_t("MemGraph nodes must be unique valid Base IDs", -1);
    }
  }
  memgraph_nodes_ = std::move(nodes);
  memgraph_entry_candidates_ = entry_candidates;
  memgraph_adjacency_.clear();
  memgraph_degree_ = 0;
  memgraph_build_time_us_ = 0;
  memgraph_build_peak_bytes_ = 0;
  if (memgraph_nodes_.size() > 1U) {
    timer_t build_timer;
    memgraph_degree_ = static_cast<uint32_t>(std::min<size_t>(16, memgraph_nodes_.size() - 1U));
    std::vector<float> reconstructed(memgraph_nodes_.size() * aligned_dim_, 0.0F);
    for (size_t node = 0; node < memgraph_nodes_.size(); ++node) {
      pq_table_.inflate_vector(data_ + static_cast<size_t>(memgraph_nodes_[node]) * num_chunks_,
                               reconstructed.data() + node * aligned_dim_);
    }
    memgraph_adjacency_.resize(memgraph_nodes_.size() * memgraph_degree_);
    std::vector<std::pair<float, uint32_t>> candidates;
    candidates.reserve(memgraph_nodes_.size() - 1U);
    memgraph_build_peak_bytes_ = reconstructed.capacity() * sizeof(float) +
                                 candidates.capacity() * sizeof(candidates.front()) +
                                 memgraph_adjacency_.capacity() * sizeof(uint32_t);
    for (size_t node = 0; node < memgraph_nodes_.size(); ++node) {
      candidates.clear();
      const float* source = reconstructed.data() + node * aligned_dim_;
      for (size_t target = 0; target < memgraph_nodes_.size(); ++target) {
        if (target == node) {
          continue;
        }
        const float distance = float_distance_comparator_->compare(
            source, reconstructed.data() + target * aligned_dim_,
            static_cast<uint32_t>(aligned_dim_));
        candidates.emplace_back(distance, static_cast<uint32_t>(target));
      }
      std::partial_sort(candidates.begin(), candidates.begin() + memgraph_degree_, candidates.end(),
                        [this](const auto& left, const auto& right) {
                          return left.first < right.first ||
                                 (left.first == right.first &&
                                  memgraph_nodes_[left.second] < memgraph_nodes_[right.second]);
                        });
      for (uint32_t edge = 0; edge < memgraph_degree_; ++edge) {
        memgraph_adjacency_[node * memgraph_degree_ + edge] = candidates[edge].second;
      }
    }
    memgraph_build_time_us_ = static_cast<uint64_t>(build_timer.elapsed());
  }
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search(const data_t* query, uint64_t top_k,
                                                  uint64_t search_list_size, uint64_t* indices,
                                                  float* distances, uint64_t beam_width,
                                                  bool use_reorder_data, query_stats_t* stats) {
  cached_beam_search(query, top_k, search_list_size, indices, distances, beam_width,
                     std::numeric_limits<uint32_t>::max(), use_reorder_data, stats);
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search(const data_t* query, uint64_t top_k,
                                                  uint64_t search_list_size, uint64_t* indices,
                                                  float* distances, uint64_t beam_width,
                                                  uint32_t io_limit, bool use_reorder_data,
                                                  query_stats_t* stats) {
  cached_beam_search_impl<false, false, false, false, false>(
      query, top_k, search_list_size, indices, distances, beam_width, io_limit, use_reorder_data,
      stats, nullptr);
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search_io_optimized(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit, query_stats_t* stats) {
  cached_beam_search_impl<false, false, false, true, false>(query, top_k, search_list_size, indices,
                                                            distances, beam_width, io_limit, false,
                                                            stats, nullptr);
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search_hybrid(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit,
    const community_polar_search_t& hybrid_search, bool use_reorder_data, query_stats_t* stats) {
  cached_beam_search_impl<true, true, false, false, false>(query, top_k, search_list_size, indices,
                                                           distances, beam_width, io_limit,
                                                           use_reorder_data, stats, &hybrid_search);
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search_hybrid_io_optimized(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit,
    const community_polar_search_t& hybrid_search, query_stats_t* stats) {
  if (forced_cell_tree_configured_) {
    cached_beam_search_impl<true, true, false, true, true>(query, top_k, search_list_size, indices,
                                                           distances, beam_width, io_limit, false,
                                                           stats, &hybrid_search);
  } else {
    cached_beam_search_impl<true, true, false, true, false>(query, top_k, search_list_size, indices,
                                                            distances, beam_width, io_limit, false,
                                                            stats, &hybrid_search);
  }
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search_observe_only(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit,
    const community_polar_search_t& hybrid_search, bool use_reorder_data, query_stats_t* stats) {
  cached_beam_search_impl<true, false, false, false, false>(
      query, top_k, search_list_size, indices, distances, beam_width, io_limit, use_reorder_data,
      stats, &hybrid_search);
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search_phase_profiled(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit, bool use_reorder_data,
    query_stats_t* stats) {
  if (stats == nullptr) {
    throw diskann_exception_t("Beam phase profiling requires per-query statistics", -1);
  }
  cached_beam_search_impl<false, false, true, false, false>(query, top_k, search_list_size, indices,
                                                            distances, beam_width, io_limit,
                                                            use_reorder_data, stats, nullptr);
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search_io_optimized_phase_profiled(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit, query_stats_t* stats) {
  if (stats == nullptr) {
    throw diskann_exception_t("IO-optimized Beam phase profiling requires statistics", -1);
  }
  cached_beam_search_impl<false, false, true, true, false>(query, top_k, search_list_size, indices,
                                                           distances, beam_width, io_limit, false,
                                                           stats, nullptr);
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search_hybrid_phase_profiled(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit,
    const community_polar_search_t& hybrid_search, bool use_reorder_data, query_stats_t* stats) {
  if (stats == nullptr) {
    throw diskann_exception_t("Hybrid Beam phase profiling requires per-query statistics", -1);
  }
  cached_beam_search_impl<true, true, true, false, false>(query, top_k, search_list_size, indices,
                                                          distances, beam_width, io_limit,
                                                          use_reorder_data, stats, &hybrid_search);
}

template <typename data_t>
void pq_flash_index_t<data_t>::cached_beam_search_hybrid_io_optimized_phase_profiled(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit,
    const community_polar_search_t& hybrid_search, query_stats_t* stats) {
  if (stats == nullptr) {
    throw diskann_exception_t("IO-optimized Hybrid phase profiling requires statistics", -1);
  }
  if (forced_cell_tree_configured_) {
    cached_beam_search_impl<true, true, true, true, true>(query, top_k, search_list_size, indices,
                                                          distances, beam_width, io_limit, false,
                                                          stats, &hybrid_search);
  } else {
    cached_beam_search_impl<true, true, true, true, false>(query, top_k, search_list_size, indices,
                                                           distances, beam_width, io_limit, false,
                                                           stats, &hybrid_search);
  }
}

template <typename data_t>
template <bool use_hybrid, bool insert_macro, bool track_beam_phase, bool use_io_layout,
          bool forced_terminal_tree>
void pq_flash_index_t<data_t>::cached_beam_search_impl(
    const data_t* query, uint64_t top_k, uint64_t search_list_size, uint64_t* indices,
    float* distances, uint64_t beam_width, uint32_t io_limit, bool use_reorder_data,
    query_stats_t* stats, const community_polar_search_t* hybrid_search) {
  if (!is_loaded_ || query == nullptr || indices == nullptr) {
    throw diskann_exception_t("Search requires a loaded index and valid buffers", -1);
  }
  if (top_k == 0 || search_list_size < top_k || beam_width == 0) {
    throw diskann_exception_t("Search requires 0 < top_k <= search_list_size and beam_width > 0",
                              -1);
  }
  const uint64_t l_increment =
      enable_l_aware_refine_budget_
          ? (search_list_size + l_aware_refine_divisor_ - 1U) / l_aware_refine_divisor_
          : 0;
  const uint32_t refine_candidate_budget =
      search_list_size < cell_pq_refine_min_l_
          ? 0
          : enable_l_aware_refine_budget_
                ? static_cast<uint32_t>(std::min<uint64_t>(cell_pq_refine_candidates_,
                                                           l_aware_refine_base_ + l_increment))
                : cell_pq_refine_candidates_;
  uint64_t ranked_cell_page_budget_unbounded =
      static_cast<uint64_t>(ranked_cell_page_base_pages_) +
      (search_list_size + ranked_cell_page_l_divisor_ - 1U) /
          ranked_cell_page_l_divisor_;
  if (ranked_cell_page_growth_start_l_ != 0 &&
      search_list_size > ranked_cell_page_growth_start_l_) {
    ranked_cell_page_budget_unbounded +=
        (search_list_size - ranked_cell_page_growth_start_l_ +
         ranked_cell_page_growth_divisor_ - 1U) /
        ranked_cell_page_growth_divisor_;
  }
  const uint32_t ranked_cell_page_budget =
      enable_ranked_cell_page_refinement_ && search_list_size <= ranked_cell_page_max_l_
          ? static_cast<uint32_t>(std::min<uint64_t>(ranked_cell_page_max_pages_,
                                                     ranked_cell_page_budget_unbounded))
          : 0;
  const uint64_t sectors_per_node = DIV_ROUND_UP(max_node_len_, defaults::SECTOR_LEN);
  if (beam_width > sectors_per_node * defaults::MAX_N_SECTOR_READS) {
    throw diskann_exception_t("Beam width exceeds the query scratch I/O capacity", -1);
  }
  static_assert(use_hybrid || !insert_macro);
  if constexpr (use_io_layout) {
    if (io_index_ == nullptr) {
      throw diskann_exception_t("IO-optimized search requires a loaded IO topology", -1);
    }
    if (use_reorder_data) {
      throw diskann_exception_t("IO-1 layout does not support Base reorder-data reads", -1);
    }
  } else if (io_index_ != nullptr) {
    throw diskann_exception_t("IO layout load requires an IO-optimized search specialization", -1);
  }
  if constexpr (use_hybrid) {
    if (hybrid_search == nullptr || hybrid_search->point_count() != num_points_ ||
        hybrid_search->dimension() != data_dim_ || hybrid_search->pq_code_width() != num_chunks_) {
      throw diskann_exception_t("Community-Polar sidecar disagrees with the loaded PQ index", -1);
    }
  }

  scratch_store_manager_t<ssd_thread_data_t<data_t>> manager(thread_data_);
  const auto* final_adaptive_cell_snapshot =
      adaptive_cell_runtime_ == nullptr ? nullptr : adaptive_cell_runtime_->final_snapshot();
  const auto adaptive_cell_snapshot =
      adaptive_cell_runtime_ == nullptr || final_adaptive_cell_snapshot != nullptr
          ? nullptr
          : adaptive_cell_runtime_->acquire();
  const auto* active_adaptive_cell_snapshot = final_adaptive_cell_snapshot == nullptr
                                                  ? adaptive_cell_snapshot.get()
                                                  : final_adaptive_cell_snapshot;
  const auto active_vector_location = [&](uint32_t node_id) {
    const auto fixed = io_index_->vector_location(node_id);
    return active_adaptive_cell_snapshot == nullptr
               ? fixed
               : active_adaptive_cell_snapshot->vector_location(node_id, fixed);
  };
  auto* thread_data = manager.scratch_space();
  auto& io_context = thread_data->ctx;
  auto* query_scratch = &thread_data->scratch;
  auto* pq_scratch = query_scratch->pq_scratch();
  query_scratch->reset();

#if defined(__linux__)
  auto* prefetch_reader = dynamic_cast<io_uring_aligned_file_reader_t*>(reader_.get());
  static thread_local std::vector<io_cache_reservation_t> pending_prefetches;
  struct gateway_prefetch_submission_t {
    std::chrono::steady_clock::time_point submitted_at;
    uint32_t submitted_hop = 0;
  };
  static thread_local std::unordered_map<uint64_t, gateway_prefetch_submission_t>
      gateway_prefetch_submissions;
  pending_prefetches.clear();
  gateway_prefetch_submissions.clear();
  auto harvest_prefetches = [&](bool wait) {
    if (pending_prefetches.empty()) {
      return;
    }
    try {
      const auto completed = wait ? prefetch_reader->complete_prefetches(io_context)
                                  : prefetch_reader->poll_prefetches(io_context);
      for (const uint64_t page : completed) {
        const auto reservation =
            std::find_if(pending_prefetches.begin(), pending_prefetches.end(),
                         [page](const auto& item) { return item.page_id == page; });
        if (reservation == pending_prefetches.end()) {
          throw diskann_exception_t("io_uring returned an unknown prefetch tag", -1);
        }
        io_cache_->publish(*reservation, true);
        io_cache_->release(*reservation);
        pending_prefetches.erase(reservation);
      }
    } catch (...) {
      for (const auto& reservation : pending_prefetches) {
        io_cache_->publish(reservation, false);
      }
      pending_prefetches.clear();
      throw;
    }
  };
  auto submit_prefetch = [&](uint64_t page_id, bool gateway_prediction = false,
                             uint32_t submitted_hop = 0) {
    if (page_id == k_no_io_prefetch_page ||
        std::any_of(pending_prefetches.begin(), pending_prefetches.end(),
                    [page_id](const auto& item) { return item.page_id == page_id; })) {
      return false;
    }
    if (gateway_prediction && (pending_prefetches.size() >= gateway_prefetch_max_outstanding_ ||
                               !prefetch_reader->has_prefetch_headroom(io_context))) {
      if (stats != nullptr) {
        ++stats->gateway_prefetch_queue_rejections;
      }
      return false;
    }
    auto reservation = io_cache_->reserve(page_id,
                                          gateway_prediction ? io_cache_access_t::GATEWAY_PREFETCH
                                                             : io_cache_access_t::PREFETCH,
                                          false);
    if (reservation.state == io_cache_reservation_state_t::HIT) {
      io_cache_->release(reservation);
      return false;
    }
    if (reservation.state != io_cache_reservation_state_t::LOAD) {
      return false;
    }
    try {
      prefetch_reader->submit_prefetch(
          aligned_read_t{page_id * defaults::SECTOR_LEN, defaults::SECTOR_LEN, reservation.data},
          io_context, page_id);
      pending_prefetches.push_back(reservation);
      if (gateway_prediction) {
        gateway_prefetch_submissions[page_id] = {std::chrono::steady_clock::now(), submitted_hop};
        if (stats != nullptr) {
          ++stats->gateway_prefetch_pages_submitted;
        }
      }
    } catch (...) {
      io_cache_->publish(reservation, false);
      throw;
    }
    return true;
  };
  if ((enable_node_prefetch_ || enable_cell_prefetch_ || enable_gateway_prefetch_) &&
      prefetch_reader == nullptr) {
    throw diskann_exception_t("IO prefetch requires the io_uring reader", -1);
  }
#else
  if (enable_node_prefetch_ || enable_cell_prefetch_ || enable_gateway_prefetch_) {
    throw diskann_exception_t("IO prefetch is supported only by the Linux io_uring backend", -1);
  }
#endif

  data_t* aligned_query = query_scratch->aligned_query();
  for (size_t dim = 0; dim < data_dim_; ++dim) {
    aligned_query[dim] = query[dim];
  }
  pq_scratch->initialize(data_dim_, aligned_query);
  float* query_float = pq_scratch->aligned_query_float;
  float* rotated_query = pq_scratch->rotated_query;
  const bool cell_u8_layout = use_io_layout && io_index_ != nullptr &&
                              io_index_->vector_layout() == io_vector_layout_t::CELL_U8_4K;

  data_t* data_buffer = query_scratch->coord_scratch;
  cpu_prefetch_t1(data_buffer);
  char* sector_scratch = query_scratch->sector_scratch;
  size_t& sector_scratch_index = query_scratch->sector_idx;
  const uint64_t num_sectors_per_node =
      nodes_per_sector_ > 0 ? 1 : DIV_ROUND_UP(max_node_len_, defaults::SECTOR_LEN);

  static thread_local std::vector<uint8_t> decoded_u8_query;
  if (enable_cell_pq_decoded_u8_ || enable_resident_u8_traversal_ || cell_u8_layout) {
    decoded_u8_query.resize(data_dim_);
    for (size_t dimension = 0; dimension < data_dim_; ++dimension) {
      const float value = static_cast<float>(aligned_query[dimension]);
      if (cell_u8_layout && (!std::isfinite(value) || value < 0.0F || value > 255.0F ||
                             std::nearbyint(value) != value)) {
        throw diskann_exception_t("Cell-u8 refinement requires integral queries in [0, 255]", -1);
      }
      decoded_u8_query[dimension] =
          static_cast<uint8_t>(std::clamp(std::lround(value * cell_pq_decoded_scale_), 0L, 255L));
    }
  } else {
    pq_table_.preprocess_query(rotated_query);
  }
  float* pq_distances = pq_scratch->aligned_pqtable_dist_scratch;
  if (enable_cell_pq_decoded_u8_ || enable_resident_u8_traversal_) {
    // Decoded vectors bypass the query-local ADC table.
  } else {
    pq_table_.populate_chunk_distances(rotated_query, pq_distances);
  }
  float* distance_scratch = pq_scratch->aligned_dist_scratch;
  uint8_t* pq_coord_scratch = pq_scratch->aligned_pq_coord_scratch;

  static thread_local std::vector<uint16_t> quantized_pq_distances;
  static thread_local std::vector<uint8_t> quantized_pq_distances_u8;
  float pq_quantization_scale = 0.0F;
  float pq_quantization_offset = 0.0F;
  if (pq_score_quantization_bits_ != 0) {
    const uint32_t maximum_quantized = pq_score_quantization_bits_ == 8 ? 255U : 65535U;
    float maximum_residual = 0.0F;
    for (size_t chunk = 0; chunk < num_chunks_; ++chunk) {
      const float* distances = pq_distances + chunk * 256U;
      const float minimum = *std::min_element(distances, distances + 256U);
      pq_quantization_offset += minimum;
      for (size_t center = 0; center < 256U; ++center) {
        maximum_residual = std::max(maximum_residual, distances[center] - minimum);
      }
    }
    pq_quantization_scale =
        maximum_residual == 0.0F ? 1.0F : static_cast<float>(maximum_quantized) / maximum_residual;
    quantized_pq_distances.resize(pq_score_quantization_bits_ == 16 ? num_chunks_ * 256U + 2U : 0U);
    quantized_pq_distances_u8.resize(pq_score_quantization_bits_ == 8 ? num_chunks_ * 256U + 4U
                                                                      : 0U);
    for (size_t chunk = 0; chunk < num_chunks_; ++chunk) {
      const float* distances = pq_distances + chunk * 256U;
      const float minimum = *std::min_element(distances, distances + 256U);
      for (size_t center = 0; center < 256U; ++center) {
        const long quantized =
            std::clamp(std::lround((distances[center] - minimum) * pq_quantization_scale), 0L,
                       static_cast<long>(maximum_quantized));
        if (pq_score_quantization_bits_ == 8) {
          quantized_pq_distances_u8[chunk * 256U + center] = static_cast<uint8_t>(quantized);
        } else {
          quantized_pq_distances[chunk * 256U + center] = static_cast<uint16_t>(quantized);
        }
      }
    }
  }

  auto compute_pq_distances = [this, pq_distances, pq_quantization_scale, pq_quantization_offset](
                                  const uint8_t* codes, uint64_t num_ids, float* output) {
    if (enable_cell_pq_traversal_ && enable_cell_pq_decoded_u8_) {
      decoded_u8_l2_lookup(codes, num_ids, data_dim_, decoded_u8_query.data(), output);
    } else if (pq_score_quantization_bits_ != 0) {
      uint64_t id = 0;
#if defined(__AVX2__)
      const __m256i mask = _mm256_set1_epi32(0xFFFF);
      for (; id + 8 <= num_ids; id += 8) {
        __m256i accumulated = _mm256_setzero_si256();
        for (size_t chunk = 0; chunk < num_chunks_; ++chunk) {
          const auto* chunk_distances = reinterpret_cast<const int*>(
              pq_score_quantization_bits_ == 8
                  ? static_cast<const void*>(quantized_pq_distances_u8.data() + chunk * 256U)
                  : static_cast<const void*>(quantized_pq_distances.data() + chunk * 256U));
          const __m256i byte_offsets = _mm256_setr_epi32(
              (pq_score_quantization_bits_ / 8U) * codes[(id + 0) * num_chunks_ + chunk],
              (pq_score_quantization_bits_ / 8U) * codes[(id + 1) * num_chunks_ + chunk],
              (pq_score_quantization_bits_ / 8U) * codes[(id + 2) * num_chunks_ + chunk],
              (pq_score_quantization_bits_ / 8U) * codes[(id + 3) * num_chunks_ + chunk],
              (pq_score_quantization_bits_ / 8U) * codes[(id + 4) * num_chunks_ + chunk],
              (pq_score_quantization_bits_ / 8U) * codes[(id + 5) * num_chunks_ + chunk],
              (pq_score_quantization_bits_ / 8U) * codes[(id + 6) * num_chunks_ + chunk],
              (pq_score_quantization_bits_ / 8U) * codes[(id + 7) * num_chunks_ + chunk]);
          const __m256i gathered = _mm256_i32gather_epi32(chunk_distances, byte_offsets, 1);
          const __m256i active_mask =
              pq_score_quantization_bits_ == 8 ? _mm256_set1_epi32(0xFF) : mask;
          accumulated = _mm256_add_epi32(accumulated, _mm256_and_si256(gathered, active_mask));
        }
        const __m256 values = _mm256_add_ps(
            _mm256_set1_ps(pq_quantization_offset),
            _mm256_div_ps(_mm256_cvtepi32_ps(accumulated), _mm256_set1_ps(pq_quantization_scale)));
        _mm256_storeu_ps(output + id, values);
      }
#endif
      for (; id < num_ids; ++id) {
        uint32_t distance = 0;
        for (size_t chunk = 0; chunk < num_chunks_; ++chunk) {
          const size_t offset = chunk * 256U + codes[id * num_chunks_ + chunk];
          distance += pq_score_quantization_bits_ == 8 ? quantized_pq_distances_u8[offset]
                                                       : quantized_pq_distances[offset];
        }
        output[id] = pq_quantization_offset + static_cast<float>(distance) / pq_quantization_scale;
      }
    } else if (pq_score_chunk_indices_.empty()) {
      pq_dist_lookup(codes, num_ids, num_chunks_, pq_distances, output);
    } else {
      const float scale =
          static_cast<float>(num_chunks_) / static_cast<float>(pq_score_chunk_indices_.size());
      std::fill(output, output + num_ids, 0.0F);
#if defined(__AVX2__)
      uint64_t id = 0;
      for (; id + 8 <= num_ids; id += 8) {
        __m256 accumulated = _mm256_setzero_ps();
        for (const uint32_t chunk : pq_score_chunk_indices_) {
          const float* chunk_distances = pq_distances + 256U * chunk;
          const __m256i centers = _mm256_setr_epi32(
              codes[(id + 0) * num_chunks_ + chunk], codes[(id + 1) * num_chunks_ + chunk],
              codes[(id + 2) * num_chunks_ + chunk], codes[(id + 3) * num_chunks_ + chunk],
              codes[(id + 4) * num_chunks_ + chunk], codes[(id + 5) * num_chunks_ + chunk],
              codes[(id + 6) * num_chunks_ + chunk], codes[(id + 7) * num_chunks_ + chunk]);
          accumulated = _mm256_add_ps(accumulated,
                                      _mm256_i32gather_ps(chunk_distances, centers, sizeof(float)));
        }
        _mm256_storeu_ps(output + id, _mm256_mul_ps(accumulated, _mm256_set1_ps(scale)));
      }
      for (; id < num_ids; ++id) {
#else
      for (uint64_t id = 0; id < num_ids; ++id) {
#endif
        float distance = 0.0F;
        for (const uint32_t chunk : pq_score_chunk_indices_) {
          distance += pq_distances[256U * chunk + codes[id * num_chunks_ + chunk]];
        }
        output[id] = distance * scale;
      }
    }
  };
  auto compute_distances = [this, pq_coord_scratch, &compute_pq_distances](
                               const uint32_t* ids, uint64_t num_ids, float* output) {
    if (enable_resident_u8_traversal_) {
      original_u8_l2_lookup(resident_u8_refinement_.data(), ids, num_ids, data_dim_,
                            decoded_u8_query.data(), output);
      return;
    }
    const uint8_t* source = decoded_node_pq_codes_.empty() ? data_ : decoded_node_pq_codes_.data();
    if (enable_cell_pq_decoded_u8_) {
      original_u8_l2_lookup(source, ids, num_ids, data_dim_, decoded_u8_query.data(), output);
      return;
    }
    const size_t width = enable_cell_pq_decoded_u8_ ? data_dim_ : num_chunks_;
    aggregate_coords(ids, num_ids, source, width, pq_coord_scratch);
    compute_pq_distances(pq_coord_scratch, num_ids, output);
  };
  auto load_hybrid_candidate_codes = [&](std::span<const uint32_t> node_ids) {
    if (enable_cell_pq_decoded_u8_) {
      const uint8_t* source =
          decoded_node_pq_codes_.empty() ? data_ : decoded_node_pq_codes_.data();
      aggregate_coords(node_ids.data(), node_ids.size(), source, data_dim_, pq_coord_scratch);
    } else {
      aggregate_coords(node_ids.data(), node_ids.size(), data_, num_chunks_, pq_coord_scratch);
    }
  };

  timer_t query_timer;
  timer_t io_timer;
  timer_t cpu_timer;
  auto& visited = query_scratch->visited;
  auto& candidate_queue = query_scratch->retset;
  auto& full_result = query_scratch->full_retset;
  static thread_local std::vector<uint64_t> dense_visited_bits;
  static thread_local std::vector<uint32_t> dense_visited_ids;
  static thread_local std::vector<uint32_t> unvisited_neighbor_ids;
  unvisited_neighbor_ids.clear();
  unvisited_neighbor_ids.reserve(max_degree_);
  if (enable_cell_pq_dense_visited_) {
    for (const uint32_t node : dense_visited_ids) {
      dense_visited_bits[node >> 6U] &= ~(uint64_t{1} << (node & 63U));
    }
    dense_visited_ids.clear();
    dense_visited_bits.resize((num_points_ + 63U) / 64U, 0);
  }
  const auto insert_visited = [&](uint32_t node) {
    if (!enable_cell_pq_dense_visited_) {
      return visited.insert(node).second;
    }
    const uint64_t mask = uint64_t{1} << (node & 63U);
    uint64_t& word = dense_visited_bits[node >> 6U];
    if ((word & mask) != 0) {
      return false;
    }
    word |= mask;
    dense_visited_ids.push_back(node);
    return true;
  };
  const auto contains_visited = [&](uint32_t node) {
    if (!enable_cell_pq_dense_visited_) {
      return visited.find(node) != visited.end();
    }
    return (dense_visited_bits[node >> 6U] & (uint64_t{1} << (node & 63U))) != 0;
  };
  candidate_queue.reserve(search_list_size);
  static thread_local std::vector<neighbor_t> terminal_cell_results;
  terminal_cell_results.clear();
  static thread_local std::vector<neighbor_t> cell_batch_exact_results;
  static thread_local std::vector<uint32_t> cell_batch_selected_cells;
  static thread_local std::vector<uint32_t> cell_batch_loaded_cells;
  static thread_local std::vector<uint32_t> cell_batch_exact_nodes;
  cell_batch_exact_results.clear();
  cell_batch_selected_cells.clear();
  cell_batch_loaded_cells.clear();
  cell_batch_exact_nodes.clear();
  uint32_t interleaved_last_cell_hop = 0;
  static thread_local std::vector<uint64_t> cell_prediction_pages;
  cell_prediction_pages.clear();
  static thread_local std::vector<uint32_t> adaptive_loaded_cells;
  adaptive_loaded_cells.clear();
  community_polar_query_state_t* hybrid_state = nullptr;
  if constexpr (use_hybrid) {
    static thread_local community_polar_query_state_t query_state_scratch;
    hybrid_search->begin_query(std::span<const float>(query_float, data_dim_), query_state_scratch,
                               search_list_size >=
                                   hybrid_search->config().cell_min_search_list_size);
    hybrid_state = &query_state_scratch;
  }
  const bool enable_hybrid_graph_hooks = [&]() {
    if constexpr (!use_hybrid) {
      return false;
    }
    const auto& hybrid_config = hybrid_search->config();
    if constexpr (forced_terminal_tree) {
      // Forced-tree search still performs its mandatory root-to-leaf route.  When explicitly
      // requested, retain only the cheap source-Cell observation from the graph approach so the
      // terminal batch can prefer a leaf already reached by the shared frontier.  Candidate-edge
      // evidence and live macro insertion remain disabled below for this specialization.
      return hybrid_config.community_skip || hybrid_config.cell_terminal_frontier_cells != 0;
    }
    return !hybrid_config.cell_defer_hierarchy_routing || hybrid_config.community_skip ||
           hybrid_config.gateway_landing_cell_handoff ||
           hybrid_config.cell_terminal_frontier_cells != 0;
  }();

  uint32_t best_medoid = 0;
  float best_medoid_distance = std::numeric_limits<float>::max();
  for (size_t medoid = 0; medoid < num_medoids_; ++medoid) {
    const float distance = float_distance_comparator_->compare(
        query_float, centroid_data_ + aligned_dim_ * medoid, static_cast<uint32_t>(aligned_dim_));
    if (distance < best_medoid_distance) {
      best_medoid = medoids_[medoid];
      best_medoid_distance = distance;
    }
  }
  compute_distances(&best_medoid, 1, distance_scratch);
  candidate_queue.insert(neighbor_t(best_medoid, distance_scratch[0]));
  insert_visited(best_medoid);

  static thread_local std::unordered_set<uint32_t> memgraph_injected;
  memgraph_injected.clear();
  if (!memgraph_nodes_.empty()) {
    using memgraph_frontier_t = std::pair<float, uint32_t>;
    static thread_local std::vector<uint8_t> memgraph_seen;
    static thread_local std::vector<uint32_t> memgraph_local_batch;
    static thread_local std::vector<uint32_t> memgraph_base_batch;
    static thread_local std::vector<neighbor_t> memgraph_scores;
    std::priority_queue<memgraph_frontier_t, std::vector<memgraph_frontier_t>,
                        std::greater<memgraph_frontier_t>>
        memgraph_frontier;
    memgraph_seen.assign(memgraph_nodes_.size(), 0);
    memgraph_local_batch.clear();
    memgraph_base_batch.clear();
    memgraph_scores.clear();
    const size_t score_budget =
        std::min(memgraph_nodes_.size(), std::max<size_t>(64, memgraph_entry_candidates_ * 32U));
    memgraph_scores.reserve(score_budget);
    if constexpr (track_beam_phase) {
      cpu_timer.reset();
    }
    auto score_memgraph_batch = [&]() {
      if (memgraph_local_batch.empty()) {
        return;
      }
      memgraph_base_batch.resize(memgraph_local_batch.size());
      for (size_t position = 0; position < memgraph_local_batch.size(); ++position) {
        memgraph_base_batch[position] = memgraph_nodes_[memgraph_local_batch[position]];
      }
      compute_distances(memgraph_base_batch.data(), memgraph_base_batch.size(), distance_scratch);
      for (size_t position = 0; position < memgraph_local_batch.size(); ++position) {
        memgraph_scores.emplace_back(memgraph_base_batch[position], distance_scratch[position]);
        memgraph_frontier.emplace(distance_scratch[position], memgraph_local_batch[position]);
      }
      memgraph_local_batch.clear();
    };
    const size_t anchor_count = std::min<size_t>(4, memgraph_nodes_.size());
    for (size_t anchor = 0; anchor < anchor_count; ++anchor) {
      const uint32_t local = static_cast<uint32_t>(anchor * memgraph_nodes_.size() / anchor_count);
      if (memgraph_seen[local] == 0) {
        memgraph_seen[local] = 1;
        memgraph_local_batch.push_back(local);
      }
    }
    score_memgraph_batch();
    while (!memgraph_frontier.empty() && memgraph_scores.size() < score_budget) {
      const uint32_t local = memgraph_frontier.top().second;
      memgraph_frontier.pop();
      for (uint32_t edge = 0; edge < memgraph_degree_ &&
                              memgraph_scores.size() + memgraph_local_batch.size() < score_budget;
           ++edge) {
        const uint32_t neighbor = memgraph_adjacency_[local * memgraph_degree_ + edge];
        if (memgraph_seen[neighbor] == 0) {
          memgraph_seen[neighbor] = 1;
          memgraph_local_batch.push_back(neighbor);
        }
      }
      score_memgraph_batch();
    }
    std::partial_sort(memgraph_scores.begin(), memgraph_scores.begin() + memgraph_entry_candidates_,
                      memgraph_scores.end());
    uint32_t inserted = 0;
    for (size_t position = 0; position < memgraph_entry_candidates_; ++position) {
      if (insert_visited(static_cast<uint32_t>(memgraph_scores[position].id))) {
        candidate_queue.insert(memgraph_scores[position]);
        memgraph_injected.insert(memgraph_scores[position].id);
        ++inserted;
      }
    }
    if (stats != nullptr) {
      stats->memgraph_nodes_scored += static_cast<uint32_t>(memgraph_scores.size());
      stats->memgraph_nodes_inserted += inserted;
      stats->n_cmps += static_cast<uint32_t>(memgraph_scores.size());
      if constexpr (track_beam_phase) {
        stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
      }
    }
  }

  uint32_t num_ios = 0;
  std::vector<uint32_t> frontier;
  std::vector<float> frontier_candidate_distances;
  std::vector<std::pair<uint32_t, char*>> frontier_neighborhoods;
  std::vector<aligned_read_t> frontier_read_requests;
  std::vector<aligned_read_t> coalesced_read_requests;
  std::vector<std::pair<char*, char*>> coalesced_page_copies;
  std::vector<uint64_t> frontier_page_ids;
  std::vector<char*> frontier_page_buffers;
  std::vector<io_cache_reservation_t> frontier_cache_reservations;
  std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t*>>> cached_neighborhoods;
  frontier.reserve(2 * beam_width);
  frontier_candidate_distances.reserve(2 * beam_width);
  frontier_neighborhoods.reserve(2 * beam_width);
  frontier_read_requests.reserve(2 * beam_width);
  coalesced_read_requests.reserve(2 * beam_width);
  coalesced_page_copies.reserve(2 * beam_width);
  frontier_page_ids.reserve(2 * beam_width);
  frontier_page_buffers.reserve(2 * beam_width);
  frontier_cache_reservations.reserve(2 * beam_width);
  cached_neighborhoods.reserve(2 * beam_width);
  static thread_local std::vector<uint64_t> refine_page_ids;
  static thread_local std::vector<char*> refine_page_buffers;
  static thread_local std::vector<io_cache_reservation_t> refine_reservations;
  static thread_local std::vector<aligned_read_t> refine_read_requests;
  static thread_local std::vector<aligned_read_t> refine_coalesced_read_requests;
  static thread_local std::vector<std::pair<char*, char*>> refine_coalesced_page_copies;
  static thread_local std::vector<uint64_t> refine_requested_pages;
  static thread_local std::vector<uint64_t> refine_candidate_pages;
  refine_page_ids.clear();
  refine_page_buffers.clear();
  refine_reservations.clear();
  refine_read_requests.clear();
  refine_coalesced_read_requests.clear();
  refine_coalesced_page_copies.clear();
  refine_requested_pages.clear();
  refine_candidate_pages.clear();
  const size_t refine_reserve =
      enable_cell_leaf_refinement_ ? defaults::MAX_N_SECTOR_READS : refine_candidate_budget;
  refine_page_ids.reserve(refine_reserve);
  refine_page_buffers.reserve(refine_reserve);
  refine_reservations.reserve(refine_reserve);
  refine_read_requests.reserve(refine_reserve);
  size_t refine_scratch_page = 0;
  bool refine_prefetch_attempted = false;
  bool refine_prefetch_submitted = false;
#if defined(__linux__)
  auto* libaio_reader = dynamic_cast<linux_aligned_file_reader_t*>(reader_.get());
  static thread_local libaio_async_batch_t refine_async_batch;
#endif
  const auto reserve_refine_pages = [&](std::span<const uint64_t> requested_pages) {
    refine_requested_pages.clear();
    for (const uint64_t page_id : requested_pages) {
      if (std::find(refine_page_ids.begin(), refine_page_ids.end(), page_id) ==
              refine_page_ids.end() &&
          std::find(refine_requested_pages.begin(), refine_requested_pages.end(), page_id) ==
              refine_requested_pages.end()) {
        refine_requested_pages.push_back(page_id);
      }
    }
    std::sort(refine_requested_pages.begin(), refine_requested_pages.end());
    if (io_cache_ == nullptr) {
      for (const uint64_t page_id : refine_requested_pages) {
        if (refine_scratch_page >= defaults::MAX_N_SECTOR_READS) {
          throw diskann_exception_t("Cell-PQ refinement exhausted query scratch pages", -1);
        }
        char* destination = sector_scratch + refine_scratch_page * defaults::SECTOR_LEN;
        ++refine_scratch_page;
        refine_read_requests.emplace_back(page_id * defaults::SECTOR_LEN, defaults::SECTOR_LEN,
                                          destination);
        refine_page_ids.push_back(page_id);
        refine_page_buffers.push_back(destination);
        if (stats != nullptr) {
          ++stats->n_4k;
          ++stats->n_ios;
          ++stats->random_ios;
          stats->read_size += defaults::SECTOR_LEN;
        }
        ++num_ios;
      }
      return;
    }
    const auto reservations = io_cache_->reserve_many(refine_requested_pages);
    for (size_t index = 0; index < refine_requested_pages.size(); ++index) {
      const uint64_t page_id = refine_requested_pages[index];
      const auto& reservation = reservations[index];
      char* destination = reservation.data;
      if (reservation.state == io_cache_reservation_state_t::HIT) {
        if (stats != nullptr) {
          ++stats->n_cache_hits;
        }
      } else {
        if (reservation.state == io_cache_reservation_state_t::BYPASS) {
          if (refine_scratch_page >= defaults::MAX_N_SECTOR_READS) {
            throw diskann_exception_t("Cell-PQ refinement exhausted query scratch pages", -1);
          }
          destination = sector_scratch + refine_scratch_page * defaults::SECTOR_LEN;
          ++refine_scratch_page;
        }
        refine_read_requests.emplace_back(page_id * defaults::SECTOR_LEN, defaults::SECTOR_LEN,
                                          destination);
        if (stats != nullptr) {
          ++stats->n_4k;
          ++stats->n_ios;
          ++stats->random_ios;
          stats->read_size += defaults::SECTOR_LEN;
        }
        ++num_ios;
      }
      refine_page_ids.push_back(page_id);
      refine_page_buffers.push_back(destination);
      refine_reservations.push_back(reservation);
    }
  };
  using phase_io_candidates_t =
      std::conditional_t<track_beam_phase, std::vector<neighbor_t>, disabled_beam_phase_profile_t>;
  phase_io_candidates_t phase_io_candidates;
  uint64_t beam_phase_loop_start_us = 0;
  size_t beam_phase_max_marker = 0;
  uint64_t beam_phase_classified_ios = 0;
  uint64_t beam_phase_useful_ios = 0;
  bool cell_convergence_ready = false;
  bool terminal_convergence_complete = false;
  bool terminal_convergence_started = false;
  bool cell_batch_exact_pending = false;
  bool cell_batch_only_round = false;
  uint64_t current_beam_width =
      enable_dynamic_width_ ? std::min<uint64_t>(dynamic_width_initial_, beam_width) : beam_width;
  uint32_t completed_base_hops = 0;
  uint32_t terminal_refine_target_hops = 0;
  if constexpr (track_beam_phase) {
    phase_io_candidates.reserve(beam_width);
    stats->beam_phase_setup_us = static_cast<float>(query_timer.elapsed());
    beam_phase_loop_start_us = query_timer.elapsed();
  }

  auto unexpanded_macro_candidates = [&]() {
    uint64_t count = 0;
    if constexpr (use_hybrid) {
      for (size_t position = 0; position < candidate_queue.size(); ++position) {
        const auto candidate = candidate_queue[position];
        if (!candidate.expanded &&
            hybrid_state->origin(candidate.id) != community_polar_candidate_origin_t::BASE_EDGE) {
          ++count;
        }
      }
    }
    return count;
  };

  auto candidate_threshold = [&]() {
    return candidate_queue.size() < candidate_queue.capacity()
               ? std::numeric_limits<float>::infinity()
               : candidate_queue[candidate_queue.size() - 1].distance;
  };

  auto cell_candidate_threshold = [&]() {
    if constexpr (!use_hybrid) {
      return candidate_threshold();
    } else {
      if (candidate_queue.size() == 0 || (!hybrid_search->config().cell_use_provisional_frontier &&
                                          candidate_queue.size() < candidate_queue.capacity())) {
        return std::numeric_limits<float>::infinity();
      }
      return candidate_queue[candidate_queue.size() - 1].distance;
    }
  };

#if defined(__linux__)
  auto prefetch_gateway_route = [&](const community_polar_candidate_batch_t& batch,
                                    uint32_t preferred_target_node) {
    if (!enable_gateway_prefetch_ ||
        batch.origin != community_polar_candidate_origin_t::COMMUNITY_GATEWAY ||
        batch.gateway_proposals.empty()) {
      return;
    }
    harvest_prefetches(false);
    const community_polar_gateway_proposal_t* best_proposal = nullptr;
    float best_landing_distance = std::numeric_limits<float>::infinity();
    for (const auto& proposal : batch.gateway_proposals) {
      const auto& hint = io_index_->gateway_prefetch_hint(proposal.gateway_id);
      if (hint.target_node != proposal.target_node) {
        throw diskann_exception_t("Gateway prefetch hint target disagrees with query sidecar", -1);
      }
      const float distance =
          hybrid_search->cell_query_distance_squared(*hybrid_state, hint.landing_cell);
      if (proposal.target_node == preferred_target_node) {
        best_proposal = &proposal;
        best_landing_distance = distance;
        break;
      }
      if (best_proposal == nullptr ||
          std::tie(distance, proposal.gateway_id) <
              std::tie(best_landing_distance, best_proposal->gateway_id)) {
        best_proposal = &proposal;
        best_landing_distance = distance;
      }
    }
    if (best_proposal == nullptr || !std::isfinite(best_landing_distance)) {
      return;
    }
    if (stats != nullptr) {
      ++stats->gateway_prefetch_predictions;
    }
    const auto& hint = io_index_->gateway_prefetch_hint(best_proposal->gateway_id);
    submit_prefetch(hint.landing_node_page, true, completed_base_hops);
    if (gateway_prefetch_max_outstanding_ < 2) {
      return;
    }
    uint64_t best_page = hint.landing_cell_first_page;
    float best_page_score = best_landing_distance * 0.5F;
    uint32_t best_count = hint.outgoing_transition_count;
    for (size_t successor = 0; successor < hint.successor_cells.size(); ++successor) {
      if (hint.successor_cells[successor] == UINT32_MAX ||
          hint.successor_first_pages[successor] == hint.landing_node_page) {
        continue;
      }
      const float distance = hybrid_search->cell_query_distance_squared(
          *hybrid_state, hint.successor_cells[successor]);
      const float probability = hint.outgoing_transition_count == 0
                                    ? 0.0F
                                    : static_cast<float>(hint.transition_counts[successor]) /
                                          hint.outgoing_transition_count;
      const float score = distance / (1.0F + 4.0F * probability);
      if (std::tie(score, best_count, hint.successor_first_pages[successor]) <
          std::tie(best_page_score, hint.transition_counts[successor], best_page)) {
        best_page = hint.successor_first_pages[successor];
        best_page_score = score;
        best_count = hint.transition_counts[successor];
      }
    }
    if (best_page != hint.landing_node_page) {
      submit_prefetch(best_page, true, completed_base_hops);
    }
  };
#endif

  auto score_macro_batch = [&](const community_polar_candidate_batch_t& batch) {
    if constexpr (!use_hybrid) {
      static_cast<void>(batch);
      return;
    } else {
      static thread_local std::vector<uint32_t> unseen;
      static thread_local std::vector<neighbor_t> scored;
      unseen.clear();
      unseen.reserve(batch.node_ids.size());
      for (const uint32_t node : batch.node_ids) {
        if (!contains_visited(node)) {
          unseen.push_back(node);
        }
      }
      if (batch.origin == community_polar_candidate_origin_t::POLAR_CELL && stats != nullptr) {
        stats->cell_nodes_already_visited +=
            static_cast<uint32_t>(batch.node_ids.size() - unseen.size());
        if (hybrid_search->config().cell_value_telemetry &&
            batch.block_ordinal < k_cell_value_max_blocks) {
          stats->cell_value_already_visited[batch.block_ordinal] =
              static_cast<uint32_t>(batch.node_ids.size() - unseen.size());
        }
      }
      if (unseen.empty()) {
        if (batch.origin == community_polar_candidate_origin_t::POLAR_CELL && stats != nullptr) {
          ++stats->cell_blocks_without_unseen_nodes;
        }
        return;
      }
      const bool is_cell = batch.origin == community_polar_candidate_origin_t::POLAR_CELL;
      const bool queue_was_full = candidate_queue.size() == candidate_queue.capacity();
      const bool threshold_available =
          queue_was_full || (is_cell && hybrid_search->config().cell_use_provisional_frontier &&
                             candidate_queue.size() != 0);
      const float batch_threshold = threshold_available
                                        ? candidate_queue[candidate_queue.size() - 1].distance
                                        : std::numeric_limits<float>::infinity();
      if (is_cell && stats != nullptr) {
        if constexpr (!insert_macro) {
          if (threshold_available && batch_threshold > 0.0F &&
              std::isfinite(batch.evidence_distance_squared)) {
            stats->cell_oracle_selected_evidence_threshold_ratio =
                batch.evidence_distance_squared / batch_threshold;
          }
        }
      }
      if (is_cell && threshold_available &&
          (!std::isfinite(batch.evidence_distance_squared) ||
           batch.evidence_distance_squared >
               batch_threshold * hybrid_search->config().cell_evidence_threshold_ratio)) {
        if (stats != nullptr) {
          ++stats->cell_blocks_low_yield;
        }
        return;
      }
      if (batch.origin == community_polar_candidate_origin_t::POLAR_CELL &&
          candidate_queue.size() == candidate_queue.capacity() &&
          batch.lower_bound_squared >= candidate_queue[candidate_queue.size() - 1].distance) {
        if (stats != nullptr) {
          ++stats->cell_blocks_hint_rejected;
        }
        return;
      }
      if (unseen.size() > defaults::MAX_GRAPH_DEGREE) {
        throw diskann_exception_t("Community-Polar batch exceeds PQ scratch capacity", -1);
      }
      load_hybrid_candidate_codes(unseen);
      if constexpr (track_beam_phase) {
        cpu_timer.reset();
      }
      compute_pq_distances(pq_coord_scratch, unseen.size(), distance_scratch);
      if (batch.origin == community_polar_candidate_origin_t::COMMUNITY_GATEWAY) {
        hybrid_search->record_gateway_scores(*hybrid_state, static_cast<uint32_t>(unseen.size()));
        if (stats != nullptr) {
          stats->gateway_nodes_scored += static_cast<uint32_t>(unseen.size());
        }
      } else if (stats != nullptr) {
        stats->cell_nodes_scored += static_cast<uint32_t>(unseen.size());
        if (hybrid_search->config().cell_value_telemetry &&
            batch.block_ordinal < k_cell_value_max_blocks) {
          stats->cell_value_nodes_scored[batch.block_ordinal] =
              static_cast<uint32_t>(unseen.size());
        }
      }
      if (stats != nullptr) {
        stats->n_cmps += static_cast<uint32_t>(unseen.size());
        if constexpr (track_beam_phase) {
          stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
        }
      }

      scored.clear();
      scored.reserve(unseen.size());
      for (size_t position = 0; position < unseen.size(); ++position) {
        scored.emplace_back(unseen[position], distance_scratch[position]);
        if (stats != nullptr && distance_scratch[position] < batch.lower_bound_squared) {
          ++stats->macro_hint_bound_violations;
        }
      }
      if (is_cell && batch.insert_cap == 1) {
        const auto best = std::min_element(scored.begin(), scored.end());
        std::iter_swap(scored.begin(), best);
      } else {
        std::sort(scored.begin(), scored.end());
      }
#if defined(__linux__)
      if (batch.origin == community_polar_candidate_origin_t::COMMUNITY_GATEWAY &&
          !scored.empty()) {
        prefetch_gateway_route(batch, scored.front().id);
      }
#endif
      if (batch.origin == community_polar_candidate_origin_t::COMMUNITY_GATEWAY &&
          stats != nullptr && hybrid_search->config().gateway_value_telemetry) {
        for (const auto& proposal : batch.gateway_proposals) {
          const auto found = std::find_if(scored.begin(), scored.end(), [&](const auto& candidate) {
            return candidate.id == proposal.target_node;
          });
          if (found == scored.end()) {
            continue;
          }
          gateway_value_record_t record;
          record.source_community = proposal.source_community;
          record.target_community = proposal.target_community;
          record.source_node = proposal.source_node;
          record.gateway_node = proposal.target_node;
          record.trigger_hop = stats->base_hops;
          record.trigger_base_reads = stats->n_ios;
          record.gateway_pq_rank = static_cast<uint32_t>(found - scored.begin());
          stats->gateway_value_records.push_back(record);
        }
      }
      if (batch.origin == community_polar_candidate_origin_t::COMMUNITY_GATEWAY &&
          hybrid_search->config().gateway_landing_cell_handoff) {
        for (const auto& candidate : scored) {
          hybrid_search->observe_gateway_landing_candidate(*hybrid_state, candidate.id,
                                                           candidate.distance, stats);
        }
      }
      uint32_t batch_competitive_count = 0;
      if (is_cell) {
        for (const auto& candidate : scored) {
          if (!threshold_available || candidate.distance <= batch_threshold) {
            ++batch_competitive_count;
          }
        }
        if (stats != nullptr) {
          stats->cell_nodes_competitive += batch_competitive_count;
          if (hybrid_search->config().cell_value_telemetry &&
              batch.block_ordinal < k_cell_value_max_blocks) {
            stats->cell_value_nodes_competitive[batch.block_ordinal] = batch_competitive_count;
            if (threshold_available && batch_threshold > 0.0F && !scored.empty()) {
              stats->cell_value_best_threshold_ratio[batch.block_ordinal] =
                  scored.front().distance / batch_threshold;
            }
          }
          if constexpr (!insert_macro) {
            stats->cell_oracle_selected_unseen_nodes = static_cast<uint32_t>(scored.size());
            stats->cell_oracle_selected_competitive_nodes = batch_competitive_count;
            if (threshold_available && batch_threshold > 0.0F && !scored.empty()) {
              stats->cell_oracle_selected_best_threshold_ratio =
                  scored.front().distance / batch_threshold;
            }
          }
        }
        const float competitive_fraction =
            static_cast<float>(batch_competitive_count) / static_cast<float>(scored.size());
        const bool strong_best =
            !threshold_available ||
            scored.front().distance <=
                batch_threshold * hybrid_search->config().cell_candidate_threshold_ratio;
        if (batch_competitive_count < hybrid_search->config().cell_min_competitive_count ||
            competitive_fraction < hybrid_search->config().cell_min_competitive_fraction ||
            !strong_best) {
          if (stats != nullptr) {
            ++stats->cell_blocks_low_yield;
          }
          return;
        }
        if (stats != nullptr && hybrid_search->config().cell_value_telemetry &&
            batch.block_ordinal < k_cell_value_max_blocks) {
          stats->cell_value_admitted[batch.block_ordinal] = 1;
        }
      }

      uint32_t adaptive_insert_cap = batch.insert_cap;
      if (is_cell) {
        const auto proportional_cap =
            static_cast<uint32_t>(std::ceil(static_cast<double>(batch_competitive_count) *
                                            hybrid_search->config().cell_insert_fraction));
        adaptive_insert_cap = std::min(adaptive_insert_cap, proportional_cap);
      }
      uint32_t inserted_from_batch = 0;
      uint64_t current_unexpanded_macro_candidates = unexpanded_macro_candidates();
      for (const auto& candidate : scored) {
        const bool competitive = candidate_queue.size() < candidate_queue.capacity() ||
                                 !(candidate_queue[candidate_queue.size() - 1] < candidate);
        if (competitive && stats != nullptr && !is_cell) {
          if (batch.origin == community_polar_candidate_origin_t::COMMUNITY_GATEWAY) {
            ++stats->gateway_nodes_competitive;
          }
        }
        if constexpr (!insert_macro) {
          continue;
        }
        if (inserted_from_batch >= adaptive_insert_cap) {
          continue;
        }
        const bool passes_cell_margin =
            !is_cell || !threshold_available ||
            candidate.distance <=
                batch_threshold * hybrid_search->config().cell_candidate_threshold_ratio;
        const bool guard_allows =
            graph_guard_allows_macro(search_list_size, hybrid_search->config().graph_guard_fraction,
                                     current_unexpanded_macro_candidates);
        if (!competitive || !passes_cell_margin || !guard_allows) {
          if (!competitive) {
            insert_visited(static_cast<uint32_t>(candidate.id));
          }
          continue;
        }
        if (!insert_visited(static_cast<uint32_t>(candidate.id))) {
          continue;
        }
        candidate_queue.insert(candidate);
        ++inserted_from_batch;
        ++current_unexpanded_macro_candidates;
        hybrid_state->mark_macro_inserted(candidate.id, batch.origin, batch.block_ordinal);
        if constexpr (use_io_layout) {
          if (is_cell) {
            const uint64_t page = active_vector_location(candidate.id).page_id;
            if (std::find(cell_prediction_pages.begin(), cell_prediction_pages.end(), page) ==
                cell_prediction_pages.end()) {
              cell_prediction_pages.push_back(page);
            }
          }
        }
        if (stats != nullptr) {
          if (batch.origin == community_polar_candidate_origin_t::COMMUNITY_GATEWAY) {
            ++stats->gateway_nodes_inserted;
          } else {
            ++stats->cell_nodes_inserted;
            if (hybrid_search->config().cell_value_telemetry &&
                batch.block_ordinal < k_cell_value_max_blocks) {
              ++stats->cell_value_nodes_inserted[batch.block_ordinal];
            }
          }
        }
      }
    }
  };

  auto score_terminal_cell_batch = [&](const community_polar_candidate_batch_t& batch) {
    if constexpr (!use_hybrid || !insert_macro) {
      static_cast<void>(batch);
      return;
    } else {
      if (batch.origin != community_polar_candidate_origin_t::POLAR_CELL ||
          batch.node_ids.empty()) {
        return;
      }
      if (enable_cell_batch_search_ || enable_cell_adj_correction_) {
        const uint32_t selected_cell = hybrid_search->cell_id(batch.node_ids.front());
        if (selected_cell != UINT32_MAX &&
            std::find(cell_batch_selected_cells.begin(), cell_batch_selected_cells.end(),
                      selected_cell) == cell_batch_selected_cells.end()) {
          cell_batch_selected_cells.push_back(selected_cell);
        }
      }
      if (batch.node_ids.size() > defaults::MAX_GRAPH_DEGREE) {
        throw diskann_exception_t("Terminal Cell batch exceeds PQ scratch capacity", -1);
      }
      if constexpr (track_beam_phase) {
        cpu_timer.reset();
      }
      if (!decoded_cell_pq_codes_.empty() && batch.packet_record_begin <= num_points_ &&
          batch.packet_record_count == batch.node_ids.size() &&
          batch.packet_record_count <= num_points_ - batch.packet_record_begin) {
        decoded_u8_l2_lookup(decoded_cell_pq_codes_.data() + batch.packet_record_begin * data_dim_,
                             batch.packet_record_count, data_dim_, decoded_u8_query.data(),
                             distance_scratch);
      } else if (enable_resident_u8_traversal_) {
        original_u8_l2_lookup(resident_u8_refinement_.data(), batch.node_ids.data(),
                              batch.node_ids.size(), data_dim_, decoded_u8_query.data(),
                              distance_scratch);
      } else {
        load_hybrid_candidate_codes(batch.node_ids);
        compute_pq_distances(pq_coord_scratch, batch.node_ids.size(), distance_scratch);
      }
      terminal_cell_results.reserve(terminal_cell_results.size() + batch.node_ids.size());
      for (size_t position = 0; position < batch.node_ids.size(); ++position) {
        terminal_cell_results.emplace_back(
            batch.node_ids[position],
            distance_scratch[position] * hybrid_search->config().cell_terminal_distance_scale);
      }
      if (stats != nullptr) {
        stats->cell_nodes_scored += static_cast<uint32_t>(batch.node_ids.size());
        stats->cell_terminal_candidates += static_cast<uint32_t>(batch.node_ids.size());
        stats->n_cmps += static_cast<uint32_t>(batch.node_ids.size());
        if constexpr (track_beam_phase) {
          stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
        }
      }
    }
  };

  auto rank_terminal_cells_for_exact_batch = [&]() {
    if ((!enable_cell_batch_search_ &&
         !(enable_cell_adj_correction_ && enable_cell_leaf_refinement_)) ||
        cell_batch_selected_cells.size() <= 1 ||
        terminal_cell_results.empty()) {
      return;
    }
    struct cell_yield_t {
      uint32_t cell = 0;
      uint32_t top_candidates = 0;
      float best_distance = std::numeric_limits<float>::infinity();
    };
    std::vector<size_t> candidate_order(terminal_cell_results.size());
    std::iota(candidate_order.begin(), candidate_order.end(), 0);
    const size_t yield_window = std::min<size_t>(search_list_size, candidate_order.size());
    std::partial_sort(candidate_order.begin(), candidate_order.begin() + yield_window,
                      candidate_order.end(), [&](size_t left, size_t right) {
                        return terminal_cell_results[left] < terminal_cell_results[right];
                      });
    std::vector<cell_yield_t> yields;
    yields.reserve(cell_batch_selected_cells.size());
    for (const uint32_t cell : cell_batch_selected_cells) {
      yields.push_back({cell, 0, std::numeric_limits<float>::infinity()});
    }
    for (size_t rank = 0; rank < yield_window; ++rank) {
      const auto& candidate = terminal_cell_results[candidate_order[rank]];
      const uint32_t cell = hybrid_search->cell_id(static_cast<uint32_t>(candidate.id));
      const auto position = std::find_if(yields.begin(), yields.end(),
                                         [cell](const auto& value) { return value.cell == cell; });
      if (position == yields.end()) {
        continue;
      }
      ++position->top_candidates;
      position->best_distance = std::min(position->best_distance, candidate.distance);
    }
    std::sort(yields.begin(), yields.end(), [](const auto& left, const auto& right) {
      return std::make_tuple(std::numeric_limits<uint32_t>::max() - left.top_candidates,
                             left.best_distance, left.cell) <
             std::make_tuple(std::numeric_limits<uint32_t>::max() - right.top_candidates,
                             right.best_distance, right.cell);
    });
    cell_batch_selected_cells.clear();
    for (const auto& value : yields) {
      cell_batch_selected_cells.push_back(value.cell);
    }
  };

  auto score_cell_adj_candidates = [&]() {
    if (!enable_cell_adj_correction_ || cell_batch_selected_cells.empty() ||
        search_list_size < cell_adj_min_search_list_size_ ||
        search_list_size > cell_adj_max_search_list_size_) {
      return;
    }
    static thread_local std::unordered_map<uint32_t, uint32_t> support;
    static thread_local std::vector<std::pair<uint32_t, uint32_t>> support_ranked;
    static thread_local std::vector<std::tuple<uint32_t, float, uint32_t>> ranked;
    static thread_local std::vector<uint32_t> corrected_cells;
    static thread_local std::vector<uint32_t> expanded_cells;
    support.clear();
    support_ranked.clear();
    ranked.clear();
    corrected_cells.clear();
    expanded_cells.clear();
    // Treat the configured expansion count as one total Cell-Adj work budget. When hierarchical
    // routing selects a forest, divide it deterministically across the leading leaves. This keeps
    // multiple graph-local directions for boundary queries without increasing edges examined. A
    // single selected leaf preserves the original repeated-hop behavior.
    const size_t source_cell_count =
        std::min<size_t>(cell_adj_expansion_cells_, cell_batch_selected_cells.size());
    if (cell_adjacency_index_ != nullptr) {
      for (size_t source = 0; source < source_cell_count; ++source) {
        uint32_t source_cell = cell_batch_selected_cells[source];
        const uint32_t expansion_budget =
            cell_adj_expansion_cells_ / static_cast<uint32_t>(source_cell_count) +
            (source < cell_adj_expansion_cells_ % source_cell_count ? 1U : 0U);
        for (uint32_t expansion = 0; expansion < expansion_budget; ++expansion) {
          expanded_cells.push_back(source_cell);
          const auto neighbors = cell_adjacency_index_->neighbors(source_cell);
          if (stats != nullptr) {
            stats->cell_adj_neighbor_edges_considered +=
                static_cast<uint32_t>(neighbors.size());
          }
          uint32_t next_cell = UINT32_MAX;
          auto next_key = std::make_tuple(std::numeric_limits<float>::infinity(), UINT32_MAX,
                                          UINT32_MAX);
          for (uint32_t rank = 0; rank < neighbors.size(); ++rank) {
            const uint32_t target_cell = neighbors[rank];
            if (std::find(cell_batch_selected_cells.begin(), cell_batch_selected_cells.end(),
                          target_cell) != cell_batch_selected_cells.end()) {
              continue;
            }
            const auto [position, inserted] = support.emplace(target_cell, rank);
            if (!inserted) {
              position->second = std::min(position->second, rank);
            }
            if (std::find(expanded_cells.begin(), expanded_cells.end(), target_cell) !=
                expanded_cells.end()) {
              continue;
            }
            const auto key = std::make_tuple(
                hybrid_search->cell_query_distance_squared(*hybrid_state, target_cell), rank,
                target_cell);
            if (key < next_key) {
              next_key = key;
              next_cell = target_cell;
            }
          }
          if (next_cell == UINT32_MAX) {
            break;
          }
          source_cell = next_cell;
        }
      }
      support_ranked.reserve(support.size());
      for (const auto& [cell, support_rank] : support) {
        support_ranked.emplace_back(support_rank, cell);
      }
    } else {
      for (size_t source = 0; source < source_cell_count; ++source) {
        const uint32_t source_cell = cell_batch_selected_cells[source];
        for (const uint32_t source_node : hybrid_search->cell_node_ids(source_cell)) {
          const auto neighbors = io_index_->neighbors(source_node);
          if (stats != nullptr) {
            stats->cell_adj_neighbor_edges_considered +=
                static_cast<uint32_t>(neighbors.size());
          }
          for (const uint32_t neighbor : neighbors) {
            const uint32_t target_cell = hybrid_search->cell_id(neighbor);
            if (target_cell == source_cell ||
                std::find(cell_batch_selected_cells.begin(), cell_batch_selected_cells.end(),
                          target_cell) != cell_batch_selected_cells.end()) {
              continue;
            }
            auto& count = support[target_cell];
            if (count != std::numeric_limits<uint32_t>::max()) {
              ++count;
            }
          }
        }
      }
      support_ranked.reserve(support.size());
      for (const auto& [cell, count] : support) {
        support_ranked.emplace_back(std::numeric_limits<uint32_t>::max() - count, cell);
      }
    }
    const size_t shortlist_count =
        std::min<size_t>(cell_adj_support_shortlist_, support_ranked.size());
    std::partial_sort(support_ranked.begin(), support_ranked.begin() + shortlist_count,
                      support_ranked.end());
    ranked.reserve(shortlist_count);
    for (size_t rank = 0; rank < shortlist_count; ++rank) {
      const auto [inverse_support, cell] = support_ranked[rank];
      ranked.emplace_back(
          inverse_support, hybrid_search->cell_query_distance_squared(*hybrid_state, cell), cell);
    }
    // The support shortlist approximates a persisted bounded Cell graph. Within it, query
    // geometry chooses the direction; support and Cell ID provide deterministic tie-breaks.
    std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
      return std::make_tuple(std::get<1>(left), std::get<0>(left), std::get<2>(left)) <
             std::make_tuple(std::get<1>(right), std::get<0>(right), std::get<2>(right));
    });
    const size_t candidate_count = std::min<size_t>(cell_adj_candidate_cells_, ranked.size());
    for (size_t rank = 0; rank < candidate_count; ++rank) {
      const uint32_t cell = std::get<2>(ranked[rank]);
      const auto nodes = hybrid_search->cell_node_ids(cell);
      community_polar_candidate_batch_t batch;
      batch.origin = community_polar_candidate_origin_t::POLAR_CELL;
      batch.node_ids = nodes;
      batch.packet_record_count = nodes.size();
      score_terminal_cell_batch(batch);
      corrected_cells.push_back(cell);
      if (stats != nullptr) {
        ++stats->cell_adj_cells_scored;
      }
    }
    // Cell-Adj is a physical landing correction, not only another PQ hint. Make the bounded,
    // deterministically ranked neighbor Cells eligible for the following whole-Cell read. Append
    // after scanning the source vector so cross-Cell candidates cannot recursively expand here.
    for (const uint32_t cell : corrected_cells) {
      if (std::find(cell_batch_selected_cells.begin(), cell_batch_selected_cells.end(), cell) ==
          cell_batch_selected_cells.end()) {
        cell_batch_selected_cells.push_back(cell);
      }
    }
  };

  auto schedule_interleaved_cell = [&]() {
    if (!enable_interleaved_cell_batch_search_ ||
        cell_batch_loaded_cells.size() >= cell_batch_max_cached_expansions_) {
      return false;
    }
    static thread_local std::unordered_map<uint32_t, uint32_t> support;
    static thread_local std::unordered_map<uint32_t, float> best_distances;
    support.clear();
    best_distances.clear();
    for (size_t position = 0; position < candidate_queue.size(); ++position) {
      const auto candidate = candidate_queue[position];
      const uint32_t cell = hybrid_search->cell_id(static_cast<uint32_t>(candidate.id));
      if (cell == std::numeric_limits<uint32_t>::max() ||
          std::find(cell_batch_loaded_cells.begin(), cell_batch_loaded_cells.end(), cell) !=
              cell_batch_loaded_cells.end()) {
        continue;
      }
      auto& count = support[cell];
      if (count != std::numeric_limits<uint32_t>::max()) {
        ++count;
      }
      const auto [distance, inserted] = best_distances.emplace(cell, candidate.distance);
      if (!inserted && candidate.distance < distance->second) {
        distance->second = candidate.distance;
      }
    }
    uint32_t selected = std::numeric_limits<uint32_t>::max();
    uint32_t selected_support = 0;
    float selected_distance = std::numeric_limits<float>::infinity();
    for (const auto& [cell, count] : support) {
      const float distance = best_distances[cell];
      if (selected == std::numeric_limits<uint32_t>::max() ||
          std::make_tuple(std::numeric_limits<uint32_t>::max() - count, distance, cell) <
              std::make_tuple(std::numeric_limits<uint32_t>::max() - selected_support,
                              selected_distance, selected)) {
        selected = cell;
        selected_support = count;
        selected_distance = distance;
      }
    }
    if (selected == std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    cell_batch_selected_cells.clear();
    cell_batch_selected_cells.push_back(selected);
    cell_batch_exact_pending = true;
    cell_batch_only_round = true;
    return true;
  };

  auto record_base_expansion = [&](uint32_t node_id) {
    if (stats != nullptr && stats->base_trace_enabled) {
      base_expansion_record_t record;
      record.node_id = node_id;
      record.hop = stats->base_hops;
      record.base_reads = stats->n_ios;
      if constexpr (use_hybrid) {
        record.community_id = hybrid_search->community_id(node_id);
        record.cell_id = hybrid_search->cell_id(node_id);
      }
      stats->base_expansion_records.push_back(record);
    }
    if constexpr (use_hybrid) {
      if constexpr (forced_terminal_tree) {
        if (!hybrid_search->config().community_skip) {
          return;
        }
      }
      if (!enable_hybrid_graph_hooks) {
        return;
      }
      if constexpr (!insert_macro) {
        hybrid_search->observe_cell_oracle_expansion(*hybrid_state, node_id);
      }
      const uint32_t cell_block_ordinal = hybrid_search->config().cell_value_telemetry
                                              ? hybrid_state->cell_block_ordinal(node_id)
                                              : std::numeric_limits<uint32_t>::max();
      const auto candidate_origin = hybrid_state->origin(node_id);
      if (stats != nullptr) {
        if (hybrid_search->config().gateway_value_telemetry) {
          for (auto& record : stats->gateway_value_records) {
            if (record.later_expanded == 0 && record.gateway_node == node_id) {
              record.later_expanded = 1;
              record.first_later_expansion_hop = stats->base_hops;
              record.first_later_expansion_read = stats->n_ios;
            }
          }
          const uint32_t expanded_community = hybrid_search->community_id(node_id);
          for (auto& record : stats->gateway_value_records) {
            if (record.later_expanded != 0 && record.target_community == expanded_community) {
              ++record.future_expansions_covered;
            }
          }
        }
        if (candidate_origin == community_polar_candidate_origin_t::COMMUNITY_GATEWAY) {
          ++stats->gateway_nodes_later_expanded;
        } else if (candidate_origin == community_polar_candidate_origin_t::POLAR_CELL) {
          ++stats->cell_nodes_later_expanded;
          if (cell_block_ordinal < k_cell_value_max_blocks) {
            ++stats->cell_value_future_expansions[cell_block_ordinal];
          }
        }
      }
      hybrid_search->observe_base_expansion(*hybrid_state, node_id, stats);
    }
  };

  auto mark_base_reached = [&](uint32_t node_id) {
    if constexpr (use_hybrid && !forced_terminal_tree) {
      if (enable_hybrid_graph_hooks) {
        hybrid_state->mark_base_reached(node_id);
      }
    }
  };

  auto record_base_candidates = [&](uint32_t source_node, const uint32_t* neighbors, uint32_t count,
                                    const float* neighbor_distances) {
    if constexpr (use_hybrid) {
      if constexpr (forced_terminal_tree) {
        return;
      }
      if (hybrid_search->config().source_cell_batching) {
        return;
      }
      if (!hybrid_search->accepts_cell_evidence(*hybrid_state)) {
        return;
      }
      static thread_local std::vector<uint32_t> evidence_nodes;
      static thread_local std::vector<float> evidence_distances;
      evidence_nodes.clear();
      evidence_distances.clear();
      evidence_nodes.reserve(count);
      evidence_distances.reserve(count);
      const float threshold = candidate_threshold();
      for (uint32_t offset = 0; offset < count; ++offset) {
        if (contains_visited(neighbors[offset]) || neighbor_distances[offset] > threshold) {
          continue;
        }
        evidence_nodes.push_back(neighbors[offset]);
        evidence_distances.push_back(neighbor_distances[offset]);
      }
      hybrid_search->observe_base_candidates(*hybrid_state, source_node, evidence_nodes,
                                             evidence_distances, stats);
    }
  };

  while (num_ios < io_limit) {
#if defined(__linux__)
    if (enable_node_prefetch_ || enable_cell_prefetch_ || enable_gateway_prefetch_) {
      harvest_prefetches(false);
    }
#endif
    if constexpr (track_beam_phase) {
      ++stats->beam_phase_total_search_rounds;
      phase_io_candidates.clear();
    }
    if constexpr (use_hybrid) {
      if (terminal_convergence_started && !cell_batch_exact_pending &&
          completed_base_hops >= terminal_refine_target_hops) {
        const bool interleaved_budget_remaining =
            enable_interleaved_cell_batch_search_ &&
            cell_batch_loaded_cells.size() < cell_batch_max_cached_expansions_;
        if (!interleaved_budget_remaining) {
          terminal_convergence_complete = true;
          break;
        }
        const bool graph_interval_complete =
            completed_base_hops - interleaved_last_cell_hop >=
            interleaved_cell_batch_graph_hops_;
        if ((graph_interval_complete || !candidate_queue.has_unexpanded_node()) &&
            !schedule_interleaved_cell()) {
          terminal_convergence_complete = true;
          break;
        }
      }
      if (hybrid_search->has_community_work(*hybrid_state)) {
        const auto batch = hybrid_search->expand_next_community(*hybrid_state, stats);
        if (batch.has_value()) {
          score_macro_batch(*batch);
        }
      }
      if constexpr (insert_macro) {
        if (hybrid_search->config().cell_terminal_convergence && !terminal_convergence_started &&
            (completed_base_hops >= hybrid_search->terminal_approach_hops(search_list_size) ||
             !candidate_queue.has_unexpanded_node())) {
          if (hybrid_search->config().cell_hierarchy_graph_guided_routing) {
            for (size_t position = 0; position < candidate_queue.size(); ++position) {
              const auto candidate = candidate_queue[position];
              hybrid_search->observe_source_cell(*hybrid_state,
                                                 static_cast<uint32_t>(candidate.id),
                                                 candidate.distance, nullptr);
            }
          }
          hybrid_search->prepare_terminal_convergence(*hybrid_state, search_list_size);
          while (hybrid_search->has_cell_work(*hybrid_state)) {
            const auto batch = hybrid_search->expand_next_cell(
                *hybrid_state, search_list_size, std::numeric_limits<float>::infinity(), stats);
            if (!batch.has_value()) {
              break;
            }
            score_terminal_cell_batch(*batch);
          }
          rank_terminal_cells_for_exact_batch();
          score_cell_adj_candidates();
          rank_terminal_cells_for_exact_batch();
          if (terminal_cell_results.size() >= top_k) {
            if (stats != nullptr) {
              stats->cell_terminal_activated = 1;
            }
            const uint32_t effective_refine_hops =
                hybrid_search->terminal_refine_hops(search_list_size);
            if (effective_refine_hops == 0 &&
                !enable_cell_batch_search_) {
              terminal_convergence_complete = true;
              break;
            }
            const size_t refine_candidates =
                std::min<size_t>(search_list_size, terminal_cell_results.size());
            std::partial_sort(terminal_cell_results.begin(),
                              terminal_cell_results.begin() + refine_candidates,
                              terminal_cell_results.end());
            for (size_t position = 0; position < refine_candidates; ++position) {
              const auto& candidate = terminal_cell_results[position];
              if (!insert_visited(static_cast<uint32_t>(candidate.id))) {
                continue;
              }
              candidate_queue.insert(candidate);
              hybrid_state->mark_macro_inserted(candidate.id,
                                                community_polar_candidate_origin_t::POLAR_CELL);
              if constexpr (use_io_layout) {
                const uint64_t page = active_vector_location(candidate.id).page_id;
                if (std::find(cell_prediction_pages.begin(), cell_prediction_pages.end(), page) ==
                    cell_prediction_pages.end()) {
                  cell_prediction_pages.push_back(page);
                }
              }
            }
            terminal_convergence_started = true;
            cell_batch_exact_pending = enable_cell_batch_search_;
            cell_batch_only_round = enable_cell_batch_search_ && effective_refine_hops == 0;
            terminal_refine_target_hops = completed_base_hops + effective_refine_hops;
          }
        }
      }
      if (!hybrid_search->config().cell_terminal_convergence &&
          hybrid_search->config().cell_expansion && !cell_convergence_ready) {
        size_t marker = 0;
        while (marker < candidate_queue.size() && candidate_queue[marker].expanded) {
          ++marker;
        }
        cell_convergence_ready = marker >= hybrid_search->config().cell_activation_marker;
      }
      const float live_candidate_threshold = cell_candidate_threshold();
      if (!hybrid_search->config().cell_terminal_convergence && cell_convergence_ready &&
          hybrid_search->has_cell_work(*hybrid_state, live_candidate_threshold)) {
        if constexpr (!insert_macro) {
          if (hybrid_search->begin_cell_oracle(*hybrid_state, stats, live_candidate_threshold)) {
            if (enable_cell_pq_dense_visited_) {
              for (const uint32_t node_id : dense_visited_ids) {
                hybrid_search->record_cell_oracle_previsited(*hybrid_state, node_id);
              }
            } else {
              for (const auto node_id : visited) {
                hybrid_search->record_cell_oracle_previsited(*hybrid_state,
                                                             static_cast<uint32_t>(node_id));
              }
            }
          }
        }
        const auto batch = hybrid_search->expand_next_cell(*hybrid_state, search_list_size,
                                                           live_candidate_threshold, stats);
        if (batch.has_value()) {
          if constexpr (!insert_macro) {
            hybrid_search->record_cell_oracle_selection(*hybrid_state, batch->object_id,
                                                        batch->block_ordinal);
          }
          score_macro_batch(*batch);
        }
      }
    }
    if (!candidate_queue.has_unexpanded_node() && !cell_batch_exact_pending) {
      if constexpr (use_hybrid && !forced_terminal_tree) {
        if (terminal_convergence_started) {
          terminal_convergence_complete = true;
          break;
        }
        if (hybrid_search->has_community_work(*hybrid_state) ||
            hybrid_search->has_cell_work(*hybrid_state, cell_candidate_threshold())) {
          continue;
        }
      }
      if constexpr (use_hybrid && forced_terminal_tree) {
        terminal_convergence_complete = terminal_convergence_started;
      }
      break;
    }
    frontier.clear();
    frontier_candidate_distances.clear();
    frontier_neighborhoods.clear();
    frontier_read_requests.clear();
    coalesced_read_requests.clear();
    coalesced_page_copies.clear();
    frontier_page_ids.clear();
    frontier_page_buffers.clear();
    frontier_cache_reservations.clear();
    cached_neighborhoods.clear();
    sector_scratch_index = 0;
    const bool community_pq_approach_round =
        use_hybrid && use_io_layout && hybrid_search->config().community_skip &&
        hybrid_search->config().community_pq_approach_hops != 0 &&
        completed_base_hops < hybrid_search->config().community_pq_approach_hops;
    static thread_local std::vector<uint64_t> cell_batch_miss_pages;
    static thread_local std::vector<size_t> ranked_cell_candidate_order;
    static thread_local std::vector<uint64_t> ranked_cell_pages;
    cell_batch_miss_pages.clear();
    ranked_cell_candidate_order.clear();
    ranked_cell_pages.clear();
    cell_batch_exact_nodes.clear();

    if constexpr (use_hybrid && use_io_layout) {
      if (enable_cell_batch_search_ && terminal_convergence_started &&
          cell_batch_loaded_cells.size() < cell_batch_max_cached_expansions_) {
        if (enable_ranked_cell_page_refinement_) {
          ranked_cell_candidate_order.resize(terminal_cell_results.size());
          std::iota(ranked_cell_candidate_order.begin(), ranked_cell_candidate_order.end(), 0);
          std::sort(ranked_cell_candidate_order.begin(), ranked_cell_candidate_order.end(),
                    [&](size_t left, size_t right) {
                      return terminal_cell_results[left] < terminal_cell_results[right];
                    });
          for (const size_t position : ranked_cell_candidate_order) {
            const uint32_t node = static_cast<uint32_t>(terminal_cell_results[position].id);
            const uint32_t cell = hybrid_search->cell_id(node);
            if (std::find(cell_batch_selected_cells.begin(), cell_batch_selected_cells.end(),
                          cell) == cell_batch_selected_cells.end()) {
              continue;
            }
            const uint64_t page = active_vector_location(node).page_id;
            if (std::find(ranked_cell_pages.begin(), ranked_cell_pages.end(), page) ==
                ranked_cell_pages.end()) {
              ranked_cell_pages.push_back(page);
              if (ranked_cell_pages.size() >= ranked_cell_page_budget) {
                break;
              }
            }
          }
          for (const size_t position : ranked_cell_candidate_order) {
            if (ranked_cell_pages.size() >= ranked_cell_page_budget) {
              break;
            }
            const uint32_t node = static_cast<uint32_t>(terminal_cell_results[position].id);
            const uint32_t cell = hybrid_search->cell_id(node);
            if (cell >= io_index_->cell_page_ranges().size() ||
                std::find(cell_batch_selected_cells.begin(), cell_batch_selected_cells.end(),
                          cell) == cell_batch_selected_cells.end()) {
              continue;
            }
            const auto& range = io_index_->cell_page_ranges()[cell];
            for (uint32_t page_offset = 0;
                 page_offset < range.page_count &&
                 ranked_cell_pages.size() < ranked_cell_page_budget;
                 ++page_offset) {
              const uint64_t page = range.first_page + page_offset;
              if (std::find(ranked_cell_pages.begin(), ranked_cell_pages.end(), page) ==
                  ranked_cell_pages.end()) {
                ranked_cell_pages.push_back(page);
              }
            }
          }
          for (const uint32_t cell : cell_batch_selected_cells) {
            if (ranked_cell_pages.size() >= ranked_cell_page_budget) {
              break;
            }
            if (cell >= io_index_->cell_page_ranges().size()) {
              continue;
            }
            const auto& range = io_index_->cell_page_ranges()[cell];
            for (uint32_t page_offset = 0;
                 page_offset < range.page_count &&
                 ranked_cell_pages.size() < ranked_cell_page_budget;
                 ++page_offset) {
              const uint64_t page = range.first_page + page_offset;
              if (std::find(ranked_cell_pages.begin(), ranked_cell_pages.end(), page) ==
                  ranked_cell_pages.end()) {
                ranked_cell_pages.push_back(page);
              }
            }
          }
          if (ranked_cell_pages.empty()) {
            throw diskann_exception_t("Ranked Cell-page refinement selected no physical page", -1);
          }
          cell_batch_miss_pages = ranked_cell_pages;
          for (const uint32_t cell : cell_batch_selected_cells) {
            bool cell_selected = false;
            for (const uint32_t node : hybrid_search->cell_node_ids(cell)) {
              if (std::find(ranked_cell_pages.begin(), ranked_cell_pages.end(),
                            active_vector_location(node).page_id) == ranked_cell_pages.end()) {
                continue;
              }
              cell_batch_exact_nodes.push_back(node);
              cell_selected = true;
            }
            if (cell_selected) {
              cell_batch_loaded_cells.push_back(cell);
            }
          }
        } else {
          for (const uint32_t cell : cell_batch_selected_cells) {
            if (cell_batch_loaded_cells.size() >= cell_batch_max_cached_expansions_ ||
                cell >= io_index_->cell_page_ranges().size() ||
                std::find(cell_batch_loaded_cells.begin(), cell_batch_loaded_cells.end(), cell) !=
                    cell_batch_loaded_cells.end()) {
              continue;
            }
            const auto& range = io_index_->cell_page_ranges()[cell];
            if (range.page_count <=
                defaults::MAX_N_SECTOR_READS - cell_batch_miss_pages.size()) {
              cell_batch_loaded_cells.push_back(cell);
              for (uint32_t page = 0; page < range.page_count; ++page) {
                cell_batch_miss_pages.push_back(range.first_page + page);
              }
              const auto nodes = hybrid_search->cell_node_ids(cell);
              cell_batch_exact_nodes.insert(cell_batch_exact_nodes.end(), nodes.begin(),
                                            nodes.end());
              if (enable_interleaved_cell_batch_search_) {
                break;
              }
            }
          }
        }
      }
      if (cell_batch_exact_pending && cell_batch_miss_pages.empty()) {
        cell_batch_exact_pending = false;
        cell_batch_only_round = false;
      }
    }

    const bool cell_batch_round_active = enable_cell_batch_search_ && cell_batch_exact_pending;
    uint32_t num_seen = 0;
    const uint64_t candidate_batch_limit =
        cell_batch_round_active
            ? beam_width + static_cast<uint64_t>(cell_batch_max_cached_expansions_)
            : current_beam_width;
    if (cell_batch_round_active && stats != nullptr) {
      ++stats->cell_batch_search_rounds;
    }
    const auto has_batch_capacity = [&]() {
      return cell_batch_round_active ? cell_batch_miss_pages.size() < current_beam_width
                                     : frontier.size() < current_beam_width;
    };
    while (!cell_batch_only_round && candidate_queue.has_unexpanded_node() &&
           has_batch_capacity() &&
           num_seen < candidate_batch_limit) {
      const auto neighbor = candidate_queue.closest_unexpanded();
      ++num_seen;
      if (memgraph_injected.erase(neighbor.id) != 0 && stats != nullptr) {
        ++stats->memgraph_nodes_later_expanded;
      }
      const auto cached = neighborhood_cache_.find(neighbor.id);
      if (cached != neighborhood_cache_.end()) {
        cached_neighborhoods.emplace_back(neighbor.id, cached->second);
        if (stats != nullptr) {
          ++stats->n_cache_hits;
          if (enable_cell_batch_search_) {
            ++stats->cell_batch_cached_expansions;
          }
        }
      } else {
        bool resident_page = false;
        if constexpr (use_io_layout) {
          if (enable_cell_batch_search_ && !enable_cell_pq_traversal_ &&
              terminal_convergence_started) {
            const uint64_t page_id = active_vector_location(neighbor.id).page_id;
            const auto existing =
                std::find(frontier_page_ids.begin(), frontier_page_ids.end(), page_id);
            if (existing != frontier_page_ids.end()) {
              resident_page = true;
            } else {
              const auto reservation = io_cache_->try_reserve_hit(page_id);
              if (reservation.state == io_cache_reservation_state_t::HIT) {
                frontier_page_ids.push_back(page_id);
                frontier_page_buffers.push_back(reservation.data);
                frontier_cache_reservations.push_back(reservation);
                resident_page = true;
                if (stats != nullptr) {
                  ++stats->n_cache_hits;
                }
              }
            }
            if (!resident_page &&
                std::find(cell_batch_miss_pages.begin(), cell_batch_miss_pages.end(), page_id) ==
                    cell_batch_miss_pages.end()) {
              cell_batch_miss_pages.push_back(page_id);
            }
          }
        }
        frontier.push_back(neighbor.id);
        frontier_candidate_distances.push_back(neighbor.distance);
        if (resident_page && stats != nullptr) {
          ++stats->cell_batch_cached_expansions;
        }
        if constexpr (track_beam_phase) {
          phase_io_candidates.push_back(neighbor);
        }
      }
      node_visit_tracker_.record(neighbor.id);
    }
#if defined(__linux__)
    if (enable_node_prefetch_ && candidate_queue.has_unexpanded_node()) {
      for (size_t position = 0; position < candidate_queue.size(); ++position) {
        if (!candidate_queue[position].expanded) {
          submit_prefetch(active_vector_location(candidate_queue[position].id).page_id);
          break;
        }
      }
    }
#endif

    if (!frontier.empty() || !cell_batch_miss_pages.empty()) {
      if (!frontier.empty() && stats != nullptr) {
        ++stats->n_hops;
        ++stats->base_hops;
      }
      if (!frontier.empty()) {
        ++completed_base_hops;
      }
      if constexpr (use_hybrid && use_io_layout) {
        if (enable_cell_batch_search_) {
          if (enable_cell_pq_traversal_ && !cell_batch_miss_pages.empty()) {
            std::sort(cell_batch_miss_pages.begin(), cell_batch_miss_pages.end());
            const auto reservations = io_cache_->reserve_many(cell_batch_miss_pages);
            for (const auto& reservation : reservations) {
              char* destination = reservation.data;
              if (reservation.state == io_cache_reservation_state_t::HIT) {
                if (stats != nullptr) {
                  ++stats->n_cache_hits;
                }
              } else {
                if (reservation.state == io_cache_reservation_state_t::BYPASS) {
                  if (sector_scratch_index >= defaults::MAX_N_SECTOR_READS) {
                    throw diskann_exception_t("Cell batch exhausted query scratch pages", -1);
                  }
                  destination = sector_scratch + sector_scratch_index * defaults::SECTOR_LEN;
                  ++sector_scratch_index;
                }
                frontier_read_requests.emplace_back(reservation.page_id * defaults::SECTOR_LEN,
                                                    defaults::SECTOR_LEN, destination);
                if (stats != nullptr) {
                  ++stats->n_4k;
                  ++stats->n_ios;
                  stats->read_size += defaults::SECTOR_LEN;
                }
                ++num_ios;
              }
              frontier_page_ids.push_back(reservation.page_id);
              frontier_page_buffers.push_back(destination);
              frontier_cache_reservations.push_back(reservation);
            }
          } else {
            for (const uint64_t page_id : cell_batch_miss_pages) {
              if (std::find(frontier_page_ids.begin(), frontier_page_ids.end(), page_id) !=
                  frontier_page_ids.end()) {
                continue;
              }
              auto reservation = io_cache_->reserve(page_id, io_cache_access_t::DEMAND, false);
              char* destination = reservation.data;
              if (reservation.state == io_cache_reservation_state_t::HIT) {
                if (stats != nullptr) {
                  ++stats->n_cache_hits;
                }
              } else {
                if (reservation.state == io_cache_reservation_state_t::BYPASS) {
                  if (sector_scratch_index >= defaults::MAX_N_SECTOR_READS) {
                    throw diskann_exception_t("Cell batch exhausted query scratch pages", -1);
                  }
                  destination = sector_scratch + sector_scratch_index * defaults::SECTOR_LEN;
                  ++sector_scratch_index;
                }
                frontier_read_requests.emplace_back(page_id * defaults::SECTOR_LEN,
                                                    defaults::SECTOR_LEN, destination);
                if (stats != nullptr) {
                  ++stats->n_4k;
                  ++stats->n_ios;
                  stats->read_size += defaults::SECTOR_LEN;
                }
                ++num_ios;
              }
              frontier_page_ids.push_back(page_id);
              frontier_page_buffers.push_back(destination);
              frontier_cache_reservations.push_back(reservation);
            }
          }
        }
      }
      for (uint32_t node_id : frontier) {
        if constexpr (use_io_layout) {
          if (enable_cell_pq_traversal_ || community_pq_approach_round) {
            frontier_neighborhoods.emplace_back(node_id, nullptr);
            if (enable_cell_pq_traversal_ && stats != nullptr) {
              ++stats->cell_pq_traversal_expansions;
            }
            continue;
          }
          if (adaptive_cell_runtime_ != nullptr && adaptive_cell_runtime_->accepts_observations()) {
            const uint32_t cell = adaptive_cell_runtime_->logical_cell(node_id);
            if (cell != UINT32_MAX) {
              adaptive_loaded_cells.push_back(cell);
            }
          }
          const uint64_t page_id = active_vector_location(node_id).page_id;
          const auto existing =
              std::find(frontier_page_ids.begin(), frontier_page_ids.end(), page_id);
          char* destination = nullptr;
          if (existing == frontier_page_ids.end()) {
            io_cache_reservation_t reservation;
            if (io_cache_ != nullptr) {
              reservation = io_cache_->reserve(page_id, io_cache_access_t::DEMAND, false);
#if defined(__linux__)
              if (reservation.state == io_cache_reservation_state_t::BYPASS &&
                  reservation.coalesced &&
                  (enable_node_prefetch_ || enable_cell_prefetch_ || enable_gateway_prefetch_)) {
                const auto gateway_submission = gateway_prefetch_submissions.find(page_id);
                if (gateway_submission != gateway_prefetch_submissions.end()) {
                  if (stats != nullptr) {
                    ++stats->gateway_prefetch_late_hits;
                  }
                  gateway_prefetch_submissions.erase(gateway_submission);
                }
                harvest_prefetches(true);
                reservation = io_cache_->reserve(page_id, io_cache_access_t::DEMAND, false);
              }
#endif
            }
            if (reservation.state == io_cache_reservation_state_t::HIT) {
#if defined(__linux__)
              if (reservation.prefetched_hit) {
                const auto gateway_submission = gateway_prefetch_submissions.find(page_id);
                if (gateway_submission != gateway_prefetch_submissions.end()) {
                  if (stats != nullptr) {
                    ++stats->gateway_prefetch_timely_hits;
                    stats->gateway_prefetch_lead_us +=
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                  std::chrono::steady_clock::now() -
                                                  gateway_submission->second.submitted_at)
                                                  .count());
                    stats->gateway_prefetch_lead_hops +=
                        completed_base_hops - gateway_submission->second.submitted_hop;
                  }
                  gateway_prefetch_submissions.erase(gateway_submission);
                }
              }
#endif
              destination = reservation.data;
              if (stats != nullptr) {
                ++stats->n_cache_hits;
              }
            } else {
              destination = reservation.state == io_cache_reservation_state_t::LOAD
                                ? reservation.data
                                : sector_scratch + sector_scratch_index * defaults::SECTOR_LEN;
              if (reservation.state == io_cache_reservation_state_t::BYPASS) {
                ++sector_scratch_index;
              }
              frontier_read_requests.emplace_back(page_id * defaults::SECTOR_LEN,
                                                  defaults::SECTOR_LEN, destination);
              if (stats != nullptr) {
                ++stats->n_4k;
                ++stats->n_ios;
                stats->read_size += defaults::SECTOR_LEN;
              }
              ++num_ios;
            }
            frontier_page_ids.push_back(page_id);
            frontier_page_buffers.push_back(destination);
            frontier_cache_reservations.push_back(reservation);
          } else {
            const size_t request_index = static_cast<size_t>(existing - frontier_page_ids.begin());
            destination = frontier_page_buffers[request_index];
          }
          frontier_neighborhoods.emplace_back(node_id, destination);
        } else {
          char* destination =
              sector_scratch + num_sectors_per_node * sector_scratch_index * defaults::SECTOR_LEN;
          ++sector_scratch_index;
          frontier_neighborhoods.emplace_back(node_id, destination);
          frontier_read_requests.emplace_back(get_node_sector(node_id) * defaults::SECTOR_LEN,
                                              num_sectors_per_node * defaults::SECTOR_LEN,
                                              destination);
          if (stats != nullptr) {
            ++stats->n_4k;
            ++stats->n_ios;
            stats->read_size += num_sectors_per_node * defaults::SECTOR_LEN;
          }
          ++num_ios;
        }
      }
      if constexpr (use_io_layout) {
        const bool can_batch =
            enable_cell_page_batch_ && frontier_read_requests.size() > 1 &&
            std::none_of(frontier_cache_reservations.begin(), frontier_cache_reservations.end(),
                         [](const auto& item) {
                           return item.state == io_cache_reservation_state_t::BYPASS;
                         });
        if (can_batch) {
          static thread_local std::vector<uint64_t> requested_pages;
          requested_pages.clear();
          requested_pages.reserve(frontier_read_requests.size());
          for (const auto& request : frontier_read_requests) {
            requested_pages.push_back(request.offset / defaults::SECTOR_LEN);
          }
          const auto page_runs = make_io_page_runs(requested_pages);
          size_t scratch_page = 0;
          for (const auto& run : page_runs) {
            const auto find_request = [&](uint64_t page) {
              return std::find_if(frontier_read_requests.begin(), frontier_read_requests.end(),
                                  [page](const auto& request) {
                                    return request.offset / defaults::SECTOR_LEN == page;
                                  });
            };
            if (run.page_count == 1) {
              coalesced_read_requests.push_back(*find_request(run.first_page));
            } else {
              char* range_buffer = sector_scratch + scratch_page * defaults::SECTOR_LEN;
              coalesced_read_requests.emplace_back(run.first_page * defaults::SECTOR_LEN,
                                                   run.page_count * defaults::SECTOR_LEN,
                                                   range_buffer);
              for (uint32_t page = 0; page < run.page_count; ++page) {
                coalesced_page_copies.emplace_back(
                    range_buffer + page * defaults::SECTOR_LEN,
                    static_cast<char*>(find_request(run.first_page + page)->buf));
              }
              scratch_page += run.page_count;
              if (stats != nullptr) {
                ++stats->cell_page_batch_requests;
                stats->cell_page_batch_pages += run.page_count;
              }
            }
          }
          frontier_read_requests.swap(coalesced_read_requests);
        }
      }
      if (stats != nullptr) {
        stats->physical_read_requests += static_cast<uint32_t>(frontier_read_requests.size());
        if constexpr (use_io_layout) {
          if (enable_cell_page_batch_) {
            for (const auto& request : frontier_read_requests) {
              const auto pages = static_cast<uint32_t>(request.len / defaults::SECTOR_LEN);
              ++stats->random_ios;
              stats->sequential_ios += pages - 1U;
            }
          } else {
            for (size_t request = 0; request < frontier_read_requests.size(); ++request) {
              const bool sequential =
                  request != 0 && frontier_read_requests[request].offset ==
                                      frontier_read_requests[request - 1U].offset +
                                          frontier_read_requests[request - 1U].len;
              if (sequential) {
                ++stats->sequential_ios;
              } else {
                ++stats->random_ios;
              }
            }
          }
        } else {
          stats->random_ios += static_cast<uint32_t>(frontier_read_requests.size());
        }
        if constexpr (!use_io_layout) {
          for (size_t request = 1; request < frontier_read_requests.size(); ++request) {
            if (frontier_read_requests[request].offset ==
                frontier_read_requests[request - 1U].offset +
                    frontier_read_requests[request - 1U].len) {
              --stats->random_ios;
              ++stats->sequential_ios;
            }
          }
        }
        if (!enable_cell_pq_traversal_ && !community_pq_approach_round) {
          stats->demand_useful_bytes += static_cast<uint64_t>(frontier.size()) *
                                        (use_io_layout ? disk_bytes_per_point_ : max_node_len_);
        }
      }
#if defined(__linux__)
      if (enable_cell_prefetch_) {
        while (!cell_prediction_pages.empty() &&
               std::find(frontier_page_ids.begin(), frontier_page_ids.end(),
                         cell_prediction_pages.front()) != frontier_page_ids.end()) {
          cell_prediction_pages.erase(cell_prediction_pages.begin());
        }
        if (!cell_prediction_pages.empty()) {
          submit_prefetch(cell_prediction_pages.front());
          cell_prediction_pages.erase(cell_prediction_pages.begin());
        }
      }
#endif
      if (!frontier_read_requests.empty()) {
        io_timer.reset();
        try {
          reader_->read(frontier_read_requests, io_context);
          for (const auto& [source, destination] : coalesced_page_copies) {
            std::memcpy(destination, source, defaults::SECTOR_LEN);
          }
          if (io_cache_ != nullptr) {
            io_cache_->publish_many(frontier_cache_reservations, true);
          }
        } catch (...) {
          if (io_cache_ != nullptr) {
            io_cache_->publish_many(frontier_cache_reservations, false);
          }
          throw;
        }
        if (stats != nullptr) {
          stats->io_us += static_cast<float>(io_timer.elapsed());
        }
      }

      if constexpr (use_hybrid && use_io_layout) {
        if (enable_cell_batch_search_ && !cell_batch_exact_nodes.empty()) {
          const bool stitch_cell_batch =
              cell_batch_graph_stitch_min_l_ != 0 &&
              search_list_size >= cell_batch_graph_stitch_min_l_;
          static thread_local std::vector<float> cell_batch_graph_distances;
          if (stitch_cell_batch) {
            cell_batch_graph_distances.resize(cell_batch_exact_nodes.size());
            compute_distances(cell_batch_exact_nodes.data(), cell_batch_exact_nodes.size(),
                              cell_batch_graph_distances.data());
          }
          for (size_t node_rank = 0; node_rank < cell_batch_exact_nodes.size(); ++node_rank) {
            const uint32_t node_id = cell_batch_exact_nodes[node_rank];
            const auto location = active_vector_location(node_id);
            const auto page =
                std::find(frontier_page_ids.begin(), frontier_page_ids.end(), location.page_id);
            if (page == frontier_page_ids.end()) {
              throw diskann_exception_t("Cell batch lost a reserved vector page", -1);
            }
            const size_t page_index = static_cast<size_t>(page - frontier_page_ids.begin());
            const auto* coordinates = reinterpret_cast<const data_t*>(
                frontier_page_buffers[page_index] + location.page_offset);
            float distance = 0.0F;
            if (cell_u8_layout) {
              decoded_u8_l2_lookup(reinterpret_cast<const uint8_t*>(coordinates), 1, data_dim_,
                                   decoded_u8_query.data(), &distance);
            } else {
              distance = distance_comparator_->compare(aligned_query, coordinates,
                                                       static_cast<uint32_t>(data_dim_));
            }
            cell_batch_exact_results.emplace_back(node_id, distance);
            if (!cell_batch_preserve_graph_ || stitch_cell_batch) {
              const bool newly_visited = insert_visited(node_id);
              const float frontier_distance = stitch_cell_batch
                                                  ? cell_batch_graph_distances[node_rank]
                                                  : distance;
              candidate_queue.insert_or_update(neighbor_t(node_id, frontier_distance));
              if (newly_visited) {
                hybrid_state->mark_macro_inserted(node_id,
                                                  community_polar_candidate_origin_t::POLAR_CELL);
              }
            }
          }
          if (stats != nullptr) {
            stats->n_cmps += static_cast<uint32_t>(cell_batch_exact_nodes.size());
            stats->cell_nodes_scored += static_cast<uint32_t>(cell_batch_exact_nodes.size());
            stats->cell_pq_refined_candidates +=
                static_cast<uint32_t>(cell_batch_exact_nodes.size());
            stats->demand_useful_bytes +=
                static_cast<uint64_t>(cell_batch_exact_nodes.size()) * disk_bytes_per_point_;
          }
          cell_batch_exact_pending = false;
          cell_batch_only_round = false;
          interleaved_last_cell_hop = completed_base_hops;
        }
      }
#if defined(__linux__)
      if (enable_node_prefetch_ || enable_cell_prefetch_ || enable_gateway_prefetch_) {
        harvest_prefetches(false);
      }
#endif
    }

    for (const auto& cached_neighborhood : cached_neighborhoods) {
      record_base_expansion(cached_neighborhood.first);
      const auto coords = coordinate_cache_.find(cached_neighborhood.first);
      if (coords == coordinate_cache_.end()) {
        throw diskann_exception_t("Neighborhood cache has no matching coordinate cache entry", -1);
      }
      float expanded_distance = 0.0F;
      if (use_disk_index_pq_) {
        expanded_distance =
            disk_pq_table_.l2_distance(query_float, reinterpret_cast<uint8_t*>(coords->second));
      } else {
        expanded_distance = distance_comparator_->compare(aligned_query, coords->second,
                                                          static_cast<uint32_t>(aligned_dim_));
      }
      full_result.emplace_back(cached_neighborhood.first, expanded_distance);
      if constexpr (use_hybrid) {
        if (enable_hybrid_graph_hooks) {
          hybrid_search->observe_source_cell(*hybrid_state, cached_neighborhood.first,
                                             expanded_distance, stats);
        }
      }

      uint32_t num_neighbors = cached_neighborhood.second.first;
      if (graph_neighbor_score_limit_ != 0) {
        num_neighbors = std::min(num_neighbors, graph_neighbor_score_limit_);
      }
      uint32_t* neighbors = cached_neighborhood.second.second;
      if (enable_cell_pq_filter_visited_) {
        unvisited_neighbor_ids.clear();
        for (uint32_t offset = 0; offset < num_neighbors; ++offset) {
          if (insert_visited(neighbors[offset])) {
            unvisited_neighbor_ids.push_back(neighbors[offset]);
          } else {
            mark_base_reached(neighbors[offset]);
          }
        }
        if constexpr (track_beam_phase) {
          cpu_timer.reset();
        }
        compute_distances(unvisited_neighbor_ids.data(), unvisited_neighbor_ids.size(),
                          distance_scratch);
        if (stats != nullptr) {
          stats->n_cmps += static_cast<uint32_t>(unvisited_neighbor_ids.size());
          stats->n_base_neighbors_scanned += num_neighbors;
          if constexpr (track_beam_phase) {
            stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
          }
        }
        record_base_candidates(cached_neighborhood.first, unvisited_neighbor_ids.data(),
                               static_cast<uint32_t>(unvisited_neighbor_ids.size()),
                               distance_scratch);
        for (size_t offset = 0; offset < unvisited_neighbor_ids.size(); ++offset) {
          candidate_queue.insert(
              neighbor_t(unvisited_neighbor_ids[offset], distance_scratch[offset]));
        }
      } else {
        if constexpr (track_beam_phase) {
          cpu_timer.reset();
        }
        compute_distances(neighbors, num_neighbors, distance_scratch);
        if (stats != nullptr) {
          stats->n_cmps += num_neighbors;
          stats->n_base_neighbors_scanned += num_neighbors;
          if constexpr (track_beam_phase) {
            stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
          }
        }
        record_base_candidates(cached_neighborhood.first, neighbors, num_neighbors,
                               distance_scratch);
        for (uint32_t i = 0; i < num_neighbors; ++i) {
          if (insert_visited(neighbors[i])) {
            candidate_queue.insert(neighbor_t(neighbors[i], distance_scratch[i]));
          } else {
            mark_base_reached(neighbors[i]);
          }
        }
      }
    }

    for (size_t frontier_position = 0; frontier_position < frontier_neighborhoods.size();
         ++frontier_position) {
      const auto& frontier_neighborhood = frontier_neighborhoods[frontier_position];
      record_base_expansion(frontier_neighborhood.first);
      const data_t* coordinates = nullptr;
      const uint32_t* neighbors = nullptr;
      uint32_t num_neighbors = 0;
      if constexpr (use_io_layout) {
        if (!enable_cell_pq_traversal_ && !community_pq_approach_round) {
          const auto location = active_vector_location(frontier_neighborhood.first);
          coordinates =
              reinterpret_cast<const data_t*>(frontier_neighborhood.second + location.page_offset);
        }
        const auto io_neighbors = io_index_->neighbors(frontier_neighborhood.first);
        neighbors = io_neighbors.data();
        num_neighbors = static_cast<uint32_t>(io_neighbors.size());
      } else {
        char* node_buffer =
            offset_to_node(frontier_neighborhood.second, frontier_neighborhood.first);
        uint32_t* neighborhood = offset_to_node_neighborhood(node_buffer);
        num_neighbors = *neighborhood;
        neighbors = neighborhood + 1;
        coordinates = offset_to_node_coords(node_buffer);
      }
      if (num_neighbors > max_degree_) {
        throw diskann_exception_t("Disk node degree exceeds index metadata during search", -1);
      }
      if (graph_neighbor_score_limit_ != 0) {
        num_neighbors = std::min(num_neighbors, graph_neighbor_score_limit_);
      }

      float expanded_distance = 0.0F;
      if (enable_cell_pq_traversal_ || community_pq_approach_round) {
        expanded_distance = frontier_candidate_distances[frontier_position];
      } else {
        std::memcpy(data_buffer, coordinates, disk_bytes_per_point_);
        if (use_disk_index_pq_) {
          expanded_distance =
              disk_pq_table_.l2_distance(query_float, reinterpret_cast<uint8_t*>(data_buffer));
        } else {
          expanded_distance = distance_comparator_->compare(aligned_query, data_buffer,
                                                            static_cast<uint32_t>(aligned_dim_));
        }
      }
      full_result.emplace_back(frontier_neighborhood.first, expanded_distance);
      if constexpr (use_hybrid) {
        if (enable_hybrid_graph_hooks) {
          hybrid_search->observe_source_cell(*hybrid_state, frontier_neighborhood.first,
                                             expanded_distance, stats);
        }
      }

      if (enable_cell_pq_filter_visited_) {
        unvisited_neighbor_ids.clear();
        for (uint32_t offset = 0; offset < num_neighbors; ++offset) {
          if (insert_visited(neighbors[offset])) {
            unvisited_neighbor_ids.push_back(neighbors[offset]);
          } else {
            mark_base_reached(neighbors[offset]);
          }
        }
        if constexpr (track_beam_phase) {
          cpu_timer.reset();
        }
        compute_distances(unvisited_neighbor_ids.data(), unvisited_neighbor_ids.size(),
                          distance_scratch);
        if (stats != nullptr) {
          stats->n_cmps += static_cast<uint32_t>(unvisited_neighbor_ids.size());
          stats->n_base_neighbors_scanned += num_neighbors;
          if constexpr (track_beam_phase) {
            stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
          }
        }
        record_base_candidates(frontier_neighborhood.first, unvisited_neighbor_ids.data(),
                               static_cast<uint32_t>(unvisited_neighbor_ids.size()),
                               distance_scratch);
        for (size_t offset = 0; offset < unvisited_neighbor_ids.size(); ++offset) {
          candidate_queue.insert(
              neighbor_t(unvisited_neighbor_ids[offset], distance_scratch[offset]));
        }
      } else {
        if constexpr (track_beam_phase) {
          cpu_timer.reset();
        }
        compute_distances(neighbors, num_neighbors, distance_scratch);
        if (stats != nullptr) {
          stats->n_cmps += num_neighbors;
          stats->n_base_neighbors_scanned += num_neighbors;
          if constexpr (track_beam_phase) {
            stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
          }
        }
        record_base_candidates(frontier_neighborhood.first, neighbors, num_neighbors,
                               distance_scratch);
        if constexpr (track_beam_phase) {
          cpu_timer.reset();
        }
        for (uint32_t i = 0; i < num_neighbors; ++i) {
          if (insert_visited(neighbors[i])) {
            if (stats != nullptr) {
              ++stats->n_cmps;
            }
            candidate_queue.insert(neighbor_t(neighbors[i], distance_scratch[i]));
          } else {
            mark_base_reached(neighbors[i]);
          }
        }
        if (stats != nullptr) {
          if constexpr (track_beam_phase) {
            stats->cpu_us += static_cast<float>(cpu_timer.elapsed());
          }
        }
      }
    }
    if (io_cache_ != nullptr) {
      io_cache_->release_many(frontier_cache_reservations);
    }
    if (enable_cell_pq_traversal_ && cell_pq_refine_prefetch_hop_ != 0 &&
        !refine_prefetch_attempted && completed_base_hops >= cell_pq_refine_prefetch_hop_) {
      refine_prefetch_attempted = true;
#if defined(__linux__)
      if (libaio_reader != nullptr) {
        const size_t predicted_count =
            std::min<size_t>(candidate_queue.size(), refine_candidate_budget);
        refine_candidate_pages.clear();
        refine_candidate_pages.reserve(predicted_count);
        for (size_t position = 0; position < predicted_count; ++position) {
          refine_candidate_pages.push_back(
              active_vector_location(candidate_queue[position].id).page_id);
        }
        reserve_refine_pages(refine_candidate_pages);
        if (!refine_read_requests.empty()) {
          libaio_reader->submit_async(refine_read_requests, io_context, refine_async_batch);
          refine_prefetch_submitted = true;
          if (stats != nullptr) {
            stats->physical_read_requests += static_cast<uint32_t>(refine_read_requests.size());
          }
        }
      }
#endif
    }
    if (enable_dynamic_width_ && !frontier_candidate_distances.empty()) {
      const float radius = candidate_queue.size() == 0
                               ? std::numeric_limits<float>::infinity()
                               : candidate_queue[candidate_queue.size() - 1U].distance;
      const uint32_t useful = static_cast<uint32_t>(
          std::count_if(frontier_candidate_distances.begin(), frontier_candidate_distances.end(),
                        [radius](float distance) { return distance <= radius; }));
      const uint32_t wasted = static_cast<uint32_t>(frontier_candidate_distances.size()) - useful;
      if (stats != nullptr) {
        ++stats->dynamic_width_rounds;
        stats->dynamic_width_sum += static_cast<uint32_t>(current_beam_width);
        stats->dynamic_width_max =
            std::max(stats->dynamic_width_max, static_cast<uint32_t>(current_beam_width));
        stats->dynamic_width_useful_ios += useful;
        stats->dynamic_width_wasted_ios += wasted;
      }
      if (completed_base_hops >= dynamic_width_marker_ && current_beam_width < beam_width) {
        const float waste_ratio =
            static_cast<float>(wasted) / static_cast<float>(frontier_candidate_distances.size());
        if (waste_ratio <= dynamic_width_waste_threshold_) {
          ++current_beam_width;
        }
      }
    }
    if constexpr (track_beam_phase) {
      size_t marker = 0;
      while (marker < candidate_queue.size() && candidate_queue[marker].expanded) {
        ++marker;
      }
      beam_phase_max_marker = std::max(beam_phase_max_marker, marker);
      if (stats->beam_phase_transition_found == 0 &&
          beam_phase_max_marker >= k_pipeann_convergence_marker &&
          candidate_queue.has_unexpanded_node() && !phase_io_candidates.empty()) {
        const float pool_radius = candidate_queue[candidate_queue.size() - 1].distance;
        for (const auto& candidate : phase_io_candidates) {
          beam_phase_useful_ios += candidate.distance <= pool_radius ? 1U : 0U;
          ++beam_phase_classified_ios;
        }
        const double waste_ratio =
            static_cast<double>(beam_phase_classified_ios - beam_phase_useful_ios) /
            static_cast<double>(beam_phase_classified_ios);
        if (waste_ratio <= k_pipeann_waste_threshold) {
          stats->beam_phase_transition_found = 1;
          stats->beam_phase_entry_io_hop = stats->base_hops;
          stats->beam_phase_entry_expansion_rank = static_cast<uint32_t>(
              std::min<size_t>(beam_phase_max_marker, std::numeric_limits<uint32_t>::max()));
          stats->beam_phase_approach_base_hops = stats->base_hops;
          stats->beam_phase_approach_ios = stats->n_ios;
          stats->beam_phase_approach_cmps = stats->n_cmps;
          stats->beam_phase_approach_community_expansions = stats->community_meta_expansions;
          stats->beam_phase_approach_gateway_nodes_scored = stats->gateway_nodes_scored;
          stats->beam_phase_approach_cell_blocks = stats->cell_blocks_expanded;
          stats->beam_phase_approach_cell_nodes_scored = stats->cell_nodes_scored;
          stats->beam_phase_approach_us =
              static_cast<float>(query_timer.elapsed() - beam_phase_loop_start_us);
        }
      }
    }
  }

#if defined(__linux__)
  if (enable_node_prefetch_ || enable_cell_prefetch_ || enable_gateway_prefetch_) {
    harvest_prefetches(true);
  }
  if (enable_gateway_prefetch_ && stats != nullptr) {
    stats->gateway_prefetch_unused += static_cast<uint32_t>(gateway_prefetch_submissions.size());
  }
#endif

  if constexpr (use_io_layout) {
    if (adaptive_cell_runtime_ != nullptr && adaptive_cell_runtime_->accepts_observations() &&
        !adaptive_loaded_cells.empty()) {
      std::sort(adaptive_loaded_cells.begin(), adaptive_loaded_cells.end());
      adaptive_loaded_cells.erase(
          std::unique(adaptive_loaded_cells.begin(), adaptive_loaded_cells.end()),
          adaptive_loaded_cells.end());
      adaptive_cell_runtime_->observe_query(adaptive_loaded_cells);
    }
  }

  if constexpr (use_hybrid && !insert_macro) {
    hybrid_search->finish_cell_oracle(*hybrid_state, stats);
  }

  if constexpr (track_beam_phase) {
    const auto loop_elapsed_us =
        static_cast<float>(query_timer.elapsed() - beam_phase_loop_start_us);
    stats->beam_phase_search_loop_us = loop_elapsed_us;
    if (stats->beam_phase_transition_found != 0) {
      stats->beam_phase_convergence_base_hops =
          stats->base_hops - stats->beam_phase_approach_base_hops;
      stats->beam_phase_convergence_ios = stats->n_ios - stats->beam_phase_approach_ios;
      stats->beam_phase_convergence_cmps = stats->n_cmps - stats->beam_phase_approach_cmps;
      stats->beam_phase_convergence_community_expansions =
          stats->community_meta_expansions - stats->beam_phase_approach_community_expansions;
      stats->beam_phase_convergence_gateway_nodes_scored =
          stats->gateway_nodes_scored - stats->beam_phase_approach_gateway_nodes_scored;
      stats->beam_phase_convergence_cell_blocks =
          stats->cell_blocks_expanded - stats->beam_phase_approach_cell_blocks;
      stats->beam_phase_convergence_cell_nodes_scored =
          stats->cell_nodes_scored - stats->beam_phase_approach_cell_nodes_scored;
      stats->beam_phase_convergence_us = loop_elapsed_us - stats->beam_phase_approach_us;
    } else {
      stats->beam_phase_approach_base_hops = stats->base_hops;
      stats->beam_phase_approach_ios = stats->n_ios;
      stats->beam_phase_approach_cmps = stats->n_cmps;
      stats->beam_phase_approach_community_expansions = stats->community_meta_expansions;
      stats->beam_phase_approach_gateway_nodes_scored = stats->gateway_nodes_scored;
      stats->beam_phase_approach_cell_blocks = stats->cell_blocks_expanded;
      stats->beam_phase_approach_cell_nodes_scored = stats->cell_nodes_scored;
      stats->beam_phase_approach_us = loop_elapsed_us;
    }
  }

  if constexpr (use_hybrid && insert_macro) {
    if (terminal_convergence_complete && !enable_cell_batch_search_) {
      if (!decoded_cell_pq_codes_.empty()) {
        const size_t base_result_count = full_result.size();
        for (const auto& candidate : terminal_cell_results) {
          const bool already_present =
              std::any_of(full_result.begin(), full_result.begin() + base_result_count,
                          [&](const auto& result) { return result.id == candidate.id; });
          if (!already_present) {
            full_result.push_back(candidate);
          }
        }
      } else {
        std::unordered_set<uint32_t> result_ids;
        result_ids.reserve(full_result.size() + terminal_cell_results.size());
        for (const auto& result : full_result) {
          result_ids.insert(result.id);
        }
        for (const auto& candidate : terminal_cell_results) {
          if (result_ids.insert(candidate.id).second) {
            full_result.push_back(candidate);
          }
        }
      }
    }
  }

  const bool exact_cell_results_only = forced_terminal_tree && enable_cell_batch_search_ &&
                                       enable_cell_pq_traversal_ && refine_candidate_budget == 0 &&
                                       !enable_cell_leaf_refinement_ && !use_reorder_data &&
                                       !enable_ranked_cell_page_refinement_;
  const bool defer_exact_cell_merge = enable_cell_batch_search_ && enable_cell_pq_traversal_ &&
                                      refine_candidate_budget != 0 &&
                                      !enable_cell_leaf_refinement_ && !use_reorder_data;
  const auto merge_exact_cell_results = [&]() {
    for (const auto& exact_candidate : cell_batch_exact_results) {
      const auto existing =
          std::find_if(full_result.begin(), full_result.end(), [&](const auto& candidate) {
            return candidate.id == exact_candidate.id;
          });
      if (existing == full_result.end()) {
        full_result.push_back(exact_candidate);
      } else {
        *existing = exact_candidate;
      }
    }
  };
  if (exact_cell_results_only) {
    // Topology/PQ graph candidates are routing evidence, not exact-distance results. When no
    // later raw-vector refinement is configured, expose only vectors read from admitted Cells.
    full_result = cell_batch_exact_results;
  } else if (!defer_exact_cell_merge && !cell_batch_exact_results.empty()) {
    std::sort(cell_batch_exact_results.begin(), cell_batch_exact_results.end(),
              [](const auto& left, const auto& right) { return left.id < right.id; });
    cell_batch_exact_results.erase(
        std::unique(cell_batch_exact_results.begin(), cell_batch_exact_results.end(),
                    [](const auto& left, const auto& right) { return left.id == right.id; }),
        cell_batch_exact_results.end());
    for (const auto& exact_candidate : cell_batch_exact_results) {
      const auto existing =
          std::find_if(full_result.begin(), full_result.end(), [&](const auto& candidate) {
            return candidate.id == exact_candidate.id;
          });
      if (existing == full_result.end()) {
        full_result.push_back(exact_candidate);
      } else {
        *existing = exact_candidate;
      }
    }
  }

  const bool terminal_top_k_only =
      refine_candidate_budget == 0 && !enable_cell_leaf_refinement_ && !use_reorder_data &&
      (!decoded_cell_pq_codes_.empty() || terminal_convergence_complete);
  if (terminal_top_k_only) {
    const size_t result_count = std::min<size_t>(top_k, full_result.size());
    std::partial_sort(full_result.begin(), full_result.begin() + result_count, full_result.end());
    full_result.resize(result_count);
  } else {
    std::sort(full_result.begin(), full_result.end());
  }
  if (enable_cell_pq_traversal_ &&
      (refine_candidate_budget != 0 || enable_cell_leaf_refinement_)) {
    if (!enable_cell_leaf_refinement_ && refine_candidate_budget < top_k) {
      throw diskann_exception_t("Cell-PQ refinement candidate count must cover top-k", -1);
    }
    static thread_local std::vector<neighbor_t> leaf_refine_results;
    auto* refine_results = &full_result;
    size_t refine_count = std::min<size_t>(full_result.size(), refine_candidate_budget);
    if (enable_cell_leaf_refinement_) {
      leaf_refine_results = terminal_cell_results;
      std::sort(leaf_refine_results.begin(), leaf_refine_results.end(),
                [](const auto& left, const auto& right) { return left.id < right.id; });
      leaf_refine_results.erase(
          std::unique(leaf_refine_results.begin(), leaf_refine_results.end(),
                      [](const auto& left, const auto& right) { return left.id == right.id; }),
          leaf_refine_results.end());
      static thread_local std::unordered_set<uint32_t> leaf_refine_ids;
      leaf_refine_ids.clear();
      leaf_refine_ids.reserve(leaf_refine_results.size() + refine_candidate_budget);
      for (const auto& candidate : leaf_refine_results) {
        leaf_refine_ids.insert(static_cast<uint32_t>(candidate.id));
      }
      const auto& terminal_config = hybrid_search->config();
      if (terminal_config.cell_terminal_stitch_promote_cells &&
          refine_candidate_budget != 0) {
        struct stitch_cell_t {
          uint32_t cell = 0;
          uint32_t support = 0;
          float best_distance = std::numeric_limits<float>::infinity();
        };
        static thread_local std::vector<stitch_cell_t> stitch_cells;
        static thread_local std::vector<neighbor_t> stitch_candidates;
        static thread_local std::unordered_set<uint32_t> promoted_stitch_cells;
        stitch_cells.clear();
        stitch_candidates.clear();
        promoted_stitch_cells.clear();
        uint32_t candidates_considered = 0;
        for (const auto& candidate : full_result) {
          if (candidates_considered >= refine_candidate_budget) {
            break;
          }
          const uint32_t node = static_cast<uint32_t>(candidate.id);
          if (leaf_refine_ids.contains(node)) {
            continue;
          }
          ++candidates_considered;
          stitch_candidates.push_back(candidate);
          const uint32_t cell = hybrid_search->cell_id(node);
          if (cell == UINT32_MAX) {
            continue;
          }
          const auto position =
              std::find_if(stitch_cells.begin(), stitch_cells.end(),
                           [cell](const auto& value) { return value.cell == cell; });
          if (position == stitch_cells.end()) {
            stitch_cells.push_back({cell, 1, candidate.distance});
          } else {
            ++position->support;
            position->best_distance = std::min(position->best_distance, candidate.distance);
          }
        }
        std::sort(stitch_cells.begin(), stitch_cells.end(), [](const auto& left, const auto& right) {
          return std::make_tuple(std::numeric_limits<uint32_t>::max() - left.support,
                                 left.best_distance, left.cell) <
                 std::make_tuple(std::numeric_limits<uint32_t>::max() - right.support,
                                 right.best_distance, right.cell);
        });
        if (stats != nullptr) {
          stats->cell_stitch_candidates_considered += candidates_considered;
          stats->cell_stitch_owner_cells += static_cast<uint32_t>(stitch_cells.size());
          for (const auto& value : stitch_cells) {
            stats->cell_stitch_max_owner_support =
                std::max(stats->cell_stitch_max_owner_support, value.support);
            if (value.support >= 2) {
              stats->cell_stitch_multi_owner_candidates += value.support;
            }
          }
        }
        stitch_cells.erase(
            std::remove_if(stitch_cells.begin(), stitch_cells.end(), [&](const auto& value) {
              return value.support < terminal_config.cell_terminal_stitch_min_support;
            }),
            stitch_cells.end());
        stitch_cells.resize(std::min<size_t>(
            stitch_cells.size(), terminal_config.cell_terminal_stitch_cell_budget));
        for (const auto& value : stitch_cells) {
          promoted_stitch_cells.insert(value.cell);
          for (const uint32_t node : hybrid_search->cell_node_ids(value.cell)) {
            if (leaf_refine_ids.insert(node).second) {
              leaf_refine_results.emplace_back(node, std::numeric_limits<float>::infinity());
            }
          }
        }
        for (const auto& candidate : stitch_candidates) {
          const uint32_t node = static_cast<uint32_t>(candidate.id);
          if (!promoted_stitch_cells.contains(hybrid_search->cell_id(node)) &&
              leaf_refine_ids.insert(node).second) {
            leaf_refine_results.push_back(candidate);
          }
        }
      } else {
        uint32_t stitched = 0;
        for (const auto& candidate : full_result) {
          if (stitched >= refine_candidate_budget) {
            break;
          }
          if (leaf_refine_ids.insert(static_cast<uint32_t>(candidate.id)).second) {
            leaf_refine_results.push_back(candidate);
            ++stitched;
          }
        }
      }
      std::sort(leaf_refine_results.begin(), leaf_refine_results.end(),
                [](const auto& left, const auto& right) { return left.id < right.id; });
      refine_results = &leaf_refine_results;
      refine_count = refine_results->size();
    }
    if (!enable_cell_leaf_refinement_ && refine_count > defaults::MAX_N_SECTOR_READS) {
      throw diskann_exception_t("Cell-PQ refinement exceeds query scratch capacity", -1);
    }
    refine_results->resize(refine_count);

    if (!resident_u8_refinement_.empty()) {
      static thread_local std::vector<uint8_t> query_u8;
      static thread_local std::vector<uint32_t> refine_ids;
      query_u8.resize(data_dim_);
      refine_ids.resize(refine_count);
      for (size_t dimension = 0; dimension < data_dim_; ++dimension) {
        const float value = query_float[dimension];
        if (!std::isfinite(value) || value < 0.0F || value > 255.0F ||
            std::nearbyint(value) != value) {
          throw diskann_exception_t(
              "Resident uint8 refinement requires integral queries in [0, 255]", -1);
        }
        query_u8[dimension] = static_cast<uint8_t>(value);
      }
      for (size_t position = 0; position < refine_count; ++position) {
        refine_ids[position] = static_cast<uint32_t>((*refine_results)[position].id);
      }
      original_u8_l2_lookup(resident_u8_refinement_.data(), refine_ids.data(), refine_count,
                            data_dim_, query_u8.data(), distance_scratch);
      for (size_t position = 0; position < refine_count; ++position) {
        (*refine_results)[position].distance = distance_scratch[position];
      }
      if (stats != nullptr) {
        stats->n_cmps += static_cast<uint32_t>(refine_count);
        stats->cell_pq_refined_candidates += static_cast<uint32_t>(refine_count);
        stats->demand_useful_bytes += refine_count * data_dim_;
      }
      std::sort(refine_results->begin(), refine_results->end());
      if (enable_cell_leaf_refinement_) {
        for (const auto& exact_candidate : *refine_results) {
          const auto existing =
              std::find_if(full_result.begin(), full_result.end(), [&](const auto& candidate) {
                return candidate.id == exact_candidate.id;
              });
          if (existing == full_result.end()) {
            full_result.push_back(exact_candidate);
          } else {
            *existing = exact_candidate;
          }
        }
        std::sort(full_result.begin(), full_result.end());
      }
    } else {

#if defined(__linux__)
      if (refine_prefetch_submitted) {
        io_timer.reset();
        try {
          libaio_reader->complete_async(io_context, refine_async_batch);
          io_cache_->publish_many(refine_reservations, true);
        } catch (...) {
          io_cache_->publish_many(refine_reservations, false);
          throw;
        }
        if (stats != nullptr) {
          stats->io_us += static_cast<float>(io_timer.elapsed());
        }
        refine_prefetch_submitted = false;
      }
#endif
      refine_read_requests.clear();
      const size_t existing_reservations = refine_reservations.size();
      refine_candidate_pages.clear();
      refine_candidate_pages.reserve(refine_count);
      for (const auto& candidate : *refine_results) {
        const auto location = active_vector_location(static_cast<uint32_t>(candidate.id));
        refine_candidate_pages.push_back(location.page_id);
      }
      reserve_refine_pages(refine_candidate_pages);
      if (refine_page_ids.size() > defaults::MAX_N_SECTOR_READS) {
        throw diskann_exception_t("Cell refinement exceeds query page capacity", -1);
      }
      if (!refine_read_requests.empty()) {
        const bool can_batch =
            enable_cell_page_batch_ && refine_read_requests.size() > 1 &&
            std::none_of(refine_reservations.begin() +
                             static_cast<std::ptrdiff_t>(existing_reservations),
                         refine_reservations.end(), [](const auto& reservation) {
                           return reservation.state == io_cache_reservation_state_t::BYPASS;
                         });
        if (can_batch) {
          refine_coalesced_read_requests.clear();
          refine_coalesced_page_copies.clear();
          static thread_local std::vector<uint64_t> requested_pages;
          requested_pages.clear();
          requested_pages.reserve(refine_read_requests.size());
          for (const auto& request : refine_read_requests) {
            requested_pages.push_back(request.offset / defaults::SECTOR_LEN);
          }
          const auto page_runs = make_io_page_runs(requested_pages);
          size_t scratch_page = 0;
          for (const auto& run : page_runs) {
            const auto find_request = [&](uint64_t page) {
              return std::find_if(refine_read_requests.begin(), refine_read_requests.end(),
                                  [page](const auto& request) {
                                    return request.offset / defaults::SECTOR_LEN == page;
                                  });
            };
            if (run.page_count == 1) {
              refine_coalesced_read_requests.push_back(*find_request(run.first_page));
              continue;
            }
            if (scratch_page + run.page_count > defaults::MAX_N_SECTOR_READS) {
              throw diskann_exception_t("Cell refinement batch exceeds query scratch capacity",
                                        -1);
            }
            char* range_buffer = sector_scratch + scratch_page * defaults::SECTOR_LEN;
            refine_coalesced_read_requests.emplace_back(run.first_page * defaults::SECTOR_LEN,
                                                        run.page_count * defaults::SECTOR_LEN,
                                                        range_buffer);
            for (uint32_t page = 0; page < run.page_count; ++page) {
              refine_coalesced_page_copies.emplace_back(
                  range_buffer + page * defaults::SECTOR_LEN,
                  static_cast<char*>(find_request(run.first_page + page)->buf));
            }
            scratch_page += run.page_count;
            if (stats != nullptr) {
              ++stats->cell_page_batch_requests;
              stats->cell_page_batch_pages += run.page_count;
              stats->random_ios -= run.page_count - 1U;
              stats->sequential_ios += run.page_count - 1U;
            }
          }
          refine_read_requests.swap(refine_coalesced_read_requests);
        }
        io_timer.reset();
        try {
          reader_->read(refine_read_requests, io_context);
          for (const auto& [source, destination] : refine_coalesced_page_copies) {
            std::memcpy(destination, source, defaults::SECTOR_LEN);
          }
          if (io_cache_ != nullptr) {
            io_cache_->publish_many(std::span<const io_cache_reservation_t>(refine_reservations)
                                        .subspan(existing_reservations),
                                    true);
          }
        } catch (...) {
          if (io_cache_ != nullptr) {
            io_cache_->publish_many(std::span<const io_cache_reservation_t>(refine_reservations)
                                        .subspan(existing_reservations),
                                    false);
          }
          throw;
        }
        if (stats != nullptr) {
          stats->io_us += static_cast<float>(io_timer.elapsed());
          stats->physical_read_requests += static_cast<uint32_t>(refine_read_requests.size());
        }
      }
      for (auto& candidate : *refine_results) {
        const auto location = active_vector_location(static_cast<uint32_t>(candidate.id));
        const auto page =
            std::find(refine_page_ids.begin(), refine_page_ids.end(), location.page_id);
        if (page == refine_page_ids.end()) {
          throw diskann_exception_t("Cell-PQ refinement lost a reserved vector page", -1);
        }
        const size_t page_index = static_cast<size_t>(page - refine_page_ids.begin());
        const auto* coordinates = refine_page_buffers[page_index] + location.page_offset;
        if (cell_u8_layout) {
          decoded_u8_l2_lookup(reinterpret_cast<const uint8_t*>(coordinates), 1, data_dim_,
                               decoded_u8_query.data(), &candidate.distance);
        } else {
          candidate.distance = distance_comparator_->compare(
              aligned_query, reinterpret_cast<const data_t*>(coordinates),
              static_cast<uint32_t>(data_dim_));
        }
      }
      if (io_cache_ != nullptr) {
        io_cache_->release_many(refine_reservations);
      }
      if (stats != nullptr) {
        stats->n_cmps += static_cast<uint32_t>(refine_count);
        stats->cell_pq_refined_candidates += static_cast<uint32_t>(refine_count);
        const uint64_t stored_vector_bytes = cell_u8_layout ? data_dim_ : disk_bytes_per_point_;
        stats->demand_useful_bytes += refine_count * stored_vector_bytes;
      }
      std::sort(refine_results->begin(), refine_results->end());
      if (enable_cell_leaf_refinement_) {
        for (const auto& exact_candidate : *refine_results) {
          const auto existing =
              std::find_if(full_result.begin(), full_result.end(), [&](const auto& candidate) {
                return candidate.id == exact_candidate.id;
              });
          if (existing == full_result.end()) {
            full_result.push_back(exact_candidate);
          } else {
            *existing = exact_candidate;
          }
        }
        std::sort(full_result.begin(), full_result.end());
      }
    }
  }
  if (defer_exact_cell_merge) {
    merge_exact_cell_results();
    std::sort(full_result.begin(), full_result.end());
  }
  if (use_reorder_data) {
    if (!reorder_data_exists_) {
      throw diskann_exception_t("Requested reorder data is not present in the disk index", -1);
    }
    const size_t reorder_count =
        std::min<size_t>(full_result.size(), top_k * k_full_precision_reorder_multiplier);
    if (reorder_count > defaults::MAX_N_SECTOR_READS) {
      throw diskann_exception_t("Reorder candidate count exceeds query scratch capacity", -1);
    }
    full_result.resize(reorder_count);
    std::vector<aligned_read_t> reorder_requests;
    reorder_requests.reserve(reorder_count);
    for (size_t i = 0; i < reorder_count; ++i) {
      const uint64_t id = full_result[i].id;
      const uint64_t sector = id / vectors_per_sector_ + reorder_data_start_sector_;
      reorder_requests.emplace_back(sector * defaults::SECTOR_LEN, defaults::SECTOR_LEN,
                                    sector_scratch + i * defaults::SECTOR_LEN);
      if (stats != nullptr) {
        ++stats->n_4k;
        ++stats->n_ios;
        stats->read_size += defaults::SECTOR_LEN;
      }
    }
    io_timer.reset();
    reader_->read(reorder_requests, io_context);
    if (stats != nullptr) {
      stats->io_us += static_cast<float>(io_timer.elapsed());
    }
    for (size_t i = 0; i < reorder_count; ++i) {
      const uint64_t id = full_result[i].id;
      const uint64_t offset = (id % vectors_per_sector_) * data_dim_ * sizeof(float);
      const char* location = sector_scratch + i * defaults::SECTOR_LEN + offset;
      full_result[i].distance =
          distance_comparator_->compare(aligned_query, reinterpret_cast<const data_t*>(location),
                                        static_cast<uint32_t>(data_dim_));
    }
    std::sort(full_result.begin(), full_result.end());
  }

  if (full_result.size() < top_k) {
    throw diskann_exception_t("Beam search produced fewer results than requested", -1);
  }
  for (size_t i = 0; i < top_k; ++i) {
    indices[i] = full_result[i].id;
    if (distances != nullptr) {
      distances[i] = full_result[i].distance;
    }
  }
  if (stats != nullptr) {
    if constexpr (use_hybrid) {
      if (stats->gateway_nodes_inserted + stats->cell_nodes_inserted == 0) {
        stats->fallback_queries = 1;
      }
    }
    stats->total_us = static_cast<float>(query_timer.elapsed());
    if constexpr (!track_beam_phase) {
      // Normal Release QPS runs avoid hundreds of fine-grained clock reads per query. The
      // phase-profiled specialization retains the detailed CPU attribution; the normal path
      // reports coarse non-I/O service time without perturbing the hot loop.
      stats->cpu_us = std::max(0.0F, stats->total_us - stats->io_us);
    }
    if constexpr (track_beam_phase) {
      stats->beam_phase_post_search_us = std::max(
          0.0F, stats->total_us - stats->beam_phase_setup_us - stats->beam_phase_search_loop_us);
    }
  }
}

template <typename data_t> void pq_flash_index_t<data_t>::start_node_visit_tracking() {
  node_visit_tracker_.start(num_points_);
}

template <typename data_t> void pq_flash_index_t<data_t>::stop_node_visit_tracking() {
  node_visit_tracker_.stop();
}

template <typename data_t>
void pq_flash_index_t<data_t>::dump_node_visit_counts(const std::string& output_path) const {
  node_visit_tracker_.dump_csv(output_path);
}

template class pq_flash_index_t<float>;

} // namespace powerlaw_ann
