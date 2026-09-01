#include "index/adaptive_cell_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace powerlaw_ann {
namespace {

constexpr uint64_t k_fnv_offset_basis = 14695981039346656037ULL;
constexpr uint64_t k_fnv_prime = 1099511628211ULL;
constexpr uint64_t k_generation_magic = 0x314E4547434C4350ULL; // "PCLCGEN1"
constexpr uint64_t k_override_filter_bits = 64U * 64U;

uint64_t mix_node_id(uint32_t node_id) noexcept {
  uint64_t value = static_cast<uint64_t>(node_id) + 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

bool filter_contains(const std::array<uint64_t, 64>& filter, uint32_t node_id) noexcept {
  const uint64_t hash = mix_node_id(node_id);
  const uint64_t first = hash % k_override_filter_bits;
  const uint64_t second = (hash >> 32U) % k_override_filter_bits;
  return (filter[first / 64U] & (uint64_t{1} << (first % 64U))) != 0 &&
         (filter[second / 64U] & (uint64_t{1} << (second % 64U))) != 0;
}

void filter_insert(std::array<uint64_t, 64>& filter, uint32_t node_id) noexcept {
  const uint64_t hash = mix_node_id(node_id);
  const uint64_t first = hash % k_override_filter_bits;
  const uint64_t second = (hash >> 32U) % k_override_filter_bits;
  filter[first / 64U] |= uint64_t{1} << (first % 64U);
  filter[second / 64U] |= uint64_t{1} << (second % 64U);
}

void hash_bytes(uint64_t& checksum, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t index = 0; index < size; ++index) {
    checksum ^= bytes[index];
    checksum *= k_fnv_prime;
  }
}

template <typename value_t>
void append_little(std::vector<uint8_t>& bytes, value_t value) {
  for (size_t byte = 0; byte < sizeof(value_t); ++byte) {
    bytes.push_back(static_cast<uint8_t>(value >> (byte * 8U)));
  }
}

void pread_exact(int descriptor, void* destination, size_t size, uint64_t offset) {
  size_t completed = 0;
  while (completed < size) {
    const auto result = ::pread(descriptor, static_cast<uint8_t*>(destination) + completed,
                                size - completed, offset + completed);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      throw std::runtime_error("Adaptive Cell physical read failed");
    }
    completed += static_cast<size_t>(result);
  }
}

void pwrite_exact(int descriptor, const void* source, size_t size, uint64_t offset) {
  size_t completed = 0;
  while (completed < size) {
    const auto result = ::pwrite(descriptor, static_cast<const uint8_t*>(source) + completed,
                                 size - completed, offset + completed);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      throw std::runtime_error("Adaptive Cell physical write failed");
    }
    completed += static_cast<size_t>(result);
  }
}

uint32_t capacity_for_population(uint32_t population) {
  for (const uint32_t capacity : {16U, 32U, 64U, 128U, 256U, 512U}) {
    if (population <= capacity) {
      return capacity;
    }
  }
  throw std::invalid_argument("Online adaptive Cell population exceeds Cell-512");
}

double relative_increase(uint64_t merged, uint64_t fixed) {
  if (fixed == 0) {
    return merged == 0 ? 0.0 : std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(merged) / static_cast<double>(fixed) - 1.0;
}

} // namespace

struct adaptive_cell_runtime_t::candidate_counters_t {
  std::atomic<uint64_t> parent_observations{0};
  std::atomic<uint64_t> joint_observations{0};
  std::atomic<uint64_t> fixed_pages{0};
  std::atomic<uint64_t> merged_pages{0};
  std::atomic<uint64_t> fixed_unseen{0};
  std::atomic<uint64_t> merged_unseen{0};
};

adaptive_cell_snapshot_t::~adaptive_cell_snapshot_t() {
  if (published_ && retirement_counter_ != nullptr) {
    retirement_counter_->fetch_add(1, std::memory_order_relaxed);
  }
}

const adaptive_cell_view_t& adaptive_cell_snapshot_t::cell(uint32_t logical_cell) const {
  if (logical_cell >= cells_.size()) {
    throw std::out_of_range("Adaptive Cell snapshot logical Cell is out of range");
  }
  return *cells_[logical_cell];
}

io_vector_location_t
adaptive_cell_snapshot_t::vector_location(uint32_t node_id,
                                          io_vector_location_t fallback) const noexcept {
  if (!filter_contains(override_filter_, node_id)) {
    return fallback;
  }
  const auto found = std::lower_bound(override_nodes_.begin(), override_nodes_.end(), node_id);
  if (found == override_nodes_.end() || *found != node_id) {
    return fallback;
  }
  return override_locations_[static_cast<size_t>(found - override_nodes_.begin())];
}

adaptive_cell_runtime_t::adaptive_cell_runtime_t(
    std::span<const io_cell_page_range_t> fixed_cells,
    std::span<const adaptive_cell_merge_candidate_t> candidates,
    adaptive_cell_runtime_config_t config)
    : config_(config), candidates_(candidates.begin(), candidates.end()),
      candidates_by_cell_(fixed_cells.size()),
      counters_(candidates.empty() ? nullptr
                                   : std::make_unique<candidate_counters_t[]>(candidates.size())),
      retirement_counter_(std::make_shared<std::atomic<uint64_t>>(0)),
      pending_(candidates.empty() ? nullptr
                                  : std::make_unique<std::atomic<bool>[]>(candidates.size())) {
  initialize(fixed_cells);
}

adaptive_cell_runtime_t::adaptive_cell_runtime_t(
    std::span<const io_cell_page_range_t> fixed_cells,
    std::span<const adaptive_cell_merge_candidate_t> candidates,
    adaptive_cell_runtime_config_t config, adaptive_cell_physical_config_t physical_config)
    : config_(config), candidates_(candidates.begin(), candidates.end()),
      candidates_by_cell_(fixed_cells.size()),
      counters_(candidates.empty() ? nullptr
                                   : std::make_unique<candidate_counters_t[]>(candidates.size())),
      retirement_counter_(std::make_shared<std::atomic<uint64_t>>(0)),
      pending_(candidates.empty() ? nullptr
                                  : std::make_unique<std::atomic<bool>[]>(candidates.size())),
      base_index_(std::move(physical_config.base_index)),
      overlay_path_(std::move(physical_config.overlay_path)),
      generation_path_(overlay_path_.string() + ".generation"),
      vector_bytes_(physical_config.vector_bytes),
      cell_nodes_(std::move(physical_config.cell_nodes)) {
  if (base_index_ == nullptr || overlay_path_.empty() || vector_bytes_ == 0 ||
      vector_bytes_ > k_io_vector_page_size || cell_nodes_.size() != fixed_cells.size() ||
      base_index_->cell_page_ranges().size() != fixed_cells.size()) {
    throw std::invalid_argument("Online adaptive Cell physical configuration is invalid");
  }
  base_page_count_ = base_index_->vector_page_count();
  node_to_cell_.assign(base_index_->point_count(), UINT32_MAX);
  for (uint32_t cell = 0; cell < cell_nodes_.size(); ++cell) {
    if (cell_nodes_[cell].size() != fixed_cells[cell].node_count) {
      throw std::invalid_argument("Online adaptive Cell membership count is invalid");
    }
    std::sort(cell_nodes_[cell].begin(), cell_nodes_[cell].end(),
              [this](uint32_t left, uint32_t right) {
                const auto left_location = base_index_->vector_location(left);
                const auto right_location = base_index_->vector_location(right);
                return std::tie(left_location.page_id, left_location.page_offset, left) <
                       std::tie(right_location.page_id, right_location.page_offset, right);
              });
    for (const uint32_t node : cell_nodes_[cell]) {
      if (node >= node_to_cell_.size() || node_to_cell_[node] != UINT32_MAX) {
        throw std::invalid_argument("Online adaptive Cell membership is not one-owner complete");
      }
      node_to_cell_[node] = cell;
    }
  }
  if (std::find(node_to_cell_.begin(), node_to_cell_.end(), UINT32_MAX) != node_to_cell_.end()) {
    throw std::invalid_argument("Online adaptive Cell membership is incomplete");
  }
  base_file_descriptor_ = ::open(base_index_->vector_path().c_str(), O_RDONLY);
  if (base_file_descriptor_ < 0) {
    throw std::runtime_error("Failed to open the Base Cell vector artifact for adaptation");
  }
  overlay_file_descriptor_ =
      ::open(overlay_path_.c_str(), O_CREAT | O_TRUNC | O_RDWR, S_IRUSR | S_IWUSR);
  if (overlay_file_descriptor_ < 0) {
    ::close(base_file_descriptor_);
    base_file_descriptor_ = -1;
    throw std::runtime_error("Failed to create the adaptive Cell overlay artifact");
  }
  initialize(fixed_cells);
}

void adaptive_cell_runtime_t::initialize(std::span<const io_cell_page_range_t> fixed_cells) {
  if (fixed_cells.empty() || config_.minimum_joint_observations == 0 ||
      !std::isfinite(config_.minimum_coaccess_ratio) ||
      !std::isfinite(config_.maximum_page_increase) ||
      !std::isfinite(config_.maximum_unseen_increase) || config_.minimum_coaccess_ratio < 0.0 ||
      config_.minimum_coaccess_ratio > 1.0 || config_.maximum_page_increase < 0.0 ||
      config_.maximum_unseen_increase < 0.0) {
    throw std::invalid_argument("Online adaptive Cell runtime configuration is invalid");
  }
  auto initial = std::make_shared<adaptive_cell_snapshot_t>();
  initial->retirement_counter_ = retirement_counter_;
  initial->published_ = true;
  initial->cells_.reserve(fixed_cells.size());
  for (uint32_t cell = 0; cell < fixed_cells.size(); ++cell) {
    const auto& range = fixed_cells[cell];
    if (range.node_count == 0 || range.page_count == 0) {
      throw std::invalid_argument("Online adaptive Cell runtime requires nonempty page ranges");
    }
    auto view = std::make_shared<adaptive_cell_view_t>();
    view->view_id = cell;
    view->capacity_class = capacity_for_population(range.node_count);
    view->node_count = range.node_count;
    view->physical_payload_ready = true;
    view->logical_cell_ids = {cell};
    view->page_runs = {{range.first_page, range.page_count}};
    initial->cells_.push_back(std::move(view));
  }
  for (uint32_t candidate_id = 0; candidate_id < candidates_.size(); ++candidate_id) {
    const auto candidate = candidates_[candidate_id];
    if (candidate.left_cell >= fixed_cells.size() || candidate.right_cell >= fixed_cells.size() ||
        candidate.left_cell == candidate.right_cell) {
      throw std::invalid_argument("Online adaptive Cell merge candidate is invalid");
    }
    candidates_by_cell_[candidate.left_cell].push_back(candidate_id);
    candidates_by_cell_[candidate.right_cell].push_back(candidate_id);
  }
  for (size_t candidate = 0; candidate < candidates_.size(); ++candidate) {
    pending_[candidate].store(false, std::memory_order_relaxed);
  }
  snapshot_ = std::move(initial);
  if (config_.enabled && config_.background_publish && !candidates_.empty()) {
    background_thread_ = std::thread([this]() { background_loop(); });
  }
}

adaptive_cell_runtime_t::~adaptive_cell_runtime_t() {
  stop_background_.store(true, std::memory_order_release);
  background_wakeup_.notify_all();
  if (background_thread_.joinable()) {
    background_thread_.join();
  }
  if (base_file_descriptor_ >= 0) {
    ::close(base_file_descriptor_);
  }
  if (overlay_file_descriptor_ >= 0) {
    ::close(overlay_file_descriptor_);
  }
  if (!overlay_path_.empty()) {
    std::error_code ignored;
    std::filesystem::remove(overlay_path_, ignored);
    std::filesystem::remove(generation_path_, ignored);
  }
}

std::shared_ptr<const adaptive_cell_snapshot_t> adaptive_cell_runtime_t::acquire() const noexcept {
  return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
}

void adaptive_cell_runtime_t::observe(uint32_t candidate, bool left_loaded, bool right_loaded,
                                      uint32_t fixed_pages, uint32_t merged_pages,
                                      uint32_t fixed_unseen, uint32_t merged_unseen) noexcept {
  if (!config_.enabled || candidate >= candidates_.size() || (!left_loaded && !right_loaded)) {
    return;
  }
  auto& counters = counters_[candidate];
  counters.joint_observations.fetch_add(left_loaded && right_loaded ? 1U : 0U,
                                        std::memory_order_relaxed);
  counters.fixed_pages.fetch_add(fixed_pages, std::memory_order_relaxed);
  counters.merged_pages.fetch_add(merged_pages, std::memory_order_relaxed);
  counters.fixed_unseen.fetch_add(fixed_unseen, std::memory_order_relaxed);
  counters.merged_unseen.fetch_add(merged_unseen, std::memory_order_relaxed);
  counters.parent_observations.fetch_add(1, std::memory_order_release);
  observations_.fetch_add(1, std::memory_order_relaxed);
}

void adaptive_cell_runtime_t::observe_query(std::span<const uint32_t> loaded_cells) noexcept {
  if (!accepts_observations() || loaded_cells.empty()) {
    return;
  }
  static thread_local std::vector<uint32_t> query_candidates;
  query_candidates.clear();
  for (const uint32_t cell : loaded_cells) {
    if (cell >= candidates_by_cell_.size()) {
      continue;
    }
    query_candidates.insert(query_candidates.end(), candidates_by_cell_[cell].begin(),
                            candidates_by_cell_[cell].end());
  }
  std::sort(query_candidates.begin(), query_candidates.end());
  query_candidates.erase(std::unique(query_candidates.begin(), query_candidates.end()),
                         query_candidates.end());
  if (config_.force_first_observed_modification && !query_candidates.empty() &&
      !forced_modification_queued_.exchange(true, std::memory_order_acq_rel)) {
    forced_candidate_.store(query_candidates.front(), std::memory_order_release);
    pending_[query_candidates.front()].store(true, std::memory_order_release);
    background_wakeup_.notify_one();
  }
  for (const uint32_t candidate : query_candidates) {
    if (!accepts_observations()) {
      break;
    }
    const auto pair = candidates_[candidate];
    const bool left_loaded =
        std::binary_search(loaded_cells.begin(), loaded_cells.end(), pair.left_cell);
    const bool right_loaded =
        std::binary_search(loaded_cells.begin(), loaded_cells.end(), pair.right_cell);
    if (!left_loaded && !right_loaded) {
      continue;
    }
    const auto current = acquire();
    const auto& left = current->cell(pair.left_cell);
    const auto& right = current->cell(pair.right_cell);
    if (left.view_id == right.view_id) {
      continue;
    }
    const auto page_count = [](const adaptive_cell_view_t& view) {
      uint32_t pages = 0;
      for (const auto& run : view.page_runs) {
        pages += run.page_count;
      }
      return pages;
    };
    const uint32_t left_pages = page_count(left);
    const uint32_t right_pages = page_count(right);
    const uint32_t fixed_pages =
        (left_loaded ? left_pages : 0U) + (right_loaded ? right_pages : 0U);
    const uint32_t merged_pages = left_pages + right_pages;
    const uint32_t fixed_unseen =
        (left_loaded ? left.node_count : 0U) + (right_loaded ? right.node_count : 0U);
    const uint32_t merged_unseen = left.node_count + right.node_count;
    observe(candidate, left_loaded, right_loaded, fixed_pages, merged_pages, fixed_unseen,
            merged_unseen);
    if (config_.background_publish && profile(candidate).eligible) {
      pending_[candidate].store(true, std::memory_order_release);
      background_wakeup_.notify_one();
    }
  }
}

uint32_t adaptive_cell_runtime_t::logical_cell(uint32_t node_id) const noexcept {
  return node_id < node_to_cell_.size() ? node_to_cell_[node_id] : UINT32_MAX;
}

adaptive_cell_merge_profile_t adaptive_cell_runtime_t::profile(uint32_t candidate) const {
  if (candidate >= candidates_.size()) {
    throw std::out_of_range("Online adaptive Cell profile candidate is out of range");
  }
  const auto& counters = counters_[candidate];
  adaptive_cell_merge_profile_t result;
  result.parent_observations = counters.parent_observations.load(std::memory_order_acquire);
  result.joint_observations = counters.joint_observations.load(std::memory_order_relaxed);
  result.fixed_pages = counters.fixed_pages.load(std::memory_order_relaxed);
  result.merged_pages = counters.merged_pages.load(std::memory_order_relaxed);
  result.fixed_unseen = counters.fixed_unseen.load(std::memory_order_relaxed);
  result.merged_unseen = counters.merged_unseen.load(std::memory_order_relaxed);
  result.coaccess_ratio = result.parent_observations == 0
                              ? 0.0
                              : static_cast<double>(result.joint_observations) /
                                    static_cast<double>(result.parent_observations);
  result.page_increase = relative_increase(result.merged_pages, result.fixed_pages);
  result.unseen_increase = relative_increase(result.merged_unseen, result.fixed_unseen);
  result.eligible = config_.enabled &&
                    result.joint_observations >= config_.minimum_joint_observations &&
                    result.coaccess_ratio >= config_.minimum_coaccess_ratio &&
                    result.page_increase <= config_.maximum_page_increase &&
                    result.unseen_increase <= config_.maximum_unseen_increase;
  return result;
}

bool adaptive_cell_runtime_t::try_publish(uint32_t candidate,
                                          std::shared_ptr<const adaptive_cell_view_t> replacement,
                                          bool force_experimental_operation) {
  if (!accepts_observations() || candidate >= candidates_.size() || replacement == nullptr ||
      (!force_experimental_operation && !profile(candidate).eligible)) {
    return false;
  }
  publish_attempts_.fetch_add(1, std::memory_order_relaxed);
  const auto pair = candidates_[candidate];
  auto current = acquire();
  const auto& left = current->cell(pair.left_cell);
  const auto& right = current->cell(pair.right_cell);
  if (left.view_id == right.view_id || !replacement->physical_payload_ready ||
      replacement->generation != current->generation() + 1 || replacement->page_runs.size() != 1 ||
      replacement->node_count != left.node_count + right.node_count ||
      replacement->capacity_class != capacity_for_population(replacement->node_count)) {
    return false;
  }
  std::vector<uint32_t> expected_cells = left.logical_cell_ids;
  expected_cells.insert(expected_cells.end(), right.logical_cell_ids.begin(),
                        right.logical_cell_ids.end());
  std::sort(expected_cells.begin(), expected_cells.end());
  expected_cells.erase(std::unique(expected_cells.begin(), expected_cells.end()),
                       expected_cells.end());
  if (replacement->logical_cell_ids != expected_cells) {
    return false;
  }
  if ((!replacement->vector_nodes.empty() || !replacement->vector_locations.empty()) &&
      (replacement->vector_nodes.size() != replacement->node_count ||
       replacement->vector_locations.size() != replacement->vector_nodes.size() ||
       replacement->payload_checksum == 0 || replacement->directory_checksum == 0 ||
       !std::is_sorted(replacement->vector_nodes.begin(), replacement->vector_nodes.end()) ||
       std::adjacent_find(replacement->vector_nodes.begin(), replacement->vector_nodes.end()) !=
           replacement->vector_nodes.end())) {
    return false;
  }

  auto next = std::make_shared<adaptive_cell_snapshot_t>();
  next->generation_ = current->generation() + 1;
  next->cells_ = current->cells_;
  for (const uint32_t logical_cell : expected_cells) {
    if (logical_cell >= next->cells_.size()) {
      return false;
    }
    next->cells_[logical_cell] = replacement;
  }
  std::map<uint32_t, io_vector_location_t> directory;
  for (size_t entry = 0; entry < current->override_nodes_.size(); ++entry) {
    directory[current->override_nodes_[entry]] = current->override_locations_[entry];
  }
  for (size_t entry = 0; entry < replacement->vector_nodes.size(); ++entry) {
    directory[replacement->vector_nodes[entry]] = replacement->vector_locations[entry];
  }
  next->override_nodes_.reserve(directory.size());
  next->override_locations_.reserve(directory.size());
  for (const auto& [node, location] : directory) {
    next->override_nodes_.push_back(node);
    next->override_locations_.push_back(location);
    filter_insert(next->override_filter_, node);
  }
  next->retirement_counter_ = retirement_counter_;
  next->published_ = true;
  std::shared_ptr<const adaptive_cell_snapshot_t> desired = next;
  auto expected = current;
  if (!std::atomic_compare_exchange_strong_explicit(
          &snapshot_, &expected, desired, std::memory_order_release, std::memory_order_acquire)) {
    next->published_ = false;
    publish_conflicts_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  const uint64_t successful = successful_publishes_.fetch_add(1, std::memory_order_acq_rel) + 1U;
  if (successful >= config_.maximum_successful_publishes) {
    final_snapshot_.store(desired.get(), std::memory_order_release);
  }
  if (!replacement->vector_nodes.empty()) {
    physical_generations_.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
}

std::shared_ptr<const adaptive_cell_view_t>
adaptive_cell_runtime_t::make_contiguous_replacement(uint32_t candidate) const {
  const auto current = acquire();
  const auto pair = candidates_[candidate];
  const auto& left = current->cell(pair.left_cell);
  const auto& right = current->cell(pair.right_cell);
  if (left.view_id == right.view_id || left.page_runs.size() != 1 || right.page_runs.size() != 1) {
    return nullptr;
  }
  const auto left_run = left.page_runs.front();
  const auto right_run = right.page_runs.front();
  const bool left_before = left_run.first_page + left_run.page_count == right_run.first_page;
  const bool right_before = right_run.first_page + right_run.page_count == left_run.first_page;
  if (!left_before && !right_before) {
    return nullptr;
  }
  auto replacement = std::make_shared<adaptive_cell_view_t>();
  replacement->view_id =
      (current->generation() + 1U) << 32U | std::min(pair.left_cell, pair.right_cell);
  replacement->generation = current->generation() + 1U;
  replacement->node_count = left.node_count + right.node_count;
  replacement->capacity_class = capacity_for_population(replacement->node_count);
  replacement->physical_payload_ready = true;
  replacement->logical_cell_ids = left.logical_cell_ids;
  replacement->logical_cell_ids.insert(replacement->logical_cell_ids.end(),
                                       right.logical_cell_ids.begin(),
                                       right.logical_cell_ids.end());
  std::sort(replacement->logical_cell_ids.begin(), replacement->logical_cell_ids.end());
  replacement->logical_cell_ids.erase(
      std::unique(replacement->logical_cell_ids.begin(), replacement->logical_cell_ids.end()),
      replacement->logical_cell_ids.end());
  replacement->page_runs = {
      {std::min(left_run.first_page, right_run.first_page),
       left_run.page_count + right_run.page_count},
  };
  return replacement;
}

std::shared_ptr<const adaptive_cell_view_t>
adaptive_cell_runtime_t::make_physical_replacement(uint32_t candidate) {
  if (base_index_ == nullptr || base_file_descriptor_ < 0 || overlay_file_descriptor_ < 0) {
    return make_contiguous_replacement(candidate);
  }
  const auto started = std::chrono::steady_clock::now();
  const auto current = acquire();
  const auto pair = candidates_[candidate];
  const auto& left = current->cell(pair.left_cell);
  const auto& right = current->cell(pair.right_cell);
  if (left.view_id == right.view_id) {
    return nullptr;
  }
  std::vector<uint32_t> logical_cells = left.logical_cell_ids;
  logical_cells.insert(logical_cells.end(), right.logical_cell_ids.begin(),
                       right.logical_cell_ids.end());
  std::sort(logical_cells.begin(), logical_cells.end());
  logical_cells.erase(std::unique(logical_cells.begin(), logical_cells.end()), logical_cells.end());
  std::vector<uint32_t> packed_nodes;
  packed_nodes.reserve(left.node_count + right.node_count);
  for (const uint32_t cell : logical_cells) {
    if (cell >= cell_nodes_.size()) {
      return nullptr;
    }
    packed_nodes.insert(packed_nodes.end(), cell_nodes_[cell].begin(), cell_nodes_[cell].end());
  }
  if (packed_nodes.size() != left.node_count + right.node_count) {
    return nullptr;
  }
  const uint32_t vectors_per_page = k_io_vector_page_size / vector_bytes_;
  if (vectors_per_page == 0) {
    return nullptr;
  }
  const uint64_t page_count = (packed_nodes.size() + vectors_per_page - 1U) / vectors_per_page;
  const uint64_t local_first_page = next_overlay_page_;
  next_overlay_page_ += page_count;
  const size_t payload_bytes = static_cast<size_t>(page_count * k_io_vector_page_size);
  void* payload_raw = nullptr;
  void* verified_raw = nullptr;
  if (posix_memalign(&payload_raw, k_io_vector_page_size, payload_bytes) != 0 ||
      posix_memalign(&verified_raw, k_io_vector_page_size, payload_bytes) != 0) {
    std::free(payload_raw);
    std::free(verified_raw);
    throw std::bad_alloc();
  }
  std::unique_ptr<void, decltype(&std::free)> payload(payload_raw, &std::free);
  std::unique_ptr<void, decltype(&std::free)> verified(verified_raw, &std::free);
  std::memset(payload.get(), 0, payload_bytes);

  std::vector<std::pair<uint32_t, io_vector_location_t>> directory;
  directory.reserve(packed_nodes.size());
  for (size_t ordinal = 0; ordinal < packed_nodes.size(); ++ordinal) {
    const uint32_t node = packed_nodes[ordinal];
    const auto source_location = current->vector_location(node, base_index_->vector_location(node));
    const bool source_is_overlay = source_location.page_id >= base_page_count_;
    const int source_descriptor =
        source_is_overlay ? overlay_file_descriptor_ : base_file_descriptor_;
    const uint64_t source_page =
        source_is_overlay ? source_location.page_id - base_page_count_ : source_location.page_id;
    const uint64_t source_offset =
        source_page * k_io_vector_page_size + source_location.page_offset;
    const uint64_t local_page = ordinal / vectors_per_page;
    const uint32_t page_offset = static_cast<uint32_t>(ordinal % vectors_per_page) * vector_bytes_;
    auto* destination =
        static_cast<uint8_t*>(payload.get()) + local_page * k_io_vector_page_size + page_offset;
    pread_exact(source_descriptor, destination, vector_bytes_, source_offset);
    directory.push_back({node, {base_page_count_ + local_first_page + local_page, page_offset}});
  }
  pwrite_exact(overlay_file_descriptor_, payload.get(), payload_bytes,
               local_first_page * k_io_vector_page_size);
  if (::fdatasync(overlay_file_descriptor_) != 0) {
    throw std::runtime_error("Adaptive Cell overlay fdatasync failed");
  }
  pread_exact(overlay_file_descriptor_, verified.get(), payload_bytes,
              local_first_page * k_io_vector_page_size);
  uint64_t payload_checksum = k_fnv_offset_basis;
  uint64_t verified_checksum = k_fnv_offset_basis;
  hash_bytes(payload_checksum, payload.get(), payload_bytes);
  hash_bytes(verified_checksum, verified.get(), payload_bytes);
  if (payload_checksum != verified_checksum) {
    checksum_failures_.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error("Adaptive Cell overlay checksum verification failed");
  }
  std::sort(directory.begin(), directory.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  uint64_t directory_checksum = k_fnv_offset_basis;
  for (const auto& [node, location] : directory) {
    hash_bytes(directory_checksum, &node, sizeof(node));
    hash_bytes(directory_checksum, &location.page_id, sizeof(location.page_id));
    hash_bytes(directory_checksum, &location.page_offset, sizeof(location.page_offset));
  }

  auto replacement = std::make_shared<adaptive_cell_view_t>();
  replacement->view_id =
      (current->generation() + 1U) << 32U | std::min(pair.left_cell, pair.right_cell);
  replacement->generation = current->generation() + 1U;
  replacement->capacity_class = capacity_for_population(static_cast<uint32_t>(packed_nodes.size()));
  replacement->node_count = static_cast<uint32_t>(packed_nodes.size());
  replacement->physical_payload_ready = true;
  replacement->logical_cell_ids = std::move(logical_cells);
  replacement->page_runs = {
      {base_page_count_ + local_first_page, static_cast<uint32_t>(page_count)}};
  replacement->payload_checksum = payload_checksum;
  replacement->directory_checksum = directory_checksum;
  replacement->vector_nodes.reserve(directory.size());
  replacement->vector_locations.reserve(directory.size());
  for (const auto& [node, location] : directory) {
    replacement->vector_nodes.push_back(node);
    replacement->vector_locations.push_back(location);
  }

  std::vector<uint8_t> manifest;
  append_little(manifest, k_generation_magic);
  append_little(manifest, static_cast<uint32_t>(1));
  append_little(manifest, replacement->generation);
  append_little(manifest, replacement->page_runs.front().first_page);
  append_little(manifest, replacement->page_runs.front().page_count);
  append_little(manifest, replacement->node_count);
  append_little(manifest, replacement->payload_checksum);
  append_little(manifest, replacement->directory_checksum);
  for (size_t entry = 0; entry < replacement->vector_nodes.size(); ++entry) {
    append_little(manifest, replacement->vector_nodes[entry]);
    append_little(manifest, replacement->vector_locations[entry].page_id);
    append_little(manifest, replacement->vector_locations[entry].page_offset);
  }
  uint64_t manifest_checksum = k_fnv_offset_basis;
  hash_bytes(manifest_checksum, manifest.data(), manifest.size());
  append_little(manifest, manifest_checksum);
  const auto temporary_manifest = std::filesystem::path(generation_path_.string() + ".tmp");
  {
    std::ofstream output(temporary_manifest, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(manifest.data()),
                 static_cast<std::streamsize>(manifest.size()));
    output.flush();
    if (!output) {
      throw std::runtime_error("Adaptive Cell generation manifest write failed");
    }
  }
  std::filesystem::rename(temporary_manifest, generation_path_);
  physical_bytes_written_.fetch_add(payload_bytes + manifest.size(), std::memory_order_relaxed);
  physical_write_time_us_.fetch_add(
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count()),
      std::memory_order_relaxed);
  return replacement;
}

void adaptive_cell_runtime_t::background_loop() noexcept {
  while (!stop_background_.load(std::memory_order_acquire)) {
    std::unique_lock lock(background_mutex_);
    background_wakeup_.wait(lock, [this]() {
      if (stop_background_.load(std::memory_order_acquire)) {
        return true;
      }
      for (size_t candidate = 0; candidate < candidates_.size(); ++candidate) {
        if (pending_[candidate].load(std::memory_order_acquire)) {
          return true;
        }
      }
      return false;
    });
    lock.unlock();
    for (uint32_t candidate = 0; candidate < candidates_.size(); ++candidate) {
      {
        std::lock_guard active_lock(background_mutex_);
        if (!pending_[candidate].exchange(false, std::memory_order_acq_rel)) {
          continue;
        }
        background_active_.fetch_add(1, std::memory_order_acq_rel);
      }
      try {
        const auto replacement = make_physical_replacement(candidate);
        if (replacement != nullptr) {
          const bool force = forced_candidate_.load(std::memory_order_acquire) == candidate;
          try_publish(candidate, replacement, force);
        }
      } catch (...) {
      }
      {
        std::lock_guard active_lock(background_mutex_);
        background_active_.fetch_sub(1, std::memory_order_acq_rel);
      }
      background_wakeup_.notify_all();
    }
  }
}

void adaptive_cell_runtime_t::drain_background() {
  if (!background_thread_.joinable()) {
    return;
  }
  std::unique_lock lock(background_mutex_);
  background_wakeup_.wait(lock, [this]() {
    if (background_active_.load(std::memory_order_acquire) != 0) {
      return false;
    }
    for (size_t candidate = 0; candidate < candidates_.size(); ++candidate) {
      if (pending_[candidate].load(std::memory_order_acquire)) {
        return false;
      }
    }
    return true;
  });
}

adaptive_cell_runtime_metrics_t adaptive_cell_runtime_t::metrics() const noexcept {
  const auto snapshot = acquire();
  const uint64_t directory_bytes =
      snapshot->override_nodes_.capacity() * sizeof(uint32_t) +
      snapshot->override_locations_.capacity() * sizeof(io_vector_location_t) +
      sizeof(snapshot->override_filter_);
  std::error_code ignored;
  const bool overlay_exists =
      !overlay_path_.empty() && std::filesystem::exists(overlay_path_, ignored) && !ignored;
  ignored.clear();
  const uint64_t overlay_bytes =
      overlay_exists ? std::filesystem::file_size(overlay_path_, ignored) : 0;
  return {observations_.load(std::memory_order_relaxed),
          publish_attempts_.load(std::memory_order_relaxed),
          successful_publishes_.load(std::memory_order_relaxed),
          publish_conflicts_.load(std::memory_order_relaxed),
          retirement_counter_->load(std::memory_order_relaxed),
          physical_generations_.load(std::memory_order_relaxed),
          physical_bytes_written_.load(std::memory_order_relaxed),
          physical_write_time_us_.load(std::memory_order_relaxed),
          checksum_failures_.load(std::memory_order_relaxed),
          directory_bytes,
          ignored ? 0 : overlay_bytes};
}

} // namespace powerlaw_ann
