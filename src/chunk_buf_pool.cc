#include "storage/chunk_buf_pool.h"

#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

chunk_level_pool_t::~chunk_level_pool_t() { destroy(); }

chunk_level_pool_t::chunk_level_pool_t(chunk_level_pool_t&& other) noexcept { move_from(other); }

chunk_level_pool_t& chunk_level_pool_t::operator=(chunk_level_pool_t&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  destroy();
  move_from(other);
  return *this;
}

void chunk_level_pool_t::init(chunk_type_t type, size_t chunk_count) {
  destroy();
  type_ = type;
  chunk_count_ = chunk_count;
  chunk_size_ = disk_manager_t::chunk_size(type);
  if (chunk_count_ == 0) {
    return;
  }

  control_bytes_ = sizeof(chunk_t) * chunk_count_;
  data_offset_ = align_up(control_bytes_, k_data_alignment);
  const size_t data_bytes = chunk_size_ * chunk_count_;
  total_bytes_ = data_offset_ + data_bytes;
  memory_ = ::operator new(total_bytes_, std::align_val_t{k_data_alignment});

  chunks_ = static_cast<chunk_t*>(memory_);
  frames_ = reinterpret_cast<char*>(memory_) + data_offset_;
  std::memset(frames_, 0, data_bytes);

  for (size_t i = 0; i < chunk_count_; ++i) {
    const chunk_id_t id = disk_manager_t::make_chunk_id(type_, i);
    new (&chunks_[i]) chunk_t(id, type_, frames_ + i * chunk_size_, chunk_size_);
  }

  free_slots_.reserve(chunk_count_);
  for (size_t i = chunk_count_; i > 0; --i) {
    free_slots_.push_back(i - 1);
  }
}

size_t chunk_level_pool_t::get_chunk_count() const { return chunk_count_; }

size_t chunk_level_pool_t::get_free_chunk_count() const { return free_slots_.size(); }

size_t chunk_level_pool_t::get_chunk_size() const { return chunk_size_; }

chunk_t* chunk_level_pool_t::get_chunk(size_t index) {
  if (index >= chunk_count_) {
    return nullptr;
  }
  return &chunks_[index];
}

const chunk_t* chunk_level_pool_t::get_chunk(size_t index) const {
  if (index >= chunk_count_) {
    return nullptr;
  }
  return &chunks_[index];
}

char* chunk_level_pool_t::get_data_frame(size_t index) {
  if (index >= chunk_count_) {
    return nullptr;
  }
  return frames_ + index * chunk_size_;
}

chunk_t* chunk_level_pool_t::get_control_base() { return chunks_; }

char* chunk_level_pool_t::get_data_base() { return frames_; }

chunk_t* chunk_level_pool_t::allocate_chunk() {
  if (free_slots_.empty()) {
    return nullptr;
  }

  const size_t index = free_slots_.back();
  free_slots_.pop_back();
  chunk_t* chunk = get_chunk(index);
  chunk->reset_memory();
  chunk->set_chunk_state(chunk_state_t::resident);
  chunk->set_referenced(true);
  return chunk;
}

void chunk_level_pool_t::release_chunk(chunk_t* chunk) {
  if (chunk == nullptr || chunk->get_chunk_state() == chunk_state_t::free) {
    return;
  }

  const chunk_no_t index = chunk->get_chunk_no();
  if (index >= chunk_count_ || get_chunk(index) != chunk) {
    throw std::invalid_argument("chunk does not belong to this level");
  }

  chunk->reset_memory();
  chunk->set_chunk_state(chunk_state_t::free);
  free_slots_.push_back(index);
}

size_t chunk_level_pool_t::align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

void chunk_level_pool_t::destroy() {
  if (chunks_ != nullptr) {
    for (size_t i = 0; i < chunk_count_; ++i) {
      chunks_[i].~chunk_t();
    }
  }

  if (memory_ != nullptr) {
    ::operator delete(memory_, std::align_val_t{k_data_alignment});
  }

  memory_ = nullptr;
  chunks_ = nullptr;
  frames_ = nullptr;
  chunk_count_ = 0;
  chunk_size_ = 0;
  control_bytes_ = 0;
  data_offset_ = 0;
  total_bytes_ = 0;
  free_slots_.clear();
}

void chunk_level_pool_t::move_from(chunk_level_pool_t& other) {
  type_ = other.type_;
  memory_ = other.memory_;
  chunks_ = other.chunks_;
  frames_ = other.frames_;
  chunk_count_ = other.chunk_count_;
  chunk_size_ = other.chunk_size_;
  control_bytes_ = other.control_bytes_;
  data_offset_ = other.data_offset_;
  total_bytes_ = other.total_bytes_;
  free_slots_ = std::move(other.free_slots_);

  other.memory_ = nullptr;
  other.chunks_ = nullptr;
  other.frames_ = nullptr;
  other.chunk_count_ = 0;
  other.chunk_size_ = 0;
  other.control_bytes_ = 0;
  other.data_offset_ = 0;
  other.total_bytes_ = 0;
}

chunk_buf_pool_t::chunk_buf_pool_t(const chunk_buf_pool_options_t& options) { init(options); }

std::unique_ptr<chunk_buf_pool_t>
chunk_buf_pool_t::chunk_buf_pool_init(const chunk_buf_pool_options_t& options) {
  return std::make_unique<chunk_buf_pool_t>(options);
}

void chunk_buf_pool_t::init(const chunk_buf_pool_options_t& options) {
  total_chunk_count_ = 0;
  for (size_t i = 0; i < k_chunk_levels; ++i) {
    const auto type = static_cast<chunk_type_t>(i);
    levels_[i].init(type, options.reserved_chunk_counts[i]);
    total_chunk_count_ += options.reserved_chunk_counts[i];
  }
}

size_t chunk_buf_pool_t::get_level_count() const { return k_chunk_levels; }

size_t chunk_buf_pool_t::get_total_chunk_count() const { return total_chunk_count_; }

size_t chunk_buf_pool_t::get_reserved_chunk_count(chunk_type_t type) const {
  return level(type).get_chunk_count();
}

size_t chunk_buf_pool_t::get_free_chunk_count(chunk_type_t type) const {
  return level(type).get_free_chunk_count();
}

size_t chunk_buf_pool_t::get_chunk_size(chunk_type_t type) const {
  return level(type).get_chunk_size();
}

chunk_t* chunk_buf_pool_t::get_chunk(chunk_type_t type, size_t index) {
  return level(type).get_chunk(index);
}

chunk_t* chunk_buf_pool_t::get_chunk(chunk_id_t chunk_id) {
  return get_chunk(disk_manager_t::chunk_type(chunk_id), disk_manager_t::chunk_no(chunk_id));
}

char* chunk_buf_pool_t::get_data_frame(chunk_type_t type, size_t index) {
  return level(type).get_data_frame(index);
}

chunk_t* chunk_buf_pool_t::allocate_chunk(chunk_type_t type) {
  return level(type).allocate_chunk();
}

void chunk_buf_pool_t::release_chunk(chunk_t* chunk) {
  if (chunk == nullptr) {
    return;
  }
  level(chunk->get_chunk_type()).release_chunk(chunk);
}

size_t chunk_buf_pool_t::level_index(chunk_type_t type) {
  const size_t index = static_cast<size_t>(type);
  if (index >= k_chunk_levels) {
    throw std::out_of_range("invalid chunk level");
  }
  return index;
}

chunk_level_pool_t& chunk_buf_pool_t::level(chunk_type_t type) {
  return levels_[level_index(type)];
}

const chunk_level_pool_t& chunk_buf_pool_t::level(chunk_type_t type) const {
  return levels_[level_index(type)];
}
