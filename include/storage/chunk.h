#ifndef STORAGE_CHUNK
#define STORAGE_CHUNK

#include "storage/disk_manager.h"
#include "storage/storage_types.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <shared_mutex>

/**
 * @brief Buffer-pool residency state for a chunk frame.
 */
enum class chunk_state_t : uint8_t {
  free = 0,
  resident = 1,
};

/**
 * @brief Control body for one fixed-size chunk frame.
 */
class alignas(64) chunk_t {
public:
  /**
   * @brief Cache-line size used for chunk control body alignment.
   */
  static constexpr size_t k_cache_line_size = 64;

  /**
   * @brief Invalid chunk identifier used by unbound frames.
   */
  static constexpr chunk_id_t k_invalid_chunk_id = std::numeric_limits<chunk_id_t>::max();

  /**
   * @brief Construct an unbound chunk control body.
   */
  chunk_t() = default;

  /**
   * @brief Construct a chunk control body bound to an external data frame.
   * @param chunk_id Encoded chunk id.
   * @param type Chunk file type.
   * @param data External data frame pointer.
   * @param size Data frame size in bytes.
   */
  chunk_t(chunk_id_t chunk_id, chunk_type_t type, void* data, size_t size) {
    bind(chunk_id, type, data, size);
  }

  /**
   * @brief Move construct a chunk control body.
   * @param other Source chunk.
   */
  chunk_t(chunk_t&& other) noexcept
      : data_(other.data_), chunk_id_(other.chunk_id_), type_(other.type_), size_(other.size_),
        pin_count_(other.pin_count_.load(std::memory_order_acquire)),
        is_dirty_(other.is_dirty_.load(std::memory_order_acquire)),
        referenced_(other.referenced_.load(std::memory_order_acquire)), next_(other.next_),
        prev_(other.prev_), state_(other.state_) {
    other.data_ = nullptr;
    other.chunk_id_ = k_invalid_chunk_id;
    other.size_ = 0;
    other.pin_count_.store(0, std::memory_order_release);
    other.is_dirty_.store(false, std::memory_order_release);
    other.referenced_.store(false, std::memory_order_release);
    other.next_ = nullptr;
    other.prev_ = nullptr;
    other.state_ = chunk_state_t::free;
  }

  /**
   * @brief Move assign a chunk control body.
   * @param other Source chunk.
   * @return This chunk.
   */
  chunk_t& operator=(chunk_t&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    data_ = other.data_;
    chunk_id_ = other.chunk_id_;
    type_ = other.type_;
    size_ = other.size_;
    pin_count_.store(other.pin_count_.load(std::memory_order_acquire), std::memory_order_release);
    is_dirty_.store(other.is_dirty_.load(std::memory_order_acquire), std::memory_order_release);
    referenced_.store(other.referenced_.load(std::memory_order_acquire), std::memory_order_release);
    next_ = other.next_;
    prev_ = other.prev_;
    state_ = other.state_;

    other.data_ = nullptr;
    other.chunk_id_ = k_invalid_chunk_id;
    other.size_ = 0;
    other.pin_count_.store(0, std::memory_order_release);
    other.is_dirty_.store(false, std::memory_order_release);
    other.referenced_.store(false, std::memory_order_release);
    other.next_ = nullptr;
    other.prev_ = nullptr;
    other.state_ = chunk_state_t::free;
    return *this;
  }

  chunk_t(const chunk_t&) = delete;
  chunk_t& operator=(const chunk_t&) = delete;

  /**
   * @brief Bind this control body to an external data frame.
   * @param chunk_id Encoded chunk id.
   * @param type Chunk file type.
   * @param data External data frame pointer.
   * @param size Data frame size in bytes.
   */
  void bind(chunk_id_t chunk_id, chunk_type_t type, void* data, size_t size) {
    chunk_id_ = chunk_id;
    type_ = type;
    data_ = reinterpret_cast<char*>(data);
    size_ = size;
    pin_count_.store(0, std::memory_order_release);
    is_dirty_.store(false, std::memory_order_release);
    referenced_.store(false, std::memory_order_release);
    next_ = nullptr;
    prev_ = nullptr;
    state_ = chunk_state_t::free;
  }

  /**
   * @brief Reset data bytes and runtime metadata while keeping the binding.
   */
  void reset_memory() {
    if (data_ != nullptr && size_ != 0) {
      std::memset(data_, 0, size_);
    }
    pin_count_.store(0, std::memory_order_release);
    is_dirty_.store(false, std::memory_order_release);
    referenced_.store(false, std::memory_order_release);
  }

  /**
   * @brief Get the bound data frame.
   * @return Mutable data pointer.
   */
  char* get_data() { return data_; }

  /**
   * @brief Get the bound data frame.
   * @return Const data pointer.
   */
  const char* get_data() const { return data_; }

  /**
   * @brief Replace the bound data frame.
   * @param data New external data frame pointer.
   * @param size Data frame size in bytes.
   */
  void set_data(void* data, size_t size) {
    data_ = reinterpret_cast<char*>(data);
    size_ = size;
  }

  /**
   * @brief Get the encoded chunk id.
   * @return Encoded chunk id.
   */
  chunk_id_t get_chunk_id() const { return chunk_id_; }

  /**
   * @brief Set the encoded chunk id.
   * @param chunk_id Encoded chunk id.
   */
  void set_chunk_id(chunk_id_t chunk_id) { chunk_id_ = chunk_id; }

  /**
   * @brief Get the chunk file type.
   * @return Chunk file type.
   */
  chunk_type_t get_chunk_type() const { return type_; }

  /**
   * @brief Set the chunk file type.
   * @param type Chunk file type.
   */
  void set_chunk_type(chunk_type_t type) { type_ = type; }

  /**
   * @brief Get the chunk number inside its chunk file.
   * @return Chunk number.
   */
  chunk_no_t get_chunk_no() const { return disk_manager_t::chunk_no(chunk_id_); }

  /**
   * @brief Get the bound data frame size.
   * @return Data frame size in bytes.
   */
  size_t get_chunk_size() const { return size_; }

  /**
   * @brief Increment the pin count.
   * @return Pin count after increment.
   */
  size_t inc_pin_count() { return pin_count_.fetch_add(1, std::memory_order_acq_rel) + 1; }

  /**
   * @brief Decrement the pin count.
   * @return Pin count after decrement.
   */
  size_t dec_pin_count() { return pin_count_.fetch_sub(1, std::memory_order_acq_rel) - 1; }

  /**
   * @brief Get the current pin count.
   * @return Current pin count.
   */
  size_t get_pin_count() const { return pin_count_.load(std::memory_order_acquire); }

  /**
   * @brief Mark the chunk as dirty.
   */
  void mark_dirty() { set_dirty(true); }

  /**
   * @brief Set the dirty flag.
   * @param is_dirty New dirty state.
   */
  void set_dirty(bool is_dirty) { is_dirty_.store(is_dirty, std::memory_order_release); }

  /**
   * @brief Check whether the chunk has been modified.
   * @return True if the chunk is dirty.
   */
  bool is_dirty() const { return is_dirty_.load(std::memory_order_acquire); }

  /**
   * @brief Set the replacement referenced bit.
   * @param referenced New referenced state.
   */
  void set_referenced(bool referenced) { referenced_.store(referenced, std::memory_order_release); }

  /**
   * @brief Get the replacement referenced bit.
   * @return True if the chunk was recently referenced.
   */
  bool get_referenced() const { return referenced_.load(std::memory_order_acquire); }

  /**
   * @brief Acquire the chunk write latch.
   */
  void w_latch() { latch_.lock(); }

  /**
   * @brief Release the chunk write latch.
   */
  void w_unlatch() { latch_.unlock(); }

  /**
   * @brief Acquire the chunk read latch.
   */
  void r_latch() { latch_.lock_shared(); }

  /**
   * @brief Release the chunk read latch.
   */
  void r_unlatch() { latch_.unlock_shared(); }

  /**
   * @brief Get the next chunk in an intrusive list.
   * @return Next chunk pointer.
   */
  chunk_t* get_next() const { return next_; }

  /**
   * @brief Set the next chunk in an intrusive list.
   * @param next Next chunk pointer.
   */
  void set_next(chunk_t* next) { next_ = next; }

  /**
   * @brief Get the previous chunk in an intrusive list.
   * @return Previous chunk pointer.
   */
  chunk_t* get_prev() const { return prev_; }

  /**
   * @brief Set the previous chunk in an intrusive list.
   * @param prev Previous chunk pointer.
   */
  void set_prev(chunk_t* prev) { prev_ = prev; }

  /**
   * @brief Get the buffer-pool state.
   * @return Current chunk state.
   */
  chunk_state_t get_chunk_state() const { return state_; }

  /**
   * @brief Set the buffer-pool state.
   * @param state New chunk state.
   */
  void set_chunk_state(chunk_state_t state) { state_ = state; }

private:
  /**
   * @brief External data frame for this chunk.
   */
  char* data_{nullptr};

  /**
   * @brief Encoded chunk id currently stored in this frame.
   */
  chunk_id_t chunk_id_{k_invalid_chunk_id};

  /**
   * @brief Chunk file type for this frame.
   */
  chunk_type_t type_{chunk_type_t::chunk_4kb};

  /**
   * @brief Data frame size in bytes.
   */
  size_t size_{0};

  /**
   * @brief Pin count used by the buffer pool.
   */
  std::atomic<size_t> pin_count_{0};

  /**
   * @brief Dirty flag for write-back.
   */
  std::atomic<bool> is_dirty_{false};

  /**
   * @brief Replacement referenced bit.
   */
  std::atomic<bool> referenced_{false};

  /**
   * @brief Latch protecting chunk data access.
   */
  std::shared_mutex latch_;

  /**
   * @brief Next chunk in an intrusive list.
   */
  chunk_t* next_{nullptr};

  /**
   * @brief Previous chunk in an intrusive list.
   */
  chunk_t* prev_{nullptr};

  /**
   * @brief Buffer-pool residency state.
   */
  chunk_state_t state_{chunk_state_t::free};
};

static_assert(alignof(chunk_t) >= chunk_t::k_cache_line_size);
static_assert(sizeof(chunk_t) % chunk_t::k_cache_line_size == 0);

#endif // STORAGE_CHUNK
