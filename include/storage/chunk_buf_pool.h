#ifndef STORAGE_CHUNK_BUF_POOL
#define STORAGE_CHUNK_BUF_POOL

#include "storage/chunk.h"

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

/**
 * @brief Runtime configuration for chunk buffer pool capacity.
 */
struct chunk_buf_pool_options_t {
  /**
   * @brief Reserved chunk count for each chunk level.
   */
  std::array<size_t, disk_manager_t::k_chunk_levels> reserved_chunk_counts{};
};

/**
 * @brief Buffer pool level with contiguous chunk controls and data frames.
 */
class chunk_level_pool_t {
public:
  /**
   * @brief Construct an empty level pool.
   */
  chunk_level_pool_t() = default;

  /**
   * @brief Destroy all chunk controls and release the level memory.
   */
  ~chunk_level_pool_t();

  chunk_level_pool_t(const chunk_level_pool_t&) = delete;
  chunk_level_pool_t& operator=(const chunk_level_pool_t&) = delete;

  /**
   * @brief Move construct a level pool.
   * @param other Source level pool.
   */
  chunk_level_pool_t(chunk_level_pool_t&& other) noexcept;

  /**
   * @brief Move assign a level pool.
   * @param other Source level pool.
   * @return This level pool.
   */
  chunk_level_pool_t& operator=(chunk_level_pool_t&& other) noexcept;

  /**
   * @brief Initialize this level with one contiguous allocation.
   * @param type Chunk level type.
   * @param chunk_count Number of chunks to reserve.
   */
  void init(chunk_type_t type, size_t chunk_count);

  /**
   * @brief Get reserved chunk count.
   * @return Number of chunks in this level.
   */
  size_t get_chunk_count() const;

  /**
   * @brief Get free chunk count.
   * @return Number of currently free chunks.
   */
  size_t get_free_chunk_count() const;

  /**
   * @brief Get chunk byte size.
   * @return Chunk size in bytes.
   */
  size_t get_chunk_size() const;

  /**
   * @brief Get a chunk control by level-local index.
   * @param index Level-local chunk index.
   * @return Chunk control pointer, or nullptr if out of range.
   */
  chunk_t* get_chunk(size_t index);

  /**
   * @brief Get a chunk control by level-local index.
   * @param index Level-local chunk index.
   * @return Const chunk control pointer, or nullptr if out of range.
   */
  const chunk_t* get_chunk(size_t index) const;

  /**
   * @brief Get a chunk data frame by level-local index.
   * @param index Level-local chunk index.
   * @return Data frame pointer, or nullptr if out of range.
   */
  char* get_data_frame(size_t index);

  /**
   * @brief Get the first control body in this level.
   * @return Chunk control base pointer.
   */
  chunk_t* get_control_base();

  /**
   * @brief Get the first data frame in this level.
   * @return Data frame base pointer.
   */
  char* get_data_base();

  /**
   * @brief Allocate one free chunk frame from this level.
   * @return Allocated chunk pointer, or nullptr if exhausted.
   */
  chunk_t* allocate_chunk();

  /**
   * @brief Release one resident chunk back to this level.
   * @param chunk Chunk to release.
   */
  void release_chunk(chunk_t* chunk);

private:
  /**
   * @brief Data frame alignment in bytes.
   */
  static constexpr size_t k_data_alignment = 4096;

  /**
   * @brief Round a size up to an alignment boundary.
   * @param value Size value.
   * @param alignment Alignment value.
   * @return Aligned size.
   */
  static size_t align_up(size_t value, size_t alignment);

  /**
   * @brief Destroy controls and release memory.
   */
  void destroy();

  /**
   * @brief Move from another level pool.
   * @param other Source level pool.
   */
  void move_from(chunk_level_pool_t& other);

  /**
   * @brief Chunk level type.
   */
  chunk_type_t type_{chunk_type_t::chunk_4kb};

  /**
   * @brief Raw allocation holding controls followed by data frames.
   */
  void* memory_{nullptr};

  /**
   * @brief Contiguous chunk control body region.
   */
  chunk_t* chunks_{nullptr};

  /**
   * @brief Contiguous chunk data frame region.
   */
  char* frames_{nullptr};

  /**
   * @brief Reserved chunk count.
   */
  size_t chunk_count_{0};

  /**
   * @brief Chunk data frame size in bytes.
   */
  size_t chunk_size_{0};

  /**
   * @brief Control region byte size.
   */
  size_t control_bytes_{0};

  /**
   * @brief Data region offset from the raw allocation base.
   */
  size_t data_offset_{0};

  /**
   * @brief Total raw allocation byte size.
   */
  size_t total_bytes_{0};

  /**
   * @brief Stack of free level-local chunk indexes.
   */
  std::vector<size_t> free_slots_;
};

/**
 * @brief Chunk buffer pool with independently configured chunk levels.
 */
class chunk_buf_pool_t {
public:
  /**
   * @brief Construct an empty chunk buffer pool.
   */
  chunk_buf_pool_t() = default;

  /**
   * @brief Construct a chunk buffer pool from runtime options.
   * @param options Pool capacity options.
   */
  explicit chunk_buf_pool_t(const chunk_buf_pool_options_t& options);

  chunk_buf_pool_t(const chunk_buf_pool_t&) = delete;
  chunk_buf_pool_t& operator=(const chunk_buf_pool_t&) = delete;
  chunk_buf_pool_t(chunk_buf_pool_t&&) = delete;
  chunk_buf_pool_t& operator=(chunk_buf_pool_t&&) = delete;

  /**
   * @brief Initialize a chunk buffer pool.
   * @param options Pool capacity options.
   * @return Pointer to the initialized pool.
   */
  static std::unique_ptr<chunk_buf_pool_t>
  chunk_buf_pool_init(const chunk_buf_pool_options_t& options);

  /**
   * @brief Initialize this pool from runtime options.
   * @param options Pool capacity options.
   */
  void init(const chunk_buf_pool_options_t& options);

  /**
   * @brief Get the number of chunk levels.
   * @return Chunk level count.
   */
  size_t get_level_count() const;

  /**
   * @brief Get total reserved chunk count.
   * @return Total chunk count.
   */
  size_t get_total_chunk_count() const;

  /**
   * @brief Get reserved chunk count for a level.
   * @param type Chunk level type.
   * @return Reserved chunk count.
   */
  size_t get_reserved_chunk_count(chunk_type_t type) const;

  /**
   * @brief Get free chunk count for a level.
   * @param type Chunk level type.
   * @return Free chunk count.
   */
  size_t get_free_chunk_count(chunk_type_t type) const;

  /**
   * @brief Get chunk byte size for a level.
   * @param type Chunk level type.
   * @return Chunk size in bytes.
   */
  size_t get_chunk_size(chunk_type_t type) const;

  /**
   * @brief Get a chunk by level and local index.
   * @param type Chunk level type.
   * @param index Level-local chunk index.
   * @return Chunk pointer, or nullptr if out of range.
   */
  chunk_t* get_chunk(chunk_type_t type, size_t index);

  /**
   * @brief Get a chunk by encoded chunk id.
   * @param chunk_id Encoded chunk id.
   * @return Chunk pointer, or nullptr if out of range.
   */
  chunk_t* get_chunk(chunk_id_t chunk_id);

  /**
   * @brief Get a chunk data frame by level and local index.
   * @param type Chunk level type.
   * @param index Level-local chunk index.
   * @return Data frame pointer, or nullptr if out of range.
   */
  char* get_data_frame(chunk_type_t type, size_t index);

  /**
   * @brief Allocate one free chunk from a level.
   * @param type Chunk level type.
   * @return Allocated chunk, or nullptr if exhausted.
   */
  chunk_t* allocate_chunk(chunk_type_t type);

  /**
   * @brief Release a chunk back to its level.
   * @param chunk Chunk to release.
   */
  void release_chunk(chunk_t* chunk);

private:
  /**
   * @brief Number of supported chunk levels.
   */
  static constexpr size_t k_chunk_levels = disk_manager_t::k_chunk_levels;

  /**
   * @brief Convert chunk type to level index.
   * @param type Chunk level type.
   * @return Level index.
   */
  static size_t level_index(chunk_type_t type);

  /**
   * @brief Get a mutable level pool.
   * @param type Chunk level type.
   * @return Level pool reference.
   */
  chunk_level_pool_t& level(chunk_type_t type);

  /**
   * @brief Get a const level pool.
   * @param type Chunk level type.
   * @return Const level pool reference.
   */
  const chunk_level_pool_t& level(chunk_type_t type) const;

  /**
   * @brief Per-level chunk pools.
   */
  std::array<chunk_level_pool_t, k_chunk_levels> levels_;

  /**
   * @brief Total reserved chunk count.
   */
  size_t total_chunk_count_{0};
};

#endif // STORAGE_CHUNK_BUF_POOL
