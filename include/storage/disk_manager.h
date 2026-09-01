#ifndef STORAGE_DISK_MANAGER
#define STORAGE_DISK_MANAGER

#include "storage/storage_types.h"

#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

/**
 * @brief Manage four fixed-size chunk files under one storage prefix.
 */
class disk_manager_t {
public:
  /**
   * @brief Chunk Level Count
   */
  static constexpr uint32_t k_chunk_levels = 4;

  /**
   * @brief Bit Count for Chunk Type
   */
  static constexpr uint32_t k_chunk_type_bits = 2;

  /**
   * @brief Bit Count for Chunk Number
   */
  static constexpr uint32_t k_chunk_no_bits = 62;

  /**
   * @brief Mask for extracting chunk number from ChunkID
   */
  static constexpr chunk_id_t k_chunk_no_mask = (chunk_id_t{1} << k_chunk_no_bits) - 1;

  /**
   * @brief Chunk Sizes
   */
  static constexpr std::array<uint32_t, k_chunk_levels> k_chunk_sizes = {
      4 * 1024,
      64 * 1024,
      512 * 1024,
      2 * 1024 * 1024,
  };

  /**
   * @brief Construct a closed disk manager.
   */
  disk_manager_t() = default;

  /**
   * @brief Open a disk manager with the given file prefix.
   * @param prefix Common path prefix for meta and chunk files.
   * @param options Open behavior.
   */
  explicit disk_manager_t(const std::string& prefix, const disk_manager_options_t& options = {});

  /**
   * @brief Close open file handles.
   */
  ~disk_manager_t();

  disk_manager_t(const disk_manager_t&) = delete;
  disk_manager_t& operator=(const disk_manager_t&) = delete;

  /**
   * @brief Move construct a disk manager.
   * @param other Source disk manager.
   */
  disk_manager_t(disk_manager_t&& other) noexcept;

  /**
   * @brief Move assign a disk manager.
   * @param other Source disk manager.
   * @return This disk manager.
   */
  disk_manager_t& operator=(disk_manager_t&& other) noexcept;

  /**
   * @brief Open a disk manager with the given file prefix.
   * @param prefix Common path prefix for meta and chunk files.
   * @param options Open behavior.
   */
  void open(const std::string& prefix, const disk_manager_options_t& options = {});

  /**
   * @brief Flush metadata and close all files.
   */
  void close();

  /**
   * @brief Check whether all files are open.
   * @return True if the manager is ready for I/O.
   */
  bool is_open() const;

  /**
   * @brief Allocate one chunk page from a chunk file.
   * @param type Chunk file type.
   * @return Encoded chunk id.
   */
  chunk_id_t allocate_chunk(chunk_type_t type);

  /**
   * @brief Release one chunk page for later reuse.
   * @param chunk_id Encoded chunk id.
   */
  void delete_chunk(chunk_id_t chunk_id);

  /**
   * @brief Write an entire chunk page.
   * @param chunk_id Encoded chunk id.
   * @param data Source buffer.
   * @param size Number of bytes to write.
   */
  void write_chunk(chunk_id_t chunk_id, const void* data, size_t size);

  /**
   * @brief Allocate and write one chunk page.
   * @param type Chunk file type.
   * @param data Source buffer.
   * @param size Number of bytes to write.
   * @return Encoded chunk id.
   */
  chunk_id_t write_new_chunk(chunk_type_t type, const void* data, size_t size);

  /**
   * @brief Load a chunk page into a caller-owned buffer.
   * @param chunk_id Encoded chunk id.
   * @param dst Destination pointer with at least chunk_size bytes.
   */
  void load_chunk(chunk_id_t chunk_id, void* dst);

  /**
   * @brief Get allocated chunk count for a chunk file.
   * @param type Chunk file type.
   * @return Total chunk slots including deleted slots.
   */
  uint64_t chunk_count(chunk_type_t type) const;

  /**
   * @brief Get reusable chunk count for a chunk file.
   * @param type Chunk file type.
   * @return Number of deleted chunk slots available for reuse.
   */
  uint64_t free_chunk_count(chunk_type_t type) const;

  /**
   * @brief Get allocated chunk counts for all chunk files.
   * @return Count array ordered by chunk_type_t value.
   */
  const std::array<uint64_t, k_chunk_levels>& chunk_counts() const { return meta_.chunk_counts; }

  /**
   * @brief Get reusable chunk counts for all chunk files.
   * @return Count array ordered by chunk_type_t value.
   */
  const std::array<uint64_t, k_chunk_levels>& free_chunk_counts() const {
    return meta_.free_counts;
  }

  /**
   * @brief Get opened file paths.
   * @return File path group.
   */
  const disk_manager_files_t& files() const { return files_; }

  /**
   * @brief Get active metadata snapshot.
   * @return Const metadata reference.
   */
  const disk_manager_meta_t& meta() const { return meta_; }

  /**
   * @brief Check whether 2MB huge page hints are active.
   * @return True if huge page support is active for chunk_2mb.
   */
  bool huge_page_enabled() const { return meta_.huge_page_enabled != 0; }

  /**
   * @brief Encode chunk type and chunk number into chunk_id_t.
   * @param type Chunk file type.
   * @param chunk_no Internal chunk number within that chunk file.
   * @return Encoded chunk id.
   */
  static chunk_id_t make_chunk_id(chunk_type_t type, chunk_no_t chunk_no);

  /**
   * @brief Decode a chunk id into physical address information.
   * @param chunk_id Encoded chunk id.
   * @return Decoded address.
   */
  static page_addr_t decode_chunk_id(chunk_id_t chunk_id);

  /**
   * @brief Extract chunk type from a chunk id.
   * @param chunk_id Encoded chunk id.
   * @return Chunk file type.
   */
  static chunk_type_t chunk_type(chunk_id_t chunk_id);

  /**
   * @brief Extract chunk number from a chunk id.
   * @param chunk_id Encoded chunk id.
   * @return Internal chunk number within the chunk file.
   */
  static chunk_no_t chunk_no(chunk_id_t chunk_id);

  /**
   * @brief Get chunk byte size for a type.
   * @param type Chunk file type.
   * @return Chunk byte size.
   */
  static uint32_t chunk_size(chunk_type_t type);

  /**
   * @brief Build file names from a common prefix.
   * @param prefix Common file prefix.
   * @return Meta and chunk file paths.
   */
  static disk_manager_files_t build_file_names(const std::string& prefix);

private:
  /**
   * @brief Prefix for this group of files
   *        Represents the name of this vector index
   */
  std::string prefix_;

  /**
   * @brief File paths for this group of files
   */
  disk_manager_files_t files_;

  /**
   * @brief Meta data of this group of files
   *
   */
  disk_manager_meta_t meta_;

  /**
   * @brief Options for this disk manager instance
   */
  disk_manager_options_t options_;

  /**
   * @brief Meta data file
   */
  std::fstream meta_file_;

  /**
   * @brief Chunk files for each chunk type
   */
  std::array<std::fstream, k_chunk_levels> chunk_files_;

  /**
   * @brief Positional-read descriptors for concurrent chunk loading.
   */
  std::array<int, k_chunk_levels> chunk_read_fds_{{-1, -1, -1, -1}};

  /**
   * @brief Slots for reuse in each chunk file
   */
  std::array<std::vector<chunk_no_t>, k_chunk_levels> free_lists_;

  /**
   * @brief Load or initialize metadata.
   */
  void load_meta();

  /**
   * @brief Persist metadata to the meta file.
   */
  void flush_meta();

  /**
   * @brief Open one chunk file.
   * @param type Chunk file type.
   */
  void open_chunk_file(chunk_type_t type);

  /**
   * @brief Apply OS-specific huge page hints when available.
   */
  void configure_huge_page_hint();

  /**
   * @brief Check whether a page id points to an allocated slot.
   * @param chunk_id Encoded chunk id.
   */
  void validate_chunk_id(chunk_id_t chunk_id) const;
};

#endif // STORAGE_DISK_MANAGER
