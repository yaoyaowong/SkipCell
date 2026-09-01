#ifndef STORAGE_STORAGE_TYPES
#define STORAGE_STORAGE_TYPES

#include <array>
#include <cstdint>
#include <string>

/**
 * @brief Encoded identifier for a chunk page.
 */
using chunk_id_t = uint64_t;

/**
 * @brief Page number inside one chunk file.
 */
using chunk_no_t = uint64_t;

/**
 * @brief Chunk file type encoded in the high bits of chunk_id_t.
 */
enum class chunk_type_t : uint8_t {
  chunk_4kb = 0,
  chunk_64kb = 1,
  chunk_512kb = 2,
  chunk_2mb = 3,
};

/**
 * @brief Runtime policy for 2MB huge page hints.
 */
enum class huge_page_policy_t : uint8_t {
  disabled = 0,
  prefer = 1,
  require = 2,
};

/**
 * @brief File paths opened by one disk manager prefix.
 */
struct disk_manager_files_t {
  std::string meta;
  std::array<std::string, 4> chunks;
};

/**
 * @brief Decoded physical chunk address.
 */
struct page_addr_t {
  chunk_type_t type;
  chunk_no_t chunk_no;
  uint64_t offset;
  uint32_t size;
};

/**
 * @brief Persistent metadata for one chunk file group.
 */
struct disk_manager_meta_t {
  static constexpr uint64_t k_magic = 0x5247434D'44574342ULL;
  static constexpr uint32_t k_version = 1;
  static constexpr uint32_t k_chunk_levels = 4;

  uint64_t magic = k_magic;
  uint32_t version = k_version;
  uint32_t chunk_levels = k_chunk_levels;
  std::array<uint64_t, 4> chunk_counts{};
  std::array<uint64_t, 4> free_counts{};
  uint8_t huge_page_enabled = 0;
};

/**
 * @brief Options for opening a chunk disk manager.
 */
struct disk_manager_options_t {
  bool create_if_missing = true;
  bool truncate = false;
  bool sync_on_write = false;
  huge_page_policy_t huge_page_policy = huge_page_policy_t::prefer;
};

#endif // STORAGE_STORAGE_TYPES
