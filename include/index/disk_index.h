#ifndef INDEX_DISK_INDEX
#define INDEX_DISK_INDEX

#include <cstdint>
#include <string>

namespace powerlaw_ann {

class hnsw_index_t;

/**
 * @brief Save/load HNSW index from a single DB file.
 */
class disk_index_t {
public:
  static constexpr uint64_t k_magic = 0x31524F54'43455742ULL;
  static constexpr uint32_t k_version = 1;

  /**
   * @brief Save index to disk.
   * @param index Source index.
   * @param path Target file path.
   */
  static void save(const hnsw_index_t& index, const std::string& path);

  /**
   * @brief Load index from disk.
   * @param index Destination index.
   * @param path Source file path.
   */
  static void load(hnsw_index_t& index, const std::string& path);

  /**
   * @brief Check file magic header.
   * @param path File path.
   * @return True if file exists and magic matches.
   */
  static bool exists(const std::string& path);
};

} // namespace powerlaw_ann

#endif // INDEX_DISK_INDEX
