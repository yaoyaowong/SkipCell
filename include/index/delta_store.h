#ifndef INDEX_DELTA_STORE
#define INDEX_DELTA_STORE

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace powerlaw_ann {

class hnsw_index_t;

/**
 * @brief One delta record.
 */
struct delta_record_t {
  /**
   * @brief Delta operation type.
   */
  enum class op_t : uint8_t { insert = 1 };

  op_t op;
  uint64_t label;
  std::vector<float> vec;
};

/**
 * @brief Buffer and replay delta records.
 */
class delta_store_t {
public:
  /**
   * @brief Construct delta store.
   * @param dim Vector dimension.
   * @param max_buffer Max buffered records.
   */
  delta_store_t(int dim, size_t max_buffer = 512);

  /**
   * @brief Append one insert record.
   * @param vec Pointer to vector data.
   * @param label External label.
   */
  void add(const float* vec, uint64_t label);

  /**
   * @brief Check whether buffer is full.
   * @return True if current size >= max_buffer.
   */
  bool is_full() const;

  /**
   * @brief Check whether buffer is empty.
   * @return True if buffer is empty.
   */
  bool empty() const;

  /**
   * @brief Get buffered record count.
   * @return Number of buffered records.
   */
  size_t size() const;

  /**
   * @brief Flush buffered records to delta file.
   * @param path Delta file path.
   */
  void flush_to_file(const std::string& path);

  /**
   * @brief Apply buffered records to index and clear buffer.
   * @param index Destination index.
   */
  void apply_and_clear(hnsw_index_t& index);

  /**
   * @brief Replay a delta file into index.
   * @param path Delta file path.
   * @param index Destination index.
   */
  static void replay_file(const std::string& path, hnsw_index_t& index);

  /**
   * @brief Get read-only record view.
   * @return Const reference to internal record storage.
   */
  const std::vector<delta_record_t>& records() const { return records_; }

private:
  int dim_;
  size_t max_buffer_;
  mutable std::mutex mu_;
  std::vector<delta_record_t> records_;
};

} // namespace powerlaw_ann

#endif // INDEX_DELTA_STORE
