#ifndef INDEX_POWERLAW_ANN
#define INDEX_POWERLAW_ANN

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace powerlaw_ann {
class hnsw_index_t;
class delta_store_t;
} // namespace powerlaw_ann

/**
 * @brief Runtime configuration for powerlaw_ann_t.
 */
struct powerlaw_ann_config_t {
  int dim = 128;
  int m = 16;
  int ef_construction = 200;
  int ef_search = 50;
  size_t max_elements = 100'000;
  size_t delta_buffer = 512;
  std::string workspace = "./workspace";
  std::string dataset_name = "default";
};

/**
 * @brief Main API entry for vector indexing.
 */
struct powerlaw_ann_t {
  using config_t = powerlaw_ann_config_t;

  /**
   * @brief Construct a powerlaw_ann instance.
   * @param cfg Runtime configuration.
   */
  explicit powerlaw_ann_t(const config_t& cfg = config_t{});

  /**
   * @brief Destroy the instance and stop background tasks.
   */
  ~powerlaw_ann_t();

  powerlaw_ann_t(const powerlaw_ann_t&) = delete;
  powerlaw_ann_t& operator=(const powerlaw_ann_t&) = delete;

  /**
   * @brief Insert one vector with an external label.
   * @param vec Pointer to vector data.
   * @param label External label.
   */
  void insert(const float* vec, uint64_t label);

  /**
   * @brief Run ANN search.
   * @param query Pointer to query vector.
   * @param k Number of results.
   * @return Results sorted by ascending distance.
   */
  std::vector<std::pair<float, uint64_t>> search(const float* query, int k) const;

  /**
   * @brief Bulk build from a row-major matrix.
   * @param data Pointer to row-major matrix.
   * @param labels Pointer to labels.
   * @param n Number of vectors.
   */
  void build(const float* data, const uint64_t* labels, size_t n);

  /**
   * @brief Persist in-memory index.
   */
  void save();

  /**
   * @brief Load index and replay delta file if present.
   */
  void load();

  /**
   * @brief Apply buffered deltas into the main index.
   */
  void consolidate();

  /**
   * @brief Get total element count including buffered deltas.
   * @return Total element count.
   */
  size_t size() const;

  /**
   * @brief Get DB file path.
   * @return Absolute/relative DB path from config.
   */
  std::string db_path() const;

  /**
   * @brief Get delta file path.
   * @return Absolute/relative delta path from config.
   */
  std::string delta_path() const;

private:
  config_t cfg_;
  std::unique_ptr<powerlaw_ann::hnsw_index_t> index_;
  std::unique_ptr<powerlaw_ann::delta_store_t> delta_;

  std::thread bg_thread_;
  std::atomic<bool> stop_bg_{false};
  mutable std::mutex bg_mu_;
  std::condition_variable bg_cv_;

  /**
   * @brief Start background consolidation thread.
   */
  void start_bg();

  /**
   * @brief Background loop for flushing and merging deltas.
   */
  void bg_loop();

  /**
   * @brief Ensure workspace directory exists.
   */
  void ensure_workspace() const;
};

#endif // INDEX_POWERLAW_ANN
