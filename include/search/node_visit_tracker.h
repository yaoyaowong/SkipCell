#ifndef SEARCH_NODE_VISIT_TRACKER
#define SEARCH_NODE_VISIT_TRACKER

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace powerlaw_ann {

/**
 * @brief Counts nodes selected for expansion by concurrent beam searches.
 *
 * Start, stop, and dump operations must run outside active searches. `record()`
 * is lock-free and safe for concurrent query workers.
 */
class node_visit_tracker_t {
public:
  node_visit_tracker_t() = default;
  node_visit_tracker_t(const node_visit_tracker_t&) = delete;
  node_visit_tracker_t& operator=(const node_visit_tracker_t&) = delete;

  void start(size_t num_nodes) {
    stop();
    auto counters = std::make_unique<std::atomic<uint64_t>[]>(num_nodes);
    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
      counters[node_id].store(0, std::memory_order_relaxed);
    }
    counters_ = std::move(counters);
    num_nodes_ = num_nodes;
    enabled_.store(true, std::memory_order_release);
  }

  void stop() noexcept { enabled_.store(false, std::memory_order_release); }

  bool is_enabled() const noexcept { return enabled_.load(std::memory_order_acquire); }

  void record(uint64_t node_id) noexcept {
    if (!is_enabled() || node_id >= num_nodes_) {
      return;
    }
    counters_[node_id].fetch_add(1, std::memory_order_relaxed);
  }

  std::vector<uint64_t> snapshot() const {
    std::vector<uint64_t> counts(num_nodes_);
    for (size_t node_id = 0; node_id < num_nodes_; ++node_id) {
      counts[node_id] = counters_[node_id].load(std::memory_order_relaxed);
    }
    return counts;
  }

  void dump_csv(const std::string& output_path) const {
    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("Unable to open node visit output file: " + output_path);
    }
    output << "node_id,visit_count\n";
    for (size_t node_id = 0; node_id < num_nodes_; ++node_id) {
      output << node_id << ',' << counters_[node_id].load(std::memory_order_relaxed) << '\n';
    }
    if (!output) {
      throw std::runtime_error("Unable to write node visit output file: " + output_path);
    }
  }

private:
  std::unique_ptr<std::atomic<uint64_t>[]> counters_;
  size_t num_nodes_ = 0;
  std::atomic<bool> enabled_{false};
};

} // namespace powerlaw_ann

#endif // SEARCH_NODE_VISIT_TRACKER
