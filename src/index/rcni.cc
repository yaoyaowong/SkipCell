#include "index/rcni.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

namespace powerlaw_ann {
namespace {

struct rcni_credit_entry_t {
  uint32_t witness_id = 0;
  uint32_t victim_id = 0;
  float event_credit = 0.0F;
};

struct rcni_source_contribution_t {
  uint32_t source_id = 0;
  uint32_t witness_id = 0;
  float source_saturation = 0.0F;
  uint64_t event_count = 0;
};

struct rcni_aggregation_scratch_t {
  std::vector<rcni_credit_entry_t> credits;
  std::vector<rcni_source_contribution_t> contributions;
};

thread_local rcni_aggregation_scratch_t rcni_aggregation_scratch;

float validate_and_convert_distance(float distance, rcni_distance_semantics_t semantics) {
  if (!std::isfinite(distance) || distance < 0.0F) {
    throw std::invalid_argument("RCNI distances must be finite and non-negative");
  }

  switch (semantics) {
  case rcni_distance_semantics_t::TRUE_L2:
    return distance;
  case rcni_distance_semantics_t::SQUARED_L2:
    return std::sqrt(distance);
  }

  throw std::invalid_argument("RCNI distance semantics are invalid");
}

float clamp_unit(double value) { return static_cast<float>(std::clamp(value, 0.0, 1.0)); }

double safe_ratio(float numerator, float denominator) {
  const double ratio = static_cast<double>(numerator) / static_cast<double>(denominator);
  return std::isfinite(ratio) ? ratio : std::numeric_limits<double>::max();
}

template <typename iterator_t, typename credit_fn_t>
float compute_sorted_rcni_source_saturation(iterator_t begin, iterator_t end,
                                            credit_fn_t get_credit) {
  long double log_survival = 0.0L;
  for (auto current = begin; current != end; ++current) {
    const float credit = get_credit(*current);
    if (credit == 1.0F) {
      return 1.0F;
    }
    log_survival += std::log1p(-static_cast<long double>(credit));
  }

  return clamp_unit(static_cast<double>(-std::expm1(log_survival)));
}

} // namespace

void update_rcni_causal_witness(rcni_causal_witness_t& state, uint32_t witness_id,
                                float witness_victim_distance, float occlusion_factor) {
  if (!state.witness_id.has_value() || occlusion_factor > state.occlusion_factor ||
      (occlusion_factor == state.occlusion_factor && witness_id < *state.witness_id)) {
    state.witness_id = witness_id;
    state.witness_victim_distance = witness_victim_distance;
    state.occlusion_factor = occlusion_factor;
  }
}

rcni_event_credit_t compute_rcni_event_credit(const rcni_event_input_t& input) {
  if (!std::isfinite(input.alpha) || input.alpha < 1.0F) {
    throw std::invalid_argument("RCNI alpha must be finite and greater than or equal to one");
  }
  if (!std::isfinite(input.epsilon) || input.epsilon <= 0.0F) {
    throw std::invalid_argument("RCNI epsilon must be finite and greater than zero");
  }

  const float source_distance =
      validate_and_convert_distance(input.source_victim_distance, input.distance_semantics);
  const float witness_distance =
      validate_and_convert_distance(input.witness_victim_distance, input.distance_semantics);

  std::optional<rcni_neighbor_distance_t> alternative;
  float alternative_distance = 0.0F;
  for (const auto& neighbor : input.selected_neighbors) {
    const float distance =
        validate_and_convert_distance(neighbor.victim_distance, input.distance_semantics);
    if (neighbor.node_id == input.witness_id) {
      continue;
    }

    if (!alternative.has_value() || distance < alternative_distance ||
        (distance == alternative_distance && neighbor.node_id < alternative->node_id)) {
      alternative = neighbor;
      alternative_distance = distance;
    }
  }

  rcni_event_credit_t result;
  if (alternative.has_value()) {
    result.alternative_id = alternative->node_id;
  }

  // A zero source-to-victim distance provides no meaningful contraction baseline.
  if (source_distance <= input.epsilon) {
    return result;
  }

  const double witness_ratio = safe_ratio(witness_distance, source_distance);
  result.witness_ratio = static_cast<float>(
      std::min(witness_ratio, static_cast<double>(std::numeric_limits<float>::max())));

  double counterfactual_ratio = 1.0;
  if (alternative.has_value()) {
    counterfactual_ratio = safe_ratio(alternative_distance, source_distance);
    counterfactual_ratio = std::max(counterfactual_ratio, witness_ratio);
    counterfactual_ratio = std::min(counterfactual_ratio, 1.0);
  }
  result.counterfactual_ratio = static_cast<float>(counterfactual_ratio);

  const double scaled_witness_ratio = static_cast<double>(input.alpha) * witness_ratio;
  const double certified_progress = scaled_witness_ratio >= 1.0 ? 0.0 : 1.0 - scaled_witness_ratio;

  double counterfactual_irreplaceability = 0.0;
  if (counterfactual_ratio > static_cast<double>(input.epsilon) &&
      witness_ratio < counterfactual_ratio) {
    counterfactual_irreplaceability = 1.0 - witness_ratio / counterfactual_ratio;
  }

  result.certified_progress = clamp_unit(certified_progress);
  result.counterfactual_irreplaceability = clamp_unit(counterfactual_irreplaceability);
  result.event_credit = clamp_unit(static_cast<double>(result.certified_progress) *
                                   result.counterfactual_irreplaceability);
  return result;
}

float compute_rcni_source_saturation(std::span<const float> event_credits) {
  std::vector<float> sorted_credits;
  sorted_credits.reserve(event_credits.size());

  for (const float credit : event_credits) {
    if (!std::isfinite(credit) || credit < -k_rcni_default_epsilon ||
        credit > 1.0F + k_rcni_default_epsilon) {
      throw std::invalid_argument("RCNI event credits must be finite and within [0, 1]");
    }
    sorted_credits.push_back(std::clamp(credit, 0.0F, 1.0F));
  }

  std::sort(sorted_credits.begin(), sorted_credits.end());
  return compute_sorted_rcni_source_saturation(sorted_credits.begin(), sorted_credits.end(),
                                               [](float credit) { return credit; });
}

void validate_candidate_hub_selection_config(const candidate_hub_selection_config_t& config) {
  double active_value = 0.0;
  switch (config.mode) {
  case candidate_hub_selection_mode_t::TOP_RATIO:
    active_value = config.top_ratio;
    break;
  case candidate_hub_selection_mode_t::IMPORTANCE_MASS:
    active_value = config.importance_mass;
    break;
  default:
    throw std::invalid_argument("candidate-Hub selection mode is invalid");
  }

  if (!std::isfinite(active_value) || active_value < 0.0 || active_value > 1.0) {
    throw std::invalid_argument("candidate-Hub selection value must be finite and within [0, 1]");
  }
}

candidate_hub_selection_result_t
select_candidate_hubs(std::span<const rcni_node_importance_t> records,
                      const candidate_hub_selection_config_t& config) {
  validate_candidate_hub_selection_config(config);

  candidate_hub_selection_result_t result;
  result.mode = config.mode;
  result.selection_value = config.mode == candidate_hub_selection_mode_t::TOP_RATIO
                               ? config.top_ratio
                               : config.importance_mass;

  std::vector<rcni_node_importance_t> ranked(records.begin(), records.end());
  std::vector<uint32_t> node_ids;
  node_ids.reserve(ranked.size());
  for (const auto& record : ranked) {
    if (!std::isfinite(record.raw_importance) || record.raw_importance < 0.0 ||
        !std::isfinite(record.normalized_importance) || record.normalized_importance < 0.0 ||
        record.normalized_importance > 1.0) {
      throw std::invalid_argument(
          "candidate-Hub importance must be finite, non-negative, and normalized within [0, 1]");
    }
    node_ids.push_back(record.node_id);
  }
  std::sort(node_ids.begin(), node_ids.end());
  if (std::adjacent_find(node_ids.begin(), node_ids.end()) != node_ids.end()) {
    throw std::invalid_argument("candidate-Hub importance contains duplicate node IDs");
  }

  std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
    if (left.normalized_importance != right.normalized_importance) {
      return left.normalized_importance > right.normalized_importance;
    }
    return left.node_id < right.node_id;
  });

  long double total_importance = 0.0L;
  for (const auto& record : ranked) {
    total_importance += static_cast<long double>(record.normalized_importance);
  }
  result.total_importance = static_cast<double>(total_importance);
  if (ranked.empty() || result.selection_value == 0.0 || total_importance == 0.0L) {
    return result;
  }

  size_t selected_count = 0;
  if (config.mode == candidate_hub_selection_mode_t::TOP_RATIO) {
    const long double requested_count =
        static_cast<long double>(ranked.size()) * static_cast<long double>(config.top_ratio);
    selected_count = static_cast<size_t>(std::ceil(requested_count));
    selected_count = std::min(selected_count, ranked.size());
  }

  long double selected_importance = 0.0L;
  result.candidate_hubs.reserve(
      config.mode == candidate_hub_selection_mode_t::TOP_RATIO ? selected_count : ranked.size());
  for (size_t rank = 0; rank < ranked.size(); ++rank) {
    if (config.mode == candidate_hub_selection_mode_t::TOP_RATIO && rank >= selected_count) {
      break;
    }

    const auto& record = ranked[rank];
    selected_importance += static_cast<long double>(record.normalized_importance);
    result.candidate_hubs.push_back(
        candidate_hub_t{record.node_id, record.raw_importance, record.normalized_importance,
                        static_cast<double>(selected_importance / total_importance)});

    if (config.mode == candidate_hub_selection_mode_t::IMPORTANCE_MASS &&
        selected_importance >=
            static_cast<long double>(config.importance_mass) * total_importance) {
      break;
    }
  }

  result.selected_importance = static_cast<double>(selected_importance);
  result.selected_importance_mass = static_cast<double>(selected_importance / total_importance);
  return result;
}

class rcni_aggregator_t::impl_t {
public:
  struct shard_t {
    mutable std::mutex mutex;
    std::vector<rcni_source_contribution_t> contributions;
  };

  impl_t(float alpha, float epsilon) : alpha_(alpha), epsilon_(epsilon) {
    if (!std::isfinite(alpha_) || alpha_ < 1.0F) {
      throw std::invalid_argument("RCNI alpha must be finite and greater than or equal to one");
    }
    if (!std::isfinite(epsilon_) || epsilon_ <= 0.0F) {
      throw std::invalid_argument("RCNI epsilon must be finite and greater than zero");
    }
  }

  void prepare(size_t source_count) {
    if (source_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::invalid_argument("RCNI source count exceeds the uint32 node-ID range");
    }

    source_count_ = source_count;
    observed_sources_ = std::make_unique<std::atomic_bool[]>(source_count_);
    for (size_t source = 0; source < source_count_; ++source) {
      observed_sources_[source].store(false, std::memory_order_relaxed);
    }
    const size_t available_workers =
        std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
    shard_count_ = std::max<size_t>(1, std::min(source_count_, available_workers));
    shards_ = std::make_unique<shard_t[]>(shard_count_);
    prepared_ = true;
  }

  void observe_source(uint32_t source_id, std::span<const rcni_prune_event_t> events) {
    if (!prepared_) {
      throw std::logic_error("RCNI aggregator must be prepared before observing sources");
    }
    if (source_id >= source_count_) {
      throw std::out_of_range("RCNI source ID is outside the prepared node range");
    }
    if (observed_sources_[source_id].exchange(true, std::memory_order_relaxed)) {
      throw std::logic_error("RCNI source was observed more than once");
    }

    auto& scratch = rcni_aggregation_scratch;
    scratch.credits.clear();
    if (scratch.credits.capacity() < events.size()) {
      scratch.credits.reserve(events.size());
    }

    for (const auto& event : events) {
      if (event.source_id != source_id) {
        throw std::invalid_argument("RCNI event source does not match its callback source");
      }
      if (event.witness_id >= source_count_ || event.victim_id >= source_count_) {
        throw std::out_of_range("RCNI event node ID is outside the prepared node range");
      }

      std::span<const rcni_neighbor_distance_t> alternative;
      if (event.alternative.has_value()) {
        if (event.alternative->node_id >= source_count_) {
          throw std::out_of_range("RCNI alternative ID is outside the prepared node range");
        }
        alternative = std::span<const rcni_neighbor_distance_t>(&*event.alternative, 1);
      }

      rcni_event_input_t input;
      input.witness_id = event.witness_id;
      input.source_victim_distance = event.source_victim_distance;
      input.witness_victim_distance = event.witness_victim_distance;
      input.selected_neighbors = alternative;
      input.alpha = alpha_;
      input.epsilon = epsilon_;
      input.distance_semantics = rcni_distance_semantics_t::SQUARED_L2;
      const auto credit = compute_rcni_event_credit(input);
      scratch.credits.push_back(
          rcni_credit_entry_t{event.witness_id, event.victim_id, credit.event_credit});
    }

    std::sort(scratch.credits.begin(), scratch.credits.end(),
              [](const auto& left, const auto& right) {
                return std::tie(left.witness_id, left.event_credit, left.victim_id) <
                       std::tie(right.witness_id, right.event_credit, right.victim_id);
              });

    scratch.contributions.clear();
    if (scratch.contributions.capacity() < scratch.credits.size()) {
      scratch.contributions.reserve(scratch.credits.size());
    }
    auto group_begin = scratch.credits.begin();
    while (group_begin != scratch.credits.end()) {
      const uint32_t witness_id = group_begin->witness_id;
      const auto group_end =
          std::find_if(group_begin, scratch.credits.end(),
                       [witness_id](const auto& entry) { return entry.witness_id != witness_id; });
      const float saturation = compute_sorted_rcni_source_saturation(
          group_begin, group_end, [](const auto& entry) { return entry.event_credit; });
      scratch.contributions.push_back(
          rcni_source_contribution_t{source_id, witness_id, saturation,
                                     static_cast<uint64_t>(std::distance(group_begin, group_end))});
      group_begin = group_end;
    }

    auto& shard = shards_[source_id % shard_count_];
    std::lock_guard<std::mutex> guard(shard.mutex);
    shard.contributions.insert(shard.contributions.end(), scratch.contributions.begin(),
                               scratch.contributions.end());
  }

  std::vector<rcni_node_importance_t> compute_importance() const {
    if (!prepared_) {
      throw std::logic_error("RCNI aggregator must be prepared before computing importance");
    }

    size_t contribution_count = 0;
    for (size_t shard = 0; shard < shard_count_; ++shard) {
      std::lock_guard<std::mutex> guard(shards_[shard].mutex);
      contribution_count += shards_[shard].contributions.size();
    }

    std::vector<rcni_source_contribution_t> contributions;
    contributions.reserve(contribution_count);
    for (size_t shard = 0; shard < shard_count_; ++shard) {
      std::lock_guard<std::mutex> guard(shards_[shard].mutex);
      contributions.insert(contributions.end(), shards_[shard].contributions.begin(),
                           shards_[shard].contributions.end());
    }
    std::sort(contributions.begin(), contributions.end(), [](const auto& left, const auto& right) {
      return std::tie(left.source_id, left.witness_id) <
             std::tie(right.source_id, right.witness_id);
    });

    std::vector<rcni_node_importance_t> importance(source_count_);
    for (size_t node = 0; node < source_count_; ++node) {
      importance[node].node_id = static_cast<uint32_t>(node);
    }
    for (const auto& contribution : contributions) {
      auto& node = importance[contribution.witness_id];
      node.raw_importance += static_cast<double>(contribution.source_saturation);
      node.witness_event_count += contribution.event_count;
      if (contribution.source_saturation > 0.0F) {
        ++node.source_support;
      }
    }

    const auto maximum = std::max_element(importance.begin(), importance.end(),
                                          [](const auto& left, const auto& right) {
                                            return left.raw_importance < right.raw_importance;
                                          });
    const double maximum_importance = maximum == importance.end() ? 0.0 : maximum->raw_importance;
    if (maximum_importance > 0.0) {
      const double denominator = maximum_importance + static_cast<double>(epsilon_);
      for (auto& node : importance) {
        node.normalized_importance = node.raw_importance / denominator;
      }

      std::vector<uint32_t> ordered_nodes(source_count_);
      std::iota(ordered_nodes.begin(), ordered_nodes.end(), 0U);
      std::sort(ordered_nodes.begin(), ordered_nodes.end(), [&](uint32_t left, uint32_t right) {
        return std::tie(importance[left].raw_importance, left) <
               std::tie(importance[right].raw_importance, right);
      });
      size_t group_begin = 0;
      while (group_begin < ordered_nodes.size()) {
        size_t group_end = group_begin + 1;
        const double value = importance[ordered_nodes[group_begin]].raw_importance;
        while (group_end < ordered_nodes.size() &&
               importance[ordered_nodes[group_end]].raw_importance == value) {
          ++group_end;
        }
        const double percentile =
            static_cast<double>(group_end) / static_cast<double>(ordered_nodes.size());
        for (size_t position = group_begin; position < group_end; ++position) {
          importance[ordered_nodes[position]].importance_percentile = percentile;
        }
        group_begin = group_end;
      }
    }

    return importance;
  }

private:
  float alpha_;
  float epsilon_;
  size_t source_count_ = 0;
  size_t shard_count_ = 0;
  bool prepared_ = false;
  std::unique_ptr<std::atomic_bool[]> observed_sources_;
  std::unique_ptr<shard_t[]> shards_;
};

rcni_aggregator_t::rcni_aggregator_t(float alpha, float epsilon)
    : impl_(std::make_unique<impl_t>(alpha, epsilon)) {}

rcni_aggregator_t::~rcni_aggregator_t() = default;

void rcni_aggregator_t::prepare(size_t source_count) { impl_->prepare(source_count); }

void rcni_aggregator_t::observe_source(uint32_t source_id,
                                       std::span<const rcni_prune_event_t> events,
                                       std::span<const uint32_t> selected_neighbors) {
  static_cast<void>(selected_neighbors);
  impl_->observe_source(source_id, events);
}

std::vector<rcni_node_importance_t> rcni_aggregator_t::compute_importance() const {
  return impl_->compute_importance();
}

void write_rcni_importance_csv(const std::filesystem::path& path,
                               std::span<const rcni_node_importance_t> records) {
  std::filesystem::path temporary_path = path;
  temporary_path += ".tmp";

  std::ofstream output(temporary_path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open RCNI output: " + temporary_path.string());
  }
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "node_id,raw_importance,normalized_importance,importance_percentile,source_support,"
            "witness_event_count\n";
  for (const auto& record : records) {
    output << record.node_id << ',' << record.raw_importance << ',' << record.normalized_importance
           << ',' << record.importance_percentile << ',' << record.source_support << ','
           << record.witness_event_count << '\n';
  }
  output.flush();
  if (!output) {
    output.close();
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::runtime_error("failed to write RCNI output: " + temporary_path.string());
  }
  output.close();

  std::error_code rename_error;
  std::filesystem::rename(temporary_path, path, rename_error);
  if (rename_error) {
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::runtime_error("failed to publish RCNI output: " + rename_error.message());
  }
}

void write_candidate_hubs_csv(const std::filesystem::path& path,
                              const candidate_hub_selection_result_t& result) {
  std::filesystem::path temporary_path = path;
  temporary_path += ".tmp";

  std::ofstream output(temporary_path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open candidate-Hub output: " + temporary_path.string());
  }
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "selection_mode,selection_value,total_importance,selected_importance,"
            "selected_importance_mass,rank,node_id,raw_importance,normalized_importance,"
            "cumulative_importance_mass\n";

  const char* mode_name = nullptr;
  switch (result.mode) {
  case candidate_hub_selection_mode_t::TOP_RATIO:
    mode_name = "top_ratio";
    break;
  case candidate_hub_selection_mode_t::IMPORTANCE_MASS:
    mode_name = "importance_mass";
    break;
  default:
    output.close();
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::invalid_argument("candidate-Hub result mode is invalid");
  }

  for (size_t rank = 0; rank < result.candidate_hubs.size(); ++rank) {
    const auto& hub = result.candidate_hubs[rank];
    output << mode_name << ',' << result.selection_value << ',' << result.total_importance << ','
           << result.selected_importance << ',' << result.selected_importance_mass << ','
           << rank + 1 << ',' << hub.node_id << ',' << hub.raw_importance << ','
           << hub.normalized_importance << ',' << hub.cumulative_importance_mass << '\n';
  }
  output.flush();
  if (!output) {
    output.close();
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::runtime_error("failed to write candidate-Hub output: " + temporary_path.string());
  }
  output.close();

  std::error_code rename_error;
  std::filesystem::rename(temporary_path, path, rename_error);
  if (rename_error) {
    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    throw std::runtime_error("failed to publish candidate-Hub output: " + rename_error.message());
  }
}

} // namespace powerlaw_ann
