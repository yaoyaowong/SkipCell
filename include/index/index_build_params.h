#ifndef INDEX_INDEX_BUILD_PARAMS
#define INDEX_INDEX_BUILD_PARAMS

#include "common/diskann_exception.h"

#include <cstdint>
#include <string>

namespace powerlaw_ann {
struct index_filter_params_t {
public:
  std::string save_path_prefix;
  std::string label_file;
  std::string tags_file;
  std::string universal_label;
  uint32_t filter_threshold = 0;

private:
  index_filter_params_t(const std::string& save_path_prefix, const std::string& label_file,
                        const std::string& universal_label, uint32_t filter_threshold)
      : save_path_prefix(save_path_prefix), label_file(label_file),
        universal_label(universal_label), filter_threshold(filter_threshold) {}

  friend class index_filter_params_builder_t;
};
class index_filter_params_builder_t {
public:
  index_filter_params_builder_t() = default;

  index_filter_params_builder_t& with_save_path_prefix(const std::string& save_path_prefix) {
    if (save_path_prefix.empty() || save_path_prefix == "")
      throw diskann_exception_t("Error: save_path_prefix can't be empty", -1);
    this->save_path_prefix_ = save_path_prefix;
    return *this;
  }

  index_filter_params_builder_t& with_label_file(const std::string& label_file) {
    this->label_file_ = label_file;
    return *this;
  }

  index_filter_params_builder_t& with_universal_label(const std::string& univeral_label) {
    this->universal_label_ = univeral_label;
    return *this;
  }

  index_filter_params_builder_t& with_filter_threshold(const std::uint32_t& filter_threshold) {
    this->filter_threshold_ = filter_threshold;
    return *this;
  }

  index_filter_params_t build() {
    return index_filter_params_t(save_path_prefix_, label_file_, universal_label_,
                                 filter_threshold_);
  }

  index_filter_params_builder_t(const index_filter_params_builder_t&) = delete;
  index_filter_params_builder_t& operator=(const index_filter_params_builder_t&) = delete;

private:
  std::string save_path_prefix_;
  std::string label_file_;
  std::string tags_file_;
  std::string universal_label_;
  uint32_t filter_threshold_ = 0;
};
} // namespace powerlaw_ann

#endif // INDEX_INDEX_BUILD_PARAMS
