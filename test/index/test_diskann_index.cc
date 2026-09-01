#include "common/ann_error.h"
#include "power_ann.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

namespace {

class temp_dir_t {
public:
  temp_dir_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_diskann_index_test_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~temp_dir_t() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

void write_dataset(const std::filesystem::path& path) {
  constexpr int32_t num_points = 32;
  constexpr int32_t dimensions = 4;
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
  output.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
  for (int32_t point = 0; point < num_points; ++point) {
    const float vector[dimensions] = {
        static_cast<float>(point),
        static_cast<float>((point * 7) % 13),
        static_cast<float>((point * point) % 17),
        static_cast<float>((point * 11) % 19),
    };
    output.write(reinterpret_cast<const char*>(vector), sizeof(vector));
  }
}

std::vector<char> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

powerlaw_ann::diskann_memory_index_config_t make_config(const std::filesystem::path& data_path,
                                                        const std::filesystem::path& index_prefix) {
  powerlaw_ann::diskann_memory_index_config_t config;
  config.data_path = data_path;
  config.index_path_prefix = index_prefix;
  config.max_degree = 8;
  config.build_list_size = 16;
  config.alpha = 1.2F;
  config.num_threads = 1;
  return config;
}

TEST(DiskannIndexTest, BuildsDeterministicFloatL2Artifacts) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto first_prefix = temp_dir.path() / "first.index";
  const auto second_prefix = temp_dir.path() / "second.index";
  write_dataset(data_path);

  powerlaw_ann::power_ann_t power_ann;
  power_ann.build_diskann_memory_index(make_config(data_path, first_prefix));
  power_ann.build_diskann_memory_index(make_config(data_path, second_prefix));

  ASSERT_TRUE(std::filesystem::is_regular_file(first_prefix));
  ASSERT_TRUE(std::filesystem::is_regular_file(first_prefix.string() + ".data"));
  EXPECT_EQ(read_file(first_prefix), read_file(second_prefix));
  EXPECT_EQ(read_file(first_prefix.string() + ".data"),
            read_file(second_prefix.string() + ".data"));
}

TEST(DiskannIndexTest, RejectsInvalidTypedConfiguration) {
  temp_dir_t temp_dir;
  auto config = make_config(temp_dir.path() / "missing.bin", temp_dir.path() / "index");

  powerlaw_ann::power_ann_t power_ann;
  EXPECT_THROW(power_ann.build_diskann_memory_index(config), powerlaw_ann::ann_exception_t);
}

TEST(DiskannIndexTest, RejectsDirectoryPrefixWithoutRemovingIt) {
  temp_dir_t temp_dir;
  const auto data_path = temp_dir.path() / "base.bin";
  const auto index_directory = temp_dir.path() / "index_directory";
  write_dataset(data_path);
  std::filesystem::create_directory(index_directory);

  auto config = make_config(data_path, index_directory);
  powerlaw_ann::power_ann_config_t power_ann_config;
  power_ann_config.enable_powerann = true;
  powerlaw_ann::power_ann_t power_ann(power_ann_config);
  EXPECT_THROW(power_ann.build_diskann_memory_index(config), powerlaw_ann::ann_exception_t);
  EXPECT_TRUE(std::filesystem::is_directory(index_directory));
  EXPECT_TRUE(std::filesystem::is_empty(index_directory));
}

} // namespace
