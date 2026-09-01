#if defined(__linux__)

#include "common/diskann_exception.h"
#include "storage/io_uring_aligned_file_reader.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

class temp_file_t {
public:
  temp_file_t() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("powerlawann_io_uring_test_" + std::to_string(suffix));
    std::vector<uint8_t> data(8192);
    for (size_t index = 0; index < data.size(); ++index) {
      data[index] = static_cast<uint8_t>(index % 251U);
    }
    std::ofstream output(path_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(data.data()), data.size());
  }

  ~temp_file_t() { std::filesystem::remove(path_); }
  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

struct aligned_free_t {
  void operator()(void* pointer) const { std::free(pointer); }
};

TEST(IoUringAlignedFileReaderTest, ReadsAlignedDemandBatchAndReportsCompletions) {
  temp_file_t file;
  void* allocation = nullptr;
  ASSERT_EQ(posix_memalign(&allocation, 4096, 8192), 0);
  std::unique_ptr<void, aligned_free_t> buffer(allocation);
  powerlaw_ann::io_uring_aligned_file_reader_t reader(8);
  reader.open(file.path().string());
  reader.register_thread();
  auto& context = reader.get_ctx();
  std::vector<powerlaw_ann::aligned_read_t> requests = {
      {0, 4096, allocation},
      {4096, 4096, static_cast<char*>(allocation) + 4096},
  };
  ASSERT_NO_THROW(reader.read(requests, context));
  const auto* bytes = static_cast<const uint8_t*>(allocation);
  for (size_t index = 0; index < 8192; ++index) {
    EXPECT_EQ(bytes[index], static_cast<uint8_t>(index % 251U));
  }
  const auto stats = reader.stats();
  EXPECT_EQ(stats.submitted, 2U);
  EXPECT_EQ(stats.completed, 2U);
  EXPECT_EQ(stats.completion_batches, 1U);
  EXPECT_EQ(stats.short_or_error_completions, 0U);
  EXPECT_EQ(stats.queue_depth, 8U);
  reader.deregister_thread();
}

TEST(IoUringAlignedFileReaderTest, RejectsInvalidDepthAndShortCompletion) {
  EXPECT_THROW(powerlaw_ann::io_uring_aligned_file_reader_t(0), powerlaw_ann::diskann_exception_t);
  temp_file_t file;
  void* allocation = nullptr;
  ASSERT_EQ(posix_memalign(&allocation, 4096, 4096), 0);
  std::unique_ptr<void, aligned_free_t> buffer(allocation);
  powerlaw_ann::io_uring_aligned_file_reader_t reader(2);
  reader.open(file.path().string());
  reader.register_thread();
  std::vector<powerlaw_ann::aligned_read_t> requests = {{8192, 4096, allocation}};
  EXPECT_THROW(reader.read(requests, reader.get_ctx()), powerlaw_ann::diskann_exception_t);
  EXPECT_EQ(reader.stats().short_or_error_completions, 1U);
  reader.deregister_thread();
}

TEST(IoUringAlignedFileReaderTest, SubmitsAndHarvestsTaggedPrefetch) {
  temp_file_t file;
  void* allocation = nullptr;
  ASSERT_EQ(posix_memalign(&allocation, 4096, 4096), 0);
  std::unique_ptr<void, aligned_free_t> buffer(allocation);
  powerlaw_ann::io_uring_aligned_file_reader_t reader(2);
  reader.open(file.path().string());
  reader.register_thread();
  auto& context = reader.get_ctx();
  EXPECT_TRUE(reader.has_prefetch_headroom(context));
  const powerlaw_ann::aligned_read_t request{4096, 4096, allocation};
  ASSERT_NO_THROW(reader.submit_prefetch(request, context, 37));
  EXPECT_FALSE(reader.has_prefetch_headroom(context));
  EXPECT_TRUE(reader.has_prefetch_headroom(context, 0));
  std::vector<powerlaw_ann::aligned_read_t> demand;
  EXPECT_NO_THROW(reader.read(demand, context));
  const auto tags = reader.complete_prefetches(context);
  ASSERT_EQ(tags.size(), 1U);
  EXPECT_EQ(tags.front(), 37U);
  EXPECT_TRUE(reader.has_prefetch_headroom(context));
  const auto* bytes = static_cast<const uint8_t*>(allocation);
  for (size_t index = 0; index < 4096; ++index) {
    EXPECT_EQ(bytes[index], static_cast<uint8_t>((index + 4096U) % 251U));
  }
  EXPECT_EQ(reader.stats().prefetch_submitted, 1U);
  EXPECT_EQ(reader.stats().prefetch_completed, 1U);
  ASSERT_NO_THROW(reader.submit_prefetch(request, context, 41));
  ASSERT_NO_THROW(reader.discard_prefetches(context));
  EXPECT_EQ(reader.stats().prefetch_submitted, 2U);
  EXPECT_EQ(reader.stats().prefetch_completed, 2U);
  EXPECT_EQ(reader.stats().prefetch_obsolete, 1U);
  reader.deregister_thread();
}

TEST(IoUringAlignedFileReaderTest, OverlapsPrefetchWithDemandAndPollsWithoutBarrier) {
  temp_file_t file;
  void* allocation = nullptr;
  ASSERT_EQ(posix_memalign(&allocation, 4096, 8192), 0);
  std::unique_ptr<void, aligned_free_t> buffer(allocation);
  powerlaw_ann::io_uring_aligned_file_reader_t reader(4);
  reader.open(file.path().string());
  reader.register_thread();
  auto& context = reader.get_ctx();
  ASSERT_NO_THROW(reader.submit_prefetch({4096, 4096, allocation}, context, 73));
  const auto early = reader.poll_prefetches(context);
  EXPECT_LE(early.size(), 1U);
  std::vector<powerlaw_ann::aligned_read_t> demand = {
      {0, 4096, static_cast<char*>(allocation) + 4096}};
  EXPECT_NO_THROW(reader.read(demand, context));
  auto completed = reader.poll_prefetches(context);
  completed.insert(completed.end(), early.begin(), early.end());
  if (completed.empty()) {
    completed = reader.complete_prefetches(context);
  }
  ASSERT_EQ(completed.size(), 1U);
  EXPECT_EQ(completed.front(), 73U);
  EXPECT_EQ(reader.stats().submitted, 2U);
  EXPECT_EQ(reader.stats().completed, 2U);
  reader.deregister_thread();
}

} // namespace

#endif // defined(__linux__)
