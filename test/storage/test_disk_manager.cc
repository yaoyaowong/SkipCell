#include "storage/disk_manager.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

static std::string make_disk_manager_prefix(const std::string& name) {
  return "/tmp/powerlaw_ann_disk_manager_" + std::to_string(::getpid()) + "_" + name;
}

static void remove_disk_manager_files(const std::string& prefix) {
  const auto files = disk_manager_t::build_file_names(prefix);
  std::filesystem::remove(files.meta);
  for (const auto& path : files.chunks) {
    std::filesystem::remove(path);
  }
}

TEST(disk_manager_t_test, encodes_page_id_with_chunk_type_and_page_number) {
  const chunk_id_t id = disk_manager_t::make_chunk_id(chunk_type_t::chunk_2mb, 1);

  EXPECT_EQ(disk_manager_t::chunk_type(id), chunk_type_t::chunk_2mb);
  EXPECT_EQ(disk_manager_t::chunk_no(id), 1u);

  const auto addr = disk_manager_t::decode_chunk_id(id);
  EXPECT_EQ(addr.type, chunk_type_t::chunk_2mb);
  EXPECT_EQ(addr.chunk_no, 1u);
  EXPECT_EQ(addr.offset, 2u * 1024u * 1024u);
  EXPECT_EQ(addr.size, 2u * 1024u * 1024u);
}

TEST(disk_manager_t_test, builds_prefixed_file_group) {
  const auto files = disk_manager_t::build_file_names("/tmp/powerlaw_ann_demo");

  EXPECT_EQ(files.meta, "/tmp/powerlaw_ann_demo_meta.db");
  EXPECT_EQ(files.chunks[0], "/tmp/powerlaw_ann_demo_chunk_0.db");
  EXPECT_EQ(files.chunks[1], "/tmp/powerlaw_ann_demo_chunk_1.db");
  EXPECT_EQ(files.chunks[2], "/tmp/powerlaw_ann_demo_chunk_2.db");
  EXPECT_EQ(files.chunks[3], "/tmp/powerlaw_ann_demo_chunk_3.db");
}

TEST(disk_manager_t_test, writes_loads_and_reuses_chunk_pages) {
  const std::string prefix = make_disk_manager_prefix("rw");
  remove_disk_manager_files(prefix);

  disk_manager_options_t options;
  options.truncate = true;
  options.sync_on_write = true;

  {
    disk_manager_t manager(prefix, options);
    ASSERT_TRUE(manager.is_open());

    std::vector<uint8_t> src(disk_manager_t::chunk_size(chunk_type_t::chunk_4kb), 0);
    for (size_t i = 0; i < src.size(); ++i) {
      src[i] = static_cast<uint8_t>(i % 251);
    }

    const chunk_id_t first =
        manager.write_new_chunk(chunk_type_t::chunk_4kb, src.data(), src.size());
    EXPECT_EQ(disk_manager_t::chunk_no(first), 0u);
    EXPECT_EQ(manager.chunk_count(chunk_type_t::chunk_4kb), 1u);

    std::vector<uint8_t> dst(src.size(), 0);
    manager.load_chunk(first, dst.data());
    EXPECT_EQ(dst, src);

    manager.delete_chunk(first);
    EXPECT_EQ(manager.free_chunk_count(chunk_type_t::chunk_4kb), 1u);

    std::vector<uint8_t> second_src(src.size(), 7);
    const chunk_id_t second =
        manager.write_new_chunk(chunk_type_t::chunk_4kb, second_src.data(), second_src.size());
    EXPECT_EQ(second, first);
    EXPECT_EQ(manager.free_chunk_count(chunk_type_t::chunk_4kb), 0u);

    std::vector<uint8_t> second_dst(second_src.size(), 0);
    manager.load_chunk(second, second_dst.data());
    EXPECT_EQ(second_dst, second_src);
  }

  remove_disk_manager_files(prefix);
}

TEST(disk_manager_t_test, persists_counts_and_free_lists_after_reopen) {
  const std::string prefix = make_disk_manager_prefix("reopen");
  remove_disk_manager_files(prefix);

  chunk_id_t deleted = 0;

  {
    disk_manager_options_t options;
    options.truncate = true;

    disk_manager_t manager(prefix, options);
    std::vector<uint8_t> src(disk_manager_t::chunk_size(chunk_type_t::chunk_64kb), 3);

    deleted = manager.write_new_chunk(chunk_type_t::chunk_64kb, src.data(), src.size());
    const chunk_id_t kept =
        manager.write_new_chunk(chunk_type_t::chunk_64kb, src.data(), src.size());
    EXPECT_EQ(disk_manager_t::chunk_no(deleted), 0u);
    EXPECT_EQ(disk_manager_t::chunk_no(kept), 1u);

    manager.delete_chunk(deleted);
    EXPECT_EQ(manager.chunk_count(chunk_type_t::chunk_64kb), 2u);
    EXPECT_EQ(manager.free_chunk_count(chunk_type_t::chunk_64kb), 1u);
  }

  {
    disk_manager_t manager(prefix);
    EXPECT_EQ(manager.chunk_count(chunk_type_t::chunk_64kb), 2u);
    EXPECT_EQ(manager.free_chunk_count(chunk_type_t::chunk_64kb), 1u);

    std::vector<uint8_t> src(disk_manager_t::chunk_size(chunk_type_t::chunk_64kb), 9);
    const chunk_id_t reused =
        manager.write_new_chunk(chunk_type_t::chunk_64kb, src.data(), src.size());
    EXPECT_EQ(reused, deleted);
    EXPECT_EQ(manager.free_chunk_count(chunk_type_t::chunk_64kb), 0u);
  }

  remove_disk_manager_files(prefix);
}

TEST(disk_manager_t_test, concurrently_loads_chunks_from_one_manager) {
  const std::string prefix = make_disk_manager_prefix("concurrent_reads");
  remove_disk_manager_files(prefix);

  constexpr size_t chunk_count = 32;
  disk_manager_options_t options;
  options.truncate = true;

  {
    disk_manager_t writer(prefix, options);
    for (size_t i = 0; i < chunk_count; ++i) {
      std::vector<uint8_t> data(disk_manager_t::chunk_size(chunk_type_t::chunk_4kb),
                                static_cast<uint8_t>(i + 1));
      writer.write_new_chunk(chunk_type_t::chunk_4kb, data.data(), data.size());
    }
  }

  disk_manager_t reader(prefix);
  std::atomic<bool> valid{true};
  std::vector<std::thread> threads;
  for (size_t thread_id = 0; thread_id < 8; ++thread_id) {
    threads.emplace_back([&, thread_id] {
      std::vector<uint8_t> data(disk_manager_t::chunk_size(chunk_type_t::chunk_4kb));
      for (size_t iteration = 0; iteration < 500; ++iteration) {
        const size_t index = (thread_id * 7 + iteration * 13) % chunk_count;
        const chunk_id_t chunk_id = disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, index);
        reader.load_chunk(chunk_id, data.data());
        const uint8_t expected = static_cast<uint8_t>(index + 1);
        if (!std::all_of(data.begin(), data.end(),
                         [expected](uint8_t value) { return value == expected; })) {
          valid.store(false, std::memory_order_relaxed);
          return;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(valid.load(std::memory_order_relaxed));
  remove_disk_manager_files(prefix);
}
