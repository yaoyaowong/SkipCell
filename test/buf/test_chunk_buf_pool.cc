#include "storage/chunk_buf_pool.h"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

TEST(chunk_buf_pool_t_test, chunk_control_body_is_cache_line_aligned) {
  EXPECT_GE(alignof(chunk_t), chunk_t::k_cache_line_size);
  EXPECT_EQ(sizeof(chunk_t) % chunk_t::k_cache_line_size, 0u);
}

TEST(chunk_buf_pool_t_test, initializes_runtime_configured_level_capacities) {
  chunk_buf_pool_options_t options;
  options.reserved_chunk_counts = {3, 2, 0, 1};

  auto pool = chunk_buf_pool_t::chunk_buf_pool_init(options);
  ASSERT_NE(pool, nullptr);

  EXPECT_EQ(pool->get_level_count(), 4u);
  EXPECT_EQ(pool->get_reserved_chunk_count(chunk_type_t::chunk_4kb), 3u);
  EXPECT_EQ(pool->get_reserved_chunk_count(chunk_type_t::chunk_64kb), 2u);
  EXPECT_EQ(pool->get_reserved_chunk_count(chunk_type_t::chunk_512kb), 0u);
  EXPECT_EQ(pool->get_reserved_chunk_count(chunk_type_t::chunk_2mb), 1u);
  EXPECT_EQ(pool->get_total_chunk_count(), 6u);
}

TEST(chunk_buf_pool_t_test, keeps_controls_and_data_frames_contiguous_per_level) {
  chunk_buf_pool_options_t options;
  options.reserved_chunk_counts = {4, 0, 0, 0};

  auto pool = chunk_buf_pool_t::chunk_buf_pool_init(options);
  ASSERT_NE(pool, nullptr);

  chunk_t* first = pool->get_chunk(chunk_type_t::chunk_4kb, 0);
  chunk_t* second = pool->get_chunk(chunk_type_t::chunk_4kb, 1);
  chunk_t* fourth = pool->get_chunk(chunk_type_t::chunk_4kb, 3);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(fourth, nullptr);

  const auto first_addr = reinterpret_cast<uintptr_t>(first);
  const auto second_addr = reinterpret_cast<uintptr_t>(second);
  EXPECT_EQ(first_addr % chunk_t::k_cache_line_size, 0u);
  EXPECT_EQ(second_addr - first_addr, sizeof(chunk_t));

  char* first_frame = pool->get_data_frame(chunk_type_t::chunk_4kb, 0);
  char* second_frame = pool->get_data_frame(chunk_type_t::chunk_4kb, 1);
  ASSERT_NE(first_frame, nullptr);
  ASSERT_NE(second_frame, nullptr);

  const size_t chunk_size = disk_manager_t::chunk_size(chunk_type_t::chunk_4kb);
  const auto data_addr = reinterpret_cast<uintptr_t>(first_frame);
  const auto controls_end = reinterpret_cast<uintptr_t>(fourth) + sizeof(chunk_t);
  EXPECT_GT(data_addr, controls_end);
  EXPECT_EQ(first->get_data(), first_frame);
  EXPECT_EQ(second->get_data(), second_frame);
  EXPECT_EQ(static_cast<size_t>(second_frame - first_frame), chunk_size);
  EXPECT_EQ(fourth->get_chunk_id(), disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, 3));
}

TEST(chunk_buf_pool_t_test, allocates_and_releases_frames_from_the_configured_free_list) {
  chunk_buf_pool_options_t options;
  options.reserved_chunk_counts = {2, 0, 0, 0};

  auto pool = chunk_buf_pool_t::chunk_buf_pool_init(options);
  ASSERT_NE(pool, nullptr);

  chunk_t* first = pool->allocate_chunk(chunk_type_t::chunk_4kb);
  chunk_t* second = pool->allocate_chunk(chunk_type_t::chunk_4kb);
  chunk_t* exhausted = pool->allocate_chunk(chunk_type_t::chunk_4kb);

  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(exhausted, nullptr);
  EXPECT_EQ(first->get_chunk_state(), chunk_state_t::resident);
  EXPECT_EQ(second->get_chunk_state(), chunk_state_t::resident);
  EXPECT_EQ(pool->get_free_chunk_count(chunk_type_t::chunk_4kb), 0u);

  pool->release_chunk(first);

  EXPECT_EQ(first->get_chunk_state(), chunk_state_t::free);
  EXPECT_EQ(pool->get_free_chunk_count(chunk_type_t::chunk_4kb), 1u);
  EXPECT_EQ(pool->allocate_chunk(chunk_type_t::chunk_4kb), first);
}

TEST(chunk_buf_pool_t_test, decodes_chunk_ids_for_direct_lookup) {
  chunk_buf_pool_options_t options;
  options.reserved_chunk_counts = {0, 3, 0, 0};

  auto pool = chunk_buf_pool_t::chunk_buf_pool_init(options);
  ASSERT_NE(pool, nullptr);

  const chunk_id_t id = disk_manager_t::make_chunk_id(chunk_type_t::chunk_64kb, 2);
  chunk_t* chunk = pool->get_chunk(id);

  ASSERT_NE(chunk, nullptr);
  EXPECT_EQ(chunk->get_chunk_type(), chunk_type_t::chunk_64kb);
  EXPECT_EQ(chunk->get_chunk_no(), 2u);
  EXPECT_EQ(pool->get_chunk(disk_manager_t::make_chunk_id(chunk_type_t::chunk_64kb, 3)), nullptr);
}
