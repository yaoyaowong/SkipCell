#include "storage/chunk.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>

TEST(chunk_t_test, default_chunk_starts_without_bound_storage) {
  chunk_t chunk;

  EXPECT_EQ(chunk.get_chunk_id(), chunk_t::k_invalid_chunk_id);
  EXPECT_EQ(chunk.get_chunk_size(), 0u);
  EXPECT_EQ(chunk.get_data(), nullptr);
  EXPECT_EQ(chunk.get_pin_count(), 0u);
  EXPECT_FALSE(chunk.is_dirty());
  EXPECT_EQ(chunk.get_chunk_state(), chunk_state_t::free);
}

TEST(chunk_t_test, binds_control_body_to_external_data_domain) {
  std::array<std::byte, 4096> frame{};
  chunk_t chunk;

  chunk.bind(disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, 7), chunk_type_t::chunk_4kb,
             frame.data(), frame.size());

  EXPECT_EQ(chunk.get_chunk_id(), disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, 7));
  EXPECT_EQ(chunk.get_chunk_type(), chunk_type_t::chunk_4kb);
  EXPECT_EQ(chunk.get_chunk_no(), 7u);
  EXPECT_EQ(chunk.get_chunk_size(), frame.size());
  ASSERT_EQ(chunk.get_data(), reinterpret_cast<char*>(frame.data()));

  std::memset(chunk.get_data(), 0x5A, chunk.get_chunk_size());
  EXPECT_EQ(frame[0], std::byte{0x5A});
  EXPECT_EQ(frame.back(), std::byte{0x5A});
}

TEST(chunk_t_test, tracks_pin_dirty_and_referenced_state) {
  std::array<std::byte, 4096> frame{};
  chunk_t chunk(disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, 2), chunk_type_t::chunk_4kb,
                frame.data(), frame.size());

  EXPECT_EQ(chunk.inc_pin_count(), 1u);
  EXPECT_EQ(chunk.inc_pin_count(), 2u);
  EXPECT_EQ(chunk.dec_pin_count(), 1u);

  chunk.mark_dirty();
  chunk.set_referenced(true);
  chunk.set_chunk_state(chunk_state_t::resident);

  EXPECT_TRUE(chunk.is_dirty());
  EXPECT_TRUE(chunk.get_referenced());
  EXPECT_EQ(chunk.get_chunk_state(), chunk_state_t::resident);
}

TEST(chunk_t_test, reset_memory_clears_data_and_runtime_metadata) {
  std::array<std::byte, 4096> frame;
  frame.fill(std::byte{0x7F});

  chunk_t chunk(disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, 3), chunk_type_t::chunk_4kb,
                frame.data(), frame.size());
  chunk.inc_pin_count();
  chunk.mark_dirty();
  chunk.set_referenced(true);

  chunk.reset_memory();

  EXPECT_EQ(chunk.get_chunk_id(), disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, 3));
  EXPECT_EQ(chunk.get_pin_count(), 0u);
  EXPECT_FALSE(chunk.is_dirty());
  EXPECT_FALSE(chunk.get_referenced());
  for (const auto byte : frame) {
    EXPECT_EQ(byte, std::byte{0});
  }
}

TEST(chunk_t_test, stores_intrusive_list_links) {
  std::array<std::byte, 4096> left_frame{};
  std::array<std::byte, 4096> right_frame{};
  chunk_t left(disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, 1), chunk_type_t::chunk_4kb,
               left_frame.data(), left_frame.size());
  chunk_t right(disk_manager_t::make_chunk_id(chunk_type_t::chunk_4kb, 2), chunk_type_t::chunk_4kb,
                right_frame.data(), right_frame.size());

  left.set_next(&right);
  right.set_prev(&left);

  EXPECT_EQ(left.get_next(), &right);
  EXPECT_EQ(right.get_prev(), &left);
}
