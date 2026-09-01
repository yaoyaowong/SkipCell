#include "storage/io_lru_buffer_pool.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <thread>

namespace {

TEST(IoLruBufferPoolTest, HandlesHitMissEvictionPinningAndZeroCapacity) {
  powerlaw_ann::io_lru_buffer_pool_t pool(2);
  auto first = pool.reserve(7);
  ASSERT_EQ(first.state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  std::memset(first.data, 0x2A, 4096);
  pool.publish(first, true);
  pool.release(first);

  const auto absent = pool.try_reserve_hit(99);
  EXPECT_EQ(absent.state, powerlaw_ann::io_cache_reservation_state_t::BYPASS);
  EXPECT_EQ(pool.stats().misses, 1U);
  auto speculative_hit = pool.try_reserve_hit(7);
  ASSERT_EQ(speculative_hit.state, powerlaw_ann::io_cache_reservation_state_t::HIT);
  EXPECT_EQ(static_cast<unsigned char>(speculative_hit.data[0]), 0x2A);
  pool.release(speculative_hit);

  auto hit = pool.reserve(7);
  ASSERT_EQ(hit.state, powerlaw_ann::io_cache_reservation_state_t::HIT);
  EXPECT_EQ(static_cast<unsigned char>(hit.data[0]), 0x2A);
  auto second = pool.reserve(8);
  pool.publish(second, true);
  pool.release(second);
  auto third = pool.reserve(9);
  EXPECT_TRUE(third.evicted);
  pool.publish(third, true);
  pool.release(third);
  pool.release(hit);

  auto stats = pool.stats();
  EXPECT_EQ(stats.demand_hits, 2U);
  EXPECT_EQ(stats.misses, 3U);
  EXPECT_EQ(stats.evictions, 1U);
  EXPECT_EQ(pool.resident_bytes(), 8192U);

  powerlaw_ann::io_lru_buffer_pool_t zero(0);
  EXPECT_EQ(zero.reserve(1).state, powerlaw_ann::io_cache_reservation_state_t::BYPASS);
  EXPECT_EQ(zero.stats().misses, 1U);
}

TEST(IoLruBufferPoolTest, CoalescesConcurrentMissAndRollsBackFailedLoad) {
  powerlaw_ann::io_lru_buffer_pool_t pool(1);
  auto loader = pool.reserve(12);
  ASSERT_EQ(loader.state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  const auto nonblocking = pool.reserve(12, powerlaw_ann::io_cache_access_t::DEMAND, false);
  EXPECT_EQ(nonblocking.state, powerlaw_ann::io_cache_reservation_state_t::BYPASS);
  EXPECT_TRUE(nonblocking.coalesced);
  auto waiter = std::async(std::launch::async, [&]() { return pool.reserve(12); });
  EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  std::memset(loader.data, 0x51, 4096);
  pool.publish(loader, true);
  auto coalesced = waiter.get();
  EXPECT_EQ(coalesced.state, powerlaw_ann::io_cache_reservation_state_t::HIT);
  EXPECT_TRUE(coalesced.coalesced);
  EXPECT_EQ(static_cast<unsigned char>(coalesced.data[0]), 0x51);
  pool.release(loader);
  pool.release(coalesced);

  auto failed = pool.reserve(13);
  ASSERT_EQ(failed.state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  pool.publish(failed, false);
  auto retry = pool.reserve(13);
  EXPECT_EQ(retry.state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  pool.publish(retry, true);
  pool.release(retry);
  EXPECT_EQ(pool.stats().failed_loads, 1U);
}

TEST(IoLruBufferPoolTest, BypassesWhenEveryVictimIsPinned) {
  powerlaw_ann::io_lru_buffer_pool_t pool(1);
  auto first = pool.reserve(1);
  pool.publish(first, true);
  auto bypass = pool.reserve(2);
  EXPECT_EQ(bypass.state, powerlaw_ann::io_cache_reservation_state_t::BYPASS);
  EXPECT_EQ(pool.stats().pinned_victim_failures, 1U);
  pool.release(first);
}

TEST(IoLruBufferPoolTest, ClearDropsPagesAndCountersBetweenCurvePoints) {
  powerlaw_ann::io_lru_buffer_pool_t pool(2);
  auto resident = pool.reserve(7);
  pool.publish(resident, true);
  pool.release(resident);
  auto pinned = pool.try_reserve_hit(7);
  ASSERT_EQ(pinned.state, powerlaw_ann::io_cache_reservation_state_t::HIT);
  EXPECT_THROW(pool.clear(), std::logic_error);
  pool.release(pinned);

  pool.clear();
  EXPECT_EQ(pool.stats().demand_hits, 0U);
  auto missing = pool.try_reserve_hit(7);
  EXPECT_EQ(missing.state, powerlaw_ann::io_cache_reservation_state_t::BYPASS);
  auto reusable = pool.reserve(8);
  EXPECT_EQ(reusable.state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  pool.publish(reusable, true);
  pool.release(reusable);
}

TEST(IoLruBufferPoolTest, CapacityResetBuildsARequestedEmptyPool) {
  powerlaw_ann::io_lru_buffer_pool_t pool(2);
  auto resident = pool.reserve(7);
  pool.publish(resident, true);
  pool.release(resident);

  pool.reset_capacity(4);
  EXPECT_EQ(pool.capacity_pages(), 4U);
  EXPECT_EQ(pool.resident_bytes(), 4U * 4096U);
  EXPECT_EQ(pool.stats().misses, 0U);
  EXPECT_EQ(pool.try_reserve_hit(7).state,
            powerlaw_ann::io_cache_reservation_state_t::BYPASS);

  auto pinned = pool.reserve(8);
  pool.publish(pinned, true);
  EXPECT_THROW(pool.reset_capacity(1), std::logic_error);
  pool.release(pinned);
  EXPECT_THROW(pool.reset_capacity(0), std::invalid_argument);
  EXPECT_NO_THROW(pool.reset_capacity(1));
  EXPECT_EQ(pool.capacity_pages(), 1U);
}

TEST(IoLruBufferPoolTest, TracksPrefetchAdmissionUseAndPollution) {
  powerlaw_ann::io_lru_buffer_pool_t pool(2);
  auto prefetched = pool.reserve(4, powerlaw_ann::io_cache_access_t::PREFETCH);
  ASSERT_EQ(prefetched.state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  pool.publish(prefetched, true);
  pool.release(prefetched);
  auto demand = pool.reserve(4);
  ASSERT_EQ(demand.state, powerlaw_ann::io_cache_reservation_state_t::HIT);
  EXPECT_TRUE(demand.prefetched_hit);
  pool.release(demand);
  auto repeated_demand = pool.reserve(4);
  EXPECT_FALSE(repeated_demand.prefetched_hit);
  pool.release(repeated_demand);
  auto second_prefetch = pool.reserve(5, powerlaw_ann::io_cache_access_t::PREFETCH);
  pool.publish(second_prefetch, true);
  pool.release(second_prefetch);
  auto replacement = pool.reserve(6);
  pool.publish(replacement, true);
  pool.release(replacement);
  const auto stats = pool.stats();
  EXPECT_EQ(stats.prefetch_misses, 2U);
  EXPECT_EQ(stats.prefetch_used, 1U);
  EXPECT_EQ(stats.prefetch_pollution_evictions, 1U);
}

TEST(IoLruBufferPoolTest, BoundsPrefetchDemandEvictionWithUseCredit) {
  powerlaw_ann::io_lru_buffer_pool_t pool(2);
  auto first = pool.reserve(1);
  pool.publish(first, true);
  pool.release(first);
  auto second = pool.reserve(2);
  pool.publish(second, true);
  pool.release(second);
  auto admitted = pool.reserve(3, powerlaw_ann::io_cache_access_t::PREFETCH, false);
  EXPECT_EQ(admitted.state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  pool.publish(admitted, true);
  pool.release(admitted);
  auto replacement = pool.reserve(4, powerlaw_ann::io_cache_access_t::PREFETCH, false);
  EXPECT_EQ(replacement.state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  pool.publish(replacement, true);
  pool.release(replacement);
  auto hit = pool.reserve(2);
  EXPECT_EQ(hit.state, powerlaw_ann::io_cache_reservation_state_t::HIT);
  pool.release(hit);
  EXPECT_EQ(pool.stats().prefetch_demand_evictions, 1U);
  EXPECT_EQ(pool.stats().prefetch_pollution_evictions, 1U);
}

TEST(IoLruBufferPoolTest, KeepsChunkSizeClassesInIndependentDomains) {
  std::array<size_t, disk_manager_t::k_chunk_levels> capacities{};
  capacities[static_cast<size_t>(chunk_type_t::chunk_4kb)] = 2;
  capacities[static_cast<size_t>(chunk_type_t::chunk_64kb)] = 1;
  powerlaw_ann::io_size_specific_lru_t pools(capacities);
  ASSERT_NE(pools.pool(chunk_type_t::chunk_4kb), nullptr);
  ASSERT_NE(pools.pool(chunk_type_t::chunk_64kb), nullptr);
  EXPECT_EQ(pools.pool(chunk_type_t::chunk_512kb), nullptr);
  EXPECT_EQ(pools.resident_bytes(), 2U * 4096U + 65536U);

  auto small = pools.pool(chunk_type_t::chunk_4kb)->reserve(1);
  auto large = pools.pool(chunk_type_t::chunk_64kb)->reserve(1);
  EXPECT_EQ(small.frame->get_chunk_size(), 4096U);
  EXPECT_EQ(large.frame->get_chunk_size(), 65536U);
  pools.pool(chunk_type_t::chunk_4kb)->publish(small, true);
  pools.pool(chunk_type_t::chunk_64kb)->publish(large, true);
  pools.pool(chunk_type_t::chunk_4kb)->release(small);
  pools.pool(chunk_type_t::chunk_64kb)->release(large);
}

TEST(IoLruBufferPoolTest, ReservesPublishesAndReleasesSortedBatches) {
  powerlaw_ann::io_lru_buffer_pool_t pool(3);
  const std::array<uint64_t, 2> pages = {4, 9};
  auto loads = pool.reserve_many(pages);
  ASSERT_EQ(loads.size(), pages.size());
  EXPECT_EQ(loads[0].state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  EXPECT_EQ(loads[1].state, powerlaw_ann::io_cache_reservation_state_t::LOAD);
  std::memset(loads[0].data, 0x14, 4096);
  std::memset(loads[1].data, 0x19, 4096);
  pool.publish_many(loads, true);
  pool.release_many(loads);

  auto hits = pool.reserve_many(pages);
  ASSERT_EQ(hits.size(), pages.size());
  EXPECT_EQ(hits[0].state, powerlaw_ann::io_cache_reservation_state_t::HIT);
  EXPECT_EQ(hits[1].state, powerlaw_ann::io_cache_reservation_state_t::HIT);
  EXPECT_EQ(static_cast<unsigned char>(hits[0].data[0]), 0x14);
  EXPECT_EQ(static_cast<unsigned char>(hits[1].data[0]), 0x19);
  pool.release_many(hits);

  const std::array<uint64_t, 2> unsorted = {9, 4};
  const std::array<uint64_t, 2> duplicate = {4, 4};
  EXPECT_THROW(pool.reserve_many(unsorted), std::invalid_argument);
  EXPECT_THROW(pool.reserve_many(duplicate), std::invalid_argument);
}

} // namespace
