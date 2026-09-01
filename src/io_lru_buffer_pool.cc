#include "storage/io_lru_buffer_pool.h"

#include <algorithm>
#include <stdexcept>

namespace powerlaw_ann {
namespace {

bool is_prefetch(io_cache_access_t access) noexcept { return access != io_cache_access_t::DEMAND; }

} // namespace

io_lru_buffer_pool_t::io_lru_buffer_pool_t(size_t capacity_pages, chunk_type_t chunk_type)
    : capacity_pages_(capacity_pages), chunk_type_(chunk_type),
      chunk_bytes_(disk_manager_t::chunk_size(chunk_type)) {
  chunk_buf_pool_options_t options;
  options.reserved_chunk_counts[static_cast<size_t>(chunk_type_)] = capacity_pages;
  frames_ = chunk_buf_pool_t::chunk_buf_pool_init(options);
  free_frames_.reserve(capacity_pages);
  resident_.reserve(capacity_pages);
  loading_.reserve(capacity_pages);
  for (size_t index = 0; index < capacity_pages; ++index) {
    free_frames_.push_back(frames_->get_chunk(chunk_type_, index));
  }
}

io_cache_reservation_t io_lru_buffer_pool_t::reserve(uint64_t page_id, io_cache_access_t access,
                                                     bool wait_for_inflight) {
  std::unique_lock lock(mutex_);
  return reserve_locked(page_id, access, wait_for_inflight, lock);
}

std::vector<io_cache_reservation_t>
io_lru_buffer_pool_t::reserve_many(std::span<const uint64_t> page_ids, io_cache_access_t access) {
  if (!std::is_sorted(page_ids.begin(), page_ids.end()) ||
      std::adjacent_find(page_ids.begin(), page_ids.end()) != page_ids.end()) {
    throw std::invalid_argument("LRU batch page IDs must be strictly increasing");
  }
  std::unique_lock lock(mutex_);
  std::vector<io_cache_reservation_t> reservations;
  reservations.reserve(page_ids.size());
  for (const uint64_t page_id : page_ids) {
    reservations.push_back(reserve_locked(page_id, access, true, lock));
  }
  return reservations;
}

io_cache_reservation_t io_lru_buffer_pool_t::reserve_locked(uint64_t page_id,
                                                            io_cache_access_t access,
                                                            bool wait_for_inflight,
                                                            std::unique_lock<std::mutex>& lock) {
  bool coalesced = false;
  while (true) {
    const auto resident = resident_.find(page_id);
    if (resident != resident_.end()) {
      resident->second.frame->inc_pin_count();
      const bool prefetched_hit =
          access == io_cache_access_t::DEMAND && resident->second.prefetched_unused;
      if (access == io_cache_access_t::DEMAND) {
        lru_.splice(lru_.begin(), lru_, resident->second.lru_position);
        ++stats_.demand_hits;
        if (resident->second.prefetched_unused) {
          resident->second.prefetched_unused = false;
          if (prefetched_unused_count_ == 0) {
            throw std::logic_error("LRU unused-prefetch count underflow");
          }
          --prefetched_unused_count_;
          prefetch_admission_credit_ = true;
          ++stats_.prefetch_used;
        }
      } else {
        ++stats_.prefetch_hits;
      }
      return {io_cache_reservation_state_t::HIT,
              page_id,
              resident->second.frame,
              resident->second.frame->get_data(),
              coalesced,
              false,
              access,
              prefetched_hit};
    }
    if (loading_.contains(page_id)) {
      coalesced = true;
      ++stats_.coalesced_misses;
      if (!wait_for_inflight) {
        return {
            io_cache_reservation_state_t::BYPASS, page_id, nullptr, nullptr, true, false, access};
      }
      completion_.wait(lock, [&]() { return !loading_.contains(page_id); });
      continue;
    }
    if (!is_prefetch(access)) {
      ++stats_.misses;
    } else {
      ++stats_.prefetch_misses;
    }
    if (capacity_pages_ == 0) {
      return {io_cache_reservation_state_t::BYPASS,
              page_id,
              nullptr,
              nullptr,
              coalesced,
              false,
              access};
    }

    chunk_t* frame = nullptr;
    bool evicted = false;
    if (!free_frames_.empty()) {
      frame = free_frames_.back();
      free_frames_.pop_back();
    } else {
      const auto select_victim = [&](bool require_unused_prefetch) -> chunk_t* {
        for (auto position = lru_.rbegin(); position != lru_.rend(); ++position) {
          auto victim = resident_.find(*position);
          if (victim->second.frame->get_pin_count() != 0 ||
              (require_unused_prefetch && !victim->second.prefetched_unused)) {
            continue;
          }
          chunk_t* selected = victim->second.frame;
          if (victim->second.prefetched_unused) {
            if (prefetched_unused_count_ == 0) {
              throw std::logic_error("LRU unused-prefetch count underflow");
            }
            --prefetched_unused_count_;
            ++stats_.prefetch_pollution_evictions;
          }
          const auto erase_position = std::next(position).base();
          lru_.erase(erase_position);
          resident_.erase(victim);
          ++stats_.evictions;
          return selected;
        }
        return nullptr;
      };
      if (is_prefetch(access)) {
        if (prefetched_unused_count_ != 0) {
          frame = select_victim(true);
        }
        const bool may_evict_demand =
            access == io_cache_access_t::GATEWAY_PREFETCH || prefetch_admission_credit_;
        if (frame == nullptr && may_evict_demand) {
          frame = select_victim(false);
          if (frame != nullptr) {
            if (access == io_cache_access_t::PREFETCH) {
              prefetch_admission_credit_ = false;
            }
            ++stats_.prefetch_demand_evictions;
          }
        }
      } else {
        frame = select_victim(false);
      }
      evicted = frame != nullptr;
    }
    if (frame == nullptr) {
      if (is_prefetch(access)) {
        ++stats_.prefetch_admission_rejections;
      } else {
        ++stats_.pinned_victim_failures;
      }
      return {io_cache_reservation_state_t::BYPASS,
              page_id,
              nullptr,
              nullptr,
              coalesced,
              false,
              access};
    }
    frame->set_chunk_id(disk_manager_t::make_chunk_id(chunk_type_, page_id));
    frame->set_chunk_state(chunk_state_t::free);
    frame->inc_pin_count();
    loading_.emplace(page_id, frame);
    return {io_cache_reservation_state_t::LOAD,
            page_id,
            frame,
            frame->get_data(),
            coalesced,
            evicted,
            access};
  }
}

io_cache_reservation_t io_lru_buffer_pool_t::try_reserve_hit(uint64_t page_id,
                                                             io_cache_access_t access) {
  std::lock_guard lock(mutex_);
  const auto resident = resident_.find(page_id);
  if (resident == resident_.end()) {
    return {io_cache_reservation_state_t::BYPASS, page_id, nullptr, nullptr, false, false, access};
  }
  resident->second.frame->inc_pin_count();
  const bool prefetched_hit =
      access == io_cache_access_t::DEMAND && resident->second.prefetched_unused;
  if (access == io_cache_access_t::DEMAND) {
    lru_.splice(lru_.begin(), lru_, resident->second.lru_position);
    ++stats_.demand_hits;
    if (resident->second.prefetched_unused) {
      resident->second.prefetched_unused = false;
      if (prefetched_unused_count_ == 0) {
        throw std::logic_error("LRU unused-prefetch count underflow");
      }
      --prefetched_unused_count_;
      prefetch_admission_credit_ = true;
      ++stats_.prefetch_used;
    }
  } else {
    ++stats_.prefetch_hits;
  }
  return {io_cache_reservation_state_t::HIT,
          page_id,
          resident->second.frame,
          resident->second.frame->get_data(),
          false,
          false,
          access,
          prefetched_hit};
}

void io_lru_buffer_pool_t::publish(const io_cache_reservation_t& reservation, bool success) {
  if (reservation.state != io_cache_reservation_state_t::LOAD || reservation.frame == nullptr) {
    return;
  }
  std::lock_guard lock(mutex_);
  const auto loading = loading_.find(reservation.page_id);
  if (loading == loading_.end() || loading->second != reservation.frame) {
    throw std::logic_error("LRU publication does not match an in-flight page");
  }
  loading_.erase(loading);
  if (success) {
    reservation.frame->set_chunk_state(chunk_state_t::resident);
    if (!is_prefetch(reservation.access)) {
      lru_.push_front(reservation.page_id);
    } else {
      lru_.push_back(reservation.page_id);
    }
    resident_.emplace(
        reservation.page_id,
        resident_entry_t{reservation.frame,
                         !is_prefetch(reservation.access) ? lru_.begin() : std::prev(lru_.end()),
                         is_prefetch(reservation.access)});
    if (is_prefetch(reservation.access)) {
      ++prefetched_unused_count_;
    }
  } else {
    reservation.frame->set_chunk_state(chunk_state_t::free);
    reservation.frame->dec_pin_count();
    free_frames_.push_back(reservation.frame);
    ++stats_.failed_loads;
  }
  completion_.notify_all();
}

void io_lru_buffer_pool_t::publish_many(std::span<const io_cache_reservation_t> reservations,
                                        bool success) {
  std::lock_guard lock(mutex_);
  for (const auto& reservation : reservations) {
    if (reservation.state != io_cache_reservation_state_t::LOAD || reservation.frame == nullptr) {
      continue;
    }
    const auto loading = loading_.find(reservation.page_id);
    if (loading == loading_.end() || loading->second != reservation.frame) {
      throw std::logic_error("LRU publication does not match an in-flight page");
    }
    loading_.erase(loading);
    if (success) {
      reservation.frame->set_chunk_state(chunk_state_t::resident);
      if (!is_prefetch(reservation.access)) {
        lru_.push_front(reservation.page_id);
      } else {
        lru_.push_back(reservation.page_id);
      }
      resident_.emplace(
          reservation.page_id,
          resident_entry_t{reservation.frame,
                           !is_prefetch(reservation.access) ? lru_.begin() : std::prev(lru_.end()),
                           is_prefetch(reservation.access)});
      if (is_prefetch(reservation.access)) {
        ++prefetched_unused_count_;
      }
    } else {
      reservation.frame->set_chunk_state(chunk_state_t::free);
      reservation.frame->dec_pin_count();
      free_frames_.push_back(reservation.frame);
      ++stats_.failed_loads;
    }
  }
  completion_.notify_all();
}

void io_lru_buffer_pool_t::release(const io_cache_reservation_t& reservation) {
  if (reservation.frame == nullptr || reservation.state == io_cache_reservation_state_t::BYPASS) {
    return;
  }
  std::lock_guard lock(mutex_);
  if (reservation.frame->get_pin_count() == 0) {
    throw std::logic_error("LRU frame pin count underflow");
  }
  reservation.frame->dec_pin_count();
}

void io_lru_buffer_pool_t::release_many(std::span<const io_cache_reservation_t> reservations) {
  std::lock_guard lock(mutex_);
  for (const auto& reservation : reservations) {
    if (reservation.frame == nullptr || reservation.state == io_cache_reservation_state_t::BYPASS) {
      continue;
    }
    if (reservation.frame->get_pin_count() == 0) {
      throw std::logic_error("LRU frame pin count underflow");
    }
    reservation.frame->dec_pin_count();
  }
}

void io_lru_buffer_pool_t::clear() {
  std::lock_guard lock(mutex_);
  if (!loading_.empty()) {
    throw std::logic_error("cannot clear LRU while page loads are in flight");
  }
  for (const auto& [page_id, entry] : resident_) {
    (void)page_id;
    if (entry.frame->get_pin_count() != 0) {
      throw std::logic_error("cannot clear LRU while a resident page is pinned");
    }
    entry.frame->set_chunk_state(chunk_state_t::free);
  }
  resident_.clear();
  lru_.clear();
  free_frames_.clear();
  for (size_t index = 0; index < capacity_pages_; ++index) {
    free_frames_.push_back(frames_->get_chunk(chunk_type_, index));
  }
  prefetched_unused_count_ = 0;
  prefetch_admission_credit_ = true;
  stats_ = {};
}

void io_lru_buffer_pool_t::reset_capacity(size_t capacity_pages) {
  if (capacity_pages == 0) {
    throw std::invalid_argument("LRU capacity reset requires at least one page");
  }
  chunk_buf_pool_options_t options;
  options.reserved_chunk_counts[static_cast<size_t>(chunk_type_)] = capacity_pages;
  auto frames = chunk_buf_pool_t::chunk_buf_pool_init(options);
  std::vector<chunk_t*> free_frames;
  free_frames.reserve(capacity_pages);
  for (size_t index = 0; index < capacity_pages; ++index) {
    free_frames.push_back(frames->get_chunk(chunk_type_, index));
  }

  std::lock_guard lock(mutex_);
  if (!loading_.empty()) {
    throw std::logic_error("cannot reset LRU capacity while page loads are in flight");
  }
  for (const auto& [page_id, entry] : resident_) {
    (void)page_id;
    if (entry.frame->get_pin_count() != 0) {
      throw std::logic_error("cannot reset LRU capacity while a resident page is pinned");
    }
  }
  capacity_pages_ = capacity_pages;
  frames_ = std::move(frames);
  free_frames_ = std::move(free_frames);
  std::unordered_map<uint64_t, resident_entry_t>().swap(resident_);
  std::unordered_map<uint64_t, chunk_t*>().swap(loading_);
  resident_.reserve(capacity_pages);
  loading_.reserve(capacity_pages);
  lru_.clear();
  prefetched_unused_count_ = 0;
  prefetch_admission_credit_ = true;
  stats_ = {};
}

io_lru_buffer_pool_stats_t io_lru_buffer_pool_t::stats() const {
  std::lock_guard lock(mutex_);
  return stats_;
}

io_size_specific_lru_t::io_size_specific_lru_t(
    const std::array<size_t, disk_manager_t::k_chunk_levels>& capacities) {
  for (size_t level = 0; level < capacities.size(); ++level) {
    if (capacities[level] != 0) {
      pools_[level] = std::make_unique<io_lru_buffer_pool_t>(capacities[level],
                                                             static_cast<chunk_type_t>(level));
    }
  }
}

io_lru_buffer_pool_t* io_size_specific_lru_t::pool(chunk_type_t type) noexcept {
  return pools_[static_cast<size_t>(type)].get();
}

const io_lru_buffer_pool_t* io_size_specific_lru_t::pool(chunk_type_t type) const noexcept {
  return pools_[static_cast<size_t>(type)].get();
}

uint64_t io_size_specific_lru_t::resident_bytes() const noexcept {
  uint64_t bytes = 0;
  for (const auto& pool : pools_) {
    if (pool != nullptr) {
      bytes += pool->resident_bytes();
    }
  }
  return bytes;
}

} // namespace powerlaw_ann
