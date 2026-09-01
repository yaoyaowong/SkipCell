#include "index/powerlaw_ann.h"

#include "index/delta_store.h"
#include "index/disk_index.h"
#include "index/hnsw.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>

using namespace powerlaw_ann;

std::string powerlaw_ann_t::db_path() const {
  return cfg_.workspace + "/" + cfg_.dataset_name + "/" + cfg_.dataset_name + ".db";
}

std::string powerlaw_ann_t::delta_path() const {
  return cfg_.workspace + "/" + cfg_.dataset_name + "/" + cfg_.dataset_name + ".delta";
}

void powerlaw_ann_t::ensure_workspace() const {
  std::filesystem::create_directories(cfg_.workspace + "/" + cfg_.dataset_name);
}

powerlaw_ann_t::powerlaw_ann_t(const config_t& cfg) : cfg_(cfg) {
  index_ =
      std::make_unique<hnsw_index_t>(cfg_.dim, cfg_.max_elements, cfg_.m, cfg_.ef_construction);
  delta_ = std::make_unique<delta_store_t>(cfg_.dim, cfg_.delta_buffer);
  start_bg();
}

powerlaw_ann_t::~powerlaw_ann_t() {
  stop_bg_.store(true);
  bg_cv_.notify_all();
  if (bg_thread_.joinable()) {
    bg_thread_.join();
  }
}

void powerlaw_ann_t::start_bg() {
  bg_thread_ = std::thread([this] { bg_loop(); });
}

void powerlaw_ann_t::bg_loop() {
  while (true) {
    std::unique_lock<std::mutex> lk(bg_mu_);
    bg_cv_.wait_for(lk, std::chrono::seconds(2),
                    [this] { return stop_bg_.load() || delta_->is_full(); });

    if (stop_bg_.load() && delta_->empty()) {
      break;
    }

    if (!delta_->empty()) {
      ensure_workspace();
      delta_->flush_to_file(delta_path());
      delta_->apply_and_clear(*index_);

      std::error_code ec;
      std::filesystem::remove(delta_path(), ec);
    }

    if (stop_bg_.load()) {
      break;
    }
  }
}

void powerlaw_ann_t::insert(const float* vec, uint64_t label) {
  delta_->add(vec, label);
  if (delta_->is_full()) {
    bg_cv_.notify_one();
  }
}

std::vector<std::pair<float, uint64_t>> powerlaw_ann_t::search(const float* query, int k) const {
  return index_->search(query, k, cfg_.ef_search);
}

void powerlaw_ann_t::build(const float* data, const uint64_t* labels, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    index_->insert(data + i * cfg_.dim, labels[i]);
  }
}

void powerlaw_ann_t::save() {
  consolidate();
  ensure_workspace();
  disk_index_t::save(*index_, db_path());
}

void powerlaw_ann_t::load() {
  if (!disk_index_t::exists(db_path())) {
    throw std::runtime_error("powerlaw_ann_t::load: db not found at " + db_path());
  }

  index_ =
      std::make_unique<hnsw_index_t>(cfg_.dim, cfg_.max_elements, cfg_.m, cfg_.ef_construction);
  disk_index_t::load(*index_, db_path());
  delta_store_t::replay_file(delta_path(), *index_);
}

void powerlaw_ann_t::consolidate() {
  if (!delta_->empty()) {
    std::lock_guard<std::mutex> lk(bg_mu_);
    delta_->apply_and_clear(*index_);
    std::error_code ec;
    std::filesystem::remove(delta_path(), ec);
  }
}

size_t powerlaw_ann_t::size() const { return index_->size() + delta_->size(); }
