#include "index/delta_store.h"

#include "index/hnsw.h"

#include <fstream>
#include <stdexcept>

namespace powerlaw_ann {

static constexpr uint8_t k_insert = 1;

delta_store_t::delta_store_t(int dim, size_t max_buffer) : dim_(dim), max_buffer_(max_buffer) {}

void delta_store_t::add(const float* vec, uint64_t label) {
  std::lock_guard<std::mutex> lk(mu_);
  delta_record_t rec;
  rec.op = delta_record_t::op_t::insert;
  rec.label = label;
  rec.vec.assign(vec, vec + dim_);
  records_.push_back(std::move(rec));
}

bool delta_store_t::is_full() const {
  std::lock_guard<std::mutex> lk(mu_);
  return records_.size() >= max_buffer_;
}

bool delta_store_t::empty() const {
  std::lock_guard<std::mutex> lk(mu_);
  return records_.empty();
}

size_t delta_store_t::size() const {
  std::lock_guard<std::mutex> lk(mu_);
  return records_.size();
}

void delta_store_t::flush_to_file(const std::string& path) {
  std::lock_guard<std::mutex> lk(mu_);
  std::ofstream out(path, std::ios::binary | std::ios::app);
  if (!out) {
    throw std::runtime_error("delta_store_t: cannot open " + path);
  }

  for (const auto& rec : records_) {
    uint8_t op = k_insert;
    out.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out.write(reinterpret_cast<const char*>(&rec.label), sizeof(rec.label));
    out.write(reinterpret_cast<const char*>(rec.vec.data()), sizeof(float) * dim_);
  }
}

void delta_store_t::apply_and_clear(hnsw_index_t& index) {
  std::lock_guard<std::mutex> lk(mu_);
  for (const auto& rec : records_) {
    index.insert(rec.vec.data(), rec.label);
  }
  records_.clear();
}

void delta_store_t::replay_file(const std::string& path, hnsw_index_t& index) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return;
  }

  const int dim = index.dim();
  std::vector<float> buf(dim);

  while (in.peek() != EOF) {
    uint8_t op;
    uint64_t label;
    in.read(reinterpret_cast<char*>(&op), sizeof(op));
    in.read(reinterpret_cast<char*>(&label), sizeof(label));
    in.read(reinterpret_cast<char*>(buf.data()), sizeof(float) * dim);
    if (!in) {
      break;
    }
    if (op == k_insert) {
      index.insert(buf.data(), label);
    }
  }
}

} // namespace powerlaw_ann
