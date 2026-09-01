#include "index/disk_index.h"

#include "index/hnsw.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace powerlaw_ann {

static void write_u64(std::ostream& out, uint64_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_u32(std::ostream& out, uint32_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static uint64_t read_u64(std::istream& in) {
  uint64_t value;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

static uint32_t read_u32(std::istream& in) {
  uint32_t value;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

void disk_index_t::save(const hnsw_index_t& index, const std::string& path) {
  std::filesystem::path file_path(path);
  if (file_path.has_parent_path()) {
    std::filesystem::create_directories(file_path.parent_path());
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("disk_index_t: cannot open for write: " + path);
  }

  write_u64(out, k_magic);
  write_u32(out, k_version);
  index.save(out);

  if (!out) {
    throw std::runtime_error("disk_index_t: write failed: " + path);
  }
}

void disk_index_t::load(hnsw_index_t& index, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("disk_index_t: cannot open: " + path);
  }

  const uint64_t magic = read_u64(in);
  const uint32_t version = read_u32(in);

  if (magic != k_magic) {
    throw std::runtime_error("disk_index_t: bad magic in " + path);
  }
  if (version != k_version) {
    throw std::runtime_error("disk_index_t: unsupported version " + std::to_string(version));
  }

  index.load(in);
  if (!in) {
    throw std::runtime_error("disk_index_t: read failed: " + path);
  }
}

bool disk_index_t::exists(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  const uint64_t magic = read_u64(in);
  return magic == k_magic;
}

} // namespace powerlaw_ann
