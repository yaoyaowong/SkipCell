#include "storage/disk_manager.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

// Convert a chunk type to the array index used by metadata and file handles.
size_t chunk_index(chunk_type_t type) {
  const auto index = static_cast<size_t>(type);
  if (index >= disk_manager_t::k_chunk_levels) {
    throw std::out_of_range("invalid chunk type");
  }
  return index;
}

// Throw a std::runtime_error that includes errno text.
void throw_errno(const std::string& action, const std::string& path) {
  throw std::runtime_error(action + " failed for " + path + ": " + std::strerror(errno));
}

// Create parent directories when the prefix includes a directory component.
void ensure_parent_dir(const std::string& path) {
  const std::filesystem::path file_path(path);
  const auto parent = file_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

// Create or truncate one file before opening it for bidirectional I/O.
void prepare_file(const std::string& path, bool create_if_missing, bool truncate) {
  ensure_parent_dir(path);
  if (truncate || !std::filesystem::exists(path)) {
    if (!create_if_missing && !std::filesystem::exists(path)) {
      throw std::runtime_error("missing file: " + path);
    }
    std::ofstream out(path, std::ios::binary | (truncate ? std::ios::trunc : std::ios::app));
    if (!out.is_open()) {
      throw std::runtime_error("failed to create file: " + path);
    }
  }
}

// Open a binary fstream with internal buffering disabled.
void open_stream(std::fstream& stream, const std::string& path) {
  stream.rdbuf()->pubsetbuf(nullptr, 0);
  stream.open(path, std::ios::binary | std::ios::in | std::ios::out);
  if (!stream.is_open()) {
    throw std::runtime_error("failed to open file: " + path);
  }
}

// Resize a file to at least the requested byte length.
void reserve_file_space(const std::string& path, uint64_t bytes) {
  const int fd = ::open(path.c_str(), O_RDWR);
  if (fd < 0) {
    throw_errno("open", path);
  }

#if defined(__APPLE__)
  fstore_t store;
  std::memset(&store, 0, sizeof(store));
  store.fst_flags = F_ALLOCATECONTIG;
  store.fst_posmode = F_PEOFPOSMODE;
  store.fst_offset = 0;
  store.fst_length = static_cast<off_t>(bytes);
  if (::fcntl(fd, F_PREALLOCATE, &store) != 0) {
    store.fst_flags = F_ALLOCATEALL;
    (void) ::fcntl(fd, F_PREALLOCATE, &store);
  }
  if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    const int saved_errno = errno;
    ::close(fd);
    errno = saved_errno;
    throw_errno("ftruncate", path);
  }
#elif defined(__linux__)
  const int fallocate_rc = ::posix_fallocate(fd, 0, static_cast<off_t>(bytes));
  if (fallocate_rc != 0) {
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      const int saved_errno = errno;
      ::close(fd);
      errno = saved_errno;
      throw_errno("ftruncate", path);
    }
  }
#else
  if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    const int saved_errno = errno;
    ::close(fd);
    errno = saved_errno;
    throw_errno("ftruncate", path);
  }
#endif

  if (::close(fd) != 0) {
    throw_errno("close", path);
  }
}

// Flush file contents to stable storage with the strongest platform primitive available.
void sync_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDWR);
  if (fd < 0) {
    throw_errno("open", path);
  }

#if defined(__APPLE__)
  if (::fcntl(fd, F_FULLFSYNC) != 0 && ::fsync(fd) != 0) {
#else
  if (::fsync(fd) != 0) {
#endif
    const int saved_errno = errno;
    ::close(fd);
    errno = saved_errno;
    throw_errno("sync", path);
  }

  if (::close(fd) != 0) {
    throw_errno("close", path);
  }
}

// Check whether Ubuntu-style transparent huge pages are available.
bool linux_huge_pages_available() {
#if defined(__linux__)
  std::ifstream in("/sys/kernel/mm/transparent_hugepage/enabled");
  std::string value;
  std::getline(in, value);
  return value.find("[always]") != std::string::npos ||
         value.find("[madvise]") != std::string::npos;
#else
  return false;
#endif
}

// Check whether an fstream is currently open.
bool stream_is_open(const std::fstream& stream) { return stream.is_open(); }

// Open a disk manager with the given file prefix.
disk_manager_t::disk_manager_t(const std::string& prefix, const disk_manager_options_t& options) {
  open(prefix, options);
}

// Close open file handles.
disk_manager_t::~disk_manager_t() {
  try {
    close();
  } catch (...) {
  }
}

// Move construct a disk manager.
disk_manager_t::disk_manager_t(disk_manager_t&& other) noexcept { *this = std::move(other); }

// Move assign a disk manager.
disk_manager_t& disk_manager_t::operator=(disk_manager_t&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  try {
    close();
  } catch (...) {
  }

  prefix_ = std::move(other.prefix_);
  files_ = std::move(other.files_);
  meta_ = other.meta_;
  options_ = other.options_;
  meta_file_ = std::move(other.meta_file_);
  chunk_files_ = std::move(other.chunk_files_);
  chunk_read_fds_ = other.chunk_read_fds_;
  other.chunk_read_fds_.fill(-1);
  free_lists_ = std::move(other.free_lists_);
  other.meta_ = disk_manager_meta_t{};
  return *this;
}

// Open a disk manager with the given file prefix.
void disk_manager_t::open(const std::string& prefix, const disk_manager_options_t& options) {
  close();
  prefix_ = prefix;
  options_ = options;
  files_ = build_file_names(prefix);

  prepare_file(files_.meta, options_.create_if_missing, options_.truncate);
  open_stream(meta_file_, files_.meta);

  for (size_t i = 0; i < k_chunk_levels; ++i) {
    prepare_file(files_.chunks[i], options_.create_if_missing, options_.truncate);
    open_chunk_file(static_cast<chunk_type_t>(i));
  }

  load_meta();
  configure_huge_page_hint();
}

// Flush metadata and close all files.
void disk_manager_t::close() {
  if (meta_file_.is_open()) {
    flush_meta();
    meta_file_.close();
  }

  for (auto& file : chunk_files_) {
    if (file.is_open()) {
      file.close();
    }
  }
  for (int& fd : chunk_read_fds_) {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
}

// Check whether all files are open.
bool disk_manager_t::is_open() const {
  return meta_file_.is_open() &&
         std::all_of(chunk_files_.begin(), chunk_files_.end(), stream_is_open) &&
         std::all_of(chunk_read_fds_.begin(), chunk_read_fds_.end(),
                     [](int fd) { return fd >= 0; });
}

// Allocate one chunk page from a chunk file.
chunk_id_t disk_manager_t::allocate_chunk(chunk_type_t type) {
  const size_t index = chunk_index(type);
  chunk_no_t chunk_no = 0;

  if (!free_lists_[index].empty()) {
    chunk_no = free_lists_[index].back();
    free_lists_[index].pop_back();
    meta_.free_counts[index] = free_lists_[index].size();
    return make_chunk_id(type, chunk_no);
  }

  chunk_no = meta_.chunk_counts[index]++;
  const uint64_t required_size = (chunk_no + 1) * static_cast<uint64_t>(chunk_size(type));
  reserve_file_space(files_.chunks[index], required_size);
  return make_chunk_id(type, chunk_no);
}

// Release one chunk page for later reuse.
void disk_manager_t::delete_chunk(chunk_id_t chunk_id) {
  validate_chunk_id(chunk_id);

  const chunk_type_t type = chunk_type(chunk_id);
  const size_t index = chunk_index(type);
  const chunk_no_t no = chunk_no(chunk_id);
  free_lists_[index].push_back(no);
  meta_.free_counts[index] = free_lists_[index].size();
}

// Write an entire chunk page.
void disk_manager_t::write_chunk(chunk_id_t chunk_id, const void* data, size_t size) {
  if (data == nullptr) {
    throw std::invalid_argument("write_chunk data is null");
  }
  validate_chunk_id(chunk_id);

  const page_addr_t addr = decode_chunk_id(chunk_id);
  if (size > addr.size) {
    throw std::invalid_argument("write size exceeds chunk size");
  }

  auto& file = chunk_files_[chunk_index(addr.type)];
  file.clear();
  file.seekp(static_cast<std::streamoff>(addr.offset), std::ios::beg);
  file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!file.good()) {
    throw std::runtime_error("failed to write chunk");
  }

  if (size < addr.size) {
    static constexpr std::array<char, 4096> k_zero_page{};
    size_t remaining = addr.size - size;
    while (remaining > 0) {
      const size_t bytes = std::min(remaining, k_zero_page.size());
      file.write(k_zero_page.data(), static_cast<std::streamsize>(bytes));
      remaining -= bytes;
    }
  }

  file.flush();
  if (options_.sync_on_write) {
    sync_file(files_.chunks[chunk_index(addr.type)]);
  }
}

// Allocate and write one chunk page.
chunk_id_t disk_manager_t::write_new_chunk(chunk_type_t type, const void* data, size_t size) {
  const chunk_id_t chunk_id = allocate_chunk(type);
  try {
    write_chunk(chunk_id, data, size);
  } catch (...) {
    delete_chunk(chunk_id);
    throw;
  }
  return chunk_id;
}

// Load a chunk page into a caller-owned buffer.
void disk_manager_t::load_chunk(chunk_id_t chunk_id, void* dst) {
  if (dst == nullptr) {
    throw std::invalid_argument("load_chunk destination is null");
  }
  validate_chunk_id(chunk_id);

  const page_addr_t addr = decode_chunk_id(chunk_id);
  const int fd = chunk_read_fds_[chunk_index(addr.type)];
  if (fd < 0) {
    throw std::runtime_error("chunk read descriptor is not open");
  }

  size_t total = 0;
  while (total < addr.size) {
    const ssize_t bytes_read = ::pread(fd, static_cast<char*>(dst) + total, addr.size - total,
                                       static_cast<off_t>(addr.offset + total));
    if (bytes_read > 0) {
      total += static_cast<size_t>(bytes_read);
      continue;
    }
    if (bytes_read == 0) {
      std::memset(static_cast<char*>(dst) + total, 0, addr.size - total);
      return;
    }
    if (errno == EINTR) {
      continue;
    }
    throw_errno("pread", files_.chunks[chunk_index(addr.type)]);
  }
}

// Get allocated chunk count for a chunk file.
uint64_t disk_manager_t::chunk_count(chunk_type_t type) const {
  return meta_.chunk_counts[chunk_index(type)];
}

// Get reusable chunk count for a chunk file.
uint64_t disk_manager_t::free_chunk_count(chunk_type_t type) const {
  return meta_.free_counts[chunk_index(type)];
}

// Encode chunk type and chunk number into chunk_id_t.
chunk_id_t disk_manager_t::make_chunk_id(chunk_type_t type, chunk_no_t chunk_no) {
  const size_t index = chunk_index(type);
  if ((chunk_no & ~k_chunk_no_mask) != 0) {
    throw std::out_of_range("chunk number exceeds chunk id capacity");
  }
  return (static_cast<chunk_id_t>(index) << k_chunk_no_bits) | chunk_no;
}

// Decode a chunk id into physical address information.
page_addr_t disk_manager_t::decode_chunk_id(chunk_id_t chunk_id) {
  const chunk_type_t type = chunk_type(chunk_id);
  const chunk_no_t no = chunk_no(chunk_id);
  const uint32_t size = chunk_size(type);
  if (no > std::numeric_limits<uint64_t>::max() / size) {
    throw std::overflow_error("chunk offset overflows uint64_t");
  }
  return page_addr_t{type, no, no * static_cast<uint64_t>(size), size};
}

// Extract chunk type from a chunk id.
chunk_type_t disk_manager_t::chunk_type(chunk_id_t chunk_id) {
  const auto index = static_cast<uint8_t>(chunk_id >> k_chunk_no_bits);
  if (index >= k_chunk_levels) {
    throw std::out_of_range("invalid encoded chunk type");
  }
  return static_cast<chunk_type_t>(index);
}

// Extract chunk number from a chunk id.
chunk_no_t disk_manager_t::chunk_no(chunk_id_t chunk_id) { return chunk_id & k_chunk_no_mask; }

// Get chunk byte size for a type.
uint32_t disk_manager_t::chunk_size(chunk_type_t type) { return k_chunk_sizes[chunk_index(type)]; }

// Build file names from a common prefix.
disk_manager_files_t disk_manager_t::build_file_names(const std::string& prefix) {
  return disk_manager_files_t{
      prefix + "_meta.db",
      {
          prefix + "_chunk_0.db",
          prefix + "_chunk_1.db",
          prefix + "_chunk_2.db",
          prefix + "_chunk_3.db",
      },
  };
}

// Load or initialize metadata.
void disk_manager_t::load_meta() {
  free_lists_ = {};

  const auto file_size = std::filesystem::file_size(files_.meta);
  if (options_.truncate || file_size < sizeof(disk_manager_meta_t)) {
    meta_ = disk_manager_meta_t{};
    flush_meta();
    return;
  }

  meta_file_.clear();
  meta_file_.seekg(0, std::ios::beg);
  meta_file_.read(reinterpret_cast<char*>(&meta_), sizeof(meta_));
  if (!meta_file_.good()) {
    throw std::runtime_error("failed to read disk manager metadata");
  }
  if (meta_.magic != disk_manager_meta_t::k_magic ||
      meta_.version != disk_manager_meta_t::k_version ||
      meta_.chunk_levels != disk_manager_meta_t::k_chunk_levels) {
    throw std::runtime_error("invalid disk manager metadata");
  }

  for (size_t i = 0; i < k_chunk_levels; ++i) {
    free_lists_[i].resize(meta_.free_counts[i]);
    if (!free_lists_[i].empty()) {
      meta_file_.read(reinterpret_cast<char*>(free_lists_[i].data()),
                      static_cast<std::streamsize>(free_lists_[i].size() * sizeof(chunk_no_t)));
      if (!meta_file_.good()) {
        throw std::runtime_error("failed to read free chunk list");
      }
    }
    for (const chunk_no_t no : free_lists_[i]) {
      if (no >= meta_.chunk_counts[i]) {
        throw std::runtime_error("invalid free chunk number in metadata");
      }
    }
    meta_.free_counts[i] = free_lists_[i].size();
  }
}

// Persist metadata to the meta file.
void disk_manager_t::flush_meta() {
  if (!meta_file_.is_open()) {
    return;
  }

  uint64_t meta_bytes = sizeof(disk_manager_meta_t);
  for (size_t i = 0; i < k_chunk_levels; ++i) {
    meta_.free_counts[i] = free_lists_[i].size();
    meta_bytes += free_lists_[i].size() * sizeof(chunk_no_t);
  }

  meta_file_.clear();
  meta_file_.seekp(0, std::ios::beg);
  meta_file_.write(reinterpret_cast<const char*>(&meta_), sizeof(meta_));
  for (const auto& free_list : free_lists_) {
    if (!free_list.empty()) {
      meta_file_.write(reinterpret_cast<const char*>(free_list.data()),
                       static_cast<std::streamsize>(free_list.size() * sizeof(chunk_no_t)));
    }
  }
  if (!meta_file_.good()) {
    throw std::runtime_error("failed to write disk manager metadata");
  }

  meta_file_.flush();
  std::filesystem::resize_file(files_.meta, meta_bytes);
  if (options_.sync_on_write) {
    sync_file(files_.meta);
  }
}

// Open one chunk file.
void disk_manager_t::open_chunk_file(chunk_type_t type) {
  const size_t index = chunk_index(type);
  open_stream(chunk_files_[index], files_.chunks[index]);
  chunk_read_fds_[index] = ::open(files_.chunks[index].c_str(), O_RDONLY);
  if (chunk_read_fds_[index] < 0) {
    throw_errno("open", files_.chunks[index]);
  }
}

// Apply OS-specific huge page hints when available.
void disk_manager_t::configure_huge_page_hint() {
  if (options_.huge_page_policy == huge_page_policy_t::disabled) {
    meta_.huge_page_enabled = 0;
    return;
  }

  const bool available = linux_huge_pages_available();
  if (!available && options_.huge_page_policy == huge_page_policy_t::require) {
    throw std::runtime_error("2MB huge page support is not available on this platform");
  }

  meta_.huge_page_enabled = available ? 1 : 0;
}

// Check whether a chunk id points to an allocated slot.
void disk_manager_t::validate_chunk_id(chunk_id_t chunk_id) const {
  const chunk_type_t type = chunk_type(chunk_id);
  const size_t index = chunk_index(type);
  const chunk_no_t no = chunk_no(chunk_id);
  if (no >= meta_.chunk_counts[index]) {
    throw std::out_of_range("chunk id is outside allocated range");
  }
  const auto& free_list = free_lists_[index];
  if (std::find(free_list.begin(), free_list.end(), no) != free_list.end()) {
    throw std::runtime_error("chunk id points to a deleted chunk");
  }
}
