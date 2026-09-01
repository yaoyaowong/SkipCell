#include "common/utils.h"

#include "common/arch_compat.h"

#include <stdio.h>

#ifdef EXEC_ENV_OLS
#include "storage/aligned_file_reader.h"
#endif

#ifdef EXEC_ENV_OLS
const uint32_t MAX_REQUEST_SIZE = 1024 * 1024 * 1024; // 1GB
const uint32_t MAX_SIMULTANEOUS_READ_REQUESTS = 128;
#endif

#ifdef _WINDOWS
#include <intrin.h>

// Taken from:
// https://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/
bool cpu_has_avx_support() {
  bool avx_supported = false;

  // Checking for AVX requires 3 things:
  // 1) CPUID indicates that the OS uses XSAVE and XRSTORE
  //     instructions (allowing saving YMM registers on context
  //     switch)
  // 2) CPUID indicates support for AVX
  // 3) XGETBV indicates the AVX registers will be saved and
  //     restored on context switch
  //
  // Note that XGETBV is only available on 686 or later CPUs, so
  // the instruction needs to be conditionally run.
  int cpu_info[4];
  __cpuid(cpu_info, 1);

  bool os_uses_xsave_xrstore = cpu_info[2] & (1 << 27) || false;
  bool cpu_avx_support = cpu_info[2] & (1 << 28) || false;

  if (os_uses_xsave_xrstore && cpu_avx_support) {
    // Check if the OS will save the YMM registers
    unsigned long long xcr_feature_mask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    avx_supported = (xcr_feature_mask & 0x6) || false;
  }

  return avx_supported;
}

bool cpu_has_avx2_support() {
  int cpu_info[4];
  __cpuid(cpu_info, 0);
  int n = cpu_info[0];
  if (n >= 7) {
    __cpuidex(cpu_info, 7, 0);
    static int avx2Mask = 0x20;
    return (cpu_info[1] & avx2Mask) > 0;
  }
  return false;
}

bool avx_supported_cpu = cpu_has_avx_support();
bool avx2_supported_cpu = cpu_has_avx2_support();

#else

bool avx2_supported_cpu = POWERLAWANN_ARCH_X86;
bool avx_supported_cpu = false;
#endif

namespace powerlaw_ann {

void block_convert(std::ofstream& writr, std::ifstream& readr, float* read_buf, size_t npts,
                   size_t ndims) {
  readr.read((char*) read_buf, npts * ndims * sizeof(float));
  uint32_t ndims_u32 = (uint32_t) ndims;
#pragma omp parallel for
  for (int64_t i = 0; i < (int64_t) npts; i++) {
    float norm_pt = std::numeric_limits<float>::epsilon();
    for (uint32_t dim = 0; dim < ndims_u32; dim++) {
      norm_pt += *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
    }
    norm_pt = std::sqrt(norm_pt);
    for (uint32_t dim = 0; dim < ndims_u32; dim++) {
      *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
    }
  }
  writr.write((char*) read_buf, npts * ndims * sizeof(float));
}

void normalize_data_file(const std::string& input_file_name, const std::string& output_file_name) {
  std::ifstream readr(input_file_name, std::ios::binary);
  std::ofstream writr(output_file_name, std::ios::binary);

  int npts_s32, ndims_s32;
  readr.read((char*) &npts_s32, sizeof(int32_t));
  readr.read((char*) &ndims_s32, sizeof(int32_t));

  writr.write((char*) &npts_s32, sizeof(int32_t));
  writr.write((char*) &ndims_s32, sizeof(int32_t));

  size_t npts = (size_t) npts_s32;
  size_t ndims = (size_t) ndims_s32;
  powerlaw_ann::cout << "Normalizing FLOAT vectors in file: " << input_file_name << std::endl;
  powerlaw_ann::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

  size_t blk_size = 131072;
  size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
  powerlaw_ann::cout << "# blks: " << nblks << std::endl;

  float* read_buf = new float[npts * ndims];
  for (size_t i = 0; i < nblks; i++) {
    size_t cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(writr, readr, read_buf, cblk_size, ndims);
  }
  delete[] read_buf;

  powerlaw_ann::cout << "Wrote normalized points to file: " << output_file_name << std::endl;
}

double calculate_recall(uint32_t num_queries, uint32_t* gold_std, float* gs_dist, uint32_t dim_gs,
                        uint32_t* our_results, uint32_t dim_or, uint32_t recall_at) {
  double total_recall = 0;
  std::set<uint32_t> gt, res;

  for (size_t i = 0; i < num_queries; i++) {
    gt.clear();
    res.clear();
    uint32_t* gt_vec = gold_std + dim_gs * i;
    uint32_t* res_vec = our_results + dim_or * i;
    size_t tie_breaker = recall_at;
    if (gs_dist != nullptr) {
      tie_breaker = recall_at - 1;
      float* gt_dist_vec = gs_dist + dim_gs * i;
      while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
        tie_breaker++;
    }

    gt.insert(gt_vec, gt_vec + tie_breaker);
    res.insert(res_vec,
               res_vec + recall_at); // change to recall_at for recall k@k
                                     // or dim_or for k@dim_or
    uint32_t cur_recall = 0;
    for (auto& v : gt) {
      if (res.find(v) != res.end()) {
        cur_recall++;
      }
    }
    total_recall += cur_recall;
  }
  return total_recall / (num_queries) * (100.0 / recall_at);
}

double calculate_recall(uint32_t num_queries, uint32_t* gold_std, float* gs_dist, uint32_t dim_gs,
                        uint32_t* our_results, uint32_t dim_or, uint32_t recall_at,
                        const tsl::robin_set<uint32_t>& active_tags) {
  double total_recall = 0;
  std::set<uint32_t> gt, res;
  bool printed = false;
  for (size_t i = 0; i < num_queries; i++) {
    gt.clear();
    res.clear();
    uint32_t* gt_vec = gold_std + dim_gs * i;
    uint32_t* res_vec = our_results + dim_or * i;
    size_t tie_breaker = recall_at;
    uint32_t active_points_count = 0;
    uint32_t cur_counter = 0;
    while (active_points_count < recall_at && cur_counter < dim_gs) {
      if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
        active_points_count++;
      }
      cur_counter++;
    }
    if (active_tags.empty())
      cur_counter = recall_at;

    if ((active_points_count < recall_at && !active_tags.empty()) && !printed) {
      powerlaw_ann::cout << "Warning: Couldn't find enough closest neighbors "
                         << active_points_count << "/" << recall_at
                         << " from "
                            "truthset for query # "
                         << i << ". Will result in under-reported value of recall." << std::endl;
      printed = true;
    }
    if (gs_dist != nullptr) {
      tie_breaker = cur_counter - 1;
      float* gt_dist_vec = gs_dist + dim_gs * i;
      while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
        tie_breaker++;
    }

    gt.insert(gt_vec, gt_vec + tie_breaker);
    res.insert(res_vec, res_vec + recall_at);
    uint32_t cur_recall = 0;
    for (auto& v : res) {
      if (gt.find(v) != gt.end()) {
        cur_recall++;
      }
    }
    total_recall += cur_recall;
  }
  return ((double) (total_recall / (num_queries))) * ((double) (100.0 / recall_at));
}

double calculate_range_search_recall(uint32_t num_queries,
                                     std::vector<std::vector<uint32_t>>& groundtruth,
                                     std::vector<std::vector<uint32_t>>& our_results) {
  double total_recall = 0;
  std::set<uint32_t> gt, res;

  for (size_t i = 0; i < num_queries; i++) {
    gt.clear();
    res.clear();

    gt.insert(groundtruth[i].begin(), groundtruth[i].end());
    res.insert(our_results[i].begin(), our_results[i].end());
    uint32_t cur_recall = 0;
    for (auto& v : gt) {
      if (res.find(v) != res.end()) {
        cur_recall++;
      }
    }
    if (gt.size() != 0)
      total_recall += ((100.0 * cur_recall) / gt.size());
    else
      total_recall += 100;
  }
  return total_recall / (num_queries);
}

#ifdef EXEC_ENV_OLS
void get_bin_metadata(aligned_file_reader_t& reader, size_t& npts, size_t& ndim, size_t offset) {
  std::vector<aligned_read_t> read_reqs;
  aligned_read_t read_req;
  uint32_t buf[2]; // npts/ndim are uint32_ts.

  read_req.buf = buf;
  read_req.offset = offset;
  read_req.len = 2 * sizeof(uint32_t);
  read_reqs.push_back(read_req);

  aligned_io_context_t& ctx = reader.get_ctx();
  reader.read(read_reqs, ctx); // synchronous
  if ((*(ctx.request_statuses_))[0] == aligned_io_context_t::READ_SUCCESS) {
    npts = buf[0];
    ndim = buf[1];
    powerlaw_ann::cout << "File has: " << npts << " points, " << ndim
                       << " dimensions at offset: " << offset << std::endl;
  } else {
    std::stringstream str;
    str << "Could not read binary metadata from index file at offset: " << offset << std::endl;
    throw powerlaw_ann::diskann_exception_t(str.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
}

template <typename T>
void load_bin(aligned_file_reader_t& reader, T*& data, size_t& npts, size_t& ndim, size_t offset) {
  // Code assumes that the reader is already setup correctly.
  get_bin_metadata(reader, npts, ndim, offset);
  data = new T[npts * ndim];

  size_t data_size = npts * ndim * sizeof(T);
  size_t write_offset = 0;
  size_t read_start = offset + 2 * sizeof(uint32_t);

  // BingAlignedFileReader can only read uint32_t bytes of data. So,
  // we limit ourselves even more to reading 1GB at a time.
  std::vector<aligned_read_t> read_reqs;
  while (data_size > 0) {
    aligned_read_t read_req;
    read_req.buf = data + write_offset;
    read_req.offset = read_start + write_offset;
    read_req.len = data_size > MAX_REQUEST_SIZE ? MAX_REQUEST_SIZE : data_size;
    read_reqs.push_back(read_req);
    // in the corner case, the loop will not execute
    data_size -= read_req.len;
    write_offset += read_req.len;
  }
  aligned_io_context_t& ctx = reader.get_ctx();
  reader.read(read_reqs, ctx);
  for (int i = 0; i < read_reqs.size(); i++) {
    // Since we are making sync calls, no request will be in the
    // READ_WAIT state.
    if ((*(ctx.request_statuses_))[i] != aligned_io_context_t::READ_SUCCESS) {
      std::stringstream str;
      str << "Could not read binary data from index file at offset: " << read_reqs[i].offset
          << std::endl;
      throw powerlaw_ann::diskann_exception_t(str.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
  }
}
template <typename T>
void load_bin(aligned_file_reader_t& reader, std::unique_ptr<T[]>& data, size_t& npts, size_t& ndim,
              size_t offset) {
  T* ptr = nullptr;
  load_bin(reader, ptr, npts, ndim, offset);
  data.reset(ptr);
}

template <typename T>
void copy_aligned_data_from_file(aligned_file_reader_t& reader, T*& data, size_t& npts,
                                 size_t& ndim, const size_t& rounded_dim, size_t offset) {
  if (data == nullptr) {
    powerlaw_ann::cerr << "Memory was not allocated for " << data
                       << " before calling the load function. Exiting..." << std::endl;
    throw powerlaw_ann::diskann_exception_t("Null pointer passed to copy_aligned_data_from_file()",
                                            -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  size_t pts, dim;
  get_bin_metadata(reader, pts, dim, offset);

  if (ndim != dim || npts != pts) {
    std::stringstream ss;
    ss << "Either file dimension: " << dim << " is != passed dimension: " << ndim
       << " or file #pts: " << pts << " is != passed #pts: " << npts << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  // Instead of reading one point of ndim size and setting (rounded_dim - dim)
  // values to zero We'll set everything to zero and read in chunks of data at
  // the appropriate locations.
  size_t read_offset = offset + 2 * sizeof(uint32_t);
  memset(data, 0, npts * rounded_dim * sizeof(T));
  int i = 0;
  std::vector<aligned_read_t> read_requests;

  while (i < npts) {
    int j = 0;
    read_requests.clear();
    while (j < MAX_SIMULTANEOUS_READ_REQUESTS && i < npts) {
      aligned_read_t read_req;
      read_req.buf = data + i * rounded_dim;
      read_req.len = dim * sizeof(T);
      read_req.offset = read_offset + i * dim * sizeof(T);
      read_requests.push_back(read_req);
      i++;
      j++;
    }
    aligned_io_context_t& ctx = reader.get_ctx();
    reader.read(read_requests, ctx);
    for (int k = 0; k < read_requests.size(); k++) {
      if ((*ctx.request_statuses_)[k] != aligned_io_context_t::READ_SUCCESS) {
        throw powerlaw_ann::diskann_exception_t("Load data from file using AlignedReader failed.",
                                                -1, __FUNCSIG__, __FILE__, __LINE__);
      }
    }
  }
}

// Unlike load_bin, assumes that data is already allocated 'size' entries
template <typename T>
void read_array(aligned_file_reader_t& reader, T* data, size_t size, size_t offset) {
  if (data == nullptr) {
    throw powerlaw_ann::diskann_exception_t("read_array requires an allocated buffer.", -1);
  }

  if (size * sizeof(T) > MAX_REQUEST_SIZE) {
    std::stringstream ss;
    ss << "Cannot read more than " << MAX_REQUEST_SIZE
       << " bytes. Current request size: " << std::to_string(size) << " sizeof(T): " << sizeof(T)
       << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
  std::vector<aligned_read_t> read_requests;
  aligned_read_t read_req;
  read_req.buf = data;
  read_req.len = size * sizeof(T);
  read_req.offset = offset;
  read_requests.push_back(read_req);
  aligned_io_context_t& ctx = reader.get_ctx();
  reader.read(read_requests, ctx);

  if ((*(ctx.request_statuses_))[0] != aligned_io_context_t::READ_SUCCESS) {
    std::stringstream ss;
    ss << "Failed to read_array() of size: " << size * sizeof(T) << " at offset: " << offset
       << " from reader. " << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
}

template <typename T>
void read_value(aligned_file_reader_t& reader, T& value, size_t offset) {
  read_array(reader, &value, 1, offset);
}

template POWERLAWANN_DLLEXPORT void load_bin<uint8_t>(aligned_file_reader_t& reader,
                                                      std::unique_ptr<uint8_t[]>& data,
                                                      size_t& npts, size_t& ndim, size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<int8_t>(aligned_file_reader_t& reader,
                                                     std::unique_ptr<int8_t[]>& data, size_t& npts,
                                                     size_t& ndim, size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<uint32_t>(aligned_file_reader_t& reader,
                                                       std::unique_ptr<uint32_t[]>& data,
                                                       size_t& npts, size_t& ndim, size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<uint64_t>(aligned_file_reader_t& reader,
                                                       std::unique_ptr<uint64_t[]>& data,
                                                       size_t& npts, size_t& ndim, size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<int64_t>(aligned_file_reader_t& reader,
                                                      std::unique_ptr<int64_t[]>& data,
                                                      size_t& npts, size_t& ndim, size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<float>(aligned_file_reader_t& reader,
                                                    std::unique_ptr<float[]>& data, size_t& npts,
                                                    size_t& ndim, size_t offset);

template POWERLAWANN_DLLEXPORT void load_bin<uint8_t>(aligned_file_reader_t& reader, uint8_t*& data,
                                                      size_t& npts, size_t& ndim, size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<int64_t>(aligned_file_reader_t& reader, int64_t*& data,
                                                      size_t& npts, size_t& ndim, size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<uint64_t>(aligned_file_reader_t& reader,
                                                       uint64_t*& data, size_t& npts, size_t& ndim,
                                                       size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<uint32_t>(aligned_file_reader_t& reader,
                                                       uint32_t*& data, size_t& npts, size_t& ndim,
                                                       size_t offset);
template POWERLAWANN_DLLEXPORT void load_bin<int32_t>(aligned_file_reader_t& reader, int32_t*& data,
                                                      size_t& npts, size_t& ndim, size_t offset);

template POWERLAWANN_DLLEXPORT void
copy_aligned_data_from_file<uint8_t>(aligned_file_reader_t& reader, uint8_t*& data, size_t& npts,
                                     size_t& dim, const size_t& rounded_dim, size_t offset);
template POWERLAWANN_DLLEXPORT void
copy_aligned_data_from_file<int8_t>(aligned_file_reader_t& reader, int8_t*& data, size_t& npts,
                                    size_t& dim, const size_t& rounded_dim, size_t offset);
template POWERLAWANN_DLLEXPORT void
copy_aligned_data_from_file<float>(aligned_file_reader_t& reader, float*& data, size_t& npts,
                                   size_t& dim, const size_t& rounded_dim, size_t offset);

template POWERLAWANN_DLLEXPORT void read_array<char>(aligned_file_reader_t& reader, char* data,
                                                     size_t size, size_t offset);

template POWERLAWANN_DLLEXPORT void read_array<uint8_t>(aligned_file_reader_t& reader,
                                                        uint8_t* data, size_t size, size_t offset);
template POWERLAWANN_DLLEXPORT void read_array<int8_t>(aligned_file_reader_t& reader, int8_t* data,
                                                       size_t size, size_t offset);
template POWERLAWANN_DLLEXPORT void
read_array<uint32_t>(aligned_file_reader_t& reader, uint32_t* data, size_t size, size_t offset);
template POWERLAWANN_DLLEXPORT void read_array<float>(aligned_file_reader_t& reader, float* data,
                                                      size_t size, size_t offset);

template POWERLAWANN_DLLEXPORT void read_value<uint8_t>(aligned_file_reader_t& reader,
                                                        uint8_t& value, size_t offset);
template POWERLAWANN_DLLEXPORT void read_value<int8_t>(aligned_file_reader_t& reader, int8_t& value,
                                                       size_t offset);
template POWERLAWANN_DLLEXPORT void read_value<float>(aligned_file_reader_t& reader, float& value,
                                                      size_t offset);
template POWERLAWANN_DLLEXPORT void read_value<uint32_t>(aligned_file_reader_t& reader,
                                                         uint32_t& value, size_t offset);
template POWERLAWANN_DLLEXPORT void read_value<uint64_t>(aligned_file_reader_t& reader,
                                                         uint64_t& value, size_t offset);

#endif

} // namespace powerlaw_ann
