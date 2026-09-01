#include "pq/pq_l2_distance.h"

#include "common/diskann_exception.h"
#include "common/utils.h"
#include "pq/pq_common.h"
#include "pq/pq_scratch.h"
#include "pq/product_quantizer.h"

// block size for reading/processing large files and matrices in blocks
#define BLOCK_SIZE 5000000

namespace powerlaw_ann {

template <typename data_t>
pq_l2_distance_t<data_t>::pq_l2_distance_t(uint32_t num_chunks, bool use_opq)
    : num_chunks_(num_chunks), is_opq_(use_opq) {}

template <typename data_t>
pq_l2_distance_t<data_t>::~pq_l2_distance_t() {
#ifndef EXEC_ENV_OLS
  if (tables_ != nullptr)
    delete[] tables_;
  if (chunk_offsets_ != nullptr)
    delete[] chunk_offsets_;
  if (centroid_ != nullptr)
    delete[] centroid_;
  if (transposed_rotation_matrix_ != nullptr)
    delete[] transposed_rotation_matrix_;
#endif
  if (transposed_tables_ != nullptr)
    delete[] transposed_tables_;
}

template <typename data_t>
bool pq_l2_distance_t<data_t>::is_opq() const {
  return this->is_opq_;
}

template <typename data_t>
std::string
pq_l2_distance_t<data_t>::get_quantized_vectors_filename(const std::string& prefix) const {
  if (num_chunks_ == 0) {
    throw powerlaw_ann::diskann_exception_t(
        "Must set num_chunks before calling get_quantized_vectors_filename", -1, __FUNCSIG__,
        __FILE__, __LINE__);
  }
  return powerlaw_ann::get_quantized_vectors_filename(prefix, is_opq_, (uint32_t) num_chunks_);
}
template <typename data_t>
std::string pq_l2_distance_t<data_t>::get_pivot_data_filename(const std::string& prefix) const {
  if (num_chunks_ == 0) {
    throw powerlaw_ann::diskann_exception_t(
        "Must set num_chunks before calling get_pivot_data_filename", -1, __FUNCSIG__, __FILE__,
        __LINE__);
  }
  return powerlaw_ann::get_pivot_data_filename(prefix, is_opq_, (uint32_t) num_chunks_);
}
template <typename data_t>
std::string
pq_l2_distance_t<data_t>::get_rotation_matrix_suffix(const std::string& pq_pivots_filename) const {
  return powerlaw_ann::get_rotation_matrix_suffix(pq_pivots_filename);
}

#ifdef EXEC_ENV_OLS
template <typename data_t>
void pq_l2_distance_t<data_t>::load_pivot_data(memory_mapped_files_t& files,
                                               const std::string& pq_table_file,
                                               size_t num_chunks) {
#else
template <typename data_t>
void pq_l2_distance_t<data_t>::load_pivot_data(const std::string& pq_table_file,
                                               size_t num_chunks) {
#endif
  size_t nr, nc;
  // std::string rotmat_file = get_opq_rot_matrix_filename(pq_table_file,
  // false);

#ifdef EXEC_ENV_OLS
  size_t* file_offset_data; // since load_bin only sets the pointer, no need
  // to delete.
  powerlaw_ann::load_bin<size_t>(files, pq_table_file, file_offset_data, nr, nc);
#else
  std::unique_ptr<size_t[]> file_offset_data;
  powerlaw_ann::load_bin<size_t>(pq_table_file, file_offset_data, nr, nc);
#endif

  bool use_old_filetype = false;

  if (nr != 4 && nr != 5) {
    powerlaw_ann::cout << "Error reading pq_pivots file " << pq_table_file
                       << ". Offsets dont contain correct metadata, # offsets = " << nr
                       << ", but expecting " << 4 << " or " << 5;
    throw powerlaw_ann::diskann_exception_t("Error reading pq_pivots file at offsets data.", -1,
                                            __FUNCSIG__, __FILE__, __LINE__);
  }

  if (nr == 4) {
    powerlaw_ann::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " "
                       << file_offset_data[2] << " " << file_offset_data[3] << std::endl;
  } else if (nr == 5) {
    use_old_filetype = true;
    powerlaw_ann::cout << "Offsets: " << file_offset_data[0] << " " << file_offset_data[1] << " "
                       << file_offset_data[2] << " " << file_offset_data[3] << file_offset_data[4]
                       << std::endl;
  } else {
    throw powerlaw_ann::diskann_exception_t("Wrong number of offsets in pq_pivots", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
  }

#ifdef EXEC_ENV_OLS
  powerlaw_ann::load_bin<float>(files, pq_table_file, tables, nr, nc, file_offset_data[0]);
#else
  powerlaw_ann::load_bin<float>(pq_table_file, tables_, nr, nc, file_offset_data[0]);
#endif

  if ((nr != NUM_PQ_CENTROIDS)) {
    powerlaw_ann::cout << "Error reading pq_pivots file " << pq_table_file
                       << ". file_num_centers  = " << nr << " but expecting " << NUM_PQ_CENTROIDS
                       << " centers";
    throw powerlaw_ann::diskann_exception_t("Error reading pq_pivots file at pivots data.", -1,
                                            __FUNCSIG__, __FILE__, __LINE__);
  }

  this->num_dimensions_ = nc;

#ifdef EXEC_ENV_OLS
  powerlaw_ann::load_bin<float>(files, pq_table_file, centroid, nr, nc, file_offset_data[1]);
#else
  powerlaw_ann::load_bin<float>(pq_table_file, centroid_, nr, nc, file_offset_data[1]);
#endif

  if ((nr != this->num_dimensions_) || (nc != 1)) {
    powerlaw_ann::cerr << "Error reading centroids from pq_pivots file " << pq_table_file
                       << ". file_dim  = " << nr << ", file_cols = " << nc << " but expecting "
                       << this->num_dimensions_ << " entries in 1 dimension.";
    throw powerlaw_ann::diskann_exception_t("Error reading pq_pivots file at centroid data.", -1,
                                            __FUNCSIG__, __FILE__, __LINE__);
  }

  int chunk_offsets_index = 2;
  if (use_old_filetype) {
    chunk_offsets_index = 3;
  }
#ifdef EXEC_ENV_OLS
  powerlaw_ann::load_bin<uint32_t>(files, pq_table_file, chunk_offsets, nr, nc,
                                   file_offset_data[chunk_offsets_index]);
#else
  powerlaw_ann::load_bin<uint32_t>(pq_table_file, chunk_offsets_, nr, nc,
                                   file_offset_data[chunk_offsets_index]);
#endif

  if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0)) {
    powerlaw_ann::cerr << "Error loading chunk offsets file. numc: " << nc
                       << " (should be 1). numr: " << nr << " (should be " << num_chunks + 1
                       << " or 0 if we need to infer)" << std::endl;
    throw powerlaw_ann::diskann_exception_t("Error loading chunk offsets file", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
  }

  this->num_chunks_ = nr - 1;
  powerlaw_ann::cout << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS
                     << ", #dims: " << this->num_dimensions_ << ", #chunks: " << this->num_chunks_
                     << std::endl;

  // For OPQ there will be a rotation matrix to load.
  if (this->is_opq_) {
    std::string rotmat_file = get_rotation_matrix_suffix(pq_table_file);
#ifdef EXEC_ENV_OLS
    powerlaw_ann::load_bin<float>(files, rotmat_file, (float*&) rotmat_tr, nr, nc);
#else
    powerlaw_ann::load_bin<float>(rotmat_file, transposed_rotation_matrix_, nr, nc);
#endif
    if (nr != this->num_dimensions_ || nc != this->num_dimensions_) {
      powerlaw_ann::cerr << "Error loading rotation matrix file" << std::endl;
      throw powerlaw_ann::diskann_exception_t("Error loading rotation matrix file", -1, __FUNCSIG__,
                                              __FILE__, __LINE__);
    }
  }

  // alloc and compute transpose
  transposed_tables_ = new float[256 * this->num_dimensions_];
  for (size_t i = 0; i < 256; i++) {
    for (size_t j = 0; j < this->num_dimensions_; j++) {
      transposed_tables_[j * 256 + i] = tables_[i * this->num_dimensions_ + j];
    }
  }
}

template <typename data_t>
uint32_t pq_l2_distance_t<data_t>::get_num_chunks() const {
  return static_cast<uint32_t>(num_chunks_);
}

// REFACTOR: Instead of doing half the work in the caller and half in this
// function, we let this function
//  do all of the work, making it easier for the caller.
template <typename data_t>
void pq_l2_distance_t<data_t>::preprocess_query(const data_t* aligned_query, uint32_t dim,
                                                pq_scratch_t<data_t>& scratch) {
  // Copy query vector to float and then to "rotated" query
  for (size_t d = 0; d < dim; d++) {
    scratch.aligned_query_float[d] = (float) aligned_query[d];
  }
  scratch.initialize(dim, aligned_query);

  for (uint32_t d = 0; d < num_dimensions_; d++) {
    scratch.rotated_query[d] -= centroid_[d];
  }
  std::vector<float> tmp(num_dimensions_, 0);
  if (is_opq_) {
    for (uint32_t d = 0; d < num_dimensions_; d++) {
      for (uint32_t d1 = 0; d1 < num_dimensions_; d1++) {
        tmp[d] += scratch.rotated_query[d1] * transposed_rotation_matrix_[d1 * num_dimensions_ + d];
      }
    }
    std::memcpy(scratch.rotated_query, tmp.data(), num_dimensions_ * sizeof(float));
  }
  this->prepopulate_chunkwise_distances(scratch.rotated_query,
                                        scratch.aligned_pqtable_dist_scratch);
}

template <typename data_t>
void pq_l2_distance_t<data_t>::preprocessed_distance(pq_scratch_t<data_t>& pq_scratch,
                                                     const uint32_t n_ids, float* dists_out) {
  pq_dist_lookup(pq_scratch.aligned_pq_coord_scratch, n_ids, num_chunks_,
                 pq_scratch.aligned_pqtable_dist_scratch, dists_out);
}

template <typename data_t>
void pq_l2_distance_t<data_t>::preprocessed_distance(pq_scratch_t<data_t>& pq_scratch,
                                                     const uint32_t n_ids,
                                                     std::vector<float>& dists_out) {
  pq_dist_lookup(pq_scratch.aligned_pq_coord_scratch, n_ids, num_chunks_,
                 pq_scratch.aligned_pqtable_dist_scratch, dists_out);
}

template <typename data_t>
float pq_l2_distance_t<data_t>::brute_force_distance(const float* query_vec, uint8_t* base_vec) {
  float res = 0;
  for (size_t chunk = 0; chunk < num_chunks_; chunk++) {
    for (size_t j = chunk_offsets_[chunk]; j < chunk_offsets_[chunk + 1]; j++) {
      const float* centers_dim_vec = transposed_tables_ + (256 * j);
      float diff = centers_dim_vec[base_vec[chunk]] - (query_vec[j]);
      res += diff * diff;
    }
  }
  return res;
}

template <typename data_t>
void pq_l2_distance_t<data_t>::prepopulate_chunkwise_distances(const float* query_vec,
                                                               float* dist_vec) {
  memset(dist_vec, 0, 256 * num_chunks_ * sizeof(float));
  // chunk wise distance computation
  for (size_t chunk = 0; chunk < num_chunks_; chunk++) {
    // sum (q-c)^2 for the dimensions associated with this chunk
    float* chunk_dists = dist_vec + (256 * chunk);
    for (size_t j = chunk_offsets_[chunk]; j < chunk_offsets_[chunk + 1]; j++) {
      const float* centers_dim_vec = transposed_tables_ + (256 * j);
      for (size_t idx = 0; idx < 256; idx++) {
        double diff = centers_dim_vec[idx] - (query_vec[j]);
        chunk_dists[idx] += (float) (diff * diff);
      }
    }
  }
}

template POWERLAWANN_DLLEXPORT class pq_l2_distance_t<int8_t>;
template POWERLAWANN_DLLEXPORT class pq_l2_distance_t<uint8_t>;
template POWERLAWANN_DLLEXPORT class pq_l2_distance_t<float>;

} // namespace powerlaw_ann
