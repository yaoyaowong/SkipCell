#include "pq/pq_data_store.h"

#include "common/distance.h"
#include "common/utils.h"
#include "pq/pq_common.h"
#include "pq/pq_scratch.h"
#include "pq/product_quantizer.h"

namespace powerlaw_ann {

// REFACTOR TODO: Assuming that num_pq_chunks is known already. Must verify if
// this is true.
template <typename data_t>
pq_data_store_t<data_t>::pq_data_store_t(
    size_t dim, location_t num_points, size_t num_pq_chunks,
    std::unique_ptr<distance_t<data_t>> distance_fn,
    std::unique_ptr<quantized_distance_t<data_t>> pq_distance_fn)
    : abstract_data_store_t<data_t>(num_points, dim), quantized_data_(nullptr),
      num_chunks_(num_pq_chunks), distance_metric_(distance_fn->get_metric()) {
  if (num_pq_chunks > dim) {
    throw powerlaw_ann::diskann_exception_t("ERROR: num_pq_chunks > dim", -1, __FUNCSIG__, __FILE__,
                                            __LINE__);
  }
  distance_fn_ = std::move(distance_fn);
  pq_distance_fn_ = std::move(pq_distance_fn);
}

template <typename data_t>
pq_data_store_t<data_t>::~pq_data_store_t() {
  if (quantized_data_ != nullptr) {
    aligned_free(quantized_data_);
    quantized_data_ = nullptr;
  }
}

template <typename data_t>
location_t pq_data_store_t<data_t>::load(const std::string& filename) {
  return load_impl(filename);
}
template <typename data_t>
size_t pq_data_store_t<data_t>::save(const std::string& filename, const location_t num_points) {
  return powerlaw_ann::save_bin(filename, quantized_data_, this->capacity(), num_chunks_, 0);
}

template <typename data_t>
size_t pq_data_store_t<data_t>::get_aligned_dim() const {
  return this->get_dims();
}

// Populate quantized data from regular data.
template <typename data_t>
void pq_data_store_t<data_t>::populate_data(const data_t* vectors, const location_t num_pts) {
  throw std::logic_error("Not implemented yet");
}

template <typename data_t>
void pq_data_store_t<data_t>::populate_data(const std::string& filename, const size_t offset) {
  if (quantized_data_ != nullptr) {
    aligned_free(quantized_data_);
  }

  size_t file_num_points = 0, file_dim = 0;
  get_bin_metadata(filename, file_num_points, file_dim, offset);
  this->capacity_ = static_cast<location_t>(file_num_points);
  this->dim_ = file_dim;

  double p_val = std::min(1.0, ((double) MAX_PQ_TRAINING_SET_SIZE / (double) file_num_points));

  auto pivots_file = pq_distance_fn_->get_pivot_data_filename(filename);
  auto compressed_file = pq_distance_fn_->get_quantized_vectors_filename(filename);

  generate_quantized_data<data_t>(filename, pivots_file, compressed_file, distance_metric_, p_val,
                                  num_chunks_, pq_distance_fn_->is_opq());

  // REFACTOR TODO: Not sure of the alignment. Just copying from index.cpp
  alloc_aligned(((void**) &quantized_data_), file_num_points * num_chunks_ * sizeof(uint8_t), 1);
  copy_aligned_data_from_file<uint8_t>(compressed_file.c_str(), quantized_data_, file_num_points,
                                       num_chunks_, num_chunks_);
#ifdef EXEC_ENV_OLS
  throw diskann_exception_t("load_pq_centroid_bin should not be called when "
                            "EXEC_ENV_OLS is defined.",
                            -1, __FUNCSIG__, __FILE__, __LINE__);
#else
  pq_distance_fn_->load_pivot_data(pivots_file.c_str(), num_chunks_);
#endif
}

template <typename data_t>
void pq_data_store_t<data_t>::extract_data_to_bin(const std::string& filename,
                                                  const location_t num_pts) {
  throw std::logic_error("Not implemented yet");
}

template <typename data_t>
void pq_data_store_t<data_t>::get_vector(const location_t i, data_t* target) const {
  // REFACTOR TODO: Should we inflate the compressed vector here?
  if (i < this->capacity()) {
    throw std::logic_error("Not implemented yet.");
  } else {
    std::stringstream ss;
    ss << "Requested vector " << i << " but only  " << this->capacity() << " vectors are present";
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1);
  }
}
template <typename data_t>
void pq_data_store_t<data_t>::set_vector(const location_t i, const data_t* const vector) {
  // REFACTOR TODO: Should we accept a normal vector and compress here?
  // memcpy (_data + i * num_chunks_, vector, num_chunks_ * sizeof(data_t));
  throw std::logic_error("Not implemented yet");
}

template <typename data_t>
void pq_data_store_t<data_t>::prefetch_vector(const location_t loc) {
  const uint8_t* ptr = quantized_data_ + ((size_t) loc) * num_chunks_ * sizeof(data_t);
  powerlaw_ann::prefetch_vector((const char*) ptr, num_chunks_ * sizeof(data_t));
}

template <typename data_t>
void pq_data_store_t<data_t>::move_vectors(const location_t old_location_start,
                                           const location_t new_location_start,
                                           const location_t num_points) {
  // REFACTOR TODO: Moving vectors is only for in-mem fresh.
  throw std::logic_error("Not implemented yet");
}

template <typename data_t>
void pq_data_store_t<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc,
                                           const location_t num_points) {
  // REFACTOR TODO: Is the number of bytes correct?
  memcpy(quantized_data_ + to_loc * num_chunks_, quantized_data_ + from_loc * num_chunks_,
         num_chunks_ * num_points);
}

// REFACTOR TODO: Currently, we take aligned_query as parameter, but this
// function should also do the alignment.
template <typename data_t>
void pq_data_store_t<data_t>::preprocess_query(const data_t* aligned_query,
                                               abstract_scratch_t<data_t>* scratch) const {
  if (scratch == nullptr) {
    throw powerlaw_ann::diskann_exception_t("Scratch space is null", -1);
  }

  pq_scratch_t<data_t>* pq_scratch = scratch->pq_scratch();

  if (pq_scratch == nullptr) {
    throw powerlaw_ann::diskann_exception_t(
        "pq_scratch_t space has not been set in the scratch object.", -1);
  }

  pq_distance_fn_->preprocess_query(aligned_query, (location_t) this->get_dims(), *pq_scratch);
}

template <typename data_t>
float pq_data_store_t<data_t>::get_distance(const data_t* query, const location_t loc) const {
  throw std::logic_error("Not implemented yet");
}

template <typename data_t>
float pq_data_store_t<data_t>::get_distance(const location_t loc1, const location_t loc2) const {
  throw std::logic_error("Not implemented yet");
}

template <typename data_t>
void pq_data_store_t<data_t>::get_distance(const data_t* preprocessed_query,
                                           const location_t* locations,
                                           const uint32_t location_count, float* distances,
                                           abstract_scratch_t<data_t>* scratch_space) const {
  if (scratch_space == nullptr) {
    throw powerlaw_ann::diskann_exception_t("Scratch space is null", -1);
  }
  pq_scratch_t<data_t>* pq_scratch = scratch_space->pq_scratch();
  if (pq_scratch == nullptr) {
    throw powerlaw_ann::diskann_exception_t("pq_scratch_t not set in scratch space.", -1);
  }
  powerlaw_ann::aggregate_coords(locations, location_count, quantized_data_, this->num_chunks_,
                                 pq_scratch->aligned_pq_coord_scratch);
  pq_distance_fn_->preprocessed_distance(*pq_scratch, location_count, distances);
}

template <typename data_t>
void pq_data_store_t<data_t>::get_distance(const data_t* preprocessed_query,
                                           const std::vector<location_t>& ids,
                                           std::vector<float>& distances,
                                           abstract_scratch_t<data_t>* scratch_space) const {
  if (scratch_space == nullptr) {
    throw powerlaw_ann::diskann_exception_t("Scratch space is null", -1);
  }
  pq_scratch_t<data_t>* pq_scratch = scratch_space->pq_scratch();
  if (pq_scratch == nullptr) {
    throw powerlaw_ann::diskann_exception_t("pq_scratch_t not set in scratch space.", -1);
  }
  powerlaw_ann::aggregate_coords(ids, quantized_data_, this->num_chunks_,
                                 pq_scratch->aligned_pq_coord_scratch);
  pq_distance_fn_->preprocessed_distance(*pq_scratch, (location_t) ids.size(), distances);
}

template <typename data_t>
location_t pq_data_store_t<data_t>::calculate_medoid() const {
  // REFACTOR TODO: Must calculate this just like we do with data store.
  size_t r = (size_t) rand() * (size_t) RAND_MAX + (size_t) rand();
  return (uint32_t) (r % (size_t) this->capacity());
}

template <typename data_t>
size_t pq_data_store_t<data_t>::get_alignment_factor() const {
  return 1;
}

template <typename data_t>
distance_t<data_t>* pq_data_store_t<data_t>::get_dist_fn() const {
  return distance_fn_.get();
}

template <typename data_t>
location_t pq_data_store_t<data_t>::load_impl(const std::string& file_prefix) {
  if (quantized_data_ != nullptr) {
    aligned_free(quantized_data_);
  }
  auto quantized_vectors_file = pq_distance_fn_->get_quantized_vectors_filename(file_prefix);

  size_t num_points;
  load_aligned_bin(quantized_vectors_file, quantized_data_, num_points, num_chunks_, num_chunks_);
  this->capacity_ = static_cast<location_t>(num_points);

  auto pivots_file = pq_distance_fn_->get_pivot_data_filename(file_prefix);
  pq_distance_fn_->load_pivot_data(pivots_file, num_chunks_);

  return this->capacity_;
}

template <typename data_t>
location_t pq_data_store_t<data_t>::expand(const location_t new_size) {
  throw std::logic_error("Not implemented yet");
}

template <typename data_t>
location_t pq_data_store_t<data_t>::shrink(const location_t new_size) {
  throw std::logic_error("Not implemented yet");
}

#ifdef EXEC_ENV_OLS
template <typename data_t>
location_t pq_data_store_t<data_t>::load_impl(aligned_file_reader_t& reader) {}
#endif

template POWERLAWANN_DLLEXPORT class pq_data_store_t<int8_t>;
template POWERLAWANN_DLLEXPORT class pq_data_store_t<float>;
template POWERLAWANN_DLLEXPORT class pq_data_store_t<uint8_t>;

} // namespace powerlaw_ann
