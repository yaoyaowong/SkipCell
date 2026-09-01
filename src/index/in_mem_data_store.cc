#include "index/in_mem_data_store.h"

#include "common/abstract_scratch.h"
#include "common/utils.h"

#include <memory>

namespace powerlaw_ann {

template <typename data_t>
in_mem_data_store_t<data_t>::in_mem_data_store_t(const location_t num_points, const size_t dim,
                                                 std::unique_ptr<distance_t<data_t>> distance_fn)
    : abstract_data_store_t<data_t>(num_points, dim), distance_fn_(std::move(distance_fn)) {
  aligned_dim_ = ROUND_UP(dim, distance_fn_->get_required_alignment());
  alloc_aligned(((void**) &data_), this->capacity_ * aligned_dim_ * sizeof(data_t),
                8 * sizeof(data_t));
  std::memset(data_, 0, this->capacity_ * aligned_dim_ * sizeof(data_t));
}

template <typename data_t>
in_mem_data_store_t<data_t>::~in_mem_data_store_t() {
  if (data_ != nullptr) {
    aligned_free(this->data_);
  }
}

template <typename data_t>
size_t in_mem_data_store_t<data_t>::get_aligned_dim() const {
  return aligned_dim_;
}

template <typename data_t>
size_t in_mem_data_store_t<data_t>::get_alignment_factor() const {
  return distance_fn_->get_required_alignment();
}

template <typename data_t>
location_t in_mem_data_store_t<data_t>::load(const std::string& filename) {
  return load_impl(filename);
}

#ifdef EXEC_ENV_OLS
template <typename data_t>
location_t in_mem_data_store_t<data_t>::load_impl(aligned_file_reader_t& reader) {
  size_t file_dim, file_num_points;

  powerlaw_ann::get_bin_metadata(reader, file_num_points, file_dim);

  if (file_dim != this->dim_) {
    std::stringstream stream;
    stream << "ERROR: Driver requests loading " << this->dim_ << " dimension," << "but file has "
           << file_dim << " dimension." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    aligned_free(data_);
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (file_num_points > this->capacity()) {
    this->resize((location_t) file_num_points);
  }
  copy_aligned_data_from_file<data_t>(reader, data_, file_num_points, file_dim, aligned_dim_);

  return (location_t) file_num_points;
}
#endif

template <typename data_t>
location_t in_mem_data_store_t<data_t>::load_impl(const std::string& filename) {
  size_t file_dim, file_num_points;
  if (!file_exists(filename)) {
    std::stringstream stream;
    stream << "ERROR: data file " << filename << " does not exist." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    aligned_free(data_);
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }
  powerlaw_ann::get_bin_metadata(filename, file_num_points, file_dim);

  if (file_dim != this->dim_) {
    std::stringstream stream;
    stream << "ERROR: Driver requests loading " << this->dim_ << " dimension," << "but file has "
           << file_dim << " dimension." << std::endl;
    powerlaw_ann::cerr << stream.str() << std::endl;
    aligned_free(data_);
    throw powerlaw_ann::diskann_exception_t(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
  }

  if (file_num_points > this->capacity()) {
    this->resize((location_t) file_num_points);
  }

  copy_aligned_data_from_file<data_t>(filename.c_str(), data_, file_num_points, file_dim,
                                      aligned_dim_);

  return (location_t) file_num_points;
}

template <typename data_t>
size_t in_mem_data_store_t<data_t>::save(const std::string& filename, const location_t num_points) {
  return save_data_in_base_dimensions(filename, data_, num_points, this->get_dims(),
                                      this->get_aligned_dim(), 0U);
}

template <typename data_t>
void in_mem_data_store_t<data_t>::populate_data(const data_t* vectors, const location_t num_pts) {
  memset(data_, 0, aligned_dim_ * sizeof(data_t) * num_pts);
  for (location_t i = 0; i < num_pts; i++) {
    std::memmove(data_ + i * aligned_dim_, vectors + i * this->dim_, this->dim_ * sizeof(data_t));
  }

  if (distance_fn_->preprocessing_required()) {
    distance_fn_->preprocess_base_points(data_, this->aligned_dim_, num_pts);
  }
}

template <typename data_t>
void in_mem_data_store_t<data_t>::populate_data(const std::string& filename, const size_t offset) {
  size_t npts, ndim;
  copy_aligned_data_from_file(filename.c_str(), data_, npts, ndim, aligned_dim_, offset);

  if ((location_t) npts > this->capacity()) {
    std::stringstream ss;
    ss << "Number of points in the file: " << filename
       << " is greater than the capacity of data store: " << this->capacity()
       << ". Must invoke resize before calling populate_data()" << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1);
  }

  if ((location_t) ndim != this->get_dims()) {
    std::stringstream ss;
    ss << "Number of dimensions of a point in the file: " << filename
       << " is not equal to dimensions of data store: " << this->capacity() << "." << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1);
  }

  if (distance_fn_->preprocessing_required()) {
    distance_fn_->preprocess_base_points(data_, this->aligned_dim_, this->capacity());
  }
}

template <typename data_t>
void in_mem_data_store_t<data_t>::extract_data_to_bin(const std::string& filename,
                                                      const location_t num_points) {
  save_data_in_base_dimensions(filename, data_, num_points, this->get_dims(),
                               this->get_aligned_dim(), 0U);
}

template <typename data_t>
void in_mem_data_store_t<data_t>::get_vector(const location_t i, data_t* dest) const {
  // REFACTOR TODO: Should we denormalize and return values?
  memcpy(dest, data_ + i * aligned_dim_, this->dim_ * sizeof(data_t));
}

template <typename data_t>
void in_mem_data_store_t<data_t>::set_vector(const location_t loc, const data_t* const vector) {
  size_t offset_in_data = loc * aligned_dim_;
  memset(data_ + offset_in_data, 0, aligned_dim_ * sizeof(data_t));
  memcpy(data_ + offset_in_data, vector, this->dim_ * sizeof(data_t));
  if (distance_fn_->preprocessing_required()) {
    distance_fn_->preprocess_base_points(data_ + offset_in_data, aligned_dim_, 1);
  }
}

template <typename data_t>
void in_mem_data_store_t<data_t>::prefetch_vector(const location_t loc) {
  powerlaw_ann::prefetch_vector((const char*) data_ + aligned_dim_ * (size_t) loc * sizeof(data_t),
                                sizeof(data_t) * aligned_dim_);
}

template <typename data_t>
void in_mem_data_store_t<data_t>::preprocess_query(
    const data_t* query, abstract_scratch_t<data_t>* query_scratch) const {
  if (query_scratch != nullptr) {
    memcpy(query_scratch->aligned_query(), query, sizeof(data_t) * this->get_dims());
  } else {
    std::stringstream ss;
    ss << "In in_mem_data_store_t::preprocess_query: Query scratch is null";
    powerlaw_ann::cerr << ss.str() << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1);
  }
}

template <typename data_t>
float in_mem_data_store_t<data_t>::get_distance(const data_t* query, const location_t loc) const {
  return distance_fn_->compare(query, data_ + aligned_dim_ * loc, (uint32_t) aligned_dim_);
}

template <typename data_t>
void in_mem_data_store_t<data_t>::get_distance(const data_t* query, const location_t* locations,
                                               const uint32_t location_count, float* distances,
                                               abstract_scratch_t<data_t>* scratch_space) const {
  for (location_t i = 0; i < location_count; i++) {
    distances[i] = distance_fn_->compare(query, data_ + locations[i] * aligned_dim_,
                                         (uint32_t) this->aligned_dim_);
  }
}

template <typename data_t>
float in_mem_data_store_t<data_t>::get_distance(const location_t loc1,
                                                const location_t loc2) const {
  return distance_fn_->compare(data_ + loc1 * aligned_dim_, data_ + loc2 * aligned_dim_,
                               (uint32_t) this->aligned_dim_);
}

template <typename data_t>
void in_mem_data_store_t<data_t>::get_distance(const data_t* preprocessed_query,
                                               const std::vector<location_t>& ids,
                                               std::vector<float>& distances,
                                               abstract_scratch_t<data_t>* scratch_space) const {
  for (int i = 0; i < ids.size(); i++) {
    distances[i] = distance_fn_->compare(preprocessed_query, data_ + ids[i] * aligned_dim_,
                                         (uint32_t) this->aligned_dim_);
  }
}

template <typename data_t>
location_t in_mem_data_store_t<data_t>::expand(const location_t new_size) {
  if (new_size == this->capacity()) {
    return this->capacity();
  } else if (new_size < this->capacity()) {
    std::stringstream ss;
    ss << "Cannot 'expand' datastore when new capacity (" << new_size << ") < existing capacity("
       << this->capacity() << ")" << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1);
  }
#ifndef _WINDOWS
  data_t* new_data;
  alloc_aligned((void**) &new_data, new_size * aligned_dim_ * sizeof(data_t), 8 * sizeof(data_t));
  memcpy(new_data, data_, this->capacity() * aligned_dim_ * sizeof(data_t));
  aligned_free(data_);
  data_ = new_data;
#else
  realloc_aligned((void**) &data_, new_size * aligned_dim_ * sizeof(data_t), 8 * sizeof(data_t));
#endif
  this->capacity_ = new_size;
  return this->capacity_;
}

template <typename data_t>
location_t in_mem_data_store_t<data_t>::shrink(const location_t new_size) {
  if (new_size == this->capacity()) {
    return this->capacity();
  } else if (new_size > this->capacity()) {
    std::stringstream ss;
    ss << "Cannot 'shrink' datastore when new capacity (" << new_size << ") > existing capacity("
       << this->capacity() << ")" << std::endl;
    throw powerlaw_ann::diskann_exception_t(ss.str(), -1);
  }
#ifndef _WINDOWS
  data_t* new_data;
  alloc_aligned((void**) &new_data, new_size * aligned_dim_ * sizeof(data_t), 8 * sizeof(data_t));
  memcpy(new_data, data_, new_size * aligned_dim_ * sizeof(data_t));
  aligned_free(data_);
  data_ = new_data;
#else
  realloc_aligned((void**) &data_, new_size * aligned_dim_ * sizeof(data_t), 8 * sizeof(data_t));
#endif
  this->capacity_ = new_size;
  return this->capacity_;
}

template <typename data_t>
void in_mem_data_store_t<data_t>::move_vectors(const location_t old_location_start,
                                               const location_t new_location_start,
                                               const location_t num_locations) {
  if (num_locations == 0 || old_location_start == new_location_start) {
    return;
  }

  /*    // Update pointers to the moved nodes. Note: the computation is correct
     even
      // when new_location_start < old_location_start given the C++ uint32_t
      // integer arithmetic rules.
      const uint32_t location_delta = new_location_start - old_location_start;
  */
  // The [start, end) interval which will contain obsolete points to be
  // cleared.
  uint32_t mem_clear_loc_start = old_location_start;
  uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

  if (new_location_start < old_location_start) {
    // If ranges are overlapping, make sure not to clear the newly copied
    // data.
    if (mem_clear_loc_start < new_location_start + num_locations) {
      // Clear only after the end of the new range.
      mem_clear_loc_start = new_location_start + num_locations;
    }
  } else {
    // If ranges are overlapping, make sure not to clear the newly copied
    // data.
    if (mem_clear_loc_end_limit > new_location_start) {
      // Clear only up to the beginning of the new range.
      mem_clear_loc_end_limit = new_location_start;
    }
  }

  // Use memmove to handle overlapping ranges.
  copy_vectors(old_location_start, new_location_start, num_locations);
  memset(data_ + aligned_dim_ * mem_clear_loc_start, 0,
         sizeof(data_t) * aligned_dim_ * (mem_clear_loc_end_limit - mem_clear_loc_start));
}

template <typename data_t>
void in_mem_data_store_t<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc,
                                               const location_t num_points) {
  assert(from_loc < this->capacity_);
  assert(to_loc < this->capacity_);
  assert(num_points < this->capacity_);
  memmove(data_ + aligned_dim_ * to_loc, data_ + aligned_dim_ * from_loc,
          num_points * aligned_dim_ * sizeof(data_t));
}

template <typename data_t>
location_t in_mem_data_store_t<data_t>::calculate_medoid() const {
  // allocate and init centroid
  float* center = new float[aligned_dim_];
  for (size_t j = 0; j < aligned_dim_; j++)
    center[j] = 0;

  for (size_t i = 0; i < this->capacity(); i++)
    for (size_t j = 0; j < aligned_dim_; j++)
      center[j] += (float) data_[i * aligned_dim_ + j];

  for (size_t j = 0; j < aligned_dim_; j++)
    center[j] /= (float) this->capacity();

  // compute all to one distance
  float* distances = new float[this->capacity()];

  // TODO: REFACTOR. Removing pragma might make this slow. Must revisit.
  //  Problem is that we need to pass num_threads here, it is not clear
  //  if data store must be aware of threads!
  // #pragma omp parallel for schedule(static, 65536)
  for (int64_t i = 0; i < (int64_t) this->capacity(); i++) {
    // extract point and distance reference
    float& dist = distances[i];
    const data_t* cur_vec = data_ + (i * (size_t) aligned_dim_);
    dist = 0;
    float diff = 0;
    for (size_t j = 0; j < aligned_dim_; j++) {
      diff = (center[j] - (float) cur_vec[j]) * (center[j] - (float) cur_vec[j]);
      dist += diff;
    }
  }
  // find imin
  uint32_t min_idx = 0;
  float min_dist = distances[0];
  for (uint32_t i = 1; i < this->capacity(); i++) {
    if (distances[i] < min_dist) {
      min_idx = i;
      min_dist = distances[i];
    }
  }

  delete[] distances;
  delete[] center;
  return min_idx;
}

template <typename data_t>
distance_t<data_t>* in_mem_data_store_t<data_t>::get_dist_fn() const {
  return this->distance_fn_.get();
}

template POWERLAWANN_DLLEXPORT class in_mem_data_store_t<float>;
template POWERLAWANN_DLLEXPORT class in_mem_data_store_t<int8_t>;
template POWERLAWANN_DLLEXPORT class in_mem_data_store_t<uint8_t>;

} // namespace powerlaw_ann