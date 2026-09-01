#ifndef PQ_PQ_DATA_STORE
#define PQ_PQ_DATA_STORE

#include "common/distance.h"
#include "index/abstract_data_store.h"
#include "pq/quantized_distance.h"

#include <memory>

namespace powerlaw_ann {
// REFACTOR TODO: By default, the pq_data_store_t is an in-memory datastore because both Vamana and
// DiskANN treat it the same way. But with DiskPQ, that may need to change.
template <typename data_t>
class pq_data_store_t : public abstract_data_store_t<data_t> {

public:
  pq_data_store_t(size_t dim, location_t num_points, size_t num_pq_chunks,
                  std::unique_ptr<distance_t<data_t>> distance_fn,
                  std::unique_ptr<quantized_distance_t<data_t>> pq_distance_fn);
  pq_data_store_t(const pq_data_store_t&) = delete;
  pq_data_store_t& operator=(const pq_data_store_t&) = delete;
  ~pq_data_store_t();

  // Load quantized vectors from a set of files. Here filename is treated
  // as a prefix and the files are assumed to be named with DiskANN
  // conventions.
  virtual location_t load(const std::string& file_prefix) override;

  // Save quantized vectors to a set of files whose names start with
  // file_prefix.
  //  Currently, the plan is to save the quantized vectors to the quantized
  //  vectors file.
  virtual size_t save(const std::string& file_prefix, const location_t num_points) override;

  // Since base class function is pure virtual, we need to declare it here, even though alignent
  // concept is not needed for Quantized data stores.
  virtual size_t get_aligned_dim() const override;

  // Populate quantized data from unaligned data using PQ functionality
  virtual void populate_data(const data_t* vectors, const location_t num_pts) override;
  virtual void populate_data(const std::string& filename, const size_t offset) override;

  virtual void extract_data_to_bin(const std::string& filename, const location_t num_pts) override;

  virtual void get_vector(const location_t i, data_t* target) const override;
  virtual void set_vector(const location_t i, const data_t* const vector) override;
  virtual void prefetch_vector(const location_t loc) override;

  virtual void move_vectors(const location_t old_location_start,
                            const location_t new_location_start,
                            const location_t num_points) override;
  virtual void copy_vectors(const location_t from_loc, const location_t to_loc,
                            const location_t num_points) override;

  virtual void preprocess_query(const data_t* query,
                                abstract_scratch_t<data_t>* scratch) const override;

  virtual float get_distance(const data_t* query, const location_t loc) const override;
  virtual float get_distance(const location_t loc1, const location_t loc2) const override;

  // NOTE: Caller must invoke "PQDistance->preprocess_query" ONCE before calling
  // this function.
  virtual void get_distance(const data_t* preprocessed_query, const location_t* locations,
                            const uint32_t location_count, float* distances,
                            abstract_scratch_t<data_t>* scratch_space) const override;

  // NOTE: Caller must invoke "PQDistance->preprocess_query" ONCE before calling
  // this function.
  virtual void get_distance(const data_t* preprocessed_query, const std::vector<location_t>& ids,
                            std::vector<float>& distances,
                            abstract_scratch_t<data_t>* scratch_space) const override;

  // We are returning the distance function that is used for full precision
  // vectors here, not the PQ distance function. This is because the callers
  // all are expecting a distance_t<T> not quantized_distance_t<T>.
  virtual distance_t<data_t>* get_dist_fn() const override;

  virtual location_t calculate_medoid() const override;

  virtual size_t get_alignment_factor() const override;

protected:
  virtual location_t expand(const location_t new_size) override;
  virtual location_t shrink(const location_t new_size) override;

  virtual location_t load_impl(const std::string& filename);
#ifdef EXEC_ENV_OLS
  virtual location_t load_impl(aligned_file_reader_t& reader);
#endif

private:
  uint8_t* quantized_data_ = nullptr;
  size_t num_chunks_ = 0;

  // REFACTOR TODO: Doing this temporarily before refactoring OPQ into
  // its own class. Remove later.
  bool use_opq_ = false;

  metric_t distance_metric_;
  std::unique_ptr<distance_t<data_t>> distance_fn_ = nullptr;
  std::unique_ptr<quantized_distance_t<data_t>> pq_distance_fn_ = nullptr;
};
} // namespace powerlaw_ann

#endif // PQ_PQ_DATA_STORE
