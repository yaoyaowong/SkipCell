#ifndef COMMON_ABSTRACT_SCRATCH
#define COMMON_ABSTRACT_SCRATCH

namespace powerlaw_ann {

template <typename data_t>
class pq_scratch_t;

// By somewhat more than a coincidence, it seems that both in_mem_query_scratch_t
// and ssd_query_scratch_t have the aligned query and pq_scratch_t objects. So we
// can put them in a neat hierarchy and keep pq_scratch_t as a standalone class.
template <typename data_t>
class abstract_scratch_t {
public:
  abstract_scratch_t() = default;
  // This class does not take any responsibilty for memory management of
  // its members. It is the responsibility of the derived classes to do so.
  virtual ~abstract_scratch_t() = default;

  // Scratch objects should not be copied
  abstract_scratch_t(const abstract_scratch_t&) = delete;
  abstract_scratch_t& operator=(const abstract_scratch_t&) = delete;

  data_t* aligned_query() { return aligned_query_; }
  pq_scratch_t<data_t>* pq_scratch() { return pq_scratch_; }

protected:
  data_t* aligned_query_ = nullptr;
  pq_scratch_t<data_t>* pq_scratch_ = nullptr;
};
} // namespace powerlaw_ann

#endif // COMMON_ABSTRACT_SCRATCH
