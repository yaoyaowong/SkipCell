#ifndef PQ_PQ
#define PQ_PQ

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace powerlaw_ann::pq {

inline constexpr uint32_t k_num_centers = 256;

struct model_t {
  uint32_t dim = 0;
  std::vector<uint32_t> chunk_offsets;
  std::vector<float> pivots;
  std::vector<float> centroid;

  uint32_t num_chunks() const {
    return chunk_offsets.empty() ? 0 : static_cast<uint32_t>(chunk_offsets.size() - 1);
  }
};

class code_store_t {
public:
  uint32_t num_vectors() const { return num_vectors_; }
  uint32_t num_chunks() const { return num_chunks_; }
  const uint8_t* code(uint32_t vector_id) const;

private:
  friend code_store_t load_codes(const std::string& path);

  uint32_t num_vectors_ = 0;
  uint32_t num_chunks_ = 0;
  std::vector<uint8_t> codes_;
};

class distance_table_t {
public:
  uint32_t num_chunks() const { return num_chunks_; }
  float distance(const uint8_t* code) const;

private:
  friend distance_table_t build_distance_table(const float* query, const model_t& model);

  uint32_t num_chunks_ = 0;
  std::vector<float> distances_;
};

std::vector<uint32_t> make_chunk_offsets(uint32_t dim, uint32_t num_pq_chunks);

void save_pivots(const model_t& model, const std::string& path);
model_t load_pivots(const std::string& path);
code_store_t load_codes(const std::string& path);
distance_table_t build_distance_table(const float* query, const model_t& model);

std::vector<uint8_t> encode(const float* data, size_t num_vectors, const model_t& model);

model_t train(const float* training_data, size_t num_vectors, uint32_t dim, uint32_t num_pq_chunks,
              uint32_t max_iterations = 15, uint32_t seed = 0);

model_t generate_pivots(const std::string& data_path, const std::string& output_prefix,
                        uint32_t num_pq_chunks, double sampling_rate, uint32_t seed = 0);

void generate_files(const std::string& data_path, const std::string& output_prefix,
                    uint32_t num_pq_chunks, double sampling_rate, uint32_t seed = 0);

} // namespace powerlaw_ann::pq

#endif // PQ_PQ
