#include "pq/pq.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace powerlaw_ann::pq {
namespace {

constexpr size_t k_metadata_size = 4096;
constexpr size_t k_encode_block_vectors = 16384;

size_t checked_matrix_bytes(size_t rows, size_t cols, size_t element_size) {
  if (rows != 0 && cols > std::numeric_limits<size_t>::max() / rows) {
    throw std::overflow_error("PQ matrix element count overflows");
  }
  const size_t elements = rows * cols;
  if (elements > (std::numeric_limits<size_t>::max() - 2 * sizeof(uint32_t)) / element_size) {
    throw std::overflow_error("PQ matrix byte size overflows");
  }
  return 2 * sizeof(uint32_t) + elements * element_size;
}

void validate_model(const model_t& model) {
  if (model.dim == 0) {
    throw std::invalid_argument("PQ model dimension must be positive");
  }
  if (model.chunk_offsets.size() < 2 || model.chunk_offsets.front() != 0 ||
      model.chunk_offsets.back() != model.dim) {
    throw std::invalid_argument("PQ model has invalid chunk offsets");
  }
  for (size_t i = 1; i < model.chunk_offsets.size(); ++i) {
    if (model.chunk_offsets[i] <= model.chunk_offsets[i - 1]) {
      throw std::invalid_argument("PQ chunk offsets must be strictly increasing");
    }
  }
  if (model.pivots.size() != static_cast<size_t>(k_num_centers) * model.dim) {
    throw std::invalid_argument("PQ model has invalid pivot count");
  }
  if (model.centroid.size() != model.dim) {
    throw std::invalid_argument("PQ model has invalid centroid count");
  }
}

template <typename T>
void write_matrix(std::fstream& out, size_t offset, uint32_t rows, uint32_t cols, const T* data) {
  out.seekp(static_cast<std::streamoff>(offset));
  out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  const size_t count = static_cast<size_t>(rows) * cols;
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
  if (!out) {
    throw std::runtime_error("failed writing PQ matrix");
  }
}

template <typename T>
std::vector<T> read_matrix(std::ifstream& in, size_t offset, uint32_t expected_rows,
                           uint32_t expected_cols) {
  in.seekg(static_cast<std::streamoff>(offset));
  uint32_t rows = 0;
  uint32_t cols = 0;
  in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  if (!in || rows != expected_rows || cols != expected_cols) {
    throw std::runtime_error("PQ matrix shape does not match expected layout");
  }
  std::vector<T> values(static_cast<size_t>(rows) * cols);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!in) {
    throw std::runtime_error("truncated PQ matrix");
  }
  return values;
}

float squared_l2(const float* lhs, const float* rhs, size_t dim) {
  float distance = 0.0f;
  for (size_t d = 0; d < dim; ++d) {
    const float diff = lhs[d] - rhs[d];
    distance += diff * diff;
  }
  return distance;
}

uint32_t nearest_center(const float* point, const float* centers, size_t num_centers, size_t dim,
                        float* distance_out = nullptr) {
  float best_distance = std::numeric_limits<float>::max();
  uint32_t best_center = 0;
  for (uint32_t center = 0; center < num_centers; ++center) {
    const float distance = squared_l2(point, centers + static_cast<size_t>(center) * dim, dim);
    if (distance < best_distance) {
      best_distance = distance;
      best_center = center;
    }
  }
  if (distance_out != nullptr) {
    *distance_out = best_distance;
  }
  return best_center;
}

void select_kmeans_pp(const std::vector<float>& data, size_t num_vectors, size_t dim,
                      std::vector<float>& centers, std::mt19937& generator) {
  std::uniform_int_distribution<size_t> first_distribution(0, num_vectors - 1);
  const size_t first = first_distribution(generator);
  std::memcpy(centers.data(), data.data() + first * dim, dim * sizeof(float));

  std::vector<float> min_distances(num_vectors);
  for (size_t point = 0; point < num_vectors; ++point) {
    min_distances[point] = squared_l2(data.data() + point * dim, data.data() + first * dim, dim);
  }

  std::uniform_real_distribution<double> unit_distribution(0.0, 1.0);
  for (uint32_t selected = 1; selected < k_num_centers; ++selected) {
    double sum = 0.0;
    for (const float distance : min_distances) {
      sum += distance;
    }

    size_t selected_point = first_distribution(generator);
    if (sum > 0.0) {
      const double dart = unit_distribution(generator) * sum;
      double prefix = 0.0;
      selected_point = num_vectors - 1;
      for (size_t point = 0; point < num_vectors; ++point) {
        prefix += min_distances[point];
        if (dart < prefix) {
          selected_point = point;
          break;
        }
      }
    }

    float* destination = centers.data() + static_cast<size_t>(selected) * dim;
    const float* source = data.data() + selected_point * dim;
    std::memcpy(destination, source, dim * sizeof(float));
    for (size_t point = 0; point < num_vectors; ++point) {
      min_distances[point] =
          std::min(min_distances[point], squared_l2(data.data() + point * dim, source, dim));
    }
  }
}

void run_lloyd(const std::vector<float>& data, size_t num_vectors, size_t dim,
               uint32_t max_iterations, std::vector<float>& centers) {
  std::vector<uint32_t> assignments(num_vectors);
  std::vector<double> sums(static_cast<size_t>(k_num_centers) * dim);
  std::vector<size_t> counts(k_num_centers);
  float previous_residual = std::numeric_limits<float>::max();

  for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t point = 0; point < num_vectors; ++point) {
      const uint32_t center =
          nearest_center(data.data() + point * dim, centers.data(), k_num_centers, dim);
      assignments[point] = center;
      ++counts[center];
      for (size_t d = 0; d < dim; ++d) {
        sums[static_cast<size_t>(center) * dim + d] += data[point * dim + d];
      }
    }

    std::fill(centers.begin(), centers.end(), 0.0f);
    for (uint32_t center = 0; center < k_num_centers; ++center) {
      if (counts[center] == 0) {
        continue;
      }
      for (size_t d = 0; d < dim; ++d) {
        centers[static_cast<size_t>(center) * dim + d] = static_cast<float>(
            sums[static_cast<size_t>(center) * dim + d] / static_cast<double>(counts[center]));
      }
    }

    double residual = 0.0;
    for (size_t point = 0; point < num_vectors; ++point) {
      residual += squared_l2(data.data() + point * dim,
                             centers.data() + static_cast<size_t>(assignments[point]) * dim, dim);
    }
    const float current_residual = static_cast<float>(residual);
    if (current_residual <= std::numeric_limits<float>::epsilon()) {
      break;
    }
    if (iteration != 0 && (previous_residual - current_residual) / current_residual < 0.00001f) {
      break;
    }
    previous_residual = current_residual;
  }
}

struct fbin_header_t {
  uint32_t rows = 0;
  uint32_t dim = 0;
};

fbin_header_t read_fbin_header(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open float32 vector file: " + path);
  }
  fbin_header_t header;
  in.read(reinterpret_cast<char*>(&header.rows), sizeof(header.rows));
  in.read(reinterpret_cast<char*>(&header.dim), sizeof(header.dim));
  if (!in || header.rows == 0 || header.dim == 0) {
    throw std::runtime_error("invalid float32 vector file header: " + path);
  }

  const uintmax_t element_count =
      static_cast<uintmax_t>(header.rows) * static_cast<uintmax_t>(header.dim);
  if (element_count >
      (std::numeric_limits<uintmax_t>::max() - 2 * sizeof(uint32_t)) / sizeof(float)) {
    throw std::overflow_error("float32 vector file size overflows");
  }
  const uintmax_t expected_size = 2 * sizeof(uint32_t) + element_count * sizeof(float);
  if (std::filesystem::file_size(path) != expected_size) {
    throw std::runtime_error("float32 vector file size does not match its header: " + path);
  }
  return header;
}

std::string temporary_path(const std::string& final_path) {
  std::random_device random_device;
  return final_path + ".tmp." + std::to_string(random_device());
}

void ensure_output_parent_directory(const std::string& output_prefix) {
  const std::filesystem::path parent = std::filesystem::path(output_prefix).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

} // namespace

std::vector<uint32_t> make_chunk_offsets(uint32_t dim, uint32_t num_pq_chunks) {
  if (dim == 0) {
    throw std::invalid_argument("PQ dimension must be positive");
  }
  if (num_pq_chunks == 0 || num_pq_chunks > dim) {
    throw std::invalid_argument("PQ chunk count must be in [1, dim]");
  }

  const uint32_t low = dim / num_pq_chunks;
  const uint32_t high_count = dim - low * num_pq_chunks;

  std::vector<uint32_t> offsets;
  offsets.reserve(static_cast<size_t>(num_pq_chunks) + 1);
  offsets.push_back(0);
  for (uint32_t chunk = 0; chunk < num_pq_chunks; ++chunk) {
    const uint32_t chunk_size = low + (chunk < high_count ? 1u : 0u);
    offsets.push_back(offsets.back() + chunk_size);
  }
  return offsets;
}

const uint8_t* code_store_t::code(uint32_t vector_id) const {
  if (vector_id >= num_vectors_) {
    throw std::out_of_range("PQ vector id is outside compressed code store");
  }
  return codes_.data() + static_cast<size_t>(vector_id) * num_chunks_;
}

float distance_table_t::distance(const uint8_t* code) const {
  if (code == nullptr) {
    throw std::invalid_argument("PQ code must not be null");
  }
  float distance = 0.0f;
  for (uint32_t chunk = 0; chunk < num_chunks_; ++chunk) {
    distance += distances_[static_cast<size_t>(chunk) * k_num_centers + code[chunk]];
  }
  return distance;
}

void save_pivots(const model_t& model, const std::string& path) {
  validate_model(model);

  const size_t pivot_bytes = checked_matrix_bytes(k_num_centers, model.dim, sizeof(float));
  const size_t centroid_bytes = checked_matrix_bytes(model.dim, 1, sizeof(float));
  const size_t chunk_bytes = checked_matrix_bytes(model.chunk_offsets.size(), 1, sizeof(uint32_t));
  if (pivot_bytes > std::numeric_limits<size_t>::max() - k_metadata_size ||
      centroid_bytes > std::numeric_limits<size_t>::max() - k_metadata_size - pivot_bytes ||
      chunk_bytes >
          std::numeric_limits<size_t>::max() - k_metadata_size - pivot_bytes - centroid_bytes) {
    throw std::overflow_error("PQ pivot file size overflows");
  }

  const std::array<size_t, 4> offsets = {k_metadata_size, k_metadata_size + pivot_bytes,
                                         k_metadata_size + pivot_bytes + centroid_bytes,
                                         k_metadata_size + pivot_bytes + centroid_bytes +
                                             chunk_bytes};

  std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot open PQ pivot output: " + path);
  }

  std::array<char, k_metadata_size> metadata{};
  out.write(metadata.data(), static_cast<std::streamsize>(metadata.size()));
  write_matrix(out, offsets[0], k_num_centers, model.dim, model.pivots.data());
  write_matrix(out, offsets[1], model.dim, 1, model.centroid.data());
  write_matrix(out, offsets[2], static_cast<uint32_t>(model.chunk_offsets.size()), 1,
               model.chunk_offsets.data());
  write_matrix(out, 0, static_cast<uint32_t>(offsets.size()), 1, offsets.data());
  out.flush();
  if (!out) {
    throw std::runtime_error("failed finalizing PQ pivot output: " + path);
  }
}

model_t load_pivots(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open PQ pivot file: " + path);
  }
  const size_t file_size = std::filesystem::file_size(path);
  if (file_size < k_metadata_size) {
    throw std::runtime_error("PQ pivot file is smaller than metadata region");
  }

  const auto offsets_vector = read_matrix<size_t>(in, 0, 4, 1);
  const std::array<size_t, 4> offsets = {offsets_vector[0], offsets_vector[1], offsets_vector[2],
                                         offsets_vector[3]};
  if (offsets[0] != k_metadata_size || offsets[0] >= offsets[1] || offsets[1] >= offsets[2] ||
      offsets[2] >= offsets[3] || offsets[3] != file_size) {
    throw std::runtime_error("PQ pivot offsets are invalid");
  }

  in.seekg(static_cast<std::streamoff>(offsets[0]));
  uint32_t centers = 0;
  uint32_t dim = 0;
  in.read(reinterpret_cast<char*>(&centers), sizeof(centers));
  in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!in || centers != k_num_centers || dim == 0) {
    throw std::runtime_error("PQ pivot matrix header is invalid");
  }

  model_t model;
  model.dim = dim;
  model.pivots = read_matrix<float>(in, offsets[0], k_num_centers, dim);
  model.centroid = read_matrix<float>(in, offsets[1], dim, 1);

  in.seekg(static_cast<std::streamoff>(offsets[2]));
  uint32_t chunk_count = 0;
  uint32_t chunk_cols = 0;
  in.read(reinterpret_cast<char*>(&chunk_count), sizeof(chunk_count));
  in.read(reinterpret_cast<char*>(&chunk_cols), sizeof(chunk_cols));
  if (!in || chunk_count < 2 || chunk_cols != 1) {
    throw std::runtime_error("PQ chunk-offset matrix header is invalid");
  }
  model.chunk_offsets = read_matrix<uint32_t>(in, offsets[2], chunk_count, 1);
  validate_model(model);
  return model;
}

code_store_t load_codes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open PQ compressed file: " + path);
  }

  code_store_t store;
  in.read(reinterpret_cast<char*>(&store.num_vectors_), sizeof(store.num_vectors_));
  in.read(reinterpret_cast<char*>(&store.num_chunks_), sizeof(store.num_chunks_));
  if (!in || store.num_vectors_ == 0 || store.num_chunks_ == 0) {
    throw std::runtime_error("invalid PQ compressed file header: " + path);
  }
  const uintmax_t code_count = static_cast<uintmax_t>(store.num_vectors_) * store.num_chunks_;
  if (code_count > std::numeric_limits<size_t>::max()) {
    throw std::overflow_error("PQ compressed code count overflows");
  }
  const uintmax_t expected_size = 2 * sizeof(uint32_t) + code_count;
  if (std::filesystem::file_size(path) != expected_size) {
    throw std::runtime_error("PQ compressed file size does not match header: " + path);
  }

  store.codes_.resize(static_cast<size_t>(code_count));
  in.read(reinterpret_cast<char*>(store.codes_.data()),
          static_cast<std::streamsize>(store.codes_.size()));
  if (!in) {
    throw std::runtime_error("truncated PQ compressed file: " + path);
  }
  return store;
}

distance_table_t build_distance_table(const float* query, const model_t& model) {
  validate_model(model);
  if (query == nullptr) {
    throw std::invalid_argument("PQ query must not be null");
  }

  distance_table_t table;
  table.num_chunks_ = model.num_chunks();
  table.distances_.resize(static_cast<size_t>(table.num_chunks_) * k_num_centers);
  for (uint32_t chunk = 0; chunk < table.num_chunks_; ++chunk) {
    const uint32_t begin = model.chunk_offsets[chunk];
    const uint32_t end = model.chunk_offsets[chunk + 1];
    for (uint32_t center = 0; center < k_num_centers; ++center) {
      float distance = 0.0f;
      for (uint32_t dim = begin; dim < end; ++dim) {
        const float centered_query = query[dim] - model.centroid[dim];
        const float diff =
            centered_query - model.pivots[static_cast<size_t>(center) * model.dim + dim];
        distance += diff * diff;
      }
      table.distances_[static_cast<size_t>(chunk) * k_num_centers + center] = distance;
    }
  }
  return table;
}

std::vector<uint8_t> encode(const float* data, size_t num_vectors, const model_t& model) {
  validate_model(model);
  if (data == nullptr && num_vectors != 0) {
    throw std::invalid_argument("PQ encoding data is null");
  }
  if (num_vectors > std::numeric_limits<size_t>::max() / model.num_chunks()) {
    throw std::overflow_error("PQ code count overflows");
  }
  if (num_vectors > std::numeric_limits<size_t>::max() / model.dim) {
    throw std::overflow_error("PQ input element count overflows");
  }

  std::vector<uint8_t> codes(num_vectors * model.num_chunks());
  std::vector<float> chunk_data(model.dim);
  std::vector<float> chunk_centers(static_cast<size_t>(k_num_centers) * model.dim);

  for (uint32_t chunk = 0; chunk < model.num_chunks(); ++chunk) {
    const uint32_t begin = model.chunk_offsets[chunk];
    const uint32_t end = model.chunk_offsets[chunk + 1];
    const size_t chunk_dim = end - begin;

    for (uint32_t center = 0; center < k_num_centers; ++center) {
      std::memcpy(chunk_centers.data() + static_cast<size_t>(center) * chunk_dim,
                  model.pivots.data() + static_cast<size_t>(center) * model.dim + begin,
                  chunk_dim * sizeof(float));
    }
    for (size_t point = 0; point < num_vectors; ++point) {
      for (size_t d = 0; d < chunk_dim; ++d) {
        chunk_data[d] = data[point * model.dim + begin + d] - model.centroid[begin + d];
      }
      const uint32_t center =
          nearest_center(chunk_data.data(), chunk_centers.data(), k_num_centers, chunk_dim);
      codes[point * model.num_chunks() + chunk] = static_cast<uint8_t>(center);
    }
  }
  return codes;
}

model_t train(const float* training_data, size_t num_vectors, uint32_t dim, uint32_t num_pq_chunks,
              uint32_t max_iterations, uint32_t seed) {
  if (training_data == nullptr) {
    throw std::invalid_argument("PQ training data is null");
  }
  if (num_vectors < k_num_centers) {
    throw std::invalid_argument("PQ training requires at least 256 vectors");
  }
  if (max_iterations == 0) {
    throw std::invalid_argument("PQ training iteration count must be positive");
  }
  if (dim != 0 && num_vectors > std::numeric_limits<size_t>::max() / dim) {
    throw std::overflow_error("PQ training element count overflows");
  }

  model_t model;
  model.dim = dim;
  model.chunk_offsets = make_chunk_offsets(dim, num_pq_chunks);
  model.pivots.resize(static_cast<size_t>(k_num_centers) * dim);
  model.centroid.assign(dim, 0.0f);

  std::mt19937 generator(seed);
  for (uint32_t chunk = 0; chunk < num_pq_chunks; ++chunk) {
    const uint32_t begin = model.chunk_offsets[chunk];
    const uint32_t end = model.chunk_offsets[chunk + 1];
    const size_t chunk_dim = end - begin;

    std::vector<float> chunk_data(num_vectors * chunk_dim);
    for (size_t point = 0; point < num_vectors; ++point) {
      std::memcpy(chunk_data.data() + point * chunk_dim, training_data + point * dim + begin,
                  chunk_dim * sizeof(float));
    }

    std::vector<float> chunk_centers(static_cast<size_t>(k_num_centers) * chunk_dim);
    select_kmeans_pp(chunk_data, num_vectors, chunk_dim, chunk_centers, generator);
    run_lloyd(chunk_data, num_vectors, chunk_dim, max_iterations, chunk_centers);

    for (uint32_t center = 0; center < k_num_centers; ++center) {
      std::memcpy(model.pivots.data() + static_cast<size_t>(center) * dim + begin,
                  chunk_centers.data() + static_cast<size_t>(center) * chunk_dim,
                  chunk_dim * sizeof(float));
    }
  }
  return model;
}

model_t generate_pivots(const std::string& data_path, const std::string& output_prefix,
                        uint32_t num_pq_chunks, double sampling_rate, uint32_t seed) {
  if (!(sampling_rate > 0.0 && sampling_rate <= 1.0)) {
    throw std::invalid_argument("PQ sampling rate must be in (0, 1]");
  }

  const fbin_header_t header = read_fbin_header(data_path);
  const auto expected_offsets = make_chunk_offsets(header.dim, num_pq_chunks);
  ensure_output_parent_directory(output_prefix);

  const std::string pivots_path = output_prefix + "_pq_pivots.bin";
  if (std::filesystem::exists(pivots_path)) {
    model_t model = load_pivots(pivots_path);
    if (model.dim != header.dim || model.chunk_offsets != expected_offsets) {
      throw std::runtime_error("existing PQ pivot file is incompatible with input");
    }
    return model;
  }

  std::ifstream sample_reader(data_path, std::ios::binary);
  sample_reader.seekg(2 * sizeof(uint32_t));
  std::vector<float> row(header.dim);
  std::vector<float> sample;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (uint32_t point = 0; point < header.rows; ++point) {
    sample_reader.read(reinterpret_cast<char*>(row.data()),
                       static_cast<std::streamsize>(row.size() * sizeof(float)));
    if (!sample_reader) {
      throw std::runtime_error("truncated float32 vector data: " + data_path);
    }
    if (sampling_rate >= 1.0 || distribution(generator) < sampling_rate) {
      sample.insert(sample.end(), row.begin(), row.end());
    }
  }

  const size_t sample_size = sample.size() / header.dim;
  if (sample_size < k_num_centers) {
    throw std::invalid_argument("PQ sampling produced fewer than 256 vectors");
  }

  model_t model = train(sample.data(), sample_size, header.dim, num_pq_chunks, 15, seed);
  const std::string pivot_temp = temporary_path(pivots_path);
  try {
    save_pivots(model, pivot_temp);
    std::filesystem::rename(pivot_temp, pivots_path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(pivot_temp, ignored);
    throw;
  }
  return model;
}

void generate_files(const std::string& data_path, const std::string& output_prefix,
                    uint32_t num_pq_chunks, double sampling_rate, uint32_t seed) {
  const fbin_header_t header = read_fbin_header(data_path);
  const model_t model =
      generate_pivots(data_path, output_prefix, num_pq_chunks, sampling_rate, seed);

  const std::string compressed_path = output_prefix + "_pq_compressed.bin";
  const std::string compressed_temp = temporary_path(compressed_path);
  try {
    std::ofstream compressed(compressed_temp, std::ios::binary | std::ios::trunc);
    if (!compressed) {
      throw std::runtime_error("cannot open PQ compressed output: " + compressed_temp);
    }
    compressed.write(reinterpret_cast<const char*>(&header.rows), sizeof(header.rows));
    compressed.write(reinterpret_cast<const char*>(&num_pq_chunks), sizeof(num_pq_chunks));

    std::ifstream data_reader(data_path, std::ios::binary);
    data_reader.seekg(2 * sizeof(uint32_t));
    const size_t block_capacity =
        std::min(static_cast<size_t>(header.rows), k_encode_block_vectors);
    std::vector<float> block(block_capacity * header.dim);
    size_t remaining = header.rows;
    while (remaining != 0) {
      const size_t current = std::min(remaining, block_capacity);
      data_reader.read(reinterpret_cast<char*>(block.data()),
                       static_cast<std::streamsize>(current * header.dim * sizeof(float)));
      if (!data_reader) {
        throw std::runtime_error("truncated float32 vector data during PQ encoding");
      }
      const auto codes = encode(block.data(), current, model);
      compressed.write(reinterpret_cast<const char*>(codes.data()),
                       static_cast<std::streamsize>(codes.size()));
      if (!compressed) {
        throw std::runtime_error("failed writing PQ compressed output");
      }
      remaining -= current;
    }
    compressed.close();
    if (!compressed) {
      throw std::runtime_error("failed finalizing PQ compressed output");
    }

    std::filesystem::rename(compressed_temp, compressed_path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(compressed_temp, ignored);
    throw;
  }
}

} // namespace powerlaw_ann::pq
