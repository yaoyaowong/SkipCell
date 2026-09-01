#ifndef INDEX_FILTER_UTILS
#define INDEX_FILTER_UTILS

#include "third/tsl/robin_map.h"
#include "third/tsl/robin_set.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE file_handle_t;
#else
#include <unistd.h>
typedef int file_handle_t;
#endif

#ifndef _WINDOWS
#include <sys/uio.h>
#endif

#include "common/platform_compat.h"
#include "common/utils.h"
#include "storage/memory_mapper.h"

// custom types (for readability)
typedef tsl::robin_set<std::string> label_set_t;
typedef std::string path_t;

// structs for returning multiple items from a function
typedef std::tuple<std::vector<label_set_t>, tsl::robin_map<std::string, uint32_t>,
                   tsl::robin_set<std::string>>
    parse_label_file_return_values_t;
typedef std::tuple<std::vector<std::vector<uint32_t>>, uint64_t> load_label_index_return_values_t;

namespace powerlaw_ann {
template <typename T>
POWERLAWANN_DLLEXPORT void generate_label_indices(path_t input_data_path,
                                                  path_t final_index_path_prefix,
                                                  label_set_t all_labels, unsigned R, unsigned L,
                                                  float alpha, unsigned num_threads);

POWERLAWANN_DLLEXPORT load_label_index_return_values_t
load_label_index(path_t label_index_path, uint32_t label_number_of_points);

template <typename label_t>
POWERLAWANN_DLLEXPORT std::tuple<std::vector<std::vector<label_t>>, tsl::robin_set<label_t>>
parse_formatted_label_file(path_t label_file);

POWERLAWANN_DLLEXPORT parse_label_file_return_values_t
parse_label_file(path_t label_data_path, std::string universal_label);

template <typename T>
POWERLAWANN_DLLEXPORT tsl::robin_map<std::string, std::vector<uint32_t>>
generate_label_specific_vector_files_compat(
    path_t input_data_path, tsl::robin_map<std::string, uint32_t> labels_to_number_of_points,
    std::vector<label_set_t> point_ids_to_labels, label_set_t all_labels);

/*
 * For each label, generates a file containing all vectors that have said label.
 * Also copies data from original bin file to new dimension-aligned file.
 *
 * Utilizes POSIX functions mmap and writev in order to minimize memory
 * overhead, so we include an STL version as well.
 *
 * Each data file is saved under the following format:
 *    input_data_path + "_" + label
 */
#ifndef _WINDOWS
template <typename T>
inline tsl::robin_map<std::string, std::vector<uint32_t>> generate_label_specific_vector_files(
    path_t input_data_path, tsl::robin_map<std::string, uint32_t> labels_to_number_of_points,
    std::vector<label_set_t> point_ids_to_labels, label_set_t all_labels) {
#ifndef _WINDOWS
  auto file_writing_timer = std::chrono::high_resolution_clock::now();
  powerlaw_ann::memory_mapper_t input_data(input_data_path);
  char* input_start = input_data.get_buf();

  uint32_t number_of_points, dimension;
  std::memcpy(&number_of_points, input_start, sizeof(uint32_t));
  std::memcpy(&dimension, input_start + sizeof(uint32_t), sizeof(uint32_t));
  const uint32_t VECTOR_SIZE = dimension * sizeof(T);
  const size_t METADATA = 2 * sizeof(uint32_t);
  if (number_of_points != point_ids_to_labels.size()) {
    std::cerr << "Error: number of points in labels file and data file differ." << std::endl;
    throw;
  }

  tsl::robin_map<std::string, iovec*> label_to_iovec_map;
  tsl::robin_map<std::string, uint32_t> label_to_curr_iovec;
  tsl::robin_map<std::string, std::vector<uint32_t>> label_id_to_orig_id;

  // setup iovec list for each label
  for (const auto& lbl : all_labels) {
    iovec* label_iovecs = (iovec*) malloc(labels_to_number_of_points[lbl] * sizeof(iovec));
    if (label_iovecs == nullptr) {
      throw;
    }
    label_to_iovec_map[lbl] = label_iovecs;
    label_to_curr_iovec[lbl] = 0;
    label_id_to_orig_id[lbl].reserve(labels_to_number_of_points[lbl]);
  }

  // each point added to corresponding per-label iovec list
  for (uint32_t point_id = 0; point_id < number_of_points; point_id++) {
    char* curr_point = input_start + METADATA + (VECTOR_SIZE * point_id);
    iovec curr_iovec;

    curr_iovec.iov_base = curr_point;
    curr_iovec.iov_len = VECTOR_SIZE;
    for (const auto& lbl : point_ids_to_labels[point_id]) {
      *(label_to_iovec_map[lbl] + label_to_curr_iovec[lbl]) = curr_iovec;
      label_to_curr_iovec[lbl]++;
      label_id_to_orig_id[lbl].push_back(point_id);
    }
  }

  // write each label iovec to resp. file
  for (const auto& lbl : all_labels) {
    int label_input_data_fd;
    path_t curr_label_input_data_path(input_data_path + "_" + lbl);
    uint32_t curr_num_pts = labels_to_number_of_points[lbl];

    label_input_data_fd = open(curr_label_input_data_path.c_str(),
                               O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, (mode_t) 0644);
    if (label_input_data_fd == -1)
      throw;

    // write metadata
    uint32_t metadata[2] = {curr_num_pts, dimension};
    int return_value = write(label_input_data_fd, metadata, sizeof(uint32_t) * 2);
    if (return_value == -1) {
      throw;
    }

    // limits on number of iovec structs per writev means we need to perform
    // multiple writevs
    size_t i = 0;
    while (curr_num_pts > IOV_MAX) {
      return_value =
          writev(label_input_data_fd, (label_to_iovec_map[lbl] + (IOV_MAX * i)), IOV_MAX);
      if (return_value == -1) {
        close(label_input_data_fd);
        throw;
      }
      curr_num_pts -= IOV_MAX;
      i += 1;
    }
    return_value =
        writev(label_input_data_fd, (label_to_iovec_map[lbl] + (IOV_MAX * i)), curr_num_pts);
    if (return_value == -1) {
      close(label_input_data_fd);
      throw;
    }

    free(label_to_iovec_map[lbl]);
    close(label_input_data_fd);
  }

  std::chrono::duration<double> file_writing_time =
      std::chrono::high_resolution_clock::now() - file_writing_timer;
  std::cout << "generated " << all_labels.size()
            << " label-specific vector files for index building in time "
            << file_writing_time.count() << "\n"
            << std::endl;

  return label_id_to_orig_id;
#endif
}
#endif

inline std::vector<uint32_t> load_tags(const std::string& tags_file, const std::string& base_file) {
  const bool tags_enabled = tags_file.empty() ? false : true;
  std::vector<uint32_t> location_to_tag;
  if (tags_enabled) {
    size_t tag_file_ndims, tag_file_npts;
    std::uint32_t* tag_data;
    powerlaw_ann::load_bin<std::uint32_t>(tags_file, tag_data, tag_file_npts, tag_file_ndims);
    if (tag_file_ndims != 1) {
      powerlaw_ann::cerr << "tags file error" << std::endl;
      throw powerlaw_ann::diskann_exception_t("tag file error", -1, __FUNCSIG__, __FILE__,
                                              __LINE__);
    }

    // check if the point count match
    size_t base_file_npts, base_file_ndims;
    powerlaw_ann::get_bin_metadata(base_file, base_file_npts, base_file_ndims);
    if (base_file_npts != tag_file_npts) {
      powerlaw_ann::cerr << "point num in tags file mismatch" << std::endl;
      throw powerlaw_ann::diskann_exception_t("point num in tags file mismatch", -1, __FUNCSIG__,
                                              __FILE__, __LINE__);
    }

    location_to_tag.assign(tag_data, tag_data + tag_file_npts);
    delete[] tag_data;
  }
  return location_to_tag;
}

} // namespace powerlaw_ann

#endif // INDEX_FILTER_UTILS
