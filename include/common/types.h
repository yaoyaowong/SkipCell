#ifndef COMMON_TYPES
#define COMMON_TYPES

#include "common/any_wrappers.h"

#include <any>
#include <cstddef>
#include <cstdint>

namespace powerlaw_ann {
typedef uint32_t location_t;

using data_type_t = std::any;
using tag_type_t = std::any;
using label_type_t = std::any;
using tag_vector_t = any_wrapper::any_vector_t;
using data_vector_t = any_wrapper::any_vector_t;
using label_vector_t = any_wrapper::any_vector_t;
using tag_robin_set_t = any_wrapper::any_robin_set_t;
} // namespace powerlaw_ann

#endif // COMMON_TYPES
