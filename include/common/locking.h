#ifndef COMMON_LOCKING
#define COMMON_LOCKING

#include <mutex>

#ifdef _WINDOWS
#include "windows_slim_lock.h"
#endif

namespace powerlaw_ann {
#ifdef _WINDOWS
using non_recursive_mutex_t = windows_exclusive_slim_lock;
using lock_guard_t = windows_exclusive_slim_lock_guard;
#else
using non_recursive_mutex_t = std::mutex;
using lock_guard_t = std::lock_guard<non_recursive_mutex_t>;
#endif
} // namespace powerlaw_ann

#endif // COMMON_LOCKING
