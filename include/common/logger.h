#ifndef COMMON_LOGGER
#define COMMON_LOGGER

#include "common/platform_compat.h"

#include <functional>
#include <iostream>

#ifdef EXEC_ENV_OLS
#ifndef ENABLE_CUSTOM_LOGGER
#define ENABLE_CUSTOM_LOGGER
#endif // !ENABLE_CUSTOM_LOGGER
#endif // EXEC_ENV_OLS

namespace powerlaw_ann {
#ifdef ENABLE_CUSTOM_LOGGER
POWERLAWANN_DLLEXPORT extern std::basic_ostream<char> cout;
POWERLAWANN_DLLEXPORT extern std::basic_ostream<char> cerr;
#else
using std::cerr;
using std::cout;
#endif

enum class POWERLAWANN_DLLEXPORT log_level_t { LL_Info = 0, LL_Error, LL_Count };

#ifdef ENABLE_CUSTOM_LOGGER
POWERLAWANN_DLLEXPORT void set_custom_logger(std::function<void(log_level_t, const char*)> logger);
#endif
} // namespace powerlaw_ann

#endif // COMMON_LOGGER
