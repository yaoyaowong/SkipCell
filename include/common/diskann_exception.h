#ifndef COMMON_DISKANN_EXCEPTION
#define COMMON_DISKANN_EXCEPTION

#include "common/platform_compat.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <system_error>

#ifndef _WINDOWS
#define __FUNCSIG__ __PRETTY_FUNCTION__
#endif

namespace powerlaw_ann {

class diskann_exception_t : public std::runtime_error {
public:
  POWERLAWANN_DLLEXPORT diskann_exception_t(const std::string& message, int error_code);
  POWERLAWANN_DLLEXPORT diskann_exception_t(const std::string& message, int error_code,
                                            const std::string& func_sig,
                                            const std::string& file_name, std::uint32_t line_num);

  int error_code() const noexcept { return error_code_; }

private:
  int error_code_;
};

class file_exception_t : public diskann_exception_t {
public:
  POWERLAWANN_DLLEXPORT file_exception_t(const std::string& filename, std::system_error& e,
                                         const std::string& func_sig, const std::string& file_name,
                                         std::uint32_t line_num);
};
} // namespace powerlaw_ann

#endif // COMMON_DISKANN_EXCEPTION
