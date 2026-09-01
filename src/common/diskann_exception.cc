#include "common/diskann_exception.h"

#include <string>

namespace powerlaw_ann {
diskann_exception_t::diskann_exception_t(const std::string& message, int error_code)
    : std::runtime_error(message), error_code_(error_code) {}

std::string package_string(const std::string& item_name, const std::string& item_val) {
  return std::string("[") + item_name + ": " + std::string(item_val) + std::string("]");
}

diskann_exception_t::diskann_exception_t(const std::string& message, int error_code,
                                         const std::string& func_sig, const std::string& file_name,
                                         uint32_t line_num)
    : diskann_exception_t(package_string(std::string("FUNC"), func_sig) +
                              package_string(std::string("FILE"), file_name) +
                              package_string(std::string("LINE"), std::to_string(line_num)) + "  " +
                              message,
                          error_code) {}

file_exception_t::file_exception_t(const std::string& filename, std::system_error& e,
                                   const std::string& func_sig, const std::string& file_name,
                                   uint32_t line_num)
    : diskann_exception_t(std::string(" While opening file \'") + filename +
                              std::string("\', error code: ") + std::to_string(e.code().value()) +
                              "  " + e.code().message(),
                          e.code().value(), func_sig, file_name, line_num) {}

} // namespace powerlaw_ann
