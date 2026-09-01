#ifndef COMMON_ANN_ERROR
#define COMMON_ANN_ERROR

#include <stdexcept>
#include <string>

namespace powerlaw_ann {

class ann_exception_t : public std::runtime_error {
public:
  explicit ann_exception_t(const std::string& message) : std::runtime_error(message) {}
};

} // namespace powerlaw_ann

#endif // COMMON_ANN_ERROR
