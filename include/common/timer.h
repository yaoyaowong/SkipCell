#ifndef COMMON_TIMER
#define COMMON_TIMER

#include <chrono>

namespace powerlaw_ann {
class timer_t {
  typedef std::chrono::high_resolution_clock high_resolution_clock_t;
  std::chrono::time_point<high_resolution_clock_t> check_point;

public:
  timer_t() : check_point(high_resolution_clock_t::now()) {}

  void reset() { check_point = high_resolution_clock_t::now(); }

  long long elapsed() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(high_resolution_clock_t::now() -
                                                                 check_point)
        .count();
  }

  float elapsed_seconds() const { return (float) elapsed() / 1000000.0f; }

  std::string elapsed_seconds_for_step(const std::string& step) const {
    return std::string("Time for ") + step + std::string(": ") + std::to_string(elapsed_seconds()) +
           std::string(" seconds");
  }
};
} // namespace powerlaw_ann

#endif // COMMON_TIMER
