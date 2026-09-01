#include "common/logger_impl.h"
#include "common/platform_compat.h"

#include <iostream>

#ifdef ENABLE_CUSTOM_LOGGER
#include "common/diskann_exception.h"

#include <cstring>
#include <functional>
#endif

namespace powerlaw_ann {

#ifdef ENABLE_CUSTOM_LOGGER
POWERLAWANN_DLLEXPORT ann_stream_buf_t cout_buf(stdout);
POWERLAWANN_DLLEXPORT ann_stream_buf_t cerr_buf(stderr);

POWERLAWANN_DLLEXPORT std::basic_ostream<char> cout(&cout_buf);
POWERLAWANN_DLLEXPORT std::basic_ostream<char> cerr(&cerr_buf);
std::function<void(log_level_t, const char*)> g_logger;

void set_custom_logger(std::function<void(log_level_t, const char*)> logger) {
  g_logger = logger;
  powerlaw_ann::cout << "Set Custom Logger" << std::endl;
}

ann_stream_buf_t::ann_stream_buf_t(FILE* fp) {
  if (fp == nullptr) {
    throw powerlaw_ann::diskann_exception_t(
        "File pointer passed to ann_stream_buf_t() cannot be null", -1);
  }
  if (fp != stdout && fp != stderr) {
    throw powerlaw_ann::diskann_exception_t("The custom logger only supports stdout and stderr.",
                                            -1);
  }
  fp_ = fp;
  log_level_ = (fp_ == stdout) ? log_level_t::LL_Info : log_level_t::LL_Error;
  buf_ = new char[BUFFER_SIZE + 1]; // See comment in the header

  std::memset(buf_, 0, (BUFFER_SIZE) * sizeof(char));
  setp(buf_, buf_ + BUFFER_SIZE - 1);
}

ann_stream_buf_t::~ann_stream_buf_t() {
  sync();
  fp_ = nullptr; // we'll not close because we can't.
  delete[] buf_;
}

int ann_stream_buf_t::overflow(int c) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (c != EOF) {
    *pptr() = (char) c;
    pbump(1);
  }
  flush();
  return c;
}

int ann_stream_buf_t::sync() {
  std::lock_guard<std::mutex> lock(mutex_);
  flush();
  return 0;
}

int ann_stream_buf_t::underflow() {
  throw powerlaw_ann::diskann_exception_t("Attempt to read on streambuf meant only for writing.",
                                          -1);
}

int ann_stream_buf_t::flush() {
  const int num = (int) (pptr() - pbase());
  log_impl(pbase(), num);
  pbump(-num);
  return num;
}
void ann_stream_buf_t::log_impl(char* str, int num) {
  str[num] = '\0'; // Safe. See the c'tor.
  // Invoke the OLS custom logging function.
  if (g_logger) {
    g_logger(log_level_, str);
  }
}
#else
using std::cerr;
using std::cout;
#endif

} // namespace powerlaw_ann
