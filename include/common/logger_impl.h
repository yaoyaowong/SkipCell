#ifndef COMMON_LOGGER_IMPL
#define COMMON_LOGGER_IMPL

#include "common/logger.h" // IWYU pragma: keep

#ifdef ENABLE_CUSTOM_LOGGER
#include <cstdio>
#include <mutex>
#include <streambuf>

namespace powerlaw_ann {
class ann_stream_buf_t : public std::basic_streambuf<char> {
public:
  POWERLAWANN_DLLEXPORT explicit ann_stream_buf_t(FILE* fp);
  POWERLAWANN_DLLEXPORT ~ann_stream_buf_t();

  POWERLAWANN_DLLEXPORT bool is_open() const {
    return true; // because stdout and stderr are always open.
  }
  POWERLAWANN_DLLEXPORT void close();
  POWERLAWANN_DLLEXPORT virtual int underflow();
  POWERLAWANN_DLLEXPORT virtual int overflow(int c);
  POWERLAWANN_DLLEXPORT virtual int sync();

private:
  FILE* fp_;
  char* buf_;
  int buf_index_;
  std::mutex mutex_;
  log_level_t log_level_;

  int flush();
  void log_impl(char* str, int numchars);

  // Why the two buffer-sizes? If we are running normally, we are basically
  // interacting with a character output system, so we short-circuit the
  // output process by keeping an empty buffer and writing each character
  // to stdout/stderr. But if we are running in OLS, we have to take all
  // the text that is written to powerlaw_ann::cout/powerlaw_ann::cerr, consolidate it
  // and push it out in one-shot, because the OLS infra does not give us
  // character based output. Therefore, we use a larger buffer that is large
  // enough to store the longest message, and continuously add characters
  // to it. When the calling code outputs a std::endl or std::flush, sync()
  // will be called and will output a log level, component name, and the text
  // that has been collected. (sync() is also called if the buffer is full, so
  // overflows/missing text are not a concern).
  // This implies calling code _must_ either print std::endl or std::flush
  // to ensure that the message is written immediately.

  static const int BUFFER_SIZE = 1024;

  ann_stream_buf_t(const ann_stream_buf_t&);
  ann_stream_buf_t& operator=(const ann_stream_buf_t&);
};
} // namespace powerlaw_ann

#endif // ENABLE_CUSTOM_LOGGER

#endif // COMMON_LOGGER_IMPL
