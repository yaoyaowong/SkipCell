#ifndef STORAGE_MEMORY_MAPPER
#define STORAGE_MEMORY_MAPPER

#ifndef _WINDOWS
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#else
#include <Windows.h>
#endif
#include <string>

namespace powerlaw_ann {
class memory_mapper_t {
private:
#ifndef _WINDOWS
  int fd_;
#else
  HANDLE bare_file_;
  HANDLE fd_;
  std::string file_name_;

#endif
  char* buf_;
  size_t file_size_;

public:
  memory_mapper_t(const char* filename);
  memory_mapper_t(const std::string& filename);

  char* get_buf();
  size_t get_file_size();

  ~memory_mapper_t();
};
} // namespace powerlaw_ann

#endif // STORAGE_MEMORY_MAPPER
