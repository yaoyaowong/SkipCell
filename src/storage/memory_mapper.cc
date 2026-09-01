#include "storage/memory_mapper.h"

#include "common/logger.h"

#include <iostream>
#include <sstream>

using namespace powerlaw_ann;

memory_mapper_t::memory_mapper_t(const std::string& filename) : memory_mapper_t(filename.c_str()) {}

memory_mapper_t::memory_mapper_t(const char* filename)
#ifdef _WINDOWS
    : file_name_(filename)
#endif
{
#ifndef _WINDOWS
  fd_ = open(filename, O_RDONLY);
  if (fd_ <= 0) {
    std::cerr << "Inner vertices file not found" << std::endl;
    return;
  }
  struct stat sb;
  if (fstat(fd_, &sb) != 0) {
    std::cerr << "Inner vertices file not dound. " << std::endl;
    return;
  }
  file_size_ = sb.st_size;
  powerlaw_ann::cout << "File Size: " << file_size_ << std::endl;
  buf_ = (char*) mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
#else
  bare_file_ = CreateFileA(filename, GENERIC_READ | GENERIC_EXECUTE, 0, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
  if (bare_file_ == nullptr) {
    std::ostringstream message;
    message << "CreateFileA(" << filename << ") failed with error " << GetLastError() << std::endl;
    std::cerr << message.str();
    throw std::exception(message.str().c_str());
  }

  fd_ = CreateFileMapping(bare_file_, NULL, PAGE_EXECUTE_READ, 0, 0, NULL);
  if (fd_ == nullptr) {
    std::ostringstream message;
    message << "CreateFileMapping(" << filename << ") failed with error " << GetLastError()
            << std::endl;
    std::cerr << message.str() << std::endl;
    throw std::exception(message.str().c_str());
  }

  buf_ = (char*) MapViewOfFile(fd_, FILE_MAP_READ, 0, 0, 0);
  if (buf_ == nullptr) {
    std::ostringstream message;
    message << "MapViewOfFile(" << filename << ") failed with error: " << GetLastError()
            << std::endl;
    std::cerr << message.str() << std::endl;
    throw std::exception(message.str().c_str());
  }

  LARGE_INTEGER file_size;
  if (TRUE == GetFileSizeEx(bare_file_, &file_size)) {
    file_size_ = file_size.QuadPart; // take the 64-bit value
    powerlaw_ann::cout << "File Size: " << file_size_ << std::endl;
  } else {
    std::cerr << "Failed to get size of file " << filename << std::endl;
  }
#endif
}
char* memory_mapper_t::get_buf() { return buf_; }

size_t memory_mapper_t::get_file_size() { return file_size_; }

memory_mapper_t::~memory_mapper_t() {
#ifndef _WINDOWS
  if (munmap(buf_, file_size_) != 0)
    std::cerr << "ERROR unmapping. CHECK!" << std::endl;
  close(fd_);
#else
  if (FALSE == UnmapViewOfFile(buf_)) {
    std::cerr << "Unmap view of file failed. Error: " << GetLastError() << std::endl;
  }

  if (FALSE == CloseHandle(fd_)) {
    std::cerr << "Failed to close memory mapped file. Error: " << GetLastError() << std::endl;
  }

  if (FALSE == CloseHandle(bare_file_)) {
    std::cerr << "Failed to close file: " << file_name_ << " Error: " << GetLastError()
              << std::endl;
  }

#endif
}
