#ifndef COMMON_ARCH_COMPAT
#define COMMON_ARCH_COMPAT

#include <cstddef>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define POWERLAWANN_ARCH_X86 1
#else
#define POWERLAWANN_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
#define POWERLAWANN_ARCH_ARM 1
#else
#define POWERLAWANN_ARCH_ARM 0
#endif

#if POWERLAWANN_ARCH_X86
#include <immintrin.h>
#endif

namespace powerlaw_ann {

inline void cpu_prefetch_t0(const void* ptr) {
#if POWERLAWANN_ARCH_X86
  _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#else
  __builtin_prefetch(ptr, 0, 3);
#endif
}

inline void cpu_prefetch_t1(const void* ptr) {
#if POWERLAWANN_ARCH_X86
  _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T1);
#else
  __builtin_prefetch(ptr, 0, 2);
#endif
}

} // namespace powerlaw_ann

#endif // COMMON_ARCH_COMPAT
