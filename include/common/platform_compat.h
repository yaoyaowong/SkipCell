#ifndef COMMON_PLATFORM_COMPAT
#define COMMON_PLATFORM_COMPAT

#ifdef _WINDOWS

#ifdef _WINDLL
#define POWERLAWANN_DLLEXPORT __declspec(dllexport)
#else
#define POWERLAWANN_DLLEXPORT __declspec(dllimport)
#endif

#else
#define POWERLAWANN_DLLEXPORT
#endif

#endif // COMMON_PLATFORM_COMPAT
