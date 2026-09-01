cmake_minimum_required(VERSION 3.14)
project(powerlawann)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
  message(STATUS "Build: type not specified, defaulting to Debug")
endif()

# Build configuration
message(STATUS "Build: type = ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Build: Debug mode enabled")
  add_definitions(-DPOWERLAW_ANN_DEBUG_MODE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Build: Release mode enabled")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(UNIX)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fcoroutines")
  endif()
endif()

# Platform configuration
if(APPLE)
  set(HOMEBREW_PREFIX /opt/homebrew)
  list(APPEND CMAKE_PREFIX_PATH ${HOMEBREW_PREFIX})
elseif(UNIX AND NOT APPLE)
  list(APPEND CMAKE_PREFIX_PATH /usr/local)
  list(APPEND CMAKE_PREFIX_PATH /usr)
endif()

# Core system package
find_package(Threads REQUIRED)

if(APPLE)
  set(OpenMP_ROOT "${HOMEBREW_PREFIX}/opt/libomp")
  list(APPEND CMAKE_PREFIX_PATH "${OpenMP_ROOT}")
endif()

find_package(OpenMP REQUIRED)
message(STATUS "Dependency OpenMP: required and found")

find_package(Boost REQUIRED)
message(STATUS "Dependency Boost headers: required and found")

# KaMinPar shared-memory graph partitioner. Prefer the locally installed package used by the
# research prototype while allowing an explicit CMake override on both supported platforms.
if(NOT DEFINED KAMINPAR_LOCAL_PATH OR KAMINPAR_LOCAL_PATH STREQUAL "")
  if(APPLE)
    set(KAMINPAR_LOCAL_PATH "$ENV{HOME}/Library/CppLib/KaMinPar")
  else()
    set(KAMINPAR_LOCAL_PATH "$ENV{HOME}/.local/lib/KaMinPar")
  endif()
endif()
list(PREPEND CMAKE_PREFIX_PATH "${KAMINPAR_LOCAL_PATH}")
find_package(KaMinPar QUIET)
if(KaMinPar_FOUND AND TARGET KaMinPar::KaMinPar)
  set(POWERLAWANN_KAMINPAR_TARGET KaMinPar::KaMinPar)
  message(STATUS "Dependency KaMinPar: found via package at ${KAMINPAR_LOCAL_PATH}")
else()
  find_path(POWERLAWANN_KAMINPAR_INCLUDE_DIR
    NAMES kaminpar.h
    PATHS
      "${KAMINPAR_LOCAL_PATH}"
      "${KAMINPAR_LOCAL_PATH}/install"
      /usr/local
      /usr
    PATH_SUFFIXES include
    REQUIRED
  )
  find_library(POWERLAWANN_KAMINPAR_LIBRARY
    NAMES KaMinPar kaminpar
    PATHS
      "${KAMINPAR_LOCAL_PATH}"
      "${KAMINPAR_LOCAL_PATH}/install"
      /usr/local
      /usr
    PATH_SUFFIXES lib build/kaminpar-shm
    REQUIRED
  )
  add_library(KaMinPar::KaMinPar UNKNOWN IMPORTED)
  set_target_properties(KaMinPar::KaMinPar PROPERTIES
    IMPORTED_LOCATION "${POWERLAWANN_KAMINPAR_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${POWERLAWANN_KAMINPAR_INCLUDE_DIR}"
  )
  set(POWERLAWANN_KAMINPAR_TARGET KaMinPar::KaMinPar)
  message(STATUS "Dependency KaMinPar: found manually (${POWERLAWANN_KAMINPAR_LIBRARY})")
endif()

# PQ/index-build math backend. macOS is a correctness-validation target and
# uses the system Accelerate framework. Ubuntu keeps DiskANN's oneMKL ABI and
# execution path for performance experiments.
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  message(STATUS "Dependency math backend: Apple Accelerate")
elseif(UNIX)
  find_path(POWERLAWANN_MKL_INCLUDE_DIR
    NAMES mkl.h
    HINTS ${MKL_INCLUDE_PATH} $ENV{MKLROOT}
    PATH_SUFFIXES include include/mkl
    PATHS
      /opt/intel/oneapi/mkl/latest
      /usr
      /opt/intel/mkl
    REQUIRED
  )
  find_library(POWERLAWANN_MKL_INTERFACE
    NAMES mkl_intel_ilp64
    HINTS ${MKL_PATH} $ENV{MKLROOT}
    PATH_SUFFIXES lib lib/intel64 lib/x86_64-linux-gnu
    PATHS
      /opt/intel/oneapi/mkl/latest
      /usr
      /opt/intel/mkl
    REQUIRED
  )
  find_library(POWERLAWANN_MKL_THREADING
    NAMES mkl_intel_thread
    HINTS ${MKL_PATH} $ENV{MKLROOT}
    PATH_SUFFIXES lib lib/intel64 lib/x86_64-linux-gnu
    PATHS
      /opt/intel/oneapi/mkl/latest
      /usr
      /opt/intel/mkl
    REQUIRED
  )
  find_library(POWERLAWANN_MKL_CORE
    NAMES mkl_core
    HINTS ${MKL_PATH} $ENV{MKLROOT}
    PATH_SUFFIXES lib lib/intel64 lib/x86_64-linux-gnu
    PATHS
      /opt/intel/oneapi/mkl/latest
      /usr
      /opt/intel/mkl
    REQUIRED
  )
  find_library(POWERLAWANN_IOMP_RUNTIME
    NAMES iomp5
    HINTS ${OMP_PATH} $ENV{ONEAPI_ROOT}
    PATH_SUFFIXES
      compiler/latest/lib
      compiler/latest/linux/compiler/lib/intel64_lin
      lib/intel64_lin
      lib/x86_64-linux-gnu
    PATHS
      /opt/intel/oneapi
      /usr
      /opt/intel
    REQUIRED
  )
  message(STATUS "Dependency math backend: oneMKL ILP64 (${POWERLAWANN_MKL_INTERFACE})")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_path(POWERLAWANN_LIBAIO_INCLUDE_DIR
    NAMES libaio.h
    PATHS /usr/include /usr/local/include
    REQUIRED
  )
  find_library(POWERLAWANN_LIBAIO_LIBRARY
    NAMES aio
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    REQUIRED
  )
  find_path(POWERLAWANN_LIBURING_INCLUDE_DIR
    NAMES liburing.h
    PATHS /usr/include /usr/local/include
    REQUIRED
  )
  find_library(POWERLAWANN_LIBURING_LIBRARY
    NAMES uring
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    REQUIRED
  )
  find_path(POWERLAWANN_SIMDCOMP_INCLUDE_DIR
    NAMES codecfactory.h
    PATHS /usr/local/include/SIMDCompressionAndIntersection /usr/include/SIMDCompressionAndIntersection
    REQUIRED
  )
  find_library(POWERLAWANN_SIMDCOMP_LIBRARY
    NAMES SIMDCompressionAndIntersection
    PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
    REQUIRED
  )
  find_library(POWERLAWANN_TCMALLOC_LIBRARY
    NAMES tcmalloc
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    REQUIRED
  )
  message(STATUS "Dependency I/O backend: libaio + O_DIRECT (${POWERLAWANN_LIBAIO_LIBRARY})")
  message(STATUS "Dependency experimental I/O backend: io_uring (${POWERLAWANN_LIBURING_LIBRARY})")
  message(STATUS "Dependency adjacency codec: SIMDCompressionAndIntersection (${POWERLAWANN_SIMDCOMP_LIBRARY})")
  message(STATUS "Dependency allocator: tcmalloc (${POWERLAWANN_TCMALLOC_LIBRARY})")
endif()

# Third-party dependencies

# GTest
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  if(APPLE)
    set(GTEST_INCLUDE_DIR ${HOMEBREW_PREFIX}/include)
    set(GTEST_LIB_DIR ${HOMEBREW_PREFIX}/lib)
  else()
    set(GTEST_INCLUDE_DIR /usr/local/include)
    set(GTEST_LIB_DIR /usr/local/lib)
  endif()

  find_library(GTEST_LIBRARY
               NAMES gtest
               PATHS ${GTEST_LIB_DIR} /usr/lib /usr/lib/x86_64-linux-gnu
               REQUIRED)

  find_library(GTEST_MAIN_LIBRARY
               NAMES gtest_main
               PATHS ${GTEST_LIB_DIR} /usr/lib /usr/lib/x86_64-linux-gnu
               REQUIRED)

  if(GTEST_LIBRARY AND GTEST_MAIN_LIBRARY)
    add_library(GTest::gtest UNKNOWN IMPORTED)
    set_target_properties(GTest::gtest PROPERTIES
      IMPORTED_LOCATION ${GTEST_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
      INTERFACE_LINK_LIBRARIES "Threads::Threads"
    )

    add_library(GTest::gtest_main UNKNOWN IMPORTED)
    set_target_properties(GTest::gtest_main PROPERTIES
      IMPORTED_LOCATION ${GTEST_MAIN_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIR}
      INTERFACE_LINK_LIBRARIES "GTest::gtest;Threads::Threads"
    )

    message(STATUS
            "Dependency GTest: found manually (gtest=${GTEST_LIBRARY}, gtest_main=${GTEST_MAIN_LIBRARY}, include=${GTEST_INCLUDE_DIR})")
  else()
    message(FATAL_ERROR
            "Dependency GTest: not found. Make sure googletest is installed.")
  endif()
else()
  message(STATUS "Dependency GTest: found via find_package")
endif()

# Targets
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cc")
list(FILTER SOURCES EXCLUDE REGEX "/src/storage/(linux|macos)_aligned_file_reader\\.cc$")
if(APPLE)
  list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/storage/macos_aligned_file_reader.cc")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/storage/linux_aligned_file_reader.cc")
else()
  message(FATAL_ERROR "PowerLawANN supports only macOS and Linux")
endif()
add_library(powerlawann ${SOURCES})

if(APPLE)
  # AppleClang 21 searches /usr/local/include before imported-system include directories. Preserve
  # the selected Homebrew dependency order when a legacy Boost is installed on that implicit path.
  set_target_properties(powerlawann PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
endif()

target_include_directories(powerlawann PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(powerlawann SYSTEM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third/louvain-communities-openmp/inc>
)

target_include_directories(powerlawann BEFORE PRIVATE
  ${Boost_INCLUDE_DIRS}
)

if(UNIX AND NOT APPLE)
  target_include_directories(powerlawann BEFORE PRIVATE
    ${POWERLAWANN_MKL_INCLUDE_DIR}
    ${POWERLAWANN_SIMDCOMP_INCLUDE_DIR}
  )
  target_include_directories(powerlawann PUBLIC
    ${POWERLAWANN_LIBAIO_INCLUDE_DIR}
  )
  target_compile_definitions(powerlawann PUBLIC
    MKL_ILP64
    POWERLAWANN_USE_LIBAIO
    POWERLAWANN_USE_PFORDELTA
  )
endif()

target_compile_definitions(powerlawann PUBLIC
  OPENMP
)

target_link_libraries(powerlawann PUBLIC
  Threads::Threads
  OpenMP::OpenMP_CXX
  ${POWERLAWANN_KAMINPAR_TARGET}
)

if(APPLE)
  target_compile_definitions(powerlawann PUBLIC
    ACCELERATE_NEW_LAPACK
    ACCELERATE_LAPACK_ILP64
  )
  target_link_libraries(powerlawann PUBLIC ${ACCELERATE_FRAMEWORK})
elseif(UNIX)
  target_link_options(powerlawann PUBLIC -Wl,--no-as-needed)
  target_link_libraries(powerlawann PUBLIC
    ${POWERLAWANN_LIBAIO_LIBRARY}
    ${POWERLAWANN_LIBURING_LIBRARY}
    ${POWERLAWANN_SIMDCOMP_LIBRARY}
    ${POWERLAWANN_TCMALLOC_LIBRARY}
    ${POWERLAWANN_MKL_INTERFACE}
    ${POWERLAWANN_MKL_THREADING}
    ${POWERLAWANN_MKL_CORE}
    ${POWERLAWANN_IOMP_RUNTIME}
    ${CMAKE_DL_LIBS}
    m
  )
endif()

target_compile_options(powerlawann PUBLIC
  $<$<CONFIG:Debug>:-g -fno-eliminate-unused-debug-types -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-unused-function>
  $<$<CONFIG:Release>:-O3 -DNDEBUG>
  $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
  $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Ubuntu Release builds use -Ofast below. Keep its finite-value assumption out of code whose
  # validation or query state intentionally uses NaN/Inf, including the missing-evidence sentinel.
  set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/index/polar.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/index/rcni.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/index/community_polar_index.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/search/community_polar_search.cc"
    PROPERTIES
    COMPILE_OPTIONS "$<$<CONFIG:Release>:-fno-finite-math-only>"
  )
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" POWERLAWANN_SYSTEM_PROCESSOR)
if(UNIX AND NOT APPLE AND POWERLAWANN_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  target_compile_definitions(powerlawann PRIVATE
    $<$<CONFIG:Release>:USE_AVX2>
  )
  target_compile_options(powerlawann PRIVATE
    $<$<CONFIG:Release>:-m64>
    $<$<CONFIG:Release>:-mavx2>
    $<$<CONFIG:Release>:-mfma>
    $<$<CONFIG:Release>:-msse2>
    $<$<CONFIG:Release>:-ftree-vectorize>
    $<$<CONFIG:Release>:-fno-builtin-malloc>
    $<$<CONFIG:Release>:-fno-builtin-calloc>
    $<$<CONFIG:Release>:-fno-builtin-realloc>
    $<$<CONFIG:Release>:-fno-builtin-free>
    $<$<CONFIG:Release>:-fopenmp-simd>
    $<$<CONFIG:Release>:-funroll-loops>
    $<$<CONFIG:Release>:-Wfatal-errors>
    $<$<CONFIG:Release>:-Ofast>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-mtune=native>
  )
  message(STATUS "DiskANN baseline backend: Ubuntu x86 AVX2/FMA release path")
elseif(APPLE)
  message(STATUS "DiskANN baseline backend: macOS portable correctness path")
endif()

file(GLOB BIN_SOURCES "bin/*.cc")
foreach(BIN_FILE ${BIN_SOURCES})
  get_filename_component(BIN_NAME ${BIN_FILE} NAME_WE)

  add_executable(${BIN_NAME} ${BIN_FILE})
  target_link_libraries(${BIN_NAME} PRIVATE powerlawann)

  set_target_properties(${BIN_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

  message(STATUS "Target executable: ${BIN_NAME} (${BIN_FILE})")
endforeach()

# Tests
enable_testing()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_subdirectory(test)
else()
  message(WARNING "Tests: directory not found at ${CMAKE_CURRENT_SOURCE_DIR}/test")
endif()
