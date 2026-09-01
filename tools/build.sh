#!/usr/bin/env bash
set -euo pipefail

build_type=Debug
clean=0
run_tests=1
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

usage() {
  cat <<'EOF'
Usage: tools/build.sh [--type debug|release] [--clean] [--jobs N] [--no-tests]
EOF
}

while (($#)); do
  case "$1" in
    -t|--type)
      case "${2:-}" in
        debug|Debug) build_type=Debug ;;
        release|Release) build_type=Release ;;
        *) echo "[Build] invalid build type: ${2:-missing}" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    -c|--clean) clean=1; shift ;;
    -j|--jobs) jobs="${2:?missing job count}"; shift 2 ;;
    --no-tests) run_tests=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[Build] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ((clean)) && [[ -d build ]]; then
  echo "[Build] removing ./build"
  cmake -E remove_directory build
fi

echo "[Build] configure type=${build_type} jobs=${jobs}"
cmake -S . -B build -DCMAKE_BUILD_TYPE="$build_type" -DBUILD_TESTS=ON
echo "[Build] compile"
cmake --build build --parallel "$jobs"

if ((run_tests)); then
  echo "[Build] test"
  (cd build && ctest --output-on-failure)
fi

echo "[Build] ready: build/bin"
