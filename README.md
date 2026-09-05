# SkipCell

SkipCell is an SSD-resident approximate nearest-neighbor search prototype. It combines
Community Skipping with Waterline Search and a Cell-contiguous 4-KiB vector layout.

Paper Title: Accelerating Disk-Resident Graph ANNS via Community Skipping and Waterline Search over Crosslinked Cell Hierarchies

## Reproduce SIFT1M

The reviewer example downloads SIFT1M when it is not already present, builds one immutable
DiskANN Base index, derives the checksummed SkipCell artifacts, runs both systems at
`L=10,20,30,40,50,75,100,125`, and writes the Recall-QPS curve to `output/example.pdf`.

```bash
bash tools/example.sh
```

The run also writes `output/example.csv` and one log per system under `output/logs/`. Artifacts are
checkpointed under `output/index/`, so an interrupted run can be restarted with the same command.

Build only:

```bash
bash tools/build.sh --type release
```

The supported platform is Ubuntu Linux with a C++20 compiler, CMake, OpenMP, Boost, oneMKL ILP64,
libaio, tcmalloc, NumPy, and KaMinPar installed under `$HOME/.local/lib/KaMinPar` (or selected with
`KAMINPAR_LOCAL_PATH`). Use `SKIPCELL_BUILD_THREADS` and `SKIPCELL_QUERY_THREADS` to override the
detected build parallelism and the default 20 query threads.

## Repository layout

```text
bin/       Thin build and search command-line programs
include/   Public C++ interfaces
src/       SkipCell and imported DiskANN-compatible implementation
python/    Deterministic Cell-tree trainer
test/      C++ and Python regression tests
tools/     Reviewer build and end-to-end example scripts
```
