#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: tools/example.sh

Build and compare DiskANN and SkipCell on SIFT1M at L=10,20,30,40,50,75,100,125.
Environment overrides:
  SKIPCELL_OUTPUT_DIR       Output root (default: output)
  SKIPCELL_BUILD_THREADS   Index-build threads (default: detected CPU count)
  SKIPCELL_QUERY_THREADS   Query threads (default: 20)
  SKIPCELL_SKIP_BUILD      Set to 1 to reuse an existing Release build
EOF
  exit 0
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

output="${SKIPCELL_OUTPUT_DIR:-output}"
build_threads="${SKIPCELL_BUILD_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 20)}"
query_threads="${SKIPCELL_QUERY_THREADS:-20}"
data_dir="$output/data/sift1m"
index_dir="$output/index"
log_dir="$output/logs"
mkdir -p "$data_dir" "$index_dir" "$log_dir"

say() { printf '[SkipCell] %s\n' "$*"; }

base="$data_dir/base.fbin"
query="$data_dir/query.fbin"
truth="$data_dir/gt100.bin"

if [[ -f data/sift-128-euclidean.train.fbin &&
      -f data/sift-128-euclidean.query.fbin &&
      -f data/sift-128-euclidean.gt100.bin ]]; then
  base="data/sift-128-euclidean.train.fbin"
  query="data/sift-128-euclidean.query.fbin"
  truth="data/sift-128-euclidean.gt100.bin"
elif [[ ! -f "$base" || ! -f "$query" || ! -f "$truth" ]]; then
  archive="$data_dir/sift.tar.gz"
  if [[ ! -f "$archive" ]]; then
    say "download SIFT1M"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --retry 5 --output "$archive" \
        ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    elif command -v wget >/dev/null 2>&1; then
      wget --tries=5 --output-document="$archive" \
        ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    else
      echo "[SkipCell] curl or wget is required to download SIFT1M" >&2
      exit 1
    fi
  fi
  if [[ ! -f "$data_dir/sift/sift_base.fvecs" ]]; then
    say "extract SIFT1M"
    tar -xzf "$archive" -C "$data_dir"
  fi
  say "convert TexMex vectors to DiskANN binary files"
  python3 - "$data_dir/sift" "$base" "$query" "$truth" <<'PY'
import struct
import sys
from pathlib import Path

import numpy as np

source, base_out, query_out, truth_out = map(Path, sys.argv[1:])

def convert_records(input_path: Path, output_path: Path) -> None:
  raw = np.memmap(input_path, mode="r", dtype=np.uint8)
  dimension = int(np.frombuffer(raw[:4], dtype="<i4")[0])
  record_bytes = 4 + 4 * dimension
  if dimension <= 0 or raw.size % record_bytes:
    raise RuntimeError(f"invalid TexMex file: {input_path}")
  count = raw.size // record_bytes
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_path.with_suffix(output_path.suffix + ".tmp")
  with temporary.open("wb") as output:
    output.write(struct.pack("<II", count, dimension))
    for begin in range(0, count, 8192):
      end = min(count, begin + 8192)
      block = raw[begin * record_bytes : end * record_bytes].reshape(-1, record_bytes)
      output.write(block[:, 4:].copy(order="C").tobytes())
  temporary.replace(output_path)

convert_records(source / "sift_base.fvecs", base_out)
convert_records(source / "sift_query.fvecs", query_out)
convert_records(source / "sift_groundtruth.ivecs", truth_out)
PY
fi

if [[ "${SKIPCELL_SKIP_BUILD:-0}" != "1" ]]; then
  bash tools/build.sh --type release --no-tests
fi

prefix="$index_dir/sift1m"
bootstrap_sidecar="${prefix}.community_polar.bin"
assignment="$index_dir/sift1m.cells.csv"
hierarchy="$index_dir/sift1m.cells.hierarchy.bin"
sidecar="$index_dir/sift1m.skipcell.bin"
topology="$index_dir/sift1m.cell4k.topology.bin"
vectors="$index_dir/sift1m.cell4k.vectors.bin"

if [[ ! -f "${prefix}_disk.index" ]]; then
  say "build immutable DiskANN Base and bootstrap Community metadata"
  build/bin/build_diskann_index \
    --data_path "$base" --index_path_prefix "$prefix" \
    --search_DRAM_budget 1 --build_DRAM_budget 8 \
    --max_degree 64 --Lbuild 100 --num_threads "$build_threads" --QD 128 \
    --enable_powerann true --community_count 256 --community_imbalance 0.05 \
    --partition_quality fast --partition_threads "$build_threads" \
    --community_projection local_vamana_k8 --partition_seed 0 \
    --gateway_count 8 --gateway_shortlist 64 --direction_count 16 \
    --cell_target_size 64 --cell_partition graph_local --block_target_size 32
fi

if [[ ! -f "$assignment" || ! -f "$hierarchy" ]]; then
  say "train deterministic global Cell-32 hierarchy"
  python3 python/train_geometric_cells.py \
    --sidecar "$bootstrap_sidecar" --base_file "$base" \
    --output "$assignment" --hierarchy_output "$hierarchy" \
    --target_size 32 --projection_dimension 128 --identity_projection \
    --iterations 8 --seed 0 --global_cells
fi

if [[ ! -f "$sidecar" ]]; then
  say "materialize checksummed SkipCell sidecar"
  build/bin/build_diskann_index \
    --data_path "$base" --index_path_prefix "$prefix" \
    --enable_powerann true --community_sidecar_only true \
    --rcni_path "${prefix}.rcni.csv" --skipcell_sidecar_path "$sidecar" \
    --community_count 256 --community_imbalance 0.05 \
    --partition_quality fast --partition_threads "$build_threads" \
    --community_projection local_vamana_k8 --partition_seed 0 \
    --gateway_count 8 --gateway_shortlist 64 --direction_count 16 \
    --cell_target_size 32 --cell_partition global_geometric \
    --cell_profile_path "$assignment" --cell_hierarchy_path "$hierarchy" \
    --block_target_size 16
fi

if [[ ! -f "$topology" || ! -f "$vectors" ]]; then
  say "build Cell-u8-4K vector layout; Base adjacency remains immutable"
  build/bin/build_io_optimized_index \
    --index_path_prefix "$prefix" --topology_path "$topology" --vector_path "$vectors" \
    --vector_layout cell_u8_4k --adjacency raw --skipcell_sidecar_path "$sidecar"
fi

common=(
  --index_path_prefix "$prefix" --query_file "$query" --gt_file "$truth"
  --recall_at 10 --search_list_series 10,20,30,40,50,75,100,125
  --beamwidth 8 --num_threads "$query_threads" --num_nodes_to_cache 0
  --warmup_queries 1000 --warmup_query_offset 9000 --output_style compact
)

say "run DiskANN curve"
build/bin/search_diskann_index "${common[@]}" \
  --enable_powerann false --powerann_mode baseline \
  | tee "$log_dir/diskann.log"

say "run SkipCell curve"
build/bin/search_diskann_index "${common[@]}" \
  --enable_powerann true --powerann_mode hybrid \
  --community_skip true --cell_expansion true --skipcell_sidecar_path "$sidecar" \
  --community_expansion_budget 1 --gateway_score_budget 8 \
  --cell_block_budget 1 --cell_support 1 --cell_insert_cap 64 \
  --cell_activation_marker 0 --cell_min_search_list_size 1 \
  --cell_min_competitive_count 1 --cell_min_competitive_fraction 0 \
  --cell_evidence_threshold_ratio 1 --cell_candidate_threshold_ratio 1 \
  --cell_insert_fraction 1 --cell_require_full_frontier false \
  --cell_use_provisional_frontier true --cell_whole_cell_scan true \
  --cell_centroid_routing true --cell_routing_community_budget 256 \
  --cell_routing_cell_budget 1 --cell_terminal_convergence true \
  --cell_terminal_approach_hops 100 --cell_terminal_refine_hops 0 \
  --cell_terminal_distance_scale 1 --cell_terminal_frontier_cells 0 \
  --cell_hierarchical_routing true --cell_hierarchy_branching 4 \
  --cell_hierarchy_leaf_size 16 --cell_hierarchy_probe_budget 1 \
  --cell_hierarchy_exact_routing false --cell_hierarchy_preserve_leaf_order true \
  --cell_hierarchy_require_persisted true --graph_guard_fraction 0 \
  --io_topology_vector true --io_topology_path "$topology" --io_vector_path "$vectors" \
  --io_cell_layout true --io_cell_u8_layout true --io_cell_page_batch true \
  --io_cell_batch_search true --io_cell_batch_max_cached_expansions 1 \
  --io_lru_cache true --io_lru_cache_pages 65536 --io_cell_pq_traversal true \
  --io_cell_pq_refine_candidates 14 --io_l_aware_refine_budget true \
  --io_l_aware_refine_base 9 --io_l_aware_refine_divisor 25 \
  --io_forced_cell_tree_search true \
  | tee "$log_dir/skipcell.log"

say "write CSV, validate matched-recall floor, and draw output/example.pdf"
python3 - "$log_dir/diskann.log" "$log_dir/skipcell.log" \
  "$output/example.csv" "$output/example.pdf" <<'PY'
import csv
import math
import sys
from pathlib import Path

baseline_log, candidate_log, csv_path, pdf_path = map(Path, sys.argv[1:])

def parse(path):
  rows = []
  for line in path.read_text().splitlines():
    fields = line.split()
    if len(fields) == 9 and fields[0] == "[Search]" and fields[1] in {"DiskANN", "SkipCell"}:
      rows.append({"system": fields[1], "L": int(fields[2]), "beam": int(fields[3]),
                   "threads": int(fields[4]), "recall": float(fields[5]),
                   "qps": float(fields[6]), "p99_us": float(fields[7]),
                   "four_kib_reads": float(fields[8])})
  if len(rows) != 8:
    raise RuntimeError(f"expected 8 curve points in {path}, found {len(rows)}")
  return rows

baseline = parse(baseline_log)
candidate = parse(candidate_log)
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="") as output:
  writer = csv.DictWriter(output, fieldnames=baseline[0].keys())
  writer.writeheader()
  writer.writerows(baseline + candidate)

candidate_by_recall = sorted((row["recall"], row["qps"]) for row in candidate)
def interpolate(recall):
  for (r0, q0), (r1, q1) in zip(candidate_by_recall, candidate_by_recall[1:]):
    if r0 <= recall <= r1:
      if r0 == r1:
        return max(q0, q1)
      ratio = (recall - r0) / (r1 - r0)
      return q0 + ratio * (q1 - q0)
  return None

ratios = []
for row in baseline:
  qps = interpolate(row["recall"])
  if qps is not None:
    ratios.append(qps / row["qps"])
if not ratios or min(ratios) < 5.0:
  raise RuntimeError(f"matched-recall 5x gate failed: ratios={ratios}")

width, height = 612, 432
left, bottom, right, top = 72, 58, 584, 392
all_rows = baseline + candidate
x_min = math.floor(min(row["recall"] for row in all_rows))
x_max = math.ceil(max(row["recall"] for row in all_rows))
if x_min == x_max:
  x_max += 1
y_max = max(row["qps"] for row in all_rows) * 1.10
def x(value): return left + (value - x_min) * (right - left) / (x_max - x_min)
def y(value): return bottom + value * (top - bottom) / y_max
def esc(value): return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

commands = ["1 1 1 rg 0 0 612 432 re f", "0 0 0 RG 1 w",
            f"{left} {bottom} m {right} {bottom} l {right} {top} l S"]
for tick in range(6):
  value = x_min + tick * (x_max - x_min) / 5
  position = x(value)
  commands += [f"{position:.2f} {bottom} m {position:.2f} {bottom-4} l S",
               f"BT /F1 9 Tf {position-10:.2f} {bottom-18} Td ({value:.1f}) Tj ET"]
for tick in range(6):
  value = tick * y_max / 5
  position = y(value)
  commands += [f"{left-4} {position:.2f} m {left} {position:.2f} l S",
               f"BT /F1 9 Tf {left-48} {position-3:.2f} Td ({value:.0f}) Tj ET"]
commands += [f"BT /F1 11 Tf 290 25 Td (Recall@10 \050%\051) Tj ET",
             "BT /F1 11 Tf 16 225 Td (QPS) Tj ET",
             "BT /F1 14 Tf 205 407 Td (SIFT1M Recall-QPS) Tj ET"]
for rows, color, label in ((baseline, "0.25 0.25 0.25", "DiskANN"),
                           (candidate, "0.10 0.35 0.85", "SkipCell")):
  ordered = sorted(rows, key=lambda row: row["recall"])
  commands.append(f"{color} RG 2 w")
  commands.append(f"{x(ordered[0]['recall']):.2f} {y(ordered[0]['qps']):.2f} m")
  commands.extend(f"{x(row['recall']):.2f} {y(row['qps']):.2f} l" for row in ordered[1:])
  commands.append("S")
  for row in ordered:
    commands.append(f"{x(row['recall'])-2:.2f} {y(row['qps'])-2:.2f} 4 4 re f")
  legend_y = 372 if label == "DiskANN" else 354
  commands += [f"{color} RG 2 w 455 {legend_y} m 480 {legend_y} l S",
               f"0 0 0 rg BT /F1 10 Tf 488 {legend_y-3} Td ({esc(label)}) Tj ET"]
content = ("\n".join(commands) + "\n").encode()
objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
           b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
           b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 432] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
           b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
           f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream"]
document = bytearray(b"%PDF-1.4\n")
offsets = [0]
for number, obj in enumerate(objects, 1):
  offsets.append(len(document))
  document.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
xref = len(document)
document.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
for offset in offsets[1:]:
  document.extend(f"{offset:010d} 00000 n \n".encode())
document.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
pdf_path.write_bytes(document)
print(f"[Result] minimum matched-recall QPS ratio: {min(ratios):.3f}x")
print(f"[Result] CSV: {csv_path}")
print(f"[Result] Figure: {pdf_path}")
PY

say "complete"
