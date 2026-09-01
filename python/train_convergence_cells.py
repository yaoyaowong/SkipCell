#!/usr/bin/env python3
"""Train deterministic Community-local Cells from Base convergence co-access."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import zlib
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


FIXED_HEADER_SIZE = 96
DIRECTORY_ENTRY_SIZE = 32
NODE_TO_COMMUNITY_SECTION = 2
NODE_TO_CELL_SECTION = 3
CELL_SECTION = 12
PACKET_SECTION = 14
DIRECTION_AXES_SECTION = 10
PQ_LOCALITY_MAGIC = b"PLCPQLOC"
PQ_LOCALITY_VERSION = 1
PQ_LOCALITY_HEADER = struct.Struct("<8sIIIIIQQQ")
BINARY_ENDIAN_MARKER = 0x01020304


class ProfileError(RuntimeError):
  pass


def read_memberships(path: Path) -> tuple[list[int], list[int]]:
  with path.open("rb") as source:
    fixed = source.read(FIXED_HEADER_SIZE)
    if len(fixed) != FIXED_HEADER_SIZE:
      raise ProfileError("Community-Polar sidecar header is truncated")
    _, version, header_size, _, section_count, point_count, *_ = struct.unpack_from(
        "<QIIIIQIIQQQQQQ", fixed
    )
    if version < 3 or header_size != FIXED_HEADER_SIZE + section_count * DIRECTORY_ENTRY_SIZE:
      raise ProfileError("Community-Polar sidecar does not expose graph-local Cell metadata")
    directory = source.read(section_count * DIRECTORY_ENTRY_SIZE)
    sections: dict[int, tuple[int, int, int]] = {}
    for index in range(section_count):
      section_id, _, offset, length, count = struct.unpack_from(
          "<IIQQQ", directory, index * DIRECTORY_ENTRY_SIZE
      )
      sections[section_id] = (offset, length, count)

    def read_u32_section(section_id: int) -> list[int]:
      offset, length, count = sections[section_id]
      if count != point_count or length != count * 4:
        raise ProfileError(f"sidecar section {section_id} has an invalid shape")
      source.seek(offset)
      payload = source.read(length)
      if len(payload) != length:
        raise ProfileError(f"sidecar section {section_id} is truncated")
      return list(struct.unpack(f"<{count}I", payload))

    return read_u32_section(NODE_TO_COMMUNITY_SECTION), read_u32_section(NODE_TO_CELL_SECTION)


def map_u32_membership(path: Path, section_id: int) -> np.ndarray:
  """Memory-map one canonical point-sized uint32 sidecar section."""
  with path.open("rb") as source:
    fixed = source.read(FIXED_HEADER_SIZE)
    if len(fixed) != FIXED_HEADER_SIZE:
      raise ProfileError("Community-Polar sidecar header is truncated")
    _, version, header_size, _, section_count, point_count, *_ = struct.unpack_from(
        "<QIIIIQIIQQQQQQ", fixed
    )
    if version < 3 or header_size != FIXED_HEADER_SIZE + section_count * DIRECTORY_ENTRY_SIZE:
      raise ProfileError("Community-Polar sidecar membership directory is incompatible")
    directory = source.read(section_count * DIRECTORY_ENTRY_SIZE)
  for index in range(section_count):
    current, _, offset, length, count = struct.unpack_from(
        "<IIQQQ", directory, index * DIRECTORY_ENTRY_SIZE
    )
    if current == section_id:
      if count != point_count or length != count * 4:
        raise ProfileError(f"sidecar section {section_id} has an invalid shape")
      return np.memmap(path, dtype="<u4", mode="r", offset=offset, shape=(count,))
  raise ProfileError(f"sidecar section {section_id} is missing")


def collect_coaccess(
    rows: Iterable[dict[str, str]], query_limit: int, minimum_hop: int, window: int,
    scope: str = "community",
) -> tuple[dict[int, Counter[int]], Counter[int]]:
  if scope not in ("community", "global"):
    raise ProfileError("co-access scope must be 'community' or 'global'")
  adjacency: dict[int, Counter[int]] = defaultdict(Counter)
  frequency: Counter[int] = Counter()
  current_query = -1
  query_rows: list[tuple[int, int]] = []

  def flush() -> None:
    by_community: dict[int, list[int]] = defaultdict(list)
    for community, node in query_rows:
      by_community[community if scope == "community" else 0].append(node)
      frequency[node] += 1
    for nodes in by_community.values():
      for left in range(len(nodes)):
        for right in range(left + 1, min(len(nodes), left + window + 1)):
          source, target = nodes[left], nodes[right]
          if source == target:
            continue
          adjacency[source][target] += 1
          adjacency[target][source] += 1

  for row in rows:
    query = int(row["query_id"])
    if query >= query_limit:
      continue
    if query != current_query:
      if current_query >= 0:
        flush()
      current_query = query
      query_rows = []
    if int(row["hop"]) >= minimum_hop:
      query_rows.append((int(row["community_id"]), int(row["node_id"])))
  if current_query >= 0:
    flush()
  return adjacency, frequency


def build_assignments(
    node_to_community: Sequence[int],
    node_to_source_cell: Sequence[int],
    adjacency: dict[int, Counter[int]],
    frequency: Counter[int],
    target_size: int,
) -> tuple[list[int], list[int]]:
  if len(node_to_community) != len(node_to_source_cell) or target_size <= 0:
    raise ProfileError("membership arrays and target size are incompatible")
  community_nodes: dict[int, list[int]] = defaultdict(list)
  source_cell_nodes: dict[tuple[int, int], list[int]] = defaultdict(list)
  for node, community in enumerate(node_to_community):
    community_nodes[community].append(node)
    source_cell_nodes[(community, node_to_source_cell[node])].append(node)

  assigned = bytearray(len(node_to_community))
  cell_ids = [-1] * len(node_to_community)
  orders = [-1] * len(node_to_community)
  next_cell = 0

  def publish(nodes: list[int]) -> None:
    nonlocal next_cell
    ordered = sorted(nodes, key=lambda node: (-frequency[node], node))
    for order, node in enumerate(ordered):
      assigned[node] = 1
      cell_ids[node] = next_cell
      orders[node] = order
    next_cell += 1

  for community in sorted(community_nodes):
    seeds = sorted(
        (node for node in community_nodes[community] if node in adjacency),
        key=lambda node: (-sum(adjacency[node].values()), -frequency[node], node),
    )
    for seed in seeds:
      if assigned[seed]:
        continue
      cluster: list[int] = []
      affinity: Counter[int] = Counter()

      def add(node: int) -> None:
        assigned[node] = 1
        cluster.append(node)
        for neighbor, weight in adjacency.get(node, {}).items():
          if not assigned[neighbor] and node_to_community[neighbor] == community:
            affinity[neighbor] += weight

      add(seed)
      while len(cluster) < target_size and affinity:
        candidate = min(
            affinity,
            key=lambda node: (-affinity[node], -frequency[node], node),
        )
        del affinity[candidate]
        if not assigned[candidate]:
          add(candidate)

      source_cells = sorted({node_to_source_cell[node] for node in cluster})
      for source_cell in source_cells:
        for candidate in source_cell_nodes[(community, source_cell)]:
          if len(cluster) == target_size:
            break
          if not assigned[candidate]:
            assigned[candidate] = 1
            cluster.append(candidate)
        if len(cluster) == target_size:
          break
      publish(cluster)

    remaining_by_source: dict[int, list[int]] = defaultdict(list)
    for node in community_nodes[community]:
      if not assigned[node]:
        remaining_by_source[node_to_source_cell[node]].append(node)
    pending: list[int] = []
    for source_cell in sorted(remaining_by_source):
      pending.extend(remaining_by_source[source_cell])
      while len(pending) >= target_size:
        publish(pending[:target_size])
        pending = pending[target_size:]
    if pending:
      publish(pending)

  if any(cell < 0 for cell in cell_ids) or any(order < 0 for order in orders):
    raise ProfileError("trained Cells do not cover every node")
  return cell_ids, orders


def build_observed_groups(
    adjacency: dict[int, Counter[int]], frequency: Counter[int], target_size: int
) -> list[np.ndarray]:
  """Group only trace-observed nodes without allocating point-count-sized Python lists."""
  if target_size <= 0:
    raise ProfileError("co-access Cell target size must be positive")
  assigned: set[int] = set()
  groups: list[np.ndarray] = []
  seeds = sorted(
      adjacency,
      key=lambda node: (-sum(adjacency[node].values()), -frequency[node], node),
  )
  for seed in seeds:
    if seed in assigned:
      continue
    cluster: list[int] = []
    affinity: Counter[int] = Counter()

    def add(node: int) -> None:
      assigned.add(node)
      cluster.append(node)
      for neighbor, weight in adjacency.get(node, {}).items():
        if neighbor not in assigned:
          affinity[neighbor] += weight

    add(seed)
    while len(cluster) < target_size and affinity:
      candidate = min(
          affinity,
          key=lambda node: (-affinity[node], -frequency[node], node),
      )
      del affinity[candidate]
      if candidate not in assigned:
        add(candidate)
    groups.append(
        np.asarray(sorted(cluster, key=lambda node: (-frequency[node], node)), dtype="<u4")
    )

  remaining = sorted(
      (node for node in frequency if node not in assigned),
      key=lambda node: (-frequency[node], node),
  )
  for begin in range(0, len(remaining), target_size):
    groups.append(np.asarray(remaining[begin:begin + target_size], dtype="<u4"))
  if sum(map(len, groups)) != len(frequency) or any(len(group) == 0 for group in groups):
    raise ProfileError("observed co-access grouping is incomplete")
  return groups


def read_sidecar_layout(path: Path) -> dict[str, object]:
  """Memory-map the minimal source-Cell fields needed for scalable trace regrouping."""
  with path.open("rb") as source:
    fixed = source.read(FIXED_HEADER_SIZE)
    if len(fixed) != FIXED_HEADER_SIZE:
      raise ProfileError("Community-Polar sidecar header is truncated")
    _, version, header_size, endian, section_count, point_count, dimension, pq_width, *_ = (
        struct.unpack_from("<QIIIIQIIQQQQQQ", fixed)
    )
    if (
        version < 5
        or endian != BINARY_ENDIAN_MARKER
        or header_size != FIXED_HEADER_SIZE + section_count * DIRECTORY_ENTRY_SIZE
    ):
      raise ProfileError("Community-Polar sidecar layout is incompatible")
    directory = source.read(section_count * DIRECTORY_ENTRY_SIZE)
  sections: dict[int, tuple[int, int, int]] = {}
  for index in range(section_count):
    section_id, reserved, offset, length, count = struct.unpack_from(
        "<IIQQQ", directory, index * DIRECTORY_ENTRY_SIZE
    )
    if reserved != 0 or section_id in sections:
      raise ProfileError("Community-Polar sidecar directory is invalid")
    sections[section_id] = (offset, length, count)
  for section_id in (NODE_TO_CELL_SECTION, CELL_SECTION, PACKET_SECTION,
                     DIRECTION_AXES_SECTION):
    if section_id not in sections:
      raise ProfileError(f"Community-Polar sidecar section {section_id} is missing")

  node_to_cell_offset, node_to_cell_bytes, node_count = sections[NODE_TO_CELL_SECTION]
  if node_count != point_count or node_to_cell_bytes != point_count * 4:
    raise ProfileError("Community-Polar node-to-Cell directory has an invalid shape")
  cell_offset, cell_bytes, cell_count = sections[CELL_SECTION]
  cell_record_bytes = 52 if version >= 6 else 48
  if cell_count == 0 or cell_bytes != cell_count * cell_record_bytes:
    raise ProfileError("Community-Polar Cell records have an invalid shape")
  packet_offset, packet_bytes, packet_count = sections[PACKET_SECTION]
  packet_record_bytes = 4 + pq_width
  if packet_count != point_count or packet_bytes != point_count * packet_record_bytes:
    raise ProfileError("Community-Polar packet records have an invalid shape")
  axes_offset, axes_bytes, axes_count = sections[DIRECTION_AXES_SECTION]
  if axes_count != cell_count * dimension or axes_bytes != axes_count * 4:
    raise ProfileError("Community-Polar Cell centroids have an invalid shape")

  cell_raw = np.memmap(path, dtype=np.uint8, mode="r", offset=cell_offset, shape=(cell_bytes,))
  populations = np.ndarray(
      shape=(cell_count,), dtype="<u8", buffer=cell_raw, offset=16,
      strides=(cell_record_bytes,),
  )
  packet_raw = np.memmap(
      path, dtype=np.uint8, mode="r", offset=packet_offset, shape=(packet_bytes,)
  )
  packet_nodes = np.ndarray(
      shape=(point_count,), dtype="<u4", buffer=packet_raw,
      strides=(packet_record_bytes,),
  )
  return {
      "point_count": point_count,
      "dimension": dimension,
      "cell_count": cell_count,
      "node_to_cell": np.memmap(
          path, dtype="<u4", mode="r", offset=node_to_cell_offset, shape=(point_count,)
      ),
      "populations": populations,
      "packet_nodes": packet_nodes,
      "centroids": np.memmap(
          path, dtype="<f4", mode="r", offset=axes_offset,
          shape=(cell_count, dimension),
      ),
      "backing": (cell_raw, packet_raw),
  }


def read_fbin_memmap(path: Path, point_count: int, dimension: int) -> np.ndarray:
  with path.open("rb") as source:
    header = source.read(8)
  if len(header) != 8:
    raise ProfileError("Base fbin header is truncated")
  rows, columns = struct.unpack("<II", header)
  if rows != point_count or columns != dimension or path.stat().st_size != 8 + rows * columns * 4:
    raise ProfileError("Base fbin shape is incompatible with the sidecar")
  return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, columns))


def replay_grouping(
    rows: Iterable[dict[str, str]], query_limit: int, minimum_hop: int,
    observed_owner: dict[int, int], source_node_to_cell: np.ndarray,
    observed_cell_count: int,
) -> dict[str, float | int]:
  per_query: list[Counter[int]] = []
  source_per_query: list[Counter[int]] = []
  current_query = -1
  counts: Counter[int] = Counter()
  source_counts: Counter[int] = Counter()

  def flush() -> None:
    if counts:
      per_query.append(counts.copy())
      source_per_query.append(source_counts.copy())

  for row in rows:
    query = int(row["query_id"])
    if query >= query_limit:
      continue
    if query != current_query:
      if current_query >= 0:
        flush()
      current_query = query
      counts.clear()
      source_counts.clear()
    if int(row["hop"]) < minimum_hop:
      continue
    node = int(row["node_id"])
    owner = observed_owner.get(node)
    if owner is None:
      owner = observed_cell_count + int(source_node_to_cell[node])
    counts[owner] += 1
    source_counts[int(source_node_to_cell[node])] += 1
  if current_query >= 0:
    flush()
  if not per_query:
    raise ProfileError("replay trace contains no convergence records")

  def coverage(values: Sequence[Counter[int]], limit: int) -> float:
    total = 0.0
    for query_counts in values:
      ordered = sorted(query_counts.items(), key=lambda item: (-item[1], item[0]))
      total += sum(count for _, count in ordered[:limit]) / sum(query_counts.values())
    return total / len(values)

  return {
      "queries": len(per_query),
      "source_distinct_cells_per_query": (
          sum(len(value) for value in source_per_query) / len(source_per_query)
      ),
      "distinct_cells_per_query": sum(len(value) for value in per_query) / len(per_query),
      "source_top4_cell_coverage": coverage(source_per_query, 4),
      "top1_cell_coverage": coverage(per_query, 1),
      "top4_cell_coverage": coverage(per_query, 4),
      "top8_cell_coverage": coverage(per_query, 8),
      "top16_cell_coverage": coverage(per_query, 16),
  }


def write_trace_locality_profile(
    path: Path, base_path: Path, layout: dict[str, object], groups: Sequence[np.ndarray],
) -> dict[str, int]:
  """Write a checksummed global Cell profile while touching raw vectors only for trace nodes."""
  point_count = int(layout["point_count"])
  dimension = int(layout["dimension"])
  source_cell_count = int(layout["cell_count"])
  node_to_cell = layout["node_to_cell"]
  source_populations = layout["populations"]
  packet_nodes = layout["packet_nodes"]
  source_centroids = layout["centroids"]
  assert isinstance(node_to_cell, np.ndarray)
  assert isinstance(source_populations, np.ndarray)
  assert isinstance(packet_nodes, np.ndarray)
  assert isinstance(source_centroids, np.ndarray)

  observed_nodes = np.concatenate(groups) if groups else np.empty(0, dtype="<u4")
  if len(observed_nodes) == 0 or len(np.unique(observed_nodes)) != len(observed_nodes):
    raise ProfileError("trace-locality groups must contain unique observed nodes")
  observed_source_cells = np.asarray(node_to_cell[observed_nodes], dtype="<u4")
  if np.any(observed_source_cells >= source_cell_count):
    raise ProfileError("source node-to-Cell mapping is out of range")
  removed_counts = np.bincount(
      observed_source_cells, minlength=source_cell_count
  ).astype("<u8", copy=False)
  if np.any(removed_counts > source_populations):
    raise ProfileError("trace-locality removal exceeds a source Cell population")
  remaining_counts = np.asarray(source_populations, dtype="<u8") - removed_counts
  nonempty_source = np.flatnonzero(remaining_counts)
  populations = np.concatenate(
      (np.asarray([len(group) for group in groups], dtype="<u8"),
       remaining_counts[nonempty_source])
  )
  offsets = np.empty(len(populations) + 1, dtype="<u8")
  offsets[0] = 0
  np.cumsum(populations, dtype=np.uint64, out=offsets[1:])
  if offsets[-1] != point_count or np.any(populations == 0) or np.any(populations > 128):
    raise ProfileError("trace-locality Cells violate complete bounded coverage")

  base = read_fbin_memmap(base_path, point_count, dimension)
  observed_vectors = np.asarray(base[observed_nodes], dtype=np.float32)
  group_centroids = np.empty((len(groups), dimension), dtype="<f4")
  cursor = 0
  for group_id, group in enumerate(groups):
    end = cursor + len(group)
    group_centroids[group_id] = np.mean(
        observed_vectors[cursor:end], axis=0, dtype=np.float64
    ).astype(np.float32)
    cursor = end

  source_order = np.argsort(observed_source_cells, kind="stable")
  sorted_cells = observed_source_cells[source_order]
  starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
  touched_cells = sorted_cells[starts]
  removed_sums = np.add.reduceat(
      observed_vectors[source_order].astype(np.float64), starts, axis=0
  )
  touched_index = {int(cell): index for index, cell in enumerate(touched_cells)}
  observed_mask = np.zeros(point_count, dtype=np.bool_)
  observed_mask[observed_nodes] = True

  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  checksum = 0
  written_nodes = 0
  with temporary.open("wb") as output:
    output.write(b"\0" * PQ_LOCALITY_HEADER.size)

    def write_payload(array: np.ndarray) -> None:
      nonlocal checksum
      view = memoryview(np.ascontiguousarray(array)).cast("B")
      output.write(view)
      checksum = zlib.crc32(view, checksum)

    write_payload(offsets)
    for group in groups:
      write_payload(np.asarray(group, dtype="<u4"))
      written_nodes += len(group)
    for begin in range(0, point_count, 1_000_000):
      source_nodes = np.asarray(packet_nodes[begin:min(point_count, begin + 1_000_000)])
      remaining = source_nodes[~observed_mask[source_nodes]]
      write_payload(np.asarray(remaining, dtype="<u4"))
      written_nodes += len(remaining)
    if written_nodes != point_count:
      raise ProfileError("trace-locality node order is not complete")

    write_payload(group_centroids)
    centroid_buffer = np.empty((8192, dimension), dtype="<f4")
    buffer_count = 0
    for cell in nonempty_source:
      source_centroid = source_centroids[cell]
      touched = touched_index.get(int(cell))
      if touched is None:
        centroid = source_centroid
      else:
        centroid = (
            source_centroid.astype(np.float64) * int(source_populations[cell])
            - removed_sums[touched]
        ) / int(remaining_counts[cell])
      centroid_buffer[buffer_count] = centroid
      buffer_count += 1
      if buffer_count == len(centroid_buffer):
        write_payload(centroid_buffer)
        buffer_count = 0
    if buffer_count:
      write_payload(centroid_buffer[:buffer_count])
    output.seek(0)
    output.write(
        PQ_LOCALITY_HEADER.pack(
            PQ_LOCALITY_MAGIC, PQ_LOCALITY_VERSION, PQ_LOCALITY_HEADER.size,
            BINARY_ENDIAN_MARKER, 128, dimension, point_count, len(populations), checksum,
        )
    )
  temporary.replace(path)
  return {
      "points": point_count,
      "cells": len(populations),
      "observed_nodes": len(observed_nodes),
      "observed_cells": len(groups),
      "source_remainder_cells": len(nonempty_source),
      "minimum_cell_size": int(np.min(populations)),
      "maximum_cell_size": int(np.max(populations)),
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sidecar", type=Path, required=True)
  parser.add_argument("--base_trace", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--pq_locality_output", type=Path)
  parser.add_argument("--base_file", type=Path)
  parser.add_argument("--scope", choices=("community", "global"), default="community")
  parser.add_argument("--replay_trace", type=Path)
  parser.add_argument("--structural_replay_only", action="store_true")
  parser.add_argument("--query_limit", type=int, required=True)
  parser.add_argument("--replay_query_limit", type=int)
  parser.add_argument("--minimum_hop", type=int, default=5)
  parser.add_argument("--window", type=int, default=8)
  parser.add_argument("--target_size", type=int, default=64)
  args = parser.parse_args()
  if args.query_limit <= 0 or args.minimum_hop < 0 or args.window <= 0 or args.target_size <= 0:
    raise ProfileError("query, hop, window, and target-size parameters are invalid")
  if args.structural_replay_only:
    if args.output is not None or args.pq_locality_output is not None or args.replay_trace is None:
      raise ProfileError("structural replay requires only an explicit replay trace")
  elif (args.output is None) == (args.pq_locality_output is None):
    raise ProfileError("select exactly one CSV or compact PQ-locality output")
  if (args.pq_locality_output is not None or args.structural_replay_only) and (
      (args.base_file is None and not args.structural_replay_only) or args.target_size != 128
  ):
    raise ProfileError("compact trace-locality output requires Base fbin and target size 128")
  if args.output is not None and args.scope != "community":
    raise ProfileError("legacy CSV output requires Community-local co-access scope")
  replay_query_limit = args.replay_query_limit or args.query_limit
  if replay_query_limit <= 0:
    raise ProfileError("replay query limit must be positive")

  if args.pq_locality_output is not None or args.structural_replay_only:
    layout = read_sidecar_layout(args.sidecar)
    communities = None
    source_cells = layout["node_to_cell"]
  else:
    communities, source_cells = read_memberships(args.sidecar)
  with args.base_trace.open(newline="", encoding="utf-8") as source:
    adjacency, frequency = collect_coaccess(
        csv.DictReader(source), args.query_limit, args.minimum_hop, args.window, args.scope
    )
  if args.pq_locality_output is not None or args.structural_replay_only:
    groups = build_observed_groups(adjacency, frequency, args.target_size)
    observed_owner = {
        int(node): cell_id for cell_id, group in enumerate(groups) for node in group
    }
    if args.structural_replay_only:
      summary: dict[str, object] = {
          "observed_nodes": len(observed_owner),
          "observed_cells": len(groups),
          "minimum_observed_cell_size": min(map(len, groups)),
          "maximum_observed_cell_size": max(map(len, groups)),
      }
    else:
      assert args.pq_locality_output is not None and args.base_file is not None
      summary = write_trace_locality_profile(
          args.pq_locality_output, args.base_file, layout, groups
      )
    if args.replay_trace is not None:
      with args.replay_trace.open(newline="", encoding="utf-8") as replay:
        summary["replay"] = replay_grouping(
            csv.DictReader(replay), replay_query_limit, args.minimum_hop,
            observed_owner, source_cells, len(groups),
        )
    print(json.dumps(summary, sort_keys=True))
    return 0

  assert communities is not None and args.output is not None
  cells, orders = build_assignments(
      communities, source_cells, adjacency, frequency, args.target_size
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  temporary = args.output.with_suffix(args.output.suffix + ".tmp")
  with temporary.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("node_id", "community_id", "cell_id", "order"))
    writer.writerows(
        (node, communities[node], cells[node], orders[node]) for node in range(len(cells))
    )
  temporary.replace(args.output)
  counts = Counter(cells)
  print(
      json.dumps(
          {
              "points": len(cells),
              "cells": len(counts),
              "observed_nodes": len(adjacency),
              "minimum_cell_size": min(counts.values()),
              "maximum_cell_size": max(counts.values()),
          },
          sort_keys=True,
      )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
