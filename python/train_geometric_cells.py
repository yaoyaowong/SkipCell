#!/usr/bin/env python3
"""Build deterministic capacity-balanced geometric Cells inside frozen Communities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mmap
import struct
import zlib
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from train_convergence_cells import NODE_TO_COMMUNITY_SECTION, ProfileError, map_u32_membership, read_memberships


HIERARCHY_MAGIC = 0x0052454948434C50
HIERARCHY_VERSION = 2
INVALID_ID = (1 << 32) - 1
ASSIGNMENT_BINARY_MAGIC = b"PLCAASGN"
ASSIGNMENT_BINARY_VERSION = 1
ASSIGNMENT_BINARY_HEADER = struct.Struct("<8sIIIIQQ32s")
ASSIGNMENT_BINARY_RECORD = struct.Struct("<IIII")
PQ_LOCALITY_MAGIC = b"PLCPQLOC"
PQ_LOCALITY_VERSION = 1
PQ_LOCALITY_HEADER = struct.Struct("<8sIIIIIQQQ")
TRACE_BINARY_MAGIC = b"PLCATRCE"
TRACE_BINARY_VERSION = 1
TRACE_BINARY_HEADER = struct.Struct("<8sIIIIIQQQ32s")
TRACE_BINARY_RECORD = struct.Struct("<IIB3x")
TRACE_SPLITS = {"train": 1, "heldout": 2}
BINARY_ENDIAN_MARKER = 0x01020304
ADAPTIVE_CAPACITY_CLASSES = (16, 32, 64, 128)
ADAPTIVE_AUDIT_FIELDS = (
    "parent_id",
    "child_ids",
    "child_count",
    "actual_population",
    "proposed_capacity_class",
    "tree_depth",
    "normalized_rms_radius",
    "p95_radius",
    "parent_sse",
    "child_sse",
    "split_gain",
    "centroid_radius_overlap",
    "train_cross_child_coaccess",
    "train_parent_visits",
    "train_joint_visits",
    "heldout_gt_coverage",
    "fixed_child_pages",
    "merged_parent_pages",
    "page_delta",
    "unseen_pq_delta",
    "decision",
    "rejection_reason",
)


@dataclass
class HierarchyNode:
  centroid: np.ndarray
  children: list[int]
  children_are_cells: bool
  descendant_node_count: int
  radius: float


@dataclass
class AdaptiveTreeNode:
  node_id: int
  members: np.ndarray
  children: list[int]


@dataclass(frozen=True)
class ReplayQuery:
  query_id: int
  accessed: frozenset[int]
  ground_truth: frozenset[int]
  already_seen: frozenset[int]


@dataclass(frozen=True)
class BinaryReplayTrace:
  path: Path
  expected_split: str
  point_count: int
  query_count: int
  record_count: int

  def __len__(self) -> int:
    return self.query_count

  def __iter__(self):
    return iter_binary_replay_trace(self)


@dataclass(frozen=True)
class AdaptivePolicy:
  maximum_normalized_rms_radius: float = 1.25
  maximum_normalized_p95_radius: float = 1.25
  maximum_split_gain: float = 0.15
  minimum_centroid_radius_overlap: float = 0.50
  minimum_train_cross_child_coaccess: float = 0.50
  maximum_unseen_pq_increase: float = 0.10
  page_bytes: int = 4096
  vector_bytes: int = 0
  lru_pages: int = 0


def read_fbin(path: Path) -> np.ndarray:
  with path.open("rb") as source:
    header = source.read(8)
  if len(header) != 8:
    raise ProfileError("Base fbin header is truncated")
  point_count, dimension = struct.unpack("<II", header)
  expected = 8 + point_count * dimension * 4
  if path.stat().st_size != expected:
    raise ProfileError("Base fbin size disagrees with its header")
  return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(point_count, dimension))


def read_pq_codes(path: Path) -> np.ndarray:
  """Memory-map one canonical DiskANN compressed-PQ matrix."""
  with path.open("rb") as source:
    header = source.read(8)
  if len(header) != 8:
    raise ProfileError("Compressed-PQ header is truncated")
  point_count, code_width = struct.unpack("<II", header)
  if point_count == 0 or code_width == 0 or path.stat().st_size != 8 + point_count * code_width:
    raise ProfileError("Compressed-PQ matrix has an invalid shape or byte size")
  return np.memmap(path, dtype="u1", mode="r", offset=8, shape=(point_count, code_width))


def read_diskann_pq_model(path: Path, expected_chunks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Read the pivots, global centroid, and chunk offsets used by DiskANN PQ."""
  with path.open("rb") as source:
    header = source.read(8)
    if len(header) != 8:
      raise ProfileError("PQ-pivots header is truncated")
    offset_count, offset_columns = struct.unpack("<II", header)
    if offset_columns != 1 or offset_count not in (4, 5):
      raise ProfileError("PQ-pivots offset table is incompatible")
    offsets = np.frombuffer(source.read(offset_count * 8), dtype="<u8")
  if len(offsets) != offset_count or np.any(offsets[1:] <= offsets[:-1]):
    raise ProfileError("PQ-pivots offsets are truncated or unordered")

  def matrix(offset: int, dtype: str) -> np.ndarray:
    with path.open("rb") as source:
      source.seek(int(offset))
      matrix_header = source.read(8)
    if len(matrix_header) != 8:
      raise ProfileError("PQ-pivots matrix header is truncated")
    rows, columns = struct.unpack("<II", matrix_header)
    array = np.memmap(path, dtype=dtype, mode="r", offset=int(offset) + 8,
                      shape=(rows, columns))
    return np.asarray(array).copy()

  pivots = matrix(int(offsets[0]), "<f4")
  centroid = matrix(int(offsets[1]), "<f4").reshape(-1)
  chunk_index = 3 if offset_count == 5 else 2
  chunk_offsets = matrix(int(offsets[chunk_index]), "<u4").reshape(-1)
  if (
      pivots.shape[0] != 256
      or pivots.shape[1] != len(centroid)
      or len(chunk_offsets) != expected_chunks + 1
      or chunk_offsets[0] != 0
      or chunk_offsets[-1] != len(centroid)
      or np.any(chunk_offsets[1:] <= chunk_offsets[:-1])
  ):
    raise ProfileError("PQ-pivots matrices disagree with the compressed code width")
  return pivots, centroid, chunk_offsets


def morton_byte_table(dimension_count: int, bits_per_dimension: int) -> np.ndarray:
  """Return deterministic quantized-rank-to-Morton contributions."""
  if (
      not 1 <= dimension_count <= 32
      or bits_per_dimension <= 0
      or dimension_count * bits_per_dimension > 64
  ):
    raise ProfileError("PQ-locality Morton dimensions do not fit a 64-bit key")
  table = np.zeros((dimension_count, 256), dtype=np.uint64)
  for dimension in range(dimension_count):
    for value in range(256):
      encoded = 0
      for bit in range(bits_per_dimension):
        encoded |= ((value >> bit) & 1) << (bit * dimension_count + dimension)
      table[dimension, value] = encoded
  return table


def pq_locality_order(codes: np.ndarray, pivots: np.ndarray, centroid: np.ndarray,
                      chunk_offsets: np.ndarray, locality_dimensions: int) -> np.ndarray:
  """Order node IDs by a Morton key spanning deterministic PQ-subspace principal axes."""
  chunk_count = codes.shape[1]
  dimension_count = min(locality_dimensions, 32, chunk_count)
  bits_per_dimension = 64 // dimension_count
  selected_chunks = np.linspace(0, chunk_count - 1, dimension_count, dtype=np.int32)
  table = morton_byte_table(dimension_count, bits_per_dimension)
  keys = np.zeros(len(codes), dtype=np.uint64)
  levels = 1 << bits_per_dimension
  for locality_dimension, chunk in enumerate(selected_chunks):
    begin, end = int(chunk_offsets[chunk]), int(chunk_offsets[chunk + 1])
    subspace = pivots[:, begin:end].astype(np.float64)
    centered = subspace - np.mean(subspace, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axis = right[0]
    sign_index = int(np.argmax(np.abs(axis)))
    if axis[sign_index] < 0.0:
      axis = -axis
    scores = centered @ axis
    ranked_codes = np.argsort(scores, kind="stable")
    quantized_codes = np.empty(256, dtype=np.uint8)
    quantized_codes[ranked_codes] = (
        np.arange(256, dtype=np.uint16) * levels // 256
    ).astype(np.uint8)
    keys |= table[locality_dimension, quantized_codes[codes[:, chunk]]]
  # Stable sorting makes canonical node ID the final tie-breaker without materializing it in key.
  return np.argsort(keys, kind="stable").astype(np.uint32, copy=False)


def pq_reconstructed_cell_centroids(
    codes: np.ndarray,
    ordered_nodes: np.ndarray,
    pivots: np.ndarray,
    centroid: np.ndarray,
    chunk_offsets: np.ndarray,
    target_size: int,
    samples_per_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Estimate routing centroids from bounded deterministic samples of the resident PQ codes."""
  point_count = len(ordered_nodes)
  cell_count = math.ceil(point_count / target_size)
  offsets = np.minimum(
      np.arange(cell_count + 1, dtype=np.uint64) * target_size, point_count
  )
  populations = np.diff(offsets).astype(np.uint32)
  centroids = np.zeros((cell_count, len(centroid)), dtype=np.float32)
  for sample in range(samples_per_cell):
    ranks = np.minimum(
        populations - 1,
        ((sample * 2 + 1) * populations.astype(np.uint64)) // (2 * samples_per_cell),
    )
    sample_nodes = ordered_nodes[offsets[:-1] + ranks]
    for chunk in range(codes.shape[1]):
      chunk_begin, chunk_end = int(chunk_offsets[chunk]), int(chunk_offsets[chunk + 1])
      centroids[:, chunk_begin:chunk_end] += (
          pivots[codes[sample_nodes, chunk], chunk_begin:chunk_end]
          + centroid[chunk_begin:chunk_end]
      ) / np.float32(samples_per_cell)
  return populations, centroids


def build_contiguous_centroid_hierarchy(
    cell_centroids: np.ndarray,
    populations: np.ndarray,
    branching: int,
    serialized_cell_centroids: np.ndarray | None = None,
    group_cell_leaves: bool = False,
) -> tuple[int, list[HierarchyNode]]:
  """Build a deterministic bottom-up hierarchy over PQ-locality ordered Cells."""
  if len(cell_centroids) == 0 or len(cell_centroids) != len(populations) or branching < 2:
    raise ProfileError("PQ-locality hierarchy inputs are invalid")
  leaf_centroid_values = (
      np.asarray(cell_centroids, dtype=np.float32)
      if serialized_cell_centroids is None
      else np.asarray(serialized_cell_centroids, dtype=np.float32)
  )
  if leaf_centroid_values.shape != cell_centroids.shape:
    raise ProfileError("Serialized Cell centroids disagree with hierarchy routing centroids")
  hierarchy: list[HierarchyNode] = []
  level: list[int] = []
  if group_cell_leaves:
    for begin in range(0, len(cell_centroids), branching):
      children = list(range(begin, min(len(cell_centroids), begin + branching)))
      weights = populations[children].astype(np.float64)
      centers = cell_centroids[children].astype(np.float64)
      center = np.average(centers, axis=0, weights=weights)
      distances = np.linalg.norm(centers - center, axis=1)
      radius = float(np.nextafter(np.float32(np.max(distances)), np.float32(np.inf)))
      level.append(len(hierarchy))
      hierarchy.append(
          HierarchyNode(center.astype(np.float32), children, True, int(np.sum(weights)), radius)
      )
    while len(level) != 1:
      next_level = []
      for begin in range(0, len(level), branching):
        children = level[begin : begin + branching]
        weights = np.asarray(
            [hierarchy[child].descendant_node_count for child in children], dtype=np.float64
        )
        centers = np.asarray(
            [hierarchy[child].centroid for child in children], dtype=np.float64
        )
        center = np.average(centers, axis=0, weights=weights)
        distances = np.linalg.norm(centers - center, axis=1) + np.asarray(
            [hierarchy[child].radius for child in children], dtype=np.float64
        )
        radius = float(np.nextafter(np.float32(np.max(distances)), np.float32(np.inf)))
        next_level.append(len(hierarchy))
        hierarchy.append(
            HierarchyNode(center.astype(np.float32), children, False, int(np.sum(weights)), radius)
        )
      level = next_level
    return level[0], hierarchy
  leaf_radius = (
      float(np.nextafter(np.float32(0.0), np.float32(np.inf)))
      if serialized_cell_centroids is not None
      else 0.0
  )
  for cell_id in range(len(cell_centroids)):
    level.append(len(hierarchy))
    hierarchy.append(
        HierarchyNode(
            leaf_centroid_values[cell_id],
            [cell_id],
            True,
            int(populations[cell_id]),
            leaf_radius,
        )
    )
  leaf_centers = np.asarray(leaf_centroid_values, dtype=np.float64)
  level_centers = np.asarray(cell_centroids, dtype=np.float64)
  level_weights = np.asarray(populations, dtype=np.float64)
  level_leaf_begins = np.arange(len(level), dtype=np.int64)
  while len(level) != 1:
    begins = np.arange(0, len(level), branching, dtype=np.int64)
    next_weights = np.add.reduceat(level_weights, begins)
    next_centers = np.empty((len(begins), level_centers.shape[1]), dtype=np.float64)
    for parent_begin in range(0, len(begins), 256):
      parent_end = min(len(begins), parent_begin + 256)
      child_begin = int(begins[parent_begin])
      child_end = int(begins[parent_end]) if parent_end < len(begins) else len(level)
      local_begins = begins[parent_begin:parent_end] - child_begin
      weighted = (
          level_centers[child_begin:child_end]
          * level_weights[child_begin:child_end, np.newaxis]
      )
      next_centers[parent_begin:parent_end] = np.add.reduceat(
          weighted, local_begins, axis=0
      ) / next_weights[parent_begin:parent_end, np.newaxis]
    serialized_next_centers = next_centers.astype(np.float32)
    next_leaf_begins = level_leaf_begins[begins]
    next_leaf_ends = np.append(next_leaf_begins[1:], len(leaf_centers))
    leaf_parent_positions = np.repeat(
        np.arange(len(begins), dtype=np.int64), next_leaf_ends - next_leaf_begins
    )
    next_radius_squared = np.zeros(len(begins), dtype=np.float64)
    for leaf_begin in range(0, len(leaf_centers), 8192):
      leaf_end = min(len(leaf_centers), leaf_begin + 8192)
      parent_slice = leaf_parent_positions[leaf_begin:leaf_end]
      differences = leaf_centers[leaf_begin:leaf_end] - serialized_next_centers[parent_slice]
      squared = np.sum(differences * differences, axis=1, dtype=np.float64)
      np.maximum.at(next_radius_squared, parent_slice, squared)
    next_radii = np.sqrt(next_radius_squared)
    serialized_next_radii = np.nextafter(
        next_radii.astype(np.float32), np.float32(np.inf)
    ).astype(np.float64)
    next_level = []
    for parent_position, begin in enumerate(begins):
      begin = int(begin)
      children = level[begin : begin + branching]
      next_level.append(len(hierarchy))
      hierarchy.append(
          HierarchyNode(
              serialized_next_centers[parent_position],
              children,
              False,
              int(next_weights[parent_position]),
              float(serialized_next_radii[parent_position]),
          )
      )
    level = next_level
    level_centers = np.asarray(serialized_next_centers, dtype=np.float64)
    level_weights = next_weights
    level_leaf_begins = next_leaf_begins
  return level[0], hierarchy


def write_pq_locality_profile(path: Path, ordered_nodes: np.ndarray, populations: np.ndarray,
                              cell_centroids: np.ndarray, target_size: int) -> None:
  """Write one compact checksummed Cell-order artifact consumed by sidecar-only repacking."""
  point_count = len(ordered_nodes)
  cell_count = len(populations)
  offsets = np.empty(cell_count + 1, dtype="<u8")
  offsets[0] = 0
  np.cumsum(populations, dtype=np.uint64, out=offsets[1:])
  if offsets[-1] != point_count or cell_centroids.shape[0] != cell_count:
    raise ProfileError("PQ-locality profile has inconsistent Cell coverage")
  payload_checksum = 0
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("wb") as output:
    output.write(b"\0" * PQ_LOCALITY_HEADER.size)
    for array in (offsets, np.asarray(ordered_nodes, dtype="<u4"),
                  np.asarray(cell_centroids, dtype="<f4")):
      view = memoryview(array).cast("B")
      output.write(view)
      payload_checksum = zlib.crc32(view, payload_checksum)
    output.seek(0)
    output.write(
        PQ_LOCALITY_HEADER.pack(
            PQ_LOCALITY_MAGIC,
            PQ_LOCALITY_VERSION,
            PQ_LOCALITY_HEADER.size,
            BINARY_ENDIAN_MARKER,
            target_size,
            cell_centroids.shape[1],
            point_count,
            cell_count,
            payload_checksum,
        )
    )
  temporary.replace(path)


def balanced_kmeans(points: np.ndarray, target_size: int, iterations: int) -> list[np.ndarray]:
  count = len(points)
  cluster_count = max(1, math.ceil(count / target_size))
  capacity = np.full(cluster_count, count // cluster_count, dtype=np.int32)
  capacity[: count % cluster_count] += 1
  if cluster_count == 1:
    return [np.arange(count, dtype=np.int32)]

  direction = np.linspace(-1.0, 1.0, points.shape[1], dtype=np.float32)
  initial_order = np.argsort(points @ direction, kind="stable")
  seeds = initial_order[
      np.minimum(
          count - 1,
          ((np.arange(cluster_count, dtype=np.int64) * 2 + 1) * count)
          // (2 * cluster_count),
      )
  ]
  centroids = np.asarray(points[seeds], dtype=np.float32).copy()
  assignment = np.full(count, -1, dtype=np.int32)

  for _ in range(iterations):
    point_norms = np.einsum("ij,ij->i", points, points, optimize=True)
    centroid_norms = np.einsum("ij,ij->i", centroids, centroids, optimize=True)
    distances = point_norms[:, None] + centroid_norms[None, :] - 2.0 * (points @ centroids.T)
    preferences = np.argsort(distances, axis=1, kind="stable")
    confidence = distances[np.arange(count), preferences[:, 1]] - distances[
        np.arange(count), preferences[:, 0]
    ]
    assignment.fill(-1)
    remaining = capacity.copy()
    for point in np.argsort(-confidence, kind="stable"):
      for cluster in preferences[point]:
        if remaining[cluster] != 0:
          assignment[point] = cluster
          remaining[cluster] -= 1
          break
    for cluster in range(cluster_count):
      members = np.flatnonzero(assignment == cluster)
      centroids[cluster] = np.mean(points[members], axis=0, dtype=np.float64)

  return [np.flatnonzero(assignment == cluster) for cluster in range(cluster_count)]


def recursive_global_kmeans(
    points: np.ndarray, nodes: np.ndarray, target_size: int, iterations: int
) -> list[np.ndarray]:
  groups: list[np.ndarray] = []
  hierarchy: list[HierarchyNode] = []
  cell_centroids: list[np.ndarray] = []
  recursive_global_hierarchy(
      points, nodes, target_size, iterations, groups, hierarchy, cell_centroids
  )
  return groups


def recursive_global_hierarchy(
    points: np.ndarray,
    nodes: np.ndarray,
    target_size: int,
    iterations: int,
    groups: list[np.ndarray],
    hierarchy: list[HierarchyNode],
    cell_centroids: list[np.ndarray],
) -> tuple[int, list[int]]:
  node_id = len(hierarchy)
  hierarchy.append(
      HierarchyNode(np.empty(0, dtype=np.float32), [], False, 0, 0.0)
  )
  if len(nodes) <= target_size:
    cell_id = len(groups)
    groups.append(nodes)
    centroid = np.asarray(
        np.mean(points[nodes], axis=0, dtype=np.float64), dtype=np.float32
    )
    cell_centroids.append(centroid)
    hierarchy[node_id] = HierarchyNode(centroid, [cell_id], True, len(nodes), 0.0)
    return node_id, [cell_id]
  cluster_count = min(16, max(2, math.ceil(len(nodes) / target_size)))
  local = np.asarray(points[nodes], dtype=np.float32)
  node_centroid = np.asarray(np.mean(local, axis=0, dtype=np.float64), dtype=np.float32)
  direction = np.linspace(-1.0, 1.0, local.shape[1], dtype=np.float32)
  order = np.argsort(local @ direction, kind="stable")
  seeds = order[
      np.minimum(
          len(nodes) - 1,
          ((np.arange(cluster_count, dtype=np.int64) * 2 + 1) * len(nodes))
          // (2 * cluster_count),
      )
  ]
  centroids = local[seeds].copy()
  assignment = np.zeros(len(nodes), dtype=np.int32)
  for _ in range(iterations):
    point_norms = np.einsum("ij,ij->i", local, local, optimize=True)
    centroid_norms = np.einsum("ij,ij->i", centroids, centroids, optimize=True)
    distances = point_norms[:, None] + centroid_norms[None, :] - 2.0 * (local @ centroids.T)
    assignment = np.argmin(distances, axis=1).astype(np.int32)
    for cluster in range(cluster_count):
      members = np.flatnonzero(assignment == cluster)
      if len(members) == 0:
        farthest = int(np.argmax(np.min(distances, axis=1)))
        assignment[farthest] = cluster
        members = np.asarray([farthest], dtype=np.int32)
      centroids[cluster] = np.mean(local[members], axis=0, dtype=np.float64)
  child_members = [nodes[np.flatnonzero(assignment == cluster)] for cluster in range(cluster_count)]
  del local, centroids, assignment
  child_nodes: list[int] = []
  descendant_cells: list[int] = []
  for cluster in range(cluster_count):
    child_node, child_cells = recursive_global_hierarchy(
        points,
        child_members[cluster],
        target_size,
        iterations,
        groups,
        hierarchy,
        cell_centroids,
    )
    child_nodes.append(child_node)
    descendant_cells.extend(child_cells)
  leaf_centers = np.asarray(
      [cell_centroids[cell_id] for cell_id in descendant_cells], dtype=np.float32
  )
  differences = np.asarray(leaf_centers, dtype=np.float64) - np.asarray(
      node_centroid, dtype=np.float64
  )
  radius64 = float(np.sqrt(np.max(np.einsum("ij,ij->i", differences, differences))))
  radius = float(np.nextafter(np.float32(radius64), np.float32(np.inf)))
  hierarchy[node_id] = HierarchyNode(
      node_centroid, child_nodes, False, len(nodes), radius
  )
  return node_id, descendant_cells


def fnv1a(data: bytes | bytearray) -> int:
  value = 14695981039346656037
  for byte in data:
    value ^= byte
    value = (value * 1099511628211) & ((1 << 64) - 1)
  return value


def write_hierarchy(
    path: Path,
    dimension: int,
    cell_count: int,
    community_count: int,
    root: int,
    hierarchy: list[HierarchyNode],
) -> None:
  children = [child for node in hierarchy for child in node.children]
  roots = [INVALID_ID] * community_count
  roots[0] = root
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("wb") as output:
    output.write(b"\0" * 40)
    checksum = 0

    def write_payload(data: bytes | bytearray | memoryview) -> None:
      nonlocal checksum
      output.write(data)
      checksum = zlib.crc32(data, checksum)

    write_payload(struct.pack(f"<{community_count}I", *roots))
    child_begin = 0
    records = bytearray()
    for node_id, node in enumerate(hierarchy):
      records.extend(
          struct.pack(
              "<IIIQfI",
              node_id,
              child_begin,
              len(node.children),
              node.descendant_node_count,
              node.radius,
              int(node.children_are_cells),
          )
      )
      child_begin += len(node.children)
      if len(records) >= 8 * 1024 * 1024:
        write_payload(records)
        records.clear()
    if records:
      write_payload(records)
    child_array = np.asarray(children, dtype="<u4")
    write_payload(memoryview(child_array).cast("B"))
    for begin in range(0, len(hierarchy), 1024):
      centroids = np.asarray(
          [node.centroid for node in hierarchy[begin : begin + 1024]], dtype="<f4"
      )
      if centroids.shape != (min(1024, len(hierarchy) - begin), dimension):
        raise ProfileError("Hierarchy centroids have an unexpected shape")
      write_payload(memoryview(centroids).cast("B"))
    output.seek(0)
    output.write(
        struct.pack(
            "<QIIIIIIQ",
            HIERARCHY_MAGIC,
            HIERARCHY_VERSION,
            dimension,
            cell_count,
            len(hierarchy),
            community_count,
            len(children),
            checksum,
        )
    )
  temporary.replace(path)


def read_hierarchy(path: Path) -> tuple[int, int, int, list[HierarchyNode]]:
  """Read and validate a deterministic geometric hierarchy artifact."""
  data = path.read_bytes()
  if len(data) < 40:
    raise ProfileError("Source hierarchy header is truncated")
  (
      magic,
      version,
      dimension,
      cell_count,
      node_count,
      community_count,
      child_count,
      checksum,
  ) = struct.unpack_from("<QIIIIIIQ", data)
  if magic != HIERARCHY_MAGIC or version not in (1, HIERARCHY_VERSION):
    raise ProfileError("Source hierarchy has an unsupported magic or version")
  if dimension == 0 or cell_count == 0 or node_count == 0 or community_count == 0:
    raise ProfileError("Source hierarchy has an invalid zero count")
  expected = 40 + community_count * 4 + node_count * 28 + child_count * 4
  expected += node_count * dimension * 4
  if len(data) != expected:
    raise ProfileError("Source hierarchy byte size disagrees with its header")
  payload = memoryview(data)[40:]
  actual_checksum = fnv1a(payload) if version == 1 else zlib.crc32(payload)
  if actual_checksum != checksum:
    raise ProfileError("Source hierarchy checksum mismatch")
  offset = 0
  roots = struct.unpack_from(f"<{community_count}I", payload, offset)
  offset += community_count * 4
  valid_roots = [root for root in roots if root != INVALID_ID]
  if len(valid_roots) != 1 or valid_roots[0] >= node_count:
    raise ProfileError("Source global hierarchy must contain exactly one valid root")
  records = []
  for expected_node_id in range(node_count):
    record = struct.unpack_from("<IIIQfI", payload, offset)
    offset += 28
    node_id, child_begin, count, descendants, radius, children_are_cells = record
    if node_id != expected_node_id or child_begin + count > child_count:
      raise ProfileError("Source hierarchy has an invalid node record")
    if count == 0 or children_are_cells not in (0, 1):
      raise ProfileError("Source hierarchy has an empty node or invalid child kind")
    if descendants == 0 or not math.isfinite(radius) or radius < 0.0:
      raise ProfileError("Source hierarchy has invalid geometry metadata")
    records.append(record)
  children = struct.unpack_from(f"<{child_count}I", payload, offset)
  offset += child_count * 4
  centroids = np.frombuffer(
      payload, dtype="<f4", count=node_count * dimension, offset=offset
  ).reshape(node_count, dimension)
  if not np.all(np.isfinite(centroids)):
    raise ProfileError("Source hierarchy contains a non-finite centroid")
  hierarchy = []
  node_parents = [-1] * node_count
  cell_parents = [-1] * cell_count
  for node_id, record in enumerate(records):
    _, child_begin, count, descendants, radius, children_are_cells = record
    node_children = list(children[child_begin : child_begin + count])
    limit = cell_count if children_are_cells else node_count
    if any(child >= limit for child in node_children):
      raise ProfileError("Source hierarchy contains an out-of-range child")
    parents = cell_parents if children_are_cells else node_parents
    for child in node_children:
      if parents[child] != -1:
        raise ProfileError("Source hierarchy has non-unique parentage")
      parents[child] = node_id
    hierarchy.append(
        HierarchyNode(
            np.asarray(centroids[node_id], dtype=np.float32).copy(),
            node_children,
            bool(children_are_cells),
            descendants,
            radius,
        )
    )
  root = valid_roots[0]
  if node_parents[root] != -1 or any(
      parent == -1 for node_id, parent in enumerate(node_parents) if node_id != root
  ):
    raise ProfileError("Source hierarchy is disconnected or its root has a parent")
  if any(parent == -1 for parent in cell_parents):
    raise ProfileError("Source hierarchy does not cover every Cell")
  return root, dimension, cell_count, hierarchy


def sha256_payload(path: Path, offset: int) -> bytes:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    source.seek(offset)
    while chunk := source.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.digest()


def _groups_from_assignment_arrays(
    node_cells: np.ndarray, node_orders: np.ndarray, cell_count: int
) -> list[np.ndarray]:
  if len(node_cells) == 0 or len(node_cells) != len(node_orders):
    raise ProfileError("Source assignment arrays have invalid lengths")
  if np.any(node_cells >= cell_count):
    raise ProfileError("Source assignment contains an out-of-range Cell")
  counts = np.bincount(node_cells, minlength=cell_count).astype(np.uint64, copy=False)
  if np.any(counts == 0):
    raise ProfileError("Source assignment contains an empty Cell")
  if np.any(node_orders >= counts[node_cells]):
    raise ProfileError("Source assignment Cell order is out of range")
  offsets = np.empty(cell_count + 1, dtype=np.uint64)
  offsets[0] = 0
  np.cumsum(counts, out=offsets[1:])
  positions = offsets[node_cells] + node_orders
  members = np.full(len(node_cells), INVALID_ID, dtype=np.uint32)
  members[positions] = np.arange(len(node_cells), dtype=np.uint32)
  if np.any(members == INVALID_ID):
    raise ProfileError("Source assignment Cell order is duplicated or incomplete")
  return [
      members[int(offsets[cell]) : int(offsets[cell + 1])]
      for cell in range(cell_count)
  ]


def read_binary_assignment_groups(
    path: Path, point_count: int, cell_count: int
) -> list[np.ndarray]:
  with path.open("rb") as source:
    header_bytes = source.read(ASSIGNMENT_BINARY_HEADER.size)
  if len(header_bytes) != ASSIGNMENT_BINARY_HEADER.size:
    raise ProfileError("Binary assignment header is truncated")
  magic, version, header_size, record_size, endian, stored_points, stored_cells, digest = (
      ASSIGNMENT_BINARY_HEADER.unpack(header_bytes)
  )
  expected_size = header_size + stored_points * record_size
  if (
      magic != ASSIGNMENT_BINARY_MAGIC
      or version != ASSIGNMENT_BINARY_VERSION
      or header_size != ASSIGNMENT_BINARY_HEADER.size
      or record_size != ASSIGNMENT_BINARY_RECORD.size
      or endian != BINARY_ENDIAN_MARKER
      or stored_points != point_count
      or stored_cells != cell_count
      or path.stat().st_size != expected_size
      or sha256_payload(path, header_size) != digest
  ):
    raise ProfileError("Binary assignment header, size, or checksum is invalid")
  records = np.memmap(
      path,
      dtype=np.dtype(
          [
              ("node_id", "<u4"),
              ("community_id", "<u4"),
              ("cell_id", "<u4"),
              ("order", "<u4"),
          ]
      ),
      mode="r",
      offset=header_size,
      shape=(point_count,),
  )
  validation_chunk = 8 * 1024 * 1024
  for begin in range(0, point_count, validation_chunk):
    end = min(point_count, begin + validation_chunk)
    expected = np.arange(begin, end, dtype=np.uint32)
    if not np.array_equal(records["node_id"][begin:end], expected):
      raise ProfileError("Binary assignment node IDs are not canonical and complete")
  return _groups_from_assignment_arrays(records["cell_id"], records["order"], cell_count)


def read_assignment_groups(
    path: Path, point_count: int, cell_count: int
) -> list[np.ndarray]:
  """Read a one-owner CSV or checksummed binary Cell assignment."""
  with path.open("rb") as source:
    if source.read(8) == ASSIGNMENT_BINARY_MAGIC:
      return read_binary_assignment_groups(path, point_count, cell_count)
  required = {"node_id", "community_id", "cell_id", "order"}
  node_cells = np.full(point_count, INVALID_ID, dtype=np.uint32)
  node_orders = np.full(point_count, INVALID_ID, dtype=np.uint32)
  with path.open(newline="", encoding="utf-8") as source:
    reader = csv.DictReader(source)
    missing = required.difference(reader.fieldnames or ())
    if missing:
      raise ProfileError(f"Source assignment is missing fields: {sorted(missing)}")
    for line_number, row in enumerate(reader, start=2):
      try:
        node_id = int(row["node_id"])
        cell_id = int(row["cell_id"])
        order = int(row["order"])
        community_id = int(row["community_id"])
      except ValueError as error:
        raise ProfileError(
            f"Source assignment row {line_number} has an invalid integer"
        ) from error
      if (
          not 0 <= node_id < point_count
          or not 0 <= cell_id < cell_count
          or not 0 <= order <= INVALID_ID
          or not 0 <= community_id <= INVALID_ID
      ):
        raise ProfileError(f"Source assignment row {line_number} is out of range")
      if node_cells[node_id] != INVALID_ID:
        raise ProfileError("Source assignment gives one vector multiple Cell owners")
      node_cells[node_id] = cell_id
      node_orders[node_id] = order
  if np.any(node_cells == INVALID_ID):
    raise ProfileError("Source assignment does not provide complete vector coverage")
  return _groups_from_assignment_arrays(node_cells, node_orders, cell_count)


def finalize_hierarchy_geometry(
    node_id: int,
    hierarchy: list[HierarchyNode],
    cell_centroids: Sequence[np.ndarray],
) -> list[int]:
  node = hierarchy[node_id]
  if node.children_are_cells:
    descendant_cells = list(node.children)
    if len(descendant_cells) == 1:
      node.centroid = cell_centroids[descendant_cells[0]]
      node.radius = float(np.nextafter(np.float32(0.0), np.float32(np.inf)))
      return descendant_cells
  else:
    descendant_cells = []
    for child in node.children:
      descendant_cells.extend(
          finalize_hierarchy_geometry(child, hierarchy, cell_centroids)
      )
  leaf_centers = np.asarray(
      [cell_centroids[cell_id] for cell_id in descendant_cells], dtype=np.float32
  )
  differences = np.asarray(leaf_centers, dtype=np.float64) - np.asarray(
      node.centroid, dtype=np.float64
  )
  squared = np.sum(differences * differences, axis=1, dtype=np.float64)
  radius64 = float(np.sqrt(np.max(squared)))
  node.radius = float(np.nextafter(np.float32(radius64), np.float32(np.inf)))
  return descendant_cells


def _parse_flag(value: str, field: str, line_number: int) -> bool:
  if value not in ("0", "1"):
    raise ProfileError(
        f"Adaptive trace row {line_number} field {field} must be zero or one"
    )
  return value == "1"


def nearest_rank_p95(values: np.ndarray) -> float:
  if len(values) == 0:
    raise ProfileError("Cannot compute a p95 radius for an empty region")
  ordered = np.sort(np.asarray(values, dtype=np.float64), kind="stable")
  return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def iter_csv_replay_trace(path: Path, expected_split: str, point_count: int):
  required = {"query_id", "node_id", "split", "accessed", "gt", "already_seen"}
  current_query = None
  flags = None
  previous_pair = None
  with path.open(newline="", encoding="utf-8") as source:
    reader = csv.DictReader(source)
    missing = required.difference(reader.fieldnames or ())
    if missing:
      raise ProfileError(f"Adaptive trace is missing fields: {sorted(missing)}")
    for line_number, row in enumerate(reader, start=2):
      try:
        query_id = int(row["query_id"])
        node_id = int(row["node_id"])
      except ValueError as error:
        raise ProfileError(f"Adaptive trace row {line_number} has an invalid ID") from error
      if query_id < 0 or node_id < 0 or node_id >= point_count:
        raise ProfileError(f"Adaptive trace row {line_number} has an out-of-range ID")
      if row["split"] != expected_split:
        raise ProfileError(
            f"Adaptive trace row {line_number} belongs to {row['split']!r}, "
            f"not {expected_split!r}"
        )
      pair = (query_id, node_id)
      if previous_pair is not None and pair <= previous_pair:
        raise ProfileError("Adaptive CSV trace must be sorted by query ID and node ID")
      previous_pair = pair
      if current_query != query_id:
        if current_query is not None:
          if not flags["accessed"]:
            raise ProfileError(f"Adaptive trace query {current_query} has no accessed node")
          yield ReplayQuery(
              current_query,
              frozenset(flags["accessed"]),
              frozenset(flags["ground_truth"]),
              frozenset(flags["already_seen"]),
          )
        current_query = query_id
        flags = {"accessed": set(), "ground_truth": set(), "already_seen": set()}
      if _parse_flag(row["accessed"], "accessed", line_number):
        flags["accessed"].add(node_id)
      if _parse_flag(row["gt"], "gt", line_number):
        flags["ground_truth"].add(node_id)
      if _parse_flag(row["already_seen"], "already_seen", line_number):
        flags["already_seen"].add(node_id)
  if current_query is None:
    raise ProfileError("Adaptive trace contains no queries")
  if not flags["accessed"]:
    raise ProfileError(f"Adaptive trace query {current_query} has no accessed node")
  yield ReplayQuery(
      current_query,
      frozenset(flags["accessed"]),
      frozenset(flags["ground_truth"]),
      frozenset(flags["already_seen"]),
  )


def open_binary_replay_trace(
    path: Path, expected_split: str, point_count: int
) -> BinaryReplayTrace:
  with path.open("rb") as source:
    header_bytes = source.read(TRACE_BINARY_HEADER.size)
  if len(header_bytes) != TRACE_BINARY_HEADER.size:
    raise ProfileError("Binary adaptive trace header is truncated")
  (
      magic,
      version,
      header_size,
      record_size,
      split,
      endian,
      stored_points,
      query_count,
      record_count,
      digest,
  ) = TRACE_BINARY_HEADER.unpack(header_bytes)
  if (
      magic != TRACE_BINARY_MAGIC
      or version != TRACE_BINARY_VERSION
      or header_size != TRACE_BINARY_HEADER.size
      or record_size != TRACE_BINARY_RECORD.size
      or split != TRACE_SPLITS.get(expected_split)
      or endian != BINARY_ENDIAN_MARKER
      or stored_points != point_count
      or query_count == 0
      or record_count == 0
      or path.stat().st_size != header_size + record_count * record_size
      or sha256_payload(path, header_size) != digest
  ):
    raise ProfileError("Binary adaptive trace header, size, split, or checksum is invalid")
  return BinaryReplayTrace(path, expected_split, point_count, query_count, record_count)


def iter_binary_replay_trace(trace: BinaryReplayTrace):
  records = np.memmap(
      trace.path,
      dtype=np.dtype([("query_id", "<u4"), ("node_id", "<u4"), ("flags", "u1"), ("pad", "V3")]),
      mode="r",
      offset=TRACE_BINARY_HEADER.size,
      shape=(trace.record_count,),
  )
  current_query = None
  accessed = set()
  ground_truth = set()
  already_seen = set()
  previous_pair = None
  yielded = 0
  for record in records:
    query_id = int(record["query_id"])
    node_id = int(record["node_id"])
    bits = int(record["flags"])
    pair = (query_id, node_id)
    if node_id >= trace.point_count or bits & ~0x7 or (
        previous_pair is not None and pair <= previous_pair
    ):
      raise ProfileError("Binary adaptive trace contains an invalid or unsorted record")
    previous_pair = pair
    if current_query != query_id:
      if current_query is not None:
        if not accessed:
          raise ProfileError(f"Adaptive trace query {current_query} has no accessed node")
        yield ReplayQuery(
            current_query,
            frozenset(accessed),
            frozenset(ground_truth),
            frozenset(already_seen),
        )
        yielded += 1
      current_query = query_id
      accessed = set()
      ground_truth = set()
      already_seen = set()
    if bits & 0x1:
      accessed.add(node_id)
    if bits & 0x2:
      ground_truth.add(node_id)
    if bits & 0x4:
      already_seen.add(node_id)
  if current_query is None or not accessed:
    raise ProfileError("Binary adaptive trace has an empty final query")
  yield ReplayQuery(
      current_query,
      frozenset(accessed),
      frozenset(ground_truth),
      frozenset(already_seen),
  )
  yielded += 1
  if yielded != trace.query_count:
    raise ProfileError("Binary adaptive trace query count disagrees with its header")


def open_replay_trace(path: Path, expected_split: str, point_count: int):
  with path.open("rb") as source:
    if source.read(8) == TRACE_BINARY_MAGIC:
      return open_binary_replay_trace(path, expected_split, point_count)
  return list(iter_csv_replay_trace(path, expected_split, point_count))


def read_replay_trace(path: Path, expected_split: str, point_count: int) -> list[ReplayQuery]:
  return list(open_replay_trace(path, expected_split, point_count))


def write_assignment_binary(
    source_path: Path, output_path: Path, point_count: int, cell_count: int
) -> None:
  required = {"node_id", "community_id", "cell_id", "order"}
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_path.with_suffix(output_path.suffix + ".tmp")
  digest = hashlib.sha256()
  record_count = 0
  with source_path.open(newline="", encoding="utf-8") as source, temporary.open(
      "wb"
  ) as output:
    output.write(b"\0" * ASSIGNMENT_BINARY_HEADER.size)
    reader = csv.DictReader(source)
    missing = required.difference(reader.fieldnames or ())
    if missing:
      raise ProfileError(f"Source assignment is missing fields: {sorted(missing)}")
    buffer = bytearray()
    for line_number, row in enumerate(reader, start=2):
      try:
        values = tuple(int(row[field]) for field in ("node_id", "community_id", "cell_id", "order"))
      except ValueError as error:
        raise ProfileError(f"Source assignment row {line_number} has an invalid integer") from error
      node_id, community_id, cell_id, order = values
      if (
          node_id != record_count
          or not 0 <= community_id <= INVALID_ID
          or not 0 <= cell_id < cell_count
          or not 0 <= order <= INVALID_ID
      ):
        raise ProfileError("Source assignment is non-canonical or out of range")
      buffer.extend(ASSIGNMENT_BINARY_RECORD.pack(*values))
      record_count += 1
      if len(buffer) >= 8 * 1024 * 1024:
        output.write(buffer)
        digest.update(buffer)
        buffer.clear()
    if buffer:
      output.write(buffer)
      digest.update(buffer)
    if record_count != point_count:
      raise ProfileError("Source assignment point count is incomplete")
    output.seek(0)
    output.write(
        ASSIGNMENT_BINARY_HEADER.pack(
            ASSIGNMENT_BINARY_MAGIC,
            ASSIGNMENT_BINARY_VERSION,
            ASSIGNMENT_BINARY_HEADER.size,
            ASSIGNMENT_BINARY_RECORD.size,
            BINARY_ENDIAN_MARKER,
            point_count,
            cell_count,
            digest.digest(),
        )
    )
  read_binary_assignment_groups(temporary, point_count, cell_count)
  temporary.replace(output_path)


def write_assignment_binary_from_order(
    output_path: Path,
    ordered_nodes: np.ndarray,
    populations: np.ndarray,
    communities: Sequence[int],
) -> None:
  """Write canonical node-order assignment records without a 100M-row CSV intermediate."""
  point_count = len(ordered_nodes)
  cell_count = len(populations)
  if point_count == 0 or cell_count == 0 or sum(map(int, populations)) != point_count:
    raise ProfileError("PQ-locality assignment coverage is invalid")
  if np.any(ordered_nodes >= point_count):
    raise ProfileError("PQ-locality order contains an out-of-range node ID")
  inverse = np.full(point_count, INVALID_ID, dtype=np.uint32)
  inverse[ordered_nodes] = np.arange(point_count, dtype=np.uint32)
  if np.any(inverse == INVALID_ID):
    raise ProfileError("PQ-locality order is not a complete node permutation")
  offsets = np.empty(cell_count + 1, dtype=np.uint64)
  offsets[0] = 0
  np.cumsum(populations, dtype=np.uint64, out=offsets[1:])
  community_array = np.asarray(communities, dtype=np.uint32)
  if len(community_array) != point_count:
    raise ProfileError("PQ-locality assignment Community coverage is invalid")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_path.with_suffix(output_path.suffix + ".tmp")
  digest = hashlib.sha256()
  record_dtype = np.dtype(
      [("node_id", "<u4"), ("community_id", "<u4"), ("cell_id", "<u4"), ("order", "<u4")]
  )
  with temporary.open("wb") as output:
    output.write(b"\0" * ASSIGNMENT_BINARY_HEADER.size)
    for begin in range(0, point_count, 1_000_000):
      end = min(point_count, begin + 1_000_000)
      ranks = inverse[begin:end]
      cell_ids = np.searchsorted(offsets, ranks, side="right").astype(np.uint32) - 1
      records = np.empty(end - begin, dtype=record_dtype)
      records["node_id"] = np.arange(begin, end, dtype=np.uint32)
      records["community_id"] = community_array[begin:end]
      records["cell_id"] = cell_ids
      records["order"] = ranks - offsets[cell_ids]
      payload = memoryview(records).cast("B")
      output.write(payload)
      digest.update(payload)
    output.seek(0)
    output.write(
        ASSIGNMENT_BINARY_HEADER.pack(
            ASSIGNMENT_BINARY_MAGIC,
            ASSIGNMENT_BINARY_VERSION,
            ASSIGNMENT_BINARY_HEADER.size,
            ASSIGNMENT_BINARY_RECORD.size,
            BINARY_ENDIAN_MARKER,
            point_count,
            cell_count,
            digest.digest(),
        )
    )
  del inverse
  expected_size = ASSIGNMENT_BINARY_HEADER.size + point_count * ASSIGNMENT_BINARY_RECORD.size
  if temporary.stat().st_size != expected_size:
    raise ProfileError("PQ-locality assignment size is invalid")
  with temporary.open("rb") as source:
    header = ASSIGNMENT_BINARY_HEADER.unpack(source.read(ASSIGNMENT_BINARY_HEADER.size))
  if (
      header[0] != ASSIGNMENT_BINARY_MAGIC
      or header[1] != ASSIGNMENT_BINARY_VERSION
      or header[5] != point_count
      or header[6] != cell_count
      or sha256_payload(temporary, ASSIGNMENT_BINARY_HEADER.size) != header[7]
  ):
    raise ProfileError("PQ-locality assignment header or checksum is invalid")
  temporary.replace(output_path)


def compute_group_centroids(
    points: np.ndarray,
    ordered_nodes: np.ndarray,
    populations: np.ndarray,
    batch_cells: int = 4096,
) -> np.ndarray:
  """Compute float64-accumulated Cell centroids in bounded vectorized batches."""
  if batch_cells <= 0 or len(populations) == 0:
    raise ProfileError("Grouped centroid batch or population count is invalid")
  counts = np.asarray(populations, dtype=np.int64)
  if np.any(counts <= 0) or int(np.sum(counts, dtype=np.int64)) != len(ordered_nodes):
    raise ProfileError("Grouped centroid populations do not cover the ordered nodes")
  centroids = np.empty((len(counts), points.shape[1]), dtype=np.float32)
  node_begin = 0
  for cell_begin in range(0, len(counts), batch_cells):
    cell_end = min(len(counts), cell_begin + batch_cells)
    batch_counts = counts[cell_begin:cell_end]
    node_end = node_begin + int(np.sum(batch_counts, dtype=np.int64))
    local = points[ordered_nodes[node_begin:node_end]]
    offsets = np.empty(len(batch_counts), dtype=np.int64)
    offsets[0] = 0
    if len(batch_counts) > 1:
      np.cumsum(batch_counts[:-1], out=offsets[1:])
    sums = np.add.reduceat(local, offsets, axis=0, dtype=np.float64)
    centroids[cell_begin:cell_end] = np.asarray(
        sums / batch_counts[:, np.newaxis], dtype=np.float32
    )
    node_begin = node_end
  return centroids


def radial_group_member_order(
    points: np.ndarray,
    ordered_nodes: np.ndarray,
    populations: np.ndarray,
    batch_cells: int = 4096,
) -> np.ndarray:
  """Stably order each Cell by raw distance to its float64 centroid in bounded batches."""
  if batch_cells <= 0 or len(populations) == 0:
    raise ProfileError("Radial-order batch or population count is invalid")
  counts = np.asarray(populations, dtype=np.int64)
  if np.any(counts <= 0) or int(np.sum(counts, dtype=np.int64)) != len(ordered_nodes):
    raise ProfileError("Radial-order populations do not cover the ordered nodes")
  output = np.empty_like(ordered_nodes)
  node_begin = 0
  cell_begin = 0
  while cell_begin < len(counts):
    population = int(counts[cell_begin])
    run_end = cell_begin + 1
    while run_end < len(counts) and counts[run_end] == population:
      run_end += 1
    for batch_begin in range(cell_begin, run_end, batch_cells):
      batch_end = min(run_end, batch_begin + batch_cells)
      node_end = node_begin + (batch_end - batch_begin) * population
      nodes = np.asarray(ordered_nodes[node_begin:node_end]).reshape(-1, population)
      vectors = points[nodes]
      centroids = np.mean(vectors, axis=1, dtype=np.float64)
      differences = vectors - centroids[:, np.newaxis, :]
      distances = np.sum(differences * differences, axis=2, dtype=np.float64)
      order = np.argsort(distances, axis=1, kind="stable")
      output[node_begin:node_end] = np.take_along_axis(nodes, order, axis=1).reshape(-1)
      node_begin = node_end
    cell_begin = run_end
  return output


def release_memmap_pages(array: np.ndarray) -> None:
  """Release construction-only mapped pages without changing the backing artifact."""
  mapping = getattr(array, "_mmap", None)
  if mapping is None or not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
    return
  try:
    mapping.madvise(mmap.MADV_DONTNEED)
  except (BufferError, OSError, ValueError):
    pass


def write_replay_queries_binary(
    queries, output_path: Path, expected_split: str, point_count: int
) -> None:
  """Write an already-validated replay-query stream in the canonical binary format."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_path.with_suffix(output_path.suffix + ".tmp")
  digest = hashlib.sha256()
  record_count = 0
  query_count = 0
  with temporary.open("wb") as output:
    output.write(b"\0" * TRACE_BINARY_HEADER.size)
    buffer = bytearray()
    for query in queries:
      query_count += 1
      nodes = sorted(query.accessed | query.ground_truth | query.already_seen)
      for node_id in nodes:
        bits = (int(node_id in query.accessed) | (int(node_id in query.ground_truth) << 1) |
                (int(node_id in query.already_seen) << 2))
        buffer.extend(TRACE_BINARY_RECORD.pack(query.query_id, node_id, bits))
        record_count += 1
        if len(buffer) >= 8 * 1024 * 1024:
          output.write(buffer)
          digest.update(buffer)
          buffer.clear()
    if buffer:
      output.write(buffer)
      digest.update(buffer)
    output.seek(0)
    output.write(
        TRACE_BINARY_HEADER.pack(
            TRACE_BINARY_MAGIC,
            TRACE_BINARY_VERSION,
            TRACE_BINARY_HEADER.size,
            TRACE_BINARY_RECORD.size,
            TRACE_SPLITS[expected_split],
            BINARY_ENDIAN_MARKER,
            point_count,
            query_count,
            record_count,
            digest.digest(),
        )
    )
  open_binary_replay_trace(temporary, expected_split, point_count)
  temporary.replace(output_path)


def write_trace_binary(
    source_path: Path, output_path: Path, expected_split: str, point_count: int
) -> None:
  write_replay_queries_binary(
      iter_csv_replay_trace(source_path, expected_split, point_count),
      output_path,
      expected_split,
      point_count,
  )


def smallest_capacity_class(population: int) -> int:
  """Return the smallest explicit capacity class that can hold a nonempty Cell."""
  if population <= 0:
    raise ProfileError("Adaptive Cell population must be positive")
  for capacity in ADAPTIVE_CAPACITY_CLASSES:
    if population <= capacity:
      return capacity
  raise ProfileError("Adaptive Cell population exceeds the Cell-128 safety bound")


def assign_fixed_leaf_capacity_classes(
    root: int,
    hierarchy: Sequence[HierarchyNode],
    groups: Sequence[np.ndarray],
) -> list[dict[str, int]]:
  """Label frozen fixed-128 leaves without changing membership, order, or topology."""
  rows: list[dict[str, int] | None] = [None] * len(groups)
  parents = [-1] * len(hierarchy)

  def visit(node_id: int, depth: int) -> int:
    node = hierarchy[node_id]
    if node.children_are_cells:
      population = 0
      for cell_id in node.children:
        if cell_id >= len(groups) or rows[cell_id] is not None:
          raise ProfileError("Fixed hierarchy has an invalid or repeated Cell leaf")
        actual = len(groups[cell_id])
        rows[cell_id] = {
            "cell_id": cell_id,
            "capacity_class": smallest_capacity_class(actual),
            "actual_population": actual,
            "tree_node_id": node_id,
            "tree_depth": depth,
        }
        population += actual
      return population
    population = 0
    for child in node.children:
      if child >= len(hierarchy) or parents[child] != -1:
        raise ProfileError("Fixed hierarchy has invalid or non-unique parentage")
      parents[child] = node_id
      population += visit(child, depth + 1)
    return population

  total = visit(root, 0)
  if total != sum(len(group) for group in groups) or any(row is None for row in rows):
    raise ProfileError("Fixed hierarchy and assignment coverage disagree")
  return [row for row in rows if row is not None]


def normalize_active_cell_leaves(
    hierarchy: list[HierarchyNode], groups: Sequence[np.ndarray]
) -> bool:
  """Give every active Cell a unique routing leaf while retaining the existing ancestors."""
  changed = False
  for node in list(hierarchy):
    if not node.children_are_cells or len(node.children) <= 1:
      continue
    cell_children = list(node.children)
    node.children = []
    node.children_are_cells = False
    for cell_id in cell_children:
      if cell_id >= len(groups):
        raise ProfileError("Hierarchy leaf normalization found an invalid Cell ID")
      leaf_id = len(hierarchy)
      hierarchy.append(
          HierarchyNode(
              node.centroid.copy(),
              [cell_id],
              True,
              len(groups[cell_id]),
              0.0,
          )
      )
      node.children.append(leaf_id)
    changed = True
  return changed


def build_fixed_replay_tree(
    root: int,
    hierarchy: Sequence[HierarchyNode],
    groups: Sequence[np.ndarray],
) -> tuple[int, list[AdaptiveTreeNode], list[int]]:
  """Expose frozen Cells as replay leaves while preserving the persisted routing topology."""
  tree: list[AdaptiveTreeNode] = []
  leaves: list[int] = []

  def append_source(source_node_id: int) -> int:
    source = hierarchy[source_node_id]
    output_id = len(tree)
    tree.append(
        AdaptiveTreeNode(
            output_id,
            np.empty(0, dtype=np.int32),
            [],
        )
    )
    if source.children_are_cells:
      children = []
      for cell_id in source.children:
        members = groups[cell_id]
        leaf_id = len(tree)
        tree.append(
            AdaptiveTreeNode(
                leaf_id,
                members,
                [],
            )
        )
        children.append(leaf_id)
        leaves.append(leaf_id)
    else:
      children = [append_source(child) for child in source.children]
    tree[output_id].children = children
    return output_id

  replay_root = append_source(root)
  if len(leaves) != len(groups):
    raise ProfileError("Fixed replay tree does not expose every source Cell exactly once")
  return replay_root, tree, leaves


def hierarchy_parent_depth_and_cell_leaves(
    root: int, hierarchy: Sequence[HierarchyNode], cell_count: int
) -> tuple[list[int], list[int], list[int]]:
  parents = [-1] * len(hierarchy)
  depths = [-1] * len(hierarchy)
  cell_leaves = [-1] * cell_count

  def visit(node_id: int, depth: int) -> None:
    if depths[node_id] != -1:
      raise ProfileError("Hierarchy contains a cycle or repeated node")
    depths[node_id] = depth
    node = hierarchy[node_id]
    if node.children_are_cells:
      for cell_id in node.children:
        if cell_id >= cell_count or cell_leaves[cell_id] != -1:
          raise ProfileError("Hierarchy contains an invalid or repeated Cell")
        cell_leaves[cell_id] = node_id
      return
    for child in node.children:
      if child >= len(hierarchy) or parents[child] != -1:
        raise ProfileError("Hierarchy contains invalid or non-unique parentage")
      parents[child] = node_id
      visit(child, depth + 1)

  visit(root, 0)
  if any(depth < 0 for depth in depths) or any(leaf < 0 for leaf in cell_leaves):
    raise ProfileError("Hierarchy is disconnected or has incomplete Cell coverage")
  return parents, depths, cell_leaves


def hierarchy_lca(left: int, right: int, parents: Sequence[int], depths: Sequence[int]) -> int:
  while depths[left] > depths[right]:
    left = parents[left]
  while depths[right] > depths[left]:
    right = parents[right]
  while left != right:
    left = parents[left]
    right = parents[right]
  return left


def build_trace_cell_index(
    queries: Sequence[ReplayQuery],
    groups: Sequence[np.ndarray],
    point_count: int,
    node_to_cell: np.ndarray | None = None,
) -> tuple[list[set[int]], list[Counter[int]], list[Counter[int]], np.ndarray]:
  if node_to_cell is None:
    populations = np.fromiter((len(group) for group in groups), dtype=np.int64)
    ordered_nodes = np.concatenate(groups).astype(np.int64, copy=False)
    if (
        len(ordered_nodes) != point_count
        or np.any(ordered_nodes < 0)
        or np.any(ordered_nodes >= point_count)
        or np.any(np.bincount(ordered_nodes, minlength=point_count) != 1)
    ):
      raise ProfileError("Cell groups do not provide one-owner complete vector coverage")
    node_to_cell = np.empty(point_count, dtype=np.int32)
    node_to_cell[ordered_nodes] = np.repeat(
        np.arange(len(groups), dtype=np.int32), populations
    )
  elif node_to_cell.shape != (point_count,) or np.any(node_to_cell < 0):
    raise ProfileError("Reused Cell ownership index is invalid")
  visits = [set() for _ in groups]
  seen = [Counter() for _ in groups]
  ground_truth = [Counter() for _ in groups]
  for query_position, query in enumerate(queries):
    for node_id in query.accessed:
      visits[int(node_to_cell[node_id])].add(query_position)
    for node_id in query.already_seen:
      seen[int(node_to_cell[node_id])][query_position] += 1
    for node_id in query.ground_truth:
      ground_truth[int(node_to_cell[node_id])][query_position] += 1
  return visits, seen, ground_truth, node_to_cell


def replay_cell_pair(
    left: int,
    right: int,
    groups: Sequence[np.ndarray],
    visits: Sequence[set[int]],
    seen: Sequence[Counter[int]],
    ground_truth: Sequence[Counter[int]],
    policy: AdaptivePolicy,
) -> dict[str, float | int]:
  reached = visits[left] | visits[right]
  joint = visits[left] & visits[right]
  left_pages = math.ceil(len(groups[left]) * policy.vector_bytes / policy.page_bytes)
  right_pages = math.ceil(len(groups[right]) * policy.vector_bytes / policy.page_bytes)
  merged_population = len(groups[left]) + len(groups[right])
  merged_region_pages = math.ceil(
      merged_population * policy.vector_bytes / policy.page_bytes
  )
  fixed_pages = 0
  fixed_runs = 0
  fixed_unseen = 0
  merged_unseen = 0
  fixed_gt = 0
  merged_gt = 0
  for query_position in reached:
    left_selected = query_position in visits[left]
    right_selected = query_position in visits[right]
    if left_selected:
      fixed_pages += left_pages
      fixed_runs += 1
      fixed_unseen += len(groups[left]) - seen[left][query_position]
      fixed_gt += ground_truth[left][query_position]
    if right_selected:
      fixed_pages += right_pages
      fixed_runs += 1
      fixed_unseen += len(groups[right]) - seen[right][query_position]
      fixed_gt += ground_truth[right][query_position]
    merged_unseen += (
        merged_population - seen[left][query_position] - seen[right][query_position]
    )
    merged_gt += ground_truth[left][query_position] + ground_truth[right][query_position]
  denominator = max(1, len(reached))
  fixed_unseen_denominator = max(1, fixed_unseen)
  return {
      "parent_visits": len(reached),
      "joint_visits": len(joint),
      "cross_child_coaccess": 0.0 if not reached else len(joint) / len(reached),
      "fixed_pages": fixed_pages / denominator,
      "merged_pages": 0.0 if not reached else float(merged_region_pages),
      "page_delta": (
          0.0 if not reached else merged_region_pages - fixed_pages / denominator
      ),
      "fixed_runs": fixed_runs / denominator,
      "merged_runs": 0.0 if not reached else 1.0,
      "unseen_delta": (merged_unseen - fixed_unseen) / denominator,
      "unseen_ratio": (merged_unseen - fixed_unseen) / fixed_unseen_denominator,
      "gt_coverage": 1.0 if merged_gt == 0 else merged_gt / merged_gt,
      "fixed_gt_coverage": 1.0 if merged_gt == 0 else fixed_gt / merged_gt,
  }


def train_pair_coaccess_merges(
    points: np.ndarray,
    root: int,
    hierarchy: Sequence[HierarchyNode],
    groups: Sequence[np.ndarray],
    training_queries: Sequence[ReplayQuery],
    heldout_queries: Sequence[ReplayQuery],
    policy: AdaptivePolicy,
    minimum_pair_support: int,
    minimum_lca_depth: int,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
  """Select deterministic, disjoint co-access pairs without repartitioning any source Cell."""
  if minimum_pair_support <= 0 or minimum_lca_depth < 0:
    raise ProfileError("Adaptive pair support and LCA depth are invalid")
  parents, depths, cell_leaves = hierarchy_parent_depth_and_cell_leaves(
      root, hierarchy, len(groups)
  )
  train_visits, train_seen, train_gt, node_to_cell = build_trace_cell_index(
      training_queries, groups, len(points)
  )
  heldout_visits, heldout_seen, heldout_gt, heldout_node_to_cell = build_trace_cell_index(
      heldout_queries, groups, len(points), node_to_cell
  )
  if not np.array_equal(node_to_cell, heldout_node_to_cell):
    raise ProfileError("Training and held-out replay disagree on Cell ownership")
  joint_counts: Counter[tuple[int, int]] = Counter()
  for query in training_queries:
    touched = sorted(
        {
            int(node_to_cell[node_id])
            for node_id in query.accessed
            if len(groups[int(node_to_cell[node_id])]) <= ADAPTIVE_CAPACITY_CLASSES[-2]
        }
    )
    for left_position, left in enumerate(touched):
      for right in touched[left_position + 1 :]:
        left_leaf = cell_leaves[left]
        right_leaf = cell_leaves[right]
        same_leaf_group = left_leaf == right_leaf
        sibling_leaf_groups = (
            left_leaf != right_leaf
            and parents[left_leaf] != -1
            and parents[left_leaf] == parents[right_leaf]
            and len(hierarchy[left_leaf].children) == 1
            and len(hierarchy[right_leaf].children) == 1
        )
        if not same_leaf_group and not sibling_leaf_groups:
          continue
        if len(groups[left]) + len(groups[right]) <= ADAPTIVE_CAPACITY_CLASSES[-1]:
          joint_counts[(left, right)] += 1

  geometry: dict[int, tuple[np.ndarray, float, float, float]] = {}

  def cell_geometry(cell_id: int) -> tuple[np.ndarray, float, float, float]:
    if cell_id not in geometry:
      local = np.asarray(points[groups[cell_id]], dtype=np.float64)
      centroid = np.asarray(np.mean(local, axis=0, dtype=np.float64), dtype=np.float32)
      squared = np.einsum(
          "ij,ij->i", local - centroid.astype(np.float64), local - centroid.astype(np.float64)
      )
      geometry[cell_id] = (
          centroid,
          float(np.sqrt(np.mean(squared, dtype=np.float64))),
          nearest_rank_p95(np.sqrt(squared)),
          float(np.sum(squared, dtype=np.float64)),
      )
    return geometry[cell_id]

  candidates = []
  for left, right in sorted(joint_counts):
    population = len(groups[left]) + len(groups[right])
    capacity = smallest_capacity_class(population)
    parent_id = hierarchy_lca(cell_leaves[left], cell_leaves[right], parents, depths)
    if depths[parent_id] < minimum_lca_depth:
      continue
    left_geometry = cell_geometry(left)
    right_geometry = cell_geometry(right)
    members = np.concatenate((groups[left], groups[right]))
    local = np.asarray(points[members], dtype=np.float64)
    centroid = np.asarray(np.mean(local, axis=0, dtype=np.float64), dtype=np.float32)
    squared = np.einsum(
        "ij,ij->i", local - centroid.astype(np.float64), local - centroid.astype(np.float64)
    )
    parent_sse = float(np.sum(squared, dtype=np.float64))
    child_sse = left_geometry[3] + right_geometry[3]
    split_gain = 0.0 if parent_sse == 0.0 else (parent_sse - child_sse) / parent_sse
    centroid_distance = float(
        np.linalg.norm(
            left_geometry[0].astype(np.float64) - right_geometry[0].astype(np.float64)
        )
    )
    radius_sum = left_geometry[2] + right_geometry[2]
    overlap = (
        1.0
        if radius_sum == 0.0
        else max(0.0, min(1.0, (radius_sum - centroid_distance) / radius_sum))
    )
    candidates.append(
        {
            "left": left,
            "right": right,
            "parent_id": parent_id,
            "depth": depths[parent_id],
            "population": population,
            "capacity": capacity,
            "rms_radius": float(np.sqrt(np.mean(squared, dtype=np.float64))),
            "p95_radius": nearest_rank_p95(np.sqrt(squared)),
            "parent_sse": parent_sse,
            "child_sse": child_sse,
            "split_gain": split_gain,
            "overlap": overlap,
            "training": replay_cell_pair(
                left, right, groups, train_visits, train_seen, train_gt, policy
            ),
            "heldout": replay_cell_pair(
                left, right, groups, heldout_visits, heldout_seen, heldout_gt, policy
            ),
        }
    )

  cohorts: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
  for candidate in candidates:
    cohorts[(int(candidate["depth"]), int(candidate["capacity"]))].append(candidate)
  cohort_medians = {
      cohort: (
          float(np.median([float(row["rms_radius"]) for row in rows])),
          float(np.median([float(row["p95_radius"]) for row in rows])),
      )
      for cohort, rows in cohorts.items()
  }

  eligible = []
  for candidate in candidates:
    rms_median, p95_median = cohort_medians[
        (int(candidate["depth"]), int(candidate["capacity"]))
    ]
    candidate["normalized_rms"] = float(candidate["rms_radius"]) / max(
        rms_median, 1.0e-12
    )
    candidate["normalized_p95"] = float(candidate["p95_radius"]) / max(
        p95_median, 1.0e-12
    )
    training = candidate["training"]
    assert isinstance(training, dict)
    checks = (
        (int(training["joint_visits"]) >= minimum_pair_support, "train_pair_support"),
        (
            float(candidate["normalized_rms"]) <= policy.maximum_normalized_rms_radius
            or float(candidate["normalized_p95"]) <= policy.maximum_normalized_p95_radius,
            "compact_region",
        ),
        (
            float(candidate["split_gain"]) <= policy.maximum_split_gain
            or float(candidate["overlap"]) >= policy.minimum_centroid_radius_overlap,
            "weak_split",
        ),
        (
            float(training["cross_child_coaccess"])
            >= policy.minimum_train_cross_child_coaccess,
            "train_cross_child_coaccess",
        ),
        (
            float(training["page_delta"]) <= 1.0e-12
            and float(training["merged_runs"]) < float(training["fixed_runs"]),
            "page_economics",
        ),
        (
            float(training["unseen_ratio"]) <= policy.maximum_unseen_pq_increase,
            "scoring_overhead",
        ),
    )
    candidate["rejection"] = next(
        (reason for passed, reason in checks if not passed), ""
    )
    if candidate["rejection"] == "":
      eligible.append(candidate)

  selected_cells: set[int] = set()
  selected_pairs = []
  for candidate in sorted(
      eligible,
      key=lambda row: (
          -float(row["training"]["cross_child_coaccess"]),
          float(row["training"]["page_delta"]),
          -int(row["training"]["joint_visits"]),
          int(row["parent_id"]),
          int(row["left"]),
          int(row["right"]),
      ),
  ):
    left = int(candidate["left"])
    right = int(candidate["right"])
    if left in selected_cells or right in selected_cells:
      candidate["rejection"] = "cell_already_merged"
      continue
    selected_cells.update((left, right))
    selected_pairs.append((left, right))

  audit = []
  for candidate in candidates:
    training = candidate["training"]
    heldout = candidate["heldout"]
    assert isinstance(training, dict) and isinstance(heldout, dict)
    rejection = str(candidate["rejection"])
    audit.append(
        {
            "parent_id": candidate["parent_id"],
            "child_ids": f"{candidate['left']};{candidate['right']}",
            "child_count": 2,
            "actual_population": candidate["population"],
            "proposed_capacity_class": candidate["capacity"],
            "tree_depth": candidate["depth"],
            "normalized_rms_radius": candidate["normalized_rms"],
            "p95_radius": candidate["p95_radius"],
            "parent_sse": candidate["parent_sse"],
            "child_sse": candidate["child_sse"],
            "split_gain": candidate["split_gain"],
            "centroid_radius_overlap": candidate["overlap"],
            "train_cross_child_coaccess": training["cross_child_coaccess"],
            "train_parent_visits": training["parent_visits"],
            "train_joint_visits": training["joint_visits"],
            "heldout_gt_coverage": heldout["gt_coverage"],
            "fixed_child_pages": heldout["fixed_pages"],
            "merged_parent_pages": heldout["merged_pages"],
            "page_delta": heldout["page_delta"],
            "unseen_pq_delta": heldout["unseen_delta"],
            "decision": "accept" if rejection == "" else "reject",
            "rejection_reason": rejection,
        }
    )
  audit.sort(
      key=lambda row: (
          int(row["tree_depth"]),
          int(row["parent_id"]),
          str(row["child_ids"]),
      )
  )
  return sorted(selected_pairs), audit


def materialize_pair_merges(
    root: int,
    hierarchy: Sequence[HierarchyNode],
    groups: Sequence[np.ndarray],
    selected_pairs: Sequence[tuple[int, int]],
) -> tuple[list[np.ndarray], list[HierarchyNode], int]:
  """Merge disjoint Cell pairs and splice one new leaf at each pair's source-tree LCA."""
  if not selected_pairs:
    # Preserve the legacy pre-order materialization without building the LCA,
    # placement, and vector-owner tables when the frozen policy accepts no
    # merge.  This is the common production path after a failed structural
    # gate, and the emitted hierarchy is byte-identical to the general path.
    output_hierarchy: list[HierarchyNode] = []

    def append_unmerged(source_node_id: int) -> int:
      source = hierarchy[source_node_id]
      output_id = len(output_hierarchy)
      output_hierarchy.append(
          HierarchyNode(
              source.centroid,
              list(source.children) if source.children_are_cells else [],
              source.children_are_cells,
              source.descendant_node_count,
              source.radius,
          )
      )
      if not source.children_are_cells:
        output_hierarchy[output_id].children = [
            append_unmerged(child) for child in source.children
        ]
      return output_id

    output_root = append_unmerged(root)
    return list(groups), output_hierarchy, output_root

  parents, depths, cell_leaves = hierarchy_parent_depth_and_cell_leaves(
      root, hierarchy, len(groups)
  )
  partner: dict[int, int] = {}
  for left, right in selected_pairs:
    if left == right or left in partner or right in partner:
      raise ProfileError("Selected Cell merge pairs must be disjoint")
    if len(groups[left]) + len(groups[right]) > ADAPTIVE_CAPACITY_CLASSES[-1]:
      raise ProfileError("Selected Cell merge exceeds the Cell-128 safety bound")
    partner[left] = right
    partner[right] = left

  units: list[tuple[tuple[int, ...], np.ndarray, int]] = []
  consumed: set[int] = set()
  for cell_id in range(len(groups)):
    if cell_id in consumed:
      continue
    if cell_id in partner:
      other = partner[cell_id]
      pair = tuple(sorted((cell_id, other)))
      consumed.update(pair)
      placement = hierarchy_lca(
          cell_leaves[pair[0]], cell_leaves[pair[1]], parents, depths
      )
      members = np.concatenate((groups[pair[0]], groups[pair[1]]))
      units.append((pair, members, placement))
    else:
      consumed.add(cell_id)
      units.append(((cell_id,), groups[cell_id], cell_leaves[cell_id]))
  units.sort(key=lambda unit: unit[0])
  output_groups = [unit[1] for unit in units]
  placements: dict[int, list[int]] = defaultdict(list)
  for output_cell, (_, _, placement) in enumerate(units):
    placements[placement].append(output_cell)

  output_hierarchy: list[HierarchyNode] = []

  def append_source(source_node_id: int) -> int | None:
    source = hierarchy[source_node_id]
    output_id = len(output_hierarchy)
    output_hierarchy.append(
        HierarchyNode(source.centroid, [], False, 0, source.radius)
    )
    child_outputs = []
    if not source.children_are_cells:
      for child in source.children:
        output_child = append_source(child)
        if output_child is not None:
          child_outputs.append(output_child)
    direct_cells = sorted(placements.get(source_node_id, ()))
    if not child_outputs and not direct_cells:
      output_hierarchy.pop()
      return None
    if not child_outputs:
      output_hierarchy[output_id].children = direct_cells
      output_hierarchy[output_id].children_are_cells = True
      output_hierarchy[output_id].descendant_node_count = sum(
          len(output_groups[cell_id]) for cell_id in direct_cells
      )
      return output_id
    if direct_cells:
      leaf_id = len(output_hierarchy)
      output_hierarchy.append(
          HierarchyNode(
              source.centroid,
              direct_cells,
              True,
              sum(len(output_groups[cell_id]) for cell_id in direct_cells),
              0.0,
          )
      )
      child_outputs.append(leaf_id)
    output_hierarchy[output_id].children = child_outputs
    output_hierarchy[output_id].descendant_node_count = sum(
        output_hierarchy[child].descendant_node_count for child in child_outputs
    )
    return output_id

  output_root = append_source(root)
  if output_root is None:
    raise ProfileError("Pair merge removed the hierarchy root")
  owners = np.full(sum(len(group) for group in groups), -1, dtype=np.int32)
  for cell_id, members in enumerate(output_groups):
    if np.any(owners[members] != -1):
      raise ProfileError("Pair merge duplicates a vector owner")
    owners[members] = cell_id
  if np.any(owners < 0):
    raise ProfileError("Pair merge loses vector coverage")
  return output_groups, output_hierarchy, output_root


def _request_pages(
    cache: OrderedDict[tuple[int, int], None],
    region_id: int,
    page_count: int,
    capacity: int,
) -> tuple[int, int]:
  hits = 0
  misses = 0
  for page in range(page_count):
    key = (region_id, page)
    if capacity != 0 and key in cache:
      cache.move_to_end(key)
      hits += 1
      continue
    misses += 1
    if capacity != 0:
      cache[key] = None
      while len(cache) > capacity:
        cache.popitem(last=False)
  return hits, misses


def evaluate_frontier(
    frontier: Sequence[int],
    tree: Sequence[AdaptiveTreeNode],
    queries: Sequence[ReplayQuery],
    policy: AdaptivePolicy,
    point_count: int,
) -> dict[str, float | int]:
  parents = [-1] * len(tree)
  for node in tree:
    for child in node.children:
      if parents[child] != -1:
        raise ProfileError("Adaptive replay tree has non-unique parentage")
      parents[child] = node.node_id
  populations = np.fromiter(
      (len(tree[node_id].members) for node_id in frontier), dtype=np.int64
  )
  ordered_nodes = np.concatenate([tree[node_id].members for node_id in frontier]).astype(
      np.int64, copy=False
  )
  if (
      len(ordered_nodes) != point_count
      or np.any(ordered_nodes < 0)
      or np.any(ordered_nodes >= point_count)
      or np.any(np.bincount(ordered_nodes, minlength=point_count) != 1)
  ):
    raise ProfileError("Adaptive replay frontier duplicates or loses a vector owner")
  owner = np.empty(point_count, dtype=np.int32)
  owner[ordered_nodes] = np.repeat(np.asarray(frontier, dtype=np.int32), populations)
  if np.any(owner < 0):
    raise ProfileError("Adaptive replay frontier does not cover every vector")
  cache: OrderedDict[tuple[int, int], None] = OrderedDict()
  hits = 0
  misses = 0
  evictions = 0
  page_requests = 0
  padding_bytes_read = 0
  unseen = 0
  gt_covered = 0
  gt_total = 0
  runs = 0
  hierarchy_nodes_scored = 0
  for query in queries:
    selected = sorted({int(owner[node]) for node in query.accessed})
    routed_nodes: set[int] = set()
    for node_id in selected:
      current = node_id
      while current != -1:
        routed_nodes.add(current)
        current = parents[current]
    hierarchy_nodes_scored += len(routed_nodes)
    for node_id in selected:
      node = tree[node_id]
      pages = math.ceil(len(node.members) * policy.vector_bytes / policy.page_bytes)
      page_requests += pages
      padding_bytes_read += pages * policy.page_bytes - len(node.members) * policy.vector_bytes
      before = len(cache)
      region_hits, region_misses = _request_pages(cache, node_id, pages, policy.lru_pages)
      hits += region_hits
      misses += region_misses
      if policy.lru_pages != 0:
        evictions += max(0, before + region_misses - policy.lru_pages)
      unseen += len(node.members) - sum(
          int(member) in query.already_seen for member in node.members
      )
    runs += len(selected)
    gt_total += len(query.ground_truth)
    selected_set = set(selected)
    gt_covered += sum(int(owner[node]) in selected_set for node in query.ground_truth)
  query_count = len(queries)
  artifact_padding_bytes = sum(
      math.ceil(len(tree[node_id].members) * policy.vector_bytes / policy.page_bytes)
      * policy.page_bytes
      - len(tree[node_id].members) * policy.vector_bytes
      for node_id in frontier
  )
  return {
      "cell_count": len(frontier),
      "physical_pages_per_query": misses / query_count,
      "logical_page_requests_per_query": page_requests / query_count,
      "unseen_pq_records_per_query": unseen / query_count,
      "raw_vector_comparisons_per_query": 0.0,
      "gt_coverage": 1.0 if gt_total == 0 else gt_covered / gt_total,
      "sequential_runs_per_query": runs / query_count,
      "hierarchy_nodes_scored_per_query": hierarchy_nodes_scored / query_count,
      "leaf_groups_probed_per_query": runs / query_count,
      "mean_run_pages": 0.0 if runs == 0 else page_requests / runs,
      "lru_hits": hits,
      "lru_misses": misses,
      "lru_evictions": evictions,
      "padding_bytes_per_query": padding_bytes_read / query_count,
      "artifact_padding_bytes": artifact_padding_bytes,
  }


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  with temporary.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  temporary.replace(path)


def write_json_atomic(path: Path, value: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(
      json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  temporary.replace(path)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sidecar", type=Path)
  parser.add_argument(
      "--single_community",
      action="store_true",
      help="Use one deterministic global Community without reading a bootstrap sidecar.",
  )
  parser.add_argument("--base_file", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--hierarchy_output", type=Path)
  parser.add_argument("--target_size", type=int, default=256)
  parser.add_argument("--projection_dimension", type=int, default=24)
  parser.add_argument("--identity_projection", action="store_true")
  parser.add_argument("--global_cells", action="store_true")
  parser.add_argument("--pq_codes", type=Path)
  parser.add_argument("--pq_pivots", type=Path)
  parser.add_argument("--pq_locality_profile", type=Path)
  parser.add_argument(
      "--adaptive_pq_locality_profile",
      type=Path,
      help="Write the selected adaptive leaves as a checksummed fast-repack profile.",
  )
  parser.add_argument("--pq_locality_dimensions", type=int, default=32)
  parser.add_argument("--pq_centroid_samples", type=int, default=8)
  parser.add_argument("--pq_hierarchy_branching", type=int, default=16)
  parser.add_argument("--iterations", type=int, default=6)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--adaptive_multi_capacity", action="store_true")
  parser.add_argument(
      "--adaptive_initial_cell_size",
      type=int,
      default=16,
      help="Maximum population of deterministic in-memory PQ bootstrap leaves.",
  )
  parser.add_argument(
      "--adaptive_radial_member_order",
      action="store_true",
      help="Stably order nodes inside each selected Cell by raw centroid distance.",
  )
  parser.add_argument(
      "--adaptive_compact_source_leaves",
      action="store_true",
      help="Bootstrap routing leaves over branching-sized Cell groups before normalization.",
  )
  parser.add_argument("--enable_coaccess_merge", action="store_true")
  parser.add_argument("--minimum_pair_support", type=int, default=20)
  parser.add_argument("--minimum_merge_lca_depth", type=int, default=2)
  parser.add_argument("--source_assignment", type=Path)
  parser.add_argument("--source_hierarchy", type=Path)
  parser.add_argument("--training_trace", type=Path)
  parser.add_argument("--heldout_trace", type=Path)
  parser.add_argument("--capacity_output", type=Path)
  parser.add_argument("--audit_output", type=Path)
  parser.add_argument("--summary_output", type=Path)
  parser.add_argument("--page_bytes", type=int, default=4096)
  parser.add_argument("--vector_bytes", type=int)
  parser.add_argument("--lru_pages", type=int, default=0)
  parser.add_argument("--maximum_normalized_rms_radius", type=float, default=1.25)
  parser.add_argument("--maximum_normalized_p95_radius", type=float, default=1.25)
  parser.add_argument("--maximum_split_gain", type=float, default=0.15)
  parser.add_argument("--minimum_centroid_radius_overlap", type=float, default=0.50)
  parser.add_argument("--minimum_train_cross_child_coaccess", type=float, default=0.50)
  parser.add_argument("--maximum_unseen_pq_increase", type=float, default=0.10)
  args = parser.parse_args()
  if args.target_size <= 0 or args.projection_dimension <= 0 or args.iterations <= 0:
    raise ProfileError("Cell training parameters must be positive")
  if args.single_community == (args.sidecar is not None):
    raise ProfileError("Specify exactly one of --sidecar and --single_community")
  pq_locality_paths = (args.pq_codes, args.pq_pivots, args.pq_locality_profile)
  pq_locality = any(path is not None for path in pq_locality_paths) and not (
      args.adaptive_multi_capacity and args.pq_locality_profile is None
  )
  if pq_locality and (
      any(path is None for path in pq_locality_paths)
      or not args.global_cells
      or not args.identity_projection
      or args.hierarchy_output is None
      or args.adaptive_multi_capacity
      or not 1 <= args.pq_locality_dimensions <= 32
      or args.pq_centroid_samples <= 0
      or args.pq_hierarchy_branching < 2
  ):
    raise ProfileError(
        "PQ-locality training requires global identity Cells, assignment/hierarchy/profile "
        "outputs, and complete positive PQ-locality parameters"
    )
  if args.hierarchy_output is not None and (
      not args.global_cells or not args.identity_projection
  ):
    raise ProfileError("A persisted hierarchy requires global Cells and identity projection")
  adaptive_paths = (
      args.training_trace,
      args.heldout_trace,
      args.capacity_output,
      args.audit_output,
      args.summary_output,
      args.hierarchy_output,
  )
  if args.adaptive_multi_capacity:
    frozen_source = args.source_assignment is not None or args.source_hierarchy is not None
    pq_bootstrap = args.pq_codes is not None or args.pq_pivots is not None
    if not args.global_cells or not args.identity_projection or any(
        path is None for path in adaptive_paths
    ):
      raise ProfileError(
          "Adaptive multi-capacity training requires global identity-projected Cells and all "
          "trace/capacity/audit/summary outputs"
      )
    if frozen_source == pq_bootstrap or (
        frozen_source
        and (args.source_assignment is None or args.source_hierarchy is None)
    ) or (
        pq_bootstrap and (args.pq_codes is None or args.pq_pivots is None)
    ):
      raise ProfileError(
          "Adaptive training requires exactly one complete frozen source or PQ bootstrap"
      )
    if args.target_size != ADAPTIVE_CAPACITY_CLASSES[-1]:
      raise ProfileError("Adaptive multi-capacity cell_target_size must be exactly 128")
    if (
        not 1 <= args.adaptive_initial_cell_size <= ADAPTIVE_CAPACITY_CLASSES[0]
        or not 1 <= args.pq_locality_dimensions <= 32
        or args.pq_centroid_samples <= 0
        or args.pq_hierarchy_branching < 2
    ):
      raise ProfileError("Adaptive PQ bootstrap parameters are invalid")
    if args.training_trace.resolve() == args.heldout_trace.resolve():
      raise ProfileError("Adaptive training and held-out traces must be different files")
    numeric_policy = (
        args.maximum_normalized_rms_radius,
        args.maximum_normalized_p95_radius,
        args.maximum_split_gain,
        args.minimum_centroid_radius_overlap,
        args.minimum_train_cross_child_coaccess,
        args.maximum_unseen_pq_increase,
    )
    if (
        any(not math.isfinite(value) or value < 0.0 for value in numeric_policy)
        or args.page_bytes <= 0
        or args.lru_pages < 0
        or args.minimum_pair_support <= 0
        or args.minimum_merge_lca_depth < 0
        or (args.vector_bytes is not None and args.vector_bytes <= 0)
    ):
      raise ProfileError("Adaptive policy thresholds and replay parameters are invalid")
  elif args.adaptive_pq_locality_profile is not None:
    raise ProfileError(
        "An adaptive PQ-locality profile requires --adaptive_multi_capacity"
    )

  base = read_fbin(args.base_file)
  if args.single_community:
    communities = np.zeros(len(base), dtype=np.uint32)
  elif pq_locality:
    communities = map_u32_membership(args.sidecar, NODE_TO_COMMUNITY_SECTION)
  else:
    communities, _ = read_memberships(args.sidecar)
  if len(communities) != len(base):
    raise ProfileError("Base vectors and sidecar memberships disagree")
  if args.identity_projection:
    projected = base
  else:
    rng = np.random.default_rng(args.seed)
    projection = rng.standard_normal((base.shape[1], args.projection_dimension)).astype(
        np.float32
    ) / math.sqrt(args.projection_dimension)
    projected = np.asarray(base @ projection, dtype=np.float32)
  cells = [-1] * len(communities)
  orders = [-1] * len(communities)
  next_cell = 0
  hierarchy: list[HierarchyNode] = []
  hierarchy_root: int | None = None
  adaptive_summary: dict[str, object] | None = None
  reuse_frozen_assignment_and_hierarchy = False
  reuse_frozen_member_order = False
  precomputed_cell_centroids: np.ndarray | None = None
  hierarchy_geometry_finalized = False
  if pq_locality:
    codes = read_pq_codes(args.pq_codes)
    if len(codes) != len(base):
      raise ProfileError("Compressed-PQ and Base point counts disagree")
    pivots, pq_centroid, chunk_offsets = read_diskann_pq_model(
        args.pq_pivots, codes.shape[1]
    )
    ordered_nodes = pq_locality_order(
        codes, pivots, pq_centroid, chunk_offsets, args.pq_locality_dimensions
    )
    populations, cell_centroids = pq_reconstructed_cell_centroids(
        codes,
        ordered_nodes,
        pivots,
        pq_centroid,
        chunk_offsets,
        args.target_size,
        args.pq_centroid_samples,
    )
    hierarchy_root, hierarchy = build_contiguous_centroid_hierarchy(
        cell_centroids, populations, args.pq_hierarchy_branching
    )
    minimum_cell_size = int(np.min(populations))
    maximum_cell_size = int(np.max(populations))
    group_count = len(populations)
    write_assignment_binary_from_order(
        args.output, ordered_nodes, populations, communities
    )
    write_pq_locality_profile(
        args.pq_locality_profile,
        ordered_nodes,
        populations,
        cell_centroids,
        args.target_size,
    )
    del ordered_nodes, populations, cell_centroids
  elif args.adaptive_multi_capacity:
    training_queries = open_replay_trace(args.training_trace, "train", len(base))
    heldout_queries = open_replay_trace(args.heldout_trace, "heldout", len(base))
    vector_bytes = args.vector_bytes or base.shape[1] * np.dtype("<f4").itemsize
    policy = AdaptivePolicy(
        maximum_normalized_rms_radius=args.maximum_normalized_rms_radius,
        maximum_normalized_p95_radius=args.maximum_normalized_p95_radius,
        maximum_split_gain=args.maximum_split_gain,
        minimum_centroid_radius_overlap=args.minimum_centroid_radius_overlap,
        minimum_train_cross_child_coaccess=args.minimum_train_cross_child_coaccess,
        maximum_unseen_pq_increase=args.maximum_unseen_pq_increase,
        page_bytes=args.page_bytes,
        vector_bytes=vector_bytes,
        lru_pages=args.lru_pages,
    )
    if args.source_hierarchy is not None:
      source_root, source_dimension, source_cell_count, source_hierarchy = read_hierarchy(
          args.source_hierarchy
      )
      if source_dimension != projected.shape[1]:
        raise ProfileError("Adaptive source hierarchy dimension disagrees with Base vectors")
      source_groups = read_assignment_groups(
          args.source_assignment, len(base), source_cell_count
      )
      source_assignment_description = str(args.source_assignment.resolve())
      source_hierarchy_description = str(args.source_hierarchy.resolve())
    else:
      codes = read_pq_codes(args.pq_codes)
      if len(codes) != len(base):
        raise ProfileError("Adaptive PQ bootstrap and Base point counts disagree")
      pivots, pq_centroid, chunk_offsets = read_diskann_pq_model(
          args.pq_pivots, codes.shape[1]
      )
      source_order = pq_locality_order(
          codes, pivots, pq_centroid, chunk_offsets, args.pq_locality_dimensions
      )
      source_populations, routing_centroids = pq_reconstructed_cell_centroids(
          codes,
          source_order,
          pivots,
          pq_centroid,
          chunk_offsets,
          args.adaptive_initial_cell_size,
          args.pq_centroid_samples,
      )
      source_cell_count = len(source_populations)
      source_offsets = np.empty(source_cell_count + 1, dtype=np.uint64)
      source_offsets[0] = 0
      np.cumsum(source_populations, dtype=np.uint64, out=source_offsets[1:])
      source_centroids = compute_group_centroids(
          projected, source_order, source_populations
      )
      if args.adaptive_radial_member_order:
        source_order = radial_group_member_order(
            projected, source_order, source_populations
        )
      release_memmap_pages(projected)
      source_root, source_hierarchy = build_contiguous_centroid_hierarchy(
          routing_centroids,
          source_populations,
          args.pq_hierarchy_branching,
          source_centroids,
          args.adaptive_compact_source_leaves,
      )
      source_groups = [
          source_order[int(source_offsets[cell_id]) : int(source_offsets[cell_id + 1])]
          for cell_id in range(source_cell_count)
      ]
      precomputed_cell_centroids = source_centroids
      hierarchy_geometry_finalized = True
      source_assignment_description = "in_memory_pq_bootstrap"
      source_hierarchy_description = "in_memory_pq_bootstrap"
      del codes, pivots, pq_centroid, chunk_offsets, routing_centroids, source_populations
    if args.enable_coaccess_merge:
      selected_pairs, audit = train_pair_coaccess_merges(
          projected,
          source_root,
          source_hierarchy,
          source_groups,
          training_queries,
          heldout_queries,
          policy,
          args.minimum_pair_support,
          args.minimum_merge_lca_depth,
      )
      if selected_pairs or args.adaptive_compact_source_leaves:
        groups, hierarchy, hierarchy_root = materialize_pair_merges(
            source_root, source_hierarchy, source_groups, selected_pairs
        )
      else:
        groups = source_groups
        hierarchy = [
            HierarchyNode(
                node.centroid,
                list(node.children),
                node.children_are_cells,
                node.descendant_node_count,
                node.radius,
            )
            for node in source_hierarchy
        ]
        hierarchy_root = source_root
    else:
      selected_pairs = []
      audit = []
      groups = source_groups
      hierarchy = [
          HierarchyNode(
              node.centroid,
              list(node.children),
              node.children_are_cells,
              node.descendant_node_count,
              node.radius,
          )
          for node in source_hierarchy
      ]
      hierarchy_root = source_root
    reuse_frozen_assignment_and_hierarchy = not selected_pairs
    reuse_frozen_member_order = not selected_pairs
    if normalize_active_cell_leaves(hierarchy, groups):
      reuse_frozen_assignment_and_hierarchy = False
      hierarchy_geometry_finalized = False
    if selected_pairs:
      precomputed_cell_centroids = None
      hierarchy_geometry_finalized = False
    capacity_rows = assign_fixed_leaf_capacity_classes(
        hierarchy_root, hierarchy, groups
    )
    _, source_replay_tree, source_replay_frontier = build_fixed_replay_tree(
        source_root, source_hierarchy, source_groups
    )
    fixed128_replay = evaluate_frontier(
        source_replay_frontier,
        source_replay_tree,
        heldout_queries,
        policy,
        len(base),
    )
    if reuse_frozen_assignment_and_hierarchy:
      adaptive_replay = dict(fixed128_replay)
    else:
      _, replay_tree, replay_frontier = build_fixed_replay_tree(
          hierarchy_root, hierarchy, groups
      )
      adaptive_replay = evaluate_frontier(
          replay_frontier, replay_tree, heldout_queries, policy, len(base)
      )
    fixed_replay = {"Cell-128": fixed128_replay}
    comparator_name = "Cell-128"
    comparator_pages = float(fixed128_replay["physical_pages_per_query"])
    page_reduction = (
        0.0
        if comparator_pages == 0.0
        else 1.0
        - float(adaptive_replay["physical_pages_per_query"]) / comparator_pages
    )
    comparator_unseen = float(fixed128_replay["unseen_pq_records_per_query"])
    unseen_increase = (
        0.0
        if comparator_unseen == 0.0
        else float(adaptive_replay["unseen_pq_records_per_query"]) / comparator_unseen
        - 1.0
    )
    structural_gate = page_reduction >= 0.10 and unseen_increase <= 0.10
    rejection_counts = Counter(
        str(row["rejection_reason"]) for row in audit if row["decision"] == "reject"
    )
    capacity_histogram = Counter(row["capacity_class"] for row in capacity_rows)
    population_histogram = Counter(row["actual_population"] for row in capacity_rows)
    adaptive_summary = {
        "schema_version": 1,
        "mode": "adaptive_multi_capacity_cell_tree",
        "capacity_classes": list(ADAPTIVE_CAPACITY_CLASSES),
        "cell_target_size": args.target_size,
        "training_trace": str(args.training_trace.resolve()),
        "heldout_trace": str(args.heldout_trace.resolve()),
        "source_assignment": source_assignment_description,
        "source_hierarchy": source_hierarchy_description,
        "adaptive_initial_cell_size": args.adaptive_initial_cell_size,
        "adaptive_radial_member_order": args.adaptive_radial_member_order,
        "adaptive_compact_source_leaves": args.adaptive_compact_source_leaves,
        "capacity_policy": (
            "direct_fit_plus_pair_coaccess_merge"
            if args.enable_coaccess_merge
            else "direct_fit_existing_fixed128_cells"
        ),
        "coaccess_merge_enabled": args.enable_coaccess_merge,
        "minimum_pair_support": args.minimum_pair_support,
        "minimum_merge_lca_depth": args.minimum_merge_lca_depth,
        "iterative_geometric_clustering": False,
        "reused_frozen_assignment_and_hierarchy": (
            reuse_frozen_assignment_and_hierarchy
        ),
        "reused_frozen_member_order": reuse_frozen_member_order,
        "training_query_count": len(training_queries),
        "heldout_query_count": len(heldout_queries),
        "policy_frozen_before_heldout_replay": True,
        "policy": {
            "maximum_normalized_rms_radius": policy.maximum_normalized_rms_radius,
            "maximum_normalized_p95_radius": policy.maximum_normalized_p95_radius,
            "maximum_split_gain": policy.maximum_split_gain,
            "minimum_centroid_radius_overlap": policy.minimum_centroid_radius_overlap,
            "minimum_train_cross_child_coaccess": policy.minimum_train_cross_child_coaccess,
            "maximum_unseen_pq_increase": policy.maximum_unseen_pq_increase,
            "page_bytes": policy.page_bytes,
            "vector_bytes": policy.vector_bytes,
            "lru_pages": policy.lru_pages,
        },
        "capacity_histogram": {
            str(capacity): capacity_histogram.get(capacity, 0)
            for capacity in ADAPTIVE_CAPACITY_CLASSES
        },
        "actual_population_histogram": {
            str(population): count
            for population, count in sorted(population_histogram.items())
        },
        "accepted_merges": sum(row["decision"] == "accept" for row in audit),
        "rejected_merges": sum(row["decision"] == "reject" for row in audit),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "fixed_size_replay": fixed_replay,
        "adaptive_replay": adaptive_replay,
        "matched_fixed_comparator": comparator_name,
        "physical_page_reduction": page_reduction,
        "unseen_pq_increase": unseen_increase,
        "structural_gate_passed": structural_gate,
    }
    write_csv_atomic(
        args.capacity_output,
        ("cell_id", "capacity_class", "actual_population", "tree_node_id", "tree_depth"),
        capacity_rows,
    )
    write_csv_atomic(args.audit_output, ADAPTIVE_AUDIT_FIELDS, audit)
  elif args.global_cells:
    if args.hierarchy_output is None:
      groups = recursive_global_kmeans(
          projected,
          np.arange(len(communities), dtype=np.int32),
          args.target_size,
          args.iterations,
      )
    else:
      groups = []
      cell_centroids: list[np.ndarray] = []
      hierarchy_root, _ = recursive_global_hierarchy(
          projected,
          np.arange(len(communities), dtype=np.int32),
          args.target_size,
          args.iterations,
          groups,
          hierarchy,
          cell_centroids,
      )
  else:
    community_nodes: dict[int, list[int]] = defaultdict(list)
    for node, community in enumerate(communities):
      community_nodes[community].append(node)
    groups = []
    for community in sorted(community_nodes):
      nodes = np.asarray(community_nodes[community], dtype=np.int32)
      groups.extend(
          nodes[local_members]
          for local_members in balanced_kmeans(
              projected[nodes], args.target_size, args.iterations
          )
      )

  serialized_cell_centroids: Sequence[np.ndarray] = []
  adaptive_ordered_nodes = (
      np.empty(len(base), dtype=np.uint32)
      if args.adaptive_pq_locality_profile is not None
      else None
  )
  adaptive_populations = (
      np.empty(len(groups), dtype=np.uint32)
      if args.adaptive_pq_locality_profile is not None
      else None
  )
  adaptive_order_cursor = 0
  fast_binary_assignment = (
      adaptive_ordered_nodes is not None and args.output.suffix == ".bin"
  )
  if not pq_locality:
    for members in groups:
      if reuse_frozen_member_order:
        ordered_members = members
      else:
        centroid = np.mean(projected[members], axis=0, dtype=np.float64)
        differences = projected[members] - centroid
        order = np.argsort(
            np.sum(differences * differences, axis=1, dtype=np.float64),
            kind="stable",
        )
        ordered_members = members[order]
      if adaptive_ordered_nodes is None:
        serialized_centroid = np.asarray(
            np.mean(projected[ordered_members], axis=0, dtype=np.float64),
            dtype=np.float32,
        )
        assert isinstance(serialized_cell_centroids, list)
        serialized_cell_centroids.append(serialized_centroid)
      if adaptive_ordered_nodes is not None:
        end = adaptive_order_cursor + len(ordered_members)
        adaptive_ordered_nodes[adaptive_order_cursor:end] = ordered_members
        adaptive_populations[next_cell] = len(ordered_members)
        adaptive_order_cursor = end
      if not fast_binary_assignment:
        for rank, node in enumerate(ordered_members):
          cells[int(node)] = next_cell
          orders[int(node)] = rank
      next_cell += 1

    if adaptive_ordered_nodes is not None:
      serialized_cell_centroids = (
          precomputed_cell_centroids
          if precomputed_cell_centroids is not None
          else compute_group_centroids(projected, adaptive_ordered_nodes, adaptive_populations)
      )

    if fast_binary_assignment:
      if adaptive_order_cursor != len(base):
        raise ProfileError("Adaptive binary assignment does not cover every Base node")
      write_assignment_binary_from_order(
          args.output, adaptive_ordered_nodes, adaptive_populations, communities
      )
    else:
      if min(cells) < 0 or min(orders) < 0:
        raise ProfileError("Geometric Cells do not cover every Base node")
      args.output.parent.mkdir(parents=True, exist_ok=True)
      temporary = args.output.with_suffix(args.output.suffix + ".tmp")
      with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("node_id", "community_id", "cell_id", "order"))
        writer.writerows(
            (node, communities[node], cells[node], orders[node]) for node in range(len(cells))
        )
      temporary.replace(args.output)
    if adaptive_ordered_nodes is not None:
      if adaptive_order_cursor != len(base):
        raise ProfileError("Adaptive PQ-locality order does not cover every Base node")
      write_pq_locality_profile(
          args.adaptive_pq_locality_profile,
          adaptive_ordered_nodes,
          adaptive_populations,
          np.asarray(serialized_cell_centroids, dtype=np.float32),
          args.target_size,
      )
  if args.hierarchy_output is not None:
    if hierarchy_root is None:
      raise ProfileError("Global hierarchy training did not produce a root")
    if (
        not pq_locality
        and not hierarchy_geometry_finalized
        and (
            not reuse_frozen_assignment_and_hierarchy
            or args.adaptive_pq_locality_profile is not None
        )
    ):
      finalize_hierarchy_geometry(hierarchy_root, hierarchy, serialized_cell_centroids)
    write_hierarchy(
        args.hierarchy_output,
        base.shape[1],
        group_count if pq_locality else len(groups),
        int(np.max(communities)) + 1,
        hierarchy_root,
        hierarchy,
    )
  if adaptive_summary is not None:
    adaptive_summary["assignment_bytes"] = args.output.stat().st_size
    adaptive_summary["hierarchy_bytes"] = args.hierarchy_output.stat().st_size
    adaptive_summary["capacity_bytes"] = args.capacity_output.stat().st_size
    adaptive_summary["audit_bytes"] = args.audit_output.stat().st_size
    if args.adaptive_pq_locality_profile is not None:
      adaptive_summary["pq_locality_profile_bytes"] = (
          args.adaptive_pq_locality_profile.stat().st_size
      )
    write_json_atomic(args.summary_output, adaptive_summary)
  group_populations = None if pq_locality else [len(group) for group in groups]
  print(
      json.dumps(
          {
              "points": len(cells),
              "cells": group_count if pq_locality else len(groups),
              "minimum_cell_size": (
                  minimum_cell_size if pq_locality else min(group_populations)
              ),
              "maximum_cell_size": (
                  maximum_cell_size if pq_locality else max(group_populations)
              ),
              "hierarchy_nodes": len(hierarchy),
              "adaptive_multi_capacity": args.adaptive_multi_capacity,
              "structural_gate_passed": (
                  None
                  if adaptive_summary is None
                  else adaptive_summary["structural_gate_passed"]
              ),
          },
          sort_keys=True,
      )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
