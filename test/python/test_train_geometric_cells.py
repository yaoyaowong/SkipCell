import importlib.util
import csv
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np


PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"
sys.path.insert(0, str(PYTHON_DIR))
MODULE_PATH = PYTHON_DIR / "train_geometric_cells.py"
SPEC = importlib.util.spec_from_file_location("train_geometric_cells", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TrainGeometricCellsTest(unittest.TestCase):
  def test_recursive_hierarchy_is_deterministic_and_covers_every_leaf(self):
    points = np.asarray(
        [[float(node), float(node % 7), float((node * 3) % 11)] for node in range(40)],
        dtype=np.float32,
    )

    def train():
      groups = []
      hierarchy = []
      cell_centroids = []
      root, descendant_cells = MODULE.recursive_global_hierarchy(
          points,
          np.arange(len(points), dtype=np.int32),
          4,
          3,
          groups,
          hierarchy,
          cell_centroids,
      )
      serialized_centroids = []
      for members in groups:
        centroid = np.zeros(points.shape[1], dtype=np.float32)
        for node in members:
          centroid += points[node]
        centroid /= np.float32(len(members))
        serialized_centroids.append(centroid)
      MODULE.finalize_hierarchy_geometry(root, hierarchy, serialized_centroids)
      return root, descendant_cells, groups, hierarchy

    first = train()
    second = train()
    self.assertEqual(first[0], 0)
    self.assertEqual(first[1], list(range(len(first[2]))))
    self.assertEqual(sorted(np.concatenate(first[2]).tolist()), list(range(len(points))))
    self.assertEqual(
        [(node.children, node.children_are_cells) for node in first[3]],
        [(node.children, node.children_are_cells) for node in second[3]],
    )

    with tempfile.TemporaryDirectory() as directory:
      first_path = Path(directory) / "first.bin"
      second_path = Path(directory) / "second.bin"
      for path, trained in ((first_path, first), (second_path, second)):
        MODULE.write_hierarchy(path, points.shape[1], len(trained[2]), 2, trained[0], trained[3])
      first_bytes = first_path.read_bytes()
      self.assertEqual(first_bytes, second_path.read_bytes())
      header = struct.unpack("<QIIIIIIQ", first_bytes[:40])
      self.assertEqual(header[0], MODULE.HIERARCHY_MAGIC)
      self.assertEqual(header[1], MODULE.HIERARCHY_VERSION)
      self.assertEqual(header[2], points.shape[1])
      self.assertEqual(header[3], len(first[2]))
      self.assertEqual(header[4], len(first[3]))
      self.assertEqual(zlib.crc32(first_bytes[40:]), header[7])

  def test_direct_fit_uses_actual_population_without_repartitioning(self):
    sizes = (16, 17, 32, 33, 64, 65, 127, 128)
    groups = []
    begin = 0
    for size in sizes:
      groups.append(np.arange(begin, begin + size, dtype=np.int32))
      begin += size
    hierarchy = [
        MODULE.HierarchyNode(
            np.zeros(2, dtype=np.float32),
            list(range(1, len(groups) + 1)),
            False,
            begin,
            0.0,
        )
    ]
    hierarchy.extend(
        MODULE.HierarchyNode(
            np.zeros(2, dtype=np.float32), [cell_id], True, size, 0.0
        )
        for cell_id, size in enumerate(sizes)
    )
    original_groups = [group.copy() for group in groups]
    rows = MODULE.assign_fixed_leaf_capacity_classes(0, hierarchy, groups)
    self.assertEqual(
        [row["capacity_class"] for row in rows],
        [16, 32, 32, 64, 64, 128, 128, 128],
    )
    self.assertEqual([row["actual_population"] for row in rows], list(sizes))
    for original, current in zip(original_groups, groups):
      np.testing.assert_array_equal(original, current)
    with self.assertRaises(MODULE.ProfileError):
      MODULE.smallest_capacity_class(129)

  def test_pq_locality_construction_is_deterministic_and_one_owner(self):
    codes = np.asarray(
        [[node % 8, (node * 3) % 8, (node * 5) % 8, (node * 7) % 8]
         for node in range(37)],
        dtype=np.uint8,
    )
    pivots = np.zeros((256, 4), dtype=np.float32)
    for code in range(256):
      pivots[code] = np.asarray((code, code * 2, code * 3, code * 4), dtype=np.float32)
    centroid = np.asarray((0.5, 1.5, 2.5, 3.5), dtype=np.float32)
    chunk_offsets = np.arange(5, dtype=np.uint32)

    first = MODULE.pq_locality_order(codes, pivots, centroid, chunk_offsets, 4)
    second = MODULE.pq_locality_order(codes, pivots, centroid, chunk_offsets, 4)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(len(codes)))
    populations, centroids = MODULE.pq_reconstructed_cell_centroids(
        codes, first, pivots, centroid, chunk_offsets, 8, 4
    )
    self.assertEqual(populations.tolist(), [8, 8, 8, 8, 5])
    root, hierarchy = MODULE.build_contiguous_centroid_hierarchy(centroids, populations, 2)
    self.assertEqual(hierarchy[root].descendant_node_count, len(codes))
    leaves = [node for node in hierarchy if node.children_are_cells]
    self.assertEqual(len(leaves), len(populations))
    self.assertTrue(all(len(node.children) == 1 for node in leaves))

    with tempfile.TemporaryDirectory() as directory:
      root_path = Path(directory)
      profiles = (root_path / "first.pqloc", root_path / "second.pqloc")
      assignments = (root_path / "first.assignment.bin", root_path / "second.assignment.bin")
      communities = np.arange(len(codes), dtype=np.uint32) % 3
      for profile, assignment in zip(profiles, assignments):
        MODULE.write_pq_locality_profile(profile, first, populations, centroids, 8)
        MODULE.write_assignment_binary_from_order(
            assignment, first, populations, communities
        )
      self.assertEqual(profiles[0].read_bytes(), profiles[1].read_bytes())
      self.assertEqual(assignments[0].read_bytes(), assignments[1].read_bytes())
      groups = MODULE.read_binary_assignment_groups(
          assignments[0], len(codes), len(populations)
      )
      np.testing.assert_array_equal(np.concatenate(groups), first)

  def test_grouped_centroids_match_scalar_float64_accumulation(self):
    points = np.arange(63, dtype=np.float32).reshape(21, 3)
    ordered = np.asarray(
        [4, 2, 7, 1, 0, 9, 8, 6, 5, 3, 10, 12, 11, 15, 14, 13, 16, 20, 19, 18, 17],
        dtype=np.uint32,
    )
    populations = np.asarray([4, 6, 3, 8], dtype=np.uint32)
    actual = MODULE.compute_group_centroids(points, ordered, populations, batch_cells=2)
    offsets = np.concatenate(([0], np.cumsum(populations)))
    expected = np.asarray(
        [
            np.mean(points[ordered[int(offsets[cell]) : int(offsets[cell + 1])]], axis=0,
                    dtype=np.float64)
            for cell in range(len(populations))
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)

  def test_radial_group_member_order_matches_scalar_stable_reference(self):
    points = np.asarray(
        [
            [3.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 4.0],
            [0.0, 1.0],
            [0.0, 3.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    ordered = np.asarray([3, 0, 2, 1, 6, 4, 7, 5], dtype=np.uint32)
    populations = np.asarray([4, 4], dtype=np.uint32)
    expected = []
    cursor = 0
    for population in populations:
      members = ordered[cursor : cursor + population]
      centroid = np.mean(points[members], axis=0, dtype=np.float64)
      differences = points[members] - centroid
      order = np.argsort(
          np.sum(differences * differences, axis=1, dtype=np.float64), kind="stable"
      )
      expected.extend(members[order])
      cursor += population
    actual = MODULE.radial_group_member_order(
        points, ordered, populations, batch_cells=1
    )
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.uint32))

  def test_training_trace_rejects_heldout_rows(self):
    with tempfile.TemporaryDirectory() as directory:
      trace_path = Path(directory) / "trace.csv"
      with trace_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("query_id", "node_id", "split", "accessed", "gt", "already_seen"))
        writer.writerow((0, 0, "heldout", 1, 1, 0))
      with self.assertRaises(MODULE.ProfileError):
        MODULE.read_replay_trace(trace_path, "train", 1)

  def test_binary_assignment_and_trace_are_deterministic_and_streamed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      assignment_csv = root / "assignment.csv"
      with assignment_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("node_id", "community_id", "cell_id", "order"))
        writer.writerows(((0, 0, 0, 1), (1, 0, 1, 0), (2, 0, 0, 0), (3, 0, 1, 1)))
      assignment_paths = (root / "first.assignment.bin", root / "second.assignment.bin")
      for path in assignment_paths:
        MODULE.write_assignment_binary(assignment_csv, path, 4, 2)
      self.assertEqual(assignment_paths[0].read_bytes(), assignment_paths[1].read_bytes())
      groups = MODULE.read_assignment_groups(assignment_paths[0], 4, 2)
      np.testing.assert_array_equal(groups[0], np.asarray([2, 0], dtype=np.uint32))
      np.testing.assert_array_equal(groups[1], np.asarray([1, 3], dtype=np.uint32))

      trace_csv = root / "trace.csv"
      with trace_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("query_id", "node_id", "split", "accessed", "gt", "already_seen"))
        writer.writerows(((0, 0, "train", 1, 1, 0), (0, 2, "train", 1, 0, 1),
                          (1, 1, "train", 1, 1, 1)))
      trace_paths = (root / "first.trace.bin", root / "second.trace.bin")
      for path in trace_paths:
        MODULE.write_trace_binary(trace_csv, path, "train", 4)
      self.assertEqual(trace_paths[0].read_bytes(), trace_paths[1].read_bytes())
      streamed = MODULE.open_replay_trace(trace_paths[0], "train", 4)
      self.assertIsInstance(streamed, MODULE.BinaryReplayTrace)
      self.assertEqual(len(streamed), 2)
      self.assertEqual(list(streamed), MODULE.read_replay_trace(trace_csv, "train", 4))

      corrupt = bytearray(trace_paths[0].read_bytes())
      corrupt[-1] ^= 0x1
      corrupt_path = root / "corrupt.trace.bin"
      corrupt_path.write_bytes(corrupt)
      with self.assertRaises(MODULE.ProfileError):
        MODULE.open_replay_trace(corrupt_path, "train", 4)

  def test_pair_coaccess_merge_is_deterministic_and_preserves_one_owner(self):
    points = np.zeros((16, 2), dtype=np.float32)
    groups = [
        np.arange(0, 8, dtype=np.int32),
        np.arange(8, 16, dtype=np.int32),
    ]
    hierarchy = [
        MODULE.HierarchyNode(np.zeros(2, dtype=np.float32), [1, 2], False, 16, 0.0),
        MODULE.HierarchyNode(np.zeros(2, dtype=np.float32), [0], True, 8, 0.0),
        MODULE.HierarchyNode(np.zeros(2, dtype=np.float32), [1], True, 8, 0.0),
    ]
    queries = [
        MODULE.ReplayQuery(
            query_id,
            frozenset({0, 8}),
            frozenset({0, 8}),
            frozenset(range(16)),
        )
        for query_id in range(20)
    ]
    policy = MODULE.AdaptivePolicy(
        maximum_normalized_rms_radius=10.0,
        maximum_normalized_p95_radius=10.0,
        maximum_split_gain=1.0,
        minimum_centroid_radius_overlap=0.0,
        minimum_train_cross_child_coaccess=1.0,
        maximum_unseen_pq_increase=0.1,
        page_bytes=4096,
        vector_bytes=1,
        lru_pages=0,
    )
    first = MODULE.train_pair_coaccess_merges(
        points, 0, hierarchy, groups, queries, queries, policy, 20, 0
    )
    second = MODULE.train_pair_coaccess_merges(
        points, 0, hierarchy, groups, queries, queries, policy, 20, 0
    )
    self.assertEqual(first, second)
    self.assertEqual(first[0], [(0, 1)])
    self.assertEqual(set(first[1][0]), set(MODULE.ADAPTIVE_AUDIT_FIELDS))
    self.assertEqual(first[1][0]["decision"], "accept")
    merged_groups, merged_hierarchy, merged_root = MODULE.materialize_pair_merges(
        0, hierarchy, groups, first[0]
    )
    self.assertEqual(len(merged_groups), 1)
    np.testing.assert_array_equal(np.sort(merged_groups[0]), np.arange(16))
    capacity_rows = MODULE.assign_fixed_leaf_capacity_classes(
        merged_root, merged_hierarchy, merged_groups
    )
    self.assertEqual(capacity_rows[0]["capacity_class"], 16)
    rejected = MODULE.train_pair_coaccess_merges(
        points, 0, hierarchy, groups, queries[:19], queries, policy, 20, 0
    )[1]
    self.assertEqual(rejected[0]["rejection_reason"], "train_pair_support")

  def test_pair_coaccess_ignores_non_sibling_regions(self):
    points = np.zeros((24, 2), dtype=np.float32)
    groups = [
        np.arange(0, 8, dtype=np.int32),
        np.arange(8, 16, dtype=np.int32),
        np.arange(16, 24, dtype=np.int32),
    ]
    hierarchy = [
        MODULE.HierarchyNode(np.zeros(2, dtype=np.float32), [1, 2], False, 24, 0.0),
        MODULE.HierarchyNode(np.zeros(2, dtype=np.float32), [0, 1], True, 16, 0.0),
        MODULE.HierarchyNode(np.zeros(2, dtype=np.float32), [2], True, 8, 0.0),
    ]
    queries = [
        MODULE.ReplayQuery(query_id, frozenset({0, 16}), frozenset(), frozenset())
        for query_id in range(20)
    ]
    selected, audit = MODULE.train_pair_coaccess_merges(
        points, 0, hierarchy, groups, queries, queries, MODULE.AdaptivePolicy(
            maximum_normalized_rms_radius=10.0,
            maximum_normalized_p95_radius=10.0,
            maximum_split_gain=1.0,
            minimum_centroid_radius_overlap=0.0,
            minimum_train_cross_child_coaccess=0.0,
            maximum_unseen_pq_increase=1.0,
            page_bytes=4096,
            vector_bytes=1,
        ), 1, 0
    )
    self.assertEqual(selected, [])
    self.assertEqual(audit, [])

  def test_active_cell_leaf_normalization_provides_unique_tree_nodes(self):
    groups = [
        np.arange(0, 4, dtype=np.int32),
        np.arange(4, 8, dtype=np.int32),
    ]
    hierarchy = [
        MODULE.HierarchyNode(np.zeros(2, dtype=np.float32), [0, 1], True, 8, 0.0)
    ]
    self.assertTrue(MODULE.normalize_active_cell_leaves(hierarchy, groups))
    rows = MODULE.assign_fixed_leaf_capacity_classes(0, hierarchy, groups)
    self.assertEqual(len({row["tree_node_id"] for row in rows}), 2)
    self.assertFalse(hierarchy[0].children_are_cells)
    self.assertTrue(all(hierarchy[child].children_are_cells for child in hierarchy[0].children))

  def test_adaptive_cli_smoke_is_byte_identical_and_reports_gate(self):
    points = np.zeros((255, 2), dtype=np.float32)
    points[:, 0] = np.arange(len(points), dtype=np.float32) * 1.0e-6
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      base_path = root / "base.fbin"
      with base_path.open("wb") as output:
        output.write(struct.pack("<II", len(points), points.shape[1]))
        output.write(np.asarray(points, dtype="<f4").tobytes())

      section_count = 18
      header_size = 96 + section_count * 32
      memberships = struct.pack(f"<{len(points)}I", *([0] * len(points)))
      sidecar_path = root / "source.community_polar.bin"
      fixed = struct.pack(
          "<QIIIIQIIQQQQQQ",
          0,
          5,
          header_size,
          0x01020304,
          section_count,
          len(points),
          points.shape[1],
          1,
          0,
          0,
          0,
          0,
          0,
          0,
      ).ljust(96, b"\0")
      directory_bytes = bytearray()
      payload = bytearray()
      offset = header_size
      for section_id in range(1, section_count + 1):
        section_payload = memberships if section_id in (2, 3) else b""
        count = len(points) if section_id in (2, 3) else 0
        directory_bytes.extend(
            struct.pack("<IIQQQ", section_id, 0, offset, len(section_payload), count)
        )
        payload.extend(section_payload)
        offset += len(section_payload)
      sidecar_path.write_bytes(fixed + directory_bytes + payload)

      source_assignment = root / "fixed128.assignment.csv"
      with source_assignment.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("node_id", "community_id", "cell_id", "order"))
        for node in range(len(points)):
          cell_id = node // 128
          writer.writerow((node, 0, cell_id, node % 128))
      source_hierarchy = root / "fixed128.hierarchy.bin"
      source_cell_count = (len(points) + 127) // 128
      source_nodes = [
          MODULE.HierarchyNode(
              np.asarray(np.mean(points, axis=0, dtype=np.float64), dtype=np.float32),
              list(range(1, source_cell_count + 1)),
              False,
              len(points),
              0.0,
          )
      ]
      for cell_id in range(source_cell_count):
        begin = cell_id * 128
        end = min(len(points), begin + 128)
        source_nodes.append(
            MODULE.HierarchyNode(
                np.asarray(
                    np.mean(points[begin:end], axis=0, dtype=np.float64),
                    dtype=np.float32,
                ),
                [cell_id],
                True,
                end - begin,
                0.0,
            )
        )
      MODULE.write_hierarchy(
          source_hierarchy,
          points.shape[1],
          source_cell_count,
          1,
          0,
          source_nodes,
      )

      def write_trace(path, split):
        with path.open("w", newline="", encoding="utf-8") as output:
          writer = csv.writer(output, lineterminator="\n")
          writer.writerow(
              ("query_id", "node_id", "split", "accessed", "gt", "already_seen")
          )
          for node in range(len(points)):
            writer.writerow((0, node, split, 1, 1, 1))

      training_trace = root / "training.csv"
      heldout_trace = root / "heldout.csv"
      write_trace(training_trace, "train")
      write_trace(heldout_trace, "heldout")

      products = []
      for repetition in ("first", "second"):
        prefix = root / repetition
        product = {
            "assignment": prefix.with_suffix(".assignment.csv"),
            "hierarchy": prefix.with_suffix(".hierarchy.bin"),
            "capacity": prefix.with_suffix(".capacity.csv"),
            "audit": prefix.with_suffix(".audit.csv"),
            "summary": prefix.with_suffix(".summary.json"),
            "pqloc": prefix.with_suffix(".adaptive.pqloc"),
        }
        command = [
            sys.executable,
            str(MODULE_PATH),
            "--sidecar",
            str(sidecar_path),
            "--base_file",
            str(base_path),
            "--output",
            str(product["assignment"]),
            "--hierarchy_output",
            str(product["hierarchy"]),
            "--target_size",
            "128",
            "--identity_projection",
            "--global_cells",
            "--adaptive_multi_capacity",
            "--source_assignment",
            str(source_assignment),
            "--source_hierarchy",
            str(source_hierarchy),
            "--training_trace",
            str(training_trace),
            "--heldout_trace",
            str(heldout_trace),
            "--capacity_output",
            str(product["capacity"]),
            "--audit_output",
            str(product["audit"]),
            "--summary_output",
            str(product["summary"]),
            "--adaptive_pq_locality_profile",
            str(product["pqloc"]),
            "--vector_bytes",
            "1",
            "--maximum_normalized_rms_radius",
            "10",
            "--maximum_normalized_p95_radius",
            "10",
            "--maximum_split_gain",
            "1",
            "--minimum_centroid_radius_overlap",
            "0",
            "--minimum_train_cross_child_coaccess",
            "1",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertIn('"adaptive_multi_capacity": true', completed.stdout)
        products.append(product)
      for name in products[0]:
        self.assertEqual(products[0][name].read_bytes(), products[1][name].read_bytes())
      self.assertEqual(products[0]["assignment"].read_bytes(), source_assignment.read_bytes())
      source_root, source_dimension, source_cells, source_nodes = MODULE.read_hierarchy(
          source_hierarchy
      )
      output_root, output_dimension, output_cells, output_nodes = MODULE.read_hierarchy(
          products[0]["hierarchy"]
      )
      self.assertEqual((output_root, output_dimension, output_cells),
                       (source_root, source_dimension, source_cells))
      self.assertEqual(
          [(node.children, node.children_are_cells) for node in output_nodes],
          [(node.children, node.children_are_cells) for node in source_nodes],
      )
      self.assertTrue(all(np.isfinite(node.centroid).all() for node in output_nodes))
      summary = __import__("json").loads(products[0]["summary"].read_text())
      self.assertIn("Cell-128", summary["fixed_size_replay"])
      self.assertEqual(
          summary["capacity_policy"], "direct_fit_existing_fixed128_cells"
      )
      self.assertFalse(summary["iterative_geometric_clustering"])
      self.assertTrue(summary["reused_frozen_assignment_and_hierarchy"])
      self.assertTrue(summary["policy_frozen_before_heldout_replay"])
      self.assertIsInstance(summary["structural_gate_passed"], bool)
      self.assertEqual(summary["pq_locality_profile_bytes"], products[0]["pqloc"].stat().st_size)

  def test_single_community_requires_no_sidecar(self):
    points = np.arange(32, dtype=np.float32).reshape(16, 2)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      base_path = root / "base.fbin"
      with base_path.open("wb") as output:
        output.write(struct.pack("<II", len(points), points.shape[1]))
        output.write(np.asarray(points, dtype="<f4").tobytes())
      assignment = root / "assignment.csv"
      completed = subprocess.run(
          [
              sys.executable,
              str(MODULE_PATH),
              "--single_community",
              "--base_file",
              str(base_path),
              "--output",
              str(assignment),
              "--target_size",
              "8",
              "--identity_projection",
              "--global_cells",
              "--iterations",
              "1",
          ],
          check=True,
          capture_output=True,
          text=True,
      )
      self.assertIn('"points": 16', completed.stdout)
      with assignment.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
      self.assertEqual({row["community_id"] for row in rows}, {"0"})


if __name__ == "__main__":
  unittest.main()
