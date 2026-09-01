import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[2] / "python" / "train_convergence_cells.py"
SPEC = importlib.util.spec_from_file_location("train_convergence_cells", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrainConvergenceCellsTest(unittest.TestCase):
  def test_coaccess_is_deterministic_and_community_local(self):
    rows = [
        {"query_id": "0", "community_id": "0", "node_id": "0", "hop": "5"},
        {"query_id": "0", "community_id": "0", "node_id": "2", "hop": "6"},
        {"query_id": "0", "community_id": "1", "node_id": "3", "hop": "7"},
        {"query_id": "1", "community_id": "0", "node_id": "0", "hop": "5"},
        {"query_id": "1", "community_id": "0", "node_id": "2", "hop": "6"},
    ]
    adjacency, frequency = MODULE.collect_coaccess(rows, 2, 5, 8)
    cells, orders = MODULE.build_assignments(
        [0, 0, 0, 1, 1, 1], [0, 0, 1, 2, 2, 2], adjacency, frequency, 2
    )
    self.assertEqual(cells[0], cells[2])
    self.assertNotEqual(cells[0], cells[3])
    for cell in set(cells):
      cell_orders = sorted(orders[node] for node in range(len(cells)) if cells[node] == cell)
      self.assertEqual(cell_orders, list(range(len(cell_orders))))

  def test_global_observed_groups_cross_communities_deterministically(self):
    rows = [
        {"query_id": "0", "community_id": "0", "node_id": "7", "hop": "5"},
        {"query_id": "0", "community_id": "1", "node_id": "3", "hop": "6"},
        {"query_id": "1", "community_id": "0", "node_id": "7", "hop": "5"},
        {"query_id": "1", "community_id": "1", "node_id": "3", "hop": "6"},
    ]
    adjacency, frequency = MODULE.collect_coaccess(rows, 2, 5, 8, "global")
    first = MODULE.build_observed_groups(adjacency, frequency, 2)
    second = MODULE.build_observed_groups(adjacency, frequency, 2)
    self.assertEqual([group.tolist() for group in first], [[3, 7]])
    self.assertEqual([group.tobytes() for group in first], [group.tobytes() for group in second])

  def test_replay_reports_source_and_trace_locality_coverage(self):
    rows = [
        {"query_id": "0", "node_id": "0", "hop": "5"},
        {"query_id": "0", "node_id": "1", "hop": "6"},
        {"query_id": "0", "node_id": "2", "hop": "7"},
        {"query_id": "0", "node_id": "3", "hop": "8"},
    ]
    replay = MODULE.replay_grouping(
        rows, 1, 5, {0: 0, 1: 0, 2: 0, 3: 0},
        np.asarray([0, 1, 2, 3], dtype=np.uint32), 1,
    )
    self.assertEqual(replay["source_distinct_cells_per_query"], 4.0)
    self.assertEqual(replay["distinct_cells_per_query"], 1.0)
    self.assertEqual(replay["top1_cell_coverage"], 1.0)


if __name__ == "__main__":
  unittest.main()
