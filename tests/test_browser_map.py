import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location('browser_map', Path(__file__).resolve().parents[1]/'tools/browser_viewer/map_store.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MapTests(unittest.TestCase):
    def points(self, store):
        store.serialize()
        return list(struct.iter_unpack('<fff', store.snapshot()['data']))

    def test_points_survive_empty_and_disjoint_active_windows(self):
        store = m.MapStore()
        store.update('run', [[1, 1, 2, 3]])
        store.update('run', [[2, 20, 30, 40]])
        store.update('run', [])
        self.assertEqual(set(self.points(store)), {(1, 2, 3), (20, 30, 40)})

    def test_retained_track_moves_without_leaving_ghost(self):
        store = m.MapStore()
        store.update('run', [[1, 1, 2, 3]])
        store.update('run', [[1, 4, 5, 6]])
        self.assertEqual(self.points(store), [(4, 5, 6)])
        self.assertEqual(len(store.ids), 1)

    def test_new_stream_resets_immediately_and_invalidates_revision(self):
        store = m.MapStore()
        store.update('a', [[1, 1, 1, 1]])
        store.serialize()
        previous = store.snapshot()['revision']
        store.update('b', [])
        snapshot = store.snapshot()
        self.assertEqual(snapshot['count'], 0)
        self.assertEqual(snapshot['stream'], 'b')
        self.assertGreater(snapshot['revision'], previous)
        self.assertEqual(len(store.ids), 0)

    def test_same_points_reuse_serialized_buffer(self):
        store = m.MapStore()
        store.update('a', [[1, 1, 1, 1]])
        store.serialize()
        snapshot = store.snapshot()
        store.update('a', [[1, 1, 1, 1]])
        store.serialize()
        self.assertIs(snapshot['data'], store.snapshot()['data'])
        self.assertEqual(snapshot['revision'], store.snapshot()['revision'])

    def test_voxel_downsampling_and_updates_across_occupied_cell(self):
        store = m.MapStore(voxel_size=1)
        store.update('a', [[1, .1, .1, .1], [2, .2, .2, .2], [3, 2, 2, 2]])
        self.assertEqual(len(self.points(store)), 2)
        store.update('a', [[1, 2.1, 2.1, 2.1]])
        self.assertEqual(len(self.points(store)), 1)
        self.assertEqual(len(store.ids), len(store.cells))

    def test_capacity_compaction_preserves_earlier_regions(self):
        store = m.MapStore(max_points=128, voxel_size=.1)
        # Fill old negative region, then a distant new region. Neither is evicted.
        for offset in (-100, 100):
            store.update('a', [[offset*1000+i+100001, offset+(i%20)*.2, (i//20)*.2, 0] for i in range(400)])
        points = self.points(store)
        self.assertLessEqual(len(points), 128)
        self.assertGreater(store.voxel_size, .1)
        self.assertTrue(any(p[0] < -90 for p in points))
        self.assertTrue(any(p[0] > 90 for p in points))
        self.assertEqual(len(store.ids), len(store.cells))
        self.assertLessEqual(len(store.snapshot()['data']), 128*12)

    def test_nonfinite_and_extreme_coordinates_are_not_serialized(self):
        store = m.MapStore()
        store.update('a', [[1, float('nan'), 0, 0], [2, 1e300, 0, 0], [3, 1, 2, 3]])
        self.assertEqual(self.points(store), [(1, 2, 3)])

    def test_large_track_ids_remain_distinct(self):
        store = m.MapStore()
        store.update('a', [[2**63, 1, 0, 0], [2**63+1, 2, 0, 0]])
        store.update('a', [[2**63+1, 3, 0, 0]])
        self.assertEqual(set(self.points(store)), {(1, 0, 0), (3, 0, 0)})


if __name__ == '__main__':
    unittest.main()
