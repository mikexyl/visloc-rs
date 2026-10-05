"""Causal streaming behavior, bounded state, PGO corrections, and live transport."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from online_landmark_map import OnlineLandmarkMap, RerunLandmarkPublisher
from visualize_multi_robot_live import JsonTail, MissionFollower
from test_landmark_visualization import identity, scene


def raw_pose(feature, poses):
    return dict(key=feature['key'], timestamp_ns=feature['key']['id']*100000000,
                body_to_odom=poses[identity(feature['key'])]['body_to_map'])


def drain(display):
    while display.dirty:
        display.refine_pending(seconds=10, max_tracks=100)


class OnlineMapTests(unittest.TestCase):
    def test_map_appears_before_any_graph_exists_and_only_with_received_views(self):
        features, poses, truth = scene(4)
        display = OnlineLandmarkMap()
        display.ingest(features[0], raw_pose(features[0], poses))
        drain(display)
        self.assertEqual(display.snapshot()[0], [])
        for feature in features[1:]:
            display.ingest(feature, raw_pose(feature, poses))
            drain(display)
            points, audit = display.snapshot()
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0]['observations'], feature['key']['id'] + 1)
            np.testing.assert_allclose(points[0]['position'], truth, atol=1e-7)
            self.assertEqual(audit['graph_revision'], -1)

    def test_pose_graph_moves_existing_point_before_refinement_and_rejects_stale_graph(self):
        features, poses, truth = scene(3)
        display = OnlineLandmarkMap()
        for feature in features:
            display.ingest(feature, raw_pose(feature, poses))
        drain(display)
        graph = dict(revision=2, poses=copy.deepcopy(list(poses.values())))
        for p in graph['poses']:
            p['body_to_map']['translation'][0] += 4
        display.update_graph(graph)
        point = display.snapshot()[0][0]
        np.testing.assert_allclose(point['position'], truth + [4,0,0], atol=1e-7)
        self.assertTrue(point['refinement_pending'])
        drain(display)
        display.update_graph(dict(revision=1, poses=list(poses.values())))
        point = display.snapshot()[0][0]
        np.testing.assert_allclose(point['position'], truth + [4,0,0], atol=1e-7)
        self.assertFalse(point['refinement_pending'])
        display.update_graph(graph)
        self.assertFalse(display.dirty)

    def test_duplicate_frames_and_colliding_sessions_do_not_merge(self):
        display = OnlineLandmarkMap()
        for robot, session in [('a','one'), ('b','one'), ('a','two')]:
            features, poses, _ = scene(3, robot, session)
            for f in features:
                display.ingest(f, raw_pose(f, poses))
                display.ingest(f, raw_pose(f, poses))
        drain(display)
        points, audit = display.snapshot()
        self.assertEqual(len(points), 3)
        self.assertEqual(audit['observations_retained'], 9)
        self.assertTrue(all(p['inliers'] == 3 for p in points))

    def test_view_track_pose_and_queue_memory_are_bounded(self):
        features, poses, _ = scene(12, baseline=.1)
        display = OnlineLandmarkMap(max_tracks=3, max_views=4)
        for track_id in range(20):
            for f in copy.deepcopy(features):
                f['track_ids'] = [track_id]
                display.ingest(f, raw_pose(f, poses))
        self.assertEqual(len(display.tracks), 3)
        self.assertEqual(len(display.dirty), 3)
        self.assertLessEqual(len(display.poses), 12)
        self.assertTrue(all(len(v['views']) == 4 for v in display.tracks.values()))
        self.assertTrue(all(next(iter(v['views']))[2] == 0 for v in display.tracks.values()))
        self.assertEqual(display.counts['evicted_tracks'], 17)
        self.assertEqual(display.refine_pending(seconds=10, max_tracks=1), 1)
        self.assertEqual(len(display.dirty), 2)

    def test_unchanged_pgo_roundoff_does_not_refine_the_entire_history(self):
        features, poses, _ = scene(3)
        display = OnlineLandmarkMap()
        for f in features:
            display.ingest(f, raw_pose(f, poses))
        drain(display)
        graph = dict(revision=1, poses=copy.deepcopy(list(poses.values())))
        for pose in graph['poses']:
            pose['body_to_map']['translation'][0] += 1e-12
        display.update_graph(graph)
        self.assertFalse(display.dirty)

    def test_bad_new_measurements_remove_the_previous_display_point(self):
        features, poses, _ = scene(3)
        display = OnlineLandmarkMap()
        for f in features:
            display.ingest(f, raw_pose(f, poses))
        drain(display)
        self.assertEqual(len(display.snapshot()[0]), 1)
        # Distinct pixel shifts destroy shared geometry; the old point must not linger.
        for i, f in enumerate(copy.deepcopy(features)):
            f['pixels'][0][1] += [30, -50, 70][i]
            display.ingest(f)
        drain(display)
        self.assertEqual(display.snapshot()[0], [])

    def test_feature_before_pose_and_component_join_replaces_cloud(self):
        features, poses, _ = scene(4)
        display = OnlineLandmarkMap()
        for f in features:
            display.ingest(f)
        drain(display)
        self.assertEqual(display.snapshot()[0], [])
        split = copy.deepcopy(list(poses.values()))
        for i, p in enumerate(split):
            p['component'] = dict(p['component'], id=i // 2)
        display.update_graph(dict(revision=1, poses=split))
        drain(display)
        self.assertEqual(len(display.snapshot()[0]), 2)
        class FakeRerun:
            Radius = type('Radius', (), dict(ui_points=lambda x:x))
            def __init__(self): self.cleared = []
            def log(self, entity, *args):
                if args[0] == 'clear': self.cleared.append(entity)
            def Clear(self, **_): return 'clear'
            def Points3D(self, *_, **__): return 'points'
            def AnyValues(self, **_): return 'attrs'
        rr = FakeRerun()
        publisher = RerunLandmarkPublisher(rr, {'a':[1,2,3]})
        publisher.publish(display.snapshot()[0])
        display.update_graph(dict(revision=2, poses=list(poses.values())))
        # The same ID must not appear twice even before the refinement queue drains.
        self.assertEqual(len(display.snapshot()[0]), 1)
        drain(display)
        publisher.publish(display.snapshot()[0])
        self.assertEqual(len(publisher.entities), 1)
        self.assertTrue(any('a_first_1' in path for path in rr.cleared))

    def test_live_journal_is_read_incrementally_without_final_graph_or_sensor_bag(self):
        features, poses, _ = scene(3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mission = root / 'mission.json'
            mission.write_text(json.dumps(dict(robots=[dict(robot='a')])))
            directory = root / 'robots/a'
            directory.mkdir(parents=True)
            path = directory / 'visualization.jsonl'
            follower = MissionFollower(mission)
            self.assertEqual(follower.poll(), 0)
            for i, f in enumerate(features):
                packet = json.dumps(dict(feature=f, pose=raw_pose(f, poses)))
                with path.open('a') as out:
                    out.write(packet[:20])
                self.assertEqual(follower.poll(), 0)
                with path.open('a') as out:
                    out.write(packet[20:] + '\n')
                self.assertEqual(follower.poll(), 1)
                drain(follower.map)
                self.assertEqual(len(follower.map.snapshot()[0]), int(i > 0))
            self.assertFalse(follower.backlog())
            self.assertEqual(follower.poll(), 0)
            self.assertEqual(follower.read_packets, 3)
            (root / 'backend').mkdir()
            (root / 'backend/finished.json').write_text('{"revision":2}')
            follower.graph['revision'] = 1
            self.assertFalse(follower.finished())
            follower.graph['revision'] = 2
            self.assertTrue(follower.finished())
            # Atomic replacement/truncation starts a fresh journal read.
            path.write_text('{}\n')
            self.assertEqual(follower.tails[path].poll(), [{}])


if __name__ == '__main__':
    unittest.main()
