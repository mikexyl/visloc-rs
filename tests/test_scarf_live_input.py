"""Causal input and correction contracts for the ScaRF streaming adapter."""
import json
import ast
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'third_party/ScaRF-SLAM'))
from run_scarf_online import LiveInput, corrected_camera_poses


def transform(x=0, yaw=0):
    return dict(translation=[x, 0, 0], rotation_xyzw=Rotation.from_euler('z', yaw, degrees=True).as_quat().tolist())


class ScarfLiveInputTests(unittest.TestCase):
    def test_native_live_correction_requires_new_solved_loop_and_ignores_roundoff(self):
        from scarf_slam.core.pose import MappingPose
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('scarf_slam/mapping_app.py', 'scarf_slam/utils/timestamp_ops.py'):
                (root/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/'third_party/ScaRF-SLAM'/name, root/name)
            subprocess.run(['patch', '-s', '-p1', '-i', str(ROOT/'tools/scarf/live_input.patch')], cwd=root, check=True)
            spec = importlib.util.spec_from_file_location('patched_timestamps', root/'scarf_slam/utils/timestamp_ops.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            rotations = Rotation.random(100, random_state=7).as_quat()
            first = {str(i): MappingPose([0., 0., 0.], q.tolist()) for i, q in enumerate(rotations)}
            same = {str(i): MappingPose([0., 0., 0.], (-q).tolist()) for i, q in enumerate(rotations)}
            self.assertTrue(module.pose_dicts_equal(first, same, rotation_tol_deg=1e-6))
            same['0'] = MappingPose([0., 0., 0.], (Rotation.from_euler('z', .01, degrees=True)
                                * Rotation.from_quat(rotations[0])).as_quat().tolist())
            self.assertFalse(module.pose_dicts_equal(first, same, rotation_tol_deg=1e-6))

            # Execute the patched upstream refresh method without loading CUDA.
            # New poses/revisions alone must not optimize the accumulated map.
            tree = ast.parse((root/'scarf_slam/mapping_app.py').read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScaRFSLAM')
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                          and n.name == '_refresh_slam_trajectory_for_batch')
            scope = dict(List=list, pose_dicts_equal=module.pose_dicts_equal)
            exec(compile(ast.Module(body=[method], type_ignores=[]), '<upstream-refresh>', 'exec'), scope)
            source = object.__new__(LiveInput)
            source.applied_loops, source.graph = set(), dict(optimized_loops=[])
            source.snapshot = dict(t=MappingPose([1., 0., 0.], [0., 0., 0., 1.]))
            calls = []
            app = type('App', (), {})()
            app.input_source = source
            app.in_ref_poses_dict = dict(t=MappingPose([0., 0., 0.], [0., 0., 0., 1.]))
            app.out_ph_poses_dict = app.in_ref_poses_dict.copy()
            app._select_traj_timestamp_for_batch = lambda _: 't'
            app._load_traj_snapshot = source.get_trajectory_snapshot
            app._get_ph_poses_from_ref = lambda: app.in_ref_poses_dict.copy()
            app._refresh_frame_covisibility_pose_features = lambda: None
            app._run_submap_scale_optimization = lambda **kw: calls.append(kw)
            refresh = scope['_refresh_slam_trajectory_for_batch']
            self.assertFalse(refresh(app, ['t']))
            source.graph['optimized_loops'] = [[dict(robot='r', session='s', id=1), dict(robot='r', session='s', id=2)]]
            source.snapshot = dict(t=MappingPose([2., 0., 0.], [0., 0., 0., 1.]))
            self.assertTrue(refresh(app, ['t']))
            source.snapshot = dict(t=MappingPose([3., 0., 0.], [0., 0., 0., 1.]))
            self.assertFalse(refresh(app, ['t']))  # Same loop, even with another pose update.
            self.assertEqual(len(calls), 1)

    def test_correction_interpolates_rotation_and_holds_latest_received_frontier(self):
        graph = [dict(timestamp_ns=10, map_from_odom=transform(0, 0)),
                 dict(timestamp_ns=20, map_from_odom=transform(2, 90))]
        raw = np.repeat([[1, 0, 0, 0, 0, 0, 1]], 3, axis=0)
        poses = corrected_camera_poses(np.array([5, 15, 30]), raw, graph)
        np.testing.assert_allclose(poses[:, :3], [[1, 0, 0], [1+2**-.5, 2**-.5, 0], [2, 1, 0]], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(poses[:, 3:]).as_euler('xyz', degrees=True)[:, 2], [0, 45, 90])

    def test_no_future_rgb_and_final_arrivals_are_drained_with_same_session_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'robots/r').mkdir(parents=True)
            (root/'backend').mkdir()
            (root/'mission.json').write_text(json.dumps(dict(robots=[dict(robot='r', frames=3)])))
            images = [(1_050_000_000, root/'a.jpg', 0), (1_150_000_000, root/'b.jpg', 0)]
            source = LiveInput(root/'mission.json', root, images, np.eye(4))
            journal = root/'robots/r/camera_frames.jsonl'
            def frame(index):
                return dict(robot='r', session='s', frame_id=index, timestamp_ns=1_000_000_000+index*100_000_000,
                            body_to_odom=transform(index))
            try:
                journal.write_text('\n'.join(json.dumps(frame(i)) for i in range(2))+'\n')
                source.poll()
                self.assertEqual(len(source.raw_rgb), 1)
                self.assertEqual(len(source.snapshot), 1)
                # A colliding local keyframe id from another session must not apply.
                graph = dict(backend_mode='global_bundle_adjustment', revision=1, loops=[], optimized_loops=[], poses=[
                    dict(key=dict(robot='r',session='other',id=0),timestamp_ns=1_000_000_000,map_from_odom=transform(999)),
                    dict(key=dict(robot='r',session='s',id=0),timestamp_ns=1_000_000_000,map_from_odom=transform(10))])
                (root/'backend/graph_snapshot.json').write_text(json.dumps(graph))
                with journal.open('a') as stream:
                    stream.write(json.dumps(frame(2))+'\n')
                (root/'backend/finished.json').write_text('{"revision":1}')
                app = type('App', (), {})()
                self.assertTrue(source.wait(app))
                self.assertEqual(len(source.snapshot), 2)
                self.assertAlmostEqual(list(source.snapshot.values())[-1].pos[0], 11.5)
                self.assertFalse(source.wait(app))
                self.assertTrue(source.finished())
            finally:
                source.events.close()


if __name__ == '__main__':
    unittest.main()
