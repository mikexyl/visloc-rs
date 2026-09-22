"""Independent geometry, gating and robust metric-scale contracts for DA3."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from da3_geometry import Keyframe, CoveredView, align_depth, coverage, resize_intrinsics
from online_da3 import Config, OnlineDepth, Pipeline, keyframe_from_vio, model_inputs


def frame(ordinal, x=None):
    x = ordinal*.1 if x is None else x
    y, u = np.mgrid[5:60:10, 5:80:10]
    pixels = np.column_stack((u.ravel(), y.ravel())).astype(float)
    k = np.array([[60., 0, 39.5], [0, 60., 29.5], [0, 0, 1.]])
    pc = np.column_stack(((pixels[:, 0]-39.5)*10/60, (pixels[:, 1]-29.5)*10/60, np.full(len(pixels), 10.)))
    pose = np.eye(4)
    pose[0, 3] = x
    return Keyframe(ordinal*7, ordinal, ordinal*350_000_000, np.full((60, 80), 100, np.uint8),
                    k, pose, np.arange(len(pixels)), pixels, pc+pose[:3, 3])


class FakeModel:
    def __init__(self, reject=False):
        self.calls, self.reject = 0, reject

    def infer(self, window):
        self.calls += 1
        inputs, gray = model_inputs(window, (80, 60))
        return (np.full((5, 60, 80), 5., np.float32),
                np.full((5, 60, 80), 0. if self.reject else 2., np.float32), inputs, gray, 1.)


class DepthGeometryTests(unittest.TestCase):
    def test_intrinsics_follow_resize_pixel_centers_and_poses_include_lever_arm(self):
        k = frame(0).intrinsics
        scaled = resize_intrinsics(k, (80, 60), (113, 91))
        point = np.array([2., -1., 10.])
        original = k @ point
        resized = scaled @ point
        np.testing.assert_allclose(resized[:2]/resized[2], (original[:2]/original[2]+.5)*[113/80, 91/60]-.5)
        r_wb = Rotation.from_euler('z', 90, degrees=True)
        r_bc = Rotation.from_euler('y', 30, degrees=True)
        translation = np.array([1., 2., 3.])
        body = np.array([4., 5., 6.])
        calibration = dict(intrinsics=[dict(intrinsics=dict(fx=60., fy=60., cx=39.5, cy=29.5))],
                           T_imu_cam=[dict(zip(('px','py','pz','qx','qy','qz','qw'), [*translation, *r_bc.as_quat()]))])
        camera_rotation = (r_wb*r_bc).as_matrix()
        camera_position = r_wb.apply(translation)+body
        world = camera_position+camera_rotation @ np.array([0., 0., 10.])
        result = dict(frame_id=7, timestamp_ns=100, position=body,
                      quaternion_xyzw=r_wb.as_quat(), map_points=[[8, *world]],
                      feature_tracks=[[0, 8, 39.5, 29.5]])
        f = keyframe_from_vio(result, np.zeros((60,80), np.uint8), calibration, 1, Config())
        np.testing.assert_allclose(f.camera_to_world[:3,3], camera_position)
        inputs, _ = model_inputs([f]*5, (80,60))
        p_camera = inputs['input_extrinsics'][0,0] @ np.r_[world, 1.]
        np.testing.assert_allclose(p_camera, [0,0,10,1], atol=1e-6)

    def test_coverage_uses_geometry_not_track_identity_and_requires_visibility(self):
        f = frame(0)
        past = CoveredView(np.eye(4), f.intrinsics, np.full((60,80),10.))
        f.track_ids += 10000
        self.assertEqual(coverage([f]*5, [past])['new_fraction'], 0.)
        self.assertGreater(coverage([frame(0,x=10)]*5, [past])['new_fraction'], .3)
        # A new foreground surface is not covered by the old background.
        f.points_world *= .5
        self.assertEqual(coverage([f]*5, [past])['new_fraction'], 1.)
        f.points_world[:,2] = -10
        self.assertFalse(past.covers(f.points_world,.2).any())

    def test_scale_is_shared_across_views_robust_to_outliers_and_independently_checked(self):
        window = [frame(i) for i in range(5)]
        depth = np.full((5,60,80),5.,np.float32)
        conf = np.full_like(depth,2.)
        # Corrupt 20% of fit-only tracks in every view, keeping held-out IDs clean.
        for f, d in zip(window, depth):
            for i in (1,2,6,7,11,12,16,17):
                u,v = f.pixels[i].astype(int)
                d[v-1:v+2,u-1:u+2] = 30
        scale, report = align_depth(window,depth,conf,Config())
        self.assertTrue(report['accepted'], report)
        self.assertAlmostEqual(scale,2.,places=5)
        self.assertLess(report['holdout_relative_median'],1e-6)
        for f,d in zip(window,depth):
            for i in np.flatnonzero(f.track_ids % 5 == 0):
                u,v = f.pixels[i].astype(int)
                d[v-1:v+2,u-1:u+2] = 15
        scale, report = align_depth(window,depth,conf,Config())
        self.assertIsNone(scale)
        self.assertEqual(report['reason'],'held_out_depth_disagreement')

    def test_overlap_gate_runs_before_model_and_sequence_gaps_reset(self):
        with tempfile.TemporaryDirectory() as output:
            model, accepted = FakeModel(), []
            pipeline = Pipeline(Config(min_interval_keyframes=1),model,output,accepted.append)
            try:
                for i in range(10):
                    pipeline.accept(frame(i))
                self.assertEqual(model.calls,1)
                self.assertEqual(len(accepted),1)
                self.assertEqual(accepted[0]['keyframe_ordinals'],[0,1,2,3,4])
                self.assertGreater(pipeline.stats['novelty_skips'],0)
                for i in range(20,25):
                    pipeline.accept(frame(i,x=20+(i-20)*.1))
                self.assertEqual(model.calls,2)
                self.assertEqual(accepted[1]['keyframe_ordinals'],[20,21,22,23,24])
                self.assertEqual(pipeline.stats['sequence_resets'],1)
            finally:
                pipeline.close()

    def test_failed_alignment_does_not_mark_area_as_covered(self):
        with tempfile.TemporaryDirectory() as output:
            model = FakeModel(reject=True)
            pipeline = Pipeline(Config(min_interval_keyframes=1),model,output,lambda _: None)
            try:
                for i in range(6):
                    pipeline.accept(frame(i))
                self.assertEqual(model.calls,2)
                self.assertEqual(pipeline.history,[])
                self.assertEqual(pipeline.stats['alignment_rejections'],2)
            finally:
                pipeline.close()

    def test_worker_queue_is_bounded_while_inference_is_busy(self):
        entered, resume = threading.Event(), threading.Event()

        class BusyModel(FakeModel):
            session = SimpleNamespace(tensors=[])
            closed = False

            def infer(self, window):
                entered.set()
                if not resume.wait(5):
                    raise RuntimeError('Test inference was never released')
                return super().infer(window)

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory)/'test.engine'
            engine.write_bytes(b'fake model for scheduling test')
            model = BusyModel()
            with patch('online_da3.Model', return_value=model):
                worker = OnlineDepth(Config(engine=str(engine), queue_capacity=8), Path(directory)/'out')
                try:
                    for i in range(5):
                        worker.submit(frame(i))
                    self.assertTrue(entered.wait(2))
                    # Inference is blocked: eight queued jobs fit; three drop.
                    for i in range(5, 16):
                        worker.submit(frame(i))
                    self.assertEqual(worker.dropped, 3)
                    self.assertEqual(worker.queue.qsize(), 8)
                    resume.set()
                    result = worker.finish()
                    self.assertEqual(result['keyframes'], 13)
                    self.assertEqual(result['submitted_keyframes'], 16)
                    self.assertEqual(result['dropped_keyframes'], 3)
                    self.assertTrue(model.closed)
                finally:
                    resume.set()
                    worker.abort()

    def test_configuration_rejects_unsafe_silent_values(self):
        for kwargs in (dict(new_area_threshold=0),dict(new_area_threshold=float('nan')),
                       dict(queue_capacity=0),dict(min_fit_landmarks=2.5),dict(max_depth_m=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs).validate()


if __name__ == '__main__':
    unittest.main()
