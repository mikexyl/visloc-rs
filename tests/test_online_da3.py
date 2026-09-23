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
from da3_geometry import (Keyframe, CoveredView, align_depth, coverage, resize_intrinsics,
                          pose_depth_scale, filter_depth_confidence, filter_depth_reprojection)
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
                np.full((5, 60, 80), 0. if self.reject else 2., np.float32), inputs, gray, 1.,
                inputs['input_extrinsics'][0].copy())


class DepthGeometryTests(unittest.TestCase):
    def test_confidence_percentile_and_floor_filter_without_changing_depth(self):
        depth = np.full((5,4,5),10.,np.float32)
        scores = np.linspace(1,11,100,dtype=np.float32).reshape(depth.shape)
        filtered,report = filter_depth_confidence(depth,scores,min_confidence=3.,percentile=70.)
        self.assertEqual(np.isfinite(filtered).sum(),30)
        self.assertAlmostEqual(report['threshold'],8.,places=5)
        np.testing.assert_array_equal(filtered[np.isfinite(filtered)],10.)
        filtered,report = filter_depth_confidence(depth,scores,min_confidence=9.5,percentile=70.)
        self.assertEqual(np.isfinite(filtered).sum(),15)
        self.assertEqual(report['threshold'],9.5)
        np.testing.assert_array_equal(depth,10.)

    def test_confidence_percentile_excludes_invalid_pixels_and_never_rescues_a_weak_window(self):
        depth = np.full((5,4,5),10.,np.float32)
        scores = np.full_like(depth,2.)
        depth[0] = np.nan
        scores[0] = 10000.
        filtered,report = filter_depth_confidence(depth,scores,min_confidence=3.,percentile=70.)
        self.assertEqual(report['percentile_threshold'],2.)
        self.assertFalse(np.isfinite(filtered).any())
        depth[:] = np.nan
        filtered,report = filter_depth_confidence(depth,scores,min_confidence=3.,percentile=70.)
        self.assertIsNone(report['percentile_threshold'])
        self.assertEqual(report['retained_fraction'],0.)
        json.dumps(report,allow_nan=False)

    def test_low_confidence_views_cannot_support_reprojection(self):
        class WeakSupportModel(FakeModel):
            def infer(self,window):
                depth,conf,inputs,gray,elapsed,poses = super().infer(window)
                conf[0] = 4.  # Only the reference view passes the absolute floor.
                return depth,conf,inputs,gray,elapsed,poses
        with tempfile.TemporaryDirectory() as output:
            accepted = []
            config = Config(landmark_alignment=False,reprojection_filter=True,min_confidence=3.,confidence_percentile=70.)
            pipeline = Pipeline(config,WeakSupportModel(),output,accepted.append)
            try:
                for i in range(5):
                    pipeline.accept(frame(i))
                self.assertFalse(accepted)
                self.assertEqual(pipeline.stats['confidence_rejections'],0)
                self.assertEqual(pipeline.stats['consistency_rejections'],1)
                self.assertFalse(pipeline.history)
            finally:
                pipeline.close()

    def test_reprojection_keeps_a_plane_with_rotated_cameras_and_different_intrinsics(self):
        y,x = np.mgrid[:60,:80]
        pixels = np.stack((x,y,np.ones_like(x)),axis=-1)
        poses = np.array([frame(i).camera_to_world for i in range(5)])
        k = np.array([frame(i).intrinsics for i in range(5)])
        depth = []
        for i in range(5):
            poses[i,:3,:3] = Rotation.from_euler('y', i-2, degrees=True).as_matrix()
            k[i,0,0] += i
            ray = pixels @ np.linalg.inv(k[i]).T @ poses[i,:3,:3].T
            depth.append((10-poses[i,2,3])/ray[:,:,2])
        depth = np.array(depth,dtype=np.float32)
        original = depth.copy()
        filtered,support,_ = filter_depth_reprojection(depth,k,poses,max_error_px=.05,max_relative_depth=.005)
        self.assertTrue(np.isfinite(filtered[:,10:-10,10:-10]).all())
        self.assertTrue((support[:,15:-15,15:-15]==4).all())
        np.testing.assert_array_equal(filtered[np.isfinite(filtered)],depth[np.isfinite(filtered)])
        np.testing.assert_array_equal(depth,original)
        change = np.eye(4)
        change[:3,:3] = Rotation.from_euler('xyz',[25,-40,80],degrees=True).as_matrix()
        change[:3,3] = [500,-200,80]
        transformed,_,_ = filter_depth_reprojection(depth,k,change@poses,max_error_px=.05,max_relative_depth=.005)
        np.testing.assert_array_equal(transformed,filtered)

    def test_reprojection_rejects_outliers_but_keeps_other_visible_views(self):
        poses = np.array([frame(i).camera_to_world for i in range(5)])
        k = np.array([frame(i).intrinsics for i in range(5)])
        depth = np.full((5,60,80),10.,np.float32)
        depth[0,20:40,30:50] = 3.
        filtered,support,report = filter_depth_reprojection(depth,k,poses)
        self.assertTrue(np.isnan(filtered[0,22:38,32:48]).all())
        self.assertTrue((support[0,22:38,32:48]==0).all())
        self.assertTrue(np.isfinite(filtered[1:,22:38,32:48]).all())
        self.assertGreater(report['rejected_pixels'],300)

    def test_reprojection_depth_check_is_needed_even_when_pixel_roundtrip_is_zero(self):
        poses = np.repeat(np.eye(4)[None],5,axis=0)
        k = np.array([frame(i).intrinsics for i in range(5)])
        depth = np.full((5,60,80),10.,np.float32)
        depth[0] = 20.
        filtered,_,_ = filter_depth_reprojection(depth,k,poses)
        self.assertTrue(np.isnan(filtered[0]).all())
        self.assertTrue(np.isfinite(filtered[1:,1:-1,1:-1]).all())
        depth[2:] = np.nan
        filtered,_,_ = filter_depth_reprojection(depth,k,poses)
        self.assertTrue(np.isnan(filtered).all())

    def test_reprojection_pixel_limit_rejects_depth_that_passes_relative_limit(self):
        poses = np.array([frame(i,x=i).camera_to_world for i in range(5)])
        k = np.array([frame(i).intrinsics for i in range(5)])
        depth = np.full((5,60,80),10.,np.float32)
        depth[0] = 10.4
        loose,_,_ = filter_depth_reprojection(depth,k,poses,max_error_px=2.,max_relative_depth=.1)
        strict,_,_ = filter_depth_reprojection(depth,k,poses,max_error_px=.05,max_relative_depth=.1)
        self.assertTrue(np.isfinite(loose[0,10:-10,40:60]).all())
        self.assertTrue(np.isnan(strict[0]).all())

    def test_reprojection_does_not_count_out_of_view_or_behind_camera_samples(self):
        poses = np.array([frame(i,x=i*100).camera_to_world for i in range(5)])
        k = np.array([frame(i).intrinsics for i in range(5)])
        depth = np.full((5,60,80),10.,np.float32)
        filtered,support,_ = filter_depth_reprojection(depth,k,poses)
        self.assertTrue(np.isnan(filtered).all())
        self.assertFalse(support.any())
        poses[:] = np.eye(4)
        poses[2:,:3,:3] = Rotation.from_euler('y',180,degrees=True).as_matrix()
        filtered,support,_ = filter_depth_reprojection(depth,k,poses)
        self.assertTrue(np.isnan(filtered[:2]).all())
        self.assertTrue((support[:2,10:-10,10:-10]==1).all())

    def test_pipeline_saves_filter_mask_and_only_filtered_depth_enters_history(self):
        class OutlierModel(FakeModel):
            def infer(self,window):
                depth,conf,inputs,gray,elapsed,poses = super().infer(window)
                depth[0,20:40,30:50] = 2.
                return depth,conf,inputs,gray,elapsed,poses
        with tempfile.TemporaryDirectory() as output:
            accepted = []
            pipeline = Pipeline(Config(landmark_alignment=False,reprojection_filter=True),OutlierModel(),output,accepted.append)
            try:
                for i in range(5):
                    pipeline.accept(frame(i))
                self.assertEqual(len(accepted),1)
                self.assertLess(accepted[0]['consistency']['retained_fraction'],1.)
                with np.load(accepted[0]['archive']) as data:
                    self.assertTrue(np.isnan(data['depth_m'][0,22:38,32:48]).all())
                    np.testing.assert_array_equal(data['consistency_mask'],np.isfinite(data['depth_m']))
                self.assertTrue(np.isnan(pipeline.history[0].depth[5:7,7:9]).all())
            finally:
                pipeline.close()

    def test_rejected_consistency_does_not_add_coverage_history(self):
        class InconsistentModel(FakeModel):
            def infer(self,window):
                depth,conf,inputs,gray,elapsed,poses = super().infer(window)
                depth[:] = np.arange(1,6,dtype=np.float32)[:,None,None]
                return depth,conf,inputs,gray,elapsed,poses
        with tempfile.TemporaryDirectory() as output:
            accepted = []
            pipeline = Pipeline(Config(landmark_alignment=False,reprojection_filter=True),InconsistentModel(),output,accepted.append)
            try:
                for i in range(5):
                    pipeline.accept(frame(i))
                self.assertFalse(accepted)
                self.assertEqual(pipeline.stats['consistency_rejections'],1)
                self.assertFalse(pipeline.history)
                self.assertFalse(list(Path(output).glob('window_*.npz')))
            finally:
                pipeline.close()

    def test_pose_scale_uses_camera_centers_and_keeps_metric_camera_poses_fixed(self):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[:, :3, 3] = [[0,0,0], [1,.2,0], [2,0,.1], [3,.4,0], [4,.5,.2]]
        poses[:, :3, :3] = Rotation.from_euler('xyz', [20,40,30], degrees=True).as_matrix()
        predicted = poses.copy()
        rotation = Rotation.from_euler('xyz', [-10,35,70], degrees=True).as_matrix()
        predicted[:, :3, :3] = rotation @ poses[:, :3, :3]
        predicted[:, :3, 3] = .25 * poses[:, :3, 3] @ rotation.T + [7,-3,2]
        supplied = np.linalg.inv(poses)
        original = supplied.copy()
        scale, report = pose_depth_scale(supplied, np.linalg.inv(predicted)[:, :3])
        self.assertAlmostEqual(scale, 4.)
        self.assertLess(report['pose_fit_rmse_m'], 1e-10)
        np.testing.assert_array_equal(supplied, original)
        self.assertIsNone(pose_depth_scale(np.repeat(np.eye(4)[None], 5, axis=0), supplied)[0])

    def test_pose_only_mode_bypasses_landmark_fit_and_held_out_rejection(self):
        with tempfile.TemporaryDirectory() as output, patch('online_da3.align_depth', side_effect=AssertionError('Landmark fit called')):
            accepted = []
            pipeline = Pipeline(Config(landmark_alignment=False), FakeModel(), output, accepted.append)
            try:
                for i in range(5):
                    f = frame(i)
                    f.track_ids[:] = 1  # Cannot satisfy independent fit/held-out landmark requirements.
                    pipeline.accept(f)
                self.assertEqual(len(accepted), 1)
                self.assertEqual(pipeline.stats['alignment_rejections'], 0)
                self.assertEqual(accepted[0]['alignment']['method'], 'input_poses')
                with np.load(accepted[0]['archive']) as data:
                    np.testing.assert_allclose(data['depth_m'], 5.)
                    self.assertAlmostEqual(float(data['scale']), 1.)
                    np.testing.assert_array_equal(data['camera_to_world'], [frame(i).camera_to_world for i in range(5)])
            finally:
                pipeline.close()

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
                       dict(queue_capacity=0),dict(min_fit_landmarks=2.5),dict(max_depth_m=-1),
                       dict(landmark_alignment='false'),dict(reprojection_filter='false'),
                       dict(max_reprojection_error_px=0),dict(max_reprojection_depth_error=float('nan')),
                       dict(min_consistent_views=0),dict(min_consistent_views=5),
                       dict(confidence_percentile=-1),dict(confidence_percentile=100),
                       dict(confidence_percentile=float('nan'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs).validate()


if __name__ == '__main__':
    unittest.main()
