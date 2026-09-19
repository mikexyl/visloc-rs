"""Coordinate-convention checks for GRACO replay and stereo refinement."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_graco_vio import prepare_calibration
from refine_graco_stereo import epipolar_error


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transforms = []
        for x in (0., .43):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix()
            transform[:3, 3] = [x, .02, .03]
            self.transforms.append(transform)
        stereo = {f'cam{i}': dict(camera_model='pinhole', distortion_model='radial-tangential',
                  intrinsics=[920., 925., 810., 552.], distortion_coeffs=[-.1, .08, .001, .0002],
                  resolution=[1600, 1100]) for i in range(2)}
        extrinsics = {f'T_Imu_cam{i}': {'data': transform.ravel().tolist()}
                      for i, transform in enumerate(self.transforms)}
        imu = dict(rate_hz=125, accelerometer_noise_density=.02, gyroscope_noise_density=.0005,
                   accelerometer_random_walk=.0006, gyroscope_random_walk=.00001)
        for name, value in [('stereo.yaml', stereo), ('stereo-imu.yaml', extrinsics), ('imu.yaml', imu)]:
            (self.root / name).write_text(yaml.safe_dump(value))

    def test_resize_preserves_calibrated_pixel_centers(self):
        calibration, maps, _ = prepare_calibration(self.root, self.root, 800)
        intrinsics = calibration['intrinsics'][0]['intrinsics']
        self.assertEqual(intrinsics['cx'], (810. + .5) / 2 - .5)
        self.assertEqual(intrinsics['cy'], (552. + .5) / 2 - .5)
        # The remap at an output pixel must sample the distorted source at the
        # location obtained by projecting the same ray in the original camera.
        u, v = 300, 200
        ray = np.array([[(u - intrinsics['cx']) / intrinsics['fx'],
                         (v - intrinsics['cy']) / intrinsics['fy'], 1.]])
        raw, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3),
                                  np.array([[920., 0, 810.], [0, 925., 552.], [0, 0, 1.]]),
                                  np.array([-.1, .08, .001, .0002]))
        expected = (raw.ravel() + .5) / 2 - .5
        np.testing.assert_allclose([maps[0][0][v, u], maps[0][1][v, u]], expected, atol=2e-5)

    def test_refinement_rotates_right_axes_without_moving_camera_centers(self):
        prepare_calibration(self.root, self.root, 800)
        report = json.loads((self.root / 'calibration_report.json').read_text())
        correction = np.array([.002, -.003, .004])
        refinement = self.root / 'refinement.json'
        refinement.write_text(json.dumps({'schema': 'visloc.graco.stereo_rotation_refinement.v1',
            'source_sha256': report['source_sha256'], 'right_rotation_correction_rotvec': correction.tolist()}))
        calibration, _, _ = prepare_calibration(self.root, self.root, 800, refinement)
        output_transforms = []
        for c in calibration['T_imu_cam']:
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat([c[k] for k in ['qx', 'qy', 'qz', 'qw']]).as_matrix()
            transform[:3, 3] = [c[k] for k in ['px', 'py', 'pz']]
            output_transforms.append(transform)
        np.testing.assert_allclose(output_transforms[0], self.transforms[0], atol=1e-14)
        np.testing.assert_allclose(output_transforms[1][:3, 3], self.transforms[1][:3, 3])
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec(correction).as_matrix()
        np.testing.assert_allclose(np.linalg.inv(output_transforms[1]) @ output_transforms[0],
                                   delta @ np.linalg.inv(self.transforms[1]) @ self.transforms[0], atol=1e-14)
        (self.root / 'imu.yaml').write_text((self.root / 'imu.yaml').read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            prepare_calibration(self.root, self.root, 800, refinement)

    def test_image_only_fit_recovers_known_rotation(self):
        rng = np.random.default_rng(0)
        points = rng.uniform([-4, -3, 6], [4, 3, 30], (1000, 3))
        nominal = np.linalg.inv(self.transforms[1]) @ self.transforms[0]
        correction = np.array([.002, -.003, .004])
        delta = Rotation.from_rotvec(correction).as_matrix()
        right = (points @ nominal[:3, :3].T + nominal[:3, 3]) @ delta.T
        rays = [points / points[:, 2, None], right / right[:, 2, None]]
        fit = least_squares(lambda v: epipolar_error(v, rays, nominal, 920.), np.zeros(3))
        np.testing.assert_allclose(fit.x, correction, atol=1e-9)

    def test_rectification_keeps_metric_baseline_and_parallel_camera_axes(self):
        calibration, _, _ = prepare_calibration(self.root, self.root, 800, rectify=True)
        poses = []
        for c in calibration['T_imu_cam']:
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat([c[k] for k in ['qx', 'qy', 'qz', 'qw']]).as_matrix()
            transform[:3, 3] = [c[k] for k in ['px', 'py', 'pz']]
            poses.append(transform)
        relative = np.linalg.inv(poses[1]) @ poses[0]
        np.testing.assert_allclose(relative[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(relative[:3, 3], [-.43, 0, 0], atol=1e-12)
        self.assertEqual(calibration['intrinsics'][0], calibration['intrinsics'][1])


if __name__ == '__main__':
    unittest.main()
