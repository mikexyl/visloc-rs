"""The adapter's time/frame contract; dense estimation is tested upstream."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_scarf_mapping import camera_pose_at, image_stamp, load_body_poses


class ScarfInputsTest(unittest.TestCase):
    def test_epoch_nanoseconds_and_replay_offset_round_trip(self):
        origin = 1790502544606403113
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'body.csv'
            with path.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['timestamp_ns', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
                writer.writerows([[10**9, 0, 0, 0, 0, 0, 0, 1], [10**9 + 33_333_333, 1, 0, 0, 0, 0, 0, 1]])
            stamps, _ = load_body_poses(path, origin - 10**9)
            self.assertEqual(stamps.tolist(), [origin, origin + 33_333_333])
            for stamp in stamps:
                local = int(stamp) - origin
                name = Path(f'image_{local // 10**9:010d}_{local % 10**9:09d}.jpg')
                self.assertEqual(image_stamp(name) + origin, stamp)

    def test_rotating_body_composes_rgb_lever_arm_after_interpolation(self):
        origin = 1790502544606403113
        stamps = np.array([origin, origin + 200_000_000], dtype=np.int64)
        poses = np.c_[[[0, 0, 0], [2, 0, 0]], Rotation.from_euler('z', [[0], [90]], degrees=True).as_quat()]
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = [1, 0, 0]
        extrinsic[:3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
        pose = camera_pose_at(origin + 100_000_000, stamps, poses, extrinsic)
        np.testing.assert_allclose(pose[:3], [1 + 2**-.5, 2**-.5, 0])
        expected = Rotation.from_euler('z', 45, degrees=True) * Rotation.from_euler('x', 30, degrees=True)
        np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), expected.as_matrix(), atol=1e-12)

    def test_extrapolation_and_large_gaps_rejected_but_exact_endpoints_kept(self):
        stamps = np.array([1_000_000_000, 2_000_000_000])
        poses = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]])
        for stamp in [stamps[0] - 1, 1_500_000_000, stamps[-1] + 1]:
            self.assertIsNone(camera_pose_at(stamp, stamps, poses, np.eye(4)))
        for stamp in stamps:
            self.assertIsNotNone(camera_pose_at(stamp, stamps, poses, np.eye(4)))


if __name__ == '__main__':
    unittest.main()
