import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/realsense'))
from sync import interpolate_imu
from calibration import prepare
import tempfile
import yaml

class SyncTest(unittest.TestCase):
    def test_interpolation_and_half_open_intervals(self):
        gyro=[(5_000_000,(1,2,3)),(10_000_000,(2,3,4)),(15_000_000,(3,4,5))]
        accel=[(0,(0,0,0)),(10_000_000,(2,4,6)),(20_000_000,(4,8,12))]
        first=interpolate_imu(gyro,accel,0,10_000_000)
        second=interpolate_imu(gyro,accel,10_000_000,20_000_000)
        self.assertEqual([r[0] for r in first+second],[5_000_000,10_000_000,15_000_000])
        np.testing.assert_allclose(first[0][4:],[1,2,3])
        np.testing.assert_allclose(second[0][4:],[3,6,9])
    def test_no_extrapolation_or_large_gaps(self):
        with self.assertRaises(ValueError):
            interpolate_imu([(5,(0,0,0))],[(10,(0,0,0)),(20,(0,0,0))],0,6)
        with self.assertRaises(ValueError):
            interpolate_imu([(100_000_000,(0,0,0))],[(0,(0,0,0)),(200_000_000,(0,0,0))],0,100_000_000)
    def test_calibration_direction_and_clock(self):
        root=Path(__file__).resolve().parents[1]/'configs/realsense/d455_1'
        with tempfile.TemporaryDirectory() as tmp:
            maps,shift,calib=prepare(root,tmp)
            self.assertEqual(shift,-9590295)
            self.assertEqual(calib['calib_gyro_bias'],[0.]*12)
            self.assertEqual(calib['cam_time_offset_ns'],0)
            for i in range(2):
                original=np.array(yaml.safe_load((root/'camchain-imucam.yaml').read_text())[f'cam{i}']['T_cam_imu'])
                transform=calib['T_imu_cam'][i]
                from scipy.spatial.transform import Rotation
                converted=np.eye(4)
                converted[:3,:3]=Rotation.from_quat([transform[k] for k in ['qx','qy','qz','qw']]).as_matrix()
                converted[:3,3]=[transform[k] for k in ['px','py','pz']]
                np.testing.assert_allclose(original@converted,np.eye(4),atol=1e-10)
                self.assertEqual(maps[i][0].shape,(480,640))
if __name__=='__main__': unittest.main()
