"""Calibration and chronology contract for the optional NVIDIA comparison."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from run_cuvslam_realsense import make_rig,imu_end_index
except ModuleNotFoundError as error:
    if error.name != 'cuvslam':
        raise
    make_rig = None


@unittest.skipIf(make_rig is None, 'Optional NVIDIA Python wheel is not installed')
class CuVslamReplayTests(unittest.TestCase):
    def test_rig_is_body_frame_without_inverting_camera_extrinsics(self):
        calibration = {
            'T_imu_cam': [dict(px=x, py=0.006, pz=0.019, qx=0., qy=0.,
                               qz=np.sin(0.15), qw=np.cos(0.15)) for x in [-0.031, 0.064]],
            'calib_accel_bias': [0.] * 9, 'calib_gyro_bias': [0.] * 12,
            'intrinsics': [{'intrinsics': dict(fx=391., fy=392., cx=320., cy=240.,
                                              xi=0., alpha=0.)}] * 2,
            'resolution': [[640, 480]] * 2, 'imu_update_rate': 200,
            'gyro_noise_std': [0.00014] * 3, 'gyro_bias_std': [0.0000017] * 3,
            'accel_noise_std': [0.0077] * 3, 'accel_bias_std': [0.00045] * 3,
        }
        rig=make_rig(calibration)
        for camera,expected in zip(rig.cameras,calibration['T_imu_cam']):
            np.testing.assert_allclose(camera.rig_from_camera.translation,[expected[k] for k in ['px','py','pz']],atol=1e-8)
            np.testing.assert_allclose(camera.rig_from_camera.rotation,[expected[k] for k in ['qx','qy','qz','qw']],atol=1e-7)
        np.testing.assert_array_equal(rig.imus[0].rig_from_imu.translation,[0,0,0])
        np.testing.assert_array_equal(rig.imus[0].rig_from_imu.rotation,[0,0,0,1])
        self.assertAlmostEqual(rig.imus[0].frequency,calibration['imu_update_rate'])
        self.assertAlmostEqual(rig.imus[0].gyroscope_noise_density,calibration['gyro_noise_std'][0])

    def test_never_submit_future_imu_before_image(self):
        start=1790415501233623047
        times=np.array([start-1,start,start+1,start+5000000],dtype=np.int64)
        self.assertEqual(imu_end_index(times,start),2)
        self.assertEqual(imu_end_index(times,start-2),0)
        self.assertEqual(imu_end_index(times,start+5000000),4)


if __name__=='__main__':unittest.main()
