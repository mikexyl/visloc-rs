"""Kalibr stereo/IMU calibration -> pinhole images and Basalt DS (xi=alpha=0)."""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def prepare(calib_dir, output):
    calib_dir, output = Path(calib_dir), Path(output)
    chain = yaml.safe_load((calib_dir/'camchain-imucam.yaml').read_text())
    imu = yaml.safe_load((calib_dir/'imu.yaml').read_text())['imu0']
    cams = [chain['cam0'], chain['cam1']]
    if any(c['resolution'] != [640,480] or c['camera_model'] != 'pinhole' or c['distortion_model'] != 'radtan' for c in cams):
        raise ValueError('Expected calibrated 640x480 pinhole/radtan infrared pair')
    shifts = [c['timeshift_cam_imu'] for c in cams]
    if abs(shifts[0]-shifts[1]) > 0.0001:
        raise ValueError('Camera/IMU time shifts disagree by more than 0.1 ms')
    transforms, intrinsics, maps = [], [], []
    for c in cams:
        fx,fy,cx,cy=c['intrinsics']
        k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        # Keep camera axes and intrinsics; remove residual radtan distortion only.
        maps.append(cv2.initUndistortRectifyMap(k,np.array(c['distortion_coeffs']),np.eye(3),k,(640,480),cv2.CV_32FC1))
        cam_imu=np.array(c['T_cam_imu'])
        if not np.allclose(cam_imu[3],[0,0,0,1]) or not np.allclose(cam_imu[:3,:3].T@cam_imu[:3,:3],np.eye(3),atol=1e-6):
            raise ValueError('Invalid rigid calibration transform')
        imu_cam=np.linalg.inv(cam_imu)
        q=Rotation.from_matrix(imu_cam[:3,:3]).as_quat()
        transforms.append(dict(zip(('px','py','pz','qx','qy','qz','qw'),[*imu_cam[:3,3],*q])))
        intrinsics.append({'camera_type':'ds','intrinsics':dict(fx=fx,fy=fy,cx=cx,cy=cy,xi=0.,alpha=0.)})
    identity=dict(px=0.,py=0.,pz=0.,qx=0.,qy=0.,qz=0.,qw=1.)
    shift_ns=round(np.mean(shifts)*1e9)
    value={'T_imu_cam':transforms,'intrinsics':intrinsics,'resolution':[[640,480]]*2,
           'calib_accel_bias':[0.]*9,'calib_gyro_bias':[0.]*12,'imu_update_rate':imu['update_rate'],
           'accel_noise_std':[imu['accelerometer_noise_density']]*3,
           'gyro_noise_std':[imu['gyroscope_noise_density']]*3,
           'accel_bias_std':[imu['accelerometer_random_walk']]*3,
           'gyro_bias_std':[imu['gyroscope_random_walk']]*3,
           'T_mocap_world':identity,'T_imu_marker':identity,'mocap_time_offset_ns':0,'mocap_to_imu_offset_ns':0,
           # The capture bridge applies the additive correction once before delivery.
           'cam_time_offset_ns':0}
    output.mkdir(parents=True,exist_ok=True)
    (output/'basalt_calibration.json').write_text(json.dumps({'value0':value},indent=2))
    source_hashes={name:hashlib.sha256((calib_dir/name).read_bytes()).hexdigest() for name in ('camchain-imucam.yaml','imu.yaml')}
    positions=np.array([[t[k] for k in ('px','py','pz')] for t in transforms])
    report={'source_sha256':source_hashes,'source':str(calib_dir),'camera_time_shift_ns':shift_ns,'camera_time_shifts_s':shifts,
            'baseline_m':float(np.linalg.norm(positions[0]-positions[1])),
            'image_processing':'radtan undistortion, unchanged camera axes; exact pinhole DS',
            'imu_noise_source':'imu.yaml (not alternate Allan file)','imu_rate_hz':imu['update_rate']}
    (output/'calibration_report.json').write_text(json.dumps(report,indent=2))
    return maps, shift_ns, value
