#!/usr/bin/env python3
"""Run NVIDIA's Python stereo-inertial odometry on our prepared sensor cache.

Uses the same paired images, calibrated undistortion, IMU samples, and integer
timestamp mapping as the native Rust replay. No GPS or trajectory reference is
read. cuVSLAM manages its own initialization and bias estimation.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import cv2
import cuvslam
import numpy as np

import replay_sensor_source
from replay_sensor_source import open_source


def make_rig(calibration):
    """Rig frame equals the recorded IMU/body frame, matching our pose output."""
    if len(calibration['T_imu_cam']) != 2:
        raise ValueError('Expected two calibrated cameras')
    if any(calibration['calib_accel_bias']) or any(calibration['calib_gyro_bias']):
        raise ValueError('Affine IMU calibration requires explicit conversion')
    cameras = []
    for i, transform in enumerate(calibration['T_imu_cam']):
        values = calibration['intrinsics'][i]['intrinsics']
        if values['xi'] != 0 or values['alpha'] != 0:
            raise ValueError('Expected pinhole calibration after undistortion')
        camera = cuvslam.Camera()
        camera.size = calibration['resolution'][i]
        camera.focal = [values['fx'], values['fy']]
        camera.principal = [values['cx'], values['cy']]
        camera.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole, [])
        camera.rig_from_camera = cuvslam.Pose(
            rotation=[transform[k] for k in ['qx', 'qy', 'qz', 'qw']],
            translation=[transform[k] for k in ['px', 'py', 'pz']])
        cameras.append(camera)
    imu = cuvslam.ImuCalibration()
    imu.rig_from_imu = cuvslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
    for target, source in [('gyroscope_noise_density', 'gyro_noise_std'),
                           ('gyroscope_random_walk', 'gyro_bias_std'),
                           ('accelerometer_noise_density', 'accel_noise_std'),
                           ('accelerometer_random_walk', 'accel_bias_std')]:
        values = calibration[source]
        if not np.allclose(values, values[0], rtol=0, atol=0) or values[0] <= 0:
            raise ValueError('cuVSLAM requires a positive scalar noise value per sensor')
        setattr(imu, target, values[0])
    imu.frequency = calibration['imu_update_rate']
    rig = cuvslam.Rig()
    rig.cameras = cameras
    rig.imus = [imu]
    return rig


def maps_for(config, calibration):
    maps = []
    for i, name in enumerate(['preprocess', 'preprocess_right']):
        c = config[name]
        if [c['raw_width'], c['raw_height']] != calibration['resolution'][i]:
            raise ValueError('This comparison expects native calibrated resolution')
        fx, fy, cx, cy = c['intrinsics']
        k = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        maps.append(cv2.initUndistortRectifyMap(
            k, np.array(c['distortion']), np.eye(3), k,
            tuple(calibration['resolution'][i]), cv2.CV_32FC1))
    return maps


def imu_end_index(times, image_ns):
    # cuVSLAM requires globally chronological, serialized Track/Register calls.
    # Unlike a buffering ROS adapter, never submit a future IMU sample first.
    return int(np.searchsorted(times, image_ns, side='right'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mission', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int)
    args = parser.parse_args()
    mission = json.loads(args.mission.read_text())
    if len(mission['robots']) != 1:
        parser.error('Select a single-robot mission')
    robot = dict(mission['robots'][0])
    if robot.get('camera_mode') != 'stereo':
        parser.error('Expected a stereo mission')
    if robot.get('input_format') != 'realsense_cache':
        parser.error('Expected a prepared RealSense sensor cache')
    if args.frames is not None:
        if not 0 < args.frames <= robot['frames']:
            parser.error('Invalid frame limit')
        robot['frames'] = args.frames
    config = json.loads(Path(robot['config']).read_text())
    calibration = json.loads(Path(config['calibration']).read_text())['value0']
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    rig = make_rig(calibration)
    maps = maps_for(config, calibration)
    odom_config = cuvslam.Tracker.OdometryConfig(
        odometry_mode=cuvslam.Tracker.OdometryMode.Inertial,
        async_sba=False, rectified_stereo_camera=False,
        enable_observations_export=True, enable_landmarks_export=True)
    effective = dict(
        library_version=cuvslam.get_version(), package_version=importlib.metadata.version('cuvslam'),
        input_mission=str(args.mission.resolve()), input_robot=robot,
        rig_frame='recorded IMU/body; rig_from_imu is identity',
        odometry_mode='Inertial', slam_enabled=False, async_sba=False,
        rectified_stereo_camera=False, use_gpu=odom_config.use_gpu,
        use_denoising=odom_config.use_denoising, use_motion_model=odom_config.use_motion_model,
        max_frame_delta_s=odom_config.max_frame_delta_s, multicam_mode=str(odom_config.multicam_mode),
        debug_imu_mode=odom_config.debug_imu_mode,
        initialization='cuVSLAM native; no Basalt state or estimated bias supplied',
        imu_noise_source='same effective calibration as our stereo VIO, including 0.75 multipliers',
        sensor_order='Each IMU measurement <= current image timestamp, then Track; no concurrent calls',
        evaluation='No GPS, no ground-truth input, no ATE',
        packages={name:importlib.metadata.version(name) for name in ['numpy','opencv-python-headless','scipy']})
    (root/'config.json').write_text(json.dumps(effective, indent=2)+'\n')
    for name in ['calibration', 'vio_config']:
        shutil.copy2(config[name], root/Path(config[name]).name)
    shutil.copy2(robot['config'], root/'source_robot_config.json')
    shutil.copy2(__file__, root/'run_cuvslam_realsense.py')
    shutil.copy2(replay_sensor_source.__file__, root/'replay_sensor_source.py')
    source = open_source(robot)
    assert len(source.frames) == robot['frames']
    status = dict(status='running', input_pairs=len(source.frames), processed_pairs=0)
    (root/'run_status.json').write_text(json.dumps(status, indent=2)+'\n')
    initialize_started = time.perf_counter()
    tracker = cuvslam.Tracker(rig, odom_config, slam_config=None)
    initialization_ms = (time.perf_counter()-initialize_started)*1000
    started = time.perf_counter()
    imu_index = 0
    pose_count = 0
    failed = []
    last_timestamp = None
    timing = []
    pixel_hash = hashlib.sha256()
    imu_hash = hashlib.sha256()
    try:
        with (root/'trajectory.csv').open('w', buffering=1) as csv_file, \
             (root/'tracking.jsonl').open('w', buffering=1) as trace:
            writer = csv.writer(csv_file)
            writer.writerow(['frame_id','timestamp_ns','tx','ty','tz','qw','qx','qy','qz'])
            for index, (original_ns, left, right) in enumerate(source.frames):
                frame_started = time.perf_counter()
                timestamp = int(original_ns+robot['offset_ns'])
                assert right is not None
                images = []
                for i, raw in enumerate([source.image(left), source.right_image(right)]):
                    gray = np.asarray(raw.data).reshape(raw.height, raw.width)
                    image = cv2.remap(gray, *maps[i], cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    images.append(image)
                    pixel_hash.update(image.tobytes())
                prepared = time.perf_counter()
                end = imu_end_index(source.times, original_ns)
                count = end-imu_index
                for sample in source.imu[imu_index:end]:
                    stamp = int(sample[0]+robot['offset_ns'])
                    assert last_timestamp is None or stamp >= last_timestamp
                    tracker.register_imu_measurement(0, cuvslam.ImuMeasurement(
                        timestamp_ns=stamp, angular_velocities=sample[1:4], linear_accelerations=sample[4:7]))
                    imu_hash.update(np.asarray([sample[0]],dtype='<i8').tobytes())
                    imu_hash.update(np.asarray(sample[1:],dtype='<f8').tobytes())
                    last_timestamp = stamp
                imu_index = end
                assert last_timestamp is None or timestamp >= last_timestamp
                before_track = time.perf_counter()
                pose_estimate, _ = tracker.track(timestamp, images)
                after_track = time.perf_counter()
                last_timestamp = timestamp
                # Failed tracking can return an unset/stale pose timestamp.
                # Preserve that evidence and continue feeding the recording;
                # never assign a stale pose to the current image timestamp.
                pose_available = pose_estimate.world_from_rig is not None
                valid = pose_available and pose_estimate.timestamp_ns == timestamp
                if valid:
                    pose = pose_estimate.world_from_rig.pose
                    p, q = list(pose.translation), list(pose.rotation)
                    if not np.isfinite([*p,*q]).all():
                        raise RuntimeError('Nonfinite cuVSLAM pose')
                    writer.writerow([index,timestamp,*p,q[3],*q[:3]])
                    pose_count += 1
                else:
                    failed.append(index)
                state = tracker.odom.get_state()
                # A returned pose can briefly be IMU-propagated. Record visual
                # support and warmup instead of equating all poses with visual success.
                observations = state.observations
                row = dict(frame_id=index,timestamp_ns=timestamp,original_timestamp_ns=int(original_ns),
                    valid_pose=valid,imu_samples=count,state_frame_id=state.frame_id,
                    returned_pose_timestamp_ns=pose_estimate.timestamp_ns,pose_available=pose_available,
                    warming_up=state.warming_up,keyframe=state.keyframe,
                    left_tracks=sum(o.camera_index==0 for o in observations),
                    right_tracks=sum(o.camera_index==1 for o in observations),
                    gravity=None if state.gravity is None else list(state.gravity),
                    last_gravity=tracker.get_last_gravity(),
                    preprocess_ms=(prepared-frame_started)*1000,
                    imu_submit_ms=(before_track-prepared)*1000,track_ms=(after_track-before_track)*1000)
                row['pipeline_ms']=(time.perf_counter()-frame_started)*1000
                timing.append(row['track_ms'])
                trace.write(json.dumps(row, allow_nan=False)+'\n')
                status.update(processed_pairs=index+1,valid_poses=pose_count,invalid_poses=len(failed))
                if index % 250 == 0 or index+1==len(source.frames):
                    print(f'cuVSLAM pairs={index+1}/{len(source.frames)} valid={pose_count} tracks={row["left_tracks"]}/{row["right_tracks"]}',flush=True)
                    (root/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')
        status.update(status='completed',processed_pairs=len(source.frames),valid_poses=pose_count,
            invalid_poses=len(failed),invalid_frame_ids=failed,imu_samples_submitted=imu_index,
            imu_samples_after_final_image=len(source.times)-imu_index,
            initialization_ms=initialization_ms,wall_seconds=time.perf_counter()-started,
            track_median_ms=float(np.median(timing)),track_p95_ms=float(np.percentile(timing,95)),
            undistorted_pixels_sha256=pixel_hash.hexdigest(),submitted_imu_sha256=imu_hash.hexdigest())
    except BaseException as error:
        status.update(status='failed',error=repr(error))
        raise
    finally:
        (root/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')
        source.close()
    print(json.dumps(status,indent=2))


if __name__ == '__main__':
    main()
