#!/usr/bin/env python3
"""Prepare a recorded RealSense ROS2 MCAP bag for our sequence-refinement SLAM.

Left IR (or exact-timestamp stereo pairs) and synchronized IMU enter the estimator.
GPS is exported separately for evaluation; original compressed images and
integer sensor timestamps are preserved in the read-only replay cache.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import yaml

from realsense.calibration import prepare
from realsense.sync import interpolate_imu

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools/browser_viewer'))
from gps_monitor import parse_gga

LEFT = '/camera/camera/infra1/image_rect_raw/compressed'
RIGHT = '/camera/camera/infra2/image_rect_raw/compressed'
GYRO = '/camera/camera/gyro/sample'
ACCEL = '/camera/camera/accel/sample'


def stamp(message):
    return int(message.header.stamp.sec) * 10**9 + int(message.header.stamp.nanosec)


def vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def ordered_samples(samples, name):
    original = [s[0] for s in samples]
    ordered = sorted(samples)
    times = np.array([s[0] for s in ordered], dtype=np.int64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f'Duplicate/missing {name} timestamps')
    if not np.isfinite(np.asarray([s[1] for s in ordered])).all():
        raise ValueError(f'Nonfinite {name} values')
    return ordered, dict(count=len(ordered), reordered=original != times.tolist(),
                        first_ns=int(times[0]), last_ns=int(times[-1]),
                        median_interval_ms=float(np.median(np.diff(times))) / 1e6,
                        max_interval_ms=float(np.max(np.diff(times))) / 1e6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration-dir', type=Path, default=REPO / 'configs/realsense/d455_1')
    p.add_argument('--loop-config', type=Path, default=REPO / '.runtime/multi_robot_models/loop_config.json')
    p.add_argument('--rate', type=float, default=1.)
    p.add_argument('--camera-mode', choices=['mono', 'stereo'], default='mono')
    p.add_argument('--trim-imu-boundaries', action='store_true',
                   help='Exclude boundary images outside synchronized IMU coverage; preserve them in the cache and report exclusions')
    args = p.parse_args()
    if not np.isfinite(args.rate) or args.rate <= 0:
        p.error('rate must be finite and positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = output / 'sensors'
    cache.mkdir()
    calibration_dir = output / 'calibration'
    _, shift, calibration = prepare(args.calibration_dir, calibration_dir)
    for key in ['accel_noise_std', 'gyro_noise_std', 'accel_bias_std', 'gyro_bias_std']:
        calibration[key] = [v * .75 for v in calibration[key]]
    (calibration_dir / 'basalt_calibration.json').write_text(json.dumps({'value0': calibration}, indent=2) + '\n')
    report = json.loads((calibration_dir / 'calibration_report.json').read_text())
    report.update(imu_noise_scale=.75, imu_bias_scale=.75,
                  camera_mode=args.camera_mode, input_timestamp='ROS header global_time + calibrated camera offset')
    (calibration_dir / 'calibration_report.json').write_text(json.dumps(report, indent=2) + '\n')
    gyro, accel, fixes, nmea, metadata = [], [], [], [], {}
    connection = sqlite3.connect(cache / 'images.sqlite3')
    connection.execute('CREATE TABLE images(timestamp_ns INTEGER PRIMARY KEY, source_timestamp_ns INTEGER, bag_timestamp_ns INTEGER, data BLOB NOT NULL)')
    if args.camera_mode == 'stereo':
        connection.execute('CREATE TABLE right_images(timestamp_ns INTEGER PRIMARY KEY, source_timestamp_ns INTEGER, bag_timestamp_ns INTEGER, data BLOB NOT NULL)')
    topics = {LEFT, GYRO, ACCEL, '/gps/fix', '/gps/nmea', '/tf_static',
              '/camera/camera/infra1/camera_info', '/camera/camera/infra1/metadata',
              '/camera/camera/extrinsics/depth_to_gyro', '/camera/camera/extrinsics/depth_to_accel',
              '/network/identity'}
    if args.camera_mode == 'stereo':
        topics.update({RIGHT, '/camera/camera/infra2/camera_info', '/camera/camera/infra2/metadata'})
    images = 0
    right_images = 0
    camera_times = []
    with AnyReader([args.bag.resolve()], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for c, bag_time, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, c.msgtype)
            if c.topic in (LEFT, RIGHT):
                if not message.format.startswith('mono8;'):
                    raise ValueError(f'Expected recorded mono8 infrared, got {message.format}')
                source_time = stamp(message)
                corrected = source_time + shift
                if (images if c.topic == LEFT else right_images) == 0:
                    image = cv2.imdecode(np.asarray(message.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if image is None or image.shape != (480, 640) or image.dtype != np.uint8:
                        raise ValueError('Recorded image resolution/format does not match D455 calibration')
                table = 'images' if c.topic == LEFT else 'right_images'
                connection.execute(f'INSERT INTO {table} VALUES (?, ?, ?, ?)',
                                   (corrected, source_time, bag_time, bytes(message.data)))
                if c.topic == RIGHT:
                    right_images += 1
                    continue
                camera_times.append(corrected)
                images += 1
                if images % 1000 == 0:
                    connection.commit()
                    print(f'Cached {images} left IR frames', flush=True)
            elif c.topic == GYRO:
                if message.header.frame_id != 'camera_gyro_optical_frame':
                    raise ValueError('Unexpected gyro coordinate frame')
                gyro.append((stamp(message), vec(message.angular_velocity)))
            elif c.topic == ACCEL:
                if message.header.frame_id != 'camera_accel_optical_frame':
                    raise ValueError('Unexpected accel coordinate frame')
                accel.append((stamp(message), vec(message.linear_acceleration)))
            elif c.topic == '/gps/fix':
                fixes.append(dict(timestamp_ns=stamp(message), bag_timestamp_ns=bag_time,
                                  status=int(message.status.status), latitude=message.latitude,
                                  longitude=message.longitude, altitude=message.altitude,
                                  covariance_type=int(message.position_covariance_type)))
            elif c.topic == '/gps/nmea':
                for line in message.data.splitlines():
                    fix = parse_gga(line)
                    if fix is not None:
                        nmea.append(dict(bag_timestamp_ns=bag_time, **fix))
            elif c.topic not in metadata:
                if c.topic.endswith('metadata'):
                    metadata[c.topic] = json.loads(message.json_data)
                elif c.topic.endswith('identity'):
                    metadata[c.topic] = json.loads(message.data)
                elif c.topic.endswith('camera_info'):
                    metadata[c.topic] = dict(width=int(message.width), height=int(message.height),
                                            k=message.k.tolist(), d=message.d.tolist(), r=message.r.tolist())
                elif 'extrinsics' in c.topic:
                    metadata[c.topic] = dict(rotation=message.rotation.tolist(), translation=message.translation.tolist())
                else:
                    metadata[c.topic] = [dict(parent=t.header.frame_id, child=t.child_frame_id,
                                             translation=vec(t.transform.translation),
                                             rotation_xyzw=[t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w])
                                        for t in message.transforms]
    connection.commit()
    pairing = None
    if args.camera_mode == 'stereo':
        camera_times = [t for (t,) in connection.execute('SELECT timestamp_ns FROM images INNER JOIN right_images USING(timestamp_ns) ORDER BY timestamp_ns')]
        pairing = dict(policy='exact header timestamp; no nearest-neighbor reassignment',
                       pairs=len(camera_times), left_images=images, right_images=right_images,
                       unpaired_left=images-len(camera_times), unpaired_right=right_images-len(camera_times))
        if len(camera_times) < 2:
            raise ValueError('No usable exact-timestamp stereo sequence')
    connection.close()
    # Gyro and accel topics use the same optical axes; their recorded factory
    # transforms must agree before applying the existing Kalibr IMU calibration.
    if metadata['/camera/camera/extrinsics/depth_to_gyro'] != metadata['/camera/camera/extrinsics/depth_to_accel']:
        raise ValueError('Separate gyro/accel axes differ; explicit IMU transformation required')
    if metadata['/camera/camera/infra1/metadata']['clock_domain'] != 'global_time':
        raise ValueError('Expected consistent ROS global-time sensor stamps')
    if args.camera_mode == 'stereo' and metadata['/camera/camera/infra2/metadata']['clock_domain'] != 'global_time':
        raise ValueError('Right camera clock domain differs')
    gyro, gyro_stats = ordered_samples(gyro, 'gyro')
    accel, accel_stats = ordered_samples(accel, 'accel')
    start = max(gyro[0][0], accel[0][0])
    end = min(gyro[-1][0], accel[-1][0])
    combined = interpolate_imu(gyro, accel, start - 1, end)
    timestamps = np.array([s[0] for s in combined], dtype=np.int64)
    values = np.array([s[1:] for s in combined], dtype=np.float64)
    np.savez(cache / 'imu.npz', timestamps=timestamps, values=values)
    camera_times.sort()
    boundary_exclusions = dict(before_imu_ns=[], after_imu_ns=[])
    if args.trim_imu_boundaries:
        boundary_exclusions['before_imu_ns'] = [t for t in camera_times if t < timestamps[0]]
        boundary_exclusions['after_imu_ns'] = [t for t in camera_times if t > timestamps[-1]]
        camera_times = [t for t in camera_times if timestamps[0] <= t <= timestamps[-1]]
        if len(camera_times) < 2:
            raise ValueError('Fewer than two images within synchronized IMU coverage')
    if timestamps[0] > camera_times[1] or timestamps[-1] < camera_times[-1]:
        raise ValueError('Synchronized IMU does not cover the complete recorded image interval')
    with (output / 'gps_fixes.jsonl').open('w') as f:
        for row in fixes:
            # Missing NavSatFix values are represented by null, never supplied
            # to the sensor replay or interpreted as a zero-valued position.
            row = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
            f.write(json.dumps(row, allow_nan=False) + '\n')
    with (output / 'gps_gga.jsonl').open('w') as f:
        for row in nmea:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    audit = dict(bag=str(args.bag.resolve()), camera_frames=len(camera_times), camera_mode=args.camera_mode,
                 stereo_pairing=pairing, gyro=gyro_stats, accel=accel_stats,
                 trim_imu_boundaries=args.trim_imu_boundaries, imu_boundary_exclusions=boundary_exclusions,
                 synchronized_imu_samples=len(combined), gyro_samples_outside_accel_coverage=len(gyro)-len(combined),
                 first_camera_ns=camera_times[0], last_camera_ns=camera_times[-1],
                 camera_time_shift_ns=shift, camera_max_gap_ms=float(np.max(np.diff(camera_times)))/1e6,
                 gps_fixes=len(fixes), gps_status_counts=dict(Counter(f['status'] for f in fixes)),
                 gga_quality_counts=dict(Counter(f['quality'] for f in nmea)), metadata=metadata)
    (output / 'input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    config = json.loads(args.loop_config.read_text())
    config['min_similarity'] = .8
    for name in ['jist_engine', 'xfeat_engine', 'lighterglue_engine']:
        model = Path(config[name])
        config[name] = str((args.loop_config.parent / model).resolve() if not model.is_absolute() else model)
    (output / 'loop_config.json').write_text(json.dumps(config, indent=2) + '\n')
    if args.loop_config.with_name('manifest.json').exists():
        shutil.copy2(args.loop_config.with_name('manifest.json'), output / 'manifest.json')
    shutil.copy2(REPO / 'configs/graco/aerial_vio.json', output / 'vio_config.json')
    chain = yaml.safe_load((args.calibration_dir / 'camchain-imucam.yaml').read_text())
    robot = 'ucy01'
    robot_config = dict(robot=robot, peers=[robot], calibration=str(calibration_dir / 'basalt_calibration.json'),
                        vio_config=str(output / 'vio_config.json'), loop_config=str(output / 'loop_config.json'),
                        output=str(output / 'robots' / robot), reliable_sensors=True, loop_enabled=True,
                        camera_mode=args.camera_mode, fixed_last_frame=False, preprocess=dict(raw_width=640, raw_height=480,
                            intrinsics=chain['cam0']['intrinsics'], distortion=chain['cam0']['distortion_coeffs']))
    if args.camera_mode == 'stereo':
        robot_config['preprocess_right'] = dict(raw_width=640, raw_height=480,
            intrinsics=chain['cam1']['intrinsics'], distortion=chain['cam1']['distortion_coeffs'])
    (output / f'{robot}.json').write_text(json.dumps(robot_config, indent=2) + '\n')
    (output / 'backend.json').write_text(json.dumps(dict(peers=[robot], output=str(output / 'backend'),
                                                       pgo=dict(min_loop_similarity=.8)), indent=2) + '\n')
    mission = dict(rate=args.rate, epoch_ns=10**9, backend_config=str(output / 'backend.json'),
                   robots=[dict(robot=robot, bag=str(args.bag.resolve()), input_format='realsense_cache',
                                sensor_cache=str(cache), config=str(output / f'{robot}.json'), frames=len(camera_times),
                                camera_mode=args.camera_mode,
                                original_first_ns=camera_times[0], original_last_ns=camera_times[-1],
                                offset_ns=10**9-camera_times[0])])
    (output / 'mission.json').write_text(json.dumps(mission, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'metadata'}, indent=2))
    print(output / 'mission.json')


if __name__ == '__main__':
    main()
