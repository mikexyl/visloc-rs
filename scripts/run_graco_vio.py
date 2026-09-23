#!/usr/bin/env python3
"""Replay a GRACO aerial or ground ROS2 SQLite bag through visloc's Rust VIO.

Reads the source bag without modification or image extraction. Only selected
grayscale cameras and IMU gyro/acceleration enter the estimator. Ground truth is
used separately by visualization and by the final trajectory evaluation.
"""
import argparse
from collections import deque
import colorsys
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation
import yaml

from evaluate_euroc_trajectory import evaluate
from online_da3 import Config as DepthConfig, OnlineDepth, keyframe_from_vio, log_result as log_depth_result

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = Path('/data/graco/aerial-calibration-20251121T084428Z-1-001/aerial-calibration')
GROUND_CALIBRATION = Path('/data/graco/ground-calibration')


def calibration_for_bag(bag, override=None):
    if override is not None:
        return override
    if bag.name.startswith('ground-'):
        return GROUND_CALIBRATION
    if bag.name.startswith('aerial-'):
        return DEFAULT_CALIBRATION
    raise ValueError('Cannot infer the GRACO rig from the bag name; specify --calibration-dir')


def stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def xyz(value):
    return [value.x, value.y, value.z]


def prepare_calibration(source, output, width, stereo_refinement=None, rectify=False,
                        imu_noise_scale=1.0, imu_bias_scale=1.0):
    if any(not np.isfinite(v) or v <= 0 for v in (imu_noise_scale, imu_bias_scale)):
        raise ValueError('IMU noise and bias scales must be finite and positive')
    names = ('stereo.yaml', 'stereo-imu.yaml', 'imu.yaml')
    stereo, extrinsics, imu = [yaml.safe_load((source / n).read_text()) for n in names]
    source_hashes = {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in names}
    refinement = json.loads(stereo_refinement.read_text()) if stereo_refinement else None
    if refinement and refinement.get('schema') != 'visloc.graco.stereo_rotation_refinement.v1':
        raise ValueError('Unsupported stereo refinement schema')
    if refinement and refinement['source_sha256'] != source_hashes:
        raise ValueError('Stereo refinement does not match the source calibration')
    if refinement:
        correction_vector = np.asarray(refinement['right_rotation_correction_rotvec'])
        if correction_vector.shape != (3,) or not np.isfinite(correction_vector).all() or np.linalg.norm(correction_vector) > np.deg2rad(2):
            raise ValueError('Invalid stereo rotation correction')
    transforms, intrinsics, maps, resolutions = [], [], [], []
    for i in range(2):
        camera = stereo[f'cam{i}']
        if camera['camera_model'] != 'pinhole' or camera['distortion_model'] != 'radial-tangential':
            raise ValueError('Expected GRACO pinhole/radial-tangential calibration')
        w, h = camera['resolution']
        height = round(h * width / w)
        fx, fy, cx, cy = camera['intrinsics']
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        # Resize first using area filtering. OpenCV's resize pixel centers are
        # (p + .5) * scale - .5; carry that convention into the intrinsics.
        scaled = k.copy()
        scaled[0] *= width / w
        scaled[1] *= height / h
        scaled[0, 2] += .5 * (width / w - 1)
        scaled[1, 2] += .5 * (height / h - 1)
        maps.append(cv2.initUndistortRectifyMap(
            scaled, np.array(camera['distortion_coeffs']), np.eye(3), scaled,
            (width, height), cv2.CV_32FC1))
        transform = np.array(extrinsics[f'T_Imu_cam{i}']['data']).reshape(4, 4)
        if i == 1 and refinement:
            # Rotate the right camera axes about its optical center. Keep both
            # camera centers, the baseline, and the left camera/IMU transform.
            correction = Rotation.from_rotvec(refinement['right_rotation_correction_rotvec']).as_matrix()
            transform[:3, :3] = transform[:3, :3] @ correction.T
        if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError('Invalid camera-to-IMU rigid transform')
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        transforms.append(dict(zip(('px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'),
                                   [*transform[:3, 3], *quaternion])))
        intrinsics.append({'camera_type': 'ds', 'intrinsics': {
            'fx': scaled[0, 0], 'fy': scaled[1, 1], 'cx': scaled[0, 2],
            'cy': scaled[1, 2], 'xi': 0., 'alpha': 0.}})
        resolutions.append([width, height])
    if resolutions[0] != resolutions[1]:
        raise ValueError('Stereo resolutions differ')
    if rectify:
        matrices, camera_transforms = [], []
        for intr, pose in zip(intrinsics, transforms):
            v = intr['intrinsics']
            matrices.append(np.array([[v['fx'], 0, v['cx']], [0, v['fy'], v['cy']], [0, 0, 1.]]))
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat([pose[k] for k in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
            transform[:3, 3] = [pose[k] for k in ('px', 'py', 'pz')]
            camera_transforms.append(transform)
        relative = np.linalg.inv(camera_transforms[1]) @ camera_transforms[0]
        distortion = [np.array(stereo[f'cam{i}']['distortion_coeffs']) for i in range(2)]
        r0, r1, p0, p1, _, _, _ = cv2.stereoRectify(
            matrices[0], distortion[0], matrices[1], distortion[1], tuple(resolutions[0]),
            relative[:3, :3], relative[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        for i, (rectification, projection) in enumerate(zip((r0, r1), (p0, p1))):
            k = projection[:3, :3]
            maps[i] = cv2.initUndistortRectifyMap(matrices[i], distortion[i], rectification,
                                               k, tuple(resolutions[i]), cv2.CV_32FC1)
            # Rectification maps original camera coordinates into virtual
            # camera coordinates; invert it when placing that camera in IMU.
            q = Rotation.from_matrix(camera_transforms[i][:3, :3] @ rectification.T).as_quat()
            transforms[i].update(dict(zip(('qx', 'qy', 'qz', 'qw'), q)))
            intrinsics[i]['intrinsics'].update(fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2])
    identity = dict(px=0., py=0., pz=0., qx=0., qy=0., qz=0., qw=1.)
    calibration = {
        'T_imu_cam': transforms, 'intrinsics': intrinsics, 'resolution': resolutions,
        'calib_accel_bias': [0.] * 9, 'calib_gyro_bias': [0.] * 12,
        'imu_update_rate': imu['rate_hz'],
        'accel_noise_std': [imu['accelerometer_noise_density'] * imu_noise_scale] * 3,
        'gyro_noise_std': [imu['gyroscope_noise_density'] * imu_noise_scale] * 3,
        'accel_bias_std': [imu['accelerometer_random_walk'] * imu_bias_scale] * 3,
        'gyro_bias_std': [imu['gyroscope_random_walk'] * imu_bias_scale] * 3,
        'T_mocap_world': identity, 'T_imu_marker': identity,
        'mocap_time_offset_ns': 0, 'mocap_to_imu_offset_ns': 0, 'cam_time_offset_ns': 0,
    }
    (output / 'basalt_calibration.json').write_text(json.dumps({'value0': calibration}, indent=2) + '\n')
    report = {
        'source': str(source),
        'source_sha256': source_hashes,
        'baseline_m': float(np.linalg.norm(
            [transforms[0][k] - transforms[1][k] for k in ('px', 'py', 'pz')])),
        'resolution': resolutions, 'imu_rate_hz': imu['rate_hz'],
        'imu_noise_scaling': {
            'white_noise_std_scale': imu_noise_scale,
            'bias_random_walk_std_scale': imu_bias_scale,
            'effective': {k: calibration[k] for k in
                          ('accel_noise_std', 'gyro_noise_std', 'accel_bias_std', 'gyro_bias_std')},
            'note': 'Multipliers apply to standard deviations; covariance scales by their square. Source YAMLs remain unchanged.'},
        'image_processing': 'area resize then radtan undistortion; unchanged camera axes; exact pinhole DS (xi=alpha=0)',
        'camera_time_offset_ns': 0,
        'camera_time_offset_note': 'Use synchronized bag header timestamps; supplied calibration provides no time shift.',
        'intrinsics_source': 'stereo.yaml; camera-to-IMU transforms from stereo-imu.yaml',
        'stereo_refinement': str(stereo_refinement) if stereo_refinement else None,
        'stereo_rectified': rectify,
    }
    if rectify:
        report['image_processing'] = 'area resize then stereo rectification; virtual camera axes and common intrinsics carried into Basalt calibration'
    if refinement:
        (output / 'stereo_refinement.json').write_text(json.dumps(refinement, indent=2) + '\n')
    (output / 'calibration_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return calibration, maps, [stereo[f'cam{i}']['resolution'] for i in range(2)]


class Bag:
    def __init__(self, path):
        metadata = yaml.safe_load((path / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
        files = metadata['relative_file_paths']
        if metadata['storage_identifier'] != 'sqlite3' or len(files) != 1:
            raise ValueError('Expected one uncompressed ROS2 SQLite database')
        self.connection = sqlite3.connect((path / files[0]).resolve().as_uri() + '?mode=ro', uri=True)
        self.types = get_typestore(Stores.ROS2_HUMBLE)
        self.topics = {name: (tid, typ) for tid, name, typ in self.connection.execute('SELECT id,name,type FROM topics')}

    def messages(self, topic):
        tid, typ = self.topics[topic]
        for (data,) in self.connection.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,)):
            yield self.types.deserialize_cdr(data, typ)

    def stereo_index(self):
        # Index small row IDs; image blobs are fetched only for the current pair.
        indices = []
        for side in ('left', 'right'):
            tid, _ = self.topics[f'/camera_{side}/image_raw']
            rows = self.connection.execute('SELECT timestamp,id FROM messages WHERE topic_id=? ORDER BY timestamp', (tid,)).fetchall()
            index = dict(rows)
            if len(index) != len(rows):
                raise ValueError(f'Duplicate {side} image timestamps')
            indices.append(index)
        if indices[0].keys() != indices[1].keys() or not indices[0]:
            raise ValueError('Stereo images must have exactly matching bag timestamps')
        return [(t, indices[0][t], indices[1][t]) for t in sorted(indices[0])]

    def image(self, row_id, expected_stamp, resolution):
        data = self.connection.execute('SELECT data FROM messages WHERE id=?', (row_id,)).fetchone()[0]
        message = self.types.deserialize_cdr(data, 'sensor_msgs/msg/Image')
        if stamp_ns(message) != expected_stamp:
            raise ValueError('Image header and bag timestamps differ; explicit synchronization required')
        if message.encoding != 'mono8' or [message.width, message.height] != resolution:
            raise ValueError('Expected mono8 images at the calibrated resolution')
        return np.asarray(message.data).reshape(message.height, message.step)[:, :message.width]


def load_imu(bag):
    samples = [[stamp_ns(m), *xyz(m.angular_velocity), *xyz(m.linear_acceleration)]
               for m in bag.messages('/gnss/imu')]
    stamps = np.array([s[0] for s in samples], dtype=np.int64)
    if len(samples) < 2 or np.any(np.diff(stamps) <= 0) or not np.isfinite(np.array(samples)[:, 1:]).all():
        raise ValueError('IMU samples must be finite and strictly ordered')
    return samples, stamps


def load_truth(bag, output):
    truth = []
    if '/gnss/ground_truth' not in bag.topics:
        return truth
    with (output / 'ground_truth.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['#timestamp_ns', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz'])
        for m in bag.messages('/gnss/ground_truth'):
            p, q = m.pose.pose.position, m.pose.pose.orientation
            t = stamp_ns(m)
            truth.append((t, np.array(xyz(p)), Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()))
            writer.writerow([t, *xyz(p), q.w, q.x, q.y, q.z])
    return truth


def viewer_blueprint(camera_mode, online_loop=False, online_depth=False):
    camera_views = [rrb.Spatial2DView(origin='stereo/cam0', name='Left camera + feature tracks')]
    if camera_mode == 'stereo':
        camera_views.append(rrb.Spatial2DView(origin='stereo/cam1', name='Right camera + feature tracks'))
    loop_views = [rrb.TimeSeriesView(origin='metrics/loop', name='Loop closure')] if online_loop else []
    depth_views = [rrb.Vertical(
        rrb.Spatial2DView(origin='da3/depth', name='DA3 depth (WIP, metres)'),
        rrb.Spatial2DView(origin='da3/input', name='DA3 last keyframe'),
        rrb.TextDocumentView(origin='da3/status', name='DA3 sequence and alignment'),
        rrb.TimeSeriesView(origin='metrics/da3', name='DA3 coverage and alignment'),
        row_shares=[3, 2, 1, 1])] if online_depth else []
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin='world', name='VIO, active landmarks, and reference'),
            rrb.Vertical(
                *camera_views,
                rrb.TimeSeriesView(origin='metrics/tracks', name='Visual observations'),
                rrb.TimeSeriesView(origin='metrics/timing', name='VIO processing (ms)'),
                *loop_views,
                row_shares=[3] * len(camera_views) + [1, 1] + [1] * len(loop_views)),
            *depth_views,
            column_shares=([2, 1, 1] if online_depth else [1, 1]) if camera_mode == 'mono' else ([2, 1, 1] if online_depth else [2, 1])),
        rrb.TimePanel(state='expanded', timeline='elapsed', play_state='Following'))


def init_rerun(args, calibration):
    rr.init(f'visloc_GRACO_{args.bag.name}')
    sinks = [rr.FileSink(args.output / 'playback.rrd')]
    if args.rerun_connect:
        sinks.append(rr.GrpcSink(args.rerun_connect))
    rr.set_sinks(*sinks)
    rr.send_blueprint(viewer_blueprint(args.camera_mode, bool(args.online_loop_config), bool(args.da3_config)))
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log('notes', rr.TextDocument(
        f'Blue: sensor-only visloc/Basalt {args.camera_mode} VIO. Orange: active landmarks. '
        + ('Green: loop-corrected trajectory. Magenta: verified loop edges. '
           if args.online_loop_config else '') +
        ('Gray-textured dense points: WIP five-keyframe DA3, conditioned on VIO camera poses/intrinsics '
         'and converted to metres using the configured scale source (see DA3 status). Depth uses raw VIO coordinates. '
         if args.da3_config else '') +
        'Gray: GRACO ground truth transformed to match the first VIO body pose, '
        'for display only. Final evaluation uses full-run rigid SE(3) alignment. '
        'Images are resized and undistorted using the selected rig calibration. '
        'GNSS position and IMU orientation are never sent to the estimator. '
        + ('Mono mode uses and displays only the left camera and IMU.'
           if args.camera_mode == 'mono' else 'Stereo mode uses both cameras and IMU.')), static=True)
    camera_count = 1 if args.camera_mode == 'mono' else 2
    for i, transform in enumerate(calibration['T_imu_cam'][:camera_count]):
        path = f'world/body/cam{i}'
        rr.log(path, rr.Transform3D(
            translation=[transform[k] for k in ('px', 'py', 'pz')],
            quaternion=rr.Quaternion(xyzw=[transform[k] for k in ('qx', 'qy', 'qz', 'qw')])), static=True)
        intrinsics = calibration['intrinsics'][i]['intrinsics']
        rr.log(path + '/frustum', rr.Pinhole(
            focal_length=[intrinsics['fx'], intrinsics['fy']],
            principal_point=[intrinsics['cx'], intrinsics['cy']],
            resolution=calibration['resolution'][i], image_plane_distance=.5), static=True)


class FeatureTrackOverlay:
    """Short, consecutive KLT histories, keyed by camera and persistent ID."""

    def __init__(self, history_length=12):
        self.history_length = history_length
        self.histories = {}

    def update(self, observations):
        current = {}
        for camera, track_id, x, y in observations:
            if not np.isfinite([x, y]).all():
                continue
            key = (int(camera), int(track_id))
            history = self.histories.get(key, deque(maxlen=self.history_length))
            history.append([x, y])
            current[key] = history
        # Retire missing observations immediately; a reappearing ID starts
        # a fresh trail rather than drawing a line across an untracked gap.
        self.histories = current

    def log(self, camera, source_size, preview_size):
        path = f'stereo/cam{camera}/features'
        tracks = [(track_id, history) for (cam, track_id), history in self.histories.items()
                  if cam == camera]
        if not tracks:
            rr.log(path, rr.Clear(recursive=True))
            return
        scale = np.array(preview_size) / np.array(source_size)
        positions, colors, labels, trails, trail_colors = [], [], [], [], []
        for track_id, history in tracks:
            hue = ((track_id * 2654435761) & 0xffffffff) / 2**32
            color = [round(v * 255) for v in colorsys.hsv_to_rgb(hue, .65, 1.)]
            pixels = (np.asarray(history) + .5) * scale - .5
            positions.append(pixels[-1])
            colors.append(color)
            labels.append(str(track_id))
            if len(pixels) > 1:
                trails.append(pixels)
                trail_colors.append([*color, 180])
        rr.log(path + '/points', rr.Points2D(
            positions, colors=colors, radii=rr.Radius.ui_points(2.5),
            labels=labels, show_labels=False, draw_order=2))
        if trails:
            rr.log(path + '/trails', rr.LineStrips2D(
                trails, colors=trail_colors, radii=rr.Radius.ui_points(.8), draw_order=1))
        else:
            rr.log(path + '/trails', rr.Clear(recursive=True))


def replay(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    calibration, maps, raw_sizes = prepare_calibration(
        args.calibration_dir, args.output, args.width, args.stereo_refinement, args.rectify,
        args.imu_noise_scale, args.imu_bias_scale)
    config = json.loads(args.config.read_text())
    (args.output / 'basalt_config.json').write_text(json.dumps(config, indent=2) + '\n')
    bag = Bag(args.bag)
    depth_worker = None
    try:
        frames = bag.stereo_index()
        imu, imu_stamps = load_imu(bag)
        truth = load_truth(bag, args.output)
        truth_stamps = np.array([p[0] for p in truth], dtype=np.int64)
        requested = len(frames)
        if args.max_frames:
            frames = frames[:args.max_frames]
        if imu_stamps[0] > frames[0][0] or imu_stamps[-1] < frames[-1][0]:
            raise ValueError('IMU does not cover the requested camera interval')
        init_rerun(args, calibration)
        if args.da3_config:
            depth_worker = OnlineDepth(DepthConfig.from_path(args.da3_config), args.output / 'da3')
        depth_keyframe_index = 0
        width, height = calibration['resolution'][0]
        print(f'Replaying {len(frames)}/{requested} camera frames, {len(imu)} IMU samples, {width}x{height}; estimator={args.camera_mode}', flush=True)
        # The stream wire format has two fixed-size image buffers. In mono
        # mode its second buffer is discarded; do not decode the right camera.
        unused_right_image = bytes(width * height) if args.camera_mode == 'mono' else None
        started = time.monotonic()
        positions, estimates, timings, observations = [], [], [], []
        feature_overlay = FeatureTrackOverlay()
        loop_status = None
        imu_index = 0
        initial_index = int(np.searchsorted(imu_stamps, frames[0][0], side='left'))
        gt_rotation, gt_translation = np.eye(3), np.zeros(3)
        with (args.output / 'vio_stderr.log').open('w') as error_log:
            command = [
                str(args.binary), str(args.output / 'basalt_calibration.json'),
                str(args.output / 'basalt_config.json'), str(args.output),
                '--scalar-mode', args.scalar_mode, '--camera-mode', args.camera_mode]
            if args.online_loop_config:
                command.extend(['--online-loop-config', str(args.online_loop_config.resolve())])
            process = subprocess.Popen(command,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log)
            try:
                for index, (stamp, left_id, right_id) in enumerate(frames):
                    image_ids = (left_id,) if args.camera_mode == 'mono' else (left_id, right_id)
                    images = [cv2.remap(
                        cv2.resize(bag.image(row, stamp, raw_sizes[i]), (width, height), interpolation=cv2.INTER_AREA),
                        *maps[i], cv2.INTER_LINEAR) for i, row in enumerate(image_ids)]
                    end = int(np.searchsorted(imu_stamps, stamp, side='right'))
                    header = json.dumps({
                        'timestamp_ns': stamp, 'width': width, 'height': height,
                        'imu': imu[imu_index:end], 'initialization_imu': imu[initial_index],
                    }, allow_nan=False).encode()
                    process.stdin.write(struct.pack('<I', len(header)) + header)
                    for image in images:
                        process.stdin.write(image.tobytes())
                    if unused_right_image is not None:
                        process.stdin.write(unused_right_image)
                    process.stdin.flush()
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError(f'VIO stopped at frame {index}; see {args.output / "vio_stderr.log"}')
                    result = json.loads(line)
                    if result['timestamp_ns'] != stamp or result['frame_id'] != index:
                        raise RuntimeError('VIO acknowledgement does not match the sensor frame')
                    if 'feature_tracks' not in result:
                        raise RuntimeError('Rebuild basalt_stream_vio to enable feature-track overlays')
                    if depth_worker:
                        if 'is_keyframe' not in result:
                            raise RuntimeError('Rebuild basalt_stream_vio to expose actual VIO keyframes')
                        if result['is_keyframe']:
                            depth_worker.submit(keyframe_from_vio(
                                result, images[0], calibration, depth_keyframe_index, depth_worker.config))
                            depth_keyframe_index += 1
                    feature_overlay.update(result['feature_tracks'])
                    imu_index = end
                    p = np.array(result['position'])
                    rotation = Rotation.from_quat(result['quaternion_xyzw']).as_matrix()
                    positions.append(p)
                    estimates.append((stamp, p, rotation))
                    timings.append(result['process_ms'])
                    observations.append(result['observations'])
                    elapsed = (stamp - frames[0][0]) / 1e9
                    rr.set_time('elapsed', duration=elapsed)
                    if index == 0 and truth:
                        nearest = int(np.argmin(np.abs(truth_stamps - stamp)))
                        if abs(int(truth_stamps[nearest]) - stamp) > 10_000_000:
                            raise ValueError('Ground truth does not match the initial image timestamp')
                        gt_rotation = rotation @ truth[nearest][2].T
                        gt_translation = p - gt_rotation @ truth[nearest][1]
                        rr.log('world/ground_truth_trajectory', rr.LineStrips3D(
                            [np.array([x[1] for x in truth]) @ gt_rotation.T + gt_translation],
                            colors=[160, 160, 160]), static=True)
                    rr.log('world/body', rr.Transform3D(translation=p, mat3x3=rotation))
                    rr.log('world/current_vio', rr.Points3D([p], colors=[40, 160, 255], radii=.1))
                    if index % 10 == 0 or index + 1 == len(frames):
                        rr.log('world/vio_trajectory', rr.LineStrips3D([positions], colors=[40, 160, 255]))
                    points = np.array([x[1:4] for x in result['map_points']]).reshape(-1, 3)
                    rr.log('world/active_landmarks', rr.Points3D(points, colors=[255, 180, 60], radii=.045))
                    if result.get('loop_closure') is not None:
                        loop_status = result['loop_closure']
                        log_loop_status(loop_status)
                    if loop_status is not None:
                        correction = loop_status['map_from_odom']
                        loop_rotation = Rotation.from_quat(correction['quaternion_xyzw']).as_matrix()
                        loop_position = loop_rotation @ p + np.array(correction['translation'])
                        rr.log('world/current_loop_corrected', rr.Points3D(
                            [loop_position], colors=[60, 230, 120], radii=.12))
                    for i, image in enumerate(images):
                        preview = image
                        if args.preview_width and image.shape[1] > args.preview_width:
                            preview = cv2.resize(image, (args.preview_width, round(image.shape[0] * args.preview_width / image.shape[1])), interpolation=cv2.INTER_AREA)
                        rr.log(f'stereo/cam{i}', rr.Image(preview).compress(jpeg_quality=85))
                        feature_overlay.log(i, (width, height), (preview.shape[1], preview.shape[0]))
                        rr.log(f'metrics/tracks/cam{i}', rr.Scalars(result['observations'][i]))
                    rr.log('metrics/timing/process_ms', rr.Scalars(result['process_ms']))
                    if depth_worker:
                        for depth_event in depth_worker.poll():
                            log_depth_result(depth_event, frames[0][0], depth_worker.config.cloud_stride)
                        rr.set_time('elapsed', duration=elapsed)
                    if truth:
                        j = int(np.searchsorted(truth_stamps, stamp))
                        nearest = min((k for k in (j - 1, j) if 0 <= k < len(truth)),
                                      key=lambda k: abs(int(truth_stamps[k]) - stamp))
                        if abs(int(truth_stamps[nearest]) - stamp) <= 10_000_000:
                            rr.log('world/current_ground_truth', rr.Points3D(
                                [gt_rotation @ truth[nearest][1] + gt_translation],
                                colors=[160, 160, 160], radii=.1))
                        else:
                            rr.log('world/current_ground_truth', rr.Clear(recursive=True))
                    if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(frames):
                        print(f'frame={index + 1}/{len(frames)} elapsed={elapsed:.2f}s '
                              f'tracks={result["observations"]} landmarks={len(points)} '
                              f'vio_ms={result["process_ms"]:.1f} position={p.round(2).tolist()}', flush=True)
                process.stdin.close()
                if process.wait(timeout=300 if args.online_loop_config else 30):
                    raise RuntimeError(f'VIO exited with {process.returncode}; see vio_stderr.log')
            except BrokenPipeError as exc:
                raise RuntimeError(f'VIO stopped while receiving sensor input; see {args.output / "vio_stderr.log"}') from exc
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
                if not process.stdin.closed:
                    process.stdin.close()
        summary = {
            'bag': str(args.bag), 'frames_available': requested, 'frames_processed': len(estimates),
            'sensor_only': True, 'imu_samples_loaded': len(imu), 'imu_samples_delivered': imu_index,
            'scalar_mode': args.scalar_mode,
            'camera_mode': args.camera_mode,
            'imu_noise_scale': args.imu_noise_scale, 'imu_bias_scale': args.imu_bias_scale,
            'feature_tracks': 'KLT observations with persistent ID colors and 12-frame trails',
            'sensor_duration_s': (frames[-1][0] - frames[0][0]) / 1e9,
            'wall_seconds': time.monotonic() - started,
            'vio_ms_median': float(np.median(timings)), 'vio_ms_p95': float(np.percentile(timings, 95)),
            'observations_min': np.min(observations, axis=0).tolist(),
            'recording': str(args.output / 'playback.rrd'),
            'visualization_alignment': 'reference transformed to first estimated body pose, no scale correction',
        }
        if depth_worker:
            summary['da3'] = depth_worker.finish()
            summary['wall_seconds'] = time.monotonic() - started
            for depth_event in depth_worker.poll():
                log_depth_result(depth_event, frames[0][0], depth_worker.config.cloud_stride)
            rr.set_time('elapsed', duration=(frames[-1][0]-frames[0][0])/1e9)
        if truth and len(estimates) >= 3:
            evaluation = evaluate(truth, estimates, 10_000_000)
            (args.output / 'evaluation.json').write_text(json.dumps(evaluation, indent=2) + '\n')
            summary['ate_se3_rmse_m'] = evaluation['ate_translation_se3_m']['rmse']
            summary['diagnostic_scale_to_reference'] = evaluation['sim3_scale']
            summary['diagnostic_excess_scale_percent'] = (1 / evaluation['sim3_scale'] - 1) * 100
        if args.online_loop_config:
            loop_status = json.loads((args.output / 'loop_closure.json').read_text())
            summary['loop_closure'] = {k: v for k, v in loop_status.items() if k != 'keyframe_positions'}
            log_loop_status(loop_status)
            with (args.output / 'trajectory_loop.csv').open() as source:
                corrected = [(int(row['timestamp_ns']),
                    np.array([float(row[k]) for k in ('tx', 'ty', 'tz')]),
                    Rotation.from_quat([float(row[k]) for k in ('qx', 'qy', 'qz', 'qw')]).as_matrix())
                    for row in csv.DictReader(source)]
            if len(corrected) != len(estimates):
                raise RuntimeError('Loop-corrected trajectory is incomplete')
            rr.log('world/loop_corrected_trajectory', rr.LineStrips3D(
                [[x[1] for x in corrected]], colors=[60, 230, 120]))
            if truth and len(corrected) >= 3:
                loop_evaluation = evaluate(truth, corrected, 10_000_000)
                (args.output / 'evaluation_loop.json').write_text(json.dumps(loop_evaluation, indent=2) + '\n')
                summary['loop_ate_se3_rmse_m'] = loop_evaluation['ate_translation_se3_m']['rmse']
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if depth_worker and depth_worker.thread.is_alive():
            depth_worker.abort()
        bag.connection.close()
        rr.disconnect()


def log_loop_status(status):
    for key in ('keyframes', 'candidates', 'candidate_queries', 'local_sequences_skipped',
                'redundant_sequences_skipped', 'accepted_loops', 'dropped_keyframes', 'capacity_skips', 'worker_ms'):
        rr.log(f'metrics/loop/{key}', rr.Scalars(status[key]))
    positions = dict(status['keyframe_positions'])
    if positions:
        rr.log('world/loop_keyframes', rr.LineStrips3D([list(positions.values())], colors=[60, 230, 120]))
    edges = [[positions[a], positions[b]] for a, b in status['edges'] if a in positions and b in positions]
    if edges:
        rr.log('world/loop_edges', rr.LineStrips3D(edges, colors=[235, 60, 220]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, default=Path('/data/graco/aerial-08-25m_ros2'))
    parser.add_argument('--calibration-dir', type=Path, help='Override rig calibration; otherwise inferred from the ground-/aerial- bag name')
    parser.add_argument('--stereo-refinement', type=Path, help='Image-only right-camera rotation refinement from refine_graco_stereo.py')
    parser.add_argument('--rectify', action='store_true', help='Rectify the stereo pair and transform the camera calibration consistently')
    parser.add_argument('--config', type=Path, default=REPO / 'configs/graco/aerial_vio.json')
    parser.add_argument('--binary', type=Path, default=REPO / 'target/release/examples/basalt_stream_vio')
    parser.add_argument('--online-loop-config', type=Path, help='JIST/XFeat/LighterGlue TensorRT engine bundle config')
    parser.add_argument('--da3-config', type=Path, help='WIP pose-conditioned five-keyframe DA3 TensorRT depth; gate inference on new FOV coverage; select pose-only or landmark depth scale in the config')
    parser.add_argument('--imu-noise-scale', type=float, default=1.0,
                        help='Multiplier for calibrated accelerometer/gyroscope white-noise standard deviations')
    parser.add_argument('--imu-bias-scale', type=float, default=1.0,
                        help='Multiplier for calibrated accelerometer/gyroscope bias random-walk standard deviations')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing outputs are never overwritten')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--scalar-mode', choices=('f32', 'f64'), default='f32', help='Estimator arithmetic; f64 is experimental')
    parser.add_argument('--camera-mode', choices=('stereo', 'mono'), default='stereo', help='Use and display both cameras or only the left camera and IMU')
    parser.add_argument('--preview-width', type=int, default=800, help='Rerun preview width; 0 keeps processing resolution')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--rerun-connect', help='Optional viewer URL, e.g. rerun+http://127.0.0.1:9878/proxy')
    args = parser.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in (args.imu_noise_scale, args.imu_bias_scale)):
        parser.error('IMU noise and bias scales must be finite and positive')
    if args.width < 64 or args.preview_width < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('width must be >=64 and max frames must be positive')
    try:
        args.calibration_dir = calibration_for_bag(args.bag, args.calibration_dir)
    except ValueError as error:
        parser.error(str(error))
    for name in ('bag', 'calibration_dir', 'config', 'binary', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    replay(args)


if __name__ == '__main__':
    main()
