#!/usr/bin/env python3
"""Offline ScaRF-SLAM reconstruction from visloc IMU/body poses and calibrated RGB.

This adapter only prepares inputs and invokes upstream. All dense estimation,
frame/submap scale optimization and fusion belong to ScaRF-SLAM.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCARF_REVISION = 'e0a34267d575d8b21d548ea8ed3cf5782192bac3'
DA3_REVISION = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_body_poses(path, time_offset_ns=0):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    stamps = np.array([int(r['timestamp_ns']) + time_offset_ns for r in rows], dtype=np.int64)
    poses = np.array([[float(r[k]) for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')]
                      for r in rows])
    if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError('Body poses must have at least two strictly ordered integer timestamps')
    if not np.isfinite(poses).all() or np.max(np.abs(np.linalg.norm(poses[:, 3:], axis=1) - 1)) > 1e-5:
        raise ValueError('Body poses must be finite with unit quaternions')
    return stamps, poses


def camera_pose_at(stamp, stamps, poses, camera_to_body, max_gap_ns=250_000_000):
    """Interpolate in the body frame before composing the camera lever arm."""
    j = int(np.searchsorted(stamps, stamp))
    if j < len(stamps) and stamps[j] == stamp:
        p, r = poses[j, :3], Rotation.from_quat(poses[j, 3:])
    else:
        if j == 0 or j == len(stamps) or stamps[j] - stamps[j - 1] > max_gap_ns:
            return None
        interval = int(stamps[j] - stamps[j - 1])
        alpha = (int(stamp) - int(stamps[j - 1])) / interval
        p = (1 - alpha) * poses[j - 1, :3] + alpha * poses[j, :3]
        r = Slerp([0., 1.], Rotation.from_quat(poses[j - 1:j + 1, 3:]))([alpha])[0]
    return np.r_[p + r.apply(camera_to_body[:3, 3]),
                 (r * Rotation.from_matrix(camera_to_body[:3, :3])).as_quat()]


def image_stamp(path):
    match = re.fullmatch(r'image_(\d+)_(\d{9})\.(?:jpg|jpeg|png)', path.name)
    if not match:
        raise ValueError(f'Expected image_<seconds>_<9 digit nanoseconds>.jpg/png: {path}')
    return int(match[1]) * 10**9 + int(match[2])


def rgb_images(args, camera):
    """Yield image clock stamps and lazy decoders, rectifying bag RGB exactly once."""
    if args.images:
        paths = sorted((p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
                       key=image_stamp)
        for path in paths:
            yield image_stamp(path), lambda p=path: cv2.imread(str(p)), path
        return
    from rosbags.highlevel import AnyReader
    fx, fy, cx, cy = camera['intrinsics']
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    size = tuple(camera['resolution'])
    maps = cv2.initUndistortRectifyMap(k, np.array(camera['distortion_coeffs']), np.eye(3), k, size, cv2.CV_32FC1)
    last = None
    with AnyReader([args.bag]) as reader:
        connections = [c for c in reader.connections if c.topic == args.rgb_topic]
        if not connections or any(c.msgtype != 'sensor_msgs/msg/CompressedImage' for c in connections):
            raise ValueError('RGB topic must contain sensor_msgs/CompressedImage messages')
        for conn, _, data in reader.messages(connections=connections):
            msg = reader.deserialize(data, conn.msgtype)
            stamp = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
            if last is not None and stamp <= last:
                raise ValueError('RGB sensor timestamps are not strictly ordered')
            last = stamp

            def decode(data=msg.data):
                image = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != size[::-1]:
                    raise ValueError('RGB resolution differs from calibration, or image failed to decode')
                return cv2.remap(image, *maps, cv2.INTER_LINEAR)

            yield stamp, decode, None


def prepare_inputs(args):
    chain = yaml.safe_load(args.calibration.read_text())
    camera = chain[args.rgb_camera]
    if camera['camera_model'] != 'pinhole' or camera['distortion_model'] != 'radtan':
        raise ValueError('This RGB adapter requires Kalibr pinhole/radtan calibration')
    camera_to_body = np.linalg.inv(np.array(camera['T_cam_imu'], dtype=float))
    if (not np.isfinite(camera_to_body).all()
            or not np.allclose(camera_to_body[3], [0, 0, 0, 1])
            or not np.allclose(camera_to_body[:3, :3].T @ camera_to_body[:3, :3], np.eye(3), atol=1e-5)):
        raise ValueError('Invalid camera-to-IMU transform')
    offset = (args.image_clock_to_imu_ns if args.images else round(camera['timeshift_cam_imu'] * 1e9))
    stamps, body = load_body_poses(args.body_trajectory, args.pose_time_offset_ns)
    images_out = args.output / 'images'
    images_out.mkdir()
    output_poses, clock_map = [], []
    last, unmatched, considered = None, 0, 0
    with (args.output / 'trajectory_rgb.txt').open('w') as trajectory:
        trajectory.write('# timestamp tx ty tz qx qy qz qw\n')
        for image_ns, decode, source_path in rgb_images(args, camera):
            if last is not None and image_ns - last < 190_000_000:
                continue
            last = image_ns
            considered += 1
            imu_ns = image_ns + offset
            pose = camera_pose_at(imu_ns, stamps, body, camera_to_body)
            if pose is None:
                unmatched += 1
                continue
            # Rebase integers before crossing upstream float-second APIs. Both
            # image filenames and camera poses use the exact same local clock.
            local_ns = imu_ns - int(stamps[0])
            image = decode()
            if image is None or image.shape[:2] != tuple(camera['resolution'])[::-1]:
                raise ValueError('Prepared RGB resolution differs from calibration')
            suffix = source_path.suffix if source_path else '.jpg'
            dest = images_out / f'image_{local_ns // 10**9:010d}_{local_ns % 10**9:09d}{suffix}'
            if source_path:
                shutil.copyfile(source_path, dest)  # Preserve the validated prepared pixels.
            elif not cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise IOError(f'Could not write {dest}')
            trajectory.write(f'{local_ns // 10**9}.{local_ns % 10**9:09d} '
                             + ' '.join(f'{v:.12g}' for v in pose) + '\n')
            output_poses.append(pose)
            clock_map.append((local_ns, image_ns, imu_ns))
            if args.max_images and len(output_poses) >= args.max_images:
                break
    if len(output_poses) < 4:
        raise ValueError('Fewer than four RGB images have valid body poses')
    with (args.output / 'timestamps.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['local_ns', 'image_clock_ns', 'recording_imu_ns'])
        writer.writerows(clock_map)
    shutil.copyfile(args.calibration, args.output / 'calibration.yaml')
    config = yaml.safe_load(args.config.read_text())
    if config.get('use_slam') or config.get('model_name') != 'da':
        raise ValueError('The offline adapter requires use_slam: false and model_name: da')
    if any(config.get(k) for k in ('publish_ros2_pointcloud', 'publish_ros2_path', 'publish_ros2_images')):
        raise ValueError('Offline reconstruction must not publish ROS2 data')
    config.update(pinhole_intrinsics=camera['intrinsics'], pinhole_resolution=camera['resolution'],
                  depth_model_id=str(args.model.resolve()), trajectory='visloc')
    (args.output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(body_trajectory=str(args.body_trajectory.resolve()), body_trajectory_sha256=sha256(args.body_trajectory),
                  body_pose_count=len(stamps), rgb_pose_count=len(output_poses), skipped_unmatched_images=unmatched,
                  sampled_images_considered=considered, source_images=str((args.images or args.bag).resolve()),
                  camera_to_body=camera_to_body.tolist(), image_clock_to_imu_ns=offset,
                  pose_time_offset_ns=args.pose_time_offset_ns, local_origin_recording_imu_ns=int(stamps[0]),
                  interpolation='Linear body translation + SLERP, then camera extrinsic; no extrapolation; max gap 0.25 s',
                  input_duration_s=(clock_map[-1][0] - clock_map[0][0]) / 1e9,
                  input_last_local_ns=clock_map[-1][0], metric_accuracy_evaluated=False)
    save_json(args.output / 'inputs.json', report)
    return report


def stage_mapper(args, extra_patches=()):
    source = ROOT / 'third_party/ScaRF-SLAM'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SCARF_REVISION or subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']):
        raise ValueError('ScaRF submodule must be clean at the pinned revision; run git submodule update --init')
    if not (args.model / 'config.json').is_file() or not list(args.model.glob('*.safetensors')):
        raise ValueError('Model must be a downloaded DA3-LARGE directory containing config.json and safetensors')
    # Export committed upstream source, then apply the small validated GPU patch.
    # The project submodule remains clean and the exact run source is retained.
    run_source = args.output / 'scarf_source'
    run_source.mkdir()
    archive = subprocess.run(['git', '-C', str(source), 'archive', revision], capture_output=True, check=True).stdout
    subprocess.run(['tar', '-x', '-C', str(run_source)], input=archive, check=True)
    patch = ROOT / 'tools/scarf/gpu_configuration.patch'
    subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(patch)], cwd=run_source, check=True)
    for extra in extra_patches:
        subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(extra)], cwd=run_source, check=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    env.pop('PYTHONPATH', None)
    model_hashes = {p.name: sha256(p) for p in sorted(args.model.iterdir())
                    if p.suffix in ('.json', '.safetensors')}
    da3_file = Path(importlib.util.find_spec('depth_anything_3.api').origin).resolve()
    da3_root = da3_file.parents[2]
    da3_revision = subprocess.check_output(['git', '-C', str(da3_root), 'rev-parse', 'HEAD'], text=True).strip()
    if da3_revision != DA3_REVISION or subprocess.check_output(['git', '-C', str(da3_root), 'status', '--porcelain']):
        raise ValueError('DA3 must be installed editable from the clean pinned source; see tools/scarf/setup.sh')
    manifest = dict(scarf_revision=revision, da3_revision=da3_revision, gpu_patch_sha256=sha256(patch),
                    model_hashes=model_hashes, python=sys.version,
                    extra_patches={p.name: sha256(p) for p in extra_patches},
                    environment={k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MPLBACKEND')})
    save_json(args.output / 'run.json', manifest)
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (args.output / 'packages.txt').write_text('\n'.join(packages) + '\n')
    return run_source, env, manifest


def run_mapper(args):
    run_source, env, manifest = stage_mapper(args)
    command = [sys.executable, str(run_source / 'run_mapping.py'), '--slam_folder', str(args.output / 'mapping'),
               '--image_folder', str(args.output / 'images'), '--poses', str(args.output / 'trajectory_rgb.txt'),
               '--config', str(args.output / 'config.yaml')]
    manifest['command'] = command
    save_json(args.output / 'run.json', manifest)
    started = time.monotonic()
    with (args.output / 'mapping.log').open('w') as log:
        process = subprocess.run(command, cwd=run_source, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=process.returncode, wall_seconds=time.monotonic() - started)
    save_json(args.output / 'run.json', manifest)
    if process.returncode:
        raise RuntimeError(f'ScaRF failed with exit code {process.returncode}; see {args.output / "mapping.log"}')
    return manifest


def summarize(args, inputs, run):
    import open3d as o3d
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config = yaml.safe_load((args.output / 'config.yaml').read_text())
    folder = config['trajectory'] + ('_slam' if config.get('use_slam') else '')
    clouds = list((args.output / 'mapping/recon' / folder).glob('pts_global*.pcd'))
    if len(clouds) != 1:
        raise ValueError(f'Expected one final ScaRF cloud, got {len(clouds)}; inspect mapping.log')
    path = clouds[0]
    cloud = o3d.io.read_point_cloud(str(path))
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('ScaRF exported an empty or non-finite cloud')
    graph = json.loads((path.parent / path.stem.replace('pts_global', 'opt_graph', 1) / 'manifest.json').read_text())
    scales = np.array([v['scale'] for v in graph['submaps'].values()])
    mapped = np.loadtxt(path.parent / 'poses_da.txt', ndmin=2)
    trajectory = np.loadtxt(args.output / 'trajectory_rgb.txt', ndmin=2)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped[:, 0]) <= 0):
        raise ValueError('ScaRF mapping poses must be finite and strictly ordered')
    summary = dict(inputs, **run, pointcloud=str(path), point_count=len(points), submap_count=graph['num_submaps'],
                   mapping_keyframes=len(mapped), mapping_duration_s=float(mapped[-1, 0] - mapped[0, 0]),
                   mapping_tail_gap_s=float(trajectory[-1, 0] - mapped[-1, 0]),
                   submap_scale=dict(min=float(scales.min()), median=float(np.median(scales)), max=float(scales.max())))
    save_json(args.output / 'summary.json', summary)
    selection = np.linspace(0, len(points) - 1, min(160000, len(points)), dtype=int)
    xyz, rgb = points[selection], colors[selection]
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    top = fig.add_subplot(121)
    top.scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=.3, rasterized=True)
    top.plot(trajectory[:, 1], trajectory[:, 2], color='#ee681f', lw=1)
    top.set(xlabel='x (m)', ylabel='y (m)', title='ScaRF-SLAM · top view', aspect='equal')
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(*xyz.T, c=rgb, s=.3, depthshade=False, rasterized=True)
    ax.plot(*trajectory[:, 1:4].T, color='#ee681f', lw=1)
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title='DA3-Large · visloc camera poses')
    ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 1))
    ax.view_init(elev=25, azim=-60)
    fig.savefig(args.output / 'preview.png', dpi=160)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body-trajectory', required=True, type=Path, help='visloc CSV: timestamp_ns,tx,ty,tz,qx,qy,qz,qw')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bag', type=Path, help='ROS2 bag with compressed raw RGB')
    source.add_argument('--images', type=Path, help='Already rectified image_<sec>_<nsec>.jpg/png folder')
    parser.add_argument('--image-clock-to-imu-ns', type=int, help='Required with --images; add to filenames to reach recording IMU clock')
    parser.add_argument('--pose-time-offset-ns', type=int, default=0, help='Add to body CSV timestamps to undo any mission replay rebasing')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'configs/realsense/d455_1/camchain-imucam.yaml')
    parser.add_argument('--rgb-camera', default='cam2')
    parser.add_argument('--rgb-topic', default='/camera/camera/color/image_raw/compressed')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/scarf/offline.yaml')
    parser.add_argument('--model', type=Path, default=ROOT / '.runtime/scarf/DA3-LARGE')
    parser.add_argument('--output', required=True, type=Path, help='Fresh result directory; never overwritten')
    parser.add_argument('--max-images', type=int, default=0, help='Smoke test limit; zero uses all images')
    parser.add_argument('--prepare-only', action='store_true', help='Validate/export inputs without GPU inference')
    args = parser.parse_args()
    if bool(args.images) != (args.image_clock_to_imu_ns is not None):
        parser.error('--image-clock-to-imu-ns is required exactly when --images is supplied')
    if args.max_images and args.max_images < 4:
        parser.error('--max-images must be zero or at least four')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    inputs = prepare_inputs(args)
    print(json.dumps(inputs, indent=2), flush=True)
    if not args.prepare_only:
        run = run_mapper(args)
        print(json.dumps(summarize(args, inputs, run), indent=2), flush=True)


if __name__ == '__main__':
    main()
