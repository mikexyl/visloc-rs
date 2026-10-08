#!/usr/bin/env python3
"""Run ScaRF alongside a single-robot visloc ROS2 mission with GPS global BA.

The native flushed camera journal and atomic graph snapshots are the local
transport. Cached RGB is released only once bracketing VIO poses have arrived.
"""
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

# Keep inference independent of the ROS Python environment and its BLAS pools.
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MPLBACKEND'] = 'Agg'

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from run_scarf_mapping import ROOT, camera_pose_at, rgb_images, save_json, stage_mapper, summarize
from visualize_multi_robot_live import JsonTail


def pose_array(transform):
    return np.array(transform['translation'] + transform['rotation_xyzw'], dtype=float)


def corrected_camera_poses(stamps, raw_cameras, graph_poses):
    """Apply only the received BA revision; hold its correction at the frontier."""
    if not graph_poses:
        return raw_cameras.copy()
    times = np.array([p['timestamp_ns'] for p in graph_poses], dtype=np.int64)
    transforms = np.array([pose_array(p['map_from_odom']) for p in graph_poses])
    if len(times) == 1:
        correction = np.repeat(transforms, len(stamps), axis=0)
    else:
        origin = int(times[0])
        seconds = (times - origin).astype(float) / 1e9
        queries = (np.clip(stamps, times[0], times[-1]) - origin).astype(float) / 1e9
        positions = np.column_stack([np.interp(queries, seconds, transforms[:, i]) for i in range(3)])
        rotations = Slerp(seconds, Rotation.from_quat(transforms[:, 3:]))(queries).as_quat()
        correction = np.column_stack([positions, rotations])
    rotations = Rotation.from_quat(correction[:, 3:])
    return np.column_stack([rotations.apply(raw_cameras[:, :3]) + correction[:, :3],
                           (rotations * Rotation.from_quat(raw_cameras[:, 3:])).as_quat()])


def key(stamp):
    return f'{stamp // 10**9:010d}_{stamp % 10**9:09d}'


def cache_rgb(args, robot, camera):
    """Predecode dataset sensors, without reading any estimated/reference poses."""
    folder = args.output / 'rgb_cache'
    folder.mkdir()
    inputs = SimpleNamespace(images=None, bag=Path(robot['bag']), rgb_topic=args.rgb_topic)
    last, records = None, []
    shift = round(camera['timeshift_cam_imu'] * 1e9)
    for sensor_ns, decode, _ in rgb_images(inputs, camera):
        if last is not None and sensor_ns - last < 190_000_000:
            continue
        last = sensor_ns
        stamp = sensor_ns + shift + robot['offset_ns']
        path = folder / f'image_{key(stamp)}.jpg'
        if not cv2.imwrite(str(path), decode(), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError(path)
        records.append((stamp, path, sensor_ns))
    with (args.output / 'timestamps.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['mission_ns', 'image_clock_ns', 'recording_imu_ns'])
        writer.writerows((t, raw, t - robot['offset_ns']) for t, _, raw in records)
    return records


class LiveInput:
    def __init__(self, mission_path, output, images, camera_to_body, process=None):
        self.root, self.output = mission_path.parent, output
        self.mission = json.loads(mission_path.read_text())
        self.robot = self.mission['robots'][0]
        self.journal = JsonTail(self.root / 'robots' / self.robot['robot'] / 'camera_frames.jsonl')
        self.images, self.camera_to_body, self.process = images, camera_to_body, process
        self.image_index, self.unmatched = 0, 0
        self.stamps, self.bodies, self.raw_rgb = [], [], {}
        self.image_paths, self.graph = {}, dict(revision=-1, poses=[], loops=[])
        self.session, self.graph_stat = None, None
        self.snapshot, self.odometry = {}, {}
        self.bag_path = self.root
        self.start = self.activity = time.monotonic()
        self.revisions_seen, self.revisions_coalesced, self.gaps = 0, 0, 0
        self.applied_revisions = set()
        self.applied_loops = set()
        self.last_frame = -1
        self.events = (output / 'online_events.jsonl').open('w', buffering=1)

    def event(self, kind, **values):
        self.events.write(json.dumps(dict(event=kind, elapsed_s=time.monotonic()-self.start,
            backend_revision=self.graph['revision'], raw_frames=len(self.stamps),
            released_rgb=len(self.raw_rgb), sensor_replay_complete=len(self.stamps) == self.robot['frames'],
            backend_finished=(self.root/'backend/finished.json').exists(), **values), allow_nan=False) + '\n')

    def finished(self):
        path = self.root / 'backend/finished.json'
        if not path.exists():
            return False
        marker = json.loads(path.read_text())
        return (self.graph['revision'] >= marker['revision'] and self.journal.path.exists()
                and self.journal.offset == self.journal.path.stat().st_size)

    def poll(self, app=None):
        from scarf_slam.core.pose import MappingPose
        changed = False
        for frame in self.journal.poll(1024):
            if self.session is None:
                self.session = frame['session']
            if frame['session'] != self.session or frame['robot'] != self.robot['robot']:
                raise ValueError('Estimator session changed; start a separate ScaRF map for the new session')
            if self.stamps and frame['timestamp_ns'] <= self.stamps[-1]:
                raise ValueError('Non-increasing VIO camera timestamps')
            self.gaps += frame['frame_id'] - self.last_frame - 1
            self.last_frame = frame['frame_id']
            pose = pose_array(frame['body_to_odom'])
            if not np.isfinite(pose).all():
                raise ValueError('Non-finite VIO pose')
            self.stamps.append(frame['timestamp_ns'])
            self.bodies.append(pose)
            changed = True
        path = self.root / 'backend/graph_snapshot.json'
        if path.exists():
            stat = path.stat()
            version = (stat.st_ino, stat.st_mtime_ns)
            if version != self.graph_stat:
                graph = json.loads(path.read_text())
                if graph['backend_mode'] != 'global_bundle_adjustment':
                    raise ValueError('Expected the GPS global BA backend')
                if 'optimized_loops' not in graph:
                    raise ValueError('Rebuild the ROS2 backend: live ScaRF requires loop-triggered BA snapshots')
                if graph['revision'] < self.graph['revision']:
                    raise ValueError('Backend revision went backwards')
                self.revisions_coalesced += max(0, graph['revision'] - self.graph['revision'] - 1)
                self.graph, self.graph_stat = graph, version
                self.revisions_seen += 1
                self.event('graph_received', loops=len(graph['loops']))
                changed = True
        added = False
        if len(self.stamps) >= 2:
            stamps, bodies = np.array(self.stamps, dtype=np.int64), np.array(self.bodies)
            while self.image_index < len(self.images) and self.images[self.image_index][0] <= self.stamps[-1]:
                stamp, path, _ = self.images[self.image_index]
                self.image_index += 1
                pose = camera_pose_at(stamp, stamps, bodies, self.camera_to_body)
                if pose is None:
                    self.unmatched += 1
                    continue
                name = key(stamp)
                self.raw_rgb[name] = (stamp, pose)
                self.image_paths[name] = path
                self.odometry[name] = MappingPose(pose[:3].tolist(), pose[3:].tolist())
                added = True
        if changed or added:
            self.activity = time.monotonic()
            graph_poses = sorted((p for p in self.graph['poses'] if
                (p['key']['robot'], p['key']['session']) == (self.robot['robot'], self.session)),
                key=lambda p:p['timestamp_ns'])
            self.snapshot = {}
            if self.raw_rgb:
                rgb_stamps = np.array([v[0] for v in self.raw_rgb.values()], dtype=np.int64)
                raw_rgb = np.array([v[1] for v in self.raw_rgb.values()])
                corrected = corrected_camera_poses(rgb_stamps, raw_rgb, graph_poses)
                self.snapshot = {name: MappingPose(pose[:3].tolist(), pose[3:].tolist())
                                 for name, pose in zip(self.raw_rgb, corrected)}
        if app is not None:
            app.odom_ref_poses_dict = self.odometry.copy()
            app.ref_timestamps = list(self.odometry)
            app.ref_traj_snapshot_timestamps = [app.ref_timestamps[-1]]
        if self.process and self.process.poll() is not None and self.process.returncode:
            raise RuntimeError(f'ROS2 mission failed ({self.process.returncode}); see mission.log')
        if time.monotonic() - self.activity > 300 and not self.finished():
            raise TimeoutError('No VIO/BA progress for 300 seconds; refusing to silently finish an incomplete map')
        return changed or added

    def wait(self, app):
        while True:
            changed = self.poll(app)
            if changed:
                return True  # Drain newly arrived frames even when EOS arrived with them.
            if self.finished():
                return False
            time.sleep(.05)

    def get_trajectory_snapshot(self, timestamp):
        return self.snapshot.copy()

    def take_loop_correction(self):
        """Consume successful loop solves, not ordinary graph publications."""
        loops = {json.dumps(pair, sort_keys=True) for pair in self.graph.get('optimized_loops', [])}
        pending = bool(loops - self.applied_loops)
        self.applied_loops.update(loops)
        return pending

    def decode_image(self, timestamp, cam_name='cam0'):
        image = cv2.imread(str(self.image_paths[timestamp]))
        if image is None:
            raise IOError(self.image_paths[timestamp])
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def write_poses(self, path):
        with path.open('w') as stream:
            stream.write('# timestamp tx ty tz qx qy qz qw\n')
            for name, pose in self.snapshot.items():
                stream.write(name.replace('_', '.') + ' ' + ' '.join(f'{x:.12g}' for x in [*pose.pos, *pose.quat]) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mission', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true', help='Launch the fresh ROS2 mission while ScaRF follows it')
    parser.add_argument('--domain-id', type=int, default=218)
    parser.add_argument('--calibration', type=Path, default=ROOT/'configs/realsense/d455_1/camchain-imucam.yaml')
    parser.add_argument('--rgb-topic', default='/camera/camera/color/image_raw/compressed')
    parser.add_argument('--model', type=Path, default=ROOT/'.runtime/scarf/DA3-LARGE')
    args = parser.parse_args()
    args.mission, args.output = args.mission.resolve(), args.output.resolve()
    mission = json.loads(args.mission.read_text())
    if len(mission['robots']) != 1:
        parser.error('This first live adapter supports one robot/session per map')
    robot = mission['robots'][0]
    backend = json.loads(Path(mission['backend_config']).read_text())
    robot_config = json.loads(Path(robot['config']).read_text())
    if (backend['pgo']['mode'] != 'global_bundle_adjustment' or not backend['pgo']['gps']['enabled']
            or not robot_config['loop_enabled'] or not robot_config['bundle_adjustment_enabled']):
        parser.error('Enable GPS global BA, bundle observations, and loop closure in the mission')
    if (args.mission.parent/'backend/finished.json').exists():
        parser.error('Mission already finished; use the offline mapper or prepare a fresh mission')
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    camera = yaml.safe_load(args.calibration.read_text())['cam2']
    camera_to_body = np.linalg.inv(np.array(camera['T_cam_imu']))
    images = cache_rgb(args, robot, camera)
    source_path, _, run = stage_mapper(args, [ROOT/'tools/scarf/live_input.patch'])
    sys.path.insert(0, str(source_path))
    from scarf_slam.mapping_app import ScaRFSLAM

    class OnlineScarf(ScaRFSLAM):
        def _select_traj_timestamp_for_batch(self, stamps):
            return self.ref_timestamps[-1]  # Latest received graph; no future snapshot lookup.

        def _refresh_slam_trajectory_for_batch(self, stamps):
            start = time.monotonic()
            updated = super()._refresh_slam_trajectory_for_batch(stamps)
            if updated:
                self.input_source.applied_revisions.add(self.input_source.graph['revision'])
                self.input_source.event('map_corrected', submaps=len(self.submaps), seconds=time.monotonic()-start)
            return updated

        def optimize_submap(self, *values, **kwargs):
            result = super().optimize_submap(*values, **kwargs)
            self.input_source.event('submap_created', submaps=len(self.submaps))
            return result

    config = yaml.safe_load((ROOT/'configs/scarf/offline.yaml').read_text())
    config.update(pinhole_intrinsics=camera['intrinsics'], pinhole_resolution=camera['resolution'],
                  depth_model_id=str(args.model.resolve()), use_slam=True)
    (args.output/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (args.output/'calibration.yaml').write_bytes(args.calibration.read_bytes())
    process, source = None, None
    started = time.monotonic()
    with (args.output/'mission.log').open('w') as mission_log, (args.output/'mapping.log').open('w', buffering=1) as mapping_log:
        try:
            if args.launch:
                command = ['bash', '-c', 'source "$1"; exec "$2" "$3" "$4" --domain-id "$5"', 'visloc',
                    str(ROOT/'scripts/source_multi_robot_ros2.bash'), str(ROOT/'.runtime/graco-venv/bin/python'),
                    str(ROOT/'scripts/run_multi_robot_mission.py'), str(args.mission), str(args.domain_id)]
                process = subprocess.Popen(command, cwd=ROOT, stdout=mission_log, stderr=subprocess.STDOUT, start_new_session=True)
                run['mission_command'] = command
            source = LiveInput(args.mission, args.output, images, camera_to_body, process)
            # Bootstrap upstream's file-based initialization with only arrived
            # poses. Subsequent inputs and revisions come from LiveInput.
            while len(source.snapshot) < 2:
                source.poll()
                if source.finished():
                    raise RuntimeError('Mission ended before two RGB poses arrived')
                time.sleep(.05)
            bootstrap = args.output/'bootstrap_images'
            bootstrap.mkdir()
            for name, path in source.image_paths.items():
                (bootstrap/path.name).symlink_to(path)
            source.write_poses(args.output/'bootstrap_poses.txt')
            bootstrap_config = dict(config, use_slam=False)
            (args.output/'bootstrap.yaml').write_text(yaml.safe_dump(bootstrap_config))
            with contextlib.redirect_stdout(mapping_log), contextlib.redirect_stderr(mapping_log):
                app = OnlineScarf(str(args.output/'mapping'), image_folder=str(bootstrap), poses=str(args.output/'bootstrap_poses.txt'))
                app.app_configure(args.output/'bootstrap.yaml')
                app.config['use_slam'] = True
                app.recon_save_folder_name = 'visloc_slam'
                app.slam_bag_data = app.input_source = source
                source.poll(app)
                app.do_processing_outer()
            if not source.finished() or source.last_frame + 1 != robot['frames'] or source.gaps:
                raise RuntimeError('Incomplete VIO camera journal; inspect online_events.jsonl')
            source.write_poses(args.output/'trajectory_rgb.txt')
            if process and process.wait(timeout=30):
                raise RuntimeError('ROS2 mission failed during final drain')
            inputs = dict(mission=str(args.mission), rgb_pose_count=len(source.raw_rgb), body_pose_count=len(source.stamps),
                mode='online GPS global BA + loop closure', metric_accuracy_evaluated=False,
                final_backend_revision=source.graph['revision'], loops=len(source.graph['loops']),
                graph_revisions_received=source.revisions_seen, graph_revisions_coalesced=source.revisions_coalesced,
                map_correction_revisions=sorted(source.applied_revisions), camera_journal_gaps=source.gaps,
                backend_gps=source.graph.get('gps', {}), skipped_unmatched_rgb=source.unmatched)
            save_json(args.output/'inputs.json', inputs)
            run.update(exit_code=0, wall_seconds=time.monotonic()-started, command=sys.argv)
            save_json(args.output/'run.json', run)
            summary = summarize(args, inputs, run)
            print(json.dumps({k:summary[k] for k in ('point_count','submap_count','loops','final_backend_revision','map_correction_revisions','wall_seconds')}, indent=2))
        finally:
            if source:
                source.events.close()
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == '__main__':
    main()
