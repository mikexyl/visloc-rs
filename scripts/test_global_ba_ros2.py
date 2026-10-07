#!/usr/bin/env python3
"""Native GTSAM BA through Rust ROS2: stereo, delayed history, final drain and restart."""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import rclpy
from visloc_msgs import msg as m, srv as s
from test_multi_robot_ros2 import Fixture, record, key, REPO


def frame(i):
    r = record('alpha', i)
    r.body_to_odom.translation = [i * .2, i * .003, 0.]
    return r


def observations(i):
    views = []
    for camera in range(2):
        model = m.Camera(width=640, height=480, intrinsics=[400., 420., 320., 240.],
                         camera_to_body=m.Transform(translation=[camera * .095095, 0., 0.], rotation_xyzw=[0., 0., 0., 1.]))
        points = []
        for track in range(12):
            p = [track * .1 - i * .2 - camera * .095095, .2, 4.]
            points.append(m.LandmarkObservation(track_id=track, pixel=[400.*p[0]/p[2]+320., 420.*p[1]/p[2]+240.],
                                                has_landmark=True, point_camera=p))
        views.append(m.CameraObservations(camera=model, observations=points))
    return m.BundleFrame(key=key('alpha', i), timestamp_ns=frame(i).timestamp_ns, views=views)


def gps_fix(i):
    # Rotate the visual map 90 degrees and translate to ENU (10, 20).
    ns = frame(i).timestamp_ns
    return m.GpsFix(key=key('alpha', i), timestamp_ns=ns, receipt_timestamp_ns=ns+10_000_000,
                    time_source='synthetic_measurement_utc', has_position=True,
                    lla=[(20.+i*.2)/6335439.32729282*180/math.pi, 10./6378137.*180/math.pi, 9000.],
                    status=0, has_quality=True, quality=1, has_hdop=True, hdop=.8,
                    has_covariance=True, covariance_enu=[.25,0.,0.,0.,.25,0.,0.,0.,1e8])


class BundleFixture(Fixture):
    def __init__(self, gps=False):
        super().__init__()
        self.bundles, self.bundle_requests = [], 0
        self.publisher = self.create_publisher(m.BundleFrame, '/alpha/slam/bundle_frames', 64)
        self.service = self.create_service(s.GetBundleHistory, '/alpha/slam/bundle_history', self.bundle_history)
        self.gps_requests = 0
        if gps:
            self.gps_service = self.create_service(s.GetGpsHistory, '/alpha/slam/gps_history', self.gps_history)

    def bundle_history(self, q, r):
        self.bundle_requests += 1
        start = min(q.cursor, len(self.bundles)); end = min(start + 2, len(self.bundles))
        r.enabled = True; r.session = self.sessions['alpha']; r.frames = self.bundles[start:end]
        r.cursor = end; r.more = end < len(self.bundles)
        return r

    def gps_history(self, q, r):
        self.gps_requests += 1
        records = self.records['alpha']
        start = min(q.cursor, len(records)); end = min(start + 3, len(records))
        r.session = self.sessions['alpha']; r.fixes = [gps_fix(f.key.id) for f in records[start:end]]
        r.cursor = end; r.more = end < len(records)
        return r


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gps', action='store_true', help='Joint BA with horizontal GPS and history recovery')
    args = parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '225':
        raise SystemExit('Use isolated ROS_DOMAIN_ID=225')
    rclpy.init(); node = BundleFixture(gps=args.gps); process = None
    with tempfile.TemporaryDirectory(prefix='visloc-ba-ros2-') as d:
        root = Path(d); out = root / 'backend'; config = root / 'config.json'
        pgo = dict(gps=dict(enabled=False))  # Exercise default BA selection.
        if args.gps:
            pgo['gps'] = dict(origin=dict(latitude=0.,longitude=0.,altitude=0.),
                              lever_arms_m={'alpha':[0.,0.,0.]}, min_alignment_fixes=4,
                              alignment_span_m=.6, min_distance_m=.1, min_interval_s=.05,
                              residual_deadband_sigma=0.)
        config.write_text(json.dumps(dict(peers=['alpha'], output=str(out), pgo=pgo)))
        log = (root / 'backend.log').open('w')
        binary = Path(os.environ.get('VISLOC_ROS_INSTALL', REPO / '.runtime/ros2_install')) / 'visloc_ros/lib/visloc_ros/backend'
        def launch():
            return subprocess.Popen([str(binary), '--ros-args', '-p', 'use_sim_time:=true'],
                                    env=dict(os.environ, VISLOC_BACKEND_CONFIG=str(config)), stdout=log, stderr=subprocess.STDOUT)
        def finish():
            f = node.finish.call_async(s.Finish.Request()); node.until(f.done, 'finish response'); assert f.result().accepted
        def complete(n):
            if not (node.shape(n, 1) and len(node.graph.landmarks) == 12):
                return False
            if args.gps:
                root_pose = next(p for p in node.graph.poses if p.key.id == 0).body_to_map.translation
                return abs(root_pose[0]-10.) < .02 and abs(root_pose[1]-20.) < .02
            return True
        try:
            process = launch()
            node.until(lambda: node.publisher.get_subscription_count() > 0, 'BA discovery')
            node.records['alpha'] = [frame(i) for i in [3, 0, 2, 1, 4, 5, 6]]
            node.discover(); node.until(lambda: node.shape(7, 1), 'keyframe catch-up')
            finish()
            start = time.monotonic()
            while time.monotonic() - start < .7:
                rclpy.spin_once(node, timeout_sec=.03)
            assert not (out / 'finished.json').exists(), 'final solve completed before observation history'
            node.bundles = [observations(i) for i in [3, 0, 2, 1, 4, 5, 6]]
            node.until(lambda: complete(7) and (out / 'finished.json').exists(), 'automatic deferred final solve')
            first = json.loads((out / 'graph_snapshot.json').read_text())
            assert first['backend_mode'] == 'global_bundle_adjustment'
            diagnostics = first['bundle_diagnostics'][0]
            assert diagnostics['accepted_tracks'] == 12
            assert diagnostics['accepted_observations'] == 168
            assert first['optimizer_reports'][0]['solver'] == 'gtsam_global_ba'
            assert first['optimizer_reports'][0]['reprojection_factors'] == 168
            assert first['optimizer_reports'][0]['gps_factors'] == (7 if args.gps else 0)
            node.until(lambda: node.graph.input_revision == first['input_revision'], 'final revision publication')
            revision = node.graph.input_revision
            for f in node.bundles * 2:
                node.publisher.publish(f)
            start = time.monotonic()
            while time.monotonic() - start < .5:
                rclpy.spin_once(node, timeout_sec=.03)
            assert node.graph.input_revision == revision
            process.terminate(); process.wait(timeout=10)
            for i in range(7, 10):
                node.records['alpha'].append(frame(i)); node.bundles.append(observations(i))
            with (out / 'graph.jsonl').open('a') as f:
                f.write('{"bundle_frame":')
            node.graph = None; process = launch()
            node.until(lambda: complete(10) and node.graph.input_revision == (30 if args.gps else 20), 'restart and observation catch-up')
            finish(); node.until(lambda: (out / 'finished.json').exists(), 'restarted final solve')
            final = json.loads((out / 'graph_snapshot.json').read_text())
            assert final['optimizer_reports'][0]['reprojection_factors'] == 240
            assert final['optimizer_reports'][0]['gps_factors'] == (10 if args.gps else 0)
            if args.gps:
                assert final['gps']['datum'] == first['gps']['datum']
                assert len(final['gps']['aligned_components']) == 1
                assert all(d['reason'] == 'active' for d in final['gps']['diagnostics'])
                assert all(abs(p['position'][0] - 9.8) < .03 for p in final['landmarks'])
                root_pose = next(p for p in final['poses'] if p['key']['id'] == 0)
                assert abs(root_pose['body_to_map']['translation'][2]) < 1e-10
            assert not (out / 'backend_error.txt').exists()
            report = dict(passed=True, stereo_observations=240, landmarks=12, keyframes=10,
                          bundle_history_requests=node.bundle_requests, frozen_ros_time=True,
                          deferred_final_drain=True, duplicate_idempotence=True,
                          interrupted_journal_restart=True, custom_message_service_cdr=True,
                          gps_factors=final['optimizer_reports'][0]['gps_factors'],
                          gps_history_requests=node.gps_requests)
            report_name = 'global_ba_gps_ros2_result.json' if args.gps else 'global_ba_ros2_result.json'
            (REPO / '.runtime' / report_name).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        except Exception:
            print((root / 'backend.log').read_text()); raise
        finally:
            if process and process.poll() is None:
                process.terminate(); process.wait(timeout=10)
            log.close(); node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
