#!/usr/bin/env python3
"""Publish recorded camera/IMU to the native Rust nodes using actual ROS2 DDS.

Original timestamps and ground truth stay outside the estimator. All robots
share one /clock, with each bag's exact integer sensor intervals preserved.
"""
import argparse
import array
import heapq
import json
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from visloc_msgs.msg import Status, SequenceAnnouncement, LoopConstraint, GraphSnapshot, GpsFix, GpsStatus
from visloc_msgs.srv import Finish, GetGpsHistory
from replay_sensor_source import open_source

def stamp(message, ns):
    message.sec, message.nanosec = divmod(int(ns), 1_000_000_000)

class Replay(Node):
    def __init__(self, mission, output):
        super().__init__('visloc_graco_replay')
        self.mission, self.output = mission, output
        self.status, self.ack, self.bytes = {}, {}, {}
        self.latest_graph = None
        self.gps_status = None
        self.subs = []
        # Keep the node attached across callbacks. The convenience spin_once
        # function otherwise adds/removes it and wakes/rebuilds the wait set
        # for every high-rate status or odometry message.
        self.replay_executor = SingleThreadedExecutor(context=self.context)
        self.replay_executor.add_node(self)
        reliable = QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE)
        status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.clock = self.create_publisher(Clock, '/clock', 10)
        self.streams = []
        for robot in mission['robots']:
            name = robot['robot']
            robot_config = json.loads(Path(robot['config']).read_text())
            stereo = robot_config.get('camera_mode', 'mono') == 'stereo'
            if stereo != (robot.get('camera_mode', 'mono') == 'stereo'):
                raise ValueError('Mission and robot camera modes disagree')
            robot['stationary_startup'] = robot_config.get('imu_startup', {}) is not None
            self.subs.append(self.create_subscription(Status, f'/{name}/slam/status', lambda m, n=name: self.status.update({n: m}), status_qos))
            self.subs.append(self.create_subscription(Odometry, f'/{name}/vio/odometry', lambda m, n=name: self.ack.update({n: m.header.stamp.sec * 10**9 + m.header.stamp.nanosec}), reliable))
            source = open_source(robot)
            frames, imu, times = source.frames, source.imu, source.times
            self.streams.append(dict(robot=robot, source=source, frames=frames, imu=imu, times=times, imu_index=0,
                image_pub=self.create_publisher(Image, f'/{name}/camera/image', reliable),
                right_pub=self.create_publisher(Image, f'/{name}/camera_right/image', reliable) if stereo else None,
                imu_pub=self.create_publisher(Imu, f'/{name}/imu', QoSProfile(depth=2048, reliability=ReliabilityPolicy.RELIABLE)),
                finish=self.create_client(Finish, f'/{name}/slam/finish')))
            stream = self.streams[-1]
            stream['gps_index'] = 0
            stream['gps'] = []
            stream['gps_pub'] = None
            if robot_config.get('gps', {}).get('enabled', True) and robot.get('gps_records'):
                stream['gps'] = [record for line in Path(robot['gps_records']).read_text().splitlines()
                                 if (record := json.loads(line))['timestamp_ns'] <= frames[-1][0]]
                stream['gps_pub'] = self.create_publisher(GpsFix, f'/{name}/gps/normalized', reliable)
                stream['gps_finish'] = self.create_client(Finish, f'/{name}/slam/gps_finish')
                stream['gps_history'] = self.create_client(GetGpsHistory, f'/{name}/slam/gps_history')
                self.subs.append(self.create_subscription(GpsFix, f'/{name}/slam/gps', lambda m: self.count('gps_normalized_output',m), reliable))
        self.subs.append(self.create_subscription(GpsStatus, '/visloc/gps_status', lambda m: setattr(self, 'gps_status', m), status_qos))
        self.backend_finish = self.create_client(Finish, '/visloc/backend/finish')
        backend_config = json.loads(Path(mission['backend_config']).read_text())
        self.backend_finished = Path(backend_config['output']) / 'finished.json'
        self.subs.append(self.create_subscription(SequenceAnnouncement, '/visloc/sequences', lambda m: self.count('sequence_announcements', m), reliable))
        self.subs.append(self.create_subscription(LoopConstraint, '/visloc/loops', lambda m: self.count('loop_constraints', m), reliable))
        self.subs.append(self.create_subscription(GraphSnapshot, '/visloc/graph', self.graph, status_qos))

    def count(self, name, message):
        self.bytes[name] = self.bytes.get(name, 0) + len(serialize_message(message))

    def graph(self, message):
        self.latest_graph = message
        self.count('optimized_graph', message)

    def spin_until(self, condition, timeout, reason):
        start = time.monotonic()
        while not condition():
            self.replay_executor.spin_once(timeout_sec=.05)
            for name, status in self.status.items():
                if status.error:
                    raise RuntimeError(f'{name}: {status.error}')
            if time.monotonic() - start > timeout:
                raise TimeoutError(reason)

    def run(self):
        self.spin_until(lambda: len(self.status) == len(self.streams) and all(s.ready for s in self.status.values()), 180, 'robot readiness')
        self.spin_until(lambda: all(s['image_pub'].get_subscription_count() and s['imu_pub'].get_subscription_count()
                                   and (s['right_pub'] is None or s['right_pub'].get_subscription_count())
                                   and (s['gps_pub'] is None or s['gps_pub'].get_subscription_count())
                                   for s in self.streams), 30, 'sensor discovery')
        queue = [(s['frames'][0][0] + s['robot']['offset_ns'], i, 0) for i, s in enumerate(self.streams)]
        heapq.heapify(queue)
        started = time.monotonic()
        timings = []
        timing_log = (self.output / 'replay_timing.jsonl').open('w', buffering=1)
        while queue:
            group = [heapq.heappop(queue)]
            while queue and queue[0][0] - group[0][0] < 1_000_000:
                group.append(heapq.heappop(queue))
            deadline = started + (group[-1][0] - self.mission['epoch_ns']) / 1e9 / self.mission['rate']
            while time.monotonic() < deadline:
                self.replay_executor.spin_once(timeout_sec=max(0., min(.02, deadline - time.monotonic())))
            batch_started, cpu_started = time.monotonic(), time.process_time()
            clock = Clock(); stamp(clock.clock, group[-1][0]); self.clock.publish(clock)
            waiting, finishing = [], []
            for ns, robot_index, index in group:
                stream = self.streams[robot_index]
                robot, frames = stream['robot'], stream['frames']
                original_ns, row, right_row = frames[index]
                # One sample at/after the image makes the IMU watermark causal and
                # reproduces the existing adapter's first-image initialization.
                end = min(len(stream['imu']), int(np.searchsorted(stream['times'], original_ns, side='left')) + 1)
                for sample in stream['imu'][stream['imu_index']:end]:
                    m = Imu(); stamp(m.header.stamp, sample[0] + robot['offset_ns']); m.header.frame_id = f"{robot['robot']}/base_link"
                    m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = sample[1:4]
                    m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = sample[4:7]
                    stream['imu_pub'].publish(m)
                stream['imu_index'] = end
                raw = stream['source'].image(row)
                m = Image(); stamp(m.header.stamp, ns); m.header.frame_id = f"{robot['robot']}/camera"
                m.width, m.height, m.step, m.encoding, m.is_bigendian = raw.width, raw.height, raw.step, raw.encoding, raw.is_bigendian
                m.data = array.array('B', np.asarray(raw.data, dtype=np.uint8).tobytes()); stream['image_pub'].publish(m)
                if stream['right_pub'] is not None:
                    if right_row is None:
                        raise ValueError('Stereo replay frame is missing its right image')
                    raw = stream['source'].right_image(right_row)
                    m = Image(); stamp(m.header.stamp, ns); m.header.frame_id = f"{robot['robot']}/camera_right"
                    m.width, m.height, m.step, m.encoding, m.is_bigendian = raw.width, raw.height, raw.step, raw.encoding, raw.is_bigendian
                    m.data = array.array('B', np.asarray(raw.data, dtype=np.uint8).tobytes()); stream['right_pub'].publish(m)
                while stream['gps_index'] < len(stream['gps']) and stream['gps'][stream['gps_index']]['timestamp_ns'] <= original_ns:
                    record = stream['gps'][stream['gps_index']]
                    gps = GpsFix()
                    gps.key.robot = robot['robot']; gps.key.session = 'replay'; gps.key.id = record['key']['id']
                    gps.timestamp_ns = record['timestamp_ns'] + robot['offset_ns']
                    gps.receipt_timestamp_ns = record['receipt_timestamp_ns'] + robot['offset_ns']
                    gps.time_source = record['time_source']
                    gps.has_position = record['lla'] is not None; gps.lla = record['lla'] or [0.] * 3
                    gps.status = record['status']
                    gps.has_quality = record['quality'] is not None; gps.quality = record['quality'] or 0
                    gps.has_hdop = record['hdop'] is not None; gps.hdop = record['hdop'] or 0.
                    gps.has_covariance = record['covariance_enu'] is not None; gps.covariance_enu = record['covariance_enu'] or [0.] * 9
                    stream['gps_pub'].publish(gps); self.count('gps_input', gps)
                    stream['gps_index'] += 1
                waiting.append((robot['robot'], ns, robot['stationary_startup']))
                if index + 1 < len(frames):
                    heapq.heappush(queue, (frames[index + 1][0] + robot['offset_ns'], robot_index, index + 1))
                else:
                    finishing.append((stream['finish'], ns))
                if index % 100 == 0:
                    print(f"{robot['robot']} frame={index+1}/{len(frames)}", flush=True)
            # Stationary startup acknowledges ingestion while buffering; waiting
            # for odometry here would deadlock before its first second arrives.
            # Legacy mode retains the original one-result-at-a-time contract.
            publish_done = time.monotonic()
            self.spin_until(lambda: all(
                (self.status[robot].input_timestamp_ns if startup and self.status[robot].initializing
                 else self.ack.get(robot, -1)) >= ns
                for robot, ns, startup in waiting), 30, 'concurrent VIO/startup acknowledgements')
            batch_done = time.monotonic()
            timing = {'elapsed_s': batch_done-started,
                      'frames': {self.streams[i]['robot']['robot']: index+1 for _, i, index in group},
                      'publish_ms': (publish_done-batch_started)*1000,
                      'ack_wait_ms': (batch_done-publish_done)*1000,
                      'cpu_ms': (time.process_time()-cpu_started)*1000}
            timings.append(timing)
            timing_log.write(json.dumps(timing)+'\n')
            for client, ns in finishing:
                future = client.call_async(Finish.Request(last_image_timestamp_ns=ns))
                self.spin_until(future.done, 15, 'robot finish request')
                if not future.result().accepted:
                    raise RuntimeError('Robot refused finish request')
        timing_log.close()
        self.spin_until(lambda: all(s.finished for s in self.status.values()), 180, 'loop workers draining')
        # Let history recovery deliver the final graph tail before the final solve.
        expected = sum(s.keyframes for s in self.status.values())
        expected_loops = sum(s.loops for s in self.status.values())
        self.spin_until(lambda: self.latest_graph is not None and len(self.latest_graph.poses) == expected and len(self.latest_graph.loops) == expected_loops, 180, 'complete centralized graph')
        expected_gps = sum(len(stream['gps']) for stream in self.streams)
        for stream in self.streams:
            if stream['gps_pub'] is None:
                continue
            expected_records = len(stream['gps'])
            deadline = time.monotonic() + 30
            while True:
                future = stream['gps_history'].call_async(GetGpsHistory.Request(cursor=expected_records))
                self.spin_until(future.done, 3, 'GPS history drain')
                page = future.result()
                if page.dropped_inputs or page.rejected_inputs:
                    raise RuntimeError(f'GPS ingestion drops/rejections: {page.dropped_inputs}/{page.rejected_inputs}')
                if page.cursor == expected_records:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('GPS input records missing from history')
            future = stream['gps_finish'].call_async(Finish.Request())
            self.spin_until(future.done, 3, 'GPS finish')
            if not future.result().accepted:
                raise RuntimeError('GPS worker refused drain')
        if expected_gps:
            self.spin_until(lambda: self.gps_status is not None and self.gps_status.total_records == expected_gps,
                            60, 'complete GPS graph records')
        previous_revision = self.latest_graph.revision
        future = self.backend_finish.call_async(Finish.Request(last_image_timestamp_ns=0))
        self.spin_until(future.done, 15, 'backend final solve request')
        if not future.result().accepted:
            raise RuntimeError('Backend refused final solve')
        def final_graph_received():
            # Finish acknowledges queueing, not completion. An ordinary solve
            # already in flight can also advance the revision. The supervised
            # local backend's atomic completion record identifies the actual
            # final solve; still require its snapshot to arrive through DDS.
            if not self.backend_finished.exists():
                return False
            finished = json.loads(self.backend_finished.read_text())
            return (self.latest_graph.revision > previous_revision
                    and self.latest_graph.revision >= finished['revision']
                    and len(self.latest_graph.poses) == expected
                    and len(self.latest_graph.loops) == expected_loops)
        self.spin_until(final_graph_received,
                        180, 'final optimized graph publication')
        summary = {'wall_seconds': time.monotonic()-started, 'robots': {}, 'topic_cdr_bytes': self.bytes,
                   'final_graph_revision': self.latest_graph.revision,
                   'final_graph_input_revision': self.latest_graph.input_revision,
                   'topic_cdr_bytes_note': 'Serialized message payloads observed once per publication; excludes DDS framing and service traffic.'}
        summary['gps_records_published'] = expected_gps
        summary['replay_timing'] = {key: {'median_ms': float(np.median([v[key] for v in timings])),
                                        'p95_ms': float(np.percentile([v[key] for v in timings], 95))}
                                   for key in ('publish_ms', 'ack_wait_ms', 'cpu_ms')}
        for name, s in self.status.items():
            summary['robots'][name] = {field: getattr(s, field) for field in ('session', 'frames', 'frames_ingested', 'initialization_skipped_frames', 'keyframes', 'sequences', 'loops', 'dropped_images', 'dropped_keyframes', 'request_failures', 'error')}
        (self.output / 'replay_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        for stream in self.streams:
            stream['source'].close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mission', type=Path)
    args = p.parse_args()
    rclpy.init()
    node = Replay(json.loads(args.mission.read_text()), args.mission.parent)
    try:
        node.run()
    finally:
        node.replay_executor.shutdown()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
