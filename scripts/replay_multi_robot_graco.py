#!/usr/bin/env python3
"""Publish GRACO camera/IMU to the native Rust nodes using actual ROS2 DDS.

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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from visloc_msgs.msg import Status, SequenceAnnouncement, LoopConstraint, GraphSnapshot
from visloc_msgs.srv import Finish
from run_graco_vio import Bag, load_imu

def stamp(message, ns):
    message.sec, message.nanosec = divmod(int(ns), 1_000_000_000)

class Replay(Node):
    def __init__(self, mission, output):
        super().__init__('visloc_graco_replay')
        self.mission, self.output = mission, output
        self.status, self.ack, self.bytes = {}, {}, {}
        self.latest_graph = None
        self.subs = []
        reliable = QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE)
        status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.clock = self.create_publisher(Clock, '/clock', 10)
        self.streams = []
        for robot in mission['robots']:
            name = robot['robot']
            robot_config = json.loads(Path(robot['config']).read_text())
            robot['stationary_startup'] = robot_config.get('imu_startup', {}) is not None
            self.subs.append(self.create_subscription(Status, f'/{name}/slam/status', lambda m, n=name: self.status.update({n: m}), status_qos))
            self.subs.append(self.create_subscription(Odometry, f'/{name}/vio/odometry', lambda m, n=name: self.ack.update({n: m.header.stamp.sec * 10**9 + m.header.stamp.nanosec}), reliable))
            bag = Bag(Path(robot['bag']))
            frames = bag.stereo_index()[:robot['frames']]
            imu, times = load_imu(bag)
            self.streams.append(dict(robot=robot, bag=bag, frames=frames, imu=imu, times=times, imu_index=0,
                image_pub=self.create_publisher(Image, f'/{name}/camera/image', reliable),
                imu_pub=self.create_publisher(Imu, f'/{name}/imu', QoSProfile(depth=2048, reliability=ReliabilityPolicy.RELIABLE)),
                finish=self.create_client(Finish, f'/{name}/slam/finish')))
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
            rclpy.spin_once(self, timeout_sec=.05)
            for name, status in self.status.items():
                if status.error:
                    raise RuntimeError(f'{name}: {status.error}')
            if time.monotonic() - start > timeout:
                raise TimeoutError(reason)

    def run(self):
        self.spin_until(lambda: len(self.status) == len(self.streams) and all(s.ready for s in self.status.values()), 180, 'robot readiness')
        self.spin_until(lambda: all(s['image_pub'].get_subscription_count() and s['imu_pub'].get_subscription_count() for s in self.streams), 30, 'sensor discovery')
        queue = [(s['frames'][0][0] + s['robot']['offset_ns'], i, 0) for i, s in enumerate(self.streams)]
        heapq.heapify(queue)
        started = time.monotonic()
        while queue:
            group = [heapq.heappop(queue)]
            while queue and queue[0][0] - group[0][0] < 1_000_000:
                group.append(heapq.heappop(queue))
            deadline = started + (group[-1][0] - self.mission['epoch_ns']) / 1e9 / self.mission['rate']
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=min(.02, deadline - time.monotonic()))
            clock = Clock(); stamp(clock.clock, group[-1][0]); self.clock.publish(clock)
            waiting, finishing = [], []
            for ns, robot_index, index in group:
                stream = self.streams[robot_index]
                robot, frames, bag = stream['robot'], stream['frames'], stream['bag']
                original_ns, row, _ = frames[index]
                # One sample at/after the image makes the IMU watermark causal and
                # reproduces the existing adapter's first-image initialization.
                end = min(len(stream['imu']), int(np.searchsorted(stream['times'], original_ns, side='left')) + 1)
                for sample in stream['imu'][stream['imu_index']:end]:
                    m = Imu(); stamp(m.header.stamp, sample[0] + robot['offset_ns']); m.header.frame_id = f"{robot['robot']}/base_link"
                    m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = sample[1:4]
                    m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = sample[4:7]
                    stream['imu_pub'].publish(m)
                stream['imu_index'] = end
                tid, _ = bag.topics['/camera_left/image_raw']
                data = bag.connection.execute('SELECT data FROM messages WHERE id=? AND topic_id=?', (row, tid)).fetchone()[0]
                raw = bag.types.deserialize_cdr(data, 'sensor_msgs/msg/Image')
                m = Image(); stamp(m.header.stamp, ns); m.header.frame_id = f"{robot['robot']}/camera"
                m.width, m.height, m.step, m.encoding, m.is_bigendian = raw.width, raw.height, raw.step, raw.encoding, raw.is_bigendian
                m.data = array.array('B', np.asarray(raw.data, dtype=np.uint8).tobytes()); stream['image_pub'].publish(m)
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
            self.spin_until(lambda: all(
                (self.status[robot].input_timestamp_ns if startup and self.status[robot].initializing
                 else self.ack.get(robot, -1)) >= ns
                for robot, ns, startup in waiting), 30, 'concurrent VIO/startup acknowledgements')
            for client, ns in finishing:
                future = client.call_async(Finish.Request(last_image_timestamp_ns=ns))
                self.spin_until(future.done, 15, 'robot finish request')
                if not future.result().accepted:
                    raise RuntimeError('Robot refused finish request')
        self.spin_until(lambda: all(s.finished for s in self.status.values()), 180, 'loop workers draining')
        # Let history recovery deliver the final graph tail before the final solve.
        expected = sum(s.keyframes for s in self.status.values())
        expected_loops = sum(s.loops for s in self.status.values())
        self.spin_until(lambda: self.latest_graph is not None and len(self.latest_graph.poses) == expected and len(self.latest_graph.loops) == expected_loops, 180, 'complete centralized graph')
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
        for name, s in self.status.items():
            summary['robots'][name] = {field: getattr(s, field) for field in ('session', 'frames', 'frames_ingested', 'initialization_skipped_frames', 'keyframes', 'sequences', 'loops', 'dropped_images', 'dropped_keyframes', 'request_failures', 'error')}
        (self.output / 'replay_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        for stream in self.streams:
            stream['bag'].connection.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mission', type=Path)
    args = p.parse_args()
    rclpy.init()
    node = Replay(json.loads(args.mission.read_text()), args.mission.parent)
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
