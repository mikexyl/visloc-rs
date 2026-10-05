#!/usr/bin/env python3
"""Follow a running mission and publish an incremental Rerun sparse map.

Uses the native workers' flushed keyframe journals and atomically replaced graph
snapshots as a local, recoverable display transport. No sensor bag, final graph,
ground truth, or ROS callback is needed for map refinement.
"""
import argparse
from collections import OrderedDict, defaultdict
import json
from pathlib import Path
import signal
import time

from landmark_visualization import _transform, identity
from online_landmark_map import OnlineLandmarkMap, RerunLandmarkPublisher
from online_camera import CameraPublisher


class JsonTail:
    def __init__(self, path):
        self.path, self.offset, self.inode = path, 0, None

    def poll(self, limit=16):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return []
        if self.inode != stat.st_ino or stat.st_size < self.offset:
            self.offset, self.inode = 0, stat.st_ino
        records = []
        with self.path.open('rb') as source:
            source.seek(self.offset)
            for _ in range(limit):
                line = source.readline(4 * 1024 * 1024)
                if not line.endswith(b'\n'):
                    if len(line) == 4 * 1024 * 1024:
                        raise ValueError(f'display record exceeds 4 MiB: {self.path}')
                    break  # A writer is partway through a record: retry next tick.
                records.append(json.loads(line))
                self.offset = source.tell()
        return records


class MissionFollower:
    def __init__(self, mission_path, **settings):
        self.root = mission_path.resolve().parent
        self.mission = json.loads(mission_path.read_text())
        self.map = OnlineLandmarkMap(**settings)
        self.tails, self.archives = {}, set()
        self.raw = OrderedDict()
        self.graph = dict(revision=-1, poses=[], loops=[], components=0)
        self.graph_stat = None
        self.read_packets = 0
        self.camera_frames = []
        self.camera_calibration = {}
        self.latest_sensor_ns = self.mission.get('epoch_ns', 0)

    def _raw(self, record):
        self.raw[identity(record['key'])] = record
        # Display trajectories are bounded too; the native graph retains full history.
        while len(self.raw) > 20000:
            self.raw.popitem(last=False)

    def poll(self):
        """Read a bounded batch of complete live records, including session restarts."""
        received = 0
        self.camera_frames = []
        for robot in self.mission['robots']:
            root = self.root / 'robots' / robot['robot']
            for directory in [root, *sorted(root.glob('session-*'))]:
                camera_journal = directory / 'camera_frames.jsonl'
                if camera_journal.exists():
                    if directory not in self.camera_calibration:
                        calibration = directory / 'camera_calibration.json'
                        if calibration.exists():
                            self.camera_calibration[directory] = json.loads(calibration.read_text())
                    if directory in self.camera_calibration:
                        for frame in self.tails.setdefault(camera_journal, JsonTail(camera_journal)).poll(128):
                            frame.update(cameras=self.camera_calibration[directory], _directory=directory)
                            self.camera_frames.append(frame)
                            self.latest_sensor_ns = max(self.latest_sensor_ns, frame['timestamp_ns'])
                keyframes = directory / 'keyframes.jsonl'
                for record in self.tails.setdefault(keyframes, JsonTail(keyframes)).poll(64):
                    self._raw(record)
                journal = directory / 'visualization.jsonl'
                if journal.exists():
                    for packet in self.tails.setdefault(journal, JsonTail(journal)).poll():
                        self._raw(packet['pose'])
                        self.map.ingest(packet['feature'], packet['pose'])
                        received += 1
                else:
                    # Compatibility with existing binaries: selected JIST frames
                    # become available at sequence completion, still during VIO.
                    files = (p for p in sorted((directory / 'sequences').glob('*.json'))
                             if p not in self.archives)
                    for _, path in zip(range(4), files):
                        archive = json.loads(path.read_text())  # Native writer uses atomic rename.
                        for feature in archive['features']:
                            self.map.ingest(feature, self.raw.get(identity(feature['key'])))
                            received += 1
                        self.archives.add(path)
        path = self.root / 'backend/graph_snapshot.json'
        try:
            stat = path.stat()
            version = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if version != self.graph_stat:
                graph = json.loads(path.read_text())
                if graph['revision'] >= self.graph['revision']:
                    self.graph = graph
                self.graph_stat = version
        except FileNotFoundError:
            pass
        self.map.update_graph(self.graph)
        self.read_packets += received
        return received

    def backlog(self):
        return any(t.path.exists() and t.path.stat().st_size > t.offset for t in self.tails.values())

    def finished(self):
        try:
            marker = json.loads((self.root / 'backend/finished.json').read_text())
        except FileNotFoundError:
            return False
        # The final snapshot may be renamed after poll() but before the finish
        # marker becomes visible. Do not exit with the penultimate graph.
        return (self.graph['revision'] >= marker['revision']
                and not self.backlog() and not self.map.dirty)


class MissionPublisher:
    def __init__(self, rr, mission):
        self.rr = rr
        palette = [[60,160,255], [255,140,50], [90,220,130], [210,100,240]]
        self.colors = {r['robot']: palette[i % len(palette)] for i, r in enumerate(mission['robots'])}
        self.landmarks = RerunLandmarkPublisher(rr, self.colors)
        self.cameras = CameraPublisher(rr, mission.get('epoch_ns', 0))
        self.components = set()
        self.camera_paths = ()

    def publish(self, follower, elapsed):
        import rerun.blueprint as rrb
        rr = self.rr
        rr.set_time('live', duration=elapsed)
        if self.cameras.active:
            import numpy as np
            rr.set_time('sensor', duration=np.timedelta64(
                follower.latest_sensor_ns - follower.mission.get('epoch_ns', 0), 'ns'))
        points, audit = follower.map.snapshot()
        poses = {identity(p['key']): p for p in follower.graph['poses']}
        for key, raw in follower.raw.items():
            poses.setdefault(key, dict(key=raw['key'], timestamp_ns=raw['timestamp_ns'],
                                      body_to_map=raw['body_to_odom'],
                                      component=dict(robot=key[0], session=key[1], id=0)))
        groups = defaultdict(list)
        for pose in poses.values():
            c = identity(pose['component'])
            groups[c].append(pose)
        components = set(groups) | {p['component'] for p in points} | {
            c['component'] for c in self.cameras.active.values()}
        def origin(c):
            return f'components/{c[0]}_{c[1]}_{c[2]}'
        for old in self.components - components:
            rr.log(origin(old), rr.Clear(recursive=True))
        camera_paths = tuple(sorted(c['body'] for c in self.cameras.active.values()))
        if components != self.components or camera_paths != self.camera_paths:
            views = [rrb.Spatial3DView(origin=origin(c), name=f'Component: {c[0]} / {c[1][:8]}')
                     for c in sorted(components)]
            follow, images = self.cameras.views()
            rr.send_blueprint(rrb.Blueprint(rrb.Vertical(rrb.Horizontal(*views),
                *([rrb.Horizontal(*follow, *images)] if follow else []),
                rrb.TextDocumentView(origin='status', name='Live map')),
                rrb.TimePanel(timeline='sensor' if camera_paths else 'live', play_state='Following'),
                collapse_panels=True))
        self.components = components
        self.camera_paths = camera_paths
        for component, group in groups.items():
            root = origin(component)
            rr.log(root, rr.ViewCoordinates.RIGHT_HAND_Z_UP)
            sessions = defaultdict(list)
            for p in group:
                sessions[identity(p['key'])[:2]].append(p)
            for (robot, session), trajectory in sessions.items():
                trajectory.sort(key=lambda p: p['timestamp_ns'])
                path = f'{root}/{robot}/trajectory/{session}'
                rr.log(path + '/corrected', rr.LineStrips3D(
                    [[p['body_to_map']['translation'] for p in trajectory]], colors=self.colors[robot]))
                correction = trajectory[0].get('map_from_odom')
                raw = [follower.raw[identity(p['key'])]['body_to_odom']['translation']
                       for p in trajectory if identity(p['key']) in follower.raw]
                if raw:
                    if correction:
                        r, t = _transform(correction)
                        raw = [r @ p + t for p in raw]
                    rr.log(path + '/raw', rr.LineStrips3D([raw], colors=[*self.colors[robot],70]))
            edges = []
            for edge in follower.graph['loops']:
                a, b = (poses.get(identity(edge[k])) for k in ('from', 'to'))
                if a and b and identity(a['component']) == identity(b['component']) == component:
                    edges.append([a['body_to_map']['translation'], b['body_to_map']['translation']])
            rr.log(root + '/loop_edges', rr.LineStrips3D(edges, colors=[255,50,170]))
        self.landmarks.publish(points)
        audit['camera_frames'] = self.cameras.image_frames
        audit['camera_images'] = self.cameras.image_count
        rr.log('status', rr.TextDocument(
            f"Live graph revision: {follower.graph['revision']} | Components: {len(components)}\n"
            f"Display landmarks: {len(points)} | Pending refinements: {audit['pending_tracks']}\n"
            f"Received keyframes: {follower.read_packets} | Evicted tracks: {audit.get('evicted_tracks', 0)}\n"
            'Online point refinement with current poses held fixed. No feedback to VIO.'))
        for key in ('displayed_landmarks', 'pending_tracks', 'observations_retained'):
            rr.log('metrics/map/' + key, rr.Scalars(audit[key]))
        return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mission', type=Path)
    p.add_argument('--connect', help='existing Rerun gRPC endpoint; no viewer/server is spawned')
    p.add_argument('--output', type=Path)
    p.add_argument('--max-tracks', type=int, default=30000)
    p.add_argument('--max-views', type=int, default=8)
    p.add_argument('--min-observations', type=int, default=2)
    p.add_argument('--min-parallax-deg', type=float, default=1)
    p.add_argument('--reprojection-px', type=float, default=3)
    p.add_argument('--exit-when-finished', action='store_true')
    args = p.parse_args()
    follower = MissionFollower(args.mission, **{k: getattr(args, k) for k in (
        'max_tracks', 'max_views', 'min_observations', 'min_parallax_deg', 'reprojection_px')})
    output = args.output or follower.root / 'online.rrd'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        p.error(f'Output already exists: {output}')
    import rerun as rr
    rr.init('visloc live multi-robot map', spawn=False)
    sinks = [rr.FileSink(output)]
    if args.connect:
        sinks.append(rr.GrpcSink(args.connect))
    rr.set_sinks(*sinks)
    publisher = MissionPublisher(rr, follower.mission)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    start, last_publish, idle = time.monotonic(), -float('inf'), 0
    print(f'Following live keyframes and graph updates into {output}', flush=True)
    with output.with_suffix('.map.jsonl').open('w') as log:
        while True:
            received = follower.poll()
            publisher.cameras.publish(follower.camera_frames, follower.graph, time.monotonic() - start)
            follower.map.refine_pending(seconds=.1, max_tracks=64)
            idle = idle + 1 if not received else 0
            finished = args.exit_when_finished and idle >= 2 and follower.finished()
            now = time.monotonic()
            if now - last_publish >= 1 or stopped or finished:
                audit = publisher.publish(follower, now - start)
                log.write(json.dumps(dict(elapsed_s=now-start, **audit)) + '\n')
                log.flush()
                last_publish = now
            if stopped or finished:
                break
            time.sleep(.02)
    rr.get_global_data_recording().flush()
    rr.disconnect()


if __name__ == '__main__':
    main()
