"""Opt-in DA3 for live mission journals, using the sparse map's landmark gates.

Inference, image decoding and fixed-pose landmark refinement belong to the depth
worker. Raw odometry is its metric coordinate frame; graph corrections only move
the displayed clouds, never depth scale, coverage history or estimator state.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
import time

import cv2
import numpy as np

from da3_geometry import Keyframe
from landmark_visualization import _transform, build_landmark_map, identity
from online_da3 import OnlineDepth, Pipeline


def matrix(value):
    rotation, translation = _transform(value)
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


@dataclass
class JournalKeyframe(Keyframe):
    feature: dict
    pose: dict


def journal_keyframe(packet, camera):
    feature, pose = packet['feature'], packet['pose']
    key = identity(feature['key'])
    if (key != identity(pose['key']) or key[:2] != (camera['robot'], camera['session'])
            or feature['timestamp_ns'] != pose['timestamp_ns']
            or feature['timestamp_ns'] != camera['timestamp_ns']):
        raise ValueError('DA3 camera and keyframe identities/timestamps must match exactly')
    directory = Path(camera['_directory']).resolve()
    image = (directory / camera['images'][0]).resolve()
    if not image.is_relative_to(directory):
        raise ValueError('DA3 image must be inside its session directory')
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    calibration = feature['camera']
    if gray is None or gray.shape != (calibration['height'], calibration['width']):
        raise ValueError(f'Invalid DA3 keyframe image: {image}')
    if camera['cameras'][0] != calibration:
        raise ValueError('DA3 image calibration differs from keyframe calibration')
    fx, fy, cx, cy = calibration['intrinsics']
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    c2w = matrix(pose['body_to_odom']) @ matrix(calibration['camera_to_body'])
    # Geometry is filled only after multi-view filtering. There is no raw fallback.
    return JournalKeyframe(camera['frame_id'], key[2], feature['timestamp_ns'], gray,
        intrinsics, c2w, np.empty(0, np.uint64), np.empty((0, 2)), np.empty((0, 3)), feature, pose)


class FilteredPipeline(Pipeline):
    def __init__(self, *args, quality=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality or dict(min_observations=2, min_parallax_deg=1., reprojection_px=3.)
        self.owner = None

    def accept(self, item):
        frame = journal_keyframe(*item)
        owner = identity(frame.feature['key'])[:2]
        if self.owner is not None and owner != self.owner:
            raise ValueError('A DA3 pipeline must not mix robots or estimator sessions')
        self.owner = owner
        if self.window and frame.ordinal <= self.window[-1].ordinal:
            self.stats['duplicate_keyframes'] = self.stats.get('duplicate_keyframes', 0) + 1
            return
        super().accept(frame)

    def prepare_window(self, window):
        started = time.monotonic()
        poses = {identity(f.feature['key']): dict(
            key=f.feature['key'], body_to_map=f.pose['body_to_odom'],
            component=dict(robot=self.owner[0], session=self.owner[1], id=0)) for f in window}
        points, audit = build_landmark_map([f.feature for f in window], poses, **self.quality)
        by_id = {p['track_id']: p['position'] for p in points}
        filtered, retained = [], []
        for f in window:
            ids, pixels, world = [], [], []
            seen = set()
            for track, pixel in zip(f.feature['track_ids'], f.feature['pixels']):
                if track in seen or track not in by_id:
                    continue
                seen.add(track)
                point = by_id[track]
                pc = (point - f.camera_to_world[:3, 3]) @ f.camera_to_world[:3, :3]
                uv = f.intrinsics @ pc
                measured = np.asarray(pixel)
                h, w = f.gray.shape
                if (not np.isfinite(measured).all() or not .1 < pc[2] < self.config.max_depth_m
                        or not 0 <= measured[0] < w or not 0 <= measured[1] < h
                        or np.linalg.norm(uv[:2] / pc[2] - measured) > self.quality['reprojection_px']):
                    continue  # A valid track can still have an outlying observation.
                ids.append(track); pixels.append(pixel); world.append(point)
            filtered.append(replace(f, track_ids=np.array(ids, np.uint64),
                pixels=np.asarray(pixels).reshape(-1, 2), points_world=np.asarray(world).reshape(-1, 3)))
            retained.append(len(ids))
        return filtered, dict(
            keyframe_keys=[f.feature['key'] for f in window],
            raw_body_poses=[f.pose['body_to_odom'] for f in window],
            camera_to_body=[f.feature['camera']['camera_to_body'] for f in window],
            landmark_filter=dict(audit, retained_observations_by_view=retained,
                filter_ms=(time.monotonic() - started) * 1000,
                scope='current five actual keyframes; raw VIO poses fixed; no future observations'))

    def archive_fields(self, window):
        return dict(landmark_view_indices=np.concatenate([np.full(len(f.track_ids), i, np.uint8)
                                                         for i, f in enumerate(window)]),
                    landmark_ids=np.concatenate([f.track_ids for f in window]),
                    landmark_pixels=np.concatenate([f.pixels for f in window]),
                    landmark_points_world=np.concatenate([f.points_world for f in window]))


class MissionDepth:
    """Bounded asynchronous join of independent image and keyframe journals."""
    def __init__(self, config, output, quality, worker_factory=OnlineDepth):
        self.config, self.output, self.quality = config, Path(output), quality
        self.worker_factory = worker_factory
        self.cameras, self.pending, self.workers = OrderedDict(), OrderedDict(), {}
        self.last_submitted, self.counts = {}, Counter()
        self.closed = False

    @staticmethod
    def camera_key(frame):
        return frame['robot'], frame['session'], frame['timestamp_ns']

    def ingest(self, packets, cameras):
        for camera in cameras:
            self.cameras[self.camera_key(camera)] = camera
        for packet in packets:
            feature = packet['feature']
            key = (*identity(feature['key'])[:2], feature['timestamp_ns'])
            self.pending[key] = packet
        for key, packet in list(self.pending.items()):
            camera = self.cameras.get(key)
            if camera is None:
                continue
            owner, ordinal = key[:2], packet['feature']['key']['id']
            if ordinal <= self.last_submitted.get(owner, -1):
                self.counts['duplicate_keyframes'] += 1
            else:
                if owner not in self.workers:
                    self.workers[owner] = self.worker_factory(self.config,
                        self.output / owner[0] / owner[1],
                        pipeline_factory=partial(FilteredPipeline, quality=self.quality))
                self.workers[owner].submit((packet, camera))
                self.last_submitted[owner] = ordinal
                self.counts['matched_keyframes'] += 1
            del self.pending[key]
        while len(self.cameras) > 2048:
            self.cameras.popitem(last=False)
            self.counts['retired_camera_metadata'] += 1
        while len(self.pending) > 64:
            self.pending.popitem(last=False)
            self.counts['missing_image_drops'] += 1

    def poll(self):
        return [event for worker in self.workers.values() for event in worker.poll()]

    def finish(self):
        self.counts['missing_image_drops'] += len(self.pending)
        self.pending.clear()
        result = dict(join=dict(self.counts), sessions={f'{a}/{b}': worker.finish()
                      for (a, b), worker in self.workers.items()})
        self.closed = True
        return result

    def abort(self):
        for worker in self.workers.values():
            worker.abort()


class DensePublisher:
    """Store camera-local samples once and move their entities with each graph."""
    def __init__(self, rr, stride):
        self.rr, self.stride = rr, stride
        self.windows = []
        self.revision = None
        self.points = 0

    def publish(self, events, graph):
        rr = self.rr
        for event in events:
            self.windows.append(dict(event=event, paths={}, logged=False))
        if not events and self.revision == graph['revision']:
            return
        poses = {identity(p['key']): p for p in graph['poses']}
        for window in self.windows:
            event = window['event']
            data = None
            for i, key in enumerate(event['keyframe_keys']):
                owner = identity(key)
                pose = poses.get(owner)
                component = identity(pose['component']) if pose else (*owner[:2], 0)
                body = pose['body_to_map'] if pose else event['raw_body_poses'][i]
                camera = matrix(body) @ matrix(event['camera_to_body'][i])
                c = component
                root = (f'components/{c[0]}_{c[1]}_{c[2]}/{owner[0]}/dense/{owner[1]}'
                        f"/window_{event['window_index']:04d}/keyframe_{owner[2]}")
                old = window['paths'].get(i)
                if old and old != root:
                    rr.log(old, rr.Clear(recursive=True))
                rr.log(root, rr.Transform3D(translation=camera[:3, 3], mat3x3=camera[:3, :3]))
                if not window['logged'] or old != root:
                    if data is None:
                        data = np.load(event['archive'], allow_pickle=False)
                    depth, gray, k = data['depth_m'][i], data['gray'][i], data['intrinsics'][i]
                    y, x = np.mgrid[0:depth.shape[0]:self.stride, 0:depth.shape[1]:self.stride]
                    z = depth[y, x]
                    good = np.isfinite(z) & (z > 0)
                    pc = np.column_stack(((x[good]-k[0, 2])*z[good]/k[0, 0],
                                          (y[good]-k[1, 2])*z[good]/k[1, 1], z[good]))
                    rr.log(root + '/points', rr.Points3D(pc,
                        colors=np.repeat(gray[y, x][good, None], 3, axis=1), radii=.025))
                    if not window['logged']:
                        self.points += len(pc)
                    rr.log(f'da3/{owner[0]}/depth', rr.DepthImage(depth, meter=1.))
                window['paths'][i] = root
            if data is not None:
                data.close()
            window['logged'] = True
        self.revision = graph['revision']
        rr.log('da3/status', rr.TextDocument(
            f'DA3 dense mapping — WIP\nAccepted windows: {len(self.windows)} | Samples: {self.points}\n'
            'Five actual VIO keyframes; shared multi-view landmark filtering before FOV gating.\n'
            'Confidence and depth reprojection gates apply before display. Clouds follow PGO poses.'))
