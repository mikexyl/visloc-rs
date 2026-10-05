"""Incremental, bounded display map. Only observations/poses received so far count.

Run in a visualization worker/process, never in a sensor callback. Landmark
coordinates are anchored to a retained observing body pose, so PGO moves existing
points immediately while bounded fixed-pose refinement catches up.
"""
from collections import Counter, OrderedDict, defaultdict
import math
import time

import numpy as np

from landmark_visualization import _transform, build_landmark_map, identity


class OnlineLandmarkMap:
    def __init__(self, max_tracks=30000, max_views=8, min_observations=2,
                 min_parallax_deg=1.0, reprojection_px=3.0):
        if max_tracks < 1 or not 2 <= min_observations <= max_views:
            raise ValueError('invalid display map capacity')
        if not 0 < min_parallax_deg < 90 or not math.isfinite(reprojection_px) or reprojection_px <= 0:
            raise ValueError('invalid display quality gates')
        self.max_tracks, self.max_views = max_tracks, max_views
        self.settings = dict(min_observations=min_observations,
                             min_parallax_deg=min_parallax_deg, reprojection_px=reprojection_px)
        self.tracks = OrderedDict()
        self.by_frame = defaultdict(set)
        self.poses = {}
        self.dirty = OrderedDict()
        self.revision = -1
        self.counts = Counter()

    def _remove_view(self, track, key):
        del self.tracks[track]['views'][key]
        self.by_frame[key].discard(track)
        if not self.by_frame[key]:
            del self.by_frame[key]
            self.poses.pop(key, None)

    def ingest(self, feature, raw_pose=None):
        """Upsert one actual keyframe. Repeated delivery never adds a view."""
        key = identity(feature['key'])
        ids, pixels, points = (feature[k] for k in ('track_ids', 'pixels', 'points_camera'))
        if not len(ids) == len(pixels) == len(points):
            raise ValueError('misaligned display feature arrays')
        for track_id, pixel, point in zip(ids, pixels, points):
            track = (*key[:2], int(track_id))
            if track not in self.tracks:
                if len(self.tracks) == self.max_tracks:
                    old = next(iter(self.tracks))
                    for view in list(self.tracks[old]['views']):
                        self._remove_view(old, view)
                    del self.tracks[old]
                    self.dirty.pop(old, None)
                    self.counts['evicted_tracks'] += 1
                self.tracks[track] = dict(views=OrderedDict(), points=[], rejected={})
            state = self.tracks[track]
            view = dict(key=feature['key'], camera=feature['camera'],
                        track_ids=[track_id], pixels=[pixel], points_camera=[point])
            if state['views'].get(key) == view:
                self.counts['duplicate_observations'] += 1
                continue
            state['views'][key] = view
            # Keep the oldest ray plus the most recent views for parallax.
            if len(state['views']) > self.max_views:
                self._remove_view(track, list(state['views'])[1])
                self.counts['retired_views'] += 1
            self.by_frame[key].add(track)
            self.tracks.move_to_end(track)
            self.dirty[track] = None
            self.counts['observations'] += 1
        if raw_pose is not None and key in self.by_frame and key not in self.poses:
            _transform(raw_pose['body_to_odom'])  # Reject nonfinite poses before storing.
            self.poses[key] = dict(key=feature['key'], body_to_map=raw_pose['body_to_odom'],
                                   component=dict(robot=key[0], session=key[1], id=0))
        self.counts['keyframe_packets'] += 1

    def update_graph(self, graph):
        """Apply a newer (or duplicate) snapshot; stale delivery cannot undo PGO."""
        if graph['revision'] < self.revision:
            self.counts['stale_graphs'] += 1
            return
        # Only retain poses referenced by the bounded display observation cache.
        updates = {}
        for pose in graph['poses']:
            key = identity(pose['key'])
            if key not in self.by_frame:
                continue
            r, t = _transform(pose['body_to_map'])
            old = self.poses.get(key)
            changed = old is None or old['component'] != pose['component']
            if old is not None and not changed:
                old_r, old_t = _transform(old['body_to_map'])
                # Ignore round-trip floating-point noise in unchanged graph poses.
                changed = (not np.allclose(t, old_t, rtol=0, atol=1e-9)
                           or not np.allclose(r, old_r, rtol=0, atol=1e-12))
            if changed:
                updates[key] = pose
        for key, pose in updates.items():
            self.poses[key] = pose
            for track in self.by_frame[key]:
                self.dirty[track] = None
        self.revision = graph['revision']
        self.counts['pose_updates'] += len(updates)

    def refine_pending(self, seconds=.1, max_tracks=64):
        """Bound work per tick; unfinished tracks remain coalesced in the queue."""
        deadline = time.monotonic() + seconds
        done = 0
        while self.dirty and done < max_tracks and time.monotonic() < deadline:
            track, _ = self.dirty.popitem(last=False)
            state = self.tracks[track]
            points, audit = build_landmark_map(state['views'].values(), self.poses, **self.settings)
            state['points'] = []
            for point in points:
                # Anchor to the oldest retained view: it is preserved by the view budget.
                anchor = next(k for k in state['views'] if k in self.poses
                              and identity(self.poses[k]['component']) == point['component'])
                r, t = _transform(self.poses[anchor]['body_to_map'])
                point['anchor'] = anchor
                point['anchor_position'] = r.T @ (point.pop('position') - t)
                point['refined_revision'] = self.revision
                state['points'].append(point)
            state['rejected'] = audit['rejected']
            done += 1
        self.counts['refinements'] += done
        return done

    def snapshot(self):
        """Fresh coordinates even if correction-triggered refinement is pending."""
        points = []
        for track, state in self.tracks.items():
            emitted = set()
            for point in state['points']:
                pose = self.poses.get(point['anchor'])
                if pose is None:
                    continue
                component = identity(pose['component'])
                # Component joining can merge two formerly independent copies.
                if component in emitted:
                    continue
                emitted.add(component)
                r, t = _transform(pose['body_to_map'])
                points.append({k: v for k, v in point.items() if k not in ('anchor', 'anchor_position')} |
                              dict(component=component, position=r @ point['anchor_position'] + t,
                                   refinement_pending=track in self.dirty))
        rejected = Counter()
        for state in self.tracks.values():
            rejected.update(state['rejected'])
        return points, dict(self.counts, graph_revision=self.revision, tracks=len(self.tracks),
                            observations_retained=sum(len(t['views']) for t in self.tracks.values()),
                            poses_retained=len(self.poses), pending_tracks=len(self.dirty),
                            displayed_landmarks=len(points), rejected=dict(rejected),
                            max_tracks=self.max_tracks, max_views=self.max_views, settings=self.settings)


class RerunLandmarkPublisher:
    """Replace stable clouds on a live timeline; clear retired component entities."""
    def __init__(self, rr, colors):
        self.rr, self.colors = rr, colors
        self.entities = set()

    def publish(self, points):
        rr = self.rr
        groups = defaultdict(list)
        for point in points:
            c = point['component']
            entity = f"components/{c[0]}_{c[1]}_{c[2]}/{point['robot']}/landmarks/{point['session']}"
            groups[entity].append(point)
        for entity in self.entities - groups.keys():
            rr.log(entity, rr.Clear(recursive=True))
        for entity, cloud in groups.items():
            rr.log(entity, rr.Points3D([p['position'] for p in cloud],
                                     colors=self.colors[cloud[0]['robot']], radii=rr.Radius.ui_points(1)),
                   rr.AnyValues(landmark_id=[str(p['track_id']) for p in cloud],
                                inlier_views=[p['inliers'] for p in cloud],
                                reprojection_rmse_px=[p['reprojection_px'] for p in cloud],
                                refinement_pending=[p['refinement_pending'] for p in cloud]))
        self.entities = set(groups)
