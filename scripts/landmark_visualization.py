"""Sparse display-map geometry with fixed supplied poses, never fed to VIO/PGO.

Identity is (graph component, robot, session, VIO track ID). This deliberately
does not merge unrelated tracks or infer cross-robot landmark associations.
"""
from collections import Counter, defaultdict
from itertools import combinations
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def identity(key):
    return key['robot'], key['session'], key['id']


def _transform(value):
    rotation = Rotation.from_quat(value['rotation_xyzw']).as_matrix()
    translation = np.asarray(value['translation'], dtype=float)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('nonfinite camera transform')
    return rotation, translation


def _intersect(centers, rays):
    """Least-squares intersection of unit bearing rays in map coordinates."""
    projectors = np.eye(3) - rays[:, :, None] * rays[:, None, :]
    a = projectors.sum(axis=0)
    if np.linalg.cond(a) > 1e10:
        return None
    return np.linalg.solve(a, np.einsum('nij,nj->i', projectors, centers))


def _refine(observations, min_observations, min_parallax_deg, threshold):
    n = len(observations)
    if n < min_observations:
        return None, 'insufficient_observations'
    centers = np.array([o['center'] for o in observations])
    rotations = np.array([o['map_to_camera'] for o in observations])
    pixels = np.array([o['pixel'] for o in observations])
    intrinsics = np.array([o['intrinsics'] for o in observations])
    rays = np.array([o['ray'] for o in observations])
    angles = np.degrees(np.arccos(np.clip(np.abs(rays @ rays.T), 0, 1)))
    pairs = sorted(combinations(range(n), 2), key=lambda ij: (-angles[ij], ij))
    if not pairs or angles[pairs[0]] < min_parallax_deg:
        return None, 'low_parallax'
    seeds = [o['seed'] for o in observations if o['seed'] is not None]
    if not seeds:
        return None, 'no_metric_landmark'
    seeds.append(np.median(seeds, axis=0))
    # Deterministic, bounded hypotheses; prioritize well-separated viewing rays.
    for i, j in pairs[:32]:
        if angles[i, j] < min_parallax_deg:
            break
        point = _intersect(centers[[i, j]], rays[[i, j]])
        if point is not None:
            seeds.append(point)

    def project(point):
        camera = np.einsum('nij,nj->ni', rotations, point - centers)
        z = np.maximum(camera[:, 2], 1e-8)
        uv = camera[:, :2] / z[:, None] * intrinsics[:, :2] + intrinsics[:, 2:]
        return uv - pixels, camera

    def consensus(point):
        residual, camera = project(point)
        errors = np.linalg.norm(residual, axis=1)
        mask = (camera[:, 2] > 1e-6) & np.isfinite(errors) & (errors <= threshold)
        return mask, errors

    best, mask, rank = None, None, None
    for point in seeds:
        if not np.isfinite(point).all():
            continue
        support, errors = consensus(point)
        score = (int(support.sum()), -float(np.median(errors[support])) if support.any() else -math.inf)
        if rank is None or score > rank:
            best, mask, rank = point, support, score
    required = max(min_observations, math.ceil(.6 * n))
    if best is None or mask.sum() < required:
        return None, 'inconsistent_reprojection'

    # Optimize a single point. Camera poses and calibration remain fixed.
    def jacobian(point, support):
        _, camera = project(point)
        z = np.maximum(camera[:, 2], 1e-8)
        j = np.zeros((n, 2, 3))
        j[:, 0, 0] = intrinsics[:, 0] / z
        j[:, 1, 1] = intrinsics[:, 1] / z
        j[:, 0, 2] = -intrinsics[:, 0] * camera[:, 0] / z**2
        j[:, 1, 2] = -intrinsics[:, 1] * camera[:, 1] / z**2
        return np.einsum('nij,njk->nik', j, rotations)[support].reshape(-1, 3)

    for _ in range(2):
        old_mask = mask.copy()
        result = least_squares(lambda p: project(p)[0][old_mask].ravel(), best,
                               jac=lambda p: jacobian(p, old_mask), loss='huber',
                               f_scale=threshold / 2, max_nfev=30)
        if not np.isfinite(result.x).all():
            return None, 'nonfinite_refinement'
        best = result.x
        mask, errors = consensus(best)
        if mask.sum() < required:
            return None, 'inconsistent_reprojection'
        if np.array_equal(mask, old_mask):
            break
    parallax = float(angles[np.ix_(mask, mask)].max())
    if parallax < min_parallax_deg:
        return None, 'low_parallax'
    return dict(position=best, observations=n, inliers=int(mask.sum()),
                reprojection_px=float(np.sqrt(np.mean(errors[mask]**2))),
                parallax_deg=parallax), None


def build_landmark_map(features, pose_by_key, *, min_observations=2,
                       min_parallax_deg=1.0, reprojection_px=3.0):
    """Return one well-supported point per composite landmark identity and an audit.

    Multiple archives of the same keyframe count as one observing view. Missing
    metric points may contribute a bearing to a track that has another metric
    observation, but never introduce a landmark absent from VIO's metric map.
    """
    if (min_observations < 2 or not math.isfinite(min_parallax_deg)
            or not 0 < min_parallax_deg < 90 or not math.isfinite(reprojection_px)
            or reprojection_px <= 0):
        raise ValueError('need at least two views, 0 < parallax < 90 degrees, and positive pixel threshold')
    groups = defaultdict(dict)
    counts = Counter()
    for feature in features:
        counts['feature_packets'] += 1
        key = identity(feature['key'])
        pose = pose_by_key.get(key)
        if pose is None:
            counts['missing_graph_pose'] += 1
            continue
        ids, pixels, points = (feature.get(k, []) for k in ('track_ids', 'pixels', 'points_camera'))
        if not len(ids) == len(pixels) == len(points):
            raise ValueError(f'misaligned feature arrays for {key}')
        camera = feature['camera']
        rb, tb = _transform(pose['body_to_map'])
        rc, tc = _transform(camera['camera_to_body'])
        rotation, center = rb @ rc, rb @ tc + tb
        intrinsics = np.asarray(camera['intrinsics'], dtype=float)
        if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0):
            raise ValueError(f'invalid pinhole intrinsics for {key}')
        for track, pixel, point in zip(ids, pixels, points):
            counts['archived_observations'] += 1
            uv = np.asarray(pixel, dtype=float)
            if (uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv < 0)
                    or uv[0] >= camera['width'] or uv[1] >= camera['height']):
                counts['invalid_pixels'] += 1
                continue
            seed = None
            if point is not None:
                p = np.asarray(point, dtype=float)
                if p.shape == (3,) and np.isfinite(p).all() and p[2] > 0:
                    seed = rotation @ p + center
                    counts['archived_metric_estimates'] += 1
                else:
                    counts['invalid_metric_estimates'] += 1
            ray = rotation @ np.r_[(uv - intrinsics[2:]) / intrinsics[:2], 1.0]
            ray /= np.linalg.norm(ray)
            landmark = (identity(pose['component']), key[0], key[1], int(track))
            observation = dict(center=center, map_to_camera=rotation.T,
                               intrinsics=intrinsics, pixel=uv, ray=ray, seed=seed)
            existing = groups[landmark].get(key)
            if existing is not None:
                counts['duplicate_observations'] += 1
                # Prefer a packet containing a metric estimate, without adding a vote.
                if existing['seed'] is not None or seed is None:
                    continue
            groups[landmark][key] = observation
    landmarks = []
    rejected = Counter()
    for (component, robot, session, track), by_view in sorted(groups.items()):
        observations = [by_view[k] for k in sorted(by_view)]
        if not any(o['seed'] is not None for o in observations):
            counts['nonmetric_tracks_omitted'] += 1
            continue
        point, reason = _refine(observations, min_observations, min_parallax_deg, reprojection_px)
        if point is None:
            rejected[reason] += 1
            continue
        landmarks.append(dict(component=component, robot=robot, session=session,
                              track_id=track, **point))
    audit = dict(counts, observed_track_groups=len(groups),
                 landmark_groups=len(groups)-counts['nonmetric_tracks_omitted'],
                 displayed_landmarks=len(landmarks),
                 rejected=dict(sorted(rejected.items())),
                 settings=dict(min_observations=min_observations,
                               min_parallax_deg=min_parallax_deg,
                               reprojection_px=reprojection_px,
                               minimum_consensus_ratio=.6),
                 method='fixed-pose robust multi-view landmark refinement for visualization',
                 backend_landmark_optimization=False)
    return landmarks, audit
