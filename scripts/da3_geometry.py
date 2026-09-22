"""Geometry-only DA3 scheduling and scale fitting. No ground-truth inputs."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Keyframe:
    frame_id: int
    ordinal: int
    timestamp_ns: int
    gray: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    track_ids: np.ndarray
    pixels: np.ndarray
    points_world: np.ndarray

    @property
    def points_camera(self):
        return (self.points_world - self.camera_to_world[:3, 3]) @ self.camera_to_world[:3, :3]


def resize_intrinsics(k, source_size, target_size):
    scale = np.asarray(target_size) / np.asarray(source_size)
    result = k.copy()
    result[0] *= scale[0]
    result[1] *= scale[1]
    result[0, 2] += .5 * (scale[0] - 1)
    result[1, 2] += .5 * (scale[1] - 1)
    return result


def sample_bilinear(image, pixels):
    h, w = image.shape
    uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
    valid = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] <= h-1)
    safe = np.clip(np.nan_to_num(uv, nan=0., posinf=0., neginf=0.), [0, 0], [w-1, h-1])
    xy = np.floor(safe).astype(int)
    hi = np.minimum(xy + 1, [w-1, h-1])
    delta = safe - xy
    a, b = delta[:, 0], delta[:, 1]
    values = ((1-a)*(1-b)*image[xy[:, 1], xy[:, 0]] + a*(1-b)*image[xy[:, 1], hi[:, 0]]
              + (1-a)*b*image[hi[:, 1], xy[:, 0]] + a*b*image[hi[:, 1], hi[:, 0]])
    return np.where(valid, values, np.nan)


def balanced_landmarks(frame, grid=(8, 6)):
    """One measured landmark per occupied image cell, regardless of feature density."""
    if not len(frame.pixels):
        return np.zeros(0, dtype=int)
    h, w = frame.gray.shape
    coordinate = frame.pixels / [w, h] * grid
    cells = np.clip(np.floor(coordinate).astype(int), [0, 0], np.array(grid)-1)
    distance = np.square(coordinate - cells - .5).sum(1)
    bins = cells[:, 1] * grid[0] + cells[:, 0]
    ordered = np.argsort(distance, kind='stable')
    _, first = np.unique(bins[ordered], return_index=True)
    return ordered[first]


@dataclass
class CoveredView:
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    depth: np.ndarray

    def covers(self, points_world, depth_tolerance):
        camera = points_world @ self.world_to_camera[:3, :3].T + self.world_to_camera[:3, 3]
        z = camera[:, 2]
        projection = camera @ self.intrinsics.T
        pixels = projection[:, :2] / np.maximum(projection[:, 2:], 1e-10)
        observed = sample_bilinear(self.depth, pixels)
        # A point behind a previously seen occluder (or a new foreground
        # surface) is not established coverage merely because it is in the FOV.
        return ((z > 0) & np.isfinite(observed) & (observed > 0)
                & (np.abs(z-observed) <= depth_tolerance * np.maximum(z, observed)))


def coverage(window, history, depth_tolerance=.15):
    """Estimate sequence FOV novelty by projecting VIO surface samples into
    the union of all successfully aligned DA3 views, with depth agreement.

    Image cells with no reliable VIO geometry remain unknown; callers enforce
    a minimum spatial support before interpreting this sparse area estimate.
    Track IDs are deliberately irrelevant to the overlap calculation.
    """
    indices = [balanced_landmarks(frame) for frame in window]
    points = np.concatenate([frame.points_world[i] for frame, i in zip(window, indices)])
    covered = np.zeros(len(points), dtype=bool)
    for previous in reversed(history):
        remaining = np.flatnonzero(~covered)
        if not len(remaining):
            break
        covered[remaining] = previous.covers(points[remaining], depth_tolerance)
    by_view, cursor = [], 0
    for i in indices:
        n = len(i)
        by_view.append(dict(cells=n, covered=int(covered[cursor:cursor+n].sum())))
        cursor += n
    depths = np.concatenate([f.points_camera[i, 2] for f, i in zip(window, indices)])
    centers = np.array([f.camera_to_world[:3, 3] for f in window])
    baseline = float(np.linalg.norm(centers[:, None]-centers[None, :], axis=2).max())
    return dict(sampled_cells=len(points), covered_cells=int(covered.sum()),
                new_fraction=float((~covered).mean()) if len(points) else None,
                by_view=by_view, baseline_m=baseline,
                baseline_depth_ratio=baseline/float(np.median(depths)) if len(depths) else 0.)


def align_depth(window, depths, confidence, config):
    """Fit one positive multiplicative depth scale shared by the five views.

    Spatial balancing and inverse track-frequency weights avoid giving dense
    texture or repeatedly seen tracks disproportionate influence. Entire track
    IDs, across every view, are held out for validation, preventing duplicate
    observations from leaking between the fit and validation sets.
    """
    ratios, ids, view_ids = [], [], []
    for view, (frame, depth, conf) in enumerate(zip(window, depths, confidence)):
        index = balanced_landmarks(frame)
        h, w = frame.gray.shape
        target_h, target_w = depth.shape
        pixels = (frame.pixels[index] + .5) * [target_w/w, target_h/h] - .5
        predicted = sample_bilinear(depth, pixels)
        quality = sample_bilinear(conf, pixels)
        measured = frame.points_camera[index, 2]
        valid = (np.isfinite(predicted) & (predicted > 1e-6) & np.isfinite(quality)
                 & (quality >= config.min_confidence) & np.isfinite(measured) & (measured > 0))
        ratios.extend(np.log(measured[valid]/predicted[valid]))
        ids.extend(frame.track_ids[index][valid])
        view_ids.extend([view] * int(valid.sum()))
    ratios, ids, view_ids = np.asarray(ratios), np.asarray(ids, dtype=np.int64), np.asarray(view_ids, dtype=int)
    result = dict(samples=len(ids), unique_landmarks=len(np.unique(ids)), accepted=False)
    holdout = (ids * 2654435761 % 5) == 0
    train = ~holdout
    if len(np.unique(ids[train])) < config.min_fit_landmarks or len(np.unique(ids[holdout])) < config.min_holdout_landmarks:
        return None, dict(result, reason='insufficient_independent_landmarks')
    if len(np.unique(view_ids[train])) < 3 or len(np.unique(view_ids[holdout])) < 2:
        return None, dict(result, reason='insufficient_supported_views')
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    log_scale = float(np.median(ratios[train]))
    mad = float(np.median(np.abs(ratios[train]-log_scale)))
    cutoff = min(np.log1p(config.max_fit_relative_error), max(.06, 3*1.4826*mad))
    inliers = train & (np.abs(ratios-log_scale) <= cutoff)
    if int(inliers.sum()) < config.min_fit_samples or len(np.unique(ids[inliers])) < config.min_fit_landmarks:
        return None, dict(result, reason='insufficient_scale_inliers')
    if np.mean(inliers[train]) < config.min_fit_inlier_ratio:
        return None, dict(result, reason='inconsistent_scale')
    for _ in range(8):
        residual = ratios[inliers] - log_scale
        robust = np.minimum(1., .05/np.maximum(np.abs(residual), 1e-12))
        log_scale = float(np.average(ratios[inliers], weights=weights[inliers]*robust))
    scale = float(np.exp(log_scale))
    relative = np.abs(np.expm1(log_scale-ratios))
    train_errors, held_errors = relative[inliers], relative[holdout]
    result.update(scale=scale, fit_samples=int(inliers.sum()),
                  fit_landmarks=len(np.unique(ids[inliers])), holdout_samples=int(holdout.sum()),
                  holdout_landmarks=len(np.unique(ids[holdout])),
                  fit_inlier_ratio=float(inliers[train].mean()),
                  fit_relative_median=float(np.median(train_errors)),
                  holdout_relative_median=float(np.median(held_errors)),
                  holdout_relative_p90=float(np.quantile(held_errors, .9)),
                  fit_samples_by_view=np.bincount(view_ids[inliers], minlength=5).tolist())
    if (not np.isfinite(scale) or scale <= 0
            or result['holdout_relative_median'] > config.max_holdout_median
            or result['holdout_relative_p90'] > config.max_holdout_p90):
        return None, dict(result, reason='held_out_depth_disagreement')
    if sum(n >= 4 for n in result['fit_samples_by_view']) < 3:
        return None, dict(result, reason='insufficient_inlier_views')
    return scale, dict(result, accepted=True, reason='aligned')
