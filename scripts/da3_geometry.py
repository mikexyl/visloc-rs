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
    the union of all accepted metric DA3 views, with depth agreement.

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


def pose_depth_scale(input_world_to_camera, predicted_world_to_camera):
    """Restore metric depth using DA3's input-pose scale convention.

    Upstream aligns the supplied camera centers to the predicted centers with
    Umeyama, then divides depth by that scale. Only the scalar is applied:
    reconstruction keeps the supplied camera poses and calibrated intrinsics.
    This is necessary because the exported graph normalizes input translations.
    """
    def centers(extrinsics):
        value = np.asarray(extrinsics, dtype=np.float64)
        if value.shape not in ((5, 3, 4), (5, 4, 4)) or not np.isfinite(value).all():
            raise ValueError('Expected five finite world-to-camera matrices')
        return -np.einsum('nji,nj->ni', value[:, :3, :3], value[:, :3, 3])

    measured, predicted = centers(input_world_to_camera), centers(predicted_world_to_camera)
    a, b = measured-measured.mean(0), predicted-predicted.mean(0)
    variance = float(np.square(a).sum())
    report = dict(method='input_poses', accepted=False, landmark_alignment=False)
    if variance <= 1e-12:
        return None, dict(report, reason='degenerate_input_baseline')
    u, singular, vt = np.linalg.svd(b.T @ a)
    signs = np.ones(3)
    signs[-1] = 1. if np.linalg.det(u @ vt) >= 0 else -1.
    input_to_prediction = float(singular @ signs / variance)
    if not np.isfinite(input_to_prediction) or input_to_prediction <= 1e-12:
        return None, dict(report, reason='degenerate_predicted_baseline')
    scale = 1. / input_to_prediction
    rotation = (u * signs) @ vt
    residual = (b-input_to_prediction*(a @ rotation.T))*scale
    return scale, dict(report, accepted=True, reason='input_pose_scale', scale=scale,
                       pose_fit_rmse_m=float(np.sqrt(np.square(residual).sum(1).mean())))


def filter_depth_confidence(depths, confidence, *, min_confidence=1., percentile=0., max_depth_m=200.):
    """Apply one confidence threshold across the entire five-keyframe window.

    The percentile is computed over finite, in-range depths BEFORE applying
    the absolute floor or reprojection mask. The floor prevents a uniformly
    low-confidence window from passing merely through relative ranking.
    """
    depth = np.asarray(depths, dtype=np.float32)
    scores = np.asarray(confidence)
    if depth.ndim != 3 or depth.shape[0] != 5 or scores.shape != depth.shape:
        raise ValueError('Expected matching five-view depth and confidence maps')
    if (not np.isfinite(min_confidence) or min_confidence <= 0
            or not np.isfinite(percentile) or not 0 <= percentile < 100
            or not np.isfinite(max_depth_m) or max_depth_m <= .1):
        raise ValueError('Invalid depth confidence thresholds')
    valid = (np.isfinite(depth) & (depth > .1) & (depth < max_depth_m)
             & np.isfinite(scores))
    before = int(valid.sum())
    quantile = float(np.percentile(scores[valid], percentile)) if before and percentile > 0 else None
    threshold = max(min_confidence, quantile) if quantile is not None else min_confidence
    keep = valid & (scores >= threshold)
    after = int(keep.sum())
    report = dict(min_confidence=min_confidence, percentile=percentile, threshold=float(threshold),
                  percentile_threshold=quantile, input_pixels=before, retained_pixels=after,
                  rejected_pixels=before-after, retained_fraction=after/before if before else 0.)
    return np.where(keep, depth, np.nan).astype(np.float32), report


def filter_depth_reprojection(depths, intrinsics, camera_to_world, *, max_error_px=1.5,
                              max_relative_depth=.05, min_consistent_views=2):
    """Mask depth without modifying surviving values, camera poses or scale.

    For every ordered pair of views, project reference depth into the other
    camera, sample that camera's depth, and project it back. Both the round-trip
    pixel error and relative camera-z error must pass. At least two OTHER views
    support a pixel by default. All votes use the original maps, so filtering
    order cannot change the result. Missing, occluded and out-of-view samples
    provide no support; a pixel can still pass using other visible views.
    """
    depth = np.asarray(depths, dtype=np.float32)
    k = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    if depth.ndim != 3 or depth.shape[0] != 5 or min(depth.shape[1:]) < 2:
        raise ValueError('Expected five nonempty depth maps')
    if k.shape != (5, 3, 3) or poses.shape != (5, 4, 4):
        raise ValueError('Expected five calibrated intrinsics and camera poses')
    if not np.isfinite(k).all() or not np.isfinite(poses).all():
        raise ValueError('Camera geometry must be finite')
    if (not np.isfinite(max_error_px) or max_error_px <= 0
            or not np.isfinite(max_relative_depth) or not 0 < max_relative_depth <= 1
            or type(min_consistent_views) is not int or not 1 <= min_consistent_views <= 4):
        raise ValueError('Invalid depth reprojection thresholds')
    inverse_k = np.linalg.inv(k).astype(np.float32)
    w2c = np.linalg.inv(poses)
    k = k.astype(np.float32)
    _, h, w = depth.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    support = np.zeros(depth.shape, dtype=np.uint8)

    def apply(matrix, xyz):
        # Elementwise arithmetic avoids starting a BLAS thread pool per image.
        return tuple(sum(matrix[i, j]*xyz[j] for j in range(3)) for i in range(3))

    def transform(matrix, xyz):
        return tuple(value + matrix[i, 3] for i, value in enumerate(apply(matrix, xyz)))

    def project(matrix, xyz):
        u, v, z = apply(matrix, xyz)
        return u/np.maximum(z, 1e-8), v/np.maximum(z, 1e-8)

    def sample(image, u, v):
        inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)
        u = np.clip(np.nan_to_num(u, nan=0., posinf=0., neginf=0.), 0, w-1)
        v = np.clip(np.nan_to_num(v, nan=0., posinf=0., neginf=0.), 0, h-1)
        x0, y0 = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
        x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
        a, b, c, d = image[y0,x0], image[y0,x1], image[y1,x0], image[y1,x1]
        low, high = np.minimum(np.minimum(a,b),np.minimum(c,d)), np.maximum(np.maximum(a,b),np.maximum(c,d))
        # Do not interpolate a fictitious surface across holes or depth edges.
        usable = inside & (low > 0) & np.isfinite(high) & (high-low <= max_relative_depth*low)
        dx, dy = u-x0, v-y0
        sampled = (1-dy)*((1-dx)*a+dx*b) + dy*((1-dx)*c+dx*d)
        return np.where(usable, sampled, np.nan).astype(np.float32)

    for ref in range(5):
        z = np.where(valid[ref], depth[ref], 0.)
        rays = apply(inverse_k[ref], (x,y,np.ones_like(x)))
        points = tuple(value*z/rays[2] for value in rays)
        for other in range(5):
            if other == ref:
                continue
            relative = (w2c[other] @ poses[ref]).astype(np.float32)
            target = transform(relative, points)
            u, v = project(k[other], target)
            measured = sample(depth[other], u, v)
            target_ray = apply(inverse_k[other], (u,v,np.ones_like(u)))
            target_points = tuple(value*measured/target_ray[2] for value in target_ray)
            back = transform((w2c[ref] @ poses[other]).astype(np.float32), target_points)
            back_u, back_v = project(k[ref], back)
            consistent = (valid[ref] & (target[2] > 0) & (back[2] > 0)
                          & np.isfinite(measured)
                          & ((back_u-x)**2 + (back_v-y)**2 <= max_error_px**2)
                          & (np.abs(back[2]-z) <= max_relative_depth*z))
            support[ref] += consistent.astype(np.uint8)
    keep = valid & (support >= min_consistent_views)
    before, after = int(valid.sum()), int(keep.sum())
    report = dict(enabled=True, max_error_px=max_error_px, max_relative_depth=max_relative_depth,
                  min_consistent_views=min_consistent_views, input_pixels=before, retained_pixels=after,
                  rejected_pixels=before-after, retained_fraction=after/before if before else 0.,
                  by_view=[dict(input_pixels=int(a.sum()), retained_pixels=int(b.sum()))
                           for a,b in zip(valid,keep)])
    return np.where(keep, depth, np.nan).astype(np.float32), support, report
