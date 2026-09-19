#!/usr/bin/env python3
"""Audit/refine GRACO stereo rotation using images only, with held-out checks.

Preserves camera intrinsics, distortion, both optical centers, and the left
camera/IMU transform. Never reads ground truth or changes the source YAMLs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from run_graco_vio import Bag, DEFAULT_CALIBRATION


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def epipolar_error(correction, rays, relative_transform, focal_length):
    delta = Rotation.from_rotvec(correction).as_matrix()
    # R'_right = R_right * delta^T about the fixed optical center, hence
    # both terms of T_right_left are premultiplied by delta.
    essential = skew(delta @ relative_transform[:3, 3]) @ delta @ relative_transform[:3, :3]
    lines = rays[0] @ essential.T
    return np.sum(lines * rays[1], axis=1) / np.linalg.norm(lines[:, :2], axis=1) * focal_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, default=Path('/data/graco/aerial-08-25m_ros2'))
    parser.add_argument('--calibration-dir', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--train-frames', type=int, nargs='+', default=[200, 400, 800, 1600])
    parser.add_argument('--check-frames', type=int, nargs='+', default=[3200, 5000])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if set(args.train_frames) & set(args.check_frames):
        parser.error('training and validation frames must be disjoint')
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    source = args.calibration_dir
    stereo = yaml.safe_load((source / 'stereo.yaml').read_text())
    extrinsics = yaml.safe_load((source / 'stereo-imu.yaml').read_text())
    transforms = [np.array(extrinsics[f'T_Imu_cam{i}']['data']).reshape(4, 4) for i in range(2)]
    relative = np.linalg.inv(transforms[1]) @ transforms[0]
    published = np.array(stereo['T_cam1_cam0']['data']).reshape(4, 4)
    if not np.allclose(relative, published, atol=1e-6):
        raise ValueError('Stereo and camera/IMU extrinsics disagree')
    intrinsics, distortion = [], []
    for i in range(2):
        fx, fy, cx, cy = stereo[f'cam{i}']['intrinsics']
        intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]))
        distortion.append(np.array(stereo[f'cam{i}']['distortion_coeffs']))
    focal = np.mean([intrinsics[1][0, 0], intrinsics[1][1, 1]])
    bag = Bag(args.bag)
    data, stamps = {}, {}
    try:
        frames = bag.stereo_index()
        detector = cv2.SIFT_create(nfeatures=2500)
        for index in args.train_frames + args.check_frames:
            if not 0 <= index < len(frames):
                raise ValueError(f'Invalid frame index {index}')
            stamp, left, right = frames[index]
            features = [detector.detectAndCompute(bag.image(row, stamp, stereo[f'cam{i}']['resolution']), None)
                        for i, row in enumerate((left, right))]
            matches = [m for m, n in cv2.BFMatcher().knnMatch(features[0][1], features[1][1], k=2)
                       if m.distance < .65 * n.distance]
            if len(matches) < 100:
                raise ValueError(f'Frame {index}: too few stereo matches ({len(matches)})')
            pixels = [np.array([features[i][0][m.queryIdx if i == 0 else m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                      for i in range(2)]
            # Remove descriptor outliers independently of the supplied calibration.
            _, mask = cv2.findFundamentalMat(*pixels, cv2.USAC_MAGSAC, 1., .999)
            if mask is None or mask.sum() < 100:
                raise ValueError(f'Frame {index}: too few robust stereo inliers')
            rays = [cv2.undistortPoints(p, k, d).reshape(-1, 2)
                    for p, k, d in zip(pixels, intrinsics, distortion)]
            data[index] = [np.c_[p, np.ones(len(p))][mask.ravel().astype(bool)] for p in rays]
            stamps[index] = stamp
    finally:
        bag.connection.close()
    training = [np.concatenate([data[i][j] for i in args.train_frames]) for j in range(2)]
    fit = least_squares(lambda v: epipolar_error(v, training, relative, focal),
                        np.zeros(3), loss='soft_l1', f_scale=1.)
    if not fit.success or np.linalg.norm(fit.x) > np.deg2rad(2):
        raise ValueError('Stereo rotation fit failed or exceeds 2 degrees')
    checks = []
    for index, rays in data.items():
        before = np.percentile(np.abs(epipolar_error(np.zeros(3), rays, relative, focal)), [50, 95])
        after = np.percentile(np.abs(epipolar_error(fit.x, rays, relative, focal)), [50, 95])
        checks.append({'frame_index': index, 'timestamp_ns': stamps[index],
                       'split': 'train' if index in args.train_frames else 'held_out',
                       'inliers': len(rays[0]), 'before_median_p95_px': before.tolist(),
                       'after_median_p95_px': after.tolist()})
        print(f'frame={index} median/p95 px: {before.round(3)} -> {after.round(3)}', flush=True)
        if index in args.check_frames and (after[0] >= before[0] or after[1] > 1.5):
            raise ValueError(f'Refinement failed held-out validation at frame {index}')
    names = ('stereo.yaml', 'stereo-imu.yaml', 'imu.yaml')
    report = {
        'schema': 'visloc.graco.stereo_rotation_refinement.v1',
        'bag': str(args.bag.resolve()), 'ground_truth_used': False,
        'source_sha256': {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in names},
        'right_rotation_correction_rotvec': fit.x.tolist(),
        'rotation_correction_degrees': float(np.rad2deg(np.linalg.norm(fit.x))),
        'preserved': ['intrinsics', 'distortion', 'camera centers', 'baseline', 'left camera/IMU transform'],
        'checks': checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
