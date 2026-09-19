#!/usr/bin/env python3
"""Export Basalt EuRoC VIO results as synchronized Rerun browser playback.

Install scripts/requirements-rerun.txt in a virtual environment, then run
  python scripts/view_euroc_rerun.py --euroc-dir /data/euroc/MH_01_easy \
    --run-dir target/euroc_vio/MH_01_easy
  rerun --web-viewer target/euroc_vio/MH_01_easy/playback.rrd
"""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb

from evaluate_euroc_trajectory import (
    associate, load_estimate, load_ground_truth, quaternion_matrix, umeyama,
)

REPO = Path(__file__).resolve().parents[1]


def manifest(sequence, camera):
    folder = sequence / 'mav0' / camera
    with (folder / 'data.csv').open() as stream:
        return {int(row[0]): folder / 'data' / row[1].strip()
                for row in csv.reader(stream)
                if row and not row[0].lstrip().startswith('#')}


def encoded_image(path, width):
    with Image.open(path) as image:
        if width and image.width > width:
            image.thumbnail((width, round(image.height * width / image.width)))
        output = io.BytesIO()
        image.convert('L').save(output, format='JPEG', quality=85)
    return rr.EncodedImage(contents=output.getvalue(), media_type='image/jpeg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--euroc-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=REPO / 'configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json')
    parser.add_argument('--save', type=Path, help='Default: RUN_DIR/playback.rrd')
    parser.add_argument('--image-width', type=int, default=480, help='Preview width; 0 preserves full resolution')
    parser.add_argument('--max-frames', type=int, help='Limit export for a smoke test')
    args = parser.parse_args()
    if args.image_width < 0 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('image width must be nonnegative and max frames must be positive')
    with (args.run_dir / 'trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))[:args.max_frames]
    if not rows:
        parser.error('trajectory is empty')
    stamps = [int(row['timestamp_ns']) for row in rows]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        parser.error('trajectory timestamps must be strictly increasing')
    positions = np.array([[float(row[k]) for k in ('tx', 'ty', 'tz')] for row in rows])
    rotations = [quaternion_matrix(*(float(row[k]) for k in ('qx', 'qy', 'qz', 'qw'))) for row in rows]
    cameras = [manifest(args.euroc_dir, f'cam{i}') for i in range(2)]
    # Basalt's CSV preserves integer timestamps; TUM may round at sub-microsecond precision.
    for stamp in stamps:
        for camera in cameras:
            if stamp not in camera or not camera[stamp].is_file():
                parser.error(f'missing stereo image at timestamp {stamp}')
    calibration = json.loads(args.calibration.read_text())['value0']
    ground_truth_path = args.euroc_dir / 'mav0/state_groundtruth_estimate0/data.csv'
    gt, pairs = [], []
    align_rotation, align_translation = np.eye(3), np.zeros(3)
    if ground_truth_path.is_file():
        gt = load_ground_truth(ground_truth_path)
        # Align on the full run, including when exporting a short preview.
        pairs = associate(gt, load_estimate(args.run_dir / 'trajectory.tum', 's'), 10_000_000)
        if len(pairs) < 3:
            parser.error('fewer than three ground-truth matches for rigid alignment')
        _, align_rotation, align_translation = umeyama(
            np.array([p[1][1] for p in pairs]), np.array([p[0][1] for p in pairs]), False)
    positions = positions @ align_rotation.T + align_translation
    rotations = [align_rotation @ rotation for rotation in rotations]
    matched_gt = {est[0]: truth for truth, est, _ in associate(
        gt, [(s, p, r) for s, p, r in zip(stamps, positions, rotations)], 10_000_000)} if gt else {}
    output = args.save or args.run_dir / 'playback.rrd'
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f'visloc_vio_{args.euroc_dir.name}')
    rr.save(output)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin='world', name='VIO and ground truth'),
            rrb.Vertical(
                rrb.Spatial2DView(origin='stereo/cam0', name='Left camera'),
                rrb.Spatial2DView(origin='stereo/cam1', name='Right camera'),
                rrb.TimeSeriesView(origin='metrics/error', name='Position error (m)'),
                rrb.TimeSeriesView(origin='metrics/tracks', name='Visual observations'),
            ), column_shares=[2, 1]),
        rrb.TimePanel(state='expanded')))
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log('world/vio_trajectory', rr.LineStrips3D([positions], colors=[40, 160, 255]), static=True)
    if gt:
        rr.log('world/ground_truth_trajectory', rr.LineStrips3D(
            [[pose[1] for pose in gt]], colors=[160, 160, 160]), static=True)
    rr.log('notes', rr.TextDocument(
        'Basalt stereo-inertial VIO playback. Blue: estimate; gray: ground truth. '
        'Estimate is rigidly aligned to ground truth for display (no scale correction). '
        'Images are raw distorted JPEG previews. Camera frustums use approximate pinhole '
        'intrinsics; images are displayed separately. No landmarks were saved by this VIO run.'
        if gt else 'VIO playback in the estimator frame; no ground truth available.'), static=True)
    for i, transform in enumerate(calibration['T_imu_cam']):
        camera_path = f'world/body/cam{i}'
        rr.log(camera_path, rr.Transform3D(
            translation=[transform[k] for k in ('px', 'py', 'pz')],
            quaternion=rr.Quaternion(xyzw=[transform[k] for k in ('qx', 'qy', 'qz', 'qw')])), static=True)
        intrinsics = calibration['intrinsics'][i]['intrinsics']
        rr.log(camera_path + '/frustum', rr.Pinhole(
            focal_length=[intrinsics['fx'], intrinsics['fy']],
            principal_point=[intrinsics['cx'], intrinsics['cy']],
            resolution=calibration['resolution'][i], image_plane_distance=0.2), static=True)
    for index, (row, stamp, position, rotation) in enumerate(zip(rows, stamps, positions, rotations)):
        rr.set_time('elapsed', duration=(stamp - stamps[0]) / 1e9)
        rr.log('world/body', rr.Transform3D(translation=position, mat3x3=rotation))
        rr.log('world/current_vio', rr.Points3D([position], colors=[40, 160, 255], radii=0.06))
        for i in range(2):
            rr.log(f'stereo/cam{i}', encoded_image(cameras[i][stamp], args.image_width))
            rr.log(f'metrics/tracks/cam{i}_observations', rr.Scalars(float(row[f'cam{i}_observations'])))
        if stamp in matched_gt:
            truth = matched_gt[stamp]
            rr.log('world/current_ground_truth', rr.Points3D([truth[1]], colors=[160, 160, 160], radii=0.06))
            rr.log('metrics/error/position_error_m', rr.Scalars(float(np.linalg.norm(position - truth[1]))))
        else:
            rr.log('world/current_ground_truth', rr.Clear(recursive=True))
            rr.log('metrics/error/position_error_m', rr.Clear(recursive=True))
        if (index + 1) % 250 == 0:
            print(f'Exported {index + 1}/{len(rows)} frames', flush=True)
    rr.disconnect()
    print(f'Saved {len(rows)} stereo frames to {output}', flush=True)


if __name__ == '__main__':
    main()
