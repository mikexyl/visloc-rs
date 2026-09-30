#!/usr/bin/env python3
"""Inspect nominal mounting datums composed with measured T_imu_cam in Rerun."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import rerun as rr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calibration', type=Path); parser.add_argument('--gap-mm', type=float, default=.6)
    parser.add_argument('--save', type=Path); args = parser.parse_args()
    values = json.loads(args.calibration.read_text())['value0']
    t = values['T_imu_cam'][0]
    rotation = Rotation.from_quat([t[k] for k in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
    translation = np.array([t[k] for k in ('px', 'py', 'pz')])
    optical_from_rear = np.array([[-1., 0, 0], [0, 0, -1.], [0, -1., 0]])
    rear_translation = -optical_from_rear @ np.array([.0475, -.02135, 0.])
    gps_rotation = optical_from_rear @ np.diag([-1., -1., 1.])
    gps_translation = rear_translation + optical_from_rear @ np.array([-.095, .087, .048])
    antenna = gps_translation + gps_rotation @ np.array([0., .01045, .0525+args.gap_mm*.001])
    rr.init('visloc raw GNSS sensor calibration', spawn=args.save is None)
    if args.save: rr.save(str(args.save))
    rr.log('imu', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log('imu/optical', rr.Transform3D(translation=translation, mat3x3=rotation), static=True)
    rr.log('imu/optical/rear_mount', rr.Transform3D(translation=rear_translation, mat3x3=optical_from_rear), static=True)
    rr.log('imu/optical/gps_casing', rr.Transform3D(translation=gps_translation, mat3x3=gps_rotation), static=True)
    rr.log('imu/optical/antenna', rr.Points3D([antenna], colors=[255, 220, 0], radii=.007, labels=['nominal CH7604A phase center']), static=True)
    rr.log('imu/lever_arm', rr.LineStrips3D([[np.zeros(3), translation+rotation@antenna]], colors=[255, 220, 0], radii=.002), static=True)
    for frame in ('imu', 'imu/optical', 'imu/optical/rear_mount', 'imu/optical/gps_casing'):
        rr.log(frame+'/axes', rr.Arrows3D(origins=np.zeros((3,3)), vectors=np.eye(3)*.06, colors=[[255,0,0],[0,255,0],[0,120,255]]), static=True)
    rr.log('calibration', rr.TextDocument(f'Nominal housing/antenna geometry; gasket {args.gap_mm} mm.\nOptical lever: {antenna.tolist()} m\nMeasured T_imu_cam composed IMU lever: {(translation+rotation@antenna).tolist()} m\nFixed 20 mm uncertainty per lever axis; not calibrated.'), static=True)


if __name__ == '__main__': main()
