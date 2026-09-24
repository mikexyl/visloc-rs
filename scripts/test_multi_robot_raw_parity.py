#!/usr/bin/env python3
"""Check exact raw-VIO parity for identically ordered ROS sensor replays."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('enabled', type=Path)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.control.open() as source:
        control = list(csv.DictReader(source))
    with args.enabled.open() as source:
        enabled = list(csv.DictReader(source))
    count = args.frames if args.frames is not None else len(control)
    if count <= 0 or min(len(control), len(enabled)) < count:
        parser.error('both completed trajectories must contain the requested positive frame count')
    if args.frames is None and len(control) != len(enabled):
        parser.error('trajectory lengths differ; select an explicit common prefix with --frames')
    maximum = 0.
    for index, (a, b) in enumerate(zip(control[:count], enabled[:count])):
        for field in ('frame_id', 'timestamp_ns'):
            assert int(a[field]) == int(b[field]), f'{field} differs at row {index}'
        for field in ('tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz'):
            x, y = float(a[field]), float(b[field])
            assert math.isfinite(x) and math.isfinite(y), f'nonfinite {field} at row {index}'
            maximum = max(maximum, abs(x-y))
            assert x.hex() == y.hex(), f'raw {field} differs at row {index}: {x} vs {y}'
    report = {'passed': True, 'frames': count, 'identical_pose_bits': True, 'max_absolute_pose_difference': maximum,
              'control': str(args.control.resolve()), 'enabled': str(args.enabled.resolve())}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
