#!/usr/bin/env python3
"""Prepare raw-GNSS South-ece comparisons from a verified cached VIO run."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(baseline, recording, output, coupling="keyframe_preintegration", include_window_only=False):
    baseline, recording, output = [p.resolve() for p in (baseline, recording, output)]
    mission = json.loads((baseline / 'mission.json').read_text())
    robot = mission['robots'][0]
    assert robot['camera_mode'] == 'stereo'
    assert Path(robot['bag']).resolve() == recording
    original = json.loads(Path(robot['config']).read_text())
    identity = {'recording': str(recording), 'files': []}
    # Hash all sensor bags and the raw receiver log; a renamed recording is not an identity.
    paths = sorted(recording.glob('*.mcap')) + [recording / name for name in (
        'metadata.yaml', 'gps-raw.ubx', 'gps-raw.timestamps.jsonl', 'clock-check.json')]
    for path in paths:
        identity['files'].append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    identity['calibration_sha256'] = sha256(Path(original['calibration']))
    identity['vio_config_sha256'] = sha256(Path(original['vio_config']))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'recording_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    modes = ['baseline', 'doppler_only', 'pseudorange_doppler']
    if include_window_only:
        if coupling != 'keyframe_preintegration':
            raise ValueError('window_only requires keyframe_preintegration coupling')
        modes.append('window_only')
    for mode in modes:
        for loops in ((False,) if mode == 'window_only' else (False, True)):
            root = output / (mode + ('_loops' if loops else '_no_loops'))
            root.mkdir(parents=True, exist_ok=True)
            config = dict(original, output=str(root / 'robots' / robot['robot']),
                          loop_enabled=loops, gnss={'enabled': mode != 'baseline',
                          'mode': mode if mode != 'baseline' else 'pseudorange_doppler',
                          'coupling': coupling},
                          gnss_typed_input=False, gnss_replay_shift_ns=robot['offset_ns'])
            robot_config = root / (robot['robot'] + '.json')
            robot_config.write_text(json.dumps(config, indent=2) + '\n')
            backend = json.loads(Path(mission['backend_config']).read_text())
            backend['output'] = str(root / 'backend')
            backend_path = root / 'backend.json'
            backend_path.write_text(json.dumps(backend, indent=2) + '\n')
            replay = dict(robot, config=str(robot_config), gnss_raw=str(recording / 'gps-raw.ubx'),
                          gnss_receipts=str(recording / 'gps-raw.timestamps.jsonl'))
            result = dict(mission, rate=0.25, backend_config=str(backend_path), robots=[replay])
            (root / 'mission.json').write_text(json.dumps(result, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('recording', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--coupling', choices=('keyframe_preintegration', 'frame_window'),
                        default='keyframe_preintegration')
    parser.add_argument('--include-window-only', action='store_true',
                        help='Also prepare a no-loop keyframe-window ablation')
    args = parser.parse_args()
    print(prepare(args.baseline, args.recording, args.output, args.coupling, args.include_window_only))
