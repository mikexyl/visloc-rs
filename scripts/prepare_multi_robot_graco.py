#!/usr/bin/env python3
"""Prepare immutable sensor calibration and a ROS2 multi-robot replay mission."""
import argparse
import json
from pathlib import Path
import yaml
from run_graco_vio import Bag, prepare_calibration, load_truth, DEFAULT_CALIBRATION, REPO

BAGS = {5: '/data/graco/aerial-05-40m', 6: '/data/graco/aerial-06-20m_full_ros2',
        7: '/data/graco/aerial-07-25m_full_ros2', 8: '/data/graco/aerial-08-25m_ros2'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--robots', type=int, nargs='+', default=[5, 6, 7, 8], choices=list(BAGS))
    p.add_argument('--max-frames', type=int)
    p.add_argument('--imu-startup', choices=['stationary', 'legacy', 'stationary-gravity', 'stationary-motion'], default='stationary')
    p.add_argument('--rate', type=float, default=.25)
    p.add_argument('--fixed-last-frame', action='store_true')
    p.add_argument('--loop-config', type=Path, default=REPO / '.runtime/multi_robot_models/loop_config.json')
    args = p.parse_args()
    if args.rate <= 0 or len(set(args.robots)) != len(args.robots):
        p.error('rate must be positive and robots unique')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    peers = [f'a{i:02}' for i in args.robots]
    mission = {'rate': args.rate, 'epoch_ns': 1_000_000_000, 'robots': [], 'fixed_last_frame': args.fixed_last_frame}
    stereo = yaml.safe_load((DEFAULT_CALIBRATION / 'stereo.yaml').read_text())
    for i, robot in zip(args.robots, peers):
        calibration_dir = output / 'calibration' / robot
        calibration_dir.mkdir(parents=True)
        calibration, _, raw_sizes = prepare_calibration(DEFAULT_CALIBRATION, calibration_dir, 800,
            imu_noise_scale=.75, imu_bias_scale=.75)
        bag = Bag(Path(BAGS[i]))
        frames = bag.stereo_index()
        if args.max_frames:
            frames = frames[:args.max_frames]
        load_truth(bag, calibration_dir)
        bag.connection.close()
        config = {'robot': robot, 'peers': peers,
                  'calibration': str(calibration_dir / 'basalt_calibration.json'),
                  'vio_config': str(REPO / 'configs/graco/aerial_vio.json'),
                  'loop_config': str(args.loop_config.resolve()),
                  'output': str(output / 'robots' / robot), 'reliable_sensors': True,
                  'fixed_last_frame': args.fixed_last_frame,
                  'preprocess': {'raw_width': raw_sizes[0][0], 'raw_height': raw_sizes[0][1],
                    'intrinsics': [calibration['intrinsics'][0]['intrinsics'][v] for v in ('fx', 'fy', 'cx', 'cy')],
                    'distortion': stereo['cam0']['distortion_coeffs']}}
        if args.imu_startup != 'legacy':
            config['imu_startup'] = {'average_gravity': args.imu_startup != 'stationary',
                                     'wait_for_motion': args.imu_startup == 'stationary-motion'}
        else:
            config['imu_startup'] = None
        config_path = output / f'{robot}.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        mission['robots'].append({'robot': robot, 'bag': BAGS[i], 'config': str(config_path),
            'frames': len(frames), 'original_first_ns': frames[0][0],
            'original_last_ns': frames[-1][0], 'offset_ns': mission['epoch_ns'] - frames[0][0],
            'truth': str(calibration_dir / 'ground_truth.csv')})
    backend = {'peers': peers, 'output': str(output / 'backend')}
    (output / 'backend.json').write_text(json.dumps(backend, indent=2) + '\n')
    mission['backend_config'] = str(output / 'backend.json')
    (output / 'mission.json').write_text(json.dumps(mission, indent=2) + '\n')
    print(output / 'mission.json')

if __name__ == '__main__':
    main()
