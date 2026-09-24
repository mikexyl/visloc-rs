#!/usr/bin/env python3
"""Launch the native rclrs robots/backend, replay ROS2 sensors, and drain PGO."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mission', type=Path)
    p.add_argument('--domain-id', type=int, default=217)
    p.add_argument('--profile-replay', action='store_true', help='Save Python replay timing to replay_profile.pstats')
    args = p.parse_args()
    mission = json.loads(args.mission.read_text())
    root = args.mission.resolve().parent
    env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain_id), ROS_LOCALHOST_ONLY='1')
    env.setdefault('ROS_LOG_DIR', str(root / 'ros_logs'))
    binaries = Path(os.environ.get('VISLOC_ROS_INSTALL', REPO / '.runtime/ros2_install')) / 'visloc_ros/lib/visloc_ros'
    logs, processes = [], []
    replay = None
    try:
        for binary, key, config, name in [('backend', 'VISLOC_BACKEND_CONFIG', mission['backend_config'], 'backend')] + [
                ('robot', 'VISLOC_ROBOT_CONFIG', r['config'], r['robot']) for r in mission['robots']]:
            log = (root / f'{name}.log').open('w'); logs.append(log)
            processes.append(subprocess.Popen([str(binaries / binary)], env=dict(env, **{key: config}), stdout=log, stderr=subprocess.STDOUT))
        replay_command = [sys.executable]
        if args.profile_replay:
            replay_command += ['-m', 'cProfile', '-o', str(root / 'replay_profile.pstats')]
        replay_command += [str(REPO / 'scripts/replay_multi_robot_graco.py'), str(args.mission.resolve())]
        replay = subprocess.Popen(replay_command, env=env)
        while replay.poll() is None:
            failed = [p.returncode for p in processes if p.poll() is not None]
            if failed:
                replay.terminate(); replay.wait()
                raise RuntimeError(f'Native node exited: {failed}; inspect {root}/*.log')
            time.sleep(.2)
        if replay.returncode:
            raise RuntimeError(f'Replay failed with {replay.returncode}; inspect {root}/*.log')
        deadline = time.monotonic() + 180
        while not (root / 'backend/finished.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Final centralized PGO did not finish')
            time.sleep(.2)
        print(f'Completed multi-robot replay: {root}', flush=True)
    finally:
        if replay and replay.poll() is None:
            replay.terminate()
            try: replay.wait(timeout=10)
            except subprocess.TimeoutExpired: replay.kill(); replay.wait()
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for log in logs:
            log.close()

if __name__ == '__main__':
    main()
