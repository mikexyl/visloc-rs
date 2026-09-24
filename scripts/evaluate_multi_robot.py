#!/usr/bin/env python3
"""Metric evaluation only: one rigid global alignment per connected component."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from evaluate_euroc_trajectory import load_ground_truth, associate, umeyama, evaluate

def transform(value):
    return Rotation.from_quat(value['rotation_xyzw']), np.array(value['translation'])

def key(value):
    return value['robot'], value['session'], value['id']

def load_dense(path, offset):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([int(r['timestamp_ns']) for r in rows], dtype=np.int64)
    points = np.array([[float(r[c]) for c in ('tx', 'ty', 'tz')] for r in rows])
    rotations = Rotation.from_quat([[float(r[c]) for c in ('qx', 'qy', 'qz', 'qw')] for r in rows])
    return times, points, rotations

def analyze(mission_path, graph_path=None, output_dir=None):
    root = mission_path.parent
    mission = json.loads(mission_path.read_text())
    graph = json.loads((graph_path or root / 'backend/graph_snapshot.json').read_text())
    output = output_dir or root
    output.mkdir(parents=True, exist_ok=True)
    results = {'alignment': 'metric SE(3), no scale fitting', 'components': graph['components'], 'robots': {}, 'joint_by_component': {}}
    component_pairs = {}
    for robot in mission['robots']:
        name, offset = robot['robot'], robot['offset_ns']
        poses = sorted((p for p in graph['poses'] if p['key']['robot'] == name), key=lambda p: p['timestamp_ns'])
        times, points, rotations = load_dense(root / 'robots' / name / 'trajectory.csv', offset)
        if not poses:
            raise ValueError(f'No optimized keyframes for {name}')
        components = {key(p['component']) for p in poses}
        if len(components) != 1:
            raise ValueError(f'{name} has multiple disconnected trajectory fragments; evaluate fragments separately')
        component = next(iter(components))
        ts = np.array([p['timestamp_ns'] for p in poses], dtype=np.int64)
        correction_r = Rotation.from_quat([p['map_from_odom']['rotation_xyzw'] for p in poses])
        correction_t = np.array([p['map_from_odom']['translation'] for p in poses])
        relative = (times - ts[0]) / 1e9
        knots = (ts - ts[0]) / 1e9
        cr = Slerp(knots, correction_r)(np.clip(relative, knots[0], knots[-1])) if len(poses) > 1 else Rotation.from_quat(np.repeat(correction_r.as_quat(), len(times), axis=0))
        ct = np.column_stack([np.interp(relative, knots, correction_t[:, d]) for d in range(3)])
        corrected = cr.apply(points) + ct
        corrected_rotations = cr * rotations
        original_times = times - offset
        truth = load_ground_truth(Path(robot['truth']))
        raw_poses = list(zip(map(int, original_times), points, rotations.as_matrix()))
        corrected_poses = list(zip(map(int, original_times), corrected, corrected_rotations.as_matrix()))
        results['robots'][name] = {'raw': evaluate(truth, raw_poses, 10_000_000), 'corrected': evaluate(truth, corrected_poses, 10_000_000), 'component': component}
        pairs = associate(truth, corrected_poses, 10_000_000)
        component_pairs.setdefault(component, []).extend(pairs)
        out = (output / f'{name}_corrected.tum') if output_dir else root / 'robots' / name / 'trajectory_corrected.tum'
        with out.open('w') as stream:
            for ns, p, q in zip(times, corrected, corrected_rotations.as_quat()):
                stream.write(f'{ns/1e9:.9f} ' + ' '.join(f'{v:.12f}' for v in [*p, *q]) + '\n')
    for component, pairs in component_pairs.items():
        truth = np.array([p[0][1] for p in pairs]); estimated = np.array([p[1][1] for p in pairs])
        _, rotation, translation = umeyama(estimated, truth, False)
        error = np.linalg.norm(estimated @ rotation.T + translation - truth, axis=1)
        results['joint_by_component']['/'.join(map(str, component))] = {
            'associated_poses': len(pairs), 'ate_rmse_m': float(np.sqrt(np.mean(error**2))),
            'rotation': rotation.tolist(), 'translation': translation.tolist()}
    (output / 'evaluation.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({'components': results['components'], 'joint_by_component': results['joint_by_component']}, indent=2))
    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mission', type=Path)
    p.add_argument('--graph', type=Path); p.add_argument('--output', type=Path)
    a=p.parse_args();analyze(a.mission,a.graph,a.output)
