#!/usr/bin/env python3
"""Rigid evo translation APE against the correlated, partially covered PPK reference."""
import argparse
import csv
from decimal import Decimal
import json
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

RUN_NAMES = {mode + suffix for mode in ('baseline', 'window_only', 'doppler_only', 'pseudorange_doppler')
             for suffix in ('_no_loops', '_loops')} | {'legacy_pseudorange_doppler'}


def gnss_statistics(files, first_ns):
    diagnostics = []
    with (files / 'gnss_diagnostics.jsonl').open() as stream:
        for line in stream:
            diagnostics.append(json.loads(line))
    if not diagnostics:
        return None
    active = [d for d in diagnostics if d['diagnostics']['status'] == 'active']
    result = dict(diagnostics[-1], initialization_latency_s=None)
    if active:
        result['initialization_latency_s'] = (active[0]['timestamp_ns'] - first_ns) * 1e-9
    latest = {}; decisions = Counter(); epoch_times = {}
    with (files / 'gnss_records.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            reason = record.get('reason')
            if reason == 'optimized_residual':
                latest[(record['epoch_id'], json.dumps(record['satellite'], sort_keys=True))] = record
                if 'sensor_time_ns' in record:
                    epoch_times[record['epoch_id']] = record['sensor_time_ns']
            elif reason:
                decisions[reason] += 1
    result['decision_records_by_reason'] = dict(decisions)
    result['used_epochs'] = len({key[0] for key in latest})
    result['unique_satellite_observations'] = len(latest)
    def distribution(values):
        return None if not values else dict(count=len(values), rms=float(np.sqrt(np.mean(np.square(values)))),
            median_abs=float(np.median(np.abs(values))), p95_abs=float(np.percentile(np.abs(values), 95)),
            max_abs=float(np.max(np.abs(values))))
    result['normalized_residuals'] = {
        channel: distribution([r[channel+'_sigma'] for r in latest.values() if r['use_'+channel]])
        for channel in ('pseudorange', 'doppler')}
    result['factor_age_ms'] = distribution([(r['solve_timestamp_ns']-r['sensor_time_ns'])*1e-6
        for r in latest.values() if 'solve_timestamp_ns' in r])
    if latest:
        clocks = np.array([r['clock']['bias_m'] + [r['clock']['drift_m_s']] for r in latest.values()])
        result['clock_min_max'] = {name: [float(clocks[:, i].min()), float(clocks[:, i].max())]
            for i, name in enumerate(('gps_bias_m', 'galileo_bias_m', 'common_drift_m_s'))}
    used_times = sorted(epoch_times.values())
    result['used_epoch_gaps_over_1s'] = ([dict(start_elapsed_s=(a-first_ns)*1e-9, duration_s=(b-a)*1e-9)
        for a, b in zip(used_times, used_times[1:]) if b-a > 1_000_000_000] if used_times else None)
    return result


def active_output(root):
    pointer = root / 'active_session.json'
    return Path(json.loads(pointer.read_text())['output']) if pointer.exists() else root


def corrected(root, times, points, rotations):
    graph = json.loads((root / 'backend/graph_snapshot.json').read_text())
    poses = sorted(graph['poses'], key=lambda p: p['timestamp_ns'])
    ts = np.array([p['timestamp_ns'] for p in poses], dtype=np.int64)
    knots = (ts - ts[0]) * 1e-9
    r = Rotation.from_quat([p['map_from_odom']['rotation_xyzw'] for p in poses])
    t = np.array([p['map_from_odom']['translation'] for p in poses])
    relative = (times - ts[0]) * 1e-9
    cr = Slerp(knots, r)(np.clip(relative, knots[0], knots[-1])) if len(poses) > 1 else r[0]
    ct = np.array([np.interp(relative, knots, t[:, i]) for i in range(3)]).T
    return cr.apply(points) + ct, cr * rotations, graph


def analyze(output, reference):
    with reference.open() as stream:
        rows = list(csv.DictReader(stream))
    truth_ns = np.array([int(Decimal(r['t']) * 10**9) for r in rows], dtype=np.int64)
    ecef = np.array([[float(r[c]) for c in ('ecef_x_m', 'ecef_y_m', 'ecef_z_m')] for r in rows])
    lat, lon = np.radians([float(rows[0]['lat']), float(rows[0]['lon'])])
    re = np.array([[-np.sin(lon), np.cos(lon), 0],
                   [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                   [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    enu = (ecef-ecef[0]) @ re.T
    # Relative timestamp origin avoids loss of nanosecond precision in association.
    origin_ns = int(truth_ns[0])
    truth = PoseTrajectory3D(enu, np.tile([1, 0, 0, 0], (len(enu), 1)), (truth_ns-origin_ns)*1e-9)
    report = {'reference': str(reference.resolve()), 'correlated_reference': True,
              'reference_rows': len(rows), 'reference_span_s': float((truth_ns[-1]-truth_ns[0])*1e-9),
              'alignment': 'one rigid SE(3) alignment per trajectory; scale fixed to 1',
              'association_max_s': .020, 'lever_arm_imu_m': [.11119323117036664, -.09490301030319159, -.07874470264976632],
              'reference_point': 'antenna; nominal lever compensated on estimates',
              'coverage_limitation': 'Only supplied accepted PPK epochs; no interpolation across gaps, no claim for uncovered path', 'runs': {}}
    for run in sorted(output.glob('*/mission.json')):
        root = run.parent
        if root.name not in RUN_NAMES:
            continue
        if not (root / 'backend/finished.json').exists():
            continue
        mission = json.loads(run.read_text()); robot = mission['robots'][0]
        files = active_output(root / 'robots' / robot['robot'])
        with (files / 'trajectory.csv').open() as stream:
            poses = list(csv.DictReader(stream))
        times = np.array([int(p['timestamp_ns']) for p in poses], dtype=np.int64)
        original_times = times - int(robot['offset_ns'])
        p = np.array([[float(v[c]) for c in ('tx', 'ty', 'tz')] for v in poses])
        r = Rotation.from_quat([[float(v[c]) for c in ('qx', 'qy', 'qz', 'qw')] for v in poses])
        cp, cr, graph = corrected(root, times, p, r)
        result = {'frames': len(poses), 'span_s': float((times[-1]-times[0])*1e-9),
                  'graph_poses': len(graph['poses']), 'graph_revision': graph['revision'],
                  'components': graph['components'], 'metrics': {}}
        if (files / 'gnss_diagnostics.jsonl').exists():
            result['gnss'] = gnss_statistics(files, int(times[0]))
        result['raw_trajectory_sha256'] = hashlib.sha256((files / 'trajectory.csv').read_bytes()).hexdigest()
        events = [json.loads(line) for line in (files / 'events.jsonl').read_text().splitlines()]
        result['process_ms'] = {k: float(np.percentile([v['process_ms'] for v in events if v.get('event') == 'vio'], quantile)) for k, quantile in [('median', 50), ('p95', 95), ('max', 100)]}
        if result.get('gnss') and result['gnss']['initialization_latency_s'] is not None:
            activation_ns = int(times[0]) + int(result['gnss']['initialization_latency_s'] * 1e9)
            active_ms = [v['process_ms'] for v in events if v.get('event') == 'vio'
                         and v['timestamp_ns'] >= activation_ns]
            result['active_process_ms'] = {k: float(np.percentile(active_ms, q))
                                          for k, q in [('median', 50), ('p95', 95), ('max', 100)]}
        config = json.loads((files / 'config.json').read_text())
        result['coupling'] = config.get('gnss', {}).get('coupling', 'frame_window') if config.get('gnss', {}).get('enabled') else 'disabled'
        result['loops'] = sum(1 for line in (files / 'loops.jsonl').read_text().splitlines() if line)
        result['communication'] = json.loads((files / 'communication.json').read_text())
        # Earlier experiment binaries charged GNSS publications to the service
        # counters. Preserve those recorded counters and label the known issue;
        # the later transport counters measure the same publication payloads.
        if result.get('gnss') and 'gnss_records_published' not in result['communication']:
            result['communication']['counter_limitation'] = (
                'This build charged GNSS publications to service_attempts and '
                'service_request_field_bytes; these are not service RPCs.')
        if (root / 'replay_summary.json').exists():
            result['replay'] = json.loads((root / 'replay_summary.json').read_text())
        directory = root / 'evaluation'; directory.mkdir(exist_ok=True)
        for name, point, rotation in [('raw', p, r), ('corrected', cp, cr)]:
            antenna = point + rotation.apply(np.tile(report['lever_arm_imu_m'], (len(point), 1)))
            quat = rotation.as_quat()[:, [3, 0, 1, 2]]
            trajectory = PoseTrajectory3D(antenna, quat, (original_times-origin_ns)*1e-9)
            ref, estimate = sync.associate_trajectories(truth, trajectory, max_diff=.020)
            estimate.align(ref, correct_scale=False)
            ape = metrics.APE(metrics.PoseRelation.translation_part); ape.process_data((ref, estimate))
            stats = ape.get_all_statistics()
            result['metrics'][name] = {**stats, 'matched_poses': len(estimate.positions_xyz), 'scale': 1.0,
                                      'matched_path_m': float(ref.distances[-1]),
                                      'first_elapsed_s': float(ref.timestamps[0]-(original_times[0]-origin_ns)*1e-9),
                                      'last_elapsed_s': float(ref.timestamps[-1]-(original_times[0]-origin_ns)*1e-9),
                                      'first_to_last_error_change_m': float(ape.error[-1]-ape.error[0])}
            # Relative displacement in the common rigid alignment; exclude
            # pairs crossing missing reference intervals longer than 0.5 s.
            gaps = np.r_[0, np.cumsum(np.diff(ref.timestamps) > .5)]
            relative_errors = []
            for i, time in enumerate(ref.timestamps):
                j = int(np.searchsorted(ref.timestamps, time+1.))
                if j < len(ref.timestamps) and ref.timestamps[j]-time <= 1.25 and gaps[j] == gaps[i]:
                    relative_errors.append(np.linalg.norm((estimate.positions_xyz[j]-estimate.positions_xyz[i])
                        - (ref.positions_xyz[j]-ref.positions_xyz[i])))
            result['metrics'][name]['displacement_error_1s'] = dict(pairs=len(relative_errors),
                rmse_m=float(np.sqrt(np.mean(np.square(relative_errors)))) if relative_errors else None,
                p95_m=float(np.percentile(relative_errors, 95)) if relative_errors else None)
            saved = ape.get_result(); saved.trajectories['reference'] = ref; saved.trajectories['estimate'] = estimate
            file_interface.save_res_file(str(directory / (name + '_ape.zip')), saved, confirm_overwrite=False)
            np.savetxt(directory / (name + '_errors.csv'), np.column_stack([ref.timestamps, ape.error]), delimiter=',', header='relative_reference_time_s,translation_error_m')
        (directory / 'report.json').write_text(json.dumps(result, indent=2)+'\n')
        report['runs'][root.name] = result
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    report['raw_bit_parity_loops'] = {mode: report['runs'][mode+'_loops']['raw_trajectory_sha256'] == report['runs'][mode+'_no_loops']['raw_trajectory_sha256']
        for mode in ('baseline', 'doppler_only', 'pseudorange_doppler')
        if mode+'_loops' in report['runs'] and mode+'_no_loops' in report['runs']}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name: {mode: values['rmse'] for mode, values in value['metrics'].items()} for name, value in report['runs'].items()}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('reference', type=Path)
    args = parser.parse_args(); analyze(args.output, args.reference)
