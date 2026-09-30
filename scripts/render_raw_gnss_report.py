#!/usr/bin/env python3
"""Render completed raw GNSS comparisons, without filling reference gaps."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evo.tools import file_interface

MODES = ('baseline', 'doppler_only', 'pseudorange_doppler')
LABELS = ('VIO baseline', 'Doppler only', 'Pseudorange + Doppler')
COLORS = ('#3569b0', '#dc7d27', '#298969')


def render(root):
    data = json.loads((root / 'evaluation.json').read_text())
    runs = data['runs']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    for i, (mode, label, color) in enumerate(zip(MODES, LABELS, COLORS)):
        for suffix, marker in (('_no_loops', 'o'), ('_loops', 's')):
            name = mode + suffix
            if name not in runs:
                continue
            row = runs[name]
            axes[0].scatter([i-.08 if suffix == '_no_loops' else i+.08],
                [row['metrics']['corrected']['rmse']], color=color, marker=marker, s=70)
            ape = file_interface.load_res_file(root / name / 'evaluation/corrected_ape.zip', load_trajectories=True)
            reference = ape.trajectories['reference']
            elapsed = reference.timestamps - reference.timestamps[0] + row['metrics']['corrected']['first_elapsed_s']
            errors = ape.np_arrays['error_array'].copy()
            errors[np.r_[False, np.diff(reference.timestamps) > .5]] = np.nan
            axes[1].plot(elapsed, errors, color=color,
                linestyle='-' if suffix == '_no_loops' else '--', linewidth=1.2,
                label=label + (' / loops' if suffix == '_loops' else ''))
    axes[0].set(xticks=range(3), xticklabels=LABELS, ylabel='Rigid-aligned ATE RMSE (m)',
        title='Corrected trajectories, scale fixed to 1')
    axes[0].tick_params(axis='x', labelsize=9)
    axes[0].set_xlim(-.5, 2.5)
    peak = max((r['metrics']['corrected']['rmse'] for r in runs.values()), default=1.)
    axes[0].set_ylim(0, max(1., peak*1.2))
    axes[0].plot([], [], 'ko', label='Loops disabled')
    axes[0].plot([], [], 'ks', label='Loops enabled')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Sensor elapsed time (s)', ylabel='Translation error (m)', title='Available PPK reference only')
    axes[1].legend(fontsize=7, ncol=2)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(root / 'accuracy.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    reference_drawn = False
    for mode, label, color in zip(MODES, LABELS, COLORS):
        name = mode+'_no_loops'
        if name not in runs: continue
        ape = file_interface.load_res_file(root / name / 'evaluation/raw_ape.zip', load_trajectories=True)
        ref, estimate = ape.trajectories['reference'], ape.trajectories['estimate']
        gaps = np.r_[False, np.diff(ref.timestamps) > .5]
        points = estimate.positions_xyz.copy(); points[gaps] = np.nan
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.3, label=label)
        if not reference_drawn:
            ax.scatter(ref.positions_xyz[:, 0], ref.positions_xyz[:, 1], color='#333333', s=3, alpha=.4, label='Correlated PPK')
            reference_drawn = True
    ax.set(xlabel='East (m)', ylabel='North (m)', title='Antenna trajectories on evaluated coverage')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.savefig(root / 'trajectories.png', dpi=180); plt.close(fig)

    lines = ['# South-ece raw GNSS VIO experiment', '',
        f'Completed full comparisons: {len(runs)}/6.', '',
        'Experimental branch `experiment/raw-gnss-vio`, sequential baseline `73b52a4`. '
        'Stereo-inertial f64 Basalt, stationary gyro-only initialization, measured calibration, '
        'fixed nominal antenna lever, and unchanged sequential PGO with relative-pose loops.', '',
        '## Accuracy', '',
        'evo translation APE uses one rigid alignment per trajectory, no scale fitting, '
        '20 ms timestamp association, and antenna lever compensation. PPK uses the same rover '
        'measurements as the estimator and is a correlated, partial reference. No PPK enters '
        'GNSS initialization or time-offset estimation.', '',
        '| Mode | Loops | Raw ATE (m) | Corrected ATE (m) | Matched poses | Accepted loops |',
        '|---|---|---:|---:|---:|---:|']
    for mode, label in zip(MODES, LABELS):
        for suffix in ('_no_loops', '_loops'):
            name = mode+suffix
            if name not in runs: continue
            r = runs[name]; raw, corrected = r['metrics']['raw'], r['metrics']['corrected']
            lines.append(f"| {label} | {'Yes' if suffix == '_loops' else 'No'} | {raw['rmse']:.3f} | {corrected['rmse']:.3f} | {raw['matched_poses']} | {r['loops']} |")
    if all(mode+'_no_loops' in runs for mode in MODES):
        baseline = runs['baseline_no_loops']['metrics']['raw']['rmse']
        changes = [runs[mode+'_no_loops']['metrics']['raw']['rmse']-baseline for mode in MODES[1:]]
        lines += ['', f'GNSS did not improve ATE on this recording: Doppler-only increased '
            f'RMSE by {changes[0]:.3f} m ({changes[0]/baseline*100:.1f}%), and code plus Doppler '
            f'by {changes[1]:.3f} m ({changes[1]/baseline*100:.1f}%). '
            'The fused runs completed with finite trajectories. GNSS remains experimental and '
            'disabled by default. The loop-enabled fused runs accepted no loops, so they do '
            'not demonstrate a loop-correction benefit.']
    lines += ['', '![Accuracy and translation-error time series](accuracy.png)', '', '![Evaluated trajectories](trajectories.png)', '',
        f"Reference: {data['reference_rows']} accepted epochs spanning {data['reference_span_s']:.1f} s; "
        'the recording spans 239.7 s. Missing intervals are excluded. Accuracy outside reference '
        'coverage is unknown. Per-run evo ZIPs, relative-displacement errors, and exact coverage '
        'are in the `evaluation` folders and `evaluation.json`.', '', '## Runtime and diagnostics', '',
        '| Run | Processing median / p95 (ms) | Wall time (s) | GNSS initialization (s) | Used GNSS epochs | Input drops |',
        '|---|---:|---:|---:|---:|---:|']
    for name, r in runs.items():
        g = r.get('gnss') or {}; init = g.get('initialization_latency_s')
        replay = r.get('replay', {}); wall = replay.get('wall_seconds')
        drops = r['communication']['sensor_ingress_drops']
        wall_text = f'{wall:.1f}' if wall is not None else 'unavailable'
        lines.append(f"| {name} | {r['process_ms']['median']:.1f} / {r['process_ms']['p95']:.1f} | {wall_text}")
        lines[-1] += f" | {init:.2f}" if init is not None else ' | inactive'
        lines[-1] += f" | {g.get('used_epochs', 0)} | {drops} |"
    lines += ['', 'Replay requested 0.25x and used backpressure; actual wall time includes slower '
        'estimator processing. This prototype does not meet the live camera frame rate. '
        'No-loop GNSS runs used the earlier dense reduction; loop GNSS runs used compact state '
        'support and reused validated landmark trial tokens. Runtime differences therefore cannot '
        'be attributed solely to loops. Fused replays ran concurrently; these timings include '
        'shared-machine contention. Raw trajectory parity across these builds is checked below.', '',
        'Raw trajectory bit parity with loops enabled/disabled: `'+json.dumps(data.get('raw_bit_parity_loops', {}))+'`.', '',
        'GNSS diagnostics include per-satellite rejection decisions, normalized residuals, receiver '
        'clocks, used-epoch gaps, alignment, queue drops, and communicated payloads. '
        'Ephemeris acquisition and pre-initialization history rejection are reported separately from '
        'queue losses. See `gnss_records.jsonl`, `gnss_diagnostics.jsonl`, `gnss_effective.json`, '
        'and `replay_summary.json` in each run.', '',
        '| Fused mode (loop-enabled run) | First screening accepted code / Doppler | First screening rejected code / Doppler | Final normalized code / Doppler RMS | Used epochs |',
        '|---|---:|---:|---:|---:|']
    for mode, label in zip(MODES[1:], LABELS[1:]):
        r = runs.get(mode+'_loops')
        if not r: continue
        g = r['gnss']; d = g['diagnostics']; residuals = g['normalized_residuals']
        rms = [f"{residuals[c]['rms']:.3f}" if residuals[c] else 'disabled'
            for c in ('pseudorange', 'doppler')]
        lines.append(f"| {label} | {d['accepted_pseudoranges']} / {d['accepted_dopplers']} | "
            f"{d['rejected_pseudoranges']} / {d['rejected_dopplers']} | {' / '.join(rms)} | {g['used_epochs']} |")
    fused = runs.get('pseudorange_doppler_loops')
    if fused:
        g = fused['gnss']; d = g['diagnostics']; c = fused['communication']
        lines += ['', f"Initialization took {g['initialization_latency_s']:.2f} s; the frozen GNSS-to-sensor "
            f"offset was {d['time_offset_s']*1000:.0f} ms (estimated standard deviation "
            f"{d['timing_std_s']*1000:.2f} ms). Of {d['epochs']} decoded epochs, {g['used_epochs']} "
            f"had optimized observations. Rejections included {d['stale_epochs']} startup/history "
            f"epochs outside the active window, {d['missing_ephemeris']} satellite observations "
            f"without usable ephemerides, {d['low_cn0']} below CN0, and {d['low_elevation']} below "
            f"elevation. Decoder checksum/malformed counts and all queue-drop counts were zero.", '',
            'Used-observation gaps longer than one second: `'+json.dumps(g['used_epoch_gaps_over_1s'])+'`. '
            'These are gaps in accepted GNSS support, not queue losses. The earlier no-loop build '
            'did not log observation times, so its gap and factor-age statistics are unavailable.', '',
            'Receiver clock ranges (meters and meters/second): `'+json.dumps(g['clock_min_max'])+'`.', '',
            f"GNSS raw input occupied {fused['replay']['topic_cdr_bytes']['gnss_raw']:,} serialized "
            f"bytes; {c['gnss_records_published']} typed GNSS publications contained "
            f"{c['gnss_record_field_bytes']:,} field bytes. Counts exclude DDS overhead. The earlier "
            'no-loop build incorrectly charged these publication counters to service traffic; '
            'the evaluation preserves and labels that limitation.', '',
            'At the final optimization before retirement, observations were about 0.52 s old. '
            'This factor-age statistic is not a USB delivery-latency measurement.']
    lines += ['', '## Limits', '',
        'The housing and antenna geometry is nominal (0.6 mm gasket, 20 mm per-axis uncertainty); '
        'extrinsics are fixed. Galileo currently uses GPS Klobuchar atmospheric correction rather '
        'than NeQuick. Satellite and atmospheric states are frozen per window. Legacy VIO '
        '`MargData` does not serialize the added GNSS state/prior; GNSS replay uses the native '
        'joint prior. Carrier phase, RTK fusion, and online extrinsic estimation are deferred.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    print(root / 'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    render(parser.parse_args().output.resolve())
