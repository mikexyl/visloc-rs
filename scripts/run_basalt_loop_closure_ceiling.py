#!/usr/bin/env python3
"""Stage-A loop-closure-ceiling sweep driver.

For each of the 11 EuRoC sequences this:
  1. Runs `basalt_loop_closure_postprocess` (release binary, already built)
     over the marg-data-enabled Basalt VIO rerun in `--vio-marg-root`,
     producing `corrected_trajectory.tum`, `loops.json`, `stats.json`.
  2. Evaluates both the untouched VIO trajectory (`trajectory.tum` from the
     same rerun) and the corrected trajectory against ground truth with
     `scripts/evaluate_euroc_trajectory.py` (SE(3) Umeyama ATE).
  3. Runs a ground-truth-informed *diagnostic* (never fed back into the
     algorithm) that checks each accepted loop's PnP-estimated relative pose
     against the GT-interpolated relative pose, to report false-loop counts.
  4. Aggregates a table against the task-provided ORB-SLAM3 stereo-inertial
     numbers into `summary.json` / `summary.md`.

This is intentionally a thin driver: all algorithmic work is in the Rust
binary and in `evaluate_euroc_trajectory.py`; this script only orchestrates
and reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

SEQUENCES = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]

# Task-provided, same-protocol reference numbers (ATE SE(3) RMSE, metres).
ORB_SLAM3_STEREO_INERTIAL = {
    "MH_01_easy": 0.036,
    "MH_02_easy": 0.033,
    "MH_03_medium": 0.028,
    "MH_04_difficult": 0.043,
    "MH_05_difficult": 0.055,
    "V1_01_easy": 0.038,
    "V1_02_medium": 0.017,
    "V1_03_difficult": 0.029,
    "V2_01_easy": 0.039,
    "V2_02_medium": 0.014,
    "V2_03_difficult": 0.056,
}
TASK_VIO_BASELINE = {
    "MH_01_easy": 0.066,
    "MH_02_easy": 0.058,
    "MH_03_medium": 0.062,
    "MH_04_difficult": 0.114,
    "MH_05_difficult": 0.144,
    "V1_01_easy": 0.043,
    "V1_02_medium": 0.045,
    "V1_03_difficult": 0.053,
    "V2_01_easy": 0.039,
    "V2_02_medium": 0.049,
    "V2_03_difficult": 0.230,
}


def run(cmd: list[str], log_path: Path, env: dict | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = None
    if env:
        import os

        full_env = {**os.environ, **env}
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=full_env)
    return proc.returncode


def evaluate(python_bin: str, repo_root: Path, gt_csv: Path, trajectory: Path, out_json: Path) -> dict | None:
    if not trajectory.exists() or trajectory.stat().st_size == 0:
        return None
    cmd = [
        python_bin,
        str(repo_root / "scripts" / "evaluate_euroc_trajectory.py"),
        "--ground-truth-csv",
        str(gt_csv),
        "--trajectory",
        str(trajectory),
        "--tum-time-unit",
        "s",
        "--out-json",
        str(out_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  evaluate failed: {result.stderr[-2000:]}", file=sys.stderr)
        return None
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    return payload["runs"][0]


def load_gt(gt_csv: Path) -> list[tuple[int, float, float, float, float, float, float, float]]:
    rows = []
    with gt_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            t_ns = int(row[0])
            tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            rows.append((t_ns, tx, ty, tz, qw, qx, qy, qz))
    rows.sort(key=lambda r: r[0])
    return rows


def gt_position_at(gt_rows, t_ns: int, max_diff_ns: int = 20_000_000):
    """Nearest-neighbour GT position lookup (diagnostic only)."""
    import bisect

    times = [r[0] for r in gt_rows]
    idx = bisect.bisect_left(times, t_ns)
    best = None
    for candidate in (idx - 1, idx):
        if 0 <= candidate < len(gt_rows):
            diff = abs(gt_rows[candidate][0] - t_ns)
            if best is None or diff < best[0]:
                best = (diff, gt_rows[candidate])
    if best is None or best[0] > max_diff_ns:
        return None
    return best[1]


def quat_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate(q, v):
    # v' = q * (0,v) * q^-1, unit quaternion so inverse == conjugate.
    qv = (0.0, v[0], v[1], v[2])
    rw, rx, ry, rz = quat_mul(quat_mul(q, qv), quat_conjugate(q))
    return (rx, ry, rz)


def relative_pose_error(gt_a, gt_b, loop_translation, loop_quat_wxyz):
    """GT-informed, post-hoc only -- never used by the algorithm itself.

    Computes ``T_gtA_to_gtB`` from the two nearest-timestamp GT body poses
    (EuRoC's `state_groundtruth_estimate0` is already `imu_to_world`, the
    same convention the Rust side uses) with the identical
    world_to_camera-style composition the pose graph uses, then compares its
    translation/rotation against the accepted loop's own
    ``measurement_translation`` / ``measurement_quaternion_wxyz``.
    """
    _, ax, ay, az, aqw, aqx, aqy, aqz = gt_a
    _, bx, by, bz, bqw, bqx, bqy, bqz = gt_b
    qa = (aqw, aqx, aqy, aqz)
    qb = (bqw, bqx, bqy, bqz)
    qa_inv = quat_conjugate(qa)
    # world_to_body_a rotates by qa_inv; translation of world_to_body_a is
    # -qa_inv * (ax,ay,az).
    t_world_to_a = tuple(-c for c in quat_rotate(qa_inv, (ax, ay, az)))
    # measurement = T_bodyA_to_bodyB s.t. T_worldToB = measurement o T_worldToA
    # => measurement.rotation = qb_inv * qa (since T_worldToB.rotation = qb_inv,
    #    T_worldToA.rotation = qa_inv, and rotation composes as q_meas * qa_inv = qb_inv
    #    => q_meas = qb_inv * qa).
    qb_inv = quat_conjugate(qb)
    q_meas = quat_mul(qb_inv, qa)
    t_world_to_b = tuple(-c for c in quat_rotate(qb_inv, (bx, by, bz)))
    # measurement.translation = t_world_to_b - q_meas * t_world_to_a
    t_meas = tuple(
        wb - r for wb, r in zip(t_world_to_b, quat_rotate(q_meas, t_world_to_a))
    )
    translation_error = math.dist(t_meas, loop_translation)
    q_loop = tuple(loop_quat_wxyz)
    q_rel = quat_mul(quat_conjugate(q_meas), q_loop)
    w = max(-1.0, min(1.0, q_rel[0]))
    rotation_error_deg = math.degrees(2.0 * math.acos(abs(w)))
    return translation_error, rotation_error_deg


def false_loop_diagnostics(loops_json: Path, gt_csv: Path, gt_relative_error_threshold_m: float = 0.2):
    """GT-informed, post-hoc only: never used by the algorithm itself.

    A loop is "GT-false" when its own relative-pose estimate
    (`measurement_translation`/`measurement_quaternion_wxyz`, logged by the
    Rust binary, GT-blind) disagrees with the GT-derived relative pose
    between the same two keyframe timestamps by more than
    `gt_relative_error_threshold_m` in translation.
    """
    if not loops_json.exists():
        return {"available": False}
    payload = json.loads(loops_json.read_text(encoding="utf-8"))
    gt_rows = load_gt(gt_csv)
    false_count = 0
    checked = 0
    translation_errors = []
    rotation_errors = []
    for loop in payload.get("accepted_loops", []):
        a = gt_position_at(gt_rows, loop["from_timestamp_ns"])
        b = gt_position_at(gt_rows, loop["to_timestamp_ns"])
        if a is None or b is None or "measurement_translation" not in loop:
            continue
        checked += 1
        t_err, r_err = relative_pose_error(
            a, b, tuple(loop["measurement_translation"]), tuple(loop["measurement_quaternion_wxyz"])
        )
        translation_errors.append(t_err)
        rotation_errors.append(r_err)
        if t_err > gt_relative_error_threshold_m:
            false_count += 1
    return {
        "available": True,
        "accepted_loops": len(payload.get("accepted_loops", [])),
        "gt_checked": checked,
        "false_loops_over_threshold": false_count,
        "gt_relative_error_threshold_m": gt_relative_error_threshold_m,
        "mean_relative_translation_error_m": statistics.mean(translation_errors) if translation_errors else None,
        "mean_relative_rotation_error_deg": statistics.mean(rotation_errors) if rotation_errors else None,
    }


def wait_for_vio_ready(vio_dir: Path, progress_jsonl: Path | None, seq: str, poll_s: float = 15.0, timeout_s: float = 3600.0) -> bool:
    """Block until the marg-data VIO rerun for `seq` has finished.

    Prefers the VIO orchestrator's `progress.jsonl` (one JSON line per
    finished sequence with a `returncode`) when given; falls back to
    checking that `trajectory.csv` and a non-empty `marg_data/` exist.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if progress_jsonl and progress_jsonl.exists():
            try:
                for line in progress_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip().lstrip("﻿")
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("sequence") == seq:
                        return rec.get("returncode") == 0
            except (OSError, json.JSONDecodeError):
                pass
        traj = vio_dir / "trajectory.csv"
        marg = vio_dir / "marg_data"
        if traj.exists() and marg.is_dir() and any(marg.iterdir()):
            # No progress.jsonl signal available; best-effort readiness.
            return True
        time.sleep(poll_s)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--vio-marg-root", type=Path, required=True)
    parser.add_argument("--vio-progress-jsonl", type=Path, default=None,
                         help="VIO rerun orchestrator's progress.jsonl, to wait on per-sequence readiness")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sequences", nargs="*", default=SEQUENCES)
    parser.add_argument("--superpoint-model", type=Path, required=True)
    parser.add_argument("--lightglue-model", type=Path, required=True)
    parser.add_argument("--ort-dylib-path", type=Path, required=True)
    parser.add_argument("--min-temporal-gap-s", type=float, default=20.0)
    parser.add_argument("--proximity-base-m", type=float, default=1.0)
    parser.add_argument("--proximity-drift-frac", type=float, default=0.03)
    parser.add_argument("--max-viewing-angle-deg", type=float, default=45.0)
    parser.add_argument("--min-inliers", type=int, default=40)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.4)
    parser.add_argument("--vio-consistency-translation-base-m", type=float, default=0.10)
    parser.add_argument("--vio-consistency-translation-per-m", type=float, default=0.02)
    parser.add_argument("--vio-consistency-rotation-base-deg", type=float, default=2.0)
    parser.add_argument("--vio-consistency-rotation-per-10m-deg", type=float, default=0.5)
    parser.add_argument("--gt-false-threshold-m", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true",
                         help="skip sequences already present in an existing summary.json")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    rows = []
    done_sequences = set()
    if args.resume and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = existing.get("rows", [])
            done_sequences = {r["sequence"] for r in rows if r.get("pg_ate_se3_rmse_m") is not None}
        except (OSError, json.JSONDecodeError):
            pass

    for seq in args.sequences:
        if seq in done_sequences:
            print(f"=== {seq} === (skipping, already in summary.json)", flush=True)
            continue
        print(f"=== {seq} ===", flush=True)
        vio_dir = args.vio_marg_root / seq
        print(f"  waiting for VIO MargData rerun readiness...", flush=True)
        ready = wait_for_vio_ready(vio_dir, args.vio_progress_jsonl, seq)
        if not ready:
            print(f"  VIO rerun for {seq} not ready / failed -- skipping", file=sys.stderr, flush=True)
            rows.append({"sequence": seq, "error": "vio_rerun_not_ready"})
            summary_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
            continue
        seq_out = args.out_dir / "pg" / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        start = time.time()
        rc = run(
            [
                str(args.binary),
                "--sequence", seq,
                "--dataset-dir", str(args.dataset_root / seq),
                "--calibration", str(args.calibration),
                "--vio-out-dir", str(vio_dir),
                "--out-dir", str(seq_out),
                "--superpoint-model", str(args.superpoint_model),
                "--lightglue-model", str(args.lightglue_model),
                "--min-temporal-gap-s", str(args.min_temporal_gap_s),
                "--proximity-base-m", str(args.proximity_base_m),
                "--proximity-drift-frac", str(args.proximity_drift_frac),
                "--max-viewing-angle-deg", str(args.max_viewing_angle_deg),
                "--min-inliers", str(args.min_inliers),
                "--min-inlier-ratio", str(args.min_inlier_ratio),
                "--vio-consistency-translation-base-m", str(args.vio_consistency_translation_base_m),
                "--vio-consistency-translation-per-m", str(args.vio_consistency_translation_per_m),
                "--vio-consistency-rotation-base-deg", str(args.vio_consistency_rotation_base_deg),
                "--vio-consistency-rotation-per-10m-deg", str(args.vio_consistency_rotation_per_10m_deg),
            ],
            seq_out / "run.log",
            env={"ORT_DYLIB_PATH": str(args.ort_dylib_path)},
        )
        wall = time.time() - start
        gt_csv = args.gt_root / seq / "mav0" / "state_groundtruth_estimate0" / "data.csv"

        vio_eval = evaluate(
            args.python, args.repo_root, gt_csv, vio_dir / "trajectory.tum", seq_out / "vio_eval.json"
        )
        pg_eval = evaluate(
            args.python,
            args.repo_root,
            gt_csv,
            seq_out / "corrected_trajectory.tum",
            seq_out / "pg_eval.json",
        )
        false_loops = false_loop_diagnostics(
            seq_out / "loops.json", gt_csv, gt_relative_error_threshold_m=args.gt_false_threshold_m
        )

        stats_path = seq_out / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

        row = {
            "sequence": seq,
            "postprocess_returncode": rc,
            "postprocess_wall_seconds": wall,
            "vio_ate_se3_rmse_m": vio_eval["ate_translation_se3_m"]["rmse"] if vio_eval else None,
            "pg_ate_se3_rmse_m": pg_eval["ate_translation_se3_m"]["rmse"] if pg_eval else None,
            "orb_slam3_stereo_inertial_ate_m": ORB_SLAM3_STEREO_INERTIAL.get(seq),
            "task_reported_vio_ate_m": TASK_VIO_BASELINE.get(seq),
            "keyframe_count": stats.get("keyframe_count"),
            "candidate_pairs": stats.get("candidate_pairs"),
            "verified_both_directions": stats.get("verified_both_directions"),
            "loops_accepted": stats.get("loops_accepted"),
            "pgo_initial_cost": stats.get("pgo_initial_cost"),
            "pgo_final_cost": stats.get("pgo_final_cost"),
            "false_loops": false_loops,
        }
        rows.append(row)
        (args.out_dir / "summary.json").write_text(
            json.dumps({"rows": rows}, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, indent=2), flush=True)

    write_markdown(rows, args.out_dir / "summary.md")
    return 0


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def write_markdown(rows, out_path: Path) -> None:
    lines = [
        "# Basalt loop-closure post-process: accuracy ceiling (Stage A, pose-graph only)",
        "",
        "ATE RMSE (m), SE(3) Umeyama alignment, `scripts/evaluate_euroc_trajectory.py`, "
        "10 ms association.",
        "",
        "| seq | VIO (measured) | VIO (task) | +PG | ORB-SLAM3 SI | +PG vs ORB-SLAM3 | "
        "keyframes | candidates | verified | accepted | GT-false (>0.2m) | wall (s) |",
        "| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pg = row["pg_ate_se3_rmse_m"]
        orb = row["orb_slam3_stereo_inertial_ate_m"]
        verdict = "n/a"
        if pg is not None and orb is not None:
            verdict = "WIN" if pg < orb else "loss"
        fl = row["false_loops"]
        fl_str = "n/a"
        if fl.get("available"):
            fl_str = f"{fl['false_loops_over_threshold']}/{fl['gt_checked']}"
        lines.append(
            "| {seq} | {vio_m} | {vio_t} | {pg} | {orb} | {verdict} | {kf} | {cand} | {ver} | {loops} | {fl} | {wall:.0f} |".format(
                seq=row["sequence"],
                vio_m=fmt(row["vio_ate_se3_rmse_m"]),
                vio_t=fmt(row["task_reported_vio_ate_m"]),
                pg=fmt(pg),
                orb=fmt(orb),
                verdict=verdict,
                kf=row["keyframe_count"] or "n/a",
                cand=row.get("candidate_pairs") if row.get("candidate_pairs") is not None else "n/a",
                ver=row.get("verified_both_directions") if row.get("verified_both_directions") is not None else "n/a",
                loops=row["loops_accepted"] if row["loops_accepted"] is not None else "n/a",
                fl=fl_str,
                wall=row["postprocess_wall_seconds"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
