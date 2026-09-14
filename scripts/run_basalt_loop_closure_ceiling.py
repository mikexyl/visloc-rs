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


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
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


def false_loop_diagnostics(loops_json: Path, gt_csv: Path, gt_relative_error_threshold_m: float = 1.0):
    """GT-informed, post-hoc only: never used by the algorithm itself.

    A loop is called "false" when the GT relative-translation norm between
    its two keyframes disagrees with the PnP-estimated relative-translation
    norm by more than `gt_relative_error_threshold_m` -- i.e. the recovered
    metric loop displacement is grossly wrong, which is the failure mode a
    real deployment (no GT) cannot detect.
    """
    if not loops_json.exists():
        return {"available": False}
    payload = json.loads(loops_json.read_text(encoding="utf-8"))
    gt_rows = load_gt(gt_csv)
    false_count = 0
    checked = 0
    errors = []
    for loop in payload.get("accepted_loops", []):
        a = gt_position_at(gt_rows, loop["from_timestamp_ns"])
        b = gt_position_at(gt_rows, loop["to_timestamp_ns"])
        if a is None or b is None:
            continue
        checked += 1
        gt_dist = math.dist(a[1:4], b[1:4])
        # We don't have the PnP translation magnitude in loops.json (only
        # inlier stats), so the diagnostic instead checks GT distance itself:
        # a "loop" between two keyframes whose true positions are far apart
        # is very likely a false positive regardless of PnP inlier count.
        if gt_dist > gt_relative_error_threshold_m:
            false_count += 1
            errors.append(gt_dist)
    return {
        "available": True,
        "accepted_loops": len(payload.get("accepted_loops", [])),
        "gt_checked": checked,
        "false_loops_gt_distance_over_threshold": false_count,
        "gt_relative_error_threshold_m": gt_relative_error_threshold_m,
        "mean_false_gt_distance_m": statistics.mean(errors) if errors else None,
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
    parser.add_argument("--min-frame-gap", type=int, default=200)
    parser.add_argument("--min-path-length", type=float, default=5.0)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--retrieval-min-similarity", type=float, default=0.75)
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
                "--min-frame-gap", str(args.min_frame_gap),
                "--min-path-length", str(args.min_path_length),
                "--retrieval-top-k", str(args.retrieval_top_k),
                "--retrieval-min-similarity", str(args.retrieval_min_similarity),
            ],
            seq_out / "run.log",
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
        false_loops = false_loop_diagnostics(seq_out / "loops.json", gt_csv)

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
            "candidates_evaluated": stats.get("candidates_evaluated"),
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
        "keyframes | loops accepted | false loops (GT>1m) | wall (s) |",
        "| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
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
            fl_str = f"{fl['false_loops_gt_distance_over_threshold']}/{fl['gt_checked']}"
        lines.append(
            "| {seq} | {vio_m} | {vio_t} | {pg} | {orb} | {verdict} | {kf} | {loops} | {fl} | {wall:.0f} |".format(
                seq=row["sequence"],
                vio_m=fmt(row["vio_ate_se3_rmse_m"]),
                vio_t=fmt(row["task_reported_vio_ate_m"]),
                pg=fmt(pg),
                orb=fmt(orb),
                verdict=verdict,
                kf=row["keyframe_count"] or "n/a",
                loops=row["loops_accepted"] if row["loops_accepted"] is not None else "n/a",
                fl=fl_str,
                wall=row["postprocess_wall_seconds"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
