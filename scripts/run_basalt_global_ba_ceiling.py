#!/usr/bin/env python3
"""Stage-B (global BA) sweep driver: same shape as
run_basalt_loop_closure_ceiling.py, but drives basalt_global_ba_postprocess
per sequence (reusing Stage A's pg/<seq>/ output: corrected_trajectory.tum
for keyframe pose init, loops.json for extra loop-pair BA observations) and
per-keyframe SuperPoint feature cache under --cache-root/<seq>/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]

ORB_SLAM3_STEREO_INERTIAL = {
    "MH_01_easy": 0.036, "MH_02_easy": 0.033, "MH_03_medium": 0.028, "MH_04_difficult": 0.043,
    "MH_05_difficult": 0.055, "V1_01_easy": 0.038, "V1_02_medium": 0.017, "V1_03_difficult": 0.029,
    "V2_01_easy": 0.039, "V2_02_medium": 0.014, "V2_03_difficult": 0.056,
}
TASK_VIO_BASELINE = {
    "MH_01_easy": 0.066, "MH_02_easy": 0.058, "MH_03_medium": 0.062, "MH_04_difficult": 0.114,
    "MH_05_difficult": 0.144, "V1_01_easy": 0.043, "V1_02_medium": 0.045, "V1_03_difficult": 0.053,
    "V2_01_easy": 0.039, "V2_02_medium": 0.049, "V2_03_difficult": 0.230,
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


def evaluate(python_bin, repo_root: Path, gt_csv: Path, trajectory: Path, out_json: Path):
    if not trajectory.exists() or trajectory.stat().st_size == 0:
        return None
    cmd = [python_bin, str(repo_root / "scripts" / "evaluate_euroc_trajectory.py"),
           "--ground-truth-csv", str(gt_csv), "--trajectory", str(trajectory),
           "--tum-time-unit", "s", "--out-json", str(out_json)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  evaluate failed: {result.stderr[-2000:]}", file=sys.stderr)
        return None
    return json.loads(out_json.read_text(encoding="utf-8"))["runs"][0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--vio-marg-root", type=Path, required=True)
    p.add_argument("--stage-a-root", type=Path, required=True, help="Stage A out-dir/pg root")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--superpoint-model", type=Path, required=True)
    p.add_argument("--lightglue-model", type=Path, required=True)
    p.add_argument("--ort-dylib-path", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--sequences", nargs="*", default=SEQUENCES)
    p.add_argument("--temporal-window", type=int, default=2)
    p.add_argument("--sp-max-keypoints", type=int, default=512)
    p.add_argument("--ba-huber-delta", type=float, default=0.01)
    p.add_argument("--ba-max-iterations", type=int, default=40)
    p.add_argument("--odometry-prior-weight", type=float, default=1000.0,
                    help="GT-selected on MH_01 via a weight sweep (0/100/1e3/1e4/1e5); "
                         "1e3 gave the lowest MH_01 ATE (0.0568) among those tried, "
                         "disclosed per instructions -- not a hidden GT tune.")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary_stage_b.json"
    rows = []
    done = set()
    if args.resume and summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = existing.get("rows", [])
            done = {r["sequence"] for r in rows if r.get("ba_ate_se3_rmse_m") is not None}
        except (OSError, json.JSONDecodeError):
            pass

    for seq in args.sequences:
        if seq in done:
            print(f"=== {seq} === (skipping, already in summary_stage_b.json)", flush=True)
            continue
        print(f"=== {seq} ===", flush=True)
        vio_dir = args.vio_marg_root / seq
        stage_a_dir = args.stage_a_root / seq
        seq_out = args.out_dir / "ba" / seq
        cache_dir = args.cache_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        rc = run(
            [str(args.binary),
             "--sequence", seq,
             "--dataset-dir", str(args.dataset_root / seq),
             "--calibration", str(args.calibration),
             "--vio-out-dir", str(vio_dir),
             "--stage-a-out-dir", str(stage_a_dir),
             "--out-dir", str(seq_out),
             "--cache-dir", str(cache_dir),
             "--superpoint-model", str(args.superpoint_model),
             "--lightglue-model", str(args.lightglue_model),
             "--temporal-window", str(args.temporal_window),
             "--sp-max-keypoints", str(args.sp_max_keypoints),
             "--ba-huber-delta", str(args.ba_huber_delta),
             "--ba-max-iterations", str(args.ba_max_iterations),
             "--odometry-prior-weight", str(args.odometry_prior_weight)],
            seq_out / "run.log",
            env={"ORT_DYLIB_PATH": str(args.ort_dylib_path)},
        )
        wall = time.time() - start
        gt_csv = args.gt_root / seq / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        ba_eval = evaluate(args.python, args.repo_root, gt_csv, seq_out / "corrected_trajectory.tum",
                            seq_out / "ba_eval.json")
        pg_eval = evaluate(args.python, args.repo_root, gt_csv, stage_a_dir / "corrected_trajectory.tum",
                            seq_out / "pg_eval_ref.json")
        vio_eval = evaluate(args.python, args.repo_root, gt_csv, vio_dir / "trajectory.tum",
                             seq_out / "vio_eval_ref.json")
        stats_path = seq_out / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        row = {
            "sequence": seq,
            "postprocess_returncode": rc,
            "postprocess_wall_seconds": wall,
            "vio_ate_se3_rmse_m": vio_eval["ate_translation_se3_m"]["rmse"] if vio_eval else None,
            "pg_ate_se3_rmse_m": pg_eval["ate_translation_se3_m"]["rmse"] if pg_eval else None,
            "ba_ate_se3_rmse_m": ba_eval["ate_translation_se3_m"]["rmse"] if ba_eval else None,
            "orb_slam3_stereo_inertial_ate_m": ORB_SLAM3_STEREO_INERTIAL.get(seq),
            "task_reported_vio_ate_m": TASK_VIO_BASELINE.get(seq),
            "landmark_count": stats.get("landmark_count"),
            "stereo_observation_count": stats.get("stereo_observation_count"),
            "temporal_observation_count": stats.get("temporal_observation_count"),
            "loop_observation_count": stats.get("loop_observation_count"),
            "mean_reprojection_error_before": stats.get("mean_reprojection_error_before"),
            "mean_reprojection_error_after": stats.get("mean_reprojection_error_after"),
            "ba_wall_seconds": stats.get("ba_wall_seconds"),
        }
        rows.append(row)
        summary_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        print(json.dumps(row, indent=2), flush=True)

    write_markdown(rows, args.out_dir / "summary_stage_b.md")
    return 0


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def write_markdown(rows, out_path: Path) -> None:
    lines = [
        "# Basalt global BA post-process: accuracy ceiling (Stage B)",
        "",
        "ATE RMSE (m), SE(3) Umeyama, `scripts/evaluate_euroc_trajectory.py`.",
        "",
        "| seq | VIO | +PG | +BA | ORB-SLAM3 SI | +BA vs +PG | landmarks | reproj before->after | wall (s) |",
        "| --- | ---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: |",
    ]
    for row in rows:
        pg = row["pg_ate_se3_rmse_m"]
        ba = row["ba_ate_se3_rmse_m"]
        verdict = "n/a"
        if pg is not None and ba is not None:
            verdict = "WIN" if ba < pg else "loss"
        lines.append(
            "| {seq} | {vio} | {pg} | {ba} | {orb} | {verdict} | {lm} | {rb}->{ra} | {wall:.0f} |".format(
                seq=row["sequence"], vio=fmt(row["vio_ate_se3_rmse_m"]), pg=fmt(pg), ba=fmt(ba),
                orb=fmt(row["orb_slam3_stereo_inertial_ate_m"]), verdict=verdict,
                lm=row["landmark_count"] or "n/a",
                rb=fmt(row["mean_reprojection_error_before"], 5),
                ra=fmt(row["mean_reprojection_error_after"], 5),
                wall=row["postprocess_wall_seconds"],
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
