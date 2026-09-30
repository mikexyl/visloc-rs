# Ground-01 validation of the f64/default IMU startup pipeline

Completed GRACO `/data/graco/ground-01` using the current f64-only estimator and default stationary gyro-only initialization. **Raw full-trajectory ATE RMSE is 2.898798 m**, passing the existing **<4 m** target on this ground sequence. No loop closure, depth, or scale correction was used.

## Profile

- Monocular left camera and IMU, 800×550 processing resolution.
- Supplied `/data/graco/ground-calibration`, including the ground IMU's calibrated 125 Hz rate and camera-to-IMU transform. Its `groundtruth.yaml` defines reference poses for the vehicle IMU frame.
- Same estimator configuration as the accepted aerial profile (`configs/graco/aerial_vio.json`), with 0.75 noise/bias multipliers applied to the **ground** calibration's IMU values.
- f64-only release build, `basalt-lm-workspace-reuse`, AVX2/FMA. Neither precision nor startup is overridden; startup uses the current gyro-only default.
- Original ordered bag samples, starting at frame zero. Ground truth is used for visualization/evaluation only.

## Results

| Metric | Result |
|---|---:|
| Input / estimated / GT-associated frames | 6,676 / 6,676 / 6,676 |
| Sensor duration | 333.75 s |
| Full rigid SE(3) ATE RMSE, no scale fitting | **2.899 m** |
| Median / 95th-percentile position error | 2.532 / 5.377 m |
| Maximum position error | 5.776 m at 313.00 s |
| Last-pose error after the same full-trajectory alignment | 5.108 m |
| Reference traveled distance | 438.46 m |
| Maximum consecutive estimated displacement (50 ms) | 0.096 m |
| Left-camera observations, minimum / median | 163 / 202.5 |
| Right-camera observations | 0 |
| Startup-skipped / missing frames | 0 / 0 |
| VIO processing median / p95 | 50.3 / 74.3 ms |
| Replay wall time | 393.1 s |

Startup passed after exactly one second using 125 IMU samples and replayed all 21 buffered frames. The estimated startup gyro offset was `[0.001164745, -0.001197834, 0.001702849]` rad/s. Averaged-gravity initialization and motion waiting were disabled.

Every camera timestamp and frame ID matches the bag, all poses and inertial states are finite, and all 6,676 poses associate with ground truth within 1 ns. The estimator stderr is empty; Rerun's recording verifier passes. An independent Kabsch alignment reproduces the replay evaluator's RMSE within 1e-9 m. No frames or error outliers were removed.

The result shows stable completion below the 4 m **RMSE** target on ground-01; it is not a claim that every position error is below 4 m or that all ground sequences pass. A separate Sim(3) diagnostic gives scale-to-reference 0.95503948, equivalent to **4.71% excess estimated scale**, and 1.207 m scale-fitted ATE. This scale fit is diagnostic only and is not used for the reported 2.899 m acceptance result.

## Artifacts and reproduction

`results/graco/ground_f64_stationary_20260924/` contains the frozen executable, exact command/source hashes in `experiment_manifest.json`, completion status, independent `assessment.json`, aligned per-frame evaluation, and `trajectory_and_error.png`. The `ground-01/` directory contains effective calibration/configuration, startup report, complete trajectory/inertial states, evaluation, and `playback.rrd` with left-camera feature tracks.

The PNG uses the full-trajectory rigid evaluation alignment. The Rerun recording uses the existing first-pose alignment for visualization, so the visible offset there differs from the ATE alignment.

```bash
RUSTFLAGS='-C target-feature=+avx2,+fma' CARGO_BUILD_JOBS=2 \
  cargo build --offline --locked --release \
  --example basalt_stream_vio --features basalt-lm-workspace-reuse
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .runtime/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/ground-01 \
  --calibration-dir /data/graco/ground-calibration \
  --camera-mode mono --width 800 \
  --imu-noise-scale 0.75 --imu-bias-scale 0.75 \
  --output results/graco/ground01_f64_new_run
```

No production code or defaults were changed for this experiment.
