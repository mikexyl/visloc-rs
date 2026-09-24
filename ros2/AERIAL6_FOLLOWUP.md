# Aerial 6: gravity/bias initialization follow-up

This is a historical experiment report. Its precision-selector commands
require the archived executable and matching runner revision. Current VIO is
f64-only with stationary gyro-only startup; see the
[current default validation](F64_DEFAULT_VALIDATION.md).

This follow-up isolates aerial 6 using the standalone raw-VIO runner. The
original ROS mission's 13.170 m raw ATE remains a failed result. All controls
use monocular left-camera input, 800 px processing, the same audit executable,
and the supplied aerial camera/IMU extrinsics. No loop closure, PGO, DA3, or
ground-truth feedback is used. ATE is translation RMSE after one rigid SE(3)
alignment with **no scale fitting**.

## Completed full-sequence results

Each control processed all 6,584 images over 329.15 s. The first 400 trajectory
and inertial-state rows of each new full f32 run exactly reproduce its earlier
short control. All four full controls have identical ordered frame timestamps,
IMU counts, and tracked-feature counts; the right camera has zero observations.
Every estimate associates to ground truth, with a maximum timestamp difference
of 1 ns. Camera intrinsics/extrinsics and estimator JSON configuration match.

| Control | Full metric ATE | Diagnostic scale to GT | Endpoint displacement error |
|---|---:|---:|---:|
| f32, noise/bias 0.75/0.75 | 12.447 m | 0.87552 | 31.293 m |
| f64, noise/bias 0.75/0.75 | 6.886 m | 1.00832 | 12.616 m |
| f32, noise/bias 1.5/1.5 | 6.091 m | 1.03928 | 12.457 m |
| f32, first-second gyro offset, noise/bias 0.75/0.75 | **1.925 m** | **1.01017** | **4.819 m** |

The gyro-only intervention reduces full ATE by 84.5% against the matched f32
standalone baseline. It retains single precision and the requested 0.75
uncertainty multipliers. The scale column is diagnostic only; no scale
correction is applied to the primary ATE. Endpoint error is the difference
between estimated and reference start-to-end displacement vectors after the
same full-trajectory rigid rotation.

The 1.5/1.5 control looks strong on the outbound leg but deforms on return:
its first-160-second ATE is 0.496 m, rising to 6.091 m over the whole sequence.
Independent outbound (40–160 s) and inbound (190–280 s) scale diagnostics are
1.0030 and 1.1050. Even full-trajectory Sim(3) fitting leaves 5.678 m ATE.
Changing uncertainty alone is therefore not an adequate aerial-6 repair.

The gyro probe still has 1.925 m ATE and 4.819 m endpoint error; it does not
eliminate all drift. It provides substantially stronger support for a
stationary gyro/monocular initialization repair than for changing arithmetic
mode or globally loosening noise. An online version must be tested on all
mission sequences before replacing the production profile. This experiment
does not establish an exact arithmetic defect or upstream C++ behavior.

![Full aerial-6 controls](../results/graco/vio_failure_audit/a06_followup.png)

## What the additional diagnostics isolate

Aerial 6 develops a large **gravity-direction error coupled to an estimated
accelerometer bias during takeoff**. This occurs in both arithmetic paths:
peak gravity tilt error in the first 20 seconds is 14.872 degrees for the f32
baseline and 14.266 degrees for f64. The f64 control therefore does not have a
sound initialization merely because its initial position ATE is much lower.

The bias direction supports the gravity/bias ambiguity interpretation. With
`R_gt` and `R_est` mapping body to world, compare the estimated accelerometer
bias to the gravity-equivalent vector
`delta_b = R_gt.T * [0,0,9.81] - R_est.T * [0,0,9.81]`.
For f64 at 8 s, the estimated bias norm is 2.208 m/s² and the
gravity-equivalent vector norm is 2.175 m/s²; their vector difference is only
0.296 m/s². At 19.95 s, the corresponding norms are 1.712, 1.759, and
0.311 m/s². This shows that much of the large estimated bias compensates for
the wrong tilt. It is an evaluation diagnostic, not an independent physical
calibration of all accelerometer axes. Ground-truth orientation and real
sensor bias also have uncertainty.

The f64 tilt error remains 7.406 degrees at 79.95 s and 5.702 degrees at
159.95 s, before falling to 0.669 degrees at 239.95 s. This explains why good
early position ATE did not establish a healthy attitude/bias estimate. The
later trajectory deformation is consistent with that estimate changing over
the flight; the data do not isolate every contribution to later drift.

## Sensor-only stationary gyro experiment

The first camera-second contains 124 IMU packets. Its mean angular velocity is
`[-0.002045017, -0.001369748, 0.000488342]` rad/s. The supplied IMU YAML has
noise/random-walk values but no static gyro offset, and the adapter initializes
both static and dynamic gyro bias to zero. Dynamic estimation does recover
approximately this offset by 3 s in the baseline, but its startup history has
already differed from a run corrected from the beginning.
The offset magnitude is about 0.144 degrees/s: it cannot directly account for
the 14-degree excursion over the few seconds before bias recovery. The much
larger excursion arises in the coupled visual/inertial optimization, rather
than being explained by integrating this small offset alone.

Image evidence supports treating the initial mean as a stationary offset:
400 forward/backward-checked optical-flow tracks over the first second have
0.0109 px median and 0.0224 px 95th-percentile displacement. Over the last
second, these are 0.0296 and 0.0415 px; its independently measured mean gyro
is `[-0.002102691, -0.001569757, 0.000468769]` rad/s. Only the **first-second**
mean is used by the experiment; the final-second measurement is corroboration.
Similar nonzero gyro means occur in successful aerial 5/8, so the offset alone
is not a sufficient explanation for sequence failure.

The experiment subtracts this first-second mean using the existing static
gyro-calibration coefficients. It does not change camera calibration,
uncertainty weights, or the estimator implementation. Its initial gravity
direction is unchanged. It reduces peak first-20-second tilt error to
2.433 degrees and metric ATE from 3.105 to 0.707 m while retaining f32 and
0.75/0.75 uncertainty multipliers. This is evidence of a sensitive startup
trajectory, not proof that one gyro-offset constant repairs monocular VIO.
An online implementation would need a stationary-data gate and initialization
buffer; this offline probe is not such an implementation.

## Controls that do not fix aerial 6

| First-400-image control | Metric ATE | Peak gravity tilt error |
|---|---:|---:|
| f32, noise/bias 0.75/0.75 | 3.105 m | 14.872° |
| f64, noise/bias 0.75/0.75 | 1.405 m | 14.266° |
| f32, noise/bias 1.5/1.5 | 0.693 m | 3.968° |
| f32, noise/bias 0.75/1.5 | 6.170 m | 26.810° |
| f32, first-second gyro offset, 0.75/0.75 | 0.707 m | 2.433° |
| Same gyro correction plus radial acceleration correction | 1.026 m | 5.061° |

Loosening only the bias random-walk weight, which helped aerial 7, makes
aerial 6 worse. The last row also forces the initial mean acceleration
magnitude to 9.81 using a radial offset; this does not identify full 3-axis
accelerometer calibration and worsens the gyro-only startup result.

A separate fresh start from camera frame 120 (6 s) yields **1,108.268 m ATE**
over its next 400 images. This crop has a different sensor interval and is
excluded from the identical-input table. Skipping ahead is not a repair.

## Reproduction and scope

All local artifacts are under `results/graco/vio_failure_audit`:

- `a06_followup_analysis.py` recomputes metric trajectory errors,
  yaw-independent gravity tilt, bias/gravity comparisons, figures, and exact
  first-400 trajectory/state parity. It requires completed full runs unless
  `--partial` is explicitly supplied; partial figures are labelled as such.
  Final measurements are in `a06_followup.json`, with `a06_followup.png` and
  `a06_followup.pdf` figures. They agree with each runner's independent
  `evaluation.json` to floating-point precision.
- `static_probe.py` derives the experimental offset solely from first-second
  IMU data and calls the normal runner. Each probe saves its derived
  calibration and `static_probe.json`; the source calibration is untouched.
- `a06_static_sensor_evidence/report.json` records the independent first/last
  stationary image-motion and IMU evidence.
- `source_manifest.json` records executable/source hashes.

The interrupted `a06_f32_noise1p5_full` directory contains only 757 frames and
an explicit incomplete status (SIGTERM after an interrupted task). It is
excluded from completed results; `a06_f32_noise1p5_full_repeat` is the full
6,584-image replacement. Original mission outputs were preserved.

Full controls start from the first image and process all 6,584 images
(329.15 s). Use a new output directory for each run:

```sh
.runtime/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/aerial-06-20m_full_ros2 \
  --output results/graco/NEW_a06_full \
  --binary results/graco/vio_failure_audit/basalt_stream_vio_audit \
  --camera-mode mono --scalar-mode f32 \
  --imu-noise-scale 1.5 --imu-bias-scale 1.5

.runtime/graco-venv/bin/python results/graco/vio_failure_audit/static_probe.py \
  --probe gyro --bag /data/graco/aerial-06-20m_full_ros2 \
  --output results/graco/NEW_a06_gyro_full \
  --binary results/graco/vio_failure_audit/basalt_stream_vio_audit \
  --camera-mode mono --scalar-mode f32 \
  --imu-noise-scale 0.75 --imu-bias-scale 0.75
```

The production profile and estimator code are unchanged by this follow-up.
These are standalone controls, not a rerun or validation of the full ROS
multi-robot mission. The original mission and standalone adapter do not have
bit-identical sensor preprocessing/output; direct numerical improvement
claims use the matched standalone baseline.

## Online initialization follow-up

The sensor-only Rust initializer and controlled validation are documented in [IMU startup validation](IMU_STARTUP_VALIDATION.md). That report preserves the failed controls and distinguishes gyro-only correction from motion-gated startup.
