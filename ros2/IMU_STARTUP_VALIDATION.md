# IMU startup validation — 24 September 2026

Implementation and method references are described in
[IMU startup](../docs/imu-startup.md). Artifacts are under
`results/graco/imu_startup_validation/`. This is a raw-VIO initialization
experiment; it does not claim improved multi-robot loop closure or PGO accuracy.

Subsequent f64 experiments and the shared-profile requirement of ATE below
4 m on every sequence are reported in [the follow-up](UNIFORM_ATE4_VALIDATION.md).
The f32 results below remain historical measurements, including their failures.
Current VIO is f64-only with gyro-only startup enabled by default. Historical
`--scalar-mode` commands below require the archived experiment binaries and
matching runner revisions;
the current tools reject that removed option.

## Protocol

- GRACO aerial 5–8, left camera only, 800 px processing width, f32 VIO,
  unchanged aerial calibration, 0.75 noise and bias multipliers.
- No loop corrections or depth inputs in standalone accuracy runs.
- All initialization decisions use arriving images and IMU only. Ground truth
  is loaded solely by the evaluator/visualizer.
- Primary ATE is position RMSE after one rigid SE(3) alignment, with no scale
  fit and 10 ms maximum association difference. Comparisons use exactly common
  estimate timestamps. Scale fits are diagnostic only.
- Motion mode intentionally omits the stationary pre-initialization prefix.
  This is reported separately from queue drops. No early poses are fabricated.
- Runtime measurements from these concurrently executed experiments are not
  throughput benchmarks.

The original ROS aerial 6 failure was **13.170 m ATE**. Its matched standalone
legacy control is **12.447 m**; ROS and standalone ingestion are not bit-identical.
Use the latter for the controlled comparisons below.

## Full native ROS2 aerial 6 validation

The online **gyro-only** initializer completes the native Rust ROS2 path at
**1.651884 m raw ATE**, versus **13.170038 m** in the original ROS mission.
Both trajectories contain exactly the same 6,584 sensor timestamps; no frames
are excluded and no scale is fitted. The new run reports 6,584 ingested and
estimated frames, zero initialization omissions, zero image/keyframe drops,
zero request failures, and 941 keyframes delivered to the final graph.

Calibration JSON and VIO configuration match the original mission exactly.
The validation uses one robot with loop inference disabled to isolate raw VIO;
the original run contained four robots with loop inference. Pose-graph
corrections do not feed back into either raw trajectory. Sensor intervals are
preserved; nominal replay rate is 4x with backpressure (552.15 s wall time).
This is an accuracy check, not a real-time throughput claim.

Artifacts: `ros_a06_stationary_full/validation_summary.json`,
`a06_ros_comparison.json`, and the reproducible `analyze_ros.py` comparison.
The initializer is still opt-in: this result validates aerial 6, while the
gyro-only aerial 5 failure below prevents adopting it universally.

![Native ROS2 aerial 6 comparison](../results/graco/imu_startup_validation/a06_ros_comparison.png)

## Completed gyro-only control

`a06_stationary_full_v2` completed all 6,584 frames with **1.925 m ATE**, versus
**12.447 m** for `vio_failure_audit/a06_f32_default_full`.
`online_gyro_full_parity.json` confirms bit-identical poses across all frames
against the earlier offline `a06_f32_static_gyro_full` experiment. The online
gate estimated the same offset from the first 124 IMU samples:
`[-0.002045016779334102, -0.0013697484368932638, 0.0004883416841039093] rad/s`.

Gyro correction alone is **not** the recommended general mode: aerial 5
`a05_stationary_400_v2` diverged to **177.653 m ATE**. On aerial 6, adding mean
gravity without waiting for motion yielded **1.113 m** over the first 400
frames, compared with **0.707 m** for gyro-only correction. These ablations
show why changing a startup mean alone is insufficient.

Reproduce the aerial 6 gyro-only mode with a fresh output directory:

```bash
.runtime/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/aerial-06-20m_full_ros2 \
  --output results/graco/my_a06_imu_startup \
  --camera-mode mono --scalar-mode f32 \
  --imu-noise-scale .75 --imu-bias-scale .75 --imu-startup stationary
```

Use `--imu-startup stationary-motion` for the separate motion-gated mode.
The tested binary paths/hashes are recorded in the artifact manifest; the
current default example binary has also been rebuilt with TensorRT loop and
Basalt workspace-reuse features.

## First 400 input frames: matched evaluation intervals

| Sequence | Common poses | Legacy ATE (m) | Motion-gated ATE (m) |
| --- | ---: | ---: | ---: |
| Aerial 5 | 287 | 1.049311 | 0.060408 |
| Aerial 6 | 254 | 2.524896 | 0.571695 |
| Aerial 7 | 323 | 318.226081 | 0.060703 |
| Aerial 8 | 325 | 0.085760 | 0.113508 |

These are startup checks, not full-sequence ATE. Aerial 8 regresses slightly
on this short interval; this is not hidden by scale fitting or removing
outlier poses. Details: `short_common_timestamp_scores.json`.

## Full motion-gated runs

All runs reached the end of their bags. The table excludes each mode's
pre-initialization prefix from both trajectories in a matched comparison.

| Sequence | Estimated / input frames | First pose (s) | Motion ATE (m) | Matched legacy ATE (m) |
| --- | ---: | ---: | ---: | ---: |
| Aerial 5 | 5813 / 5926 | 5.65 | 1.700900 | 2.461243 |
| Aerial 6 | 6438 / 6584 | 7.30 | 7.928259 | 12.027322 |
| Aerial 7 | 7809 / 7886 | 3.85 | 2.584229 | — |
| Aerial 8 | 5488 / 5563 | 3.75 | 0.670077 | — |

There is no matched full f32 standalone legacy control claimed for aerial 7
or 8 in this table. The earlier native ROS results use a different ingestion
path. Aerial 7's matched short baseline diverged, whereas its new full run
completed with the error above. All four short runs exactly reproduce the
corresponding full-run prefixes (`motion_short_full_parity.json`).

`analyze_runs.py` reproduces the independent scores in `full_scores.json`.
The following plot uses the same common timestamps per panel. Consequently,
the aerial 6 gyro-only curve excludes the first 146 frames and scores
1.849 m; the complete 6,584-frame gyro-only score remains 1.925 m.

![Full sequence position errors](../results/graco/imu_startup_validation/full_sequence_errors.png)

The motion-gated aerial 6 run is **rejected as an aerial 6 repair**, despite
its better initial trajectory. Its full ATE is 7.928 m, substantially worse
than gyro-only initialization. Its peak first-20-second gravity tilt is only
2.189°, but return-leg deformation remains: independently aligned outbound
(40–160 s) / inbound (190–280 s) diagnostic scale factors are 1.0073 / 1.1076,
with segment ATE 0.515 / 4.557 m. Gyro-only inbound ATE is 0.507 m with a
0.9891 scale diagnostic. These segment fits are diagnostic and do not replace
the single-alignment full ATE. See `a06_motion_diagnostics.json` and
`accuracy_assessment.json`.

This establishes a useful aerial 6 online gyro initializer, but not one
universally validated initialization profile for all four sequences. The
results do not isolate every contribution to the remaining long-run drift.

## Regressions and integration checks

- Legacy aerial 5 and 6 outputs match the pre-change controls bit-for-bit over
  400 frames (`a05_legacy_parity.json`, `legacy_parity.json`).
- Basalt unit and integration suites pass. New tests cover static statistics,
  constant-rate visual motion, missing/vibrating/nonfinite IMU, bounded buffers,
  packet ordering/release, affine calibration, motion waiting/timeout, and IMU
  gaps while waiting. Both precision paths pass the independent closed-form
  deterministic integration check described in the method note.
- Native Humble/rclrs packages build with the regenerated Rust custom status
  message. The existing bounded-input and estimator-session restart test passes
  (48 submitted images, 32 retained, 16 explicitly dropped, distinct session IDs).
- The preliminary native ROS aerial 6 startup replay
  `ros_a06_motion_400_retry` accepts 400 images and estimates 254 poses after
  omitting 146 pre-initialization images: 37 keyframes, zero image/keyframe
  drops, zero request failures, and a completed final graph solve.

- The final two-robot ROS test `ros_motion_5_7_gpu_retry` completed with GPU
  sequence inference enabled: 400 inputs each, 287/323 estimated poses,
  113/77 initialization omissions, 41/43 keyframes, and three JIST sequences
  per robot. No image/keyframe drops or request failures occurred. All 84
  keyframes reached the backend; 39 accepted updates had non-increasing
  objective and zero rejected solutions. The short route produced no loop
  verification candidates, so this does not validate matching accuracy. Raw
  ROS startup ATE was 0.0633/0.0611 m. `validation_summary.json` accounts for
  every image despite intentionally delayed pose initialization.
- The final standalone executable reproduces the aerial 6 motion-gated
  254-pose startup test bit-for-bit (`final_motion_parity.json`).

Local ROS tests initially used DDS shared memory. The sandbox denied UDP and
CUDA access for the test with inference enabled; the GPU retry succeeded
outside the sandbox. Remote-host networking is not validated by these tests.

## Experimental failures retained

Early v1 runs rejected newly detected pyramid-border features because their
first temporal match retained fewer than half the original detections. v2
establishes a mature track pool on the first match while keeping absolute
support and displacement checks. The first v3 aerial 8 motion wait later
rejected an aging reference; v4 permits reference refresh only when sufficient
surviving tracks still verify small displacement. No failed run is counted as
a completed accuracy result.

The first ROS aerial 6 launch failed because the default ROS log path was
read-only; the runner now uses a mission-local log directory. The successful
retry is separately named and preserved.

`ros_motion_5_7_final` failed before sensor replay with a CUDA access error in
the sandbox. The separately preserved GPU retry above completed successfully.

`implementation_manifest.json` records source and executable hashes. Full
motion runs use the saved v4 executable. The final implementation additionally
rejects IMU gaps during the motion wait and extends diagnostics; its accepted
sensor processing equations are unchanged. Legacy remains the default, with
`--imu-startup stationary-motion` selecting the new path.
