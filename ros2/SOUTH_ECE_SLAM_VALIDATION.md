# South-ece-rtk-test: our sequence/refinement SLAM

The dataset path was replaced with a new recording on 2026-09-27. This report describes the archived 2026-09-26 recording; rerunning a command against the current `/data/ucy/South-ece-rtk-test` will use the replacement recording.

Run date: 2026-09-26. **Accuracy failed:** all recorded frames were processed,
but monocular VIO diverged and no loop constraint passed verification. This run
does not demonstrate the less-than-4 m ATE target.

## Pipeline and input

- Our native Rust ROS2 robot/backend nodes: f64 Basalt, stationary gyro-only
  initialization, JIST TensorRT, XFeat at measured VIO observations, trained
  LighterGlue TensorRT, bidirectional geometric verification, centralized SE(3) PGO.
- One robot (`ucy01`); monocular left IR, native calibrated **640 × 480**.
  Calibration: `configs/realsense/d455_1` for the same Jetson setup. IMU white-noise
  and bias multipliers: 0.75. Existing `configs/graco/aerial_vio.json` estimator
  parameters. No VIO or loop algorithm changes for this experiment.
- JIST threshold **0.8**, ten retained actual keyframes per sequence, five selected
  frames, full 5×5 descriptor refinement, existing temporal/covisibility exclusions.
- Dataset: `/data/ucy/South-ece-rtk-test`. Left PNG images are decoded losslessly.
  Gyro and acceleration use matching optical axes; acceleration is interpolated
  onto gyro timestamps without extrapolation. Camera offset **−9,590,295 ns** is
  applied once. Integer nanosecond timestamps are preserved and mapped into the
  common ROS playback epoch. GPS never enters the estimator.
- Initialization accepted the first stationary second: 199 IMU samples,
  31 buffered images released, zero skipped images.

## Full run

| Measurement | Result |
|---|---:|
| Recorded duration | 312.296 s |
| Frames ingested / processed | 9,364 / 9,364 |
| Wall time | 504.659 s |
| Keyframes / sequences | 1,338 / 124 |
| Candidate verification attempts / accepted loops | 78 / 0 |
| Dropped images / keyframes / request failures | 0 / 0 / 0 |
| Final graph keyframes / components | 1,338 / 1 |
| Estimated path length | 2,962.67 m |
| Maximum estimated speed | 55.26 m/s |

The complete trajectory has severe late drift. Without loop factors, corrected
and raw trajectories are effectively identical. Every emitted keyframe reached
the final graph; published optimization objectives were finite and non-increasing.
These transport/optimizer checks do not imply accurate tracking.

## Failure evidence

Of 78 selected candidate pairs, 76 had empty feature packets at both endpoints,
one had an empty packet at one endpoint, and one had 128 features at both.
The first 77 produced zero matches. The remaining pair produced 15 matches and
13 fundamental-matrix inliers, insufficient to reach the minimum 15 metric PnP
correspondences in either direction. Its production reason is
`pnp_geometric_support_failed`.

An empty feature packet means fewer than the required 128 measured VIO
observations; it does **not** establish that VIO had zero tracks. All selected
frames after **164.40 s** produced empty packets. Estimated speed rises from
about 1.82 m/s at 180 s to 7.53 m/s at 220 s and 53.32 m/s at 310 s.

The recording metadata reports `frame_emitter_mode=1` and
`frame_laser_power=150`. Strong projected IR dots are visible on indoor surfaces
by 200 s; the existing live VIO capture disables the emitter. This is a suspected
contributor requiring a controlled comparison, not a proven sole cause. Feature
shortage already appears outdoors before the indoor transition. Outdoor shadows,
the entrance, and indoor projected texture are retained in the image examples.

## Evaluation reference

The user confirmed that GPS is unreliable and requested that it not be used for
comparison. Earlier GPS-based comparisons are withdrawn. No ATE is reported for
this dataset. The monocular failure diagnosis rests on its runaway estimated
speed and trajectory. See the [stereo rerun](SOUTH_ECE_STEREO_VALIDATION.md) for
a reference-free comparison of tracking behavior.

## Artifacts and reproduction

All artifacts are in `results/ucy/south_ece_rtk_20260926/`:

- `assessment.json`: measured results, explicit failed accuracy status.
- `recorded_ir_examples.png`: recorded-image evidence. The earlier GPS plot is
  superseded by the stereo report's reference-free tracking figure.
- `playback.rrd`: verified Rerun recording with trajectories, sparse landmarks,
  selected pairs, and graph status.
- `robots/ucy01/`, `backend/`: original trajectories, sequence features, retrieval
  and verification logs, graph journal, final graph, timing and communication.
- `input_audit.json`, `input_manifest.json`, `experiment_manifest.json`: input
  checks, source/configuration/model hashes, and exact timestamp mapping.
- `source_at_launch/`, `runtime_install/`: frozen launch scripts and executables.

Prepare a **new output directory** to repeat the experiment:

```bash
.runtime/graco-venv/bin/python scripts/prepare_realsense_slam.py \
  --bag /data/ucy/South-ece-rtk-test --output /tmp/south-ece-slam-repeat
source scripts/source_multi_robot_ros2.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/visloc_ros/config/fastdds_replay.xml" \
  .runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py \
  /tmp/south-ece-slam-repeat/mission.json --domain-id 221
.runtime/graco-venv/bin/python scripts/visualize_multi_robot.py \
  /tmp/south-ece-slam-repeat/mission.json
```

The first launch used an unsupported FastDDS domain ID and exited before sensor
processing; its logs are retained under `attempts/domain234_startup_failure/`.
The complete run used domain 221. Preparation defaults to nominal 1× playback;
reliable delivery and processing acknowledgements slowed actual throughput.

Replay adapter checks passed: four timestamp/image/cache tests, three IMU sync
tests, and unchanged GRACO frame indices, IMU samples, and sampled image bytes.
The Rerun archive passed `rerun rrd verify`. No estimator parameters were tuned
against GNSS and no sensor/GPS hardware was opened by this offline replay.
