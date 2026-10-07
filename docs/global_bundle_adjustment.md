# Global bundle adjustment with GPS

The native Rust ROS2 pipeline defaults to **GTSAM global BA with horizontal GPS**.
The solver jointly optimizes body poses and shared landmarks. PGO remains an
explicit alternative. Basalt remains f64 with stationary gyro-only initialization;
backend corrections never feed back into raw VIO.

## Run and configure

Build the ROS messages and nodes together, then prepare a fresh mission:

```bash
CARGO_NET_OFFLINE=true bash scripts/build_multi_robot_ros2.sh
source scripts/source_multi_robot_ros2.bash
.runtime/graco-venv/bin/python scripts/prepare_realsense_slam.py \
  --bag /data/ucy/South-ece-rtk-test --output results/ucy/south_ece_gps_ba_new \
  --camera-mode stereo --trim-imu-boundaries
.runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py \
  results/ucy/south_ece_gps_ba_new/mission.json
```

RealSense preparation preserves calibrated timing and normalizes receiver GPS
from GGA/RMC UTC. PPK ground truth is never an estimator input. The nominal antenna
lever arm is specific to the calibrated South-ece rig; another rig must provide
`--gps-lever-arm X Y Z` in IMU/body metres. Missing GPS produces a visual-only
mission. `--no-gps` explicitly selects visual BA; add `--backend pose_graph` for
visual PGO. GRACO preparation defaults to visual BA because it has no receiver
GPS stream; its ground truth is never substituted for GPS.

Manual configuration uses the historical `pgo` object:

| Setting | Default | Alternative |
| --- | --- | --- |
| Backend `pgo.mode` | `global_bundle_adjustment` | `pose_graph` |
| Backend `pgo.gps.enabled` | `true` | `false` |
| Robot `bundle_adjustment_enabled` | `true` | Disable for PGO-only transport |
| Robot `gps.enabled` | `true` | `false` |

Explicit settings in existing configurations remain authoritative. Historical
snapshots without a backend-mode field are interpreted as PGO. Start a fresh
robot output directory when enabling BA on an archive without BA observations.
Live GPS needs normalized `GpsFix` metadata (`gps.normalized_input=true`) to pass
the default HDOP gate; bare `NavSatFix` has no HDOP. Neither node opens a GPS device.

GPS uses the same [quality gates and measurement conventions](gps_pose_graph.md)
as PGO: HDOP ≤ 1, 5 m assumed horizontal sigma for unknown covariance, switchable
loss with λ=9, and a one-sigma deadband. Missing or rejected fixes leave visual BA
available. A calibrated/declared antenna lever arm is required per robot.

## Solver and observation contract

Every actual VIO keyframe supplies calibrated pixel observations before sequence
selection or descriptor subsampling. Stereo contributes both cameras with their
own intrinsics and camera-to-body extrinsics. Non-pinhole models are rejected.
Track identity is `(robot, session, track ID)`; different IDs across loops or
robots are not merged. Verified relative-pose loops connect those submaps.

GTSAM optimizes body SE(3) poses and XYZ landmarks with fixed calibration. Raw VIO
relative measurements retain metric scale. Velocities, IMU biases, and time
offsets are not batch-optimized. Odometry/loop uncertainty defaults remain
0.15 m / 0.03 rad and 0.20 m / 0.04 rad; these are tuning parameters, not measured
independent covariances.

Admission uses cheap ray triangulation and metric VIO seeds, positive depth,
at least two body keyframes, sufficient parallax, and a reprojection gate.
Previously optimized points provide warm starts. **Separate fixed-pose landmark
pre-refinement has been removed.** The BA settings under `pgo.bundle_adjustment` are:

```json
{
  "pixel_sigma": 1.5,
  "huber_delta": 3.0,
  "initial_reprojection_gate_px": 3.0,
  "min_parallax_deg": 1.0,
  "min_keyframes": 2,
  "max_landmarks": 100000,
  "max_observations": 2000000
}
```

Capacity overruns fail explicitly. Obsolete `pre_refine_landmarks` and
`minimum_consensus_ratio` settings must be removed; unknown settings are rejected.

The solver uses point-first constrained COLAMD, sparse multifrontal Cholesky,
Huber pixel loss, and at most 20 LM iterations. Components are optimized
independently with one anchor each. GPS alignment releases east/north/yaw while
preserving root height and gravity. A verified connecting loop joins components;
GPS alone does not merge them. The backend schedules at most one solve per
second while dirty; solves never overlap. Only finite solutions with
non-increasing robust objective and valid depths replace the previous snapshot.

Robots journal reliable `/<robot>/slam/bundle_frames` and serve 16-frame pages at
`/<robot>/slam/bundle_history`. History recovery uses steady-time two-second
requests and three retries. The backend deduplicates immutable records and
persists observations in `graph.jsonl`. Final drain waits for observations for
every known keyframe; a final BA graph without admitted landmarks is a failure.
Snapshots include optimized points, component IDs, admission counts, solver
cost/timing and reprojection RMSE. Rerun displays the optimized points directly
in batched clouds and uses optimized camera poses; it does not re-refine BA points.

## Measurements and known limitations

Frozen South-ece backend comparison, 2026-10-06:

| Mode | Matched-keyframe ATE | Cold backend time |
| --- | ---: | ---: |
| Visual BA | 0.583 m | 6.29 s |
| GPS BA | **0.544 m** | 6.38 s |

Both runs used the same 1,015 keyframes, 5,874 landmarks, 45,931 pixel factors,
and one verified loop; GPS added 104 factors. Evaluation used 114 common
PPK-matched keyframes, rigid alignment without scale/time fitting, and a 20 ms
association limit. This is an archived backend comparison using left-camera
observations, not a full online stereo accuracy result. PPK has partial coverage,
shares receiver measurements with GPS, and the primary comparison retains an
uncorrected antenna offset. Local evidence:
`results/ucy/south_ece_gps_ba_20261006/evaluation/report.json`.

Five earlier paired timing runs showed pre-refinement slowing total BA from
6.50 s to 8.51 s, while matched ATE worsened from 0.583 m to 0.641 m. Its code,
configuration and benchmark harness were removed. A separate 360-image native
stereo replay verified bit-for-bit raw VIO parity with BA enabled/disabled,
zero sensor/keyframe drops, and 120 optimized landmarks. These limited checks do
not establish uniformly better full-trajectory accuracy.

The 2026-10-07 endpoint diagnosis confirms a visual return to the starting area,
but no accepted endpoint constraint. The last completed sequence scored 0.7947
against an early sequence, below the 0.8 gate; temporal/covisibility exclusions
did not block it. Forced verification found 42 matches and 30 fundamental
inliers, but only 10/7 forward/reverse PnP inliers versus the required 15. None of
all 25 frame pairs passed. The final 6.77 s formed only two retained keyframes
and no complete ten-keyframe block. Lowering retrieval alone is insufficient;
metric landmark support and incomplete final sequences remain limitations.
Evidence: `results/ucy/south_ece_endpoint_diagnosis_20261007/report.json`.

## Reproduction and focused validation

```bash
# Frozen backend comparison, with no VIO or ground-truth inputs:
cargo run --release --offline -p visloc-multi-robot --example global_ba_replay -- \
  MISSION_ROOT NEW_RESULT_DIRECTORY [MAX_KEYFRAMES]
# Prepare paired GPS-off/on missions reusing an existing sensor cache:
.runtime/graco-venv/bin/python scripts/prepare_gps_pose_graph.py \
  SOURCE_MISSION/mission.json NEW_PAIRED_RESULT --rate .25

cargo test --offline -p visloc-gtsam -p visloc-multi-robot
# After sourcing the ROS environment:
(cd ros2/visloc_ros && RUSTFLAGS='-C target-feature=+avx2,+fma' cargo test --release --offline)
ROS_DOMAIN_ID=225 ROS_LOCALHOST_ONLY=1 /usr/bin/python3 scripts/test_global_ba_ros2.py --gps
.runtime/graco-venv/bin/python -m unittest discover -s tests -p test_gps_preparation.py
.runtime/graco-venv/bin/python -m unittest discover -s tests -p test_online_landmark_map.py
```

`global_ba_replay` saves PGO/BA snapshots, configurations, input paths, keyframe
TUM trajectories, and diagnostics. Legacy display journals supply only measured
left-camera observations and are labeled `legacy_left_camera_only`. Generated
results remain local under ignored `results/`; reusable code lives outside it.
The focused suite covers solver geometry, robust loss, invalid-output atomicity,
identity/component handling, stereo calibration, GPS alignment, ROS history and
restart recovery, and display ownership. Run the ROS test without `--gps` when
changing the GPS-free path; native Jacobian tests are documented in
[gtsam_pose_graph.md](gtsam_pose_graph.md#tests-and-reproduction).
