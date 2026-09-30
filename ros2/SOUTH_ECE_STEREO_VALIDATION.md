# South-ece-rtk-test: stereo-inertial rerun

The dataset path was replaced with a new recording on 2026-09-27. This report describes the archived 2026-09-26 recording; rerunning a command against the current `/data/ucy/South-ece-rtk-test` will use the replacement recording.

Completed 2026-09-26 using **our native Rust ROS2 sequence/refinement SLAM**.
The stereo run avoids the monocular run's runaway speed. **GPS is excluded from
evaluation at the user's request. No ATE or metric-accuracy claim is made.**

## Configuration

- f64 Basalt, stationary gyro-only initialization, existing 0.75 IMU noise/bias
  multipliers, unchanged estimator parameters.
- Both D455 IR cameras at 640×480; calibrated baseline **95.095 mm**. Each camera
  uses its own intrinsics, distortion, and camera-to-IMU transform from
  `configs/realsense/d455_1`.
- JIST TensorRT threshold **0.8**, ten retained actual VIO keyframes, five-view
  sequence encoding and full 5×5 refinement, XFeat/LighterGlue, centralized PGO.
  Loops remain enabled; left-camera observations provide the verification
  packets, while both cameras contribute to VIO and metric landmarks.
- The camera offset remains −9,590,295 ns. Gyro/acceleration samples and the
  complete calibration are identical to the preceding monocular run.
- Exact timestamp pairing yields **9,360 pairs** from 9,364 left and 9,362 right
  images. Four unpaired left images and two unpaired right images are excluded
  before replay. No nearest-exposure substitution or mono fallback is used.

## Results without an external reference

| Measurement | Monocular | Stereo |
|---|---:|---:|
| Processed frames / pairs | 9,364 | 9,360 |
| Sensor duration | 312.296 s | 312.296 s |
| Estimated path length | 2,962.67 m | 347.36 m |
| Maximum estimated speed | 55.26 m/s | 2.24 m/s |
| Final estimated speed | 55.26 m/s | 0.0041 m/s |
| Accepted loops / verification attempts | 0 / 78 | 0 / 84 |

The stereo replay took 708.63 s. All 9,360 pairs were ingested and processed,
with zero skipped initialization frames, runtime image drops, keyframe drops,
or request failures. All 1,337 keyframes reached the final graph. The run encoded
127 sequences; the final graph has one component and revision 686.

Left observations have median 114, range 29–183. Right observations have median
58, range 0–122; seven frames have zero surviving right observations despite
having both input images. The raw and PGO-corrected positions agree to numerical
precision because no loop factors were accepted. The behavioral change therefore
comes from stereo-inertial VIO, not loop correction.

All **84 loop candidates** selected endpoints whose packets were both empty:
each endpoint had fewer than the required 128 measured left-camera observations.
They produced zero local matches and were rejected before geometric verification.
An empty packet does not mean VIO had zero tracks. The fixed 128-feature contract
was retained, without padding or relaxed verification thresholds.

Plausible speed and a bounded trajectory establish that the monocular runaway
did not recur; they do not establish positional accuracy. Estimated final Z is
−9.14 m and the minimum is −11.95 m. Without a trusted reference, actual elevation
change cannot be separated from vertical drift. The less-than-4 m ATE target
remains unverified for this dataset.

## Implementation and checks

The ROS robot accepts `camera_mode: "stereo"` plus
`/<robot>/camera_right/image`. Its bounded queue pairs matching timestamps in
either callback order and waits for IMU coverage. Each processed frame records
left/right observations in `tracking.jsonl`. Mono remains the default.

Eight Rust unit tests, five replay-source tests, and three IMU synchronization
tests passed. A 120-pair native ROS2 smoke run had nonzero right-camera tracks in
every frame. A separate 120-frame mono replay reproduced the original mono pose
bits exactly. The full stereo output was checked against the complete timestamp
intersection of the recorded cameras, and all published robust optimization
objectives were finite and non-increasing.

## Artifacts

Root: `results/ucy/south_ece_stereo_20260926/`.

- `assessment.json`: reference-free result and tracking/verification counts.
- `stereo_tracking_comparison.png`: trajectories, speeds, heights, and tracks;
  no GPS overlay or GPS-derived alignment.
- `playback.rrd`: verified Rerun recording of raw/corrected paths, sparse
  landmarks, selected loop pairs, and graph status.
- `robots/ucy01/`, `backend/`: trajectories, tracking counts, frozen features,
  retrieval decisions, verification outcomes, timing and communication journals.
- `input_audit.json`, `experiment_manifest.json`, `source_at_launch/`,
  `runtime_install/`, `checks/`: pairing audit, calibration/model/source hashes,
  exact executables, and regression results.

Reproduce with a new output directory:

```bash
.runtime/graco-venv/bin/python scripts/prepare_realsense_slam.py \
  --bag /data/ucy/South-ece-rtk-test \
  --output /tmp/south-ece-stereo-repeat --camera-mode stereo
source scripts/source_multi_robot_ros2.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/visloc_ros/config/fastdds_replay.xml" \
  .runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py \
  /tmp/south-ece-stereo-repeat/mission.json --domain-id 221
```
