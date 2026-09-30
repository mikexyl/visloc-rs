# South-ece: replacement recording, 27 September 2026

A subsequent evaluation against the user-supplied PPK reference is available in the [evo report](SOUTH_ECE_EVO_20260927.md): 0.781 m raw and 0.787 m corrected RMSE on 656 matched epochs, with partial coverage and reference-quality limitations. The assessment below records the original reference-free replay checks.

Completed the replacement recording with **our f64 stereo-inertial sequence/refinement SLAM**, using the same estimator binaries, calibration, configuration and TensorRT models as the previous run. This is the new recording at `/data/ucy/South-ece-rtk-test`, session `20260927-094857-961a3e`; the older VIO and cuVSLAM results describe different recorded data and are not accuracy comparisons for this run.

**GPS is excluded from estimation and evaluation. No ATE or absolute-accuracy claim is made.**

## Result

| Measurement | Result |
|---|---:|
| Processed stereo pairs | 7,182 |
| Processed recording duration | 239.716 s |
| Estimated raw path length | 299.03 m |
| Maximum estimated speed | 4.094 m/s |
| Final estimated speed | 0.0054 m/s |
| Maximum adjacent pose translation | 0.137 m |
| Keyframes / encoded sequences | 1015 / 91 |
| Verified loops / matching attempts | 1 / 17 |
| Final graph components | 1 |
| Runtime image / keyframe drops | 0 / 0 |
| Skipped initialization frames | 0 |
| Communication request failures | 0 |
| Replay wall time | 652.52 s |

Raw final position is [1.806, 3.637, -0.296] m; estimated height ranges from -3.17 to 0.45 m. These are estimator outputs, not position errors. The maximum difference between raw and final PGO-corrected positions is 0.751 m. PGO corrections do not feed back into raw VIO; without a trusted reference, their accuracy benefit is unverified.

The output contains every selected camera timestamp in order, all positions are finite, and all emitted keyframes and verified loops reached the final graph. Published robust optimization objectives were finite and non-increasing within each solve.

## Input and configuration

- f64 Basalt, stationary gyro-only initialization, 0.75 IMU noise/bias multipliers.
- Left and right D455 IR at 640×480, the existing `configs/realsense/d455_1` calibration, 95.0955 mm stereo baseline, and −9,590,295 ns camera offset.
- The Jetson serial, both camera-info records and gyro/accelerometer factory extrinsics match the previous recording. The estimator executables, complete effective calibration, VIO configuration, loop configuration and all three TensorRT engine hashes match the previous run.
- JIST threshold 0.8, ten retained VIO keyframes per sequence, five-view encoding and full 5×5 refinement, XFeat/LighterGlue verification, centralized sparse SE(3) PGO. Loop closure is enabled.
- All 7,191 metadata messages for each IR camera report emitter mode 0 and laser power 0: the emitter was disabled throughout this recording.
- 7,190 left and 7,185 right images produce 7,184 exact-timestamp pairs. 6 left and 1 right images have no exact counterpart.
- The first two paired exposures precede synchronized IMU coverage. They are explicitly excluded, leaving 7,182 pairs; their original images remain in the cache. No future IMU samples are fabricated or extrapolated. These preparation exclusions are distinct from runtime drops.
- The stationary initializer passed with 200 IMU samples and 31 buffered camera pairs. All buffered pairs were subsequently processed.

The preparation script now offers `--trim-imu-boundaries`; replay honors the mission's selected first/last timestamps. Default preparation remains strict. Six replay-source tests passed, including exact nanosecond interval selection for both mono and stereo, inclusive end bounds, and preservation of the read-only cache. Selection on the previous mission was also checked and still yields all 9,360 original pairs. Estimator code was unchanged.

## Tracking and loop verification

| Camera | Minimum observations | Median | Maximum | Zero-observation frames |
|---|---:|---:|---:|---:|
| Left | 7 | 128 | 179 | 0 |
| Right | 0 | 79 | 124 | 2 |

| Verification outcome | Attempts |
|---|---:|
| pnp_geometric_support_failed | 7 |
| verified | 1 |
| insufficient_matches | 9 |

Accepted constraints' matches, fundamental-matrix/PnP inliers, direction, coverage, reprojection error and timings are saved in `assessment.json` and the robot event journal. Rejected candidates are retained in diagnostics. Thresholds and the 128-real-feature contract were not relaxed.

## Artifacts and reproduction

Run root: `results/ucy/south_ece_stereo_20260927/`.

- `tracking_diagnostics.png` / `.pdf`: raw/corrected trajectories, height, raw speed and left/right track counts.
- `playback.rrd`: Rerun trajectories, sparse landmarks, selected frame pairs and loop edges.
- `assessment.json`, `trajectory_corrected.csv`, `robots/ucy01/`, `backend/`: measurements and complete estimation/verification/graph journals.
- `input_manifest.json`, `recording_metadata.yaml`: SHA-256 fingerprints for the six new MCAP files and recording metadata; filesystem dates alone are not used to identify the recording.
- `input_audit.json`, `camera_metadata_audit.json`, `experiment_manifest.json`, `source_at_launch/`, `runtime_install/`, `checks/`: exclusions, hardware metadata, configuration/model/source hashes, frozen executables and validation records.

```bash
.runtime/graco-venv/bin/python scripts/prepare_realsense_slam.py \
  --bag /data/ucy/South-ece-rtk-test \
  --output /tmp/south-ece-stereo-new-repeat --camera-mode stereo --trim-imu-boundaries
source scripts/source_multi_robot_ros2.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/visloc_ros/config/fastdds_replay.xml" \
  .runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py \
  /tmp/south-ece-stereo-new-repeat/mission.json --domain-id 222
```

Use a new output directory and an unused ROS domain. The frozen executables in the run archive identify the exact binaries used here. No physical camera or GPS device is opened for replay.
