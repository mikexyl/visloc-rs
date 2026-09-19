# GRACO aerial VIO with Rerun

Run `aerial-08-25m_ros2` directly from its SQLite bag through visloc's Rust
`basalt_stream_vio`. No ROS installation or image extraction is required.
The source bag and calibration YAMLs are opened read-only.

From the repository root, install/build and run:

```bash
bash scripts/run_local_graco_vio.sh \
  --bag /data/graco/aerial-08-25m_ros2 \
  --camera-mode mono \
  --output target/graco_vio/aerial-08-25m_run

target/graco-venv/bin/rerun target/graco_vio/aerial-08-25m_run/playback.rrd
```

Choose a fresh output directory each time. The setup script creates
`target/graco-venv`, installs `scripts/requirements-graco.txt`, and builds the
release estimator with AVX2/FMA and LM workspace reuse.

The command above uses **left-camera + IMU VIO**, retaining the supplied
intrinsics and camera/IMU extrinsics. The right image is previewed but excluded
from estimation. This avoids the excessive scale seen with stereo constraints
on this sequence; no baseline or trajectory scale adjustment is applied.
For stereo comparisons, explicitly select `--camera-mode stereo` and optionally
add `--stereo-refinement configs/graco/aerial-08-25m_stereo_refinement.json`.
The generic replay CLI retains stereo as its default when the mode is omitted.

The completed left-camera/IMU run at
`target/graco_vio/aerial-08-25m_mono_20260920` processed all **5,563 frames**
(278.10 seconds). Full-run SE(3) ATE RMSE is **1.058 m** and diagnostic excess
scale is **0.928%**, compared with **4.934 m / 8.995%** for the corrected stereo
run. No ground-truth alignment or scale correction enters estimation. The
trajectory still has ordinary odometry drift; this is a working monocular
alternative, not a repair of the unresolved stereo calibration.

Reproduce this exact estimator mode after setup:

```bash
target/graco-venv/bin/python scripts/run_graco_vio.py \
  --camera-mode mono --output target/graco_vio/aerial-08-25m_mono_repeat
```

For live visualization, start the viewer first in a separate terminal:

```bash
target/graco-venv/bin/rerun --bind 127.0.0.1 --port 9878 --memory-limit 2GiB
```

Then add `--rerun-connect rerun+http://127.0.0.1:9878/proxy` to the replay
command. Once dependencies and the binary are built, use
`target/graco-venv/bin/python scripts/run_graco_vio.py` with the same arguments
to skip setup/build. A recording is saved even when a viewer is connected.
Use `--max-frames 1000` for a takeoff check; the first 100 frames are stationary
and do **not** validate metric scale during flight.

The default replay processes every stereo pair at 800 × 550 pixels. `--width`
changes image size with matching intrinsics. The display contains undistorted
left/right images, blue VIO poses, orange active landmarks, a gray reference
trajectory, observation counts, and estimator processing time. The timeline
uses sensor time, independent of replay processing speed.

## Sensor and coordinate conventions

- `/camera_left/image_raw` and `/camera_right/image_raw`: paired at exact bag
  timestamps; both header timestamps are checked before delivery.
- `/gnss/imu`: gyro in rad/s and acceleration in m/s², delivered once per sample
  in `(previous_camera_time, camera_time]`. The first sample at or after the
  first image provides the estimator's gravity initialization.
- The published aerial calibration is expected at
  `/data/graco/aerial-calibration-20251121T084428Z-1-001/aerial-calibration`;
  override with `--calibration-dir`.
- `T_Imu_cam*` maps camera coordinates into IMU coordinates and is already in
  Basalt's `T_imu_cam` direction. The baseline is 0.43395255 m.
- Area downsampling includes the pixel-center correction in the intrinsics;
  radial-tangential distortion is removed. Camera axes are retained. The
  undistorted pinhole camera is represented exactly by Double Sphere
  `xi=alpha=0`.
- IMU rate/noise come from the aerial `imu.yaml`. No camera time shift is given
  by this calibration, so the replay uses synchronized header timestamps with
  zero additional shift.
- `/gnss/ground_truth` and the IMU orientation field never enter VIO. For live
  display the reference is transformed to the initial estimated body pose.
  Final evaluation separately uses nearest-time association within 10 ms and
  full-run rigid SE(3) alignment. Sim(3) scale is only an evaluation diagnostic.

## Calibration audit and sequence-specific correction

The published stereo and camera/IMU transforms agree algebraically to under
`1e-8`; the bag's older `CameraInfo` intrinsics have larger epipolar residuals
than the supplied aerial YAMLs. Nevertheless, the YAML stereo rotation leaves
systematic image residuals on this sequence. A 0.5579° correction of the right
camera rotation reduces held-out median epipolar errors from 2.62/4.18 pixels
to 0.22/0.37 pixels at the original 1600 × 1100 resolution. Held-out 95th
percentiles are 0.68/0.91 pixels.

`configs/graco/aerial-08-25m_stereo_refinement.json` records this **sequence-specific,
image-derived correction**, its source calibration hashes, and validation
measurements. This is not a replacement official calibration. The correction
preserves both optical centers, baseline, camera intrinsics/distortion, and
the left camera/IMU transform. Loading it with different calibration hashes
fails rather than silently applying it. It is applied only when explicitly
selected with `--stereo-refinement`.

To reproduce the fit without reading ground truth:

```bash
target/graco-venv/bin/python scripts/refine_graco_stereo.py \
  --bag /data/graco/aerial-08-25m_ros2 \
  --output target/graco_vio/new_stereo_refinement.json
```

The fit uses SIFT correspondences and robust epipolar residuals from frames
200, 400, 800, and 1600; frames 3200 and 5000 are held out. Thus it is an
offline calibration pass over this bag, not an online calibration algorithm
or a calibration-independent benchmark. Validate a new fit for other bags.

The GRACO config also increases `config.optical_flow_levels` from 3 to 5.
At takeoff, the wide stereo baseline and nearby ground defeat the smaller
EuRoC tracking pyramid and produce poor scale initialization. On the first
1,000 frames (49.95 s), SE(3) position RMSE was 4.694 m with the initial EuRoC
config, 1.257 m with the larger pyramid, and 0.422 m with both the larger
pyramid and stereo rotation correction. These are prefix checks, not full-run
accuracy claims.

The full 5,563-frame corrected run **still has excess scale**: SE(3) ATE RMSE
is 4.934 m, and the diagnostic Sim(3) scale is 0.91748, meaning the estimate
is approximately 9.0% too large. The stereo rotation correction improves
epipolar geometry and takeoff behavior but does not resolve this metric error.
No Sim(3) correction is applied to the estimator output or Rerun trajectory.

Controlled comparisons over the same first 2,000 frames (99.95 s):

| Replay | SE(3) ATE RMSE | Excess scale |
| --- | ---: | ---: |
| 800-pixel width, stereo rotation correction | 2.085 m | 7.53% |
| 1600-pixel width, more KLT iterations | 1.973 m | 7.11% |
| 800-pixel width, stereo rectification | 2.129 m | 7.70% |
| Additional camera/IMU rotation fit | 2.049 m | 7.38% |
| Double-precision estimator arithmetic | 2.200 m | 7.98% |
| Image-only varying right-camera rotation/warp | 2.177 m | 7.79% |
| Left camera + IMU, published calibration | **0.221 m** | **0.44%** |

The stereo alternatives did not fix scale and are not promoted to defaults.
`--rectify` is available for further diagnosis; `--preview-width` controls
Rerun image size independently of processing resolution.
`--scalar-mode f64` enables experimental double-precision arithmetic; the
default remains the upstream-compatible `f32` path.

Starting separately at frame 300 (the first hover), then processing through
frame 1999, still gives 7.66% excess scale and 2.075 m SE(3) ATE. This uses a
different evaluation interval, but shows that bypassing the poorly conditioned
takeoff initialization does not resolve the error. A normal replay with added
bias logging exactly reproduces the existing 2,000-frame result. The reference
trajectory also agrees with the bag's GPS distances to about 0.3%; changing
the reference scale is not justified.

Additional checks found approximately 2 ms camera/IMU timing offset and
1–2 ms inter-camera timing offset. The bag's alternate `CameraInfo`
intrinsics failed held-out epipolar validation even after rotation fitting.
An image/gyro camera/IMU rotation fit was consistent between separate halves
of the sequence but did not materially change scale. The LiDAR depth check
was inconclusive across vegetation, occlusions, and different ground levels;
it does not establish a replacement baseline. The remaining metric-scale
error in stereo mode is unresolved. The left-camera/IMU comparison removes
the stereo baseline entirely while retaining metric inertial measurements;
its substantially smaller scale error implicates the stereo constraints.
It does not establish which individual stereo calibration parameter is wrong.

## Outputs and checks

Each run saves `trajectory.csv`, `trajectory.tum`, `playback.rrd`, the exact
Basalt calibration/config, `calibration_report.json`, the selected refinement,
`ground_truth.csv`, `evaluation.json`, `summary.json`, and `vio_stderr.log`.
`summary.json` reports processed frames and timing. Ground truth is reference
output only. Active landmarks are logged into Rerun; no persistent SLAM map or
loop closure is performed.
`inertial_state.csv` records world velocity (m/s), gyro bias (rad/s), and
accelerometer bias (m/s²) for diagnosing initialization and subsequent drift.

```bash
target/graco-venv/bin/python -m unittest discover -s tests -p test_graco_calibration.py -v
target/graco-venv/bin/rerun rrd verify target/graco_vio/aerial-08-25m_run/playback.rrd
```

The calibration tests check resized pixel coordinates against independent
camera projection, transform direction/optical-center preservation, source
hash rejection, and recovery of a known synthetic stereo rotation.
