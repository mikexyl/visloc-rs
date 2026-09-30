# South-ece: cuVSLAM versus our stereo VIO

The dataset path was replaced with a new recording on 2026-09-27. This report describes the archived 2026-09-26 recording; rerunning a command against the current `/data/ucy/South-ece-rtk-test` will use the replacement recording.

The completed cuVSLAM stereo-inertial run **diverges late in the recording**. Our f64 stereo VIO maintains continuous output with bounded estimated speed on the same inputs. This is a stability comparison, not an accuracy measurement: GPS is excluded, neither estimate is ground truth, and no ATE or fitted scale is reported.

## Completed experiment

Run date: 2026-09-26. Recording: `/data/ucy/South-ece-rtk-test`.

| Measurement | Our f64 stereo VIO | cuVSLAM 17.0.0 stereo-inertial |
|---|---:|---:|
| Input stereo pairs | 9,360 | 9,360 |
| Recording duration | 312.296 s | 312.296 s |
| Returned poses | 9,360 | 9,359 |
| Invalid pose responses | 0 | 1, at 284.551 s |
| Maximum estimated speed before first cuVSLAM failure | 2.24 m/s | 284.85 m/s |
| Estimated distance over the same initial 284.518 s | 329.89 m | 1,424.51 m |
| Largest adjacent pose translation before failure | 0.089 m | 9.499 m |

A returned, finite pose does not establish successful visual tracking. cuVSLAM's large trajectory drift precedes its explicit invalid response by many seconds. At frame 8527 it returns `world_from_rig=None` with timestamp zero. It resumes returning poses at frame 8528, but the following segment also deteriorates: its final estimated speed is 91.00 m/s. No application reset, relocalization, interpolation over the missing pose, or stitching of segments was performed.

The two trajectories were placed in a common coordinate system using **only the first common body pose**. This accounts for the arbitrary world orientation; it does not fit later trajectory positions or scale. Their positional disagreement is 0.32 m at 30 s, 1.99 m at 120 s, 3.44 m at 180 s, 20.09 m at 220 s, and 97.02 m at 260 s. These values are estimator disagreement, **not error against a reference**. The initial uninterrupted cuVSLAM segment is about 974 m from our estimate by 284.518 s.

Our full-run path length is 347.36 m. Its continuous output does not establish its absolute accuracy or validate an ATE target. See [our stereo baseline assessment](SOUTH_ECE_STEREO_VALIDATION.md).

## Configuration and input checks

- Used NVIDIA's official [cuVSLAM v17.0.0 release](https://github.com/nvidia-isaac/cuVSLAM/releases/tag/v17.0.0), commit `57f42cc92d93eef47577d726789852a350b2a369`, through its Python 3.10 CUDA 12 wheel. Its [Python API](https://nvidia-isaac.github.io/cuVSLAM/python/api.html) is called directly from an isolated environment.
- Host: NVIDIA RTX 4070 Laptop GPU, 8,188 MiB, driver 580.178.04; CUDA libraries from `/usr/local/cuda-12.9/lib64`.
- NVIDIA `OdometryMode.Inertial`, GPU enabled, synchronous SBA (`async_sba=False`, as in the upstream offline examples). Motion model enabled, denoising disabled, other tuning at defaults. No SLAM object or loop closure.
- Our comparison uses raw Basalt f64 VIO, stationary gyro-only initialization and 0.75 IMU noise/bias multipliers. The existing JIST/XFeat/LighterGlue run accepted no loops; PGO therefore did not change this trajectory.
- Reused the completed baseline's exact 9,360 timestamp-paired D455 left/right IR images at 640×480. No nearest-neighbor stereo pairing, frame subsampling, RGB, depth, GPS, or ground truth.
- Reused both calibrated intrinsics, radial/tangential undistortion, both camera-to-IMU transforms, and the 95.0955 mm baseline. cuVSLAM's rig frame is the recorded IMU/body frame: `rig_from_camera=T_imu_cam`, `rig_from_imu=identity`. Camera extrinsics are not inverted.
- Images are individually undistorted with their original pinhole intrinsics. They are not epipolar-rectified, so `rectified_stereo_camera=False`.
- The initial OpenCV 5 installation produced different remapped pixels. The final run pins OpenCV 4.14.0.94. An independent pass with system OpenCV 4.5.4 reproduced the Rust adapter's preprocessing operations for **every stereo pair** and matched the final run's entire undistorted image stream byte-for-byte. SHA-256: `4a47b1c89b774160c575293782f75f0e54871d0d236f9ef1cc8b1a5bf5f2e6fd`.
- Reused the existing gyro samples with accelerometer values interpolated at gyro timestamps. Camera/IMU clock correction remains the baseline's −9,590,295 ns camera offset, already applied in the cache. The same integer timestamp mapping places the first camera frame at ROS time 1 s.
- Submitted 62,331 IMU samples through the last image, in chronological order before each corresponding `track` call. The 11 cached samples after the final image were not submitted. No future IMU samples are registered before an earlier image. All 9,360 input image timestamps exactly match our baseline's ordered output timestamps.
- cuVSLAM uses its own initializer and bias estimates; it receives no Basalt pose or estimated startup bias. IMU noise and random-walk settings come from the same effective calibration as our VIO. Its first exported gravity vector appears at 43.118 s. The `warming_up` flag was false throughout; it is not used as proof that inertial initialization completed immediately.

The checks above address sensor-conversion errors; they do **not** isolate the underlying cause of cuVSLAM's late divergence. No cuVSLAM noise or tracking parameter sweep was performed.

## Timing and validation

The full Python batch run took 143.887 s, plus 0.248 s for tracker construction. Median / p95 time inside `track` was 1.428 / 42.404 ms. Our Rust VIO adapter's recorded processing events were 51.434 / 98.707 ms. The boundaries and CPU/GPU work differ: these are stage timings, not a controlled end-to-end speedup benchmark. In particular, do not compare Python batch wall time directly to paced ROS replay wall time. Diagnostic exports were enabled and an independent CPU pixel audit overlapped the tail of the cuVSLAM run.

Validation completed:

- Calibration-frame conversion and nanosecond IMU-ordering tests: 2 passed.
- Every final input frame processed; timestamps checked; invalid output recorded without assigning it a fabricated current-frame pose.
- Full-stream image parity audit passed, and configuration, source, dependency, wheel and input provenance were saved.
- Comparison PNG inspected; Rerun recording verified without errors; repository whitespace check passed.

A smoke run used 180 frames. The first full attempt stopped at frame 8535 because the runner asserted timestamp equality before checking whether cuVSLAM returned a pose. That attempt already exhibited severe divergence and is preserved at `results/ucy/south_ece_cuvslam_20260926`. The runner was corrected to log missing/stale poses and continue. The completed repeat uses unchanged estimator settings and the same image/IMU streams; its late trajectory and exact failure frame differ slightly. Results in this report refer only to the completed repeat.

## Reproduce and inspect

From the `visloc-rs` repository, using the installed isolated environment:

```bash
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .runtime/cuvslam-venv/bin/python scripts/run_cuvslam_realsense.py \
  results/ucy/south_ece_stereo_20260926/mission.json \
  --output results/ucy/south_ece_cuvslam_new_run
```

The output directory must be new. This processes cached recording data without opening a camera or GPS device. The NVIDIA package is optional and isolated from ordinary visloc/ROS builds.

Completed artifacts are in `results/ucy/south_ece_cuvslam_20260926_full/`:

- `comparison.png`, `comparison.pdf`: trajectory, height, and estimated-speed comparisons.
- `comparison.rrd`: Rerun comparison with trajectory detail and synchronized sampled stereo images. The shared map stops cuVSLAM at its first invalid pose; subsequent output is stored separately under `post_failure_unregistered`.
- `comparison.json`, `trajectory.csv`, `trajectory_in_vio_initial_frame.csv`, `tracking.jsonl`: measurements, native trajectory, initial-pose-transformed output, and per-input diagnostics. The transformed CSV retains later output for inspection; it is not a stitched, validated global trajectory.
- `config.json`, calibration, `manifest.json`, `requirements.lock.txt`, frozen runner/input reader, `pixel_audit.json`, and `run.log`: reproducibility records.
- `assess.py`, `plot.py`, `visualize.py`, `audit_pixels.py`: scripts used for the saved analysis and visualizations.

Open the Rerun recording with:

```bash
.runtime/graco-venv/bin/rerun results/ucy/south_ece_cuvslam_20260926_full/comparison.rrd
```
