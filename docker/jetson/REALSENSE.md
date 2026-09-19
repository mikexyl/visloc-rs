# Live D455F stereo-inertial VIO

The camera input bridge captures infrared 1/2 and gyro/accelerometer data with
librealsense, then sends synchronized sensor packets directly to the Rust
`basalt_stream_vio` executable. VIO runs continuously; no image dataset is
written or replayed. The Python bridge handles camera acquisition and timestamp
synchronization only. Estimation remains in Rust.

## Build on the Jetson

```bash
cd ~/workspaces/visloc-rs
bash docker/jetson/build_realsense.sh
```

Image: `visloc-rs:realsense`. The multi-stage build compiles librealsense v2.58.4
with `FORCE_RSUSB_BACKEND=ON`. This supports the D455 IMU without IIO kernel
drivers or changes to the Jetson host kernel. The camera’s existing firmware is
left unchanged. CUDA is not used.

## Run

The camera must be free: stop any other camera consumer first. On this board,
`dpvo-online-robot0` was explicitly stopped by the user for visloc testing.

```bash
cd ~/workspaces/visloc-rs
# 30-second live check, after two seconds of exposure warmup:
bash docker/jetson/run_realsense.sh results/d455-check --duration 30
# Continuous run (Ctrl-C stops cleanly):
bash docker/jetson/run_realsense.sh results/d455-live
```

Default serial: `425122300073`. Override with `--serial` only if the supplied
calibration belongs to that camera. The default image stream is 640x480 Y8 at
30 Hz. `--vio-fps 5` caps processed image pairs to keep CPU load bounded while
retaining every IMU sample between processed frames. Images superseded during
processing are skipped; IMU buffer overrun or gaps over 50 ms terminate the run
with an error. This frame-rate cap is configurable and does not promise
real-time performance for every scene or motion.

`--duration 0` (default) runs until stopped. Each run requires a fresh output
directory. Use `docker stop -t 40 visloc-realsense` for a container running in
another terminal. The launch script does not stop other applications or alter
their restart policies. It passes USB devices to the container without
`--privileged` and enables outbound networking for the Rerun SDK. USB acquisition runs as root;
output files consequently have root ownership on the Jetson.

## Calibration

The source files supplied in `/home/mikexyl/Downloads/d455_1_calib` are preserved
under `configs/realsense/d455_1/`:

- `camchain-imucam.yaml`: stereo/IMU calibration, infrared cam0/cam1; RGB ignored.
- `imu.yaml`: calibrated IMU noise and random-walk values at 200 Hz. The separate
  `_alan` file has different accelerometer noise values and is not used.

The bridge inverts Kalibr `T_cam_imu` into Basalt `T_imu_cam` (camera to IMU).
It undistorts each infrared image using its supplied radtan coefficients,
keeping the calibrated focal length, principal point, resolution, and camera
axes. The resulting pinhole model is represented exactly by Basalt’s Double
Sphere model with `xi=alpha=0`. The stereo baseline is approximately 95.095 mm.

Kalibr’s convention is `t_imu = t_camera + timeshift_cam_imu`. The two infrared
offsets differ by about 3 microseconds, so their mean, **-9,590,295 ns**, is
applied once to stereo timestamps. The generated Basalt calibration has zero
additional time offset. Gyro/accel samples use the device hardware clock, in
rad/s and m/s². Acceleration is linearly interpolated onto each gyro timestamp,
without extrapolation. The bridge checks that SDK gyro and accelerometer axes
agree and logs the SDK extrinsics for audit. It does not import EuRoC IMU biases.
The supplied calibration assumes the same SDK optical IMU axes used by the ROS
camera IMU stream during calibration.

## Outputs and inspection

Each run writes:

- `trajectory.tum`, `trajectory.csv`: poses flushed after each estimator update.
- `status.json`: latest pose, visual observations, IMU sample count, processing
  time, sensor lag, acceleration/gyro norms.
- `live.jsonl`: per-frame status history.
- `cam0_latest.jpg`, `cam1_latest.jpg`: bounded-size undistorted previews.
- `device.json`, `basalt_calibration.json`, `calibration_report.json`: device,
  calibration and timestamp provenance.
- `capture_summary.json`, `summary.json`: shutdown/processing summaries.

No ground truth is used. A stationary live check verifies acquisition, clock
synchronization and estimator operation; motion accuracy requires a separate
controlled trajectory test. These scripts do not command any robot motion.

## Verification

```bash
python3 -m unittest discover -s tests -p test_realsense_sync.py
```

The unit checks cover IMU interval boundaries, interpolation, missing-data
rejection, calibration transform inversion, and time-offset sign. The Rust
streaming adapter was also fed 80 EuRoC frames and reproduced the batch
trajectory within 1e-9 on the development machine.

## Verified deployment (2026-09-16)

The USB backend exposes the D455F Motion Module on this Jetson. A 30-second
live test processed 151 stereo frames and delivered 6,011 synchronized IMU
samples to the Rust estimator, with no synchronization errors. Mean estimator
time was 96.3 ms (p95 108.2 ms); median reported sensor-to-pose lag was 126.3 ms.
The stationary-scene run ended 7.0 mm from its starting estimate, with a maximum
excursion of 12.7 mm. This is not a ground-truth accuracy measurement.

Active deployment:

- Container: `visloc-realsense` (background, no automatic restart policy).
- Image: `visloc-rs:realsense`.
- Live output: `/home/mikexyl/workspaces/visloc-rs/results/d455-live`.
- Passed test output: `results/d455-live-check-2`.
- Previous camera consumer `dpvo-online-robot0` remains stopped.

```bash
# Inspect the current stream:
docker logs --tail 5 visloc-realsense
cat ~/workspaces/visloc-rs/results/d455-live/status.json
# Stop cleanly:
docker stop -t 40 visloc-realsense
# Start another background run into a fresh directory:
cd ~/workspaces/visloc-rs
VISLOC_DETACH=1 bash docker/jetson/run_realsense.sh
```

## Direct Rerun SDK connection

The live VIO process logs poses, calibrated stereo images, the trajectory, and
performance metrics directly to Rerun. No SSH exporter or local bridge runs.
`run_realsense.sh` sets `RERUN_CONNECT=rerun+http://192.168.0.243:9878/proxy`;
set `RERUN_CONNECT` in the launching shell to override the destination.
The receiver must listen on the LAN interface, not only localhost:

```bash
rerun --serve-web --bind 0.0.0.0 --port 9878 --web-viewer-port 9096
```

The local web viewer is at `http://127.0.0.1:9096` (connect it to
`rerun+http://127.0.0.1:9878/proxy`). The SDK is installed in the VIO container,
using a Python environment with NumPy-2-compatible OpenCV and SciPy. The board's
cached ARM64 wheels are in `native/rerun-wheels` and are used during image builds.
Current direct-stream output: `results/d455-rerun-live`.
