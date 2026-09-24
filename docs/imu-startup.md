# Stationary IMU initialization

The shared Rust `visloc_basalt::startup::StationaryStartup` gate improves
stationary-start initialization without changing Basalt's preintegration,
optimizer, scalar precision, calibrated noise values, or tracking frontend.
It is used by both `basalt_stream_vio` and the native rclrs robot node.

## Why this change

The aerial 6 investigation isolated a coupled startup failure: the original
single accelerometer sample / zero gyro bias initialization developed a large
gravity-direction error while still stationary. Changing f32 to f64 or widening
noise helped, but did not eliminate the problem. Correcting a gyro offset
measured from the first stationary second substantially improved the full run.
That correction alone destabilized aerial 5 in f32, so the f32 result must not
be promoted as a general fix. A stationary monocular optimizer also lacks translational
excitation; the motion-gated mode addresses that part of startup.

This is a stationary-start initializer, not a moving-start initializer or a
replacement for full observability-aware dynamic initialization. It assumes a
textured static scene and a stationary first second. Scene motion, weak
texture, or movement during that interval can prevent initialization. The
optional motion-gated mode additionally requires subsequent takeoff-like
excitation; slow motion may not trigger it.
It does not estimate three-axis accelerometer bias from one orientation.

## Modes

Both `scripts/run_graco_vio.py` and `scripts/prepare_multi_robot_graco.py`
accept `--imu-startup MODE`. The stream executable, GRACO replay, and native
ROS2 robot default to **`stationary` gyro-only initialization**. VIO always
runs in **f64**; there is no estimator precision selector.
The initial f32 experiments found that `stationary` improves native ROS2
aerial 6 raw ATE from 13.170 m to 1.652 m, but fails on aerial 5.
In f32, `stationary-motion` completes aerial 5/7/8 at 1.701/2.584/0.670 m,
but aerial 6 reaches 7.928 m. These are not universal f32 profiles.
See [the shared-profile follow-up](../ros2/UNIFORM_ATE4_VALIDATION.md) for
the f64 comparison against the requirement of ATE below 4 m on every sequence.
The passing shared profile is **f64 + `stationary` + 0.75 noise/bias scales**:
full standalone and native ROS2 runs give 2.887/2.262/2.642/0.964 m on aerial
5/6/7/8, with identical raw poses and no omitted frames or drops. This is an
RMSE target; aerial 5 still has a brief 14.175 m startup error peak.
The f64 `stationary-motion` profile fails on aerial 8 and is not recommended
as a shared profile despite its good aerial 6 result.

| Mode | Behavior |
| --- | --- |
| `legacy` | Original immediate initialization; no startup gate. |
| `stationary` | Default gyro-only correction; replays the buffered first second. |
| `stationary-gravity` | Experimental gyro correction plus averaged initial gravity; replays the buffer. |
| `stationary-motion` | Gyro correction and averaged gravity, followed by a motion wait. |

Equivalent native ROS robot configuration:

```json
"imu_startup": {
  "average_gravity": false,
  "wait_for_motion": false
}
```

An absent `imu_startup` or an empty object selects the default gyro-only mode.
Explicit `"imu_startup": null` selects legacy behavior. The remaining
`StationaryStartupConfig` fields can override thresholds; unknown fields are
rejected. Replay acknowledges ingestion while this default gate is buffering,
including when the configuration field is omitted.

The old `--scalar-mode` option and ROS `scalar_mode` field have been removed;
delete them from launch commands and robot configurations. Both are rejected
rather than silently accepting a precision setting. The core estimator also
has no precision field. Low-level upstream f32 arithmetic reference tests are
retained separately from the fixed f64 VIO path. Startup statistics use f64.

## Sensor handling

1. A separate KLT stream checks visual stationarity while original sensor
   packets are held. The VIO estimator has not processed any frames yet.
2. Calibrated IMU samples in the half-open first one-second interval produce
   the mean gyro offset and accelerometer vector. Existing affine calibration
   is respected. Statistics and the VIO estimator use f64.
3. The accepted gyro offset rebases the session's static gyro correction;
   dynamic gyro bias estimates residual bias about zero. The source calibration
   file is unchanged. The default keeps the original single-sample gravity
   initialization; averaged acceleration is used only by the optional gravity
   and motion modes.
4. In motion mode, wait until persistent visual tracks move at least three
   pixels (90th percentile) and a post-window IMU acceleration-norm deviation
   exceeds 0.2 m/s². The latter is latched. These are configurable heuristics,
   not a complete observability test.
5. By default, start VIO by replaying every buffered frame. Only motion mode
   starts with the last waiting frame and the triggering frame, omitting
   earlier pre-initialization frames and counting them separately from
   transport/queue drops. There are no fabricated poses for that interval.
   Retained packets and later inputs preserve their original IDs, timestamps,
   images, and IMU ordering.

While waiting, an aging visual reference can be refreshed only if at least 30
surviving tracks still verify displacement below the motion threshold. Tracking
failure does not count as motion. Calibration statistics never incorporate
later moving samples. Startup uses sensors only; ground truth is confined to
evaluation and visualization.

Default calibration-window gates are: at least 50 IMU samples, maximum 50 ms
IMU gaps, at least 30 common image tracks, at least 50% retention of the mature
track pool, image p90 displacement ≤0.75 px, gyro standard-deviation norm
≤0.005 rad/s, acceleration standard-deviation norm ≤0.15 m/s², gyro mean norm
≤0.03 rad/s, and acceleration norm within 0.5 m/s² of 9.81. The first temporal
match establishes the mature pool because newly detected pyramid-border
features may not be trackable. Absolute support is always required.

Buffers are bounded by 64 frames and 4096 IMU samples. The motion wait keeps
only the latest waiting frame. The sensor-time deadline is 30 seconds. Invalid
ordering, nonfinite input, missing IMU coverage, movement during calibration,
insufficient texture, overflow, timeout, or early EOF produces an explicit
error. There is no automatic fallback to legacy initialization.

`imu_startup.json` records thresholds, sample count, accepted statistics,
calibration/release timestamps, and omitted-frame count. Standalone replay
also reports ingested, estimated, and initialization-skipped frame counts.
ROS status publishes `initializing` and `input_timestamp_ns`; replay uses this
ingestion acknowledgment during startup instead of waiting for a pose that
cannot yet exist. ROS callbacks still only enqueue work.

## Reference methods considered

- [GTSAM CombinedImuFactor](https://borglab.github.io/gtsam/combinedimufactor/)
  accounts for bias evolution and correlations between integrated motion and
  bias uncertainty. This is useful for a future covariance-model comparison;
  it does not independently make a stationary monocular startup observable.
- [OpenVINS StaticInitializer](https://docs.openvins.com/classov__init_1_1StaticInitializer.html)
  uses a stationary interval for gravity/bias initialization and normally waits
  for motion. Its caution about immediate startup without zero-velocity
  updates informed the separate motion gate here. No OpenVINS code is copied.
- [Eckenhoff, Geneva, Huang: Closed-form Preintegration Methods for Graph-based
  Visual-Inertial Navigation](https://arxiv.org/abs/1805.02774) develops analytical
  integration under piecewise measurement/acceleration assumptions. This is
  different from fitting a continuous-time spline trajectory. Neither a
  closed-form integrator nor a spline backend is introduced by this change.

The existing SO(3) integrator already has bias Jacobians and covariance
propagation. A new independent rotating-body-force closed-form test exercises
both f32/f64 paths at the GRACO 125 Hz IMU rate, including the held partial
endpoint. Both pass within 2e-6 m in position, 2e-6 m/s in velocity, and
2e-7 rad in rotation over a 50 ms interval.
This checks deterministic integration, not all covariance consistency or
long-run numerical behavior. The measured startup failure gives stronger
evidence for this targeted fix than for replacing the integration method.

See [the experiment report](../ros2/IMU_STARTUP_VALIDATION.md) for completed
runs, failures, matched comparisons, and limitations.
