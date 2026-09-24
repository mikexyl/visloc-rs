# f64-only VIO and default stationary gyro initialization

The accepted aerial 5–8 profile is now the default precision/startup behavior:
the VIO estimator uses f64 and the sensor entry points enable stationary
gyro-only initialization. Camera mode and IMU noise multipliers are unchanged.

## Code and configuration contract

- Removed the precision field from `EstimatorConfig`, the ROS robot config,
  and both replay CLIs. The stream executable also rejects `--scalar-mode`.
- Removed estimator f32 branches for calibrated IMU input, initial orientation,
  propagation, camera bearings, triangulation, and marginal-data reduction.
  Window construction and solver calls use the existing f64 path.
- Retained low-level upstream arithmetic reference helpers/tests. They cannot
  select an f32 VIO estimator. Image/feature/TensorRT data types are unchanged.
- The stream executable, GRACO replay, mission preparer, and ROS robot default
  to the same verified one-second gyro-only startup gate. Averaged gravity and
  motion waiting are off. Every buffered frame is replayed.
- A missing ROS `imu_startup` field or `{}` enables the default gate. Explicit
  JSON `null` or `--imu-startup legacy` bypasses it. Replay acknowledges input
  ingestion while the default gate buffers, even when the field is omitted.
- Old `scalar_mode` JSON fields and `--scalar-mode` flags must be removed from
  active configurations/commands. They fail clearly rather than being ignored.
  Historical experiment artifacts are preserved.

## Validation

Artifacts are under `results/graco/f64_default_validation`. Basalt unit and
integration suites pass after removal of the f32 estimator branches: 422
passed, 0 failed, and 66 existing ignored tests. All four native ROS tests
pass, covering omitted startup, explicit null, rejection of both obsolete
precision values, and preservation of archived session IDs. Release builds
of the stream executable and native ROS nodes pass.

Full monocular aerial 5–8 default replays use the same 800 px calibration and
0.75 noise/bias multipliers as the accepted control. They omit precision and
startup flags. Aerial 5 additionally omits the stream executable's startup
argument, exercising its own default with real sensor packets. Every raw pose
is compared against `uniform_ate4/a0N_f64_stationary`, with no alignment or
tolerance hiding differences. All **25,959 frames** pass: complete trajectory
CSV files, startup reports, calibration, and VIO configuration are byte-identical
to the accepted pre-change controls. No initialization frames are omitted.

| Sequence | Frames | Raw ATE RMSE | Full trajectory parity |
| --- | ---: | ---: | --- |
| Aerial 5 | 5,926 | 2.886619 m | Byte-identical |
| Aerial 6 | 6,584 | 2.261530 m | Byte-identical |
| Aerial 7 | 7,886 | 2.641899 m | Byte-identical |
| Aerial 8 | 5,563 | 0.963876 m | Byte-identical |

ATE uses one rigid SE(3) alignment per full sequence, with no scale fitting.
All four remain below the strict 4 m RMSE target. This preserves the existing
accuracy result; it is not a new accuracy improvement or a bound on peak error.
The audit script, final JSON report, executable/source hashes, and focused
source diff are saved in the artifact directory.

The native ROS test uses real aerial 5/6 images and IMU, 400 frames each,
omitted `imu_startup` fields, and loop inference disabled. This exercises
configuration defaults, initialization acknowledgements, and complete graph
drain. Both robots' 400 poses are bit-identical to the accepted controls after
undoing the replay timestamp offsets. Startup statistics and calibration also
match exactly. There are no omitted frames, input/keyframe drops, or request
failures; all 116 keyframes reach the backend and its final solve completes.

The separate bounded-queue/restart fixture passes with 48 inputs, 32 retained
images, 16 intentional queue drops, and distinct archived/new session IDs.
It explicitly disables startup because its inputs intentionally contain
blank images and missing IMU. The real-image test above covers default startup.

All three CLIs reject both obsolete precision values. A real one-frame replay
with explicit `--imu-startup legacy` confirms that this override still works,
and the mission preparer writes explicit JSON null for that mode. Python
syntax checks and `git diff --check` pass.
