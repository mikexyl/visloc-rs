# One shared profile with ATE below 4 m

The acceptance target is **raw position ATE RMSE < 4 m on each of GRACO aerial
5, 6, 7, and 8, using one shared configuration**. Evaluate a complete bag with
one rigid SE(3) alignment and no scale fitting. Ground truth is used only for
evaluation. An aggregate mean below 4 m is insufficient; every sequence must
pass independently.

Common settings are monocular left-camera VIO, the supplied aerial
calibration, 800-pixel processing width, and the same ordered sensor packets.
Initial experiments keep both IMU noise multipliers at 0.75 and change only
precision/initialization. No loop correction or depth enters the raw accuracy
tests. Motion-wait initialization, when tested, reports its omitted prefix;
it never filters poses after seeing evaluation errors.

The previous f32 results do not satisfy this shared-profile requirement:
gyro-only startup fails aerial 5, while motion-gated startup gives aerial 6
7.928 m despite passing the other three sequences. Choosing a different mode
for aerial 6 is excluded from acceptance here.

## Experiment records

Artifacts are in `results/graco/uniform_ate4`. The saved executable and
`experiment_manifest.json` preserve the initial source/configuration hashes.
Each replay saves its effective calibration, estimator configuration,
initialization statistics, complete trajectory, inertial states, and Rerun
recording. Completed-run scores are distinguished from provisional prefixes.

These experiments initially added a native ROS2 precision selector so the
f64 profile could be tested. Following acceptance, VIO was made f64-only and
stationary gyro-only startup became the default. The old selector is removed;
see [the default-change regression](F64_DEFAULT_VALIDATION.md). Archived
experiment binaries/configurations retain their original settings.

## Full standalone results

One shared profile passes every sequence: **f64, `stationary` gyro-only
initialization, 0.75 IMU noise and bias multipliers**. The startup gate checks
the first stationary second, measures the gyro offset, and replays every
buffered frame. It neither averages gravity nor waits for motion in this mode.
No pre-initialization frames are omitted.

| Sequence | Estimated / input frames | Raw SE(3) ATE RMSE (m) |
| --- | ---: | ---: |
| Aerial 5 | 5926 / 5926 | 2.886619 |
| Aerial 6 | 6584 / 6584 | 2.261530 |
| Aerial 7 | 7886 / 7886 | 2.641899 |
| Aerial 8 | 5563 / 5563 | 0.963876 |

The independent scorer verifies all input frames, full ground-truth association,
identical calibration/configuration across the four sequences, and agreement
with the replay's own metric. The worst score is 2.887 m; all four satisfy the
strict 4 m threshold. Results are in `sweep_scores.json`.

The additional f64 `stationary-motion` aerial 6 ablation gives **1.457217 m**
over 6,438 poses, with 146 explicitly omitted startup frames. This beats the
previous f32 motion-mode result of 7.928259 m. However, extending the same f64
motion profile to aerial 5/7/8 exposes **catastrophic aerial 8 divergence**:
the first 576 estimated poses have 862.409 m rigid ATE. Their squared error
alone imposes a lower bound of **277.504 m** on the eventual full ATE, even
using all 5,563 input frames as the maximum possible denominator. A full
trajectory's best rigid-fit squared error cannot be smaller than the best
fit to this prefix.

The motion profile is therefore rejected, and the three extension runs were
stopped. Aerial 5/7 prefixes are incomplete and are not counted as passes.
`motion_profile_rejection.json` retains the measured prefix and conservative
bound. No sequence-specific mix of modes is used for acceptance.

The passing gyro-only profile still has a **14.175 m pointwise error peak**
on aerial 5 at 6.90 seconds, during startup. The 4 m requirement in this report
is full position ATE **RMSE**, not a guarantee that every pose is within 4 m.
The peak is included in the reported 2.886619 m ATE; no outliers are removed.

An independent full aerial 5 repeat gives **2.886619 m**, with a byte-identical
trajectory CSV across all 5,926 frames (`a05_repeat_parity.json`).

## Native ROS2 validation

The combined four-robot replay was stopped incomplete after approximately
2,400 frames per robot because replay throughput deteriorated to about
0.5 frames/s per robot. Preserved artifacts are under
`ros_f64_stationary_full`; `run_assessment.json` explicitly records that this
is not a completed mission or full-sequence accuracy result. Its earlier
2,200-frame prefixes had identical positions to the standalone runs.

Full native ROS2 accuracy validation completed under
`ros_f64_stationary_isolated`, with one robot/backend mission per localhost
DDS domain (228–231). All use the same native executables, estimator profile,
ordered sensor data, and calibration; loop inference is disabled. These runs
check raw VIO accuracy rather than inter-robot loop closure or joint PGO.
Normal localhost DDS is enabled for these replays; the preceding combined
run was sandboxed with shared-memory transport available and UDP blocked.
Both domain separation and transport environment changed, so this is not a
controlled diagnosis of the earlier replay slowdown.

**All four full native ROS2 runs pass.** Their position and quaternion values
are exactly identical to the standalone results across all **25,959 poses**,
after restoring the original sensor timestamps. Consequently, their ATE values
are the same as the standalone table. No startup frames were omitted; image
drops, keyframe drops, and communication request failures are zero on every
robot. Aerial 5 additionally reproduced its entire standalone trajectory in a
third full replay.

| Sequence | Native ROS2 frames | Graph keyframes | Raw ATE RMSE (m) | Wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| Aerial 5 | 5926 | 847 | 2.886619 | 783.3 |
| Aerial 6 | 6584 | 941 | 2.261530 | 854.4 |
| Aerial 7 | 7886 | 1123 | 2.641899 | 1047.6 |
| Aerial 8 | 5563 | 795 | 0.963876 | 726.6 |

Each backend drained its complete odometry graph and final solve, with finite,
non-increasing published objectives and no rejected solutions. These isolated
missions each have one component and no verified loop constraints. Nominal
replay rate was 4x with backpressure; measured effective rate was 0.376–0.385x.
These were concurrently executed accuracy checks, not a dedicated real-time
throughput benchmark.

`final_validation.json` records the strict shared-profile acceptance, full
frame/timestamp accounting, canonical calibration/configuration hashes, pose
parity, and each final graph revision. `finalize_validation.py` reproduces the
audit and fails if any sequence does not meet the 4 m threshold. Per-mission
`accuracy_assessment.json` records the same target in the native ROS summaries.
The passing worst-case ATE has **1.113 m margin** below the target.

![Shared profile raw ATE](../results/graco/uniform_ate4/uniform_ate4.png)

The native ROS build and the existing bounded-input/session-restart regression
also pass. The default f32 regression retains the expected 32 frames from 48
submitted images and reports all 16 deliberate queue drops. Logs are
`ros_scalar_build.log` and `ros_default_regression.log`; the accuracy missions
above have zero drops. Python syntax checks and `git diff --check` pass.

## Reproduction and interpretation

Use the same options on each of the four bags, with a fresh output directory:

```bash
.runtime/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/aerial-06-20m_full_ros2 \
  --output results/graco/my_uniform_ate4_a06 \
  --camera-mode mono \
  --imu-noise-scale .75 --imu-bias-scale .75
```

For native ROS2, prepare a fresh mission using its defaults; the aerial
calibration preparation uses 800 px and the same 0.75 multipliers. Set `loop_enabled: false`
in each robot configuration for the raw-VIO control, then use the documented
mission runner and evaluator. The experiment manifest saves the exact tested
executable/source hashes, and each run archives its effective configuration.

The accepted profile now runs without precision or startup flags: VIO is
f64-only and stationary gyro-only initialization is the default. No integration
formula, tracking frontend, camera calibration, or IMU noise setting was changed
by this default change. At the time of the comparison, both precision modes
supported the online initializer. Their estimator paths also differed in some
solver/marginalization routines, so these results
do not isolate floating-point precision as the only cause of different
behavior. In particular, the rejected f64 motion run prevents calling f64
uniformly superior across all initialization modes.
