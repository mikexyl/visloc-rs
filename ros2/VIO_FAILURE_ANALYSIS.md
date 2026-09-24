# GRACO aerial 6/7 raw VIO failure investigation

This is a historical investigation. Its precision-selector commands require
the archived executable and matching runner revision. Current VIO is f64-only
with stationary gyro-only startup; see the
[current default validation](F64_DEFAULT_VALIDATION.md).

The original full mission **fails accuracy validation on both aerial 6 and
aerial 7**. Its raw metric ATE values are 13.170 m and 64,656.661 m,
respectively; PGO does not repair either trajectory. These failures remain on
record in the mission's `accuracy_assessment.json` and regenerated
`validation_summary.json`. Completing a replay with finite poses is not an
accuracy pass.

## Findings

The evidence points to fragile monocular initialization interacting with the
f32 estimator path and the tight IMU bias random-walk constraints. This is not
yet a demonstrated single-line arithmetic defect, nor proof that upstream C++
Basalt has the same failure. No upstream C++ comparison was performed.

The subsequent [aerial-6 follow-up](AERIAL6_FOLLOWUP.md) identifies a large
gravity/bias initialization error in both precision paths. On matched full
standalone runs, correcting only the gyro offset measured from the first
stationary second reduces aerial-6 ATE from 12.447 m to 1.925 m while retaining
f32 and 0.75/0.75 uncertainty multipliers. The full 1.5/1.5 control still has
6.091 m ATE. This strengthens the case for robust stationary initialization;
the production profile and original failed mission remain unchanged.

- **The failure already exists without ROS2, JIST, loop closure, or PGO.** The
  original standalone aerial-7 runner's first 400 poses were reproduced
  bit-for-bit. A standalone aerial-6 run also has a large initial scale error.
- **Startup has almost no translation.** Ground truth, used only for
  evaluation, moves about 1 mm on aerial 6 and 3 mm on aerial 7 over the first
  1.4 s. Both first images show the nearby tiled ground and landing gear.
- **The estimator invents enough motion to triangulate.** Initialization uses
  one accelerometer reading to orient gravity, with zero initial velocity and
  biases. Optimization starts at frame 4; there is no monocular observability
  gate. Temporal triangulation checks estimated camera baseline >=5 cm,
  without a separate minimum parallax-angle gate. The measured initial
  acceleration magnitude is about 9.761–9.762 m/s², versus the model's 9.81;
  this initial mismatch can seed motion before bias/scale become observable.
  It also occurs in the successful aerial-5/8 data, so it is not sufficient
  by itself to explain the failures.
- **Aerial 7 bootstraps a bad depth map before the visible explosion.** At
  frame 28 (1.4 s), 84 landmarks are triangulated. Their median host distance
  is initially 21.78 m, then 3,978.75 m at frame 35 and 5,333.55 m at frame 80.
  At frame 98, the baseline still has a 292.83 m median, while the looser-noise
  control has recovered to 0.344 m. These are map-state distances, not DA3
  depths. The accelerometer bias then grows to 15.407 m/s² and speed to
  145.49 m/s by 19.95 s, despite at least 176 tracked image features.
- **Aerial 6 has a bad scale estimate, not merely accumulated end-of-run
  drift.** Its standalone first-20-second scale diagnostic indicates 63.0%
  excess scale and 3.105 m metric ATE. The original full ROS run has about
  15.2% excess scale; even diagnostic scale fitting leaves 9.836 m ATE.
  The trajectory is also deformed, so a global rescale cannot fix it.

The relevant initialization and triangulation code is in
[`estimator.rs`](../pipelines/basalt/src/vio/estimator.rs), and the f32/f64
solver paths are in [`aom.rs`](../pipelines/basalt/src/vio/aom.rs) and
[`window.rs`](../pipelines/basalt/src/vio/window.rs).

## Controlled first-20-second ablations

All rows start from the first image and process 400 identical ordered sensor
frames through the standalone runner, except the explicitly separate cropped
startup test below. All use monocular left-camera input, 800 px processing,
the supplied aerial calibration, no loop/depth processing, and metric SE(3)
ATE without scale fitting. Multipliers apply to standard deviations, so
changing 0.75 to 1.5 reduces the affected information weights fourfold.

| Sequence | Arithmetic path | White-noise multiplier | Bias-walk multiplier | Metric ATE |
|---|---|---:|---:|---:|
| aerial 6 | f32 | 0.75 | 0.75 | 3.105 m |
| aerial 6 | f64 | 0.75 | 0.75 | 1.405 m |
| aerial 6 | f32 | 1.5 | 1.5 | 0.693 m |
| aerial 7 | f32 | 0.75 | 0.75 | 311.539 m |
| aerial 7 | f64 | 0.75 | 0.75 | 0.241 m |
| aerial 7 | f32 | 1.5 | 0.75 | 960.274 m |
| aerial 7 | f32 | 0.75 | 1.5 | 0.376 m |
| aerial 7 | f32 | 1.5 | 1.5 | 0.118 m |

This isolates **bias-walk weighting as an important sensitivity** on aerial 7:
loosening white noise alone does not help; loosening bias walk does, while
retaining f32. It does not independently establish the true physical noise
densities. Likewise, f64 means double rather than single precision, but the
implementation also uses different numerical routines (for example f32 LDLT
versus f64 Cholesky, plus separate triangulation/marginalization paths).
The mode comparison cannot prove rounding alone is the defect.

A frame-98 solver capture has an undamped normal-matrix eigenvalue ratio of
about 8.91e11 under the failed baseline, versus 7.28e9 in the successful
looser-noise control. This supports numerical-conditioning concerns; it is a
comparison of already different estimator states, not an isolated proof of
which operation first fails. Neither captured matrix is indefinite.
The trace-enabled and trace-disabled poses match bit-for-bit over all 130
captured frames in both controls.

![First 20 seconds](../results/graco/vio_failure_audit/startup_comparison.png)

## Completed full f64 controls

Both standalone controls processed every image with the original 0.75/0.75
uncertainty multipliers and unchanged calibration, with no loop correction:

| Sequence | Images | Full metric ATE | Diagnostic scale to ground truth |
|---|---:|---:|---:|
| aerial 6 | 6,584 | 6.886 m | 1.0083 |
| aerial 7 | 7,886 | 2.637 m | 1.0415 |

**Changing float to double is not a complete repair.** Aerial 7 no longer
diverges, but aerial 6 still has unacceptable drift. Aerial 6's first 80 s
have 0.857 m ATE, followed by substantial later trajectory deformation; even
diagnostic scale fitting leaves 6.870 m full ATE. Thus the original gross
scale/startup error is only part of its failure. The follow-up examines the
gravity/bias coupling further; these controls do not validate
the multi-robot mission or justify silently replacing its requested f32 mode.

## Sensor and integration checks

- IMU intervals are approximately 7.989–8.011 ms (125 Hz), without large gaps
  or ordering errors. The full mission reported zero image/keyframe drops.
- Aerial 6's first image precedes its first IMU by 10.858 ms. The first future
  IMU initializes gravity; there is no preceding interval to integrate.
  The standalone runner unnecessarily rejected this case, while the ROS
  adapter already allowed it. Its guard now matches ROS semantics; subsequent
  empty intervals are still rejected. Skipping the first image does **not**
  cure the failure: a fresh f32 run starting at image 1 has 6.235 m ATE over
  its next 400 images. This cropped run is not part of the identical-input
  table above.
- Camera calibration and preprocessing parameters were held fixed in the
  arithmetic/uncertainty ablations. The same calibration works much better
  under the alternative paths, so the available evidence does not justify
  changing the measured baseline, intrinsics, or extrinsics.
- The original ROS and standalone runs are different ingestion paths and do
  not have bit-identical aerial-6/7 poses. Thus their full-run ATE values are
  not claimed as an exact sensor-payload parity comparison. The controlled
  ablations above all use the same standalone path; their conclusions do not
  depend on cross-wrapper parity.

## Scope and next repair

The production f32/0.75 profile, calibration, and tracking algorithm have not
been changed. Aerial 7's 1.5 controls cover only startup; the completed aerial-6
1.5 control still has 6.091 m ATE. These are not a validated replacement
mission configuration. The diagnosis calls for a robust stationary/monocular
initialization and an audit of the f32 bias/prior/linear-solve path, rather
than allowing failed VIO to enter the pose graph and expecting SE(3) PGO to
recover scale. Finite-output checks alone also need a separate estimator
health assessment; no arbitrary health cutoff was silently introduced.

The only runner changes are the first-interval guard and an explicit
`--start-frame` diagnostic option. All experiment files live in
`results/graco/vio_failure_audit`; original mission outputs are preserved.
The audit includes `analysis.json`, `sensor_audit.json`, exact-repeat and
trace-parity reports, executable/source hashes, per-run calibration and
configuration, trajectories, bias states, Rerun recordings, and the frame-98
solver/landmark captures. The first-interval guard's positive and negative
stream checks passed; Python compilation, Rust formatting, and diff checks
also passed.

Example reproductions (from the repository root):

```sh
.runtime/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/aerial-07-25m_full_ros2 \
  --output results/graco/NEW_a07_f32_control \
  --binary results/graco/vio_failure_audit/basalt_stream_vio_audit \
  --camera-mode mono --scalar-mode f32 --max-frames 400 \
  --imu-noise-scale 0.75 --imu-bias-scale 0.75
```

Change only `--scalar-mode f64` for the precision-path comparison, or only
`--imu-bias-scale 1.5` for the bias-walk comparison. Use a new output directory
for every run. Omitting `--max-frames` runs the entire bag. Ground truth is
never passed to the Rust estimator.
