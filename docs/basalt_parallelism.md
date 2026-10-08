# Basalt CPU parallelism

The f64 VIO solver uses Rayon for independent landmark linearization,
QR elimination and normal-matrix products, and landmark recovery. The image
update schedule is unchanged: each processed frame updates the local window.

Results are collected in landmark order. Objective and normal-matrix sums
retain their original serial order; a parallel floating-point reduction would
change rounding and potentially later LM decisions. Normal-matrix contributions
are processed in batches of at most 32 to bound temporary storage. Small lists
use the serial path. IMU integration, state updates, and marginalization policy
remain sequential. Diagnostic captures and the retained f32 reference path
keep serial landmark linearization.

A dedicated, process-wide pool defaults to `min(4, available CPU threads)`.
It does not change Rayon's global pool, used elsewhere by visloc. Set the
worker count before starting the process:

```bash
VISLOC_BASALT_THREADS=4 <vio-command>
VISLOC_BASALT_THREADS=1 <vio-command>  # serial comparison
```

The setting is read once at first use. Invalid values or failure to create the
pool produce a warning and fall back to serial execution. Multiple estimators
in one process share this pool; separate robot processes each have their own.
Choose the total CPU budget together with loop inference and other workers.

This adds no TBB/C++ dependency. Rayon was already a workspace dependency and
is now a direct dependency of `visloc-basalt`.

## Avoiding repeated f64 work

The production f64 solver evaluates visual, IMU, and bias residual costs
without constructing their Jacobians. Camera projection and Huber arithmetic,
factor accumulation order, covariance whitening, and the constant-free
marginal-prior cost retain the existing conventions. Poses are decoded once
per objective evaluation. Initial/marginal prior construction is unchanged.

Each LM linearization retains its landmark QR decomposition, transformed rows,
and triangular factor. Normal reduction, predicted model decrease, and trial
landmark recovery reuse that decomposition. Recovery still applies Qᵀ to the
raw assembled residual-plus-state increment to preserve floating-point order.
Accepted steps consume the trial's recovered landmark increments through the
existing state/topology-bound token; rejected steps discard them. Missing
landmark identity metadata falls back to the problem's own recovery hook.
Diagnostic captures retain their original evaluation path.

These changes do not reduce the iteration limit, skip frame updates, alter
feature counts, or change initialization/calibration. Rejected LM attempts
still follow the existing relinearization schedule. Frontend KLT parallelism
and overlap between frontend and estimator processing are separate work.

The stream executable can save cumulative phase timings when built with
`--features basalt-timing-breakdown` and run with
`VISLOC_BASALT_TIMING_BREAKDOWN=1`. Its `timing_breakdown.json` file is emitted
only when instrumentation is enabled.

## Initial measurement

On an Intel Core i7-13650HX, replaying the first 1,000 South-ece stereo pairs
(143 keyframes, 640×480, f64, stationary gyro-only initialization, existing
0.75 IMU noise/bias multipliers) through the direct stream adapter gave:

| Build / workers | Mean processing time per pair |
|---|---:|
| Original serial implementation | 76.0 ms |
| Rayon implementation, 1 worker | 80.3 ms |
| Rayon implementation, 2 workers | 74.6 ms |
| Rayon implementation, 4 workers, initial / finished build | 57.7 / 59.8 ms |

Four workers reduced mean frame processing time by 21–24% against the original
serial run. Exported trajectory, velocity/bias states, and initialization
reports were byte-identical across all five runs. The 384 active library tests
passed (59 ignored), including ordered matrix reduction and landmark recovery.
The stream executable and ROS2 robot/backend release builds passed.

These are short desktop measurements, with two four-worker runs and one run
per other variant. The harness excludes ROS, loop inference, GPS BA, ScaRF,
and visualization; this is not a full-mission or Jetson benchmark. Processing
still exceeds the 33.3 ms budget for every 30 Hz image pair. Evidence and the
replay harness are saved under
`results/ucy/south_ece_rayon_20261007/`.

## Residual-only evaluation and QR reuse measurement

A matched 1,000-pair South-ece replay on the same desktop, using four Rayon
workers in both builds and otherwise identical settings, measured:

| Metric | Before reuse | After reuse |
|---|---:|---:|
| Mean processing time per stereo pair | 53.33 ms | 42.57 ms |
| 95th percentile | 76.65 ms | 58.68 ms |
| Replay wall time, including input/output | 62.28 s | 50.76 s |
| Peak resident memory | 50.05 MiB | 51.76 MiB |

Mean processing time fell by 20.2%, with 1.71 MiB additional peak memory.
Both runs produced 143 keyframes, and the trajectory, inertial states, and
initialization report were byte-identical. This is one matched pair of runs;
the earlier Rayon measurements were separate runs with different observed
timings. The 386 active Basalt library tests pass (59 ignored), and both
release stream and ROS2 builds pass. Evidence is under
`results/ucy/south_ece_f64_reuse_20261008/`.

The subsequent full 7,182-pair replay, with phase instrumentation enabled,
produced 1,015 keyframes. Every exported timestamp and body-pose component
exactly matched the previous complete ROS replay. Mean processing time was
44.71 ms/pair (p95 58.34 ms), wall time 376.50 s, and peak memory 53.39 MiB.
This full run validates trajectory preservation; it does not have a newly
measured full-length baseline for a timing percentage.

The full-run timing breakdown, averaged across all input pairs, is:

| Region | ms/pair | Relationship |
|---|---:|---|
| Frontend | 34.37 | Top-level adapter region |
| Temporal KLT | 19.68 | Included in frontend |
| Feature replenishment | 13.65 | Included in frontend |
| Estimator | 10.34 | Top-level adapter region |
| LM solve | 9.64 | Included in estimator |
| Landmark QR / normal reduction | 5.67 | Included in LM solve |
| Linearization | 1.37 | Included in LM solve |
| Trial residual cost | 0.45 | Included in LM solve |
| Marginalization | 0.33 | Included in estimator |

The frontend accounts for about 77% of processing time. Temporal KLT and
feature replenishment are the next optimization targets. The current desktop
run still exceeds the 33.3 ms budget for every 30 Hz pair; Jetson performance
has not been measured with these changes.
