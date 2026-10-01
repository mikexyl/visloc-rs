# Experimental raw GNSS VIO

Branch `experiment/raw-gnss-vio` starts at the sequential SLAM baseline
`73b52a4`. GNSS is opt-in. Tracking uses the existing f64 Basalt frontend,
stationary gyro-only startup, measured stereo calibration, and the unchanged
sequential PGO / relative-pose loop backend. No PPK trajectory, receiver position
solution, or GPS fix topic enters the estimator.

The design follows the raw pseudorange/Doppler coupling of
[GVINS](https://arxiv.org/abs/2103.07899) and the square-root marginalization and
clock-state treatment discussed by
[SRI-GVINS](https://arxiv.org/html/2405.10874v1). It is an adaptation to our
sliding-window solver, not a reproduction of either complete system.

## Calibration

`visloc-gnss::calibration` composes the supplied mounting geometry and measured
`T_imu_cam[0]`. At the nominal 0.6 mm gasket gap the optical lever is
`[0.1425, -0.1011, -0.0979] m`; the South-ece measured calibration gives
`[0.111193231, -0.094903010, -0.078744703] m` in the IMU frame. Extrinsics remain
fixed. A configurable 20 mm standard deviation per lever axis contributes to
code and rotational-velocity measurement weights. Housing geometry, gasket
compression, and antenna phase center are nominal, not calibrated.

`gnss_effective.json` saves the composed lever, configuration, and replay clock
mapping. `scripts/visualize_raw_gnss_calibration.py` logs the IMU, optical center,
rear mounting face, receiver casing, and antenna phase center to Rerun.

## Estimator

The native `visloc-gnss` crate decodes fragmented UBX RAWX/SFRBX packets, GPS L1
LNAV and Galileo E1 I/NAV, and broadcast ephemerides. It validates framing,
checksums, Galileo CRC, issue consistency, health, and age. Other constellations
and frequencies are counted as unsupported. No application C/C++ bridge is used;
see `crates/gnss/THIRD_PARTY.md` for broadcast-navigation attribution.

Pseudorange and Doppler are optimized jointly with visual and inertial rows.
The antenna point includes the rotated lever arm; antenna velocity includes
body angular velocity after gyro-bias subtraction. Satellite clocks,
relativistic clock correction, Earth rotation, and atmosphere corrections are
included. GPS uses broadcast Klobuchar plus Saastamoinen; Galileo E1 currently
uses the same conservative Klobuchar model, **not Galileo NeQuick**. Satellite
transmission states and atmospheric delays are frozen within each window
problem; their small derivatives are not part of the state Jacobian.

By default, `coupling: "keyframe_preintegration"` retains the newest actual
keyframe as Nav15 (pose, velocity, and biases), alongside Basalt's ordinary
short recent-camera window. Older keyframes become Pose6 when replaced by a
new keyframe. Actual keyframe selection remains unchanged. GNSS does not add
optimized navigation states or replace IMU links in this mode.

Each epoch binds to a retained actual keyframe. Boundary-interpolated IMU
preintegration evaluates its navigation at the true GNSS timestamp, forward
or backward. The backward reconstruction follows SRI-GVINS Section III-C,
Equations 22–29; left-tangent analytical Jacobians chain the GNSS rows to the
keyframe. The measurement covariance includes propagated IMU uncertainty,
a frozen-dynamics bias-diffusion approximation, and the bootstrapped timing
uncertainty. Code/rate covariance is whitened jointly when both channels are
active; an inactive channel cannot contaminate the active channel.

The propagation reuses IMU data also supporting VIO. Treating its uncertainty
as additional measurement noise omits cross-correlations with the VIO prior
and across epochs/satellites. This is an approximation, not exact independent
fusion or a reproduction of the SRI-GVINS filter. Satellite and atmosphere
states, propagation covariance, and robust residual decisions are refreshed
for each window construction and frozen during its LM solve.

An epoch stays a live factor until its owning keyframe retires. Its GNSS rows
then enter the square-root FEJ prior exactly once. Marginalizing ordinary
non-keyframes does not assimilate that live GNSS factor. The joint prior
retains camera pose/navigation blocks, ENU translation/yaw, and one clock
frontier. Rank-aware projection eliminates old clocks and retired velocity/
bias states without adding information in nullspaces. Rotation charts include
their left-tangent derivatives.

`coupling: "frame_window"` retains the previous experimental implementation
for comparisons: hidden GNSS epoch Nav15 blocks replace affected IMU links
with boundary-interpolated subintervals, and camera navigation retirement is
delayed by `min_active_window_s` (0.5 s) up to `max_navigation_states` (32).
These legacy settings do not expand the new keyframe window. The ordinary
GNSS-disabled VIO covariance and solver paths remain unchanged.

RAWX timing is corrected for estimated receiver clock bias, independently of
USB receipt time. Raw code/Doppler bootstrap ECEF position, velocity, separate
GPS/Galileo biases, and a common drift. Motion aligns VIO to ENU and searches a
constant GNSS-to-sensor time offset; timing must pass the curvature/uncertainty
gate before activation. The offset is frozen after initialization. A known-PPS
configuration may explicitly supply `fixed_time_offset_s`; absent means estimate,
not zero. PPK is never used for this step.

The measured indexed arrival age after clock mapping is 68 ms median,
85 ms p95, and 202 ms maximum at the next camera update on this recording.
The previous 0.5 s full-state window was a conservative implementation choice;
the -80 ms clock offset alone does not require 16–18 camera navigation states.

`max_epoch_age_s` (0.5 s by default) rejects stale epochs when first binding
them, without retaining extra camera states. This removes the startup backlog
and bounds delayed live input. Epochs already owned by an active keyframe
remain live until retirement. On activation, if the last actual keyframe has
already become Pose6, GNSS waits for the next actual keyframe instead of
inventing velocity/bias values. Missing keyframe-to-epoch IMU support, retired
owners, and epochs older than the marginalized clock frontier are rejected
and logged. If the chosen keyframe needs the next IMU packet, the epoch is
deferred until that packet reaches the estimator; VIO inputs are not modified.
A keyframe omitted from a pre-marginalization prefix remains a live owner. Epoch IDs remain deduplicated after retirement. Queues, bootstrap
history, ten-second IMU history, and ephemeris history stay bounded. Outages
continue VIO and clock random-walk uncertainty grows over elapsed sensor time.

`mode: "window_only"` is an ablation: bootstrap determines the same activation
time and keyframe-retention policy, but no GNSS measurement/clock rows are
added. It isolates effects of retaining keyframe navigation and the joint
marginalization path.

Doppler-only is a diagnostic: raw code still bootstraps position/clocks/timing,
but code factors are disabled. Geographic translation remains fixed because
Doppler provides negligible translation observability; velocity, yaw, and clock
drift are fused. A conservative bootstrap clock prior fixes clock-bias gauges.

## ROS2 and replay

Native `rclrs` callbacks enqueue inputs for the existing sensor worker. Raw UBX
bytes arrive on `/<robot>/gnss/raw` (`std_msgs/UInt8MultiArray`); alternatively
`gnss_typed_input=true` receives typed epochs/ephemerides. The two modes are
mutually exclusive. The node never opens a GPS serial device; existing
mapping-gated ownership remains the source's responsibility.

Typed records are `GnssEpoch`, `GnssObservation`, `GnssEphemeris`, and `GnssStatus`.
Epoch delivery is reliable; ephemerides/status use transient-local durability.
Replay raw input uses backpressure. Live queue drops are counted and logged.
Decoded records are persisted before publication. Per-satellite decisions,
normalized optimized residuals, clocks, queue drops, and initialization status
are saved to `gnss_records.jsonl` / `gnss_diagnostics.jsonl`.

Fused local odometry keeps `odom -> base_link`. Geographic odometry is separately
published on `/<robot>/gnss/geographic_odometry`. The GNSS ENU alignment uses
`earth -> <robot>/gnss_enu -> <robot>/gnss_local` to avoid giving PGO's odometry
frame two parents. GNSS alignment is independent of PGO map corrections.

Replay uses original UBX byte-receipt ordering, preserving the recorded RAWX
GNSS time. `gnss_replay_shift_ns` must equal the camera/IMU replay mapping.
Recorded USB receipts remain in the dataset index; the decoded epoch's receipt
field records this run's callback wall time and is diagnostic only. Sensor time
and `/clock` remain estimation time. Communication deadlines use steady time.

Prepare and run (source ROS2 first):

```sh
python scripts/prepare_raw_gnss_experiment.py BASELINE_RUN RECORDING OUTPUT --include-window-only
python scripts/run_multi_robot_mission.py OUTPUT/baseline_no_loops/mission.json --domain-id 211
python scripts/run_multi_robot_mission.py OUTPUT/doppler_only_no_loops/mission.json --domain-id 212
python scripts/run_multi_robot_mission.py OUTPUT/pseudorange_doppler_no_loops/mission.json --domain-id 213
# Repeat the corresponding *_loops missions after the no-loop validation.
python scripts/evaluate_raw_gnss_experiment.py OUTPUT RECORDING/ppk/RUN/ground-truth.csv
python scripts/render_raw_gnss_report.py OUTPUT
# Match the mission runner's local DDS settings for an independent client:
ROS_DOMAIN_ID=213 ROS_LOCALHOST_ONLY=1 python scripts/test_raw_gnss_ros_contracts.py --output OUTPUT/ros_contracts.json
```

The preparation script hashes recording/calibration identities and creates all
six 0.25x stereo-inertial missions plus the optional window-only ablation.
Use `--coupling frame_window` to explicitly reproduce the legacy configuration. Results are excluded from Git. Evaluation
uses evo translation APE with one rigid alignment, scale fixed to 1, 20 ms
association, and nominal antenna lever compensation on every estimate.
South-ece's supplied PPK is a correlated reference derived from the same rover
observations and covers only part of the route. No interpolation across missing
reference intervals or full-route accuracy claim is made.

The legacy expanded navigation window is much slower than ordinary VIO.
The GNSS solver reduces exact nonzero factor columns and reuses Basalt's bound,
one-shot landmark trial token; ordinary VIO retains its existing solver path.
The regression checks compare reduced systems and landmark recovery, and
ordered replay comparisons check raw pose bits across these implementations.
Replay backpressure preserves every sensor input when computation exceeds the
requested playback interval. Live-rate acceptance must be measured separately for each coupling mode. Enable `visloc-basalt/basalt-timing-breakdown` at build time and
`VISLOC_BASALT_TIMING_BREAKDOWN=1` at runtime for the optional timing sidecar.

The joint square-root prior is estimator-owned. The legacy VIO `MargData`
export does not serialize the additional GNSS navigation, alignment, and clock
blocks; it must not be used as a GNSS estimator checkpoint. The ROS replay
uses the native joint prior directly and records its diagnostics separately.

Carrier-phase RTK, additional constellations, online extrinsic calibration,
NeQuick, and changes to loop/PGO factors remain outside this experiment.

## Legacy frame-window South-ece validation

The six complete 0.25x stereo replays each processed 7,182 frames with no sensor,
graph, communication, or GNSS queue drops. Against the same 656 associated PPK
poses (174 seconds of a 239.7-second recording), rigid-aligned raw ATE was
0.767 m for disabled GNSS, 0.861 m for Doppler-only, and 0.856 m for pseudorange
plus Doppler. GNSS **did not improve ATE**: the increases were 12.2% and 11.5%.
The baseline accepted one loop (corrected ATE 0.768 m); fused runs accepted none.
Raw trajectory files were bit-identical with loop processing enabled/disabled
for every mode, including the dense-versus-compact GNSS solver comparison.

GNSS initialized after 32.11 seconds without PPK, freezing a -80 ms time offset
with a 19.76 ms estimated standard deviation. It used 865 epochs. There were
184 startup/history epochs outside the active window and accepted-support gaps
of 3.2 and 31.0 seconds. Decoder checksum and malformed-packet counts were zero.
The compact solver's median frame processing time was about 0.52 seconds under
concurrent replay load; live-rate acceptance is not met.

Native tests passed: 394 Basalt, 8 GNSS unit, 2 independently checked broadcast
fixtures, 10 ROS2, 18 multi-robot, and 9 online-loop tests. The existing 59
Basalt fixture-dependent ignored tests remain ignored. Disabled-input numerical
parity, split-interval noise composition, rank-deficient square-root prior,
rotation-chart Jacobians, clock outages/resets, and timing observability are
covered. A Python ROS2 subscriber received the Rust node's GPS/Galileo custom
messages and retained ephemerides after delayed startup. Full typed-input replay
has not been separately benchmarked; the acceptance replays used RAWX/SFRBX.

Generated report: `results/gnss/south_ece_20260927/report.md`; evo results,
recording/model identities, launch-build hashes, effective configurations,
diagnostics, and communication counts remain alongside it. Per-run manifests
describe the binaries used for each replay; the final-source manifest is stored
separately. The implementation remains opt-in and experimental.

## Keyframe coupling validation

The replacement couples GNSS through retained actual keyframe navigation,
without hidden epoch Nav15 blocks. Three complete 0.25x stereo replays processed
7,182 frames each: window-only, Doppler-only, and code plus Doppler. All used
3–4 camera Nav15 blocks before marginalization, zero GNSS Nav15 blocks, and
zero input/GNSS queue drops. All 6,219 joint solves per run returned finite,
non-increasing objectives. The fused runs deferred 145 epochs until their
keyframe IMU bracket arrived, rejected no epochs for missing brackets/retired
owners/clock-frontier age, discarded 183 startup backlog epochs, and used 866
GNSS epochs. Initialization and frozen timing stayed at 32.11 s and -80 ms.

On the same 656 associated PPK poses, raw ATE was 0.778310 m for window-only,
0.778236 m for Doppler-only, and 0.777796 m for code plus Doppler. The verified
ordinary VIO baseline remains 0.767399 m; legacy code plus Doppler was
0.855707 m. Thus the new implementation removes most of the legacy regression
but does not demonstrate an accuracy benefit over ordinary VIO (+0.010398 m,
+1.35%). GNSS has little effect on the shape here: direct unaligned position
changes from the window-only run were 5.6 mm RMS / 17.4 mm maximum for code plus
Doppler. This comparison does not establish geographic absolute accuracy.

A repeated same-build 1,250-frame diagnostic at 4x requested speed measured
78.4 ms median active processing for keyframe coupling versus 497.9 ms for the
legacy frame window, a 6.3x reduction. Full new runs measured about 52 ms median
active processing at 0.25x. Replays ran concurrently, so timings include machine
contention; even the faster path exceeds the 30 Hz camera budget. Calibration,
raw sensor intervals, VIO input ordering, and loop/PGO formulation are unchanged.

The final regression suite has 450 active passes (402 Basalt, 9 GNSS unit,
2 broadcast, 10 ROS2, 18 multi-robot, 9 loop), with the same 59 existing Basalt
fixture-dependent ignores. New coverage checks forward/backward manifold
Jacobians, propagated covariance, correlated whitening and channel masking,
actual satellite information entering the prior once, retained-keyframe
selection, deferred IMU brackets, and incomplete marginalization prefixes.
GNSS-disabled numerical parity passes, and every replay preserves baseline
raw pose bits through frame 962, before GNSS activation at frame 963.

Report and evo results: `results/gnss/south_ece_keyframes_20261001/report.md`.
Recording identities, per-run source/binary hashes, the runtime benchmark,
complete-run checks, configurations, and diagnostics are saved alongside it.
The final fused runs use loops disabled; loop-enabled full replay has not been
repeated for the new coupling. The default coupling is keyframe preintegration
when GNSS is explicitly enabled; GNSS itself remains disabled by default.
