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

GNSS epochs own separate navigation/clock identities. Camera blocks retain their
existing indices; hidden epoch states are appended. An observation inside an IMU
interval replaces that interval with boundary-interpolated subintervals; the
unsplit factor is removed. Split intervals integrate continuous position/rotation/velocity process-noise covariance, with the calibrated packet period converting the existing per-packet noise standard deviations. This avoids singular covariance for sub-packet intervals. The ordinary VIO covariance path remains unchanged. The joint prior retains camera pose/navigation
blocks, ENU translation/yaw, and a clock frontier. Rank-aware orthogonal
projection eliminates epoch states and old clocks without adding information
in nullspaces. Retained rotation charts include their left-tangent derivatives.

RAWX timing is corrected for estimated receiver clock bias, independently of
USB receipt time. Raw code/Doppler bootstrap ECEF position, velocity, separate
GPS/Galileo biases, and a common drift. Motion aligns VIO to ENU and searches a
constant GNSS-to-sensor time offset; timing must pass the curvature/uncertainty
gate before activation. The offset is frozen after initialization. A known-PPS
configuration may explicitly supply `fixed_time_offset_s`; absent means estimate,
not zero. PPK is never used for this step.

The original three-state window was too short for this recording's USB latency.
Once GNSS activates, its navigation window retains at least 0.5 seconds where
capacity permits, bounded by 32 camera navigation states. These are tuning
parameters (`min_active_window_s`, `max_navigation_states`). Tracking and
keyframe selection are unchanged; navigation retirement is delayed. Disabled
GNSS and pre-activation VIO keep the original path. Epochs older than the active
window are rejected and logged. Queues, bootstrap history, IMU history, and
broadcast ephemeris history are bounded. During outages VIO continues and the
clock frontier's random-walk uncertainty grows with elapsed sensor time.

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
python scripts/prepare_raw_gnss_experiment.py BASELINE_RUN RECORDING OUTPUT
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
six 0.25x stereo-inertial missions. Results are excluded from Git. Evaluation
uses evo translation APE with one rigid alignment, scale fixed to 1, 20 ms
association, and nominal antenna lever compensation on every estimate.
South-ece's supplied PPK is a correlated reference derived from the same rover
observations and covers only part of the route. No interpolation across missing
reference intervals or full-route accuracy claim is made.

The expanded navigation window is currently much slower than ordinary VIO.
The GNSS solver reduces exact nonzero factor columns and reuses Basalt's bound,
one-shot landmark trial token; ordinary VIO retains its existing solver path.
The regression checks compare reduced systems and landmark recovery, and
ordered replay comparisons check raw pose bits across these implementations.
Replay backpressure preserves every sensor input when computation exceeds the
requested playback interval. This version is not ready for the live camera
frame rate. Enable `visloc-basalt/basalt-timing-breakdown` at build time and
`VISLOC_BASALT_TIMING_BREAKDOWN=1` at runtime for the optional timing sidecar.

The joint square-root prior is estimator-owned. The legacy VIO `MargData`
export does not serialize the additional GNSS navigation, alignment, and clock
blocks; it must not be used as a GNSS estimator checkpoint. The ROS replay
uses the native joint prior directly and records its diagnostics separately.

Carrier-phase RTK, additional constellations, online extrinsic calibration,
NeQuick, and changes to loop/PGO factors remain outside this experiment.

## South-ece validation

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
