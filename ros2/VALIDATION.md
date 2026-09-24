# Native Rust ROS2 multi-robot SLAM validation

This report preserves the initial f32/legacy-startup experiments, including
their accuracy failures. The current estimator is f64-only and defaults to
stationary gyro-only startup. See the [shared-profile results](UNIFORM_ATE4_VALIDATION.md)
and [current-default regression](F64_DEFAULT_VALIDATION.md): all four raw VIO
sequences pass the 4 m ATE RMSE target. These later raw-VIO checks do not replace
the historical multi-robot loop/PGO measurements below.

Validated on 2026-09-23 with Ubuntu 22.04, ROS2 Humble, Rust 1.94,
TensorRT 10.13.2, and an RTX 4070. The checked-in ROS dependency pins and build
instructions are in [README.md](README.md). The robot and centralized backend
are native `rclrs` executables; Python publishes bags, tests interoperability,
evaluates trajectories, and records Rerun views.

## Algorithm and regression checks

| Check | Result |
|---|---|
| Multi-robot algorithm suite | 16 passed |
| Existing Basalt and SLAM suites | 1,391 passed; 77 pre-existing ignored tests remained ignored |
| Existing online-loop unit suite | 9 passed |
| Existing online-loop GPU tests | Both explicitly enabled tests passed |
| ROS archive recovery unit test | Passed, including multiple sessions and incomplete journal tail |
| Identical ordered ROS sensors, loops enabled vs disabled | All seven raw aerial-5 pose columns identical, maximum absolute difference 0.0 over 400 images |

The algorithm suite covers strict >95% landmark overlap, empty sets, intentional
skips versus missing keyframes, all 56 optimal selections, full 25-pair
refinement, exclusions before ranking, two-hop covisibility over omitted
keyframes, robot/session ID collisions, endpoint-specific calibration, reverse
PnP, stronger-direction selection, out-of-order graph edges, component joining,
anchor removal, and invalid pose/information rejection.

The raw-VIO control is `results/graco/multi_robot_loop_disabled_control`.
The final 400-image-per-robot 5/7 smoke run is
`results/graco/multi_robot_final_smoke_5_7`: 800 images, 114 keyframes, seven
sequences, no verified loops, no image/keyframe/communication queue drops, and
no request failures. Its aerial-5 poses also match the earlier enabled smoke
run exactly. It completed in 40.63 seconds at nominal 1x with backpressure.
The initial smoke validation used the default 0.25x rate.

## ROS2 transport and lifecycle checks

These tests launch the actual native executables and exchange generated ROS2
messages/services with `rclpy`, rather than mocking a transport interface.

- Backend integration: delayed discovery, two-record history pages, an induced
  two-second timeout while `/clock` is frozen, clock jumps, duplicate records,
  a loop arriving before its endpoint, disconnection, backend restart,
  interrupted journal-tail recovery, a new estimator session, and distinct map
  frame IDs for disconnected fragments within one session. Final state:
  15 keyframes, two components, one unique loop; revision increased from 5 to 10.
- Robot queue/restart integration: 48 images arrived before IMU coverage;
  the 32-image buffer retained 32 and reported exactly 16 drops. Restart kept
  the old keyframe and added a distinct-session keyframe with local ID zero.
- Real TensorRT reconnect test: withhold peer services for all four request
  attempts, then reconnect and retransmit the same announcement repeatedly.
  The restored native robot performed one matching attempt and published one
  verified constraint with 16 PnP inliers. Four unanswered calls were recorded;
  no duplicate factor was produced.

The final request helper replaces a client after a timeout, bounding the pinned
rclrs implementation's retained unanswered-request state. Both outage/reconnect
and backend integration tests passed again after this change. The complete
four-robot replay started before the final transport/output hardening (client
replacement, fragment-specific map frame names, and atomic status files); its
VIO, sequence, verification, and optimization algorithms are identical.

The latest request helper and final-publication barrier were also exercised in
`results/graco/multi_robot_final_publication_smoke`: all four robots processed
160 images each, producing 92 keyframes with zero drops or request failures in
14.03 seconds. Replay received the exact final solve's revision through DDS
before shutting down. Its first 160 aerial-5 raw poses remained bit-identical
to the loop-disabled control. The full mission predates this replay shutdown
hardening; its final persisted graph is evaluated separately.
The final node build also passed a 256-image four-robot shutdown smoke run in
`results/graco/multi_robot_atomic_status_smoke`, with valid atomic status and
communication reports, no drops or request failures, and matching final graph
and completion revisions.

Machine-readable results are `.runtime/ros2_integration_result.json`,
`.runtime/ros2_robot_integration_result.json`, and
`.runtime/ros2_reconnect_result.json`. Runtime fixtures use isolated ROS domains
219–221 and temporary output directories.

## TensorRT parity

The FP32 bundle uses no TF32 and exactly 128 real matcher features. Compared
with ONNX Runtime fixtures:

| Output | Maximum absolute difference |
|---|---:|
| JIST sequence descriptor | 1.34e-7 |
| JIST frame descriptors | 4.06e-7 |
| XFeat | 1.57e-5 |
| Common LighterGlue match scores | 5.92e-5 |

Stable match membership agrees completely. One fixture has one additional
TensorRT match at score 0.1000007838, immediately above the 0.1 cutoff; ONNX
Runtime excludes it. The validator explicitly reports such changes within
1e-5 of the cutoff, and still requires complete agreement for stable matches.
This is numerical parity within the documented boundary tolerance, not
bit-identical discrete matcher output. Full results and model hashes are in
`.runtime/multi_robot_models/validation/report.json` and the bundle manifest.

## Dataset accuracy limitation

**Aerial 6 and aerial 7 both fail accuracy validation.** Aerial 6's full raw
13.170 m ATE (11.841 m after PGO) is a failure, not an acceptable drift result;
aerial 7 catastrophically diverges. Neither successful replay completion nor
the absence of estimator exceptions constitutes a successful VIO run.
The subsequent [failure investigation](VIO_FAILURE_ANALYSIS.md) reproduces
the failure without ROS or loops and isolates sensitivity to the f32
estimator path and tight IMU bias-walk weighting during monocular startup.
The [aerial-6 follow-up](AERIAL6_FOLLOWUP.md) further isolates gravity/bias
initialization sensitivity: a sensor-only stationary gyro-offset experiment
reduces full standalone f32 ATE from 12.447 m to 1.925 m. At the time of this
experiment it had not yet been integrated as an online initializer. The later
online startup implementation and full f64 raw-VIO validation are linked above.

The supplied aerial-7 start diverges in raw monocular VIO with the requested
calibration, f32 arithmetic, and 0.75 noise/bias multipliers. Over its first 400
images, the new ROS path has 303.38 m metric ATE; the pre-existing single-robot
runner also diverges, at 311.54 m. Aerial-5 over the same 400-image smoke length
has 1.0303 m ATE. The same-input enabled/disabled test above isolates loop
processing from this pre-existing raw-VIO divergence. Those integration tests
alone do not diagnose its cause; see the separate controlled investigation.

No calibration changes, scale fitting, ground-truth feedback, or Basalt tracking
changes were made to mask this result. Passing transport and loop-verification
tests does not establish useful map accuracy. SE(3) PGO cannot repair the
underlying scale failure of a diverged metric VIO trajectory.

## Completed short 6/7 refinement experiment

`results/graco/multi_robot_postchange_6_7` contains 1,600 images per robot,
457 keyframes, 27 sequences, and two cross-robot constraints joining the robots
into one component. There were 24 unique verification attempts, no dropped
images/keyframes, and no request failures. Wall time was 170.98 seconds.

The controlled comparison uses the identical 24 recorded retrieval pairs and
the same raw odometry and batch solver settings:

| Verification mode | Accepted loops | Total accepted PnP inliers | Joint metric ATE |
|---|---:|---:|---:|
| Full 5×5 refinement | 2 | 34 | 4,669.71 m |
| Fixed last frame | 2 | 33 | 5,588.45 m |

The large errors reflect aerial-7's raw divergence. These results do not
demonstrate an accurate multi-robot map. The actual asynchronous online solve
is reported separately from the controlled batch comparison.

## Full aerial 5–8 experiment

The final full mission is `results/graco/multi_robot_aerial_5_8_final`, with
nominal 1x playback after the initial 0.25x validation, and per-robot sensor
acknowledgement backpressure. Sensor timestamps and intervals remain unchanged.
The superseded slow replay in `multi_robot_aerial_5_8_v4` was stopped early and
is explicitly marked incomplete; it must not be used as a full-mission result.

All 25,959 images completed: 3,709 actual keyframes, 274 sequences, 119 unique
matching attempts, and three verified constraints (2.52% verification success).
Every emitted keyframe and loop reached the final graph and every submitted
matching attempt received a result. There were zero image/keyframe drops and
three exhausted history requests, one each on aerial 6, 7, and 8. No node
reported an estimator error. There were 30 insufficient-match rejections and
86 geometric-support rejections.

The final graph has three components: `{a05}`, `{a06, a07}`, and `{a08}`.
The accepted constraints are aerial-5 keyframes 43–775 (16 PnP inliers,
0.973 px, reverse direction), aerial-6/7 keyframes 108–112 (16 inliers,
0.732 px), and aerial-6/7 keyframes 118–125 (18 inliers, 0.878 px).

| Robot | Images | Keyframes | Sequences | Raw metric ATE | Online corrected metric ATE |
|---|---:|---:|---:|---:|---:|
| aerial 5 | 5,926 | 847 | 72 | 2.439 m | 2.597 m |
| aerial 6 | 6,584 | 941 | 66 | 13.170 m | 11.841 m |
| aerial 7 | 7,886 | 1,126 | 74 | 64,656.661 m | 64,912.597 m |
| aerial 8 | 5,563 | 795 | 62 | 1.059 m | 1.059 m |

Per-robot ATE uses one rigid alignment per trajectory. Joint corrected ATE
uses **one rigid alignment per connected component**, with no scale fitting:
2.597 m for aerial 5, **53,254.154 m for aerial 6+7**, and 1.059 m for aerial 8.
All 25,959 poses were associated with ground truth. The first 400 aerial-5 raw
poses in this full run also matched the loop-disabled ROS control bit for bit.

The aerial-5 loop slightly worsens metric ATE, aerial 6 improves in the online
solve, aerial 8 is unchanged, and aerial 7 remains severely divergent. These
measurements do **not** establish an accurate combined map or a general
accuracy improvement from refinement.

The controlled batch experiment reuses the same 119 retrieval pairs, raw
odometry, and solver settings. It is separate from the asynchronous online
solution above:

| Mode | Accepted loops | Accepted PnP inliers | Aerial 5 ATE | Aerial 6+7 joint ATE | Aerial 8 ATE |
|---|---:|---:|---:|---:|---:|
| Full 5×5 refinement | 3 | 50 | 2.597 m | 53,039.799 m | 1.059 m |
| Fixed last frame | 2 | 33 | 2.439 m | 53,170.337 m | 1.059 m |

Both modes still leave three components. Refinement finds the extra aerial-5
self-loop, which does not improve its metric accuracy. The two batch modes'
individual aerial-6 ATE values are 13.250 m and 716.171 m, respectively;
aerial-7 values are 64,322.902 m and 64,606.767 m. This sensitivity, together
with the large joint errors, reinforces the underlying VIO limitation.

### Timing and communication

The mission took **7,130.62 seconds (118.84 minutes)**. Despite nominal 1x
publication, bounded backpressure produced an effective **0.0553x** rate.
The full replay is not a real-time throughput demonstration; the cause of its
slowdown has not been established. Timings came from a shared development
machine, including intermittent separate validation jobs.

| Robot | VIO median / P95 | Sequence encoding median / P95 | Exchange through verification median / P95 |
|---|---:|---:|---:|
| aerial 5 | 55.41 / 83.64 ms | 32.69 / 46.31 ms | 10.78 / 24.45 ms |
| aerial 6 | 63.76 / 93.60 ms | 31.53 / 55.84 ms | 10.37 / 20.46 ms |
| aerial 7 | 55.25 / 95.66 ms | 33.72 / 57.63 ms | 2.92 / 11.26 ms |
| aerial 8 | 56.75 / 86.99 ms | 31.24 / 50.39 ms | 9.19 / 13.09 ms |

Exchange-through-verification timings exclude earlier failed attempts and
retrieval backlog. VIO timing covers the estimator call, not full replay
latency. The backend accepted 1,881 finite updates, rejected none, and every
published solve had a non-increasing robust objective. Its solve median/P95
were 61.18/182.00 ms; the final solve took 119.09 ms, at input revision 3,712.

Observed serialized topic payloads were 927,320,544 bytes of graph snapshots,
995,560 bytes of sequence announcements, and 1,863 bytes of loop constraints.
These cover only those three subscribed topic types; latest-only graph
subscriptions can skip revisions. They exclude other topics, DDS framing,
transport retransmission, and service traffic.

The full run's older non-atomic diagnostics writer left **all four robot
`communication.json` files empty at shutdown**. Their final service totals
and communication-queue counters are unavailable and have not been fabricated
or backfilled. The backend's atomic report survived: 112,205 service attempts,
112,196 responses, 2,917,330 request field bytes, and 7,889,850 response field
bytes. These are **backend-only partial service counts**, excluding CDR
padding/lengths and DDS overhead. The final code fixes diagnostic writes with
atomic replacement; the separate shutdown smoke test above passed.

Regenerate this legacy run's summary with
`scripts/summarize_multi_robot.py MISSION --allow-incomplete-traffic` to retain
the explicit missing-counter warnings. Normal runs require complete counters.
The exact full-run executables and SHA-256 hashes are preserved in the mission's
`executables/` and `executable_manifest.json` before installing the final fixes.

Machine-readable results are `evaluation.json`, `validation_summary.json`,
`refinement_comparison.json`, and `refinement/{refined,fixed_last}/evaluation.json`
inside the final mission directory. `playback.rrd` shows colored raw/corrected
paths, sparse landmarks, selected images and observations, verification results,
loop edges, component membership, and the metric ATE values.
