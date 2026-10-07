# Experimental horizontal GPS pose graph

GPS is enabled by default in the centralized [joint global BA backend](global_bundle_adjustment.md#gps-factors-in-joint-ba). SE(3) pose-graph optimization remains available with `pgo.mode="pose_graph"`. Both `pgo.gps.enabled` and the robot adapter's `gps.enabled` default to `true`; explicit `false` disables them. Missing or rejected GPS leaves visual optimization available. New RealSense missions prepare normalized receiver records by default; use `--no-gps` to disable them. GRACO missions have no receiver stream and explicitly disable GPS.

Basalt, image/IMU calibration, initialization, tracking, and raw odometry are unchanged. No code from the abandoned tightly coupled GNSS estimator is imported. Neither node opens a GPS serial device.

The [native GTSAM C++ wrapper](gtsam_pose_graph.md) is now the sole optimizer for the centralized backend and the single-robot loop worker. The custom Rust optimizer path, experimental Rust GPS factors and Python worker have been removed. There is no optimizer selector or fallback.

The experiment now uses **horizontal GPS only**. Following user feedback, receiver altitude is ignored even when projecting fixes: latitude/longitude are projected at the fixed mission datum height. An automatic datum has height zero; an explicitly configured datum may have a fixed height. Root height and gravity remain fixed, while east/north translation and yaw are free after GPS initialization. There is no altitude-enabled configuration or active altitude ablation.

## Quality and correction policy

Ordinary standalone GPS and differential GPS are accepted; **RTK is not required**. Receiver quality checks precede association and alignment:

- Require a valid satellite fix. GGA quality 1, 2, 4, and 5 are accepted when supplied; dead reckoning/manual/simulation quality is rejected. A valid `NavSatFix` without GGA quality can pass the fix check.
- Default maximum HDOP is **1.0**. HDOP is required by the strict experiment profile. A bare `NavSatFix` has no HDOP, so an upstream normalized `GpsFix` carrying receiver metadata is needed for that profile; missing metadata is reported explicitly. `require_hdop=false` is an explicit relaxation for another input.
- If covariance is known, require the largest horizontal standard deviation to be at most **5 m**. Unknown covariance is not zero noise: assume **5 m horizontal standard deviation**, multiplied by `max(1, HDOP)`. Floors and thresholds are configurable assumptions, not calibrated receiver noise.
- Retain at most one factor per second and require 2 m of raw body displacement. Initialization requires ten retained fixes, 20 m of horizontal motion and 70% consensus within three horizontal standard deviations. The refined yaw/translation fit must also meet these requirements. No scale is fitted.
- Default `residual_deadband_sigma=1`: discrepancies inside one whitened standard deviation cause **no GPS force**. For this recording that is approximately 5 m. This is a tolerance for occasional long-drift correction, not a promise that GPS detects drift correctly. A common GPS bias remains indistinguishable from global translation.

Quality filtering uses receiver metadata, never PPK or distance from the drifting VIO estimate. Measurement records remain journaled; geometric robust weights are recomputed and can recover after graph corrections. Setting the deadband to zero recovers the original position-factor objective for controlled comparisons.

## Measurement and solver conventions

For bracketing raw keyframes in one robot/session, interpolate translation linearly and orientation by shortest-path SLERP. Predict antenna position `p + R * lever_arm` and whiten its east/north error by the common-ENU horizontal covariance. Exact timestamp matches produce a unary factor. Missing right endpoints remain pending; extrapolation, missing predecessor links, and brackets wider than two seconds are excluded.

GTSAM stores body-to-map poses and uses rotation-then-translation tangent coordinates. The C++ wrapper permutes the complete incoming translation-then-rotation information matrix and uses `BetweenFactorPose3(to, from, T_to_from)`. Analytical derivatives chain native `GPSFactorArm` through the two SLERP endpoints and horizontal projection. The root is either constant or an exact east/north/yaw state, preserving its height and gravity direction. Native C++ tests check the endpoint derivatives numerically, including unary fixes and near-pi rotations.

Let `d² = rᵀ Ω r`, `u = max(0, sqrt(d²) - residual_deadband_sigma)`. The GPS-specific switchable objective is `s² u² + λ(1-s)²`, with exact elimination `s=λ/(λ+u²)`. Default `λ=9`. The Huber comparison uses `δ=3` on `u²`. Native GTSAM robust loss and the adapter Jacobian include the deadband chain rule; diagnostics distinguish switch/Huber robustness from the total optimization weight. A compatible fix can have robust weight one and optimization weight zero. Existing loop/odometry Huber behavior is unchanged.

GPS factors use GTSAM sparse multifrontal Cholesky LM, 20 iterations per update, and one-second scheduling. They trigger solves without loops. Each disconnected graph component is independently initialized; GPS never creates a verified inter-robot loop or merges component identities. Georeferenced warm starts survive joining and restart. Finite solutions with non-increasing robust objective are published; failures preserve the last solution. The active backends no longer depend on the legacy `visloc-slam` pose-graph implementation.

The antenna lever arm is fixed. South-ece nominal optical geometry `[0.1425, -0.1011, -0.0979] m`, composed with the calibrated D455 optical-to-IMU transform, gives `[0.111193231, -0.094903010, -0.078744703] m`. Nominal 20 mm per-axis uncertainty is included conservatively. This is not a calibrated antenna phase centre.

## Transport and recovery

`visloc_multi_robot::{GpsRecord,GpsConfig,Backend::insert_gps}` are independent of ROS. `GpsFix.msg`, `GpsStatus.msg`, and `GetGpsHistory.srv` provide native Rust `rclrs` transport. GPS callbacks feed an independent bounded 2,048-entry queue; the worker validates, journals and publishes records. The graph queue is bounded at 8,192 entries. Overflow counts are exposed, and reliable normalized histories recover missed graph records. Services paginate 64 fixes and use the existing two-second timeout / three-retry policy (four attempts total) with steady time.

Robot/session/sample identities distinguish estimator restarts. Identical duplicates are idempotent and conflicting records are logged and rejected. Journals recover complete prefixes after interrupted writes. The ENU datum and optimized alignment state are persisted; a conflicting datum is an error. Each solution records per-fix associations, rejection reasons, covariance assumptions, residuals, robust/optimization weights, tolerance state and graph revision in `gps_revisions.jsonl` plus the latest snapshot.

The online Rerun recorder uses four batched GPS entities per robot/component (compatible fixes, downweighted fixes, unused fixes, residual segments). Display markers use trajectory height and never depict GPS altitude as trusted. No Rerun viewer or server is started.

## South-ece reproduction

Use the September 27 stereo cache with f64 Basalt, stationary gyro-only initialization, calibrated camera timing, and 0.75 IMU multipliers. The preparation script reads matching GGA/RMC measurement UTC and retains receipt times separately; it does not read PPK. Both timestamps receive the same integer replay offset as the existing camera/IMU cache. There is no timing fit.

```sh
bash scripts/build_gtsam_native.sh
CARGO_NET_OFFLINE=true bash scripts/build_multi_robot_ros2.sh
cargo build --offline --release -p visloc-multi-robot --examples
.runtime/graco-venv/bin/python scripts/prepare_gps_pose_graph.py SOURCE_MISSION NEW_RESULT --backend pose_graph --rate .25
# Source scripts/source_multi_robot_ros2.bash before either native ROS replay.
.runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py NEW_RESULT/gps_off/mission.json --domain-id 222
.runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py NEW_RESULT/gps_on/mission.json --domain-id 223
python3 scripts/run_gps_pose_graph_ablation.py NEW_RESULT
.runtime/evo-venv/bin/python scripts/evaluate_gps_pose_graph.py NEW_RESULT PPK_GROUND_TRUTH.csv
```

The preparation script defaults to GPS-off/on joint BA profiles with keyframe
observation transport enabled. Pass `--backend pose_graph` for the
pose-graph comparison above. The six-way `run_gps_pose_graph_ablation.py`
comparison remains a pose-graph experiment; use the paired ROS missions for BA.

All six backend comparisons share frozen VIO keyframes and verified loops: raw, loops, horizontal GPS with Huber/switchable, loops plus horizontal GPS with Huber/switchable. Evaluation uses external evo, common valid timestamps, 20 ms association, rigid alignment, no scale or time-offset fitting, and the same nominal antenna compensation. Final graph corrected full trajectories and final keyframes are reported separately. PPK supplies positions only, so short-interval displacement-vector error is reported instead of inventing reference orientations for SE(3) RPE. Retain the earlier uncompensated result for continuity. PPK coverage is incomplete and shares rover observations with the GPS input; neither aligned ATE nor the reference supports independent global-accuracy claims.

## References

The architecture follows [VINS-Fusion's separate global estimator](https://arxiv.org/html/1901.03642). The antenna prediction follows [GTSAM's GPS factor with lever arm](https://borglab.github.io/gtsam/gpsfactor/). Receiver thresholds, robust-loss parameters and the residual tolerance above are engineering choices for this experiment, not claims of a universally optimal GPS estimator.
