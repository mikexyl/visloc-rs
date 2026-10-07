# Native Rust multi-robot visual-inertial SLAM

The `robot` and `backend` executables are **Rust ROS2 nodes using rclrs directly**. They call the Basalt, TensorRT, and native GTSAM crates directly. GTSAM global BA with horizontal GPS is the default centralized backend, with PGO as an alternative. The ROS nodes have no C++ wrapper; the solver uses the existing GTSAM C ABI. Python scripts only publish dataset sensors, launch processes, evaluate results, and visualize them.

The ROS-free algorithm crate is `pipelines/multi-robot`. The existing single-robot loop mode remains available unchanged. DA3 is an optional WIP display worker; distributed optimization, hybrid tracking, and joint estimator landmark optimization are excluded.

```mermaid
flowchart LR
    Sensors[Camera and IMU] --> Queue[ROS input queues]
    Queue --> VIO[Basalt worker]
    VIO --> Odom[Continuous raw odometry]
    VIO --> Keyframes[All actual keyframes]
    Keyframes --> Sequences[Ten retained keyframes / five selected views]
    Sequences --> GPU[JIST and XFeat worker]
    GPU --> Retrieval[Sequence retrieval and 5 x 5 refinement]
    Retrieval <-->|Typed announcements and services| Peers[Other Rust robot nodes]
    Retrieval --> Verify[LighterGlue / F-RANSAC / bidirectional P3P]
    Keyframes --> Backend[Central sparse SE3 PGO]
    Verify --> Backend
    Backend --> Outputs[Revisioned poses, components, paths and map corrections]
    Outputs --> Rerun[Rerun recording and evaluation]
```

## Build

Validated on Ubuntu 22.04, ROS2 Humble, Rust 1.94 (minimum 1.85 for this ROS workspace), OpenCV 4.5, and TensorRT 10.13.2. Ordinary visloc builds do not require ROS or the ROS Rust toolchain.

`dependencies.lock.json` and `visloc_ros/Cargo.lock` pin the ROS integration. The rclrs release is exactly 0.7.0; rosidl_generator_rs is 0.5.0 at the recorded commit. Its generated `Sequence<T> -> Vec<T>` conversions need the recorded rosidl_runtime_rs 0.6.1 commit, **not runtime 0.7**, whose ABI differs. The separate Cargo workspace prevents these patches from affecting ordinary visloc builds.

Prerequisites: ROS Humble including development headers, libclang 14, OpenCV development libraries, CUDA/TensorRT, `uv`, and Rust. Standard messages must have compatible generated Rust packages (`share/sensor_msgs/rust/Cargo.toml`, etc.) in a sourced ROS overlay. The validated machine already has them in `/opt/ros/humble`. On a fresh Humble installation, first build the standard interfaces according to the [upstream Humble instructions](https://github.com/ros2-rust/ros2_rust#ros-2-humble-hawksbill), using this repository's pinned rosidl generator/runtime. `ros2_cargo_paths.py` verifies this prerequisite and selects generated crates from the sourced overlay.

From the repository root:

```bash
bash scripts/bootstrap_multi_robot_ros2.sh
bash scripts/build_multi_robot_ros2.sh
source scripts/source_multi_robot_ros2.bash
```

Bootstrap installs Python dependencies, the pinned generator, and extracted test_msgs/Clang packages inside ignored `.runtime/` directories. It does not modify `/opt/ros` or require `sudo`. `colcon-cargo` and `colcon-ros-cargo` build/install the Rust nodes. The local `test_msgs` library works around rclrs 0.7's unconditional vendor linkage. The generated `.cargo` patches work around older colcon plugins expecting a `rust_packages` resource index.

Build the existing five-view JIST, XFeat, and XFeat-trained LighterGlue engines on the deployment GPU:

```bash
.runtime/graco-venv/bin/python scripts/build_loop_tensorrt.py \
  --model-dir /path/to/onnx_model --output .runtime/multi_robot_models
```

If the existing GRACO Python environment is absent, create it with `uv venv --python /usr/bin/python3 --system-site-packages .runtime/graco-venv`, then install the validated replay/evaluation dependencies:

```bash
uv pip install --python .runtime/graco-venv/bin/python \
  numpy==2.2.6 scipy==1.15.3 opencv-python-headless==4.14.0.94 \
  rosbags==0.11.5 rerun-sdk==0.37.2 onnxruntime==1.23.2 PyYAML
```

Use Humble's Python 3.10 and source the ROS setup for `rclpy` and generated interfaces. Python is only needed for replay, evaluation, tests, and visualization; the robot/backend executables are Rust.

Required exports are `JIST_r18_512_seqgem_frames.onnx`, `xfeat_320x224.onnx`, and `lg_320x224_dyn.onnx`. Engines use FP32, no TF32, and exactly 128 real matcher features. TensorRT plans are GPU/SDK-specific and are not committed. The bundle manifest records ONNX and engine hashes; every robot saves the effective model configuration and manifest.

## GRACO replay

For concurrent local replay with Fast DDS, use the supplied shared-memory
profile. It reserves a 64 MiB segment with room for the 1.76 MB raw images and
growing graph messages. The default transport buffers caused repeated sensor
delivery stalls in the four-robot run. This opt-in profile applies to the
processes launched from this shell; live-node configuration is unchanged.

```bash
source scripts/source_multi_robot_ros2.bash
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/visloc_ros/config/fastdds_replay.xml"
export OPENBLAS_NUM_THREADS=1
.runtime/graco-venv/bin/python scripts/prepare_multi_robot_graco.py \
  --output results/graco/my_multi_robot_run --robots 5 6 7 8
.runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py \
  results/graco/my_multi_robot_run/mission.json --domain-id 217
.runtime/graco-venv/bin/python scripts/evaluate_multi_robot.py \
  results/graco/my_multi_robot_run/mission.json
.runtime/graco-venv/bin/python scripts/visualize_multi_robot.py \
  results/graco/my_multi_robot_run/mission.json
```

The mission launcher also starts an independent **online** Rerun map process.
It writes `online.rrd` as VIO runs; add `--rerun-connect rerun+http://HOST:9876/proxy`
to stream to an existing viewer. It never starts a Rerun server. Use
`--no-online-viewer` to omit this process. The `visualize_multi_robot.py` command
above remains a separate, optional final-result playback.

Replay mission directories must be new. A robot restarted with its existing config creates a new session subdirectory, restores its durable keyframe/loop records and sparse-feature archives, and continues serving earlier sessions. `active_session.json` identifies the current output directory. The keyframe capacity includes archived sessions. Use `--robots 5 7 --max-frames 400` for the two-robot smoke test. Defaults are **left camera only**, f64 Basalt with stationary gyro-only initialization, the existing aerial calibration, 800-pixel width, and 0.75 noise/bias multipliers. Right images and ground truth never enter the estimator. The source bags remain unchanged.

Each first camera timestamp maps to a shared playback epoch; integer camera/IMU intervals remain exact. `mission.json` records every original-to-replay timestamp offset. One `/clock` accompanies playback. The first frame can initialize from the first IMU sample at/after its timestamp; an empty first integration interval is allowed, while later empty intervals are errors. The nominal rate is 0.25x with bounded backpressure, so overload slows replay. Robots receive concurrent batches, with at most one unacknowledged camera frame per robot. The final solve waits for VIO, loop workers, and graph history to drain.

The validation summary reports both nominal and effective playback rate. The
replay keeps its executor attached and writes per-batch publication,
acknowledgement-wait and CPU timing to `replay_timing.jsonl`, with median/p95
values in `replay_summary.json`. Add `--profile-replay` to
`run_multi_robot_mission.py` to save a standard Python `replay_profile.pstats`
file for diagnosing publisher/callback delays; this does not change sensor
timestamps or estimator settings.

Set `loop_enabled: false` in a robot JSON config to run the raw-VIO regression control. This bypasses model construction and loop inference without changing sensor preprocessing or Basalt calls.

Use `prepare_multi_robot_graco.py --min-similarity 0.7` to try a different
JIST global cosine threshold. Preparation snapshots the loop configuration
and sets both its `min_similarity` and the backend's
`pgo.min_loop_similarity` to the same value. Without this override, it uses
the loop configuration's value (0.8 by default). Native robot retrieval and
proposal rechecking honor `min_similarity`; geometric verification thresholds
are unchanged. For hand-written missions, set both robot and backend
thresholds consistently. Thresholds must be finite and in [0, 1].
The [full aerial 5–8 threshold comparison](JIST_07_VALIDATION.md) found no
accuracy or connectivity gain at 0.7: matching attempts increased from 128 to
912, while the same three constraints passed verification. The default
therefore remains 0.8.

VIO precision is fixed to f64. The `--scalar-mode` option, ROS `scalar_mode`
configuration field, and core estimator precision selector have been removed.
Remove the old option/field from existing commands/configurations; obsolete
settings are rejected. Previous experiment archives are not modified.

The shared profile validated below 4 m raw ATE RMSE on aerial 5–8 is now the
default when preparing a mission. It
estimates gyro offset from a verified stationary first second and replays all
buffered frames. The 0.75 noise/bias multipliers and 800 px aerial calibration
remain the same. See [full results and limitations](UNIFORM_ATE4_VALIDATION.md).
The [four-robot rerun](MULTI_ROBOT_F64_VALIDATION.md) validates this profile
with online loops and centralized PGO: all four trajectories stay below 4 m
ATE RMSE, while the map remains split into three connected components.
The separate `stationary-motion` mode also averages gravity and waits for
motion, explicitly counting omitted startup frames; it is not a uniformly
passing profile in either precision mode tested. To explicitly bypass startup,
use `--imu-startup legacy` or `"imu_startup": null` in robot JSON. Missing
startup configuration or `{}` enables the default gyro-only gate;
see [startup behavior and reference methods](../docs/imu-startup.md) and
[the initial f32 experiments](IMU_STARTUP_VALIDATION.md).

Compare completed enabled/control trajectories with `python3 scripts/test_multi_robot_raw_parity.py CONTROL_CSV ENABLED_CSV --frames 400 --output parity.json`. This requires identical frame IDs, timestamps, and every raw pose value; it does not align or rescale trajectories to hide differences.

## Live ROS2

Prepare the same calibration and JSON configuration, set `reliable_sensors: false`, and launch:

```bash
ros2 launch visloc_ros multi_robot.launch.py mission:=/absolute/path/mission.json
```

Or run a single native node with `VISLOC_ROBOT_CONFIG=/path/robot.json ros2 run visloc_ros robot`; the backend uses `VISLOC_BACKEND_CONFIG`. Configure every participating robot in `peers`, use a shared `ROS_DOMAIN_ID`, and configure DDS networking normally across hosts. Only the local replay runner forces `ROS_LOCALHOST_ONLY=1`.

Inputs are `/<robot>/camera/image` (`sensor_msgs/Image`: mono8, rgb8, or bgr8) and `/<robot>/imu` (`sensor_msgs/Imu`). ROS remapping supports existing camera/IMU topic names. Images must use the calibrated raw resolution; preprocessing applies area resizing then radtan undistortion. With `preprocess: null`, supply already rectified pinhole images at the calibrated processing resolution. Inputs must arrive in timestamp order within each sensor topic. Sensor callbacks enqueue only; VIO waits for the IMU watermark to cover each image.

Set `camera_mode: "stereo"` to also subscribe to `/<robot>/camera_right/image`.
Stereo inputs require exact matching exposure timestamps and two calibrated
cameras. Provide `preprocess_right` with the right camera's own parameters when
preprocessing raw images; leave both preprocessing fields null for rectified
inputs. Pairing tolerates callback order between topics, waits for both images
and IMU coverage, and bounds pending exposures to 32. Missing/evicted exposures
are counted once, with no monocular fallback. The default remains `"mono"`.
`tracking.jsonl` records left/right observation counts and the input mode for
every processed frame. Sequence retrieval and verification use the left camera;
stereo observations contribute to VIO and its metric landmarks.

Live inputs use best-effort sensor QoS; replay uses reliable QoS. SLAM records are reliable. Current status, corrected paths, and graph snapshots are transient-local. Communication deadlines use monotonic time; estimation uses message timestamps. Basalt never consumes corrected poses.

## Sequence and verification contracts

- Every actual VIO keyframe enters the odometry graph. For retrieval, skip a keyframe only when more than 95% of its triangulated landmark IDs occur in the last retained keyframe. This runs before feature subsampling; an empty set is retained.
- Each non-overlapping block contains ten retained keyframes. Enumerate all 56 first/last-plus-three selections, minimizing squared consecutive camera-center distances, with lexicographic ties. Missing keyframe ordinals reset incomplete blocks; intentional skips do not.
- Broadcast a normalized 512-dimensional JIST sequence descriptor, selected/member identities, local covisibility neighborhood, and model identity. Cache all five normalized frame descriptors.
- Before scoring, exclude same-session overlapping/adjacent blocks, the 20-second neighborhood, and two-hop covisibility with at least 15 shared landmarks per edge. All sequence members participate. IDs and times never exclude a different robot/session. Keep cosine similarity above the configured inclusive threshold (0.8 by default) and up to three distinct neighborhoods.
- Trigger retrieval on local completion and remote announcements/history. Canonical sequence pairs assign one verification owner by robot ID; that robot also serves its archived sessions. Exchange the frame matrix first; maximize the entire 5x5 similarity matrix, then fetch only that pair's sparse packets.
- Sample XFeat at measured VIO feature positions. Fewer than 128 real features produces an empty feature packet and a rejected verification, never padding. Match with the existing trained LighterGlue engine.
- Fundamental-matrix RANSAC precedes both P3P directions and pose refinement. Require at least 15 PnP inliers, a 0.5 ratio among that direction's available 2D–3D correspondences, a 3-pixel RANSAC threshold, mean error <=2.25 px, 2% image-area coverage and four occupied 4x4 cells. Either direction can pass; choose more inliers, then lower reprojection error. Matching attempts are journaled before GPU execution and survive restart. Match a canonical sequence pair once; the old temporal confirmation gate is disabled.
- Landmarks are frozen in their observing camera's metric frame. Every endpoint carries its own intrinsics and camera-to-body extrinsics. No common global alignment is needed for verification.

Composite IDs are `(robot, session, local_id)`; every estimator start generates a new UUID session. Robot-local track IDs are scoped by the enclosing feature packet's robot/session. Typed interfaces are in `visloc_msgs/msg` and `srv`. Sequence, selected-feature, and paginated-history services live under `/<robot>/slam/`. Requests wait at most two seconds per attempt, with up to three retries after the first attempt. Timed-out clients are replaced because rclrs 0.7 otherwise retains unanswered requests after their promises are dropped. Failed requests are logged and deferred for reconnection; no feature matching is repeated after a successfully submitted verification. History cursor changes recover robot restarts.

## Graph conventions and execution

Edges store **T_to_from**, transforming points from the source body to the destination body. Information matrices are row-major, with tangent ordering **translation, then rotation**. Sequential measurements always derive from raw VIO poses. Stable internal solver IDs map composite identities.

The central backend reuses visloc's sparse SE(3) LM with Huber delta 3, 20 iterations, and no more than one solve per second while dirty. Disconnected components solve separately with one anchor each. A verified bridge initializes the joining alignment and removes the redundant anchor. Only finite solutions whose robust objective does not increase are published. Failure retains the last published solution. The graph journal supports restart and peer history recovers its missing tail.

Defaults are odometry 0.15 m / 0.03 rad and loop 0.20 m / 0.04 rad. These are **tuning parameters, not measured covariances**. Configure them in backend `pgo` fields `odometry_translation_sigma`, `odometry_rotation_sigma`, `loop_translation_sigma`, and `loop_rotation_sigma`.

The active implementation uses sequential raw-VIO edges and verified relative-pose
loops, with the corrected SE3 log Jacobian. The covisibility-Schur and pose-only
reprojection experiments are preserved on branch
`experiment/covisibility-schur-reprojection` (snapshot `668b0ef`). Their reports
and code live on that branch; saved experiment data remains under `results/okvis2/`.

Raw `/<robot>/vio/odometry` and `robot/odom -> robot/base_link` stay continuous. Revisioned `/visloc/graph`, `/<robot>/slam/path`, and `map_<component robot>_<session>_<root keyframe> -> robot/odom` provide corrections. Including the root keyframe distinguishes temporary disconnected fragments within the same session when records arrive out of order.

Queues are bounded: sensor ingress 8192, pending camera images 32, loop worker 4, ROS events 1024, communications 512, graph updates 8192, graph results 2. A full image buffer drops the oldest image and counts it. A full keyframe queue counts a dropped loop keyframe; its ordinal gap resets sequence construction while the odometry keyframe remains in the graph. Sensor ingress overflow reports an error. The configured 4000-keyframe capacity bounds per-robot feature history; exceeding it reports an error rather than silently evicting data needed by a peer.

## Diagnostics, evaluation, and tests

Outputs include raw/corrected trajectories, calibration snapshots and model hashes, effective configs, frozen sequence feature archives, retrieval decisions, selected pairs, geometric outcomes, VIO/encoding/verification timing, queue drops, service failure counts, graph revisions, and component memberships. Exchange-through-verification latency includes descriptor/feature service calls and GPU queueing for a successful attempt, measured with steady time; it excludes earlier failed attempts and retrieval backlog. Replay counts actual serialized topic payload bytes. Each requesting node separately records service calls and typed field payload bytes, explicitly excluding CDR framing/padding and DDS overhead. These are application payload measurements, not total network traffic. `communication.json` also counts sensor-ingress, communication, and backend graph queue overflows; image-buffer and loop-keyframe drops are in robot status.

`evaluate_multi_robot.py` reports metric SE(3) ATE: per-robot raw and corrected results, plus **one rigid alignment for each connected component**, with no scale fitting for reported accuracy. Existing evaluator Sim(3) fields are diagnostic only. Dense corrected poses interpolate graph corrections. Rerun records robot-colored raw/corrected trajectories, selected frame pairs, sparse landmarks, loop edges, and disconnected-component status. Only robots sharing a component appear in a shared map; isolated trajectories retain separate views.

The online sparse map renders one point per `(component, robot, session,
landmark ID)`. Every actual VIO keyframe supplies its measured pixels, metric
camera-frame landmarks, calibration, and raw body pose through a dedicated
display writer, including when loop closure is disabled. This writer has a
four-packet queue with nonblocking submission; a slow disk drops display packets
and reports them in `visualization_status.json`. The viewer follows its flushed
`visualization.jsonl` and the backend's atomically published graph snapshots.
This is local display transport; robot/backend communication remains ROS2.

Every processed camera frame also supplies a matched body pose and the same
undistorted grayscale images consumed by VIO. A separate four-packet camera
worker encodes JPEGs (quality 90) and then flushes `camera_frames.jsonl`; encoding
and disk I/O never run on the estimator thread. `camera_status.json` reports
written/dropped image pairs. The camera stream is independent of keyframe
selection and loop inference. Monocular mode records only the left camera;
stereo records both with their own intrinsics and camera-to-body transforms.

Rerun logs `component -> body -> cam0/cam1 -> image`, with explicit optical
RDF coordinates (+X right, +Y down, +Z forward), calibrated pinholes, and images
on the same exact `sensor` timestamp as their body pose. The `sensor` playback
timeline preserves every exposure; `live` additionally records arrival and map
processing time. Camera poses use the latest available map/odometry correction
at or before their timestamp, scoped to the robot/session. A late graph revision
also moves the held camera after input drains. Component changes clear the old
camera entity. The default blueprint includes a map overview, a **Follow camera**
3D view tracking the left pinhole, and separate left/right image views.

The display process refines changed landmarks against observations received so
far with the **current** camera poses held fixed. It does not wait for the final
graph. Defaults require two distinct keyframes, 1 degree of viewing-ray parallax,
and at most 3 px reprojection error in 60% of retained views. Each observation
uses its own intrinsics and camera-to-body transform. Duplicates add no support;
IDs never merge across robots or sessions. Tracks without a metric VIO estimate
are omitted. Bounds are 30,000 tracks, eight views per track (oldest plus latest),
and 20,000 trajectory records. Least recently observed tracks are evicted at
capacity and counted. The refinement queue coalesces updates by landmark ID;
each tick processes at most 64 tracks or 100 ms, whichever comes first (a running
single-point solve completes). Rerun clouds publish at most once per second.

Retained points are anchored to an observing body pose: graph corrections move
them immediately, then queue geometric refinement. Point inspection marks
`refinement_pending` until that work finishes. A component merge clears retired
entities rather than leaving duplicate surfaces. `online.map.jsonl` records live
point counts, pending work, memory bounds, rejection reasons, and evictions.
Older binaries without the display journal are supported by following selected
sequence feature archives as they arrive, at the lower sequence-completion rate.

Attach a viewer to an already running mission with:

```bash
python scripts/visualize_multi_robot_live.py /path/to/mission.json \
  --output /path/to/new/online.rrd --connect rerun+http://HOST:9876/proxy
```

Use `--min-observations`, `--min-parallax-deg`, `--reprojection-px`,
`--max-tracks`, and `--max-views` to configure the live display. Set the robot
config's `visualization_enabled` to false to disable its display writer for
parity measurements. Neither the writer nor viewer changes Basalt inputs/state.

For WIP dense mapping, add `--da3-config configs/graco/da3_five_view.json` to
the mission launcher or live recorder. DA3 uses five actual VIO keyframes and
the same robust landmark filtering before its coverage gate. The supplied
profile uses pose-only depth scale, strict confidence/reprojection filtering,
and camera-local clouds that follow PGO. See [DA3 depth](../docs/online_da3_depth.md)
for queue diagnostics, input conventions, archives and current limitations.

This is a visualization refinement; the backend still optimizes poses, and the
refined display points are not fed to VIO or loop verification. Single-view and
weak-parallax landmarks are hidden by default to prioritize stable geometry.
The optional offline viewer uses the final graph and exports `.landmarks.csv`
and `.landmarks.json` beside its RRD for inspection:

```bash
python scripts/visualize_multi_robot.py /path/to/mission.json \
  --output /path/to/new/playback_refined.rrd
```

Use `--landmark-min-observations`, `--landmark-min-parallax-deg`, and
`--landmark-max-reprojection-px` to adjust display quality. The optional
`--show-archived-landmarks` adds the old unmerged estimates in a separate faint
diagnostic layer. Saving an RRD does not start a Rerun server.

```bash
cargo test -p visloc-multi-robot
cargo test -p visloc-online-loop --features native
source scripts/source_multi_robot_ros2.bash
ROS_DOMAIN_ID=219 ROS_LOCALHOST_ONLY=1 .runtime/graco-venv/bin/python scripts/test_multi_robot_ros2.py
ROS_DOMAIN_ID=220 ROS_LOCALHOST_ONLY=1 .runtime/graco-venv/bin/python scripts/test_multi_robot_robot_ros2.py /path/to/a05.json
ROS_DOMAIN_ID=221 ROS_LOCALHOST_ONLY=1 .runtime/graco-venv/bin/python scripts/test_multi_robot_reconnect_ros2.py /path/to/completed/6_7/mission.json
ROS_DOMAIN_ID=217 .runtime/graco-venv/bin/python scripts/probe_multi_robot_ros2.py /path/to/running/mission.json
cargo build --release -p visloc-tensorrt --features native --example infer_fixtures
.runtime/graco-venv/bin/python scripts/validate_loop_tensorrt.py --bundle .runtime/multi_robot_models
cargo run --release -p visloc-multi-robot --features native --example compare_refinement -- \
  /path/to/completed/mission_directory .runtime/multi_robot_models/loop_config.json
```

The comparison reuses exactly the recorded retrieval pairs and frozen feature packets, rerunning only matching/verification for 5x5 refinement versus fixed-last-frame pairing. It cannot credit different retrieval candidates to refinement. Both variants also write a batch-optimized graph, using identical raw odometry and solver settings; evaluate each with `evaluate_multi_robot.py MISSION --graph refinement/refined_graph.json --output refinement/refined` (and `fixed_last`). These controlled batch results are distinct from the actual online graph. The TensorRT validator reports exact match agreement and separately identifies FP32 membership changes within 1e-5 of the matcher's 0.1 cutoff; all stable matches must agree.

After evaluation, `scripts/summarize_multi_robot.py MISSION` writes `validation_summary.json` with accuracy, connectivity, matching outcomes, latency percentiles, drops, and payload accounting. It checks that all emitted keyframes/loops reached the final graph and each submitted refinement received exactly one verification result. The reconnect test requires a completed archive containing a cross-robot loop; it copies a single pair into a temporary fixture and leaves the original experiment untouched.

Status and communication snapshots use atomic replacement. For legacy runs with interrupted diagnostic writes, `--allow-incomplete-traffic` produces a summary with explicitly unavailable nodes and partial service totals; it never reconstructs missing counters as measurements.

Validation results and known dataset limitations are recorded separately in `VALIDATION.md`. Successful transport or synthetic graph tests do not establish improved dataset accuracy.

## Recorded RealSense inputs

`scripts/prepare_realsense_slam.py --bag BAG --output NEW_DIRECTORY` prepares
the same native Rust sequence/refinement pipeline for the Jetson's ROS2 MCAP
recordings. It uses the D455 calibration, left IR only at 640×480, f64 stationary
gyro-only initialization, 0.75 IMU noise/bias multipliers, and JIST threshold 0.8.
Separate acceleration samples are interpolated onto gyro timestamps; original
integer timestamps and lossless compressed images are retained in a replay cache.
Recorded receiver GPS is normalized from GGA/RMC UTC and included in the default
GTSAM global BA backend with its calibrated nominal antenna lever arm. Use
`--no-gps` for visual BA; missing GPS also leaves visual BA available. PPK is never
an estimator input. Run the generated `mission.json`
through `scripts/run_multi_robot_mission.py`; visualization uses the same cached
images. The default nominal playback rate is 1×, configurable with `--rate`.
Use `--camera-mode stereo` to cache both IR streams and replay only exact-stamp
pairs. The preparation audit reports paired and unpaired image counts; the
original per-camera stamps remain in the cache. Both modes use the same f64
estimator, initialization, IMU parameters, and sequence/refinement backend.
If capture starts before synchronized IMU is available, pass
`--trim-imu-boundaries` to select only images inside IMU coverage. The excluded
timestamps are audited and the images remain in the cache; replay honors the
mission's selected first/last timestamps without extrapolating IMU samples.

The [South-ece-rtk-test run](SOUTH_ECE_SLAM_VALIDATION.md) completed all 9,364
frames but **failed accuracy**, with late VIO divergence and no accepted loops.
GPS is unreliable on this dataset and is excluded from evaluation. The
[stereo-inertial rerun](SOUTH_ECE_STEREO_VALIDATION.md) avoided the monocular
runaway speed but still accepted no loops. No ATE is claimed. Full logs, calibration/model
hashes, trajectories, image evidence, and a Rerun recording are retained.

The [cuVSLAM Python comparison](SOUTH_ECE_CUVSLAM_COMPARISON.md) runs NVIDIA's
stereo-inertial odometry on the same paired images and IMU cache. The final
cuVSLAM run develops severe late drift; the report includes input-parity checks,
failure diagnostics, timing limitations, plots, and a Rerun comparison. The
optional NVIDIA environment is isolated from ordinary visloc and ROS builds.

The dataset path was replaced on 2026-09-27. The
[replacement-recording rerun](SOUTH_ECE_STEREO_20260927_VALIDATION.md) processed
7,182 stereo pairs with zero runtime drops and one accepted loop from 17
verification attempts. Two initial pairs were excluded because IMU coverage
had not started. The previous VIO/cuVSLAM results remain archived and describe
the older recording; GPS remains excluded and no ATE is claimed.

A subsequent [evo evaluation against the supplied PPK reference](SOUTH_ECE_EVO_20260927.md)
reports 0.781 m raw and 0.787 m loop-corrected translation RMSE, with rigid
alignment and no scale fitting. It covers 656 matched reference epochs;
reference gaps, strict quality flags, and the uncompensated antenna offset are
documented. This offline evaluation does not feed GNSS into SLAM.

## Default global bundle adjustment with GPS

Both mission preparation scripts default to `global_bundle_adjustment`, sending
full calibrated keyframe observations through Rust ROS2. RealSense missions
enable receiver GPS by default; GRACO uses visual BA because it has no receiver
stream. `--backend pose_graph` selects the alternative PGO backend. Robot BA
observations and GPS ingestion are enabled by default, while explicit overrides
remain supported. See [configuration, scope, and validation](../docs/global_bundle_adjustment.md).
