# ScaRF-SLAM dense reconstruction

ScaRF-SLAM replaces the custom DA3 dense mapping experiment. Visloc continues
to produce stereo-inertial body poses. The adapter supplies rectified **RGB**
images, calibrated intrinsics, and camera poses to upstream ScaRF. ScaRF owns
depth prediction, frame scale optimization, submap scale optimization, matching,
filtering, and point cloud fusion. No VIO landmarks are passed to the mapper.

Two entry points are available: an offline reconstruction from completed poses,
and an online mapper following VIO and GPS global BA during ROS2 dataset replay.
Neither changes estimator poses or feeds dense points back into tracking.
JIST/XFeat/LighterGlue loop closure and the sparse Rerun recorder remain in the
native visloc mission.

## Source and environment

`third_party/ScaRF-SLAM` is a Git submodule pinned to the working experiment's
revision `e0a34267d575d8b21d548ea8ed3cf5782192bac3`. Its source and GPLv3 license
remain in that repository. The adapter executes it in a separate process.
`tools/scarf/gpu_configuration.patch` exposes model, dtype, and inference
resolution settings; it preserves upstream defaults and estimation algorithms.
The runner applies the patch to an archived source copy inside each result
directory, leaving the submodule clean.

```bash
git submodule update --init third_party/ScaRF-SLAM
bash tools/scarf/setup.sh
```

Setup uses `uv`, Python 3.11, and an isolated `.runtime/scarf/venv`, pins DA3 to
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, and downloads DA3-Large snapshot
`c54c26b16ec04d218e8d584ecf4bce082a9fcc20`. CUDA/PyTorch GPU access is required
for mapping. No OpenVINS, ROS node build, TensorRT conversion, or Rerun server
is required. The existing validated ScaRF Python environment can also execute
the adapter directly; pass its downloaded model directory with `--model`.

## Online VIO, loop closure, and GPS global BA

Prepare a fresh mission, then let the adapter launch it:

```bash
.runtime/graco-venv/bin/python scripts/prepare_realsense_slam.py \
  --bag /data/ucy/South-ece-rtk-test \
  --output results/ucy/south_ece_scarf_online_new \
  --camera-mode stereo --trim-imu-boundaries \
  --backend global_bundle_adjustment --rate 1.0
env -u PYTHONPATH .runtime/scarf/venv/bin/python scripts/run_scarf_online.py \
  results/ucy/south_ece_scarf_online_new/mission.json \
  --output results/ucy/south_ece_scarf_online_new/scarf --launch
```

ROS2 and the native visloc binaries must already be built as described in
[global BA](global_bundle_adjustment.md). `--launch` sources the ROS environment
only for the mission subprocess; ScaRF keeps its own Python 3.11/CUDA runtime.
Omit `--launch` to follow an already running mission. A completed mission is
rejected; use the offline entry point for completed inputs. This first live
adapter supports one robot/session and the D455 RGB topic/calibration.

The nodes communicate over ROS2. ScaRF consumes the same flushed camera-pose
journal and atomically published graph snapshot used by visloc's online
visualizer. It does not reconstruct a completed corrected trajectory afterward.
RGB pixels are cached from the dataset before launch, but an image cannot enter
mapping until its bracketing VIO poses have arrived. Ground truth is not read.
Raw camera poses drive ScaRF's motion selection; the latest received BA revision
conditions depth. A new accepted loop triggers GPS BA; its successful result
then triggers ScaRF correction of existing submap anchors and scales.

For intermediate RGB timestamps, the adapter interpolates the available
`map_from_odom` transforms (translation and quaternion SLERP) and applies the
result to the calibrated raw RGB pose. Beyond the optimized frontier it holds
the last received correction. Robot/session identities are checked; a restart
requires a new map. BA snapshots received while ScaRF is busy are coalesced to
the latest revision. The final revision is drained before saving the cloud,
without forcing an extra optimization.

`tools/scarf/live_input.patch` adds input polling/waiting hooks to upstream's
existing processing loop and enables its existing rotation-change check for
live corrections. Quaternion comparison uses chord distance to avoid the
near-zero roundoff of `acos(dot)`. Bootstrap uses only already arrived poses, then enables
native `use_slam` processing. Successful optimized loop IDs gate map correction,
so pose publications, duplicate deliveries, and unchanged loops cannot trigger
another solve. In live mode the periodic, per-insertion submap-scale, and
unconditional final map solves are disabled. Depth inference, within-batch frame
scale fitting, submap construction, and fusion continue between loops. The
optimization mathematics and the offline workflow remain upstream methods.
Before the first loop the mapper uses raw VIO poses; after it, new poses carry
the most recent BA correction. Rebuild the ROS nodes before using this adapter.

The `scarf/` output contains `online_events.jsonl` with received revisions,
submap creation, correction timings, and input counts; `mapping.log` contains
upstream optimization diagnostics. `mission.log` and the mission's native logs,
`replay_summary.json`, `backend/revisions.jsonl`, and GPS diagnostics establish
loop counts, dropped inputs, actual GPS factors, and backend completion.
`summary.json`, `preview.png`, and `mapping/recon/visloc_slam/` hold the final
map. The normal mission recorder writes `online.rrd` for sparse visualization;
no Rerun server is launched.

`--rate` is a replay pacing limit, not a real-time performance guarantee.
The recorder does not wait for dense inference; ScaRF may trail the sensor
frontier while a loop-triggered map correction is running.

## South-ece from the source recording

```bash
env -u PYTHONPATH .runtime/scarf/venv/bin/python scripts/run_scarf_mapping.py \
  --bag /data/ucy/South-ece-rtk-test \
  --body-trajectory /path/to/completed/trajectory.csv \
  --calibration configs/realsense/d455_1/camchain-imucam.yaml \
  --output results/ucy/south_ece_scarf_new
```

Use a fresh output directory. The CSV must contain `timestamp_ns, tx, ty, tz,
qx, qy, qz, qw`: finite world-from-IMU/body poses with strictly increasing
integer timestamps on the **recording IMU clock**. Raw or corrected visloc
trajectories may be supplied explicitly. Do not pass camera poses or GPS
ground truth. Rebased mission exports require `--pose-time-offset-ns`: add
the negative of the mission's replay offset to recover recording time.
Global BA JSON snapshots are not this CSV format and cannot be passed directly.

The default RGB topic is `/camera/camera/color/image_raw/compressed`, and
Kalibr `cam2` is the RGB camera. The adapter samples at approximately 5 Hz,
undistorts RGB once using its own intrinsics, and computes
`T_world_rgb = T_world_imu * inverse(T_rgb_imu)`. It applies the calibrated
RGB-to-IMU time correction once, interpolates body translation and orientation
(SLERP), and rejects extrapolation or interpolation over gaps above 0.25 s.

Timestamp rebasing uses integer subtraction **before** floating-point
conversion. Both the image filenames and camera trajectory use this local
clock. `timestamps.csv` preserves the exact mapping to the source image and
recording IMU clocks.

For the already prepared images in the validated ScaRF workspace:

```bash
env -u PYTHONPATH /home/mikexyl/workspaces/scarf_slam_ws/.venv/bin/python \
  scripts/run_scarf_mapping.py \
  --images /home/mikexyl/workspaces/scarf_slam_ws/data/south_ece_rtk/rgb_rectified \
  --image-clock-to-imu-ns=-9591775 \
  --body-trajectory /home/mikexyl/workspaces/scarf_slam_ws/results/south_ece_rtk_visloc/vio/trajectory.csv \
  --model /home/mikexyl/workspaces/scarf_slam_ws/models/DA3-LARGE \
  --output results/ucy/south_ece_scarf_prepared
```

Prepared images must already be undistorted with the supplied RGB intrinsics.
Their image clock offset is mandatory, since the validated workspace names
them on the IR0 clock. The adapter copies their bytes without another
rectification or JPEG encoding. `--max-images 180` runs a short smoke test;
`--prepare-only` checks inputs without invoking the GPU mapper.

## Configuration and outputs

`configs/scarf/offline.yaml` retains the validated settings: DA3-Large float32
weights at resolution 336, four views per submap, one shared view, frame and
submap scale optimization, and point cloud fusion. Candidate frames must meet
ScaRF's time/motion thresholds; they are not restricted to visloc keyframes.
This is upstream PyTorch inference with its native mixed precision behavior.

The output contains:

- `config.yaml`, `calibration.yaml`, `inputs.json`, `timestamps.csv`, and
  `trajectory_rgb.txt`: effective calibration, clock conversion, and inputs.
- `run.json`, `packages.txt`, `scarf_source/`, and `mapping.log`: command,
  source revisions, patch/model hashes, environment, runtime, and diagnostics.
- `mapping/recon/visloc/`: upstream point cloud, optimized submap graph, and
  selected camera poses.
- `summary.json` and `preview.png`: finite point count, mapped coverage, tail
  gap, submap scales, and a static map preview. No viewer server is launched.

Incomplete final submaps can leave a tail gap; it is reported instead of
silently claiming complete coverage. A completed finite cloud is not an
independent metric reconstruction-accuracy measurement.

The old `--da3-config` options, depth workers, VIO-landmark depth alignment,
depth-specific TensorRT tools, and associated tests/configurations are removed.
The historical implementation is recoverable at commit `1e46ecf` on
`experiment/global-bundle-adjustment`.

## Validation, 2026-10-07

The adapter completed the full South-ece recording using the validated f64
stereo-inertial raw VIO trajectory (7,182 body poses, no GPS/loop correction).
The run is saved under `results/ucy/south_ece_scarf_20261007`.

| Measurement | Result |
| --- | ---: |
| RGB images with interpolated poses | 1,195 |
| Selected mapping frames / submaps | 520 / 173 |
| Finite exported points | 23,334,815 |
| Mapped duration / final tail gap | 231.446 s / 7.803 s |
| Mapper wall time, including final solve/export | 219.65 s |
| Final submap scales, min / median / max | 0.444 / 0.994 / 1.166 |

All 1,195 exported RGB JPEGs are byte-identical to the independently prepared
working ScaRF input. Camera poses differ by at most 3.72e-9 m and 3.05e-9 rad;
the adapter now rebases timestamps and retains greater decimal precision.
Matching and GPU inference can vary slightly between runs; identical input
geometry does not imply bit-identical reconstruction.

Thirty focused tests passed: three input time/frame contract tests and 27
existing calibration, sparse landmark, and camera visualization checks. CLI
imports and removed-hook checks passed. This validates the integration and
finite outputs, not dense metric accuracy. No Rerun server was started.

The isolated project environment was also installed and checked with a
90-image GPU run: seven submaps and 742,543 finite points in 19.81 s. This uses
the project-local DA3 checkout and `.runtime/scarf/venv`, without relying on
the original workspace's Python installation.

## Online GPS-BA validation, 2026-10-07

The initial complete online run, **before loop-triggered scheduling**, is saved under
`results/ucy/south_ece_scarf_online_gps_ba_20261007`. It used the then-current mission
defaults: stereo f64 Basalt, stationary gyro-only startup, calibrated D455
intrinsics/extrinsics, 0.75 IMU noise/bias multipliers, JIST similarity 0.8,
and native GTSAM global BA with recorded receiver GPS. ScaRF uses the same
DA3-Large profile as the offline experiment.

| Measurement | Result |
| --- | ---: |
| VIO input/output frames | 7,182 / 7,182 |
| Keyframes / accepted loops | 1,015 / 1 |
| Active GPS factors / optimized landmarks | 104 / 6,549 |
| Dropped images / dropped keyframes / request failures | 0 / 0 / 0 |
| Final backend revision | 294 |
| Dense submaps / finite points | 175 / 23,278,312 |
| Submaps created before all camera poses arrived | 136 |
| First submap after mission launch | 21.66 s |
| Dense tail gap | 7.403 s |
| ROS2 mission / end-to-end wall time | 814.92 s / 1,162.74 s |

`online_validation.json` checks backend mode, loops, actual GPS factors, raw
coverage, drops, robust objective acceptance, and final revision agreement.
Every saved ScaRF camera pose agrees with its final GPS-BA-corrected RGB input:
maximum position difference 6.20e-11 m and rotation difference 1.83e-12 rad.
This is an integration consistency check, not a dense accuracy measurement.

The initial run exposed an adapter performance issue: the upstream `acos(dot)`
rotation comparison reported approximately 1.71e-6 degrees for identical
quaternions, causing redundant global submap solves at the unchanged final
revision. The live patch now uses a stable quaternion chord comparison, tested
with identical/sign-flipped poses and a genuine rotation-only change. Six
focused adapter tests pass. The full run above predates this numeric fix and
loop-triggered BA/ScaRF scheduling; its timing includes the redundant solves,
and no improved full-run runtime is claimed.
It demonstrates online reconstruction and correction, not real-time throughput.

The loop-triggered implementation was checked with a fresh 400-frame South-ece
ROS2/GPU replay in `results/ucy/south_ece_loop_trigger_smoke_20261007_172815`.
It ingested 67 recorded GPS fixes, accepted no loops, and completed with zero
BA solves and zero ScaRF map solves, including final drain. Incremental mapping
produced three submaps and 400,076 points, with no dropped images. The mission
took 31.88 s and the mapper finished at 33.82 s; this short check is not a
full-sequence throughput comparison. Backend tests exercise a new accepted loop,
delayed endpoints/observations, duplicate delivery, restart, and correction
propagation; the patched ScaRF refresh test checks that only a new solved loop
permits map correction. All 35 multi-robot tests and six adapter tests passed.

## Parallel native implementation

The independent `mikexyl/native-slam` package contains native Basalt, TensorRT loop closure, GTSAM GPS BA and an experimental C++ ScaRF port. South-ece visual inspection found poorer dense-map quality in that port than this original Python ScaRF workflow. Both are retained; this integration is not superseded by the C++ implementation. Dense-quality parity remains unresolved.
