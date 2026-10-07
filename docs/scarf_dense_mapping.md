# Offline ScaRF-SLAM dense reconstruction

ScaRF-SLAM replaces the custom DA3 dense mapping experiment. Visloc continues
to produce stereo-inertial body poses. The adapter supplies rectified **RGB**
images, calibrated intrinsics, and camera poses to upstream ScaRF. ScaRF owns
depth prediction, frame scale optimization, submap scale optimization, matching,
filtering, and point cloud fusion. No VIO landmarks are passed to the mapper.

This first integration is **offline**: run it after VIO/SLAM finishes. It does
not subscribe to live VIO, modify estimator poses, or feed dense points back
into tracking. The online sparse Rerun recorder, GPS/global BA backend, and
JIST/XFeat/LighterGlue loop closure remain independent.

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
