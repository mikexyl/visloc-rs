# Jetson Docker deployment

Native ARM64 CPU build of the Basalt stereo-inertial EuRoC runner. The default
CPU target is Cortex-A78 (Orin/A78AE); set `TARGET_CPU=generic` for other ARM64
boards. CUDA and the NVIDIA container runtime are not required by this pipeline.
The runtime is Ubuntu 24.04 and the build uses the repository's Rust 1.94.0 pin.

## Deploy from the development machine

```bash
ssh mikexyl@192.168.0.156 'mkdir -p ~/workspaces/visloc-rs ~/data/euroc/MH_01_easy/mav0'
rsync -az --exclude=target --exclude=.git --exclude=work --exclude=docs \
  --exclude=benchmarks --exclude=__pycache__ ./ \
  mikexyl@192.168.0.156:workspaces/visloc-rs/
rsync -a /data/euroc/MH_01_easy/mav0/ \
  mikexyl@192.168.0.156:data/euroc/MH_01_easy/mav0/
ssh mikexyl@192.168.0.156 \
  'cd ~/workspaces/visloc-rs && bash docker/jetson/build.sh'
```

Build on the board, not on the x86 development host. The image tag defaults to
`visloc-rs:jetson`. `VISLOC_IMAGE`, `BUILD_JOBS` (default 4), `TARGET_CPU`, and
`SOURCE_REVISION` can be set when building. The build context excludes local
artifacts and datasets. No existing containers need to be stopped.

## Run on the Jetson

```bash
cd ~/workspaces/visloc-rs
# Short functional check:
bash docker/jetson/run.sh ~/data/euroc/MH_01_easy "$PWD/results/mh01-smoke" --max-frames 80
# Full sequence:
bash docker/jetson/run.sh ~/data/euroc/MH_01_easy "$PWD/results/mh01-full"
```

Choose a new output directory for each run. Outputs include `trajectory.tum`,
`trajectory.csv`, `summary.txt`, `timing.txt`, `run.log`, and `evaluation.json`
(when ground truth exists). Trace and mapper packets are disabled. Calibration
and configuration default to the official EuRoC Double Sphere variant bundled
with the image. Ground truth is read only after VIO exits, by the evaluator.

For custom calibration, use `docker run` directly and supply read-only mounts
plus `VISLOC_CALIBRATION` and `VISLOC_CONFIG` environment variables. For direct
binary access, override the entrypoint with `--entrypoint basalt_euroc_vio_demo`.

The batch container exits on completion. The image remains installed for
subsequent runs; it is not a camera/ROS service. The Python Rerun playback tool
runs on the development machine using copied-back output trajectories and the
local EuRoC images (see `docs/rerun_euroc_viewer.md`).

## Installed deployment (2026-09-16)

- Host: `mikexyl@192.168.0.156`, ARM64 Cortex-A78AE, L4T R39.2 / Ubuntu 24.04.
- Workspace: `/home/mikexyl/workspaces/visloc-rs`.
- Image: `visloc-rs:jetson`.
- Validation input: `/home/mikexyl/data/euroc/MH_01_easy_80` (first 80 stereo frames).
- Validation output: `/home/mikexyl/workspaces/visloc-rs/results/mh01-smoke`.
- Build log: `/tmp/visloc-jetson-build.log`.

Only the compact validation dataset was deployed. Transfer the full `mav0/`
directory using the command above before attempting a full-sequence run. The
ROS bag/database alongside `mav0/` is not needed.

Validation completed successfully: 80/80 frames, 8.39 s wall time, 17,216 KiB
peak RSS, exit code 0. The evaluator matched 58 poses to ground truth and
reported 0.00523 m rigid ATE RMSE on this short initialization segment. This is
a functional smoke check, not full-sequence accuracy or real-time validation.
The native ARM64 runtime image ID was
`sha256:264098c00cd1fc79b2dbc747f7cf80f1d1f1d17a53dba87f01092780750b84fb`.

## Viewer/map update (2026-09-17)

The host viewer and `visloc-rs:realsense-browser` image now include accumulated
map visualization and live CPU/RAM/GPU telemetry. Start/Stop controls remain in
the integrated viewer. The Orin NX GPU load path under `platform/bus@0` is
supported, and memory is correctly labeled as shared system RAM.

The ARM64 image passed an 80-frame recorded EuRoC sensor-stream smoke test
without opening the camera. Deployment details, image IDs, backups, and
screenshots are in [the deployment report](../../work/jetson_viewer_update_20260917/REPORT.md).
