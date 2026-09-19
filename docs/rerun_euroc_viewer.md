# EuRoC VIO browser viewer

`scripts/view_euroc_rerun.py` exports completed Basalt VIO results into a Rerun
recording. This is a Python playback tool for the Rust estimator; it does not
add a Rerun dependency to the Rust workspace or change VIO estimation.

## Setup and run

From the repository root:

```bash
uv venv target/rerun-venv --python python3
uv pip install --python target/rerun-venv/bin/python -r scripts/requirements-rerun.txt

target/rerun-venv/bin/python scripts/view_euroc_rerun.py \
  --euroc-dir /data/euroc/MH_01_easy \
  --run-dir target/euroc_vio/MH_01_easy

target/rerun-venv/bin/rerun --web-viewer --bind 127.0.0.1 \
  --port 9877 --web-viewer-port 9095 --server-memory-limit 2GiB \
  target/euroc_vio/MH_01_easy/playback.rrd
```

Use the bottom timeline to play or scrub the recording. The layout contains a
3D trajectory view, left/right camera images, position error, and visual
observation counts. Blue is estimated motion and gray is ground truth.

The input run must contain `trajectory.csv` and `trajectory.tum`, as produced by
`bash scripts/run_local_euroc_vio.sh /data/euroc/MH_01_easy`.
`--save PATH` changes the recording destination; `--max-frames 10` creates a
short smoke recording. Images are JPEG previews, 480 pixels wide by default;
`--image-width 0` retains original resolution, with JPEG encoding still applied.

## Coordinate conventions and scope

- Estimated body poses are rigidly aligned to ground truth using the same
  full-run association and SE(3) alignment as the evaluation script. Scale is
  unchanged. Ground truth is only used by this offline visualization.
- When ground truth is absent, poses remain in the estimator's world frame.
- `T_imu_cam` places each camera relative to the estimated body/IMU pose.
- The default calibration is the official EuRoC Double Sphere variant used by
  the local runner. Supply `--calibration` if the run used another calibration.
- Frustums approximate Double Sphere cameras with pinhole intrinsics. Raw
  distorted images are shown separately, not projected into the 3D scene.
- Full paths are visible throughout playback; markers and cameras follow time.
- No landmark map is shown because the local VIO run did not export landmarks.
- This is recorded playback, not live logging from the estimator.
