# Online, pose-conditioned DA3 depth with GRACO VIO

The GRACO replay runs a five-view DA3-SMALL TensorRT engine in a bounded
background worker. Each view is an actual left-camera **VIO keyframe**. DA3
receives the images, calibrated intrinsics and VIO camera poses. Depth is
aligned to VIO landmarks and displayed in Rerun alongside monocular VIO.

## Run

After building the estimator, installing `scripts/requirements-graco.txt` in
`.runtime/graco-venv`, and building the engine described below:

```bash
python3 scripts/build_tensorrt_python_bridge.py
.runtime/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/aerial-05-40m \
  --camera-mode mono --imu-noise-scale 0.75 --imu-bias-scale 0.75 \
  --da3-config configs/graco/da3_five_view.json \
  --output results/graco/aerial-05-40m_da3_repeat

.runtime/graco-venv/bin/rerun \
  results/graco/aerial-05-40m_da3_repeat/playback.rrd
```

Use a fresh output directory. `--max-frames 1200` gives a takeoff smoke test.
`--rerun-connect rerun+http://127.0.0.1:9878/proxy` also streams to a running
viewer. Recordings and depth archives belong under `results/graco/`, outside
the disposable Cargo `target/` directory.

DA3 is opt-in through `--da3-config`. The existing `--online-loop-config` can
also be supplied when a JIST/XFeat/LighterGlue engine bundle is available. The
DA3 cloud uses raw VIO coordinates; loop graph corrections do not deform it.
The DA3 experiment below runs VIO and DA3, without loop closure.

## Camera conditioning

The fixed engine preserves the export's FP16/FP32 tensor types and has these inputs:

| Input | Shape | Convention |
| --- | --- | --- |
| `images` | `1 × 5 × 3 × 350 × 504` | Undistorted grayscale replicated to RGB; ImageNet normalization |
| `input_extrinsics` | `1 × 5 × 4 × 4` | World-to-camera matrices |
| `input_intrinsics` | `1 × 5 × 3 × 3` | Calibrated pinhole matrices resized with the images |

Camera poses include the calibrated camera/IMU lever arm and rotation:
`T_world_camera = T_world_body @ T_body_camera`. Its inverse is passed to DA3.
Pixel-center intrinsics follow OpenCV bilinear resizing. The graph performs
upstream DA3's camera normalization exactly once: make poses relative to the
first camera and divide translation by the median camera baseline, clamped
to 0.1 m. The reference-view strategy is `middle`.

The engine exports depth, confidence and camera predictions. Reconstruction
uses the supplied VIO poses and intrinsics. One robust positive depth
multiplier per five-view window establishes metric scale from VIO landmark
camera-z values.

## Scheduling and scale validation

The worker gathers a sliding window of five consecutive keyframes. A full
queue drops the new depth job without blocking the estimator. A gap in
keyframe ordinals resets the window: no raw-frame substitution, repeated views,
or silently bridged missing keyframes.

Before inference, spatially balanced VIO landmarks are projected into the
union of previously accepted depth views. A point counts as previously seen
only if it falls inside an earlier camera's FOV, has positive depth, and agrees
with the stored surface depth within 20%. A surface behind an occluder is not
automatically covered. Feature IDs do not determine coverage.

Defaults require **30% new sampled area**, at least 80 occupied image cells
across the five views, support in at least three views, and a baseline/depth
ratio of at least 0.005. Attempts are separated by at least five keyframes.
Coverage is a sparse surface estimate: cells without reliable landmarks remain
unknown and do not increase novelty. Rejected depth never establishes coverage.
History includes every accepted window, up to an explicit 256-window capacity;
it is not silently evicted.

Scale fitting uses one landmark per occupied 8 × 6 image cell, inverse
track-frequency weights, a median/MAD initialization and robust log-scale
refinement. Entire landmark IDs are assigned to fit or validation sets across
all five views. Defaults require 20 independent fit landmarks, five held-out
landmarks, 30 inlier observations and support in three views. Held-out median
relative error must be at most 20%, and its 90th percentile at most 40%.
Failure discards the window; no estimator state is changed.

Scale is tied to **VIO**, including its residual scale error. Ground truth is
used only by the existing trajectory evaluator and reference display. Held-out
error measures consistency with sparse VIO landmarks, not dense ground-truth
depth accuracy. Passing these checks does not validate every pixel, especially
occlusion boundaries and distant or poorly textured areas.

## Runtime and outputs

`scripts/tensorrt_session.py` binds the same native
`crates/tensorrt-runtime/native/bridge.cpp` used by visloc's Rust runtime. The
worker owns its CUDA session and calls the bridge through `ctypes`, releasing
the GIL during native inference. There is no PyTorch, ONNX Runtime or CPU
inference fallback in the online path.

Rerun adds aligned depth, its keyframe, scale/coverage metrics, and accumulated
gray-textured clouds from all five views of each accepted window. The regular
left image retains feature tracks. Outputs include:

- `da3/config.json`, `model.json`: thresholds, engine hash and tensor contract.
- `da3/events.jsonl`: candidate frame IDs/ordinals, gate decisions, timing and
  alignment diagnostics.
- `da3/window_*.npz`: depth in metres, raw depth, confidence, images, calibrated
  intrinsics, VIO poses, timestamps, frame IDs and scale metadata.
- `da3/first_inference.npz`: exact camera-conditioned inputs and raw outputs for
  independent validation, even if this first window is rejected.
- `da3/summary.json`: inference/acceptance/skip counts and dropped keyframes.
- `playback.rrd`, `trajectory.csv`, `evaluation.json`, `summary.json`: normal
  replay recording and VIO evaluation.

## Reproduce the local engine

The available weights are DA3-SMALL. Existing two-view DPVO and single-view
DA3METRIC engines are incompatible. The tested engine reuses the multiview
exporter in `xfeat_hier_vpr`, which supports camera conditioning and official
pose normalization. Set paths to the corresponding local checkouts:

```bash
DA3_REPO=/home/mikexyl/workspaces/dpvo_ws/src/DPVO/thirdparty/depth-anything-3
DA3_MODEL=/home/mikexyl/workspaces/dpvo_ws/src/DPVO/models/DA3-SMALL
DA3_EXPORTER=/home/mikexyl/workspaces/xfeat_ws/src/xfeat_hier_vpr/python/export_depth_anything_v3_multiview_to_tensorrt.py
DA3_EXPORT_PYTHON=.runtime/da3-export-venv/bin/python

# This machine's layered environment uses the original PyTorch 2.6 installation.
export PYTHONPATH=/home/mikexyl/workspaces/xfeat_ws/src/xfeat_hier_vpr/.pixi/envs/export/lib/python3.11/site-packages
"$DA3_EXPORT_PYTHON" "$DA3_EXPORTER" \
  --da3-repo "$DA3_REPO" --no-clone --model "$DA3_MODEL" \
  --views 5 --height 350 --width 504 --camera-inputs \
  --onnx .runtime/da3/da3-small-pose-5view-350x504.onnx \
  --engine .runtime/da3/da3-small-pose-5view-350x504-typed.engine \
  --hf-home .runtime/da3/hf-cache --skip-engine

/usr/src/tensorrt/bin/trtexec \
  --onnx=.runtime/da3/da3-small-pose-5view-350x504.onnx \
  --saveEngine=.runtime/da3/da3-small-pose-5view-350x504-typed.engine \
  --stronglyTyped --noTF32 --memPoolSize=workspace:2048 --skipInference

"$DA3_EXPORT_PYTHON" scripts/validate_da3_tensorrt.py \
  --da3-repo "$DA3_REPO" --model "$DA3_MODEL" \
  --fixture results/graco/aerial-05-40m_da3_final_20260922/da3/first_inference.npz \
    results/graco/aerial-05-40m_da3_final_20260922/da3/window_*.npz \
  --output results/graco/aerial-05-40m_da3_final_20260922/da3/parity.json
```

The export environment needs upstream DA3 dependencies, PyTorch, ONNX and
ONNXScript. The tested versions are PyTorch 2.6.0 and TensorRT 10.13.2 on the
RTX 4070 Laptop GPU. Rebuild engines for another GPU/runtime as needed.

Validation loads unpatched upstream DA3, uses its official pose normalization
and matches export FP16 autocast. An initial weakly typed `--fp16` engine passed
the takeoff fixture but showed substantial confidence drift later in the
flight. Do not use that build. `--stronglyTyped` preserves explicit FP32
operations alongside the FP16 network and fixed the mismatch.

Across 20 real five-keyframe fixtures from the completed final replay, the
corrected engine's worst per-window median relative depth difference was
**0.0633%**, and its worst 99th percentile was **0.4109%**. Corresponding
confidence differences were **0.2109% / 1.4150%**. All outputs were finite.
Independently perturbing poses and intrinsics changed TensorRT depth, confirming
both conditioning inputs are active. `--engine` and `--library` can be added
to the validator to rerun a rebuilt engine on archived inputs; by default it
validates the recorded outputs.

## Completed aerial-05 experiment (2026-09-22)

Final artifacts: `results/graco/aerial-05-40m_da3_final_20260922/`.
Control: `results/graco/aerial-05-40m_vio_control_20260922/`.
Both use left-camera + IMU VIO with noise and bias multipliers of 0.75.

| Measurement | Result |
| --- | ---: |
| Processed frames / actual keyframes | 5,926 / 847 |
| Candidate five-keyframe windows | 843 |
| DA3 inferences / accepted depth windows | 40 / 19 |
| Windows skipped for existing coverage | 611 |
| Skipped for geometry / minimum interval | 32 / 160 |
| Rejected depth alignments | 21 |
| Dropped keyframes / sequence resets | 0 / 0 |
| Native inference median / p95 | 30.53 / 44.77 ms |
| Median of accepted-window held-out relative errors | 7.98% |
| Maximum accepted-window held-out median error | 16.21% |
| VIO SE(3) ATE RMSE | 2.445646 m |
| VIO excess-scale diagnostic | 2.09672% |

The entire `trajectory.csv` is byte-for-byte identical to the control. Every
inference passed the 30% novelty gate, every window contains five consecutive
actual keyframe ordinals, and all accepted windows had at least 14 fit
observations in each individual view. Of the rejected alignments, 20 failed
held-out depth agreement and one failed scale consistency.

The replay took 508.2 seconds for 296.25 seconds of sensor data. Host load
varied between runs, so the wall times are not a controlled overhead comparison.
The inference timings exclude coverage checks, compression and visualization.

`da3/audit.json` contains the complete comparison, `da3/parity.json` validates
all 19 accepted windows plus the first attempted inference, and
`da3/conditioning_check.json` records the independent pose/intrinsic input
perturbation checks. The engine build provenance is in `da3/engine_build.json`.
`da3/depth_preview.png` shows three aligned depth samples. The final
`playback.rrd` passed `rerun rrd verify`.

Geometry and scheduling checks:

```bash
.runtime/graco-venv/bin/python -m unittest discover -s tests -p 'test_online_da3.py'
.runtime/graco-venv/bin/python -m unittest discover -s tests -p 'test_graco_calibration.py'
```
