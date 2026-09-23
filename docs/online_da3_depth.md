# Online, pose-conditioned DA3 depth with GRACO VIO (WIP)

**Status: work in progress (WIP).** The DA3 depth and dense-map path is
experimental. Metric depth quality, consistency between windows, and the
confidence/coverage tradeoff remain unresolved; passing runtime and geometry
checks does not establish dense reconstruction accuracy. Current settings are
experiment defaults. DA3 remains opt-in and does not update the VIO estimator.

The GRACO replay runs a five-view DA3-SMALL TensorRT engine in a bounded
background worker. Each view is an actual left-camera **VIO keyframe**. DA3
receives the images, calibrated intrinsics and VIO camera poses. The supplied
configuration disables landmark depth alignment: metric scale comes only from
the camera poses. Depth is displayed in Rerun alongside monocular VIO.

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

`landmark_alignment: false` in `configs/graco/da3_five_view.json` selects pose-only
depth scaling and bypasses landmark fitting and held-out landmark rejection.
Set it to `true` to reproduce the earlier landmark-aligned experiment. Older
configs that omit this field retain landmark alignment for compatibility.
Both modes retain the same confidence/depth validity mask and FOV-overlap gate.

The supplied config also enables `reprojection_filter`. This checks depth
agreement between the five keyframes before archiving, visualization and
coverage-history updates. See the reprojection-filter experiment below.
It now uses the strict confidence settings described at the end of this page:
`confidence_percentile: 70.0` and `min_confidence: 3.0`. Use the saved configuration
from a previous run to reproduce its earlier filtering settings.

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
uses the supplied VIO poses and intrinsics. Because the graph normalizes camera
translations, raw network depth is not yet in metres even with metric inputs.
Pose-only mode follows [upstream DA3's `align_to_input_ext_scale=True`
postprocessing](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/api.py):
fit an Umeyama similarity from the five supplied camera centers to predicted
centers, then divide depth by that scale. Only the depth multiplier is applied;
the reconstruction camera poses and intrinsics remain fixed. This step has no
landmark inputs or landmark-based acceptance test. Degenerate camera baselines
are rejected. Pose-fit residuals are reported but do not reject depth windows.

When landmark alignment is enabled, one robust positive depth multiplier per
window instead establishes metric scale from VIO landmark camera-z values.

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

Optional landmark scale fitting uses one landmark per occupied 8 × 6 image cell, inverse
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

Rerun adds metric depth, its keyframe, scale/coverage metrics, and accumulated
gray-textured clouds from all five views of each accepted window. The regular
left image retains feature tracks. Outputs include:

- `da3/config.json`, `model.json`: thresholds, engine hash and tensor contract.
- `da3/events.jsonl`: candidate frame IDs/ordinals, gate decisions, timing and
  alignment diagnostics.
- `da3/window_*.npz`: depth in metres, raw depth, confidence, images, calibrated
  intrinsics, VIO poses, predicted camera extrinsics, timestamps, frame IDs and
  scale metadata identifying `input_poses` or `landmarks`.
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
Both use left-camera + IMU VIO with noise and bias multipliers of 0.75. This
earlier experiment used `landmark_alignment: true`.

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

## Landmark alignment disabled (2026-09-22)

Artifacts: `results/graco/aerial-05-40m_da3_pose_only_20260922/`. The full
5,926-frame replay used the same engine, monocular VIO settings, calibrated
camera intrinsics and pose inputs, with `landmark_alignment: false`.

All 37 attempted depth windows were accepted; 626 candidates were skipped for
coverage, 32 for geometry, and 148 for the keyframe interval. No keyframes were
dropped. All 37 archived depth multipliers match the upstream Umeyama routine
to numerical precision, with no landmark fitting or held-out rejection.
The trajectory is byte-for-byte identical to the earlier run, retaining
2.445646 m VIO SE(3) ATE RMSE. The dense-only recording contains 916,860 points
at the configured visualization stride; these are unmerged depth samples.

On the identical 19 previously accepted windows, pose-only depth scale is
1.1749 times the archived landmark scale at the median (range 0.8890–1.5011).
The side-by-side map shows more variation in surface height across windows;
disabling landmark alignment does not visibly improve that comparison.
This is a consistency/visual comparison, not dense ground-truth accuracy.
The FOV gate uses depth history, so the complete pose-only replay selects a
different set of windows; the paired comparison holds window selection fixed.

- `playback.rrd`: complete VIO, depth and feature-track playback.
- `dense_map.rrd`: full pose-only cloud with a readable point size and overview.
- `paired_dense_comparison.rrd`, `paired_dense_comparison.png`: identical 19
  windows and confident pixels, with the two depth scales shown side by side.
- `paired_scale_comparison.json`: per-window scale ratios from the same inputs.
- `audit.json`: archive checks and unchanged-trajectory verification.

All three Rerun recordings passed `rerun rrd verify`. The nine DA3 geometry and
scheduling tests and six GRACO calibration tests pass.

## Depth reprojection consistency filter (2026-09-22)

`reprojection_filter: true` enables a geometric mask after metric depth scaling
and confidence masking. For every reference pixel, project its 3D point into
each of the other four cameras, sample the other depth map, then reproject that
sample back into the reference camera. Keep the reference depth only if at
least `min_consistent_views: 2` **other** views satisfy both:

- Round-trip pixel error at most `max_reprojection_error_px: 1.5`, measured at
  the DA3 depth resolution (504 × 350), not the original camera resolution.
- Relative round-trip camera-z disagreement at most
  `max_reprojection_depth_error: 0.05` (5%). The depth check is necessary when
  low parallax allows a small pixel error despite incorrect depth.

This follows the forward/backward geometric-consistency approach described by
[MVSNet](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf),
with configurable thresholds for this VIO/DA3 experiment. Sampling rejects
missing depth and interpolation across depth discontinuities. Occluded,
behind-camera or out-of-view points provide no supporting vote. A pixel may
still survive through agreement with other visible cameras. All votes use
the original five depth maps; no sequential erosion or depth averaging occurs.

Surviving depth values, scale, calibrated intrinsics and VIO poses are unchanged.
Rejected depths become NaN and cannot enter the dense cloud or coverage history.
Online windows with less than 10% finite depth after filtering are rejected and
counted separately as `consistency_rejections`. Archives include per-pixel
`depth_support` (0–4 other views) and `consistency_mask`. Events and Rerun status
include retained fractions and filter time. Older configs that omit the filter
flag retain the previous behavior.

This is a consistency check **within each five-keyframe window**. It does not
correct a shared scale error or guarantee agreement between separate windows.
Landmark depth alignment remains disabled in the supplied configuration.

To compare the same saved windows without rerunning inference or VIO:

```bash
.runtime/graco-venv/bin/python scripts/filter_da3_depths.py \
  --source results/graco/aerial-05-40m_da3_pose_only_20260922 \
  --config results/graco/aerial-05-40m_da3_reprojection_20260922/config.json \
  --output results/graco/aerial-05-40m_da3_reprojection_repeat
.runtime/graco-venv/bin/rerun \
  results/graco/aerial-05-40m_da3_reprojection_repeat/filter_comparison.rrd
```

The script uses a new output directory, preserves source archives, saves masks
and filtered depths, and creates a before/after Rerun map using identical
camera views. It keeps the original window selection fixed; an online run's
FOV decisions can change because its coverage history uses filtered depth.

Completed outputs: `results/graco/aerial-05-40m_da3_reprojection_20260922/`.
Across all 37 saved windows, the filter retained **28,870,195 / 32,634,000 valid
pixels (88.47%)**. Displayed cloud samples at stride 6 fell from **916,860 to
806,388**. Median filter time was **213.5 ms per window** on this machine.
The first window (frames 126–154) retained 6.88%, below the online 10% minimum;
the offline comparison retains its surviving pixels and flags the window in
`summary.json`. Every surviving depth is exactly equal to its input value.

A 400-frame VIO/TensorRT smoke replay completed with 58 keyframes, five inference
windows, four accepted windows, one consistency rejection and zero dropped
keyframes. This checks online integration; the complete 37-window comparison
above is offline filtering of the prior full run. All 16 DA3 tests and six GRACO
calibration tests pass, covering rotated cameras, differing intrinsics, world
frame invariance, isolated outliers, low parallax, occlusions, missing samples,
both error limits, and filtered coverage history.

## Strict confidence trial (2026-09-22)

The supplied config now applies confidence >= `max(3.0, P70)` before the
reprojection check. P70 is the 70th percentile across the five keyframes'
finite, in-range depth pixels, computed before the absolute confidence floor
or geometric mask. This nominally retains the highest-scoring 30%; ties may
retain more and the absolute floor may retain less. The floor prevents a weak
window from passing just because some of its pixels rank higher than others.
DA3 confidence is a score, not a probability of correctness.

Both reference and supporting depth maps receive the confidence mask, so
low-confidence pixels cannot validate other depths. There is no rescaling or
depth replacement. Archives retain the raw confidence and add `confidence_mask`;
events and Rerun status report the actual per-window cutoff. Windows with less
than 10% valid image area after confidence masking are counted as
`confidence_rejections`. The default percentile for old configs is 0, preserving
absolute-threshold-only behavior. Landmark alignment stays disabled.

```bash
.runtime/graco-venv/bin/python scripts/filter_da3_depths.py \
  --source results/graco/aerial-05-40m_da3_reprojection_20260922 \
  --trajectory results/graco/aerial-05-40m_da3_pose_only_20260922/trajectory.csv \
  --output results/graco/aerial-05-40m_da3_confidence70_repeat
```

The offline tool reconstructs metric depth from archived `raw_depth` and the
unchanged scale, then applies confidence followed by reprojection. The before
panel shows the source archive's existing mask. This avoids computing a
percentile over a subset already selected by a previous geometric filter.

Completed output: `results/graco/aerial-05-40m_da3_confidence70_20260922/`.
On the same 37 windows, confidence alone retained 9,525,604 pixels. Both filters
together retained **8,136,892 / 32,634,000 original valid pixels (24.93%)**, or
28.18% of the previous reprojection-filtered map. Displayed samples fell from
**806,388 to 227,367**. Actual confidence cutoffs ranged from **3.0 to 9.1398**,
with median **4.8341**. The first window still falls below the online 10% image
support requirement. Filtering took 229.5 ms per window at the median.

The before/after rendering shows substantially less peripheral coverage and
large gaps: this is deliberately a quality-over-completeness setting. Confidence
and multiview agreement do not prove absolute depth accuracy or remove shared
scale errors between windows. All surviving depths are bit-for-bit unchanged,
and no pixel removed by the preceding reprojection filter was reintroduced.

A 400-frame online smoke replay completed with 58 keyframes, eight inferences,
seven accepted windows, one confidence rejection and zero dropped keyframes.
Its VIO trajectory matches the prior run's prefix exactly. Smaller coverage
history explains the additional inference attempts. Both recordings passed
`rerun rrd verify`; all 19 DA3 tests and six GRACO calibration tests pass.
`audit.json` verifies pixel masks, unchanged geometry and trajectory; the
`filter_comparison.rrd` and `filter_comparison.png` show the same-window comparison.
