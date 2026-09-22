# Online VIO loop closure with TensorRT

`visloc-online-loop` runs JIST, XFeat and **XFeat-trained LighterGlue** on a
background thread alongside Basalt. Its output is a separate corrected map
trajectory; it does not change VIO's filter, IMU biases, or raw trajectory.
All three networks execute through the native Rust/C++ TensorRT runtime.
Python and ONNX Runtime are only needed for replay and conversion validation.

## Data flow

1. `EstimatorOutput::is_keyframe` selects actual Basalt keyframes. JIST receives
   a causal window of **five consecutive VIO keyframes**, never fixed intervals
   of raw camera frames. A dropped keyframe resets the window; no frame is
   duplicated for padding. The newest member is the query keyframe.
2. Build a covisibility graph from persistent VIO landmark IDs using **all**
   triangulated observations, before limiting features for the matcher. Edges
   require 15 shared landmarks by default (`covisibility_min_shared`).
   Before computing similarity or selecting top-k, remove database sequences
   that overlap the query, contain consecutive keyframes, or contain any
   keyframe within two covisibility hops of **any** query member
   (`covisibility_hops`). An additional 20-second exclusion applies between
   the database sequence's last keyframe and the query's first keyframe.
   Local sequences never occupy candidate slots or reach LightGlue/PnP.
3. JIST ResNet-18/SeqGeM produces normalized 512-dimensional sequence and
   per-keyframe descriptors. Rank only the remaining search pool, with global
   cosine similarity **>= 0.8** (`min_similarity`). Group overlapping or
   covisible database windows so top-k selects up to three distinct local
   neighborhoods. Per-keyframe descriptors select the image within each
   retrieved sequence.
4. XFeat computes its dense 64-dimensional descriptor map. Bicubic sampling at
   measured VIO feature positions preserves the exact association with metric
   VIO landmarks. This frontend uses VIO's KLT positions, rather than XFeat's
   keypoint detector. A spatially balanced selection supplies 128 real features
   to each matcher input. Images with fewer tracks still contribute to JIST
   but skip geometric matching; missing features are never padded or invented.
5. LighterGlue matches those XFeat descriptors. P3P RANSAC and pose refinement
   use frozen metric landmarks from either keyframe, with minimum inlier count,
   ratio, reprojection, image coverage and temporal confirmation gates.
   If candidate-side landmarks fail verification, reverse PnP uses query-side
   landmarks and inverts the resulting constraint to preserve edge direction.
6. Verified camera constraints are converted into body constraints using the
   calibrated camera-to-IMU transform, including its lever arm. The existing
   sparse robust SE(3) pose-graph optimizer refines the map trajectory. The old
   VLAD retrieval and brute-force descriptor matcher are not invoked.

The input queue is bounded and nonblocking. GPU sessions are created on and
remain on the worker thread. Queue drops, capacity limits, candidate decisions
and accepted constraints are recorded. The default database limit is 4,000
keyframes; further keyframes are counted as capacity skips. There is no hidden
CPU inference or legacy retrieval fallback. Ground truth is used only by the
Python evaluator and visualization.

## Build models and run

The source ONNX directory must contain:

- `JIST_r18_512_seqgem_frames.onnx`: input `[1,5,3,288,512]`, outputs
  `output [1,512]` and `frame_descriptors [5,512]`.
- `xfeat_320x224.onnx`: RGB `[1,3,224,320]`, dense output `[1,64,28,40]`.
- `lg_320x224_dyn.onnx`: the six-input, 64-dimensional XFeat-trained matcher.
  SuperPoint-trained LightGlue weights are incompatible.

JIST preprocessing includes RGB ImageNet mean/std normalization **outside**
the graph, as in [official JIST evaluation](https://github.com/ga1i13o/JIST/blob/main/evaluation.py).
XFeat uses RGB floats in `[0,1]`; both adapters replicate the monocular gray
image across RGB channels. The pixel coordinates and metric geometry remain
in the calibrated, undistorted VIO image resolution.

```bash
python3 scripts/build_loop_tensorrt.py --model-dir /path/to/onnx_model
RUSTFLAGS='-C target-feature=+avx2,+fma' cargo build --locked --release \
  --example basalt_stream_vio --features basalt-lm-workspace-reuse,tensorrt-loop
target/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/ground-01 --camera-mode mono \
  --online-loop-config target/loop_models/loop_config.json \
  --output target/graco_vio/ground-01_loops
```

The ground launch helper also accepts `--online-loop-config` and builds the
required Cargo feature. Engine paths in a custom JSON config are relative to
that config's directory. The builder writes absolute paths plus SHA-256 hashes
and build commands in `target/loop_models/manifest.json`. Engines are specific
to the TensorRT version and target GPU and are not committed.

The default engines use **FP32, TF32 disabled**, and a fixed 128-feature matcher
profile. `--keypoints` changes this count and writes the corresponding config.
Validate any newly built bundle before deployment. On the tested TensorRT
10.13.2 installation, broad dynamic FP32 attention profiles failed inside
Myelin, while FP16 materially changed match confidence. Fixed profiles avoid
both issues without a custom plugin. Match **outputs** remain data dependent;
the native runtime uses an aligned output allocator, including for zero matches.

## Validation and outputs

```bash
uv pip install --python target/graco-venv/bin/python onnxruntime==1.22.1
cargo build --release -p visloc-tensorrt --features native --example infer_fixtures
target/graco-venv/bin/python scripts/validate_loop_tensorrt.py
cargo test -p visloc-online-loop --features native --lib
VISLOC_LOOP_CONFIG=target/loop_models/loop_config.json cargo test --release \
  -p visloc-online-loop --features native --test online_gpu -- --ignored --nocapture
```

Parity checks compare each engine with its original ONNX graph. The matcher
checks populated, empty, and repopulated outputs in one session, including
exact match-pair agreement. These fixtures verify numerical conversion, not
place-recognition accuracy. Dataset replay provides the integration check.
The GPU integration regressions check that repeated keyframes with new track
IDs can close a loop, while persistent covisibility prevents even similarity
comparisons and matcher calls, including beyond the temporal exclusion.

Alongside the existing raw VIO files, a loop-enabled replay writes:

- `loop_config.json`, `loop_events.jsonl`, `loop_closure.json`: effective config,
  actual keyframe IDs, sequence membership, verification decisions and counters.
  Keyframe events include weighted covisibility edges. Retrieval events record
  the local sequences skipped **before ranking**, remaining descriptor
  comparisons, and sequences passing the score threshold. `candidates` counts
  matching attempts; `candidate_queries` counts distinct query keyframes with
  an attempt. Neither number counts distinct physical loop closures.
- `trajectory_loop.csv` and `trajectory_loop.tum`: all sensor-frame body poses,
  using interpolated optimized keyframe corrections after the worker drains.
- `evaluation_loop.json`: independent corrected-trajectory evaluation against
  ground truth, using the same association and rigid alignment as raw VIO.
- `playback.rrd`: raw VIO in blue, corrected keyframe trajectory/current pose
  in green, loop edges in magenta, plus existing images and feature tracks.
  The final dense corrected trajectory appears when replay finishes.

The pose graph is metric SE(3), not Sim(3), and does not estimate camera
calibration or a global scale factor. It can distribute accumulated drift
when genuine revisits are verified; it cannot repair a calibration scale error.

### Measured validation (2026-09-20)

On the RTX 4070 Laptop / TensorRT 10.13.2, all three FP32 engines pass ONNX
parity. Maximum absolute errors were 4.0e-7 for JIST outputs and 1.6e-5 for
XFeat outputs. LighterGlue's 120/0/114-match fixtures agree exactly on pairs,
with maximum score error 7.9e-5. Two real GRACO sequence cosine scores also
agree with original ONNX inference within 4e-7. The controlled GPU loop test
accepts two synthetic revisits; persistent covisibility yields zero descriptor
comparisons and zero matcher calls even after the time exclusion expires.

The complete ground-01 replay is in
`target/graco_vio/ground-01_jist_covisibility_20260920`. It processed 6,676
sensor frames and 954 actual keyframes with zero drops. The causal graph audit
checks each query against only the covisibility edges available at that time.
It confirms 36,808 local sequence entries were skipped before descriptor
scoring. After grouping redundant database windows, there were **127 matching
attempts across 78 query keyframes**, compared with 224 attempts before grouping.
These are repeated retrievals along a traverse, not counts of physical places.

There are **zero accepted loops** on this replay: 126 attempts fail geometric
support and one passes geometry but lacks a subsequent valid confirmation.
Both raw and corrected SE(3) ATE RMSE are **2.837377510 m**. Raw poses exactly
match the VIO-only baseline, and corrected poses exactly match raw poses when
no loop is accepted. Filtering local sequences is validated; improved ATE on
this sequence has not been demonstrated. See `covisibility_audit.json`,
`loop_events.jsonl`, `summary.json`, and `evaluation_loop.json` in that directory.

The aerial-05-40m replay on 2026-09-21, in
`target/graco_vio/aerial-05-40m_jist_covisibility_20260921`, used the same loop
settings (similarity 0.8, 15 shared landmarks, two covisibility hops), monocular
VIO and the supplied aerial calibration. All 5,926 frames / 847 keyframes
completed, with zero drops. The causal audit confirms local exclusion before
scoring. There were 139 matching attempts across 121 query keyframes, with one
accepted loop from keyframe 301 to 5439 (similarity 0.9640, 15 PnP inliers,
1.105 px mean reprojection error), following confirmation at query 5418.

The accepted loop **worsened metric SE(3) ATE RMSE from 3.471971 m to
3.846168 m**. Diagnostic scale-aligned ATE improves from 1.680087 m to
1.406166 m, but the fitted reference scale changes from 0.961446 to 0.954888;
that scale alignment is never applied to the estimator or displayed path.
An independent rigid alignment of all saved poses confirms the metric ATE.
The run demonstrates an accepted online loop, not an accuracy improvement.
Its Rerun recording includes the raw and corrected trajectories and loop edge.

A subsequent [IMU uncertainty sweep](graco_imu_tuning.md) reduced aerial-05
raw ATE to **2.445646 m** and loop-corrected ATE to **2.636699 m**, using
0.75× white-noise and bias-walk standard deviations. The loop settings were
unchanged; the accepted loop still worsens the corresponding raw result.

```bash
target/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/aerial-05-40m --camera-mode mono \
  --online-loop-config target/loop_models/loop_config.json \
  --output target/graco_vio/aerial-05-40m_loops_repeat
```

## Existing implementations reviewed

- [Derkai52/XFeat-Lightglue-TRT](https://github.com/Derkai52/XFeat-Lightglue-TRT)
  provides XFeat/LighterGlue export and C++ TensorRT deployment with a fixed
  feature count. Its documented environment is TensorRT 8.5.2/JetPack 5.1.3.
  We use that fixed-count deployment approach with the existing local weights
  and our TensorRT 10 runtime; no source code from that repository is vendored.
- [noahzhy/xfeat_lightglue_onnx](https://github.com/noahzhy/xfeat_lightglue_onnx)
  provides XFeat-specific ONNX exports, including LighterGlue.
- [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)
  provides maintained export and TensorRT workflows. Its other extractors'
  weights cannot replace the XFeat-trained matcher.
- [qdLMF/LightGlue-with-FlashAttentionV2-TensorRT](https://github.com/qdLMF/LightGlue-with-FlashAttentionV2-TensorRT)
  provides a head-dimension-64 attention plugin for TensorRT 8.5.2. The local
  LighterGlue model uses head dimension 96, so it is not a drop-in replacement.
- [Official JIST](https://github.com/ga1i13o/JIST) provides the trained
  ResNet-18/SeqGeM model. No dedicated public TensorRT JIST implementation was
  found in this review; its local ONNX export converts directly and passes
  numerical parity checks.
