# JIST 0.8: loop-verification failure diagnosis

Replayed the **128 unique candidate sequence pairs** from the completed improved-VIO aerial 5–8 experiment at **JIST cosine similarity ≥0.8**. All selected frame pairs, match counts, fundamental-matrix inlier counts, original accept/reject reasons, chosen PnP inlier counts/directions, and accepted reprojection/coverage metrics reproduce (float metrics checked within 1e-6). The diagnostic verifier also agrees with the production verifier on all 128 decisions. No thresholds or estimator settings were changed.

## Candidate funnel

A candidate succeeds if either PnP direction passes. Each rejected candidate is counted once at the furthest gate reached by either direction.

| Stage | Entering | Rejected here | Remaining |
|---|---:|---:|---:|
| XFeat/LighterGlue local matching: at least 15 matches | 128 | **36** | 92 |
| Fundamental-matrix RANSAC: return a model | 92 | **0** | 92 |
| Surviving fundamental-matrix inliers: at least 15 | 92 | **33** | 59 |
| Matched metric landmarks: at least 15 in either direction | 59 | **11** | 48 |
| P3P RANSAC/refinement: return a pose | 48 | **0** | 48 |
| PnP consensus: at least 15 inliers in either direction | 48 | **45** | 3 |
| Inlier ratio ≥0.5, finite mean reprojection ≤2.25 px | 3 | **0** | 3 |
| Image coverage ≥2%, at least four occupied 4×4 cells, valid transform | 3 | **0** | **3** |

**125 rejected; 3 accepted (2.34%).** The original logs' 89 `pnp_geometric_support_failed` events split into **33 + 11 + 45**. Fundamental RANSAC returned a model for all 92 pairs reaching it; for 33, only 10–14 inliers survived. Production has no explicit minimum-F-inlier gate: these 33 subsequently fail the minimum 15 2D–3D correspondence check. The separate row identifies the earlier source of the shortage.

The final ratio/reprojection/coverage rows are conditional, not a claim that every rejected pose satisfies those criteria. Among 83 actual directional PnP pose reports, 78 have fewer than 15 inliers; 65 also have an inlier ratio below 0.5, 14 also occupy fewer than four cells, and one also covers less than 2% of the image. These overlapping failures are already counted at the first failed gate and must not be added to the candidate totals. No pose report had nonfinite or >2.25 px mean reprojection error.

## What limits verification

- All 128 pairs had **128 real features in each frame**. No rejection came from an empty feature packet or padded features.
- The 36 matching failures had 0–14 matches (median 8).
- The 33 fundamental-consensus shortages retained 10–14 matches (median 12).
- The 11 metric-landmark shortages retained 15–23 F inliers, but the better direction had only 8–14 metric correspondences (median 13).
- The 45 PnP-support failures produced poses, but their better direction had only **4–14 inliers (median 7)**. Only three of these candidates reached 12, 13, or 14 inliers; the other 42 reached at most 11.
- The three accepted candidates had 15, 16, and 17 PnP inliers. Two candidates passed both directions; one passed forward only. Winner selection used reverse for two accepted constraints.

The measured bottleneck is local correspondence/metric geometric support. Raising acceptance by relaxing only the final reprojection or coverage gates would recover **zero** candidates in this run. These counts alone do not distinguish incorrect correspondences from inaccurate landmark geometry, or prove that rejected candidates are real loops.

## Robot-pair breakdown

| Sequence endpoints | Attempts | Matches <15 | F inliers <15 | Metric correspondences <15 | PnP inliers <15 | Accepted |
|---|---:|---:|---:|---:|---:|---:|
| a05 / a05 | 10 | 0 | 2 | 1 | 6 | 1 |
| a05 / a06 | 1 | 1 | 0 | 0 | 0 | 0 |
| a05 / a07 | 13 | 1 | 7 | 3 | 2 | 0 |
| a05 / a08 | 5 | 2 | 3 | 0 | 0 | 0 |
| a06 / a06 | 31 | 10 | 12 | 3 | 6 | 0 |
| a06 / a07 | 32 | 6 | 2 | 1 | 21 | 2 |
| a06 / a08 | 10 | 5 | 3 | 1 | 1 | 0 |
| a07 / a07 | 8 | 2 | 1 | 0 | 5 | 0 |
| a07 / a08 | 11 | 8 | 3 | 0 | 0 | 0 |
| a08 / a08 | 7 | 1 | 0 | 2 | 4 | 0 |

Self-pairs are within-robot loops; other rows are cross-robot candidates. These are candidate identities, not the node that owned verification.

## Directional accounting

Both PnP directions are examined for each of the 92 pairs with a fundamental model: **184 directions** total.

| First failed gate, per direction | Count |
|---|---:|
| F inliers <15 | 66 |
| F inliers ≥15, but metric correspondences <15 | 35 |
| PnP inliers <15 | 78 |
| Passed all gates | 5 |
| All other failure gates | 0 |

There were 83 actual directional PnP solves (78 failed the inlier-count gate and 5 passed). The 5 passing directions represent only 3 accepted candidate pairs.

## Verification-owner breakdown

| Owner | Attempts | Matches <15 | F inliers <15 | Metric correspondences <15 | PnP inliers <15 | Accepted |
|---|---:|---:|---:|---:|---:|---:|
| a05 | 29 | 4 | 12 | 4 | 8 | 1 |
| a06 | 73 | 21 | 17 | 5 | 28 | 2 |
| a07 | 19 | 10 | 4 | 0 | 5 | 0 |
| a08 | 7 | 1 | 0 | 2 | 4 | 0 |

## Reproduction and artifacts

Historical experiment: `results/graco/multi_robot_f64_stationary_5_8_final/`.

- `verification_diagnosis/details.json`: every pair's selected keyframes, matching scores, F-inlier count, both directional metric counts, PnP counts/ratios/errors/coverage, overlapping failed predicates, original diagnostic, and aggregate counts.
- `verification_diagnosis/provenance.json`: original experiment commit and verified model/config/source hashes.
- `pipelines/multi-robot/examples/diagnose_verification.rs`: standalone diagnostic replay. It does not change production geometry, VIO, retrieval, or PGO. It refuses to save results if recorded outcomes or production-verifier outcomes disagree.

JIST and XFeat descriptors are read from the frozen archives. Only LighterGlue and geometry are rerun. All three current engine hashes and the archived loop-config hash match the original experiment manifest.

```bash
CARGO_BUILD_JOBS=2 cargo build --offline --release \
  -p visloc-multi-robot --features native --example diagnose_verification
source scripts/source_multi_robot_ros2.bash
export OPENBLAS_NUM_THREADS=1
target/release/examples/diagnose_verification \
  results/graco/multi_robot_f64_stationary_5_8_final \
  results/graco/multi_robot_f64_stationary_5_8_final/loop_config.json \
  /tmp/jist08_verification_diagnosis.json
```

Choose a new output filename; the helper will not overwrite existing diagnostic results. The gate instrumentation mirrors the current production implementation; its embedded source hash and mandatory parity checks identify drift if production changes later.
