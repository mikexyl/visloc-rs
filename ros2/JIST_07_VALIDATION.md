# JIST global threshold 0.7: four-robot experiment

**Verdict:** 0.7 does not improve this mission. It increases matching attempts from 128 to 912 (7.125×), while preserving exactly the same three accepted constraints and three map components. All 784 additional candidates fail verification: 596 have fewer than 15 matches, and 188 fail PnP geometric support. Retain 0.8 as the default for this configuration.

Completed aerial 5–8 in one native Rust ROS2 mission with the accepted f64 stationary gyro-only VIO. Only the global JIST cosine threshold changes from 0.8 to 0.7. The comparison preserves monocular left-camera inputs, 800-pixel preprocessing, calibration, 0.75 IMU multipliers, the three-neighborhood limit, temporal/covisibility exclusions, 5×5 refinement, 128-feature matching, geometric thresholds, and graph weights.

## Accuracy

Position ATE RMSE in meters, using rigid SE(3) alignment without scale fitting. Every raw pose is bit-identical to the accepted control, including all startup frames.

| Sequence | Raw VIO | Corrected, 0.8 | Corrected, 0.7 | Change |
|---|---:|---:|---:|---:|
| a05 | 2.887 | 2.300 | 2.300 | <0.001 |
| a06 | 2.262 | 2.252 | 2.252 | <0.001 |
| a07 | 2.642 | 2.643 | 2.643 | <0.001 |
| a08 | 0.964 | 0.964 | 0.964 | <0.001 |

ATE is unchanged at the reported precision. The small numerical differences reflect the online optimization path, with identical accepted constraints. These per-robot alignments must not be substituted for joint map accuracy.

| Trial connected component | Poses | Joint ATE RMSE (m) |
|---|---:|---:|
| a05 | 5,926 | 2.300 |
| a06, a07 | 14,470 | 3.748 |
| a08 | 5,563 | 0.964 |

The 0.8 baseline has components {a05}, {a06,a07}, {a08}, with joint ATE 2.300, 3.748 and 0.964 m respectively. Each reported joint result uses one rigid alignment for the entire connected component.

## Retrieval and verification

| Metric | 0.8 baseline | 0.7 trial |
|---|---:|---:|
| Matching attempts | 128 | 912 |
| Verified constraints | 3 | 3 |
| Verification success | 2.34% | 0.33% |
| Components | 3 | 3 |
| Wall time (min) | 20.12 | 19.71 |

The trial accepts 0 constraints with similarity below 0.8. Lowering the global threshold does not relax fundamental-matrix RANSAC, bidirectional P3P, the minimum 15 PnP inliers, the 0.5 inlier ratio, reprojection thresholds, or spatial coverage checks.

Accepted loop constraints:

| Endpoints (keyframes) | JIST similarity | PnP inliers | Reprojection error (px) |
|---|---:|---:|---:|
| a05:42 → a05:774 | 0.946064 | 16 | 1.191 |
| a06:107 → a07:105 | 0.910685 | 17 | 0.729 |
| a06:118 → a07:121 | 0.961015 | 15 | 0.667 |

Verification outcomes and score bands:

```json
{
  "outcomes": {
    "pnp_geometric_support_failed": 277,
    "insufficient_matches": 632,
    "verified": 3
  },
  "by_score_band": {
    "a05": {
      "0.7_to_0.8/insufficient_matches": 292,
      "0.7_to_0.8/pnp_geometric_support_failed": 83,
      "at_least_0.8/insufficient_matches": 4,
      "at_least_0.8/pnp_geometric_support_failed": 24,
      "at_least_0.8/verified": 1
    },
    "a06": {
      "0.7_to_0.8/insufficient_matches": 163,
      "0.7_to_0.8/pnp_geometric_support_failed": 57,
      "at_least_0.8/insufficient_matches": 21,
      "at_least_0.8/pnp_geometric_support_failed": 50,
      "at_least_0.8/verified": 2
    },
    "a07": {
      "0.7_to_0.8/insufficient_matches": 118,
      "0.7_to_0.8/pnp_geometric_support_failed": 37,
      "at_least_0.8/insufficient_matches": 10,
      "at_least_0.8/pnp_geometric_support_failed": 9
    },
    "a08": {
      "0.7_to_0.8/insufficient_matches": 23,
      "0.7_to_0.8/pnp_geometric_support_failed": 11,
      "at_least_0.8/insufficient_matches": 1,
      "at_least_0.8/pnp_geometric_support_failed": 6
    }
  }
}
```

## Reproducibility and validation

All 25,959 images completed. Image drops: 0; loop-keyframe drops: 0; exhausted service requests: 0. All keyframes and constraints reached the final graph. All published robust objectives are finite and non-increasing. Rejected optimizer updates: 0.

Archive parity against 0.8 ignores only session identifiers; it compares sequence membership, global/frame descriptors, selected features, metric landmarks and calibration:

```json
{
  "a05": {
    "baseline_sequences": 72,
    "trial_sequences": 72,
    "shared_sequences": 72,
    "identical_archives": 72,
    "changed_sequences": [],
    "missing_from_trial": [],
    "new_in_trial": []
  },
  "a06": {
    "baseline_sequences": 67,
    "trial_sequences": 67,
    "shared_sequences": 67,
    "identical_archives": 67,
    "changed_sequences": [],
    "missing_from_trial": [],
    "new_in_trial": []
  },
  "a07": {
    "baseline_sequences": 80,
    "trial_sequences": 80,
    "shared_sequences": 80,
    "identical_archives": 80,
    "changed_sequences": [],
    "missing_from_trial": [],
    "new_in_trial": []
  },
  "a08": {
    "baseline_sequences": 63,
    "trial_sequences": 63,
    "shared_sequences": 63,
    "identical_archives": 63,
    "changed_sequences": [],
    "missing_from_trial": [],
    "new_in_trial": []
  }
}
```

The previous native implementation hard-coded 0.8 in retrieval, proposal rechecking, and backend acceptance. Robots now read the existing loop configuration’s `min_similarity`; the backend persists `pgo.min_loop_similarity`. The default remains 0.8. Mission preparation snapshots the loop configuration and sets both thresholds consistently. The native GPU models and VIO source are unchanged.

Validation: 18 multi-robot algorithm tests pass, including threshold boundaries and unchanged inlier requirements. A native ROS2 contract test accepts a 0.75-similarity constraint with a 0.7 backend threshold, then checks duplicates, delayed discovery, simulated time, timeout/retry, component joining, backend restart and history recovery. The full dataset run checks actual robot retrieval and backend settings and raw-VIO parity.

The replay uses the validated large-buffer Fast DDS profile, nominal 1× playback with backpressure, and detached supervision. Model/configuration/binary/source hashes and the exact source patch are in `experiment_manifest.json`. Communication and latency measurements, including their payload-accounting limits, are in `validation_summary.json`. Different runtime load and asynchronous message order mean wall time is descriptive rather than a controlled performance benchmark.

## Running this configuration

```bash
source scripts/source_multi_robot_ros2.bash
export OPENBLAS_NUM_THREADS=1
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/visloc_ros/config/fastdds_replay.xml"
.runtime/graco-venv/bin/python scripts/prepare_multi_robot_graco.py \
  --robots 5 6 7 8 --rate 1 --min-similarity 0.7 \
  --output results/graco/my_jist07_trial
.runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py \
  results/graco/my_jist07_trial/mission.json --domain-id 227
```

Artifacts: `results/graco/multi_robot_f64_jist07/`. The audited comparison is `threshold_comparison.json`; the interactive recording is `playback.rrd`. The fixed-last-frame ablation uses the identical trial retrieval pairs and is recorded separately in `refinement_comparison.json` and the summary.
