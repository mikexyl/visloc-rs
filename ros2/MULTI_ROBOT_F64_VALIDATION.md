# Four-robot experiment with improved VIO

Completed GRACO aerial 5–8 together in one ROS2 domain, with four native Rust robot nodes, online JIST/XFeat/LighterGlue loop processing, and centralized SE(3) PGO. All raw VIO trajectories meet the per-sequence **ATE RMSE < 4 m** target. Corrected-map accuracy and connectivity are reported separately.

## Accuracy

All values are full-trajectory position ATE RMSE in meters after rigid SE(3) alignment, with **no scale fitting**. Individual rows use one alignment per robot; the joint results below use one alignment for each connected component.

| Sequence | Frames | Previous raw | Improved raw | Online corrected |
|---|---:|---:|---:|---:|
| a05 | 5,926 | 2.439 | 2.887 | 2.300 |
| a06 | 6,584 | 13.170 | 2.262 | 2.252 |
| a07 | 7,886 | 64656.661 | 2.642 | 2.643 |
| a08 | 5,563 | 1.059 | 0.964 | 0.964 |

The previous run used f32 with legacy initialization and failed badly on aerial 6/7. This rerun uses the accepted f64, stationary gyro-only initializer, monocular left camera, 800-pixel processing width, and 0.75 IMU noise/bias multipliers. All 25,959 raw poses are bit-identical to the accepted standalone controls after undoing the replay timestamp offset. No startup frames are omitted.

| Connected component | Associated poses | Joint ATE RMSE (m) |
|---|---:|---:|
| a05 | 5,926 | 2.300 |
| a06, a07 | 14,470 | 3.748 |
| a08 | 5,563 | 0.964 |

Final graph: **3,706 keyframes, 3 verified constraints, 3 components**. Verification accepted 3/128 attempts (2.3%). Separate components do not constitute one globally connected four-robot map.

## Controlled refinement comparison

Both variants use exactly the same 128 recorded retrieval pairs, frozen features, raw odometry, and batch optimizer settings. The table compares full 5×5 frame refinement with fixed-last-frame verification. These are batch ablations, distinct from the actual online result above.

| Verification | Accepted constraints | Components | a05 ATE | a06 ATE | a07 ATE | a08 ATE |
|---|---:|---:|---:|---:|---:|---:|
| Full 5×5 | 3 | 3 | 2.300 | 2.252 | 2.643 | 0.964 |
| Fixed last | 3 | 3 | 2.887 | 2.259 | 2.646 | 0.964 |

Full refinement recovers the aerial 5 self-loop that fixed-last-frame verification misses, reducing aerial 5 ATE by 0.587 m. It accepts two aerial 6–7 links; fixed-last accepts three aerial 6–7 links. Their joint aerial 6–7 ATE is 3.748 m versus 3.734 m, respectively. This is evidence of a benefit on aerial 5, not uniform superiority of refinement.

The online run rejected 36 pairs for insufficient matches and 89 for insufficient PnP geometric support. All three accepted constraints reproduce in the controlled refined ablation.

## Runtime and transport

Replay completed in 20.12 minutes (0.327× effective sensor time). The requested rate was 1.00×, with per-frame backpressure preserving ordered inputs. The replay uses a persistent Python executor and records publication, acknowledgement, and CPU timing for every camera batch.

Image drops: 0; loop-keyframe drops: 0; exhausted service requests: 0. All emitted keyframes and verified constraints reached the final graph. All published robust objectives were finite and non-increasing; 0 updates were rejected.

| Robot | VIO median / p95 (ms) | Sequence encoding median / p95 (ms) | Exchange through verification median / p95 (ms) |
|---|---:|---:|---:|
| a05 | 103.6 / 146.6 | 30.4 / 44.1 | 14.5 / 21.8 |
| a06 | 103.4 / 163.7 | 32.2 / 53.0 | 11.4 / 26.2 |
| a07 | 103.2 / 168.9 | 31.0 / 44.3 | 8.8 / 19.2 |
| a08 | 103.0 / 153.9 | 31.0 / 45.9 | 10.0 / 14.9 |

PGO median / p95: 29.5 / 69.4 ms; final solve: 33.8 ms.

Two incomplete attempts exposed approximately three-second camera-delivery stalls after about 1,500 frames. An independent subscriber observed affected frames had not yet been ingested by VIO. Replacing repeated attach/detach of the replay executor alone did not remove the stalls. The completed run opts into a 64 MiB shared-memory segment, 4 MiB transport messages, and 4,096 port queue entries via `FASTRTPS_DEFAULT_PROFILES_FILE`. The native robot/backend binaries, sensor values/order, and algorithm parameters are unchanged.

The [Fast DDS 2.6 documentation](https://fast-dds.docs.eprosima.com/en/2.6.x/fastdds/transport/shared_memory/shared_memory.html) lists a default 512 KiB shared-memory segment and cautions against undersized segments. The raw images here are 1.76 MB each. The local replay profile is opt-in and does not change live deployments.

The first buffered-transport attempt was also interrupted by command-session termination near aerial 7 frame 7,750, without an estimator error. The completed replay ran under a detached supervisor with its exit status recorded. All incomplete attempts remain explicitly marked as incomplete.

Observed application payload counts are in `validation_summary.json`: topic counts cover sequence announcements, verified loops and graph snapshots; service counts exclude CDR padding, DDS framing and retransmissions. They are not total network traffic. The recorded counters contain 33,278 service attempts and 33,277 responses, with the one-call difference at the backend shutdown snapshot. No exhausted service failure was reported, and the final graph audit confirms that all required keyframes and constraints arrived.

```json
{
  "topic_cdr_bytes": {
    "optimized_graph": 495670732,
    "sequence_announcements": 1043240,
    "loop_constraints": 1863
  },
  "service_totals": {
    "communication_queue_drops": 0,
    "graph_queue_drops": 0,
    "sensor_ingress_drops": 0,
    "service_attempts": 33278,
    "service_request_field_bytes": 867676,
    "service_response_field_bytes": 9869230,
    "service_responses": 33277
  },
  "replay_timing": {
    "publish_ms": {
      "median_ms": 23.030651493172627,
      "p95_ms": 31.00756398998783
    },
    "ack_wait_ms": {
      "median_ms": 135.4432925072615,
      "p95_ms": 193.4867350028071
    },
    "cpu_ms": {
      "median_ms": 39.58943549999816,
      "p95_ms": 108.45596350000619
    }
  }
}
```

## Reproduction and artifacts

Artifacts: `results/graco/multi_robot_f64_stationary_5_8_final/`. `experiment_manifest.json` records the committed native source, effective settings, model/config/executable hashes and replay changes. `experiment_results.json` contains the complete audited results; `refinement_comparison.json` records each identical retrieval pair; `playback.rrd` visualizes raw/corrected trajectories, sparse landmarks, selected pairs, loop edges and component status.

```bash
source scripts/source_multi_robot_ros2.bash
export OPENBLAS_NUM_THREADS=1
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/ros2/visloc_ros/config/fastdds_replay.xml"
.runtime/graco-venv/bin/python scripts/prepare_multi_robot_graco.py \
  --robots 5 6 7 8 --rate 1 --output results/graco/my_multi_robot_f64
.runtime/graco-venv/bin/python scripts/run_multi_robot_mission.py \
  results/graco/my_multi_robot_f64/mission.json --domain-id 226
```

The full experiment validates the existing inference engines in their online workload. No model conversion, VIO algorithm change, or PGO tuning was performed in this rerun. The earlier TensorRT parity and native estimator regression results remain recorded separately.
