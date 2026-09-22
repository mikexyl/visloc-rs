# GRACO aerial IMU uncertainty tuning

On 2026-09-21, aerial-05-40m was replayed with the left camera and IMU,
800 × 550 images, the supplied aerial calibration, the existing VIO config,
and f32 arithmetic. Only the IMU white-noise and bias random-walk standard
deviations changed. No camera geometry, timestamps, frontend settings,
trajectory scale, or loop thresholds were adjusted.

The original **raw VIO** SE(3) ATE was **3.471971 m**. The previously reported
**3.846168 m** was the separate loop-corrected trajectory. All full aerial-05
comparisons below contain the same 5,926 timestamps, covering 296.25 seconds.
ATE uses rigid alignment without scale fitting. The Sim(3) scale is diagnostic
only; neither estimation nor Rerun applies a reference scale correction.

| White-noise std multiplier | Bias-walk std multiplier | Full raw ATE | Diagnostic excess scale |
| ---: | ---: | ---: | ---: |
| 1.0 | 1.0 | 3.471971 m | 4.010% |
| **0.75** | **0.75** | **2.445646 m** | **2.097%** |
| 0.9 | 0.9 | 3.713637 m | 4.367% |
| 4.0 | 4.0 | 17.374689 m | 22.218% |

The 0.75/0.75 setting reduced raw ATE by **29.56%** on the tuning sequence.
Both types of standard deviation are 25% smaller, so their covariances are
0.5625 times the calibrated values.

The full loop-enabled repeat reproduced all raw poses exactly. It accepted
one loop, dropped zero keyframes, and finished at **2.636699 m** loop-corrected
ATE, compared with the original **3.846168 m**. Loop closure still worsens the
tuned raw result by 0.191 m; this tuning does not fix that separate issue.
JIST retains its 0.8 threshold, five-keyframe sequences, and pre-ranking
covisibility exclusion. The saved event audit verifies that excluded local
sequences never enter candidate selection.

As a separate check, all **5,563 frames** of aerial-08-25m were replayed with
0.75/0.75, without further tuning. Its raw ATE was essentially unchanged:
**1.057391 m** versus **1.058060 m** at 1.0/1.0. Diagnostic excess scale changed
from 0.928% to 1.042%.

Reproduce the aerial-05 setting with the existing TensorRT loop bundle:

```bash
target/graco-venv/bin/python scripts/run_graco_vio.py \
  --bag /data/graco/aerial-05-40m --camera-mode mono \
  --imu-noise-scale 0.75 --imu-bias-scale 0.75 \
  --online-loop-config target/loop_models/loop_config.json \
  --output target/graco_vio/aerial-05-40m_tight_imu_repeat
```

Omit `--online-loop-config` to reproduce raw VIO alone. Each invocation needs
a fresh output directory. The replay CLI retains 1.0/1.0 defaults; these are
explicit experimental settings, not a replacement sensor calibration.

The completed loop-enabled recording, including left-camera feature tracks,
is `target/graco_vio/aerial-05-40m_imu_n0p75_b0p75_jist_20260921/playback.rrd`.
The independent aerial-08 result is in
`target/graco_vio/aerial-08-25m_imu_n0p75_b0p75_20260921`.

Several other settings diverged during takeoff: 0.5/0.5, 0.75/1.0,
1.0/0.75, 1.0/1.5, 1.0/4.0, 1.25/4.0, 2.0/2.0 and 2.0/10.0. Diverged
trials were rejected, rather than scored on a truncated stable interval.
This non-monotonic takeoff sensitivity needs further diagnosis. The unchanged
1,000-frame control exactly reproduces the corresponding original raw poses.

Detailed local results, including rejected trials, are saved in
`target/graco_vio/aerial-05-40m_imu_tuning_20260921/comparison.json` and
`comparison.md`. Their audit independently recomputes ATE from CSV positions
using an SVD rigid fit, checks full-run timestamp coverage, verifies zero right
camera observations, and compares all non-noise calibration fields, source
YAML hashes and the VIO configuration against the baseline. Ground truth is
used for offline tuning/evaluation, never as an estimator input.
