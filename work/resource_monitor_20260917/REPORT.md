# Resource monitor verification — 2026-09-17

Implemented live host CPU/RAM and NVIDIA GPU/VRAM charts in the online and replay
web viewers. One background sampler per server, 1 Hz, bounded 120-sample history.
All measurements describe the viewer host and are independent of the VIO stream.

Passed:

- Seven collector tests: CPU deltas/guest time, RAM availability, multi-GPU CSV,
  missing/timeout driver, bounded independent snapshots, counter resets/missing
  proc files, Jetson load/shared memory, and sampler shutdown.
- Seven existing viewer-controller lifecycle tests.
- Python compilation and JavaScript syntax checks.
- Playwright 1.62 / installed Chrome: live laptop CPU/RAM/GPU/VRAM, advancing
  samples, bounded history, desktop and 390×844 mobile layout without horizontal
  overflow, stale measurements, absent GPU metrics, Jetson shared-memory labels,
  two GPUs, network failure/recovery, and no page JavaScript errors.

Real API sample: RTX 4070 Laptop GPU, 40% GPU, 0.90/8.00 GiB VRAM,
6.42% host CPU across 20 logical CPUs, 8.04/38.84 GiB system RAM. These are
momentary desktop measurements, not an inference benchmark.

[Desktop screenshot](desktop.png) · [Mobile screenshot](mobile.png)

Local preview: http://localhost:8092. Remote Jetson deployment/hardware testing
was not performed. Jetson load interpretation follows NVIDIA's documented sysfs
interface; its UI and parser branches were tested using fixtures.
