# Accumulated map visualization — 2026-09-17

The online viewer now shows a session's explored sparse map rather than replacing
it with the current active landmark window. Points survive retirement, retained
IDs receive position corrections, and clients joining/reloading receive the
server's accumulated map. New publisher sessions reset it immediately.

## Real VIO validation

Ran the rebuilt Rust `basalt_stream_vio` on 600 EuRoC MH_01_easy stereo/IMU frames
through `test_online_euroc.py` and the asynchronous BrowserStream publisher.

- Final received frame: **599** (600 frames / approximately 30 seconds of sensor data).
- Accumulated voxel representatives: **1,063**, voxel size **0.10 m**.
- Logged active-window point counts at frames 0/100/200/300/400/500:
  **61 / 136 / 90 / 43 / 57 / 58**. These are sampled active counts, not the peak
  over every frame or a unique-landmark total.
- Projection of the real accumulated cloud: **0.5 ms** in the local Chrome check.
- Desktop and 390×844 layout visually inspected; no page JavaScript errors.

[Real VIO desktop](euroc_desktop.png) · [Real VIO mobile](euroc_mobile.png) ·
[Machine-readable result](real_vio.json)

Output trajectory: `target/browser-viewer-test/retained-map-20260917/`.
The VIO publisher has finished. The local viewer at http://localhost:8092 retains
its map until a new publisher session arrives or the server restarts.

## Bounded cost

- Maximum **30,000 points**, one per voxel; adaptive global voxel coarsening
  preserves old/new regions instead of dropping the oldest history.
- Maximum map body **360,000 bytes** (little-endian Float32 XYZ).
- Map serialization and browser map polling: **1 Hz**; unchanged map returns 304.
  Point arrays are removed from pose/image response packets.
- Cached canvas projection: unchanged map/view needs only a bitmap composite.
  Transform trigonometry is calculated once per projection, with no per-point
  array allocation. Redraws are coalesced and limited to 30 FPS.
- Browser geometry buffer at capacity: **360,000 bytes**, plus a viewport-sized
  canvas cache. Hidden tabs/hidden maps do not fetch geometry.
- Local Python microcheck at 30,000 points: approximately **9.52 MiB peak traced
  Python allocations**, **7.81 ms serialization**, **5.44 ms** updating 3,000
  moved points. Tracing includes input construction; it is not total process RSS.
- Local Chrome stress check at 30,000 points: **15 ms** full CPU projection,
  **zero additional projections** over 20 unchanged redraws. The entire sequence
  (one rebuild + 20 cached redraws) took **19.8 ms**. These are local observations,
  not GPU completion timings or performance guarantees for Jetson/mobile devices.
- Adding a distant area beyond capacity coarsened to 0.40 m and 16,082 points;
  both the original area and the new area remained represented.

[Synthetic capacity test](browser.json) · [30k-point desktop](desktop.png) ·
[30k-point phone layout](mobile.png)

## Checks

- 8 map unit tests: retirement, empty windows, position corrections without
  duplicate ghosts, stream reset, cached bytes/revision, voxel collisions,
  capacity/coverage, invalid coordinates, and exact large track IDs.
- 14 existing resource/controller tests passed.
- Browser integration: retention across disjoint windows, corrected coordinates,
  empty updates, HTTP 304, stream mismatch, reload/late join, capacity and cached
  projection reuse, mobile layout, and new-session isolation.
- Rust release example build, Python compilation, JavaScript syntax checks,
  and whitespace checks passed.

This is an in-memory sparse visualization map using the last received world
coordinates. It does not globally optimize retired landmarks or apply later
loop closures, and missing IDs cannot distinguish retirement from rejected
outliers with the present publisher protocol. Publisher backpressure can skip
frames; the accumulated map covers received landmarks. Remote Jetson deployment
and performance testing have not been performed.

Reproduce with `tests/test_browser_map.py`, `tests/test_browser_map_ui.cjs`, and
`tools/browser_viewer/test_online_euroc.py`. The synthetic browser test deliberately
publishes new sessions and should run against a disposable viewer.
