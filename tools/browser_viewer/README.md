# Online browser VIO viewer

## Jetson deployment

The viewer and its Start/Stop controls are at http://192.168.55.1:8090 over Jetson USB networking, or http://192.168.0.156:8090 over Wi-Fi. The `visloc-viewer.service` system service hosts the page and starts at boot; VIO stays off until **Start visloc** is pressed. **Stop visloc** releases the camera and keeps the page available. The laptop or phone is only a browser client. VIO publishes through `host.docker.internal` (Docker host gateway), independently of Wi-Fi. The viewer binds to all interfaces and does not wait for Wi-Fi or Docker readiness at boot.

Install the service on the Jetson from the repository root:

```bash
sudo install -m 644 docker/jetson/visloc-viewer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now visloc-viewer.service
```

The service runs as `mikexyl` with Docker access. Only fixed VIO start/stop actions are exposed, guarded by a per-process control token and same-origin checks. Operations are serialized. Start attempts camera initialization regardless of other containers' running state; other workloads are never stopped. This is a trusted-LAN interface without a login.

VIO uses `visloc-rs:realsense-browser`, launched by `docker/jetson/run_browser.sh`. Capture does not start automatically at boot. The obsolete separate panel and viewer container are no longer used.

Run lifecycle tests with `python3 tests/test_browser_controller.py`. Desktop and phone-sized layout are checked in Chromium; physical iPhone Safari has not been tested.

## Accumulated 3D map

The orange point cloud is the accumulated explored map for the current publisher
session, not a snapshot of currently tracked landmarks. The server retains points
when their IDs leave the active VIO window. A retained track's coordinates are
replaced when optimization provides a new position. New browser connections and
reloads receive the accumulated map; a new publisher stream or server restart
clears it. The map is kept in memory, not saved to disk.

**Fit map** frames the whole retained cloud. **Fit path** frames the displayed
trajectory. Drag/scroll disables automatic fitting so incoming points do not
fight a manually chosen view. **Map points** toggles the cloud and pauses map
fetches while hidden. New sessions automatically fit the accumulating map.

This is a sparse visualization map of last-known VIO world coordinates. Retired
landmarks are not globally reoptimized, and later loop-closure corrections are
not available in this stream. Missing IDs are retained because the wire protocol
does not distinguish marginalization from outlier rejection. It is not dense
reconstruction or an authoritative persistent SLAM map.

### Computation and memory budget

- One representative per 0.10 m voxel, up to **30,000 retained points**. On reaching
  the cap, voxel width doubles and the whole cloud is merged. This lowers density
  across the explored area instead of throwing away older regions. The panel
  reports actual retained count and voxel width.
- The retained ID index is bounded by the same cap. Each incoming update examines
  at most 3,000 points; it does not scan the whole historical map. The Rust exporter
  samples across the active ID range when over budget rather than sending only
  the oldest IDs. The existing asynchronous publisher still drops pending frames
  under backpressure; the map therefore covers points received by the server.
- A worker serializes at most once per second. `GET /api/map?stream=...&since=...`
  returns cached little-endian Float32 XYZ triples (maximum **360,000 bytes**),
  metadata in `X-Map-*` headers, or **304** with no body when unchanged. A mismatched
  stream returns 409. Pose/image packets no longer contain point arrays.
- Browser map polling runs at 1 Hz independently of pose/image polling and pauses
  in hidden tabs. Geometry stays in a Float32Array. Bounds are calculated only
  on map updates. A cached transparent canvas reuses projected points while the
  view/map is unchanged; transforms are computed once per projection with no
  per-point array allocation. Scene redraws are coalesced and capped at 30 FPS.
- Coordinates must be finite and within ±10,000 km per world axis for safe Float32
  visualization. Track IDs remain 64-bit integers server-side; IDs are not sent
  to JavaScript, avoiding precision loss above 2^53.

Checks:

```sh
python3 tests/test_browser_map.py
# With Playwright and Chrome, against a disposable local live viewer:
node tests/test_browser_map_ui.cjs http://127.0.0.1:8092 /tmp/visloc-map-ui
```

The browser test publishes synthetic map sessions, checks retention, corrections,
late joining, session isolation/reset, capacity coarsening, mobile layout, and
cached projection reuse at 30,000 points. Do not run it against a live capture
server: its synthetic publisher intentionally replaces the displayed session.
Real EuRoC VIO and local performance evidence are recorded in
[the retained-map report](../../work/retained_map_20260917/REPORT.md).

## Optional laptop hosting

Run on the laptop:

```bash
python3 tools/browser_viewer/live_server.py --port 8090
```

Open http://localhost:8090 or, from a phone on the same LAN, `http://<laptop-IP>:8090`.

The RealSense capture process can send each new VIO result and its matching stereo JPEGs directly to this server:

```bash
python3 scripts/realsense/live.py --browser-connect http://<laptop-IP>:8090 [other capture arguments]
```

`BROWSER_CONNECT` is the equivalent environment variable. The publisher uses a background thread and a single pending frame so a slow or disconnected viewer cannot accumulate a backlog or block estimation. The browser requests the latest coherent pose/image packet every 150 ms, keeps up to 3,000 displayed poses, follows incoming updates automatically, and shows a stopped-stream status after three seconds without input. A new publisher session resets the displayed trajectory. There is no replay timeline in online mode.

## Laptop test with running VIO

With the Rust `basalt_stream_vio` example built and Pillow installed:

```bash
python3 tools/browser_viewer/test_online_euroc.py \
  --output target/browser-viewer-test/new-online-run --frames 3600
```

Choose a fresh output directory each time. This feeds EuRoC stereo and IMU measurements to a running Rust estimator and publishes newly computed results. It exercises the online pipeline with recorded sensor input; it is not a live physical-camera test and does not read a saved trajectory. It ends when the requested frame count is reached.

Validated locally in Chromium: advancing online frames, stereo images, hidden replay controls, desktop and 390×844 phone layout, touch rotation, and no JavaScript exceptions. Physical iPhone Safari has not been tested.

## Optional saved-run replay

The older replay mode remains separately available:

```bash
python3 tools/browser_viewer/server.py \
  --run-dir target/euroc_vio/MH_01_easy \
  --euroc-dir /data/euroc/MH_01_easy --port 8091
```

## Live system resources

Both online and replay viewers include a **System resources** panel. It refreshes
once per second and charts the last two minutes of CPU, RAM, GPU, and VRAM usage.
The displayed hostname identifies the **viewer server host**, not the browser's
machine or a remote VIO publisher. Replay mode still shows current resource use,
not historical resource use from the recording. All values are host-wide; these
are not per-process/container measurements or resource limits.

- CPU: interval utilization averaged across all logical cores from `/proc/stat`.
  Guest time is not double-counted; idle and I/O-wait time are excluded from busy
  time. The first sample has no CPU percentage until a second counter is read.
- RAM: `(MemTotal - MemAvailable) / MemTotal` from `/proc/meminfo`, with used/total
  GiB displayed. Reclaimable cache is excluded via the kernel's availability
  estimate. Swap is not included.
- NVIDIA GPUs: utilization, used/total dedicated VRAM, and temperature from
  `nvidia-smi`. Every GPU gets separate utilization and memory cards.
- Jetson: integrated GPU utilization comes from the readable nvgpu sysfs `load`
  node. The value is divided by ten, as documented in the
  [NVIDIA Orin guide](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html).
  GPU memory is labeled **shared system RAM**; a separate VRAM allocation count
  is not available through this collector. Unreadable load nodes show unavailable.

A single background `ResourceMonitor` serves every client from a bounded
120-sample cache at `GET /api/resources`. The handler never starts GPU queries.
NVIDIA queries time out after 800 ms, so a missing/blocked driver does not block
VIO requests. No Python dependencies or extra privileges are required beyond
read access to proc/sysfs and NVIDIA driver access for `nvidia-smi`.

Polling pauses in hidden browser tabs. Stale measurements (over three seconds)
and disconnected servers are visibly marked and their current values cleared;
unsupported/unavailable metrics show a dash rather than a misleading zero.
Graphs preserve gaps when samples are unavailable. Monitoring continues while
VIO is stopped. Restart the viewer server after updating Python files, then
reload the page; no VIO restart is necessary for the panel itself.

Checks:

```sh
python3 tests/test_browser_resources.py
python3 tests/test_browser_controller.py
# With a current Playwright package and Chrome installed, against a running viewer:
node tests/test_browser_resources_ui.cjs http://127.0.0.1:8092 /tmp/visloc-resource-ui
```

The browser smoke test requires real NVIDIA metrics, checks advancing data,
responsive desktop/390px layout, and uses mocked API responses to verify stale,
missing-GPU, Jetson, multiple-GPU, and disconnect/reconnect presentation.
`PLAYWRIGHT_PATH` can point to an installed Playwright module; `CHROME_PATH`
overrides `/opt/google/chrome/chrome`. Deployed to the Jetson on 2026-09-17. Its Orin NX GPU load is read from
`/sys/devices/platform/bus@0/17000000.gpu/load`. Live CPU/RAM/GPU telemetry and
shared-memory labeling were verified on the board; desktop and mobile Chrome
checks confirmed the Start/Stop controls remain visible. The retained-map HTTP
protocol also passed on a disposable on-board server.
