# Jetson viewer deployment — 2026-09-17

Deployed successfully to `mikexyl@192.168.55.1`, workspace
`/home/mikexyl/workspaces/visloc-rs`.

Viewer: http://192.168.55.1:8090 (USB), http://192.168.0.156:8090 (configured Wi-Fi).
USB URL verified; Wi-Fi was not separately tested in this deployment.

## Installed changes

- Host viewer: accumulated map store, cached binary map endpoint, cached map
  rendering, resource collector and live charts, all required JS/Python assets.
- Jetson GPU counter discovery updated for the board's actual path:
  `/sys/devices/platform/bus@0/17000000.gpu/load`.
- Rebuilt ARM64 publisher with active-ID-range sampling rather than first-ID
  truncation. Camera image includes all new viewer helper modules/assets.
- Restarted `visloc-viewer.service`; it is active and retains `--controls`.
- **Start visloc** and **Stop visloc** remain visible on desktop and phone layouts.
  Start is enabled; Stop is disabled because capture remains stopped.
- Other existing DPVO/coordinator/network containers were left running.

## Verification

- On-board unit tests: 8 resource, 8 map, 7 controller tests passed.
- On-board HTTP integration test passed using a disposable loopback server:
  map retention, binary XYZ, unchanged response 304, session isolation/reset,
  input validation, and required static assets. Synthetic frames were not sent
  to the deployed viewer service.
- Chrome checks against the actual deployed page: desktop and 390×844 layout,
  Start/Stop visibility and enabled state, fresh CPU/RAM/GPU measurements,
  shared-memory labeling, and no JavaScript errors.
- New camera image ran 80 recorded EuRoC frames through its actual
  `basalt_stream_vio` binary: **80/80**, valid finite poses/map points,
  **54–209 active map points**, **9.02 s** wall time. This was a functional smoke
  check using `--network none`, no USB devices, and no physical camera access.
- The first test launcher needed Pillow, absent from the production camera
  image. The successful test used the image's existing OpenCV support; no
  production dependencies were added solely for testing.

[Deployed desktop](desktop.png) · [Deployed phone](mobile.png) ·
[Browser result](browser.json)

Image IDs:

- `visloc-rs:jetson`:
  `sha256:3d78ffb08d6400f654d23ebc397c112c59c64bcad3d03a44628895e0a8d2374a`
- `visloc-rs:realsense-browser`:
  `sha256:dd69fc5fcb914995668d3f7e4ab1da8f09986fdd06bf2ecc03509b60085a4bca`

Build log on board: `/tmp/visloc-map-resource-build.log`.
Recorded-data test output:
`results/map-resource-image-smoke-20260917/validated.json` and `native/`.

Before deployment, files were backed up to
`deploy-backups/20260917-map-resources/`. Previous images remain tagged
`visloc-rs:jetson-before-map-resources-20260917` and
`visloc-rs:realsense-browser-before-map-resources-20260917`.
