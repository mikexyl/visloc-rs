# Native GTSAM pose-graph backend

GTSAM C++ is the **sole pose-graph optimizer** for the centralized multi-robot
backend and the single-robot JIST loop worker. The custom optimizer path,
experimental Rust GPS solver additions and Python process adapter were removed.
There is no optimizer selector or fallback. ROS2 nodes remain native Rust
`rclrs`; VIO, calibration, initialization, IMU processing and raw odometry are
unchanged. GPS remains disabled in ordinary profiles.

`crates/gtsam` provides a safe Rust snapshot API around a small C++ wrapper. Rust
owns the pose/factor arrays and passes fixed-layout POD buffers through a
synchronous C ABI. C++ catches exceptions and writes into caller-owned output
buffers. No Eigen/STL types, allocated buffers, Python interpreter, JSON IPC or
child process cross this boundary. Both callers execute the solver on their
existing optimization workers. Native `libgtsam.a` and `libcephes-gtsam.a` are
statically linked into the executables.

## Dependency and build

Pinned upstream: **GTSAM 4.3.0**, commit
`71a25ca36c084cbad1f872e812d6d97fbadfdb05`. The setup script checks this revision
and refuses a dependency with tracked modifications. Build prerequisites are a
C++17 compiler, CMake, Ninja and Eigen3 headers. No Python or ROS dependencies
are needed by the wrapper.

```sh
bash scripts/build_gtsam_native.sh
cargo test --offline -p visloc-gtsam -p visloc-multi-robot
cargo build --offline --release -p visloc-multi-robot --examples
# Native ROS build performs the local dependency setup if needed:
CARGO_NET_OFFLINE=true bash scripts/build_multi_robot_ros2.sh
```

The dependency is installed under `.runtime/gtsam-native/install`. Override
`VISLOC_GTSAM_ROOT` for a compatible static installation, or
`VISLOC_GTSAM_EIGEN_INCLUDE` for a nonstandard Eigen include directory. The local
build disables Boost, TBB, GPU/Python, unstable modules and architecture-specific
instructions. It uses GTSAM's standard Pose3 exponential chart and correct
BetweenFactor Jacobians. `VISLOC_GTSAM_BUILD_JOBS` controls build parallelism
(default four). Ordinary core/SfM builds that do not use either active VIO loop
pipeline do not depend on this wrapper.

Remove obsolete `optimizer` and `gtsam` process settings from backend `pgo`
configuration. They are rejected explicitly. Existing uncertainty and `gps`
settings retain their meanings; no new runtime solver configuration is needed.

## Factors and conventions

- Poses are **body-to-map**. Native
  `BetweenFactorPose3(to, from, T_to_from)` preserves the existing measured
  transform and residual frame. The complete information matrix, including
  cross-block entries, is reordered from translation/rotation to GTSAM's
  rotation/translation convention. Native Huber delta 3 applies to relative
  factors. Odometry measurements still come exclusively from raw VIO.
- Native `GPSFactorArm` supplies antenna prediction and its pose Jacobian. A
  C++ factor adapter chains them through linear-position/shortest-path-SLERP
  interpolation, horizontal covariance whitening, and the existing no-force
  radius. Up is removed entirely; any nonzero Up information is rejected.
  Native Huber or Geman–McClure handles GPS robustness. With `c=sqrt(lambda)`,
  Geman–McClure exactly implements the eliminated switch objective.
- Each disconnected component is solved separately. The root is constant until
  GPS alignment; afterwards it has exactly three parameters, east/north/yaw.
  Height and gravity remain fixed without an artificial high-weight prior.
  Factors touching that root chain native factor derivatives through this
  parameterization. Other relative factors are directly native GTSAM factors.
- GTSAM performs all linearization assembly, sparse multifrontal Cholesky,
  damping and Levenberg–Marquardt steps: at most 20 iterations, initial lambda
  `1e-4`, relative tolerance `1e-7`, absolute tolerance `1e-9`. There are no
  custom normal equations, sparse solvers or LM loops in either active backend.

C++ checks finite/non-increasing robust objective and preserved gauge before
returning a solution. Rust validates identities, finite rigid poses, gauge and
reports before replacing the snapshot. Errors retain the previous solution;
C++ exceptions cannot cross the ABI. GPS residuals and weights are returned by
these same native factors for diagnostics. Existing one-second scheduling,
final drain solve, histories, session alignment and restart recovery remain.

The old upstream `visloc-slam` graph library remains for its existing SfM and
legacy APIs; both active VIO loop backends have removed that dependency. The
experimental modifications to that old solver were reverted.

## Tests and reproduction

```sh
cmake -S crates/gtsam/native -B .runtime/gtsam-wrapper-tests -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/.runtime/gtsam-native/install"
cmake --build .runtime/gtsam-wrapper-tests --parallel 2
ctest --test-dir .runtime/gtsam-wrapper-tests --output-on-failure
cargo test --offline -p visloc-gtsam -p visloc-multi-robot
# After sourcing the built ROS environment:
ROS_DOMAIN_ID=224 .runtime/ros2-venv/bin/python scripts/test_gps_pose_graph_ros2.py
```

Native numerical tests cover both interpolation endpoints, exact timestamp
fixes, tilted-root derivatives, near-pi rotations, correlated horizontal
covariance, deadband, explicit switch-cost equivalence and altitude invariance.
Rust tests check ABI conventions, anisotropic edge cost, gauge, invalid inputs,
exception recovery, component joins, GPS jumps/bias, outages and session
isolation. ROS2 tests exercise the actual Rust executables with C++ GTSAM linked.

The six frozen South-ece ablations and all 879 recorded graph revisions are
replayed in `results/ucy/south_ece_gtsam_cpp_20261006`. These use immutable VIO
poses/loops and no estimator or reference input to optimization. External evo
uses identical reference timestamps, rigid alignment, no scale/timing fit,
20 ms association and common nominal antenna compensation. See its `REPORT.md`
for parity, timings, reference limitations and the previous Python comparison.

GPS policy and ingestion are documented in [gps_pose_graph.md](gps_pose_graph.md).
Sources: [GTSAM 4.3.0 release](https://github.com/borglab/gtsam/releases/tag/4.3.0),
[native GPS factors](https://borglab.github.io/gtsam/gpsfactor/).
