Generate the GRACO mission with `scripts/prepare_multi_robot_graco.py`.
Robot JSON files contain the calibration, VIO and TensorRT configuration paths,
robot roster, sensor QoS, image preprocessing, and output directory.

For a live camera, use the calibrated raw resolution/intrinsics/distortion in
`preprocess`, or set it to `null` for already undistorted images matching the
Basalt calibration. Set `reliable_sensors` to `false` for sensor-data QoS.
Remap `/<robot>/camera/image` and `/<robot>/imu` to the live sensor topics.
All camera timestamps and IMU timestamps must share the robot's sensor clock.

The centralized backend defaults to GPS-assisted global BA. Robot configs enable
`bundle_adjustment_enabled` and `gps.enabled` by default; backend configs use
`pgo.mode = "global_bundle_adjustment"` and `pgo.gps.enabled = true`. For visual
PGO, set the mode to `pose_graph`, disable GPS in both configs, and disable robot
BA observation export. Explicit settings in existing files remain authoritative.

Live GPS needs an IMU-frame antenna lever arm in `pgo.gps.lever_arms_m` and receiver
quality metadata. Use `gps.normalized_input=true` for normalized `GpsFix` messages;
a bare `NavSatFix` does not supply the default required HDOP. Missing/rejected GPS
leaves visual BA active. See [backend configuration](../../../docs/global_bundle_adjustment.md).
