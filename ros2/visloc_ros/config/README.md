Generate the GRACO mission with `scripts/prepare_multi_robot_graco.py`.
Robot JSON files contain the calibration, VIO and TensorRT configuration paths,
robot roster, sensor QoS, image preprocessing, and output directory.

For a live camera, use the calibrated raw resolution/intrinsics/distortion in
`preprocess`, or set it to `null` for already undistorted images matching the
Basalt calibration. Set `reliable_sensors` to `false` for sensor-data QoS.
Remap `/<robot>/camera/image` and `/<robot>/imu` to the live sensor topics.
All camera timestamps and IMU timestamps must share the robot's sensor clock.
