//! Basalt-style deterministic IMU-only initial state construction.
use nalgebra::{Unit, UnitQuaternion, Vector3};
use thiserror::Error;

use crate::imu::{integrate_between, ImuPreintegratedDelta, SamplingError};
use crate::{BasaltNavState, ImuSample};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitializationConfig {
    pub gravity_m_s2: f64,
    pub minimum_static_samples: usize,
}

impl Default for InitializationConfig {
    fn default() -> Self {
        Self {
            gravity_m_s2: 9.81,
            minimum_static_samples: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialState {
    pub nav: BasaltNavState,
    pub gravity_world_m_s2: Vector3<f64>,
    pub first_camera_timestamp_ns: i64,
    pub preintegrated_to_camera: ImuPreintegratedDelta,
}

#[derive(Debug, Error, PartialEq)]
pub enum InitializationError {
    #[error("not enough static IMU samples")]
    InsufficientStaticSamples,
    #[error("static IMU average acceleration is degenerate")]
    DegenerateAcceleration,
    #[error("camera timestamp must be after the first IMU timestamp")]
    InvalidCameraTimestamp,
    #[error("IMU sampling failed: {0}")]
    Sampling(#[from] SamplingError),
}

/// Estimate gyro bias and gravity direction from the initial stationary window.
/// The measured specific force points opposite world gravity; the returned
/// rotation maps the body-frame force direction to world up.
pub fn estimate_initial_state(
    samples: &[ImuSample],
    camera_timestamp_ns: i64,
    config: InitializationConfig,
) -> Result<InitialState, InitializationError> {
    if samples.len() < config.minimum_static_samples.max(2) {
        return Err(InitializationError::InsufficientStaticSamples);
    }
    if camera_timestamp_ns <= samples[0].timestamp_ns {
        return Err(InitializationError::InvalidCameraTimestamp);
    }
    let mean_gyro = samples
        .iter()
        .fold(Vector3::zeros(), |sum, s| sum + s.gyro_rad_s)
        / samples.len() as f64;
    let mean_accel = samples
        .iter()
        .fold(Vector3::zeros(), |sum, s| sum + s.accel_m_s2)
        / samples.len() as f64;
    let norm = mean_accel.norm();
    if !norm.is_finite() || norm < 1e-9 {
        return Err(InitializationError::DegenerateAcceleration);
    }
    let up_body = Unit::new_normalize(mean_accel);
    let up_world = Vector3::z_axis();
    let rotation = UnitQuaternion::rotation_between(&up_body, &up_world)
        .unwrap_or_else(UnitQuaternion::identity);
    let gravity = -config.gravity_m_s2.abs() * up_world.into_inner();
    let delta = integrate_between(
        samples,
        samples[0].timestamp_ns,
        camera_timestamp_ns,
        mean_gyro,
        Vector3::zeros(),
        None,
    )?;
    let delta_velocity_world = rotation.transform_vector(&delta.delta_velocity);
    let nav = BasaltNavState {
        imu_to_world: visloc_core::geometry::SE3::new(
            rotation * delta.delta_rotation,
            rotation.transform_vector(&delta.delta_position)
                + gravity * (0.5 * delta.delta_time * delta.delta_time),
        ),
        velocity_world_m_s: delta_velocity_world + gravity * delta.delta_time,
        gyro_bias_rad_s: mean_gyro,
        accel_bias_m_s2: Vector3::zeros(),
    };
    Ok(InitialState {
        nav,
        gravity_world_m_s2: gravity,
        first_camera_timestamp_ns: camera_timestamp_ns,
        preintegrated_to_camera: delta,
    })
}

/// Estimate the body-frame "up" (specific-force) direction over a window that
/// starts at `start_timestamp_ns`, compensating for device rotation.
///
/// Upstream Basalt derives the world frame's roll/pitch from a *single* IMU
/// sample, which assumes the device is static at the first camera.  On
/// sequences that start while the wearer is already rotating (LaMAria's
/// head-worn capture), that one sample can be several degrees off true
/// gravity.  Here every accelerometer sample in the window is rotated into the
/// start frame with the gyro before averaging, so a pure rotation cancels out
/// and the mean is a much more stable estimate.  It is still biased by genuine
/// linear acceleration, so callers should keep the window short (~0.2 s) and
/// treat the result as an opt-in improvement over the single-sample default.
pub fn estimate_up_body_from_window(
    samples: &[ImuSample],
    start_timestamp_ns: i64,
    window_ns: i64,
) -> Option<Vector3<f64>> {
    if window_ns <= 0 {
        return None;
    }
    let end_timestamp_ns = start_timestamp_ns.saturating_add(window_ns);
    let mut rotation = UnitQuaternion::identity();
    let mut previous: Option<ImuSample> = None;
    let mut accel_sum = Vector3::zeros();
    let mut count = 0usize;
    for sample in samples {
        if sample.timestamp_ns < start_timestamp_ns {
            previous = Some(*sample);
            continue;
        }
        if sample.timestamp_ns > end_timestamp_ns {
            break;
        }
        if let Some(previous) = previous {
            let dt = (sample.timestamp_ns - previous.timestamp_ns) as f64 * 1e-9;
            if dt > 0.0 && dt.is_finite() {
                rotation *= UnitQuaternion::from_scaled_axis(sample.gyro_rad_s * dt);
            }
        }
        let accel = sample.accel_m_s2;
        if accel.iter().all(|value| value.is_finite()) && accel.norm_squared() > 1.0e-18 {
            // `rotation` maps body@start vectors to body@sample-relative-to-start,
            // so rotating the measured specific force by it returns the start-frame
            // direction.
            accel_sum += rotation.transform_vector(&accel);
            count += 1;
        }
        previous = Some(*sample);
    }
    if count < 2 {
        return None;
    }
    let mean = accel_sum / count as f64;
    if !mean.iter().all(|value| value.is_finite()) || mean.norm_squared() <= 1.0e-18 {
        return None;
    }
    Some(mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(t: i64, g: Vector3<f64>, a: Vector3<f64>) -> ImuSample {
        ImuSample::new(t, g, a)
    }

    #[test]
    fn stationary_fixture_initializes_gravity_and_zero_motion() {
        let samples = [
            s(0, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81)),
            s(
                1_000_000_000,
                Vector3::zeros(),
                Vector3::new(0.0, 0.0, 9.81),
            ),
            s(
                2_000_000_000,
                Vector3::zeros(),
                Vector3::new(0.0, 0.0, 9.81),
            ),
        ];
        let x =
            estimate_initial_state(&samples, 500_000_000, InitializationConfig::default()).unwrap();
        assert!((x.gravity_world_m_s2 - Vector3::new(0.0, 0.0, -9.81)).norm() < 1e-12);
        assert!(x.nav.velocity_world_m_s.norm() < 1e-12);
        assert!(x.nav.imu_to_world.rotation.angle() < 1e-12);
    }

    #[test]
    fn tilted_gravity_fixture_aligns_body_force_to_world_up() {
        let tilt = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.4);
        let a = tilt
            .inverse()
            .transform_vector(&Vector3::new(0.0, 0.0, 9.81));
        let samples = [
            s(0, Vector3::zeros(), a),
            s(1_000_000_000, Vector3::zeros(), a),
        ];
        let x =
            estimate_initial_state(&samples, 500_000_000, InitializationConfig::default()).unwrap();
        assert!(
            (x.nav.imu_to_world.rotation.transform_vector(&a.normalize())
                - Vector3::new(0.0, 0.0, 1.0))
            .norm()
                < 1e-10
        );
    }

    #[test]
    fn constant_gyro_bias_is_recorded_and_removed() {
        let bias = Vector3::new(0.01, -0.02, 0.03);
        let samples = [
            s(0, bias, Vector3::new(0.0, 0.0, 9.81)),
            s(1_000_000_000, bias, Vector3::new(0.0, 0.0, 9.81)),
        ];
        let x =
            estimate_initial_state(&samples, 500_000_000, InitializationConfig::default()).unwrap();
        assert!((x.nav.gyro_bias_rad_s - bias).norm() < 1e-12);
        assert!(x.preintegrated_to_camera.delta_rotation.angle() < 1e-12);
    }

    #[test]
    fn first_camera_endpoint_is_integrated_by_interpolation() {
        let samples = [
            s(0, Vector3::zeros(), Vector3::new(2.0, 0.0, 0.0)),
            s(1_000_000_000, Vector3::zeros(), Vector3::new(2.0, 0.0, 0.0)),
        ];
        let x =
            estimate_initial_state(&samples, 500_000_000, InitializationConfig::default()).unwrap();
        assert!((x.preintegrated_to_camera.delta_time - 0.5).abs() < 1e-12);
        assert!((x.preintegrated_to_camera.delta_velocity.x - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gyro_compensated_window_recovers_up_during_rotation() {
        // Device rotates about X while gravity stays fixed in the world; the
        // measured specific force is gravity expressed in the rotating body
        // frame.  The gyro-compensated mean must recover the body-frame up at
        // the window start (the axis-angle of `R(t)` applied to world up).
        let omega = Vector3::new(0.7, 0.0, 0.3); // rad/s
        let mut samples = Vec::new();
        for k in 0..=200 {
            let t = k as i64 * 1_000_000; // 1 kHz, 200 ms
            let rotation = UnitQuaternion::from_scaled_axis(omega * (t as f64 * 1e-9));
            let accel = rotation.inverse_transform_vector(&Vector3::new(0.0, 0.0, 9.806));
            samples.push(s(t, omega, accel));
        }
        let up = estimate_up_body_from_window(&samples, 0, 200_000_000).unwrap();
        let expected = UnitQuaternion::from_scaled_axis(omega * 0.0)
            .inverse_transform_vector(&Vector3::new(0.0, 0.0, 1.0));
        assert!(
            (up.normalize() - expected).norm() < 2.0e-3,
            "recovered {:?} vs {:?}",
            up.normalize(),
            expected
        );
        // The naive single-sample direction would be off once rotation is large.
        let naive = samples[samples.len() - 1].accel_m_s2.normalize();
        assert!((naive - expected).norm() > 0.1);
    }
}
