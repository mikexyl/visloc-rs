//! Navigation at the true observation time, without an extra optimized Nav15.
use super::{integrate_split, BasaltNavState, ImuSample};
use crate::imu::{BiasRandomWalkNoise, ImuNoiseModel, ImuPreintegratedDelta};
use nalgebra::{Matrix3, SMatrix, Vector3};
pub(super) type Matrix15 = SMatrix<f64, 15, 15>;

#[derive(Debug, Clone)]
pub(super) struct EpochMotion {
    delta: ImuPreintegratedDelta,
    backward: bool,
    gravity: Vector3<f64>,
    bias_noise: BiasRandomWalkNoise,
    pub sample: ImuSample,
}
fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0., -v.z, v.y, v.z, 0., -v.x, -v.y, v.x, 0.)
}
fn left_jacobian(phi: Vector3<f64>) -> Matrix3<f64> {
    let k = skew(phi);
    let t = phi.norm();
    if t < 1e-6 {
        Matrix3::identity() + k * 0.5 + k * k / 6.
    } else {
        Matrix3::identity() + k * ((1. - t.cos()) / t.powi(2)) + k * k * ((t - t.sin()) / t.powi(3))
    }
}
impl EpochMotion {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        samples: &[ImuSample],
        anchor: i64,
        epoch: i64,
        nav: &BasaltNavState,
        gravity: Vector3<f64>,
        noise: ImuNoiseModel,
        bias_noise: BiasRandomWalkNoise,
        nominal_period: f64,
    ) -> Result<Self, String> {
        let sample = crate::imu::interpolate_at(samples, epoch)
            .map_err(|e| format!("GNSS epoch IMU sample: {e:?}"))?;
        let delta = if anchor == epoch {
            ImuPreintegratedDelta::identity(nav.gyro_bias_rad_s, nav.accel_bias_m_s2)
        } else {
            integrate_split(
                samples,
                anchor.min(epoch),
                anchor.max(epoch),
                nav.gyro_bias_rad_s,
                nav.accel_bias_m_s2,
                Some(noise),
                nominal_period,
            )
            .map_err(|e| format!("keyframe GNSS preintegration: {e:?}"))?
        };
        Ok(Self {
            delta,
            backward: epoch < anchor,
            gravity,
            bias_noise,
            sample,
        })
    }
    fn forward_jacobian(&self, start: &BasaltNavState) -> Matrix15 {
        let (_, dv, dp) = self
            .delta
            .corrected(start.gyro_bias_rad_s, start.accel_bias_m_s2);
        let r = start
            .imu_to_world
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let mut f = Matrix15::identity();
        f.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(r * dp)));
        f.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(Matrix3::identity() * self.delta.delta_time));
        f.fixed_view_mut::<3, 3>(0, 9)
            .copy_from(&(r * self.delta.jacobian_position_gyro_bias));
        f.fixed_view_mut::<3, 3>(0, 12)
            .copy_from(&(r * self.delta.jacobian_position_accel_bias));
        let phi =
            self.delta.jacobian_rotation_gyro_bias * (start.gyro_bias_rad_s - self.delta.bias_gyro);
        f.fixed_view_mut::<3, 3>(3, 9)
            .copy_from(&(r * left_jacobian(phi) * self.delta.jacobian_rotation_gyro_bias));
        f.fixed_view_mut::<3, 3>(6, 3).copy_from(&(-skew(r * dv)));
        f.fixed_view_mut::<3, 3>(6, 9)
            .copy_from(&(r * self.delta.jacobian_velocity_gyro_bias));
        f.fixed_view_mut::<3, 3>(6, 12)
            .copy_from(&(r * self.delta.jacobian_velocity_accel_bias));
        f
    }
    pub fn evaluate(&self, anchor: &BasaltNavState) -> (BasaltNavState, Matrix15) {
        let (dr, dv, dp) = self
            .delta
            .corrected(anchor.gyro_bias_rad_s, anchor.accel_bias_m_s2);
        let dt = self.delta.delta_time;
        let mut n = anchor.clone();
        if self.backward {
            n.imu_to_world.rotation *= dr.inverse();
            n.velocity_world_m_s -= self.gravity * dt + n.imu_to_world.rotation * dv;
            n.imu_to_world.translation -= n.velocity_world_m_s * dt
                + self.gravity * (0.5 * dt * dt)
                + n.imu_to_world.rotation * dp;
            let f = self
                .forward_jacobian(&n)
                .try_inverse()
                .expect("invertible navigation transition");
            (n, f)
        } else {
            n.imu_to_world.translation += anchor.velocity_world_m_s * dt
                + self.gravity * (0.5 * dt * dt)
                + anchor.imu_to_world.rotation * dp;
            n.velocity_world_m_s += self.gravity * dt + anchor.imu_to_world.rotation * dv;
            n.imu_to_world.rotation *= dr;
            (n, self.forward_jacobian(anchor))
        }
    }
    pub fn covariance(&self, anchor: &BasaltNavState) -> Matrix15 {
        let (epoch, transition) = self.evaluate(anchor);
        let start = if self.backward { &epoch } else { anchor };
        let r = start
            .imu_to_world
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let mut g = SMatrix::<f64, 9, 9>::zeros();
        for i in [0, 3, 6] {
            g.fixed_view_mut::<3, 3>(i, i).copy_from(&r);
        }
        let mut q = Matrix15::zeros();
        q.fixed_view_mut::<9, 9>(0, 0)
            .copy_from(&(g * self.delta.covariance * g.transpose()));
        // Frozen-dynamics Brownian bias bridge: endpoint variance dt, cross
        // covariance dt/2, integrated motion variance dt/3. This is an
        // approximation to bias diffusion during the short propagation.
        let dt = self.delta.delta_time;
        let f = self.forward_jacobian(start);
        let b = f.fixed_view::<9, 6>(0, 9).into_owned();
        let mut qb = SMatrix::<f64, 6, 6>::zeros();
        for i in 0..3 {
            qb[(i, i)] = self.bias_noise.gyro_density.powi(2) * dt;
            qb[(i + 3, i + 3)] = self.bias_noise.accel_density.powi(2) * dt;
        }
        let upper = q.fixed_view::<9, 9>(0, 0).into_owned() + b * qb * b.transpose() / 3.;
        q.fixed_view_mut::<9, 9>(0, 0).copy_from(&upper);
        q.fixed_view_mut::<9, 6>(0, 9).copy_from(&(b * qb * 0.5));
        q.fixed_view_mut::<6, 9>(9, 0)
            .copy_from(&(qb * b.transpose() * 0.5));
        q.fixed_view_mut::<6, 6>(9, 9).copy_from(&qb);
        if self.backward {
            q = transition * q * transition.transpose();
        }
        (q + q.transpose()) * 0.5
    }
    pub fn time_derivative(&self, nav: &BasaltNavState) -> nalgebra::SVector<f64, 15> {
        let mut d = nalgebra::SVector::<f64, 15>::zeros();
        d.fixed_rows_mut::<3>(0).copy_from(&nav.velocity_world_m_s);
        d.fixed_rows_mut::<3>(3).copy_from(
            &(nav.imu_to_world.rotation * (self.sample.gyro_rad_s - nav.gyro_bias_rad_s)),
        );
        d.fixed_rows_mut::<3>(6).copy_from(
            &(self.gravity
                + nav.imu_to_world.rotation * (self.sample.accel_m_s2 - nav.accel_bias_m_s2)),
        );
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DVector, UnitQuaternion};
    fn fixture(backward: bool) -> (EpochMotion, BasaltNavState) {
        let imu: Vec<_> = (0..=100)
            .map(|i| {
                ImuSample::new(
                    i * 1_000_000,
                    Vector3::new(0.2, -0.1, 0.3),
                    Vector3::new(1., -0.5, 9.8),
                )
            })
            .collect();
        let mut nav = BasaltNavState::default();
        nav.imu_to_world.rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.3, -0.2, 0.1));
        nav.imu_to_world.translation = Vector3::new(4., -2., 1.);
        nav.velocity_world_m_s = Vector3::new(2., 1., -0.2);
        nav.gyro_bias_rad_s = Vector3::new(0.01, -0.02, 0.03);
        nav.accel_bias_m_s2 = Vector3::new(0.1, -0.05, 0.02);
        let m = EpochMotion::new(
            &imu,
            if backward { 100_000_000 } else { 0 },
            if backward { 0 } else { 100_000_000 },
            &nav,
            Vector3::new(0., 0., -9.81),
            ImuNoiseModel {
                gyro_density: 0.01,
                accel_density: 0.1,
            },
            BiasRandomWalkNoise {
                gyro_density: 0.001,
                accel_density: 0.01,
            },
            0.005,
        )
        .unwrap();
        (m, nav)
    }
    fn difference(a: &BasaltNavState, b: &BasaltNavState) -> DVector<f64> {
        let mut d = DVector::zeros(15);
        d.rows_mut(0, 3)
            .copy_from(&(a.imu_to_world.translation - b.imu_to_world.translation));
        d.rows_mut(3, 3).copy_from(
            &(a.imu_to_world.rotation * b.imu_to_world.rotation.inverse()).scaled_axis(),
        );
        d.rows_mut(6, 3)
            .copy_from(&(a.velocity_world_m_s - b.velocity_world_m_s));
        d.rows_mut(9, 3)
            .copy_from(&(a.gyro_bias_rad_s - b.gyro_bias_rad_s));
        d.rows_mut(12, 3)
            .copy_from(&(a.accel_bias_m_s2 - b.accel_bias_m_s2));
        d
    }
    #[test]
    fn forward_and_backward_keyframe_jacobians_match_manifold_differences() {
        for backward in [false, true] {
            let (m, mut nav) = fixture(backward);
            // Exercise the finite bias correction, not just its origin.
            nav.gyro_bias_rad_s += Vector3::new(0.03, 0.02, -0.01);
            nav.accel_bias_m_s2 += Vector3::new(-0.1, 0.02, 0.04);
            let (_, j) = m.evaluate(&nav);
            for col in 0..15 {
                let eval = |h: f64| {
                    let mut n = nav.clone();
                    match col {
                        0..=2 => n.imu_to_world.translation[col] += h,
                        3..=5 => {
                            let mut v = Vector3::zeros();
                            v[col - 3] = h;
                            n.imu_to_world.rotation =
                                UnitQuaternion::from_scaled_axis(v) * n.imu_to_world.rotation;
                        }
                        6..=8 => n.velocity_world_m_s[col - 6] += h,
                        9..=11 => n.gyro_bias_rad_s[col - 9] += h,
                        _ => n.accel_bias_m_s2[col - 12] += h,
                    }
                    m.evaluate(&n).0
                };
                let numeric = difference(&eval(1e-6), &eval(-1e-6)) * 5e5;
                assert!(
                    (numeric - j.column(col)).norm() < 1e-7,
                    "backward={backward} col={col}"
                );
            }
            let q = m.covariance(&nav);
            assert!(q.symmetric_eigen().eigenvalues.min() > -1e-15);
            assert!(q.trace() > 0.);
        }
    }
    #[test]
    fn backward_navigation_inverts_forward_motion() {
        let (m, n) = fixture(false);
        let end = m.evaluate(&n).0;
        let mut inverse = m.clone();
        inverse.backward = true;
        assert!(difference(&inverse.evaluate(&end).0, &n).norm() < 1e-12);
        let f = m.evaluate(&n).1;
        let b = inverse.evaluate(&end).1;
        assert!((b * f - Matrix15::identity()).norm() < 1e-12);
        assert!((inverse.covariance(&end) - b * m.covariance(&n) * b.transpose()).norm() < 1e-14);
    }
}
