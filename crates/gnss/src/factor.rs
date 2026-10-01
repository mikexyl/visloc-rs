use crate::navigation::{enu_to_ecef, SatelliteState};
use crate::{Config, Mode, Observation, C, L1_HZ};
use nalgebra::{Matrix2, Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alignment {
    pub origin_ecef_m: Vector3<f64>,
    pub yaw_rad: f64,
    pub translation_enu_m: Vector3<f64>,
}
impl Alignment {
    pub fn rotation(&self) -> Matrix3<f64> {
        enu_to_ecef(self.origin_ecef_m)
            * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), self.yaw_rad)
                .to_rotation_matrix()
                .into_inner()
    }
    pub fn position(&self, p: Vector3<f64>) -> Vector3<f64> {
        self.origin_ecef_m
            + enu_to_ecef(self.origin_ecef_m) * self.translation_enu_m
            + self.rotation() * p
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Clock {
    pub bias_m: [f64; 2],
    pub drift_m_s: f64,
}
#[derive(Debug, Clone)]
pub struct Body {
    pub position: Vector3<f64>,
    pub rotation: UnitQuaternion<f64>,
    pub velocity: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedObservation {
    pub observation: Observation,
    pub satellite: SatelliteState,
    pub atmosphere_m: f64,
    pub gyro_rad_s: Vector3<f64>,
    pub use_pseudorange: bool,
    pub use_doppler: bool,
}
pub struct Linearization {
    pub residual: SVector<f64, 2>,
    pub jacobian: SMatrix<f64, 2, 22>,
    pub cost: f64,
}
fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0., -v.z, v.y, v.z, 0., -v.x, -v.y, v.x, 0.)
}
/// Column order: body pose6, velocity3, gyro bias3, accel bias3,
/// ENU translation3/yaw1, GPS bias/Galileo bias/common drift3.
fn raw(
    o: &PreparedObservation,
    body: &Body,
    alignment: &Alignment,
    clock: &Clock,
    lever: Vector3<f64>,
) -> Linearization {
    let r = alignment.rotation();
    let rb = body.rotation.to_rotation_matrix().into_inner();
    let angular = o.gyro_rad_s - body.gyro_bias;
    let rotated_lever = rb * lever;
    let lever_velocity = rb * angular.cross(&lever);
    let local_p = body.position + rotated_lever;
    let local_v = body.velocity + lever_velocity;
    let p = alignment.position(local_p);
    let v = r * local_v;
    let delta = o.satellite.position_m - p;
    let rho = delta.norm();
    let n = delta / rho;
    let relative_v = o.satellite.velocity_m_s - v;
    let pr = rho + clock.bias_m[o.observation.satellite.system.index()] - o.satellite.clock_m
        + o.atmosphere_m
        - o.observation.pseudorange_m;
    let rate = n.dot(&relative_v) + clock.drift_m_s - o.satellite.clock_drift_m_s
        + o.observation.doppler_hz * C / L1_HZ;
    let dp_rate = -(Matrix3::identity() - n * n.transpose()) * relative_v / rho;
    let mut j = SMatrix::<f64, 2, 22>::zeros();
    let dp = -n.transpose();
    let dv = -n.transpose();
    j.fixed_view_mut::<1, 3>(0, 0).copy_from(&(dp * r));
    j.fixed_view_mut::<1, 3>(0, 3)
        .copy_from(&(dp * r * (-skew(rotated_lever))));
    j.fixed_view_mut::<1, 3>(1, 0)
        .copy_from(&(dp_rate.transpose() * r));
    j.fixed_view_mut::<1, 3>(1, 3).copy_from(
        &(dp_rate.transpose() * r * (-skew(rotated_lever)) + dv * r * (-skew(lever_velocity))),
    );
    j.fixed_view_mut::<1, 3>(1, 6).copy_from(&(dv * r));
    j.fixed_view_mut::<1, 3>(1, 9)
        .copy_from(&(dv * r * rb * skew(lever)));
    let re = enu_to_ecef(alignment.origin_ecef_m);
    j.fixed_view_mut::<1, 3>(0, 15).copy_from(&(dp * re));
    j.fixed_view_mut::<1, 3>(1, 15)
        .copy_from(&(dp_rate.transpose() * re));
    j[(0, 18)] = (dp * r * Vector3::z().cross(&local_p))[0];
    j[(1, 18)] = (dp_rate.transpose() * r * Vector3::z().cross(&local_p)
        + dv * r * Vector3::z().cross(&local_v))[0];
    j[(0, 19 + o.observation.satellite.system.index())] = 1.;
    j[(1, 21)] = 1.;
    Linearization {
        residual: SVector::<f64, 2>::new(pr, rate),
        jacobian: j,
        cost: 0.,
    }
}

/// Unwhitened residual/Jacobian, used to propagate navigation/time uncertainty.
pub fn raw_linearize(
    o: &PreparedObservation,
    body: &Body,
    alignment: &Alignment,
    clock: &Clock,
    lever: Vector3<f64>,
) -> Linearization {
    raw(o, body, alignment, clock, lever)
}

pub fn linearize(
    o: &PreparedObservation,
    body: &Body,
    alignment: &Alignment,
    clock: &Clock,
    lever: Vector3<f64>,
    config: &Config,
) -> Linearization {
    linearize_with_covariance(o, body, alignment, clock, lever, config, Matrix2::zeros())
}

/// Additional covariance is frozen at the factor's current linearization point.
/// Whiten correlated code/rate noise jointly only when both channels are active.
pub fn linearize_with_covariance(
    o: &PreparedObservation,
    body: &Body,
    alignment: &Alignment,
    clock: &Clock,
    lever: Vector3<f64>,
    config: &Config,
    extra: Matrix2<f64>,
) -> Linearization {
    let lin = raw(o, body, alignment, clock, lever);
    let mut j = lin.jacobian;
    let angular = o.gyro_rad_s - body.gyro_bias;
    let r = alignment.rotation();
    let rb = body.rotation.to_rotation_matrix().into_inner();
    let dv = -(o.satellite.position_m - alignment.position(body.position + rb * lever))
        .normalize()
        .transpose();
    let sigma_pr = o
        .observation
        .pseudorange_std_m
        .max(config.pseudorange_floor_m)
        .hypot(config.lever_arm_std_m);
    let sigma_rate = (o.observation.doppler_std_hz * C / L1_HZ)
        .max(config.range_rate_floor_m_s)
        .hypot(config.lever_arm_std_m * (dv * r * rb * skew(angular)).norm());
    let mut covariance = extra;
    covariance[(0, 0)] += sigma_pr.powi(2);
    covariance[(1, 1)] += sigma_rate.powi(2);
    let active = [
        o.use_pseudorange && config.mode == Mode::PseudorangeDoppler,
        o.use_doppler && config.mode != Mode::WindowOnly,
    ];
    let mut residual = lin.residual;
    if active.iter().all(|a| *a) {
        let whitening = covariance
            .cholesky()
            .expect("positive measurement covariance")
            .l()
            .try_inverse()
            .expect("positive diagonal");
        residual = whitening * residual;
        j = whitening * j;
    } else {
        for row in 0..2 {
            if active[row] {
                residual[row] /= covariance[(row, row)].sqrt();
                j.row_mut(row).scale_mut(1. / covariance[(row, row)].sqrt());
            } else {
                residual[row] = 0.;
                j.row_mut(row).fill(0.);
            }
        }
    }
    let mut cost = 0.;
    for row in 0..2 {
        if !active[row] {
            residual[row] = 0.;
            j.row_mut(row).fill(0.);
            continue;
        }
        let x = residual[row].abs();
        let w = if x <= config.huber_sigma {
            1.
        } else {
            config.huber_sigma / x
        };
        cost += if x <= config.huber_sigma {
            0.5 * x * x
        } else {
            config.huber_sigma * (x - 0.5 * config.huber_sigma)
        };
        j.row_mut(row).scale_mut(w.sqrt());
        residual[row] *= w.sqrt();
    }
    Linearization {
        residual,
        jacobian: j,
        cost,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Satellite, System};
    fn fixture() -> (
        PreparedObservation,
        Body,
        Alignment,
        Clock,
        Config,
        Vector3<f64>,
    ) {
        let align = Alignment {
            origin_ecef_m: Vector3::new(4e6, 3e6, 4e6),
            yaw_rad: 0.7,
            translation_enu_m: Vector3::new(2., 3., 4.),
        };
        let body = Body {
            position: Vector3::new(10., -3., 2.),
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.1, 0.3)),
            velocity: Vector3::new(2., 1., -0.1),
            gyro_bias: Vector3::new(0.01, -0.02, 0.01),
        };
        let o = PreparedObservation {
            observation: Observation {
                satellite: Satellite {
                    system: System::Galileo,
                    prn: 7,
                },
                signal: 0,
                pseudorange_m: 2e7,
                doppler_hz: -500.,
                pseudorange_std_m: 3.,
                doppler_std_hz: 2.,
                cn0_dbhz: 45,
                lock_ms: 1000,
            },
            satellite: SatelliteState {
                position_m: Vector3::new(20e6, 10e6, 15e6),
                velocity_m_s: Vector3::new(-1000., 1500., 2000.),
                clock_m: 3.,
                clock_drift_m_s: 0.01,
            },
            atmosphere_m: 6.,
            gyro_rad_s: Vector3::new(0.1, 0.2, 0.3),
            use_pseudorange: true,
            use_doppler: true,
        };
        let cfg = Config {
            huber_sigma: 1e12,
            lever_arm_std_m: 1e-20,
            ..Config::default()
        };
        (
            o,
            body,
            align,
            Clock {
                bias_m: [3., 4.],
                drift_m_s: 0.2,
            },
            cfg,
            Vector3::new(0.11, -0.095, -0.079),
        )
    }
    #[test]
    fn metric_antenna_velocity_and_negative_doppler_close_the_measurement() {
        let (mut o, b, a, c, cfg, l) = fixture();
        let p = a.position(b.position + b.rotation * l);
        let v = a.rotation() * (b.velocity + b.rotation * (o.gyro_rad_s - b.gyro_bias).cross(&l));
        let delta = o.satellite.position_m - p;
        o.observation.pseudorange_m =
            delta.norm() + c.bias_m[1] - o.satellite.clock_m + o.atmosphere_m;
        let rate = delta.normalize().dot(&(o.satellite.velocity_m_s - v)) + c.drift_m_s
            - o.satellite.clock_drift_m_s;
        o.observation.doppler_hz = -rate * L1_HZ / C;
        assert!(linearize(&o, &b, &a, &c, l, &cfg).residual.norm() < 1e-7);
        let mut no_lever = b.clone();
        no_lever.gyro_bias = o.gyro_rad_s;
        assert!(linearize(&o, &no_lever, &a, &c, l, &cfg).residual[1].abs() > 1e-3);
    }
    #[test]
    fn full_manifold_jacobian_and_doppler_sign() {
        let (o, b, a, c, cfg, l) = fixture();
        let lin = linearize(&o, &b, &a, &c, l, &cfg);
        for col in 0..22 {
            let h = if col < 3 || (15..=17).contains(&col) || col >= 19 {
                0.01
            } else {
                1e-4
            };
            let eval = |d: f64| {
                let (mut bb, mut aa, mut cc) = (b.clone(), a.clone(), c.clone());
                match col {
                    0..=2 => bb.position[col] += d,
                    3..=5 => {
                        let mut axis = Vector3::zeros();
                        axis[col - 3] = d;
                        bb.rotation = UnitQuaternion::from_scaled_axis(axis) * bb.rotation;
                    }
                    6..=8 => bb.velocity[col - 6] += d,
                    9..=11 => bb.gyro_bias[col - 9] += d,
                    12..=14 => (),
                    15..=17 => aa.translation_enu_m[col - 15] += d,
                    18 => aa.yaw_rad += d,
                    19..=20 => cc.bias_m[col - 19] += d,
                    21 => cc.drift_m_s += d,
                    _ => unreachable!(),
                };
                linearize(&o, &bb, &aa, &cc, l, &cfg).residual
            };
            let numeric = (eval(h) - eval(-h)) / (2. * h);
            assert!(
                (numeric - lin.jacobian.column(col)).norm() < 2e-5,
                "column {col}: {numeric:?} / {:?}",
                lin.jacobian.column(col)
            );
        }
        let mut opposite = o.clone();
        opposite.observation.doppler_hz = -o.observation.doppler_hz;
        assert!(linearize(&opposite, &b, &a, &c, l, &cfg).residual[1] != lin.residual[1]);
    }
    #[test]
    fn correlated_whitening_preserves_objective_and_disabled_channels_do_not_leak() {
        let (mut o, b, a, c, cfg, l) = fixture();
        let extra = Matrix2::new(100., 2., 2., 0.2);
        let raw = raw_linearize(&o, &b, &a, &c, l);
        let mut covariance = extra;
        covariance[(0, 0)] += o
            .observation
            .pseudorange_std_m
            .max(cfg.pseudorange_floor_m)
            .powi(2);
        covariance[(1, 1)] += (o.observation.doppler_std_hz * C / L1_HZ)
            .max(cfg.range_rate_floor_m_s)
            .powi(2);
        let f = linearize_with_covariance(&o, &b, &a, &c, l, &cfg, extra);
        assert!(
            (f.residual.norm_squared()
                - raw
                    .residual
                    .dot(&(covariance.try_inverse().unwrap() * raw.residual)))
            .abs()
                < 1e-4
        );
        o.use_pseudorange = false;
        let rate = linearize_with_covariance(&o, &b, &a, &c, l, &cfg, extra);
        assert_eq!(rate.residual[0], 0.);
        assert_eq!(rate.jacobian.row(0).norm(), 0.);
        let mut changed = o.clone();
        changed.observation.pseudorange_m += 1e6;
        assert_eq!(
            rate.residual,
            linearize_with_covariance(&changed, &b, &a, &c, l, &cfg, extra).residual
        );
        assert!(rate.residual[1].abs() < linearize(&o, &b, &a, &c, l, &cfg).residual[1].abs());
    }
}
