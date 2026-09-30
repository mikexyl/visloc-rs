//! Initialization uses only raw code/Doppler and VIO. No reference trajectory.
use crate::factor::{Alignment, Clock};
use crate::navigation::{atmospheric_delay, azimuth_elevation, enu_to_ecef, Navigation};
use crate::{Config, Epoch, C, L1_HZ};
use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub utc_ns: i64,
    pub ecef_m: Vector3<f64>,
    pub velocity_ecef_m_s: Vector3<f64>,
    pub clock: Clock,
    pub code_rms_m: f64,
    pub rate_rms_m_s: f64,
    pub satellites: usize,
}
pub fn solve_epoch(
    epoch: &Epoch,
    nav: &Navigation,
    config: &Config,
    seed: Option<Vector3<f64>>,
) -> Option<Solution> {
    let mut usable = Vec::new();
    for o in &epoch.observations {
        if o.cn0_dbhz < config.min_cn0_dbhz {
            continue;
        }
        let Some(e) = nav.select(o.satellite, epoch.gps_seconds()) else {
            continue;
        };
        let Some(sat) = e.at_transmission(epoch.gps_seconds(), o.pseudorange_m) else {
            continue;
        };
        usable.push((o, sat));
    }
    let systems: [bool; 2] =
        std::array::from_fn(|i| usable.iter().any(|(o, _)| o.satellite.system.index() == i));
    let clock_col: [Option<usize>; 2] = [
        systems[0].then_some(3),
        systems[1].then_some(3 + usize::from(systems[0])),
    ];
    let n = 3 + systems.iter().filter(|x| **x).count();
    if usable.len() < n + 1 {
        return None;
    }
    let mut x = DVector::zeros(n);
    x.rows_mut(0, 3)
        .copy_from(&seed.unwrap_or(Vector3::zeros()));
    let mut code_rms = 0.;
    for iteration in 0..12 {
        let p = Vector3::new(x[0], x[1], x[2]);
        let mut j = DMatrix::zeros(usable.len(), n);
        let mut b = DVector::zeros(usable.len());
        let mut count = 0;
        for (o, sat) in &usable {
            if iteration > 2
                && azimuth_elevation(p, sat.position_m).1 < config.min_elevation_deg.to_radians()
            {
                continue;
            }
            let delta = sat.position_m - p;
            let rho = delta.norm();
            let col = clock_col[o.satellite.system.index()]?;
            let atmo = if p.norm() > 5e6 {
                atmospheric_delay(p, sat.position_m, epoch.tow_s, nav.ionosphere)
            } else {
                0.
            };
            let residual = rho + x[col] - sat.clock_m + atmo - o.pseudorange_m;
            let sigma = o.pseudorange_std_m.max(config.pseudorange_floor_m);
            let w = if iteration > 2 {
                (config.huber_sigma * sigma / residual.abs()).min(1.).sqrt() / sigma
            } else {
                1. / sigma
            };
            j.view_mut((count, 0), (1, 3))
                .copy_from(&(-delta.transpose() / rho * w));
            j[(count, col)] = w;
            b[count] = residual * w;
            count += 1;
        }
        if count < n + 1 {
            return None;
        }
        let j = j.rows(0, count).into_owned();
        let b = b.rows(0, count).into_owned();
        let svd = j.svd(true, true);
        if svd.singular_values.min() < 1e-5 {
            return None;
        }
        let step = svd.solve(&(-&b), 1e-10).ok()?;
        x += &step;
        code_rms = (b.norm_squared() / count as f64).sqrt() * config.pseudorange_floor_m;
        if step.norm() < 1e-4 {
            break;
        }
    }
    let p = Vector3::new(x[0], x[1], x[2]);
    if !(6e6..7e6).contains(&p.norm()) || !code_rms.is_finite() || code_rms > 30. {
        return None;
    }
    let mut j = DMatrix::zeros(usable.len(), 4);
    let mut b = DVector::zeros(usable.len());
    let mut count = 0;
    for (o, sat) in &usable {
        if azimuth_elevation(p, sat.position_m).1 < config.min_elevation_deg.to_radians() {
            continue;
        }
        let n = (sat.position_m - p).normalize();
        let sigma = (o.doppler_std_hz * C / L1_HZ).max(config.range_rate_floor_m_s);
        j.view_mut((count, 0), (1, 3))
            .copy_from(&(-n.transpose() / sigma));
        j[(count, 3)] = 1. / sigma;
        b[count] =
            (-o.doppler_hz * C / L1_HZ - n.dot(&sat.velocity_m_s) + sat.clock_drift_m_s) / sigma;
        count += 1;
    }
    if count < 5 {
        return None;
    }
    let j = j.rows(0, count).into_owned();
    let b = b.rows(0, count).into_owned();
    let svd = j.clone().svd(true, true);
    if svd.singular_values.min() < 1e-4 {
        return None;
    }
    let v = svd.solve(&b, 1e-10).ok()?;
    let rate_rms = (j * &v - b).norm() / ((count - 4) as f64).sqrt() * config.range_rate_floor_m_s;
    if !rate_rms.is_finite() || rate_rms > 2. {
        return None;
    }
    let clock = Clock {
        bias_m: std::array::from_fn(|i| clock_col[i].map_or(0., |col| x[col])),
        drift_m_s: v[3],
    };
    let reference = if systems[0] {
        clock.bias_m[0]
    } else {
        clock.bias_m[1]
    };
    Some(Solution {
        utc_ns: epoch.nominal_utc_ns()? - (reference / C * 1e9).round() as i64,
        ecef_m: p,
        velocity_ecef_m_s: Vector3::new(v[0], v[1], v[2]),
        clock,
        code_rms_m: code_rms,
        rate_rms_m_s: rate_rms,
        satellites: count,
    })
}
#[derive(Debug, Clone)]
pub struct VioSample {
    pub timestamp_ns: i64,
    pub position: Vector3<f64>,
    pub rotation: UnitQuaternion<f64>,
    pub velocity: Vector3<f64>,
    pub omega: Vector3<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bootstrap {
    pub alignment: Alignment,
    pub time_offset_s: f64,
    pub timing_std_s: f64,
    pub clock: Clock,
}
fn interpolate(
    vio: &[VioSample],
    t: i64,
    lever: Vector3<f64>,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let i = vio.partition_point(|s| s.timestamp_ns < t);
    if i == 0 || i == vio.len() {
        return None;
    }
    let (a, b) = (&vio[i - 1], &vio[i]);
    let f = (t - a.timestamp_ns) as f64 / (b.timestamp_ns - a.timestamp_ns) as f64;
    if b.timestamp_ns - a.timestamp_ns > 200_000_000 {
        return None;
    }
    let q = a.rotation.slerp(&b.rotation, f);
    let p = a.position * (1. - f) + b.position * f + q * lever;
    let v = a.velocity * (1. - f)
        + b.velocity * f
        + q * (a.omega * (1. - f) + b.omega * f).cross(&lever);
    Some((p, v))
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InitializationDiagnostics {
    pub reason: String,
    pub solutions: usize,
    pub pairs: usize,
    pub motion_m: f64,
    pub velocity_rms_m_s: Option<f64>,
    pub time_offset_s: Option<f64>,
    pub timing_std_s: Option<f64>,
}
pub fn initialize(
    solutions: &[Solution],
    vio: &[VioSample],
    lever: Vector3<f64>,
    config: &Config,
) -> Option<Bootstrap> {
    initialize_diagnosed(solutions, vio, lever, config).0
}
pub fn initialize_diagnosed(
    solutions: &[Solution],
    vio: &[VioSample],
    lever: Vector3<f64>,
    config: &Config,
) -> (Option<Bootstrap>, InitializationDiagnostics) {
    let mut diagnostics = InitializationDiagnostics {
        reason: "insufficient_raw_solutions".into(),
        solutions: solutions.len(),
        ..Default::default()
    };
    if solutions.len() < 20 || vio.len() < 20 {
        return (None, diagnostics);
    }
    let first = vio.first().unwrap();
    let last = vio.last().unwrap();
    diagnostics.motion_m = (last.position - first.position).norm();
    if diagnostics.motion_m < config.min_bootstrap_motion_m {
        diagnostics.reason = "insufficient_motion".into();
        return (None, diagnostics);
    }
    let origin = solutions[0].ecef_m;
    let re = enu_to_ecef(origin);
    let eval = |offset: f64| -> Option<(f64, f64, Vector3<f64>, usize)> {
        let mut pairs = Vec::new();
        for s in solutions {
            // All search offsets use the same support interval.
            if s.utc_ns < first.timestamp_ns + (config.timing_search_s * 1e9) as i64
                || s.utc_ns > last.timestamp_ns - (config.timing_search_s * 1e9) as i64
            {
                continue;
            }
            let (p, v) = interpolate(vio, s.utc_ns + (offset * 1e9).round() as i64, lever)?;
            pairs.push((
                p,
                v,
                re.transpose() * (s.ecef_m - origin),
                re.transpose() * s.velocity_ecef_m_s,
            ));
        }
        if pairs.len() < 15 {
            return None;
        }
        let (mut dot, mut cross) = (0., 0.);
        for (_, v, _, g) in &pairs {
            dot += v.x * g.x + v.y * g.y;
            cross += v.x * g.y - v.y * g.x;
        }
        if dot.hypot(cross) < 1. {
            return None;
        }
        let yaw = cross.atan2(dot);
        let r = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
        let mut error = 0.;
        let mut translation = Vector3::zeros();
        for (p, v, g, gv) in &pairs {
            error += (r * v - gv).norm_squared();
            translation += g - r * p;
        }
        Some((
            error / pairs.len() as f64,
            yaw,
            translation / pairs.len() as f64,
            pairs.len(),
        ))
    };
    let (offset, best) = if let Some(offset) = config.fixed_time_offset_s {
        (
            offset,
            match eval(offset) {
                Some(e) => e,
                None => {
                    diagnostics.reason = "insufficient_overlap".into();
                    return (None, diagnostics);
                }
            },
        )
    } else {
        let mut candidates = Vec::new();
        for i in 0..=100 {
            let d = -config.timing_search_s + 2. * config.timing_search_s * i as f64 / 100.;
            if let Some(e) = eval(d) {
                candidates.push((d, e));
            }
        }
        match candidates
            .into_iter()
            .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
        {
            Some(e) => e,
            None => {
                diagnostics.reason = "insufficient_overlap".into();
                return (None, diagnostics);
            }
        }
    };
    diagnostics.pairs = best.3;
    diagnostics.velocity_rms_m_s = Some(best.0.sqrt());
    diagnostics.time_offset_s = Some(offset);
    let h = 0.01;
    let curvature = match (eval(offset + h), eval(offset - h)) {
        (Some(a), Some(b)) => (a.0 + b.0 - 2. * best.0) / (h * h),
        _ => {
            diagnostics.reason = "insufficient_overlap".into();
            return (None, diagnostics);
        }
    };
    let std = if config.fixed_time_offset_s.is_some() {
        0.
    } else {
        if offset.abs() >= config.timing_search_s - h || curvature <= 1e-5 {
            diagnostics.reason = "timing_unobservable".into();
            return (None, diagnostics);
        }
        (2. * best.0 / (best.3 as f64 * curvature)).sqrt()
    };
    diagnostics.timing_std_s = Some(std);
    if std > config.max_timing_std_s {
        diagnostics.reason = "timing_uncertain".into();
        return (None, diagnostics);
    }
    if best.0.sqrt() > 2. {
        diagnostics.reason = "velocity_inconsistent".into();
        return (None, diagnostics);
    }
    diagnostics.reason = "active".into();
    (
        Some(Bootstrap {
            alignment: Alignment {
                origin_ecef_m: origin,
                yaw_rad: best.1,
                translation_enu_m: best.2,
            },
            time_offset_s: offset,
            timing_std_s: std,
            clock: solutions.last().unwrap().clock.clone(),
        }),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(informative: bool) -> (Vec<Solution>, Vec<VioSample>, Config) {
        let origin = Vector3::new(4e6, 3e6, 4e6);
        let re = enu_to_ecef(origin);
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.7);
        let path = |t: f64| {
            if informative {
                (
                    Vector3::new(
                        t + 2. * (0.7 * t).sin(),
                        2. * (0.4 * t).sin(),
                        0.1 * (0.3 * t).sin(),
                    ),
                    Vector3::new(
                        1. + 1.4 * (0.7 * t).cos(),
                        0.8 * (0.4 * t).cos(),
                        0.03 * (0.3 * t).cos(),
                    ),
                )
            } else {
                (Vector3::new(t, 0., 0.), Vector3::new(1., 0., 0.))
            }
        };
        let vio = (0..=1000)
            .map(|i| {
                let t = i as f64 * 0.02;
                let (p, v) = path(t);
                VioSample {
                    timestamp_ns: (t * 1e9).round() as i64,
                    position: p,
                    rotation: UnitQuaternion::identity(),
                    velocity: v,
                    omega: Vector3::zeros(),
                }
            })
            .collect();
        let solutions = (0..100)
            .map(|i| {
                let t = i as f64 * 0.2;
                let (p, v) = path(t + 0.12);
                Solution {
                    utc_ns: (t * 1e9).round() as i64,
                    ecef_m: origin + re * (yaw * p + Vector3::new(3., 4., 2.)),
                    velocity_ecef_m_s: re * (yaw * v),
                    clock: Clock::default(),
                    code_rms_m: 0.1,
                    rate_rms_m_s: 0.01,
                    satellites: 10,
                }
            })
            .collect();
        let cfg = Config {
            min_bootstrap_motion_m: 1.,
            ..Config::default()
        };
        (solutions, vio, cfg)
    }
    #[test]
    fn raw_velocity_bootstrap_recovers_yaw_and_frozen_timing() {
        let (s, v, c) = fixture(true);
        let (b, d) = initialize_diagnosed(&s, &v, Vector3::zeros(), &c);
        let b = b.expect(&d.reason);
        assert!((b.time_offset_s - 0.12).abs() < 1e-8);
        assert!((b.alignment.yaw_rad - 0.7).abs() < 0.002);
        assert!(b.timing_std_s < 0.01);
    }
    #[test]
    fn constant_velocity_does_not_observe_timing() {
        let (s, v, c) = fixture(false);
        let (b, d) = initialize_diagnosed(&s, &v, Vector3::zeros(), &c);
        assert!(b.is_none());
        assert_eq!(d.reason, "timing_unobservable");
    }
}
