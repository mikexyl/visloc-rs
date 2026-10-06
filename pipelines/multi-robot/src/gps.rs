//! Backend-only GNSS fix conditioning, association and ENU alignment.
use crate::*;
use nalgebra::{Matrix2, Matrix3, UnitQuaternion, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use visloc_core::geometry::SE3;
use visloc_gtsam::{GpsPositionFactor, GpsRobustKernel};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GpsRecord {
    pub key: Key,
    pub timestamp_ns: i64,
    pub receipt_timestamp_ns: i64,
    pub time_source: String,
    /// WGS84 latitude/longitude (degrees), ellipsoidal height (metres).
    /// None for unavailable position, never fabricate a zero fix.
    pub lla: Option<[f64; 3]>,
    pub status: i8,
    pub quality: Option<u8>,
    pub hdop: Option<f64>,
    /// Receiver ENU covariance at this fix, row-major. None means unknown.
    pub covariance_enu: Option<[f64; 9]>,
}
impl GpsRecord {
    pub fn validate(&self) -> Result<()> {
        self.key.validate()?;
        if self.timestamp_ns <= 0
            || self.receipt_timestamp_ns <= 0
            || self.time_source.is_empty()
            || self.lla.is_some_and(|p| {
                !p.iter().all(|v| v.is_finite()) || p[0].abs() > 90. || p[1].abs() > 180.
            })
            || self.hdop.is_some_and(|v| !v.is_finite() || v < 0.)
        {
            return Err(Error("malformed GPS record".into()));
        }
        if let Some(c) = self.covariance_enu {
            let c = Matrix3::from_row_slice(&c)
                .fixed_view::<2, 2>(0, 0)
                .into_owned();
            if !c.iter().all(|v| v.is_finite())
                || (c - c.transpose()).norm() > 1e-6
                || c.cholesky().is_none()
            {
                return Err(Error("invalid known GPS covariance".into()));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct GpsDatum {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
}
impl GpsDatum {
    pub fn lla(self) -> [f64; 3] {
        [self.latitude, self.longitude, self.altitude]
    }
    pub fn from_lla(p: [f64; 3]) -> Self {
        Self {
            latitude: p[0],
            longitude: p[1],
            altitude: p[2],
        }
    }
    pub fn rotation(self) -> Matrix3<f64> {
        let (s, c) = self.latitude.to_radians().sin_cos();
        let (sl, cl) = self.longitude.to_radians().sin_cos();
        Matrix3::new(-sl, cl, 0., -s * cl, -s * sl, c, c * cl, c * sl, s)
    }
    pub fn project(self, lla: [f64; 3]) -> Vector3<f64> {
        self.rotation() * (ecef(lla) - ecef(self.lla()))
    }
}
pub fn ecef(lla: [f64; 3]) -> Vector3<f64> {
    let (s, c) = lla[0].to_radians().sin_cos();
    let (sl, cl) = lla[1].to_radians().sin_cos();
    let e2 = 6.6943799901413165e-3;
    let n = 6378137. / (1. - e2 * s * s).sqrt();
    Vector3::new(
        (n + lla[2]) * c * cl,
        (n + lla[2]) * c * sl,
        (n * (1. - e2) + lla[2]) * s,
    )
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GpsRobustMode {
    Huber,
    #[default]
    Switchable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpsConfig {
    pub enabled: bool,
    pub origin: Option<GpsDatum>,
    pub lever_arms_m: BTreeMap<String, [f64; 3]>,
    pub lever_sigma_m: f64,
    pub unknown_horizontal_sigma_m: f64,
    pub min_horizontal_sigma_m: f64,
    pub min_interval_s: f64,
    pub min_distance_m: f64,
    pub max_bracket_s: f64,
    pub min_alignment_fixes: usize,
    pub alignment_span_m: f64,
    pub alignment_consensus_fraction: f64,
    pub robust_mode: GpsRobustMode,
    pub switch_lambda: f64,
    pub huber_delta: f64,
    pub max_hdop: f64,
    pub require_hdop: bool,
    pub max_reported_horizontal_sigma_m: f64,
    /// Whitened residual radius inside which GPS applies no corrective force.
    pub residual_deadband_sigma: f64,
}
impl Default for GpsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            origin: None,
            lever_arms_m: BTreeMap::new(),
            lever_sigma_m: 0.02,
            unknown_horizontal_sigma_m: 5.,
            min_horizontal_sigma_m: 0.5,
            min_interval_s: 1.,
            min_distance_m: 2.,
            max_bracket_s: 2.,
            min_alignment_fixes: 10,
            alignment_span_m: 20.,
            alignment_consensus_fraction: 0.7,
            robust_mode: GpsRobustMode::Switchable,
            switch_lambda: 9.,
            huber_delta: 3.,
            max_hdop: 1.0,
            require_hdop: true,
            max_reported_horizontal_sigma_m: 5.,
            residual_deadband_sigma: 1.,
        }
    }
}
impl GpsConfig {
    pub fn validate(&self) -> Result<()> {
        if [
            self.lever_sigma_m,
            self.unknown_horizontal_sigma_m,
            self.min_horizontal_sigma_m,
            self.min_interval_s,
            self.min_distance_m,
            self.max_bracket_s,
            self.alignment_span_m,
            self.switch_lambda,
            self.huber_delta,
            self.max_hdop,
            self.max_reported_horizontal_sigma_m,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.)
            || !self.residual_deadband_sigma.is_finite()
            || self.residual_deadband_sigma < 0.
            || self.min_alignment_fixes < 2
            || !(0.5..=1.).contains(&self.alignment_consensus_fraction)
            || self.lever_arms_m.values().flatten().any(|v| !v.is_finite())
            || self.origin.is_some_and(|d| {
                !d.lla().iter().all(|v| v.is_finite())
                    || d.latitude.abs() > 90.
                    || d.longitude.abs() > 180.
            })
        {
            return Err(Error("invalid GPS configuration".into()));
        }
        Ok(())
    }
    pub fn kernel(&self) -> GpsRobustKernel {
        match self.robust_mode {
            GpsRobustMode::Huber => GpsRobustKernel::Huber {
                delta: self.huber_delta,
            },
            GpsRobustMode::Switchable => GpsRobustKernel::Switchable {
                lambda: self.switch_lambda,
            },
        }
    }
    pub fn quality_reason(&self, record: &GpsRecord) -> Option<&'static str> {
        if record.status < 0 || record.quality == Some(0) || record.lla.is_none() {
            return Some("no_fix");
        }
        // Standalone (1) and differential (2) are ordinary valid GPS. RTK is
        // neither required nor given artificially precise weights.
        if record.quality.is_some_and(|q| ![1, 2, 4, 5].contains(&q)) {
            return Some("quality_not_satellite_fix");
        }
        if record.hdop.is_none() && self.require_hdop {
            return Some("quality_hdop_unknown");
        }
        if record.hdop.is_some_and(|x| x > self.max_hdop) {
            return Some("quality_hdop_exceeded");
        }
        if let Some(c) = record.covariance_enu {
            let c = Matrix3::from_row_slice(&c)
                .fixed_view::<2, 2>(0, 0)
                .into_owned();
            if c.symmetric_eigen().eigenvalues.max() > self.max_reported_horizontal_sigma_m.powi(2)
            {
                return Some("quality_covariance_exceeded");
            }
        }
        None
    }
    /// Horizontal covariance transported from the receiver EN tangent plane to
    /// the mission EN plane. The Up row/column is zero and never used.
    pub fn covariance(&self, record: &GpsRecord, datum: GpsDatum) -> Matrix3<f64> {
        let mut horizontal = match record.covariance_enu {
            Some(c) => {
                let rotation = datum.rotation()
                    * GpsDatum::from_lla(record.lla.unwrap())
                        .rotation()
                        .transpose();
                let rotation = rotation.fixed_view::<2, 2>(0, 0).into_owned();
                let c = Matrix3::from_row_slice(&c)
                    .fixed_view::<2, 2>(0, 0)
                    .into_owned();
                rotation * c * rotation.transpose()
            }
            None => {
                Matrix2::identity()
                    * (self.unknown_horizontal_sigma_m * record.hdop.unwrap_or(1.).max(1.)).powi(2)
            }
        };
        let floor = (self.min_horizontal_sigma_m.powi(2)
            - horizontal.symmetric_eigen().eigenvalues.min())
        .max(0.);
        horizontal += Matrix2::identity() * (floor + self.lever_sigma_m.powi(2));
        let mut result = Matrix3::zeros();
        result.fixed_view_mut::<2, 2>(0, 0).copy_from(&horizontal);
        result
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GpsSnapshot {
    pub datum: Option<GpsDatum>,
    pub aligned_components: Vec<Key>,
    pub diagnostics: Vec<GpsDiagnostic>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpsDiagnostic {
    pub key: Key,
    pub timestamp_ns: i64,
    pub reason: String,
    pub from: Option<Key>,
    pub to: Option<Key>,
    pub alpha: Option<f64>,
    pub position_enu: Option<[f64; 3]>,
    pub sigma_enu: Option<[f64; 3]>,
    pub predicted_enu: Option<[f64; 3]>,
    pub residual_enu: Option<[f64; 3]>,
    pub squared_mahalanobis: Option<f64>,
    pub robust_weight: Option<f64>,
    #[serde(default)]
    pub optimization_weight: Option<f64>,
    #[serde(default)]
    pub within_tolerance: bool,
}
impl GpsDiagnostic {
    pub(crate) fn new(r: &GpsRecord) -> Self {
        Self {
            key: r.key.clone(),
            timestamp_ns: r.timestamp_ns,
            reason: "pending_keyframes".into(),
            from: None,
            to: None,
            alpha: None,
            position_enu: None,
            sigma_enu: None,
            predicted_enu: None,
            residual_enu: None,
            squared_mahalanobis: None,
            robust_weight: None,
            optimization_weight: None,
            within_tolerance: false,
        }
    }
}
#[derive(Clone)]
pub(crate) struct AssociatedGps {
    pub diagnostic: usize,
    pub from: Key,
    pub to: Key,
    pub alpha: f64,
    pub lever: Vector3<f64>,
    pub raw_position: Vector3<f64>,
    pub measured: Vector3<f64>,
    pub information: Matrix3<f64>,
    pub sigma_horizontal: f64,
}
impl AssociatedGps {
    pub fn factor(&self, ids: &BTreeMap<Key, u64>, config: &GpsConfig) -> GpsPositionFactor {
        GpsPositionFactor {
            from: ids[&self.from],
            to: ids[&self.to],
            alpha: self.alpha,
            position: self.measured,
            lever_arm: self.lever,
            information: self.information,
            kernel: config.kernel(),
            deadband_sigma: config.residual_deadband_sigma,
        }
    }
}

pub(crate) fn associate(
    config: &GpsConfig,
    datum: GpsDatum,
    fixes: &BTreeMap<Key, GpsRecord>,
    records: &BTreeMap<Key, KeyframeRecord>,
) -> (Vec<AssociatedGps>, Vec<GpsDiagnostic>) {
    let mut sessions: BTreeMap<(&str, &str), Vec<&KeyframeRecord>> = BTreeMap::new();
    for r in records.values() {
        sessions
            .entry((&r.key.robot, &r.key.session))
            .or_default()
            .push(r);
    }
    for frames in sessions.values_mut() {
        frames.sort_by_key(|r| r.timestamp_ns);
    }
    let mut ordered: Vec<_> = fixes.values().collect();
    ordered.sort_by_key(|r| (&r.key.robot, &r.key.session, r.timestamp_ns, r.key.id));
    let mut previous: BTreeMap<(&str, &str), (i64, Vector3<f64>)> = BTreeMap::new();
    let mut retained = Vec::new();
    let mut diagnostics = Vec::new();
    for r in ordered {
        let mut d = GpsDiagnostic::new(r);
        let result = (|| {
            if r.status < 0 || r.quality == Some(0) || r.lla.is_none() {
                return "no_fix";
            }
            let mut lla = r.lla.unwrap();
            lla[2] = datum.altitude;
            let measured = datum.project(lla);
            d.position_enu = Some(measured.into());
            let cov = config.covariance(r, datum);
            d.sigma_enu = Some([cov[(0, 0)].sqrt(), cov[(1, 1)].sqrt(), cov[(2, 2)].sqrt()]);
            if let Some(reason) = config.quality_reason(r) {
                return reason;
            }
            let Some(lever) = config.lever_arms_m.get(&r.key.robot) else {
                return "missing_lever_arm";
            };
            let lever = Vector3::from(*lever);
            let Some(frames) = sessions.get(&(r.key.robot.as_str(), r.key.session.as_str())) else {
                return "pending_keyframes";
            };
            let index = frames.partition_point(|f| f.timestamp_ns < r.timestamp_ns);
            let (a, b) = if index < frames.len() && frames[index].timestamp_ns == r.timestamp_ns {
                (frames[index], frames[index])
            } else if index == 0 {
                return "before_first_keyframe";
            } else if index == frames.len() {
                return "pending_keyframes";
            } else {
                (frames[index - 1], frames[index])
            };
            if a.key != b.key && b.previous.as_ref() != Some(&a.key) {
                return "keyframe_gap";
            }
            if (b.timestamp_ns - a.timestamp_ns) as f64 * 1e-9 > config.max_bracket_s {
                return "bracket_too_wide";
            }
            let alpha = if a.key == b.key {
                0.
            } else {
                (r.timestamp_ns - a.timestamp_ns) as f64 / (b.timestamp_ns - a.timestamp_ns) as f64
            };
            d.from = Some(a.key.clone());
            d.to = Some(b.key.clone());
            d.alpha = Some(alpha);
            let ta = a.body_to_odom.se3().unwrap();
            let tb = b.body_to_odom.se3().unwrap();
            let body_position = (1. - alpha) * ta.translation + alpha * tb.translation;
            let position = body_position + ta.rotation.slerp(&tb.rotation, alpha) * lever;
            let session = (r.key.robot.as_str(), r.key.session.as_str());
            if let Some((t, p)) = previous.get(&session) {
                if (r.timestamp_ns - *t) as f64 * 1e-9 < config.min_interval_s {
                    return "interval_skip";
                }
                if (body_position - p).norm() < config.min_distance_m {
                    return "distance_skip";
                }
            }
            previous.insert(session, (r.timestamp_ns, body_position));
            let information = {
                let xy: Matrix2<f64> = cov.fixed_view::<2, 2>(0, 0).into_owned();
                let mut m = Matrix3::zeros();
                m.fixed_view_mut::<2, 2>(0, 0)
                    .copy_from(&xy.try_inverse().unwrap());
                m
            };
            retained.push(AssociatedGps {
                diagnostic: diagnostics.len(),
                from: a.key.clone(),
                to: b.key.clone(),
                alpha,
                lever,
                raw_position: position,
                measured,
                information,
                sigma_horizontal: cov[(0, 0)].max(cov[(1, 1)]).sqrt(),
            });
            "alignment_pending"
        })();
        d.reason = result.into();
        diagnostics.push(d);
    }
    (retained, diagnostics)
}

/// Deterministic two-point RANSAC, then covariance-weighted rigid SE(2) fit.
/// Input positions already share the component's local frame; never fit scale.
pub(crate) fn initialize(
    config: &GpsConfig,
    samples: &[(Vector3<f64>, Vector3<f64>, f64)],
) -> Option<SE3> {
    if samples.len() < config.min_alignment_fixes {
        return None;
    }
    let mut best: Option<(Vec<usize>, f64, f64, Vector2<f64>)> = None;
    // Bound startup work using evenly spaced hypotheses; score against ALL fixes.
    let step = (samples.len() / 48).max(1);
    for i in (0..samples.len()).step_by(step) {
        for j in (i + 1..samples.len()).step_by(step) {
            let u = (samples[j].0 - samples[i].0).xy();
            let v = (samples[j].1 - samples[i].1).xy();
            if u.norm() < config.alignment_span_m || v.norm() < 1e-6 {
                continue;
            }
            let yaw = v.y.atan2(v.x) - u.y.atan2(u.x);
            let rot = nalgebra::Rotation2::new(yaw);
            let tr = ((samples[i].1.xy() - rot * samples[i].0.xy())
                + (samples[j].1.xy() - rot * samples[j].0.xy()))
                * 0.5;
            let mut inliers = Vec::new();
            let mut error = 0.;
            for (k, (p, z, sigma)) in samples.iter().enumerate() {
                let e = (rot * p.xy() + tr - z.xy()).norm() / sigma;
                if e <= 3. {
                    inliers.push(k);
                    error += e * e;
                }
            }
            if best.as_ref().is_none_or(|(old, e, _, _)| {
                inliers.len() > old.len() || (inliers.len() == old.len() && error < *e)
            }) {
                best = Some((inliers, error, yaw, tr));
            }
        }
    }
    let (indices, _, _, _) = best?;
    if indices.len() < config.min_alignment_fixes
        || (indices.len() as f64) < config.alignment_consensus_fraction * samples.len() as f64
    {
        return None;
    }
    let weight: f64 = indices.iter().map(|&i| 1. / samples[i].2.powi(2)).sum();
    let p = indices.iter().fold(Vector3::zeros(), |p, &i| {
        p + samples[i].0 / samples[i].2.powi(2)
    }) / weight;
    let z = indices.iter().fold(Vector3::zeros(), |p, &i| {
        p + samples[i].1 / samples[i].2.powi(2)
    }) / weight;
    let (c, s) = indices.iter().fold((0., 0.), |(c, s), &i| {
        let u = (samples[i].0 - p).xy();
        let v = (samples[i].1 - z).xy();
        let w = 1. / samples[i].2.powi(2);
        (c + w * u.dot(&v), s + w * (u.x * v.y - u.y * v.x))
    });
    if c.hypot(s) < 1e-9 {
        return None;
    }
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), s.atan2(c));
    let mut translation = z - rotation * p;
    translation.z = 0.;
    // The refined fit must itself meet the consensus and motion requirements.
    let inliers: Vec<_> = samples
        .iter()
        .filter(|(p, z, sigma)| ((rotation * p + translation - z).xy().norm() / sigma) <= 3.)
        .collect();
    if inliers.len() < config.min_alignment_fixes
        || (inliers.len() as f64) < config.alignment_consensus_fraction * samples.len() as f64
    {
        return None;
    }
    let span = inliers.iter().any(|a| {
        inliers
            .iter()
            .any(|b| (a.0 - b.0).xy().norm() >= config.alignment_span_m)
    });
    if !span {
        return None;
    }
    Some(SE3::new(rotation, translation))
}
