//! Bounded, causal stationary gyro initialization shared by replay and ROS.
//!
//! A separate KLT stream checks motion while original sensor packets are held.
//! Accepted packets reach a fresh VIO adapter unchanged and in order. The
//! motion-gated mode omits earlier stationary packets and starts with the last
//! waiting frame and the triggering frame. No future bag access, ground truth,
//! or estimator reset is involved.
use std::collections::BTreeMap;

use nalgebra::{Point2, Vector3};
use serde::{Deserialize, Serialize};

use crate::{
    adapter::direct_klt_config,
    config::BasaltConfig,
    vio::estimator::{calibrate_accel, calibrate_gyro},
    BasaltCalibration, DirectKltStream, EurocSensorFrame, ImuSample, StereoFrame,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StationaryStartupConfig {
    pub window_ns: i64,
    pub max_frames: usize,
    pub max_imu_samples: usize,
    pub min_imu_samples: usize,
    pub max_imu_gap_ns: i64,
    pub min_common_tracks: usize,
    pub min_track_fraction: f64,
    pub max_disparity_p90_px: f64,
    pub max_gyro_std_rad_s: f64,
    pub max_accel_std_m_s2: f64,
    pub max_gyro_mean_rad_s: f64,
    pub max_gravity_norm_error_m_s2: f64,
    /// Experimental alternative to preserving the calibrated first gravity sample.
    pub average_gravity: bool,
    pub wait_for_motion: bool,
    pub motion_disparity_px: f64,
    pub motion_accel_deviation_m_s2: f64,
    pub max_wait_ns: i64,
}

impl Default for StationaryStartupConfig {
    fn default() -> Self {
        Self {
            window_ns: 1_000_000_000,
            max_frames: 64,
            max_imu_samples: 4096,
            min_imu_samples: 50,
            max_imu_gap_ns: 50_000_000,
            min_common_tracks: 30,
            min_track_fraction: 0.5,
            max_disparity_p90_px: 0.75,
            max_gyro_std_rad_s: 0.005,
            max_accel_std_m_s2: 0.15,
            max_gyro_mean_rad_s: 0.03,
            max_gravity_norm_error_m_s2: 0.5,
            average_gravity: false,
            wait_for_motion: false,
            motion_disparity_px: 3.0,
            motion_accel_deviation_m_s2: 0.2,
            max_wait_ns: 30_000_000_000,
        }
    }
}

impl StationaryStartupConfig {
    fn validate(&self) -> Result<(), String> {
        if self.max_wait_ns < self.window_ns
            || self.window_ns <= 0
            || self.max_frames < 2
            || self.min_imu_samples < 2
            || self.max_imu_samples < self.min_imu_samples
            || self.max_imu_gap_ns <= 0
            || self.min_common_tracks < 3
            || !(0.0..=1.0).contains(&self.min_track_fraction)
            || ![
                self.motion_disparity_px,
                self.motion_accel_deviation_m_s2,
                self.max_disparity_p90_px,
                self.max_gyro_std_rad_s,
                self.max_accel_std_m_s2,
                self.max_gyro_mean_rad_s,
                self.max_gravity_norm_error_m_s2,
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        {
            return Err("invalid stationary IMU startup configuration".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StationaryStartupReport {
    pub config: StationaryStartupConfig,
    pub first_camera_ns: i64,
    pub ready_camera_ns: i64,
    pub buffered_frames: usize,
    pub released_camera_ns: Option<i64>,
    pub discarded_stationary_frames: usize,
    pub imu_samples: usize,
    pub gyro_offset_rad_s: [f64; 3],
    pub mean_accel_m_s2: [f64; 3],
    pub gyro_std_rad_s: f64,
    pub accel_std_m_s2: f64,
    pub max_disparity_p90_px: f64,
    pub minimum_common_tracks: usize,
}

/// One-shot gate: movement, insufficient texture, gaps, and short streams fail
/// explicitly. Moving-start initialization is a different estimation problem.
pub struct StationaryStartup {
    config: StationaryStartupConfig,
    calibration: BasaltCalibration,
    frontend: Option<DirectKltStream>,
    frames: Vec<EurocSensorFrame>,
    samples: Vec<ImuSample>,
    reference: BTreeMap<u64, Point2<f64>>,
    first_ns: Option<i64>,
    last_ns: Option<i64>,
    last_imu_ns: Option<i64>,
    disparity: f64,
    minimum_common: usize,
    failed: bool,
    released: bool,
    motion_excited: bool,
    pub report: Option<StationaryStartupReport>,
}

impl StationaryStartup {
    pub fn new(
        calibration: &BasaltCalibration,
        vio: &BasaltConfig,
        config: StationaryStartupConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let frontend = DirectKltStream::new(
            calibration.clone(),
            direct_klt_config(vio).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            config,
            calibration: calibration.clone(),
            frontend: Some(frontend),
            frames: Vec::new(),
            samples: Vec::new(),
            reference: BTreeMap::new(),
            first_ns: None,
            last_ns: None,
            last_imu_ns: None,
            disparity: 0.0,
            minimum_common: usize::MAX,
            failed: false,
            released: false,
            motion_excited: false,
            report: None,
        })
    }

    /// Empty output means initialization is still collecting data. A successful
    /// first release includes the remaining buffered inputs (two when waiting
    /// for motion); subsequent releases have one.
    pub fn push(&mut self, frame: EurocSensorFrame) -> Result<Vec<EurocSensorFrame>, String> {
        if self.failed {
            return Err("stationary startup previously failed; start a new session".into());
        }
        let result = self.push_inner(frame);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn push_inner(&mut self, frame: EurocSensorFrame) -> Result<Vec<EurocSensorFrame>, String> {
        if self.released {
            return Ok(vec![frame]);
        }
        let first = *self.first_ns.get_or_insert(frame.timestamp_ns);
        if self.last_ns.is_some_and(|t| frame.timestamp_ns <= t) {
            return Err("stationary startup: camera timestamps are not increasing".into());
        }
        self.last_ns = Some(frame.timestamp_ns);
        if frame.timestamp_ns.saturating_sub(first) > self.config.max_wait_ns {
            return Err(
                "stationary startup: timed out waiting for motion; no VIO poses published".into(),
            );
        }
        let end = first
            .checked_add(self.config.window_ns)
            .ok_or("startup timestamp overflow")?;
        if self.frames.len() >= self.config.max_frames {
            return Err("stationary startup: frame buffer limit exceeded".into());
        }
        if self
            .frames
            .iter()
            .map(|f| f.imu.len())
            .sum::<usize>()
            .saturating_add(frame.imu.len())
            > self.config.max_imu_samples
        {
            return Err("stationary startup: buffered IMU packet limit exceeded".into());
        }
        for sample in &frame.imu {
            if self.last_imu_ns.is_some_and(|t| sample.timestamp_ns <= t)
                || sample.timestamp_ns > frame.timestamp_ns
                || !sample
                    .gyro_rad_s
                    .iter()
                    .chain(sample.accel_m_s2.iter())
                    .all(|v| v.is_finite())
            {
                return Err("stationary startup: invalid IMU ordering or nonfinite input".into());
            }
            if self.report.is_some()
                && self.last_imu_ns.is_some_and(|t| {
                    sample.timestamp_ns.saturating_sub(t) > self.config.max_imu_gap_ns
                })
            {
                return Err(
                    "stationary startup: gap in IMU coverage while waiting for motion".into(),
                );
            }
            self.last_imu_ns = Some(sample.timestamp_ns);
            if let Some(report) = &self.report {
                let accel = calibrate_accel(&self.calibration.calib_accel_bias, sample.accel_m_s2);
                if (accel.norm() - Vector3::from(report.mean_accel_m_s2).norm()).abs()
                    > self.config.motion_accel_deviation_m_s2
                {
                    self.motion_excited = true;
                }
            }
            if self.report.is_none() && sample.timestamp_ns >= first && sample.timestamp_ns < end {
                if self.samples.len() >= self.config.max_imu_samples {
                    return Err("stationary startup: IMU buffer limit exceeded".into());
                }
                self.samples.push(ImuSample::new(
                    sample.timestamp_ns,
                    calibrate_gyro(&self.calibration.calib_gyro_bias, sample.gyro_rad_s),
                    calibrate_accel(&self.calibration.calib_accel_bias, sample.accel_m_s2),
                ));
            }
        }
        let tracks = self
            .frontend
            .as_mut()
            .unwrap()
            .process_frame(StereoFrame::new(
                frame.frame_id,
                frame.timestamp_ns,
                frame.cam0.clone(),
                None,
            ))
            .map_err(|e| e.to_string())?;
        let current: BTreeMap<_, _> = tracks
            .observations
            .iter()
            .filter(|o| o.camera_id == 0)
            .map(|o| (o.track_id, o.pixel))
            .collect();
        if self.frames.is_empty() && self.report.is_none() {
            self.reference = current.clone();
        }
        // Newly detected corners can be untrackable at coarse pyramid borders.
        // Establish the persistent pool on the first temporal match, then apply
        // the retention fraction to that pool. The absolute support and motion
        // gates apply on every frame, including this first match.
        let mut visual_config = self.config.clone();
        if self.frames.len() == 1 && self.report.is_none() {
            visual_config.min_track_fraction = 0.0;
        }
        if self.report.is_some() {
            visual_config.max_disparity_p90_px = f64::MAX;
            visual_config.min_track_fraction = 0.0;
        }
        let (common, disparity) = visual_motion(&self.reference, &current, &visual_config)?;
        if self.frames.len() == 1 && self.report.is_none() {
            self.reference.retain(|id, _| current.contains_key(id));
        }
        // During the potentially long motion wait, replace an aging anchor
        // only while its surviving tracks still verify negligible motion.
        // Absolute support remains required; failed tracking is never motion.
        if self.report.is_some()
            && disparity < self.config.motion_disparity_px
            && (common as f64) < self.reference.len() as f64 * self.config.min_track_fraction
        {
            self.reference = current;
        }
        self.minimum_common = self.minimum_common.min(common);
        self.disparity = self.disparity.max(disparity);
        let ready_ns = frame.timestamp_ns;
        self.frames.push(frame);
        if self.report.is_some() {
            if disparity >= self.config.motion_disparity_px && self.motion_excited {
                self.report.as_mut().unwrap().released_camera_ns = Some(ready_ns);
                self.released = true;
                self.frontend = None;
                self.reference.clear();
                return Ok(std::mem::take(&mut self.frames));
            }
            self.retain_last_stationary_frame();
            return Ok(Vec::new());
        }
        if ready_ns < end {
            return Ok(Vec::new());
        }
        let (gyro, accel, gyro_std, accel_std) =
            stationary_statistics(&self.samples, first, end, &self.config)?;
        self.report = Some(StationaryStartupReport {
            config: self.config.clone(),
            first_camera_ns: first,
            ready_camera_ns: ready_ns,
            buffered_frames: self.frames.len(),
            released_camera_ns: if self.config.wait_for_motion {
                None
            } else {
                Some(ready_ns)
            },
            discarded_stationary_frames: 0,
            imu_samples: self.samples.len(),
            gyro_offset_rad_s: gyro.into(),
            mean_accel_m_s2: accel.into(),
            gyro_std_rad_s: gyro_std,
            accel_std_m_s2: accel_std,
            max_disparity_p90_px: self.disparity,
            minimum_common_tracks: self.minimum_common,
        });
        self.samples.clear();
        if self.config.wait_for_motion {
            self.retain_last_stationary_frame();
            return Ok(Vec::new());
        }
        self.released = true;
        self.reference.clear();
        self.frontend = None;
        Ok(std::mem::take(&mut self.frames))
    }

    fn retain_last_stationary_frame(&mut self) {
        let count = self.frames.len().saturating_sub(1);
        self.frames.drain(..count);
        self.report.as_mut().unwrap().discarded_stationary_frames += count;
    }

    pub fn oldest_buffered_frame_id(&self) -> Option<u64> {
        self.frames.first().map(|f| f.frame_id)
    }

    pub fn is_initialized(&self) -> bool {
        self.released && !self.failed
    }

    pub fn finish(&self) -> Result<(), String> {
        if self.is_initialized() {
            Ok(())
        } else {
            Err(
                "stream ended before a valid stationary IMU startup window; no VIO poses published"
                    .into(),
            )
        }
    }
}

fn visual_motion(
    reference: &BTreeMap<u64, Point2<f64>>,
    current: &BTreeMap<u64, Point2<f64>>,
    c: &StationaryStartupConfig,
) -> Result<(usize, f64), String> {
    let mut distances: Vec<_> = reference
        .iter()
        .filter_map(|(id, p)| current.get(id).map(|q| (p - q).norm()))
        .collect();
    if distances.len() < c.min_common_tracks
        || (distances.len() as f64) < reference.len() as f64 * c.min_track_fraction
    {
        return Err(format!("stationary startup: insufficient persistent image tracks ({} of {}, need {} and fraction {})", distances.len(), reference.len(), c.min_common_tracks, c.min_track_fraction));
    }
    if distances.iter().any(|v| !v.is_finite()) {
        return Err("stationary startup: nonfinite image tracks".into());
    }
    distances.sort_by(f64::total_cmp);
    let p90 = distances[((distances.len() - 1) as f64 * 0.9).ceil() as usize];
    if p90 > c.max_disparity_p90_px {
        return Err(format!(
            "stationary startup: image motion {p90:.3} px exceeds limit"
        ));
    }
    Ok((distances.len(), p90))
}

fn stationary_statistics(
    samples: &[ImuSample],
    start: i64,
    end: i64,
    c: &StationaryStartupConfig,
) -> Result<(Vector3<f64>, Vector3<f64>, f64, f64), String> {
    if samples.len() < c.min_imu_samples {
        return Err("stationary startup: insufficient IMU coverage".into());
    }
    if samples[0].timestamp_ns - start > c.max_imu_gap_ns
        || end - samples.last().unwrap().timestamp_ns > c.max_imu_gap_ns
        || samples.windows(2).any(|s| {
            s[1].timestamp_ns <= s[0].timestamp_ns
                || s[1].timestamp_ns - s[0].timestamp_ns > c.max_imu_gap_ns
        })
    {
        return Err("stationary startup: gap in IMU coverage".into());
    }
    let n = samples.len() as f64;
    let gyro = samples
        .iter()
        .fold(Vector3::zeros(), |a, s| a + s.gyro_rad_s)
        / n;
    let accel = samples
        .iter()
        .fold(Vector3::zeros(), |a, s| a + s.accel_m_s2)
        / n;
    let gyro_std = (samples
        .iter()
        .map(|s| (s.gyro_rad_s - gyro).norm_squared())
        .sum::<f64>()
        / n)
        .sqrt();
    let accel_std = (samples
        .iter()
        .map(|s| (s.accel_m_s2 - accel).norm_squared())
        .sum::<f64>()
        / n)
        .sqrt();
    if !gyro.iter().chain(accel.iter()).all(|v| v.is_finite())
        || !gyro_std.is_finite()
        || !accel_std.is_finite()
        || gyro_std > c.max_gyro_std_rad_s
        || accel_std > c.max_accel_std_m_s2
        || gyro.norm() > c.max_gyro_mean_rad_s
        || (accel.norm() - 9.81).abs() > c.max_gravity_norm_error_m_s2
    {
        return Err(format!("stationary startup: IMU is not stationary (gyro std {gyro_std:.5}, accel std {accel_std:.5})"));
    }
    Ok((gyro, accel, gyro_std, accel_std))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn samples() -> Vec<ImuSample> {
        (0..125)
            .map(|i| {
                ImuSample::new(
                    i * 8_000_000,
                    Vector3::new(-0.002, 0.001, 0.0005),
                    Vector3::new(0.1, -6.8, -7.0),
                )
            })
            .collect()
    }
    #[test]
    fn stationary_bias_and_gravity_are_estimated_without_accel_bias_claim() {
        let (g, a, gs, as_) = stationary_statistics(
            &samples(),
            0,
            1_000_000_000,
            &StationaryStartupConfig::default(),
        )
        .unwrap();
        assert!((g - samples()[0].gyro_rad_s).norm() < 1e-14);
        assert!((a - samples()[0].accel_m_s2).norm() < 1e-12);
        assert!(gs < 1e-12 && as_ < 1e-12);
    }
    #[test]
    fn missing_vibrating_rotating_and_nonfinite_imu_are_rejected() {
        let c = StationaryStartupConfig::default();
        let mut s = samples();
        s.drain(30..50);
        assert!(stationary_statistics(&s, 0, 1_000_000_000, &c).is_err());
        let mut s = samples();
        s[50].accel_m_s2.x += 10.0;
        assert!(stationary_statistics(&s, 0, 1_000_000_000, &c).is_err());
        let mut s = samples();
        for v in &mut s {
            v.gyro_rad_s.x = 0.2;
        }
        assert!(stationary_statistics(&s, 0, 1_000_000_000, &c).is_err());
        let mut s = samples();
        s[50].gyro_rad_s.x = f64::NAN;
        assert!(stationary_statistics(&s, 0, 1_000_000_000, &c).is_err());
    }
    #[test]
    fn image_motion_detects_constant_rotation_that_imu_variance_cannot() {
        let c = StationaryStartupConfig::default();
        let r: BTreeMap<_, _> = (0..40).map(|i| (i, Point2::new(i as f64, 10.0))).collect();
        assert!(visual_motion(&r, &r, &c).is_ok());
        let moving = r
            .iter()
            .map(|(i, p)| (*i, p + nalgebra::Vector2::new(2.0, 0.0)))
            .collect();
        assert!(visual_motion(&r, &moving, &c).is_err());
        assert!(visual_motion(&r, &BTreeMap::new(), &c).is_err());
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;
    use crate::{camera::DoubleSphereCamera, BasaltVioEstimatorAdapter, RawU16Image};
    use visloc_core::geometry::SE3;

    fn fixture() -> (
        BasaltCalibration,
        BasaltConfig,
        Vec<EurocSensorFrame>,
        StationaryStartupConfig,
    ) {
        let calibration = BasaltCalibration {
            t_imu_cam: vec![SE3::identity()],
            cameras: vec![
                DoubleSphereCamera::new(250., 250., 160., 120., 0., 0., 320, 240).unwrap(),
            ],
            resolutions: vec![(320, 240)],
            calib_accel_bias: vec![0.; 9],
            calib_gyro_bias: vec![0.; 12],
            imu_update_rate_hz: 100.,
            accel_noise_std: Vector3::repeat(0.01),
            gyro_noise_std: Vector3::repeat(0.001),
            accel_bias_std: Vector3::repeat(0.001),
            gyro_bias_std: Vector3::repeat(0.0001),
            t_mocap_world: SE3::identity(),
            t_imu_marker: SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        };
        let config =
            BasaltConfig::from_json(include_str!("../../../configs/basalt/euroc_config.json"))
                .unwrap();
        let mut seed = 42u32;
        let image = RawU16Image::new(
            320,
            240,
            (0..320 * 240)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 24) as u16) << 8
                })
                .collect(),
        )
        .unwrap();
        let sample = |t| {
            ImuSample::new(
                t,
                Vector3::new(-0.002, 0.001, 0.0005),
                Vector3::new(0., 0., 9.81),
            )
        };
        let frames = (0..3)
            .map(|i| EurocSensorFrame {
                frame_id: i,
                timestamp_ns: i as i64 * 50_000_000,
                cam0: image.clone(),
                cam1: None,
                cam0_path: Default::default(),
                cam1_path: None,
                imu: if i == 0 {
                    vec![sample(0)]
                } else {
                    ((i - 1) * 5 + 1..=i * 5)
                        .map(|k| sample(k as i64 * 10_000_000))
                        .collect()
                },
                initialization_imu: if i == 0 { Some(sample(0)) } else { None },
            })
            .collect();
        let startup = StationaryStartupConfig {
            window_ns: 100_000_000,
            min_imu_samples: 10,
            min_common_tracks: 3,
            ..Default::default()
        };
        (calibration, config, frames, startup)
    }

    #[test]
    fn packets_are_released_once_in_order_only_after_stationarity_is_verified() {
        let (c, v, f, s) = fixture();
        let mut gate = StationaryStartup::new(&c, &v, s).unwrap();
        assert!(gate.finish().is_err());
        assert!(gate.push(f[0].clone()).unwrap().is_empty());
        assert!(gate.push(f[1].clone()).unwrap().is_empty());
        assert!(gate.report.is_none());
        assert_eq!(gate.push(f[2].clone()).unwrap(), f);
        assert!(gate.finish().is_ok());
        let report = gate.report.as_ref().unwrap();
        assert_eq!(report.imu_samples, 10); // half-open sensor window
        let mut adapter = BasaltVioEstimatorAdapter::from_config(&c, &v).unwrap();
        adapter.estimator.apply_stationary_startup(report).unwrap();
        assert!(adapter.estimator.apply_stationary_startup(report).is_err());
        let out = adapter
            .process_without_marg_data_no_trace(f[0].clone())
            .unwrap();
        assert!(out.estimator.state.accel_bias_m_s2.norm() == 0.0);
        assert!(adapter.estimator.apply_stationary_startup(report).is_err());
    }

    #[test]
    fn startup_buffers_and_camera_order_are_bounded_without_silent_fallback() {
        let (c, v, f, mut s) = fixture();
        s.max_frames = 2;
        let mut gate = StationaryStartup::new(&c, &v, s.clone()).unwrap();
        gate.push(f[0].clone()).unwrap();
        gate.push(f[1].clone()).unwrap();
        assert!(gate
            .push(f[2].clone())
            .unwrap_err()
            .contains("buffer limit"));
        assert!(gate.finish().is_err());
        let mut gate = StationaryStartup::new(&c, &v, s).unwrap();
        gate.push(f[0].clone()).unwrap();
        assert!(gate.push(f[0].clone()).unwrap_err().contains("timestamps"));
    }

    #[test]
    fn motion_wait_is_bounded_and_does_not_recalibrate_from_moving_samples() {
        let (c, v, f, mut s) = fixture();
        s.wait_for_motion = true;
        s.average_gravity = true;
        s.max_wait_ns = 200_000_000;
        let mut gate = StationaryStartup::new(&c, &v, s).unwrap();
        for frame in &f {
            assert!(gate.push(frame.clone()).unwrap().is_empty());
        }
        assert!(!gate.is_initialized());
        assert_eq!(gate.oldest_buffered_frame_id(), Some(2));
        assert_eq!(gate.report.as_ref().unwrap().discarded_stationary_frames, 2);
        let mut moving = f[2].clone();
        moving.frame_id = 3;
        moving.timestamp_ns = 150_000_000;
        moving.imu = (11..=15)
            .map(|i| {
                ImuSample::new(
                    i * 10_000_000,
                    Vector3::new(-0.002, 0.001, 0.0005),
                    Vector3::new(0., 0., 10.31),
                )
            })
            .collect();
        let raw = moving.cam0.pixels();
        let shifted = (0..320 * 240)
            .map(|i| raw[i / 320 * 320 + (i % 320 + 316) % 320])
            .collect();
        moving.cam0 = RawU16Image::new(320, 240, shifted).unwrap();
        let released = gate.push(moving).unwrap();
        assert_eq!(
            released.iter().map(|f| f.frame_id).collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert!(gate.is_initialized());
        assert_eq!(
            gate.report.as_ref().unwrap().mean_accel_m_s2,
            [0., 0., 9.81]
        );
        let (c, v, f, mut s) = fixture();
        s.wait_for_motion = true;
        s.max_wait_ns = 100_000_000;
        let mut gate = StationaryStartup::new(&c, &v, s).unwrap();
        for frame in &f {
            gate.push(frame.clone()).unwrap();
        }
        assert!(gate.finish().is_err());
        let mut late = f[2].clone();
        late.frame_id = 3;
        late.timestamp_ns = 150_000_000;
        assert!(gate.push(late).unwrap_err().contains("timed out"));
    }

    #[test]
    fn an_existing_affine_sensor_calibration_is_respected() {
        let (mut c, v, mut f, s) = fixture();
        c.calib_gyro_bias[0] = 0.1;
        c.calib_gyro_bias[3] = 1.0;
        for frame in &mut f {
            for sample in &mut frame.imu {
                sample.gyro_rad_s.x = 0.049;
            }
        }
        let mut gate = StationaryStartup::new(&c, &v, s).unwrap();
        for frame in f {
            gate.push(frame).unwrap();
        }
        assert!((gate.report.unwrap().gyro_offset_rad_s[0] + 0.002).abs() < 1e-14);
    }

    #[test]
    fn imu_gap_during_motion_wait_cannot_release_a_stale_seed() {
        let (c, v, f, mut s) = fixture();
        s.wait_for_motion = true;
        let mut gate = StationaryStartup::new(&c, &v, s).unwrap();
        for frame in &f {
            gate.push(frame.clone()).unwrap();
        }
        let mut late = f[2].clone();
        late.frame_id = 3;
        late.timestamp_ns = 200_000_000;
        late.imu.truncate(1);
        late.imu[0].timestamp_ns = 200_000_000;
        assert!(gate.push(late).unwrap_err().contains("gap in IMU coverage"));
        assert!(!gate.is_initialized());
    }
}
