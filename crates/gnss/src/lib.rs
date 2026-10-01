//! Experimental GPS L1 / Galileo E1 code and Doppler integration.
//! All distances, including clock biases, are metres; clock drift is m/s.
pub mod bootstrap;
pub mod calibration;
pub mod factor;
pub mod navigation;
pub mod ubx;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
pub const C: f64 = 299_792_458.0;
pub const L1_HZ: f64 = 1_575_420_000.0;
pub const OMEGA_E: f64 = 7.292_115_146_7e-5;
pub const WEEK: f64 = 604_800.0;
pub const GPS_UNIX_EPOCH: i64 = 315_964_800;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum System {
    Gps,
    Galileo,
}
impl System {
    pub fn index(self) -> usize {
        match self {
            Self::Gps => 0,
            Self::Galileo => 1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Satellite {
    pub system: System,
    pub prn: u8,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub satellite: Satellite,
    pub signal: u8,
    pub pseudorange_m: f64,
    pub doppler_hz: f64,
    pub pseudorange_std_m: f64,
    pub doppler_std_hz: f64,
    pub cn0_dbhz: u8,
    pub lock_ms: u16,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Epoch {
    pub week: u16,
    pub tow_s: f64,
    pub leap_seconds: i8,
    pub leap_seconds_valid: bool,
    pub clock_reset: bool,
    /// Receipt time is diagnostic; it is never treated as the observation time.
    pub receipt_ns: i64,
    pub observations: Vec<Observation>,
}
impl Epoch {
    pub fn gps_seconds(&self) -> f64 {
        self.week as f64 * WEEK + self.tow_s
    }
    pub fn nominal_utc_ns(&self) -> Option<i64> {
        if !self.leap_seconds_valid
            || self.week == 0
            || !self.tow_s.is_finite()
            || !(0. ..WEEK).contains(&self.tow_s)
        {
            return None;
        }
        let seconds =
            GPS_UNIX_EPOCH as i128 + self.week as i128 * 604_800 - self.leap_seconds as i128;
        let nanoseconds = seconds * 1_000_000_000 + (self.tow_s * 1e9).round() as i128;
        i64::try_from(nanoseconds).ok()
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    PseudorangeDoppler,
    DopplerOnly,
    /// Same retained-keyframe window, without GNSS factors, for ablation.
    WindowOnly,
}
/// The legacy epoch-state path is retained for controlled comparisons.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coupling {
    FrameWindow,
    #[default]
    KeyframePreintegration,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub enabled: bool,
    pub mode: Mode,
    pub coupling: Coupling,
    pub gasket_gap_mm: f64,
    pub lever_arm_std_m: f64,
    pub min_cn0_dbhz: u8,
    pub min_elevation_deg: f64,
    pub pseudorange_floor_m: f64,
    pub range_rate_floor_m_s: f64,
    pub huber_sigma: f64,
    pub reject_sigma: f64,
    pub clock_bias_walk_m_sqrt_s: f64,
    pub clock_drift_walk_m_s_sqrt_s: f64,
    pub max_queue_epochs: usize,
    /// Legacy frame-window limits; keyframe coupling keeps the normal short window.
    pub nominal_imu_period_s: f64,
    /// Maximum age when first binding a delayed epoch; does not grow the Nav window.
    pub max_epoch_age_s: f64,
    pub min_active_window_s: f64,
    pub max_navigation_states: usize,
    pub max_bootstrap_seconds: f64,
    pub min_bootstrap_motion_m: f64,
    pub max_timing_std_s: f64,
    pub timing_search_s: f64,
    /// Explicit sensor-clock mapping override, for recordings with known PPS.
    /// Absent means bootstrap and freeze; zero is not an implicit default.
    pub fixed_time_offset_s: Option<f64>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: Mode::PseudorangeDoppler,
            coupling: Coupling::KeyframePreintegration,
            gasket_gap_mm: 0.6,
            lever_arm_std_m: 0.02,
            min_cn0_dbhz: 25,
            min_elevation_deg: 15.,
            pseudorange_floor_m: 3.,
            range_rate_floor_m_s: 0.3,
            huber_sigma: 2.5,
            reject_sigma: 8.,
            clock_bias_walk_m_sqrt_s: 10.,
            clock_drift_walk_m_s_sqrt_s: 1.,
            max_queue_epochs: 256,
            nominal_imu_period_s: 0.005,
            max_epoch_age_s: 0.5,
            min_active_window_s: 0.5,
            max_navigation_states: 32,
            max_bootstrap_seconds: 120.,
            min_bootstrap_motion_m: 10.,
            max_timing_std_s: 0.02,
            timing_search_s: 0.5,
            fixed_time_offset_s: None,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            self.lever_arm_std_m,
            self.pseudorange_floor_m,
            self.range_rate_floor_m_s,
            self.huber_sigma,
            self.reject_sigma,
            self.clock_bias_walk_m_sqrt_s,
            self.clock_drift_walk_m_s_sqrt_s,
            self.max_bootstrap_seconds,
            self.min_bootstrap_motion_m,
            self.max_timing_std_s,
            self.timing_search_s,
            self.min_active_window_s,
            self.nominal_imu_period_s,
            self.max_epoch_age_s,
        ];
        if positive.iter().any(|x| !x.is_finite() || *x <= 0.)
            || !self.gasket_gap_mm.is_finite()
            || !(0. ..=10.).contains(&self.gasket_gap_mm)
            || !(0. ..90.).contains(&self.min_elevation_deg)
            || self.max_queue_epochs == 0
            || !(5..=128).contains(&self.max_navigation_states)
            || self.fixed_time_offset_s.is_some_and(|x| !x.is_finite())
        {
            return Err("invalid raw GNSS configuration".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Diagnostics {
    pub epochs: u64,
    pub checksum_errors: u64,
    pub malformed_packets: u64,
    pub unsupported_signals: u64,
    #[serde(default)]
    pub unsupported_signal_counts: std::collections::BTreeMap<String, u64>,
    #[serde(default)]
    pub invalid_observation_counts: std::collections::BTreeMap<String, u64>,
    pub missing_ephemeris: u64,
    pub dropped_epochs: u64,
    pub dropped_records: u64,
    pub stale_epochs: u64,
    pub clock_resets: u64,
    pub low_cn0: u64,
    pub low_elevation: u64,
    pub rejected_pseudoranges: u64,
    pub rejected_dopplers: u64,
    pub accepted_pseudoranges: u64,
    pub accepted_dopplers: u64,
    pub status: String,
    pub time_offset_s: Option<f64>,
    pub timing_std_s: Option<f64>,
    pub yaw_rad: Option<f64>,
    pub origin_ecef_m: Option<Vector3<f64>>,
}
