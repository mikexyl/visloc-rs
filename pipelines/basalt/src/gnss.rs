//! Optional joint raw-GNSS window. Camera identities and retirement remain
//! estimator-owned; epoch navigation and clock identities live in this module.
use crate::vio::{
    aom::{
        landmark_nullspace_projection, solve_lm_with_timing, LandmarkBackSubstitution, LmConfig,
        LmFailure, LmLinearization, LmProblem, LmTrialToken, ReducedNormalSystem,
        WhitenedFactorRowStack,
    },
    window::{
        decode_state, flatten_nav, projected_rows, stack_rows, WindowDiagnostics, WindowImuLink,
        WindowProblem, WindowSolveError, WindowSolveResult, WindowState,
    },
};
use crate::{
    imu::{integrate_between, interpolate_at},
    timing::TimingBreakdown,
    BasaltNavState, ImuSample,
};
mod motion;
use motion::{EpochMotion, Matrix15};
use nalgebra::{DMatrix, DVector, Matrix2, UnitQuaternion, Vector3};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use visloc_gnss::{
    bootstrap::{self, Bootstrap, Solution, VioSample},
    calibration::{lever_arm, LeverArm},
    factor::{self, Alignment, Body, Clock, PreparedObservation},
    navigation::{atmospheric_delay, azimuth_elevation, Ephemeris, Navigation},
    ubx::{Decoder, Message},
    Config, Coupling, Diagnostics, Epoch, Mode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Pose(u64),
    Nav(u64),
    Epoch(u64),
    Clock(u64),
    Alignment,
}
#[derive(Debug, Clone)]
struct Block {
    key: Key,
    offset: usize,
    dof: usize,
    fej: DVector<f64>,
}
#[derive(Debug, Clone)]
struct Prior {
    blocks: Vec<Block>,
    j: DMatrix<f64>,
    r: DVector<f64>,
}
#[derive(Debug, Clone)]
struct TimedEpoch {
    id: u64,
    time: i64,
    epoch: Epoch,
    clock: Clock,
}
#[derive(Debug, Clone)]
struct ClockEpoch {
    id: u64,
    time: i64,
    clock: Clock,
    reset: bool,
}

#[derive(Debug)]
pub enum Record {
    Epoch(Epoch),
    Ephemeris(Ephemeris),
    Decision(serde_json::Value),
}

#[derive(Debug)]
pub struct GnssRuntime {
    pub config: Config,
    pub lever_arm: LeverArm,
    pub diagnostics: Diagnostics,
    pub navigation: Navigation,
    pub records: VecDeque<Record>,
    decoder: Decoder,
    week: u16,
    next_id: u64,
    epochs: VecDeque<TimedEpoch>,
    solutions: Vec<Solution>,
    vio: Vec<VioSample>,
    imu: Vec<ImuSample>,
    pub bootstrap: Option<Bootstrap>,
    pub initialization: bootstrap::InitializationDiagnostics,
    prior: Option<Prior>,
    hidden: BTreeMap<u64, BasaltNavState>,
    epoch_keyframes: BTreeMap<u64, u64>,
    motions: BTreeMap<u64, EpochMotion>,
    seen_epochs: BTreeSet<(u16, u64)>,
    seen_order: VecDeque<(u16, u64)>,
    clocks: BTreeMap<u64, ClockEpoch>,
    decisions: BTreeMap<(u64, visloc_gnss::Satellite), (bool, bool)>,
    consumed_before: i64,
    pub replay_shift_ns: i64,
}
impl GnssRuntime {
    pub fn new(
        config: Config,
        r_imu_cam: &UnitQuaternion<f64>,
        t_imu_cam: Vector3<f64>,
    ) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            lever_arm: lever_arm(
                config.gasket_gap_mm,
                config.lever_arm_std_m,
                r_imu_cam,
                t_imu_cam,
            ),
            config,
            diagnostics: Diagnostics {
                status: "awaiting_ephemeris_and_motion".into(),
                ..Default::default()
            },
            navigation: Navigation::default(),
            records: VecDeque::new(),
            decoder: Decoder::default(),
            week: 0,
            next_id: 0,
            epochs: VecDeque::new(),
            solutions: Vec::new(),
            vio: Vec::new(),
            imu: Vec::new(),
            bootstrap: None,
            initialization: Default::default(),
            prior: None,
            hidden: BTreeMap::new(),
            epoch_keyframes: BTreeMap::new(),
            motions: BTreeMap::new(),
            seen_epochs: BTreeSet::new(),
            seen_order: VecDeque::new(),
            clocks: BTreeMap::new(),
            decisions: BTreeMap::new(),
            consumed_before: i64::MIN,
            replay_shift_ns: 0,
        })
    }
    fn record(&mut self, record: Record) {
        self.records.push_back(record);
        while self.records.len() > self.config.max_queue_epochs.saturating_mul(64) {
            self.records.pop_front();
            self.diagnostics.dropped_records += 1;
        }
    }
    fn store_solution(&mut self, solution: Solution) {
        let index = self
            .solutions
            .partition_point(|s| s.utc_ns < solution.utc_ns);
        self.solutions.insert(index, solution);
        // GNSS can continue while camera input is absent. Bound history here,
        // independently of sensor_frame's normal sensor-time retirement.
        let latest = self.solutions.last().unwrap().utc_ns;
        let cutoff = latest.saturating_sub((self.config.max_bootstrap_seconds * 1e9) as i64);
        self.solutions.retain(|s| s.utc_ns >= cutoff);
        let excess = self
            .solutions
            .len()
            .saturating_sub(self.config.max_queue_epochs.saturating_mul(16));
        self.solutions.drain(..excess);
    }
    pub fn push_raw(&mut self, bytes: &[u8], receipt_ns: i64) {
        for message in self.decoder.push(bytes, receipt_ns) {
            match message {
                Message::Subframe(f) => {
                    if let Some(e) = self.navigation.push(&f, self.week) {
                        self.record(Record::Ephemeris(e));
                    }
                }
                Message::Epoch(e) => {
                    self.record(Record::Epoch(e.clone()));
                    self.push_epoch(e);
                }
            }
        }
        self.diagnostics.checksum_errors = self.decoder.diagnostics.checksum_errors;
        self.diagnostics.malformed_packets = self.decoder.diagnostics.malformed_packets;
        self.diagnostics.unsupported_signals = self.decoder.diagnostics.unsupported_signals;
        self.diagnostics.unsupported_signal_counts =
            self.decoder.diagnostics.unsupported_signal_counts.clone();
        self.diagnostics.invalid_observation_counts =
            self.decoder.diagnostics.invalid_observation_counts.clone();
    }
    pub fn push_ephemeris(&mut self, e: Ephemeris) {
        self.navigation.insert(e);
    }
    pub fn push_epoch(&mut self, e: Epoch) {
        if !self.config.enabled {
            return;
        }
        let Some(rawtime) = e.nominal_utc_ns() else {
            self.diagnostics.status = "invalid_gnss_time".into();
            self.record(Record::Decision(serde_json::json!({
                "reason":"invalid_gnss_time", "week":e.week, "tow_s":e.tow_s,
            })));
            return;
        };
        self.week = e.week;
        self.diagnostics.epochs += 1;
        if e.clock_reset {
            self.diagnostics.clock_resets += 1;
        }
        let identity = (e.week, e.tow_s.to_bits());
        if !self.seen_epochs.insert(identity) {
            self.record(Record::Decision(serde_json::json!({
                "reason":"duplicate_epoch", "week":e.week, "tow_s":e.tow_s,
            })));
            return;
        }
        self.seen_order.push_back(identity);
        while self.seen_order.len() > self.config.max_queue_epochs.saturating_mul(16) {
            self.seen_epochs
                .remove(&self.seen_order.pop_front().unwrap());
        }
        let seed = self.solutions.last().map(|s| s.ecef_m);
        let solution = bootstrap::solve_epoch(&e, &self.navigation, &self.config, seed);
        let clock = solution.as_ref().map_or_else(
            || {
                if let Some(front) = self.clocks.values().max_by_key(|c| c.time) {
                    let dt = (rawtime + self.replay_shift_ns - front.time) as f64 * 1e-9;
                    let mut c = front.clock.clone();
                    for b in &mut c.bias_m {
                        *b += c.drift_m_s * dt.max(0.);
                    }
                    c
                } else {
                    self.bootstrap
                        .as_ref()
                        .map_or_else(Clock::default, |b| b.clock.clone())
                }
            },
            |s| s.clock.clone(),
        );
        let reference = if e
            .observations
            .iter()
            .any(|o| o.satellite.system == visloc_gnss::System::Gps)
        {
            clock.bias_m[0]
        } else {
            clock.bias_m[1]
        };
        let physical = rawtime - (reference / visloc_gnss::C * 1e9).round() as i64;
        if let Some(mut s) = solution {
            s.utc_ns += self.replay_shift_ns;
            self.store_solution(s);
        }
        let time = physical + self.replay_shift_ns;
        let offset = self
            .bootstrap
            .as_ref()
            .map_or(0, |b| (b.time_offset_s * 1e9).round() as i64);
        if self.config.coupling == Coupling::FrameWindow && time + offset <= self.consumed_before {
            self.diagnostics.stale_epochs += 1;
            self.record(Record::Decision(serde_json::json!({
                "reason":"outside_active_window", "week":e.week, "tow_s":e.tow_s,
                "timestamp_ns":time+offset, "consumed_before_ns":self.consumed_before,
            })));
            return;
        }
        let id = self.next_id;
        self.next_id += 1;
        let i = self.epochs.partition_point(|x| x.time < time);
        self.epochs.insert(
            i,
            TimedEpoch {
                id,
                time,
                epoch: e,
                clock,
            },
        );
        if self.epochs.len() > self.config.max_queue_epochs {
            self.epochs.pop_front();
            self.diagnostics.dropped_epochs += 1;
        }
    }
    pub fn record_imu(&mut self, imu: &[ImuSample]) {
        for s in imu {
            if self
                .imu
                .last()
                .is_none_or(|last| s.timestamp_ns > last.timestamp_ns)
            {
                self.imu.push(*s);
            }
        }
    }
    pub fn sensor_frame(&mut self, t: i64, nav: &BasaltNavState, imu: &[ImuSample]) {
        for s in imu {
            if self
                .imu
                .last()
                .is_none_or(|last| s.timestamp_ns > last.timestamp_ns)
            {
                self.imu.push(*s);
            }
        }
        let omega = interpolate_at(&self.imu, t).map_or_else(
            |_| self.imu.last().map_or(Vector3::zeros(), |s| s.gyro_rad_s),
            |s| s.gyro_rad_s,
        ) - nav.gyro_bias_rad_s;
        self.vio.push(VioSample {
            timestamp_ns: t,
            position: nav.imu_to_world.translation,
            rotation: nav.imu_to_world.rotation,
            velocity: nav.velocity_world_m_s,
            omega,
        });
        let oldest = t - (self.config.max_bootstrap_seconds * 1e9) as i64;
        self.vio.retain(|s| s.timestamp_ns >= oldest);
        self.solutions.retain(|s| s.utc_ns >= oldest);
        if self.bootstrap.is_none() {
            (self.bootstrap, self.initialization) = bootstrap::initialize_diagnosed(
                &self.solutions,
                &self.vio,
                self.lever_arm.imu_to_antenna_m,
                &self.config,
            );
            self.diagnostics.status = self.initialization.reason.clone();
            if let Some(b) = &self.bootstrap {
                self.diagnostics.status = "active".into();
                self.diagnostics.time_offset_s = Some(b.time_offset_s);
                self.diagnostics.timing_std_s = Some(b.timing_std_s);
                self.diagnostics.origin_ecef_m = Some(b.alignment.origin_ecef_m);
                self.diagnostics.yaw_rad = Some(b.alignment.yaw_rad);
            }
        }
        // Keep the active camera window plus one interpolation packet. The
        // bootstrap trajectory retains no raw IMU beyond this bounded horizon.
        let imu_oldest = t - 10_000_000_000;
        let n = self
            .imu
            .partition_point(|s| s.timestamp_ns < imu_oldest)
            .saturating_sub(1);
        self.imu.drain(..n);
    }
    pub fn active(&self) -> bool {
        self.config.enabled && self.bootstrap.is_some()
    }
    fn build(&mut self, source: &WindowProblem) -> Result<JointProblem, String> {
        if self.config.coupling == Coupling::KeyframePreintegration {
            return self.build_keyframes(source, None);
        }
        self.build_frame_window(source)
    }
    fn build_frame_window(&mut self, source: &WindowProblem) -> Result<JointProblem, String> {
        let boot = self
            .bootstrap
            .as_ref()
            .ok_or("GNSS is not initialized")?
            .clone();
        let mut base = source.clone();
        let actual = base.states.len();
        if self.prior.is_some() {
            base.prior = None;
            base.anchor_point = None;
        }
        let first = base
            .states
            .first()
            .ok_or("empty camera state window")?
            .timestamp_ns;
        let last = base.states.last().unwrap().timestamp_ns;
        let offset = (boot.time_offset_s * 1e9).round() as i64;
        while self.epochs.front().is_some_and(|e| e.time + offset < first) {
            let old = self.epochs.pop_front().unwrap();
            self.record_decision(
                old.id,
                None,
                "outside_active_window",
                None,
                None,
                false,
                false,
            );
            self.diagnostics.stale_epochs += 1;
        }
        let selected: Vec<_> = self
            .epochs
            .iter()
            .filter(|e| {
                e.time + offset > self.consumed_before
                    && e.time + offset >= first
                    && e.time + offset <= last
            })
            .cloned()
            .collect();
        let mut epochs = Vec::new();
        let mut observations = Vec::new();
        let original_links = base.imu_links.clone();
        let mut replacements = BTreeMap::<usize, Vec<(usize, i64)>>::new();
        for e in selected {
            let t = e.time + offset;
            let Some((link_id, link)) = original_links.iter().enumerate().find(|(_, l)| {
                base.states[l.from_index].timestamp_ns <= t
                    && t <= base.states[l.to_index].timestamp_ns
            }) else {
                continue;
            };
            let from = &base.states[link.from_index];
            let sample = interpolate_at(&self.imu, t)
                .map_err(|_| format!("GNSS epoch {t} lacks bracketing IMU"))?;
            let index = if t == from.timestamp_ns {
                link.from_index
            } else if t == base.states[link.to_index].timestamp_ns {
                link.to_index
            } else {
                let nav = if let Some(nav) = self.hidden.get(&e.id) {
                    nav.clone()
                } else {
                    let delta = integrate_split(
                        &self.imu,
                        from.timestamp_ns,
                        t,
                        from.nav.gyro_bias_rad_s,
                        from.nav.accel_bias_m_s2,
                        Some(base.imu_noise),
                        self.config.nominal_imu_period_s,
                    )
                    .map_err(|x| format!("split GNSS preintegration: {x:?}"))?;
                    predict(&from.nav, &delta, base.gravity_world)
                };
                let mut solver_id = u64::MAX - e.id;
                while base.states.iter().any(|s| s.frame_id == solver_id)
                    || base.poses.iter().any(|p| p.frame_id == solver_id)
                {
                    solver_id = solver_id
                        .checked_sub(1)
                        .ok_or("GNSS solver identity exhausted")?;
                }
                let index = base.states.len();
                base.states.push(WindowState {
                    frame_id: solver_id,
                    timestamp_ns: t,
                    nav: nav.clone(),
                    stored_current_nav: nav.clone(),
                    linearized_nav: nav,
                    linearized_delta: DVector::zeros(15),
                    is_keyframe: false,
                    is_latest: false,
                    linearized: false,
                });
                replacements.entry(link_id).or_default().push((index, t));
                index
            };
            self.clocks.entry(e.id).or_insert(ClockEpoch {
                id: e.id,
                time: t,
                clock: e.clock.clone(),
                reset: e.epoch.clock_reset,
            });
            let (screened, _) =
                self.screen_epoch(&e, index, &base.states[index].nav, &sample, &boot, None);
            observations.extend(screened);
            epochs.push((e.id, index));
        }
        // Replace each affected IMU link; no original measurement survives.
        let mut links = Vec::new();
        for (i, link) in original_links.into_iter().enumerate() {
            if let Some(mut interior) = replacements.remove(&i) {
                interior.sort_by_key(|x| x.1);
                let mut indices = vec![link.from_index];
                indices.extend(interior.iter().map(|x| x.0));
                indices.push(link.to_index);
                for pair in indices.windows(2) {
                    let (a, b) = (&base.states[pair[0]], &base.states[pair[1]]);
                    let delta = integrate_split(
                        &self.imu,
                        a.timestamp_ns,
                        b.timestamp_ns,
                        a.nav.gyro_bias_rad_s,
                        a.nav.accel_bias_m_s2,
                        Some(base.imu_noise),
                        self.config.nominal_imu_period_s,
                    )
                    .map_err(|e| format!("split IMU interval: {e:?}"))?;
                    links.push(WindowImuLink {
                        from_index: pair[0],
                        to_index: pair[1],
                        delta,
                    });
                }
            } else {
                links.push(link);
            }
        }
        base.imu_links = links;
        let mut clocks: Vec<_> = epochs
            .iter()
            .filter_map(|(id, _)| self.clocks.get(id).cloned())
            .collect();
        if let Some(prior) = &self.prior {
            for block in &prior.blocks {
                if let Key::Clock(id) = block.key {
                    if !clocks.iter().any(|c| c.id == id) {
                        clocks.push(
                            self.clocks
                                .get(&id)
                                .ok_or("missing clock frontier")?
                                .clone(),
                        );
                    }
                }
            }
        }
        clocks.sort_by_key(|c| c.time);
        JointProblem::new(
            base,
            actual,
            epochs,
            clocks,
            observations,
            boot.alignment,
            self.config.clone(),
            self.lever_arm.imu_to_antenna_m,
            self.prior.clone(),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn screen_epoch(
        &mut self,
        e: &TimedEpoch,
        index: usize,
        nav: &BasaltNavState,
        sample: &ImuSample,
        boot: &Bootstrap,
        motion: Option<(&EpochMotion, &BasaltNavState)>,
    ) -> (
        Vec<(usize, u64, PreparedObservation)>,
        BTreeMap<(u64, visloc_gnss::Satellite), Matrix2<f64>>,
    ) {
        let body = body(nav);
        let p = boot
            .alignment
            .position(body.position + body.rotation * self.lever_arm.imu_to_antenna_m);
        let mut observations = Vec::new();
        let mut covariances = BTreeMap::new();
        for o in &e.epoch.observations {
            let unique = !self.decisions.contains_key(&(e.id, o.satellite));
            if o.cn0_dbhz < self.config.min_cn0_dbhz {
                if unique {
                    self.diagnostics.low_cn0 += 1;
                    self.record_decision(
                        e.id,
                        Some(o.satellite),
                        "low_cn0",
                        None,
                        None,
                        false,
                        false,
                    );
                }
                continue;
            }
            let Some(satellite) = self
                .navigation
                .select(o.satellite, e.epoch.gps_seconds())
                .and_then(|ep| ep.at_transmission(e.epoch.gps_seconds(), o.pseudorange_m))
            else {
                if unique {
                    self.diagnostics.missing_ephemeris += 1;
                    self.record_decision(
                        e.id,
                        Some(o.satellite),
                        "missing_ephemeris",
                        None,
                        None,
                        false,
                        false,
                    );
                }
                continue;
            };
            if azimuth_elevation(p, satellite.position_m).1
                < self.config.min_elevation_deg.to_radians()
            {
                if unique {
                    self.diagnostics.low_elevation += 1;
                    self.record_decision(
                        e.id,
                        Some(o.satellite),
                        "low_elevation",
                        None,
                        None,
                        false,
                        false,
                    );
                }
                continue;
            }
            let mut prepared = PreparedObservation {
                observation: o.clone(),
                atmosphere_m: atmospheric_delay(
                    p,
                    satellite.position_m,
                    e.epoch.tow_s,
                    self.navigation.ionosphere,
                ),
                satellite,
                gyro_rad_s: sample.gyro_rad_s,
                use_pseudorange: self.config.mode != Mode::DopplerOnly,
                use_doppler: true,
            };
            let mut gate = self.config.clone();
            gate.huber_sigma = f64::MAX;
            let clock = &self.clocks[&e.id].clock;
            let covariance = motion.map_or_else(Matrix2::zeros, |(m, anchor)| {
                measurement_covariance(
                    m,
                    anchor,
                    &prepared,
                    &body,
                    &boot.alignment,
                    clock,
                    self.lever_arm.imu_to_antenna_m,
                    boot.timing_std_s,
                )
            });
            let residual = factor::linearize_with_covariance(
                &prepared,
                &body,
                &boot.alignment,
                clock,
                self.lever_arm.imu_to_antenna_m,
                &gate,
                covariance,
            )
            .residual;
            if residual[0].abs() > self.config.reject_sigma {
                prepared.use_pseudorange = false;
            }
            if residual[1].abs() > self.config.reject_sigma {
                prepared.use_doppler = false;
            }
            if unique {
                if self.config.mode != Mode::DopplerOnly {
                    if prepared.use_pseudorange {
                        self.diagnostics.accepted_pseudoranges += 1;
                    } else {
                        self.diagnostics.rejected_pseudoranges += 1;
                    }
                }
                if prepared.use_doppler {
                    self.diagnostics.accepted_dopplers += 1;
                } else {
                    self.diagnostics.rejected_dopplers += 1;
                }
            }
            let flags = (prepared.use_pseudorange, prepared.use_doppler);
            if self.decisions.get(&(e.id, o.satellite)) != Some(&flags) {
                self.record_decision(
                    e.id,
                    Some(o.satellite),
                    "residual_gate",
                    Some(residual[0]),
                    Some(residual[1]),
                    flags.0,
                    flags.1,
                );
            }
            if prepared.use_pseudorange || prepared.use_doppler {
                observations.push((index, e.id, prepared.clone()));
                covariances.insert((e.id, o.satellite), covariance);
            }
        }
        (observations, covariances)
    }
    fn build_keyframes(
        &mut self,
        source: &WindowProblem,
        retiring: Option<&[u64]>,
    ) -> Result<JointProblem, String> {
        let boot = self
            .bootstrap
            .as_ref()
            .ok_or("GNSS is not initialized")?
            .clone();
        let mut base = source.clone();
        if self.prior.is_some() {
            base.prior = None;
            base.anchor_point = None;
        }
        let actual = base.states.len();
        let last = base
            .states
            .last()
            .ok_or("empty camera state window")?
            .timestamp_ns;
        let offset = (boot.time_offset_s * 1e9).round() as i64;
        let frontier_time = self
            .prior
            .as_ref()
            .into_iter()
            .flat_map(|p| &p.blocks)
            .filter_map(|b| {
                if let Key::Clock(id) = b.key {
                    self.clocks.get(&id).map(|c| c.time)
                } else {
                    None
                }
            })
            .max();
        let mut epochs = Vec::new();
        let mut observations = Vec::new();
        let mut motions = BTreeMap::new();
        let mut covariances = BTreeMap::new();
        let mut rejected = BTreeSet::new();
        for e in self.epochs.iter().cloned().collect::<Vec<_>>() {
            let t = e.time + offset;
            if self.config.mode == Mode::WindowOnly {
                if t <= last {
                    rejected.insert(e.id);
                }
                continue;
            }
            if retiring.is_none() && t > last {
                continue;
            }
            if !self.epoch_keyframes.contains_key(&e.id)
                && last.saturating_sub(t) as f64 * 1e-9 > self.config.max_epoch_age_s
            {
                rejected.insert(e.id);
                self.diagnostics.stale_epochs += 1;
                self.record_decision(
                    e.id,
                    None,
                    "delayed_epoch_age_limit",
                    None,
                    None,
                    false,
                    false,
                );
                continue;
            }
            // A pre-marginalization snapshot omits newer retained keyframes.
            // Missing from this prefix is not the same as having retired.
            if retiring.is_some_and(|ids| {
                self.epoch_keyframes
                    .get(&e.id)
                    .is_none_or(|owner| !ids.contains(owner))
            }) {
                continue;
            }
            let index = if let Some(owner) = self.epoch_keyframes.get(&e.id) {
                base.states.iter().position(|s| s.frame_id == *owner)
            } else if retiring.is_none() {
                base.states
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.is_keyframe)
                    .min_by_key(|(_, s)| ((s.timestamp_ns as i128 - t as i128).abs(), s.frame_id))
                    .map(|(i, _)| i)
            } else {
                None
            };
            let Some(index) = index else {
                // On activation the previous keyframe may already be Pose6.
                // Wait for the next actual keyframe rather than inventing velocity.
                if self.epoch_keyframes.contains_key(&e.id) {
                    rejected.insert(e.id);
                    self.diagnostics.stale_epochs += 1;
                    self.record_decision(e.id, None, "retired_keyframe", None, None, false, false);
                }
                continue;
            };
            let anchor = &base.states[index];
            if retiring.is_some_and(|ids| !ids.contains(&anchor.frame_id)) {
                continue;
            }
            if frontier_time.is_some_and(|f| t <= f) && !self.epoch_keyframes.contains_key(&e.id) {
                rejected.insert(e.id);
                self.diagnostics.stale_epochs += 1;
                self.record_decision(
                    e.id,
                    None,
                    "older_than_clock_frontier",
                    None,
                    None,
                    false,
                    false,
                );
                continue;
            }
            if !self.motions.contains_key(&e.id)
                && self
                    .imu
                    .last()
                    .is_some_and(|s| anchor.timestamp_ns.max(t) > s.timestamp_ns)
            {
                // The sensor worker intentionally gives VIO past IMU packets
                // only. Wait for the next update to bracket a newly selected
                // keyframe instead of extrapolating or losing the epoch.
                self.record(Record::Decision(serde_json::json!({"epoch_id":e.id,
                    "reason":"awaiting_keyframe_imu","keyframe_id":anchor.frame_id,
                    "keyframe_time_ns":anchor.timestamp_ns,"epoch_time_ns":t})));
                continue;
            }
            let motion = if let Some(m) = self.motions.get(&e.id) {
                m.clone()
            } else {
                match EpochMotion::new(
                    &self.imu,
                    anchor.timestamp_ns,
                    t,
                    &anchor.nav,
                    base.gravity_world,
                    base.imu_noise,
                    base.bias_walk_noise,
                    self.config.nominal_imu_period_s,
                ) {
                    Ok(m) => m,
                    Err(error) => {
                        rejected.insert(e.id);
                        self.diagnostics.stale_epochs += 1;
                        self.record(Record::Decision(serde_json::json!({"epoch_id":e.id,"reason":"missing_keyframe_imu","error":error})));
                        continue;
                    }
                }
            };
            self.epoch_keyframes.insert(e.id, anchor.frame_id);
            self.motions.insert(e.id, motion.clone());
            self.clocks.entry(e.id).or_insert(ClockEpoch {
                id: e.id,
                time: t,
                clock: e.clock.clone(),
                reset: e.epoch.clock_reset,
            });
            let (nav, _) = motion.evaluate(&anchor.nav);
            let (screened, noise) = self.screen_epoch(
                &e,
                index,
                &nav,
                &motion.sample,
                &boot,
                Some((&motion, &anchor.nav)),
            );
            observations.extend(screened);
            covariances.extend(noise);
            motions.insert(e.id, motion);
            epochs.push((e.id, index));
        }
        self.epochs.retain(|e| !rejected.contains(&e.id));
        let mut clocks: Vec<_> = epochs
            .iter()
            .filter_map(|(id, _)| self.clocks.get(id).cloned())
            .collect();
        if let Some(prior) = &self.prior {
            for block in &prior.blocks {
                if let Key::Clock(id) = block.key {
                    if !clocks.iter().any(|c| c.id == id) {
                        clocks.push(
                            self.clocks
                                .get(&id)
                                .ok_or("missing clock frontier")?
                                .clone(),
                        );
                    }
                }
            }
        }
        clocks.sort_by_key(|c| (c.time, c.id));
        let mut joint = JointProblem::new(
            base,
            actual,
            epochs,
            clocks,
            observations,
            boot.alignment,
            self.config.clone(),
            self.lever_arm.imu_to_antenna_m,
            self.prior.clone(),
        )?;
        joint.motions = motions;
        joint.covariances = covariances;
        Ok(joint)
    }
    pub(crate) fn solve(
        &mut self,
        source: &mut WindowProblem,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let error = |message: String| WindowSolveError {
            message,
            diagnostics: empty_diagnostics(),
        };
        let camera_links = source.imu_links.clone();
        let mut problem = self.build(source).map_err(error)?;
        let initial = problem.initial();
        let result =
            solve_lm_with_timing(&mut problem, initial.clone(), config, true, true, timing)
                .map_err(|e| error(format!("joint GNSS LM: {e:?}")))?;
        let initial_cost = result.trace.first().map_or(result.cost, |t| t.cost_before);
        let state = result.state;
        if state.iter().any(|v| !v.is_finite() || v.abs() > 1e9) {
            return Err(error("invalid joint GNSS solution magnitude".into()));
        }
        self.record(Record::Decision(serde_json::json!({"reason":"joint_solve","coupling":self.config.coupling,"camera_nav_states":problem.actual_states,"gnss_nav_states":problem.base.states.len()-problem.actual_states,"epoch_clocks":problem.clocks.len(),"state_dof":state.len(),"timestamp_ns":source.states.last().map(|s|s.timestamp_ns),"initial_cost":initial_cost,"final_cost":result.cost,"iterations":result.iterations,"trace":result.trace.iter().map(|t|serde_json::json!({"iteration":t.iteration,"cost_before":t.cost_before,"actual_cost":t.actual_cost,"step_norm":t.step_norm,"decision":format!("{:?}",t.decision)})).collect::<Vec<_>>()})));
        let (factor_count, factor_rows) = problem.factor_shape.get();
        let joint_landmarks = problem.base.landmarks.len();
        let joint_links = problem.base.imu_links.len();
        let prior_rows = problem.prior.as_ref().map_or(0, |p| p.j.nrows());
        self.writeback(&problem, &state);
        *source = problem.base;
        source.states.truncate(problem.actual_states);
        // source owns the camera-only IMU graph for the regular retirement
        // policy. Joint marginalization reconstructs replacements explicitly.
        source.imu_links = camera_links;
        let camera_dof = source.state_dof();
        let camera_state = state.rows(0, camera_dof).into_owned();
        let mut diagnostics = empty_diagnostics();
        diagnostics.status = "gnss_joint_success".into();
        diagnostics.attempted = true;
        diagnostics.state_dof = state.len();
        diagnostics.factor_count = factor_count;
        diagnostics.factor_rows = factor_rows;
        diagnostics.landmark_count = joint_landmarks;
        diagnostics.imu_link_count = joint_links;
        diagnostics.prior_rows = prior_rows;
        diagnostics.lm.push(crate::vio::window::LmRunDiagnostics {
            pass: "joint_gnss".into(),
            iterations: result.iterations,
            lambda: result.lambda,
            initial_cost,
            final_cost: result.cost,
            accepted: result
                .trace
                .iter()
                .filter(|t| matches!(t.decision, crate::vio::aom::LmDecision::Accepted))
                .count(),
            rejected: result
                .trace
                .iter()
                .filter(|t| matches!(t.decision, crate::vio::aom::LmDecision::Rejected))
                .count(),
            trace: result.trace.clone(),
            failure: None,
        });
        diagnostics.state_writeback = true;
        Ok(WindowSolveResult {
            state: camera_state,
            factors: Vec::new(),
            cost: result.cost,
            iterations: result.iterations,
            diagnostics,
        })
    }
    fn record_decision(
        &mut self,
        epoch: u64,
        satellite: Option<visloc_gnss::Satellite>,
        reason: &str,
        pr: Option<f64>,
        rate: Option<f64>,
        use_pr: bool,
        use_rate: bool,
    ) {
        if let Some(s) = satellite {
            self.decisions.insert((epoch, s), (use_pr, use_rate));
        }
        self.record(Record::Decision(serde_json::json!({"epoch_id":epoch,"satellite":satellite,"reason":reason,"pseudorange_sigma":pr,"doppler_sigma":rate,"use_pseudorange":use_pr,"use_doppler":use_rate})));
    }
    fn writeback(&mut self, p: &JointProblem, x: &DVector<f64>) {
        for (id, index) in &p.epochs {
            if self.config.coupling == Coupling::KeyframePreintegration {
                continue;
            }
            let offset = p.base.poses.len() * 6 + index * 15;
            self.hidden
                .insert(*id, decode_state(x.rows(offset, 15).as_slice()));
        }
        for (i, c) in p.clocks.iter().enumerate() {
            let offset = p.clock_offset(i);
            let mut c = c.clone();
            c.clock = Clock {
                bias_m: [x[offset], x[offset + 1]],
                drift_m_s: x[offset + 2],
            };
            self.clocks.insert(c.id, c);
        }
        if let Some(b) = &mut self.bootstrap {
            b.alignment = p.alignment(x);
            self.diagnostics.yaw_rad = Some(b.alignment.yaw_rad);
            if let Some(c) = self.clocks.values().max_by_key(|c| c.time) {
                b.clock = c.clock.clone();
            }
        }
        let mut gate = self.config.clone();
        gate.huber_sigma = f64::MAX;
        for (index, id, o) in &p.observations {
            let (nav, _) = p.epoch_nav(x, *index, *id);
            let c = &self.clocks[id].clock;
            let f = factor::linearize_with_covariance(
                o,
                &body(&nav),
                &p.alignment(x),
                c,
                p.lever,
                &gate,
                p.covariance(*id, o.observation.satellite),
            );
            self.record(Record::Decision(serde_json::json!({"epoch_id":id,"sensor_time_ns":self.clocks[id].time,"solve_timestamp_ns":p.base.states[p.actual_states-1].timestamp_ns,"satellite":o.observation.satellite,"reason":"optimized_residual","pseudorange_sigma":f.residual[0],"doppler_sigma":f.residual[1],"clock":c,"body_position":nav.imu_to_world.translation,"use_pseudorange":o.use_pseudorange,"use_doppler":o.use_doppler})));
        }
    }
    pub(crate) fn marginalize(
        &mut self,
        source: &WindowProblem,
        drop_poses: &[u64],
        drop_states: &[u64],
        convert: &[u64],
    ) -> Result<(), String> {
        let problem = if self.config.coupling == Coupling::KeyframePreintegration {
            let retiring: Vec<_> = drop_states.iter().chain(convert).copied().collect();
            self.build_keyframes(source, Some(&retiring))?
        } else {
            self.build(source)?
        };
        let consumed: BTreeSet<_> = problem.epochs.iter().map(|(id, _)| *id).collect();
        let x = problem.initial();
        let lin = problem
            .linearize(&x)
            .map_err(|e| format!("joint marginalization: {e:?}"))?;
        let projected = projected_rows(&lin.factors, 1e-10);
        let (j, r) = stack_rows(&projected, x.len());
        let boundary = source
            .states
            .last()
            .ok_or("empty marginal window")?
            .timestamp_ns;
        let frontier = problem
            .clocks
            .iter()
            .rfind(|c| {
                self.config.coupling == Coupling::KeyframePreintegration || c.time <= boundary
            })
            .map(|c| c.id);
        let mut keep = Vec::new();
        let mut marginal = Vec::new();
        let mut blocks = Vec::new();
        for b in &problem.blocks {
            let mut b = b.clone();
            let mut n = b.dof;
            let remove = match b.key {
                Key::Pose(id) => drop_poses.contains(&id),
                Key::Nav(id) => {
                    if convert.contains(&id) {
                        b.key = Key::Pose(id);
                        n = 6;
                    }
                    drop_states.contains(&id) || drop_poses.contains(&id)
                }
                Key::Epoch(_) => true,
                Key::Clock(id) => Some(id) != frontier,
                Key::Alignment => false,
            };
            if remove {
                marginal.extend(b.offset..b.offset + b.dof);
            } else {
                keep.extend(b.offset..b.offset + n);
                marginal.extend(b.offset + n..b.offset + b.dof);
                b.fej = b.fej.rows(0, n).into_owned();
                b.offset = blocks.iter().map(|b: &Block| b.dof).sum();
                b.dof = n;
                blocks.push(b);
            }
        }
        let (mut j, r) = compress(&j, &r, &keep, &marginal)?;
        self.record(Record::Decision(serde_json::json!({"reason":"joint_marginalization","timestamp_ns":boundary,"jacobian_norm":j.norm(),"residual_norm":r.norm(),"keep_dof":keep.len(),"marginal_dof":marginal.len(),"rows":j.nrows()})));
        let mut delta = DVector::zeros(keep.len());
        let mut k = 0;
        for b in &blocks {
            let original = problem.find_block(b.key).ok_or("missing marginal key")?;
            let d = block_delta(b.key, &x.rows(original.offset, b.dof).into_owned(), &b.fej);
            delta.rows_mut(k, b.dof).copy_from(&d);
            // The reduced rows are in the current left tangent. Store them in
            // the retained first-estimate chart, including its SO(3) derivative.
            let chart = chart_jacobian(b.key, &d)
                .try_inverse()
                .ok_or("singular prior chart")?;
            let adjusted = j.columns(k, b.dof) * chart;
            j.columns_mut(k, b.dof).copy_from(&adjusted);
            k += b.dof;
        }
        let r = r - &j * delta;
        self.prior = Some(Prior { blocks, j, r });
        self.consumed_before = boundary;
        let offset = (self.bootstrap.as_ref().unwrap().time_offset_s * 1e9).round() as i64;
        if self.config.coupling == Coupling::KeyframePreintegration {
            self.epochs.retain(|e| !consumed.contains(&e.id));
        } else {
            self.epochs.retain(|e| e.time + offset > boundary);
        }
        self.epoch_keyframes
            .retain(|id, _| self.epochs.iter().any(|e| e.id == *id));
        self.motions
            .retain(|id, _| self.epochs.iter().any(|e| e.id == *id));
        self.hidden
            .retain(|id, _| self.epochs.iter().any(|e| e.id == *id));
        self.clocks
            .retain(|id, _| Some(*id) == frontier || self.epochs.iter().any(|e| e.id == *id));
        self.decisions
            .retain(|(id, _), _| self.epochs.iter().any(|e| e.id == *id));
        Ok(())
    }
}

/// Continuous-noise covariance for intervals split at GNSS epochs. The
/// ordinary preintegrator receives per-packet standard deviations, despite
/// their legacy `*_density` field names. Convert them using the calibrated
/// packet period before integrating the position/rotation/velocity process
/// noise. Three-point Gauss-Legendre is exact for the frozen nilpotent dynamics.
fn integrate_split(
    samples: &[ImuSample],
    start: i64,
    end: i64,
    bg: Vector3<f64>,
    ba: Vector3<f64>,
    noise: Option<crate::imu::ImuNoiseModel>,
    nominal_period: f64,
) -> Result<crate::imu::ImuPreintegratedDelta, crate::imu::SamplingError> {
    let mut delta = integrate_between(samples, start, end, bg, ba, None)?;
    if let Some(n) = noise {
        let mut q = nalgebra::SMatrix::<f64, 9, 9>::zeros();
        for axis in 0..3 {
            q[(3 + axis, 3 + axis)] = n.gyro_density.powi(2) * nominal_period;
            q[(6 + axis, 6 + axis)] = n.accel_density.powi(2) * nominal_period;
        }
        let mut previous = 0;
        for trace in &delta.update_trace {
            let dt = (trace.t_ns - previous) as f64 * 1e-9;
            previous = trace.t_ns;
            let mut a = nalgebra::SMatrix::<f64, 9, 9>::zeros();
            a.fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&nalgebra::Matrix3::identity());
            a.fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(trace.f.fixed_view::<3, 3>(6, 3) / dt));
            let mut injection = nalgebra::SMatrix::<f64, 9, 9>::zeros();
            for (fraction, weight) in [
                (0.5 - (3f64 / 5.).sqrt() / 2., 5. / 18.),
                (0.5, 4. / 9.),
                (0.5 + (3f64 / 5.).sqrt() / 2., 5. / 18.),
            ] {
                let t = dt * fraction;
                let phi =
                    nalgebra::SMatrix::<f64, 9, 9>::identity() + a * t + a * a * (0.5 * t * t);
                injection += phi * q * phi.transpose() * (weight * dt);
            }
            delta.covariance = trace.f * delta.covariance * trace.f.transpose() + injection;
        }
        delta.covariance = (delta.covariance + delta.covariance.transpose()) * 0.5;
    }
    Ok(delta)
}
fn predict(
    a: &BasaltNavState,
    d: &crate::imu::ImuPreintegratedDelta,
    g: Vector3<f64>,
) -> BasaltNavState {
    let dt = d.delta_time;
    let mut b = a.clone();
    b.imu_to_world.translation += a.velocity_world_m_s * dt
        + g * (0.5 * dt * dt)
        + a.imu_to_world.rotation * d.delta_position;
    b.imu_to_world.rotation *= d.delta_rotation;
    b.velocity_world_m_s += g * dt + a.imu_to_world.rotation * d.delta_velocity;
    b
}
#[allow(clippy::too_many_arguments)]
fn measurement_covariance(
    m: &EpochMotion,
    anchor: &BasaltNavState,
    o: &PreparedObservation,
    body: &Body,
    alignment: &Alignment,
    clock: &Clock,
    lever: Vector3<f64>,
    timing_std: f64,
) -> Matrix2<f64> {
    let raw = factor::raw_linearize(o, body, alignment, clock, lever);
    let j = raw.jacobian.fixed_columns::<15>(0);
    let (nav, _) = m.evaluate(anchor);
    let time = j * m.time_derivative(&nav);
    let q = j * m.covariance(anchor) * j.transpose() + time * time.transpose() * timing_std.powi(2);
    (q + q.transpose()) * 0.5
}
fn body(n: &BasaltNavState) -> Body {
    Body {
        position: n.imu_to_world.translation,
        rotation: n.imu_to_world.rotation,
        velocity: n.velocity_world_m_s,
        gyro_bias: n.gyro_bias_rad_s,
    }
}
struct JointProblem {
    base: WindowProblem,
    actual_states: usize,
    epochs: Vec<(u64, usize)>,
    clocks: Vec<ClockEpoch>,
    observations: Vec<(usize, u64, PreparedObservation)>,
    motions: BTreeMap<u64, EpochMotion>,
    covariances: BTreeMap<(u64, visloc_gnss::Satellite), Matrix2<f64>>,
    alignment_seed: Alignment,
    config: Config,
    lever: Vector3<f64>,
    prior: Option<Prior>,
    blocks: Vec<Block>,
    factor_shape: Cell<(usize, usize)>,
    trial_token: RefCell<Option<LmTrialToken>>,
}
impl JointProblem {
    fn new(
        base: WindowProblem,
        actual: usize,
        epochs: Vec<(u64, usize)>,
        clocks: Vec<ClockEpoch>,
        observations: Vec<(usize, u64, PreparedObservation)>,
        alignment: Alignment,
        config: Config,
        lever: Vector3<f64>,
        prior: Option<Prior>,
    ) -> Result<Self, String> {
        let mut blocks = Vec::new();
        let mut offset = 0;
        for p in &base.poses {
            blocks.push(Block {
                key: Key::Pose(p.frame_id),
                offset,
                dof: 6,
                fej: crate::vio::window::flatten_pose(&p.linearized_pose),
            });
            offset += 6;
        }
        for (i, n) in base.states.iter().enumerate() {
            let key = if i < actual {
                Key::Nav(n.frame_id)
            } else {
                Key::Epoch(
                    epochs
                        .iter()
                        .find(|(_, index)| *index == i)
                        .ok_or("missing GNSS identity")?
                        .0,
                )
            };
            blocks.push(Block {
                key,
                offset,
                dof: 15,
                fej: flatten_nav(&n.linearized_nav),
            });
            offset += 15;
        }
        blocks.push(Block {
            key: Key::Alignment,
            offset,
            dof: 4,
            fej: DVector::from_vec(vec![
                alignment.translation_enu_m.x,
                alignment.translation_enu_m.y,
                alignment.translation_enu_m.z,
                alignment.yaw_rad,
            ]),
        });
        offset += 4;
        for c in &clocks {
            blocks.push(Block {
                key: Key::Clock(c.id),
                offset,
                dof: 3,
                fej: DVector::from_vec(vec![
                    c.clock.bias_m[0],
                    c.clock.bias_m[1],
                    c.clock.drift_m_s,
                ]),
            });
            offset += 3;
        }
        // A block's first estimate stays fixed for its lifetime in the joint prior.
        if let Some(p) = &prior {
            for b in &mut blocks {
                if let Some(old) = p.blocks.iter().find(|old| old.key == b.key) {
                    b.fej = old.fej.clone();
                }
            }
        }
        Ok(Self {
            base,
            actual_states: actual,
            epochs,
            clocks,
            observations,
            motions: BTreeMap::new(),
            covariances: BTreeMap::new(),
            alignment_seed: alignment,
            config,
            lever,
            prior,
            blocks,
            factor_shape: Cell::new((0, 0)),
            trial_token: RefCell::new(None),
        })
    }
    fn find_block(&self, key: Key) -> Option<&Block> {
        self.blocks.iter().find(|b| b.key == key).or_else(|| {
            if let Key::Pose(id) = key {
                self.blocks.iter().find(|b| b.key == Key::Nav(id))
            } else {
                None
            }
        })
    }
    fn initial(&self) -> DVector<f64> {
        let mut x = DVector::zeros(self.base.state_dof() + 4 + self.clocks.len() * 3);
        x.rows_mut(0, self.base.state_dof())
            .copy_from(&self.base.initial_state());
        let o = self.base.state_dof();
        x.rows_mut(o, 3)
            .copy_from(&self.alignment_seed.translation_enu_m);
        x[o + 3] = self.alignment_seed.yaw_rad;
        for (i, c) in self.clocks.iter().enumerate() {
            let o = self.clock_offset(i);
            x[o] = c.clock.bias_m[0];
            x[o + 1] = c.clock.bias_m[1];
            x[o + 2] = c.clock.drift_m_s;
        }
        x
    }
    fn alignment(&self, x: &DVector<f64>) -> Alignment {
        let o = self.base.state_dof();
        Alignment {
            origin_ecef_m: self.alignment_seed.origin_ecef_m,
            translation_enu_m: Vector3::new(x[o], x[o + 1], x[o + 2]),
            yaw_rad: x[o + 3],
        }
    }
    fn clock_offset(&self, i: usize) -> usize {
        self.base.state_dof() + 4 + i * 3
    }
    fn epoch_nav(&self, x: &DVector<f64>, index: usize, id: u64) -> (BasaltNavState, Matrix15) {
        let nav = decode_state(
            x.rows(self.base.poses.len() * 6 + index * 15, 15)
                .as_slice(),
        );
        self.motions
            .get(&id)
            .map_or_else(|| (nav.clone(), Matrix15::identity()), |m| m.evaluate(&nav))
    }
    fn covariance(&self, id: u64, satellite: visloc_gnss::Satellite) -> Matrix2<f64> {
        self.covariances
            .get(&(id, satellite))
            .copied()
            .unwrap_or_default()
    }
    fn extra(&self, x: &DVector<f64>) -> Result<Vec<WhitenedFactorRowStack>, LmFailure> {
        let mut factors = Vec::new();
        let n = x.len();
        if let Some(p) = &self.prior {
            let mut j = DMatrix::zeros(p.j.nrows(), n);
            let mut delta = DVector::zeros(p.j.ncols());
            let mut col = 0;
            for b in &p.blocks {
                let current = self.find_block(b.key).ok_or(LmFailure::LinearSolve)?;
                let d = block_delta(b.key, &x.rows(current.offset, b.dof).into_owned(), &b.fej);
                let local_j = p.j.columns(col, b.dof) * chart_jacobian(b.key, &d);
                j.view_mut((0, current.offset), (p.j.nrows(), b.dof))
                    .copy_from(&local_j);
                delta.rows_mut(col, b.dof).copy_from(&d);
                col += b.dof;
            }
            factors.push(
                WhitenedFactorRowStack::new(j, DMatrix::zeros(p.j.nrows(), 0), &p.j * delta + &p.r)
                    .ok_or(LmFailure::NonFinite)?,
            );
        }
        for (index, id, o) in &self.observations {
            let offset = self.base.poses.len() * 6 + index * 15;
            let (nav, transition) = self.epoch_nav(x, *index, *id);
            let clock_index = self
                .clocks
                .iter()
                .position(|c| c.id == *id)
                .ok_or(LmFailure::LinearSolve)?;
            let c = self.clock_offset(clock_index);
            let clock = Clock {
                bias_m: [x[c], x[c + 1]],
                drift_m_s: x[c + 2],
            };
            let lin = factor::linearize_with_covariance(
                o,
                &body(&nav),
                &self.alignment(x),
                &clock,
                self.lever,
                &self.config,
                self.covariance(*id, o.observation.satellite),
            );
            let mut j = DMatrix::zeros(2, n);
            j.view_mut((0, offset), (2, 15))
                .copy_from(&(lin.jacobian.fixed_columns::<15>(0) * transition));
            j.view_mut((0, self.base.state_dof()), (2, 4))
                .copy_from(&lin.jacobian.columns(15, 4));
            if self.config.mode == Mode::DopplerOnly {
                j.columns_mut(self.base.state_dof(), 3).fill(0.);
            }
            j.view_mut((0, c), (2, 3))
                .copy_from(&lin.jacobian.columns(19, 3));
            factors.push(
                WhitenedFactorRowStack::with_objective_cost(
                    j,
                    DMatrix::zeros(2, 0),
                    DVector::from_column_slice(lin.residual.as_slice()),
                    lin.cost,
                )
                .ok_or(LmFailure::NonFinite)?,
            );
        }
        // A conservative bootstrap clock prior fixes Doppler-only clock
        // bias gauges. Its RAWX support predates the active factor window.
        if !self
            .prior
            .as_ref()
            .is_some_and(|p| p.blocks.iter().any(|b| matches!(b.key, Key::Clock(_))))
            && !self.clocks.is_empty()
        {
            let mut j = DMatrix::zeros(3, n);
            let mut r = DVector::zeros(3);
            let c = &self.clocks[0];
            let o = self.clock_offset(0);
            for (row, (seed, sigma)) in [
                (c.clock.bias_m[0], 30.),
                (c.clock.bias_m[1], 30.),
                (c.clock.drift_m_s, 3.),
            ]
            .into_iter()
            .enumerate()
            {
                j[(row, o + row)] = 1. / sigma;
                r[row] = (x[o + row] - seed) / sigma;
            }
            factors.push(
                WhitenedFactorRowStack::new(j, DMatrix::zeros(3, 0), r)
                    .ok_or(LmFailure::NonFinite)?,
            );
        }
        for (i, pair) in self.clocks.windows(2).enumerate() {
            let (a, b) = (&pair[0], &pair[1]);
            if b.reset {
                continue;
            }
            let dt = (b.time - a.time) as f64 * 1e-9;
            if dt <= 0. {
                return Err(LmFailure::LinearSolve);
            }
            let (oa, ob) = (self.clock_offset(i), self.clock_offset(i + 1));
            let mut j = DMatrix::zeros(3, n);
            let mut r = DVector::zeros(3);
            for row in 0..2 {
                let sigma = self.config.clock_bias_walk_m_sqrt_s * dt.sqrt();
                j[(row, ob + row)] = 1. / sigma;
                j[(row, oa + row)] = -1. / sigma;
                j[(row, oa + 2)] = -0.5 * dt / sigma;
                j[(row, ob + 2)] = -0.5 * dt / sigma;
                r[row] = (x[ob + row] - x[oa + row] - 0.5 * (x[oa + 2] + x[ob + 2]) * dt) / sigma;
            }
            let sigma = self.config.clock_drift_walk_m_s_sqrt_s * dt.sqrt();
            j[(2, ob + 2)] = 1. / sigma;
            j[(2, oa + 2)] = -1. / sigma;
            r[2] = (x[ob + 2] - x[oa + 2]) / sigma;
            factors.push(
                WhitenedFactorRowStack::new(j, DMatrix::zeros(3, 0), r)
                    .ok_or(LmFailure::NonFinite)?,
            );
        }
        Ok(factors)
    }
}
impl LmProblem for JointProblem {
    fn reduce_f64(
        &self,
        factors: &[WhitenedFactorRowStack],
        state_dof: usize,
        tolerance: f64,
    ) -> ReducedNormalSystem {
        reduce_supported_columns(factors, state_dof, tolerance)
    }
    fn linearize(&self, x: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
        let n = self.base.state_dof();
        let mut lin = self.base.linearize(&x.rows(0, n).into_owned())?;
        for f in &mut lin.factors {
            let old = f.state_jacobian.clone();
            f.state_jacobian = DMatrix::zeros(old.nrows(), x.len());
            f.state_jacobian.columns_mut(0, n).copy_from(&old);
        }
        let extra = self.extra(x)?;
        lin.cost += extra.iter().map(|f| f.objective_cost).sum::<f64>();
        lin.factors.extend(extra);
        self.factor_shape.set((
            lin.factors.len(),
            lin.factors.iter().map(|f| f.rows()).sum(),
        ));
        Ok(lin)
    }
    fn cost(&self, x: &DVector<f64>) -> Result<f64, LmFailure> {
        Ok(self.linearize(x)?.cost)
    }
    fn apply_step(&self, x: &DVector<f64>, d: &DVector<f64>) -> DVector<f64> {
        let n = self.base.state_dof();
        let mut trial = x + d;
        if self.config.mode == Mode::DopplerOnly {
            trial.rows_mut(n, 3).copy_from(&x.rows(n, 3));
        }
        trial.rows_mut(0, n).copy_from(
            &self
                .base
                .apply_step(&x.rows(0, n).into_owned(), &d.rows(0, n).into_owned()),
        );
        trial
    }
    fn trial_cost(
        &self,
        x: &DVector<f64>,
        d: &DVector<f64>,
        trial: &DVector<f64>,
    ) -> Result<f64, LmFailure> {
        let n = self.base.state_dof();
        let (cost, token) = self.base.trial_cost_timed_with_token(
            &x.rows(0, n).into_owned(),
            &d.rows(0, n).into_owned(),
            &trial.rows(0, n).into_owned(),
            &mut TimingBreakdown::default(),
        )?;
        self.trial_token.replace(Some(token));
        Ok(cost
            + self
                .extra(trial)?
                .iter()
                .map(|f| f.objective_cost)
                .sum::<f64>())
    }
    fn accept_step(&mut self, x: &DVector<f64>, d: &DVector<f64>) -> Result<(), LmFailure> {
        let n = self.base.state_dof();
        self.base.accept_step_with_token(
            &x.rows(0, n).into_owned(),
            &d.rows(0, n).into_owned(),
            self.trial_token.take().unwrap_or_default(),
        )
    }
}

/// Exact structural sparsity: no thresholding of small Jacobian entries.
/// QR eliminates the same landmark variables, then the small normal system is
/// scattered into the joint layout. Back-substitution retains the full rows.
fn reduce_supported_columns(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> ReducedNormalSystem {
    let mut h = DMatrix::zeros(state_dof, state_dof);
    let mut b = DVector::zeros(state_dof);
    let mut back = Vec::with_capacity(factors.len());
    for f in factors {
        let columns: Vec<_> = (0..state_dof)
            .filter(|&col| f.state_jacobian.column(col).iter().any(|&v| v != 0.))
            .collect();
        let compact = WhitenedFactorRowStack::with_objective_cost_kind(
            DMatrix::from_fn(f.rows(), columns.len(), |row, col| {
                f.state_jacobian[(row, columns[col])]
            }),
            f.landmark_jacobian.clone(),
            f.residual.clone(),
            f.objective_cost,
            f.kind,
        )
        .expect("existing factor has valid dimensions");
        let (j, r, rank) = landmark_nullspace_projection(&compact, tolerance);
        let local_h = j.transpose() * &j;
        let local_b = j.transpose() * &r;
        for (i, &col) in columns.iter().enumerate() {
            b[col] += local_b[i];
            for (k, &other) in columns.iter().enumerate() {
                h[(col, other)] += local_h[(i, k)];
            }
        }
        back.push(LandmarkBackSubstitution {
            state_jacobian: f.state_jacobian.clone(),
            landmark_jacobian: f.landmark_jacobian.clone(),
            residual: f.residual.clone(),
            rank,
        });
    }
    ReducedNormalSystem {
        h,
        b,
        back_substitution: back,
        diagnostic_stages: None,
    }
}
fn block_delta(key: Key, x: &DVector<f64>, point: &DVector<f64>) -> DVector<f64> {
    let mut d = x - point;
    if matches!(key, Key::Pose(_) | Key::Nav(_) | Key::Epoch(_)) {
        let r = UnitQuaternion::from_scaled_axis(Vector3::new(x[3], x[4], x[5]));
        let r0 = UnitQuaternion::from_scaled_axis(Vector3::new(point[3], point[4], point[5]));
        d.rows_mut(3, 3)
            .copy_from(&(r * r0.inverse()).scaled_axis());
    }
    d
}
fn chart_jacobian(key: Key, delta: &DVector<f64>) -> DMatrix<f64> {
    let mut j = DMatrix::identity(delta.len(), delta.len());
    if matches!(key, Key::Pose(_) | Key::Nav(_) | Key::Epoch(_)) {
        let phi = Vector3::new(delta[3], delta[4], delta[5]);
        let theta = phi.norm();
        let k = nalgebra::Matrix3::new(0., -phi.z, phi.y, phi.z, 0., -phi.x, -phi.y, phi.x, 0.);
        let a = if theta < 1e-5 {
            1. / 12. + theta * theta / 720.
        } else {
            (1. - 0.5 * theta / (0.5 * theta).tan()) / (theta * theta)
        };
        j.view_mut((3, 3), (3, 3))
            .copy_from(&(nalgebra::Matrix3::identity() - 0.5 * k + a * k * k));
    }
    j
}
/// Rank-aware square-root elimination using orthogonal projection. No normal
/// equations or invented information in unobservable nuisance directions.
fn compress(
    j: &DMatrix<f64>,
    r: &DVector<f64>,
    keep: &[usize],
    marginal: &[usize],
) -> Result<(DMatrix<f64>, DVector<f64>), String> {
    let mut a = DMatrix::from_fn(j.nrows(), keep.len(), |row, col| j[(row, keep[col])]);
    let mut b = r.clone();
    let mut basis: Vec<DVector<f64>> = Vec::new();
    let scale = j.norm().max(1.);
    for &col in marginal {
        let mut v = j.column(col).into_owned();
        for _ in 0..2 {
            for q in &basis {
                v -= q * q.dot(&v);
            }
        }
        let norm = v.norm();
        if norm > 1e-12 * scale {
            basis.push(v / norm);
        }
    }
    for q in basis {
        let products = q.transpose() * &a;
        a -= &q * products;
        let v = q.dot(&b);
        b -= q * v;
    }
    let qr = a.qr();
    let out_j = qr.r();
    let out_r = qr.q().transpose() * b;
    if out_j.iter().chain(out_r.iter()).any(|x| !x.is_finite()) {
        return Err("nonfinite joint prior".into());
    }
    Ok((out_j, out_r))
}
fn empty_diagnostics() -> WindowDiagnostics {
    WindowDiagnostics {
        attempted: false,
        state_dof: 0,
        factor_count: 0,
        factor_rows: 0,
        landmark_count: 0,
        imu_link_count: 0,
        prior_rows: 0,
        prior_factor_rows: 0,
        visual_factor_rows: 0,
        imu_factor_rows: 0,
        bias_factor_rows: 0,
        prior_cost: 0.,
        visual_cost: 0.,
        imu_cost: 0.,
        bias_cost: 0.,
        imu_links: Vec::new(),
        lm: Vec::new(),
        status: "gnss".into(),
        failure: None,
        state_writeback: false,
        landmark_writeback: 0,
        prior_carry: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vio::window::local_state_difference;
    use crate::{
        camera::DoubleSphereCamera,
        imu::{BiasRandomWalkNoise, ImuNoiseModel},
        vio::scalar::ScalarMode,
    };
    #[test]
    fn histories_stay_bounded_without_camera_processing() {
        let cfg = Config {
            enabled: true,
            max_queue_epochs: 1,
            ..Config::default()
        };
        let mut runtime =
            GnssRuntime::new(cfg, &UnitQuaternion::identity(), Vector3::zeros()).unwrap();
        for i in (0..100).rev() {
            runtime.store_solution(Solution {
                utc_ns: i * 1_000_000_000,
                ecef_m: Vector3::zeros(),
                velocity_ecef_m_s: Vector3::zeros(),
                clock: Clock::default(),
                code_rms_m: 0.,
                rate_rms_m_s: 0.,
                satellites: 4,
            });
            runtime.push_epoch(Epoch {
                week: 2400,
                tow_s: f64::NAN,
                leap_seconds: 18,
                leap_seconds_valid: true,
                clock_reset: false,
                receipt_ns: 0,
                observations: Vec::new(),
            });
        }
        assert_eq!(runtime.solutions.len(), 16);
        assert_eq!(runtime.solutions.first().unwrap().utc_ns, 84_000_000_000);
        assert_eq!(runtime.records.len(), 64);
        assert_eq!(runtime.diagnostics.dropped_records, 36);
        runtime.store_solution(Solution {
            utc_ns: 300_000_000_000,
            ecef_m: Vector3::zeros(),
            velocity_ecef_m_s: Vector3::zeros(),
            clock: Clock::default(),
            code_rms_m: 0.,
            rate_rms_m_s: 0.,
            satellites: 4,
        });
        assert_eq!(runtime.solutions.len(), 1);
    }
    #[test]
    fn invalid_epoch_time_is_rejected_before_navigation_bootstrap() {
        let cfg = Config {
            enabled: true,
            ..Config::default()
        };
        let mut runtime =
            GnssRuntime::new(cfg, &UnitQuaternion::identity(), Vector3::zeros()).unwrap();
        runtime.push_epoch(Epoch {
            week: 2400,
            tow_s: f64::NAN,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: Vec::new(),
        });
        assert!(runtime.epochs.is_empty());
        assert!(runtime.solutions.is_empty());
        assert_eq!(runtime.diagnostics.status, "invalid_gnss_time");
        assert!(
            matches!(runtime.records.back(), Some(Record::Decision(v)) if v["reason"]=="invalid_gnss_time")
        );
    }
    #[test]
    fn clock_outage_weights_grow_and_reset_breaks_the_previous_clock_link() {
        let (p, _) = source(0);
        let alignment = Alignment {
            origin_ecef_m: Vector3::new(6378137., 0., 0.),
            yaw_rad: 0.,
            translation_enu_m: Vector3::zeros(),
        };
        let first = ClockEpoch {
            id: 10,
            time: 0,
            clock: Clock {
                bias_m: [100., 200.],
                drift_m_s: 5.,
            },
            reset: false,
        };
        let second = ClockEpoch {
            id: 11,
            time: 4_000_000_000,
            clock: Clock {
                bias_m: [120., 220.],
                drift_m_s: 5.,
            },
            reset: false,
        };
        let build = |reset| {
            let mut second = second.clone();
            second.reset = reset;
            JointProblem::new(
                p.clone(),
                p.states.len(),
                Vec::new(),
                vec![first.clone(), second],
                Vec::new(),
                alignment.clone(),
                Config::default(),
                Vector3::zeros(),
                None,
            )
            .unwrap()
        };
        let joint = build(false);
        let mut x = joint.initial();
        x[joint.clock_offset(1)] += 10.;
        let rows = joint.extra(&x).unwrap();
        assert_eq!(rows.len(), 2);
        assert!((rows[1].residual[0] - 0.5).abs() < 1e-12);
        assert!((rows[1].state_jacobian[(2, joint.clock_offset(1) + 2)] - 0.5).abs() < 1e-12);
        let reset = build(true);
        assert_eq!(reset.extra(&reset.initial()).unwrap().len(), 1);
    }
    #[test]
    fn compact_joint_reduction_matches_dense_with_landmarks_and_null_columns() {
        let factors: Vec<_> = (0..12)
            .map(|i| {
                let rows = 18;
                let j = DMatrix::from_fn(rows, 160, |r, c| {
                    if (i * 9..i * 9 + 12).contains(&c) {
                        ((r * 17 + c * 3 + i) as f64).sin()
                    } else {
                        0.
                    }
                });
                let l = DMatrix::from_fn(rows, if i % 3 == 0 { 0 } else { 3 }, |r, c| {
                    ((r * 3 + c * 7 + i) as f64).cos()
                });
                WhitenedFactorRowStack::new(j, l, DVector::from_fn(rows, |r, _| (r as f64).sin()))
                    .unwrap()
            })
            .collect();
        let dense = crate::vio::aom::reduce_landmark_factors(&factors, 160, 1e-10);
        let compact = reduce_supported_columns(&factors, 160, 1e-10);
        assert!((&dense.h - &compact.h).norm() < 1e-11);
        assert!((&dense.b - &compact.b).norm() < 1e-11);
        assert_eq!(dense.back_substitution, compact.back_substitution);
    }
    #[test]
    fn rank_deficient_joint_prior_preserves_linearized_objective() {
        let j = DMatrix::from_row_slice(
            5,
            4,
            &[
                1., 2., 1., 2., 0., 0., 2., 1., 2., 4., 0., 2., 3., 6., 1., 0., 0., 0., 3., -1.,
            ],
        );
        let r = DVector::from_vec(vec![0.1, -0.4, 0.2, 0.6, -0.2]);
        let (a, b) = compress(&j, &r, &[2, 3], &[0, 1]).unwrap();
        for delta in [
            DVector::from_vec(vec![0.2, -0.1]),
            DVector::from_vec(vec![-0.3, 0.7]),
        ] {
            let jm = j.columns(0, 2).into_owned();
            let target = &r + j.columns(2, 2) * &delta;
            let nuisance = jm
                .clone()
                .svd(true, true)
                .solve(&(-&target), 1e-10)
                .unwrap();
            let explicit = (&target + jm * nuisance).norm_squared();
            let jm = j.columns(0, 2).into_owned();
            let at_zero = jm.clone().svd(true, true).solve(&(-&r), 1e-10).unwrap();
            let zero = (&r + jm * at_zero).norm_squared();
            assert!(
                (explicit - zero - ((&a * delta + &b).norm_squared() - b.norm_squared())).abs()
                    < 1e-10
            );
        }
    }
    fn state(id: u64, time: i64, nav: BasaltNavState) -> WindowState {
        WindowState {
            frame_id: id,
            timestamp_ns: time,
            nav: nav.clone(),
            stored_current_nav: nav.clone(),
            linearized_nav: nav,
            linearized_delta: DVector::zeros(15),
            is_keyframe: false,
            is_latest: false,
            linearized: false,
        }
    }
    fn source(start: i64) -> (WindowProblem, Vec<ImuSample>) {
        let samples: Vec<_> = (0..=100)
            .map(|i| {
                ImuSample::new(
                    start + i * 1_000_000,
                    Vector3::new(0., 0., 0.3),
                    Vector3::new(0., 0., 9.81),
                )
            })
            .collect();
        let noise = ImuNoiseModel {
            gyro_density: 0.01,
            accel_density: 0.1,
        };
        let delta = integrate_between(
            &samples,
            start,
            start + 100_000_000,
            Vector3::zeros(),
            Vector3::zeros(),
            Some(noise),
        )
        .unwrap();
        let nav = BasaltNavState::default();
        let second = predict(&nav, &delta, Vector3::new(0., 0., -9.81));
        let camera = DoubleSphereCamera::new(300., 300., 320., 240., 0., 0.5, 640, 480).unwrap();
        (
            WindowProblem {
                trial_host_order: Vec::new(),
                camera,
                cameras: vec![camera],
                t_imu_cam: vec![Default::default()],
                poses: Vec::new(),
                states: vec![
                    state(u64::MAX, start, nav),
                    state(1, start + 100_000_000, second),
                ],
                landmarks: Vec::new(),
                imu_links: vec![WindowImuLink {
                    from_index: 0,
                    to_index: 1,
                    delta,
                }],
                imu_noise: noise,
                bias_walk_noise: BiasRandomWalkNoise {
                    gyro_density: 0.01,
                    accel_density: 0.01,
                },
                initial_pose_weight: 1e4,
                initial_accel_bias_weight: 10.,
                initial_gyro_bias_weight: 100.,
                prior: None,
                anchor_point: Some(DVector::zeros(15)),
                gravity_world: Vector3::new(0., 0., -9.81),
                scalar_mode: ScalarMode::ExtendedF64,
            },
            samples,
        )
    }
    #[test]
    fn sub_packet_covariance_is_full_rank_and_composes_continuous_noise() {
        let noise = crate::imu::ImuNoiseModel {
            gyro_density: 0.01,
            accel_density: 0.1,
        };
        let samples = vec![
            ImuSample::new(0, Vector3::zeros(), Vector3::zeros()),
            ImuSample::new(10_000_000, Vector3::zeros(), Vector3::zeros()),
        ];
        let full = integrate_split(
            &samples,
            0,
            10_000_000,
            Vector3::zeros(),
            Vector3::zeros(),
            Some(noise),
            0.01,
        )
        .unwrap();
        let first = integrate_split(
            &samples,
            0,
            1_000_000,
            Vector3::zeros(),
            Vector3::zeros(),
            Some(noise),
            0.01,
        )
        .unwrap();
        let second = integrate_split(
            &samples,
            1_000_000,
            10_000_000,
            Vector3::zeros(),
            Vector3::zeros(),
            Some(noise),
            0.01,
        )
        .unwrap();
        assert!(first.covariance.symmetric_eigen().eigenvalues.min() > 1e-16);
        let mut f = nalgebra::SMatrix::<f64, 9, 9>::identity();
        f.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(nalgebra::Matrix3::identity() * 0.009));
        let combined = f * first.covariance * f.transpose() + second.covariance;
        assert!((combined - full.covariance).norm() < 1e-15);
        let w = crate::imu::sqrt_information(&first.covariance).unwrap();
        assert!(w.iter().all(|v| v.is_finite() && v.abs() < 2e7));
    }
    #[test]
    fn retained_rotation_chart_has_correct_left_tangent_jacobian() {
        let point = DVector::from_vec(vec![1., 2., 3., 0.3, -0.2, 0.1]);
        let x = DVector::from_vec(vec![1.1, 1.9, 3.2, 0.7, 0.1, -0.2]);
        let delta = block_delta(Key::Pose(4), &x, &point);
        let j = chart_jacobian(Key::Pose(4), &delta);
        let r = UnitQuaternion::from_scaled_axis(Vector3::new(x[3], x[4], x[5]));
        for col in 0..6 {
            let eval = |h: f64| {
                let mut y = x.clone();
                if col < 3 {
                    y[col] += h;
                } else {
                    let mut d = Vector3::zeros();
                    d[col - 3] = h;
                    let rotation = (UnitQuaternion::from_scaled_axis(d) * r).scaled_axis();
                    y.rows_mut(3, 3).copy_from(&rotation);
                }
                block_delta(Key::Pose(4), &y, &point)
            };
            assert!(((eval(1e-6) - eval(-1e-6)) * 5e5 - j.column(col)).norm() < 1e-8);
        }
    }
    #[test]
    fn epoch_queue_is_sorted_deduplicated_bounded_and_reset_is_counted() {
        let mut cfg = Config::default();
        cfg.enabled = true;
        cfg.max_queue_epochs = 2;
        let mut runtime =
            GnssRuntime::new(cfg, &UnitQuaternion::identity(), Vector3::zeros()).unwrap();
        let mut e = Epoch {
            week: 2400,
            tow_s: 10.,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: Vec::new(),
        };
        runtime.push_epoch(e.clone());
        runtime.push_epoch(e.clone());
        e.tow_s = 9.;
        runtime.push_epoch(e.clone());
        e.tow_s = 11.;
        e.clock_reset = true;
        runtime.push_epoch(e);
        assert_eq!(runtime.epochs.len(), 2);
        assert_eq!(runtime.diagnostics.dropped_epochs, 1);
        assert_eq!(runtime.diagnostics.clock_resets, 1);
        assert_eq!(runtime.epochs[0].epoch.tow_s, 10.);
        assert_eq!(runtime.epochs[1].epoch.tow_s, 11.);
    }
    #[test]
    fn epoch_states_replace_original_imu_and_preserve_camera_identity() {
        let epoch = Epoch {
            week: 2400,
            tow_s: 123.45,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: Vec::new(),
        };
        let start = epoch.nominal_utc_ns().unwrap() - 50_000_000;
        let (p, imu) = source(start);
        let mut cfg = Config::default();
        cfg.enabled = true;
        cfg.coupling = Coupling::FrameWindow;
        let mut runtime =
            GnssRuntime::new(cfg, &UnitQuaternion::identity(), Vector3::zeros()).unwrap();
        runtime.record_imu(&imu);
        runtime.bootstrap = Some(Bootstrap {
            alignment: Alignment {
                origin_ecef_m: Vector3::new(6378137., 0., 0.),
                yaw_rad: 0.,
                translation_enu_m: Vector3::zeros(),
            },
            time_offset_s: 0.,
            timing_std_s: 0.,
            clock: Clock::default(),
        });
        runtime.push_epoch(epoch);
        let joint = runtime.build(&p).unwrap();
        assert_eq!(joint.actual_states, 2);
        assert_eq!(joint.base.states.len(), 3);
        assert_eq!(joint.base.imu_links.len(), 2);
        assert_ne!(joint.base.states[2].frame_id, u64::MAX);
        assert_eq!(joint.base.states[0].frame_id, u64::MAX);
        assert_eq!(joint.base.states[1].frame_id, 1);
        assert!(
            (joint
                .base
                .imu_links
                .iter()
                .map(|l| l.delta.delta_time)
                .sum::<f64>()
                - 0.1)
                .abs()
                < 1e-12
        );
        assert!(joint
            .base
            .imu_links
            .iter()
            .all(|l| !(l.from_index == 0 && l.to_index == 1)));
        let mut predicted = p.states[0].nav.clone();
        for link in &joint.base.imu_links {
            predicted = predict(&predicted, &link.delta, p.gravity_world);
        }
        assert!(local_state_difference(&predicted, &p.states[1].nav).norm() < 1e-10);
        runtime.marginalize(&p, &[], &[u64::MAX], &[]).unwrap();
        assert!(runtime
            .prior
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .any(|b| b.key == Key::Nav(1)));
        assert!(runtime
            .prior
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .all(|b| !matches!(b.key, Key::Epoch(_))));
    }
    #[test]
    fn keyframe_epochs_keep_original_imu_and_are_consumed_only_with_their_owner() {
        let epoch = Epoch {
            week: 2400,
            tow_s: 123.45,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: Vec::new(),
        };
        let start = epoch.nominal_utc_ns().unwrap() - 50_000_000;
        let (mut p, imu) = source(start);
        p.states[0].is_keyframe = true;
        let mut runtime = GnssRuntime::new(
            Config {
                enabled: true,
                ..Default::default()
            },
            &UnitQuaternion::identity(),
            Vector3::zeros(),
        )
        .unwrap();
        runtime.record_imu(&imu);
        runtime.bootstrap = Some(Bootstrap {
            alignment: Alignment {
                origin_ecef_m: Vector3::new(6378137., 0., 0.),
                yaw_rad: 0.,
                translation_enu_m: Vector3::zeros(),
            },
            time_offset_s: 0.,
            timing_std_s: 0.02,
            clock: Clock::default(),
        });
        runtime.push_epoch(epoch.clone());
        let joint = runtime.build(&p).unwrap();
        assert_eq!(joint.base.states.len(), p.states.len());
        assert_eq!(joint.base.imu_links, p.imu_links);
        assert_eq!(joint.epochs, vec![(0, 0)]);
        assert!(joint.blocks.iter().all(|b| !matches!(b.key, Key::Epoch(_))));
        assert!(joint.motions.contains_key(&0));
        runtime.marginalize(&p, &[], &[1], &[]).unwrap();
        assert_eq!(runtime.epochs.len(), 1);
        assert!(runtime
            .prior
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .all(|b| !matches!(b.key, Key::Clock(_))));
        runtime.marginalize(&p, &[], &[], &[u64::MAX]).unwrap();
        assert!(runtime.epochs.is_empty());
        let prior = runtime.prior.as_ref().unwrap();
        assert!(prior
            .blocks
            .iter()
            .any(|b| b.key == Key::Pose(u64::MAX) && b.dof == 6));
        assert!(prior.blocks.iter().any(|b| b.key == Key::Clock(0)));
        runtime.push_epoch(epoch);
        assert!(
            runtime.epochs.is_empty(),
            "consumed epochs must remain deduplicated"
        );
    }
    #[test]
    fn new_mode_waits_for_actual_keyframe_velocity_and_window_only_has_no_gnss_rows() {
        let epoch = Epoch {
            week: 2400,
            tow_s: 123.45,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: Vec::new(),
        };
        let start = epoch.nominal_utc_ns().unwrap() - 50_000_000;
        let (p, imu) = source(start);
        let mut runtime = GnssRuntime::new(
            Config {
                enabled: true,
                ..Default::default()
            },
            &UnitQuaternion::identity(),
            Vector3::zeros(),
        )
        .unwrap();
        runtime.record_imu(&imu);
        runtime.bootstrap = Some(Bootstrap {
            alignment: Alignment {
                origin_ecef_m: Vector3::new(6378137., 0., 0.),
                yaw_rad: 0.,
                translation_enu_m: Vector3::zeros(),
            },
            time_offset_s: 0.,
            timing_std_s: 0.02,
            clock: Clock::default(),
        });
        runtime.push_epoch(epoch);
        assert!(runtime.build(&p).unwrap().epochs.is_empty());
        assert_eq!(runtime.epochs.len(), 1);
        runtime.config.mode = Mode::WindowOnly;
        let joint = runtime.build(&p).unwrap();
        assert!(joint.epochs.is_empty());
        assert!(joint.clocks.is_empty());
        assert!(joint.observations.is_empty());
        assert!(runtime.epochs.is_empty());
    }
    #[test]
    fn startup_backlog_is_not_replayed_as_long_backward_keyframe_factors() {
        let epoch = Epoch {
            week: 2400,
            tow_s: 123.45,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: Vec::new(),
        };
        let (p, imu) = source(epoch.nominal_utc_ns().unwrap() + 1_000_000_000);
        let mut runtime = GnssRuntime::new(
            Config {
                enabled: true,
                ..Default::default()
            },
            &UnitQuaternion::identity(),
            Vector3::zeros(),
        )
        .unwrap();
        runtime.record_imu(&imu);
        runtime.bootstrap = Some(Bootstrap {
            alignment: Alignment {
                origin_ecef_m: Vector3::new(6378137., 0., 0.),
                yaw_rad: 0.,
                translation_enu_m: Vector3::zeros(),
            },
            time_offset_s: 0.,
            timing_std_s: 0.,
            clock: Default::default(),
        });
        runtime.push_epoch(epoch);
        assert!(runtime.build(&p).unwrap().epochs.is_empty());
        assert!(runtime.epochs.is_empty());
        assert_eq!(runtime.diagnostics.stale_epochs, 1);
        assert!(
            matches!(runtime.records.back(),Some(Record::Decision(v)) if v["reason"]=="delayed_epoch_age_limit")
        );
    }
    #[test]
    fn actual_satellite_rows_enter_prior_once_when_keyframe_retires() {
        let satellite = visloc_gnss::Satellite {
            system: visloc_gnss::System::Gps,
            prn: 1,
        };
        let epoch = Epoch {
            week: 2400,
            tow_s: 123.45,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: vec![visloc_gnss::Observation {
                satellite,
                signal: 0,
                pseudorange_m: 20_183_000.,
                doppler_hz: 0.,
                pseudorange_std_m: 3.,
                doppler_std_hz: 1.,
                cn0_dbhz: 45,
                lock_ms: 1000,
            }],
        };
        let (mut p, imu) = source(epoch.nominal_utc_ns().unwrap() - 50_000_000);
        p.states[0].is_keyframe = true;
        let mut runtime = GnssRuntime::new(
            Config {
                enabled: true,
                reject_sigma: 1e12,
                ..Default::default()
            },
            &UnitQuaternion::identity(),
            Vector3::zeros(),
        )
        .unwrap();
        runtime.record_imu(&imu);
        runtime.push_ephemeris(Ephemeris {
            satellite,
            week: 2400,
            toe: 123.45,
            toc: 123.45,
            issue: 1,
            healthy: true,
            sqrt_a: 5153.795,
            eccentricity: 0.,
            m0: 0.,
            delta_n: 0.,
            omega0: visloc_gnss::OMEGA_E * 123.45,
            inclination: 0.9,
            argument: 0.,
            omega_dot: 0.,
            inclination_dot: 0.,
            cuc: 0.,
            cus: 0.,
            crc: 0.,
            crs: 0.,
            cic: 0.,
            cis: 0.,
            af0: 0.,
            af1: 0.,
            af2: 0.,
            group_delay_s: 0.,
        });
        runtime.bootstrap = Some(Bootstrap {
            alignment: Alignment {
                origin_ecef_m: Vector3::new(6378137., 0., 0.),
                yaw_rad: 0.,
                translation_enu_m: Vector3::zeros(),
            },
            time_offset_s: 0.,
            timing_std_s: 0.02,
            clock: Default::default(),
        });
        runtime.push_epoch(epoch);
        let full = runtime.build(&p).unwrap();
        assert_eq!(full.observations.len(), 1);
        let f = full.extra(&full.initial()).unwrap();
        assert!(f[0].state_jacobian.columns(0, 15).norm() > 0.);
        let unrelated = runtime.build_keyframes(&p, Some(&[1])).unwrap();
        assert!(unrelated.observations.is_empty());
        runtime.marginalize(&p, &[], &[1], &[]).unwrap();
        assert_eq!(runtime.build(&p).unwrap().observations.len(), 1);
        runtime.marginalize(&p, &[], &[], &[u64::MAX]).unwrap();
        let prior = runtime.prior.as_ref().unwrap();
        let align = prior
            .blocks
            .iter()
            .find(|b| b.key == Key::Alignment)
            .unwrap();
        assert!(prior.j.columns(align.offset, align.dof).norm() > 1e-10);
        let consumed = runtime.build(&p).unwrap();
        assert!(consumed.epochs.is_empty());
        assert!(consumed.observations.is_empty());
    }
    #[test]
    fn new_keyframe_epoch_waits_for_imu_bracket_and_survives_an_unrelated_prefix() {
        let epoch = Epoch {
            week: 2400,
            tow_s: 123.45,
            leap_seconds: 18,
            leap_seconds_valid: true,
            clock_reset: false,
            receipt_ns: 0,
            observations: Vec::new(),
        };
        let (mut p, imu) = source(epoch.nominal_utc_ns().unwrap() - 50_000_000);
        p.states[1].is_keyframe = true;
        let mut runtime = GnssRuntime::new(
            Config {
                enabled: true,
                ..Default::default()
            },
            &UnitQuaternion::identity(),
            Vector3::zeros(),
        )
        .unwrap();
        runtime.record_imu(&imu[..imu.len() - 1]);
        runtime.bootstrap = Some(Bootstrap {
            alignment: Alignment {
                origin_ecef_m: Vector3::new(6378137., 0., 0.),
                yaw_rad: 0.,
                translation_enu_m: Vector3::zeros(),
            },
            time_offset_s: 0.,
            timing_std_s: 0.,
            clock: Default::default(),
        });
        runtime.push_epoch(epoch);
        assert!(runtime.build(&p).unwrap().epochs.is_empty());
        assert_eq!(runtime.epochs.len(), 1);
        assert_eq!(runtime.diagnostics.stale_epochs, 0);
        runtime.record_imu(&imu[imu.len() - 1..]);
        assert_eq!(runtime.build(&p).unwrap().epochs, vec![(0, 1)]);
        let mut prefix = p.clone();
        prefix.states.truncate(1);
        prefix.imu_links.clear();
        assert!(runtime
            .build_keyframes(&prefix, Some(&[u64::MAX]))
            .unwrap()
            .epochs
            .is_empty());
        assert_eq!(runtime.epochs.len(), 1);
        assert_eq!(runtime.diagnostics.stale_epochs, 0);
        assert_eq!(runtime.epoch_keyframes.get(&0), Some(&1));
    }
}
