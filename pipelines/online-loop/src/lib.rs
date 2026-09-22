//! Causal sequence retrieval and metric loop constraints alongside VIO.
//!
//! This module never updates the VIO filter, IMU biases, or local map. Its
//! output is a separate map<-odometry correction and corrected trajectory.
//! All GPU sessions are constructed and owned on one background thread.
use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};
use std::{fmt, path::PathBuf};
use visloc_core::geometry::SE3;

#[cfg(any(feature = "native", test))]
mod covisibility;
#[cfg(feature = "native")]
pub mod models;
#[cfg(feature = "native")]
mod worker;
#[cfg(feature = "native")]
pub use worker::OnlineLoop;

#[derive(Debug, Clone)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub jist_engine: PathBuf,
    pub xfeat_engine: PathBuf,
    pub lighterglue_engine: PathBuf,
    pub device: i32,
    /// Fixed input length used by the deployed LighterGlue TensorRT profile.
    /// Keyframes with fewer measured tracks still enter JIST, but skip matching.
    pub matcher_keypoints: usize,
    pub queue_capacity: usize,
    pub max_keyframes: usize,
    pub temporal_exclusion_s: f64,
    /// Minimum number of shared, triangulated VIO landmark IDs for an edge.
    pub covisibility_min_shared: usize,
    /// Local graph radius excluded around every member of the query sequence.
    pub covisibility_hops: usize,
    pub retrieval_top_k: usize,
    pub min_similarity: f32,
    pub min_matches: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub reprojection_threshold_px: f64,
    pub min_image_coverage: f64,
    pub loop_cooldown_s: f64,
    pub confirmation_window_s: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            jist_engine: PathBuf::new(),
            xfeat_engine: PathBuf::new(),
            lighterglue_engine: PathBuf::new(),
            device: 0,
            matcher_keypoints: 128,
            queue_capacity: 4,
            max_keyframes: 4000,
            temporal_exclusion_s: 20.0,
            covisibility_min_shared: 15,
            covisibility_hops: 2,
            retrieval_top_k: 3,
            min_similarity: 0.8,
            min_matches: 15,
            min_inliers: 15,
            min_inlier_ratio: 0.5,
            reprojection_threshold_px: 3.0,
            min_image_coverage: 0.02,
            loop_cooldown_s: 5.0,
            confirmation_window_s: 3.0,
        }
    }
}
impl Config {
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path).map_err(|e| Error(e.to_string()))?;
        let mut config: Self = serde_json::from_str(&data).map_err(|e| Error(e.to_string()))?;
        for model in [
            &mut config.jist_engine,
            &mut config.xfeat_engine,
            &mut config.lighterglue_engine,
        ] {
            if model.as_os_str().is_empty() {
                return Err(Error("All three TensorRT engines are required".into()));
            }
            if model.is_relative() {
                *model = path
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .join(&model);
            }
        }
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        if self.device < 0
            || !(1..=64).contains(&self.queue_capacity)
            || !(16..=1024).contains(&self.matcher_keypoints)
            || self.max_keyframes < 6
            || !(1..=1024).contains(&self.covisibility_min_shared)
            || !(1..=5).contains(&self.covisibility_hops)
            || self.retrieval_top_k == 0
            || self.retrieval_top_k > 20
            || !(0.0..=1.0).contains(&self.min_similarity)
            || !(0.0..=1.0).contains(&self.min_inlier_ratio)
            || !(0.0..=1.0).contains(&self.min_image_coverage)
            || self.min_inliers < 6
            || self.min_matches < self.min_inliers
            || [
                self.temporal_exclusion_s,
                self.reprojection_threshold_px,
                self.loop_cooldown_s,
                self.confirmation_window_s,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(Error("Invalid online loop configuration".into()));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Observation {
    /// Persistent VIO track/landmark identity, never a descriptor-match index.
    pub track_id: u64,
    pub pixel: Point2<f64>,
    /// Metric world point estimated by VIO; absent for an untriangulated track.
    pub point_world: Option<Point3<f64>>,
}
#[derive(Clone)]
pub struct Frame {
    pub id: u64,
    /// Ordinal of Basalt's actual keyframe decision (not the raw frame ID).
    /// Increment even if submission drops, so sequences cannot span a lost KF.
    pub keyframe_index: u64,
    pub timestamp_ns: i64,
    pub width: usize,
    pub height: usize,
    pub gray: Vec<u8>,
    pub body_to_world: SE3,
    pub observations: Vec<Observation>,
}
impl Frame {
    pub fn validate(&self) -> Result<()> {
        if self.width < 2
            || self.height < 2
            || self.width.checked_mul(self.height) != Some(self.gray.len())
            || self.observations.len() > 1024
            || !self.body_to_world.matrix().iter().all(|v| v.is_finite())
            || self.body_to_world.rotation.norm() < 1e-12
            || self.observations.iter().any(|o| {
                !o.pixel.coords.iter().all(|v| v.is_finite())
                    || o.pixel.x < 0.0
                    || o.pixel.y < 0.0
                    || o.pixel.x >= self.width as f64
                    || o.pixel.y >= self.height as f64
                    || o.point_world
                        .is_some_and(|p| !p.coords.iter().all(|v| v.is_finite()))
            })
        {
            return Err(Error(
                "Invalid loop frame dimensions, observations, or pose".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct Snapshot {
    pub keyframes: usize,
    pub sequence_queries: usize,
    pub candidate_queries: usize,
    pub local_sequences_skipped: usize,
    pub redundant_sequences_skipped: usize,
    pub candidates: usize,
    pub accepted_loops: usize,
    pub dropped_keyframes: usize,
    pub capacity_skips: usize,
    pub revision: usize,
    pub worker_ms: f64,
    pub error: Option<String>,
    /// Sorted keyframe timestamps and map<-odometry corrections.
    pub corrections: Vec<(i64, SE3)>,
    pub edges: Vec<(u64, u64)>,
    pub keyframe_positions: Vec<(u64, [f64; 3])>,
}
impl Snapshot {
    pub fn correction_at(&self, timestamp_ns: i64) -> SE3 {
        let i = self
            .corrections
            .partition_point(|(t, _)| *t <= timestamp_ns);
        if i == 0 {
            return self
                .corrections
                .first()
                .map(|x| x.1.clone())
                .unwrap_or_default();
        }
        if i == self.corrections.len() {
            return self.corrections[i - 1].1.clone();
        }
        let (t0, a) = &self.corrections[i - 1];
        let (t1, b) = &self.corrections[i];
        let alpha = (timestamp_ns - t0) as f64 / (t1 - t0) as f64;
        SE3::new(
            a.rotation.slerp(&b.rotation, alpha),
            a.translation * (1.0 - alpha) + b.translation * alpha,
        )
    }
    pub fn json(&self) -> serde_json::Value {
        let c = self
            .corrections
            .last()
            .map(|x| x.1.clone())
            .unwrap_or_default();
        let q = c.rotation.quaternion();
        serde_json::json!({"keyframes":self.keyframes,"candidates":self.candidates,
            "sequence_queries":self.sequence_queries,"candidate_queries":self.candidate_queries,
            "local_sequences_skipped":self.local_sequences_skipped,
            "redundant_sequences_skipped":self.redundant_sequences_skipped,
            "accepted_loops":self.accepted_loops,"dropped_keyframes":self.dropped_keyframes,
            "capacity_skips":self.capacity_skips,"revision":self.revision,"worker_ms":self.worker_ms,
            "error":self.error,"map_from_odom":{"translation":c.translation.as_slice(),
            "quaternion_xyzw":[q.i,q.j,q.k,q.w]},"edges":self.edges,
            "keyframe_positions":self.keyframe_positions})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    #[test]
    fn interpolated_correction_and_endpoints() {
        let s = Snapshot {
            corrections: vec![
                (10, SE3::identity()),
                (
                    20,
                    SE3::new(
                        UnitQuaternion::from_euler_angles(0.0, 0.0, 1.0),
                        Vector3::new(2.0, 0.0, 0.0),
                    ),
                ),
            ],
            ..Snapshot::default()
        };
        assert_eq!(s.correction_at(0), SE3::identity());
        assert!((s.correction_at(15).translation.x - 1.0).abs() < 1e-12);
        assert!((s.correction_at(15).rotation.angle() - 0.5).abs() < 1e-12);
        assert_eq!(s.correction_at(100), s.correction_at(20));
        assert_eq!(Snapshot::default().correction_at(100), SE3::identity());
    }
    #[test]
    fn bad_configuration_rejected() {
        let mut c = Config::default();
        c.validate().unwrap();
        c.queue_capacity = 0;
        assert!(c.validate().is_err());
        c.queue_capacity = 4;
        c.min_similarity = f32::NAN;
        assert!(c.validate().is_err());
    }
}
