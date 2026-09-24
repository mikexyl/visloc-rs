use crate::{Error, Result};
use nalgebra::{Matrix6, Quaternion, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use visloc_core::{geometry::SE3, types::Camera};

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Key {
    pub robot: String,
    pub session: String,
    pub id: u64,
}
impl Key {
    pub fn new(robot: &str, session: &str, id: u64) -> Self {
        Self {
            robot: robot.into(),
            session: session.into(),
            id,
        }
    }
    pub fn same_session(&self, other: &Self) -> bool {
        self.robot == other.robot && self.session == other.session
    }
    pub fn validate(&self) -> Result<()> {
        let valid = |s: &str| {
            !s.is_empty()
                && s.len() <= 128
                && s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        };
        if !valid(&self.robot) || !valid(&self.session) {
            return Err(Error("invalid robot/session identity".into()));
        }
        Ok(())
    }
}

/// Canonical pair identity is also the deterministic owner: the first robot
/// performs verification. Ownership is independent of message arrival order.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Pair(pub Key, pub Key);
impl Pair {
    pub fn new(a: Key, b: Key) -> Self {
        if a <= b {
            Self(a, b)
        } else {
            Self(b, a)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: [f64; 3],
    pub rotation_xyzw: [f64; 4],
}
impl Default for Transform {
    fn default() -> Self {
        Self::from(&SE3::identity())
    }
}
impl From<&SE3> for Transform {
    fn from(t: &SE3) -> Self {
        let q = t.rotation.quaternion();
        Self {
            translation: t.translation.into(),
            rotation_xyzw: [q.i, q.j, q.k, q.w],
        }
    }
}
impl Transform {
    pub fn se3(&self) -> Result<SE3> {
        let [x, y, z, w] = self.rotation_xyzw;
        let q = Quaternion::new(w, x, y, z);
        if !self
            .translation
            .iter()
            .chain(self.rotation_xyzw.iter())
            .all(|v| v.is_finite())
            || (q.norm() - 1.).abs() > 1e-3
        {
            return Err(Error("nonfinite pose or non-unit quaternion".into()));
        }
        Ok(SE3::new(
            UnitQuaternion::new_normalize(q),
            Vector3::from(self.translation),
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraModel {
    pub width: u32,
    pub height: u32,
    pub intrinsics: [f64; 4],
    pub camera_to_body: Transform,
}
impl CameraModel {
    pub fn camera(&self) -> Result<Camera> {
        let [fx, fy, cx, cy] = self.intrinsics;
        if self.width < 2
            || self.height < 2
            || fx <= 0.
            || fy <= 0.
            || !self.intrinsics.iter().all(|v| v.is_finite())
        {
            return Err(Error("invalid pinhole camera".into()));
        }
        self.camera_to_body.se3()?;
        Ok(Camera::pinhole(0, self.width, self.height, fx, fy, cx, cy))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyframeRecord {
    pub key: Key,
    pub timestamp_ns: i64,
    pub body_to_odom: Transform,
    pub previous: Option<Key>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sequence {
    pub key: Key,
    pub members: Vec<Key>,
    pub selected: Vec<Key>,
    pub start_ns: i64,
    pub end_ns: i64,
    pub model_id: String,
    pub descriptor: Vec<f32>,
    /// Local two-hop covisibility neighborhood, computed before feature selection.
    pub excluded_keyframes: Vec<u64>,
}
pub fn normalized(v: &[f32], n: usize) -> bool {
    v.len() == n
        && v.iter().all(|x| x.is_finite())
        && (v.iter().map(|x| x * x).sum::<f32>() - 1.).abs() < 1e-3
}
impl Sequence {
    pub fn validate(&self) -> Result<()> {
        self.key.validate()?;
        if self.members.len() != 10
            || self.selected.len() != 5
            || self.model_id.is_empty()
            || self.start_ns > self.end_ns
            || !normalized(&self.descriptor, 512)
            || self.members.iter().any(|k| !k.same_session(&self.key))
            || !self.members.windows(2).all(|w| w[0].id < w[1].id)
            || self.selected.iter().any(|k| !self.members.contains(k))
            || !self.selected.windows(2).all(|w| w[0].id < w[1].id)
        {
            return Err(Error("invalid ten-keyframe sequence".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameDescriptors {
    pub sequence: Key,
    pub frames: Vec<Key>,
    /// Five row-major, normalized 512-dimensional descriptors.
    pub descriptors: Vec<f32>,
}
impl FrameDescriptors {
    pub fn validate(&self) -> Result<()> {
        if self.frames.len() != 5
            || self.descriptors.len() != 2560
            || !self.descriptors.chunks(512).all(|v| normalized(v, 512))
        {
            return Err(Error("invalid JIST frame descriptor matrix".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureFrame {
    pub key: Key,
    pub timestamp_ns: i64,
    pub camera: CameraModel,
    pub pixels: Vec<[f32; 2]>,
    pub descriptors: Vec<f32>,
    pub track_ids: Vec<u64>,
    pub points_camera: Vec<Option<[f64; 3]>>,
}
impl FeatureFrame {
    pub fn validate(&self) -> Result<()> {
        self.key.validate()?;
        self.camera.camera()?;
        let n = self.pixels.len();
        if n > 1024
            || self.descriptors.len() != n * 64
            || self.track_ids.len() != n
            || self.points_camera.len() != n
            || !self.descriptors.iter().all(|x| x.is_finite())
            || self.pixels.iter().any(|p| {
                !p.iter().all(|x| x.is_finite())
                    || p[0] < 0.
                    || p[1] < 0.
                    || p[0] >= self.camera.width as f32
                    || p[1] >= self.camera.height as f32
            })
            || self
                .points_camera
                .iter()
                .flatten()
                .any(|p| !p.iter().all(|x| x.is_finite()))
        {
            return Err(Error("invalid sparse feature packet".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Verification {
    pub matches: usize,
    pub two_d_inliers: usize,
    pub pnp_inliers: usize,
    pub reprojection_px: f64,
    pub coverage: f64,
    pub reverse: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopConstraint {
    pub pair: Pair,
    pub from: Key,
    pub to: Key,
    /// T_to_from: converts a point in the 'from' body into the 'to' body.
    pub to_from: Transform,
    /// Row-major information for Log(Z^-1 T_to_world T_world_from), [rho; omega].
    pub information: Vec<f64>,
    pub similarity: f32,
    pub verification: Verification,
}
pub fn information(translation_sigma: f64, rotation_sigma: f64) -> Matrix6<f64> {
    Matrix6::from_diagonal(&nalgebra::Vector6::new(
        translation_sigma.powi(-2),
        translation_sigma.powi(-2),
        translation_sigma.powi(-2),
        rotation_sigma.powi(-2),
        rotation_sigma.powi(-2),
        rotation_sigma.powi(-2),
    ))
}
pub fn matrix_values(m: &Matrix6<f64>) -> Vec<f64> {
    (0..6)
        .flat_map(|i| (0..6).map(move |j| m[(i, j)]))
        .collect()
}
pub fn checked_information(v: &[f64]) -> Result<Matrix6<f64>> {
    if v.len() != 36 || !v.iter().all(|x| x.is_finite()) {
        return Err(Error("invalid information matrix".into()));
    }
    let m = Matrix6::from_row_slice(v);
    if (m - m.transpose()).norm() > 1e-8 || m.cholesky().is_none() {
        return Err(Error(
            "information must be symmetric positive definite".into(),
        ));
    }
    Ok(m)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizedPose {
    pub key: Key,
    pub timestamp_ns: i64,
    pub component: Key,
    pub body_to_map: Transform,
    pub map_from_odom: Transform,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub revision: u64,
    pub input_revision: u64,
    pub poses: Vec<OptimizedPose>,
    pub loops: Vec<LoopConstraint>,
    pub components: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub solve_ms: f64,
}
