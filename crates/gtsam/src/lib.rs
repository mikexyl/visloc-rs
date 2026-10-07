//! Safe, synchronous snapshot wrapper over GTSAM C++. This crate owns graph
//! records and validates returned values; all optimization is performed by
//! GTSAM. Call it on the existing optimization worker, never a sensor callback.
use nalgebra::{Matrix3, Matrix6, Quaternion, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::{c_char, CStr},
};
use visloc_core::geometry::SE3;

#[derive(Debug, thiserror::Error)]
#[error("GTSAM C++: {0}")]
pub struct Error(pub String);
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpsRobustKernel {
    Huber { delta: f64 },
    Switchable { lambda: f64 },
}
#[derive(Clone, Debug, PartialEq)]
pub struct GpsPositionFactor {
    pub from: u64,
    pub to: u64,
    pub alpha: f64,
    pub position: Vector3<f64>,
    pub lever_arm: Vector3<f64>,
    /// ENU information, with zero Up row and column (enforced in C++).
    pub information: Matrix3<f64>,
    pub kernel: GpsRobustKernel,
    pub deadband_sigma: f64,
}
#[derive(Clone, Debug)]
pub struct BetweenFactor {
    pub from: u64,
    pub to: u64,
    pub to_from: SE3,
    /// Translation then rotation, including full cross-block information.
    pub information: Matrix6<f64>,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpsDiagnostic {
    pub residual: Vector3<f64>,
    pub squared_mahalanobis: f64,
    pub robust_weight: f64,
    pub optimization_weight: f64,
    pub cost: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerReport {
    #[serde(default)]
    pub landmarks: usize,
    #[serde(default)]
    pub reprojection_factors: usize,
    #[serde(default)]
    pub reprojection_rmse_before_px: Option<f64>,
    #[serde(default)]
    pub reprojection_rmse_after_px: Option<f64>,
    pub solver: String,
    pub version: String,
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub solve_ms: f64,
    pub relative_factors: usize,
    pub gps_factors: usize,
}
/// A measured pixel in a calibrated camera attached rigidly to a body pose.
#[derive(Clone, Debug)]
pub struct ProjectionFactor {
    pub pose: u64,
    pub landmark: u64,
    pub camera_to_body: SE3,
    pub intrinsics: [f64; 4],
    pub pixel: [f64; 2],
    pub sigma: f64,
    pub huber: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PoseGraph {
    /// Body-to-map poses (never the old solver's inverse-pose convention).
    pub poses: BTreeMap<u64, SE3>,
    pub edges: Vec<BetweenFactor>,
    pub gps_factors: Vec<GpsPositionFactor>,
    pub anchor: Option<u64>,
    pub horizontal_anchor: bool,
    pub gps_diagnostics: Vec<GpsDiagnostic>,
}
impl PoseGraph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_pose(&mut self, id: u64, body_to_map: SE3) {
        self.poses.insert(id, body_to_map);
    }
    pub fn anchor(&mut self, id: u64) {
        self.anchor = Some(id);
        self.horizontal_anchor = false;
    }
    pub fn add_edge_with_information(
        &mut self,
        from: u64,
        to: u64,
        to_from: SE3,
        information: Matrix6<f64>,
    ) {
        self.edges.push(BetweenFactor {
            from,
            to,
            to_from,
            information,
        });
    }
    /// On failure, poses and diagnostics are unchanged. Optimization is native GTSAM.
    pub fn optimize(&mut self) -> Result<OptimizerReport> {
        self.optimize_internal(None, &[])
    }
    /// Joint full-graph visual BA with metric between factors and fixed calibration.
    /// Landmarks are eliminated before poses. All outputs commit atomically.
    pub fn optimize_bundle(
        &mut self,
        landmarks: &mut BTreeMap<u64, Vector3<f64>>,
        projections: &[ProjectionFactor],
    ) -> Result<OptimizerReport> {
        self.optimize_internal(Some(landmarks), projections)
    }
    fn optimize_internal(
        &mut self,
        landmarks: Option<&mut BTreeMap<u64, Vector3<f64>>>,
        projections: &[ProjectionFactor],
    ) -> Result<OptimizerReport> {
        let anchor = self.anchor.ok_or_else(|| Error("missing anchor".into()))?;
        let poses: Vec<_> = self
            .poses
            .iter()
            .map(|(&id, p)| FfiPose::from_pose(id, p))
            .collect();
        let edges: Vec<_> = self
            .edges
            .iter()
            .map(|e| FfiBetween {
                from: e.from,
                to: e.to,
                measurement: FfiPose::from_pose(0, &e.to_from),
                information: std::array::from_fn(|i| e.information[(i / 6, i % 6)]),
            })
            .collect();
        let gps: Vec<_> = self
            .gps_factors
            .iter()
            .map(|f| {
                let (kernel, parameter) = match f.kernel {
                    GpsRobustKernel::Huber { delta } => (0, delta),
                    GpsRobustKernel::Switchable { lambda } => (1, lambda),
                };
                FfiGps {
                    from: f.from,
                    to: f.to,
                    alpha: f.alpha,
                    position: f.position.into(),
                    lever: f.lever_arm.into(),
                    information: std::array::from_fn(|i| f.information[(i / 3, i % 3)]),
                    kernel,
                    parameter,
                    deadband: f.deadband_sigma,
                }
            })
            .collect();
        let points: Vec<_> = landmarks
            .as_ref()
            .map(|m| {
                m.iter()
                    .map(|(&id, p)| FfiLandmark {
                        id,
                        position: (*p).into(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let measurements: Vec<_> = projections
            .iter()
            .map(|p| FfiProjection {
                pose: p.pose,
                landmark: p.landmark,
                camera_to_body: FfiPose::from_pose(0, &p.camera_to_body),
                intrinsics: p.intrinsics,
                pixel: p.pixel,
                sigma: p.sigma,
                huber: p.huber,
            })
            .collect();
        let mut point_output = points.clone();
        let rmse_before = landmarks
            .as_ref()
            .and_then(|m| projection_rmse(&self.poses, m, projections));
        let mut output = poses.clone();
        let mut diagnostics = vec![FfiGpsDiagnostic::default(); gps.len()];
        let mut report = FfiReport::default();
        let mut error = [0 as c_char; 1024];
        // SAFETY: POD layouts match bridge.h, all input/output buffers remain
        // owned and live for this synchronous call, and C++ catches exceptions.
        let status = unsafe {
            if landmarks.is_some() {
                visloc_gtsam_solve_bundle(
                    poses.as_ptr(),
                    poses.len(),
                    edges.as_ptr(),
                    edges.len(),
                    gps.as_ptr(),
                    gps.len(),
                    anchor,
                    self.horizontal_anchor as u32,
                    points.as_ptr(),
                    points.len(),
                    measurements.as_ptr(),
                    measurements.len(),
                    output.as_mut_ptr(),
                    point_output.as_mut_ptr(),
                    diagnostics.as_mut_ptr(),
                    &mut report,
                    error.as_mut_ptr(),
                    error.len(),
                )
            } else {
                visloc_gtsam_solve(
                    poses.as_ptr(),
                    poses.len(),
                    edges.as_ptr(),
                    edges.len(),
                    gps.as_ptr(),
                    gps.len(),
                    anchor,
                    self.horizontal_anchor as u32,
                    output.as_mut_ptr(),
                    diagnostics.as_mut_ptr(),
                    &mut report,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        };
        if status != 0 {
            // C++ always null-terminates this pre-zeroed error buffer.
            return Err(Error(
                unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
            ));
        }
        if ![report.initial_cost, report.final_cost, report.solve_ms]
            .iter()
            .all(|x| x.is_finite())
            || report.final_cost > report.initial_cost + 1e-9
            || report.iterations > 20
        {
            return Err(Error("invalid optimization report".into()));
        }
        let mut candidate = BTreeMap::new();
        for (input, p) in poses.iter().zip(output) {
            if p.id != input.id {
                return Err(Error("output pose identity mismatch".into()));
            }
            let pose = p.pose()?;
            candidate.insert(p.id, pose);
        }
        let before = self
            .poses
            .get(&anchor)
            .ok_or_else(|| Error("missing anchor".into()))?;
        let after = &candidate[&anchor];
        let gauge_ok = if self.horizontal_anchor {
            (before.translation.z - after.translation.z).abs() < 1e-8
                && (before.rotation.inverse() * Vector3::z()
                    - after.rotation.inverse() * Vector3::z())
                .norm()
                    < 1e-8
        } else {
            before.inverse().compose(after).log().norm() < 1e-8
        };
        if !gauge_ok {
            return Err(Error("changed fixed gauge".into()));
        }
        let diagnostics = diagnostics
            .into_iter()
            .map(|d| {
                if !d
                    .residual
                    .iter()
                    .chain(
                        [
                            d.squared_mahalanobis,
                            d.robust_weight,
                            d.optimization_weight,
                            d.cost,
                        ]
                        .iter(),
                    )
                    .all(|x| x.is_finite())
                {
                    return Err(Error("non-finite GPS diagnostic".into()));
                }
                Ok(GpsDiagnostic {
                    residual: d.residual.into(),
                    squared_mahalanobis: d.squared_mahalanobis,
                    robust_weight: d.robust_weight,
                    optimization_weight: d.optimization_weight,
                    cost: d.cost,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut candidate_points = BTreeMap::new();
        for (input, point) in points.iter().zip(point_output) {
            if input.id != point.id || !point.position.iter().all(|x| x.is_finite()) {
                return Err(Error("invalid optimized landmark".into()));
            }
            candidate_points.insert(point.id, Vector3::from(point.position));
        }
        let rmse_after = projection_rmse(&candidate, &candidate_points, projections);
        let is_bundle = landmarks.is_some();
        if let Some(landmarks) = landmarks {
            *landmarks = candidate_points;
        }
        self.poses = candidate;
        self.gps_diagnostics = diagnostics;
        Ok(OptimizerReport {
            solver: if is_bundle {
                "gtsam_global_ba"
            } else {
                "gtsam_cpp"
            }
            .into(),
            landmarks: points.len(),
            reprojection_factors: projections.len(),
            reprojection_rmse_before_px: rmse_before,
            reprojection_rmse_after_px: rmse_after,
            version: "4.3.0".into(),
            iterations: report.iterations as usize,
            initial_cost: report.initial_cost,
            final_cost: report.final_cost,
            solve_ms: report.solve_ms,
            relative_factors: edges.len(),
            gps_factors: gps.len(),
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FfiPose {
    id: u64,
    translation: [f64; 3],
    quaternion_xyzw: [f64; 4],
}
impl FfiPose {
    fn from_pose(id: u64, p: &SE3) -> Self {
        let q = p.rotation.quaternion();
        Self {
            id,
            translation: p.translation.into(),
            quaternion_xyzw: [q.i, q.j, q.k, q.w],
        }
    }
    fn pose(&self) -> Result<SE3> {
        if !self
            .translation
            .iter()
            .chain(self.quaternion_xyzw.iter())
            .all(|x| x.is_finite())
        {
            return Err(Error("non-finite output pose".into()));
        }
        let [x, y, z, w] = self.quaternion_xyzw;
        let q = Quaternion::new(w, x, y, z);
        if (q.norm() - 1.).abs() > 1e-6 {
            return Err(Error("non-unit output rotation".into()));
        }
        Ok(SE3::new(
            UnitQuaternion::new_normalize(q),
            self.translation.into(),
        ))
    }
}
#[repr(C)]
struct FfiBetween {
    from: u64,
    to: u64,
    measurement: FfiPose,
    information: [f64; 36],
}
#[repr(C)]
struct FfiGps {
    from: u64,
    to: u64,
    alpha: f64,
    position: [f64; 3],
    lever: [f64; 3],
    information: [f64; 9],
    kernel: u32,
    parameter: f64,
    deadband: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FfiGpsDiagnostic {
    residual: [f64; 3],
    squared_mahalanobis: f64,
    robust_weight: f64,
    optimization_weight: f64,
    cost: f64,
}
#[repr(C)]
#[derive(Default)]
struct FfiReport {
    initial_cost: f64,
    final_cost: f64,
    solve_ms: f64,
    iterations: u64,
}
extern "C" {
    fn visloc_gtsam_solve(
        poses: *const FfiPose,
        pose_count: usize,
        edges: *const FfiBetween,
        edge_count: usize,
        gps: *const FfiGps,
        gps_count: usize,
        anchor: u64,
        horizontal_anchor: u32,
        output: *mut FfiPose,
        gps_output: *mut FfiGpsDiagnostic,
        report: *mut FfiReport,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

fn projection_rmse(
    poses: &BTreeMap<u64, SE3>,
    points: &BTreeMap<u64, Vector3<f64>>,
    factors: &[ProjectionFactor],
) -> Option<f64> {
    if factors.is_empty() {
        return None;
    }
    let mut cost = 0.;
    for f in factors {
        let camera = poses.get(&f.pose)?.compose(&f.camera_to_body).inverse();
        let p = camera.rotation * points.get(&f.landmark)? + camera.translation;
        if p.z <= 0. {
            return None;
        }
        let [fx, fy, cx, cy] = f.intrinsics;
        cost +=
            (fx * p.x / p.z + cx - f.pixel[0]).powi(2) + (fy * p.y / p.z + cy - f.pixel[1]).powi(2);
    }
    Some((cost / factors.len() as f64).sqrt())
}
#[repr(C)]
#[derive(Clone, Copy)]
struct FfiLandmark {
    id: u64,
    position: [f64; 3],
}
#[repr(C)]
struct FfiProjection {
    pose: u64,
    landmark: u64,
    camera_to_body: FfiPose,
    intrinsics: [f64; 4],
    pixel: [f64; 2],
    sigma: f64,
    huber: f64,
}
extern "C" {
    fn visloc_gtsam_solve_bundle(
        poses: *const FfiPose,
        pose_count: usize,
        edges: *const FfiBetween,
        edge_count: usize,
        gps: *const FfiGps,
        gps_count: usize,
        anchor: u64,
        horizontal_anchor: u32,
        landmarks: *const FfiLandmark,
        landmark_count: usize,
        projections: *const FfiProjection,
        projection_count: usize,
        output: *mut FfiPose,
        landmark_output: *mut FfiLandmark,
        gps_output: *mut FfiGpsDiagnostic,
        report: *mut FfiReport,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}
