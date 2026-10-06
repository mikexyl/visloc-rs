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
    pub solver: String,
    pub version: String,
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub solve_ms: f64,
    pub relative_factors: usize,
    pub gps_factors: usize,
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
    /// On failure, neither poses nor diagnostics are changed. There is no
    /// optimizer selector, fallback, subprocess, or Rust optimization code.
    pub fn optimize(&mut self) -> Result<OptimizerReport> {
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
        let mut output = poses.clone();
        let mut diagnostics = vec![FfiGpsDiagnostic::default(); gps.len()];
        let mut report = FfiReport::default();
        let mut error = [0 as c_char; 1024];
        // SAFETY: POD layouts match bridge.h, all input/output buffers remain
        // owned and live for this synchronous call, and C++ catches exceptions.
        let status = unsafe {
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
            candidate.insert(p.id, p.pose()?);
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
        self.poses = candidate;
        self.gps_diagnostics = diagnostics;
        Ok(OptimizerReport {
            solver: "gtsam_cpp".into(),
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
