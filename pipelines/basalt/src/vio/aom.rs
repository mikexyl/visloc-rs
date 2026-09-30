//! Basalt ABS_QR-style AOM reduction core.
//!
//! The ordering follows the upstream fixed-SHA contract: pose (6), velocity
//! (3), gyro bias (3), accelerometer bias (3). This module is deliberately a
//! numeric core and has no dependency on the repository bundle optimizer.
use crate::camera::DoubleSphereCamera;
use crate::imu::{eigen_ldlt_solve_f32, ImuPreintegratedDelta};
use crate::timing::{TimingBreakdown, TimingBucket};
use crate::vio::landmarks::{sophus_so3_inverse, sophus_so3_product, InverseDistanceLandmark};
use crate::vio::scalar::ScalarMode;
use nalgebra::{
    DMatrix, DVector, Matrix2x3, Matrix3, Matrix3x2, Matrix6, Point2, Point3, Quaternion, SMatrix,
    UnitQuaternion, Vector2, Vector3, Vector6,
};
use serde_json::json;
use std::{
    cell::Cell,
    collections::HashSet,
    fs::OpenOptions,
    io::Write as IoWrite,
    path::Path,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Mutex, OnceLock,
    },
};
use visloc_core::geometry::SE3;

/// Upstream AOM navigation block: pose6, velocity3, gyro-bias3, accel-bias3.
pub const AOM_NAV_DOF: usize = 15;

// Keep the state/step provenance check shared by the producer and the
// concrete WindowProblem consumer.  The preparation is one-shot, so a
// deterministic bit fingerprint is preferable to a borrowed state reference:
// it rejects a stale solve before any trial-side values are materialized.
const LM_VECTOR_FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const LM_VECTOR_FNV_PRIME: u64 = 1_099_511_628_211;

#[inline]
pub(crate) fn lm_trial_vector_fingerprint(value: &DVector<f64>) -> u64 {
    let mut hash = LM_VECTOR_FNV_OFFSET;
    hash ^= value.len() as u64;
    hash = hash.wrapping_mul(LM_VECTOR_FNV_PRIME);
    for component in value.iter() {
        hash ^= component.to_bits();
        hash = hash.wrapping_mul(LM_VECTOR_FNV_PRIME);
    }
    hash
}

// The full solver's `linearize` trait method intentionally has no iteration
// argument because legacy/synthetic callers use it directly.  Keep diagnostic
// identity in a thread-local slot so concurrent windows cannot relabel one
// another's sidecars.  `None` remains the inactive state for pre-loop costs
// and direct callers.
#[derive(Clone, Copy, Debug, Default)]
struct DiagnosticLmContext {
    run_id: Option<u128>,
    frame_id: Option<u64>,
    iteration: Option<usize>,
}

thread_local! {
    static ACTIVE_DIAGNOSTIC_LM_CONTEXT: Cell<Option<DiagnosticLmContext>> = const { Cell::new(None) };
    // Marginalization re-linearizes the truncated AOM after the LM guard has
    // ended, so its diagnostic projection records need an explicit event
    // identity instead of borrowing the (already cleared) LM frame slot.
    static ACTIVE_DIAGNOSTIC_PROJECTION_EVENT: Cell<Option<DiagnosticProjectionEvent>> =
        const { Cell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DiagnosticProjectionEvent {
    pub(crate) frame_id: u64,
    pub(crate) timestamp_ns: i64,
}

pub(crate) fn set_active_diagnostic_projection_event(event: Option<DiagnosticProjectionEvent>) {
    ACTIVE_DIAGNOSTIC_PROJECTION_EVENT.with(|context| context.set(event));
}

fn active_diagnostic_projection_event() -> Option<DiagnosticProjectionEvent> {
    ACTIVE_DIAGNOSTIC_PROJECTION_EVENT.with(Cell::get)
}

static NEXT_DIAGNOSTIC_LM_RUN_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct DiagnosticLmRunGuard {
    previous: Option<DiagnosticLmContext>,
}

impl Drop for DiagnosticLmRunGuard {
    fn drop(&mut self) {
        ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(|context| context.set(self.previous));
    }
}

#[cfg(test)]
mod m11_frame39_translation_probe {
    use super::*;

    #[test]
    #[ignore = "frame39 native current-translation schedule probe"]
    fn m11_frame39_track1827_camera_translation_schedule_probe() {
        fn pose(words: [u32; 7]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(words[3]),
                    f32::from_bits(words[0]),
                    f32::from_bits(words[1]),
                    f32::from_bits(words[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(words[4]),
                    f32::from_bits(words[5]),
                    f32::from_bits(words[6]),
                ),
            }
        }

        let target_camera_from_imu = pose([
            0x3b19188b, 0xbc55012e, 0xbf33d4ed, 0x3f362af6, 0xbd27e3b1, 0xbc8044de, 0xbb7a8e6e,
        ]);
        let target_imu_from_anchor_imu = pose([
            0xbae89e11, 0x3cacb89c, 0xbb36a8c7, 0x3f7ff113, 0x3de32d30, 0xbb8d4bec, 0xbd09edaa,
        ]);
        let anchor_t_imu_cam = pose([
            0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
        ]);
        let captured_prefix = pose([
            0x3c826a39, 0x3b559009, 0xbf344ab5, 0x3f35b244, 0xbd30c0af, 0xbe022c22, 0xbd130180,
        ]);
        let expected = [0xbde5ffa4, 0xbde359ce, 0xbd013192];

        let exact = |value: Vector3<f32>| {
            [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()]
                .iter()
                .zip(expected)
                .filter(|(left, right)| **left == *right)
                .count()
        };
        let bits = |value: Vector3<f32>| {
            format!(
                "{:08x} {:08x} {:08x}",
                value.x.to_bits(),
                value.y.to_bits(),
                value.z.to_bits()
            )
        };

        for (prefix_name, prefix_action) in [
            (
                "generic",
                sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_anchor_imu.translation,
                ),
            ),
            (
                "step_packet",
                sophus_rotate_step_packet_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_anchor_imu.translation,
                ),
            ),
        ] {
            let prefix = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_anchor_imu.rotation,
                ),
                translation: prefix_action + target_camera_from_imu.translation,
            };
            for (suffix_name, suffix_action) in [
                (
                    "generic",
                    sophus_rotate_f32(prefix.rotation, anchor_t_imu_cam.translation),
                ),
                (
                    "step_packet",
                    sophus_rotate_step_packet_f32(prefix.rotation, anchor_t_imu_cam.translation),
                ),
            ] {
                let result = suffix_action + prefix.translation;
                println!(
                    "m11_frame39_translation prefix={prefix_name} suffix={suffix_name} prefix_t={} result={} exact={}/3",
                    bits(prefix.translation), bits(result), exact(result)
                );
            }
        }

        for (suffix_name, suffix_action) in [
            (
                "generic",
                sophus_rotate_f32(captured_prefix.rotation, anchor_t_imu_cam.translation),
            ),
            (
                "step_packet",
                sophus_rotate_step_packet_f32(
                    captured_prefix.rotation,
                    anchor_t_imu_cam.translation,
                ),
            ),
        ] {
            let result = suffix_action + captured_prefix.translation;
            println!(
                "m11_frame39_translation captured_prefix suffix={suffix_name} result={} exact={}/3",
                bits(result),
                exact(result)
            );
        }
    }
}

pub(crate) fn begin_diagnostic_lm_run() -> DiagnosticLmRunGuard {
    let run_id = next_diagnostic_lm_run_id();
    let previous = ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(|context| {
        let previous = context.get();
        context.set(Some(DiagnosticLmContext {
            run_id: Some(run_id),
            ..DiagnosticLmContext::default()
        }));
        previous
    });
    DiagnosticLmRunGuard { previous }
}

fn next_diagnostic_lm_run_id() -> u128 {
    let sequence = NEXT_DIAGNOSTIC_LM_RUN_ID.fetch_add(1, Ordering::Relaxed);
    (u128::from(std::process::id()) << 64) | u128::from(sequence)
}

fn active_diagnostic_lm_context() -> Option<DiagnosticLmContext> {
    ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(Cell::get)
}

pub(crate) fn active_diagnostic_lm_run_id() -> Option<u128> {
    active_diagnostic_lm_context().and_then(|context| context.run_id)
}

pub(crate) fn set_active_diagnostic_lm_iteration(iteration: Option<usize>) {
    ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(|context| {
        let mut value = context.get().unwrap_or_default();
        value.iteration = iteration;
        if value.run_id.is_none() && value.frame_id.is_none() && value.iteration.is_none() {
            context.set(None);
        } else {
            context.set(Some(value));
        }
    });
}

pub(crate) fn active_diagnostic_lm_iteration() -> Option<usize> {
    active_diagnostic_lm_context().and_then(|context| context.iteration)
}

pub(crate) fn set_active_diagnostic_lm_frame(frame_id: Option<u64>) {
    ACTIVE_DIAGNOSTIC_LM_CONTEXT.with(|context| {
        let mut value = context.get().unwrap_or_default();
        value.frame_id = frame_id;
        if value.run_id.is_none() && value.frame_id.is_none() && value.iteration.is_none() {
            context.set(None);
        } else {
            context.set(Some(value));
        }
    });
}

pub(crate) fn active_diagnostic_lm_frame() -> Option<u64> {
    active_diagnostic_lm_context().and_then(|context| context.frame_id)
}

pub type Matrix2x6 = SMatrix<f64, 2, 6>;

/// Opt-in f32 visual-chain snapshot used to compare the current-value pose
/// path with the pinned Eigen/Sophus reference after an accepted LM step.
/// These fields are populated only when `VISLOC_BASALT_VISUAL_CHAIN_TRACE`
/// is set and never participate in the normal factor arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisualChainF32 {
    /// `[qx, qy, qz, qw, tx, ty, tz]` for the intermediate rigid transforms.
    pub target_camera_from_imu: [f32; 7],
    pub target_imu_from_anchor_imu: [f32; 7],
    pub target_camera_from_anchor_imu: [f32; 7],
    /// FEJ relative camera transform used exclusively by d_rel/d_h.
    pub target_camera_from_anchor_imu_fej: [f32; 7],
    pub target_camera_from_anchor_camera: [f32; 7],
    /// Reconstructed with the same matrix builder as the production GEMV.
    pub target_camera_matrix: [f32; 16],
    /// `d_rel_d_h` in nalgebra's column-major storage order. This is only
    /// emitted by the opt-in chain trace while auditing the FEJ Jacobian.
    pub relative_wrt_anchor: [f32; 36],
    pub point4_target: [f32; 4],
    pub point_target: [f32; 3],
    pub projection: [f32; 2],
    /// Projection Jacobian in row-major order.
    pub projection_jacobian: [f32; 6],
}

/// Eigen/GCC's pinned scalar schedule for the float visual residual.  The
/// native `LandmarkBlockAbsDynamic<float, 6>` Double-Sphere path computes
/// `x*x` first, then contracts `y*y + x2` (`vfmadd231ss`); spelling that
/// reduction explicitly keeps the robust Huber boundary at native f32 bits
/// without changing the f64 compatibility path or IMU factors.
#[inline]
fn eigen_vector2_squared_norm_f32(x: f32, y: f32) -> f32 {
    y.mul_add(y, x * x)
}

/// Literal visual row before it is scattered into the global AOM columns.
/// Upstream stores a relative-pose Jacobian and multiplies it by the host and
/// target absolute-pose Jacobians. Keeping those blocks explicit here makes
/// the host contribution (missing from the former world-XYZ factor) auditable.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchoredVisualFactor {
    /// The raw pixel residual (`projection - observation`) before robust
    /// whitening.  This is retained for the frame-4 parity audit; the solver
    /// still consumes `residual` below.
    pub raw_residual: Vector2<f64>,
    /// Projection at the linearization point, in pixel coordinates.
    pub projection: Vector2<f64>,
    /// Unsquared Huber weight used for the row stack.
    pub huber_weight: f64,
    /// Scalar multiplying raw residual/Jacobians (`sqrt(huber_weight) /
    /// observation_stddev`).
    pub sqrt_weight: f64,
    pub residual: Vector2<f64>,
    pub anchor_pose_jacobian: Matrix2x6,
    pub target_pose_jacobian: Matrix2x6,
    /// Columns are `[stereographic_u, stereographic_v, inverse_distance]`.
    pub landmark_jacobian: Matrix2x3<f64>,
    /// The robust objective contribution in Basalt's `computeError`.
    ///
    /// This is intentionally kept separate from `residual.norm_squared()`:
    /// the latter is the quadratic row-stack used by ABS_QR, while Basalt
    /// evaluates Huber rows with `0.5 * (2 - w) * w * ||r||²`.
    pub objective_cost: f64,
    /// Optional current-value f32 chain snapshot for a production-vs-oracle
    /// visual boundary audit.  This is `None` unless the opt-in environment
    /// variable is present.
    pub(crate) debug_chain: Option<VisualChainF32>,
}

/// Port of `ba_utils.h::linearizePoint` plus the absolute host/target pose
/// chain used by `LandmarkBlockAbsDynamic` at the pinned upstream revision.
///
/// Poses are `T_w_i`, extrinsics are `T_i_c`, and Basalt's pose increment is
/// `[world_translation; left_rotation]`. The residual sign is
/// `projection - observation`, exactly as upstream.
#[allow(clippy::too_many_arguments)]
pub fn anchored_visual_reprojection_factor(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    same_time_camera: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    anchored_visual_reprojection_factor_with_time_cam(
        camera,
        anchor_imu_pose,
        anchor_t_imu_cam,
        target_imu_pose,
        target_t_imu_cam,
        landmark,
        observation,
        same_time_camera,
        same_time_camera,
        config,
    )
}

/// Absolute visual factor with the two upstream identity predicates kept
/// separate.  Basalt's absolute landmark block receives a `TimeCamId` for the
/// host and target.  Only equal frame *and* camera takes the exact identity
/// branch; a same-timestamp stereo pair still calls `computeRelPose` and
/// therefore carries both relative pose Jacobian blocks.  The window
/// scatterer later combines those two blocks when both TimeCamIds share one
/// navigation-state column block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn anchored_visual_reprojection_factor_with_time_cam(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    _same_timestamp: bool,
    same_time_cam_id: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    if !config.observation_stddev.is_finite() || config.observation_stddev <= 0.0 {
        return None;
    }

    let target_camera_from_imu = target_t_imu_cam.inverse();
    let target_imu_from_anchor_imu = target_imu_pose.inverse().compose(anchor_imu_pose);
    let target_camera_from_anchor_imu = target_camera_from_imu.compose(&target_imu_from_anchor_imu);
    let target_camera_from_anchor_camera = if same_time_cam_id {
        // `TimeCamId == TimeCamId` is an explicit source branch in
        // linearization_abs_qr.cpp.  Do not compose the four constituent
        // SE(3)s here: even mathematically cancelling f64 operations can
        // leave a tiny rotation/translation that rounds into the landmark
        // Jacobian.  The identity is also independent of the current state
        // and calibration values, as upstream's branch is.
        SE3::identity()
    } else {
        target_camera_from_anchor_imu.compose(anchor_t_imu_cam)
    };

    let bearing = landmark.direction.bearing();
    let point_target = target_camera_from_anchor_camera
        .rotation
        .transform_vector(&bearing)
        + target_camera_from_anchor_camera.translation * landmark.inverse_distance;
    let (predicted, projection_jacobian) =
        project_double_sphere_with_jacobian(camera, point_target)?;
    let raw = predicted - observation.coords;

    // LandmarkBlock applies the Huber norm in raw pixel units, then divides
    // both residual and Jacobians by obs_std_dev.
    let residual_squared = raw.norm_squared();
    let huber_weight =
        if config.huber_delta > 0.0 && residual_squared > config.huber_delta * config.huber_delta {
            config.huber_delta / residual_squared.sqrt()
        } else {
            1.0
        };
    let sqrt_weight = huber_weight.sqrt() / config.observation_stddev;
    let objective_cost =
        robust_objective_from_raw(residual_squared, huber_weight, config.observation_stddev);

    let mut point_wrt_relative_pose = Matrix3x6::zeros();
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(Matrix3::identity() * landmark.inverse_distance));
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew3(point_target)));
    let residual_wrt_relative_pose = projection_jacobian * point_wrt_relative_pose;

    let (relative_wrt_anchor, relative_wrt_target) = if same_time_cam_id {
        // The complete TimeCamId equality branch bypasses computeRelPose and
        // leaves both output Jacobians zero. Keep this explicit and
        // independent of numerical SE(3) cancellation.
        (Matrix6::zeros(), Matrix6::zeros())
    } else {
        let mut anchor_rotation_blocks = Matrix6::zeros();
        let r_w_i_anchor_inv = anchor_imu_pose
            .rotation
            .inverse()
            .to_rotation_matrix()
            .into_inner();
        anchor_rotation_blocks
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&r_w_i_anchor_inv);
        anchor_rotation_blocks
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&r_w_i_anchor_inv);

        let mut target_rotation_blocks = Matrix6::zeros();
        let r_w_i_target_inv = target_imu_pose
            .rotation
            .inverse()
            .to_rotation_matrix()
            .into_inner();
        target_rotation_blocks
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&r_w_i_target_inv);
        target_rotation_blocks
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&r_w_i_target_inv);

        (
            target_camera_from_anchor_imu.adjoint() * anchor_rotation_blocks,
            -(target_camera_from_imu.adjoint() * target_rotation_blocks),
        )
    };

    let direction_jacobian: Matrix3x2<f64> = landmark.direction.bearing_jacobian();
    let mut point_wrt_landmark = Matrix3::zeros();
    point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
        &(target_camera_from_anchor_camera
            .rotation
            .to_rotation_matrix()
            .into_inner()
            * direction_jacobian),
    );
    point_wrt_landmark.set_column(2, &target_camera_from_anchor_camera.translation);

    Some(AnchoredVisualFactor {
        raw_residual: raw,
        projection: predicted,
        huber_weight,
        sqrt_weight,
        residual: raw * sqrt_weight,
        anchor_pose_jacobian: residual_wrt_relative_pose * relative_wrt_anchor * sqrt_weight,
        target_pose_jacobian: residual_wrt_relative_pose * relative_wrt_target * sqrt_weight,
        landmark_jacobian: projection_jacobian * point_wrt_landmark * sqrt_weight,
        objective_cost,
        debug_chain: None,
    })
}

type Matrix3x6 = SMatrix<f64, 3, 6>;

fn skew3(value: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -value.z, value.y, value.z, 0.0, -value.x, -value.y, value.x, 0.0,
    )
}

fn project_double_sphere_with_jacobian(
    camera: &DoubleSphereCamera,
    point: Vector3<f64>,
) -> Option<(Vector2<f64>, Matrix2x3<f64>)> {
    if !point.iter().all(|value| value.is_finite()) {
        return None;
    }
    let d1 = point.norm();
    if d1 <= 1e-12 {
        return None;
    }
    let zeta = camera.xi * d1 + point.z;
    let d2 = (point.x * point.x + point.y * point.y + zeta * zeta).sqrt();
    if d2 <= 1e-12 {
        return None;
    }
    let denominator = camera.alpha * d2 + (1.0 - camera.alpha) * zeta;
    if !denominator.is_finite() || denominator <= 1e-12 {
        return None;
    }

    let d_zeta = Vector3::new(
        camera.xi * point.x / d1,
        camera.xi * point.y / d1,
        camera.xi * point.z / d1 + 1.0,
    );
    let d_d2 = (Vector3::new(point.x, point.y, 0.0) + d_zeta * zeta) / d2;
    let d_denominator = d_d2 * camera.alpha + d_zeta * (1.0 - camera.alpha);
    let denominator_sq = denominator * denominator;
    let mut jacobian = Matrix2x3::zeros();
    for column in 0..3 {
        jacobian[(0, column)] = camera.fx
            * ((if column == 0 { denominator } else { 0.0 }) - point.x * d_denominator[column])
            / denominator_sq;
        jacobian[(1, column)] = camera.fy
            * ((if column == 1 { denominator } else { 0.0 }) - point.y * d_denominator[column])
            / denominator_sq;
    }
    Some((
        Vector2::new(
            camera.fx * point.x / denominator + camera.cx,
            camera.fy * point.y / denominator + camera.cy,
        ),
        jacobian,
    ))
}

/// A minimal float-owned pose used by the upstream compatibility path.  The
/// public `SE3` type is intentionally f64 for API stability; this local pose
/// keeps composition, inverse, adjoint, and point transforms in f32 until the
/// factor crosses back into the f64 row-stack representation.
#[derive(Clone, Copy)]
struct F32Pose {
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
}

impl F32Pose {
    fn from_se3(pose: &SE3) -> Self {
        Self {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                pose.rotation.w as f32,
                pose.rotation.i as f32,
                pose.rotation.j as f32,
                pose.rotation.k as f32,
            )),
            translation: pose.translation.map(|value| value as f32),
        }
    }

    fn compose(self, other: Self) -> Self {
        Self {
            // Sophus::SO3 multiplication constructs a fresh SO3 and
            // normalizes its quaternion. nalgebra's UnitQuaternion product
            // is intentionally unchecked, which otherwise leaks norm drift
            // into every f32 visual factor.
            rotation: sophus_quat_product_f32(self.rotation, other.rotation),
            // Sophus::SO3::operator*(Vector3) uses the explicit
            // `uv = q.vec().cross(p); p + q.w() * (2*uv) + q.vec().cross(2*uv)`
            // action. Eigen's Quaternion::transformVector is algebraically
            // equivalent but has a different f32 association.
            translation: sophus_rotate_f32(self.rotation, other.translation) + self.translation,
        }
    }

    fn inverse(self) -> Self {
        // Sophus::SO3::inverse() constructs a fresh SO3 from the conjugate;
        // that constructor normalizes the f32 quaternion.  A nalgebra
        // UnitQuaternion inverse only conjugates its unchecked coefficients,
        // which leaks the norm error introduced when an f64 state is cast to
        // f32 into the subsequent camera chain.
        let rotation = sophus_so3_inverse(self.rotation);
        Self {
            rotation,
            translation: -sophus_rotate_f32(rotation, self.translation),
        }
    }
}

#[inline]
fn visual_chain_pose_snapshot(pose: F32Pose) -> [f32; 7] {
    let q = pose.rotation.quaternion();
    [
        q.i,
        q.j,
        q.k,
        q.w,
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
    ]
}

fn sophus_relative_imu_f32(target: F32Pose, host: F32Pose) -> F32Pose {
    let target_inverse_rotation = sophus_so3_inverse(target.rotation);
    F32Pose {
        rotation: sophus_quat_product_f32(target_inverse_rotation, host.rotation),
        translation: sophus_rotate_difference_f32(
            target_inverse_rotation,
            host.translation,
            target.translation,
        ),
    }
}

fn sophus_quat_product_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    sophus_so3_product(first, second)
}

/// Packet quaternion product used by the inlined current-pose branch of
/// `computeRelPose` (libbasalt 0x305543..0x3055b8).
///
/// The current relative rotation is formed as `target.inverse() * host`.
/// Eigen does not evaluate that product with the scalar Sophus expression in
/// [`sophus_quat_product_f32`]: it keeps the two signed packet partials in
/// separate registers, blends the lane-3 accumulator after each partial, and
/// only then applies the final cross-term packet.  Keeping this call-site
/// schedule explicit is important because the final normalized quaternion is
/// consumed by the f32 visual chain.
#[inline(never)]
fn sophus_quat_product_current_packet_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let a = first.quaternion();
    let b = second.quaternion();
    let ax = a.i;
    let ay = a.j;
    let az = a.k;
    let aw = a.w;
    let bx = b.i;
    let by = b.j;
    let bz = b.k;
    let bw = b.w;

    // 0x305585..0x30559e: a.vec()*b.w, then the signed b.vec()*a.vec
    // partials.  The arrays are the exact Packet4f lane permutations from
    // the native `vpermilps` instructions.
    let a24 = [ax, ay, az, ax];
    let b3f = [bw, bw, bw, bx];
    let mut product = [
        a24[0] * b3f[0],
        a24[1] * b3f[1],
        a24[2] * b3f[2],
        a24[3] * b3f[3],
    ];
    let a_ff = [aw; 4];
    let b_vec = [bx, by, bz, bw];
    let mut negative = [0.0_f32; 4];
    for lane in 0..4 {
        negative[lane] = b_vec[lane].mul_add(a_ff[lane], -product[lane]);
        product[lane] = b_vec[lane].mul_add(a_ff[lane], product[lane]);
    }
    product[3] = negative[3];

    // 0x3055a8..0x3055b2: second signed packet partial and lane-3 blend.
    let b52 = [bz, bx, by, by];
    let a49 = [ay, az, ax, ay];
    let mut second_negative = [0.0_f32; 4];
    let mut second_positive = [0.0_f32; 4];
    for lane in 0..4 {
        second_negative[lane] = (-b52[lane]).mul_add(a49[lane], product[lane]);
        second_positive[lane] = a49[lane].mul_add(b52[lane], product[lane]);
    }
    product = second_positive;
    product[3] = second_negative[3];

    // 0x3055b8: final signed packet cross-term.
    let b89 = [by, bz, bx, bz];
    let a92 = [az, ax, ay, az];
    for lane in 0..4 {
        product[lane] = (-a92[lane]).mul_add(b89[lane], product[lane]);
    }

    // 0x3055bd..0x3055e4: vmulps followed by the Packet4f pair reduction,
    // scalar sqrt, broadcast, and vdivps.  Spell the reduction as the two
    // packet pairs so it cannot be reassociated with the product lanes.
    let squares = [
        product[0] * product[0],
        product[1] * product[1],
        product[2] * product[2],
        product[3] * product[3],
    ];
    let pair0 = squares[0] + squares[2];
    let pair1 = squares[1] + squares[3];
    let norm = (pair0 + pair1).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        product[3] / norm,
        product[0] / norm,
        product[1] / norm,
        product[2] / norm,
    ))
}

/// Packet inverse/normalization used by the inlined current visual branch
/// before its relative quaternion product.  Keep this separate from the
/// shared Sophus inverse so the current-chain fixture can audit the complete
/// native boundary (the pinned pass-3 input is sensitive to the pairwise
/// reduction's final ulp).
#[inline(never)]
fn sophus_quat_inverse_current_packet_f32(rotation: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    let q = rotation.quaternion();
    let product = [-q.i, -q.j, -q.k, q.w];
    let squares = [
        product[0] * product[0],
        product[1] * product[1],
        product[2] * product[2],
        product[3] * product[3],
    ];
    let pair0 = squares[0] + squares[2];
    let pair1 = squares[1] + squares[3];
    let norm = (pair0 + pair1).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        product[3] / norm,
        product[0] / norm,
        product[1] / norm,
        product[2] / norm,
    ))
}

/// Packet quaternion product emitted by the out-of-line `computeRelPose<float>`
/// body for `tmp = tmp2 * T_t_i_h_i` (2bcbc1..2bcc25).  This is kept separate
/// from the inlined visual-chain product because the native body materializes
/// `b * a.w`, then performs the two signed packet partials and their lane-3
/// blends before the final signed partial.  `mul_add` is the scalar spelling
/// of each packet FMA lane.
#[inline(never)]
fn sophus_quat_product_relpose_out_of_line_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let a = first.quaternion();
    let b = second.quaternion();
    let a24 = [a.i, a.j, a.k, a.i];
    let a49 = [a.j, a.k, a.i, a.j];
    let a92 = [a.k, a.i, a.j, a.k];
    let b3f = [b.w, b.w, b.w, b.i];
    let b52 = [b.k, b.i, b.j, b.j];
    let b89 = [b.j, b.k, b.i, b.k];

    // 2bcb c1: vmulps(relative_q, broadcast(tmp2.w), relative_q).
    let b_times_aw = [b.i * a.w, b.j * a.w, b.k * a.w, b.w * a.w];

    // 2bcbdf..2bcbfb: signed first partial and lane-3 blend.
    let mut first_negative = [0.0_f32; 4];
    let mut first_positive = [0.0_f32; 4];
    for lane in 0..4 {
        first_negative[lane] = (-b3f[lane]).mul_add(a24[lane], b_times_aw[lane]);
        first_positive[lane] = b3f[lane].mul_add(a24[lane], b_times_aw[lane]);
    }
    first_positive[3] = first_negative[3];

    // 2bcc01..2bcc18: second signed partial and lane-3 blend.
    let mut second_negative = [0.0_f32; 4];
    let mut second_positive = [0.0_f32; 4];
    for lane in 0..4 {
        second_negative[lane] = (-b52[lane]).mul_add(a49[lane], first_positive[lane]);
        second_positive[lane] = b52[lane].mul_add(a49[lane], first_positive[lane]);
    }
    second_positive[3] = second_negative[3];

    // 2bcc1e: final signed partial using the saved 0x89 permutation.
    let mut product = [0.0_f32; 4];
    for lane in 0..4 {
        product[lane] = (-a92[lane]).mul_add(b89[lane], second_positive[lane]);
    }

    let x2 = product[0] * product[0];
    let z2 = product[2] * product[2];
    let y2 = product[1] * product[1];
    let w2 = product[3] * product[3];
    let norm = (x2 + z2 + (y2 + w2)).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        product[3] / norm,
        product[0] / norm,
        product[1] / norm,
        product[2] / norm,
    ))
}

/// Scalar lanes interleaved with the out-of-line quaternion packet in
/// `computeRelPose<float>` (2bcb10..2bcc67).  This is Sophus's explicit
/// `q * p` action, but with the exact temporary reuse and FMA association seen
/// in the clean disassembly.  In particular, the first `vunpckhps` leaves
/// `q.z` in the scalar lane used for the initial cross product.
#[inline(never)]
fn sophus_rotate_relpose_out_of_line_f32(
    rotation: UnitQuaternion<f32>,
    point: Vector3<f32>,
) -> Vector3<f32> {
    let q = rotation.quaternion();
    let vx = point.x;
    let vy = point.y;
    let vz = point.z;

    let qz = q.k;
    let mut x5 = qz * vy;
    let mut x4 = vz * q.i;
    let mut x3 = vx * q.j;

    // First cross product, doubled (2bcb34..2bcb77).
    x5 = q.j.mul_add(vz, -x5);
    x4 = qz.mul_add(vx, -x4);
    x3 = vy.mul_add(q.i, -x3);
    x5 += x5;
    x4 += x4;
    x3 += x3;

    // q.vec().cross(2 * uv), using the same scalar temporaries and FMA order.
    let mut x11 = x5 * q.j;
    x11 = x4.mul_add(q.i, -x11);
    let mut x2 = x3 * q.i;
    x2 = qz.mul_add(x5, -x2);
    let qz_x4 = qz * x4;
    let x1 = x3.mul_add(q.j, -qz_x4);

    // Add p + q.w() * (2 * uv), then contract the cross terms.
    x4 = x4.mul_add(q.w, vy);
    x5 = x5.mul_add(q.w, vx);
    x3 = x3.mul_add(q.w, vz);
    let y = x2 + x4;
    let x = x1 + x5;
    let z = x3 + x11;
    Vector3::new(x, y, z)
}

/// Packet schedule used by the step-only native `SO3::operator*(Matrix3x1)`
/// body (libbasalt offset 0x2f0050).  Eigen forms three redundant cross
/// products in packed lanes; the duplicate lanes intentionally retain their
/// own FMA accumulator, so algebraically identical terms can differ by one
/// ulp.  This helper keeps those six scalar lanes separate before the final
/// packet-style cross/add stage.
#[inline(never)]
pub(crate) fn sophus_rotate_step_packet_f32(
    rotation: UnitQuaternion<f32>,
    point: Vector3<f32>,
) -> Vector3<f32> {
    let q = rotation.quaternion();
    let px = point.x;
    let py = point.y;
    let pz = point.z;

    // 0x2f0094..0x2f00b9: three packed partial products.  The duplicated
    // lanes below correspond to different source packets and therefore use
    // different FMA accumulators in the native body.
    let uv_x_a = (-q.k).mul_add(py, q.j * pz);
    let uv_y_a = (-q.i).mul_add(pz, q.k * px);
    let uv_y_b = q.k.mul_add(px, -(q.i * pz));
    let uv_z_b = q.i.mul_add(py, -(q.j * px));
    let uv_z_c = q.i.mul_add(py, -(px * q.j));
    let uv_x_c = q.j.mul_add(pz, -(py * q.k));

    let uv2_x_a = uv_x_a + uv_x_a;
    let uv2_y_a = uv_y_a + uv_y_a;
    let uv2_y_b = uv_y_b + uv_y_b;
    let uv2_z_b = uv_z_b + uv_z_b;
    let uv2_z_c = uv_z_c + uv_z_c;
    let uv2_x_c = uv_x_c + uv_x_c;

    // 0x2f00ca..0x2f00e8: packed x/y cross and scalar z cross.
    let cross_x = q.j.mul_add(uv2_z_c, -(q.k * uv2_y_b));
    let cross_y = q.k.mul_add(uv2_x_c, -(q.i * uv2_z_b));
    let cross_z = q.i.mul_add(uv2_y_b, -(q.j * uv2_x_c));

    // 0x2f00fa..0x2f0108: packet q.w action plus the second cross.
    Vector3::new(
        q.w.mul_add(uv2_x_a, px) + cross_x,
        q.w.mul_add(uv2_y_a, py) + cross_y,
        q.w.mul_add(uv2_z_c, pz) + cross_z,
    )
}

/// Source-faithful FEJ boundary for the out-of-line `computeRelPose<float>`:
/// `tmp2 * (T_w_i_t.inverse() * T_w_i_h)`.  The production FEJ path calls
/// this with the frozen target/anchor IMU poses so no current-pose value-chain
/// rounding can enter the relative Jacobian boundary.
#[inline(never)]
fn sophus_compute_relpose_tmp_out_of_line_f32(
    tmp2: F32Pose,
    target_imu: F32Pose,
    host_imu: F32Pose,
) -> F32Pose {
    let target_inverse_rotation = sophus_so3_inverse(target_imu.rotation);
    let relative_rotation = sophus_quat_product_f32(target_inverse_rotation, host_imu.rotation);
    let relative_translation = sophus_rotate_difference_f32(
        target_inverse_rotation,
        host_imu.translation,
        target_imu.translation,
    );
    F32Pose {
        rotation: sophus_quat_product_relpose_out_of_line_f32(tmp2.rotation, relative_rotation),
        translation: sophus_rotate_relpose_out_of_line_f32(tmp2.rotation, relative_translation)
            + tmp2.translation,
    }
}

/// Packet product used by the inlined `computeRelPose` camera prefix.  Eigen's
/// fixed `Packet4f` product keeps the first scalar product as the accumulator,
/// performs each following term as an FMA, and blends the w lane after the
/// two signed/unsigned partial products.  This is intentionally separate from
/// the generic Sophus product: the native compiler emits a different packet
/// schedule at this call site.
fn sophus_quat_product_camera_prefix_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let a = first.quaternion();
    let b = second.quaternion();
    let ax = a.i;
    let ay = a.j;
    let az = a.k;
    let aw = a.w;
    let bx = b.i;
    let by = b.j;
    let bz = b.k;
    let bw = b.w;
    let c24 = [ax, ay, az, ax];
    let p3f = [bw, bw, bw, bx];
    let mut x5 = [0.0_f32; 4];
    let mut x8 = [aw * bx, aw * by, aw * bz, aw * bw];
    for lane in 0..4 {
        x5[lane] = (-p3f[lane]).mul_add(c24[lane], x8[lane]);
        x8[lane] = p3f[lane].mul_add(c24[lane], x8[lane]);
    }
    x8[3] = x5[3];
    let p52 = [bz, bx, by, by];
    let c49 = [ay, az, ax, ay];
    let mut x2 = [0.0_f32; 4];
    for lane in 0..4 {
        x2[lane] = (-p52[lane]).mul_add(c49[lane], x8[lane]);
        x8[lane] = p52[lane].mul_add(c49[lane], x8[lane]);
    }
    x8[3] = x2[3];
    let p89 = [by, bz, bx, bz];
    let c92 = [az, ax, ay, az];
    for lane in 0..4 {
        x8[lane] = (-p89[lane]).mul_add(c92[lane], x8[lane]);
    }
    let pair0 = x8[0] * x8[0] + x8[2] * x8[2];
    let pair1 = x8[1] * x8[1] + x8[3] * x8[3];
    let norm = (pair0 + pair1).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        x8[3] / norm,
        x8[0] / norm,
        x8[1] / norm,
        x8[2] / norm,
    ))
}

/// Packet product used by the inlined host-extrinsic suffix of
/// `computeRelPose`.  This is the corresponding Eigen register schedule for
/// `q_prefix * q_host`; it has a distinct first partial product from the
/// camera prefix above.
fn sophus_quat_product_camera_suffix_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let a = first.quaternion();
    let b = second.quaternion();
    let ax = a.i;
    let ay = a.j;
    let az = a.k;
    let aw = a.w;
    let bx = b.i;
    let by = b.j;
    let bz = b.k;
    let bw = b.w;

    let a24 = [ax, ay, az, ax];
    let b3f = [bw, bw, bw, bx];
    let mut x1 = [
        a24[0] * b3f[0],
        a24[1] * b3f[1],
        a24[2] * b3f[2],
        a24[3] * b3f[3],
    ];
    let a_ff = [aw; 4];
    let b_vec = [bx, by, bz, bw];
    let mut x11 = [0.0_f32; 4];
    let mut x8 = [0.0_f32; 4];
    for lane in 0..4 {
        x11[lane] = b_vec[lane].mul_add(a_ff[lane], -x1[lane]);
        x8[lane] = b_vec[lane].mul_add(a_ff[lane], x1[lane]);
    }
    x8[3] = x11[3];

    let b52 = [bz, bx, by, by];
    let a49 = [ay, az, ax, ay];
    for lane in 0..4 {
        x1[lane] = (-b52[lane]).mul_add(a49[lane], x8[lane]);
        x8[lane] = a49[lane].mul_add(b52[lane], x8[lane]);
    }
    x8[3] = x1[3];

    let b89 = [by, bz, bx, bz];
    let a92 = [az, ax, ay, az];
    for lane in 0..4 {
        x8[lane] = (-b89[lane]).mul_add(a92[lane], x8[lane]);
    }
    let pair0 = x8[0] * x8[0] + x8[2] * x8[2];
    let pair1 = x8[1] * x8[1] + x8[3] * x8[3];
    let norm = (pair0 + pair1).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        x8[3] / norm,
        x8[0] / norm,
        x8[1] / norm,
        x8[2] / norm,
    ))
}

fn skew3_f32(value: Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(
        0.0, -value.z, value.y, value.z, 0.0, -value.x, -value.y, value.x, 0.0,
    )
}

/// Fixed-size `3x3 * 3x3` reduction used by Eigen for Sophus adjoint blocks.
/// Keep the first product as the accumulator and contract the next two terms
/// in source order; nalgebra's generic product can choose a different f32
/// association for these small blocks.
fn eigen_matrix_product_3x3_f32(left: Matrix3<f32>, right: Matrix3<f32>) -> Matrix3<f32> {
    let mut result = Matrix3::<f32>::zeros();
    for row in 0..3 {
        for column in 0..3 {
            let value = left[(row, 1)] * right[(1, column)];
            let value = left[(row, 2)].mul_add(right[(2, column)], value);
            result[(row, column)] = left[(row, 0)].mul_add(right[(0, column)], value);
        }
    }
    result
}

/// Eigen's fixed-size six-term packet reduction.  The native `6x6` product
/// processes four output lanes together and contracts the six terms in
/// increasing k order; even when a block-diagonal right operand makes terms
/// zero, it does not collapse to a smaller `3x3` product.
fn eigen_matrix_product_6x6_f32(
    left: SMatrix<f32, 6, 6>,
    right: SMatrix<f32, 6, 6>,
) -> SMatrix<f32, 6, 6> {
    let mut result = SMatrix::<f32, 6, 6>::zeros();
    for row in 0..6 {
        for column in 0..6 {
            let value = left[(row, 0)] * right[(0, column)];
            let value = left[(row, 1)].mul_add(right[(1, column)], value);
            let value = left[(row, 2)].mul_add(right[(2, column)], value);
            let value = left[(row, 3)].mul_add(right[(3, column)], value);
            let value = left[(row, 4)].mul_add(right[(4, column)], value);
            result[(row, column)] = left[(row, 5)].mul_add(right[(5, column)], value);
        }
    }
    result
}

/// Evaluate the fixed-size Eigen product used to chain a visual residual
/// Jacobian (`2x6`) into an absolute-pose Jacobian (`6x6`).
///
/// The pinned x86 Eigen assignment kernel does not reduce the six terms in
/// increasing-k order.  Its two-row packet/tail schedule forms the two
/// three-term groups `(k4 + k5) + k3` and `(k1 + k2) + k0`, then adds those
/// groups.  Although the two trees are algebraically equivalent, their f32
/// rounding (and signed-zero propagation) is observable in the ABS_QR rows.
/// Keep this helper separate from the ordinary nalgebra product so the source
/// boundary remains explicit and testable.
fn eigen_matrix_product_2x6_f32(
    left: SMatrix<f32, 2, 6>,
    right: SMatrix<f32, 6, 6>,
) -> SMatrix<f32, 2, 6> {
    let mut result = SMatrix::<f32, 2, 6>::zeros();
    for row in 0..2 {
        for column in 0..6 {
            // This is the exact scalar spelling of Eigen's `2x6 * 6x6`
            // packet assignment observed in the pinned clean libbasalt:
            // (k4 + k5) + k3, then (k1 + k2) + k0, then the pair add.
            let high = left[(row, 4)] * right[(4, column)];
            let high = left[(row, 5)].mul_add(right[(5, column)], high);
            let high = left[(row, 3)].mul_add(right[(3, column)], high);
            let low = left[(row, 1)] * right[(1, column)];
            let low = left[(row, 2)].mul_add(right[(2, column)], low);
            let low = left[(row, 0)].mul_add(right[(0, column)], low);
            result[(row, column)] = high + low;
        }
    }
    result
}

/// Source-faithful absolute-pose chain for one whitened visual row.
///
/// `LandmarkBlockAbsDynamic` first scales its relative residual Jacobian in
/// place and only then evaluates the fixed `2x6 * 6x6` product.  Keeping the
/// scalar multiplication before [`eigen_matrix_product_2x6_f32`] matters for
/// the final binary32 lanes; multiplying the completed product instead is a
/// different rounding path.
fn eigen_weighted_pose_jacobian_f32(
    relative_pose_jacobian: SMatrix<f32, 2, 6>,
    absolute_pose_jacobian: SMatrix<f32, 6, 6>,
    sqrt_weight: f32,
) -> SMatrix<f32, 2, 6> {
    let mut weighted_relative = relative_pose_jacobian;
    for value in weighted_relative.as_mut_slice() {
        *value *= sqrt_weight;
    }
    eigen_matrix_product_2x6_f32(weighted_relative, absolute_pose_jacobian)
}

/// Compute a signed Sophus `Adj() * diag(R, R)` while retaining the source
/// fixed block/product boundaries used by `computeRelPose` for both absolute
/// pose derivatives.
///
/// The sign belongs to the left operand in the upstream expression.  In
/// particular, `-tmp2.Adj() * RR_t` is not equivalent at the binary32 bit
/// level to negating the completed product: the latter changes the signs of
/// zeros produced by the six-term Eigen reduction.  Keeping the sign here
/// makes that ordering explicit without introducing a factor- or
/// observation-specific path.
fn eigen_adjoint_times_rotation_blocks_f32(
    pose: F32Pose,
    rotation: Matrix3<f32>,
    left_sign: f32,
) -> SMatrix<f32, 6, 6> {
    let pose_rotation = eigen_quaternion_matrix_f32(pose.rotation);
    let cross_rotation = eigen_matrix_product_3x3_f32(skew3_f32(pose.translation), pose_rotation);
    let mut adjoint = SMatrix::<f32, 6, 6>::zeros();
    adjoint
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&pose_rotation);
    adjoint
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&cross_rotation);
    adjoint
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&pose_rotation);
    let mut rotation_blocks = SMatrix::<f32, 6, 6>::zeros();
    rotation_blocks
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation);
    rotation_blocks
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&rotation);
    eigen_matrix_product_6x6_f32(adjoint * left_sign, rotation_blocks)
}

fn sophus_rotate_f32(rotation: UnitQuaternion<f32>, point: Vector3<f32>) -> Vector3<f32> {
    let q = rotation.quaternion();
    // Sophus `SO3Base::operator*(Point)` is `p + q.w() * uv +
    // q.vec().cross(uv)` after `uv = q.vec().cross(p); uv += uv`.
    // Spell the pinned native cross/FMA lane order explicitly.
    let uv_x = q.j.mul_add(point.z, -(q.k * point.y));
    let uv_y = q.k.mul_add(point.x, -(q.i * point.z));
    let uv_z = q.i.mul_add(point.y, -(q.j * point.x));
    let uv2_x = uv_x + uv_x;
    let uv2_y = uv_y + uv_y;
    let uv2_z = uv_z + uv_z;
    let cross_x = q.j.mul_add(uv2_z, -(q.k * uv2_y));
    let cross_y = q.k.mul_add(uv2_x, -(q.i * uv2_z));
    let cross_z = q.i.mul_add(uv2_y, -(q.j * uv2_x));
    Vector3::new(
        q.w.mul_add(uv2_x, point.x) + cross_x,
        q.w.mul_add(uv2_y, point.y) + cross_y,
        q.w.mul_add(uv2_z, point.z) + cross_z,
    )
}

/// Eigen's packetized `SO3::operator*(host_translation - target_translation)`
/// path used by `computeRelPose`.  This differs from the scalar `Vector3`
/// overload above at the intermediate cross-product boundary: Eigen keeps
/// three independently packed products, then contracts each signed pair with
/// a distinct FMA.  Keep the difference fused into this schedule instead of
/// materializing a `Vector3` before calling the generic action.
fn sophus_rotate_difference_f32(
    rotation: UnitQuaternion<f32>,
    first: Vector3<f32>,
    second: Vector3<f32>,
) -> Vector3<f32> {
    let q = rotation.quaternion();
    let dx = first.x - second.x;
    let dy = first.y - second.y;
    let dz = first.z - second.z;

    // The packet layout is [x,y], [y,z], and [z,x].  Keep each independently
    // formed lane: Eigen's three products are not common-subexpression
    // eliminated, and their duplicate cross terms can round differently.
    let uv_a_x = (-q.k).mul_add(dy, q.j * dz);
    let uv_a_y = (-q.i).mul_add(dz, q.k * dx);
    let uv_b_y = q.k.mul_add(dx, -(q.i * dz));
    let uv_b_z = q.i.mul_add(dy, -(q.j * dx));
    let uv_c_z = q.i.mul_add(dy, -(q.j * dx));
    let uv_c_x = q.j.mul_add(dz, -(q.k * dy));

    let uv2_a_x = uv_a_x + uv_a_x;
    let uv2_a_y = uv_a_y + uv_a_y;
    let uv2_b_y = uv_b_y + uv_b_y;
    let uv2_b_z = uv_b_z + uv_b_z;
    let uv2_c_z = uv_c_z + uv_c_z;
    let uv2_c_x = uv_c_x + uv_c_x;

    // Second cross packet: [cross.x, cross.y].
    let cross_x = q.j.mul_add(uv2_c_z, -(q.k * uv2_b_y));
    let cross_y = q.k.mul_add(uv2_c_x, -(q.i * uv2_b_z));
    let cross_z = q.i.mul_add(uv2_b_y, -(q.j * uv2_c_x));

    Vector3::new(
        q.w.mul_add(uv2_a_x, dx) + cross_x,
        q.w.mul_add(uv2_a_y, dy) + cross_y,
        q.w.mul_add(uv2_c_z, dz) + cross_z,
    )
}

/// Current-pose visual relinearization uses a distinct inline Eigen path from
/// the FEJ/relative-Jacobian `computeRelPose` call.  The inline path first
/// materializes `T_w_i_h.translation() - T_w_i_t.translation()` and then
/// applies the quaternion action.  Keeping that difference as a concrete
/// vector is observable at the f32 boundary (frame 4's prefix `t.y` is two
/// ulps below the packetized FEJ helper above), so do not reuse
/// [`sophus_rotate_difference_f32`] here.
fn sophus_rotate_difference_visual_f32(
    rotation: UnitQuaternion<f32>,
    first: Vector3<f32>,
    second: Vector3<f32>,
) -> Vector3<f32> {
    let difference = first - second;
    sophus_rotate_f32(rotation, difference)
}

/// Eigen::QuaternionBase::toRotationMatrix operation order.
///
/// nalgebra's `UnitQuaternion::to_rotation_matrix` is algebraically the same
/// conversion, but its expression tree is not the one used by the pinned
/// Eigen implementation at the f32 boundary.  In particular, the native
/// fixed-size packet path contracts the two x-diagonal products before the
/// subtraction from one.  Keep that operation boundary explicit for every
/// upstream quaternion-to-matrix call, including the d_rel rotation blocks.
fn eigen_quaternion_matrix_f32(rotation: UnitQuaternion<f32>) -> Matrix3<f32> {
    let q = rotation.quaternion();
    let tx = 2.0_f32 * q.i;
    let ty = 2.0_f32 * q.j;
    let tz = 2.0_f32 * q.k;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;
    Matrix3::new(
        1.0_f32 - (tyy + tzz),
        (-tz).mul_add(q.w, txy),
        ty.mul_add(q.w, txz),
        tz.mul_add(q.w, txy),
        1.0_f32 - q.i.mul_add(tx, tzz),
        (-tx).mul_add(q.w, tyz),
        (-ty).mul_add(q.w, txz),
        tx.mul_add(q.w, tyz),
        1.0_f32 - q.i.mul_add(tx, tyy),
    )
}

/// Eigen/Sophus `QuaternionBase::toRotationMatrix` scalar schedule used by
/// the native relative-pose temporary.  Eigen materializes the doubled
/// quaternion products, then the compiler contracts each signed cross term
/// as one multiply-add.  Keep the sign on the multiplicand (rather than
/// materializing `tw*` and subtracting afterward): that is the f32 boundary
/// exposed by the pinned `T_t_h` packets.
fn eigen_quaternion_matrix_native_f32(rotation: UnitQuaternion<f32>) -> Matrix3<f32> {
    let q = rotation.quaternion();
    let tx = 2.0_f32 * q.i;
    let ty = 2.0_f32 * q.j;
    let tz = 2.0_f32 * q.k;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;
    Matrix3::new(
        1.0_f32 - (tyy + tzz),
        (-tz).mul_add(q.w, txy),
        ty.mul_add(q.w, txz),
        tz.mul_add(q.w, txy),
        1.0_f32 - q.i.mul_add(tx, tzz),
        (-tx).mul_add(q.w, tyz),
        (-ty).mul_add(q.w, txz),
        tx.mul_add(q.w, tyz),
        1.0_f32 - q.i.mul_add(tx, tyy),
    )
}

/// Reproduce the fixed-size `3x4 * 4x2` reduction used for the stereographic
/// part of `linearizePoint`'s homogeneous landmark block.  `source_jup` is
/// deliberately four rows wide: the fourth row is homogeneous and zero, but
/// Eigen still evaluates the four-term product.  Its scalar assignment kernel
/// starts the `(k2,k3)` and `(k0,k1)` pairs separately, contracts the second
/// term of each pair, and then adds the pair sums.  A `3x3 * 3x2` nalgebra
/// product has a different f32 association and can move a single Jpp lane.
fn eigen_homogeneous_landmark_direction_f32(
    transform_top_left: SMatrix<f32, 3, 4>,
    source_jup: SMatrix<f32, 4, 2>,
) -> SMatrix<f32, 3, 2> {
    let mut result = SMatrix::<f32, 3, 2>::zeros();
    for row in 0..3 {
        for column in 0..2 {
            let pair23 = transform_top_left[(row, 2)] * source_jup[(2, column)];
            let pair23 = transform_top_left[(row, 3)].mul_add(source_jup[(3, column)], pair23);
            let pair01 = transform_top_left[(row, 0)] * source_jup[(0, column)];
            let pair01 = transform_top_left[(row, 1)].mul_add(source_jup[(1, column)], pair01);
            result[(row, column)] = pair23 + pair01;
        }
    }
    result
}

/// Apply the pinned `ba_utils.h::linearizePoint` homogeneous action:
/// `T_t_h * [bearing; inverse_distance]`.  Keeping the 4x4 product explicit
/// matters because the Eigen matrix path rounds differently from the
/// algebraically equivalent `R * bearing + t * rho` spelling.
#[cfg(test)]
fn sophus_homogeneous_point_f32(
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
    bearing: Vector3<f32>,
    inverse_distance: f32,
) -> Vector3<f32> {
    let rotation = eigen_quaternion_matrix_native_f32(rotation);
    let mut transform = SMatrix::<f32, 4, 4>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);
    let mut homogeneous = SMatrix::<f32, 4, 1>::zeros();
    homogeneous.fixed_view_mut::<3, 1>(0, 0).copy_from(&bearing);
    homogeneous[(3, 0)] = inverse_distance;
    (transform * homogeneous).fixed_rows::<3>(0).into_owned()
}

/// `SE3::matrix()` consumes the unit quaternion already held by Sophus.  Keep
/// the stored `UnitQuaternion` unchanged at this boundary: renormalizing its
/// coefficients here would create a second f32 rounding point that is absent
/// from the native matrix conversion.
#[cfg(test)]
fn sophus_homogeneous_point_normalized_rotation_f32(
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
    bearing: Vector3<f32>,
    inverse_distance: f32,
) -> Vector3<f32> {
    eigen_homogeneous_point_gemv_f32(rotation, translation, bearing, inverse_distance)
        .fixed_rows::<3>(0)
        .into_owned()
}

/// Evaluate the source-shaped Eigen product used by
/// `linearizePoint` for the target point:
///
/// ```text
/// T_t_h.matrix() * [bearing.x, bearing.y, bearing.z, inverse_distance]
/// ```
///
/// The native code materializes the homogeneous `4x4` transform and invokes
/// Eigen's fixed-size `4x4 * 4x1` product.  Keep that boundary explicit here:
/// Keep the fixed-size product materialized rather than spelling it as
/// `R * bearing + t * rho`.  In particular, the `k3` term is the homogeneous
/// translation-times-rho lane, and the fourth output row is retained as part
/// of the product even though callers generally consume only xyz.
fn eigen_homogeneous_point_gemv_f32(
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
    bearing: Vector3<f32>,
    inverse_distance: f32,
) -> SMatrix<f32, 4, 1> {
    let transform = eigen_homogeneous_transform_f32(rotation, translation);
    eigen_homogeneous_point_product_f32(transform, bearing, inverse_distance)
}

fn eigen_homogeneous_transform_f32(
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
) -> SMatrix<f32, 4, 4> {
    let rotation = eigen_quaternion_matrix_native_f32(rotation);
    let mut transform = SMatrix::<f32, 4, 4>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}

fn eigen_homogeneous_point_product_f32(
    transform: SMatrix<f32, 4, 4>,
    bearing: Vector3<f32>,
    inverse_distance: f32,
) -> SMatrix<f32, 4, 1> {
    // Eigen's column-major Packet4 kernel keeps one accumulator lane per
    // output row. It seeds all four lanes from column zero, then folds the
    // remaining columns with one FMA per lane. This is not the horizontal
    // even/odd reduction used by Eigen's row-dot kernels.
    let x = bearing.x;
    let y = bearing.y;
    let z = bearing.z;
    let rho = inverse_distance;
    let mut accumulator = [
        transform[(0, 0)] * x,
        transform[(1, 0)] * x,
        transform[(2, 0)] * x,
        transform[(3, 0)] * x,
    ];
    for row in 0..4 {
        accumulator[row] = transform[(row, 1)].mul_add(y, accumulator[row]);
        accumulator[row] = transform[(row, 2)].mul_add(z, accumulator[row]);
        accumulator[row] = transform[(row, 3)].mul_add(rho, accumulator[row]);
    }
    SMatrix::<f32, 4, 1>::from_column_slice(&accumulator)
}

fn project_double_sphere_with_jacobian_f32(
    camera: &DoubleSphereCamera,
    point: Vector3<f32>,
) -> Option<(Vector2<f32>, SMatrix<f32, 2, 3>)> {
    if !point.iter().all(|value| value.is_finite()) {
        return None;
    }
    let x = point.x;
    let y = point.y;
    let z = point.z;
    let xx = x * x;
    let yy = y * y;
    let r2 = xx + yy;
    // Keep the native DoubleSphere `d1_2` sequence materialized: Eigen's
    // scalar path rounds `z*z` and then adds it to `r2` separately, rather
    // than contracting the sum into one FMA.  This boundary feeds both the
    // projection and every camera-Jacobian derivative below.
    let zz = z * z;
    let d1_2 = r2 + zz;
    let d1 = d1_2.sqrt();
    if d1 <= f32::EPSILON {
        return None;
    }
    let xi = camera.xi as f32;
    let alpha = camera.alpha as f32;
    let fx = camera.fx as f32;
    let fy = camera.fy as f32;
    let cx = camera.cx as f32;
    let cy = camera.cy as f32;
    let w1 = if alpha > 0.5_f32 {
        (1.0_f32 - alpha) / alpha
    } else {
        alpha / (1.0_f32 - alpha)
    };
    let w2_denominator = 2.0_f32 * w1 * xi + xi * xi + 1.0_f32;
    let w2 = (w1 + xi) / w2_denominator.sqrt();
    let valid = z > -w2 * d1;

    let k = xi.mul_add(d1, z);
    let d2 = k.mul_add(k, r2).sqrt();
    let one_minus_alpha = 1.0_f32 - alpha;
    let norm = alpha.mul_add(d2, one_minus_alpha * k);
    if !valid || !norm.is_finite() || norm <= f32::EPSILON {
        return None;
    }
    let mx = x / norm;
    let my = y / norm;
    let norm_sq = norm * norm;
    let xy = x * y;
    let tt2 = xi * z / d1 + 1.0_f32;
    let d_norm_d_r2 =
        (xi * (1.0_f32 - alpha) / d1 + alpha * (xi * k / d1 + 1.0_f32) / d2) / norm_sq;
    let tmp2_numerator = tt2.mul_add(one_minus_alpha, alpha * k * tt2 / d2);
    let tmp2 = tmp2_numerator / norm_sq;
    let mut jacobian = SMatrix::<f32, 2, 3>::zeros();
    jacobian[(0, 0)] = (-xx).mul_add(d_norm_d_r2, 1.0_f32 / norm) * fx;
    jacobian[(1, 0)] = -fy * xy * d_norm_d_r2;
    jacobian[(0, 1)] = -fx * xy * d_norm_d_r2;
    jacobian[(1, 1)] = (-yy).mul_add(d_norm_d_r2, 1.0_f32 / norm) * fy;
    jacobian[(0, 2)] = -fx * x * tmp2;
    jacobian[(1, 2)] = -fy * y * tmp2;
    Some((
        Vector2::new(fx.mul_add(mx, cx), fy.mul_add(my, cy)),
        jacobian,
    ))
}

/// Reproduce Eigen's fixed-size `2x4 * 4x3` reduction used by
/// `linearizePoint` for the landmark block.
///
/// Eigen evaluates the first four column-major output lanes as one Packet4f:
/// `[row0-col0, row1-col0, row0-col1, row1-col1]`.  Its fixed-size product
/// computes the two pair reductions `(k0 + k1)` and `(k2 + k3)` with the
/// second term contracted, then adds those pair results.  The final column is
/// the scalar tail with the same two FMA/add pair reductions.  Keep the
/// homogeneous fourth row and column in the inputs even when they are zero;
/// collapsing this to a `2x3 * 3x3` product changes f32 lanes.
fn eigen_landmark_jacobian_f32(
    projection_jacobian: SMatrix<f32, 2, 4>,
    point_wrt_landmark: SMatrix<f32, 4, 3>,
) -> SMatrix<f32, 2, 3> {
    // This is the scalar spelling of Eigen's Packet4f operation.  Arrays are
    // deliberately used instead of target-specific intrinsics so the helper
    // remains portable while preserving the packet lane order and operation
    // grouping in source.
    let pair01 = [
        projection_jacobian[(0, 1)].mul_add(
            point_wrt_landmark[(1, 0)],
            projection_jacobian[(0, 0)] * point_wrt_landmark[(0, 0)],
        ),
        projection_jacobian[(1, 1)].mul_add(
            point_wrt_landmark[(1, 0)],
            projection_jacobian[(1, 0)] * point_wrt_landmark[(0, 0)],
        ),
        projection_jacobian[(0, 1)].mul_add(
            point_wrt_landmark[(1, 1)],
            projection_jacobian[(0, 0)] * point_wrt_landmark[(0, 1)],
        ),
        projection_jacobian[(1, 1)].mul_add(
            point_wrt_landmark[(1, 1)],
            projection_jacobian[(1, 0)] * point_wrt_landmark[(0, 1)],
        ),
    ];
    let pair23 = [
        projection_jacobian[(0, 3)].mul_add(
            point_wrt_landmark[(3, 0)],
            projection_jacobian[(0, 2)] * point_wrt_landmark[(2, 0)],
        ),
        projection_jacobian[(1, 3)].mul_add(
            point_wrt_landmark[(3, 0)],
            projection_jacobian[(1, 2)] * point_wrt_landmark[(2, 0)],
        ),
        projection_jacobian[(0, 3)].mul_add(
            point_wrt_landmark[(3, 1)],
            projection_jacobian[(0, 2)] * point_wrt_landmark[(2, 1)],
        ),
        projection_jacobian[(1, 3)].mul_add(
            point_wrt_landmark[(3, 1)],
            projection_jacobian[(1, 2)] * point_wrt_landmark[(2, 1)],
        ),
    ];
    let packet = [
        pair01[0] + pair23[0],
        pair01[1] + pair23[1],
        pair01[2] + pair23[2],
        pair01[3] + pair23[3],
    ];

    let tail_pair01 = projection_jacobian[(0, 1)].mul_add(
        point_wrt_landmark[(1, 2)],
        projection_jacobian[(0, 0)] * point_wrt_landmark[(0, 2)],
    );
    let tail_pair23 = projection_jacobian[(0, 3)].mul_add(
        point_wrt_landmark[(3, 2)],
        projection_jacobian[(0, 2)] * point_wrt_landmark[(2, 2)],
    );
    let tail_row0 = tail_pair01 + tail_pair23;
    let tail_pair01 = projection_jacobian[(1, 1)].mul_add(
        point_wrt_landmark[(1, 2)],
        projection_jacobian[(1, 0)] * point_wrt_landmark[(0, 2)],
    );
    let tail_pair23 = projection_jacobian[(1, 3)].mul_add(
        point_wrt_landmark[(3, 2)],
        projection_jacobian[(1, 2)] * point_wrt_landmark[(2, 2)],
    );
    let tail_row1 = tail_pair01 + tail_pair23;

    let mut result = SMatrix::<f32, 2, 3>::zeros();
    result[(0, 0)] = packet[0];
    result[(1, 0)] = packet[1];
    result[(0, 1)] = packet[2];
    result[(1, 1)] = packet[3];
    result[(0, 2)] = tail_row0;
    result[(1, 2)] = tail_row1;
    result
}

/// Reproduce the same Eigen `2x4 * 4x6` reduction used for the relative-pose
/// block returned by `linearizePoint`.  The fourth projection/point row is
/// homogeneous and zero, but Eigen still keeps it in the fixed-size product's
/// reduction tree.  The pairwise `(k0 + k1) + (k2 + k3)` order is observably
/// different from nalgebra's left-associated `2x3 * 3x6` product in f32.
fn eigen_relative_pose_jacobian_f32(
    projection_jacobian: SMatrix<f32, 2, 3>,
    point_wrt_relative_pose: SMatrix<f32, 3, 6>,
) -> SMatrix<f32, 2, 6> {
    let mut result = SMatrix::<f32, 2, 6>::zeros();
    for row in 0..2 {
        for column in 0..6 {
            let pair01 = projection_jacobian[(row, 0)] * point_wrt_relative_pose[(0, column)];
            let pair01 =
                projection_jacobian[(row, 1)].mul_add(point_wrt_relative_pose[(1, column)], pair01);
            let pair23 = projection_jacobian[(row, 2)] * point_wrt_relative_pose[(2, column)];
            result[(row, column)] = pair01 + pair23;
        }
    }
    result
}

/// Float-owned counterpart of [`anchored_visual_reprojection_factor`].  This
/// is deliberately separate from the f64 API so ExtendedF64 retains its
/// historical path and unit-test tolerances.
#[allow(clippy::too_many_arguments)]
pub fn anchored_visual_reprojection_factor_f32(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    same_time_camera: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    anchored_visual_reprojection_factor_f32_with_time_cam(
        camera,
        anchor_imu_pose,
        anchor_t_imu_cam,
        target_imu_pose,
        target_t_imu_cam,
        landmark,
        observation,
        same_time_camera,
        same_time_camera,
        config,
    )
}

/// Float-owned counterpart of
/// [`anchored_visual_reprojection_factor_with_time_cam`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn anchored_visual_reprojection_factor_f32_with_time_cam(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    _same_timestamp: bool,
    same_time_cam_id: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    anchored_visual_reprojection_factor_f32_with_time_cam_fej(
        camera,
        anchor_imu_pose,
        anchor_imu_pose,
        anchor_t_imu_cam,
        target_imu_pose,
        target_imu_pose,
        target_t_imu_cam,
        landmark,
        observation,
        _same_timestamp,
        same_time_cam_id,
        config,
    )
}

/// Float visual factor with the upstream split between the current value
/// state and the FEJ state used for pose derivatives.
///
/// The value-side camera chain remains the source for the residual and
/// landmark/value Jacobian.  The anchor absolute-pose block uses the separate
/// FEJ endpoint together with the FEJ host inverse rotation, while the target
/// block retains its existing target-side FEJ rotation input.  This keeps the
/// schedule distinction explicit: the single-host/target endpoint fixture
/// does not contain a distinct FEJ state, so it cannot by itself select this
/// production endpoint semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn anchored_visual_reprojection_factor_f32_with_time_cam_fej(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_imu_pose_fej: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_imu_pose_fej: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    _same_timestamp: bool,
    same_time_cam_id: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    // Compatibility callers have no lifecycle flags; retain their explicit
    // current-value contract. Window callers use the lifecycle-aware entry.
    anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
        camera,
        anchor_imu_pose,
        anchor_imu_pose_fej,
        anchor_t_imu_cam,
        target_imu_pose,
        target_imu_pose_fej,
        target_t_imu_cam,
        landmark,
        observation,
        _same_timestamp,
        same_time_cam_id,
        true,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_imu_pose_fej: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_imu_pose_fej: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    _same_timestamp: bool,
    same_time_cam_id: bool,
    either_endpoint_linearized: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    let observation_stddev = config.observation_stddev as f32;
    if !observation_stddev.is_finite() || observation_stddev <= 0.0 {
        return None;
    }
    let anchor_imu_pose = F32Pose::from_se3(anchor_imu_pose);
    let anchor_imu_pose_fej = F32Pose::from_se3(anchor_imu_pose_fej);
    let anchor_t_imu_cam = F32Pose::from_se3(anchor_t_imu_cam);
    let target_imu_pose = F32Pose::from_se3(target_imu_pose);
    let target_imu_pose_fej = F32Pose::from_se3(target_imu_pose_fej);
    let target_t_imu_cam = F32Pose::from_se3(target_t_imu_cam);
    let target_camera_from_imu = target_t_imu_cam.inverse();
    // Keep the translation expression in the same source order as
    // `computeRelPose`: form the world-frame difference first, then apply the
    // target inverse rotation.  Building `target_pose.inverse()` and composing
    // the host pose is algebraically equivalent, but its two point actions
    // expose a different f32 rounding path for small cross-time translations.
    let target_inverse_rotation = sophus_so3_inverse(target_imu_pose.rotation);
    let target_imu_from_anchor_imu = F32Pose {
        // The current value-side computeRelPose branch is inlined into
        // linearizePoint in the pinned Eigen build.  Its relative quaternion
        // product has a distinct Packet4f/FMA lane schedule from the
        // out-of-line FEJ product; keep this substitution local to the
        // current chain so d_rel and all other products retain their source
        // paths.
        rotation: sophus_quat_product_current_packet_f32(
            target_inverse_rotation,
            anchor_imu_pose.rotation,
        ),
        translation: sophus_rotate_difference_visual_f32(
            target_inverse_rotation,
            anchor_imu_pose.translation,
            target_imu_pose.translation,
        ),
    };
    let target_camera_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.translation,
        ) + target_camera_from_imu.translation,
    };
    // Keep the value-side camera chain above for the residual, landmark
    // projection, and target-camera endpoint.  The separately reconstructed
    // FEJ endpoint below is the production source for the anchor absolute-pose
    // block.  Its inputs are frozen FEJ poses, but its arithmetic intentionally
    // retains the prepatch packet-quaternion/camera-prefix schedule so this
    // control isolates endpoint inputs from endpoint rounding order.
    let target_camera_from_anchor_imu_fej = if same_time_cam_id {
        F32Pose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    } else {
        let target_inverse_rotation_fej = sophus_so3_inverse(target_imu_pose_fej.rotation);
        let target_imu_from_anchor_imu_fej = F32Pose {
            rotation: sophus_quat_product_current_packet_f32(
                target_inverse_rotation_fej,
                anchor_imu_pose_fej.rotation,
            ),
            translation: sophus_rotate_difference_f32(
                target_inverse_rotation_fej,
                anchor_imu_pose_fej.translation,
                target_imu_pose_fej.translation,
            ),
        };
        F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu_fej.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu_fej.translation,
            ) + target_camera_from_imu.translation,
        }
    };
    // Upstream linearizeProblem retains the poseLin value unless at least
    // one endpoint is linearized. Equal pose bits do not determine this flag.
    let target_camera_from_anchor_imu = if either_endpoint_linearized {
        target_camera_from_anchor_imu
    } else {
        target_camera_from_anchor_imu_fej
    };
    let target_camera_from_anchor_camera = if same_time_cam_id {
        // Preserve Basalt's exact `TimeCamId` equality branch.  In
        // particular, do not let f32 inverse/compose roundoff turn an
        // identity same-camera observation into a tiny relative transform.
        F32Pose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    } else {
        F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_t_imu_cam.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_t_imu_cam.translation,
            ) + target_camera_from_anchor_imu.translation,
        }
    };

    let bearing = landmark.direction.bearing_f32();
    let point4_target = eigen_homogeneous_point_gemv_f32(
        target_camera_from_anchor_camera.rotation,
        target_camera_from_anchor_camera.translation,
        bearing,
        landmark.inverse_distance as f32,
    );
    let point_target = point4_target.fixed_rows::<3>(0).into_owned();
    let (predicted, projection_jacobian) =
        project_double_sphere_with_jacobian_f32(camera, point_target)?;
    let raw = predicted - Vector2::new(observation.x as f32, observation.y as f32);
    let residual_squared = eigen_vector2_squared_norm_f32(raw.x, raw.y);
    let huber_delta = config.huber_delta as f32;
    let huber_weight = if huber_delta > 0.0 && residual_squared > huber_delta * huber_delta {
        huber_delta / residual_squared.sqrt()
    } else {
        1.0_f32
    };
    let sqrt_weight = huber_weight.sqrt() / observation_stddev;
    let objective_cost = 0.5_f32 * (2.0_f32 - huber_weight) * huber_weight * residual_squared
        / (observation_stddev * observation_stddev);

    let mut point_wrt_relative_pose = SMatrix::<f32, 3, 6>::zeros();
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(SMatrix::<f32, 3, 3>::identity() * landmark.inverse_distance as f32));
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew3_f32(point_target)));
    let residual_wrt_relative_pose =
        eigen_relative_pose_jacobian_f32(projection_jacobian, point_wrt_relative_pose);

    let (relative_wrt_anchor, relative_wrt_target) = if same_time_cam_id {
        // Exact TimeCamId identity is the only branch that skips
        // computeRelPose and supplies zero pose Jacobians. Same-timestamp
        // stereo uses the ordinary relative camera chain below.
        (SMatrix::<f32, 6, 6>::zeros(), SMatrix::<f32, 6, 6>::zeros())
    } else {
        let r_w_i_anchor_inv =
            eigen_quaternion_matrix_f32(sophus_so3_inverse(anchor_imu_pose_fej.rotation));
        let r_w_i_target_inv =
            eigen_quaternion_matrix_f32(sophus_so3_inverse(target_imu_pose_fej.rotation));
        (
            eigen_adjoint_times_rotation_blocks_f32(
                target_camera_from_anchor_imu_fej,
                r_w_i_anchor_inv,
                1.0,
            ),
            eigen_adjoint_times_rotation_blocks_f32(target_camera_from_imu, r_w_i_target_inv, -1.0),
        )
    };

    let direction_jacobian = landmark.direction.bearing_jacobian_f32();
    let rotation = eigen_quaternion_matrix_f32(target_camera_from_anchor_camera.rotation);
    let mut transform_top_left = SMatrix::<f32, 3, 4>::zeros();
    transform_top_left
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation);
    let mut source_jup = SMatrix::<f32, 4, 2>::zeros();
    source_jup
        .fixed_view_mut::<3, 2>(0, 0)
        .copy_from(&direction_jacobian);
    let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
    point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
        &eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup),
    );
    point_wrt_landmark.set_column(2, &target_camera_from_anchor_camera.translation);
    let anchor = eigen_weighted_pose_jacobian_f32(
        residual_wrt_relative_pose,
        relative_wrt_anchor,
        sqrt_weight,
    );
    let target = eigen_weighted_pose_jacobian_f32(
        residual_wrt_relative_pose,
        relative_wrt_target,
        sqrt_weight,
    );
    let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
    camera_jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&projection_jacobian);
    let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
    homogeneous_point_wrt_landmark
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&point_wrt_landmark);
    // `Jpp.col(2)` is the full homogeneous transform's fourth column.  Its
    // final row is the exact affine homogeneous scale, even though the
    // camera's fourth Jacobian column is zero.
    homogeneous_point_wrt_landmark[(3, 2)] = 1.0_f32;
    let landmark_jacobian =
        eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark) * sqrt_weight;
    let debug_chain = if crate::vio::window::diagnostic_env_snapshot().visual_chain_trace {
        Some(VisualChainF32 {
            target_camera_from_imu: visual_chain_pose_snapshot(target_camera_from_imu),
            target_imu_from_anchor_imu: visual_chain_pose_snapshot(target_imu_from_anchor_imu),
            target_camera_from_anchor_imu: visual_chain_pose_snapshot(
                target_camera_from_anchor_imu,
            ),
            target_camera_from_anchor_imu_fej: visual_chain_pose_snapshot(
                target_camera_from_anchor_imu_fej,
            ),
            target_camera_from_anchor_camera: visual_chain_pose_snapshot(
                target_camera_from_anchor_camera,
            ),
            target_camera_matrix: {
                let transform = eigen_homogeneous_transform_f32(
                    target_camera_from_anchor_camera.rotation,
                    target_camera_from_anchor_camera.translation,
                );
                let mut values = [0.0_f32; 16];
                values.copy_from_slice(transform.as_slice());
                values
            },
            // Copy the matrix used by production above so the opt-in sidecar
            // cannot silently drift to a separately reconstructed endpoint.
            relative_wrt_anchor: {
                let mut values = [0.0_f32; 36];
                values.copy_from_slice(relative_wrt_anchor.as_slice());
                values
            },
            point4_target: [
                point4_target[0],
                point4_target[1],
                point4_target[2],
                point4_target[3],
            ],
            point_target: [point_target.x, point_target.y, point_target.z],
            projection: [predicted.x, predicted.y],
            projection_jacobian: [
                projection_jacobian[(0, 0)],
                projection_jacobian[(0, 1)],
                projection_jacobian[(0, 2)],
                projection_jacobian[(1, 0)],
                projection_jacobian[(1, 1)],
                projection_jacobian[(1, 2)],
            ],
        })
    } else {
        None
    };
    Some(AnchoredVisualFactor {
        raw_residual: raw.map(|value| value as f64),
        projection: predicted.map(|value| value as f64),
        huber_weight: huber_weight as f64,
        sqrt_weight: sqrt_weight as f64,
        residual: raw.map(|value| (value * sqrt_weight) as f64),
        anchor_pose_jacobian: anchor.map(|value| value as f64),
        target_pose_jacobian: target.map(|value| value as f64),
        landmark_jacobian: landmark_jacobian.map(|value| value as f64),
        objective_cost: objective_cost as f64,
        debug_chain,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct SqrtPrior {
    pub jacobian: DMatrix<f64>,
    pub rhs: DVector<f64>,
    pub fej_point: DVector<f64>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginalizationError {
    InvalidColumns,
    RankDeficient,
}

impl SqrtPrior {
    pub fn re_reference(
        &mut self,
        new_fej_point: &DVector<f64>,
    ) -> Result<(), MarginalizationError> {
        if new_fej_point.len() != self.fej_point.len() {
            return Err(MarginalizationError::InvalidColumns);
        }
        self.rhs += &self.jacobian * (&self.fej_point - new_fej_point);
        self.fej_point = new_fej_point.clone();
        Ok(())
    }
}

/// Eliminate `marginal_columns` from a whitened square-root system and emit a
/// new square-root prior over `keep_columns`. QR is used both for the
/// landmark/marginal nullspace projection and for the final SqrtToSqrt factor.
pub fn sqrt_to_sqrt_marginalize(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    keep_columns: &[usize],
    marginal_columns: &[usize],
    fej_point: DVector<f64>,
) -> Result<SqrtPrior, MarginalizationError> {
    let n = jacobian.ncols();
    if rhs.len() != jacobian.nrows()
        || keep_columns.iter().chain(marginal_columns).any(|&c| c >= n)
        || keep_columns.iter().any(|c| marginal_columns.contains(c))
        || fej_point.len() != keep_columns.len()
    {
        return Err(MarginalizationError::InvalidColumns);
    }
    let mut jm = DMatrix::zeros(jacobian.nrows(), marginal_columns.len());
    for (k, &c) in marginal_columns.iter().enumerate() {
        jm.set_column(k, &jacobian.column(c));
    }
    let projector = if marginal_columns.is_empty() {
        DMatrix::identity(jacobian.nrows(), jacobian.nrows())
    } else {
        let qr = jm.clone().qr();
        let _q = qr.q();
        let gram = jm.transpose() * &jm;
        if let Some(inv) = gram.try_inverse() {
            DMatrix::identity(jacobian.nrows(), jacobian.nrows()) - &jm * inv * jm.transpose()
        } else {
            return Err(MarginalizationError::RankDeficient);
        }
    };
    let mut jk = DMatrix::zeros(jacobian.nrows(), keep_columns.len());
    for (k, &c) in keep_columns.iter().enumerate() {
        jk.set_column(k, &jacobian.column(c));
    }
    let projected = projector.clone() * jk;
    let projected_rhs = projector * rhs;
    let qr = projected.clone().qr();
    let rank = qr.r().diagonal().iter().filter(|x| x.abs() > 1e-10).count();
    if rank == 0 && !keep_columns.is_empty() {
        return Err(MarginalizationError::RankDeficient);
    }
    let q = qr.q();
    let out_j = q.transpose() * projected;
    let out_r = q.transpose() * projected_rhs;
    Ok(SqrtPrior {
        jacobian: out_j,
        rhs: out_r,
        fej_point,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmConfig {
    pub lambda_initial: f64,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub max_iterations: usize,
    pub convergence_step: f64,
}
impl Default for LmConfig {
    fn default() -> Self {
        Self {
            lambda_initial: 1e-4,
            lambda_min: 1e-6,
            lambda_max: 1e2,
            max_iterations: 7,
            convergence_step: 1e-4,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmDecision {
    Accepted,
    Rejected,
    Converged,
    Failed,
}
#[derive(Debug, Clone, PartialEq)]
pub struct LmTraceEntry {
    pub iteration: usize,
    pub lambda_before: f64,
    pub lambda_after: f64,
    pub cost_before: f64,
    pub model_cost: f64,
    pub actual_cost: f64,
    pub step_norm: f64,
    pub decision: LmDecision,
}
#[derive(Debug, Clone, PartialEq)]
pub struct LmLinearization {
    pub factors: Vec<WhitenedFactorRowStack>,
    pub cost: f64,
}

/// Observation-only callback payload for the pinned per-iteration oracle.
///
/// The default implementation on [`LmProblem`] is empty, so production runs
/// do not allocate or write diagnostic state. The active-window problem uses
/// this payload only when `VISLOC_BASALT_DETAIL_ITERATIONS=1` is set.
pub struct LmDiagnosticEvent<'a> {
    pub iteration: usize,
    pub trial: usize,
    pub phase: &'static str,
    pub lambda: f64,
    pub lambda_after: f64,
    pub cost_before: f64,
    pub model_cost: Option<f64>,
    /// Unrounded model decrease, observed directly before cost subtraction.
    pub model_decrease: Option<f64>,
    pub actual_cost: Option<f64>,
    pub step_norm: Option<f64>,
    pub decision: &'static str,
    /// State at the beginning of the iteration. For an after-decision event
    /// this remains the pre-trial state so landmark back-substitution can be
    /// reproduced exactly.
    pub base_state: &'a DVector<f64>,
    /// State represented by this snapshot (pre-trial, trial, or accepted
    /// state, depending on `phase`).
    pub state: &'a DVector<f64>,
    pub trial_state: Option<&'a DVector<f64>>,
    pub step: Option<&'a DVector<f64>>,
    pub damping_diag: &'a DVector<f64>,
    /// The solver's binary32 damped normal matrix, when the active
    /// `UpstreamF32` path owns one.  This is an observation-only checkpoint:
    /// callers must not rebuild it from the reduced f64 mirror and damping
    /// diagonal because that would introduce a different cast/add schedule.
    pub damped_h: Option<&'a DMatrix<f32>>,
    pub linearization: &'a LmLinearization,
    pub reduced: &'a ReducedNormalSystem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmFailure {
    NonFinite,
    RankDeficient,
    LinearSolve,
}
pub trait LmProblem {
    fn linearize(&self, state: &DVector<f64>) -> Result<LmLinearization, LmFailure>;
    fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure>;

    /// Expanded problems may reduce only the nonzero state columns of each
    /// factor. Ordinary VIO retains its historical arithmetic and ordering.
    fn reduce_f64(
        &self,
        factors: &[WhitenedFactorRowStack],
        state_dof: usize,
        tolerance: f64,
    ) -> ReducedNormalSystem {
        reduce_landmark_factors(factors, state_dof, tolerance)
    }

    /// Optional frame identity for the diagnostic-only IMU reduction audit.
    /// Generic/synthetic problems remain unlabelled and incur no work.
    fn diagnostic_frame_id(&self) -> Option<u64> {
        None
    }

    /// Numeric owner for the active solver. The default keeps synthetic and
    /// legacy callers on the historical f64 path; WindowProblem overrides it
    /// for the pinned `SqrtKeypointVioEstimator<float>` compatibility mode.
    fn scalar_mode(&self) -> ScalarMode {
        ScalarMode::ExtendedF64
    }

    /// Optional observation-only hook used by the full upstream iteration
    /// oracle. Keeping this a default no-op preserves the normal solver path
    /// and avoids imposing a trace implementation on synthetic test problems.
    fn diagnostic_lm_event(&mut self, _event: LmDiagnosticEvent<'_>) {}

    /// Optional, explicitly opt-in normal-system patch used by numeric
    /// counterfactuals. The default is a no-op, so ordinary solver behavior
    /// and all production callers remain unchanged. WindowProblem uses this
    /// only when a diagnostic IMU H/b delta file is named in the environment.
    fn diagnostic_patch_reduced_f32(
        &self,
        _iteration: usize,
        _h: &mut DMatrix<f32>,
        _b: &mut DVector<f32>,
    ) {
    }

    /// Apply a solver increment to a state. Generic problems retain the
    /// historical additive contract; manifold problems override this so the
    /// trial cost uses the same update rule as their Jacobians.
    fn apply_step(&self, state: &DVector<f64>, step: &DVector<f64>) -> DVector<f64> {
        state + step
    }

    /// Evaluate a trial state with any eliminated variables recovered from
    /// the same linearization.  Basalt back-substitutes landmarks before
    /// `computeError`; state-only problems keep the ordinary cost path.
    fn trial_cost(
        &self,
        _state: &DVector<f64>,
        _step: &DVector<f64>,
        trial: &DVector<f64>,
    ) -> Result<f64, LmFailure> {
        self.cost(trial)
    }

    /// Timed wrapper for a trial objective.  Keeping the historical
    /// `trial_cost` signature preserves generic callers; WindowProblem
    /// overrides this hook to split full-manifold trial construction from
    /// objective evaluation without changing its immutable borrow contract.
    fn trial_cost_timed(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<f64, LmFailure> {
        timing.measure(TimingBucket::LmTrialCost, || {
            self.trial_cost(state, step, trial)
        })
    }

    /// Evaluate a trial and optionally retain exact eliminated-variable work
    /// for the immediately following accepted step.  The default hook keeps
    /// existing `LmProblem` implementations source-compatible and carries no
    /// payload; concrete manifold problems may override it when trial work
    /// already computed a safe, one-shot commit token.
    fn trial_cost_timed_with_token(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<(f64, LmTrialToken), LmFailure> {
        self.trial_cost_timed(state, step, trial, timing)
            .map(|cost| (cost, LmTrialToken::default()))
    }

    /// Evaluate a trial with an optional clean-path landmark preparation.
    /// Generic and retained implementations deliberately discard this opaque
    /// payload and continue through their existing token hook; only a future
    /// concrete window override may consume the prepared increments.
    fn trial_cost_timed_with_preparation(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        preparation: Option<LmTrialPreparation>,
        timing: &mut TimingBreakdown,
    ) -> Result<(f64, LmTrialToken), LmFailure> {
        drop(preparation);
        self.trial_cost_timed_with_token(state, step, trial, timing)
    }

    /// Commit eliminated-variable increments after a successful trial.
    fn accept_step(
        &mut self,
        _state: &DVector<f64>,
        _step: &DVector<f64>,
    ) -> Result<(), LmFailure> {
        Ok(())
    }

    /// Commit a trial's one-shot token.  The default discards the token and
    /// preserves the historical accepted-step hook for generic and retained
    /// diagnostic implementations.
    fn accept_step_with_token(
        &mut self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        token: LmTrialToken,
    ) -> Result<(), LmFailure> {
        let _ = token;
        self.accept_step(state, step)
    }
}

#[derive(Debug)]
struct LmPreparedLandmarkStep {
    landmark_index: usize,
    track_id: u64,
    step: Option<Vector3<f64>>,
}

/// Opaque one-shot preparation produced by the clean UpstreamF32 reducer
/// after a state step is solved.  The vector is intentionally private: a
/// concrete consumer must validate its own landmark topology before using it;
/// generic `LmProblem` implementations simply drop it via the default hook.
#[derive(Debug)]
pub struct LmTrialPreparation {
    landmark_steps: Vec<LmPreparedLandmarkStep>,
    tolerance_bits: u64,
    state_fingerprint: u64,
    step_fingerprint: u64,
}

impl LmTrialPreparation {
    /// Consume this one-shot preparation and expose only the mapped values a
    /// concrete window consumer needs for validation.  The compact f32
    /// payload itself never crosses the public trait boundary.
    pub(crate) fn take_landmark_steps(
        self,
    ) -> (u64, u64, u64, Vec<(usize, u64, Option<Vector3<f64>>)>) {
        (
            self.tolerance_bits,
            self.state_fingerprint,
            self.step_fingerprint,
            self.landmark_steps
                .into_iter()
                .map(|entry| (entry.landmark_index, entry.track_id, entry.step))
                .collect(),
        )
    }

    #[cfg(test)]
    pub(crate) fn from_test_entries(
        tolerance: f64,
        entries: Vec<(usize, u64, Option<Vector3<f64>>)>,
    ) -> Self {
        Self {
            landmark_steps: entries
                .into_iter()
                .map(|(landmark_index, track_id, step)| LmPreparedLandmarkStep {
                    landmark_index,
                    track_id,
                    step,
                })
                .collect(),
            tolerance_bits: tolerance.to_bits(),
            state_fingerprint: 0,
            step_fingerprint: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_test_entries_for_state_step(
        state: &DVector<f64>,
        step: &DVector<f64>,
        tolerance: f64,
        entries: Vec<(usize, u64, Option<Vector3<f64>>)>,
    ) -> Self {
        Self {
            landmark_steps: entries
                .into_iter()
                .map(|(landmark_index, track_id, step)| LmPreparedLandmarkStep {
                    landmark_index,
                    track_id,
                    step,
                })
                .collect(),
            tolerance_bits: tolerance.to_bits(),
            state_fingerprint: lm_trial_vector_fingerprint(state),
            step_fingerprint: lm_trial_vector_fingerprint(step),
        }
    }
}

/// Opaque, one-shot work product passed from an LM trial to its immediate
/// accepted-step commit.  The default token is empty; `WindowProblem` stores
/// only the exact landmark increments it already computed while building the
/// eager trial view.  Keeping this type public is required because
/// [`LmProblem`] is public, while its payload remains private to this module.
#[derive(Debug, Default)]
pub struct LmTrialToken {
    landmark_steps: Option<Vec<Option<Vector3<f64>>>>,
    binding: Option<LmTrialBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LmTrialBinding {
    window_identity: usize,
    window_generation: u64,
    scalar_mode: ScalarMode,
    state_fingerprint: u64,
    step_fingerprint: u64,
}

impl LmTrialBinding {
    pub(crate) fn new(
        window_identity: usize,
        window_generation: u64,
        scalar_mode: ScalarMode,
        state_fingerprint: u64,
        step_fingerprint: u64,
    ) -> Self {
        Self {
            window_identity,
            window_generation,
            scalar_mode,
            state_fingerprint,
            step_fingerprint,
        }
    }
}

impl LmTrialToken {
    pub(crate) fn with_landmark_steps(landmark_steps: Vec<Option<Vector3<f64>>>) -> Self {
        Self {
            landmark_steps: Some(landmark_steps),
            binding: None,
        }
    }

    pub(crate) fn with_bound_landmark_steps(
        landmark_steps: Vec<Option<Vector3<f64>>>,
        binding: LmTrialBinding,
    ) -> Self {
        Self {
            landmark_steps: Some(landmark_steps),
            binding: Some(binding),
        }
    }

    pub(crate) fn take_landmark_steps(self) -> Option<Vec<Option<Vector3<f64>>>> {
        self.landmark_steps
    }

    pub(crate) fn take_landmark_steps_for(
        self,
        expected_binding: LmTrialBinding,
    ) -> Result<Option<Vec<Option<Vector3<f64>>>>, ()> {
        let Self {
            landmark_steps,
            binding,
        } = self;
        match landmark_steps {
            None => Ok(None),
            Some(landmark_steps) if binding == Some(expected_binding) => Ok(Some(landmark_steps)),
            Some(_) => Err(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LmResult {
    pub state: DVector<f64>,
    pub cost: f64,
    pub lambda: f64,
    pub iterations: usize,
    pub trace: Vec<LmTraceEntry>,
}

/// Lean UpstreamF32 LM loop used only by the explicit no-diagnostics window
/// path.  The retained/diagnostic implementation below deliberately keeps its
/// historical f64 mirror and event payloads.  In this path the active solve
/// owns only the f32 reduced H/b objects: no diagnostic ReducedNormalSystem,
/// f64 H/b mirror, damping payload, or event snapshot is materialized.
fn solve_lm_without_diagnostics_f32<P: LmProblem>(
    problem: &mut P,
    initial: DVector<f64>,
    config: LmConfig,
    retain_trace: bool,
    timing: &mut TimingBreakdown,
) -> Result<LmResult, LmFailure> {
    let _diagnostic_run_guard = begin_diagnostic_lm_run();
    set_active_diagnostic_lm_iteration(None);
    set_active_diagnostic_lm_frame(None);
    if !config.lambda_initial.is_finite()
        || config.lambda_initial <= 0.0
        || config.lambda_min <= 0.0
        || config.lambda_max < config.lambda_min
    {
        return Err(LmFailure::NonFinite);
    }

    let mut state = initial;
    let mut lambda = (config.lambda_initial as f32)
        .clamp(config.lambda_min as f32, config.lambda_max as f32) as f64;
    let mut cost = problem.cost(&state)?;
    if !cost.is_finite() {
        return Err(LmFailure::NonFinite);
    }
    let mut trace = retain_trace.then(Vec::new);
    // Upstream uses `it <= vio_max_iterations`, i.e. a configured value of
    // seven permits eight accepted/rejected attempts in total.
    let mut lambda_vee = 2.0;
    for iteration in 0..=config.max_iterations {
        set_active_diagnostic_lm_iteration(Some(iteration));
        set_active_diagnostic_lm_frame(problem.diagnostic_frame_id());
        let lin = timing.measure(TimingBucket::LmLinearize, || problem.linearize(&state))?;
        if !lin.cost.is_finite() {
            return Err(LmFailure::NonFinite);
        }

        // Native refreshes error_total from linearizeProblem each iteration;
        // the previous trial's expression schedule can yield different bits.
        cost = lin.cost;
        let reduced = timing.measure(TimingBucket::LmLandmarkReduction, || {
            reduce_landmark_factors_f32_checked_with_compact_back_substitution(
                &lin.factors,
                state.len(),
                1e-10,
            )
            .map_err(|_| LmFailure::LinearSolve)
        })?;
        let mut h32 = reduced.h.clone();
        if !h32.iter().all(|value| value.is_finite())
            || !reduced.b.iter().all(|value| value.is_finite())
        {
            return Err(LmFailure::NonFinite);
        }
        for i in 0..h32.nrows() {
            // Literal `vio_lm_pose_damping_variant == 1`: use the undamped
            // f32 normal diagonal and the configured f32 lambda floor.
            let damping = (h32[(i, i)] * lambda as f32).max(config.lambda_min as f32);
            h32[(i, i)] += damping;
        }

        let step = timing.measure(
            TimingBucket::LmLinearSystemSolve,
            || match eigen_ldlt_solve_f32(&h32, &reduced.b) {
                Some(value) => Ok(value.map(|entry| f64::from(-entry))),
                None => Err(LmFailure::LinearSolve),
            },
        )?;
        if !step.iter().all(|value| value.is_finite()) {
            return Err(LmFailure::NonFinite);
        }

        // Keep the complete transformed row-stack model decrease.  This is
        // the historical local evaluator; the compact payload is used only
        // for the one-shot landmark recovery path below.
        let model_decrease = timing.measure(TimingBucket::LmModelDecrease, || {
            model_cost_decrease_f32(&lin.factors, &step, 1e-10).ok_or(LmFailure::RankDeficient)
        })?;
        let model = (cost as f32 - model_decrease as f32) as f64;
        let step_norm = step
            .iter()
            .map(|value| (*value as f32).abs())
            .fold(0.0_f32, f32::max) as f64;

        // The clean f32 reducer has already retained the Q1/R rows.  Recover
        // each mapped landmark exactly once after the state solve and pass the
        // opaque result through the default-compatible trial hook.  The
        // concrete WindowProblem consumer may use it directly; generic and
        // retained implementations discard it and keep their legacy recovery
        // behavior.
        let preparation = timing.measure(TimingBucket::LmCompactBackSubstitution, || {
            reduced.into_trial_preparation(&state, &step, 1e-10)
        });

        let trial = timing.measure(TimingBucket::LmCompactApplyStep, || {
            problem.apply_step(&state, &step)
        });
        let (actual, trial_token) = problem.trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            preparation,
            timing,
        )?;
        if !actual.is_finite() {
            return Err(LmFailure::NonFinite);
        }

        let convergence_step = config.convergence_step as f32 as f64;
        let function_decrease = (cost as f32 - actual as f32) as f64;

        let before = lambda;
        let cost_before = cost;
        let (_predicted_decrease, relative_decrease) =
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                // Native uses backSubstitute's l_diff directly. Recovering it
                // from the rounded model cost can erase a small decrease.
                let predicted_decrease = model_decrease as f32 as f64;
                let relative_decrease = if predicted_decrease > 0.0 {
                    ((cost as f32 - actual as f32) / predicted_decrease as f32) as f64
                } else {
                    f64::NEG_INFINITY
                };
                (predicted_decrease, relative_decrease)
            });

        let (decision, after) = if actual < cost && relative_decrease > 0.0 {
            timing.measure(TimingBucket::LmAccept, || {
                problem.accept_step_with_token(&state, &step, trial_token)
            })?;
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                state = trial.clone();
                cost = actual;
                let scale = (1.0_f32 - (2.0_f32 * relative_decrease as f32 - 1.0_f32).powi(3))
                    .max(1.0_f32 / 3.0_f32) as f64;
                lambda_vee = 2.0;
                (
                    LmDecision::Accepted,
                    ((lambda as f32 * scale as f32).max(config.lambda_min as f32)) as f64,
                )
            })
        } else {
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                let next = (lambda as f32 * lambda_vee as f32) as f64;
                lambda_vee = (lambda_vee as f32 * 2.0_f32) as f64;
                (LmDecision::Rejected, next)
            })
        };
        lambda = after;
        if let Some(trace) = trace.as_mut() {
            trace.push(LmTraceEntry {
                iteration,
                lambda_before: before,
                lambda_after: after,
                cost_before,
                model_cost: model,
                actual_cost: actual,
                step_norm,
                decision,
            });
        }
        // Basalt does not clamp at the maximum: the rejected attempt is
        // recorded, then optimization terminates once damping crosses it.
        if decision == LmDecision::Rejected && lambda > config.lambda_max {
            break;
        }
        if decision == LmDecision::Accepted
            && (step_norm < convergence_step
                || (function_decrease > 0.0 && function_decrease < 1e-6_f32 as f64))
        {
            return Ok(LmResult {
                state,
                cost,
                lambda,
                iterations: iteration + 1,
                trace: trace.unwrap_or_default(),
            });
        }
    }
    Ok(LmResult {
        state,
        cost,
        lambda,
        iterations: config.max_iterations + 1,
        trace: trace.unwrap_or_default(),
    })
}

pub fn solve_lm<P: LmProblem>(
    problem: &mut P,
    initial: DVector<f64>,
    config: LmConfig,
) -> Result<LmResult, LmFailure> {
    let mut timing = TimingBreakdown::from_env();
    solve_lm_with_timing(problem, initial, config, true, true, &mut timing)
}

pub(crate) fn solve_lm_without_trace<P: LmProblem>(
    problem: &mut P,
    initial: DVector<f64>,
    config: LmConfig,
) -> Result<LmResult, LmFailure> {
    let mut timing = TimingBreakdown::from_env();
    // Retaining the result trace is independent from retaining the diagnostic
    // event payload.  Keep the historical no-trace helper's event behavior;
    // the explicit lean path is selected only by WindowProblem's
    // no-diagnostics entry point below.
    solve_lm_with_timing(problem, initial, config, false, true, &mut timing)
}

pub(crate) fn solve_lm_with_timing<P: LmProblem>(
    problem: &mut P,
    initial: DVector<f64>,
    config: LmConfig,
    retain_trace: bool,
    retain_diagnostics: bool,
    timing: &mut TimingBreakdown,
) -> Result<LmResult, LmFailure> {
    let scalar_mode = problem.scalar_mode();
    if !retain_diagnostics && scalar_mode == ScalarMode::UpstreamF32 {
        return solve_lm_without_diagnostics_f32(problem, initial, config, retain_trace, timing);
    }
    let _diagnostic_run_guard = begin_diagnostic_lm_run();
    // Keep direct `cost()`/`linearize()` calls outside the LM loop
    // unselected by the optional per-iteration IMU diagnostic.
    set_active_diagnostic_lm_iteration(None);
    set_active_diagnostic_lm_frame(None);
    if !config.lambda_initial.is_finite()
        || config.lambda_initial <= 0.0
        || config.lambda_min <= 0.0
        || config.lambda_max < config.lambda_min
    {
        return Err(LmFailure::NonFinite);
    }
    let mut state = initial;
    let mut lambda = if scalar_mode == ScalarMode::UpstreamF32 {
        (config.lambda_initial as f32).clamp(config.lambda_min as f32, config.lambda_max as f32)
            as f64
    } else {
        config
            .lambda_initial
            .clamp(config.lambda_min, config.lambda_max)
    };
    let mut cost = problem.cost(&state)?;
    if !cost.is_finite() {
        return Err(LmFailure::NonFinite);
    }
    let mut trace = retain_trace.then(Vec::new);
    // Upstream uses `it <= vio_max_iterations`, i.e. a configured value of
    // seven permits eight accepted/rejected attempts in total.
    let mut lambda_vee = 2.0;
    for iteration in 0..=config.max_iterations {
        set_active_diagnostic_lm_iteration(Some(iteration));
        set_active_diagnostic_lm_frame(problem.diagnostic_frame_id());
        let lin = timing.measure(TimingBucket::LmLinearize, || problem.linearize(&state))?;
        if !lin.cost.is_finite() {
            return Err(LmFailure::NonFinite);
        }
        if scalar_mode == ScalarMode::UpstreamF32 {
            cost = lin.cost;
        }
        let reduced_f32 = timing.measure(TimingBucket::LmLandmarkReduction, || {
            if scalar_mode == ScalarMode::UpstreamF32 {
                reduce_landmark_factors_f32_checked(&lin.factors, state.len(), 1e-10)
                    .map(Some)
                    .map_err(|_| LmFailure::LinearSolve)
            } else {
                Ok(None)
            }
        })?;
        let reduced_f64 = if scalar_mode == ScalarMode::ExtendedF64 {
            Some(timing.measure(TimingBucket::LmLandmarkReduction, || {
                problem.reduce_f64(&lin.factors, state.len(), 1e-10)
            }))
        } else {
            None
        };
        let (reduced, h, b, h32, b32, damping_diag) =
            timing.measure(TimingBucket::LmNormalSystemPrep, || {
                if let Some(reduced_f32) = reduced_f32.as_ref() {
                    emit_imu_reduction_diagnostic(
                        problem.diagnostic_frame_id(),
                        iteration,
                        reduced_f32,
                    );
                }
                let reduced = reduced_f32
                    .as_ref()
                    .map(ReducedNormalSystemF32::as_f64)
                    .or_else(|| reduced_f64)
                    .expect("one scalar-mode normal reduction");
                let mut h = reduced.h.clone();
                let mut b = reduced.b.clone();
                let mut h32 = reduced_f32.as_ref().map(|reduced| reduced.h.clone());
                let mut b32 = reduced_f32.as_ref().map(|reduced| reduced.b.clone());
                if let (Some(h32), Some(b32)) = (h32.as_mut(), b32.as_mut()) {
                    problem.diagnostic_patch_reduced_f32(iteration, h32, b32);
                    h = DMatrix::from_fn(h32.nrows(), h32.ncols(), |row, col| {
                        h32[(row, col)] as f64
                    });
                    // Keep the f64 mirror used by diagnostic payloads and the
                    // non-f32 fallback coherent with the counterfactual solve.
                    // The active f32 path still solves the patched f32 objects.
                    b = DVector::from_iterator(b32.len(), b32.iter().copied().map(f64::from));
                }
                if !h.iter().all(|x| x.is_finite()) || !b.iter().all(|x| x.is_finite()) {
                    return Err(LmFailure::NonFinite);
                }
                let mut damping_diag = DVector::zeros(h.nrows());
                for i in 0..h.nrows() {
                    // Literal `vio_lm_pose_damping_variant == 1`: damp by the
                    // undamped normal diagonal, with the configured minimum lambda
                    // as an absolute floor.
                    if let Some(h32) = h32.as_mut() {
                        let damping = (h32[(i, i)] * lambda as f32).max(config.lambda_min as f32);
                        damping_diag[i] = damping as f64;
                        h[(i, i)] = h32[(i, i)] as f64 + damping as f64;
                        h32[(i, i)] += damping;
                    } else {
                        damping_diag[i] = (reduced.h[(i, i)] * lambda).max(config.lambda_min);
                        h[(i, i)] += damping_diag[i];
                    }
                }
                Ok((reduced, h, b, h32, b32, damping_diag))
            })?;
        problem.diagnostic_lm_event(LmDiagnosticEvent {
            iteration,
            trial: 0,
            phase: "iteration_start",
            lambda,
            lambda_after: lambda,
            cost_before: cost,
            model_cost: None,
            model_decrease: None,
            actual_cost: None,
            step_norm: None,
            decision: "pending",
            base_state: &state,
            state: &state,
            trial_state: None,
            step: None,
            damping_diag: &damping_diag,
            damped_h: h32.as_ref(),
            linearization: &lin,
            reduced: &reduced,
        });
        let step = timing.measure(TimingBucket::LmLinearSystemSolve, || {
            if let (Some(h32), Some(b32)) = (h32.as_ref(), b32.as_ref()) {
                // Upstream's `MatrixXf::ldlt().solve(b)` is a diagonal-pivoted
                // lower LDLT solve.  Keep the native sign convention: Eigen
                // returns `H^-1 b`, then ABS_QR applies the negative increment.
                match eigen_ldlt_solve_f32(h32, b32) {
                    Some(value) => Ok(value.map(|entry| f64::from(-entry))),
                    None => Err(LmFailure::LinearSolve),
                }
            } else {
                match h.cholesky() {
                    Some(factor) => Ok(factor.solve(&(-&b))),
                    None => Err(LmFailure::LinearSolve),
                }
            }
        })?;
        if !step.iter().all(|x| x.is_finite()) {
            return Err(LmFailure::NonFinite);
        }
        // ABS_QR's backSubstitute evaluates the complete transformed row
        // stack, including each landmark block's Q1 rows.  The reduced
        // camera quadratic alone therefore underestimates the model
        // decrease and changes the LM accept/reject schedule.
        let model_decrease = timing.measure(TimingBucket::LmModelDecrease, || {
            if scalar_mode == ScalarMode::UpstreamF32 {
                model_cost_decrease_f32(&lin.factors, &step, 1e-10)
            } else {
                model_cost_decrease(&lin.factors, &step, 1e-10)
            }
            .ok_or(LmFailure::RankDeficient)
        })?;
        // Basalt's linearized marginal-prior error deliberately drops the
        // FEJ-independent residual constant and may therefore be negative.
        // Its LM loop compares that reduced cost directly; clamping the model
        // to zero changes `l_diff`/relative-decrease at the first window
        // shift and can alter the accept/reject schedule.
        let model = if scalar_mode == ScalarMode::UpstreamF32 {
            (cost as f32 - model_decrease as f32) as f64
        } else {
            cost - model_decrease
        };
        let step_norm = if scalar_mode == ScalarMode::UpstreamF32 {
            step.iter()
                .map(|value| (*value as f32).abs())
                .fold(0.0_f32, f32::max) as f64
        } else {
            step.iter().map(|value| value.abs()).fold(0.0, f64::max)
        };
        let trial = timing.measure(TimingBucket::LmCompactApplyStep, || {
            problem.apply_step(&state, &step)
        });
        let actual = problem.trial_cost_timed(&state, &step, &trial, timing)?;
        if !actual.is_finite() {
            return Err(LmFailure::NonFinite);
        }
        let before = lambda;
        let cost_before = cost;
        let state_before =
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || state.clone());
        problem.diagnostic_lm_event(LmDiagnosticEvent {
            iteration,
            trial: 0,
            phase: "trial",
            lambda,
            lambda_after: lambda,
            cost_before,
            model_cost: Some(model),
            model_decrease: Some(model_decrease),
            actual_cost: Some(actual),
            step_norm: Some(step_norm),
            decision: "pending",
            base_state: &state_before,
            state: &state,
            trial_state: Some(&trial),
            step: Some(&step),
            damping_diag: &damping_diag,
            damped_h: h32.as_ref(),
            linearization: &lin,
            reduced: &reduced,
        });
        let convergence_step = if scalar_mode == ScalarMode::UpstreamF32 {
            config.convergence_step as f32 as f64
        } else {
            config.convergence_step
        };
        let function_decrease = if scalar_mode == ScalarMode::UpstreamF32 {
            (cost as f32 - actual as f32) as f64
        } else {
            cost - actual
        };
        let (_predicted_decrease, relative_decrease) =
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                let predicted_decrease = if scalar_mode == ScalarMode::UpstreamF32 {
                    // Keep l_diff independent of the diagnostic model cost.
                    model_decrease as f32 as f64
                } else {
                    cost - model
                };
                let relative_decrease = if predicted_decrease > 0.0 {
                    if scalar_mode == ScalarMode::UpstreamF32 {
                        ((cost as f32 - actual as f32) / predicted_decrease as f32) as f64
                    } else {
                        (cost - actual) / predicted_decrease
                    }
                } else {
                    f64::NEG_INFINITY
                };
                (predicted_decrease, relative_decrease)
            });
        let (decision, after) = if actual < cost && relative_decrease > 0.0 {
            timing.measure(TimingBucket::LmAccept, || {
                problem.accept_step(&state, &step)
            })?;
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                state = trial.clone();
                cost = actual;
                let scale = if scalar_mode == ScalarMode::UpstreamF32 {
                    (1.0_f32 - (2.0_f32 * relative_decrease as f32 - 1.0_f32).powi(3))
                        .max(1.0_f32 / 3.0_f32) as f64
                } else {
                    (1.0 - (2.0 * relative_decrease - 1.0).powi(3)).max(1.0 / 3.0)
                };
                lambda_vee = 2.0;
                (
                    LmDecision::Accepted,
                    if scalar_mode == ScalarMode::UpstreamF32 {
                        ((lambda as f32 * scale as f32).max(config.lambda_min as f32)) as f64
                    } else {
                        (lambda * scale).max(config.lambda_min)
                    },
                )
            })
        } else {
            timing.measure(TimingBucket::LmDecisionStateBookkeeping, || {
                let next = if scalar_mode == ScalarMode::UpstreamF32 {
                    (lambda as f32 * lambda_vee as f32) as f64
                } else {
                    lambda * lambda_vee
                };
                lambda_vee = if scalar_mode == ScalarMode::UpstreamF32 {
                    (lambda_vee as f32 * 2.0_f32) as f64
                } else {
                    lambda_vee * 2.0
                };
                (LmDecision::Rejected, next)
            })
        };
        lambda = after;
        let phase = if decision == LmDecision::Accepted {
            "accepted"
        } else {
            "rejected"
        };
        problem.diagnostic_lm_event(LmDiagnosticEvent {
            iteration,
            trial: 0,
            phase,
            lambda: before,
            lambda_after: after,
            cost_before,
            model_cost: Some(model),
            model_decrease: Some(model_decrease),
            actual_cost: Some(actual),
            step_norm: Some(step_norm),
            decision: if decision == LmDecision::Accepted {
                "accepted"
            } else {
                "rejected"
            },
            base_state: &state_before,
            state: &state,
            trial_state: Some(&trial),
            step: Some(&step),
            damping_diag: &damping_diag,
            damped_h: h32.as_ref(),
            linearization: &lin,
            reduced: &reduced,
        });
        if let Some(trace) = trace.as_mut() {
            trace.push(LmTraceEntry {
                iteration,
                lambda_before: before,
                lambda_after: after,
                cost_before,
                model_cost: model,
                actual_cost: actual,
                step_norm,
                decision,
            });
        }
        // Basalt does not clamp at the maximum: the rejected attempt is
        // recorded, then optimization terminates once damping crosses it.
        if decision == LmDecision::Rejected && lambda > config.lambda_max {
            break;
        }
        let convergence_function = if scalar_mode == ScalarMode::UpstreamF32 {
            1e-6_f32 as f64
        } else {
            1e-6_f64
        };
        if decision == LmDecision::Accepted
            && (step_norm < convergence_step
                || (function_decrease > 0.0 && function_decrease < convergence_function))
        {
            return Ok(LmResult {
                state,
                cost,
                lambda,
                iterations: iteration + 1,
                trace: trace.unwrap_or_default(),
            });
        }
    }
    Ok(LmResult {
        state,
        cost,
        lambda,
        iterations: config.max_iterations + 1,
        trace: trace.unwrap_or_default(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorConfig {
    pub observation_stddev: f64,
    pub huber_delta: f64,
    pub outlier_threshold: f64,
}
impl Default for FactorConfig {
    fn default() -> Self {
        Self {
            observation_stddev: 0.5,
            huber_delta: 1.0,
            outlier_threshold: 3.0,
        }
    }
}

fn robust_whiten(residual: &DVector<f64>, config: FactorConfig) -> Option<DVector<f64>> {
    if !config.observation_stddev.is_finite() || config.observation_stddev <= 0.0 {
        return None;
    }
    let scaled = residual / config.observation_stddev;
    if scaled.norm() > config.outlier_threshold {
        return None;
    }
    let norm = scaled.norm();
    let weight = if norm > config.huber_delta {
        (config.huber_delta / norm).sqrt()
    } else {
        1.0
    };
    Some(scaled * weight)
}

/// Basalt's robust visual objective for one residual row block.
///
/// `huber_weight` is the unsquared Huber weight used to build the whitened
/// row (`sqrt(huber_weight) * raw / sigma`).  Keeping the objective formula
/// explicit prevents the common but incorrect `||whitened_row||²` shortcut
/// for outliers.
fn robust_objective_from_raw(
    residual_squared: f64,
    huber_weight: f64,
    observation_stddev: f64,
) -> f64 {
    0.5 * (2.0 - huber_weight) * huber_weight * residual_squared
        / (observation_stddev * observation_stddev)
}

pub fn visual_reprojection_factor(
    camera: &DoubleSphereCamera,
    pose_world_camera: &SE3,
    point_world: Point3<f64>,
    observation: Point2<f64>,
    config: FactorConfig,
) -> Option<WhitenedFactorRowStack> {
    let point_camera = pose_world_camera.inverse().transform_point(&point_world);
    let predicted = camera.project(&point_camera)?;
    let raw = DVector::from_vec(vec![
        observation.x - predicted.x,
        observation.y - predicted.y,
    ]);
    let residual = robust_whiten(&raw, config)?;
    let eps = 1e-6;
    let mut js = DMatrix::zeros(2, AOM_NAV_DOF);
    let mut jl = DMatrix::zeros(2, 3);
    for k in 0..6 {
        let mut d = Vector6::zeros();
        d[k] = eps;
        let plus = SE3::exp(&d).compose(pose_world_camera);
        d[k] = -eps;
        let minus = SE3::exp(&d).compose(pose_world_camera);
        let rp = projection_residual(camera, &plus, &point_world, observation)?;
        let rm = projection_residual(camera, &minus, &point_world, observation)?;
        for row in 0..2 {
            js[(row, k)] = (rp[row] - rm[row]) / (2.0 * eps) / config.observation_stddev;
        }
    }
    for k in 0..3 {
        let mut p = point_world;
        p.coords[k] += eps;
        let rp = projection_residual(camera, pose_world_camera, &p, observation)?;
        p.coords[k] -= 2.0 * eps;
        let rm = projection_residual(camera, pose_world_camera, &p, observation)?;
        for row in 0..2 {
            jl[(row, k)] = (rp[row] - rm[row]) / (2.0 * eps) / config.observation_stddev;
        }
    }
    let scaled_norm = raw.norm() / config.observation_stddev;
    if scaled_norm > config.huber_delta {
        let w = (config.huber_delta / scaled_norm).sqrt();
        js *= w;
        jl *= w;
    }
    let huber_weight = if scaled_norm > config.huber_delta {
        config.huber_delta / scaled_norm
    } else {
        1.0
    };
    Some(WhitenedFactorRowStack::with_objective_cost_kind(
        js,
        jl,
        residual,
        robust_objective_from_raw(raw.norm_squared(), huber_weight, config.observation_stddev),
        FactorKind::Visual,
    )?)
}
fn projection_residual(
    camera: &DoubleSphereCamera,
    pose: &SE3,
    point: &Point3<f64>,
    obs: Point2<f64>,
) -> Option<Vector3<f64>> {
    let p = camera.project(&pose.inverse().transform_point(point))?;
    Some(Vector3::new(obs.x - p.x, obs.y - p.y, 0.0))
}

pub fn stereo_reprojection_factor(
    camera_left: &DoubleSphereCamera,
    pose_left: &SE3,
    camera_right: &DoubleSphereCamera,
    pose_right: &SE3,
    point: Point3<f64>,
    observations: (Point2<f64>, Point2<f64>),
    config: FactorConfig,
) -> Option<WhitenedFactorRowStack> {
    let a = visual_reprojection_factor(camera_left, pose_left, point, observations.0, config)?;
    let b = visual_reprojection_factor(camera_right, pose_right, point, observations.1, config)?;
    let mut js = DMatrix::zeros(4, AOM_NAV_DOF);
    let mut jl = DMatrix::zeros(4, 3);
    js.rows_mut(0, 2).copy_from(&a.state_jacobian);
    js.rows_mut(2, 2).copy_from(&b.state_jacobian);
    jl.rows_mut(0, 2).copy_from(&a.landmark_jacobian);
    jl.rows_mut(2, 2).copy_from(&b.landmark_jacobian);
    let mut r = DVector::zeros(4);
    r.rows_mut(0, 2).copy_from(&a.residual);
    r.rows_mut(2, 2).copy_from(&b.residual);
    WhitenedFactorRowStack::with_objective_cost_kind(
        js,
        jl,
        r,
        a.objective_cost + b.objective_cost,
        FactorKind::Visual,
    )
}

pub fn imu_preintegration_factor(
    delta: &ImuPreintegratedDelta,
    config: FactorConfig,
) -> Option<WhitenedFactorRowStack> {
    let mut js = DMatrix::zeros(9, AOM_NAV_DOF);
    js.view_mut((0, 9), (3, 3))
        .copy_from(&delta.jacobian_rotation_gyro_bias);
    js.view_mut((3, 9), (3, 3))
        .copy_from(&delta.jacobian_velocity_gyro_bias);
    js.view_mut((3, 12), (3, 3))
        .copy_from(&delta.jacobian_velocity_accel_bias);
    js.view_mut((6, 9), (3, 3))
        .copy_from(&delta.jacobian_position_gyro_bias);
    js.view_mut((6, 12), (3, 3))
        .copy_from(&delta.jacobian_position_accel_bias);
    let raw = DVector::from_iterator(
        9,
        delta
            .delta_velocity
            .iter()
            .chain(delta.delta_position.iter())
            .chain(delta.delta_rotation.scaled_axis().iter())
            .copied(),
    );
    let residual = robust_whiten(&raw, config)?;
    WhitenedFactorRowStack::with_objective_cost_kind(
        js,
        DMatrix::zeros(9, 0),
        residual.clone(),
        0.5 * residual.norm_squared(),
        FactorKind::Imu,
    )
}

pub fn bias_random_walk_factor(
    delta_gyro: Vector3<f64>,
    delta_accel: Vector3<f64>,
    stddev: f64,
) -> Option<WhitenedFactorRowStack> {
    if !stddev.is_finite() || stddev <= 0.0 {
        return None;
    }
    let mut j = DMatrix::zeros(6, AOM_NAV_DOF);
    j.view_mut((0, 9), (3, 3)).fill(-1.0 / stddev);
    j.view_mut((3, 12), (3, 3)).fill(-1.0 / stddev);
    let r = DVector::from_iterator(
        6,
        delta_gyro
            .iter()
            .chain(delta_accel.iter())
            .map(|x| x / stddev),
    );
    WhitenedFactorRowStack::with_objective_cost_kind(
        j,
        DMatrix::zeros(6, 0),
        r.clone(),
        0.5 * r.norm_squared(),
        FactorKind::Bias,
    )
}

pub fn prior_factor(
    jacobian: DMatrix<f64>,
    residual: DVector<f64>,
) -> Option<WhitenedFactorRowStack> {
    WhitenedFactorRowStack::with_objective_cost_kind(
        jacobian,
        DMatrix::zeros(residual.len(), 0),
        residual.clone(),
        0.5 * residual.norm_squared(),
        FactorKind::Prior,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AomBlock {
    Pose6,
    Velocity3,
    GyroBias3,
    AccelBias3,
}

impl AomBlock {
    pub const fn offset(self) -> usize {
        match self {
            Self::Pose6 => 0,
            Self::Velocity3 => 6,
            Self::GyroBias3 => 9,
            Self::AccelBias3 => 12,
        }
    }
    pub const fn dof(self) -> usize {
        match self {
            Self::Pose6 => 6,
            _ => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AomState {
    pub pose6: DVector<f64>,
    pub velocity: DVector<f64>,
    pub gyro_bias: DVector<f64>,
    pub accel_bias: DVector<f64>,
}
impl Default for AomState {
    fn default() -> Self {
        Self {
            pose6: DVector::zeros(6),
            velocity: DVector::zeros(3),
            gyro_bias: DVector::zeros(3),
            accel_bias: DVector::zeros(3),
        }
    }
}
impl AomState {
    pub fn flatten(&self) -> DVector<f64> {
        let mut x = DVector::zeros(AOM_NAV_DOF);
        x.rows_mut(0, 6).copy_from(&self.pose6);
        x.rows_mut(6, 3).copy_from(&self.velocity);
        x.rows_mut(9, 3).copy_from(&self.gyro_bias);
        x.rows_mut(12, 3).copy_from(&self.accel_bias);
        x
    }
}

/// Semantic owner of a whitened row stack.
///
/// The row shape is not sufficient to identify an IMU block: a square-root
/// marginal prior can legitimately have nine rows, while the upstream IMU
/// `DenseAccumulator` receives a 15-row `[imu9 | gyro_bias3 | accel_bias3]`
/// block.  Keep this tag beside the rows so the f32 reduction never routes an
/// accidental nine-row prior (or a synthetic test factor) through the IMU
/// packet schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorKind {
    /// A factor assembled by a generic caller or a numerical test.
    Generic,
    /// A square-root marginal/anchor prior.
    Prior,
    /// A grouped visual landmark factor, before landmark elimination.
    Visual,
    /// The nine residual rows of one preintegrated IMU link.
    Imu,
    /// The six gyro/accelerometer bias random-walk rows paired with an IMU
    /// link.
    Bias,
}

/// Absolute state-column ownership of one chronological IMU link.
///
/// Basalt's `ImuBlock` receives two adjacent navigation blocks as a local
/// 30-column matrix.  The blocks are not necessarily at columns `0` and
/// `15`: pose-only keyframes are kept in the prefix of the AOM order.  Keep
/// those absolute offsets on the factor pair instead of recovering them from
/// a row/column count (or from `% 15`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImuLinkOffsets {
    pub start: usize,
    pub end: usize,
}

/// Optional identity carried by a grouped visual landmark factor.  The
/// compact UpstreamF32 landmark-recovery plan uses this identity to move a
/// precomputed increment back to the owning landmark without depending on
/// factor position.  It is deliberately an adapter rather than a required
/// constructor argument so generic/external factor builders keep their
/// historical legacy path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LandmarkFactorMetadata {
    pub landmark_index: usize,
    pub track_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhitenedFactorRowStack {
    pub state_jacobian: DMatrix<f64>,
    pub landmark_jacobian: DMatrix<f64>,
    pub residual: DVector<f64>,
    /// Semantic source used by the f32 normal-equation reduction.  This is
    /// deliberately explicit rather than inferred from row/column counts.
    pub kind: FactorKind,
    /// Explicit absolute columns for an `Imu`/`Bias` pair.  Ordinary factors
    /// leave this unset; constructors for the active-window pair attach it
    /// after scattering their local 15-column blocks into the AOM.
    pub imu_link_offsets: Option<ImuLinkOffsets>,
    /// Absolute state columns corresponding to the compact columns of a
    /// square-root marginal prior.  Upstream evaluates the stored prior as
    /// its compact `H` matrix before scattering the resulting normal block
    /// into the active AOM.  Ordinary factors leave this unset and retain the
    /// historical global-width product path.
    pub prior_state_columns: Option<Vec<usize>>,
    /// Optional identity of the landmark represented by this grouped visual
    /// factor.  Missing metadata selects the legacy back-substitution path.
    pub landmark_metadata: Option<LandmarkFactorMetadata>,
    /// Objective contribution associated with these whitened rows.
    ///
    /// Most factors are Gaussian and use `0.5 * ||residual||²`.  Robust
    /// visual factors override this with Basalt's exact Huber objective.
    pub objective_cost: f64,
    /// Optional IMU input-stage payload used only by the f32 reduction audit.
    /// Keeping the payload on the semantic IMU row lets the reducer attach
    /// the exact factor inputs to the matching 15x30 local packet without
    /// changing any production arithmetic or relying on row ordering.
    pub imu_input_diagnostic: Option<serde_json::Value>,
    /// Optional `(state_index, camera_id)` identities for the observations
    /// belonging to a grouped visual factor.  This is metadata for the
    /// visual-prefix sidecar only; the reducer never reads it on the normal
    /// path.
    visual_observation_ids: Option<Vec<(usize, u16)>>,
}
impl WhitenedFactorRowStack {
    pub fn new(
        state_jacobian: DMatrix<f64>,
        landmark_jacobian: DMatrix<f64>,
        residual: DVector<f64>,
    ) -> Option<Self> {
        let objective_cost = 0.5 * residual.norm_squared();
        Self::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual,
            objective_cost,
            FactorKind::Generic,
        )
    }

    pub fn with_objective_cost(
        state_jacobian: DMatrix<f64>,
        landmark_jacobian: DMatrix<f64>,
        residual: DVector<f64>,
        objective_cost: f64,
    ) -> Option<Self> {
        Self::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual,
            objective_cost,
            FactorKind::Generic,
        )
    }

    /// Construct a factor while retaining its semantic source tag.
    pub fn with_objective_cost_kind(
        state_jacobian: DMatrix<f64>,
        landmark_jacobian: DMatrix<f64>,
        residual: DVector<f64>,
        objective_cost: f64,
        kind: FactorKind,
    ) -> Option<Self> {
        if state_jacobian.nrows() != landmark_jacobian.nrows()
            || residual.len() != state_jacobian.nrows()
            || !objective_cost.is_finite()
        {
            None
        } else {
            Some(Self {
                state_jacobian,
                landmark_jacobian,
                residual,
                objective_cost,
                kind,
                imu_link_offsets: None,
                prior_state_columns: None,
                landmark_metadata: None,
                imu_input_diagnostic: None,
                visual_observation_ids: None,
            })
        }
    }

    /// Assign a semantic source to a factor returned by an existing
    /// constructor.  Keeping this small adapter preserves the historical
    /// constructor API for downstream synthetic callers.
    pub fn with_kind(mut self, kind: FactorKind) -> Self {
        self.kind = kind;
        self
    }

    /// Attach the absolute start/end navigation-block columns owned by an
    /// IMU link.  This is deliberately a separate adapter so legacy factor
    /// constructors remain source-compatible.
    pub fn with_imu_link_offsets(mut self, start: usize, end: usize) -> Self {
        self.imu_link_offsets = Some(ImuLinkOffsets { start, end });
        self
    }

    /// Attach the active absolute AOM columns represented by a compact
    /// square-root prior.  The f32 reducer uses this metadata to reproduce
    /// Eigen's compact `H.transpose() * H` schedule before scattering it into
    /// the global state matrix.
    pub fn with_prior_state_columns(mut self, columns: Vec<usize>) -> Self {
        self.prior_state_columns = Some(columns);
        self
    }

    /// Attach the optional landmark identity used by the clean UpstreamF32
    /// compact recovery plan.  Callers that do not provide it retain the
    /// historical per-factor recovery fallback.
    pub fn with_landmark_metadata(mut self, landmark_index: usize, track_id: u64) -> Self {
        self.landmark_metadata = Some(LandmarkFactorMetadata {
            landmark_index,
            track_id,
        });
        self
    }

    /// Attach an already materialized diagnostic-only IMU input payload.
    /// Production callers leave this unset; the active-window factor builder
    /// populates it only when `VISLOC_BASALT_DIAGNOSTIC_IMU_ROWS` is enabled.
    pub fn with_imu_input_diagnostic(mut self, payload: serde_json::Value) -> Self {
        self.imu_input_diagnostic = Some(payload);
        self
    }

    /// Attach observation identities for the opt-in visual-prefix capture.
    /// The compact pair keeps this adapter independent of the active-window
    /// observation type and has no effect on numeric factor evaluation.
    pub fn with_visual_observation_ids(mut self, observations: Vec<(usize, u16)>) -> Self {
        self.visual_observation_ids = Some(observations);
        self
    }
    pub fn rows(&self) -> usize {
        self.residual.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkBackSubstitution {
    pub state_jacobian: DMatrix<f64>,
    pub landmark_jacobian: DMatrix<f64>,
    pub residual: DVector<f64>,
    pub rank: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReducedNormalSystem {
    pub h: DMatrix<f64>,
    pub b: DVector<f64>,
    pub back_substitution: Vec<LandmarkBackSubstitution>,
    /// Optional f32 normal-system checkpoints used by the frame-level
    /// parity logger.  This is populated only when an explicit diagnostic
    /// environment variable is set; the production solver still consumes
    /// only `h`/`b`.
    pub diagnostic_stages: Option<DiagnosticNormalSystem>,
}

/// Source-order normal-system checkpoints for the pinned float32 path.
///
/// Basalt's `LinearizationAbsQR::get_dense_H_b` has four observable phases:
/// visual landmark reduction, the separate IMU `DenseAccumulator`, pose
/// damping, and finally the marginal prior.  Keeping these snapshots as
/// f64 containers is intentional: every value is converted from an already
/// rounded f32 operand, so the logger can serialize the original f32 bits
/// without changing the active arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticNormalSystem {
    pub visual_h: DMatrix<f64>,
    pub visual_b: DVector<f64>,
    pub visual_imu_h: DMatrix<f64>,
    pub visual_imu_b: DVector<f64>,
    pub prior_h: DMatrix<f64>,
    pub prior_b: DVector<f64>,
}

#[derive(Debug, Clone)]
struct ReducedNormalSystemF32 {
    h: DMatrix<f32>,
    b: DVector<f32>,
    back_substitution: Vec<LandmarkBackSubstitution>,
    /// Optional clean-path payload extracted from the same landmark QR that
    /// produced `projected`.  Retained/diagnostic callers leave this unset so
    /// their historical f64 payload and call graph remain unchanged.
    compact_back_substitution: Option<CompactLandmarkBackSubstitutionBatchF32>,
    /// Clean-path model-decrease rows moved from the reducer's projected
    /// visual tuples.  This is deliberately separate from Q1 recovery: the
    /// Q2 rows are consumed only by the clean model evaluator and are absent
    /// from retained/diagnostic/F64 reductions.
    model_decrease_payload: Option<ModelDecreasePayloadF32>,
    imu_diagnostic: Option<ImuReductionDiagnostic>,
    diagnostic_stages: Option<DiagnosticNormalSystemF32>,
}

/// Per-landmark state retained from the clean UpstreamF32 ABS-QR walk.  The
/// first `landmark_cols` transformed rows are the only rows needed after the
/// state solve: `Q1 * J_state`, `Q1 * r`, and the upper triangular landmark
/// block `R`.  Keeping these f32 values avoids rebuilding the visual factor
/// and running Householder QR again for every LM trial.
#[derive(Debug, Clone)]
pub(crate) struct CompactLandmarkBackSubstitutionF32 {
    pub(crate) landmark_index: usize,
    pub(crate) track_id: u64,
    /// Row-major Q1 state rows, followed by Q1 residual and upper-R values.
    ///
    /// The old representation kept three independent nalgebra objects here
    /// (and therefore three heap allocations) for every visual factor.  The
    /// values are only read by the exact compact back-substitution helper, so
    /// one flat allocation is sufficient and keeps the source values/order
    /// unchanged.  `state_cols` and `landmark_cols` describe the slices.
    storage: Vec<f32>,
    state_cols: usize,
    landmark_cols: usize,
    pub(crate) rank: usize,
    /// Whether the original upstream recovery preconditions hold.  Keeping
    /// rank-deficient payloads observable lets the clean caller fail closed
    /// without confusing an ineligible block with a missing mapping.
    pub(crate) eligible: bool,
}

impl CompactLandmarkBackSubstitutionF32 {
    #[inline]
    fn q1_state_value(&self, row: usize, column: usize) -> f32 {
        self.storage[row * self.state_cols + column]
    }

    #[inline]
    fn q1_residual_value(&self, row: usize) -> f32 {
        let offset = self.landmark_cols * self.state_cols;
        self.storage[offset + row]
    }

    #[inline]
    fn upper_r_value(&self, row: usize, column: usize) -> f32 {
        let offset = self.landmark_cols * self.state_cols + self.landmark_cols;
        self.storage[offset + row * self.landmark_cols + column]
    }

    #[inline]
    fn storage_len(&self) -> Option<usize> {
        checked_compact_storage_len(self.landmark_cols, self.state_cols)
    }

    #[cfg(test)]
    fn q1_state_shape(&self) -> (usize, usize) {
        (self.landmark_cols, self.state_cols)
    }

    #[cfg(test)]
    fn q1_residual_len(&self) -> usize {
        self.landmark_cols
    }

    #[cfg(test)]
    fn upper_r_shape(&self) -> (usize, usize) {
        (self.landmark_cols, self.landmark_cols)
    }

    #[cfg(test)]
    fn view(&self) -> CompactLandmarkBackSubstitutionViewF32<'_> {
        CompactLandmarkBackSubstitutionViewF32 {
            storage: &self.storage,
            storage_offset: 0,
            state_cols: self.state_cols,
            landmark_cols: self.landmark_cols,
            rank: self.rank,
            eligible: self.eligible,
        }
    }
}

/// Borrowed view of compact Q1/R storage.  It can address either the
/// standalone test payload or an entry in the reducer's shared arena without
/// copying any f32 values.
struct CompactLandmarkBackSubstitutionViewF32<'a> {
    storage: &'a [f32],
    storage_offset: usize,
    state_cols: usize,
    landmark_cols: usize,
    rank: usize,
    eligible: bool,
}

impl<'a> CompactLandmarkBackSubstitutionViewF32<'a> {
    #[inline]
    fn q1_state_value(&self, row: usize, column: usize) -> f32 {
        self.storage[self.storage_offset + row * self.state_cols + column]
    }

    #[inline]
    fn q1_residual_value(&self, row: usize) -> f32 {
        let offset = self.landmark_cols * self.state_cols;
        self.storage[self.storage_offset + offset + row]
    }

    #[inline]
    fn upper_r_value(&self, row: usize, column: usize) -> f32 {
        let offset = self.landmark_cols * self.state_cols + self.landmark_cols;
        self.storage[self.storage_offset + offset + row * self.landmark_cols + column]
    }

    #[inline]
    fn q1_residual_len(&self) -> usize {
        self.landmark_cols
    }
}

trait QrModelQ1ViewF32 {
    fn q1_state_value(&self, row: usize, column: usize) -> f32;
    fn q1_residual_value(&self, row: usize) -> f32;
    fn upper_r_value(&self, row: usize, column: usize) -> f32;
    fn landmark_cols(&self) -> usize;
    fn state_cols(&self) -> usize;
    fn rank(&self) -> usize;
    fn eligible(&self) -> bool;
}

impl QrModelQ1ViewF32 for CompactLandmarkBackSubstitutionF32 {
    #[inline]
    fn q1_state_value(&self, row: usize, column: usize) -> f32 {
        CompactLandmarkBackSubstitutionF32::q1_state_value(self, row, column)
    }

    #[inline]
    fn q1_residual_value(&self, row: usize) -> f32 {
        CompactLandmarkBackSubstitutionF32::q1_residual_value(self, row)
    }

    #[inline]
    fn upper_r_value(&self, row: usize, column: usize) -> f32 {
        CompactLandmarkBackSubstitutionF32::upper_r_value(self, row, column)
    }

    #[inline]
    fn landmark_cols(&self) -> usize {
        self.landmark_cols
    }

    #[inline]
    fn state_cols(&self) -> usize {
        self.state_cols
    }

    #[inline]
    fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    fn eligible(&self) -> bool {
        self.eligible
    }
}

impl<'a> QrModelQ1ViewF32 for CompactLandmarkBackSubstitutionViewF32<'a> {
    #[inline]
    fn q1_state_value(&self, row: usize, column: usize) -> f32 {
        CompactLandmarkBackSubstitutionViewF32::q1_state_value(self, row, column)
    }

    #[inline]
    fn q1_residual_value(&self, row: usize) -> f32 {
        CompactLandmarkBackSubstitutionViewF32::q1_residual_value(self, row)
    }

    #[inline]
    fn upper_r_value(&self, row: usize, column: usize) -> f32 {
        CompactLandmarkBackSubstitutionViewF32::upper_r_value(self, row, column)
    }

    #[inline]
    fn landmark_cols(&self) -> usize {
        self.landmark_cols
    }

    #[inline]
    fn state_cols(&self) -> usize {
        self.state_cols
    }

    #[inline]
    fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    fn eligible(&self) -> bool {
        self.eligible
    }
}

/// Metadata for one compact payload stored in a reducer-owned arena.  The
/// arena keeps all Q1/R values for a reduction in one allocation; descriptors
/// remain small and are moved into the one-shot preparation path after the
/// state solve.
#[derive(Debug, Clone)]
struct CompactLandmarkBackSubstitutionEntryF32 {
    landmark_index: usize,
    track_id: u64,
    storage_offset: usize,
    state_cols: usize,
    landmark_cols: usize,
    rank: usize,
    eligible: bool,
}

#[derive(Debug, Clone)]
struct CompactLandmarkBackSubstitutionBatchF32 {
    storage: Vec<f32>,
    entries: Vec<CompactLandmarkBackSubstitutionEntryF32>,
}

/// One visual Q2 row stack retained for the clean model-decrease pass.  The
/// matrices are moved out of the reducer's projected tuple after H/b assembly;
/// no second QR or Q2 clone is needed.  Q1/R remain in the compact arena and
/// are addressed by `compact_entry_index`.
#[derive(Debug, Clone)]
struct ModelDecreaseVisualF32 {
    factor_index: usize,
    compact_entry_index: usize,
    q2_state: DMatrix<f32>,
    q2_residual: DVector<f32>,
}

#[derive(Debug, Clone)]
struct ModelDecreasePayloadF32 {
    visual: Vec<ModelDecreaseVisualF32>,
}

/// Move the clean reducer's projected visual Q2 allocations into the model
/// sidecar after H/b assembly.  The tuple vector is consumed here, so the
/// matrices and rhs vectors are transferred without cloning.  A complete
/// mapping is required; otherwise the caller keeps the legacy evaluator.
fn move_model_decrease_payload(
    factors: &[WhitenedFactorRowStack],
    projected: Vec<(DMatrix<f32>, DVector<f32>, usize)>,
    compact: Option<&CompactLandmarkBackSubstitutionBatchF32>,
) -> Option<ModelDecreasePayloadF32> {
    let compact = compact?;
    // `zip` would silently truncate on a malformed producer result.  The
    // model sidecar is all-or-nothing: a count mismatch must select the
    // historical full-QR evaluator instead of mixing partial Q2 rows with
    // legacy visual factors.
    if projected.len() != factors.len() {
        return None;
    }
    let mut visual = Vec::new();
    let mut compact_entry_index = 0;
    for (factor_index, (factor, (q2_state, q2_residual, _rank))) in
        factors.iter().zip(projected).enumerate()
    {
        if factor.landmark_jacobian.ncols() == 0 {
            continue;
        }
        if factor.kind != FactorKind::Visual {
            return None;
        }
        let metadata = factor.landmark_metadata?;
        let entry = compact.entries.get(compact_entry_index)?;
        if entry.landmark_index != metadata.landmark_index
            || entry.track_id != metadata.track_id
            || entry.state_cols != factor.state_jacobian.ncols()
            || entry.landmark_cols != factor.landmark_jacobian.ncols()
        {
            return None;
        }
        visual.push(ModelDecreaseVisualF32 {
            factor_index,
            compact_entry_index,
            q2_state,
            q2_residual,
        });
        compact_entry_index += 1;
    }
    (compact_entry_index == compact.entries.len()).then_some(ModelDecreasePayloadF32 { visual })
}

impl CompactLandmarkBackSubstitutionEntryF32 {
    #[inline]
    fn storage_len(&self) -> Option<usize> {
        checked_compact_storage_len(self.landmark_cols, self.state_cols)
    }

    #[inline]
    fn view<'a>(&self, storage: &'a [f32]) -> Option<CompactLandmarkBackSubstitutionViewF32<'a>> {
        let end = self.storage_offset.checked_add(self.storage_len()?)?;
        storage.get(self.storage_offset..end)?;
        Some(CompactLandmarkBackSubstitutionViewF32 {
            storage,
            storage_offset: self.storage_offset,
            state_cols: self.state_cols,
            landmark_cols: self.landmark_cols,
            rank: self.rank,
            eligible: self.eligible,
        })
    }
}

impl CompactLandmarkBackSubstitutionBatchF32 {
    #[cfg(test)]
    fn as_slice(&self) -> &[CompactLandmarkBackSubstitutionEntryF32] {
        &self.entries
    }
}

#[derive(Debug, Clone)]
struct DiagnosticNormalSystemF32 {
    visual_h: DMatrix<f32>,
    visual_b: DVector<f32>,
    visual_imu_h: DMatrix<f32>,
    visual_imu_b: DVector<f32>,
    prior_h: DMatrix<f32>,
    prior_b: DVector<f32>,
}

#[derive(Debug, Clone)]
struct ImuReductionDiagnostic {
    local_blocks: Vec<ImuLocalBlockDiagnostic>,
    /// Cumulative IMU DenseAccumulator snapshots after each semantic
    /// IMU+bias pair.  These are observation-only copies used to distinguish
    /// a per-link product from the later visual/prior whole-matrix adds.
    imu_cumulative_stages: Vec<ImuCumulativeStageDiagnostic>,
    imu_h: DMatrix<f32>,
    imu_b: DVector<f32>,
}

#[derive(Debug, Clone)]
struct ImuCumulativeStageDiagnostic {
    active_offsets: Vec<usize>,
    imu_input_diagnostic: Option<serde_json::Value>,
    imu_h: DMatrix<f32>,
    imu_b: DVector<f32>,
}

#[derive(Debug, Clone)]
struct ImuLocalBlockDiagnostic {
    active_offsets: Vec<usize>,
    local_jacobian: DMatrix<f32>,
    residual: DVector<f32>,
    local_h: DMatrix<f32>,
    local_b: DVector<f32>,
    /// Input-boundary stages for this semantic IMU link.  This is populated
    /// only for the diagnostic path; the normal-equation reducer never reads
    /// it while assembling H/b.
    imu_input_diagnostic: Option<serde_json::Value>,
    local_vs_global_h_mismatches: usize,
    local_vs_global_b_mismatches: usize,
    /// Bit mismatches after embedding the 30x30 local product into the
    /// global state-width matrix.  This includes both the active block and
    /// the expected zero padding around it, so a native Eigen packet-boundary
    /// change is visible even when the active block happens to compare equal.
    local_vs_global_padded_h_mismatches: usize,
    local_vs_global_padded_b_mismatches: usize,
}

impl ReducedNormalSystemF32 {
    fn as_f64(&self) -> ReducedNormalSystem {
        ReducedNormalSystem {
            h: DMatrix::from_fn(self.h.nrows(), self.h.ncols(), |row, col| {
                self.h[(row, col)] as f64
            }),
            b: DVector::from_iterator(self.b.len(), self.b.iter().copied().map(f64::from)),
            back_substitution: self.back_substitution.clone(),
            diagnostic_stages: self.diagnostic_stages.as_ref().map(|stages| {
                DiagnosticNormalSystem {
                    visual_h: stages.visual_h.map(f64::from),
                    visual_b: stages.visual_b.map(f64::from),
                    visual_imu_h: stages.visual_imu_h.map(f64::from),
                    visual_imu_b: stages.visual_imu_b.map(f64::from),
                    prior_h: stages.prior_h.map(f64::from),
                    prior_b: stages.prior_b.map(f64::from),
                }
            }),
        }
    }

    /// Consume the clean reducer's compact payload after the state step has
    /// been solved.  Each entry is recovered exactly once with the existing
    /// f32 GEMV/triangular schedule and retained as an opaque mapped option;
    /// the concrete WindowProblem hook is responsible for validating the
    /// topology before accepting any of these values.
    fn into_trial_preparation(
        self,
        state: &DVector<f64>,
        state_step: &DVector<f64>,
        tolerance: f64,
    ) -> Option<LmTrialPreparation> {
        let compact = self.compact_back_substitution?;
        let CompactLandmarkBackSubstitutionBatchF32 { storage, entries } = compact;
        let landmark_steps = entries
            .into_iter()
            .map(|data| {
                let landmark_index = data.landmark_index;
                let track_id = data.track_id;
                let step = back_substitute_landmark_compact_entry_f32(
                    &data, &storage, state_step, tolerance,
                )
                .and_then(|step| {
                    (step.len() == 3 && step.iter().all(|value| value.is_finite()))
                        .then(|| Vector3::new(step[0], step[1], step[2]))
                });
                LmPreparedLandmarkStep {
                    landmark_index,
                    track_id,
                    step,
                }
            })
            .collect();
        Some(LmTrialPreparation {
            landmark_steps,
            tolerance_bits: tolerance.to_bits(),
            state_fingerprint: lm_trial_vector_fingerprint(state),
            step_fingerprint: lm_trial_vector_fingerprint(state_step),
        })
    }

    /// Evaluate the complete f32 model decrease from the clean reducer's
    /// retained QR sidecar.  Plain prior/IMU/bias rows remain on their
    /// existing per-factor path; only mapped visual rows use moved Q2 data.
    /// Any topology, rank, dimension, identity, or finite-value mismatch
    /// returns `None`, allowing the caller to use the historical full-QR
    /// evaluator without changing failure semantics.
    fn model_cost_decrease_from_payload(
        &self,
        factors: &[WhitenedFactorRowStack],
        state_step: &DVector<f64>,
        tolerance: f64,
    ) -> Option<f64> {
        let compact = self.compact_back_substitution.as_ref()?;
        let payload = self.model_decrease_payload.as_ref()?;
        let step = as_f32_vector(state_step);
        let mut decrease = 0.0_f32;
        let mut deferred_prior = Vec::new();
        let mut visual_ordinal = 0;
        let mut paired_bias_index = None;
        for (factor_index, factor) in factors.iter().enumerate() {
            if paired_bias_index == Some(factor_index) {
                paired_bias_index = None;
                continue;
            }
            if factor.state_jacobian.ncols() != step.len() {
                return None;
            }
            if factor.kind == FactorKind::Imu {
                if let Some(bias) = factors.get(factor_index + 1) {
                    if let Some(dot) = imu_bias_pair_model_dot_f32(factor, bias, &step) {
                        decrease -= dot;
                        paired_bias_index = Some(factor_index + 1);
                        continue;
                    }
                }
            }
            if let Some((j, rhs, compact_step)) = prior_model_inputs_f32(factor, state_step) {
                deferred_prior.push(prior_model_f32(&j, &rhs, &compact_step));
                continue;
            }
            if factor.landmark_jacobian.ncols() == 0 {
                let state = as_f32_matrix(&factor.state_jacobian);
                let residual = as_f32_vector(&factor.residual);
                let increment = &state * &step;
                let contribution = -increment.dot(&(0.5_f32 * &increment + &residual));
                if factor.kind == FactorKind::Prior {
                    deferred_prior.push(contribution);
                } else {
                    decrease += contribution;
                }
                continue;
            }
            if factor.kind != FactorKind::Visual {
                return None;
            }
            let visual = payload.visual.get(visual_ordinal)?;
            if visual.factor_index != factor_index {
                return None;
            }
            let metadata = factor.landmark_metadata?;
            let entry = compact.entries.get(visual.compact_entry_index)?;
            if entry.landmark_index != metadata.landmark_index
                || entry.track_id != metadata.track_id
                || entry.state_cols != step.len()
                || entry.landmark_cols != factor.landmark_jacobian.ncols()
            {
                return None;
            }
            let q1 = entry.view(&compact.storage)?;
            let dot = qr_payload_model_dot_reuse_f32(
                &q1,
                &visual.q2_state,
                &visual.q2_residual,
                state_step,
                tolerance,
            )?;
            decrease -= dot;
            visual_ordinal += 1;
        }
        for contribution in deferred_prior {
            decrease += contribution;
        }
        if visual_ordinal != payload.visual.len()
            || visual_ordinal != compact.entries.len()
            || !decrease.is_finite()
        {
            return None;
        }
        Some(decrease as f64)
    }
}

/// Compute the model-cost decrease for a linearized row stack after the
/// eliminated landmark block is back-substituted.
///
/// Basalt's `LinearizationAbsQR::backSubstitute` returns more than the
/// reduced-camera quadratic change.  Each landmark block contributes the
/// complete transformed `[Jp | Jl | r]` stack, including its first three
/// (`Q1`) rows.  Those rows account for the landmark-only part of the model
/// decrease and must be included in LM's `relative_decrease`; using only
/// `bᵀs + 1/2 sᵀHs` omits them and changes the lambda schedule.
pub fn model_cost_decrease(
    factors: &[WhitenedFactorRowStack],
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f64> {
    let mut decrease = 0.0;
    for factor in factors {
        if factor.state_jacobian.ncols() != state_step.len() {
            return None;
        }

        let landmark_columns = factor.landmark_jacobian.ncols();
        if landmark_columns == 0 {
            let j_inc = &factor.state_jacobian * state_step;
            decrease -= j_inc.dot(&(0.5 * &j_inc + &factor.residual));
            continue;
        }
        let (state_jacobian, landmark_jacobian, residual) = augmented_landmark_rows(factor);
        let rows = state_jacobian.nrows();
        if rows < landmark_columns {
            // There are not enough rows to recover the full landmark block;
            // landmark_steps() likewise leaves this point unchanged.
            let j_inc = &state_jacobian * state_step;
            decrease -= j_inc.dot(&(0.5 * &j_inc + &residual));
            continue;
        }

        // Mirror the upstream Householder path: transform the complete row
        // stack with the same Q, then solve the leading R block for the
        // landmark increment before evaluating the full model change.
        let qr = landmark_jacobian.clone().qr();
        let r = qr.r();
        let rank = (0..r.nrows().min(r.ncols()))
            .filter(|&i| r[(i, i)].abs() > tolerance)
            .count();
        if rank < landmark_columns {
            // WindowProblem::landmark_steps deliberately leaves a
            // rank-deficient landmark unchanged.  Match that trial path in
            // the model calculation instead of turning a newly observed,
            // underconstrained point into a solver-wide failure.
            let j_inc = &state_jacobian * state_step;
            decrease -= j_inc.dot(&(0.5 * &j_inc + &residual));
            continue;
        }

        let mut transformed_state = state_jacobian;
        let mut transformed_residual = DMatrix::from_column_slice(rows, 1, residual.as_slice());
        qr.q_tr_mul(&mut transformed_state);
        qr.q_tr_mul(&mut transformed_residual);

        let mut qj_inc = &transformed_state * state_step;
        let mut rhs = transformed_residual
            .column(0)
            .rows(0, landmark_columns)
            .into_owned();
        for row in 0..landmark_columns {
            rhs[row] += qj_inc[row];
        }
        rhs = -rhs;
        let mut landmark_inc = DVector::zeros(landmark_columns);
        for row in (0..landmark_columns).rev() {
            let mut value = rhs[row];
            for column in (row + 1)..landmark_columns {
                value -= r[(row, column)] * landmark_inc[column];
            }
            let diagonal = r[(row, row)];
            if diagonal.abs() <= tolerance {
                return None;
            }
            landmark_inc[row] = value / diagonal;
        }
        let q1_landmark_inc = r * landmark_inc;
        for row in 0..landmark_columns {
            qj_inc[row] += q1_landmark_inc[row];
        }

        let qres = transformed_residual.column(0).into_owned();
        decrease -= qj_inc.dot(&(0.5 * &qj_inc + qres));
    }
    decrease.is_finite().then_some(decrease)
}

fn model_cost_decrease_f32(
    factors: &[WhitenedFactorRowStack],
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f64> {
    let step = as_f32_vector(state_step);
    let threshold = tolerance as f32;
    let mut decrease = 0.0_f32;
    let mut deferred_prior = Vec::new();
    let mut paired_bias_index = None;
    for (factor_index, factor) in factors.iter().enumerate() {
        if paired_bias_index == Some(factor_index) {
            paired_bias_index = None;
            continue;
        }
        if factor.state_jacobian.ncols() != step.len() {
            return None;
        }
        if factor.kind == FactorKind::Imu {
            if let Some(bias) = factors.get(factor_index + 1) {
                if let Some(dot) = imu_bias_pair_model_dot_f32(factor, bias, &step) {
                    decrease -= dot;
                    paired_bias_index = Some(factor_index + 1);
                    continue;
                }
            }
        }
        if let Some((j, rhs, compact_step)) = prior_model_inputs_f32(factor, state_step) {
            deferred_prior.push(prior_model_f32(&j, &rhs, &compact_step));
            continue;
        }
        let state = as_f32_matrix(&factor.state_jacobian);
        let residual = as_f32_vector(&factor.residual);
        let landmark_columns = factor.landmark_jacobian.ncols();
        if landmark_columns == 0 {
            let increment = state * &step;
            let contribution = -increment.dot(&(0.5_f32 * &increment + &residual));
            if factor.kind == FactorKind::Prior {
                deferred_prior.push(contribution);
            } else {
                decrease += contribution;
            }
            continue;
        }
        let landmark = as_f32_matrix(&factor.landmark_jacobian);
        let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual)?;
        let rank = (0..landmark_columns)
            .filter(|&index| qr.pivots[index].abs() > threshold)
            .count();
        if rank < landmark_columns {
            let increment = state * &step;
            decrease -= increment.dot(&(0.5_f32 * &increment + &residual));
            continue;
        }
        let transformed_state = qr.transformed_state();
        let transformed_residual =
            DMatrix::from_column_slice(qr.rows, 1, qr.transformed_residual().as_slice());
        let r = qr.upper_r();
        let mut qj_inc = eigen_row_major_gemv_f32(&transformed_state, &step);
        let mut rhs = transformed_residual
            .column(0)
            .rows(0, landmark_columns)
            .into_owned();
        for row in 0..landmark_columns {
            rhs[row] += qj_inc[row];
        }
        let mut landmark_inc = DVector::<f32>::zeros(landmark_columns);
        if landmark_columns == 3 {
            let d2 = r[(2, 2)];
            let d1 = r[(1, 1)];
            let d0 = r[(0, 0)];
            if d2.abs() <= threshold || d1.abs() <= threshold || d0.abs() <= threshold {
                return None;
            }
            let x2 = rhs[2] / d2;
            let x1 = (-r[(1, 2)]).mul_add(x2, rhs[1]) / d1;
            let row0_dot = r[(0, 2)].mul_add(x2, r[(0, 1)] * x1);
            let x0 = (rhs[0] - row0_dot) / d0;
            landmark_inc[0] = -x0;
            landmark_inc[1] = -x1;
            landmark_inc[2] = -x2;
        } else {
            rhs = -rhs;
            for row in (0..landmark_columns).rev() {
                let mut value = rhs[row];
                for column in (row + 1)..landmark_columns {
                    value -= r[(row, column)] * landmark_inc[column];
                }
                let diagonal = r[(row, row)];
                if diagonal.abs() <= threshold {
                    return None;
                }
                landmark_inc[row] = value / diagonal;
            }
        }
        let q1_inc = r * landmark_inc;
        for row in 0..landmark_columns {
            qj_inc[row] += q1_inc[row];
        }
        let qres = transformed_residual.column(0).into_owned();
        decrease -= eigen_visual_model_dot_f32(&qj_inc, &qres);
    }
    // Upstream starts with the visual parallel-reduction result, applies all
    // ImuBlock terms, and only then adds the marginal-prior model change.
    for contribution in deferred_prior {
        decrease += contribution;
    }
    decrease.is_finite().then_some(decrease as f64)
}

/// Evaluate one native `ImuBlock` model term. Rust exposes the nine
/// preintegration rows and six bias-walk rows as adjacent semantic factors,
/// while upstream owns a single row-major 15x30 block and evaluates one GEMV
/// and one scalar-FMA dot over it.
fn imu_bias_pair_model_dot_f32(
    imu: &WhitenedFactorRowStack,
    bias: &WhitenedFactorRowStack,
    state_step: &DVector<f32>,
) -> Option<f32> {
    if imu.kind != FactorKind::Imu
        || bias.kind != FactorKind::Bias
        || imu.rows() != 9
        || bias.rows() != 6
        || imu.landmark_jacobian.ncols() != 0
        || bias.landmark_jacobian.ncols() != 0
        || imu.state_jacobian.ncols() != state_step.len()
        || bias.state_jacobian.ncols() != state_step.len()
    {
        return None;
    }
    let offsets = imu.imu_link_offsets?;
    if bias.imu_link_offsets != Some(offsets)
        || offsets.start + AOM_NAV_DOF > state_step.len()
        || offsets.end + AOM_NAV_DOF > state_step.len()
        || offsets.start == offsets.end
    {
        return None;
    }

    let imu_jacobian = as_f32_matrix(&imu.state_jacobian);
    let bias_jacobian = as_f32_matrix(&bias.state_jacobian);
    let mut local_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, IMU_LOCAL_COLS);
    for (target_row, source) in imu_jacobian.row_iter().enumerate() {
        for (block, global_offset) in [offsets.start, offsets.end].into_iter().enumerate() {
            local_jacobian
                .view_mut((target_row, block * AOM_NAV_DOF), (1, AOM_NAV_DOF))
                .copy_from(&source.columns(global_offset, AOM_NAV_DOF));
        }
    }
    for (source_row, source) in bias_jacobian.row_iter().enumerate() {
        for (block, global_offset) in [offsets.start, offsets.end].into_iter().enumerate() {
            local_jacobian
                .view_mut((9 + source_row, block * AOM_NAV_DOF), (1, AOM_NAV_DOF))
                .copy_from(&source.columns(global_offset, AOM_NAV_DOF));
        }
    }
    let mut local_residual = DVector::<f32>::zeros(IMU_LOCAL_ROWS);
    local_residual
        .rows_mut(0, 9)
        .copy_from(&as_f32_vector(&imu.residual));
    local_residual
        .rows_mut(9, 6)
        .copy_from(&as_f32_vector(&bias.residual));
    let local_step = DVector::from_iterator(
        IMU_LOCAL_COLS,
        (0..IMU_LOCAL_COLS).map(|local_column| {
            let block = local_column / AOM_NAV_DOF;
            let global_offset = if block == 0 {
                offsets.start
            } else {
                offsets.end
            };
            state_step[global_offset + local_column % AOM_NAV_DOF]
        }),
    );
    let increment = eigen_imu_15x30_gemv_f32(&local_jacobian, &local_step);
    Some(eigen_imu_15_model_dot_f32(&increment, &local_residual))
}

/// Reproduce the pinned Eigen column-major `15x30 * Vector30f` kernel used
/// inside `ImuBlock<float>::backSubstitute`. The native kernel vectorizes
/// across the 15 output rows, so every output lane accumulates columns 0..29
/// with one FMA per column; it does not horizontally reduce input packets.
#[inline]
fn eigen_imu_15x30_gemv_f32(jacobian: &DMatrix<f32>, step: &DVector<f32>) -> DVector<f32> {
    assert_eq!(jacobian.shape(), (IMU_LOCAL_ROWS, IMU_LOCAL_COLS));
    assert_eq!(step.len(), IMU_LOCAL_COLS);
    let mut result = DVector::<f32>::zeros(IMU_LOCAL_ROWS);
    for column in 0..IMU_LOCAL_COLS {
        for row in 0..IMU_LOCAL_ROWS {
            result[row] = jacobian[(row, column)].mul_add(step[column], result[row]);
        }
    }
    result
}

/// Reproduce the pinned 15-element Eigen dot in `ImuBlock::backSubstitute`.
/// Elements 0..7 are formed lane-wise and reduced with AVX Packet8 predux;
/// the seven remaining elements are then folded into that scalar with FMA.
#[inline]
fn eigen_imu_15_model_dot_f32(increment: &DVector<f32>, residual: &DVector<f32>) -> f32 {
    assert_eq!(increment.len(), IMU_LOCAL_ROWS);
    assert_eq!(residual.len(), IMU_LOCAL_ROWS);
    let lanes: [f32; 8] = std::array::from_fn(|index| {
        let right = 0.5_f32.mul_add(increment[index], residual[index]);
        increment[index] * right
    });
    let mut value = eigen_predux8_f32(lanes);
    for index in 8..IMU_LOCAL_ROWS {
        let right = 0.5_f32.mul_add(increment[index], residual[index]);
        value = increment[index].mul_add(right, value);
    }
    value
}

fn prior39_model_f32(j: &DMatrix<f32>, rhs: &DVector<f32>, step: &DVector<f32>) -> f32 {
    assert_eq!(j.shape(), (39, 39));
    assert_eq!(rhs.len(), 39);
    assert_eq!(step.len(), 39);
    let mut inc = [0.0_f32; 39];
    for row in 0..39 {
        for col in 0..39 {
            inc[row] = j[(row, col)].mul_add(step[col], inc[row]);
        }
    }
    let right: [f32; 39] = std::array::from_fn(|i| 0.5_f32.mul_add(inc[i], rhs[i]));
    let products: [f32; 32] = std::array::from_fn(|i| -inc[i] * right[i]);
    let lanes: [f32; 8] = std::array::from_fn(|i| {
        products[i] + ((products[i + 16] + products[i + 24]) + products[i + 8])
    });
    let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    let mut total = (half[0] + half[2]) + (half[1] + half[3]);
    // Native 39f0dd half FMA, then 39f0e3 vfnmadd231ss.
    for i in 32..39 {
        total = (-inc[i]).mul_add(right[i], total);
    }
    total
}

/// Reproduce Eigen's AVX inner-product kernel for the 45-row marginal prior.
/// The first and fifth packets are combined with FMA, packets two through four
/// use Eigen's fixed addition tree, and rows 40..44 form the scalar FMA tail.
fn prior45_model_f32(j: &DMatrix<f32>, rhs: &DVector<f32>, step: &DVector<f32>) -> f32 {
    assert_eq!(j.shape(), (45, 45));
    assert_eq!(rhs.len(), 45);
    assert_eq!(step.len(), 45);
    let mut inc = [0.0_f32; 45];
    for row in 0..45 {
        for col in 0..45 {
            inc[row] = j[(row, col)].mul_add(step[col], inc[row]);
        }
    }
    let right: [f32; 45] = std::array::from_fn(|i| 0.5_f32.mul_add(inc[i], rhs[i]));
    let lanes: [f32; 8] = std::array::from_fn(|i| {
        let first_and_fifth = (-inc[i + 32]).mul_add(right[i + 32], -inc[i] * right[i]);
        let middle = (-inc[i + 8] * right[i + 8])
            + ((-inc[i + 16] * right[i + 16]) + (-inc[i + 24] * right[i + 24]));
        first_and_fifth + middle
    });
    let mut total = eigen_predux8_f32(lanes);
    for i in 40..45 {
        total = (-inc[i]).mul_add(right[i], total);
    }
    total
}

fn prior_model_f32(j: &DMatrix<f32>, rhs: &DVector<f32>, step: &DVector<f32>) -> f32 {
    match j.shape() {
        (39, 39) => prior39_model_f32(j, rhs, step),
        (45, 45) => prior45_model_f32(j, rhs, step),
        shape => panic!("unsupported marginal-prior model shape {shape:?}"),
    }
}

/// Compact only explicitly mapped prior columns; other shapes use the generic evaluator.
fn prior_model_inputs_f32(
    factor: &WhitenedFactorRowStack,
    step: &DVector<f64>,
) -> Option<(DMatrix<f32>, DVector<f32>, DVector<f32>)> {
    let size = factor.rows();
    if factor.kind != FactorKind::Prior
        || !matches!(size, 39 | 45)
        || factor.landmark_jacobian.ncols() != 0
    {
        return None;
    }
    let columns = factor.prior_state_columns.as_ref()?;
    if columns.len() != size || factor.state_jacobian.ncols() != step.len() {
        return None;
    }
    for (i, &column) in columns.iter().enumerate() {
        if column >= step.len() || columns[..i].contains(&column) {
            return None;
        }
    }
    for col in 0..step.len() {
        if !columns.contains(&col) && factor.state_jacobian.column(col).iter().any(|x| *x != 0.0) {
            return None;
        }
    }
    let j = DMatrix::from_fn(size, size, |row, col| {
        factor.state_jacobian[(row, columns[col])] as f32
    });
    let rhs = as_f32_vector(&factor.residual);
    let compact_step = DVector::from_iterator(size, columns.iter().map(|&col| step[col] as f32));
    Some((j, rhs, compact_step))
}

fn diagnostic_prior39_model_input(
    factor: &WhitenedFactorRowStack,
    step: &DVector<f64>,
) -> Option<serde_json::Value> {
    let (j, rhs, compact_step) = prior_model_inputs_f32(factor, step)?;
    let columns = factor.prior_state_columns.as_ref()?;
    let candidate = prior_model_f32(&j, &rhs, &compact_step);
    Some(serde_json::json!({
        "columns": columns,
        "jacobian_column_major_bits": j.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>(),
        "rhs_bits": rhs.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>(),
        "step_bits": compact_step.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>(),
        "candidate_bits": format!("{:08x}", candidate.to_bits()),
    }))
}

/// Observation-only per-factor decomposition of the current model evaluator.
/// This never supplies LM's acceptance value or changes its factor order.
pub(crate) fn diagnostic_model_decrease_parts_f32(
    factors: &[WhitenedFactorRowStack],
    step: &DVector<f64>,
    actual: f64,
) -> Option<serde_json::Value> {
    let mut parts = Vec::with_capacity(factors.len());
    let mut original = 0.0_f32;
    let mut visual = 0.0_f32;
    let mut non_prior = 0.0_f32;
    let mut prior = 0.0_f32;
    let step_f32 = as_f32_vector(step);
    let mut paired_bias_index = None;
    for (index, factor) in factors.iter().enumerate() {
        let (value, paired_with_previous) = if paired_bias_index == Some(index) {
            paired_bias_index = None;
            (0.0_f32, true)
        } else if factor.kind == FactorKind::Imu {
            if let Some(bias) = factors.get(index + 1) {
                if let Some(dot) = imu_bias_pair_model_dot_f32(factor, bias, &step_f32) {
                    paired_bias_index = Some(index + 1);
                    (-dot, false)
                } else {
                    (
                        model_cost_decrease_f32(std::slice::from_ref(factor), step, 1e-10)? as f32,
                        false,
                    )
                }
            } else {
                (
                    model_cost_decrease_f32(std::slice::from_ref(factor), step, 1e-10)? as f32,
                    false,
                )
            }
        } else {
            (
                model_cost_decrease_f32(std::slice::from_ref(factor), step, 1e-10)? as f32,
                false,
            )
        };
        if factor.kind != FactorKind::Prior {
            original += value;
        }
        if factor.kind == FactorKind::Visual {
            visual += value;
        }
        if factor.kind == FactorKind::Prior {
            prior += value;
        } else {
            non_prior += value;
        }
        parts.push(serde_json::json!({
            "factor_index": index,
            "kind": full70_factor_kind_name(factor.kind),
            "rows": factor.rows(),
            "decrease_bits": format!("{:08x}", value.to_bits()),
            "original_cumulative_bits": format!("{:08x}", original.to_bits()),
            "prior39_input": diagnostic_prior39_model_input(factor, step),
            "paired_with_previous": paired_with_previous,
        }));
    }
    // Match LinearizationAbsQR::backSubstitute: visual reduction, IMU
    // blocks, then the marginal-prior contribution.
    original += prior;
    Some(serde_json::json!({
        "parts": parts,
        "original_total_bits": format!("{:08x}", original.to_bits()),
        "direct_total_bits": format!("{:08x}", (actual as f32).to_bits()),
        "reconstruction_exact": original.to_bits() == (actual as f32).to_bits(),
        "visual_bits": format!("{:08x}", visual.to_bits()),
        "non_prior_bits": format!("{:08x}", non_prior.to_bits()),
        "prior_bits": format!("{:08x}", prior.to_bits()),
        "prior_last_candidate_bits": format!("{:08x}", (non_prior + prior).to_bits()),
    }))
}

/// Test-only reduced normal-system model-decrease candidate.
///
/// This is deliberately not used by either LM loop.  It evaluates the usual
/// reduced quadratic, `-(bᵀs + 1/2 sᵀHs)`, using the already-audited f32
/// row-major GEMV schedule.  The production model evaluator above operates
/// on the complete transformed factor rows and therefore also includes each
/// landmark block's Q1 contribution.  Keeping this candidate private makes
/// that semantic distinction executable without changing the accept/reject
/// contract.
#[cfg(test)]
fn reduced_model_cost_decrease_f32(
    reduced: &ReducedNormalSystemF32,
    state_step: &DVector<f64>,
) -> Option<f64> {
    if reduced.h.nrows() != reduced.h.ncols()
        || reduced.h.nrows() != reduced.b.len()
        || reduced.h.nrows() != state_step.len()
    {
        return None;
    }
    let step = as_f32_vector(state_step);
    let h_step = eigen_row_major_gemv_f32(&reduced.h, &step);
    let mut linear = 0.0_f32;
    let mut quadratic = 0.0_f32;
    for index in 0..step.len() {
        linear = step[index].mul_add(reduced.b[index], linear);
        quadratic = step[index].mul_add(h_step[index], quadratic);
    }
    let decrease = -(linear + 0.5_f32 * quadratic);
    decrease.is_finite().then_some(decrease as f64)
}

/// Test-only reduced model candidate with the visual Q1 residual constant.
///
/// For a full-rank visual ABS-QR block, landmark elimination makes its Q1
/// state increment exactly `-Q1^T r`; its contribution to the model decrease
/// is therefore `+0.5 * ||Q1^T r||^2`, independent of the solved state step.
/// The reduced H/b system supplies the Q2 and non-visual terms.  This helper
/// intentionally remains disconnected from both LM loops until its arithmetic
/// and accumulation order are proven against the complete transformed-row
/// evaluator.
#[cfg(test)]
fn reduced_model_cost_decrease_with_q1_constant_f32(
    reduced: &ReducedNormalSystemF32,
    q1_entries: &[CompactLandmarkBackSubstitutionF32],
    state_step: &DVector<f64>,
) -> Option<f64> {
    let reduced_decrease = reduced_model_cost_decrease_f32(reduced, state_step)? as f32;
    let mut q1_constant = 0.0_f32;
    for entry in q1_entries {
        if !entry.eligible || entry.rank != entry.landmark_cols {
            return None;
        }
        let mut squared_norm = 0.0_f32;
        for row in 0..entry.q1_residual_len() {
            let residual = entry.q1_residual_value(row);
            squared_norm = residual.mul_add(residual, squared_norm);
        }
        let factor_constant = 0.5_f32 * squared_norm;
        q1_constant = add_f32_exact(q1_constant, factor_constant);
    }
    let decrease = add_f32_exact(reduced_decrease, q1_constant);
    decrease.is_finite().then_some(decrease as f64)
}

/// Test-only payload containing both sides of one already-computed landmark
/// QR.  The helper below deliberately consumes these values without calling
/// [`LandmarkHouseholderF32::factor`] again: it is the primitive we would need
/// if the clean reducer retained Q2 rows for the later model-decrease pass.
#[cfg(test)]
struct QrModelReusePayloadF32 {
    q1: CompactLandmarkBackSubstitutionF32,
    q2_state: DMatrix<f32>,
    q2_residual: DVector<f32>,
}

/// Evaluate the complete transformed-row dot term from a retained Q1/Q2 QR
/// payload.  This mirrors the visual branch of `model_cost_decrease_f32`,
/// including its pinned Eigen GEMV, triangular solve, R*landmark-inc product,
/// row order, and final scalar-FMA dot, but performs no QR.
fn qr_payload_model_dot_reuse_f32<Q1: QrModelQ1ViewF32>(
    q1: &Q1,
    q2_state: &DMatrix<f32>,
    q2_residual: &DVector<f32>,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f32> {
    let n = q1.landmark_cols();
    let state_cols = q1.state_cols();
    if !q1.eligible()
        || q1.rank() != n
        || n == 0
        || n > 3
        || state_cols != state_step.len()
        || q2_state.ncols() != state_cols
        || q2_residual.len() != q2_state.nrows()
    {
        return None;
    }

    let step = as_f32_vector(state_step);
    let q2_rows = q2_state.nrows();
    let rows = n + q2_rows;
    let transformed_state = DMatrix::from_fn(rows, state_cols, |row, column| {
        if row < n {
            q1.q1_state_value(row, column)
        } else {
            q2_state[(row - n, column)]
        }
    });
    let transformed_residual = DVector::from_iterator(
        rows,
        (0..rows).map(|row| {
            if row < n {
                q1.q1_residual_value(row)
            } else {
                q2_residual[row - n]
            }
        }),
    );
    let upper_r = DMatrix::from_fn(n, n, |row, column| q1.upper_r_value(row, column));

    let mut qj_inc = eigen_row_major_gemv_f32(&transformed_state, &step);
    let mut rhs = transformed_residual.rows(0, n).into_owned();
    for row in 0..n {
        rhs[row] += qj_inc[row];
    }
    let threshold = tolerance as f32;
    let mut landmark_inc = DVector::<f32>::zeros(n);
    if n == 3 {
        let d2 = upper_r[(2, 2)];
        let d1 = upper_r[(1, 1)];
        let d0 = upper_r[(0, 0)];
        if d2.abs() <= threshold || d1.abs() <= threshold || d0.abs() <= threshold {
            return None;
        }
        let x2 = rhs[2] / d2;
        let x1 = (-upper_r[(1, 2)]).mul_add(x2, rhs[1]) / d1;
        let row0_dot = upper_r[(0, 2)].mul_add(x2, upper_r[(0, 1)] * x1);
        let x0 = (rhs[0] - row0_dot) / d0;
        landmark_inc[0] = -x0;
        landmark_inc[1] = -x1;
        landmark_inc[2] = -x2;
    } else {
        rhs = -rhs;
        for row in (0..n).rev() {
            let mut value = rhs[row];
            for column in (row + 1)..n {
                value -= upper_r[(row, column)] * landmark_inc[column];
            }
            let diagonal = upper_r[(row, row)];
            if diagonal.abs() <= threshold {
                return None;
            }
            landmark_inc[row] = value / diagonal;
        }
    }
    let q1_landmark_inc = &upper_r * &landmark_inc;
    for row in 0..n {
        qj_inc[row] += q1_landmark_inc[row];
    }
    let dot = eigen_visual_model_dot_f32(&qj_inc, &transformed_residual);
    dot.is_finite().then_some(dot)
}

#[cfg(test)]
fn qr_payload_model_dot_f32(
    payload: &QrModelReusePayloadF32,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f32> {
    qr_payload_model_dot_reuse_f32(
        &payload.q1,
        &payload.q2_state,
        &payload.q2_residual,
        state_step,
        tolerance,
    )
}

#[cfg(test)]
fn model_cost_decrease_from_qr_payload_f32(
    payload: &QrModelReusePayloadF32,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f64> {
    let dot = qr_payload_model_dot_f32(payload, state_step, tolerance)?;
    let mut decrease = 0.0_f32;
    decrease -= dot;
    decrease.is_finite().then_some(decrease as f64)
}

#[cfg(test)]
enum ModelReusePayloadF32 {
    Visual(QrModelReusePayloadF32),
    Plain {
        kind: FactorKind,
        state: DMatrix<f32>,
        residual: DVector<f32>,
    },
}

#[cfg(test)]
fn model_cost_decrease_from_payloads_f32(
    payloads: &[ModelReusePayloadF32],
    factors: &[WhitenedFactorRowStack],
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f64> {
    if payloads.len() != factors.len() {
        return None;
    }
    let step = as_f32_vector(state_step);
    let mut decrease = 0.0_f32;
    let mut deferred_prior = Vec::new();
    let mut paired_bias_index = None;
    for (index, payload) in payloads.iter().enumerate() {
        if paired_bias_index == Some(index) {
            paired_bias_index = None;
            continue;
        }
        let factor = &factors[index];
        if factor.kind == FactorKind::Imu {
            if let Some(bias) = factors.get(index + 1) {
                if let Some(dot) = imu_bias_pair_model_dot_f32(factor, bias, &step) {
                    decrease -= dot;
                    paired_bias_index = Some(index + 1);
                    continue;
                }
            }
        }
        let (dot, defer) = match payload {
            ModelReusePayloadF32::Visual(payload) => (
                qr_payload_model_dot_f32(payload, state_step, tolerance)?,
                false,
            ),
            ModelReusePayloadF32::Plain {
                kind,
                state,
                residual,
            } => {
                if state.ncols() != step.len() || state.nrows() != residual.len() {
                    return None;
                }
                let increment = state * &step;
                (
                    increment.dot(&(0.5_f32 * &increment + residual)),
                    *kind == FactorKind::Prior,
                )
            }
        };
        if defer {
            deferred_prior.push(-dot);
        } else {
            decrease -= dot;
        }
    }
    for contribution in deferred_prior {
        decrease += contribution;
    }
    decrease.is_finite().then_some(decrease as f64)
}

/// Append the zero landmark-damping rows that ABS_QR allocates in every
/// landmark block.  The upstream block has `obs_rows + 3` storage rows and
/// removes only its first three Q1 rows, so its Q2 contribution has
/// `obs_rows` rows (the final three are zero directions before damping).  A
/// factor stored in Rust contains only the observation rows; omitting these
/// rows changes the QR reflector sequence and the canonical row spans.
fn augmented_landmark_rows(
    factor: &WhitenedFactorRowStack,
) -> (DMatrix<f64>, DMatrix<f64>, DVector<f64>) {
    let landmark_columns = factor.landmark_jacobian.ncols();
    let rows = factor.rows() + landmark_columns;
    let mut state_jacobian = DMatrix::zeros(rows, factor.state_jacobian.ncols());
    let mut landmark_jacobian = DMatrix::zeros(rows, landmark_columns);
    let mut residual = DVector::zeros(rows);
    state_jacobian
        .view_mut((0, 0), factor.state_jacobian.shape())
        .copy_from(&factor.state_jacobian);
    landmark_jacobian
        .view_mut((0, 0), factor.landmark_jacobian.shape())
        .copy_from(&factor.landmark_jacobian);
    residual
        .rows_mut(0, factor.rows())
        .copy_from(&factor.residual);
    (state_jacobian, landmark_jacobian, residual)
}

/// Project each landmark's rows into the left nullspace of J_l using QR.
/// This is the ABS_QR operation: the three Q1 rows are removed from the
/// augmented storage block while the zero damping directions remain in Q2.
pub fn landmark_nullspace_projection(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
) -> (DMatrix<f64>, DVector<f64>, usize) {
    // Basalt's ABS_QR implementation applies the landmark Householder
    // reflectors to the complete `[Jp | Jl | r]` row stack and retains rows
    // after the three landmark columns.  Forming `I - Jl(JlᵀJl)⁻¹Jlᵀ`
    // appears algebraically equivalent, but squares the landmark condition
    // number and is observably different on the float f4 system.  Keep the
    // QR row operation explicit so the reduced H/b follows the upstream
    // grouped-row ordering and numerical path.
    let landmark_columns = factor.landmark_jacobian.ncols();
    if landmark_columns == 0 {
        return (factor.state_jacobian.clone(), factor.residual.clone(), 0);
    }
    let (state_jacobian, landmark_jacobian, residual) = augmented_landmark_rows(factor);
    let qr = landmark_jacobian.qr();
    let r = qr.r();
    let rank = (0..r.nrows().min(r.ncols()))
        .filter(|&i| r[(i, i)].abs() > tolerance)
        .count();

    let mut transformed_state = state_jacobian;
    let mut transformed_residual =
        DMatrix::from_column_slice(residual.len(), 1, residual.as_slice());
    qr.q_tr_mul(&mut transformed_state);
    qr.q_tr_mul(&mut transformed_residual);

    if residual.len() <= landmark_columns {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            rank,
        );
    }
    let reduced_rows = residual.len() - landmark_columns;
    (
        transformed_state
            .rows(landmark_columns, reduced_rows)
            .into_owned(),
        transformed_residual
            .rows(landmark_columns, reduced_rows)
            .column(0)
            .into_owned(),
        rank,
    )
}

pub fn reduce_landmark_factors(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> ReducedNormalSystem {
    let mut h = DMatrix::zeros(state_dof, state_dof);
    let mut b = DVector::zeros(state_dof);
    let mut back = Vec::with_capacity(factors.len());
    for f in factors {
        assert_eq!(f.state_jacobian.ncols(), state_dof);
        let (j, r, rank) = landmark_nullspace_projection(f, tolerance);
        h += j.transpose() * &j;
        b += j.transpose() * &r;
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

fn as_f32_matrix(matrix: &DMatrix<f64>) -> DMatrix<f32> {
    DMatrix::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)] as f32
    })
}

fn as_f32_vector(vector: &DVector<f64>) -> DVector<f32> {
    DVector::from_iterator(
        vector.len(),
        vector.iter().copied().map(|value| value as f32),
    )
}

fn as_f64_vector(vector: &DVector<f32>) -> DVector<f64> {
    DVector::from_iterator(vector.len(), vector.iter().copied().map(f64::from))
}

// Eigen's pinned x86 f32 path uses eight-lane AVX packets (and four-lane
// half-packets).  Keeping this emulation local to ABS_QR makes the storage and
// arithmetic boundary explicit without target-specific intrinsics or unsafe
// code.
#[derive(Clone, Copy)]
struct LandmarkPacket4([f32; 4]);

#[inline]
fn landmark_packet_load(data: &[f32], start: usize) -> LandmarkPacket4 {
    LandmarkPacket4([
        data[start],
        data[start + 1],
        data[start + 2],
        data[start + 3],
    ])
}

#[inline]
fn landmark_packet_store(data: &mut [f32], start: usize, value: LandmarkPacket4) {
    data[start] = value.0[0];
    data[start + 1] = value.0[1];
    data[start + 2] = value.0[2];
    data[start + 3] = value.0[3];
}

#[inline]
fn landmark_packet_add(lhs: LandmarkPacket4, rhs: LandmarkPacket4) -> LandmarkPacket4 {
    LandmarkPacket4([
        lhs.0[0] + rhs.0[0],
        lhs.0[1] + rhs.0[1],
        lhs.0[2] + rhs.0[2],
        lhs.0[3] + rhs.0[3],
    ])
}

#[inline]
fn landmark_packet_fused_sub(
    destination: LandmarkPacket4,
    workspace: LandmarkPacket4,
    scale: f32,
) -> LandmarkPacket4 {
    LandmarkPacket4([
        (-scale).mul_add(workspace.0[0], destination.0[0]),
        (-scale).mul_add(workspace.0[1], destination.0[1]),
        (-scale).mul_add(workspace.0[2], destination.0[2]),
        (-scale).mul_add(workspace.0[3], destination.0[3]),
    ])
}

#[inline]
fn landmark_packet_mul_add(
    lhs: LandmarkPacket4,
    rhs: LandmarkPacket4,
    accumulator: LandmarkPacket4,
) -> LandmarkPacket4 {
    // Eigen's SSE/AVX pmadd uses fused multiply-add when FMA is enabled.
    LandmarkPacket4([
        lhs.0[0].mul_add(rhs.0[0], accumulator.0[0]),
        lhs.0[1].mul_add(rhs.0[1], accumulator.0[1]),
        lhs.0[2].mul_add(rhs.0[2], accumulator.0[2]),
        lhs.0[3].mul_add(rhs.0[3], accumulator.0[3]),
    ])
}

#[derive(Clone, Copy)]
struct LandmarkPacket8([f32; 8]);

#[inline]
fn landmark_packet8_load(data: &[f32], start: usize) -> LandmarkPacket8 {
    LandmarkPacket8([
        data[start],
        data[start + 1],
        data[start + 2],
        data[start + 3],
        data[start + 4],
        data[start + 5],
        data[start + 6],
        data[start + 7],
    ])
}

#[inline]
fn landmark_packet8_store(data: &mut [f32], start: usize, value: LandmarkPacket8) {
    data[start] = value.0[0];
    data[start + 1] = value.0[1];
    data[start + 2] = value.0[2];
    data[start + 3] = value.0[3];
    data[start + 4] = value.0[4];
    data[start + 5] = value.0[5];
    data[start + 6] = value.0[6];
    data[start + 7] = value.0[7];
}

#[inline]
fn landmark_packet8_add(lhs: LandmarkPacket8, rhs: LandmarkPacket8) -> LandmarkPacket8 {
    LandmarkPacket8([
        lhs.0[0] + rhs.0[0],
        lhs.0[1] + rhs.0[1],
        lhs.0[2] + rhs.0[2],
        lhs.0[3] + rhs.0[3],
        lhs.0[4] + rhs.0[4],
        lhs.0[5] + rhs.0[5],
        lhs.0[6] + rhs.0[6],
        lhs.0[7] + rhs.0[7],
    ])
}

#[inline]
fn landmark_packet8_fused_sub(
    destination: LandmarkPacket8,
    workspace: LandmarkPacket8,
    scale: f32,
) -> LandmarkPacket8 {
    LandmarkPacket8([
        (-scale).mul_add(workspace.0[0], destination.0[0]),
        (-scale).mul_add(workspace.0[1], destination.0[1]),
        (-scale).mul_add(workspace.0[2], destination.0[2]),
        (-scale).mul_add(workspace.0[3], destination.0[3]),
        (-scale).mul_add(workspace.0[4], destination.0[4]),
        (-scale).mul_add(workspace.0[5], destination.0[5]),
        (-scale).mul_add(workspace.0[6], destination.0[6]),
        (-scale).mul_add(workspace.0[7], destination.0[7]),
    ])
}

#[inline]
fn landmark_packet8_mul_add(
    lhs: LandmarkPacket8,
    rhs: LandmarkPacket8,
    accumulator: LandmarkPacket8,
) -> LandmarkPacket8 {
    LandmarkPacket8([
        lhs.0[0].mul_add(rhs.0[0], accumulator.0[0]),
        lhs.0[1].mul_add(rhs.0[1], accumulator.0[1]),
        lhs.0[2].mul_add(rhs.0[2], accumulator.0[2]),
        lhs.0[3].mul_add(rhs.0[3], accumulator.0[3]),
        lhs.0[4].mul_add(rhs.0[4], accumulator.0[4]),
        lhs.0[5].mul_add(rhs.0[5], accumulator.0[5]),
        lhs.0[6].mul_add(rhs.0[6], accumulator.0[6]),
        lhs.0[7].mul_add(rhs.0[7], accumulator.0[7]),
    ])
}

/// Packet workspace used by Eigen's column-major GEMV path for
/// `essential.adjoint() * bottom`.
///
/// Eigen transposes this vector-on-the-left product.  The resulting
/// `bottom.transpose()` is treated as a column-major matrix, so AVX packets
/// span eight output columns (with a four-lane half-packet tail) while the
/// essential dimension is accumulated scalar-by-scalar.  This is subtly
/// different from packetizing the dot product itself and is the boundary that
/// makes the f4 trace bitwise exact.
#[inline]
fn landmark_gemv_workspace(
    essential: &[f32],
    storage: &[f32],
    row_start: usize,
    storage_cols: usize,
) -> Vec<f32> {
    let mut result = Vec::new();
    landmark_gemv_workspace_into(essential, storage, row_start, storage_cols, &mut result);
    result
}

/// Capacity-reusing form of [`landmark_gemv_workspace`].  The evaluator
/// schedule is intentionally identical: only the destination Vec ownership
/// changes, so the clean reducer can reuse this scratch between landmark
/// factors without changing any f32 operation or traversal order.
#[inline]
fn landmark_gemv_workspace_into(
    essential: &[f32],
    storage: &[f32],
    row_start: usize,
    storage_cols: usize,
    result: &mut Vec<f32>,
) {
    debug_assert!(row_start + essential.len() <= storage.len() / storage_cols);
    result.clear();
    result.resize(storage_cols, 0.0_f32);
    let packet_end = storage_cols / 8 * 8;
    let mut column = 0;
    while column < packet_end {
        let mut packet = LandmarkPacket8([0.0; 8]);
        for (offset, &value) in essential.iter().enumerate() {
            let rhs = LandmarkPacket8([value; 8]);
            let lhs = landmark_packet8_load(storage, (row_start + offset) * storage_cols + column);
            packet = landmark_packet8_mul_add(lhs, rhs, packet);
        }
        landmark_packet8_store(result, column, packet);
        column += 8;
    }
    let half_packet_end = storage_cols / 4 * 4;
    while column < half_packet_end {
        let mut packet = LandmarkPacket4([0.0; 4]);
        for (offset, &value) in essential.iter().enumerate() {
            let rhs = LandmarkPacket4([value; 4]);
            let lhs = landmark_packet_load(storage, (row_start + offset) * storage_cols + column);
            packet = landmark_packet_mul_add(lhs, rhs, packet);
        }
        landmark_packet_store(result, column, packet);
        column += 4;
    }
    while column < storage_cols {
        let mut value = 0.0_f32;
        for (offset, &essential_value) in essential.iter().enumerate() {
            value += essential_value * storage[(row_start + offset) * storage_cols + column];
        }
        result[column] = value;
        column += 1;
    }
}

/// Scratch buffers moved into and out of one [`LandmarkHouseholderF32`] at a
/// time.  A QR result still owns its full transformed storage because Q2 rows
/// are returned to the caller, but the storage/pivot/tau vectors and the
/// Householder essential/GEMV workspaces are recycled after each projection.
/// This is deliberately capacity-only reuse; no matrix arithmetic or source
/// order is changed.
#[derive(Default)]
struct LandmarkHouseholderWorkspace {
    storage: Vec<f32>,
    pivots: Vec<f32>,
    tau: Vec<f32>,
    essential: Vec<f32>,
    gemv: Vec<f32>,
}

impl LandmarkHouseholderWorkspace {
    #[inline]
    fn take_zeroed(buffer: &mut Vec<f32>, len: usize) -> Vec<f32> {
        let mut value = std::mem::take(buffer);
        value.clear();
        value.resize(len, 0.0_f32);
        // `resize` does not overwrite elements when the old capacity/length
        // already covers `len`; the upstream storage starts zeroed on every
        // factor, including padding and damping rows.
        value.fill(0.0_f32);
        value
    }

    #[inline]
    fn recycle(buffer: &mut Vec<f32>, mut value: Vec<f32>) {
        value.clear();
        *buffer = value;
    }
}

#[derive(Debug, Clone, Copy)]
struct LandmarkHouseholderLayout {
    rows: usize,
    landmark_offset: usize,
    residual_offset: usize,
    storage_cols: usize,
    storage_len: usize,
}

/// Compute every derived ABS_QR dimension before moving any reusable buffer.
/// The production factors are small, but keeping this arithmetic checked makes
/// malformed or adversarial matrix metadata fail closed instead of wrapping
/// into a short allocation followed by an indexing panic.
#[inline]
fn checked_landmark_householder_layout(
    observation_rows: usize,
    state_cols: usize,
    landmark_cols: usize,
) -> Option<LandmarkHouseholderLayout> {
    if landmark_cols == 0 || landmark_cols > 3 {
        return None;
    }
    let rows = observation_rows.checked_add(landmark_cols)?;
    let padding_cols = 4usize.checked_sub(state_cols % 4)?;
    let landmark_offset = state_cols.checked_add(padding_cols)?;
    let residual_offset = landmark_offset.checked_add(landmark_cols)?;
    let storage_cols = residual_offset.checked_add(1)?;
    let storage_len = rows.checked_mul(storage_cols)?;
    Some(LandmarkHouseholderLayout {
        rows,
        landmark_offset,
        residual_offset,
        storage_cols,
        storage_len,
    })
}

#[inline]
fn checked_compact_storage_len(landmark_cols: usize, state_cols: usize) -> Option<usize> {
    let q1_state_len = landmark_cols.checked_mul(state_cols)?;
    let residual_end = q1_state_len.checked_add(landmark_cols)?;
    residual_end.checked_add(landmark_cols.checked_mul(landmark_cols)?)
}

#[inline]
fn checked_compact_storage_capacity(
    factors: &[WhitenedFactorRowStack],
    state_cols: usize,
) -> Option<usize> {
    factors.iter().try_fold(0usize, |capacity, factor| {
        let landmark_cols = factor.landmark_jacobian.ncols();
        if landmark_cols == 0 || landmark_cols > 3 {
            return Some(capacity);
        }
        let entry_len = checked_compact_storage_len(landmark_cols, state_cols)?;
        capacity.checked_add(entry_len)
    })
}

/// Safe f32 implementation of the three-landmark ABS_QR Householder walk.
///
/// Basalt's `LandmarkBlockAbsDynamic` owns a row-major matrix laid out as
/// `[Jp | pad | Jl | r]`.  It allocates one zero damping row per landmark
/// column, then calls `performQRHouseholder` with
/// `remainingRows = num_rows - k - 3` for `k = 0..2`; consequently the final
/// three rows never enter a reflector and remain zero.  This type intentionally
/// keeps the full row-major storage so fixture tests can check Q1, Q2,
/// pivots, and signed-zero behavior at the same boundary.
#[derive(Clone)]
struct LandmarkHouseholderF32 {
    rows: usize,
    state_cols: usize,
    landmark_cols: usize,
    landmark_offset: usize,
    residual_offset: usize,
    storage: Vec<f32>,
    pivots: Vec<f32>,
    tau: Vec<f32>,
}

impl LandmarkHouseholderF32 {
    fn factor(
        state: &DMatrix<f32>,
        landmark: &DMatrix<f32>,
        residual: &DVector<f32>,
    ) -> Option<Self> {
        let observation_rows = state.nrows();
        if landmark.nrows() != observation_rows || residual.len() != observation_rows {
            return None;
        }
        let landmark_cols = landmark.ncols();
        if landmark_cols == 0 || landmark_cols > 3 {
            return None;
        }

        // Compute all derived dimensions before allocating any storage.  The
        // checked helper preserves the upstream non-modulo-4 padding rule
        // while making malformed/adversarial dimensions fail closed.
        let layout =
            checked_landmark_householder_layout(observation_rows, state.ncols(), landmark_cols)?;
        let mut result = Self {
            rows: layout.rows,
            state_cols: state.ncols(),
            landmark_cols,
            landmark_offset: layout.landmark_offset,
            residual_offset: layout.residual_offset,
            storage: vec![0.0; layout.storage_len],
            pivots: vec![0.0; landmark_cols],
            tau: vec![0.0; landmark_cols],
        };

        for row in 0..observation_rows {
            let destination = row * layout.storage_cols;
            for column in 0..state.ncols() {
                result.storage[destination + column] = state[(row, column)];
            }
            for column in 0..landmark_cols {
                result.storage[destination + layout.landmark_offset + column] =
                    landmark[(row, column)];
            }
            result.storage[destination + layout.residual_offset] = residual[row];
        }
        result.perform();
        Some(result)
    }

    /// Build the same row-major ABS_QR storage directly from an active
    /// whitened factor.  The ordinary `factor` constructor above remains the
    /// materialized f32 fixture/oracle path; the clean production reducer uses
    /// this variant so the three temporary f32 state/landmark/residual
    /// matrices are not allocated and copied into storage a second time.
    ///
    /// The conversion, zero padding, offsets, and `perform` call deliberately
    /// mirror `factor` above.  The small norm accumulator is the same
    /// eight-way column dot schedule used by nalgebra's dynamic `Matrix::norm`
    /// and is returned only for the compact-entry eligibility bit.
    fn factor_from_whitened(factor: &WhitenedFactorRowStack) -> Option<(Self, f32)> {
        let observation_rows = factor.state_jacobian.nrows();
        if factor.landmark_jacobian.nrows() != observation_rows
            || factor.residual.len() != observation_rows
        {
            return None;
        }
        let landmark_cols = factor.landmark_jacobian.ncols();
        if landmark_cols == 0 || landmark_cols > 3 {
            return None;
        }

        let state_cols = factor.state_jacobian.ncols();
        // Keep the upstream non-modulo-4 padding rule exactly aligned with
        // `factor`: an already aligned state width still gets four zeros.
        let layout =
            checked_landmark_householder_layout(observation_rows, state_cols, landmark_cols)?;
        let mut result = Self {
            rows: layout.rows,
            state_cols,
            landmark_cols,
            landmark_offset: layout.landmark_offset,
            residual_offset: layout.residual_offset,
            storage: vec![0.0; layout.storage_len],
            pivots: vec![0.0; landmark_cols],
            tau: vec![0.0; landmark_cols],
        };

        // `DMatrix::norm()` reduces each landmark column with eight scalar
        // accumulators, then folds (0+4),(1+5),(2+6),(3+7).  Keep those
        // accumulators on the stack while the source values are converted
        // and packed, avoiding a second f32 matrix and a second conversion.
        let mut norm_accumulators = [[0.0_f32; 8]; 3];
        let mut norm_tail = [0.0_f32; 3];
        let packet_end = observation_rows / 8 * 8;
        for row in 0..observation_rows {
            let destination = row * layout.storage_cols;
            for column in 0..state_cols {
                result.storage[destination + column] = factor.state_jacobian[(row, column)] as f32;
            }
            for column in 0..landmark_cols {
                let value = factor.landmark_jacobian[(row, column)] as f32;
                result.storage[destination + layout.landmark_offset + column] = value;
                let product = value * value;
                if row < packet_end {
                    norm_accumulators[column][row % 8] += product;
                } else {
                    // nalgebra's dynamic dot product reduces the complete
                    // packet prefix first, then adds the scalar remainder.
                    // Keep that boundary explicit: folding the remainder
                    // into an accumulator lane changes the final f32 bits
                    // for rows that are not a multiple of eight.
                    norm_tail[column] += product;
                }
            }
            result.storage[destination + layout.residual_offset] = factor.residual[row] as f32;
        }
        let mut norm_squared = 0.0_f32;
        for (column, accumulators) in norm_accumulators.iter().enumerate().take(landmark_cols) {
            // nalgebra's `norm_squared` first finishes one column's local
            // dot-product accumulator and only then adds that scalar to the
            // matrix-wide result.  Keeping this local variable is important
            // for f32 rounding when there is more than one landmark column.
            let mut column_squared = 0.0_f32;
            column_squared += accumulators[0] + accumulators[4];
            column_squared += accumulators[1] + accumulators[5];
            column_squared += accumulators[2] + accumulators[6];
            column_squared += accumulators[3] + accumulators[7];
            column_squared += norm_tail[column];
            norm_squared += column_squared;
        }

        result.perform();
        Some((result, norm_squared.sqrt()))
    }

    /// Workspace-backed counterpart to [`Self::factor_from_whitened`].  The
    /// packed source conversion and norm schedule are intentionally copied
    /// exactly; only the QR-owned vectors are taken from reusable capacity and
    /// returned by the projection wrapper after Q2/compact extraction.
    fn factor_from_whitened_with_workspace(
        factor: &WhitenedFactorRowStack,
        workspace: &mut LandmarkHouseholderWorkspace,
    ) -> Option<(Self, f32)> {
        let observation_rows = factor.state_jacobian.nrows();
        if factor.landmark_jacobian.nrows() != observation_rows
            || factor.residual.len() != observation_rows
        {
            return None;
        }
        let landmark_cols = factor.landmark_jacobian.ncols();
        if landmark_cols == 0 || landmark_cols > 3 {
            return None;
        }

        let state_cols = factor.state_jacobian.ncols();
        // Resolve the complete layout before taking any reusable workspace
        // vector.  Overflow therefore cannot leave the workspace consumed or
        // turn a malformed factor into a short-buffer indexing panic.
        let layout =
            checked_landmark_householder_layout(observation_rows, state_cols, landmark_cols)?;
        let mut result = Self {
            rows: layout.rows,
            state_cols,
            landmark_cols,
            landmark_offset: layout.landmark_offset,
            residual_offset: layout.residual_offset,
            storage: LandmarkHouseholderWorkspace::take_zeroed(
                &mut workspace.storage,
                layout.storage_len,
            ),
            pivots: LandmarkHouseholderWorkspace::take_zeroed(&mut workspace.pivots, landmark_cols),
            tau: LandmarkHouseholderWorkspace::take_zeroed(&mut workspace.tau, landmark_cols),
        };

        let mut norm_accumulators = [[0.0_f32; 8]; 3];
        let mut norm_tail = [0.0_f32; 3];
        let packet_end = observation_rows / 8 * 8;
        for row in 0..observation_rows {
            let destination = row * layout.storage_cols;
            for column in 0..state_cols {
                result.storage[destination + column] = factor.state_jacobian[(row, column)] as f32;
            }
            for column in 0..landmark_cols {
                let value = factor.landmark_jacobian[(row, column)] as f32;
                result.storage[destination + layout.landmark_offset + column] = value;
                let product = value * value;
                if row < packet_end {
                    norm_accumulators[column][row % 8] += product;
                } else {
                    norm_tail[column] += product;
                }
            }
            result.storage[destination + layout.residual_offset] = factor.residual[row] as f32;
        }
        let mut norm_squared = 0.0_f32;
        for (column, accumulators) in norm_accumulators.iter().enumerate().take(landmark_cols) {
            let mut column_squared = 0.0_f32;
            column_squared += accumulators[0] + accumulators[4];
            column_squared += accumulators[1] + accumulators[5];
            column_squared += accumulators[2] + accumulators[6];
            column_squared += accumulators[3] + accumulators[7];
            column_squared += norm_tail[column];
            norm_squared += column_squared;
        }

        result.perform_with_workspace(workspace);
        Some((result, norm_squared.sqrt()))
    }

    #[inline]
    fn index(&self, row: usize, column: usize) -> usize {
        row * self.storage_cols() + column
    }

    #[inline]
    fn storage_cols(&self) -> usize {
        self.residual_offset + 1
    }

    fn perform(&mut self) {
        let mut essential = Vec::new();
        let mut workspace = Vec::new();
        self.perform_with_scratch(&mut essential, &mut workspace);
    }

    #[inline]
    fn perform_with_workspace(&mut self, workspace: &mut LandmarkHouseholderWorkspace) {
        self.perform_with_scratch(&mut workspace.essential, &mut workspace.gemv);
    }

    fn perform_with_scratch(&mut self, essential: &mut Vec<f32>, workspace: &mut Vec<f32>) {
        let storage_cols = self.storage_cols();
        let damping_rows = self.landmark_cols;
        for k in 0..self.landmark_cols {
            // The production ABS_QR path has three landmark columns and three
            // trailing damping rows.  Keep the same formula for the small
            // synthetic one/two-column rank tests as well.
            let remaining_rows = self.rows.saturating_sub(k + damping_rows);
            if remaining_rows == 0 {
                self.pivots[k] = 0.0;
                self.tau[k] = 0.0;
                continue;
            }
            let landmark_column = self.landmark_offset + k;
            let pivot_index = self.index(k, landmark_column);
            let c0 = self.storage[pivot_index];
            let tail_len = remaining_rows - 1;
            let mut tail_sq_norm = 0.0_f32;
            for offset in 0..tail_len {
                let value = self.storage[self.index(k + 1 + offset, landmark_column)];
                // Eigen's row-major column-stride VectorBlock reduction uses
                // a fused scalar multiply-add for this tail path.  Keep this
                // boundary explicit; the all61 fixture has a one-ULP beta
                // witness at k=1 (track 63).
                tail_sq_norm = value.mul_add(value, tail_sq_norm);
            }

            let (tau, beta) = if tail_sq_norm <= f32::MIN_POSITIVE {
                // Real f32 has zero imaginary component.  Do not normalize or
                // rewrite c0 here: this preserves both +0 and -0 exactly.
                essential.clear();
                essential.resize(tail_len, 0.0_f32);
                (0.0_f32, c0)
            } else {
                // Eigen's makeHouseholder adds c0^2 to the tail norm with a
                // scalar fused multiply-add before sqrt (vfmadd231ss in the
                // pinned clean build).  Keep this separate from the tail
                // reduction so both rounding boundaries are explicit.
                let beta_norm = c0.mul_add(c0, tail_sq_norm).sqrt();
                let beta = if c0 >= 0.0 { -beta_norm } else { beta_norm };
                let denominator = c0 - beta;
                essential.clear();
                essential.extend((0..tail_len).map(|offset| {
                    self.storage[self.index(k + 1 + offset, landmark_column)] / denominator
                }));
                let tau = (beta - c0) / beta;
                (tau, beta)
            };
            self.pivots[k] = beta;
            self.tau[k] = tau;

            if tau == 0.0 {
                continue;
            }

            // applyHouseholderOnTheLeft first materializes all workspace
            // entries from the old bottom and row-0 values.  Eigen
            // transposes this vector-on-the-left product and uses its
            // column-major GEMV path: each packet spans eight output columns,
            // while the essential dimension is accumulated in scalar order.
            landmark_gemv_workspace_into(essential, &self.storage, k + 1, storage_cols, workspace);
            // The row-0 update is a contiguous row-major vector operation.
            let packet_end = storage_cols / 8 * 8;
            let mut column = 0;
            while column < packet_end {
                let row_index = self.index(k, column);
                let updated_workspace = landmark_packet8_add(
                    landmark_packet8_load(workspace, column),
                    landmark_packet8_load(&self.storage, row_index),
                );
                landmark_packet8_store(workspace, column, updated_workspace);
                let updated_row = landmark_packet8_fused_sub(
                    landmark_packet8_load(&self.storage, row_index),
                    updated_workspace,
                    tau,
                );
                landmark_packet8_store(&mut self.storage, row_index, updated_row);
                column += 8;
            }
            let half_packet_end = storage_cols / 4 * 4;
            while column < half_packet_end {
                let row_index = self.index(k, column);
                let updated_workspace = landmark_packet_add(
                    landmark_packet_load(workspace, column),
                    landmark_packet_load(&self.storage, row_index),
                );
                landmark_packet_store(workspace, column, updated_workspace);
                let updated_row = landmark_packet_fused_sub(
                    landmark_packet_load(&self.storage, row_index),
                    updated_workspace,
                    tau,
                );
                landmark_packet_store(&mut self.storage, row_index, updated_row);
                column += 4;
            }
            while column < storage_cols {
                workspace[column] += self.storage[self.index(k, column)];
                let row_index = self.index(k, column);
                self.storage[row_index] =
                    (-tau).mul_add(workspace[column], self.storage[row_index]);
                column += 1;
            }
            // For a row-major destination Eigen's outer-product evaluator
            // visits rows in order and updates each contiguous row with packet
            // mul/sub operations.  Materialize tau*essential per row before
            // touching the destination, preserving the no-alias order.
            for offset in 0..tail_len {
                let row = k + 1 + offset;
                let scale = tau * essential[offset];
                let row_index = self.index(row, 0);
                let mut column = 0;
                while column < packet_end {
                    let destination = row_index + column;
                    let updated = landmark_packet8_fused_sub(
                        landmark_packet8_load(&self.storage, destination),
                        landmark_packet8_load(&workspace, column),
                        scale,
                    );
                    landmark_packet8_store(&mut self.storage, destination, updated);
                    column += 8;
                }
                while column < half_packet_end {
                    let destination = row_index + column;
                    let updated = landmark_packet_fused_sub(
                        landmark_packet_load(&self.storage, destination),
                        landmark_packet_load(&workspace, column),
                        scale,
                    );
                    landmark_packet_store(&mut self.storage, destination, updated);
                    column += 4;
                }
                while column < storage_cols {
                    let index = row_index + column;
                    self.storage[index] = (-scale).mul_add(workspace[column], self.storage[index]);
                    column += 1;
                }
            }
        }
    }

    fn recycle_into(self, workspace: &mut LandmarkHouseholderWorkspace) {
        LandmarkHouseholderWorkspace::recycle(&mut workspace.storage, self.storage);
        LandmarkHouseholderWorkspace::recycle(&mut workspace.pivots, self.pivots);
        LandmarkHouseholderWorkspace::recycle(&mut workspace.tau, self.tau);
    }

    fn transformed_state(&self) -> DMatrix<f32> {
        DMatrix::from_fn(self.rows, self.state_cols, |row, column| {
            self.storage[self.index(row, column)]
        })
    }

    fn transformed_residual(&self) -> DVector<f32> {
        DVector::from_iterator(
            self.rows,
            (0..self.rows).map(|row| self.storage[self.index(row, self.residual_offset)]),
        )
    }

    fn transformed_landmark(&self) -> DMatrix<f32> {
        DMatrix::from_fn(self.rows, self.landmark_cols, |row, column| {
            self.storage[self.index(row, self.landmark_offset + column)]
        })
    }

    fn q2_state(&self) -> DMatrix<f32> {
        let q2_rows = self.rows.saturating_sub(self.landmark_cols);
        DMatrix::from_fn(q2_rows, self.state_cols, |row, column| {
            self.storage[self.index(self.landmark_cols + row, column)]
        })
    }

    fn q2_residual(&self) -> DVector<f32> {
        let q2_rows = self.rows.saturating_sub(self.landmark_cols);
        DVector::from_iterator(
            q2_rows,
            (0..q2_rows).map(|row| {
                self.storage[self.index(self.landmark_cols + row, self.residual_offset)]
            }),
        )
    }

    fn upper_r(&self) -> DMatrix<f32> {
        DMatrix::from_fn(self.landmark_cols, self.landmark_cols, |row, column| {
            if column < row {
                0.0
            } else {
                self.storage[self.index(row, self.landmark_offset + column)]
            }
        })
    }

    /// Copy only the Q1 rows needed by landmark back-substitution.
    ///
    /// This is deliberately an extraction-only operation: the Householder
    /// walk has already happened in [`Self::factor`], and no arithmetic or
    /// row traversal is repeated here.  The lower triangle is materialized
    /// with the same positive zero convention as [`Self::upper_r`].
    fn compact_back_substitution(
        &self,
        landmark_index: usize,
        track_id: u64,
        rank: usize,
        eligible: bool,
    ) -> Option<CompactLandmarkBackSubstitutionF32> {
        let mut storage = Vec::new();
        let descriptor = self.compact_back_substitution_into(
            &mut storage,
            landmark_index,
            track_id,
            rank,
            eligible,
        )?;
        debug_assert_eq!(descriptor.storage_offset, 0);
        Some(CompactLandmarkBackSubstitutionF32 {
            landmark_index,
            track_id,
            storage,
            state_cols: descriptor.state_cols,
            landmark_cols: descriptor.landmark_cols,
            rank,
            eligible,
        })
    }

    /// Extract compact Q1/R values into a caller-owned arena.  The arithmetic
    /// and source traversal are identical to [`Self::compact_back_substitution`];
    /// only ownership changes so a reducer can retain one allocation for all
    /// visual factors in the current LM iteration.
    fn compact_back_substitution_into(
        &self,
        arena: &mut Vec<f32>,
        landmark_index: usize,
        track_id: u64,
        rank: usize,
        eligible: bool,
    ) -> Option<CompactLandmarkBackSubstitutionEntryF32> {
        let rows = self.landmark_cols;
        let q1_state_len = rows.checked_mul(self.state_cols)?;
        let residual_offset = q1_state_len;
        let upper_r_offset = residual_offset.checked_add(rows)?;
        let storage_offset = arena.len();
        let storage_len = checked_compact_storage_len(rows, self.state_cols)?;
        let storage_end = storage_offset.checked_add(storage_len)?;
        arena.resize(storage_end, 0.0_f32);
        // Keep this extraction in the same row/column order as the old
        // DMatrix values.  Only the container changes; no f32 arithmetic is
        // introduced or reordered here.
        for row in 0..rows {
            for column in 0..self.state_cols {
                arena[storage_offset + row * self.state_cols + column] =
                    self.storage[self.index(row, column)];
            }
            arena[storage_offset + residual_offset + row] =
                self.storage[self.index(row, self.residual_offset)];
            for column in row..rows {
                arena[storage_offset + upper_r_offset + row * rows + column] =
                    self.storage[self.index(row, self.landmark_offset + column)];
            }
        }
        Some(CompactLandmarkBackSubstitutionEntryF32 {
            landmark_index,
            track_id,
            storage_offset,
            state_cols: self.state_cols,
            landmark_cols: rows,
            rank,
            eligible,
        })
    }
}

/// Project one landmark block and, optionally, extract its compact Q1/R
/// payload from the very same Householder storage.  The compact caller uses
/// this function so enabling the reducer option cannot introduce the second
/// QR that the old trial recovery performed.
fn landmark_nullspace_projection_f32_with_compact(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
    metadata: Option<LandmarkFactorMetadata>,
) -> (
    DMatrix<f32>,
    DVector<f32>,
    usize,
    Option<CompactLandmarkBackSubstitutionF32>,
) {
    let state = as_f32_matrix(&factor.state_jacobian);
    let landmark = as_f32_matrix(&factor.landmark_jacobian);
    let residual = as_f32_vector(&factor.residual);
    let landmark_columns = landmark.ncols();
    if landmark_columns == 0 || landmark_columns > 3 {
        return (
            if landmark_columns == 0 {
                state
            } else {
                DMatrix::zeros(0, factor.state_jacobian.ncols())
            },
            if landmark_columns == 0 {
                residual
            } else {
                DVector::zeros(0)
            },
            0,
            None,
        );
    }
    let norm_ok = landmark.norm() > tolerance as f32;
    let Some(qr) = LandmarkHouseholderF32::factor(&state, &landmark, &residual) else {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            0,
            None,
        );
    };
    let threshold = tolerance as f32;
    let rank = (0..landmark_columns)
        .filter(|&index| qr.pivots[index].abs() > threshold)
        .count();
    let compact = metadata.and_then(|metadata| {
        qr.compact_back_substitution(
            metadata.landmark_index,
            metadata.track_id,
            rank,
            norm_ok && rank == landmark_columns,
        )
    });
    if qr.rows <= landmark_columns {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            rank,
            compact,
        );
    }
    (qr.q2_state(), qr.q2_residual(), rank, compact)
}

/// Arena-backed counterpart to
/// [`landmark_nullspace_projection_f32_with_compact`].  It deliberately
/// duplicates only the small projection wrapper so the Householder
/// implementation and all projected Q2 arithmetic remain shared; the
/// compact extraction writes directly into the reducer's single arena.
///
/// The QR scratch argument is reused across visual factors.  It owns no Q2
/// output: the transformed storage is recycled only after this function has
/// materialized the returned Q2 matrices and compact arena entry.
fn landmark_nullspace_projection_f32_with_compact_into_with_workspace(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
    metadata: Option<LandmarkFactorMetadata>,
    arena: &mut Vec<f32>,
    qr_workspace: &mut LandmarkHouseholderWorkspace,
) -> (
    DMatrix<f32>,
    DVector<f32>,
    usize,
    Option<CompactLandmarkBackSubstitutionEntryF32>,
) {
    let landmark_columns = factor.landmark_jacobian.ncols();
    if landmark_columns == 0 {
        // Landmark-free factors are not part of the compact visual path; keep
        // the historical materialized f32 projection for their ordinary
        // state/residual contribution.
        return (
            as_f32_matrix(&factor.state_jacobian),
            as_f32_vector(&factor.residual),
            0,
            None,
        );
    }
    if landmark_columns > 3 {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            0,
            None,
        );
    }
    // Valid visual compact factors are packed directly from the f64 source;
    // this is the only production compact projection constructor.  The
    // materialized `factor` constructor remains available above for fixtures
    // and independent parity tests.
    let Some((qr, landmark_norm)) =
        LandmarkHouseholderF32::factor_from_whitened_with_workspace(factor, qr_workspace)
    else {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            0,
            None,
        );
    };
    let norm_ok = landmark_norm > tolerance as f32;
    let threshold = tolerance as f32;
    let rank = (0..landmark_columns)
        .filter(|&index| qr.pivots[index].abs() > threshold)
        .count();
    let compact = metadata.and_then(|metadata| {
        qr.compact_back_substitution_into(
            arena,
            metadata.landmark_index,
            metadata.track_id,
            rank,
            norm_ok && rank == landmark_columns,
        )
    });
    let result = if qr.rows <= landmark_columns {
        (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            rank,
            compact,
        )
    } else {
        (qr.q2_state(), qr.q2_residual(), rank, compact)
    };
    qr.recycle_into(qr_workspace);
    result
}

/// Compatibility wrapper for fixture/legacy callers that do not own a
/// reducer-scoped scratch workspace.  Production clean reduction calls the
/// `_with_workspace` variant above so all visual factors share capacity.
fn landmark_nullspace_projection_f32_with_compact_into(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
    metadata: Option<LandmarkFactorMetadata>,
    arena: &mut Vec<f32>,
) -> (
    DMatrix<f32>,
    DVector<f32>,
    usize,
    Option<CompactLandmarkBackSubstitutionEntryF32>,
) {
    let mut qr_workspace = LandmarkHouseholderWorkspace::default();
    landmark_nullspace_projection_f32_with_compact_into_with_workspace(
        factor,
        tolerance,
        metadata,
        arena,
        &mut qr_workspace,
    )
}

/// Factor one visual block once and retain the compact native-f32 payload
/// required after the state step is known.  This helper is intentionally
/// separate from the reducer until the clean trial-view wiring is proven.
pub(crate) fn compact_landmark_back_substitution_f32(
    factor: &WhitenedFactorRowStack,
    landmark_index: usize,
    track_id: u64,
    tolerance: f64,
) -> Option<CompactLandmarkBackSubstitutionF32> {
    landmark_nullspace_projection_f32_with_compact(
        factor,
        tolerance,
        Some(LandmarkFactorMetadata {
            landmark_index,
            track_id,
        }),
    )
    .3
}

pub(crate) fn landmark_nullspace_projection_f32(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
) -> (DMatrix<f32>, DVector<f32>, usize) {
    let state = as_f32_matrix(&factor.state_jacobian);
    let landmark = as_f32_matrix(&factor.landmark_jacobian);
    let residual = as_f32_vector(&factor.residual);
    let landmark_columns = landmark.ncols();
    if landmark_columns == 0 {
        return (state, residual, 0);
    }
    let threshold = tolerance as f32;
    let Some(qr) = LandmarkHouseholderF32::factor(&state, &landmark, &residual) else {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            0,
        );
    };
    let rank = (0..landmark_columns)
        .filter(|&index| qr.pivots[index].abs() > threshold)
        .count();
    if let Some(metadata) = factor.landmark_metadata {
        emit_landmark_projection_probe(
            metadata,
            factor,
            &state,
            &landmark,
            &residual,
            &qr,
            rank,
            // The retained legacy wrapper does not compute the compact
            // path's separate landmark norm eligibility.  Keep this probe
            // side-effect-free on the production path and report the exact
            // rank-based condition available at this boundary.
            rank == landmark_columns,
        );
    }
    if qr.rows <= landmark_columns {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            rank,
        );
    }
    (qr.q2_state(), qr.q2_residual(), rank)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ImuReductionError {
    MissingBias {
        imu_index: usize,
    },
    UnexpectedBias {
        bias_index: usize,
    },
    InvalidRows {
        index: usize,
        expected: usize,
        actual: usize,
    },
    MissingOffsets {
        index: usize,
        kind: FactorKind,
    },
    MismatchedOffsets {
        imu_index: usize,
        bias_index: usize,
    },
    InvalidOffsets {
        index: usize,
        start: usize,
        end: usize,
        state_dof: usize,
    },
    UnexpectedColumns {
        index: usize,
    },
    StateWidth {
        index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidLocalProduct {
        h_rows: usize,
        h_cols: usize,
        b_len: usize,
    },
    InvalidAccumulator {
        h_rows: usize,
        h_cols: usize,
        b_len: usize,
    },
    InvalidFactorShape {
        index: usize,
        state_rows: usize,
        landmark_rows: usize,
        residual_len: usize,
    },
    InvalidProjectionCount {
        expected: usize,
        actual: usize,
    },
    VisualPrefixTraceIo,
    VisualPrefixTraceInvalid {
        index: usize,
    },
    Full70OracleIo,
    Full70OracleInvalid {
        index: usize,
    },
    Full70OracleProvenanceMissing {
        key: &'static str,
    },
}

fn validate_factor_shapes(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<(), ImuReductionError> {
    for (index, factor) in factors.iter().enumerate() {
        if factor.state_jacobian.ncols() != state_dof {
            return Err(ImuReductionError::StateWidth {
                index,
                expected: state_dof,
                actual: factor.state_jacobian.ncols(),
            });
        }
        if factor.state_jacobian.nrows() != factor.landmark_jacobian.nrows()
            || factor.residual.len() != factor.state_jacobian.nrows()
        {
            return Err(ImuReductionError::InvalidFactorShape {
                index,
                state_rows: factor.state_jacobian.nrows(),
                landmark_rows: factor.landmark_jacobian.nrows(),
                residual_len: factor.residual.len(),
            });
        }
    }
    Ok(())
}

fn validate_imu_pairing(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<(), ImuReductionError> {
    for (index, factor) in factors.iter().enumerate() {
        match factor.kind {
            FactorKind::Imu => {
                if factor.landmark_jacobian.ncols() != 0 {
                    return Err(ImuReductionError::UnexpectedColumns { index });
                }
                if factor.rows() != 9 {
                    return Err(ImuReductionError::InvalidRows {
                        index,
                        expected: 9,
                        actual: factor.rows(),
                    });
                }
                if factor.state_jacobian.ncols() != state_dof {
                    return Err(ImuReductionError::StateWidth {
                        index,
                        expected: state_dof,
                        actual: factor.state_jacobian.ncols(),
                    });
                }
                let Some(offsets) = factor.imu_link_offsets else {
                    return Err(ImuReductionError::MissingOffsets {
                        index,
                        kind: factor.kind,
                    });
                };
                validate_imu_offsets(index, offsets, state_dof)?;
                let Some(bias) = factors.get(index + 1) else {
                    return Err(ImuReductionError::MissingBias { imu_index: index });
                };
                if bias.kind != FactorKind::Bias {
                    return Err(ImuReductionError::MissingBias { imu_index: index });
                }
                if bias.landmark_jacobian.ncols() != 0 {
                    return Err(ImuReductionError::UnexpectedColumns { index: index + 1 });
                }
                if bias.rows() != 6 {
                    return Err(ImuReductionError::InvalidRows {
                        index: index + 1,
                        expected: 6,
                        actual: bias.rows(),
                    });
                }
                if bias.state_jacobian.ncols() != state_dof {
                    return Err(ImuReductionError::StateWidth {
                        index: index + 1,
                        expected: state_dof,
                        actual: bias.state_jacobian.ncols(),
                    });
                }
                if bias.imu_link_offsets != Some(offsets) {
                    return Err(ImuReductionError::MismatchedOffsets {
                        imu_index: index,
                        bias_index: index + 1,
                    });
                }
                validate_pair_columns(index, &factor.state_jacobian, offsets)?;
                validate_pair_columns(index + 1, &bias.state_jacobian, offsets)?;
            }
            FactorKind::Bias => {
                if index == 0 || factors[index - 1].kind != FactorKind::Imu {
                    return Err(ImuReductionError::UnexpectedBias { bias_index: index });
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_imu_offsets(
    index: usize,
    offsets: ImuLinkOffsets,
    state_dof: usize,
) -> Result<(), ImuReductionError> {
    if offsets.start == offsets.end
        || offsets.start.checked_add(AOM_NAV_DOF).is_none()
        || offsets.end.checked_add(AOM_NAV_DOF).is_none()
        || offsets.start + AOM_NAV_DOF > state_dof
        || offsets.end + AOM_NAV_DOF > state_dof
    {
        return Err(ImuReductionError::InvalidOffsets {
            index,
            start: offsets.start,
            end: offsets.end,
            state_dof,
        });
    }
    Ok(())
}

fn validate_pair_columns(
    index: usize,
    jacobian: &DMatrix<f64>,
    offsets: ImuLinkOffsets,
) -> Result<(), ImuReductionError> {
    for column in 0..jacobian.ncols() {
        let in_pair = (offsets.start..offsets.start + AOM_NAV_DOF).contains(&column)
            || (offsets.end..offsets.end + AOM_NAV_DOF).contains(&column);
        if !in_pair && (0..jacobian.nrows()).any(|row| jacobian[(row, column)] != 0.0) {
            return Err(ImuReductionError::UnexpectedColumns { index });
        }
    }
    Ok(())
}

fn reduce_landmark_factors_f32_checked(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    reduce_landmark_factors_f32_checked_with_options(
        factors, state_dof, tolerance, true, false, true,
    )
}

/// Lean LM reduction variant which omits the f64 landmark back-substitution
/// payload.  The active f32 solve computes its model decrease directly from
/// the original row stack and never consumes this payload; retaining it would
/// only recreate diagnostic/retained-path storage.  The default checked
/// reducer above deliberately preserves its historical payload for callers
/// that do need landmark recovery or diagnostic conversion.
fn reduce_landmark_factors_f32_checked_without_back_substitution(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    reduce_landmark_factors_f32_checked_with_options(
        factors, state_dof, tolerance, false, false, true,
    )
}

/// Clean-only reducer option used by the upcoming one-shot landmark recovery
/// wiring.  It retains no legacy f64 payload, but keeps compact Q1/R data
/// alongside the projected Q2 rows produced by one QR walk.
fn reduce_landmark_factors_f32_checked_with_compact_back_substitution(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    reduce_landmark_factors_f32_checked_with_options(
        factors, state_dof, tolerance, false, true, true,
    )
}

/// Standalone landmark recovery re-reduces one already linearized factor.  It
/// is a numeric compatibility path, not a window visual-prefix boundary, so
/// an enabled prefix sidecar must not try to classify it as an independent
/// visual event.
fn reduce_landmark_factors_f32_checked_without_visual_prefix_trace(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    reduce_landmark_factors_f32_checked_with_options(
        factors, state_dof, tolerance, true, false, false,
    )
}

/// Test-only bridge for the active-window integration fixture.  Keeping the
/// reducer and preparation construction in this module exercises the actual
/// producer path without widening the public API or duplicating the compact
/// recovery implementation in a Window test.
#[cfg(test)]
pub(crate) fn compact_trial_preparation_for_test(
    factors: &[WhitenedFactorRowStack],
    state: &DVector<f64>,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<LmTrialPreparation> {
    reduce_landmark_factors_f32_checked_with_compact_back_substitution(
        factors,
        state.len(),
        tolerance,
    )
    .ok()?
    .into_trial_preparation(state, state_step, tolerance)
}

fn reduce_landmark_factors_f32_checked_with_options(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
    retain_back_substitution: bool,
    retain_compact_back_substitution: bool,
    trace_visual_prefix: bool,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    validate_factor_shapes(factors, state_dof)?;
    validate_imu_pairing(factors, state_dof)?;
    let mut visual_prefix_trace = if trace_visual_prefix {
        visual_prefix_trace_writer(factors, state_dof)?
    } else {
        None
    };
    // Keep the source assembly phases separate.  Basalt's absolute QR
    // linearization first reduces all landmark blocks (the visual rows) in
    // its TBB `parallel_reduce`, then adds the IMU/bias blocks through a
    // separate `DenseAccumulator`, and only then folds that accumulator into
    // the visual result.  Treating the landmark-free rows as just another
    // item in one left-to-right loop changes the f32 rounding tree even when
    // every individual factor row is already bit exact.
    let mut visual_h = DMatrix::<f32>::zeros(state_dof, state_dof);
    let mut visual_b = DVector::<f32>::zeros(state_dof);
    let mut imu_h = DMatrix::<f32>::zeros(state_dof, state_dof);
    let mut imu_b = DVector::<f32>::zeros(state_dof);
    let diagnostic_policy = crate::vio::window::diagnostic_env_snapshot();
    let diagnostic_enabled = diagnostic_policy.diagnostic_imu_rows.is_some();
    // The normal-system sidecar is independently gated so a solver frontier
    // capture can request only the four H/b boundaries without materializing
    // the larger per-IMU input records.
    let diagnostic_stages_enabled = diagnostic_enabled
        || diagnostic_policy.diagnostic_normal_stages
        || diagnostic_policy.solver_frontier_trace.is_some()
        || diagnostic_policy.full70_factor_oracle.is_some();
    let mut local_blocks = Vec::new();
    let mut imu_cumulative_stages = Vec::new();
    let mut prior_h = DMatrix::<f32>::zeros(state_dof, state_dof);
    let mut prior_b = DVector::<f32>::zeros(state_dof);
    let first_visual = factors
        .iter()
        .position(|factor| factor.landmark_jacobian.ncols() != 0);
    let last_visual = factors
        .iter()
        .rposition(|factor| factor.landmark_jacobian.ncols() != 0);
    // Project all factors once.  Keeping these row stacks materialized lets
    // the IMU9 and bias6 entries be joined into the single 15-row `J/r` that
    // upstream `ImuBlock::add_dense_H_b` multiplies, without changing the
    // public factor list or its diagnostic row topology.  The compact option
    // takes the same path, but extracts Q1/R from the already-factored
    // Householder storage instead of factoring a visual block again.
    let (projected, compact_back_substitution) = if retain_compact_back_substitution {
        let compact_capacity = factors
            .iter()
            .filter(|factor| {
                let landmark_cols = factor.landmark_jacobian.ncols();
                landmark_cols != 0 && landmark_cols <= 3
            })
            .count();
        // Capacity arithmetic must fail closed before either arena or
        // reusable QR storage is acquired.  A saturating sum can turn an
        // impossible layout into a wrapped/oversized allocation request.
        let compact_storage_capacity = checked_compact_storage_capacity(factors, state_dof);
        let compact_capacity_valid = compact_storage_capacity.is_some();
        let mut compact_entries = if compact_capacity_valid {
            Vec::with_capacity(compact_capacity)
        } else {
            Vec::new()
        };
        let mut compact_storage = match compact_storage_capacity {
            Some(capacity) => Vec::with_capacity(capacity),
            None => Vec::new(),
        };
        #[cfg(feature = "basalt-lm-workspace-reuse")]
        let mut landmark_workspace = LandmarkHouseholderWorkspace::default();
        let mut compact_mapping_valid = compact_capacity_valid;
        let mut seen_landmark_indices = if compact_capacity_valid {
            Vec::with_capacity(compact_capacity)
        } else {
            Vec::new()
        };
        let mut projected = Vec::with_capacity(factors.len());
        for factor in factors {
            if factor.landmark_jacobian.ncols() == 0 {
                projected.push(landmark_nullspace_projection_f32(factor, tolerance));
                continue;
            }
            let metadata = (factor.kind == FactorKind::Visual)
                .then_some(factor.landmark_metadata)
                .flatten();
            if factor.kind != FactorKind::Visual || metadata.is_none() {
                // The factor still contributes its ordinary Q2 rows, but a
                // missing/non-visual identity makes the compact mapping
                // unsafe.  The caller can observe `None` and use legacy
                // recovery without guessing from factor position.
                compact_mapping_valid = false;
            }
            let use_direct_visual_pack = factor.kind == FactorKind::Visual
                && compact_capacity_valid
                && metadata.is_some()
                && factor.landmark_jacobian.ncols() <= 3;
            #[cfg(feature = "basalt-lm-workspace-reuse")]
            let (jacobian, residual, rank, compact) = if use_direct_visual_pack {
                landmark_nullspace_projection_f32_with_compact_into_with_workspace(
                    factor,
                    tolerance,
                    metadata,
                    &mut compact_storage,
                    &mut landmark_workspace,
                )
            } else {
                // Keep generic/non-visual and invalid visual mappings on the
                // historical materialized constructor.  They can still
                // contribute their ordinary Q2 rows, but they must not enter
                // the direct visual compact allocation path.
                let (jacobian, residual, rank, _) =
                    landmark_nullspace_projection_f32_with_compact(factor, tolerance, None);
                (jacobian, residual, rank, None)
            };
            #[cfg(not(feature = "basalt-lm-workspace-reuse"))]
            let (jacobian, residual, rank, compact) = if use_direct_visual_pack {
                // The non-reuse compatibility wrapper allocates a fresh
                // scratch workspace for this factor.  It is the historical
                // ownership path and intentionally shares the exact packed
                // conversion, Householder walk, and Q2/compact extraction.
                landmark_nullspace_projection_f32_with_compact_into(
                    factor,
                    tolerance,
                    metadata,
                    &mut compact_storage,
                )
            } else {
                // Keep generic/non-visual and invalid visual mappings on the
                // historical materialized constructor.  They can still
                // contribute their ordinary Q2 rows, but they must not enter
                // the direct visual compact allocation path.
                let (jacobian, residual, rank, _) =
                    landmark_nullspace_projection_f32_with_compact(factor, tolerance, None);
                (jacobian, residual, rank, None)
            };
            if let Some(compact) = compact {
                if seen_landmark_indices.contains(&compact.landmark_index) {
                    compact_mapping_valid = false;
                }
                seen_landmark_indices.push(compact.landmark_index);
                compact_entries.push(compact);
            } else {
                compact_mapping_valid = false;
            }
            projected.push((jacobian, residual, rank));
        }
        let compact = compact_mapping_valid.then_some(CompactLandmarkBackSubstitutionBatchF32 {
            storage: compact_storage,
            entries: compact_entries,
        });
        (projected, compact)
    } else {
        (
            factors
                .iter()
                .map(|factor| landmark_nullspace_projection_f32(factor, tolerance))
                .collect::<Vec<_>>(),
            None,
        )
    };
    let mut back = retain_back_substitution.then(|| Vec::with_capacity(factors.len()));
    if let Some(back) = back.as_mut() {
        for (factor, (_, _, rank)) in factors.iter().zip(&projected) {
            back.push(LandmarkBackSubstitution {
                state_jacobian: factor.state_jacobian.clone(),
                landmark_jacobian: factor.landmark_jacobian.clone(),
                residual: factor.residual.clone(),
                rank: *rank,
            });
        }
    }

    let mut factor_index = 0;
    while factor_index < factors.len() {
        let factor = &factors[factor_index];
        let (jacobian, residual, _) = &projected[factor_index];
        if factor.landmark_jacobian.ncols() != 0 {
            // Standalone landmark back-substitution callers use Generic
            // factors.  They still follow the historical visual H/b path,
            // but are not visual-prefix records; tracing them as Visual would
            // make an enabled diagnostic sidecar fail at ordinal zero before
            // the actual window visual factors are visited.
            let prefix = if factor.kind == FactorKind::Visual {
                visual_prefix_trace
                    .as_mut()
                    .map(|writer| {
                        writer.begin_visual_prefix(
                            factor_index,
                            factor,
                            jacobian,
                            residual,
                            projected[factor_index].2,
                            &visual_h,
                            &visual_b,
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            if factor.kind == FactorKind::Visual {
                visual_h += eigen_visual_gram_packet_tail_f32(jacobian);
            } else {
                visual_h += jacobian.transpose() * jacobian;
            }
            accumulate_transpose_vector_f32_eigen(&mut visual_b, jacobian, residual, false);
            if let (Some(writer), Some(prefix)) = (visual_prefix_trace.as_mut(), prefix) {
                writer.finish_visual_prefix(prefix, &visual_h, &visual_b)?;
            }
            factor_index += 1;
            continue;
        }

        // Semantic tags are authoritative.  The positional fallback exists
        // only for legacy Generic factors supplied by external callers; it
        // keeps their historical prefix/suffix behavior while ensuring that a
        // Generic nine-row prior never receives the IMU packet schedule.
        let is_imu_phase = match factor.kind {
            FactorKind::Imu | FactorKind::Bias => true,
            FactorKind::Prior => false,
            FactorKind::Visual => false,
            FactorKind::Generic => last_visual.is_some_and(|end| factor_index > end),
        };
        let (phase_h, phase_b) = if is_imu_phase {
            (&mut imu_h, &mut imu_b)
        } else if factor.kind == FactorKind::Prior
            || first_visual.is_some_and(|start| factor_index < start)
        {
            (&mut prior_h, &mut prior_b)
        } else {
            // A malformed/interleaved untagged factor is safer as a prior than
            // as an accidental IMU contribution.
            (&mut prior_h, &mut prior_b)
        };

        // A stored square-root marginal prior is compact in upstream
        // `MargLinData` (one column per retained state block).  Its normal
        // product is evaluated on that compact matrix and only then embedded
        // in the active AOM.  Do not feed the zero-padded global-width view
        // through nalgebra's generic GEMM: the padding changes Eigen's
        // packet traversal and therefore the f32 reduction tree.  The
        // optional metadata is absent for synthetic/legacy Prior factors, so
        // those callers retain the historical dynamic path below.
        if factor.kind == FactorKind::Prior {
            emit_prior_factor_input_diagnostic(
                jacobian,
                residual,
                factor.prior_state_columns.as_deref(),
            );
            accumulate_prior_gram_f32_eigen(
                phase_h,
                jacobian,
                factor.prior_state_columns.as_deref(),
            );
            accumulate_prior_transpose_vector_f32_eigen(
                phase_b,
                jacobian,
                residual,
                factor.prior_state_columns.as_deref(),
            );
            factor_index += 1;
            continue;
        }

        let exact_imu_rows = factor.kind == FactorKind::Imu
            && factor.landmark_jacobian.ncols() == 0
            && jacobian.nrows() == 9;
        if exact_imu_rows {
            // Upstream ImuBlock owns one local 15-row/30-column stack:
            // [preintegration 9 | gyro-bias 3 | accel-bias 3].  Build that
            // stack explicitly from the absolute columns carried by the
            // pair.  Reducing a zero-padded global matrix changes Eigen's
            // packet tails and is not source-equivalent.
            let Some(offsets) = factor.imu_link_offsets else {
                return Err(ImuReductionError::MissingOffsets {
                    index: factor_index,
                    kind: factor.kind,
                });
            };
            let Some((bias_jacobian, bias_residual, _)) = projected.get(factor_index + 1) else {
                return Err(ImuReductionError::MissingBias {
                    imu_index: factor_index,
                });
            };
            let mut local_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, IMU_LOCAL_COLS);
            for (local_row, source) in jacobian.row_iter().enumerate() {
                for block in 0..2 {
                    let global_offset = if block == 0 {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    local_jacobian
                        .view_mut((local_row, block * AOM_NAV_DOF), (1, AOM_NAV_DOF))
                        .copy_from(&source.columns(global_offset, AOM_NAV_DOF));
                }
            }
            for (local_row, source) in bias_jacobian.row_iter().enumerate() {
                for block in 0..2 {
                    let global_offset = if block == 0 {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    local_jacobian
                        .view_mut((9 + local_row, block * AOM_NAV_DOF), (1, AOM_NAV_DOF))
                        .copy_from(&source.columns(global_offset, AOM_NAV_DOF));
                }
            }
            let mut local_residual = DVector::<f32>::zeros(IMU_LOCAL_ROWS);
            local_residual.rows_mut(0, 9).copy_from(residual);
            local_residual.rows_mut(9, 6).copy_from(bias_residual);
            let (local_h, local_b) = local_imu_h_b_15x30(&local_jacobian, &local_residual);
            if diagnostic_enabled {
                // The local packet is deliberately only 30 columns wide, but
                // the comparison side must retain the source AOM width.  In
                // particular, frame-0 links can start after a pose-only
                // prefix, so indexing the local product with absolute AOM
                // offsets would be invalid.  Materialize the corresponding
                // 15-row global source stack for the diagnostic only.
                let mut global_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, state_dof);
                global_jacobian.rows_mut(0, 9).copy_from(jacobian);
                global_jacobian.rows_mut(9, 6).copy_from(bias_jacobian);
                local_blocks.push(imu_local_block_diagnostic(
                    &local_jacobian,
                    &local_residual,
                    &global_jacobian,
                    offsets,
                    factor.imu_input_diagnostic.clone(),
                ));
            }
            scatter_local_imu_h_b_15x30(&mut imu_h, &mut imu_b, &local_h, &local_b, offsets)?;
            if diagnostic_enabled {
                imu_cumulative_stages.push(ImuCumulativeStageDiagnostic {
                    active_offsets: vec![offsets.start, offsets.end],
                    imu_input_diagnostic: factor.imu_input_diagnostic.clone(),
                    imu_h: imu_h.clone(),
                    imu_b: imu_b.clone(),
                });
            }
            factor_index += 2;
        } else {
            accumulate_gram_f32_eigen(phase_h, jacobian, exact_imu_rows);
            accumulate_transpose_vector_f32_eigen(phase_b, jacobian, residual, exact_imu_rows);
            factor_index += 1;
        }
    }
    // These whole-matrix adds mirror LinearizationAbsQR::get_dense_H_b:
    // visual TBB result, then IMU DenseAccumulator, then marginal prior.  The
    // stage copies below are diagnostic-only and preserve the source order
    // needed to compare the later damping/prior boundary without changing
    // this production reduction.
    let diagnostic_imu_h = diagnostic_enabled.then(|| imu_h.clone());
    let diagnostic_imu_b = diagnostic_enabled.then(|| imu_b.clone());
    let diagnostic_stages = if diagnostic_stages_enabled {
        let mut visual_imu_h = visual_h.clone();
        visual_imu_h += &imu_h;
        let mut visual_imu_b = visual_b.clone();
        visual_imu_b += &imu_b;
        Some(DiagnosticNormalSystemF32 {
            visual_h: visual_h.clone(),
            visual_b: visual_b.clone(),
            visual_imu_h,
            visual_imu_b,
            prior_h: prior_h.clone(),
            prior_b: prior_b.clone(),
        })
    } else {
        None
    };
    if let Some(writer) = visual_prefix_trace.as_mut() {
        if writer.next_visual_ordinal != writer.expected_visual_count {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: writer.next_visual_ordinal,
            });
        }
        writer.write_stage("visual_total", &visual_h, &visual_b)?;
    }
    let mut h = visual_h;
    h += imu_h;
    let mut b = visual_b;
    b += imu_b;
    if let Some(writer) = visual_prefix_trace.as_mut() {
        writer.write_stage("imu_total", &h, &b)?;
        writer.write_prior_before(&h, &b, &prior_h, &prior_b)?;
    }
    h += prior_h;
    b += prior_b;
    if let Some(writer) = visual_prefix_trace.as_mut() {
        writer.write_stage("prior_after", &h, &b)?;
        writer.write_stage("final", &h, &b)?;
    }
    // The historical reducer retained compact recovery independently of the
    // model evaluator.  The later Q2 model payload transfer and all-or-nothing
    // invalidation are intentionally absent from this rollback variant.
    let (compact_back_substitution, model_decrease_payload) = if retain_compact_back_substitution {
        drop(projected);
        (compact_back_substitution, None)
    } else {
        drop(projected);
        (None, None)
    };
    Ok(ReducedNormalSystemF32 {
        h,
        b,
        back_substitution: back.unwrap_or_default(),
        compact_back_substitution,
        model_decrease_payload,
        imu_diagnostic: diagnostic_enabled.then(|| ImuReductionDiagnostic {
            local_blocks,
            imu_cumulative_stages,
            imu_h: diagnostic_imu_h.expect("diagnostic IMU H snapshot"),
            imu_b: diagnostic_imu_b.expect("diagnostic IMU b snapshot"),
        }),
        diagnostic_stages,
    })
}

/// Infallible compatibility entry point for the historical f32 reducer.
/// This rollback variant keeps the old reachable wrapper; malformed factors
/// retain the historical panic behavior.
fn reduce_landmark_factors_f32(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> ReducedNormalSystemF32 {
    reduce_landmark_factors_f32_checked(factors, state_dof, tolerance)
        .unwrap_or_else(|error| panic!("invalid f32 IMU reduction factors: {error:?}"))
}

/// Form the absolute normal system from the already reduced Q2 rows.
///
/// This is the boundary used by Basalt's `LinearizationAbsQR::get_dense_H_b`:
/// each visual landmark has already undergone its ABS-QR/null-space step and
/// the returned rows are stacked in native Q2 order.  The input is widened
/// only by the public Rust representation; the product itself is performed
/// in f32 with the pinned Eigen AVX2 GEMM/GEMV traversal (`mr=24`, `nr=4`,
/// packet width eight).  Keeping this separate from the factor reducer is
/// important: rebuilding H/b from the original factor list observes a
/// different arithmetic boundary and is not the source MargData contract.
pub(crate) fn q2_f32_normal_system(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>) {
    assert_eq!(jacobian.nrows(), rhs.len());
    let jacobian_f32 = as_f32_matrix(jacobian);
    let rhs_f32 = as_f32_vector(rhs);
    let h = eigen_q2_gram_f32(&jacobian_f32);
    let b = eigen_q2_transpose_gemv_f32(&jacobian_f32, &rhs_f32);
    emit_q2_abs_hb_diagnostic(&jacobian_f32, &rhs_f32, &h, &b);
    (
        DMatrix::from_fn(h.nrows(), h.ncols(), |row, column| {
            f64::from(h[(row, column)])
        }),
        DVector::from_iterator(b.len(), b.iter().copied().map(f64::from)),
    )
}

/// Optional post-QR Q2/ABS sidecar.  This is intentionally attached to the
/// Q2 helper rather than the factor reducer so a capture cannot accidentally
/// report the wrong pre-QR boundary.  It is disabled unless the caller sets
/// `VISLOC_BASALT_Q2_ABS_HB` and never participates in estimator arithmetic.
#[inline]
fn emit_q2_abs_hb_diagnostic(
    jacobian: &DMatrix<f32>,
    rhs: &DVector<f32>,
    h: &DMatrix<f32>,
    b: &DVector<f32>,
) {
    let Some(path) = crate::vio::window::diagnostic_env_snapshot()
        .q2_abs_hb
        .as_ref()
    else {
        return;
    };
    static EVENT_COUNT: AtomicUsize = AtomicUsize::new(0);
    let event_ordinal = EVENT_COUNT.fetch_add(1, Ordering::Relaxed);
    let payload = json!({
        "schema": "visloc.m7im15.rust_q2_abs_hb.v1",
        "event_ordinal": event_ordinal,
        "q2_jacobian_f32": diagnostic_f32_matrix(jacobian),
        "q2_rhs_f32": diagnostic_f32_vector(rhs),
        "aom_abs_h_f32": diagnostic_f32_matrix(h),
        "aom_abs_b_f32": b
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
    });
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = serde_json::to_writer(&mut file, &payload);
        let _ = file.write_all(b"\n");
    }
}

/// Build the local 15-row/30-column view owned by one semantic IMU+bias pair.
/// The production reducer still consumes the global state-width matrix; this
/// sidecar records the corresponding local dynamic product so padding/state
/// width effects can be compared without a detail trace.
fn imu_local_block_diagnostic(
    local_jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
    global_jacobian: &DMatrix<f32>,
    offsets: ImuLinkOffsets,
    imu_input_diagnostic: Option<serde_json::Value>,
) -> ImuLocalBlockDiagnostic {
    let active_offsets = vec![offsets.start, offsets.end];
    let (local_h, local_b) = local_imu_h_b_15x30(local_jacobian, residual);
    let global_h = global_jacobian.transpose() * global_jacobian;
    let global_b = global_jacobian.transpose() * residual;
    let mut padded_h = DMatrix::zeros(global_jacobian.ncols(), global_jacobian.ncols());
    let mut padded_b = DVector::zeros(global_jacobian.ncols());
    for (local_row_block, &global_row_offset) in active_offsets.iter().enumerate() {
        for (local_column_block, &global_column_offset) in active_offsets.iter().enumerate() {
            padded_h
                .view_mut(
                    (global_row_offset, global_column_offset),
                    (AOM_NAV_DOF, AOM_NAV_DOF),
                )
                .copy_from(&local_h.view(
                    (
                        local_row_block * AOM_NAV_DOF,
                        local_column_block * AOM_NAV_DOF,
                    ),
                    (AOM_NAV_DOF, AOM_NAV_DOF),
                ));
        }
    }
    for (block, &offset) in active_offsets.iter().enumerate() {
        padded_b
            .rows_mut(offset, AOM_NAV_DOF)
            .copy_from(&local_b.rows(block * AOM_NAV_DOF, AOM_NAV_DOF));
    }
    let mut h_mismatches = 0;
    for row in 0..local_h.nrows() {
        for column in 0..local_h.ncols() {
            let global_row = active_offsets[row / AOM_NAV_DOF] + row % AOM_NAV_DOF;
            let global_column = active_offsets[column / AOM_NAV_DOF] + column % AOM_NAV_DOF;
            if local_h[(row, column)].to_bits() != global_h[(global_row, global_column)].to_bits() {
                h_mismatches += 1;
            }
        }
    }
    let mut b_mismatches = 0;
    for column in 0..local_b.len() {
        let global_column = active_offsets[column / AOM_NAV_DOF] + column % AOM_NAV_DOF;
        if local_b[column].to_bits() != global_b[global_column].to_bits() {
            b_mismatches += 1;
        }
    }
    let padded_h_mismatches = (0..global_h.nrows())
        .flat_map(|row| (0..global_h.ncols()).map(move |column| (row, column)))
        .filter(|&(row, column)| {
            padded_h[(row, column)].to_bits() != global_h[(row, column)].to_bits()
        })
        .count();
    let padded_b_mismatches = (0..global_b.len())
        .filter(|&column| padded_b[column].to_bits() != global_b[column].to_bits())
        .count();
    ImuLocalBlockDiagnostic {
        active_offsets,
        local_jacobian: local_jacobian.clone(),
        residual: residual.clone(),
        local_h,
        local_b,
        imu_input_diagnostic,
        local_vs_global_h_mismatches: h_mismatches,
        local_vs_global_b_mismatches: b_mismatches,
        local_vs_global_padded_h_mismatches: padded_h_mismatches,
        local_vs_global_padded_b_mismatches: padded_b_mismatches,
    }
}

fn diagnostic_f32_matrix(value: &DMatrix<f32>) -> serde_json::Value {
    json!({
        "rows": value.nrows(),
        "cols": value.ncols(),
        "layout": "column_major",
        "bits": value.iter().map(|entry| format!("{:08x}", entry.to_bits())).collect::<Vec<_>>(),
    })
}

fn diagnostic_f32_vector(value: &DVector<f32>) -> serde_json::Value {
    json!({
        "rows": value.len(),
        "cols": 1,
        "layout": "column_major_vector",
        "bits": value.iter().map(|entry| format!("{:08x}", entry.to_bits())).collect::<Vec<_>>(),
    })
}

/// Serialize an f64-owned matrix/vector at the exact f32 boundary used by the
/// UpstreamF32 reducer.  These helpers are crate-visible only so the active
/// window can put the stored prior/FEJ source beside the ordered factor oracle;
/// they are never called from the nominal solver path.
pub(crate) fn full70_f32_bits_matrix(value: &DMatrix<f64>) -> serde_json::Value {
    json!({
        "rows": value.nrows(),
        "cols": value.ncols(),
        "layout": "column_major",
        "bits": value
            .iter()
            .map(|entry| format!("{:08x}", (*entry as f32).to_bits()))
            .collect::<Vec<_>>(),
    })
}

pub(crate) fn full70_f32_bits_vector(value: &DVector<f64>) -> serde_json::Value {
    json!({
        "rows": value.len(),
        "cols": 1,
        "layout": "column_major_vector",
        "bits": value
            .iter()
            .map(|entry| format!("{:08x}", (*entry as f32).to_bits()))
            .collect::<Vec<_>>(),
    })
}

const FULL70_ORACLE_REQUIRED_PROVENANCE: &[&str] = &[
    "VISLOC_BASALT_FULL70_EXECUTABLE_SHA256",
    "VISLOC_BASALT_FULL70_SOURCE_SHA256",
    "VISLOC_BASALT_FULL70_CONFIG_SHA256",
    "VISLOC_BASALT_FULL70_CALIBRATION_SHA256",
    "VISLOC_BASALT_FULL70_INPUT_SHA256",
];

fn full70_hash_like(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn full70_oracle_provenance_from_env() -> Result<serde_json::Value, ImuReductionError> {
    let mut required = serde_json::Map::new();
    for &key in FULL70_ORACLE_REQUIRED_PROVENANCE {
        let value = std::env::var(key)
            .ok()
            .filter(|value| full70_hash_like(value))
            .ok_or(ImuReductionError::Full70OracleProvenanceMissing { key })?;
        required.insert(
            key.to_owned(),
            serde_json::Value::String(value.to_ascii_lowercase()),
        );
    }
    let optional = [
        "VISLOC_BASALT_FULL70_FEATURES",
        "VISLOC_BASALT_FULL70_TARGET",
        "VISLOC_BASALT_FULL70_DIRTY_SCOPE_SHA256",
    ]
    .into_iter()
    .filter_map(|key| {
        std::env::var(key)
            .ok()
            .map(|value| (key.to_owned(), json!(value)))
    })
    .collect::<serde_json::Map<_, _>>();
    Ok(json!({
        "status": "declared_external_recompute_required",
        "required_hashes": required,
        "optional": optional,
        "verification": "the harness must recompute every hash from the declared path/source and reject declaration-only evidence",
    }))
}

fn full70_checked_f32_bits(value: f64, index: usize) -> Result<String, ImuReductionError> {
    let value = value as f32;
    value
        .is_finite()
        .then(|| format!("{:08x}", value.to_bits()))
        .ok_or(ImuReductionError::Full70OracleInvalid { index })
}

fn full70_checked_matrix_bits(
    value: &DMatrix<f64>,
    index: usize,
) -> Result<serde_json::Value, ImuReductionError> {
    let bits = value
        .iter()
        .map(|entry| full70_checked_f32_bits(*entry, index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "rows": value.nrows(),
        "cols": value.ncols(),
        "layout": "column_major",
        "bits": bits,
    }))
}

fn full70_checked_vector_bits(
    value: &DVector<f64>,
    index: usize,
) -> Result<serde_json::Value, ImuReductionError> {
    let bits = value
        .iter()
        .map(|entry| full70_checked_f32_bits(*entry, index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "rows": value.len(),
        "cols": 1,
        "layout": "column_major_vector",
        "bits": bits,
    }))
}

fn full70_f32_storage_bits(values: &[f32]) -> serde_json::Value {
    json!({
        "rows": values.len(),
        "cols": 1,
        "layout": "flat_storage",
        "bits": values
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
    })
}

fn full70_factor_kind_name(kind: FactorKind) -> &'static str {
    match kind {
        FactorKind::Generic => "Generic",
        FactorKind::Prior => "Prior",
        FactorKind::Visual => "Visual",
        FactorKind::Imu => "Imu",
        FactorKind::Bias => "Bias",
    }
}

/// The full70 capture is intentionally strict: accepting a count-correct but
/// interleaved or metadata-incomplete vector would make the resulting artifact
/// impossible to compare to the native source order.  The checked reducer has
/// more general validation; this additional frame-4 contract is the oracle
/// boundary itself.
fn validate_full70_frame4_factors(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<(), ImuReductionError> {
    validate_factor_shapes(factors, state_dof)?;
    validate_imu_pairing(factors, state_dof)?;
    if state_dof != 75 || factors.len() != 70 {
        return Err(ImuReductionError::Full70OracleInvalid {
            index: factors.len(),
        });
    }
    let Some(prior) = factors.first() else {
        return Err(ImuReductionError::Full70OracleInvalid { index: 0 });
    };
    if prior.kind != FactorKind::Prior || prior.rows() != 15 || prior.landmark_jacobian.ncols() != 0
    {
        return Err(ImuReductionError::Full70OracleInvalid { index: 0 });
    }
    let mut visual_end = 1usize;
    let mut previous_landmark_index = None;
    while visual_end < factors.len() && factors[visual_end].kind == FactorKind::Visual {
        let factor = &factors[visual_end];
        let Some(metadata) = factor.landmark_metadata else {
            return Err(ImuReductionError::Full70OracleInvalid { index: visual_end });
        };
        if factor.landmark_jacobian.ncols() != 3
            || previous_landmark_index.is_some_and(|previous| metadata.landmark_index <= previous)
        {
            return Err(ImuReductionError::Full70OracleInvalid { index: visual_end });
        }
        previous_landmark_index = Some(metadata.landmark_index);
        visual_end += 1;
    }
    if visual_end - 1 != 61 {
        return Err(ImuReductionError::Full70OracleInvalid { index: visual_end });
    }
    let suffix = &factors[visual_end..];
    if suffix.len() != 8
        || suffix
            .chunks_exact(2)
            .any(|pair| pair[0].kind != FactorKind::Imu || pair[1].kind != FactorKind::Bias)
    {
        return Err(ImuReductionError::Full70OracleInvalid { index: visual_end });
    }
    let row_count = factors.iter().try_fold(0usize, |sum, factor| {
        sum.checked_add(factor.rows())
            .ok_or(ImuReductionError::Full70OracleInvalid { index: sum })
    })?;
    if row_count != 1243 {
        return Err(ImuReductionError::Full70OracleInvalid { index: row_count });
    }
    Ok(())
}

fn full70_imu_local_payload(
    imu: &WhitenedFactorRowStack,
    bias: &WhitenedFactorRowStack,
    index: usize,
    state_dof: usize,
) -> Result<serde_json::Value, ImuReductionError> {
    let offsets = imu
        .imu_link_offsets
        .ok_or(ImuReductionError::Full70OracleInvalid { index })?;
    validate_imu_offsets(index, offsets, state_dof)?;
    let mut local_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, IMU_LOCAL_COLS);
    for (local_row, source) in imu.state_jacobian.row_iter().enumerate() {
        for block in 0..2 {
            let global_offset = if block == 0 {
                offsets.start
            } else {
                offsets.end
            };
            for local_column in 0..AOM_NAV_DOF {
                let value = source[global_offset + local_column] as f32;
                if !value.is_finite() {
                    return Err(ImuReductionError::Full70OracleInvalid { index });
                }
                local_jacobian[(local_row, block * AOM_NAV_DOF + local_column)] = value;
            }
        }
    }
    for (local_row, source) in bias.state_jacobian.row_iter().enumerate() {
        for block in 0..2 {
            let global_offset = if block == 0 {
                offsets.start
            } else {
                offsets.end
            };
            for local_column in 0..AOM_NAV_DOF {
                let value = source[global_offset + local_column] as f32;
                if !value.is_finite() {
                    return Err(ImuReductionError::Full70OracleInvalid { index });
                }
                local_jacobian[(9 + local_row, block * AOM_NAV_DOF + local_column)] = value;
            }
        }
    }
    let local_residual = DVector::from_iterator(
        IMU_LOCAL_ROWS,
        imu.residual
            .iter()
            .chain(bias.residual.iter())
            .map(|value| *value as f32),
    );
    if local_residual.iter().any(|value| !value.is_finite()) {
        return Err(ImuReductionError::Full70OracleInvalid { index });
    }
    Ok(json!({
        "offsets": { "start": offsets.start, "end": offsets.end },
        "shape": [IMU_LOCAL_ROWS, IMU_LOCAL_COLS],
        "jacobian_f32": diagnostic_f32_matrix(&local_jacobian),
        "residual_f32": diagnostic_f32_vector(&local_residual),
    }))
}

fn full70_factor_record(
    factor: &WhitenedFactorRowStack,
    ordinal: usize,
    row_start: usize,
    paired_imu: Option<serde_json::Value>,
) -> Result<serde_json::Value, ImuReductionError> {
    let state_jacobian_f32 = full70_checked_matrix_bits(&factor.state_jacobian, ordinal)?;
    let landmark_jacobian_f32 = full70_checked_matrix_bits(&factor.landmark_jacobian, ordinal)?;
    let residual_f32 = full70_checked_vector_bits(&factor.residual, ordinal)?;
    let objective_cost_f32_bits = full70_checked_f32_bits(factor.objective_cost, ordinal)?;
    let visual_metadata = factor.landmark_metadata.map(|metadata| {
        json!({
            "landmark_index": metadata.landmark_index,
            "track_id": metadata.track_id,
        })
    });
    let observation_ids = factor.visual_observation_ids.as_ref().map(|observations| {
        observations
            .iter()
            .map(|&(state_index, camera_id)| json!({ "state_index": state_index, "camera_id": camera_id }))
            .collect::<Vec<_>>()
    });
    let paired_bias_ordinal = (factor.kind == FactorKind::Imu).then_some(ordinal + 1);
    let paired_imu_ordinal = (factor.kind == FactorKind::Bias).then_some(ordinal.saturating_sub(1));
    Ok(json!({
        "ordinal": ordinal,
        "kind": full70_factor_kind_name(factor.kind),
        "row_start": row_start,
        "rows": factor.rows(),
        "state_cols": factor.state_jacobian.ncols(),
        "landmark_cols": factor.landmark_jacobian.ncols(),
        "objective_cost_f32_bits": objective_cost_f32_bits,
        "state_jacobian_f32": state_jacobian_f32,
        "landmark_jacobian_f32": landmark_jacobian_f32,
        "residual_f32": residual_f32,
        "source_layout": "column_major_dmatrix_cast_f32",
        "landmark_metadata": visual_metadata.unwrap_or(serde_json::Value::Null),
        "visual_observation_ids": observation_ids.unwrap_or_default(),
        "imu_link_offsets": factor.imu_link_offsets.map(|offsets| json!({
            "start": offsets.start,
            "end": offsets.end,
        })).unwrap_or(serde_json::Value::Null),
        "paired_imu_ordinal": paired_imu_ordinal,
        "paired_bias_ordinal": paired_bias_ordinal,
        "prior_state_columns": factor.prior_state_columns.clone(),
        "imu_input_diagnostic": factor.imu_input_diagnostic.clone(),
        "imu_local_15x30": paired_imu,
    }))
}

fn full70_stage_json(stage: Option<&DiagnosticNormalSystemF32>) -> serde_json::Value {
    stage.map_or(serde_json::Value::Null, |stage| {
        json!({
            "visual": {
                "h": diagnostic_f32_matrix(&stage.visual_h),
                "b": diagnostic_f32_vector(&stage.visual_b),
            },
            "visual_plus_imu": {
                "h": diagnostic_f32_matrix(&stage.visual_imu_h),
                "b": diagnostic_f32_vector(&stage.visual_imu_b),
            },
            "prior": {
                "h": diagnostic_f32_matrix(&stage.prior_h),
                "b": diagnostic_f32_vector(&stage.prior_b),
            },
        })
    })
}

fn full70_reduced_json(
    reduced: &ReducedNormalSystemF32,
    label: &str,
) -> Result<serde_json::Value, ImuReductionError> {
    let compact = if let Some(batch) = reduced.compact_back_substitution.as_ref() {
        let entries = batch
            .entries
            .iter()
            .map(|entry| {
                let len = entry
                    .storage_len()
                    .ok_or(ImuReductionError::Full70OracleInvalid {
                        index: entry.landmark_index,
                    })?;
                let end = entry.storage_offset.checked_add(len).ok_or(
                    ImuReductionError::Full70OracleInvalid {
                        index: entry.landmark_index,
                    },
                )?;
                if end > batch.storage.len() {
                    return Err(ImuReductionError::Full70OracleInvalid {
                        index: entry.landmark_index,
                    });
                }
                Ok(json!({
                    "landmark_index": entry.landmark_index,
                    "track_id": entry.track_id,
                    "storage_offset": entry.storage_offset,
                    "storage_len": len,
                    "state_cols": entry.state_cols,
                    "landmark_cols": entry.landmark_cols,
                    "rank": entry.rank,
                    "eligible": entry.eligible,
                }))
            })
            .collect::<Result<Vec<_>, ImuReductionError>>()?;
        Some(json!({
            "storage": full70_f32_storage_bits(&batch.storage),
            "entries": entries,
        }))
    } else {
        None
    };
    Ok(json!({
        "label": label,
        "h": diagnostic_f32_matrix(&reduced.h),
        "b": diagnostic_f32_vector(&reduced.b),
        "back_substitution_ranks": reduced
            .back_substitution
            .iter()
            .map(|entry| entry.rank)
            .collect::<Vec<_>>(),
        "compact_back_substitution": compact.unwrap_or(serde_json::Value::Null),
        "diagnostic_stages": full70_stage_json(reduced.diagnostic_stages.as_ref()),
        "model_decrease_payload": reduced.model_decrease_payload.is_some(),
    }))
}

fn full70_matrix_bits_match(left: &DMatrix<f32>, right: &DMatrix<f32>) -> bool {
    left.shape() == right.shape()
        && left
            .as_slice()
            .iter()
            .zip(right.as_slice())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn full70_vector_bits_match(left: &DVector<f32>, right: &DVector<f32>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn full70_factor_input_fingerprint(
    factors: &[WhitenedFactorRowStack],
) -> Result<u64, ImuReductionError> {
    let mut hash = LM_VECTOR_FNV_OFFSET;
    for (ordinal, factor) in factors.iter().enumerate() {
        visual_prefix_hash_mix(&mut hash, ordinal as u64);
        visual_prefix_hash_mix(&mut hash, factor.kind as u64);
        visual_prefix_hash_mix(&mut hash, factor.rows() as u64);
        visual_prefix_hash_mix(&mut hash, factor.state_jacobian.ncols() as u64);
        visual_prefix_hash_mix(&mut hash, factor.landmark_jacobian.ncols() as u64);
        for value in factor
            .state_jacobian
            .iter()
            .chain(factor.landmark_jacobian.iter())
            .chain(factor.residual.iter())
        {
            let value = *value as f32;
            if !value.is_finite() {
                return Err(ImuReductionError::Full70OracleInvalid { index: ordinal });
            }
            visual_prefix_hash_mix(&mut hash, u64::from(value.to_bits()));
        }
        if let Some(metadata) = factor.landmark_metadata {
            visual_prefix_hash_mix(&mut hash, metadata.landmark_index as u64);
            visual_prefix_hash_mix(&mut hash, metadata.track_id);
        }
        if let Some(offsets) = factor.imu_link_offsets {
            visual_prefix_hash_mix(&mut hash, offsets.start as u64);
            visual_prefix_hash_mix(&mut hash, offsets.end as u64);
        }
    }
    Ok(hash)
}

fn full70_recovery_json(
    factors: &[WhitenedFactorRowStack],
    legacy: &ReducedNormalSystemF32,
    compact: &ReducedNormalSystemF32,
    state_step: Option<&DVector<f64>>,
    tolerance: f64,
) -> Result<serde_json::Value, ImuReductionError> {
    let Some(state_step) = state_step else {
        return Ok(json!({ "status": "not_exposed_without_solver_step" }));
    };
    let Some(compact_batch) = compact.compact_back_substitution.as_ref() else {
        return Err(ImuReductionError::Full70OracleInvalid { index: 0 });
    };
    let mut records = Vec::with_capacity(compact_batch.entries.len());
    for entry in &compact_batch.entries {
        let Some((factor_index, factor)) = factors.iter().enumerate().find(|(_, factor)| {
            factor.landmark_metadata.is_some_and(|metadata| {
                metadata.landmark_index == entry.landmark_index
                    && metadata.track_id == entry.track_id
            })
        }) else {
            return Err(ImuReductionError::Full70OracleInvalid {
                index: entry.landmark_index,
            });
        };
        let legacy_data = legacy.back_substitution.get(factor_index).ok_or(
            ImuReductionError::Full70OracleInvalid {
                index: factor_index,
            },
        )?;
        let legacy_step = back_substitute_landmark_f32_with_track(
            legacy_data,
            state_step,
            tolerance,
            Some(entry.track_id),
        );
        let compact_step = back_substitute_landmark_compact_entry_f32(
            entry,
            &compact_batch.storage,
            state_step,
            tolerance,
        );
        let legacy_bits = legacy_step.as_ref().map(|step| {
            step.iter()
                .map(|value| format!("{:08x}", (*value as f32).to_bits()))
                .collect::<Vec<_>>()
        });
        let compact_bits = compact_step.as_ref().map(|step| {
            step.iter()
                .map(|value| format!("{:08x}", (*value as f32).to_bits()))
                .collect::<Vec<_>>()
        });
        records.push(json!({
            "factor_index": factor_index,
            "landmark_index": entry.landmark_index,
            "track_id": entry.track_id,
            "eligible": entry.eligible,
            "legacy_f32_bits": legacy_bits,
            "compact_f32_bits": compact_bits,
            "bitwise_equal": legacy_bits == compact_bits,
        }));
        let _ = factor;
    }
    let model = model_cost_decrease_f32(factors, state_step, tolerance)
        .map(|value| format!("{:08x}", (value as f32).to_bits()));
    Ok(json!({
        "status": "captured_from_same_ordered_rows",
        "state_step_f32": full70_checked_vector_bits(state_step, 0)?,
        "landmark_steps": records,
        "model_decrease_f32_row_reference": model,
        "model_decrease_compact_payload": compact.model_decrease_payload.is_some(),
        "model_decrease_note": "current reducer does not expose a compact Q2 model payload; the row reference is captured and compact recovery is compared bitwise",
    }))
}

static FULL70_ORACLE_EVENT: AtomicUsize = AtomicUsize::new(0);

fn full70_oracle_payload(
    frame_id: u64,
    event: &LmDiagnosticEvent<'_>,
    context: &serde_json::Value,
    provenance: &serde_json::Value,
) -> Result<serde_json::Value, ImuReductionError> {
    let factors = &event.linearization.factors;
    let state_dof = event.reduced.h.nrows();
    validate_full70_frame4_factors(factors, state_dof)?;
    let input_fingerprint = full70_factor_input_fingerprint(factors)?;
    let mut row_start = 0usize;
    let mut factor_records = Vec::with_capacity(factors.len());
    let mut visual_count = 0usize;
    for (ordinal, factor) in factors.iter().enumerate() {
        let paired_imu = if factor.kind == FactorKind::Imu {
            let bias = factors
                .get(ordinal + 1)
                .ok_or(ImuReductionError::MissingBias { imu_index: ordinal })?;
            Some(full70_imu_local_payload(factor, bias, ordinal, state_dof)?)
        } else {
            None
        };
        if factor.kind == FactorKind::Visual {
            visual_count += 1;
        }
        factor_records.push(full70_factor_record(
            factor, ordinal, row_start, paired_imu,
        )?);
        row_start = row_start
            .checked_add(factor.rows())
            .ok_or(ImuReductionError::Full70OracleInvalid { index: ordinal })?;
    }
    let legacy = reduce_landmark_factors_f32_checked(factors, state_dof, 1e-10)?;
    let compact = reduce_landmark_factors_f32_checked_with_compact_back_substitution(
        factors, state_dof, 1e-10,
    )?;
    let compact_batch = compact.compact_back_substitution.as_ref().ok_or(
        ImuReductionError::Full70OracleInvalid {
            index: visual_count,
        },
    )?;
    if compact_batch.entries.len() != visual_count {
        return Err(ImuReductionError::InvalidProjectionCount {
            expected: visual_count,
            actual: compact_batch.entries.len(),
        });
    }
    let compact_json = full70_reduced_json(&compact, "compact_f32")?;
    let legacy_json = full70_reduced_json(&legacy, "legacy_f32")?;
    let q2_rows = factors
        .iter()
        .enumerate()
        .map(|(index, factor)| {
            let (jacobian, residual, rank) = landmark_nullspace_projection_f32(factor, 1e-10);
            Ok(json!({
                "factor_index": index,
                "kind": full70_factor_kind_name(factor.kind),
                "rank": rank,
                "jacobian_f32": diagnostic_f32_matrix(&jacobian),
                "residual_f32": diagnostic_f32_vector(&residual),
            }))
        })
        .collect::<Result<Vec<_>, ImuReductionError>>()?;
    let h_equal = full70_matrix_bits_match(&legacy.h, &compact.h);
    let b_equal = full70_vector_bits_match(&legacy.b, &compact.b);
    let recovery = full70_recovery_json(factors, &legacy, &compact, event.step, 1e-10)?;
    let run_id = active_diagnostic_lm_run_id().unwrap_or(0);
    let event_id = FULL70_ORACLE_EVENT.fetch_add(1, Ordering::Relaxed);
    Ok(json!({
        "schema": "basalt.m11.full70_factor_oracle.v1",
        "record": "frame4_factor_event",
        "run_id": format!("{run_id:032x}"),
        "event_id": event_id,
        "frame_id": frame_id,
        "iteration": event.iteration,
        "trial": event.trial,
        "phase": event.phase,
        "state_dof": state_dof,
        "factor_count": factors.len(),
        "row_count": row_start,
        "visual_factor_count": visual_count,
        "factor_order_fingerprint": format!("{:016x}", visual_prefix_factor_order_fingerprint(factors)),
        "factor_input_fingerprint": format!("{:016x}", input_fingerprint),
        "provenance": provenance,
        "context": context,
        "state_f32": full70_checked_vector_bits(event.state, factors.len())?,
        "base_state_f32": full70_checked_vector_bits(event.base_state, factors.len())?,
        "step_f32": event.step.map(|step| full70_checked_vector_bits(step, factors.len())).transpose()?,
        "trial_state_f32": event.trial_state.map(|state| full70_checked_vector_bits(state, factors.len())).transpose()?,
        "factors": factor_records,
        "q2_rows_recomputed_from_ordered_factors": q2_rows,
        "reducer": {
            "legacy": legacy_json,
            "compact": compact_json,
            "h_bitwise_equal": h_equal,
            "b_bitwise_equal": b_equal,
            "stage_order": ["visual", "visual_plus_imu", "prior", "whole_h_b"],
        },
        "recovery": recovery,
        "solver_frontier": {
            "lambda_f32_bits": format!("{:08x}", (event.lambda as f32).to_bits()),
            "lambda_after_f32_bits": format!("{:08x}", (event.lambda_after as f32).to_bits()),
            "decision": event.decision,
            "model_cost": event.model_cost.map(|value| format!("{:08x}", (value as f32).to_bits())),
            "actual_cost": event.actual_cost.map(|value| format!("{:08x}", (value as f32).to_bits())),
        },
        "status": if h_equal && b_equal { "parity_pass" } else { "parity_mismatch" },
    }))
}

/// Emit one complete frame-4 factor event.  All arithmetic in this function is
/// diagnostic-only: the active solver has already produced its event, and the
/// checked reducer calls below operate on immutable copies of the same factors.
/// The path is created with an atomic temp/rename and an existing target is a
/// hard error, preventing an accidental append or partial fixture.
pub(crate) fn emit_full70_factor_oracle(
    path: &Path,
    frame_id: u64,
    event: &LmDiagnosticEvent<'_>,
    context: &serde_json::Value,
) -> Result<(), ImuReductionError> {
    let provenance = full70_oracle_provenance_from_env()?;
    let payload = full70_oracle_payload(frame_id, event, context, &provenance)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(ImuReductionError::Full70OracleIo)?;
    std::fs::create_dir_all(parent).map_err(|_| ImuReductionError::Full70OracleIo)?;
    let event_id = FULL70_ORACLE_EVENT.load(Ordering::Relaxed);
    let mut temporary = path.to_path_buf();
    let temporary_name = format!(
        ".{}.full70.{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or(ImuReductionError::Full70OracleIo)?,
        std::process::id(),
        event_id
    );
    temporary.set_file_name(temporary_name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|_| ImuReductionError::Full70OracleIo)?;
    let write_result = serde_json::to_writer(&mut file, &payload)
        .map_err(|_| ImuReductionError::Full70OracleIo)
        .and_then(|_| {
            file.write_all(b"\n")
                .map_err(|_| ImuReductionError::Full70OracleIo)
        })
        .and_then(|_| file.flush().map_err(|_| ImuReductionError::Full70OracleIo));
    if let Err(error) = write_result {
        drop(file);
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    drop(file);
    if std::fs::rename(&temporary, path).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return Err(ImuReductionError::Full70OracleIo);
    }
    Ok(())
}

const VISUAL_PREFIX_TRACE_CURRENT_TRIAL: usize = 0;

/// The prefix sidecar is an explicitly diagnostic operation.  Consult the
/// process-lifetime diagnostic snapshot before reading its path so the normal
/// reducer does not enumerate the environment (or allocate a path) on every
/// LM iteration.  The sidecar key is intentionally an unknown diagnostic key
/// to the window allowlist, so it selects the retained diagnostic policy at
/// startup when present.
static VISUAL_PREFIX_TRACE_EVENT: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static VISUAL_PREFIX_TRACE_OPEN_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);
static VISUAL_PREFIX_TRACE_FILTER: OnceLock<Result<VisualPrefixTraceFilter, ()>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VisualPrefixTraceFilter {
    frame_id: Option<u64>,
    iteration: Option<usize>,
    trial: Option<usize>,
}

impl VisualPrefixTraceFilter {
    #[inline]
    fn matches(self, frame_id: Option<u64>, iteration: Option<usize>, trial: usize) -> bool {
        self.frame_id.is_none_or(|target| frame_id == Some(target))
            && self
                .iteration
                .is_none_or(|target| iteration == Some(target))
            && self.trial.is_none_or(|target| trial == target)
    }
}

fn visual_prefix_trace_filter_from_cached(
    frame_id: Option<Result<u64, ()>>,
    iteration: Option<Result<usize, ()>>,
    trial: Option<Result<usize, ()>>,
) -> Result<VisualPrefixTraceFilter, ()> {
    Ok(VisualPrefixTraceFilter {
        frame_id: frame_id.transpose()?,
        iteration: iteration.transpose()?,
        trial: trial.transpose()?,
    })
}

fn visual_prefix_trace_filter() -> Result<VisualPrefixTraceFilter, ImuReductionError> {
    match *VISUAL_PREFIX_TRACE_FILTER.get_or_init(|| {
        let policy = crate::vio::window::diagnostic_env_snapshot();
        visual_prefix_trace_filter_from_cached(
            policy.visual_prefix_trace_frame,
            policy.visual_prefix_trace_iteration,
            policy.visual_prefix_trace_trial,
        )
    }) {
        Ok(filter) => Ok(filter),
        Err(()) => Err(ImuReductionError::VisualPrefixTraceInvalid { index: 0 }),
    }
}

fn visual_prefix_trace_path() -> Option<&'static Path> {
    let policy = crate::vio::window::diagnostic_env_snapshot();
    if !policy.active {
        return None;
    }
    policy.visual_prefix_trace.as_deref()
}

#[inline]
fn visual_prefix_hash_mix(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(LM_VECTOR_FNV_PRIME);
}

fn visual_prefix_factor_order_fingerprint(factors: &[WhitenedFactorRowStack]) -> u64 {
    let mut hash = LM_VECTOR_FNV_OFFSET;
    visual_prefix_hash_mix(&mut hash, factors.len() as u64);
    for (factor_index, factor) in factors.iter().enumerate() {
        visual_prefix_hash_mix(&mut hash, factor_index as u64);
        visual_prefix_hash_mix(
            &mut hash,
            match factor.kind {
                FactorKind::Generic => 0,
                FactorKind::Prior => 1,
                FactorKind::Visual => 2,
                FactorKind::Imu => 3,
                FactorKind::Bias => 4,
            },
        );
        visual_prefix_hash_mix(&mut hash, factor.state_jacobian.nrows() as u64);
        visual_prefix_hash_mix(&mut hash, factor.state_jacobian.ncols() as u64);
        visual_prefix_hash_mix(&mut hash, factor.landmark_jacobian.ncols() as u64);
        if let Some(metadata) = factor.landmark_metadata {
            visual_prefix_hash_mix(&mut hash, metadata.landmark_index as u64);
            visual_prefix_hash_mix(&mut hash, metadata.track_id);
        } else {
            visual_prefix_hash_mix(&mut hash, u64::MAX);
            visual_prefix_hash_mix(&mut hash, u64::MAX);
        }
        if let Some(observations) = factor.visual_observation_ids.as_ref() {
            visual_prefix_hash_mix(&mut hash, observations.len() as u64);
            for &(state_index, camera_id) in observations {
                visual_prefix_hash_mix(&mut hash, state_index as u64);
                visual_prefix_hash_mix(&mut hash, camera_id as u64);
            }
        } else {
            visual_prefix_hash_mix(&mut hash, u64::MAX);
        }
    }
    hash
}

static VISUAL_PREFIX_TRACE_WRITERS: OnceLock<Mutex<HashSet<std::path::PathBuf>>> = OnceLock::new();

fn visual_prefix_trace_writer_registry() -> &'static Mutex<HashSet<std::path::PathBuf>> {
    VISUAL_PREFIX_TRACE_WRITERS.get_or_init(|| Mutex::new(HashSet::new()))
}

struct VisualPrefixTraceLease {
    key: std::path::PathBuf,
    lock_path: std::path::PathBuf,
    lock_file: Option<std::fs::File>,
}

impl Drop for VisualPrefixTraceLease {
    fn drop(&mut self) {
        // Release the OS handle before removing the adjacent lock file.  A
        // stale lock is never removed by acquisition; only this live lease
        // cleans up the file it successfully created.
        self.lock_file.take();
        let _ = std::fs::remove_file(&self.lock_path);
        let Ok(mut paths) = visual_prefix_trace_writer_registry().lock() else {
            // A poisoned registry is already fail-closed for new writers; do
            // not panic while releasing a diagnostic lease.
            return;
        };
        paths.remove(&self.key);
    }
}

fn canonical_visual_prefix_trace_key(path: &Path) -> Result<std::path::PathBuf, ImuReductionError> {
    if path.as_os_str().is_empty() || path.file_name().is_none() {
        return Err(ImuReductionError::VisualPrefixTraceIo);
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?
            .join(path)
    };
    let parent = absolute
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
    std::fs::create_dir_all(parent).map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
    let canonical_parent =
        std::fs::canonicalize(parent).map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
    let file_name = absolute
        .file_name()
        .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
    // Resolve an existing file/symlink when possible.  For a new file, the
    // canonical parent plus filename is the stable same-volume key.
    Ok(std::fs::canonicalize(&absolute).unwrap_or_else(|_| canonical_parent.join(file_name)))
}

fn visual_prefix_trace_lock_path(
    canonical_target: &Path,
) -> Result<std::path::PathBuf, ImuReductionError> {
    let parent = canonical_target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
    let file_name = canonical_target
        .file_name()
        .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
    let mut lock_name = std::ffi::OsString::from(".");
    lock_name.push(file_name);
    lock_name.push(".lock");
    Ok(parent.join(lock_name))
}

fn visual_prefix_path_hash(canonical_target: &Path) -> u64 {
    let mut hash = LM_VECTOR_FNV_OFFSET;
    for byte in canonical_target.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(LM_VECTOR_FNV_PRIME);
    }
    hash
}

fn acquire_visual_prefix_trace_lock(
    canonical_target: &Path,
    run_id: u128,
    event_id: usize,
) -> Result<(std::path::PathBuf, std::fs::File), ImuReductionError> {
    let lock_path = visual_prefix_trace_lock_path(canonical_target)?;
    // `create_new` is the cross-process ownership boundary.  Existing,
    // malformed, or stale locks are deliberately not inspected or removed:
    // the caller must resolve them explicitly.
    let mut lock_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
    let payload = json!({
        "schema": "basalt.m11.visual_prefix_trace_lock.v1",
        "run_id": format!("{run_id:032x}"),
        "pid": std::process::id(),
        "event_id": event_id,
        "target_path_hash": format!("{:016x}", visual_prefix_path_hash(canonical_target)),
    });
    let write_result = serde_json::to_writer(&mut lock_file, &payload)
        .map_err(|_| ImuReductionError::VisualPrefixTraceIo)
        .and_then(|_| {
            lock_file
                .write_all(b"\n")
                .map_err(|_| ImuReductionError::VisualPrefixTraceIo)
        });
    if let Err(error) = write_result {
        drop(lock_file);
        let _ = std::fs::remove_file(&lock_path);
        return Err(error);
    }
    Ok((lock_path, lock_file))
}

fn reserve_visual_prefix_trace_path(
    path: &Path,
    run_id: u128,
    event_id: usize,
) -> Result<VisualPrefixTraceLease, ImuReductionError> {
    let key = canonical_visual_prefix_trace_key(path)?;
    let mut paths = visual_prefix_trace_writer_registry()
        .lock()
        .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
    if !paths.insert(key.clone()) {
        return Err(ImuReductionError::VisualPrefixTraceIo);
    }
    drop(paths);
    let (lock_path, lock_file) = match acquire_visual_prefix_trace_lock(&key, run_id, event_id) {
        Ok(lock) => lock,
        Err(error) => {
            if let Ok(mut paths) = visual_prefix_trace_writer_registry().lock() {
                paths.remove(&key);
            }
            return Err(error);
        }
    };
    Ok(VisualPrefixTraceLease {
        key,
        lock_path,
        lock_file: Some(lock_file),
    })
}

struct VisualPrefixTraceWriter {
    file: std::fs::File,
    _lease: VisualPrefixTraceLease,
    event_id: usize,
    run_id: u128,
    record_sequence: usize,
    frame_id: Option<u64>,
    iteration: Option<usize>,
    state_dof: usize,
    expected_visual_count: usize,
    next_visual_ordinal: usize,
    factor_order_fingerprint: u64,
}

struct VisualPrefixPending {
    visual_ordinal: usize,
    factor_index: usize,
    landmark_index: usize,
    track_id: u64,
    observations: Vec<serde_json::Value>,
    row_count: usize,
    landmark_dof: usize,
    rank: usize,
    active_state_columns: Vec<usize>,
    global_h_before: serde_json::Value,
    global_b_before: serde_json::Value,
}

impl VisualPrefixTraceWriter {
    fn open(
        path: impl AsRef<Path>,
        factors: &[WhitenedFactorRowStack],
        state_dof: usize,
    ) -> Result<Option<Self>, ImuReductionError> {
        #[cfg(test)]
        VISUAL_PREFIX_TRACE_OPEN_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
        let path = path.as_ref();
        if path.as_os_str().is_empty() || path.file_name().is_none() {
            return Err(ImuReductionError::VisualPrefixTraceIo);
        }
        let event_id = VISUAL_PREFIX_TRACE_EVENT.fetch_add(1, Ordering::Relaxed);
        let run_id = active_diagnostic_lm_run_id().unwrap_or_else(next_diagnostic_lm_run_id);
        let lease = reserve_visual_prefix_trace_path(path, run_id, event_id)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
        let frame_id = active_diagnostic_lm_frame();
        let iteration = active_diagnostic_lm_iteration();
        let expected_visual_count = factors
            .iter()
            .filter(|factor| {
                factor.kind == FactorKind::Visual && factor.landmark_jacobian.ncols() != 0
            })
            .count();
        let factor_order_fingerprint = visual_prefix_factor_order_fingerprint(factors);
        let mut writer = Self {
            file,
            _lease: lease,
            event_id,
            run_id,
            record_sequence: 0,
            frame_id,
            iteration,
            state_dof,
            expected_visual_count,
            next_visual_ordinal: 0,
            factor_order_fingerprint,
        };
        let run_id = writer.run_id_string();
        let record_sequence = writer.record_sequence;
        writer.write_record(&json!({
            "schema": "basalt.m11.absqr.visual_prior_boundary.v1",
            "record": "header",
            "event_id": event_id,
            "run_id": run_id,
            "record_sequence": record_sequence,
            "frame_id": frame_id,
            "iteration": iteration,
            "trial": 0,
            "state_dof": state_dof,
            "factor_count": factors.len(),
            "visual_factor_count": expected_visual_count,
            "factor_order_fingerprint": format!("{:016x}", factor_order_fingerprint),
        }))?;
        Ok(Some(writer))
    }

    fn run_id_string(&self) -> String {
        format!("{:032x}", self.run_id)
    }

    fn write_record(&mut self, record: &serde_json::Value) -> Result<(), ImuReductionError> {
        let next_sequence = self
            .record_sequence
            .checked_add(1)
            .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
        serde_json::to_writer(&mut self.file, record)
            .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
        self.file
            .write_all(b"\n")
            .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
        self.record_sequence = next_sequence;
        Ok(())
    }

    fn begin_visual_prefix(
        &mut self,
        factor_index: usize,
        factor: &WhitenedFactorRowStack,
        projected_jacobian: &DMatrix<f32>,
        projected_residual: &DVector<f32>,
        rank: usize,
        global_h: &DMatrix<f32>,
        global_b: &DVector<f32>,
    ) -> Result<VisualPrefixPending, ImuReductionError> {
        if factor.kind != FactorKind::Visual
            || factor.landmark_jacobian.ncols() == 0
            || projected_jacobian.ncols() != self.state_dof
            || projected_residual.len() != projected_jacobian.nrows()
            || self.next_visual_ordinal >= self.expected_visual_count
        {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: factor_index,
            });
        }
        let Some(metadata) = factor.landmark_metadata else {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: factor_index,
            });
        };
        let Some(observation_ids) = factor.visual_observation_ids.as_ref() else {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: factor_index,
            });
        };
        let observations = observation_ids
            .iter()
            .map(|&(state_index, camera_id)| {
                json!({
                    "state_index": state_index,
                    "camera_id": camera_id,
                })
            })
            .collect();
        let active_state_columns = (0..factor.state_jacobian.ncols())
            .filter(|&column| {
                (0..factor.state_jacobian.nrows())
                    .any(|row| factor.state_jacobian[(row, column)] != 0.0)
            })
            .collect();
        Ok(VisualPrefixPending {
            visual_ordinal: self.next_visual_ordinal,
            factor_index,
            landmark_index: metadata.landmark_index,
            track_id: metadata.track_id,
            observations,
            row_count: projected_jacobian.nrows(),
            landmark_dof: factor.landmark_jacobian.ncols(),
            rank,
            active_state_columns,
            global_h_before: diagnostic_f32_matrix(global_h),
            global_b_before: diagnostic_f32_vector(global_b),
        })
    }

    fn finish_visual_prefix(
        &mut self,
        pending: VisualPrefixPending,
        global_h: &DMatrix<f32>,
        global_b: &DVector<f32>,
    ) -> Result<(), ImuReductionError> {
        if pending.visual_ordinal != self.next_visual_ordinal
            || global_h.nrows() != self.state_dof
            || global_h.ncols() != self.state_dof
            || global_b.len() != self.state_dof
        {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: pending.factor_index,
            });
        }
        let run_id = self.run_id_string();
        let record_sequence = self.record_sequence;
        self.write_record(&json!({
            "schema": "basalt.m11.absqr.visual_prior_boundary.v1",
            "record": "visual_prefix",
            "event_id": self.event_id,
            "run_id": run_id,
            "record_sequence": record_sequence,
            "frame_id": self.frame_id,
            "iteration": self.iteration,
            "trial": 0,
            "visual_ordinal": pending.visual_ordinal,
            "factor_index": pending.factor_index,
            "landmark_index": pending.landmark_index,
            "track_id": pending.track_id,
            "observations": pending.observations,
            "row_count": pending.row_count,
            "landmark_dof": pending.landmark_dof,
            "rank": pending.rank,
            "factor_order_fingerprint": format!("{:016x}", self.factor_order_fingerprint),
            "aom_scatter": {
                "state_dof": self.state_dof,
                "active_state_columns": pending.active_state_columns,
                "landmark_columns_eliminated": pending.landmark_dof,
            },
            "global_h_before": pending.global_h_before,
            "global_b_before": pending.global_b_before,
            "global_h_after": diagnostic_f32_matrix(global_h),
            "global_b_after": diagnostic_f32_vector(global_b),
        }))?;
        self.next_visual_ordinal += 1;
        Ok(())
    }

    fn write_stage(
        &mut self,
        stage: &str,
        global_h: &DMatrix<f32>,
        global_b: &DVector<f32>,
    ) -> Result<(), ImuReductionError> {
        if global_h.nrows() != self.state_dof
            || global_h.ncols() != self.state_dof
            || global_b.len() != self.state_dof
        {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: self.next_visual_ordinal,
            });
        }
        let run_id = self.run_id_string();
        let record_sequence = self.record_sequence;
        self.write_record(&json!({
            "schema": "basalt.m11.absqr.visual_prior_boundary.v1",
            "record": "stage",
            "stage": stage,
            "event_id": self.event_id,
            "run_id": run_id,
            "record_sequence": record_sequence,
            "frame_id": self.frame_id,
            "iteration": self.iteration,
            "trial": 0,
            "visual_ordinal_count": self.next_visual_ordinal,
            "factor_order_fingerprint": format!("{:016x}", self.factor_order_fingerprint),
            "global_h": diagnostic_f32_matrix(global_h),
            "global_b": diagnostic_f32_vector(global_b),
        }))
    }

    fn write_prior_before(
        &mut self,
        global_h: &DMatrix<f32>,
        global_b: &DVector<f32>,
        prior_h: &DMatrix<f32>,
        prior_b: &DVector<f32>,
    ) -> Result<(), ImuReductionError> {
        if self.next_visual_ordinal != self.expected_visual_count {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: self.next_visual_ordinal,
            });
        }
        let run_id = self.run_id_string();
        let record_sequence = self.record_sequence;
        self.write_record(&json!({
            "schema": "basalt.m11.absqr.visual_prior_boundary.v1",
            "record": "prior_boundary",
            "stage": "prior_before",
            "event_id": self.event_id,
            "run_id": run_id,
            "record_sequence": record_sequence,
            "frame_id": self.frame_id,
            "iteration": self.iteration,
            "trial": 0,
            "visual_ordinal_count": self.next_visual_ordinal,
            "factor_order_fingerprint": format!("{:016x}", self.factor_order_fingerprint),
            "global_h": diagnostic_f32_matrix(global_h),
            "global_b": diagnostic_f32_vector(global_b),
            "prior_h": diagnostic_f32_matrix(prior_h),
            "prior_b": diagnostic_f32_vector(prior_b),
        }))
    }
}

fn visual_prefix_trace_writer(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<Option<VisualPrefixTraceWriter>, ImuReductionError> {
    if !crate::vio::window::diagnostic_env_active() {
        return Ok(None);
    }
    // Apply the cheap cached selector before reading/cloning the output path
    // or opening the file.  A non-target event therefore allocates neither a
    // header/prefix record nor any sidecar metadata/full H/b payload.  The
    // current prefix records are iteration-level reductions and use trial 0;
    // retaining trial in the selector keeps the contract explicit for future
    // trial-level producers without guessing a trial for this producer.
    let filter = visual_prefix_trace_filter()?;
    let frame_id = active_diagnostic_lm_frame();
    let iteration = active_diagnostic_lm_iteration();
    let trial = VISUAL_PREFIX_TRACE_CURRENT_TRIAL;
    if !filter.matches(frame_id, iteration, trial) {
        return Ok(None);
    }
    let path = visual_prefix_trace_path();
    visual_prefix_trace_writer_selected(
        filter, frame_id, iteration, trial, path, factors, state_dof,
    )
}

/// Apply the cheap event selector before resolving the output path or opening
/// the sidecar.  Keeping this boundary separate makes the no-op contract
/// testable: a non-target event must not invoke [`VisualPrefixTraceWriter`]
/// (and therefore must not copy factor metadata or materialize any H/b JSON).
#[inline]
fn visual_prefix_trace_writer_selected(
    filter: VisualPrefixTraceFilter,
    frame_id: Option<u64>,
    iteration: Option<usize>,
    trial: usize,
    path: Option<&Path>,
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<Option<VisualPrefixTraceWriter>, ImuReductionError> {
    if !filter.matches(frame_id, iteration, trial) {
        return Ok(None);
    }
    let Some(path) = path else {
        return Ok(None);
    };
    VisualPrefixTraceWriter::open(path, factors, state_dof)
}

/// Emit the exact transformed Q1 inputs used by the float32 landmark
/// back-substitution.  This is intentionally opt-in: the production solver
/// never opens a sidecar and the probe is restricted to the two tracks that
/// first exposed the Eigen 3x75 GEMV reduction boundary.
fn emit_landmark_backsub_probe(
    track_id: Option<u64>,
    state_step: &DVector<f32>,
    q1_state: &DMatrix<f32>,
    q1_residual: &DVector<f32>,
    q1_state_step: &DVector<f32>,
    upper_r: &DMatrix<f32>,
) {
    let Some(track_id) = track_id else {
        return;
    };
    let policy = crate::vio::window::diagnostic_env_snapshot();
    let Some(path) = policy.landmark_backsub_probe.as_ref() else {
        return;
    };
    let selected = if let Some(filter) = policy.landmark_backsub_probe_tracks.as_deref() {
        filter.iter().any(|value| *value == track_id)
    } else {
        track_id == 19 || track_id == 49
    };
    if !selected {
        return;
    }
    let record = json!({
        "schema": "basalt.m7im15_landmark_backsub_probe.v1",
        "track_id": track_id,
        "state_step": diagnostic_f32_vector(state_step),
        "q1_state": diagnostic_f32_matrix(q1_state),
        "q1_residual": diagnostic_f32_vector(q1_residual),
        "q1_state_step": diagnostic_f32_vector(q1_state_step),
        "upper_r": diagnostic_f32_matrix(upper_r),
    });
    let Ok(line) = serde_json::to_string(&record) else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

/// Emit the complete per-landmark QR boundary for the selected diagnostic
/// track.  This reuses the existing back-sub probe path so callers do not
/// need another environment key or a second factor construction.  The hook
/// is called only after the production Householder walk has completed; it
/// never participates in Q2 assembly or landmark acceptance.
fn emit_landmark_projection_probe(
    metadata: LandmarkFactorMetadata,
    factor: &WhitenedFactorRowStack,
    state: &DMatrix<f32>,
    landmark: &DMatrix<f32>,
    residual: &DVector<f32>,
    qr: &LandmarkHouseholderF32,
    rank: usize,
    norm_ok: bool,
) {
    let policy = crate::vio::window::diagnostic_env_snapshot();
    let Some(path) = policy.landmark_backsub_probe.as_ref() else {
        return;
    };
    let selected = if let Some(filter) = policy.landmark_backsub_probe_tracks.as_deref() {
        filter.iter().any(|value| *value == metadata.track_id)
    } else {
        metadata.track_id == 19 || metadata.track_id == 49
    };
    if !selected {
        return;
    }
    let bits = |value: f32| format!("{:08x}", value.to_bits());
    let vector_bits = |values: &[f32]| json!(values.iter().copied().map(bits).collect::<Vec<_>>());
    let transformed_state = qr.transformed_state();
    let transformed_landmark = qr.transformed_landmark();
    let transformed_residual = qr.transformed_residual();
    let q1_state = DMatrix::from_fn(qr.landmark_cols, qr.state_cols, |row, column| {
        transformed_state[(row, column)]
    });
    let q1_residual = DVector::from_iterator(
        qr.landmark_cols,
        (0..qr.landmark_cols).map(|row| transformed_residual[row]),
    );
    let q2_state = if qr.rows > qr.landmark_cols {
        qr.q2_state()
    } else {
        DMatrix::zeros(0, qr.state_cols)
    };
    let q2_residual = if qr.rows > qr.landmark_cols {
        qr.q2_residual()
    } else {
        DVector::zeros(0)
    };
    let projection_event = active_diagnostic_projection_event();
    let run_id = active_diagnostic_lm_run_id().map(|value| format!("{value:032x}"));
    let observations = factor
        .visual_observation_ids
        .as_ref()
        .map(|values| {
            values
                .iter()
                .map(|(state_index, camera_id)| {
                    json!({"state_index": state_index, "camera_id": camera_id})
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let record = json!({
        "schema": "basalt.m11.track_numeric_projection.v1",
        "source": "rust",
        "stage": "landmark_projection_after_householder",
        "run_id": run_id,
        "frame_id": active_diagnostic_lm_frame(),
        "event_frame_id": projection_event.map(|event| event.frame_id),
        "event_timestamp_ns": projection_event.map(|event| event.timestamp_ns),
        "iteration": active_diagnostic_lm_iteration(),
        "trial": 0,
        "landmark_index": metadata.landmark_index,
        "track_id": metadata.track_id,
        "factor_rows": factor.rows(),
        "state_cols": factor.state_jacobian.ncols(),
        "landmark_cols": factor.landmark_jacobian.ncols(),
        "observations": observations,
        "rank": rank,
        "norm_ok": norm_ok,
        "tolerance_f32_bits": bits(1e-10_f64 as f32),
        "pre_qr": {
            "state": diagnostic_f32_matrix(state),
            "landmark": diagnostic_f32_matrix(landmark),
            "residual": diagnostic_f32_vector(residual),
        },
        "post_qr": {
            "rows": qr.rows,
            "state_cols": qr.state_cols,
            "landmark_cols": qr.landmark_cols,
            "landmark_offset": qr.landmark_offset,
            "residual_offset": qr.residual_offset,
            "transformed_state": diagnostic_f32_matrix(&transformed_state),
            "transformed_landmark": diagnostic_f32_matrix(&transformed_landmark),
            "transformed_residual": diagnostic_f32_vector(&transformed_residual),
            "pivots_f32_bits": vector_bits(&qr.pivots),
            "tau_f32_bits": vector_bits(&qr.tau),
        },
        "q1": {
            "state": diagnostic_f32_matrix(&q1_state),
            "residual": diagnostic_f32_vector(&q1_residual),
            "upper_r": diagnostic_f32_matrix(&qr.upper_r()),
        },
        "q2": {
            "state": diagnostic_f32_matrix(&q2_state),
            "residual": diagnostic_f32_vector(&q2_residual),
        },
    });
    let Ok(line) = serde_json::to_string(&record) else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

/// Emit only the selected frame/iterations.  This is JSONL rather than the
/// ordinary detail schema: two frame-8 snapshots are roughly tens of KB and
/// contain no landmarks, observations, or state dumps.
fn emit_imu_reduction_diagnostic(
    frame_id: Option<u64>,
    iteration: usize,
    reduced: &ReducedNormalSystemF32,
) {
    let policy = crate::vio::window::diagnostic_env_snapshot();
    let Some(path) = policy.diagnostic_imu_rows.as_ref() else {
        return;
    };
    let target_frame = policy.diagnostic_frame;
    if target_frame.is_some() && target_frame != frame_id {
        return;
    }
    if let Some(filter) = policy.diagnostic_imu_iterations.as_deref() {
        let selected = filter.iter().any(|value| *value == iteration);
        if !selected {
            return;
        }
    }
    let Some(imu) = reduced.imu_diagnostic.as_ref() else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    if INITIALIZED.get().is_none() {
        let _ = INITIALIZED.set(());
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, b"");
    }
    let local_blocks = imu
        .local_blocks
        .iter()
        .map(|block| {
            let mut record = json!({
                "active_offsets": block.active_offsets,
                "local_jacobian": diagnostic_f32_matrix(&block.local_jacobian),
                "residual": diagnostic_f32_vector(&block.residual),
                "local_h": diagnostic_f32_matrix(&block.local_h),
                "local_b": diagnostic_f32_vector(&block.local_b),
                "local_vs_global_active": {
                    "h_bit_mismatches": block.local_vs_global_h_mismatches,
                    "b_bit_mismatches": block.local_vs_global_b_mismatches,
                },
                "local_vs_global_padded": {
                    "h_bit_mismatches": block.local_vs_global_padded_h_mismatches,
                    "b_bit_mismatches": block.local_vs_global_padded_b_mismatches,
                    "h_total": reduced.h.nrows() * reduced.h.ncols(),
                    "b_total": reduced.b.len(),
                },
            });
            // Keep the stage names at the local-block level so an audit can
            // consume one block without knowing an additional wrapper
            // schema.  Retain the complete nested copy as well for callers
            // that prefer to treat the input boundary as one payload.
            if let Some(input) = block.imu_input_diagnostic.as_ref() {
                if let Some(fields) = input.as_object() {
                    let object = record
                        .as_object_mut()
                        .expect("local IMU diagnostic record is an object");
                    for (key, value) in fields {
                        object.insert(key.clone(), value.clone());
                    }
                    object.insert("factor_input".into(), input.clone());
                }
            }
            record
        })
        .collect::<Vec<_>>();
    let imu_cumulative_stages = imu
        .imu_cumulative_stages
        .iter()
        .map(|stage| {
            let mut record = json!({
                "active_offsets": stage.active_offsets,
                "imu_h": diagnostic_f32_matrix(&stage.imu_h),
                "imu_b": diagnostic_f32_vector(&stage.imu_b),
            });
            if let Some(input) = stage.imu_input_diagnostic.as_ref() {
                if let Some(fields) = input.as_object() {
                    let object = record
                        .as_object_mut()
                        .expect("cumulative IMU diagnostic record is an object");
                    for (key, value) in fields {
                        object.insert(key.clone(), value.clone());
                    }
                    object.insert("factor_input".into(), input.clone());
                }
            }
            record
        })
        .collect::<Vec<_>>();
    let record = json!({
        "schema": "basalt.m7im15_reduction_diagnostic.v3",
        "frame_id": frame_id,
        "iteration": iteration,
        "state_dof": reduced.h.nrows(),
        "local_blocks": local_blocks,
        "imu_cumulative_stages": imu_cumulative_stages,
        "imu_accumulator": {
            "h": diagnostic_f32_matrix(&imu.imu_h),
            "b": diagnostic_f32_vector(&imu.imu_b),
        },
        "full": {
            "h": diagnostic_f32_matrix(&reduced.h),
            "b": diagnostic_f32_vector(&reduced.b),
        },
    });
    let Ok(line) = serde_json::to_string(&record) else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

const IMU_LOCAL_ROWS: usize = 15;
const IMU_LOCAL_COLS: usize = AOM_NAV_DOF * 2;

/// Multiply one local IMU/bias row stack using Eigen's fixed 15x30 dynamic
/// product schedule.
///
/// `ImuBlock::add_dense_H_b` materializes a 15-row by 30-column `MatrixXf`
/// before evaluating `Jp.transpose() * Jp` and `Jp.transpose() * r`.  The
/// product is not equivalent, bit-for-bit, to reducing a zero-padded global
/// matrix: the AVX GEBP kernel dispatches output rows 0..23, 24..27 and
/// 28..29 through different packet tails.  Keep this helper independent of
/// the production reducer until the active-window metadata is available.
///
/// The matrix and vector are intentionally dynamic at the boundary to match
/// upstream's `MatrixXf`/`VectorXf` expressions.  Shape assertions make an
/// accidental visual/prior row stack fail at the call site rather than being
/// silently interpreted as an IMU packet.
#[inline]
fn local_imu_h_b_15x30(
    jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
) -> (DMatrix<f32>, DVector<f32>) {
    assert_eq!(jacobian.nrows(), IMU_LOCAL_ROWS);
    assert_eq!(jacobian.ncols(), IMU_LOCAL_COLS);
    assert_eq!(residual.len(), IMU_LOCAL_ROWS);

    let mut local_h = DMatrix::<f32>::zeros(IMU_LOCAL_COLS, IMU_LOCAL_COLS);
    for row in 0..IMU_LOCAL_COLS {
        for column in 0..IMU_LOCAL_COLS {
            let value = if (24..28).contains(&row) && column < 28 {
                // The 24..27 half-packet tail has two Packet4 C/D
                // accumulators.  Reduce the first eight depths before the
                // scalar FMA tail, preserving the source association.
                let mut even = 0.0_f32;
                let mut odd = 0.0_f32;
                for depth in (0..8).step_by(2) {
                    even = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], even);
                    odd = jacobian[(depth + 1, row)].mul_add(jacobian[(depth + 1, column)], odd);
                }
                let mut value = add_f32_exact(even, odd);
                for depth in 8..IMU_LOCAL_ROWS {
                    value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
                }
                value
            } else if row >= 28 && column < 28 {
                // The remaining-row 1x4 kernel uses SwappedTraits with
                // spk=2.  Four even and four odd accumulators are reduced
                // as (C0+C1)+(C2+C3), then paired depths 8..13 are fused
                // into the corresponding lane.  Depth 14 is the scalar
                // FMA after the parity reduction.
                let mut even = [0.0_f32; 4];
                let mut odd = [0.0_f32; 4];
                for (lane, depth) in (0..8).step_by(2).enumerate() {
                    even[lane] = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], 0.0);
                    odd[lane] =
                        jacobian[(depth + 1, row)].mul_add(jacobian[(depth + 1, column)], 0.0);
                }
                let mut even_sum = add_f32_exact(
                    add_f32_exact(even[0], even[1]),
                    add_f32_exact(even[2], even[3]),
                );
                let mut odd_sum =
                    add_f32_exact(add_f32_exact(odd[0], odd[1]), add_f32_exact(odd[2], odd[3]));
                for depth in (8..14).step_by(2) {
                    even_sum = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], even_sum);
                    odd_sum =
                        jacobian[(depth + 1, row)].mul_add(jacobian[(depth + 1, column)], odd_sum);
                }
                let reduced = add_f32_exact(even_sum, odd_sum);
                jacobian[(14, row)].mul_add(jacobian[(14, column)], reduced)
            } else {
                // Full packet rows and the X1 columns retain direct depth
                // order, including the scalar tail.
                let mut value = 0.0_f32;
                for depth in 0..IMU_LOCAL_ROWS {
                    value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
                }
                value
            };
            local_h[(row, column)] = value;
        }
    }

    let mut local_b = DVector::<f32>::zeros(IMU_LOCAL_COLS);
    for column in 0..IMU_LOCAL_COLS {
        let mut packet_lanes = [0.0_f32; 8];
        for depth in 0..8 {
            packet_lanes[depth] = mul_f32_exact(jacobian[(depth, column)], residual[depth]);
        }

        // AVX Packet8f predux: q0=l0+l4, q1=l1+l5, q2=l2+l6,
        // q3=l3+l7; (q0+q2)+(q1+q3).
        let q0 = add_f32_exact(packet_lanes[0], packet_lanes[4]);
        let q1 = add_f32_exact(packet_lanes[1], packet_lanes[5]);
        let q2 = add_f32_exact(packet_lanes[2], packet_lanes[6]);
        let q3 = add_f32_exact(packet_lanes[3], packet_lanes[7]);
        let low = add_f32_exact(q0, q2);
        let high = add_f32_exact(q1, q3);
        let mut value = add_f32_exact(low, high);

        // Eigen's dynamic cleanup emits ordinary multiply/add for depths
        // 8..11, followed by scalar FMAs for depths 12..14.
        for depth in 8..12 {
            value = add_f32_exact(
                value,
                mul_f32_exact(jacobian[(depth, column)], residual[depth]),
            );
        }
        for depth in 12..IMU_LOCAL_ROWS {
            value = jacobian[(depth, column)].mul_add(residual[depth], value);
        }
        local_b[column] = value;
    }

    (local_h, local_b)
}

/// Add one local 30x30/30 IMU product to the global state accumulator.
///
/// The four matrix blocks and two vector blocks are visited in upstream's
/// order (`H00`, `H10`, `H01`, `H11`, `b0`, `b1`).  In particular, `H10` is
/// copied as-is into the `(end,start)` block; it must not be transposed to
/// repair the mathematical symmetry because the local packet schedule can
/// produce distinct low bits in the two cross blocks.
#[inline]
fn scatter_local_imu_h_b_15x30(
    accumulator_h: &mut DMatrix<f32>,
    accumulator_b: &mut DVector<f32>,
    local_h: &DMatrix<f32>,
    local_b: &DVector<f32>,
    offsets: ImuLinkOffsets,
) -> Result<(), ImuReductionError> {
    if local_h.nrows() != IMU_LOCAL_COLS
        || local_h.ncols() != IMU_LOCAL_COLS
        || local_b.len() != IMU_LOCAL_COLS
    {
        return Err(ImuReductionError::InvalidLocalProduct {
            h_rows: local_h.nrows(),
            h_cols: local_h.ncols(),
            b_len: local_b.len(),
        });
    }
    if accumulator_h.nrows() != accumulator_h.ncols()
        || accumulator_b.len() != accumulator_h.nrows()
    {
        return Err(ImuReductionError::InvalidAccumulator {
            h_rows: accumulator_h.nrows(),
            h_cols: accumulator_h.ncols(),
            b_len: accumulator_b.len(),
        });
    }
    validate_imu_offsets(0, offsets, accumulator_h.nrows())?;

    let add_matrix_block = |accumulator: &mut DMatrix<f32>,
                            local_row_offset: usize,
                            global_row_offset: usize,
                            local_column_offset: usize,
                            global_column_offset: usize| {
        for row in 0..AOM_NAV_DOF {
            for column in 0..AOM_NAV_DOF {
                accumulator[(global_row_offset + row, global_column_offset + column)] +=
                    local_h[(local_row_offset + row, local_column_offset + column)];
            }
        }
    };

    // Keep these calls separate: block order is part of the f32 contract.
    add_matrix_block(accumulator_h, 0, offsets.start, 0, offsets.start);
    add_matrix_block(accumulator_h, AOM_NAV_DOF, offsets.end, 0, offsets.start);
    add_matrix_block(accumulator_h, 0, offsets.start, AOM_NAV_DOF, offsets.end);
    add_matrix_block(
        accumulator_h,
        AOM_NAV_DOF,
        offsets.end,
        AOM_NAV_DOF,
        offsets.end,
    );
    for row in 0..AOM_NAV_DOF {
        accumulator_b[offsets.start + row] += local_b[row];
    }
    for row in 0..AOM_NAV_DOF {
        accumulator_b[offsets.end + row] += local_b[AOM_NAV_DOF + row];
    }
    Ok(())
}

#[inline]
fn add_f32_exact(left: f32, right: f32) -> f32 {
    let result = left + right;
    result
}

#[inline(never)]
fn mul_f32_exact(left: f32, right: f32) -> f32 {
    let result = left * right;
    result
}

#[inline]
fn eigen_predux8_f32(lanes: [f32; 8]) -> f32 {
    let q0 = add_f32_exact(lanes[0], lanes[4]);
    let q1 = add_f32_exact(lanes[1], lanes[5]);
    let q2 = add_f32_exact(lanes[2], lanes[6]);
    let q3 = add_f32_exact(lanes[3], lanes[7]);
    add_f32_exact(add_f32_exact(q0, q2), add_f32_exact(q1, q3))
}

#[inline]
fn eigen_predux4_f32(lanes: [f32; 4]) -> f32 {
    let q0 = add_f32_exact(lanes[0], lanes[2]);
    let q1 = add_f32_exact(lanes[1], lanes[3]);
    add_f32_exact(q0, q1)
}

#[inline]
fn eigen_q2_dot_fma_f32(jacobian: &DMatrix<f32>, row: usize, column: usize) -> f32 {
    let mut value = 0.0_f32;
    for depth in 0..jacobian.nrows() {
        value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
    }
    value
}

#[inline]
fn eigen_q2_dot_packet_parity_f32(jacobian: &DMatrix<f32>, row: usize, column: usize) -> f32 {
    // `lhs_process_one_packet` (and its half-packet specialization) keeps
    // two accumulators for each four-column RHS panel.  The peeled depth is
    // processed as K=0,2,... into C and K=1,3,... into D; only after the
    // packet loop does Eigen add C+D and visit the scalar remainder.
    let peeled_depth = jacobian.nrows() / 8 * 8;
    let mut even = 0.0_f32;
    let mut odd = 0.0_f32;
    for depth in (0..peeled_depth).step_by(2) {
        even = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], even);
        odd = jacobian[(depth + 1, row)].mul_add(jacobian[(depth + 1, column)], odd);
    }
    let mut value = add_f32_exact(even, odd);
    for depth in peeled_depth..jacobian.nrows() {
        value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
    }
    value
}

/// Reproduce Eigen's AVX2 `gebp` reduction for a dynamic f32 `JᵀJ`.
///
/// Eigen's packet kernel has `mr=24`, `nr=4`, and packet width eight.  The
/// 24-row and 16-row panels accumulate each depth in order; the 8-row and
/// 4-row tails use the packet kernel's doubled C/D accumulators for complete
/// four-column RHS panels.  A non-multiple-of-four RHS column remains on the
/// scalar one-column path and therefore uses the ordered FMA dot product.
#[inline]
fn eigen_q2_gram_f32(jacobian: &DMatrix<f32>) -> DMatrix<f32> {
    let columns = jacobian.ncols();
    let mut product = DMatrix::<f32>::zeros(columns, columns);
    if columns == 0 {
        return product;
    }

    let peeled_mc3 = columns / 24 * 24;
    let peeled_mc2 = peeled_mc3 + (columns - peeled_mc3) / 16 * 16;
    let peeled_mc1 = peeled_mc2 + (columns - peeled_mc2) / 8 * 8;
    let peeled_mc_half = peeled_mc1 + (columns - peeled_mc1) / 4 * 4;
    let packet_columns = columns / 4 * 4;

    for row in 0..peeled_mc3 {
        for column in 0..columns {
            product[(row, column)] = eigen_q2_dot_fma_f32(jacobian, row, column);
        }
    }
    for row in peeled_mc3..peeled_mc2 {
        for column in 0..columns {
            product[(row, column)] = eigen_q2_dot_fma_f32(jacobian, row, column);
        }
    }
    for row in peeled_mc2..peeled_mc1 {
        for column in 0..columns {
            product[(row, column)] = if column < packet_columns {
                eigen_q2_dot_packet_parity_f32(jacobian, row, column)
            } else {
                eigen_q2_dot_fma_f32(jacobian, row, column)
            };
        }
    }
    for row in peeled_mc1..peeled_mc_half {
        for column in 0..columns {
            product[(row, column)] = if column < packet_columns {
                eigen_q2_dot_packet_parity_f32(jacobian, row, column)
            } else {
                eigen_q2_dot_fma_f32(jacobian, row, column)
            };
        }
    }
    for row in peeled_mc_half..columns {
        for column in 0..columns {
            product[(row, column)] = eigen_q2_dot_fma_f32(jacobian, row, column);
        }
    }
    product
}

// Per-landmark LM product, not the stacked Q2 exporter. Pinned Eigen's
// one/half-packet four-column panels split peeled depth into C/D, merge,
// then process the remaining depth. Keep other panel kernels unchanged.
fn eigen_visual_gram_packet_tail_f32(jacobian: &DMatrix<f32>) -> DMatrix<f32> {
    let columns = jacobian.ncols();
    let mut product = jacobian.transpose() * jacobian;
    let end24 = columns / 24 * 24;
    let end16 = end24 + (columns - end24) / 16 * 16;
    let end8 = end16 + (columns - end16) / 8 * 8;
    let end4 = end8 + (columns - end8) / 4 * 4;
    for row in end16..end4 {
        for col in 0..columns / 4 * 4 {
            product[(row, col)] = eigen_q2_dot_packet_parity_f32(jacobian, row, col);
        }
    }
    product
}

#[inline]
fn eigen_q2_packet_dot_fma_f32(
    jacobian: &DMatrix<f32>,
    rhs: &DVector<f32>,
    row: usize,
    peeled_depth: usize,
) -> f32 {
    let mut lanes = [0.0_f32; 8];
    for depth in (0..peeled_depth).step_by(8) {
        for lane in 0..8 {
            lanes[lane] = jacobian[(depth + lane, row)].mul_add(rhs[depth + lane], lanes[lane]);
        }
    }
    eigen_predux8_f32(lanes)
}

/// Reproduce Eigen's row-major `general_matrix_vector_product` for Q2
/// `Jᵀr`.  Eight-, four-, and two-row output groups use Packet8 accumulators
/// followed by the AVX horizontal reduction and an explicitly non-fused
/// scalar depth tail.  A final one-row group additionally uses Packet4 for
/// the remaining complete four-depth block, exactly as Eigen's `HasHalf`
/// branch does.
#[inline]
fn eigen_q2_transpose_gemv_f32(jacobian: &DMatrix<f32>, rhs: &DVector<f32>) -> DVector<f32> {
    assert_eq!(jacobian.nrows(), rhs.len());
    let rows = jacobian.ncols();
    let depth = jacobian.nrows();
    let peeled_depth = depth / 8 * 8;
    let half_depth = depth / 4 * 4;
    let mut result = DVector::<f32>::zeros(rows);

    let mut output_row = 0;
    while output_row + 8 <= rows {
        for row in output_row..output_row + 8 {
            let mut value = eigen_q2_packet_dot_fma_f32(jacobian, rhs, row, peeled_depth);
            for depth in peeled_depth..jacobian.nrows() {
                value = add_f32_exact(value, mul_f32_exact(jacobian[(depth, row)], rhs[depth]));
            }
            result[row] = value;
        }
        output_row += 8;
    }
    while output_row + 4 <= rows {
        for row in output_row..output_row + 4 {
            let mut value = eigen_q2_packet_dot_fma_f32(jacobian, rhs, row, peeled_depth);
            for depth in peeled_depth..jacobian.nrows() {
                value = add_f32_exact(value, mul_f32_exact(jacobian[(depth, row)], rhs[depth]));
            }
            result[row] = value;
        }
        output_row += 4;
    }
    while output_row + 2 <= rows {
        for row in output_row..output_row + 2 {
            let mut value = eigen_q2_packet_dot_fma_f32(jacobian, rhs, row, peeled_depth);
            for depth in peeled_depth..jacobian.nrows() {
                value = add_f32_exact(value, mul_f32_exact(jacobian[(depth, row)], rhs[depth]));
            }
            result[row] = value;
        }
        output_row += 2;
    }
    while output_row < rows {
        let mut value = eigen_q2_packet_dot_fma_f32(jacobian, rhs, output_row, peeled_depth);
        if half_depth > peeled_depth {
            let mut lanes = [0.0_f32; 4];
            for depth in (peeled_depth..half_depth).step_by(4) {
                for lane in 0..4 {
                    lanes[lane] = jacobian[(depth + lane, output_row)]
                        .mul_add(rhs[depth + lane], lanes[lane]);
                }
            }
            value = add_f32_exact(value, eigen_predux4_f32(lanes));
        }
        for depth in half_depth..jacobian.nrows() {
            value = add_f32_exact(
                value,
                mul_f32_exact(jacobian[(depth, output_row)], rhs[depth]),
            );
        }
        result[output_row] = value;
        output_row += 1;
    }
    result
}

/// Accumulate `J.transpose() * J` in the source ABS_QR order.  Only an
/// explicitly tagged nine-row IMU block uses the currently pinned local
/// packet schedule. All other shapes—including a nine-row prior—use the
/// generic dynamic path. The 15-row IMU+bias product is intentionally routed
/// through that generic path until its Eigen packet tree has its own oracle.
#[inline]
fn accumulate_gram_f32_eigen(
    accumulator: &mut DMatrix<f32>,
    jacobian: &DMatrix<f32>,
    exact_imu_rows: bool,
) {
    assert_eq!(accumulator.nrows(), accumulator.ncols());
    assert_eq!(accumulator.ncols(), jacobian.ncols());
    if !exact_imu_rows || jacobian.nrows() != 9 {
        *accumulator += jacobian.transpose() * jacobian;
        return;
    }
    for row in 0..jacobian.ncols() {
        for column in 0..jacobian.ncols() {
            let mut value = 0.0_f32;
            for depth in 0..9 {
                value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
            }
            accumulator[(row, column)] += value;
        }
    }
}

/// Accumulate a stored square-root prior's compact normal product and scatter
/// it into the absolute AOM columns.  `jacobian` is the global-width view
/// carried by the factor, while `compact_state_columns` identifies the
/// columns that were present in upstream's compact `MargLinData::H`.
#[inline]
fn accumulate_prior_gram_f32_eigen(
    accumulator: &mut DMatrix<f32>,
    jacobian: &DMatrix<f32>,
    compact_state_columns: Option<&[usize]>,
) {
    assert_eq!(accumulator.nrows(), accumulator.ncols());
    assert_eq!(accumulator.ncols(), jacobian.ncols());
    let Some(columns) = compact_state_columns else {
        *accumulator += jacobian.transpose() * jacobian;
        return;
    };
    if columns.len() != jacobian.nrows()
        || columns.iter().any(|&column| column >= accumulator.ncols())
    {
        // Keep malformed metadata from changing the behavior of legacy
        // callers.  The active-window prior builder only attaches a complete
        // compact-to-global map.
        *accumulator += jacobian.transpose() * jacobian;
        return;
    }

    // Materialize the compact column-major MatrixXf that Eigen receives from
    // MargLinData.  This is intentionally a separate product: multiplying
    // the zero-padded AOM view changes the packet/tail traversal even though
    // all omitted entries are zero.
    let compact = DMatrix::<f32>::from_fn(jacobian.nrows(), columns.len(), |row, column| {
        jacobian[(row, columns[column])]
    });
    let compact_product = eigen_prior_compact_gram_f32(&compact);
    for local_row in 0..columns.len() {
        let global_row = columns[local_row];
        for local_column in 0..columns.len() {
            accumulator[(global_row, columns[local_column])] +=
                compact_product[(local_row, local_column)];
        }
    }
}

/// Reproduce Eigen's AVX2 `gebp` tail for the 21x21 compact prior product.
/// The main 16-row block is already equivalent to the established dynamic
/// product.  Rows 16..19 use the Packet4 half-kernel's even/odd accumulators;
/// row 20 uses the swapped 2-lane/4-column kernel, except for the final
/// scalar column (column 20), which takes the ordinary 1x1 path.
#[inline]
fn eigen_prior_packet_tail_gram_candidate_f32(j: &DMatrix<f32>) -> DMatrix<f32> {
    let columns = j.ncols();
    let end24 = columns / 24 * 24;
    let end16 = end24 + (columns - end24) / 16 * 16;
    let end8 = end16 + (columns - end16) / 8 * 8;
    let end4 = end8 + (columns - end8) / 4 * 4;
    let depth8 = j.nrows() / 8 * 8;
    let depth2 = j.nrows() / 2 * 2;
    let mut product = j.transpose() * j;
    for row in end16..end4 {
        for col in 0..columns / 4 * 4 {
            product[(row, col)] = eigen_q2_dot_packet_parity_f32(j, row, col);
        }
    }
    for row in end4..columns {
        for col in 0..columns / 4 * 4 {
            let mut accum = [[0.0_f32; 2]; 4];
            for depth in 0..depth8 {
                let group = (depth & 7) / 2;
                let lane = depth & 1;
                accum[group][lane] = j[(depth, row)].mul_add(j[(depth, col)], accum[group][lane]);
            }
            let mut low = (accum[0][0] + accum[1][0]) + (accum[2][0] + accum[3][0]);
            let mut high = (accum[0][1] + accum[1][1]) + (accum[2][1] + accum[3][1]);
            for depth in (depth8..depth2).step_by(2) {
                low = j[(depth, row)].mul_add(j[(depth, col)], low);
                high = j[(depth + 1, row)].mul_add(j[(depth + 1, col)], high);
            }
            let mut value = low + high;
            for depth in depth2..j.nrows() {
                value = j[(depth, row)].mul_add(j[(depth, col)], value);
            }
            product[(row, col)] = value;
        }
    }
    product
}

fn eigen_prior_compact_gram_f32(compact: &DMatrix<f32>) -> DMatrix<f32> {
    if matches!(
        compact.shape(),
        (33, 33) | (39, 39) | (45, 45) | (51, 51) | (57, 57)
    ) {
        return eigen_prior_packet_tail_gram_candidate_f32(compact);
    }
    if compact.shape() == (27, 27) {
        return eigen_prior27_tail_gram_f32(compact);
    }
    let mut product = compact.transpose() * compact;
    if compact.nrows() != 21 || compact.ncols() != 21 {
        return product;
    }

    for row in 16..20 {
        // The four-column panel uses the half-packet two-accumulator
        // kernel.  The final scalar RHS column is dispatched to the
        // one-column remainder kernel below.
        for column in 0..20 {
            let mut even = 0.0_f32;
            let mut odd = 0.0_f32;
            for depth in 0..16 {
                if depth & 1 == 0 {
                    even = compact[(depth, row)].mul_add(compact[(depth, column)], even);
                } else {
                    odd = compact[(depth, row)].mul_add(compact[(depth, column)], odd);
                }
            }
            let mut value = even + odd;
            for depth in 16..21 {
                value = compact[(depth, row)].mul_add(compact[(depth, column)], value);
            }
            product[(row, column)] = value;
        }
        let mut scalar = 0.0_f32;
        for depth in 0..21 {
            scalar = compact[(depth, row)].mul_add(compact[(depth, 20)], scalar);
        }
        product[(row, 20)] = scalar;
    }

    for column in 0..20 {
        let mut accum = [[0.0_f32; 2]; 4];
        for depth in 0..16 {
            let group = (depth & 7) / 2;
            let lane = depth & 1;
            accum[group][lane] =
                compact[(depth, 20)].mul_add(compact[(depth, column)], accum[group][lane]);
        }
        let low = (accum[0][0] + accum[1][0]) + (accum[2][0] + accum[3][0]);
        let high = (accum[0][1] + accum[1][1]) + (accum[2][1] + accum[3][1]);
        let mut merged_low = compact[(16, 20)].mul_add(compact[(16, column)], low);
        let mut merged_high = compact[(17, 20)].mul_add(compact[(17, column)], high);
        merged_low = compact[(18, 20)].mul_add(compact[(18, column)], merged_low);
        merged_high = compact[(19, 20)].mul_add(compact[(19, column)], merged_high);
        product[(20, column)] =
            compact[(20, 20)].mul_add(compact[(20, column)], merged_low + merged_high);
    }
    // Column 20 is handled by Eigen's scalar 1x1 remainder kernel, not the
    // swapped four-column panel above.
    let mut scalar = 0.0_f32;
    for depth in 0..21 {
        scalar = compact[(depth, 20)].mul_add(compact[(depth, 20)], scalar);
    }
    product[(20, 20)] = scalar;
    product
}

// Pinned prior GEMM dispatch: 2cdee0 -> 2c62f0 -> gebp 29cba0.
// The 24-row body and scalar-column remainders retain the dynamic product;
// three residual rows use the paired-depth/four-column panel reduction.
fn eigen_prior27_tail_gram_f32(compact: &DMatrix<f32>) -> DMatrix<f32> {
    assert_eq!(compact.shape(), (27, 27));
    let mut product = compact.transpose() * compact;
    for row in 24..27 {
        for column in 0..24 {
            let mut accum = [[0.0_f32; 2]; 4];
            for depth in 0..24 {
                let group = (depth & 7) / 2;
                let lane = depth & 1;
                accum[group][lane] =
                    compact[(depth, row)].mul_add(compact[(depth, column)], accum[group][lane]);
            }
            let low = (accum[0][0] + accum[1][0]) + (accum[2][0] + accum[3][0]);
            let high = (accum[0][1] + accum[1][1]) + (accum[2][1] + accum[3][1]);
            let low = compact[(24, row)].mul_add(compact[(24, column)], low);
            let high = compact[(25, row)].mul_add(compact[(25, column)], high);
            product[(row, column)] = compact[(26, row)].mul_add(compact[(26, column)], low + high);
        }
    }
    product
}

/// Emit the exact compact prior factor fed to the f32 normal reducer.  This
/// opt-in sidecar is used only to separate a Jᵀr arithmetic mismatch from a
/// frame-5 residual/current-point producer mismatch; it never participates in
/// the reducer and is disabled unless an explicit path is supplied.
#[inline]
fn emit_prior_factor_input_diagnostic(
    jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
    compact_state_columns: Option<&[usize]>,
) {
    let Some(path) = crate::vio::window::diagnostic_env_snapshot()
        .diagnostic_prior_input
        .as_ref()
    else {
        return;
    };
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let compact_bits = compact_state_columns.map(|columns| {
        columns
            .iter()
            .flat_map(|&column| (0..jacobian.nrows()).map(move |row| jacobian[(row, column)]))
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>()
    });
    let record = json!({
        "schema": "basalt.m7im15_prior_factor_input.v1",
        "call": COUNT.fetch_add(1, Ordering::Relaxed),
        "rows": jacobian.nrows(),
        "global_cols": jacobian.ncols(),
        "columns": compact_state_columns,
        "jacobian_global_bits": jacobian
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
        "jacobian_compact_bits": compact_bits,
        "residual_bits": residual
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
    });
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{}", record);
    }
}

/// Accumulate a prior `H.transpose() * r` using the same compact-column map
/// as [`accumulate_prior_gram_f32_eigen`].  The current f32 dot schedule is
/// shared with the generic transpose-vector path; compact extraction keeps
/// the source shape explicit while preserving the established RHS bits until
/// its dedicated Eigen GEMV audit is complete.
#[inline]
fn accumulate_prior_transpose_vector_f32_eigen(
    accumulator: &mut DVector<f32>,
    jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
    compact_state_columns: Option<&[usize]>,
) {
    assert_eq!(jacobian.nrows(), residual.len());
    assert_eq!(jacobian.ncols(), accumulator.len());
    let Some(columns) = compact_state_columns else {
        accumulate_transpose_vector_f32_eigen(accumulator, jacobian, residual, false);
        return;
    };
    if columns.len() != jacobian.nrows()
        || columns.iter().any(|&column| column >= accumulator.len())
    {
        accumulate_transpose_vector_f32_eigen(accumulator, jacobian, residual, false);
        return;
    }
    let compact = DMatrix::<f32>::from_fn(jacobian.nrows(), columns.len(), |row, column| {
        jacobian[(row, columns[column])]
    });
    // `compact.transpose() * residual` is dispatched by Eigen's row-major
    // GeneralMatrixVector kernel.  Its output rows are reduced in 8/4/2
    // groups, with Packet8/Packet4 horizontal reductions followed by the
    // scalar depth tail.  The generic column-wise helper has the same
    // mathematical result but a different f32 association for this 21x21
    // prior, so keep the compact transpose explicit here.
    let compact_transpose = compact.transpose().into_owned();
    let compact_result = eigen_prior_row_major_gemv_f32(&compact_transpose, residual);
    for (local_column, &global_column) in columns.iter().enumerate() {
        accumulator[global_column] += compact_result[local_column];
    }
}

/// Evaluate Eigen's row-major GEMV used by a compact marginal prior.
///
/// The 21x21 prior has an audited Eigen tail schedule, so retain that exact
/// path.  Later windows carry larger compact priors (for example 27x27 at
/// MH01 frame 9); those shapes use the generic dynamic Eigen GEMV instead of
/// asserting the first window's dimensions.  Keeping the dispatch here makes
/// the 21x21 parity path immutable while allowing the prior state dimension to
/// grow normally during a long run.
#[inline]
fn eigen_prior_row_major_gemv_f32(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    if matrix.nrows() == 21 && matrix.ncols() == 21 && vector.len() == 21 {
        return eigen_prior_row_major_gemv_21(matrix, vector);
    }
    if matrix.shape() == (27, 27) && vector.len() == 27 {
        return eigen_prior_row_major_gemv_27(matrix, vector);
    }
    if matrix.shape() == (33, 33) && vector.len() == 33 {
        return eigen_prior_row_major_gemv_33(matrix, vector);
    }
    if matrix.shape() == (39, 39) && vector.len() == 39 {
        return eigen_prior_row_major_gemv_39_f32(matrix, vector);
    }
    if matrix.shape() == (45, 45) && vector.len() == 45 {
        return eigen_prior_row_major_gemv_45_f32(matrix, vector);
    }
    if matrix.shape() == (51, 51) && vector.len() == 51 {
        return eigen_prior_row_major_gemv_51_f32(matrix, vector);
    }
    if matrix.shape() == (57, 57) && vector.len() == 57 {
        return eigen_prior_row_major_gemv_57_f32(matrix, vector);
    }
    eigen_row_major_gemv_f32(matrix, vector)
}

// Pinned 390b40 GEMV: paired outputs use sequential four-product adds
// (391eb6..391f09), then three FMA tail updates (391f1c..391f83).
fn eigen_prior_row_major_gemv_39_f32(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.shape(), (39, 39));
    assert_eq!(vector.len(), 39);
    DVector::from_fn(39, |row, _| {
        let mut lanes = [0.0_f32; 8];
        for k in 0..32 {
            lanes[k % 8] = matrix[(row, k)].mul_add(vector[k], lanes[k % 8]);
        }
        let mut value = eigen_predux8_f32(lanes);
        if row == 38 {
            let half: [f32; 4] = std::array::from_fn(|k| matrix[(row, 32 + k)] * vector[32 + k]);
            value += (half[0] + half[2]) + (half[1] + half[3]);
        } else {
            for k in 32..36 {
                value += matrix[(row, k)] * vector[k];
            }
        }
        for k in 36..39 {
            if row >= 36 {
                value = matrix[(row, k)].mul_add(vector[k], value);
            } else {
                value += matrix[(row, k)] * vector[k];
            }
        }
        value
    })
}

/// Pinned Eigen 45x45 prior GEMV. Output rows 0..39 use
/// the Packet8 depth body followed by five scalar multiply/add updates; the
/// four-row output tail contracts those five updates, while the final scalar
/// output row consumes depths 40..43 as one Packet4 reduction and depth 44
/// as a scalar FMA.
fn eigen_prior_row_major_gemv_45_f32(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.shape(), (45, 45));
    assert_eq!(vector.len(), 45);
    DVector::from_fn(45, |row, _| {
        let mut lanes = [0.0_f32; 8];
        for depth in 0..40 {
            lanes[depth % 8] = matrix[(row, depth)].mul_add(vector[depth], lanes[depth % 8]);
        }
        let mut value = eigen_predux8_f32(lanes);
        if row == 44 {
            let half: [f32; 4] =
                std::array::from_fn(|k| mul_f32_exact(matrix[(row, 40 + k)], vector[40 + k]));
            value = add_f32_exact(
                value,
                add_f32_exact(
                    add_f32_exact(half[0], half[2]),
                    add_f32_exact(half[1], half[3]),
                ),
            );
            matrix[(row, 44)].mul_add(vector[44], value)
        } else if row >= 40 {
            for depth in 40..45 {
                value = add_f32_exact(value, mul_f32_exact(matrix[(row, depth)], vector[depth]));
            }
            value
        } else {
            for depth in 40..45 {
                value = add_f32_exact(value, mul_f32_exact(matrix[(row, depth)], vector[depth]));
            }
            value
        }
    })
}

/// Candidate for the pinned Eigen 51x51 compact-prior GEMV.  The first 48
/// depths form the Packet8 body.  Paired/scalar output tail rows 48..50
/// contract the final three depths, while complete output packets keep those
/// multiply/add operations separate.
fn eigen_prior_row_major_gemv_51_f32(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.shape(), (51, 51));
    assert_eq!(vector.len(), 51);
    DVector::from_fn(51, |row, _| {
        let mut lanes = [0.0_f32; 8];
        for depth in 0..48 {
            lanes[depth % 8] = matrix[(row, depth)].mul_add(vector[depth], lanes[depth % 8]);
        }
        let mut value = eigen_predux8_f32(lanes);
        for depth in 48..51 {
            value = if row >= 48 {
                matrix[(row, depth)].mul_add(vector[depth], value)
            } else {
                add_f32_exact(value, mul_f32_exact(matrix[(row, depth)], vector[depth]))
            };
        }
        value
    })
}

/// Diagnostic candidate for Eigen's 57x57 compact-prior GEMV. Seven complete
/// Packet8 depth groups feed the standard horizontal reduction; the single
/// scalar depth/output remainder is fused only for Eigen's scalar output row.
fn eigen_prior_row_major_gemv_57_f32(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.shape(), (57, 57));
    assert_eq!(vector.len(), 57);
    DVector::from_fn(57, |row, _| {
        let mut lanes = [0.0_f32; 8];
        for depth in 0..56 {
            lanes[depth % 8] = matrix[(row, depth)].mul_add(vector[depth], lanes[depth % 8]);
        }
        let value = eigen_predux8_f32(lanes);
        if row == 56 {
            matrix[(row, 56)].mul_add(vector[56], value)
        } else {
            add_f32_exact(value, mul_f32_exact(matrix[(row, 56)], vector[56]))
        }
    })
}

fn eigen_prior_row_major_gemv_33(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.shape(), (33, 33));
    assert_eq!(vector.len(), 33);
    let mut result = eigen_row_major_gemv_f32(matrix, vector);
    // Native scalar-output kernel: Packet8 redux 392070..39209e,
    // no Packet4 for depth33, then vfma231ss at392204.
    let mut lanes = [0.0_f32; 8];
    for depth in 0..32 {
        lanes[depth % 8] = matrix[(32, depth)].mul_add(vector[depth], lanes[depth % 8]);
    }
    result[32] = matrix[(32, 32)].mul_add(vector[32], eigen_predux8_f32(lanes));
    result
}

fn eigen_prior_row_major_gemv_27(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.shape(), (27, 27));
    assert_eq!(vector.len(), 27);
    let mut result = eigen_row_major_gemv_f32(matrix, vector);
    // Native 390b40: output pair 24/25 (391f1c..391f83), scalar
    // output 26 (392204..39223b) contract all three depth-tail updates.
    for row in 24..27 {
        let mut lanes = [0.0_f32; 8];
        for depth in 0..24 {
            lanes[depth % 8] = matrix[(row, depth)].mul_add(vector[depth], lanes[depth % 8]);
        }
        let q: [f32; 4] = std::array::from_fn(|k| lanes[k] + lanes[k + 4]);
        let mut value = (q[0] + q[2]) + (q[1] + q[3]);
        for depth in 24..27 {
            value = matrix[(row, depth)].mul_add(vector[depth], value);
        }
        result[row] = value;
    }
    result
}

/// Evaluate the audited Eigen row-major `21x21 * VectorXf` GEMV used by the
/// first compact marginal prior.  `GeneralMatrixVector.h` processes output
/// rows in 8/4/2 groups, but only the final scalar row receives the Packet4
/// depth tail: rows 0..19 use Packet8 depth 0..15 followed by scalar depths
/// 16..20, while row 20 uses Packet8 depth 0..15, Packet4 depth 16..19, then
/// scalar depth 20.  Keep the packet and scalar associations explicit; using
/// the landmark helper here would incorrectly apply Packet4 to every row.
#[inline]
fn eigen_prior_row_major_gemv_21(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.nrows(), 21);
    assert_eq!(matrix.ncols(), 21);
    assert_eq!(vector.len(), 21);
    let mut result = DVector::<f32>::zeros(21);

    for row in 0..20 {
        let mut lanes = [0.0_f32; 8];
        for depth in (0..16).step_by(8) {
            for lane in 0..8 {
                lanes[lane] =
                    matrix[(row, depth + lane)].mul_add(vector[depth + lane], lanes[lane]);
            }
        }
        let q0 = add_f32_exact(lanes[0], lanes[4]);
        let q1 = add_f32_exact(lanes[1], lanes[5]);
        let q2 = add_f32_exact(lanes[2], lanes[6]);
        let q3 = add_f32_exact(lanes[3], lanes[7]);
        let mut value = add_f32_exact(add_f32_exact(q0, q2), add_f32_exact(q1, q3));
        for depth in 16..21 {
            value = add_f32_exact(value, mul_f32_exact(matrix[(row, depth)], vector[depth]));
        }
        result[row] = value;
    }

    let row = 20;
    let mut lanes = [0.0_f32; 8];
    for depth in (0..16).step_by(8) {
        for lane in 0..8 {
            lanes[lane] = matrix[(row, depth + lane)].mul_add(vector[depth + lane], lanes[lane]);
        }
    }
    let q0 = add_f32_exact(lanes[0], lanes[4]);
    let q1 = add_f32_exact(lanes[1], lanes[5]);
    let q2 = add_f32_exact(lanes[2], lanes[6]);
    let q3 = add_f32_exact(lanes[3], lanes[7]);
    let mut value = add_f32_exact(add_f32_exact(q0, q2), add_f32_exact(q1, q3));

    let mut half = [0.0_f32; 4];
    for lane in 0..4 {
        half[lane] = matrix[(row, 16 + lane)].mul_add(vector[16 + lane], half[lane]);
    }
    let h0 = add_f32_exact(half[0], half[2]);
    let h1 = add_f32_exact(half[1], half[3]);
    value = add_f32_exact(value, add_f32_exact(h0, h1));
    // GCC contracts Eigen's scalar remainder at the final depth even though
    // the earlier scalar tail updates are ordinary multiply/add operations.
    value = matrix[(row, 20)].mul_add(vector[20], value);
    result[row] = value;
    result
}

/// Accumulate `J.transpose() * r` in the source ABS_QR order.
///
/// Eigen's `LandmarkBlockAbsDynamic::add_dense_H_b` uses a row-major Q2 block
/// and its packet GEMV evaluator.  For this transpose-times-vector shape the
/// pinned f32 result is the left-to-right row reduction with a fused multiply
/// add per row.  Keep the helper safe and local to the f32 reduction path;
/// f64 accumulation intentionally retains the existing nalgebra expression.
#[inline]
fn accumulate_transpose_vector_f32_eigen(
    accumulator: &mut DVector<f32>,
    jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
    exact_imu_rows: bool,
) {
    assert_eq!(jacobian.nrows(), residual.len());
    assert_eq!(jacobian.ncols(), accumulator.len());
    if exact_imu_rows && jacobian.nrows() == 9 {
        for column in 0..jacobian.ncols() {
            let lane0 = jacobian[(0, column)].mul_add(residual[0], 0.0_f32);
            let lane1 = jacobian[(1, column)].mul_add(residual[1], 0.0_f32);
            let lane2 = jacobian[(2, column)].mul_add(residual[2], 0.0_f32);
            let lane3 = jacobian[(3, column)].mul_add(residual[3], 0.0_f32);
            let lane4 = jacobian[(4, column)].mul_add(residual[4], 0.0_f32);
            let lane5 = jacobian[(5, column)].mul_add(residual[5], 0.0_f32);
            let lane6 = jacobian[(6, column)].mul_add(residual[6], 0.0_f32);
            let lane7 = jacobian[(7, column)].mul_add(residual[7], 0.0_f32);
            let q0 = lane0 + lane4;
            let q1 = lane1 + lane5;
            let q2 = lane2 + lane6;
            let q3 = lane3 + lane7;
            let packet_sum = (q0 + q2) + (q1 + q3);
            accumulator[column] += packet_sum + jacobian[(8, column)] * residual[8];
        }
        return;
    }
    for column in 0..jacobian.ncols() {
        let mut value = 0.0_f32;
        for row in 0..jacobian.nrows() {
            value = jacobian[(row, column)].mul_add(residual[row], value);
        }
        accumulator[column] += value;
    }
}

/// Evaluate Eigen's row-major dynamic MatrixXf-by-VectorXf GEMV for the
/// landmark Q1 block.  The pinned AVX2 kernel keeps one Packet8 accumulator
/// per output row across all complete eight-column packets, performs one
/// horizontal reduction, and only then visits the scalar tail.  Reducing
/// every packet immediately (the tempting scalar translation) changes the
/// association and produces different landmark increments.
#[inline]
fn eigen_row_major_gemv_f32(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.ncols(), vector.len());
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    let full_end = columns / 8 * 8;
    let half_end = columns / 4 * 4;
    let mut result = DVector::<f32>::zeros(rows);
    for row in 0..rows {
        let mut lanes = [0.0_f32; 8];
        let mut column = 0;
        while column < full_end {
            for lane in 0..8 {
                lanes[lane] =
                    matrix[(row, column + lane)].mul_add(vector[column + lane], lanes[lane]);
            }
            column += 8;
        }
        let q0 = lanes[0] + lanes[4];
        let q1 = lanes[1] + lanes[5];
        let q2 = lanes[2] + lanes[6];
        let q3 = lanes[3] + lanes[7];
        // Eigen's AVX2 Packet8f predux lowers the two half-packet sums as
        // `(q0 + q2) + (q1 + q3)` (the SSE movehl/movehdup tree).  Preserve
        // that association exactly; the alternative pairing changes f32
        // landmark increments.
        let mut value = (q0 + q2) + (q1 + q3);

        // On the pinned AVX2 Eigen build PacketSizeHalf/Quarter both have a
        // four-scalar lane width for f32.  Keep that packet tail explicit so
        // the helper remains faithful for state widths other than 75.
        let mut half_lanes = [0.0_f32; 4];
        while column < half_end {
            for lane in 0..4 {
                half_lanes[lane] =
                    matrix[(row, column + lane)].mul_add(vector[column + lane], half_lanes[lane]);
            }
            column += 4;
        }
        // Packet4f uses the same movehl/movehdup tree: (h0+h2)+(h1+h3).
        let h0 = half_lanes[0] + half_lanes[2];
        let h1 = half_lanes[1] + half_lanes[3];
        value += h0 + h1;
        while column < columns {
            value += matrix[(row, column)] * vector[column];
            column += 1;
        }
        result[row] = value;
    }
    result
}

/// Reproduce the scalar dot emitted by the pinned native visual
/// `backSubstitute` for `QJinc.dot(0.5 * QJinc + Qr)`. The first product is a
/// plain multiply; every remaining element is folded by one FMA. The caller
/// performs the final `l_diff -= dot` separately.
#[inline]
fn eigen_visual_model_dot_f32(increment: &DVector<f32>, residual: &DVector<f32>) -> f32 {
    assert_eq!(increment.len(), residual.len());
    if increment.is_empty() {
        return 0.0_f32;
    }
    let first_right = 0.5_f32.mul_add(increment[0], residual[0]);
    let mut value = increment[0] * first_right;
    for index in 1..increment.len() {
        let right = 0.5_f32.mul_add(increment[index], residual[index]);
        value = increment[index].mul_add(right, value);
    }
    value
}

/// Recover a landmark increment from a compact Q1/R payload produced by one
/// [`LandmarkHouseholderF32::factor`] call.  The arithmetic intentionally
/// mirrors `back_substitute_landmark_f32_with_track`: the same row-major
/// GEMV, positive-Q1 RHS, and pinned 3x3 triangular schedule are used.  The
/// compact payload is consumed by the clean solver's one-shot preparation.
pub(crate) fn back_substitute_landmark_compact_f32(
    data: &CompactLandmarkBackSubstitutionF32,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    back_substitute_compact_flat_f32(
        data.landmark_cols,
        data.state_cols,
        &data.storage,
        data.rank,
        data.eligible,
        state_step,
        tolerance,
    )
}

fn back_substitute_landmark_compact_entry_f32(
    data: &CompactLandmarkBackSubstitutionEntryF32,
    arena: &[f32],
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    let end = data.storage_offset.checked_add(data.storage_len()?)?;
    let storage = arena.get(data.storage_offset..end)?;
    back_substitute_compact_flat_f32(
        data.landmark_cols,
        data.state_cols,
        storage,
        data.rank,
        data.eligible,
        state_step,
        tolerance,
    )
}

fn back_substitute_compact_flat_f32(
    n: usize,
    state_cols: usize,
    storage: &[f32],
    rank: usize,
    eligible: bool,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    let expected_storage_len = checked_compact_storage_len(n, state_cols)?;
    if !eligible
        || n == 0
        || n > 3
        || rank < n
        || state_cols != state_step.len()
        || storage.len() != expected_storage_len
    {
        return None;
    }

    // This is the same Eigen row-major Packet8/Packet4/scalar association as
    // `eigen_row_major_gemv_f32`, expressed directly over the compact flat
    // storage.  Keeping the fixed three-lane result avoids materializing a
    // temporary DMatrix/DVector for every landmark while preserving every
    // f32 multiply/add and its order.
    let q1_state_step = eigen_compact_q1_state_gemv_f32(state_cols, n, storage, state_step);
    let mut rhs = [0.0_f32; 3];
    for row in 0..n {
        let residual_offset = n * state_cols;
        rhs[row] = storage[residual_offset + row] + q1_state_step[row];
    }
    let threshold = tolerance as f32;
    let mut increment = [0.0_f32; 3];
    let upper_r_offset = n * state_cols + n;
    if n == 3 {
        // Keep this operation order identical to the existing native-f32
        // recovery path.  In particular, row 1 uses one FMA and row 0 folds
        // r02*x2 into r01*x1 with an FMA before the final subtraction.
        let d2 = storage[upper_r_offset + 2 * n + 2];
        let d1 = storage[upper_r_offset + n + 1];
        let d0 = storage[upper_r_offset];
        if d2.abs() <= threshold || d1.abs() <= threshold || d0.abs() <= threshold {
            return None;
        }
        let x2 = rhs[2] / d2;
        let row1 = (-storage[upper_r_offset + n + 2]).mul_add(x2, rhs[1]);
        let x1 = row1 / d1;
        let row0_dot = storage[upper_r_offset + 2].mul_add(x2, storage[upper_r_offset + 1] * x1);
        let row0_numerator = rhs[0] - row0_dot;
        let x0 = row0_numerator / d0;
        increment[0] = -x0;
        increment[1] = -x1;
        increment[2] = -x2;
    } else {
        // Retain the small generic path used by synthetic one/two-column
        // tests; production visual blocks use three landmark columns.
        let mut solved = [0.0_f32; 3];
        for row in (0..n).rev() {
            let mut value = rhs[row];
            let mut row_product = 0.0_f32;
            for column in (row + 1)..n {
                row_product = add_f32_exact(
                    row_product,
                    storage[upper_r_offset + row * n + column] * solved[column],
                );
            }
            value -= row_product;
            let diagonal = storage[upper_r_offset + row * n + row];
            if diagonal.abs() <= threshold {
                return None;
            }
            solved[row] = value / diagonal;
        }
        for row in 0..n {
            increment[row] = -solved[row];
        }
    }
    Some(DVector::from_iterator(
        n,
        increment[..n].iter().copied().map(f64::from),
    ))
}

/// Flat-storage equivalent of [`eigen_row_major_gemv_f32`] for the compact
/// Q1 state rows.  The result is fixed-size because visual landmark blocks
/// have at most three columns; this keeps the per-factor recovery path free
/// of temporary nalgebra allocations.
#[inline]
fn eigen_compact_q1_state_gemv_f32(
    state_cols: usize,
    landmark_cols: usize,
    storage: &[f32],
    state_step: &DVector<f64>,
) -> [f32; 3] {
    let columns = state_cols;
    let full_end = columns / 8 * 8;
    let half_end = columns / 4 * 4;
    let mut result = [0.0_f32; 3];
    for row in 0..landmark_cols {
        let mut lanes = [0.0_f32; 8];
        let mut column = 0;
        while column < full_end {
            for lane in 0..8 {
                lanes[lane] = storage[row * state_cols + column + lane]
                    .mul_add(state_step[column + lane] as f32, lanes[lane]);
            }
            column += 8;
        }
        let q0 = lanes[0] + lanes[4];
        let q1 = lanes[1] + lanes[5];
        let q2 = lanes[2] + lanes[6];
        let q3 = lanes[3] + lanes[7];
        let mut value = (q0 + q2) + (q1 + q3);

        let mut half_lanes = [0.0_f32; 4];
        while column < half_end {
            for lane in 0..4 {
                half_lanes[lane] = storage[row * state_cols + column + lane]
                    .mul_add(state_step[column + lane] as f32, half_lanes[lane]);
            }
            column += 4;
        }
        let h0 = half_lanes[0] + half_lanes[2];
        let h1 = half_lanes[1] + half_lanes[3];
        value += h0 + h1;
        while column < columns {
            value += storage[row * state_cols + column] * state_step[column] as f32;
            column += 1;
        }
        result[row] = value;
    }
    result
}

fn back_substitute_landmark_f32(
    data: &LandmarkBackSubstitution,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    back_substitute_landmark_f32_with_track(data, state_step, tolerance, None)
}

fn back_substitute_landmark_f32_with_track(
    data: &LandmarkBackSubstitution,
    state_step: &DVector<f64>,
    tolerance: f64,
    track_id: Option<u64>,
) -> Option<DVector<f64>> {
    let state = as_f32_matrix(&data.state_jacobian);
    let landmark = as_f32_matrix(&data.landmark_jacobian);
    let residual = as_f32_vector(&data.residual);
    let step = as_f32_vector(state_step);
    if data.rank < data.landmark_jacobian.ncols() || landmark.norm() <= tolerance as f32 {
        return None;
    }
    // Upstream transforms the complete row-major [Jp | Jl | r] storage once,
    // then forms Q1r + Q1Jp * pose_inc in source order.  Factoring only Jl
    // with a pre-combined rhs is algebraically equivalent but changes the f32
    // operation path and the resulting landmark increment.
    let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual)?;
    let transformed_state = qr.transformed_state();
    let transformed_rhs = qr.transformed_residual();
    let r = qr.upper_r();
    let n = landmark.ncols();
    // Keep the positive Q1 right-hand side until after the triangular solve.
    // Eigen evaluates `-Q1Jl.solve(rhs)` as a solve followed by a vector
    // negation.  Negating `rhs` first is algebraically equivalent, but it
    // changes the f32 operation path at each triangular update.
    let rhs = transformed_rhs.rows(0, n).into_owned();
    let q1_state_step = eigen_row_major_gemv_f32(&transformed_state.rows(0, n).into_owned(), &step);
    emit_landmark_backsub_probe(
        track_id,
        &step,
        &transformed_state.rows(0, n).into_owned(),
        &transformed_rhs.rows(0, n).into_owned(),
        &q1_state_step,
        &r,
    );
    let mut rhs = rhs;
    rhs += q1_state_step;
    let mut increment = DVector::<f32>::zeros(n);
    if n == 3 {
        // Pinned Eigen's 3x3 dynamic triangular kernel uses one FMA for the
        // row-1 update.  Row 0 first multiplies r01*x1, then folds r02*x2
        // into that product with an FMA, and only then subtracts the complete
        // dot product from the RHS.  The association is observable in the
        // f32 landmark increment, so do not turn this into two nested FMAs.
        let d2 = r[(2, 2)];
        let d1 = r[(1, 1)];
        let d0 = r[(0, 0)];
        if d2.abs() <= tolerance as f32
            || d1.abs() <= tolerance as f32
            || d0.abs() <= tolerance as f32
        {
            return None;
        }
        let x2 = rhs[2] / d2;
        let row1 = (-r[(1, 2)]).mul_add(x2, rhs[1]);
        let x1 = row1 / d1;
        let row0_dot = r[(0, 2)].mul_add(x2, r[(0, 1)] * x1);
        let row0_numerator = rhs[0] - row0_dot;
        let x0 = row0_numerator / d0;
        increment[0] = -x0;
        increment[1] = -x1;
        increment[2] = -x2;
    } else {
        // Keep the generic fallback for unit-test dimensions other than the
        // production 3x3 landmark block.
        let mut solved = DVector::<f32>::zeros(n);
        for row in (0..n).rev() {
            let mut value = rhs[row];
            let mut row_product = 0.0_f32;
            for column in (row + 1)..n {
                row_product = add_f32_exact(row_product, r[(row, column)] * solved[column]);
            }
            value -= row_product;
            let diagonal = r[(row, row)];
            if diagonal.abs() <= tolerance as f32 {
                return None;
            }
            solved[row] = value / diagonal;
        }
        increment = -solved;
    }
    Some(as_f64_vector(&increment))
}

pub(crate) fn back_substitute_landmark_upstream_f32_with_track(
    factor: &WhitenedFactorRowStack,
    state_step: &DVector<f64>,
    tolerance: f64,
    track_id: Option<u64>,
) -> Option<DVector<f64>> {
    let reduced = reduce_landmark_factors_f32_checked_without_visual_prefix_trace(
        std::slice::from_ref(factor),
        factor.state_jacobian.ncols(),
        tolerance,
    )
    .unwrap_or_else(|error| panic!("invalid f32 landmark reduction factors: {error:?}"));
    let data = reduced.back_substitution.first()?;
    back_substitute_landmark_f32_with_track(data, state_step, tolerance, track_id)
}

pub(crate) fn back_substitute_landmark_upstream_f32(
    factor: &WhitenedFactorRowStack,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    back_substitute_landmark_upstream_f32_with_track(factor, state_step, tolerance, None)
}

/// Recover a landmark increment after the reduced state increment is known.
pub fn back_substitute_landmark(
    data: &LandmarkBackSubstitution,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    // The linearized residual convention is r + J_s * dx + J_l * dl = 0.
    // Solve the landmark block for the negative residual after the state
    // increment has been applied.
    let rhs = -(&data.residual + &data.state_jacobian * state_step);
    let landmark_factor = WhitenedFactorRowStack {
        state_jacobian: data.state_jacobian.clone(),
        landmark_jacobian: data.landmark_jacobian.clone(),
        residual: data.residual.clone(),
        objective_cost: 0.0,
        kind: FactorKind::Generic,
        imu_link_offsets: None,
        prior_state_columns: None,
        landmark_metadata: None,
        imu_input_diagnostic: None,
        visual_observation_ids: None,
    };
    let (augmented_state, augmented_landmark, augmented_residual) =
        augmented_landmark_rows(&landmark_factor);
    let mut augmented_rhs = DVector::zeros(augmented_residual.len());
    augmented_rhs.rows_mut(0, rhs.len()).copy_from(&rhs);
    let qr = augmented_landmark.qr();
    if data.rank < data.landmark_jacobian.ncols() {
        return None;
    }
    if data.landmark_jacobian.norm() <= tolerance {
        return None;
    }

    // `nalgebra::QR::solve` intentionally only accepts square systems.  The
    // upstream block solves the first three rows of the Householder-transformed
    // rectangular stack, i.e. R * dl = -(Qᵀ(r + Jp dx))[:3].  Reproduce that
    // triangular backsolve instead of switching to an SVD least-squares path.
    let mut transformed_rhs =
        DMatrix::from_column_slice(augmented_state.nrows(), 1, augmented_rhs.as_slice());
    qr.q_tr_mul(&mut transformed_rhs);
    let r = qr.r();
    let n = data.landmark_jacobian.ncols();
    let mut increment = DVector::zeros(n);
    for row in (0..n).rev() {
        let mut value = transformed_rhs[(row, 0)];
        for column in (row + 1)..n {
            value -= r[(row, column)] * increment[column];
        }
        let diagonal = r[(row, row)];
        if diagonal.abs() <= tolerance {
            return None;
        }
        increment[row] = value / diagonal;
    }
    Some(increment)
}

// Trial computeError's DS visitor uses a different expression schedule from
// the linearized factor objective. Keep this separate until trial wiring.
pub(crate) fn upstream_trial_bearing_f32(direction: Vector2<f32>) -> Vector3<f32> {
    let scale = 2.0_f32 / (direction.x.mul_add(direction.x, direction.y * direction.y) + 1.0_f32);
    Vector3::new(direction.x * scale, direction.y * scale, scale - 1.0_f32)
}

/// Coordinate arithmetic only: the trial evaluator must apply the native
/// camera-domain validity check separately. This does not accept/reject rows.
/// Do not substitute this schedule into the linearization/Jacobian path.
pub(crate) fn upstream_trial_projection_coordinates_f32(
    camera: &DoubleSphereCamera,
    point: Vector3<f32>,
) -> Vector2<f32> {
    let r2 = point.x * point.x + point.y * point.y;
    let d1 = point.z.mul_add(point.z, r2).sqrt();
    let k = (camera.xi as f32).mul_add(d1, point.z);
    let d2 = k.mul_add(k, r2).sqrt();
    let alpha = camera.alpha as f32;
    let denominator = alpha.mul_add(d2, (1.0_f32 - alpha) * k);
    Vector2::new(
        (camera.fx as f32).mul_add(point.x / denominator, camera.cx as f32),
        (camera.fy as f32).mul_add(point.y / denominator, camera.cy as f32),
    )
}

/// Native `linearizePoint` checks finite projected coordinates and the strict
/// DS domain inequality, without image bounds or an epsilon depth cutoff.
pub(crate) fn upstream_trial_project_f32(
    camera: &DoubleSphereCamera,
    point: Vector3<f32>,
) -> Option<Vector2<f32>> {
    let projected = upstream_trial_projection_coordinates_f32(camera, point);
    let r2 = point.x * point.x + point.y * point.y;
    let d1 = point.z.mul_add(point.z, r2).sqrt();
    let alpha = camera.alpha as f32;
    let xi = camera.xi as f32;
    let w1 = if alpha > 0.5_f32 {
        (1.0_f32 - alpha) / alpha
    } else {
        alpha / (1.0_f32 - alpha)
    };
    let w2 = (w1 + xi) / ((w1 + w1).mul_add(xi, xi * xi) + 1.0_f32).sqrt();
    if projected.iter().all(|value| value.is_finite()) && point.z > (-w2) * d1 {
        Some(projected)
    } else {
        None
    }
}

pub(crate) fn upstream_trial_visual_cost_f32(x: f32, y: f32, sigma: f32, huber: f32) -> f32 {
    let norm = (y.mul_add(y, x * x) + 0.0_f32).sqrt();
    let weight = if norm < huber { 1.0_f32 } else { huber / norm };
    let coefficient = (weight / (sigma * sigma)) * (0.5_f32 * (2.0_f32 - weight));
    (coefficient * y).mul_add(y, (coefficient * x) * x)
}

/// Trial computeError constructs one matrix per host/target TimeCamId pair.
/// Callers must supply getPose-equivalent current endpoints, not FEJ poses.
pub(crate) fn upstream_trial_transform_f32(
    host: &SE3,
    target: &SE3,
    host_camera: &SE3,
    target_camera: &SE3,
    same_time_cam_id: bool,
) -> SMatrix<f32, 4, 4> {
    if same_time_cam_id {
        return SMatrix::identity();
    }
    let result = upstream_trial_transform_stages_f32(
        F32Pose::from_se3(host),
        F32Pose::from_se3(target),
        F32Pose::from_se3(host_camera),
        F32Pose::from_se3(target_camera),
    )
    .4;
    eigen_homogeneous_transform_f32(result.rotation, result.translation)
}

/// Evaluate a trial observation without constructing linearization Jacobians.
/// Invalid projection is omitted by computeError; no outlier threshold is
/// applied to its cost. Parameters are native [stereographic u, v, rho].
pub(crate) fn upstream_trial_observation_f32(
    camera: &DoubleSphereCamera,
    transform: SMatrix<f32, 4, 4>,
    parameters: Vector3<f32>,
    pixel: Vector2<f32>,
    config: FactorConfig,
) -> Option<(Vector2<f32>, f32)> {
    let bearing = upstream_trial_bearing_f32(parameters.fixed_rows::<2>(0).into_owned());
    let point = eigen_homogeneous_point_product_f32(transform, bearing, parameters.z);
    let projected = upstream_trial_project_f32(camera, point.fixed_rows::<3>(0).into_owned())?;
    let raw = projected - pixel;
    let cost = upstream_trial_visual_cost_f32(
        raw.x,
        raw.y,
        config.observation_stddev as f32,
        config.huber_delta as f32,
    );
    Some((raw, cost))
}

/// The pinned native trial computeRelPose clone uses a packet difference
/// rotation before the generic camera actions. Keep this separate from the
/// linearization transform: their f32 instruction schedules are different.
fn upstream_trial_transform_stages_f32(
    host: F32Pose,
    target: F32Pose,
    host_camera: F32Pose,
    target_camera: F32Pose,
) -> (F32Pose, UnitQuaternion<f32>, F32Pose, F32Pose, F32Pose) {
    let camera_inverse = target_camera.inverse();
    let inverse = sophus_so3_inverse(target.rotation);
    let relative = F32Pose {
        rotation: sophus_quat_product_current_packet_f32(inverse, host.rotation),
        translation: sophus_rotate_difference_f32(inverse, host.translation, target.translation),
    };
    let prefix = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(camera_inverse.rotation, relative.rotation),
        translation: sophus_rotate_f32(camera_inverse.rotation, relative.translation)
            + camera_inverse.translation,
    };
    let result = F32Pose {
        rotation: sophus_quat_product_camera_suffix_f32(prefix.rotation, host_camera.rotation),
        translation: sophus_rotate_f32(prefix.rotation, host_camera.translation)
            + prefix.translation,
    };
    (camera_inverse, inverse, relative, prefix, result)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires M11_VISUAL_CAPTURE_ROOT pinned frame17 capture"]
    fn frame17_visual_packet_tail_native_full_matrix() {
        let root = std::path::PathBuf::from(std::env::var("M11_VISUAL_CAPTURE_ROOT").unwrap());
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("factor_storage.json")).unwrap())
                .unwrap();
        let native = std::fs::read(root.join("frame17_visual_total_iter0.bin"))
            .or_else(|_| {
                // Historical call26 capture is labelled iter0; dense_call above
                // binds its actual iteration rather than trusting the filename.
                std::fs::read(root.join("frame7_visual_total_iter0.bin"))
            })
            .unwrap();
        let width = u64::from_le_bytes(native[0..8].try_into().unwrap()) as usize;
        assert!(matches!(
            (meta["dense_call"].as_u64().unwrap(), width),
            (100, 63) | (25, 51) | (26, 51)
        ));
        let factors = meta["factors"].as_array().unwrap();
        assert_eq!(factors.len(), meta["count"].as_u64().unwrap() as usize);
        let mut old = DMatrix::<f32>::zeros(width, width);
        let mut candidate = old.clone();
        for factor in factors {
            let rows = factor["rows"].as_u64().unwrap() as usize;
            let cols = factor["cols"].as_u64().unwrap() as usize;
            let used = factor["num_rows"].as_u64().unwrap() as usize;
            assert_eq!(factor["padding_idx"].as_u64(), Some(width as u64));
            let bytes = std::fs::read(root.join(factor["storage"].as_str().unwrap())).unwrap();
            assert_eq!(bytes.len(), rows * cols * 4);
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            let j = DMatrix::from_fn(used - 3, width, |r, c| values[(r + 3) * cols + c]);
            old += j.transpose() * &j;
            candidate += eigen_visual_gram_packet_tail_f32(&j);
        }
        assert_eq!(native.len(), 24 + (width * width + width) * 4);
        for offset in [0, 8, 16] {
            assert_eq!(
                u64::from_le_bytes(native[offset..offset + 8].try_into().unwrap()),
                width as u64
            );
        }
        let expected: Vec<u32> = native[24..24 + width * width * 4]
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        if width == 63 {
            assert_eq!(
                old.iter()
                    .zip(&expected)
                    .filter(|(x, y)| x.to_bits() == **y)
                    .count(),
                3864
            );
        }
        for (i, (actual, expected)) in candidate.iter().zip(expected).enumerate() {
            assert_eq!(actual.to_bits(), expected, "lane {i}");
        }
    }

    #[test]
    #[ignore = "requires M11_PRIOR_CAPTURE_ROOT pinned external capture"]
    fn frame10_prior27_native_gram_and_normal_rhs() {
        let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
        let read = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .lines()
                .map(|x| serde_json::from_str(x).unwrap())
                .collect()
        };
        let inputs = read("prior_inputs.jsonl");
        let stages = read("prior_stages.jsonl");
        assert_eq!(inputs.len(), 8);
        assert_eq!(stages.len(), 32);
        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
                .collect()
        };
        for (iteration, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64().unwrap(), iteration as u64);
            assert_eq!(input["rows"].as_u64().unwrap(), 27);
            assert_eq!(input["cols"].as_u64().unwrap(), 27);
            let j = DMatrix::from_column_slice(27, 27, &parse(&input["jacobian_bits"]));
            let gram = stages
                .iter()
                .find(|x| {
                    x["iteration"].as_u64() == Some(iteration as u64)
                        && x["stage"].as_str() == Some("gram")
                })
                .unwrap();
            let expected = parse(&gram["bits"]);
            assert_eq!(expected.len(), 729);
            let actual = eigen_prior_compact_gram_f32(&j);
            for i in 0..729 {
                assert_eq!(
                    actual.as_slice()[i].to_bits(),
                    expected[i].to_bits(),
                    "iteration {iteration}, lane {i}"
                );
            }
            let old = j.transpose() * &j;
            assert_eq!(
                old.iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                41
            );
            let stage = |name: &str| -> Vec<f32> {
                parse(
                    &stages
                        .iter()
                        .find(|x| {
                            x["iteration"].as_u64() == Some(iteration as u64)
                                && x["stage"].as_str() == Some(name)
                        })
                        .unwrap()["bits"],
                )
            };
            let residual = DVector::from_vec(stage("adjusted_rhs"));
            let normal_expected = stage("normal_rhs");
            let jt = j.transpose().into_owned();
            let normal = eigen_prior_row_major_gemv_f32(&jt, &residual);
            for lane in 0..27 {
                assert_eq!(
                    normal[lane].to_bits(),
                    normal_expected[lane].to_bits(),
                    "normal RHS iteration {iteration} lane {lane}"
                );
            }
            if iteration == 0 {
                let old_normal = eigen_row_major_gemv_f32(&jt, &residual);
                assert_eq!(
                    old_normal
                        .iter()
                        .zip(&normal_expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count(),
                    3
                );
            }
        }
    }
    #[test]
    #[ignore = "requires validated external native prior capture"]
    fn prior_packet_tail_candidate_native_gram() {
        let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
        let read = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .lines()
                .map(|x| serde_json::from_str(x).unwrap())
                .collect()
        };
        let inputs = read("prior_inputs.jsonl");
        let stages = read("prior_stages.jsonl");
        assert!(!inputs.is_empty());
        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
                .collect()
        };
        for (i, input) in inputs.iter().enumerate() {
            let size = input["rows"].as_u64().unwrap() as usize;
            assert_eq!(input["cols"].as_u64(), Some(size as u64));
            assert!([21, 27, 33, 39].contains(&size));
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            let j = DMatrix::from_column_slice(size, size, &parse(&input["jacobian_bits"]));
            let gram = stages
                .iter()
                .find(|s| {
                    s["iteration"].as_u64() == Some(i as u64) && s["stage"].as_str() == Some("gram")
                })
                .unwrap();
            let expected = parse(&gram["bits"]);
            assert_eq!(expected.len(), size * size);
            let actual = eigen_prior_packet_tail_gram_candidate_f32(&j);
            for (lane, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "size {size} iteration {i} lane {lane}"
                );
            }
            if size == 33 {
                let old = j.transpose() * &j;
                assert_eq!(
                    old.iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count(),
                    191
                );
            }
            if size == 39 {
                let production = eigen_prior_compact_gram_f32(&j);
                assert!(production
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
                let old = j.transpose() * &j;
                assert_eq!(
                    old.iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count(),
                    339
                );
            }
        }
    }

    #[test]
    #[ignore = "requires external frame31 prior input and native stage capture"]
    fn m11_frame31_prior45_candidate_probe() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME31_PRIOR_ROOT").expect("frame31 capture root"),
        );
        let input_path =
            root.join("m11_trial_wired_frame10_20260908/r34_frame31_prior/prior_inputs.jsonl");
        let native = root.join("m11_native_frame31_strict_capture_20260912/r1");
        let records: Vec<serde_json::Value> = std::fs::read_to_string(input_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|record: &serde_json::Value| {
                record["rows"].as_u64() == Some(45) && record["global_cols"].as_u64() == Some(75)
            })
            .collect();
        assert_eq!(records.len(), 8);
        let native_prior_stages: Vec<serde_json::Value> = std::fs::read_to_string(
            root.join("m11_native_frame31_prior_stages_20260912/r1/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        let native_model_parts: Vec<serde_json::Value> = std::fs::read_to_string(
            root.join("m11_native_frame31_model_parts_20260912/r1/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        let bits = |value: &serde_json::Value| -> Vec<f32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap())
                })
                .collect()
        };
        let dense = |path: &std::path::Path| -> (DMatrix<f32>, DVector<f32>) {
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(bytes.len(), 22_824);
            assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 75);
            assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 75);
            assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 75);
            let values: Vec<f32> = bytes[24..]
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect();
            (
                DMatrix::from_column_slice(75, 75, &values[..75 * 75]),
                DVector::from_column_slice(&values[75 * 75..]),
            )
        };
        for (iteration, record) in records.iter().enumerate() {
            assert_eq!(record["call"].as_u64(), Some((209 + iteration) as u64));
            let columns: Vec<usize> = record["columns"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(columns.len(), 45);
            let j = DMatrix::from_column_slice(45, 45, &bits(&record["jacobian_compact_bits"]));
            let residual = DVector::from_vec(bits(&record["residual_bits"]));
            let (before_h, before_b) =
                dense(&native.join(format!("frame31_prior_before_iter{iteration}.bin")));
            let (expected_h, expected_b) =
                dense(&native.join(format!("frame31_prior_after_iter{iteration}.bin")));
            let mut candidate_h = before_h.clone();
            let gram = eigen_prior_packet_tail_gram_candidate_f32(&j);
            for local_row in 0..45 {
                for local_column in 0..45 {
                    candidate_h[(columns[local_row], columns[local_column])] +=
                        gram[(local_row, local_column)];
                }
            }
            let mut candidate_b = before_b.clone();
            let transpose = j.transpose().into_owned();
            let normal = eigen_prior_row_major_gemv_45_f32(&transpose, &residual);
            for (local, &global) in columns.iter().enumerate() {
                candidate_b[global] += normal[local];
            }
            let h_exact = candidate_h
                .iter()
                .zip(expected_h.iter())
                .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
                .count();
            let b_exact = candidate_b
                .iter()
                .zip(expected_b.iter())
                .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
                .count();
            let native_stage = |name: &str| {
                native_prior_stages
                    .iter()
                    .find(|entry| {
                        entry["kind"].as_str() == Some("prior_stage")
                            && entry["stage"].as_str() == Some(name)
                            && entry["iteration"].as_u64() == Some(iteration as u64)
                    })
                    .unwrap()
            };
            let native_adjusted = DVector::from_vec(bits(&native_stage("adjusted_rhs")["bits"]));
            let expected_normal = bits(&native_stage("normal_rhs")["bits"]);
            let native_normal = eigen_prior_row_major_gemv_45_f32(&transpose, &native_adjusted);
            let normal_exact = native_normal
                .iter()
                .zip(&expected_normal)
                .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
                .count();
            let normal_mismatch = native_normal
                .iter()
                .zip(&expected_normal)
                .enumerate()
                .filter(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
                .map(|(lane, (actual, expected))| {
                    format!("{lane}:{:08x}/{:08x}", actual.to_bits(), expected.to_bits())
                })
                .collect::<Vec<_>>();
            let step_bytes =
                std::fs::read(native.join(format!("frame31_inc_entry_iter{iteration}.f32")))
                    .unwrap();
            assert_eq!(step_bytes.len(), 75 * 4);
            let global_step: Vec<f32> = step_bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect();
            let compact_step =
                DVector::from_iterator(45, columns.iter().map(|&column| global_step[column]));
            let model = prior45_model_f32(&j, &native_adjusted, &compact_step);
            let expected_model = native_model_parts
                .iter()
                .find(|entry| {
                    entry["kind"].as_str() == Some("model_part")
                        && entry["stage"].as_str() == Some("prior")
                        && entry["iteration"].as_u64() == Some(iteration as u64)
                })
                .unwrap()["f32_bits"]
                .as_str()
                .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
                .unwrap();
            println!(
                "M11_FRAME31_PRIOR45 iter={iteration} H={h_exact}/5625 b={b_exact}/75 normal={normal_exact}/45 model={:08x}/{:08x} mismatch={normal_mismatch:?}",
                model.to_bits(),
                expected_model.to_bits(),
            );
            assert_eq!(
                model.to_bits(),
                expected_model.to_bits(),
                "model iteration {iteration}"
            );
            let mut global_j = DMatrix::<f64>::zeros(45, 75);
            for (local_column, &global_column) in columns.iter().enumerate() {
                for row in 0..45 {
                    global_j[(row, global_column)] = j[(row, local_column)] as f64;
                }
            }
            let factor = WhitenedFactorRowStack::new(
                global_j,
                DMatrix::zeros(45, 0),
                native_adjusted.map(|value| value as f64),
            )
            .unwrap()
            .with_kind(FactorKind::Prior)
            .with_prior_state_columns(columns.clone());
            let dispatched = model_cost_decrease_f32(
                &[factor],
                &DVector::from_iterator(75, global_step.iter().map(|&value| value as f64)),
                1e-10,
            )
            .unwrap() as f32;
            assert_eq!(
                dispatched.to_bits(),
                expected_model.to_bits(),
                "production dispatch iteration {iteration}"
            );
        }
    }

    #[test]
    #[ignore = "requires validated native frame38 prior capture"]
    fn m11_frame38_prior51_gram_and_normal_candidate() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME38_PRIOR_ROOT").expect("frame38 capture root"),
        );
        let records: Vec<serde_json::Value> = std::fs::read_to_string(
            root.join("m11_native_frame38_strict_capture_20260913/r2/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        let bits = |value: &serde_json::Value| -> Vec<f32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap())
                })
                .collect()
        };
        for iteration in 0..8 {
            let input = records
                .iter()
                .find(|record| {
                    record["kind"].as_str() == Some("prior_input")
                        && record["iteration"].as_u64() == Some(iteration)
                })
                .unwrap();
            let stage = |name: &str| {
                records
                    .iter()
                    .find(|record| {
                        record["kind"].as_str() == Some("prior_stage")
                            && record["stage"].as_str() == Some(name)
                            && record["iteration"].as_u64() == Some(iteration)
                    })
                    .unwrap()
            };
            let j = DMatrix::from_column_slice(51, 51, &bits(&input["jacobian_bits"]));
            let gram_expected = bits(&stage("gram")["bits"]);
            let gram = eigen_prior_packet_tail_gram_candidate_f32(&j);
            for lane in 0..51 * 51 {
                assert_eq!(
                    gram.as_slice()[lane].to_bits(),
                    gram_expected[lane].to_bits(),
                    "Gram iteration {iteration} lane {lane}"
                );
            }

            let adjusted = DVector::from_vec(bits(&stage("adjusted_rhs")["bits"]));
            let normal_expected = bits(&stage("normal_rhs")["bits"]);
            let normal = eigen_prior_row_major_gemv_51_f32(&j.transpose(), &adjusted);
            for lane in 0..51 {
                assert_eq!(
                    normal[lane].to_bits(),
                    normal_expected[lane].to_bits(),
                    "normal RHS iteration {iteration} lane {lane}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires validated native frame45 prior capture"]
    fn m11_frame45_prior57_gram_and_normal_candidate() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME45_PRIOR_ROOT").expect("frame45 capture root"),
        );
        let records: Vec<serde_json::Value> = std::fs::read_to_string(
            root.join("m11_native_frame45_strict_capture_20260913/r1/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        let bits = |value: &serde_json::Value| -> Vec<f32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap())
                })
                .collect()
        };
        for iteration in 0..8 {
            let input = records
                .iter()
                .find(|record| {
                    record["kind"].as_str() == Some("prior_input")
                        && record["iteration"].as_u64() == Some(iteration)
                })
                .unwrap();
            let stage = |name: &str| {
                records
                    .iter()
                    .find(|record| {
                        record["kind"].as_str() == Some("prior_stage")
                            && record["stage"].as_str() == Some(name)
                            && record["iteration"].as_u64() == Some(iteration)
                    })
                    .unwrap()
            };
            let j = DMatrix::from_column_slice(57, 57, &bits(&input["jacobian_bits"]));
            let gram_expected = bits(&stage("gram")["bits"]);
            let gram = eigen_prior_packet_tail_gram_candidate_f32(&j);
            for lane in 0..57 * 57 {
                assert_eq!(
                    gram.as_slice()[lane].to_bits(),
                    gram_expected[lane].to_bits(),
                    "Gram iteration {iteration} lane {lane}"
                );
            }

            let adjusted = DVector::from_vec(bits(&stage("adjusted_rhs")["bits"]));
            let normal_expected = bits(&stage("normal_rhs")["bits"]);
            let normal = eigen_prior_row_major_gemv_57_f32(&j.transpose(), &adjusted);
            for lane in 0..57 {
                assert_eq!(
                    normal[lane].to_bits(),
                    normal_expected[lane].to_bits(),
                    "normal RHS iteration {iteration} lane {lane}"
                );
            }
        }
    }
    #[test]
    #[ignore = "requires validated external native frame17 prior capture"]
    fn frame17_prior33_normal_rhs_candidate() {
        let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
        let read = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .lines()
                .map(|x| serde_json::from_str(x).unwrap())
                .collect()
        };
        let inputs = read("prior_inputs.jsonl");
        let stages = read("prior_stages.jsonl");
        assert_eq!((inputs.len(), stages.len()), (8, 32));
        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
                .collect()
        };
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            assert_eq!(
                (input["rows"].as_u64(), input["cols"].as_u64()),
                (Some(33), Some(33))
            );
            let j = DMatrix::from_column_slice(33, 33, &parse(&input["jacobian_bits"]));
            let stage = |name: &str| -> Vec<f32> {
                parse(
                    &stages
                        .iter()
                        .find(|s| {
                            s["iteration"].as_u64() == Some(i as u64)
                                && s["stage"].as_str() == Some(name)
                        })
                        .unwrap()["bits"],
                )
            };
            let r = DVector::from_vec(stage("adjusted_rhs"));
            let expected = stage("normal_rhs");
            let jt = j.transpose().into_owned();
            let actual = eigen_prior_row_major_gemv_33(&jt, &r);
            for lane in 0..33 {
                assert_eq!(
                    actual[lane].to_bits(),
                    expected[lane].to_bits(),
                    "iteration {i} lane {lane}"
                );
            }
            let old = eigen_row_major_gemv_f32(&jt, &r);
            assert_eq!(
                old.iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                usize::from([0, 5, 6, 7].contains(&i))
            );
        }
    }
    #[test]
    #[ignore = "requires validated native trial computeRelPose capture on E"]
    fn frame24_prior39_model_same_inputs_candidate() {
        let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
        let input_root = root.join("m11_native_frame24_prior_stages_20260909/r1");
        let model_root = root.join("m11_native_frame24_model_parts_20260909/r1");
        let records = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(input_root.join(name))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        };
        let inputs = records("prior_inputs.jsonl");
        let stages = records("prior_stages.jsonl");
        let oracle: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_root.join("model_parts_validated.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(oracle["status"], "PASS");
        assert_eq!(inputs.len(), 8);
        let parse = |x: &serde_json::Value| {
            f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap())
        };
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            assert_eq!(input["rows"].as_u64(), Some(39));
            assert_eq!(input["cols"].as_u64(), Some(39));
            let j = DMatrix::from_iterator(
                39,
                39,
                input["jacobian_bits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(&parse),
            );
            let matching: Vec<_> = stages
                .iter()
                .filter(|x| {
                    x["iteration"].as_u64() == Some(i as u64) && x["stage"] == "adjusted_rhs"
                })
                .collect();
            assert_eq!(matching.len(), 1);
            let rhs = DVector::from_iterator(
                39,
                matching[0]["bits"].as_array().unwrap().iter().map(&parse),
            );
            let name = format!("frame24_inc_entry_iter{i}.f32");
            let bytes = std::fs::read(input_root.join(&name)).unwrap();
            assert_eq!(bytes, std::fs::read(model_root.join(&name)).unwrap());
            assert_eq!(bytes.len(), 69 * 4);
            let step = DVector::from_iterator(
                39,
                bytes
                    .chunks_exact(4)
                    .take(39)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap())),
            );
            let actual = prior39_model_f32(&j, &rhs, &step);
            let expected = parse(&oracle["iterations"][i]["prior"]);
            println!(
                "M11_PRIOR39_MODEL iter={i} native={:08x} candidate={:08x}",
                expected.to_bits(),
                actual.to_bits()
            );
            assert_eq!(actual.to_bits(), expected.to_bits(), "iteration {i}");
            let mut global_j = DMatrix::<f64>::zeros(39, 69);
            global_j.columns_mut(0, 39).copy_from(&j.map(|x| x as f64));
            let factor =
                WhitenedFactorRowStack::new(global_j, DMatrix::zeros(39, 0), rhs.map(|x| x as f64))
                    .unwrap()
                    .with_kind(FactorKind::Prior)
                    .with_prior_state_columns((0..39).collect());
            let global_step = DVector::from_iterator(
                69,
                bytes
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64),
            );
            let dispatched =
                model_cost_decrease_f32(&[factor], &global_step, 1e-10).unwrap() as f32;
            assert_eq!(
                dispatched.to_bits(),
                expected.to_bits(),
                "production dispatch iteration {i}"
            );
        }
    }

    #[test]
    #[ignore = "requires validated native frame24 visual capture and r28 detail on E"]
    fn frame24_track1259_model_qr_boundary_probe() {
        use std::io::BufRead;
        let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
        let native = root.join("m11_native_frame24_visual_model_callsite_20260909/r2");
        for gate in [
            "engine.rc",
            "gdb.wrapper.rc",
            "probe.validation.rc",
            "binding.pre_post.cmp.rc",
        ] {
            assert_eq!(
                std::fs::read_to_string(native.join(gate)).unwrap().trim(),
                "0"
            );
        }
        let input = std::fs::File::open(
            root.join("m11_trial_wired_frame10_20260908/r28_frame24/detail_iterations.jsonl"),
        )
        .unwrap();
        let mut seen = [false; 8];
        for line in std::io::BufReader::new(input).lines() {
            let record: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            if record["frame_id"] != 24 || record["phase"] != "iteration_start" {
                continue;
            }
            let i = record["iteration"].as_u64().unwrap() as usize;
            assert!(i < 8 && !seen[i]);
            seen[i] = true;
            let factors: Vec<_> = record["landmark_factors"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|f| f["track_id"] == 1259)
                .collect();
            assert_eq!(factors.len(), 1);
            let f = factors[0];
            let matrix = |key: &str, cols: usize| {
                DMatrix::<f32>::from_fn(10, cols, |r, c| f[key][r][c].as_f64().unwrap() as f32)
            };
            let state = matrix("state_jacobian", 69);
            let landmark = matrix("landmark_jacobian", 3);
            let residual = DVector::from_iterator(
                10,
                f["residual"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32),
            );
            let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual).unwrap();
            let bytes =
                std::fs::read(native.join(format!("visual_iter{i}_ordinal0_storage.f32"))).unwrap();
            assert_eq!(bytes.len(), 13 * 76 * 4);
            let words: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let transformed = qr.transformed_state();
            let qres = qr.transformed_residual();
            let upper = qr.upper_r();
            let mut mismatch = Vec::new();
            for row in 0..10 {
                for col in 0..69 {
                    if transformed[(row, col)].to_bits() != words[row * 76 + col].to_bits() {
                        mismatch.push(format!(
                            "state[{row},{col}] native={:08x} rust={:08x}",
                            words[row * 76 + col].to_bits(),
                            transformed[(row, col)].to_bits()
                        ));
                    }
                }
                if qres[row].to_bits() != words[row * 76 + 75].to_bits() {
                    mismatch.push(format!("residual[{row}]"));
                }
            }
            for row in 0..3 {
                for col in row..3 {
                    if upper[(row, col)].to_bits() != words[row * 76 + 72 + col].to_bits() {
                        mismatch.push(format!("R[{row},{col}]"));
                    }
                }
            }
            println!(
                "M11_TRACK1259_QR iter={i} mismatch={} first={:?}",
                mismatch.len(),
                mismatch.first()
            );
            assert!(mismatch.is_empty(), "QR boundary iteration {i}");
            let step_bytes =
                std::fs::read(native.join(format!("frame24_inc_entry_iter{i}.f32"))).unwrap();
            assert_eq!(step_bytes.len(), 69 * 4);
            let step = DVector::from_iterator(
                69,
                step_bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
            );
            let blocks: Vec<serde_json::Value> =
                std::fs::read_to_string(native.join("visual_model_blocks.jsonl"))
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str(l).unwrap())
                    .collect();
            let expected = blocks
                .iter()
                .find(|b| b["iteration"] == i && b["ordinal"] == 0)
                .unwrap();
            assert_eq!(expected["before_bits"], "00000000");
            let expected =
                u32::from_str_radix(expected["after_bits"].as_str().unwrap(), 16).unwrap();
            for mode in 0..16 {
                let mut inc = if mode & 1 != 0 {
                    eigen_row_major_gemv_f32(&transformed, &step)
                } else {
                    &transformed * &step
                };
                let head = if mode & 2 != 0 {
                    eigen_row_major_gemv_f32(&transformed.rows(0, 3).into_owned(), &step)
                } else {
                    inc.rows(0, 3).into_owned()
                };
                let rhs = qres.rows(0, 3).into_owned() + head;
                let mut lm = DVector::<f32>::zeros(3);
                if mode & 4 != 0 {
                    let x2 = rhs[2] / upper[(2, 2)];
                    let x1 = (-upper[(1, 2)]).mul_add(x2, rhs[1]) / upper[(1, 1)];
                    let dot = upper[(0, 2)].mul_add(x2, upper[(0, 1)] * x1);
                    let x0 = (rhs[0] - dot) / upper[(0, 0)];
                    lm[0] = -x0;
                    lm[1] = -x1;
                    lm[2] = -x2;
                } else {
                    for row in (0..3).rev() {
                        let mut value = -rhs[row];
                        for col in row + 1..3 {
                            value -= upper[(row, col)] * lm[col];
                        }
                        lm[row] = value / upper[(row, row)];
                    }
                }
                let extra = &upper * lm;
                for row in 0..3 {
                    inc[row] += extra[row];
                }
                let actual = if mode & 8 != 0 {
                    // Candidate only: 8-lane product reduction followed by scalar FMA tail.
                    let products: [f32; 8] =
                        std::array::from_fn(|k| -inc[k] * 0.5_f32.mul_add(inc[k], qres[k]));
                    let half: [f32; 4] = std::array::from_fn(|k| products[k] + products[k + 4]);
                    let mut sum = (half[0] + half[2]) + (half[1] + half[3]);
                    for k in 8..10 {
                        sum = (-inc[k]).mul_add(0.5_f32.mul_add(inc[k], qres[k]), sum);
                    }
                    sum
                } else {
                    -inc.dot(&(0.5_f32 * &inc + &qres))
                };
                println!("M11_TRACK1259_MODEL iter={i} mode={mode} native={expected:08x} actual={:08x} exact={}", actual.to_bits(), actual.to_bits()==expected);
            }
        }
        assert!(seen.into_iter().all(|x| x));
    }

    #[test]
    #[ignore = "requires validated native frame24 visual capture and r28 detail on E"]
    fn frame24_all_visual_model_packet_schedule_exact() {
        use std::collections::HashMap;
        use std::io::BufRead;

        let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
        let native = root.join("m11_native_frame24_visual_model_callsite_20260909/r2");
        for gate in [
            "engine.rc",
            "gdb.wrapper.rc",
            "probe.validation.rc",
            "binding.pre_post.cmp.rc",
        ] {
            assert_eq!(
                std::fs::read_to_string(native.join(gate)).unwrap().trim(),
                "0"
            );
        }

        let blocks: Vec<serde_json::Value> =
            std::fs::read_to_string(native.join("visual_model_blocks.jsonl"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
        assert_eq!(blocks.len(), 65 * 8);
        let mut by_iteration: [Vec<&serde_json::Value>; 8] = std::array::from_fn(|_| Vec::new());
        for block in &blocks {
            let iteration = block["iteration"].as_u64().unwrap() as usize;
            assert!(iteration < 8);
            by_iteration[iteration].push(block);
        }
        for entries in &mut by_iteration {
            entries.sort_by_key(|block| block["ordinal"].as_u64().unwrap());
            assert_eq!(entries.len(), 65);
        }

        let input = std::fs::File::open(
            root.join("m11_trial_wired_frame10_20260908/r28_frame24/detail_iterations.jsonl"),
        )
        .unwrap();
        let mut seen = [false; 8];
        for line in std::io::BufReader::new(input).lines() {
            let record: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            if record["frame_id"] != 24 || record["phase"] != "iteration_start" {
                continue;
            }
            let iteration = record["iteration"].as_u64().unwrap() as usize;
            assert!(iteration < 8 && !seen[iteration]);
            seen[iteration] = true;

            let factors: HashMap<(u64, u64, [u32; 2], u32), &serde_json::Value> = record
                ["landmark_factors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|factor| {
                    let direction: [u32; 2] = std::array::from_fn(|index| {
                        (factor["direction"][index].as_f64().unwrap() as f32).to_bits()
                    });
                    (
                        (
                            factor["host_timestamp_ns"].as_u64().unwrap(),
                            factor["host_cam"].as_u64().unwrap(),
                            direction,
                            (factor["rho"].as_f64().unwrap() as f32).to_bits(),
                        ),
                        factor,
                    )
                })
                .collect();
            assert_eq!(factors.len(), 65);

            let step_bytes =
                std::fs::read(native.join(format!("frame24_inc_entry_iter{iteration}.f32")))
                    .unwrap();
            assert_eq!(step_bytes.len(), 69 * 4);
            let step = DVector::from_iterator(
                69,
                step_bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())),
            );

            let mut accumulator = 0.0_f32;
            let mut production_factors = Vec::with_capacity(65);
            for block in &by_iteration[iteration] {
                let ordinal = block["ordinal"].as_u64().unwrap() as usize;
                let hex = |value: &serde_json::Value| {
                    u32::from_str_radix(value.as_str().unwrap(), 16).unwrap()
                };
                let key = (
                    block["host_timestamp_ns"].as_u64().unwrap(),
                    block["host_camera"].as_u64().unwrap(),
                    [
                        hex(&block["direction_bits"][0]),
                        hex(&block["direction_bits"][1]),
                    ],
                    hex(&block["inverse_distance_bits"]),
                );
                let factor = factors[&key];
                let rows = factor["residual"].as_array().unwrap().len();
                let matrix = |key: &str, columns: usize| {
                    DMatrix::<f32>::from_fn(rows, columns, |row, column| {
                        factor[key][row][column].as_f64().unwrap() as f32
                    })
                };
                let state = matrix("state_jacobian", 69);
                let landmark = matrix("landmark_jacobian", 3);
                let residual = DVector::from_iterator(
                    rows,
                    factor["residual"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_f64().unwrap() as f32),
                );
                production_factors.push(
                    WhitenedFactorRowStack::new(
                        state.map(|value| value as f64),
                        landmark.map(|value| value as f64),
                        residual.map(|value| value as f64),
                    )
                    .unwrap()
                    .with_kind(FactorKind::Visual),
                );
                let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual).unwrap();
                assert_eq!(
                    qr.pivots
                        .iter()
                        .filter(|pivot| pivot.abs() > 1.0e-10_f32)
                        .count(),
                    3,
                    "iteration {iteration} ordinal {ordinal} rank"
                );
                let transformed_state = qr.transformed_state();
                let transformed_residual = qr.transformed_residual();
                let upper = qr.upper_r();
                let storage_bytes = std::fs::read(native.join(format!(
                    "visual_iter{iteration}_ordinal{ordinal}_storage.f32"
                )))
                .unwrap();
                assert_eq!(storage_bytes.len(), (rows + 3) * 76 * 4);
                let storage: Vec<f32> = storage_bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect();
                for row in 0..rows {
                    for column in 0..69 {
                        assert_eq!(
                            transformed_state[(row, column)].to_bits(),
                            storage[row * 76 + column].to_bits(),
                            "iteration {iteration} ordinal {ordinal} QJ[{row},{column}]"
                        );
                    }
                    assert_eq!(
                        transformed_residual[row].to_bits(),
                        storage[row * 76 + 75].to_bits(),
                        "iteration {iteration} ordinal {ordinal} Qr[{row}]"
                    );
                }
                for row in 0..3 {
                    for column in row..3 {
                        assert_eq!(
                            upper[(row, column)].to_bits(),
                            storage[row * 76 + 72 + column].to_bits(),
                            "iteration {iteration} ordinal {ordinal} R[{row},{column}]"
                        );
                    }
                }
                let mut increment = eigen_row_major_gemv_f32(&transformed_state, &step);
                let rhs = transformed_residual.rows(0, 3).into_owned()
                    + increment.rows(0, 3).into_owned();
                let mut landmark_increment = DVector::<f32>::zeros(3);
                let x2 = rhs[2] / upper[(2, 2)];
                let x1 = (-upper[(1, 2)]).mul_add(x2, rhs[1]) / upper[(1, 1)];
                let upper_dot = upper[(0, 2)].mul_add(x2, upper[(0, 1)] * x1);
                let x0 = (rhs[0] - upper_dot) / upper[(0, 0)];
                landmark_increment[0] = -x0;
                landmark_increment[1] = -x1;
                landmark_increment[2] = -x2;
                let q1_increment = &upper * landmark_increment;
                for row in 0..3 {
                    increment[row] += q1_increment[row];
                }

                let before =
                    u32::from_str_radix(block["before_bits"].as_str().unwrap(), 16).unwrap();
                let after = u32::from_str_radix(block["after_bits"].as_str().unwrap(), 16).unwrap();
                assert_eq!(
                    accumulator.to_bits(),
                    before,
                    "iteration {iteration} ordinal {ordinal} before"
                );
                accumulator -= eigen_visual_model_dot_f32(&increment, &transformed_residual);
                assert_eq!(
                    accumulator.to_bits(),
                    after,
                    "iteration {iteration} ordinal {ordinal} after"
                );
            }
            let production_step = step.map(|value| value as f64);
            let production = model_cost_decrease_f32(
                &production_factors,
                &production_step,
                1.0e-10_f64,
            )
            .unwrap() as f32;
            assert_eq!(
                production.to_bits(),
                accumulator.to_bits(),
                "iteration {iteration} production visual total"
            );
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    #[ignore = "requires external pinned-native frame24 prior capture"]
    fn frame24_prior39_normal_rhs_candidate() {
        let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
        let read = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .lines()
                .map(|x| serde_json::from_str(x).unwrap())
                .collect()
        };
        let inputs = read("prior_inputs.jsonl");
        let stages = read("prior_stages.jsonl");
        assert_eq!((inputs.len(), stages.len()), (8, 32));
        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
                .collect()
        };
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            assert_eq!(
                (input["rows"].as_u64(), input["cols"].as_u64()),
                (Some(39), Some(39))
            );
            let j = DMatrix::from_column_slice(39, 39, &parse(&input["jacobian_bits"]));
            let stage = |name: &str| -> Vec<f32> {
                parse(
                    &stages
                        .iter()
                        .find(|s| {
                            s["iteration"].as_u64() == Some(i as u64)
                                && s["stage"].as_str() == Some(name)
                        })
                        .unwrap()["bits"],
                )
            };
            let r = DVector::from_vec(stage("adjusted_rhs"));
            let expected = stage("normal_rhs");
            let jt = j.transpose().into_owned();
            let actual = eigen_prior_row_major_gemv_39_f32(&jt, &r);
            for lane in 0..39 {
                assert_eq!(
                    actual[lane].to_bits(),
                    expected[lane].to_bits(),
                    "iteration {i} lane {lane}"
                );
            }
            let old = eigen_row_major_gemv_f32(&jt, &r);
            assert_eq!(
                old.iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                [3, 4, 3, 2, 3, 4, 2, 5][i]
            );
        }
    }
    #[test]
    #[ignore = "requires validated native trial computeRelPose capture on E"]
    fn m11_native_trial_full_transform_capture_exact() {
        use std::io::BufRead;
        let path = std::env::var("M11_NATIVE_RELPOSE_STAGES").unwrap();
        let mut counts = [[0_usize; 5]; 8];
        let mut calls = [0_usize; 8];
        let mut first = None;
        for line in std::io::BufReader::new(std::fs::File::open(path).unwrap()).lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            let word =
                |v: &serde_json::Value| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap();
            let pose = |v: &serde_json::Value| {
                let values = v
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| f32::from_bits(word(x)))
                    .collect::<Vec<_>>();
                assert_eq!(values.len(), 7);
                F32Pose {
                    rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                        values[3], values[0], values[1], values[2],
                    )),
                    translation: Vector3::new(values[4], values[5], values[6]),
                }
            };
            let host = pose(&row["inputs"]["host_imu"]);
            let target = pose(&row["inputs"]["target_imu"]);
            let host_camera = pose(&row["inputs"]["host_camera"]);
            let target_camera = pose(&row["inputs"]["target_camera"]);
            let (tmp2, inverse, relative, prefix, result) =
                super::upstream_trial_transform_stages_f32(
                    host,
                    target,
                    host_camera,
                    target_camera,
                );
            let stage1 = &row["stages"][0];
            let stage2 = &row["stages"][1];
            assert_eq!(stage1["elf_pc"], "394998");
            assert_eq!(stage2["elf_pc"], "394b07");
            let words =
                |v: &serde_json::Value| v.as_array().unwrap().iter().map(word).collect::<Vec<_>>();
            let stack1 = words(&stage1["stack_80"]);
            let stack2 = words(&stage2["stack_80"]);
            let mut relative_expected = words(&stage1["xmm"]["3"]);
            relative_expected.extend([stack1[12], stack1[13], word(&stage1["stack_210"][2])]);
            let mut prefix_expected = stack2[16..20].to_vec();
            prefix_expected.extend([word(&stage2["xmm"]["6"][0]), word(&stage2["xmm"]["7"][0])]);
            let prefix_values = visual_chain_pose_snapshot(prefix);
            let inverse_values = visual_chain_pose_snapshot(F32Pose {
                rotation: inverse,
                translation: Vector3::zeros(),
            });
            let actual = [
                visual_chain_pose_snapshot(tmp2).to_vec(),
                inverse_values[..4].to_vec(),
                visual_chain_pose_snapshot(relative).to_vec(),
                prefix_values[..4]
                    .iter()
                    .chain(prefix_values[5..7].iter())
                    .copied()
                    .collect(),
                visual_chain_pose_snapshot(result).to_vec(),
            ];
            let expected = [
                stack1[..7].to_vec(),
                words(&stage1["stack_180"])[..4].to_vec(),
                relative_expected,
                prefix_expected,
                words(&row["result"]),
            ];
            let iteration = row["iteration"].as_u64().unwrap() as usize;
            for stage in 0..5 {
                assert_eq!(actual[stage].len(), expected[stage].len());
                for lane in 0..actual[stage].len() {
                    if actual[stage][lane].to_bits() == expected[stage][lane] {
                        counts[iteration][stage] += 1;
                    } else if first.is_none() {
                        first = Some(
                            serde_json::json!({"iteration":iteration,"call":calls[iteration],"stage":stage,"lane":lane,
                            "native":format!("{:08x}",expected[stage][lane]),"rust":format!("{:08x}",actual[stage][lane].to_bits())}),
                        );
                    }
                }
            }
            calls[iteration] += 1;
        }
        eprintln!(
            "M11_TRIAL_TRANSFORM {}",
            serde_json::json!({"stages":["camera_inverse","target_inverse_q","relative","prefix_q_y_z","result"],"exact_lanes":counts,"calls":calls,"first_mismatch":first})
        );
        assert_eq!(calls, [13; 8]);
        assert_eq!(counts, [[91, 52, 91, 78, 91]; 8]);
    }

    #[test]
    fn m11_native_trial_projection_domain_matches_source() {
        let camera = DoubleSphereCamera::new(100.0, 100.0, 0.0, 0.0, 0.0, 1.0, 640, 480).unwrap();
        // alpha=1, xi=0 makes w2 exactly zero, so the z boundary is strict.
        assert!(super::upstream_trial_project_f32(&camera, Vector3::new(1.0, 0.0, 0.0)).is_none());
        assert!(
            super::upstream_trial_project_f32(&camera, Vector3::new(1.0, 0.0, 1e-20)).is_some()
        );
        assert!(
            super::upstream_trial_project_f32(&camera, Vector3::new(1.0, 0.0, -1e-20)).is_none()
        );
        assert!(super::upstream_trial_project_f32(&camera, Vector3::zeros()).is_none());
        assert!(
            super::upstream_trial_project_f32(&camera, Vector3::new(f32::NAN, 0.0, 1.0)).is_none()
        );
        let pinhole = DoubleSphereCamera::new(100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 640, 480).unwrap();
        assert!(
            super::upstream_trial_project_f32(&pinhole, Vector3::new(0.0, 0.0, 1e-12)).is_some()
        );
        let outside =
            super::upstream_trial_project_f32(&pinhole, Vector3::new(10.0, 0.0, 1.0)).unwrap();
        assert_eq!(outside.x, 1000.0);
    }

    #[test]
    #[ignore = "requires validated native point capture and matching Rust observation inputs on E"]
    fn m11_native_trial_point_and_projection_all_observations_exact() {
        use std::io::BufRead;
        let native = std::env::var("M11_NATIVE_VISUAL_OBSERVATIONS").unwrap();
        let detail = std::env::var("M11_RUST_TRIAL_DETAIL").unwrap();
        let calibration = std::env::var("M11_NATIVE_CALIBRATION").unwrap();
        let calibration: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(calibration).unwrap()).unwrap();
        let cameras = calibration["value0"]["intrinsics"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                assert_eq!(entry["camera_type"], "ds");
                let c = &entry["intrinsics"];
                let resolution = &calibration["value0"]["resolution"][index];
                DoubleSphereCamera::new(
                    c["fx"].as_f64().unwrap(),
                    c["fy"].as_f64().unwrap(),
                    c["cx"].as_f64().unwrap(),
                    c["cy"].as_f64().unwrap(),
                    c["xi"].as_f64().unwrap(),
                    c["alpha"].as_f64().unwrap(),
                    resolution[0].as_u64().unwrap() as u32,
                    resolution[1].as_u64().unwrap() as u32,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut pixels = std::collections::BTreeMap::new();
        for line in std::io::BufReader::new(std::fs::File::open(detail).unwrap()).lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            if row["phase"] != "trial" {
                continue;
            }
            assert_eq!(row["iteration"], 0);
            assert_eq!(
                row["trial_visual_costs"]["state_boundary"],
                "candidate_after_step"
            );
            for landmark in row["landmarks"].as_array().unwrap() {
                for observation in landmark["observations"].as_array().unwrap() {
                    let key = (
                        landmark["track_id"].as_u64().unwrap(),
                        observation["target_timestamp_ns"].as_u64().unwrap(),
                        observation["target_cam"].as_u64().unwrap(),
                    );
                    let pixel = Vector2::new(
                        observation["pixel"][0].as_f64().unwrap() as f32,
                        observation["pixel"][1].as_f64().unwrap() as f32,
                    );
                    assert!(pixels.insert(key, pixel).is_none());
                }
            }
            break;
        }
        assert_eq!(pixels.len(), 739);
        let mut counts = [0_usize; 8];
        for line in std::io::BufReader::new(std::fs::File::open(native).unwrap()).lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            let parse = |v: &serde_json::Value| {
                f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
            };
            let key = (
                row["track_id"].as_u64().unwrap(),
                row["target_timestamp_ns"].as_u64().unwrap(),
                row["target_cam"].as_u64().unwrap(),
            );
            let parameters = &row["landmark_parameter"];
            let bearing = super::upstream_trial_bearing_f32(Vector2::new(
                parse(&parameters[0]),
                parse(&parameters[1]),
            ));
            let matrix = SMatrix::<f32, 4, 4>::from_iterator(
                row["T_t_h_column_major"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(parse),
            );
            let point =
                super::eigen_homogeneous_point_product_f32(matrix, bearing, parse(&parameters[2]));
            for lane in 0..4 {
                assert_eq!(
                    point[lane].to_bits(),
                    parse(&row["point4_target"][lane]).to_bits(),
                    "point {key:?} lane{lane}"
                );
            }
            let (raw, cost) = super::upstream_trial_observation_f32(
                &cameras[key.2 as usize],
                matrix,
                Vector3::new(
                    parse(&parameters[0]),
                    parse(&parameters[1]),
                    parse(&parameters[2]),
                ),
                pixels[&key],
                FactorConfig::default(),
            )
            .expect("native captured observation is valid");
            assert_eq!(
                cost.to_bits(),
                parse(&row["objective"]).to_bits(),
                "cost {key:?}"
            );
            for lane in 0..2 {
                assert_eq!(
                    raw[lane].to_bits(),
                    parse(&row["raw_residual"][lane]).to_bits(),
                    "residual {key:?} lane{lane}"
                );
            }
            counts[row["iteration"].as_u64().unwrap() as usize] += 1;
        }
        assert_eq!(counts, [739; 8]);
    }

    #[test]
    #[ignore = "requires validated native frame8 observation capture on E"]
    fn m11_native_trial_visual_cost_all_observations_exact() {
        use std::io::BufRead;
        let path = std::env::var("M11_NATIVE_VISUAL_OBSERVATIONS")
            .expect("explicit validated native observation path required");
        let file = std::fs::File::open(path).unwrap();
        let mut counts = [0_usize; 8];
        for line in std::io::BufReader::new(file).lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            let iteration = row["iteration"].as_u64().unwrap() as usize;
            let word =
                |v: &serde_json::Value| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap();
            let x = f32::from_bits(word(&row["raw_residual"][0]));
            let y = f32::from_bits(word(&row["raw_residual"][1]));
            let actual = super::upstream_trial_visual_cost_f32(x, y, 0.5, 1.0);
            assert_eq!(
                actual.to_bits(),
                word(&row["objective"]),
                "iteration {iteration}, track {}",
                row["track_id"]
            );
            counts[iteration] += 1;
        }
        assert_eq!(counts, [739; 8]);
    }
    #[test]
    #[ignore = "requires external frame7 native chain fixture on E"]
    fn m11_frame7_current_fej_matrix_probe() {
        let root = std::env::var("M11_TRI_PROBE_ROOT").unwrap();
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!(
                "{root}/m11_native_frame7_iter1_relpose_20260908/r1/current_chain_fixture.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let calibration: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("../../target/euroc_ds_calib.json").unwrap(),
        )
        .unwrap();
        let parse = |v: &serde_json::Value| {
            let f =
                |i: usize| f32::from_bits(u32::from_str_radix(v[i].as_str().unwrap(), 16).unwrap());
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(f(3), f(0), f(1), f(2))),
                translation: Vector3::new(f(4), f(5), f(6)),
            }
        };
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 15);
        for case in cases {
            let chain = &case["rust_chain"];
            let current = parse(&chain["target_camera_from_anchor_camera_f32_bits"]);
            let prefix = parse(&chain["target_camera_from_anchor_imu_f32_bits"]);
            let fej = parse(&chain["target_camera_from_anchor_imu_fej_f32_bits"]);
            let host_cam = case["endpoint"][1].as_u64().unwrap() as usize;
            let e = &calibration["value0"]["T_imu_cam"][host_cam];
            let t = Vector3::new(
                e["px"].as_f64().unwrap() as f32,
                e["py"].as_f64().unwrap() as f32,
                e["pz"].as_f64().unwrap() as f32,
            );
            let alternative_t = sophus_rotate_f32(fej.rotation, t) + fej.translation;
            let native: Vec<u32> = case["native"]["T_t_h"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
                .collect();
            let rotation = eigen_quaternion_matrix_f32(current.rotation);
            let mut matrix = [0u32; 16];
            for c in 0..3 {
                for r in 0..3 {
                    matrix[c * 4 + r] = rotation[(r, c)].to_bits();
                }
            }
            for i in 0..3 {
                matrix[12 + i] = current.translation[i].to_bits();
            }
            matrix[15] = 1.0f32.to_bits();
            let current_exact = matrix.iter().zip(&native).filter(|(a, b)| a == b).count();
            for i in 0..3 {
                matrix[12 + i] = alternative_t[i].to_bits();
            }
            let alternative_exact = matrix.iter().zip(&native).filter(|(a, b)| a == b).count();
            println!(
                "M11_FEJ_MATRIX {}",
                serde_json::json!({"endpoint":case["endpoint"],"host":case["host_frame"],"target":case["target_frame"],"current_exact":current_exact,"fej_translation_only_exact":alternative_exact,"prefix_rotation_equal":prefix.rotation.coords.iter().zip(fej.rotation.coords.iter()).all(|(a,b)|a.to_bits()==b.to_bits())})
            );
        }
    }
    use super::*;
    use crate::vio::landmarks::StereographicDirection;
    use nalgebra::{UnitQuaternion, Vector2};
    use std::cell::Cell;
    fn factor(
        js: &[f64],
        jl: &[f64],
        r: &[f64],
        rows: usize,
        state: usize,
        ld: usize,
    ) -> WhitenedFactorRowStack {
        WhitenedFactorRowStack::new(
            DMatrix::from_row_slice(rows, state, js),
            DMatrix::from_row_slice(rows, ld, jl),
            DVector::from_column_slice(r),
        )
        .unwrap()
    }

    fn assert_f32_bits(actual: &[f32], expected: &[u32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&value, &bits)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                value.to_bits(),
                bits,
                "f32 lane {index}: got {:08x}, expected {bits:08x}",
                value.to_bits()
            );
        }
    }

    fn assert_matrix_f32_bitwise_equal(
        actual: &DMatrix<f32>,
        expected: &DMatrix<f32>,
        label: &str,
    ) {
        assert_eq!(actual.shape(), expected.shape(), "{label} shape");
        for (index, (&actual, &expected)) in actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label} lane {index}: got {:08x}, expected {:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    fn assert_vector_f32_bitwise_equal(
        actual: &DVector<f32>,
        expected: &DVector<f32>,
        label: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label} lane {index}: got {:08x}, expected {:08x}",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    fn assert_landmark_qr_bitwise_equal(
        actual: &LandmarkHouseholderF32,
        expected: &LandmarkHouseholderF32,
        label: &str,
    ) {
        assert_eq!(actual.rows, expected.rows, "{label} rows");
        assert_eq!(actual.state_cols, expected.state_cols, "{label} state cols");
        assert_eq!(
            actual.landmark_cols, expected.landmark_cols,
            "{label} landmark cols"
        );
        assert_eq!(
            actual.landmark_offset, expected.landmark_offset,
            "{label} landmark offset"
        );
        assert_eq!(
            actual.residual_offset, expected.residual_offset,
            "{label} residual offset"
        );
        assert_f32_bits(
            &actual.storage,
            &expected
                .storage
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
        assert_f32_bits(
            &actual.pivots,
            &expected
                .pivots
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
        assert_f32_bits(
            &actual.tau,
            &expected
                .tau
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
        let _ = label;
    }

    fn assert_compact_payload_bitwise_equal(
        materialized: &CompactLandmarkBackSubstitutionF32,
        arena: &[f32],
        entry: &CompactLandmarkBackSubstitutionEntryF32,
        label: &str,
    ) {
        assert_eq!(
            materialized.landmark_index, entry.landmark_index,
            "{label} landmark index"
        );
        assert_eq!(materialized.track_id, entry.track_id, "{label} track id");
        assert_eq!(
            materialized.state_cols, entry.state_cols,
            "{label} state cols"
        );
        assert_eq!(
            materialized.landmark_cols, entry.landmark_cols,
            "{label} landmark cols"
        );
        assert_eq!(materialized.rank, entry.rank, "{label} rank");
        assert_eq!(materialized.eligible, entry.eligible, "{label} eligible");
        let end = entry
            .storage_offset
            .checked_add(materialized.storage.len())
            .expect("compact payload offset overflow");
        assert!(end <= arena.len(), "{label} arena bounds");
        assert_f32_bits(
            &materialized.storage,
            &arena[entry.storage_offset..end]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    /// Test-only spelling of the native quaternion-to-matrix diagonal path.
    ///
    /// The pinned Eigen/Sophus codegen contracts the two diagonal products
    /// before subtracting from one (`fma(x, 2*x, yy/zz)`).  The production
    /// helper above intentionally remains unchanged until this candidate has
    /// passed every captured relative-pose fixture.  The inverse is supplied
    /// by `sophus_so3_inverse`, whose packet implementation already performs
    /// the native pairwise normalization.
    fn m7_candidate_quaternion_matrix_f32(rotation: UnitQuaternion<f32>) -> Matrix3<f32> {
        let q = rotation.quaternion();
        let tx = 2.0_f32 * q.i;
        let ty = 2.0_f32 * q.j;
        let tz = 2.0_f32 * q.k;
        let txy = ty * q.i;
        let txz = tz * q.i;
        let tyy = ty * q.j;
        let tyz = tz * q.j;
        let tzz = tz * q.k;
        Matrix3::new(
            1.0_f32 - (tyy + tzz),
            (-tz).mul_add(q.w, txy),
            ty.mul_add(q.w, txz),
            tz.mul_add(q.w, txy),
            1.0_f32 - q.i.mul_add(tx, tzz),
            (-tx).mul_add(q.w, tyz),
            (-ty).mul_add(q.w, txz),
            tx.mul_add(q.w, tyz),
            1.0_f32 - q.i.mul_add(tx, tyy),
        )
    }

    /// Test-only candidate retaining the production signed-left-operand
    /// boundary and increasing-k six-term Eigen reduction.
    fn m7_candidate_adjoint_times_rotation_blocks_f32(
        pose: F32Pose,
        rotation: Matrix3<f32>,
        left_sign: f32,
    ) -> SMatrix<f32, 6, 6> {
        let pose_rotation = m7_candidate_quaternion_matrix_f32(pose.rotation);
        let cross_rotation =
            eigen_matrix_product_3x3_f32(skew3_f32(pose.translation), pose_rotation);
        let mut adjoint = SMatrix::<f32, 6, 6>::zeros();
        adjoint
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&pose_rotation);
        adjoint
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&cross_rotation);
        adjoint
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&pose_rotation);
        let mut rotation_blocks = SMatrix::<f32, 6, 6>::zeros();
        rotation_blocks
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation);
        rotation_blocks
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&rotation);
        eigen_matrix_product_6x6_f32(adjoint * left_sign, rotation_blocks)
    }

    fn tagged_imu_bias_pair(
        state_dof: usize,
        imu_offsets: Option<ImuLinkOffsets>,
        bias_offsets: Option<ImuLinkOffsets>,
        unexpected_column: Option<usize>,
    ) -> (WhitenedFactorRowStack, WhitenedFactorRowStack) {
        let mut imu_jacobian = DMatrix::<f32>::zeros(9, state_dof);
        let mut bias_jacobian = DMatrix::<f32>::zeros(6, state_dof);
        if let Some(offsets) = imu_offsets {
            for row in 0..9 {
                for column in 0..AOM_NAV_DOF * 2 {
                    let block = column / AOM_NAV_DOF;
                    let global_offset = if block == 0 {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    let global_column = global_offset + column % AOM_NAV_DOF;
                    if global_column < state_dof {
                        imu_jacobian[(row, global_column)] =
                            1.0 + (row * AOM_NAV_DOF * 2 + column) as f32 * 0.03125;
                    }
                }
            }
            for row in 0..6 {
                for column in 0..AOM_NAV_DOF * 2 {
                    let block = column / AOM_NAV_DOF;
                    let global_offset = if block == 0 {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    let global_column = global_offset + column % AOM_NAV_DOF;
                    if global_column < state_dof {
                        bias_jacobian[(row, global_column)] =
                            -0.5 + (row * AOM_NAV_DOF * 2 + column) as f32 * 0.0625;
                    }
                }
            }
        }
        if let Some(column) = unexpected_column {
            imu_jacobian[(0, column)] = 7.0;
        }
        let imu_residual = DVector::from_fn(9, |row, _| 0.25 + row as f32 * 0.125);
        let bias_residual = DVector::from_fn(6, |row, _| -0.5 + row as f32 * 0.03125);
        let imu = WhitenedFactorRowStack::with_objective_cost_kind(
            imu_jacobian.map(f64::from),
            DMatrix::zeros(9, 0),
            imu_residual.map(f64::from),
            0.0,
            FactorKind::Imu,
        )
        .unwrap();
        let bias = WhitenedFactorRowStack::with_objective_cost_kind(
            bias_jacobian.map(f64::from),
            DMatrix::zeros(6, 0),
            bias_residual.map(f64::from),
            0.0,
            FactorKind::Bias,
        )
        .unwrap();
        let imu = if let Some(offsets) = imu_offsets {
            imu.with_imu_link_offsets(offsets.start, offsets.end)
        } else {
            imu
        };
        let bias = if let Some(offsets) = bias_offsets {
            bias.with_imu_link_offsets(offsets.start, offsets.end)
        } else {
            bias
        };
        (imu, bias)
    }

    fn fixture_f32_bits(value: &serde_json::Value, field: &str) -> Vec<u32> {
        value[field]
            .as_array()
            .unwrap_or_else(|| panic!("fixture field {field} is not an array"))
            .iter()
            .map(|word| {
                let word = word
                    .as_str()
                    .unwrap_or_else(|| panic!("fixture field {field} contains a non-string"));
                u32::from_str_radix(word, 16)
                    .unwrap_or_else(|error| panic!("invalid fixture f32 word {word}: {error}"))
            })
            .collect()
    }

    fn fixture_matrix_bits(value: &serde_json::Value) -> Vec<u32> {
        let rows = value["rows"].as_u64().unwrap() as usize;
        let cols = value["cols"].as_u64().unwrap() as usize;
        let bits = fixture_f32_bits(value, "bits_row_major");
        assert_eq!(bits.len(), rows * cols);
        bits
    }

    #[test]
    #[ignore = "requires pinned external m7im15 full-product capture"]
    fn m7im15_local_product_matches_full_900_30_oracle_bits() {
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7im15_test_new.json");
        let fixture_text = std::fs::read_to_string(&fixture_path).unwrap_or_else(|error| {
            panic!(
                "m7im15 oracle fixture {} is required for this bit gate: {error}",
                fixture_path.display()
            )
        });
        let fixture: serde_json::Value = serde_json::from_str(&fixture_text).unwrap();
        let input = &fixture["inputs"]["Jp"];
        assert_eq!(input["rows"].as_u64(), Some(IMU_LOCAL_ROWS as u64));
        assert_eq!(input["cols"].as_u64(), Some(IMU_LOCAL_COLS as u64));
        let jacobian_bits = fixture_matrix_bits(input);
        let residual_bits = fixture_f32_bits(&fixture["inputs"]["r"], "bits");
        let jacobian = DMatrix::from_row_slice(
            IMU_LOCAL_ROWS,
            IMU_LOCAL_COLS,
            &jacobian_bits
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect::<Vec<_>>(),
        );
        let residual = DVector::from_column_slice(
            &residual_bits
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect::<Vec<_>>(),
        );
        let (actual_h, actual_b) = local_imu_h_b_15x30(&jacobian, &residual);

        let expected_h = fixture_matrix_bits(&fixture["native_eigen"]["H"]);
        let expected_b = fixture_f32_bits(&fixture["native_eigen"]["b"], "bits");
        let actual_h_ref = &actual_h;
        let actual_h_row_major = (0..IMU_LOCAL_COLS)
            .flat_map(|row| (0..IMU_LOCAL_COLS).map(move |column| actual_h_ref[(row, column)]))
            .collect::<Vec<_>>();
        assert_f32_bits(&actual_h_row_major, &expected_h);
        assert_f32_bits(actual_b.as_slice(), &expected_b);
        assert_eq!(expected_h.len(), 900);
        assert_eq!(expected_b.len(), 30);
    }

    #[test]
    fn m7im15_scatter_at_nonzero_offsets_preserves_cross_block_order() {
        let mut accumulator_h = DMatrix::<f32>::zeros(40, 40);
        let mut accumulator_b = DVector::<f32>::zeros(40);
        let local_h = DMatrix::from_fn(IMU_LOCAL_COLS, IMU_LOCAL_COLS, |row, column| {
            (row * IMU_LOCAL_COLS + column) as f32 + 0.25
        });
        let local_b = DVector::from_fn(IMU_LOCAL_COLS, |row, _| row as f32 + 0.5);
        scatter_local_imu_h_b_15x30(
            &mut accumulator_h,
            &mut accumulator_b,
            &local_h,
            &local_b,
            ImuLinkOffsets { start: 6, end: 21 },
        )
        .unwrap();

        for row in 0..IMU_LOCAL_COLS {
            for column in 0..IMU_LOCAL_COLS {
                let (global_row, global_column) = if row < AOM_NAV_DOF {
                    if column < AOM_NAV_DOF {
                        (6 + row, 6 + column)
                    } else {
                        (6 + row, 21 + column - AOM_NAV_DOF)
                    }
                } else if column < AOM_NAV_DOF {
                    (21 + row - AOM_NAV_DOF, 6 + column)
                } else {
                    (21 + row - AOM_NAV_DOF, 21 + column - AOM_NAV_DOF)
                };
                assert_eq!(
                    accumulator_h[(global_row, global_column)].to_bits(),
                    local_h[(row, column)].to_bits(),
                    "scatter mismatch local ({row},{column}) -> global ({global_row},{global_column})"
                );
            }
        }
        for row in 0..AOM_NAV_DOF {
            assert_eq!(accumulator_b[6 + row].to_bits(), local_b[row].to_bits());
            assert_eq!(
                accumulator_b[21 + row].to_bits(),
                local_b[AOM_NAV_DOF + row].to_bits()
            );
        }
        // Distinct cross blocks make an accidental transpose immediately
        // visible.  H10/H01 are copied in their native local orientation.
        assert_eq!(
            accumulator_h[(21 + 2, 6 + 4)].to_bits(),
            local_h[(17, 4)].to_bits()
        );
        assert_eq!(
            accumulator_h[(6 + 2, 21 + 4)].to_bits(),
            local_h[(2, 19)].to_bits()
        );

        let mut too_small_h = DMatrix::<f32>::zeros(35, 35);
        let mut too_small_b = DVector::<f32>::zeros(35);
        assert!(scatter_local_imu_h_b_15x30(
            &mut too_small_h,
            &mut too_small_b,
            &local_h,
            &local_b,
            ImuLinkOffsets { start: 6, end: 21 },
        )
        .is_err());
    }

    #[test]
    fn m7im15_diagnostic_uses_global_width_for_absolute_offsets() {
        let offsets = ImuLinkOffsets { start: 6, end: 21 };
        let local_jacobian = DMatrix::from_fn(IMU_LOCAL_ROWS, IMU_LOCAL_COLS, |row, column| {
            0.25_f32 + (row * IMU_LOCAL_COLS + column) as f32 * 0.03125
        });
        let residual = DVector::from_fn(IMU_LOCAL_ROWS, |row, _| -0.75_f32 + row as f32 * 0.0625);
        let mut global_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, 40);
        for row in 0..IMU_LOCAL_ROWS {
            for column in 0..IMU_LOCAL_COLS {
                let global_column = if column < AOM_NAV_DOF {
                    offsets.start + column
                } else {
                    offsets.end + column - AOM_NAV_DOF
                };
                global_jacobian[(row, global_column)] = local_jacobian[(row, column)];
            }
        }

        // The local product must remain 30x30 even though its active columns
        // live at absolute offsets 6 and 21 in the 40-column source stack.
        // The old helper indexed a 30x30 product with offset 21 and panicked
        // as soon as the second 15-column block was reached.
        let diagnostic =
            imu_local_block_diagnostic(&local_jacobian, &residual, &global_jacobian, offsets, None);
        assert_eq!(diagnostic.active_offsets, vec![6, 21]);
        assert_eq!(
            diagnostic.local_jacobian.shape(),
            (IMU_LOCAL_ROWS, IMU_LOCAL_COLS)
        );
        assert_eq!(diagnostic.residual.len(), IMU_LOCAL_ROWS);
        assert_eq!(diagnostic.local_jacobian[(0, 0)].to_bits(), 0x3e80_0000);
        assert_eq!(diagnostic.local_jacobian[(14, 29)].to_bits(), 0x4164_8000);
        assert_eq!(diagnostic.residual[0].to_bits(), 0xbf40_0000);
        assert_eq!(diagnostic.residual[14].to_bits(), 0x3e00_0000);
        assert_eq!(diagnostic.local_h.shape(), (IMU_LOCAL_COLS, IMU_LOCAL_COLS));
        assert_eq!(diagnostic.local_b.len(), IMU_LOCAL_COLS);
        assert_eq!(diagnostic.local_vs_global_h_mismatches, 0);
        assert_eq!(diagnostic.local_vs_global_b_mismatches, 0);
        assert_eq!(diagnostic.local_vs_global_padded_h_mismatches, 0);
        assert_eq!(diagnostic.local_vs_global_padded_b_mismatches, 0);
    }

    fn assert_pose_f32_bits(
        pose: F32Pose,
        expected_translation: [u32; 3],
        expected_quaternion_xyzw: [u32; 4],
    ) {
        assert_f32_bits(
            &[pose.translation.x, pose.translation.y, pose.translation.z],
            &expected_translation,
        );
        let q = pose.rotation.quaternion();
        assert_f32_bits(&[q.i, q.j, q.k, q.w], &expected_quaternion_xyzw);
    }

    fn m7_fixed_relative_point() -> (UnitQuaternion<f32>, Vector3<f32>, Vector3<f32>, f32) {
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f7ffd31),
            f32::from_bits(0xbc0e2956),
            f32::from_bits(0xbb507eff),
            f32::from_bits(0xba002940),
        ));
        let translation = Vector3::new(
            f32::from_bits(0xbde1e254),
            f32::from_bits(0xbaf1ecc0),
            f32::from_bits(0xba884364),
        );
        let bearing = Vector3::new(
            f32::from_bits(0xbf2a746e),
            f32::from_bits(0xbe9aed26),
            f32::from_bits(0x3f2e9644),
        );
        (rotation, translation, bearing, f32::from_bits(0x3e160ef3))
    }

    fn m7_fixed_camera() -> DoubleSphereCamera {
        DoubleSphereCamera::new(
            361.6713883800533,
            360.5856493689301,
            379.40818394080869,
            255.9772968522045,
            -0.21300835384809328,
            0.5767008625037023,
            752,
            480,
        )
        .unwrap()
    }

    fn m7_fixed_factor_inputs() -> (
        DoubleSphereCamera,
        SE3,
        SE3,
        SE3,
        SE3,
        InverseDistanceLandmark,
        Point2<f64>,
    ) {
        let anchor_pose = SE3::new(
            UnitQuaternion::new_unchecked(Quaternion::new(
                0.5944822430610657,
                -0.052778493613004684,
                -0.8023747801780701,
                0.0,
            )),
            Vector3::zeros(),
        );
        let target_pose = SE3::new(
            UnitQuaternion::new_unchecked(Quaternion::new(
                0.5954498648643494,
                -0.05392327159643173,
                -0.801576554775238,
                -0.002603980479761958,
            )),
            Vector3::new(
                0.0003828657791018486,
                -9.85765946097672e-5,
                -0.002031802199780941,
            ),
        );
        let anchor_extrinsic = SE3::new(
            UnitQuaternion::new_normalize(Quaternion::new(
                0.7123125505904486,
                -0.007239825785317818,
                0.007541278561558601,
                0.7017845426564943,
            )),
            Vector3::new(
                -0.016774788924641534,
                -0.068938940687127,
                0.005139123188382424,
            ),
        );
        let target_extrinsic = SE3::new(
            UnitQuaternion::new_normalize(Quaternion::new(
                0.7115930283929829,
                -0.0023360576185881625,
                0.013000769689092388,
                0.7024677108343111,
            )),
            Vector3::new(
                -0.01507436282032619,
                0.0412627204046637,
                0.00316287258752953,
            ),
        );
        let landmark = InverseDistanceLandmark {
            anchor_pose: 0,
            anchor_camera_id: 0,
            direction: StereographicDirection {
                xy: Point2::new(-0.39586612582206726, -0.1799013614654541),
            },
            inverse_distance: 0.14654140174388885,
        };
        (
            m7_fixed_camera(),
            anchor_pose,
            anchor_extrinsic,
            target_pose,
            target_extrinsic,
            landmark,
            Point2::new(27.31320571899414, 106.39038848876953),
        )
    }

    #[test]
    fn m7_eigen_quaternion_matrix_matches_pinned_lanes() {
        let (rotation, _, _, _) = m7_fixed_relative_point();
        let matrix = eigen_quaternion_matrix_f32(rotation);
        assert_f32_bits(
            matrix.as_slice(),
            &[
                0x3f7ffea4, 0xba71d6ad, 0x3bd0c3e1, 0x3a87645a, 0x3f7ff61a, 0xbc8e2141, 0xbbd0358a,
                0x3c8e2e4d, 0x3f7ff4ce,
            ],
        );
    }

    #[test]
    fn m7_frame4_iter1_relative_pose_matrix_and_translation_match_native() {
        // Native Basalt frame-4/iter-1 track-120 observations, host
        // frame0/cam0 -> target frame3/cam0 and cam1.  These endpoint
        // quaternions are captured immediately before `linearizePoint`;
        // checking the complete Eigen column-major matrix guards the native
        // f32 conversion schedule used by the current value path.
        let cases = [
            (
                [0x3bcfedd3, 0xbbc0c558, 0x3b389897, 0x3f7ffd4a],
                [
                    0x3f7ffa6d, 0x3bb62458, 0x3c41593c, 0xbbbb08ed, 0x3f7ff9af, 0x3c4f609f,
                    0xbc402d5f, 0xbc5076a0, 0x3f7ff630,
                ],
            ),
            (
                [0xba7fd817, 0xbbcefd9a, 0x3aeb32ff, 0x3f7ffe8f],
                [
                    0x3f7ffa59, 0x3b6c0089, 0x3c4eedbf, 0xbb6a62cf, 0x3f7fff74, 0xbb0167ab,
                    0xbc4f0b21, 0x3afcddf6, 0x3f7ffaa5,
                ],
            ),
        ];
        for (q_xyzw, expected_matrix) in cases {
            let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q_xyzw[3]),
                f32::from_bits(q_xyzw[0]),
                f32::from_bits(q_xyzw[1]),
                f32::from_bits(q_xyzw[2]),
            ));
            let matrix = eigen_quaternion_matrix_native_f32(rotation);
            assert_f32_bits(matrix.as_slice(), &expected_matrix);
        }
    }

    #[test]
    fn m7_probe_relpose_candidate_vs_generic_iter1_frame3_cam1() {
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            let q = std::hint::black_box(q);
            let t = std::hint::black_box(t);
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }
        fn drel(pose: F32Pose, host: F32Pose) -> SMatrix<f32, 6, 6> {
            eigen_adjoint_times_rotation_blocks_f32(
                pose,
                eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                1.0,
            )
        }
        let host = pose([0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd], [0, 0, 0]);
        let target = pose(
            [0xbd5e3af6, 0xbf4e66c7, 0xbbc319f3, 0x3f16cb91],
            [0x3b2b6242, 0xbb51bff3, 0xbd4c7b9e],
        );
        let target_ext = pose(
            [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
            [0xbc76fa76, 0x3d290319, 0x3b4f4832],
        );
        let target_camera_from_imu = std::hint::black_box(target_ext.inverse());
        let relative = sophus_relative_imu_f32(target, host);
        let generic = std::hint::black_box(F32Pose {
            rotation: sophus_quat_product_f32(target_camera_from_imu.rotation, relative.rotation),
            translation: sophus_rotate_f32(target_camera_from_imu.rotation, relative.translation)
                + target_camera_from_imu.translation,
        });
        let candidate = std::hint::black_box(sophus_compute_relpose_tmp_out_of_line_f32(
            target_camera_from_imu,
            target,
            host,
        ));
        let candidate_raw = m7_relpose_packet_product_raw_for_test(
            target_camera_from_imu.rotation,
            relative.rotation,
        );
        let generic_raw = m7_generic_packet_product_raw_for_test(
            target_camera_from_imu.rotation,
            relative.rotation,
        );
        let candidate_norm = m7_normalize_packet_product_for_test(candidate_raw);
        let generic_norm = m7_normalize_packet_product_for_test(generic_raw);
        let candidate_drel = drel(candidate, host);
        let generic_drel = drel(generic, host);
        let d_diffs = candidate_drel
            .as_slice()
            .iter()
            .zip(generic_drel.as_slice())
            .enumerate()
            .filter_map(|(index, (actual, expected))| {
                (actual.to_bits() != expected.to_bits()).then_some(format!(
                    "{index}:{:08x}/{:08x}",
                    actual.to_bits(),
                    expected.to_bits()
                ))
            })
            .collect::<Vec<_>>();
        // First actual_eigen_site record in
        // target/m7im15_native_actual_eigen_site_iter1_track119_20260827.jsonl:
        // iteration 1, track 119, target frame 3/cam 1.  The native right36
        // is Eigen's column-major memory order, matching SMatrix::as_slice().
        let native_right36 = [
            0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3f01,
            0xbd87ecbc, 0x3dc4f985, 0x00000000, 0x00000000, 0x00000000, 0xbd118207, 0xbf74a0e5,
            0xbe95cd73, 0x00000000, 0x00000000, 0x00000000, 0x3d8687dc, 0xbd228425, 0xbb8c8d8a,
            0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0xbbecbb09, 0xbc3f8e51, 0x3d8841e8, 0x3f7e3f01,
            0xbd87ecbc, 0x3dc4f985, 0x3b7e5e61, 0xbc360bff, 0x3d12b628, 0xbd118207, 0xbf74a0e5,
            0xbe95cd73,
        ];
        let drel_bits = |matrix: &SMatrix<f32, 6, 6>| {
            matrix
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let candidate_bits = drel_bits(&candidate_drel);
        let generic_bits = drel_bits(&generic_drel);
        let candidate_native_exact = candidate_bits
            .iter()
            .zip(native_right36)
            .filter(|(actual, expected)| **actual == *expected)
            .count();
        let generic_native_exact = generic_bits
            .iter()
            .zip(native_right36)
            .filter(|(actual, expected)| **actual == *expected)
            .count();
        let format_bits = |bits: &[u32]| {
            bits.iter()
                .map(|bits| format!("{bits:08x}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        println!(
            "iter1=3/1 drel36_native={} drel36_candidate={} drel36_generic={} drel36_exact_candidate={}/36 drel36_exact_generic={}/36",
            format_bits(&native_right36),
            format_bits(&candidate_bits),
            format_bits(&generic_bits),
            candidate_native_exact,
            generic_native_exact,
        );
        assert_eq!(
            generic_native_exact, 24,
            "generic iter1 frame3/cam1 drel must match native right36 in 24/36 lanes"
        );
        assert_eq!(
            candidate_native_exact, 18,
            "candidate iter1 frame3/cam1 drel must match native right36 in 18/36 lanes"
        );
        let q = |pose: F32Pose| {
            [
                pose.rotation.i.to_bits(),
                pose.rotation.j.to_bits(),
                pose.rotation.k.to_bits(),
                pose.rotation.w.to_bits(),
            ]
        };
        let t = |pose: F32Pose| {
            [
                pose.translation.x.to_bits(),
                pose.translation.y.to_bits(),
                pose.translation.z.to_bits(),
            ]
        };
        println!(
            "iter1=3/1 candidate_q={:08x},{:08x},{:08x},{:08x} generic_q={:08x},{:08x},{:08x},{:08x} candidate_t={:08x},{:08x},{:08x} generic_t={:08x},{:08x},{:08x} drel_diffs=[{}]",
            q(candidate)[0],
            q(candidate)[1],
            q(candidate)[2],
            q(candidate)[3],
            q(generic)[0],
            q(generic)[1],
            q(generic)[2],
            q(generic)[3],
            t(candidate)[0],
            t(candidate)[1],
            t(candidate)[2],
            t(generic)[0],
            t(generic)[1],
            t(generic)[2],
            d_diffs.join(","),
        );
        println!(
            "iter1=3/1 raw_candidate={:08x},{:08x},{:08x},{:08x} raw_generic={:08x},{:08x},{:08x},{:08x} norm_candidate={:08x},{:08x},{:08x},{:08x} norm_generic={:08x},{:08x},{:08x},{:08x}",
            candidate_raw[0].to_bits(),
            candidate_raw[1].to_bits(),
            candidate_raw[2].to_bits(),
            candidate_raw[3].to_bits(),
            generic_raw[0].to_bits(),
            generic_raw[1].to_bits(),
            generic_raw[2].to_bits(),
            generic_raw[3].to_bits(),
            candidate_norm[0].to_bits(),
            candidate_norm[1].to_bits(),
            candidate_norm[2].to_bits(),
            candidate_norm[3].to_bits(),
            generic_norm[0].to_bits(),
            generic_norm[1].to_bits(),
            generic_norm[2].to_bits(),
            generic_norm[3].to_bits(),
        );
    }

    #[test]
    fn m7im15_pose_lin_generic_drel_matches_tied_native_and_current_is_negative() {
        let packet: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7im15_native_iter1_track119_packet_inputs_20260827.json"
        ))
        .expect("packet-input fixture must be valid JSON");

        fn pose(packet: &serde_json::Value, field: &str) -> F32Pose {
            let pose = &packet[field];
            let q = pose["q_xyzw"]
                .as_array()
                .expect("pose quaternion must be an array")
                .iter()
                .map(|value| {
                    u32::from_str_radix(value.as_str().expect("pose bits must be strings"), 16)
                        .expect("pose bits must be hexadecimal")
                })
                .collect::<Vec<_>>();
            let t = pose["t_xyz"]
                .as_array()
                .expect("pose translation must be an array")
                .iter()
                .map(|value| {
                    u32::from_str_radix(value.as_str().expect("pose bits must be strings"), 16)
                        .expect("pose bits must be hexadecimal")
                })
                .collect::<Vec<_>>();
            assert_eq!(q.len(), 4);
            assert_eq!(t.len(), 3);
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }

        fn drel(packet_pose: F32Pose, host: F32Pose) -> SMatrix<f32, 6, 6> {
            eigen_adjoint_times_rotation_blocks_f32(
                packet_pose,
                eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                1.0,
            )
        }

        fn candidate_drel(packet_pose: F32Pose, host: F32Pose) -> SMatrix<f32, 6, 6> {
            m7_candidate_adjoint_times_rotation_blocks_f32(
                packet_pose,
                m7_candidate_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                1.0,
            )
        }

        fn generic_drel(
            host: F32Pose,
            target: F32Pose,
            target_extrinsic: F32Pose,
        ) -> SMatrix<f32, 6, 6> {
            let target_camera_from_imu = target_extrinsic.inverse();
            let relative = sophus_relative_imu_f32(target, host);
            let target_camera_from_anchor_imu = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    relative.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    relative.translation,
                ) + target_camera_from_imu.translation,
            };
            drel(target_camera_from_anchor_imu, host)
        }

        fn expected(packet: &serde_json::Value, field: &str) -> Vec<u32> {
            packet[field]
                .as_array()
                .expect("drel bits must be an array")
                .iter()
                .map(|value| {
                    u32::from_str_radix(value.as_str().expect("drel bits must be strings"), 16)
                        .expect("drel bits must be hexadecimal")
                })
                .collect()
        }

        let host_lin = pose(&packet, "host_pose_lin");
        let target_lin = pose(&packet, "target_pose_lin");
        let host_current = pose(&packet, "host_pose_current");
        let target_current = pose(&packet, "target_pose_current");
        let target_extrinsic = pose(&packet, "target_extrinsic");
        let native_h = expected(&packet, "d_rel_d_h36");
        let native_t = expected(&packet, "d_rel_d_t36");
        assert_eq!(native_h.len(), 36);
        assert_eq!(native_t.len(), 36);

        let lin_h = generic_drel(host_lin, target_lin, target_extrinsic);
        let candidate_lin_h = {
            let target_camera_from_imu = target_extrinsic.inverse();
            let relative = sophus_relative_imu_f32(target_lin, host_lin);
            let target_camera_from_anchor_imu = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    relative.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    relative.translation,
                ) + target_camera_from_imu.translation,
            };
            candidate_drel(target_camera_from_anchor_imu, host_lin)
        };
        let target_camera_from_imu = target_extrinsic.inverse();
        let lin_t = eigen_adjoint_times_rotation_blocks_f32(
            target_camera_from_imu,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(target_lin.rotation)),
            -1.0,
        );
        let candidate_lin_t = m7_candidate_adjoint_times_rotation_blocks_f32(
            target_camera_from_imu,
            m7_candidate_quaternion_matrix_f32(sophus_so3_inverse(target_lin.rotation)),
            -1.0,
        );
        let current_h = generic_drel(host_current, target_current, target_extrinsic);
        let current_t = eigen_adjoint_times_rotation_blocks_f32(
            target_camera_from_imu,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(target_current.rotation)),
            -1.0,
        );
        let candidate_current_t = m7_candidate_adjoint_times_rotation_blocks_f32(
            target_camera_from_imu,
            m7_candidate_quaternion_matrix_f32(sophus_so3_inverse(target_current.rotation)),
            -1.0,
        );

        let exact = |actual: &SMatrix<f32, 6, 6>, expected: &[u32]| {
            actual
                .as_slice()
                .iter()
                .zip(expected.iter())
                .filter(|(actual, expected)| actual.to_bits() == **expected)
                .count()
        };
        let lin_h_exact = exact(&lin_h, &native_h);
        let lin_t_exact = exact(&lin_t, &native_t);
        let candidate_lin_h_exact = exact(&candidate_lin_h, &native_h);
        let candidate_lin_t_exact = exact(&candidate_lin_t, &native_t);
        let current_h_exact = exact(&current_h, &native_h);
        let current_t_exact = exact(&current_t, &native_t);
        let candidate_current_t_exact = exact(&candidate_current_t, &native_t);
        eprintln!(
            "m7im15 poseLin generic drel exact h={lin_h_exact}/36 candidate_h={candidate_lin_h_exact}/36 t={lin_t_exact}/36 candidate_t={candidate_lin_t_exact}/36; current-input negative h={current_h_exact}/36 t={current_t_exact}/36 candidate_t={candidate_current_t_exact}/36"
        );
        assert_eq!(candidate_lin_h_exact, 36);
        assert_eq!(candidate_lin_t_exact, 36);
        assert_eq!(
            lin_h_exact, 36,
            "poseLin generic drel_h must match tied native"
        );
        assert_eq!(
            lin_t_exact, 36,
            "poseLin generic drel_t must match tied native"
        );
        assert!(
            current_h_exact < 36 || current_t_exact < 36,
            "current-input generic drel must remain a negative witness"
        );
    }

    #[test]
    fn m7im15_frame6_iter4_track120_obs6_target_drel_matches_native_call8() {
        // Native `computeRelPose` call 8 is the target absolute-pose derivative
        // packet for frame 6/iter 4 track 120 observation 6.  Keep the
        // comparison at the raw 6x6 boundary: this distinguishes a d_rel
        // producer mismatch from the subsequent 2x6*6x6 Eigen assignment.
        fn pose(q_xyzw: [u32; 4], t_xyz: [u32; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q_xyzw[3]),
                    f32::from_bits(q_xyzw[0]),
                    f32::from_bits(q_xyzw[1]),
                    f32::from_bits(q_xyzw[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t_xyz[0]),
                    f32::from_bits(t_xyz[1]),
                    f32::from_bits(t_xyz[2]),
                ),
            }
        }

        let target_camera_from_imu = pose(
            [0x3bed3c0f, 0xbbf71cd4, 0xbf33a827, 0x3f365a1e],
            [0x3d8ddf2e, 0xbc810144, 0xbb71aa03],
        );
        let target_pose_fej = pose(
            [0xbd61da53, 0xbf524aba, 0xbbd8d37a, 0x3f114c04],
            [0xba89dd02, 0xbbe8ec4f, 0xbe135a1f],
        );
        let target_inverse = sophus_so3_inverse(target_pose_fej.rotation);
        let r_w_i_target_inv = eigen_quaternion_matrix_f32(target_inverse);
        let candidate_r_w_i_target_inv = m7_candidate_quaternion_matrix_f32(target_inverse);
        eprintln!(
            "target_inverse_q={:08x} {:08x} {:08x} {:08x} matrix={} candidate_matrix={}",
            target_inverse.i.to_bits(),
            target_inverse.j.to_bits(),
            target_inverse.k.to_bits(),
            target_inverse.w.to_bits(),
            r_w_i_target_inv
                .as_slice()
                .iter()
                .map(|value| format!("{:08x}", value.to_bits()))
                .collect::<Vec<_>>()
                .join(" "),
            candidate_r_w_i_target_inv
                .as_slice()
                .iter()
                .map(|value| format!("{:08x}", value.to_bits()))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let actual =
            eigen_adjoint_times_rotation_blocks_f32(target_camera_from_imu, r_w_i_target_inv, -1.0);
        let candidate = m7_candidate_adjoint_times_rotation_blocks_f32(
            target_camera_from_imu,
            candidate_r_w_i_target_inv,
            -1.0,
        );
        let expected = [
            0xbde613b7, 0xbeb39f91, 0x3f6dff58, 0x00000000, 0x00000000, 0x00000000, 0xbf7e42b2,
            0x3d8bc630, 0xbdc10d9f, 0x00000000, 0x00000000, 0x80000000, 0x3cf8ddd0, 0x3f6f175d,
            0x3eb65413, 0x00000000, 0x00000000, 0x00000000, 0xbc8287e0, 0xbd830bee, 0xbcd59519,
            0xbde613b7, 0xbeb39f91, 0x3f6dff58, 0x3ae38e3c, 0x3c26fe37, 0xbc32cbb7, 0xbf7e42b2,
            0x3d8bc630, 0xbdc10d9f, 0xbb0dd14b, 0xbccb0173, 0x3d857b21, 0x3cf8ddd0, 0x3f6f175d,
            0x3eb65413,
        ];
        assert_f32_bits(actual.as_slice(), &expected);
        assert_f32_bits(candidate.as_slice(), &expected);
    }

    #[test]
    fn m7im15_historical_packet_tmp_drel_matches_variant_fixture() {
        // This is a historical step-only call-15 packet. The helper under
        // test is a test-only reconstruction of that out-of-line boundary;
        // it is not the active anchored_visual_reprojection_factor_f32 path.
        fn bits(value: &serde_json::Value) -> u32 {
            u32::from_str_radix(
                value
                    .as_str()
                    .expect("f32 fixture bits must be hexadecimal strings"),
                16,
            )
            .expect("f32 fixture bits must be hexadecimal")
        }

        fn pose(packet: &serde_json::Value, field: &str) -> F32Pose {
            let value = &packet[field];
            let q = value["q_xyzw"]
                .as_array()
                .expect("pose quaternion must be an array");
            let t = value["t_xyz"]
                .as_array()
                .expect("pose translation must be an array");
            assert_eq!(q.len(), 4);
            assert_eq!(t.len(), 3);
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(bits(&q[3])),
                    f32::from_bits(bits(&q[0])),
                    f32::from_bits(bits(&q[1])),
                    f32::from_bits(bits(&q[2])),
                )),
                translation: Vector3::new(
                    f32::from_bits(bits(&t[0])),
                    f32::from_bits(bits(&t[1])),
                    f32::from_bits(bits(&t[2])),
                ),
            }
        }

        fn pose_bits(pose: F32Pose) -> [u32; 7] {
            [
                pose.rotation.i.to_bits(),
                pose.rotation.j.to_bits(),
                pose.rotation.k.to_bits(),
                pose.rotation.w.to_bits(),
                pose.translation.x.to_bits(),
                pose.translation.y.to_bits(),
                pose.translation.z.to_bits(),
            ]
        }

        fn matrix_bits(packet: &serde_json::Value, field: &str) -> Vec<u32> {
            packet[field]
                .as_array()
                .expect("matrix fixture must be an array")
                .iter()
                .map(bits)
                .collect()
        }

        let packet: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7im15_native_iter1_track119_packet_inputs_20260827.json"
        ))
        .expect("packet-input fixture must be valid JSON");
        assert_eq!(
            packet["schema"].as_str(),
            Some("basalt.relpose_packet_inputs.v1")
        );
        assert_eq!(packet["record"].as_str(), Some("relpose_packet_inputs"));
        assert_eq!(packet["iteration"].as_u64(), Some(1));
        assert_eq!(packet["trial"].as_u64(), Some(0));
        assert_eq!(packet["host_cam"].as_u64(), Some(0));
        assert_eq!(packet["target_cam"].as_u64(), Some(1));
        assert_eq!(packet["host_is_linearized"].as_bool(), Some(true));
        assert_eq!(packet["target_is_linearized"].as_bool(), Some(false));

        let host = pose(&packet, "host_pose_lin");
        let target = pose(&packet, "target_pose_lin");
        let target_camera_from_imu = pose(&packet, "target_extrinsic").inverse();
        let tmp = sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
        assert_eq!(
            pose_bits(tmp),
            pose_bits(pose(&packet, "tmp_for_jacobian")),
            "captured out-of-line variant relpose must reproduce the packet pose"
        );

        let d_rel_d_h = eigen_adjoint_times_rotation_blocks_f32(
            tmp,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        );
        let d_rel_d_t = eigen_adjoint_times_rotation_blocks_f32(
            target_camera_from_imu,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(target.rotation)),
            -1.0,
        );
        let expected_h = matrix_bits(&packet, "d_rel_d_h36");
        let expected_t = matrix_bits(&packet, "d_rel_d_t36");
        assert_eq!(expected_h.len(), 36);
        assert_eq!(expected_t.len(), 36);
        assert_f32_bits(d_rel_d_h.as_slice(), &expected_h);
        assert_f32_bits(d_rel_d_t.as_slice(), &expected_t);

        // Keep the captured current-value chain as a negative witness: the
        // Jacobian boundary is tied to the linearized packet, not an arbitrary
        // current-pose recomputation.
        let current_host = pose(&packet, "host_pose_current");
        let current_target = pose(&packet, "target_pose_current");
        let current_tmp = sophus_compute_relpose_tmp_out_of_line_f32(
            target_camera_from_imu,
            current_target,
            current_host,
        );
        let current_h = eigen_adjoint_times_rotation_blocks_f32(
            current_tmp,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(current_host.rotation)),
            1.0,
        );
        let current_t = eigen_adjoint_times_rotation_blocks_f32(
            target_camera_from_imu,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(current_target.rotation)),
            -1.0,
        );
        let current_matches = current_h
            .as_slice()
            .iter()
            .zip(expected_h.iter())
            .all(|(actual, expected)| actual.to_bits() == *expected)
            && current_t
                .as_slice()
                .iter()
                .zip(expected_t.iter())
                .all(|(actual, expected)| actual.to_bits() == *expected);
        assert!(
            !current_matches,
            "current-value relpose must remain a negative witness"
        );
    }

    #[test]
    fn m7im15_found_tmp_translation_reproduces_native_drel_cross() {
        // Standalone-search witness for the exact native cross block.  The
        // temporary rotation is the tied true-pose operand; only translation
        // is replaced by the ULP-search hit.  Keep this focused regression
        // test-only until the authoritative step-only producer is recovered.
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }

        let found_tmp = pose(
            [0x3c342b3c, 0xbc4f602d, 0xbf3355a4, 0x3f36a35f],
            [0xbd235394, 0xbd83bd75, 0xbc801294],
        );
        let host = pose([0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd], [0, 0, 0]);
        let expected = [
            0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3f01,
            0xbd87ecbc, 0x3dc4f985, 0x00000000, 0x00000000, 0x00000000, 0xbd118207, 0xbf74a0e5,
            0xbe95cd73, 0x00000000, 0x00000000, 0x00000000, 0x3d8687db, 0xbd228425, 0xbb8c8d8c,
            0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0xbbecbb07, 0xbc3f8e51, 0x3d8841e7, 0x3f7e3f01,
            0xbd87ecbc, 0x3dc4f985, 0x3b7e5e58, 0xbc360bff, 0x3d12b627, 0xbd118207, 0xbf74a0e5,
            0xbe95cd73,
        ];
        let right = eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation));
        let actual = eigen_adjoint_times_rotation_blocks_f32(found_tmp, right, 1.0);
        assert_eq!(
            actual
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected,
            "ULP-search translation must reproduce native drel 36/36"
        );
    }

    #[test]
    fn m7_clean_frame4_iter1_cam1_fej_tmp_and_drel_h_match_native() {
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }
        let host = pose([0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd], [0, 0, 0]);
        let _host_ext = pose(
            [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
            [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
        );
        let target = pose(
            [0xbd5e3af6, 0xbf4e66c7, 0xbbc319f3, 0x3f16cb91],
            [0x3b2b6242, 0xbb51bff3, 0xbd4c7b9e],
        );
        let target_ext = pose(
            [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
            [0xbc76fa76, 0x3d290319, 0x3b4f4832],
        );
        let tmp2 = target_ext.inverse();
        let tmp = sophus_compute_relpose_tmp_out_of_line_f32(tmp2, target, host);
        assert_pose_f32_bits(
            tmp,
            [0xbd235393, 0xbd83bd7a, 0xbc8012a4],
            [0x3c342b39, 0xbc4f6029, 0xbf3355a5, 0x3f36a35f],
        );
        let drel = eigen_adjoint_times_rotation_blocks_f32(
            tmp,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        );
        assert_f32_bits(
            drel.as_slice(),
            &[
                0x3de3fe94, 0x3e9306e1, 0xbf738e5b, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3f02,
                0xbd87ecd4, 0x3dc4f984, 0x00000000, 0x00000000, 0x00000000, 0xbd118236, 0xbf74a0e5,
                0xbe95cd73, 0x00000000, 0x00000000, 0x00000000, 0x3d8687e1, 0xbd228425, 0xbb8c8d7c,
                0x3de3fe94, 0x3e9306e1, 0xbf738e5b, 0xbbecbb19, 0xbc3f8e72, 0x3d8841ee, 0x3f7e3f02,
                0xbd87ecd4, 0x3dc4f984, 0x3b7e5e03, 0xbc360bf9, 0x3d12b625, 0xbd118236, 0xbf74a0e5,
                0xbe95cd73,
            ],
        );
    }

    #[test]
    fn m7_clean_frame4_iter0_cam1_fej_tmp_and_drel_h_match_native() {
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }
        let host = pose(
            [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
            [0x00000000, 0x00000000, 0x00000000],
        );
        let target = pose(
            [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
            [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
        );
        let target_ext = pose(
            [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
            [0xbc76fa76, 0x3d290319, 0x3b4f4832],
        );
        let target_camera_from_imu = target_ext.inverse();
        let target_inverse_rotation = sophus_so3_inverse(target.rotation);
        let relative_rotation = sophus_quat_product_f32(target_inverse_rotation, host.rotation);
        assert_f32_bits(
            &[
                relative_rotation.i,
                relative_rotation.j,
                relative_rotation.k,
                relative_rotation.w,
            ],
            &[0x3bc353bb, 0x3bc9740e, 0x3b240734, 0x3f7ffd65],
        );
        let tmp = sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
        assert_pose_f32_bits(
            tmp,
            [0xbd27a81a, 0xbd05555f, 0xbb8ddf30],
            [0x3c31fe46, 0xbc520562, 0xbf33585b, 0x3f36a0a7],
        );
        let drel = eigen_adjoint_times_rotation_blocks_f32(
            tmp,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        );
        assert_f32_bits(
            drel.as_slice(),
            &[
                0x3de4276f, 0x3e92d17a, 0xbf7395cf, 0x00000000, 0x00000000, 0x00000000, 0x3f7e3e30,
                0xbd88188a, 0x3dc51f14, 0x00000000, 0x00000000, 0x00000000, 0xbd11f0e1, 0xbf74a889,
                0xbe9599d5, 0x00000000, 0x00000000, 0x00000000, 0x3d03f3e6, 0xbd21806f, 0xbc04e3d5,
                0x3de4276f, 0x3e92d17a, 0xbf7395cf, 0xbb6030cf, 0xb9bcd31b, 0x3d0f8f43, 0x3f7e3e30,
                0xbd88188a, 0x3dc51f14, 0x3bb0151e, 0xbc416c25, 0x3d1b7a6d, 0xbd11f0e1, 0xbf74a889,
                0xbe9599d5,
            ],
        );
    }

    #[test]
    fn m7_clean_frame4_iter0_cam0_fej_tmp_and_drel_h_match_native() {
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }
        let host = pose(
            [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
            [0x00000000, 0x00000000, 0x00000000],
        );
        let target = pose(
            [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
            [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
        );
        let target_ext = pose(
            [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
            [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
        );
        let target_camera_from_imu = target_ext.inverse();
        let target_inverse_rotation = sophus_so3_inverse(target.rotation);
        let relative_rotation = sophus_quat_product_f32(target_inverse_rotation, host.rotation);
        let relative_translation = sophus_rotate_difference_f32(
            target_inverse_rotation,
            host.translation,
            target.translation,
        );
        assert_f32_bits(
            &[
                relative_rotation.i,
                relative_rotation.j,
                relative_rotation.k,
                relative_rotation.w,
            ],
            &[0x3bc353bb, 0x3bc9740e, 0x3b240734, 0x3f7ffd65],
        );
        assert_f32_bits(
            relative_translation.as_slice(),
            &[0x3c8a503f, 0xb9376126, 0xba473200],
        );
        let relative_product = m7_relpose_packet_product_raw_for_test(
            target_camera_from_imu.rotation,
            relative_rotation,
        );
        eprintln!(
            "m7 cam0 raw_product={:08x} {:08x} {:08x} {:08x}",
            relative_product[0].to_bits(),
            relative_product[1].to_bits(),
            relative_product[2].to_bits(),
            relative_product[3].to_bits(),
        );
        let rotated_translation = sophus_rotate_relpose_out_of_line_f32(
            target_camera_from_imu.rotation,
            relative_translation,
        );
        // Keep the packet-stage value visible while auditing the native
        // target-frame-3/cam0 capture; the final add below is the value
        // materialized at the computeRelPose return boundary.
        eprintln!(
            "m7 cam0 rel_t={:08x} {:08x} {:08x} rotated_t={:08x} {:08x} {:08x}",
            relative_translation.x.to_bits(),
            relative_translation.y.to_bits(),
            relative_translation.z.to_bits(),
            rotated_translation.x.to_bits(),
            rotated_translation.y.to_bits(),
            rotated_translation.z.to_bits(),
        );
        let tmp = sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
        assert_pose_f32_bits(
            tmp,
            [0x3d8e0f98, 0xbd05a9be, 0xbb91861b],
            [0x3c814783, 0xbbf146bb, 0xbf332b9e, 0x3f36cb95],
        );
        let drel = eigen_adjoint_times_rotation_blocks_f32(
            tmp,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        );
        assert_f32_bits(
            drel.as_slice(),
            &[
                0x3de12055, 0x3e9a0cd6, 0xbf7282a1, 0x00000000, 0x00000000, 0x00000000, 0x3f7e4d08,
                0xbd869b07, 0x3dc151a6, 0x00000000, 0x00000000, 0x00000000, 0xbd0ab1a0, 0xbf738e9a,
                0xbe9cba11, 0x00000000, 0x00000000, 0x00000000, 0x3d0417c9, 0x3d859348, 0x3cc85bc6,
                0x3de12055, 0x3e9a0cd6, 0xbf7282a1, 0xbb5d0047, 0xbc338e7d, 0x3ce4342e, 0x3f7e4d08,
                0xbd869b07, 0x3dc151a6, 0x3bbcdefd, 0x3caf2cdf, 0xbd896b41, 0xbd0ab1a0, 0xbf738e9a,
                0xbe9cba11,
            ],
        );
    }

    #[cfg(test)]
    fn m7_relpose_packet_product_raw_for_test(
        first: UnitQuaternion<f32>,
        second: UnitQuaternion<f32>,
    ) -> [f32; 4] {
        let a = first.quaternion();
        let b = second.quaternion();
        let a24 = [a.i, a.j, a.k, a.i];
        let a49 = [a.j, a.k, a.i, a.j];
        let a92 = [a.k, a.i, a.j, a.k];
        let b3f = [b.w, b.w, b.w, b.i];
        let b52 = [b.k, b.i, b.j, b.j];
        let b89 = [b.j, b.k, b.i, b.k];
        let b_times_aw = [b.i * a.w, b.j * a.w, b.k * a.w, b.w * a.w];
        let mut first_negative = [0.0_f32; 4];
        let mut first_positive = [0.0_f32; 4];
        for lane in 0..4 {
            first_negative[lane] = (-b3f[lane]).mul_add(a24[lane], b_times_aw[lane]);
            first_positive[lane] = b3f[lane].mul_add(a24[lane], b_times_aw[lane]);
        }
        first_positive[3] = first_negative[3];
        let mut second_negative = [0.0_f32; 4];
        let mut second_positive = [0.0_f32; 4];
        for lane in 0..4 {
            second_negative[lane] = (-b52[lane]).mul_add(a49[lane], first_positive[lane]);
            second_positive[lane] = b52[lane].mul_add(a49[lane], first_positive[lane]);
        }
        second_positive[3] = second_negative[3];
        let mut product = [0.0_f32; 4];
        for lane in 0..4 {
            product[lane] = (-a92[lane]).mul_add(b89[lane], second_positive[lane]);
        }
        product
    }

    #[cfg(test)]
    fn m7_generic_packet_product_raw_for_test(
        first: UnitQuaternion<f32>,
        second: UnitQuaternion<f32>,
    ) -> [f32; 4] {
        let a = first.quaternion();
        let b = second.quaternion();
        [
            (-a.k).mul_add(b.j, a.j.mul_add(b.k, a.w.mul_add(b.i, a.i * b.w))),
            (-a.i).mul_add(b.k, a.k.mul_add(b.i, a.w.mul_add(b.j, a.j * b.w))),
            (-a.j).mul_add(b.i, a.i.mul_add(b.j, a.w.mul_add(b.k, a.k * b.w))),
            (-a.k).mul_add(b.k, (-a.j).mul_add(b.j, b.w.mul_add(a.w, -(b.i * a.i)))),
        ]
    }

    #[cfg(test)]
    fn m7_normalize_packet_product_for_test(raw: [f32; 4]) -> [f32; 4] {
        let x2 = raw[0] * raw[0];
        let z2 = raw[2] * raw[2];
        let y2 = raw[1] * raw[1];
        let w2 = raw[3] * raw[3];
        let norm = (x2 + z2 + (y2 + w2)).sqrt();
        [raw[0] / norm, raw[1] / norm, raw[2] / norm, raw[3] / norm]
    }

    #[test]
    fn m7_relative_pose_chain_matches_pinned_lanes() {
        let (_, anchor_pose, anchor_extrinsic, target_pose, target_extrinsic, _, _) =
            m7_fixed_factor_inputs();
        let anchor_pose = F32Pose::from_se3(&anchor_pose);
        let target_pose = F32Pose::from_se3(&target_pose);
        let anchor_extrinsic = F32Pose::from_se3(&anchor_extrinsic);
        let target_extrinsic = F32Pose::from_se3(&target_extrinsic);
        let target_camera_from_imu = target_extrinsic.inverse();
        let target_imu_from_anchor_imu = target_pose.inverse().compose(anchor_pose);
        let target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.translation,
            ) + target_camera_from_imu.translation,
        };
        let target_camera_from_anchor_camera = F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_extrinsic.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_extrinsic.translation,
            ) + target_camera_from_anchor_imu.translation,
        };

        assert_pose_f32_bits(
            target_camera_from_imu,
            [0xbd27e3b1, 0xbc8044de, 0xbb7a8e6e],
            [0x3b19188b, 0xbc55012e, 0xbf33d4ed, 0x3f362af6],
        );
        assert_pose_f32_bits(
            target_imu_from_anchor_imu,
            [0x3b06d6cc, 0xb8746f41, 0xb9657ee0],
            [0x3b322ebb, 0xbab5f931, 0x3a19f84f, 0x3f7fffaf],
        );
        assert_pose_f32_bits(
            target_camera_from_anchor_imu,
            [0xbd28004c, 0xbc912752, 0xbb837666],
            [0x3b577913, 0xbc82408f, 0xbf33b736, 0x3f36442d],
        );
        assert_pose_f32_bits(
            target_camera_from_anchor_camera,
            [0xbde1e255, 0xbaf1ec60, 0xba884368],
            [0xbc0e2957, 0xbb507f02, 0xba002d40, 0x3f7ffd31],
        );
    }

    #[test]
    fn m7dx_clean_track1_relative_pose_jacobians_are_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            factor: String,
            host_pose_q_xyzw: [String; 4],
            host_pose_t_xyz: [String; 3],
            target_pose_q_xyzw: [String; 4],
            target_pose_t_xyz: [String; 3],
            host_extrinsic_q_xyzw: [String; 4],
            host_extrinsic_t_xyz: [String; 3],
            target_extrinsic_q_xyzw: [String; 4],
            target_extrinsic_t_xyz: [String; 3],
            d_rel_d_h_bits_column_major: Vec<String>,
            d_rel_d_t_bits_column_major: Vec<String>,
        }

        fn bits(value: &str) -> u32 {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        }

        fn pose(q_xyzw: &[String; 4], t_xyz: &[String; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(bits(&q_xyzw[3])),
                    f32::from_bits(bits(&q_xyzw[0])),
                    f32::from_bits(bits(&q_xyzw[1])),
                    f32::from_bits(bits(&q_xyzw[2])),
                )),
                translation: Vector3::new(
                    f32::from_bits(bits(&t_xyz[0])),
                    f32::from_bits(bits(&t_xyz[1])),
                    f32::from_bits(bits(&t_xyz[2])),
                ),
            }
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7dx_clean_track1_relpose_jacobian.json"
        ))
        .expect("m7dx clean relative-pose Jacobian fixture must parse");
        assert_eq!(fixture.d_rel_d_h_bits_column_major.len(), 36);
        assert_eq!(fixture.d_rel_d_t_bits_column_major.len(), 36);

        let host_pose = pose(&fixture.host_pose_q_xyzw, &fixture.host_pose_t_xyz);
        let target_pose = pose(&fixture.target_pose_q_xyzw, &fixture.target_pose_t_xyz);
        let host_extrinsic = pose(
            &fixture.host_extrinsic_q_xyzw,
            &fixture.host_extrinsic_t_xyz,
        );
        let target_extrinsic = pose(
            &fixture.target_extrinsic_q_xyzw,
            &fixture.target_extrinsic_t_xyz,
        );
        let target_camera_from_imu = target_extrinsic.inverse();
        let target_imu_from_anchor_imu = sophus_relative_imu_f32(target_pose, host_pose);
        let target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.translation,
            ) + target_camera_from_imu.translation,
        };
        // Keep the complete clean factor chain in the fixture exercise, even
        // though the raw d_rel buffers stop before this host-extrinsic suffix.
        let _relative_camera = F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                target_camera_from_anchor_imu.rotation,
                host_extrinsic.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_anchor_imu.rotation,
                host_extrinsic.translation,
            ) + target_camera_from_anchor_imu.translation,
        };

        let r_w_i_anchor_inv = eigen_quaternion_matrix_f32(host_pose.rotation.inverse());
        let r_w_i_target_inv = eigen_quaternion_matrix_f32(target_pose.rotation.inverse());
        let d_rel_d_h = eigen_adjoint_times_rotation_blocks_f32(
            target_camera_from_anchor_imu,
            r_w_i_anchor_inv,
            1.0,
        );
        let d_rel_d_t =
            eigen_adjoint_times_rotation_blocks_f32(target_camera_from_imu, r_w_i_target_inv, -1.0);
        let expected_h = fixture
            .d_rel_d_h_bits_column_major
            .iter()
            .map(|value| bits(value))
            .collect::<Vec<_>>();
        let expected_t = fixture
            .d_rel_d_t_bits_column_major
            .iter()
            .map(|value| bits(value))
            .collect::<Vec<_>>();
        assert_eq!(
            expected_t
                .iter()
                .filter(|&&value| value == 0x8000_0000)
                .count(),
            1
        );
        assert_f32_bits(d_rel_d_h.as_slice(), &expected_h);
        assert_f32_bits(d_rel_d_t.as_slice(), &expected_t);
    }

    #[test]
    fn m7ga_weighted_pose_product_matches_native_packet_tree() {
        // Retained native M7aq pose-boundary probe (track 1, frame 0 cam0 ->
        // frame 1 cam1).  The probe records the post-whitening relative
        // Jacobian and both post-product absolute pose blocks.  These words
        // are deliberately kept in Eigen column-major order, including the
        // signed zero in the target chain.
        fn matrix<const R: usize, const C: usize>(words: &[u32]) -> SMatrix<f32, R, C> {
            assert_eq!(words.len(), R * C);
            let values = words
                .iter()
                .copied()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            SMatrix::from_column_slice(&values)
        }

        let relative = matrix(&[
            0x4247c89d, 0xc129fef8, 0xc12a8201, 0x428cdaf2, 0x4236ccb5, 0x419a2414, 0xc2239d4b,
            0xc3b7255f, 0x43df6969, 0x42231f8c, 0x4314e606, 0xc3af8621,
        ]);
        let d_rel_d_h = matrix(&[
            0x3dda79ad, 0x3e8b3905, 0xbf74d5e7, 0x00000000, 0x00000000, 0x00000000, 0x3f7e5131,
            0xbd8df5f8, 0x3dba92eb, 0x00000000, 0x00000000, 0x00000000, 0xbd2a12ca, 0xbf75b6c8,
            0xbe8e17f1, 0x00000000, 0x00000000, 0x00000000, 0x3c93c295, 0xbd226d6d, 0xbc17c30c,
            0x3dda79ad, 0x3e8b3905, 0xbf74d5e7, 0xbaf80701, 0xb9828894, 0x3ca77d72, 0x3f7e5131,
            0xbd8df5f8, 0x3dba92eb, 0x3a8bd267, 0xbc37c50e, 0x3d1e3cc6, 0xbd2a12ca, 0xbf75b6c8,
            0xbe8e17f1,
        ]);
        let d_rel_d_t = matrix(&[
            0xbdda79af, 0xbe8b3907, 0x3f74d5e9, 0x00000000, 0x00000000, 0x00000000, 0xbf7e5131,
            0x3d8df601, 0xbdba92eb, 0x00000000, 0x00000000, 0x80000000, 0x3d2a12d7, 0x3f75b6ca,
            0x3e8e17f1, 0x00000000, 0x00000000, 0x00000000, 0xbc833104, 0x3d223cf7, 0x3c1b3e27,
            0xbdda79af, 0xbe8b3907, 0x3f74d5e9, 0x3addb39c, 0x388626ba, 0xbc96b372, 0xbf7e5131,
            0x3d8df601, 0xbdba92eb, 0xba312e45, 0x3c37c62b, 0xbd1e7b0f, 0x3d2a12d7, 0x3f75b6ca,
            0x3e8e17f1,
        ]);
        let sqrt_weight = f32::from_bits(0x3ff04066);
        let expected_anchor = [
            0xc29af308, 0xbf450fd7, 0x42cca9a4, 0xc1cd6fc5, 0xc107fd35, 0xc3081664, 0xc236f40c,
            0x440eed0a, 0xc2d6b96d, 0xc43ae580, 0xc45aed81, 0x4309d5bd,
        ];
        let expected_target = [
            0x429af30a, 0x3f450fa6, 0xc2cca9a4, 0x41cd6fca, 0x4107fd38, 0x43081665, 0x4237c9cc,
            0xc40eef86, 0x42d70bb8, 0x443ae8ef, 0x445aef88, 0xc309d848,
        ];

        let anchor = eigen_weighted_pose_jacobian_f32(relative, d_rel_d_h, sqrt_weight);
        let target = eigen_weighted_pose_jacobian_f32(relative, d_rel_d_t, sqrt_weight);
        assert_f32_bits(anchor.as_slice(), &expected_anchor);
        assert_f32_bits(target.as_slice(), &expected_target);
    }

    #[test]
    fn m7fx_clean_same_timestamp_stereo_relative_pose_jacobians_are_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            factor: String,
            #[allow(dead_code)]
            provenance: String,
            host_pose_q_xyzw: [String; 4],
            host_pose_t_xyz: [String; 3],
            target_pose_q_xyzw: [String; 4],
            target_pose_t_xyz: [String; 3],
            host_extrinsic_q_xyzw: [String; 4],
            host_extrinsic_t_xyz: [String; 3],
            target_extrinsic_q_xyzw: [String; 4],
            target_extrinsic_t_xyz: [String; 3],
            d_rel_d_h_bits_column_major: Vec<String>,
            d_rel_d_t_bits_column_major: Vec<String>,
        }

        fn bits(value: &str) -> u32 {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        }

        fn pose(q_xyzw: &[String; 4], t_xyz: &[String; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(bits(&q_xyzw[3])),
                    f32::from_bits(bits(&q_xyzw[0])),
                    f32::from_bits(bits(&q_xyzw[1])),
                    f32::from_bits(bits(&q_xyzw[2])),
                )),
                translation: Vector3::new(
                    f32::from_bits(bits(&t_xyz[0])),
                    f32::from_bits(bits(&t_xyz[1])),
                    f32::from_bits(bits(&t_xyz[2])),
                ),
            }
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7fx_same_timestamp_clean_relpose_jacobian.json"
        ))
        .expect("m7fx same-timestamp relative-pose Jacobian fixture must parse");
        assert_eq!(fixture.d_rel_d_h_bits_column_major.len(), 36);
        assert_eq!(fixture.d_rel_d_t_bits_column_major.len(), 36);

        let host_pose = pose(&fixture.host_pose_q_xyzw, &fixture.host_pose_t_xyz);
        let target_pose = pose(&fixture.target_pose_q_xyzw, &fixture.target_pose_t_xyz);
        let host_extrinsic = pose(
            &fixture.host_extrinsic_q_xyzw,
            &fixture.host_extrinsic_t_xyz,
        );
        let target_extrinsic = pose(
            &fixture.target_extrinsic_q_xyzw,
            &fixture.target_extrinsic_t_xyz,
        );
        let target_camera_from_imu = target_extrinsic.inverse();
        let target_imu_from_anchor_imu = sophus_relative_imu_f32(target_pose, host_pose);
        let target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.translation,
            ) + target_camera_from_imu.translation,
        };
        // The clean relation is a stereo call: host and target share the
        // state timestamp but differ in camera id, so no identity shortcut
        // applies to either relative-pose Jacobian.
        let _relative_camera = F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                target_camera_from_anchor_imu.rotation,
                host_extrinsic.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_anchor_imu.rotation,
                host_extrinsic.translation,
            ) + target_camera_from_anchor_imu.translation,
        };

        let r_w_i_anchor_inv = eigen_quaternion_matrix_f32(host_pose.rotation.inverse());
        let r_w_i_target_inv = eigen_quaternion_matrix_f32(target_pose.rotation.inverse());
        let d_rel_d_h = eigen_adjoint_times_rotation_blocks_f32(
            target_camera_from_anchor_imu,
            r_w_i_anchor_inv,
            1.0,
        );
        let d_rel_d_t =
            eigen_adjoint_times_rotation_blocks_f32(target_camera_from_imu, r_w_i_target_inv, -1.0);
        let expected_h = fixture
            .d_rel_d_h_bits_column_major
            .iter()
            .map(|value| bits(value))
            .collect::<Vec<_>>();
        let expected_t = fixture
            .d_rel_d_t_bits_column_major
            .iter()
            .map(|value| bits(value))
            .collect::<Vec<_>>();
        assert_f32_bits(d_rel_d_h.as_slice(), &expected_h);
        assert_f32_bits(d_rel_d_t.as_slice(), &expected_t);
    }

    #[test]
    fn m7gh_clean_all_relative_pose_jacobians_are_bitwise_exact() {
        // Replay all nine non-identity TimeCam relations from the clean M7fw
        // capture with the clean m7db f32 states and pinned EuRoC calibration
        // words. Compare the raw d_rel buffers before any residual/Jacobian
        // product, so a failure identifies the relative-pose boundary.
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }
        fn bits(value: &serde_json::Value) -> u32 {
            u32::from_str_radix(value.as_str().unwrap(), 16).unwrap()
        }
        fn assert_matrix_bits(
            actual: &SMatrix<f32, 6, 6>,
            expected: &serde_json::Value,
            label: &str,
        ) {
            let expected = expected["f32_bits_column_major"].as_array().unwrap();
            assert_eq!(expected.len(), 36);
            for (index, value) in expected.iter().enumerate() {
                let got = actual.as_slice()[index].to_bits();
                let want = bits(value);
                assert_eq!(got, want, "{label} lane {index}: {got:08x} != {want:08x}");
            }
        }
        fn assert_pose_bits(actual: F32Pose, expected: &serde_json::Value, label: &str) {
            let expected = expected["q_t"]["f32_bits"].as_array().unwrap();
            assert_eq!(expected.len(), 7);
            let actual = [
                actual.rotation.i.to_bits(),
                actual.rotation.j.to_bits(),
                actual.rotation.k.to_bits(),
                actual.rotation.w.to_bits(),
                actual.translation.x.to_bits(),
                actual.translation.y.to_bits(),
                actual.translation.z.to_bits(),
            ];
            for (index, value) in expected.iter().enumerate() {
                let want = bits(value);
                assert_eq!(
                    actual[index], want,
                    "{label} lane {index}: {:08x} != {want:08x}",
                    actual[index]
                );
            }
        }

        let states = [
            pose(
                [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
                [0x00000000, 0x00000000, 0x00000000],
            ),
            pose(
                [0xbd5cdea6, 0xbf4d341f, 0xbb2aa78b, 0x3f186f67],
                [0x39c8bb60, 0xb8cebae8, 0xbb0527fc],
            ),
            pose(
                [0xbd5cf5f2, 0xbf4d97c0, 0xbbac4997, 0x3f17e7a6],
                [0x3af3e90a, 0xb9fd97e2, 0xbbfc3924],
            ),
            pose(
                [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
                [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
            ),
            pose(
                [0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719],
                [0x3c02c75a, 0xbb0c3bda, 0xbcdc2b7f],
            ),
        ];
        let extrinsics = [
            pose(
                [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
                [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
            ),
            pose(
                [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
                [0xbc76fa76, 0x3d290319, 0x3b4f4832],
            ),
        ];
        let clean: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7gh_clean_all_relpose_jacobians.json"
        ))
        .unwrap();
        assert_eq!(
            clean["schema"].as_str(),
            Some("visloc-rs.basalt.m7gh.clean-all-relpose-jacobians.v1")
        );
        assert_eq!(
            clean["source"]["upstream_commit"].as_str(),
            Some("0f3b2b52c807f70ff4e2973ce253c73329eea7bc")
        );
        assert_eq!(clean["relation_count"].as_u64(), Some(9));
        for target_frame in 0..5 {
            for target_cam in 0..2 {
                if target_frame == 0 && target_cam == 0 {
                    continue;
                }
                let record = clean["records"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|record| {
                        record["caller"]["host_frame_id"] == 0
                            && record["caller"]["host_cam"] == 0
                            && record["caller"]["target_frame_id"] == target_frame
                            && record["caller"]["target_cam"] == target_cam
                    })
                    .unwrap_or_else(|| panic!("missing relation {target_frame}/{target_cam}"));
                let target_camera_from_imu = extrinsics[target_cam].inverse();
                let target_inverse_rotation = states[target_frame].rotation.inverse();
                let target_imu_from_anchor_imu = F32Pose {
                    rotation: sophus_quat_product_f32(target_inverse_rotation, states[0].rotation),
                    translation: sophus_rotate_difference_f32(
                        target_inverse_rotation,
                        states[0].translation,
                        states[target_frame].translation,
                    ),
                };
                let target_camera_from_anchor_imu = F32Pose {
                    rotation: sophus_quat_product_camera_prefix_f32(
                        target_camera_from_imu.rotation,
                        target_imu_from_anchor_imu.rotation,
                    ),
                    translation: sophus_rotate_f32(
                        target_camera_from_imu.rotation,
                        target_imu_from_anchor_imu.translation,
                    ) + target_camera_from_imu.translation,
                };
                let relative_camera = F32Pose {
                    rotation: sophus_quat_product_camera_suffix_f32(
                        target_camera_from_anchor_imu.rotation,
                        extrinsics[0].rotation,
                    ),
                    translation: sophus_rotate_f32(
                        target_camera_from_anchor_imu.rotation,
                        extrinsics[0].translation,
                    ) + target_camera_from_anchor_imu.translation,
                };
                assert_pose_bits(
                    relative_camera,
                    record,
                    &format!("target={target_frame}/{target_cam} q_t"),
                );
                let anchor_rotation = eigen_quaternion_matrix_f32(states[0].rotation.inverse());
                let target_rotation =
                    eigen_quaternion_matrix_f32(states[target_frame].rotation.inverse());
                let actual_anchor = eigen_adjoint_times_rotation_blocks_f32(
                    target_camera_from_anchor_imu,
                    anchor_rotation,
                    1.0,
                );
                let actual_target = eigen_adjoint_times_rotation_blocks_f32(
                    target_camera_from_imu,
                    target_rotation,
                    -1.0,
                );
                assert_matrix_bits(
                    &actual_anchor,
                    &record["d_rel_d_h"],
                    &format!("target={target_frame}/{target_cam} anchor"),
                );
                assert_matrix_bits(
                    &actual_target,
                    &record["d_rel_d_t"],
                    &format!("target={target_frame}/{target_cam} target"),
                );
            }
        }
    }

    #[test]
    fn m7gu_clean_frame4_point_projection_and_relative_jacobians_are_bitwise_exact() {
        // Keep this fixture bounded to the eleven frame-4 ordinals which still
        // differ after the absolute-pose chain was made exact.  The native
        // capture is the raw `linearizePoint` boundary: target_point4,
        // projection, residual, and d_res_d_xi are checked independently from
        // the caller-owned landmark/absolute-pose buffers.  The selected
        // records live under tests/fixtures so this contract does not depend
        // on a developer-local raw capture.
        let clean: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7gu_clean_frame4_visual.json"
        ))
        .unwrap();
        assert_eq!(
            clean["schema"].as_str(),
            Some("visloc-rs.basalt.m7gu.clean-frame4-visual.v1")
        );
        assert_eq!(
            clean["source"]["upstream_commit"].as_str(),
            Some("0f3b2b52c807f70ff4e2973ce253c73329eea7bc")
        );
        let ordinals = [75_usize, 79, 162, 257, 267, 283, 306, 478, 494, 515, 579];
        assert_eq!(clean["records"].as_array().unwrap().len(), ordinals.len());
        let bits =
            |value: &serde_json::Value| u32::from_str_radix(value.as_str().unwrap(), 16).unwrap();
        let lane_bits = |record: &serde_json::Value, key: &str| {
            record[key]
                .as_array()
                .unwrap()
                .iter()
                .map(bits)
                .collect::<Vec<_>>()
        };
        for ordinal in ordinals {
            let record = clean["records"]
                .as_array()
                .unwrap()
                .iter()
                .find(|record| record["ordinal"].as_u64() == Some(ordinal as u64))
                .unwrap_or_else(|| panic!("missing bounded visual ordinal {ordinal}"));
            let transform_bits = record["T_t_h_f32_bits"].as_array().unwrap();
            let transform = SMatrix::<f32, 4, 4>::from_column_slice(
                &transform_bits
                    .iter()
                    .map(bits)
                    .map(f32::from_bits)
                    .collect::<Vec<_>>(),
            );
            let direction = record["direction2_f32_bits"].as_array().unwrap();
            let u = f32::from_bits(bits(&direction[0]));
            let v = f32::from_bits(bits(&direction[1]));
            let r2 = u * u + v * v;
            let scale = 2.0_f32 / (1.0_f32 + r2);
            let bearing = Vector3::new(u * scale, v * scale, scale - 1.0_f32);
            let rho = f32::from_bits(bits(&record["inv_dist_f32_bits"][0]));
            let point4 = eigen_homogeneous_point_product_f32(transform, bearing, rho);
            let point = point4.fixed_rows::<3>(0).into_owned();
            // The bounded ordinals span both target cameras.  M7ef stores the
            // native camera pointer but not its compact id in each record;
            // retain the keyed target-cam mapping from the clean relation
            // audit here so projection/Jxi are evaluated against the right
            // calibration rather than silently using cam1 for cam0 rows.
            let target_cam = match ordinal {
                79 | 257 | 306 | 494 => 1,
                _ => 0,
            };
            let camera = if target_cam == 0 {
                DoubleSphereCamera::new(
                    349.7560023050409,
                    348.72454229977037,
                    365.89440762590149,
                    249.32995565708704,
                    -0.2409573942178872,
                    0.566996899163044,
                    752,
                    480,
                )
                .unwrap()
            } else {
                m7_fixed_camera()
            };
            let (projection, projection_jacobian) =
                project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
            let pixel = record["pixel2_f32_bits"].as_array().unwrap();
            let observation = Vector2::new(
                f32::from_bits(bits(&pixel[0])),
                f32::from_bits(bits(&pixel[1])),
            );
            let raw = projection - observation;
            let mut point_wrt_relative_pose = SMatrix::<f32, 3, 6>::zeros();
            point_wrt_relative_pose
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(SMatrix::<f32, 3, 3>::identity() * rho));
            point_wrt_relative_pose
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-skew3_f32(point)));
            let d_xi =
                eigen_relative_pose_jacobian_f32(projection_jacobian, point_wrt_relative_pose);
            for (name, actual, expected) in [
                (
                    "point4",
                    point4.as_slice().to_vec(),
                    lane_bits(record, "target_point4_f32_bits"),
                ),
                (
                    "projection",
                    projection.as_slice().to_vec(),
                    lane_bits(record, "projection2_f32_bits"),
                ),
                (
                    "raw",
                    raw.as_slice().to_vec(),
                    lane_bits(record, "raw_residual2_f32_bits"),
                ),
                (
                    "d_res_d_xi",
                    d_xi.as_slice().to_vec(),
                    lane_bits(record, "d_res_d_xi12_f32_bits"),
                ),
            ] {
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "ordinal {ordinal} {name} length"
                );
                for (lane, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        want,
                        "ordinal {ordinal} {name} lane {lane}: {:08x} != {want:08x}",
                        got.to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn m7_homogeneous_relative_point_matches_pinned_lanes() {
        let (rotation, translation, bearing, inverse_distance) = m7_fixed_relative_point();
        let point = sophus_homogeneous_point_f32(rotation, translation, bearing, inverse_distance);
        assert_f32_bits(point.as_slice(), &[0xbf2fc73d, 0xbe94aaaa, 0x3f2ec6b2]);
    }

    #[test]
    fn m7_double_sphere_boundary_matches_pinned_lanes() {
        let (_, _, _, point_inverse_distance) = m7_fixed_relative_point();
        let point = Vector3::new(
            f32::from_bits(0xbf2fc73d),
            f32::from_bits(0xbe94aaaa),
            f32::from_bits(0x3f2ec6b2),
        );
        let (predicted, jacobian) =
            project_double_sphere_with_jacobian_f32(&m7_fixed_camera(), point).unwrap();
        assert_f32_bits(predicted.as_slice(), &[0x41da6d22, 0x42d70d2e]);
        assert_f32_bits(
            jacobian.as_slice(),
            &[
                0x43aa6a6b, 0xc29101bc, 0xc2917182, 0x43f04ca7, 0x439beda6, 0x43037b7f,
            ],
        );
        assert_eq!(point_inverse_distance.to_bits(), 0x3e160ef3);
        let raw = predicted - Vector2::new(27.31320571899414_f32, 106.39038848876953_f32);
        assert_f32_bits(raw.as_slice(), &[0xbc228000, 0x3f915340]);
    }

    #[test]
    fn m7_anchored_factor_matches_pinned_projection_and_raw() {
        let (
            camera,
            anchor_pose,
            anchor_extrinsic,
            target_pose,
            target_extrinsic,
            landmark,
            observation,
        ) = m7_fixed_factor_inputs();
        let factor = anchored_visual_reprojection_factor_f32(
            &camera,
            &anchor_pose,
            &anchor_extrinsic,
            &target_pose,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            FactorConfig::default(),
        )
        .unwrap();
        let projection = factor.projection.map(|value| value as f32);
        let raw = factor.raw_residual.map(|value| value as f32);
        assert_f32_bits(projection.as_slice(), &[0x41da6d17, 0x42d70d32]);
        assert_f32_bits(raw.as_slice(), &[0xbc22d800, 0x3f915440]);
    }

    #[test]
    fn m7ec_clean_track2_stereo_factor_is_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            factor: String,
            direction_xy_f32_bits: [String; 2],
            inverse_distance_f32_bits: String,
            observation_f32_bits: [String; 2],
            q_xyzw_f32_bits: [String; 4],
            t_xyz_f32_bits: [String; 3],
            target_point_xyz_f32_bits: [String; 3],
            projection_uv_f32_bits: [String; 2],
            raw_residual_uv_f32_bits: [String; 2],
            landmark_jacobian_column_major_f32_bits: [String; 6],
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7ec_clean_track2_stereo_factor.json"
        ))
        .expect("m7ec clean track-2 stereo fixture must parse");
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let (camera, anchor_pose, anchor_extrinsic, _target_pose, target_extrinsic, _, _) =
            m7_fixed_factor_inputs();
        let landmark = InverseDistanceLandmark {
            anchor_pose: 0,
            anchor_camera_id: 0,
            direction: StereographicDirection {
                xy: Point2::new(
                    f32::from_bits(bits(&fixture.direction_xy_f32_bits[0])) as f64,
                    f32::from_bits(bits(&fixture.direction_xy_f32_bits[1])) as f64,
                ),
            },
            inverse_distance: f32::from_bits(bits(&fixture.inverse_distance_f32_bits)) as f64,
        };
        let anchor_pose_f32 = F32Pose::from_se3(&anchor_pose);
        let target_pose_f32 = F32Pose::from_se3(&anchor_pose);
        let anchor_extrinsic_f32 = F32Pose::from_se3(&anchor_extrinsic);
        let target_extrinsic_f32 = F32Pose::from_se3(&target_extrinsic);
        let target_camera_from_imu = target_extrinsic_f32.inverse();
        let target_imu_from_anchor_imu = sophus_relative_imu_f32(target_pose_f32, anchor_pose_f32);
        let target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.translation,
            ) + target_camera_from_imu.translation,
        };
        let relative_camera = F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_extrinsic_f32.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_extrinsic_f32.translation,
            ) + target_camera_from_anchor_imu.translation,
        };
        assert_pose_f32_bits(
            relative_camera,
            std::array::from_fn(|index| bits(&fixture.t_xyz_f32_bits[index])),
            std::array::from_fn(|index| bits(&fixture.q_xyzw_f32_bits[index])),
        );
        let point_target = sophus_homogeneous_point_normalized_rotation_f32(
            relative_camera.rotation,
            relative_camera.translation,
            landmark.direction.bearing_f32(),
            landmark.inverse_distance as f32,
        );
        let expected_point = [
            bits(&fixture.target_point_xyz_f32_bits[0]),
            bits(&fixture.target_point_xyz_f32_bits[1]),
            bits(&fixture.target_point_xyz_f32_bits[2]),
        ];
        assert_f32_bits(point_target.as_slice(), &expected_point);
        let observation = Point2::new(
            f32::from_bits(bits(&fixture.observation_f32_bits[0])) as f64,
            f32::from_bits(bits(&fixture.observation_f32_bits[1])) as f64,
        );
        let factor = anchored_visual_reprojection_factor_f32_with_time_cam(
            &camera,
            &anchor_pose,
            &anchor_extrinsic,
            &anchor_pose,
            &target_extrinsic,
            &landmark,
            observation,
            true,
            false,
            FactorConfig::default(),
        )
        .unwrap();
        let projection = factor.projection.map(|value| value as f32);
        let raw = factor.raw_residual.map(|value| value as f32);
        let jl = factor
            .landmark_jacobian
            .map(|value| (value as f32) / (factor.sqrt_weight as f32));
        let expected_projection = [
            bits(&fixture.projection_uv_f32_bits[0]),
            bits(&fixture.projection_uv_f32_bits[1]),
        ];
        let expected_raw = [
            bits(&fixture.raw_residual_uv_f32_bits[0]),
            bits(&fixture.raw_residual_uv_f32_bits[1]),
        ];
        let expected_jl = [
            bits(&fixture.landmark_jacobian_column_major_f32_bits[0]),
            bits(&fixture.landmark_jacobian_column_major_f32_bits[1]),
            bits(&fixture.landmark_jacobian_column_major_f32_bits[2]),
            bits(&fixture.landmark_jacobian_column_major_f32_bits[3]),
            bits(&fixture.landmark_jacobian_column_major_f32_bits[4]),
            bits(&fixture.landmark_jacobian_column_major_f32_bits[5]),
        ];
        assert_f32_bits(projection.as_slice(), &expected_projection);
        assert_f32_bits(raw.as_slice(), &expected_raw);
        assert_f32_bits(jl.as_slice(), &expected_jl);
    }

    #[test]
    fn m7et_ordinal0_jp_packet4_is_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            #[allow(dead_code)]
            camera: String,
            camera_params_f32_bits: [String; 6],
            #[allow(dead_code)]
            observation_f32_bits: [String; 2],
            direction_xy_f32_bits: [String; 2],
            inverse_distance_f32_bits: String,
            target_point_xyz_f32_bits: [String; 3],
            projection_uv_f32_bits: [String; 2],
            camera_jacobian_row_major_f32_bits: [String; 8],
            homogeneous_point_jacobian_column_major_f32_bits: [String; 12],
            raw_landmark_jacobian_column_major_f32_bits: [String; 6],
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7et_ordinal0_jp_packet4.json"
        ))
        .expect("m7et ordinal-0 packet-4 fixture must parse");
        assert_eq!(fixture.ordinal, 0);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let direction = StereographicDirection {
            xy: Point2::new(
                f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
                f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
            ),
        };
        // Ordinal 0 uses an identity target transform, so the active
        // homogeneous Jpp block is the stereographic unprojection Jup
        // itself.  Keep this direct assertion at the first mismatch boundary
        // instead of only checking the downstream packet product.
        let expected_bearing_jacobian = [
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[0]),
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[1]),
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[2]),
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[4]),
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[5]),
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[6]),
        ];
        assert_f32_bits(
            direction.bearing_jacobian_f32().as_slice(),
            &expected_bearing_jacobian,
        );
        let inverse_distance = f32_value(&fixture.inverse_distance_f32_bits);
        let point = sophus_homogeneous_point_normalized_rotation_f32(
            UnitQuaternion::identity(),
            Vector3::zeros(),
            direction.bearing_f32(),
            inverse_distance,
        );
        let expected_point: [u32; 3] =
            std::array::from_fn(|index| bits(&fixture.target_point_xyz_f32_bits[index]));
        assert_f32_bits(point.as_slice(), &expected_point);

        let (projection, projection_jacobian) =
            project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let expected_projection: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
        assert_f32_bits(projection.as_slice(), &expected_projection);

        let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
        camera_jacobian
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&projection_jacobian);
        let expected_camera: [u32; 8] =
            std::array::from_fn(|index| bits(&fixture.camera_jacobian_row_major_f32_bits[index]));
        for row in 0..2 {
            for column in 0..4 {
                let index = row * 4 + column;
                assert_eq!(
                    camera_jacobian[(row, column)].to_bits(),
                    expected_camera[index],
                    "camera J row-major lane {index}: got {:08x}, expected {:08x}",
                    camera_jacobian[(row, column)].to_bits(),
                    expected_camera[index]
                );
            }
        }

        let expected_point_jacobian: [u32; 12] = std::array::from_fn(|index| {
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[index])
        });
        let point_wrt_landmark =
            SMatrix::<f32, 4, 3>::from_column_slice(&expected_point_jacobian.map(f32::from_bits));
        assert_f32_bits(point_wrt_landmark.as_slice(), &expected_point_jacobian);

        let raw_landmark_jacobian =
            eigen_landmark_jacobian_f32(camera_jacobian, point_wrt_landmark);
        let expected_raw: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.raw_landmark_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw);
    }

    #[test]
    fn m7ez_ordinal3_camera_jacobian_is_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            camera_params_f32_bits: [String; 6],
            target_point_xyz_f32_bits: [String; 3],
            projection_uv_f32_bits: [String; 2],
            camera_jacobian_column_major_f32_bits: [String; 6],
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7ez_ordinal3_camera_jacobian.json"
        ))
        .expect("M7ez ordinal-3 camera-J fixture must parse");
        assert_eq!(fixture.ordinal, 3);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let point = Vector3::new(
            f32_value(&fixture.target_point_xyz_f32_bits[0]),
            f32_value(&fixture.target_point_xyz_f32_bits[1]),
            f32_value(&fixture.target_point_xyz_f32_bits[2]),
        );
        let (projection, jacobian) =
            project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let expected_projection: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
        let expected_jacobian: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.camera_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(projection.as_slice(), &expected_projection);
        assert_f32_bits(jacobian.as_slice(), &expected_jacobian);
    }

    #[test]
    fn m7ez_ordinal38_projection_is_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            camera_params_f32_bits: [String; 6],
            target_point_xyz_f32_bits: [String; 3],
            projection_uv_f32_bits: [String; 2],
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7ez_ordinal38_projection.json"
        ))
        .expect("M7ez ordinal-38 projection fixture must parse");
        assert_eq!(fixture.ordinal, 38);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let point = Vector3::new(
            f32_value(&fixture.target_point_xyz_f32_bits[0]),
            f32_value(&fixture.target_point_xyz_f32_bits[1]),
            f32_value(&fixture.target_point_xyz_f32_bits[2]),
        );
        let (projection, _) = project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let expected: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
        assert_f32_bits(projection.as_slice(), &expected);
    }

    #[test]
    fn m7fi_ordinal13_homogeneous_point_and_jp_are_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            #[allow(dead_code)]
            relation: String,
            camera_params_f32_bits: [String; 6],
            direction_xy_f32_bits: [String; 2],
            inverse_distance_f32_bits: String,
            observation_f32_bits: [String; 2],
            sqrt_weight_f32_bits: String,
            transform_t_t_h_f32_bits_column_major: [String; 16],
            target_point4_f32_bits: [String; 4],
            projection_uv_f32_bits: [String; 2],
            raw_residual_uv_f32_bits: [String; 2],
            bearing_j3x2_f32_bits_column_major: [String; 6],
            camera_j2x4_f32_bits_column_major: [String; 8],
            jpp4x3_f32_bits_column_major: [String; 12],
            raw_jp2x3_f32_bits_column_major: [String; 6],
            weighted_jp2x3_f32_bits_column_major: [String; 6],
        }

        let fixture: Fixture =
            serde_json::from_str(include_str!("../../tests/fixtures/m7fi_ordinal13_jp.json"))
                .expect("M7fi ordinal-13 Jp fixture must parse");
        assert_eq!(fixture.ordinal, 13);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let direction = StereographicDirection {
            xy: Point2::new(
                f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
                f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
            ),
        };
        let expected_transform: [u32; 16] = std::array::from_fn(|index| {
            bits(&fixture.transform_t_t_h_f32_bits_column_major[index])
        });
        let transform =
            SMatrix::<f32, 4, 4>::from_column_slice(&expected_transform.map(f32::from_bits));
        assert_f32_bits(transform.as_slice(), &expected_transform);

        let expected_bearing_jacobian: [u32; 6] =
            std::array::from_fn(|index| bits(&fixture.bearing_j3x2_f32_bits_column_major[index]));
        assert_f32_bits(
            direction.bearing_jacobian_f32().as_slice(),
            &expected_bearing_jacobian,
        );
        let inverse_distance = f32_value(&fixture.inverse_distance_f32_bits);
        let point4 = eigen_homogeneous_point_product_f32(
            transform,
            direction.bearing_f32(),
            inverse_distance,
        );
        let expected_point: [u32; 4] =
            std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
        assert_f32_bits(point4.as_slice(), &expected_point);
        let point = point4.fixed_rows::<3>(0).into_owned();

        let (projection, projection_jacobian) =
            project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let expected_projection: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
        assert_f32_bits(projection.as_slice(), &expected_projection);
        let observation = Vector2::new(
            f32_value(&fixture.observation_f32_bits[0]),
            f32_value(&fixture.observation_f32_bits[1]),
        );
        let raw = projection - observation;
        let expected_raw: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
        assert_f32_bits(raw.as_slice(), &expected_raw);

        let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
        camera_jacobian
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&projection_jacobian);
        let expected_camera: [u32; 8] =
            std::array::from_fn(|index| bits(&fixture.camera_j2x4_f32_bits_column_major[index]));
        assert_f32_bits(camera_jacobian.as_slice(), &expected_camera);

        let mut transform_top_left = SMatrix::<f32, 3, 4>::zeros();
        transform_top_left.copy_from(&transform.fixed_view::<3, 4>(0, 0));
        let mut source_jup = SMatrix::<f32, 4, 2>::zeros();
        source_jup
            .fixed_view_mut::<3, 2>(0, 0)
            .copy_from(&direction.bearing_jacobian_f32());
        let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
        point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
            &eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup),
        );
        let translation = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);
        point_wrt_landmark.set_column(2, &translation);
        let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
        homogeneous_point_wrt_landmark
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&point_wrt_landmark);
        homogeneous_point_wrt_landmark[(3, 2)] = 1.0_f32;
        let expected_jpp: [u32; 12] =
            std::array::from_fn(|index| bits(&fixture.jpp4x3_f32_bits_column_major[index]));
        assert_f32_bits(homogeneous_point_wrt_landmark.as_slice(), &expected_jpp);

        let raw_landmark_jacobian =
            eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark);
        let expected_raw_jp: [u32; 6] =
            std::array::from_fn(|index| bits(&fixture.raw_jp2x3_f32_bits_column_major[index]));
        assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw_jp);
        let weighted_landmark_jacobian =
            raw_landmark_jacobian * f32_value(&fixture.sqrt_weight_f32_bits);
        let expected_weighted_jp: [u32; 6] =
            std::array::from_fn(|index| bits(&fixture.weighted_jp2x3_f32_bits_column_major[index]));
        assert_f32_bits(weighted_landmark_jacobian.as_slice(), &expected_weighted_jp);
    }

    #[test]
    fn m7fi_ordinal311_homogeneous_point_and_projection_are_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            #[allow(dead_code)]
            relation: String,
            camera_params_f32_bits: [String; 6],
            direction_xy_f32_bits: [String; 2],
            inverse_distance_f32_bits: String,
            observation_f32_bits: [String; 2],
            transform_t_t_h_f32_bits_column_major: [String; 16],
            target_point4_f32_bits: [String; 4],
            projection_uv_f32_bits: [String; 2],
            raw_residual_uv_f32_bits: [String; 2],
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7fi_ordinal311_projection.json"
        ))
        .expect("M7fi ordinal-311 projection fixture must parse");
        assert_eq!(fixture.ordinal, 311);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let direction = StereographicDirection {
            xy: Point2::new(
                f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
                f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
            ),
        };
        let expected_transform: [u32; 16] = std::array::from_fn(|index| {
            bits(&fixture.transform_t_t_h_f32_bits_column_major[index])
        });
        let transform =
            SMatrix::<f32, 4, 4>::from_column_slice(&expected_transform.map(f32::from_bits));
        assert_f32_bits(transform.as_slice(), &expected_transform);
        let point4 = eigen_homogeneous_point_product_f32(
            transform,
            direction.bearing_f32(),
            f32_value(&fixture.inverse_distance_f32_bits),
        );
        let expected_point: [u32; 4] =
            std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
        assert_f32_bits(point4.as_slice(), &expected_point);
        let point = point4.fixed_rows::<3>(0).into_owned();
        let (projection, _) = project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let expected_projection: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
        assert_f32_bits(projection.as_slice(), &expected_projection);
        let raw = projection
            - Vector2::new(
                f32_value(&fixture.observation_f32_bits[0]),
                f32_value(&fixture.observation_f32_bits[1]),
            );
        let expected_raw: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
        assert_f32_bits(raw.as_slice(), &expected_raw);
    }

    #[test]
    fn m7fa_ordinal43_transform_point_and_jp_are_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            #[allow(dead_code)]
            relation: String,
            camera_params_f32_bits: [String; 6],
            direction_xy_f32_bits: [String; 2],
            inverse_distance_f32_bits: String,
            observation_f32_bits: [String; 2],
            q_xyzw_f32_bits: [String; 4],
            t_xyz_f32_bits: [String; 3],
            transform_t_t_h_f32_bits_column_major: [String; 16],
            target_point4_f32_bits: [String; 4],
            projection_uv_f32_bits: [String; 2],
            raw_residual_uv_f32_bits: [String; 2],
            camera_jacobian_column_major_f32_bits: [String; 8],
            homogeneous_point_jacobian_column_major_f32_bits: [String; 12],
            raw_landmark_jacobian_column_major_f32_bits: [String; 6],
            weighted_landmark_jacobian_column_major_f32_bits: [String; 6],
        }

        let fixture: Fixture =
            serde_json::from_str(include_str!("../../tests/fixtures/m7fa_ordinal43_jp.json"))
                .expect("M7fa ordinal-43 fixture must parse");
        assert_eq!(fixture.ordinal, 43);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32_value(&fixture.q_xyzw_f32_bits[3]),
            f32_value(&fixture.q_xyzw_f32_bits[0]),
            f32_value(&fixture.q_xyzw_f32_bits[1]),
            f32_value(&fixture.q_xyzw_f32_bits[2]),
        ));
        let translation = Vector3::new(
            f32_value(&fixture.t_xyz_f32_bits[0]),
            f32_value(&fixture.t_xyz_f32_bits[1]),
            f32_value(&fixture.t_xyz_f32_bits[2]),
        );
        let rotation_matrix = eigen_quaternion_matrix_native_f32(rotation);
        let expected_transform: [u32; 16] = std::array::from_fn(|index| {
            bits(&fixture.transform_t_t_h_f32_bits_column_major[index])
        });
        let mut transform = SMatrix::<f32, 4, 4>::identity();
        transform
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation_matrix);
        transform
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&translation);
        assert_f32_bits(transform.as_slice(), &expected_transform);

        let direction = StereographicDirection {
            xy: Point2::new(
                f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
                f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
            ),
        };
        let inverse_distance = f32_value(&fixture.inverse_distance_f32_bits);
        let point = sophus_homogeneous_point_normalized_rotation_f32(
            rotation,
            translation,
            direction.bearing_f32(),
            inverse_distance,
        );
        let expected_point: [u32; 3] =
            std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
        assert_f32_bits(point.as_slice(), &expected_point);

        let (projection, projection_jacobian) =
            project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let expected_projection: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
        assert_f32_bits(projection.as_slice(), &expected_projection);
        let observation = Vector2::new(
            f32_value(&fixture.observation_f32_bits[0]),
            f32_value(&fixture.observation_f32_bits[1]),
        );
        let raw = projection - observation;
        let expected_raw: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
        assert_f32_bits(raw.as_slice(), &expected_raw);

        let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
        camera_jacobian
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&projection_jacobian);
        let expected_camera: [u32; 8] = std::array::from_fn(|index| {
            bits(&fixture.camera_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(camera_jacobian.as_slice(), &expected_camera);

        let direction_jacobian = direction.bearing_jacobian_f32();
        let mut transform_top_left = SMatrix::<f32, 3, 4>::zeros();
        transform_top_left
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation_matrix);
        let mut source_jup = SMatrix::<f32, 4, 2>::zeros();
        source_jup
            .fixed_view_mut::<3, 2>(0, 0)
            .copy_from(&direction_jacobian);
        let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
        point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
            &eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup),
        );
        point_wrt_landmark.set_column(2, &translation);
        let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
        homogeneous_point_wrt_landmark
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&point_wrt_landmark);
        homogeneous_point_wrt_landmark[(3, 2)] = 1.0;
        let expected_jpp: [u32; 12] = std::array::from_fn(|index| {
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(homogeneous_point_wrt_landmark.as_slice(), &expected_jpp);

        let raw_landmark_jacobian =
            eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark);
        let expected_raw_jp: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.raw_landmark_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw_jp);
        let weighted_landmark_jacobian = raw_landmark_jacobian * 2.0_f32;
        let expected_weighted_jp: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.weighted_landmark_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(weighted_landmark_jacobian.as_slice(), &expected_weighted_jp);
    }

    #[test]
    fn m7fq_live_ordinal105_bearing_jacobian_and_jp_are_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            #[allow(dead_code)]
            relation: String,
            camera_params_f32_bits: [String; 6],
            direction_xy_f32_bits: [String; 2],
            inverse_distance_f32_bits: String,
            observation_f32_bits: [String; 2],
            q_xyzw_f32_bits: [String; 4],
            t_xyz_f32_bits: [String; 3],
            transform_t_t_h_f32_bits_column_major: [String; 16],
            target_point4_f32_bits: [String; 4],
            projection_uv_f32_bits: [String; 2],
            raw_residual_uv_f32_bits: [String; 2],
            bearing_jacobian_column_major_f32_bits: [String; 6],
            camera_jacobian_column_major_f32_bits: [String; 8],
            homogeneous_point_jacobian_column_major_f32_bits: [String; 12],
            raw_landmark_jacobian_column_major_f32_bits: [String; 6],
            weighted_landmark_jacobian_column_major_f32_bits: [String; 6],
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7fq_live_ordinal105.json"
        ))
        .expect("M7fq live ordinal-105 fixture must parse");
        assert_eq!(fixture.ordinal, 105);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32_value(&fixture.q_xyzw_f32_bits[3]),
            f32_value(&fixture.q_xyzw_f32_bits[0]),
            f32_value(&fixture.q_xyzw_f32_bits[1]),
            f32_value(&fixture.q_xyzw_f32_bits[2]),
        ));
        let translation = Vector3::new(
            f32_value(&fixture.t_xyz_f32_bits[0]),
            f32_value(&fixture.t_xyz_f32_bits[1]),
            f32_value(&fixture.t_xyz_f32_bits[2]),
        );
        let rotation_matrix = eigen_quaternion_matrix_native_f32(rotation);
        let expected_transform: [u32; 16] = std::array::from_fn(|index| {
            bits(&fixture.transform_t_t_h_f32_bits_column_major[index])
        });
        let mut transform = SMatrix::<f32, 4, 4>::identity();
        transform
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation_matrix);
        transform
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&translation);
        assert_f32_bits(transform.as_slice(), &expected_transform);

        let direction = StereographicDirection {
            xy: Point2::new(
                f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
                f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
            ),
        };
        let direction_jacobian = direction.bearing_jacobian_f32();
        let expected_bearing_jacobian: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.bearing_jacobian_column_major_f32_bits[index])
        });
        // These are the actual live in-context Jpp operands' active bearing
        // lanes.  In particular, lane 0 is 3fba8d4f, not the isolated M7fn
        // replay's 3fba8d50.
        assert_f32_bits(direction_jacobian.as_slice(), &expected_bearing_jacobian);

        let inverse_distance = f32_value(&fixture.inverse_distance_f32_bits);
        let point = sophus_homogeneous_point_normalized_rotation_f32(
            rotation,
            translation,
            direction.bearing_f32(),
            inverse_distance,
        );
        let expected_point: [u32; 3] =
            std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
        assert_f32_bits(point.as_slice(), &expected_point);

        let (projection, projection_jacobian) =
            project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let expected_projection: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
        assert_f32_bits(projection.as_slice(), &expected_projection);
        let observation = Vector2::new(
            f32_value(&fixture.observation_f32_bits[0]),
            f32_value(&fixture.observation_f32_bits[1]),
        );
        let raw = projection - observation;
        let expected_raw: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
        assert_f32_bits(raw.as_slice(), &expected_raw);

        let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
        camera_jacobian
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&projection_jacobian);
        let expected_camera: [u32; 8] = std::array::from_fn(|index| {
            bits(&fixture.camera_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(camera_jacobian.as_slice(), &expected_camera);

        let mut transform_top_left = SMatrix::<f32, 3, 4>::zeros();
        transform_top_left
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation_matrix);
        let mut source_jup = SMatrix::<f32, 4, 2>::zeros();
        source_jup
            .fixed_view_mut::<3, 2>(0, 0)
            .copy_from(&direction_jacobian);
        let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
        point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
            &eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup),
        );
        point_wrt_landmark.set_column(2, &translation);
        let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
        homogeneous_point_wrt_landmark
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&point_wrt_landmark);
        homogeneous_point_wrt_landmark[(3, 2)] = 1.0;
        let expected_jpp: [u32; 12] = std::array::from_fn(|index| {
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[index])
        });
        // This is the live pre-product Jpp capture, including its one-ULP
        // lane-0 difference from the old isolated replay.
        assert_f32_bits(homogeneous_point_wrt_landmark.as_slice(), &expected_jpp);

        let raw_landmark_jacobian =
            eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark);
        let expected_raw_jp: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.raw_landmark_jacobian_column_major_f32_bits[index])
        });
        // This is the actual live post-product Jp capture, rather than the
        // isolated M7fn product's surrogate operand/result.
        assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw_jp);
        let weighted_landmark_jacobian = raw_landmark_jacobian * 2.0_f32;
        let expected_weighted_jp: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.weighted_landmark_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(weighted_landmark_jacobian.as_slice(), &expected_weighted_jp);
    }

    #[test]
    fn m7fu_live_ordinal109_camera_jacobian_and_jp_are_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            #[allow(dead_code)]
            relation: String,
            camera_params_f32_bits: [String; 6],
            target_point_xyz_f32_bits: [String; 3],
            camera_jacobian_column_major_f32_bits: [String; 8],
            homogeneous_point_jacobian_column_major_f32_bits: [String; 12],
            raw_landmark_jacobian_column_major_f32_bits: [String; 6],
            sqrt_weight_f32_bits: String,
            weighted_landmark_jacobian_column_major_f32_bits: [String; 6],
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7fu_live_ordinal109_camera_jp.json"
        ))
        .expect("M7fu live ordinal-109 camera-J/Jp fixture must parse");
        assert_eq!(fixture.ordinal, 109);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let point = Vector3::new(
            f32_value(&fixture.target_point_xyz_f32_bits[0]),
            f32_value(&fixture.target_point_xyz_f32_bits[1]),
            f32_value(&fixture.target_point_xyz_f32_bits[2]),
        );
        let (_, projection_jacobian) =
            project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
        camera_jacobian
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&projection_jacobian);
        let expected_camera: [u32; 8] = std::array::from_fn(|index| {
            bits(&fixture.camera_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(camera_jacobian.as_slice(), &expected_camera);

        let expected_jpp: [u32; 12] = std::array::from_fn(|index| {
            bits(&fixture.homogeneous_point_jacobian_column_major_f32_bits[index])
        });
        let point_wrt_landmark =
            SMatrix::<f32, 4, 3>::from_column_slice(&expected_jpp.map(f32::from_bits));
        assert_f32_bits(point_wrt_landmark.as_slice(), &expected_jpp);

        let raw_landmark_jacobian =
            eigen_landmark_jacobian_f32(camera_jacobian, point_wrt_landmark);
        let expected_raw: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.raw_landmark_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(raw_landmark_jacobian.as_slice(), &expected_raw);

        let weighted_landmark_jacobian =
            raw_landmark_jacobian * f32_value(&fixture.sqrt_weight_f32_bits);
        let expected_weighted: [u32; 6] = std::array::from_fn(|index| {
            bits(&fixture.weighted_landmark_jacobian_column_major_f32_bits[index])
        });
        assert_f32_bits(weighted_landmark_jacobian.as_slice(), &expected_weighted);
    }

    #[test]
    fn m7fb_ordinal110_transform_point_and_projection_are_bitwise_exact() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            #[allow(dead_code)]
            schema: String,
            #[allow(dead_code)]
            source: String,
            ordinal: u32,
            #[allow(dead_code)]
            relation: String,
            camera_params_f32_bits: [String; 6],
            direction_xy_f32_bits: [String; 2],
            inverse_distance_f32_bits: String,
            observation_f32_bits: [String; 2],
            q_xyzw_f32_bits: [String; 4],
            t_xyz_f32_bits: [String; 3],
            transform_t_t_h_f32_bits_column_major: [String; 16],
            target_point4_f32_bits: [String; 4],
            projection_uv_f32_bits: [String; 2],
            raw_residual_uv_f32_bits: [String; 2],
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7fb_ordinal110_projection.json"
        ))
        .expect("M7fb ordinal-110 fixture must parse");
        assert_eq!(fixture.ordinal, 110);
        let bits = |value: &str| {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("f32 fixture bit pattern")
        };
        let f32_value = |value: &str| f32::from_bits(bits(value));
        let camera = DoubleSphereCamera::new(
            f32_value(&fixture.camera_params_f32_bits[0]) as f64,
            f32_value(&fixture.camera_params_f32_bits[1]) as f64,
            f32_value(&fixture.camera_params_f32_bits[2]) as f64,
            f32_value(&fixture.camera_params_f32_bits[3]) as f64,
            f32_value(&fixture.camera_params_f32_bits[4]) as f64,
            f32_value(&fixture.camera_params_f32_bits[5]) as f64,
            752,
            480,
        )
        .unwrap();
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32_value(&fixture.q_xyzw_f32_bits[3]),
            f32_value(&fixture.q_xyzw_f32_bits[0]),
            f32_value(&fixture.q_xyzw_f32_bits[1]),
            f32_value(&fixture.q_xyzw_f32_bits[2]),
        ));
        let translation = Vector3::new(
            f32_value(&fixture.t_xyz_f32_bits[0]),
            f32_value(&fixture.t_xyz_f32_bits[1]),
            f32_value(&fixture.t_xyz_f32_bits[2]),
        );
        let rotation_matrix = eigen_quaternion_matrix_native_f32(rotation);
        let mut transform = SMatrix::<f32, 4, 4>::identity();
        transform
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&rotation_matrix);
        transform
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&translation);
        let expected_transform: [u32; 16] = std::array::from_fn(|index| {
            bits(&fixture.transform_t_t_h_f32_bits_column_major[index])
        });
        assert_f32_bits(transform.as_slice(), &expected_transform);

        let direction = StereographicDirection {
            xy: Point2::new(
                f32_value(&fixture.direction_xy_f32_bits[0]) as f64,
                f32_value(&fixture.direction_xy_f32_bits[1]) as f64,
            ),
        };
        let point = sophus_homogeneous_point_normalized_rotation_f32(
            rotation,
            translation,
            direction.bearing_f32(),
            f32_value(&fixture.inverse_distance_f32_bits),
        );
        let expected_point: [u32; 3] =
            std::array::from_fn(|index| bits(&fixture.target_point4_f32_bits[index]));
        assert_f32_bits(point.as_slice(), &expected_point);
        let (projection, _) = project_double_sphere_with_jacobian_f32(&camera, point).unwrap();
        let expected_projection: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.projection_uv_f32_bits[index]));
        assert_f32_bits(projection.as_slice(), &expected_projection);
        let raw = projection
            - Vector2::new(
                f32_value(&fixture.observation_f32_bits[0]),
                f32_value(&fixture.observation_f32_bits[1]),
            );
        let expected_raw: [u32; 2] =
            std::array::from_fn(|index| bits(&fixture.raw_residual_uv_f32_bits[index]));
        assert_f32_bits(raw.as_slice(), &expected_raw);
    }

    #[test]
    fn same_time_cam_id_has_exact_identity_and_stereo_keeps_extrinsic() {
        // This fixture deliberately gives the host and target the same frame
        // but nontrivial, otherwise unrelated state/calibration values.  A
        // same-camera observation must nevertheless be identical to an
        // all-identity chain: upstream branches on the complete TimeCamId,
        // rather than relying on the four SE(3) factors to numerically cancel.
        let camera =
            DoubleSphereCamera::new(300.0, 301.0, 320.0, 240.0, 0.4, 0.7, 640, 480).unwrap();
        let pose = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.03, -0.02, 0.01)),
            Vector3::new(0.2, -0.1, 0.4),
        );
        let anchor_extrinsic = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.01, 0.02, 0.03)),
            Vector3::new(0.04, -0.03, 0.02),
        );
        let target_extrinsic = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.02, 0.01, -0.02)),
            Vector3::new(-0.08, 0.01, 0.03),
        );
        let landmark = InverseDistanceLandmark {
            anchor_pose: 7,
            anchor_camera_id: 0,
            direction: StereographicDirection {
                xy: Point2::new(0.08, -0.04),
            },
            inverse_distance: 0.3,
        };
        let observation = Point2::new(321.0, 239.0);
        let config = FactorConfig {
            observation_stddev: 1.0,
            huber_delta: 0.0,
            outlier_threshold: f64::INFINITY,
        };

        let same_camera = anchored_visual_reprojection_factor_f32_with_time_cam(
            &camera,
            &pose,
            &anchor_extrinsic,
            &SE3::new(pose.rotation, pose.translation),
            &target_extrinsic,
            &landmark,
            observation,
            true,
            true,
            config,
        )
        .unwrap();
        let identity_chain = anchored_visual_reprojection_factor_f32_with_time_cam(
            &camera,
            &SE3::identity(),
            &SE3::identity(),
            &SE3::identity(),
            &SE3::identity(),
            &landmark,
            observation,
            true,
            true,
            config,
        )
        .unwrap();
        assert_eq!(same_camera.projection, identity_chain.projection);
        assert_eq!(same_camera.raw_residual, identity_chain.raw_residual);
        assert_eq!(same_camera.residual, identity_chain.residual);
        assert_eq!(
            same_camera.landmark_jacobian,
            identity_chain.landmark_jacobian
        );
        assert!(same_camera
            .anchor_pose_jacobian
            .iter()
            .all(|value| *value == 0.0));
        assert!(same_camera
            .target_pose_jacobian
            .iter()
            .all(|value| *value == 0.0));

        // The other camera at the same timestamp keeps the relative
        // extrinsic transform.  Compare it to a temporal evaluation with
        // the same equal poses: only the timestamp predicate changes, so the
        // complete factor—including both pose rows—must be identical.
        let stereo = anchored_visual_reprojection_factor_f32_with_time_cam(
            &camera,
            &pose,
            &anchor_extrinsic,
            &pose,
            &target_extrinsic,
            &landmark,
            observation,
            true,
            false,
            config,
        )
        .unwrap();
        let temporal = anchored_visual_reprojection_factor_f32_with_time_cam(
            &camera,
            &pose,
            &anchor_extrinsic,
            &pose,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            false,
            config,
        )
        .unwrap();
        assert_eq!(stereo.projection, temporal.projection);
        assert_eq!(stereo.raw_residual, temporal.raw_residual);
        assert_eq!(stereo.residual, temporal.residual);
        assert_eq!(stereo.landmark_jacobian, temporal.landmark_jacobian);
        assert_eq!(stereo.anchor_pose_jacobian, temporal.anchor_pose_jacobian);
        assert_eq!(stereo.target_pose_jacobian, temporal.target_pose_jacobian);
        assert!(temporal.anchor_pose_jacobian.norm() > 0.0);
        assert!(temporal.target_pose_jacobian.norm() > 0.0);
    }

    #[test]
    fn fej_anchor_endpoint_regression_runs_in_fresh_process() {
        let executable = std::env::current_exe().expect("test executable path");
        let mut command = std::process::Command::new(executable);
        // The diagnostic environment is process-lifetime state.  Remove all
        // Basalt keys before installing the child marker and chain switch so
        // an unrelated test or parent shell cannot decide this child path.
        for (key, _) in std::env::vars_os() {
            if key
                .to_string_lossy()
                .get(.."VISLOC_BASALT_".len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("VISLOC_BASALT_"))
            {
                command.env_remove(key);
            }
        }
        let output = command
            .env("VISLOC_BASALT_FEJ_ANCHOR_ENDPOINT_CHILD", "1")
            .env("VISLOC_BASALT_VISUAL_CHAIN_TRACE", "1")
            .args(["fej_anchor_endpoint_regression_child", "--nocapture"])
            .output()
            .expect("spawn fresh FEJ endpoint regression child");
        assert!(
            output.status.success(),
            "FEJ endpoint child failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn fej_anchor_endpoint_regression_child() {
        if std::env::var_os("VISLOC_BASALT_FEJ_ANCHOR_ENDPOINT_CHILD").is_none() {
            return;
        }
        let (
            camera,
            anchor_pose,
            anchor_extrinsic,
            target_pose,
            target_extrinsic,
            landmark,
            observation,
        ) = m7_fixed_factor_inputs();
        let anchor_pose_fej = anchor_pose.clone();
        let target_pose_fej = target_pose.clone();
        let mut moved_anchor_pose = anchor_pose.clone();
        moved_anchor_pose.translation.x += 0.125;
        let config = FactorConfig {
            observation_stddev: 1.0,
            huber_delta: 0.0,
            outlier_threshold: f64::INFINITY,
        };
        let baseline = anchored_visual_reprojection_factor_f32_with_time_cam_fej(
            &camera,
            &anchor_pose,
            &anchor_pose_fej,
            &anchor_extrinsic,
            &target_pose,
            &target_pose_fej,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            false,
            config,
        )
        .expect("baseline FEJ factor");
        let moved = anchored_visual_reprojection_factor_f32_with_time_cam_fej(
            &camera,
            &moved_anchor_pose,
            &anchor_pose_fej,
            &anchor_extrinsic,
            &target_pose,
            &target_pose_fej,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            false,
            config,
        )
        .expect("moved current-value FEJ factor");
        let baseline_chain = baseline
            .debug_chain
            .as_ref()
            .expect("visual chain sidecar enabled");
        let moved_chain = moved
            .debug_chain
            .as_ref()
            .expect("visual chain sidecar enabled");
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };

        // Current-value motion must still affect the value-side projection,
        // while the frozen FEJ endpoint and the production anchor matrix stay
        // fixed.  The latter assertion rejects a regression to the removed
        // value-side `for_anchor_j` reconstruction.
        assert_ne!(baseline.projection, moved.projection);
        assert_ne!(
            baseline_chain.target_camera_from_anchor_imu,
            moved_chain.target_camera_from_anchor_imu
        );
        assert_eq!(
            bits(&baseline_chain.target_camera_from_anchor_imu_fej),
            bits(&moved_chain.target_camera_from_anchor_imu_fej)
        );
        assert_eq!(
            bits(&baseline_chain.relative_wrt_anchor),
            bits(&moved_chain.relative_wrt_anchor)
        );
    }

    #[test]
    fn frame4_mh01_huber_objective_matches_upstream_golden() {
        // Actual MH01 frame-4 landmark trace at the pinned upstream SHA:
        // target frame 1403636579963555584, camera 0, pixel
        // (649.7265625, 176.98886108398438).
        let raw = Vector2::new(-0.44049072265625_f64, 0.9515228271484375_f64);
        let huber_weight = 0.95371061563491821_f64;
        let objective = robust_objective_from_raw(
            raw.norm_squared(),
            huber_weight,
            FactorConfig::default().observation_stddev,
        );
        assert!((objective - 2.194144031284797).abs() < 1e-12);
        // The row itself is still sqrt(w)/sigma weighted; only the objective
        // uses the Huber loss correction.
        let whitened_norm_squared = raw.norm_squared() * huber_weight / 0.25;
        assert!((whitened_norm_squared - 4.19414373130865).abs() < 1e-12);
        assert!(objective < whitened_norm_squared);
    }

    #[test]
    fn m7_frame4_visual_f32_vector2_norm_matches_eigen_packet() {
        // Track 120, frame 4 / iteration 0 / target frame 4 cam 0.  This is
        // the first clean factor whose residual crosses the Huber boundary.
        // The pinned Double-Sphere assembly computes x*x first and contracts
        // y*y + x2.  The standalone nalgebra reduction is one ulp different
        // here on this fixture.
        let x = f32::from_bits(0xbee18800);
        let y = f32::from_bits(0x3f739700);
        let squared = eigen_vector2_squared_norm_f32(x, y);
        assert_eq!(squared.to_bits(), 0x3f8cba0d);

        let huber_weight = 1.0_f32 / squared.sqrt();
        let sqrt_weight = huber_weight.sqrt() / 0.5_f32;
        assert_eq!(sqrt_weight.to_bits(), 0x3ffa0138);
        assert_eq!((x * sqrt_weight).to_bits(), 0xbf5c3fe3);
        assert_eq!((y * sqrt_weight).to_bits(), 0x3fede29f);

        // Asymmetric cross-time witness: clean native Huber capture for
        // track 73 / observation 2.  This distinguishes y.mul_add(y, x*x)
        // from the opposite FMA orientation while checking every scalar
        // robust boundary used by the production factor.
        let x = f32::from_bits(0xbe24d800);
        let y = f32::from_bits(0x3f9279c0);
        let squared = eigen_vector2_squared_norm_f32(x, y);
        assert_eq!(squared.to_bits(), 0x3faaef5d);
        let norm = squared.sqrt();
        assert_eq!(norm.to_bits(), 0x3f93eaf6);
        let huber_weight = 1.0_f32 / norm;
        assert_eq!(huber_weight.to_bits(), 0x3f5d8746);
        let sqrt_numerator = huber_weight.sqrt();
        assert_eq!(sqrt_numerator.to_bits(), 0x3f6e242c);
        assert_eq!((sqrt_numerator / 0.5_f32).to_bits(), 0x3fee242c);
    }

    #[test]
    fn ordering_is_pose_velocity_bias_bias() {
        assert_eq!(AomBlock::Pose6.offset(), 0);
        assert_eq!(AomBlock::Velocity3.offset(), 6);
        assert_eq!(AomBlock::GyroBias3.offset(), 9);
        assert_eq!(AomBlock::AccelBias3.offset(), 12);
        assert_eq!(AomBlock::Pose6.dof(), 6);
        assert_eq!(AOM_NAV_DOF, 15);
    }
    #[test]
    fn nullspace_reduction_matches_dense_schur_reference() {
        let f = factor(
            &[1., 0., 2., 1., 0., 1., 1., -1., 2., 1., 0., 1.],
            &[1., 0., 0., 1., 1., 1.],
            &[2., 1., 3.],
            3,
            4,
            2,
        );
        let red = reduce_landmark_factors(&[f.clone()], 4, 1e-10);
        let a = &f.state_jacobian;
        let l = &f.landmark_jacobian;
        let rr = &f.residual;
        let dense = a.transpose() * a
            - a.transpose() * l * (l.transpose() * l).try_inverse().unwrap() * l.transpose() * a;
        let db = a.transpose() * rr
            - a.transpose() * l * (l.transpose() * l).try_inverse().unwrap() * l.transpose() * rr;
        assert!((&red.h - dense).norm() < 1e-10);
        assert!((&red.b - db).norm() < 1e-10);
    }

    #[test]
    fn abs_qr_keeps_observation_rows_after_zero_landmark_damping_rows() {
        // Basalt's ABS_QR storage appends one zero row per landmark column
        // before applying the landmark QR.  The three Q1 rows are removed,
        // leaving all original observation rows in Q2.  A factor with four
        // observations and a three-column landmark is the smallest fixture
        // that makes the row-span contract visible.
        let f = factor(
            &[0., 0., 0., 0.],
            &[1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.],
            &[1., 2., 3., 4.],
            4,
            1,
            3,
        );
        let (reduced_jacobian, reduced_residual, _) = landmark_nullspace_projection(&f, 1e-10);
        assert_eq!(reduced_jacobian.nrows(), 4);
        assert_eq!(reduced_residual.len(), 4);
    }

    #[test]
    fn compact_landmark_backsub_full_rank_matches_legacy_bitwise() {
        let rows = 8;
        let state_cols = 11;
        let state = (0..rows * state_cols)
            .map(|index| (index as f64 - 13.0) / 17.0)
            .collect::<Vec<_>>();
        let landmark = (0..rows * 3)
            .map(|index| {
                let row = index / 3;
                let column = index % 3;
                match column {
                    0 => {
                        if row == 0 {
                            1.25
                        } else {
                            0.125 * (row + 1) as f64
                        }
                    }
                    1 => {
                        if row == 1 {
                            -1.5
                        } else {
                            0.09375 * (row + 2) as f64
                        }
                    }
                    _ => {
                        if row == 2 {
                            0.75
                        } else {
                            -0.0625 * (row + 3) as f64
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
        let residual = (0..rows)
            .map(|row| (row as f64 - 2.0) * 0.1875)
            .collect::<Vec<_>>();
        let factor = factor(&state, &landmark, &residual, rows, state_cols, 3);
        let state_step = DVector::from_iterator(
            state_cols,
            (0..state_cols).map(|column| (column as f64 - 4.0) * 0.03125),
        );

        let legacy =
            back_substitute_landmark_upstream_f32_with_track(&factor, &state_step, 1e-10, Some(73))
                .expect("synthetic landmark block must be full rank");
        let compact =
            compact_landmark_back_substitution_f32(&factor, 5, 73, 1e-10).expect("compact payload");
        assert_eq!(compact.landmark_index, 5);
        assert_eq!(compact.track_id, 73);
        assert_eq!(compact.rank, 3);
        assert!(compact.eligible);
        assert_eq!(compact.q1_state_shape(), (3, state_cols));
        assert_eq!(compact.q1_residual_len(), 3);
        assert_eq!(compact.upper_r_shape(), (3, 3));

        let recovered = back_substitute_landmark_compact_f32(&compact, &state_step, 1e-10)
            .expect("compact landmark recovery");
        assert_eq!(recovered.len(), legacy.len());
        for (index, (&actual, &expected)) in recovered.iter().zip(legacy.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "landmark increment {index} differs"
            );
        }

        // The compact fields are extraction-only views of the same QR
        // storage consumed by the legacy path, including signed zeroes.
        let state32 = as_f32_matrix(&factor.state_jacobian);
        let landmark32 = as_f32_matrix(&factor.landmark_jacobian);
        let residual32 = as_f32_vector(&factor.residual);
        let qr = LandmarkHouseholderF32::factor(&state32, &landmark32, &residual32).unwrap();
        for row in 0..3 {
            for column in 0..state_cols {
                assert_eq!(
                    compact.q1_state_value(row, column).to_bits(),
                    qr.storage[qr.index(row, column)].to_bits()
                );
            }
            assert_eq!(
                compact.q1_residual_value(row).to_bits(),
                qr.storage[qr.index(row, qr.residual_offset)].to_bits()
            );
            for column in 0..3 {
                let expected = if column < row {
                    0.0_f32
                } else {
                    qr.storage[qr.index(row, qr.landmark_offset + column)]
                };
                assert_eq!(
                    compact.upper_r_value(row, column).to_bits(),
                    expected.to_bits()
                );
            }
        }
    }

    #[test]
    fn compact_landmark_backsub_rank_deficiency_fails_closed_like_legacy() {
        let factor = factor(
            &[0.25, -0.5, 0.75, 1.0, -1.25, 1.5, 1.75, -2.0],
            &[1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 3.0, 4.0, 0.0, 4.0],
            &[1.0, -2.0, 3.0, -4.0],
            4,
            2,
            3,
        );
        let state_step = DVector::from_row_slice(&[0.125, -0.25]);
        let compact = compact_landmark_back_substitution_f32(&factor, 9, 101, 1e-10)
            .expect("rank-deficient payload remains inspectable");
        assert!(compact.rank < 3);
        assert!(!compact.eligible);
        assert!(
            back_substitute_landmark_compact_f32(&compact, &state_step, 1e-10).is_none(),
            "ineligible compact payload must not be solved"
        );
        assert!(
            back_substitute_landmark_upstream_f32_with_track(&factor, &state_step, 1e-10, None)
                .is_none(),
            "legacy rank-deficient path must also fail"
        );
    }

    #[test]
    #[ignore = "requires pinned external frame-4 Householder capture"]
    fn m7_householder_track1_fixture_is_bitwise_eigen_compatible() {
        // This is the frame-4/track-1/iteration-0 storage captured from the
        // pinned upstream ABS_QR run.  Keep this gate next to the helper so a
        // production call-site cannot silently fall back to nalgebra QR (or
        // change the packet/reduction order) without failing the exact trace.
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7_householder_track1_f4_i0.txt");
        let fixture = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
        let mut tokens = fixture.split_whitespace();
        let rows: usize = tokens.next().unwrap().parse().unwrap();
        let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!((rows, storage_cols), (15, 80));
        let input = (0..rows * storage_cols)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(tokens.next(), Some("Q1_JP"));
        let expected_q1_state = (0..3 * 75)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(tokens.next(), Some("Q1_JL"));
        let expected_q1_landmark = (0..9)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(tokens.next(), Some("Q1_R"));
        let expected_q1_residual = (0..3)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        assert!(tokens.next().is_none());

        let state = DMatrix::from_fn(12, 75, |row, column| input[row * storage_cols + column]);
        let landmark =
            DMatrix::from_fn(12, 3, |row, column| input[row * storage_cols + 76 + column]);
        let residual =
            DVector::from_iterator(12, (0..12).map(|row| input[row * storage_cols + 79]));
        let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual).unwrap();
        let fixture_factor = WhitenedFactorRowStack::new(
            state.map(f64::from),
            landmark.map(f64::from),
            residual.map(f64::from),
        )
        .unwrap();
        let compact = compact_landmark_back_substitution_f32(&fixture_factor, 0, 1, 1e-10)
            .expect("fixture compact payload");
        assert_eq!(compact.rank, 3);
        assert!(compact.eligible);
        for row in 0..3 {
            for column in 0..75 {
                assert_eq!(
                    compact.q1_state_value(row, column).to_bits(),
                    qr.storage[qr.index(row, column)].to_bits()
                );
            }
            assert_eq!(
                compact.q1_residual_value(row).to_bits(),
                qr.storage[qr.index(row, qr.residual_offset)].to_bits()
            );
            for column in 0..3 {
                let expected = if column < row {
                    0.0_f32
                } else {
                    qr.storage[qr.index(row, qr.landmark_offset + column)]
                };
                assert_eq!(
                    compact.upper_r_value(row, column).to_bits(),
                    expected.to_bits()
                );
            }
        }
        let fixture_step =
            DVector::from_iterator(75, (0..75).map(|column| (column as f64 - 37.0) * 0.015625));
        let legacy = back_substitute_landmark_upstream_f32_with_track(
            &fixture_factor,
            &fixture_step,
            1e-10,
            Some(1),
        )
        .expect("fixture legacy recovery");
        let recovered = back_substitute_landmark_compact_f32(&compact, &fixture_step, 1e-10)
            .expect("fixture compact recovery");
        for (index, (&actual, &expected)) in recovered.iter().zip(legacy.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "fixture landmark increment {index} differs"
            );
        }

        let mut actual_q1_state = Vec::with_capacity(3 * 75);
        let mut actual_q1_landmark = Vec::with_capacity(9);
        let mut actual_q1_residual = Vec::with_capacity(3);
        for row in 0..3 {
            for column in 0..75 {
                actual_q1_state.push(qr.storage[qr.index(row, column)]);
            }
            for column in 0..3 {
                actual_q1_landmark.push(qr.storage[qr.index(row, qr.landmark_offset + column)]);
            }
            actual_q1_residual.push(qr.storage[qr.index(row, qr.residual_offset)]);
        }
        let count_mismatches = |actual: &[f32], expected: &[f32]| {
            actual
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                .count()
        };
        assert_eq!(
            count_mismatches(&actual_q1_state, &expected_q1_state),
            0,
            "q1 state bit mismatches"
        );
        assert_eq!(
            count_mismatches(&actual_q1_landmark, &expected_q1_landmark),
            0,
            "q1 landmark bit mismatches"
        );
        assert_eq!(
            count_mismatches(&actual_q1_residual, &expected_q1_residual),
            0,
            "q1 residual bit mismatches"
        );

        assert_f32_bits(&qr.pivots, &[0xc54060ff, 0xc54e1be3, 0x4285df56]);
        assert_f32_bits(&qr.tau, &[0x3fc25a51, 0x3fc213e7, 0x3fcbca3c]);
        assert_f32_bits(
            &[
                qr.storage[qr.index(0, qr.landmark_offset)],
                qr.storage[qr.index(1, qr.landmark_offset + 1)],
                qr.storage[qr.index(2, qr.landmark_offset + 2)],
            ],
            &[0xc54060fe, 0xc54e1be4, 0x4285df56],
        );
        for row in 0..rows {
            assert_eq!(
                qr.storage[qr.index(row, 75)].to_bits(),
                0,
                "padding column changed at row {row}"
            );
        }
        for row in 12..15 {
            for column in 0..qr.storage_cols() {
                assert_eq!(
                    qr.storage[qr.index(row, column)].to_bits(),
                    0,
                    "damping row {row}, column {column} changed"
                );
            }
        }

        let json_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7al_upstream_f4_20260821T000200Z/iteration.jsonl");
        let mut expected_q2_state = None;
        let mut expected_q2_residual = None;
        for line in std::fs::read_to_string(&json_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", json_path.display()))
            .lines()
        {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            if value.get("record").and_then(serde_json::Value::as_str) == Some("landmark_qr")
                && value.get("track_id").and_then(serde_json::Value::as_u64) == Some(1)
                && value.get("iteration").and_then(serde_json::Value::as_u64) == Some(0)
            {
                let rows = value["reduced_rows"].as_array().unwrap();
                expected_q2_state = Some(
                    rows.iter()
                        .flat_map(|row| {
                            row.as_array()
                                .unwrap()
                                .iter()
                                .map(|value| value.as_f64().unwrap() as f32)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                );
                expected_q2_residual = Some(
                    value["reduced_rhs"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_f64().unwrap() as f32)
                        .collect::<Vec<_>>(),
                );
                break;
            }
        }
        let expected_q2_state = expected_q2_state.expect("track 1 q2 fixture");
        let expected_q2_residual = expected_q2_residual.expect("track 1 q2 rhs fixture");
        let actual_q2_state = qr.q2_state();
        let actual_q2_residual = qr.q2_residual();
        assert_eq!(actual_q2_state.shape(), (12, 75));
        assert_eq!(expected_q2_state.len(), 12 * 75);
        assert_eq!(expected_q2_residual.len(), 12);
        let actual_q2_state_ref = &actual_q2_state;
        let actual_q2_state = (0..actual_q2_state_ref.nrows())
            .flat_map(|row| {
                (0..actual_q2_state_ref.ncols())
                    .map(move |column| actual_q2_state_ref[(row, column)])
            })
            .collect::<Vec<_>>();
        assert_eq!(
            count_mismatches(&actual_q2_state, &expected_q2_state),
            0,
            "q2 state bit mismatches"
        );
        assert_eq!(
            count_mismatches(actual_q2_residual.as_slice(), &expected_q2_residual),
            0,
            "q2 residual bit mismatches"
        );
    }

    #[test]
    fn m7_landmark_gemv_packet_reduction_and_tails_match_golden_bits() {
        // These widths exercise the scalar-only, Packet4-plus-tail, and
        // Packet8-plus-tail paths.  The expected words come from the pinned
        // Eigen AVX2 row-major GEMV schedule: Packet8/Packet4 FMA lanes are
        // reduced once with the predux tree, followed by plain scalar tail
        // products.
        let fixture_value = |index: usize| {
            let mantissa = ((index.wrapping_mul(0x01f1_2345) + 0x2a5) & 0x007f_ffff) as u32;
            let sign: u32 = if index % 3 == 0 { 0x8000_0000 } else { 0 };
            f32::from_bits(sign | 0x3e80_0000 | mantissa)
        };
        let expected = [
            (5, [0xbdeae61a, 0xbe2ec959, 0x3f571bd7]),
            (11, [0xbecfde58, 0xbee5f40c, 0x3fcd585d]),
            (14, [0xbf19e8df, 0xbf08d17c, 0x40048312]),
            (75, [0xc0669959, 0xc05b5d00, 0xc05f46cd]),
        ];
        for (columns, expected_bits) in expected {
            let matrix = DMatrix::from_fn(3, columns, |row, column| {
                fixture_value(row * columns + column)
            });
            let vector = DVector::from_iterator(
                columns,
                (0..columns).map(|column| fixture_value(1000 + column)),
            );
            let actual = eigen_row_major_gemv_f32(&matrix, &vector);
            assert_f32_bits(actual.as_slice(), &expected_bits);
        }
    }

    #[test]
    fn m7_householder_tiny_and_signed_zero_match_eigen_semantics() {
        let state = DMatrix::<f32>::zeros(3, 0);
        let residual = DVector::<f32>::zeros(3);
        let mut landmark = DMatrix::<f32>::zeros(3, 3);
        landmark[(0, 0)] = f32::from_bits(0x8000_0000);
        landmark[(1, 0)] = f32::from_bits(0x0000_0001);
        landmark[(2, 0)] = f32::from_bits(0x0000_0001);
        let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual).unwrap();
        assert_eq!(qr.pivots[0].to_bits(), 0x8000_0000);
        assert_eq!(qr.tau[0].to_bits(), 0x0000_0000);
        assert_eq!(
            qr.storage[qr.index(0, qr.landmark_offset)].to_bits(),
            0x8000_0000
        );
        assert_eq!(
            qr.storage[qr.index(1, qr.landmark_offset)].to_bits(),
            0x0000_0001
        );
        assert_eq!(
            qr.storage[qr.index(2, qr.landmark_offset)].to_bits(),
            0x0000_0001
        );

        let mut positive = DMatrix::<f32>::zeros(3, 3);
        positive[(0, 0)] = f32::from_bits(0x0000_0001);
        let qr = LandmarkHouseholderF32::factor(&state, &positive, &residual).unwrap();
        assert_eq!(qr.pivots[0].to_bits(), 0x0000_0001);
        assert_eq!(qr.tau[0].to_bits(), 0x0000_0000);
    }

    #[test]
    #[ignore = "requires pinned external Q2/Eigen Householder captures"]
    fn m7_q2_one_factor_f32_reduction_is_bitwise_eigen_compatible() {
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7_track1_q2_fixture.txt");
        let fixture_text = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
        let mut tokens = fixture_text.split_whitespace();
        let rows: usize = tokens.next().unwrap().parse().unwrap();
        let cols: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!((rows, cols), (12, 75));
        let input = (0..rows * cols)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        let jacobian = DMatrix::from_fn(rows, cols, |row, column| input[row * cols + column]);
        assert_eq!(tokens.next(), Some("RHS"));
        let rhs = DVector::from_iterator(
            rows,
            (0..rows).map(|_| tokens.next().unwrap().parse::<f32>().unwrap()),
        );

        let oracle_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7_track1_q2_eigen_hb.txt");
        let oracle_text = std::fs::read_to_string(&oracle_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", oracle_path.display()));
        let mut oracle = oracle_text.split_whitespace();
        assert_eq!(oracle.next(), Some("SHAPE"));
        assert_eq!(oracle.next().unwrap().parse::<usize>().unwrap(), cols);
        assert_eq!(oracle.next().unwrap().parse::<usize>().unwrap(), cols);
        assert_eq!(oracle.next().unwrap().parse::<usize>().unwrap(), cols);
        assert_eq!(oracle.next(), Some("H_BITS"));
        let expected_h = (0..cols * cols)
            .map(|_| u32::from_str_radix(oracle.next().unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(oracle.next(), Some("B_BITS"));
        let expected_b = (0..cols)
            .map(|_| u32::from_str_radix(oracle.next().unwrap(), 16).unwrap())
            .collect::<Vec<_>>();

        let h = jacobian.transpose() * &jacobian;
        let mut b = DVector::<f32>::zeros(cols);
        accumulate_transpose_vector_f32_eigen(&mut b, &jacobian, &rhs, false);
        let mut h_mismatch = 0;
        let mut first_h = None;
        for row in 0..cols {
            for column in 0..cols {
                let index = row * cols + column;
                if h[(row, column)].to_bits() != expected_h[index] {
                    if first_h.is_none() {
                        first_h =
                            Some((row, column, h[(row, column)].to_bits(), expected_h[index]));
                    }
                    h_mismatch += 1;
                }
            }
        }
        let mut b_mismatch = 0;
        let mut first_b = None;
        for row in 0..cols {
            if b[row].to_bits() != expected_b[row] {
                if first_b.is_none() {
                    first_b = Some((row, b[row].to_bits(), expected_b[row]));
                }
                b_mismatch += 1;
            }
        }
        assert_eq!(
            (h_mismatch, b_mismatch),
            (0, 0),
            "current f32 Q2 accumulation mismatch: first H={first_h:?}, first b={first_b:?}"
        );
    }

    #[test]
    fn m7_q2_abs_normal_system_matches_native_events() {
        // The native logger writes the post-QR Q2 matrix and the ABS H/b
        // assignment in a compact binary record.  Keep this test local to
        // the captured parity corpus: ordinary builds without the diagnostic
        // artifacts simply have no oracle to load.
        fn read_u32(data: &[u8], offset: &mut usize) -> u32 {
            let end = *offset + 4;
            let bytes = data
                .get(*offset..end)
                .unwrap_or_else(|| panic!("truncated native Q2 fixture at {}", *offset));
            *offset = end;
            u32::from_le_bytes(bytes.try_into().unwrap())
        }
        fn read_i64(data: &[u8], offset: &mut usize) -> i64 {
            let end = *offset + 8;
            let bytes = data
                .get(*offset..end)
                .unwrap_or_else(|| panic!("truncated native Q2 fixture at {}", *offset));
            *offset = end;
            i64::from_le_bytes(bytes.try_into().unwrap())
        }
        fn read_matrix(data: &[u8], offset: &mut usize) -> (usize, usize, Vec<f32>) {
            let rows = read_u32(data, offset) as usize;
            let columns = read_u32(data, offset) as usize;
            let values = (0..rows * columns)
                .map(|_| f32::from_bits(read_u32(data, offset)))
                .collect();
            (rows, columns, values)
        }
        fn read_vector(data: &[u8], offset: &mut usize) -> Vec<f32> {
            let length = read_u32(data, offset) as usize;
            (0..length)
                .map(|_| f32::from_bits(read_u32(data, offset)))
                .collect()
        }

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture_root = root.join("../../target/m7im15_native_abs_hb_frame6_max7_20260829");
        let first_fixture = fixture_root.join("abs.event00.bin");
        if !first_fixture.exists() {
            return;
        }

        for event in ["00", "01", "02"] {
            let path = fixture_root.join(format!("abs.event{event}.bin"));
            let data = std::fs::read(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            assert_eq!(&data[..11], b"M7IM15ABS1\0");
            let mut offset = 11;
            let _version = read_u32(&data, &mut offset);
            let _event = read_u32(&data, &mut offset);
            let q2_rows = read_u32(&data, &mut offset) as usize;
            let q2_columns = read_u32(&data, &mut offset) as usize;
            let q2_rhs_len = read_u32(&data, &mut offset) as usize;
            let order_count = read_u32(&data, &mut offset) as usize;
            for _ in 0..order_count {
                let _timestamp = read_i64(&data, &mut offset);
                let _offset = read_u32(&data, &mut offset);
                let _dof = read_u32(&data, &mut offset);
            }
            let kfs_count = read_u32(&data, &mut offset) as usize;
            for _ in 0..kfs_count {
                let _frame_id = read_i64(&data, &mut offset);
            }
            let (matrix_rows, matrix_columns, q2_values) = read_matrix(&data, &mut offset);
            let q2_rhs = read_vector(&data, &mut offset);
            let (abs_rows, abs_columns, expected_h) = read_matrix(&data, &mut offset);
            let expected_b = read_vector(&data, &mut offset);
            let _sentinel = read_u32(&data, &mut offset);
            assert_eq!(offset, data.len());
            assert_eq!((matrix_rows, matrix_columns), (q2_rows, q2_columns));
            assert_eq!(q2_rhs.len(), q2_rhs_len);
            assert_eq!((abs_rows, abs_columns), (q2_columns, q2_columns));
            assert_eq!(expected_b.len(), q2_columns);

            let jacobian = DMatrix::from_column_slice(q2_rows, q2_columns, &q2_values)
                .map(|value| f64::from(value));
            let rhs = DVector::from_iterator(q2_rhs.len(), q2_rhs.iter().copied().map(f64::from));
            let (actual_h, actual_b) = q2_f32_normal_system(&jacobian, &rhs);
            let mut h_mismatches = 0;
            let mut b_mismatches = 0;
            for index in 0..expected_h.len() {
                let row = index % q2_columns;
                let column = index / q2_columns;
                if actual_h[(row, column)].to_bits() != f64::from(expected_h[index]).to_bits() {
                    h_mismatches += 1;
                }
            }
            for index in 0..expected_b.len() {
                if actual_b[index].to_bits() != f64::from(expected_b[index]).to_bits() {
                    b_mismatches += 1;
                }
            }
            assert_eq!(
                (h_mismatches, b_mismatches),
                (0, 0),
                "native Q2 ABS event {event} mismatch"
            );
        }
    }

    #[test]
    #[ignore = "requires pinned external frame-4 and frame-6 model captures"]
    fn m7_reduced_model_decrease_candidate_is_not_full_qr_model() {
        // This pinned frame-4/track-1 block contains both Q1 and Q2 rows.
        // Using several deterministic trial steps makes the missing Q1
        // contribution observable without depending on a particular LM
        // damping value or on a diagnostic run being present.
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7_householder_track1_f4_i0.txt");
        let fixture_text = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
        let mut tokens = fixture_text.split_whitespace();
        let rows: usize = tokens.next().unwrap().parse().unwrap();
        let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!((rows, storage_cols), (15, 80));
        let input = (0..rows * storage_cols)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        let state = DMatrix::from_fn(12, 75, |row, column| {
            f64::from(input[row * storage_cols + column])
        });
        let landmark = DMatrix::from_fn(12, 3, |row, column| {
            f64::from(input[row * storage_cols + 76 + column])
        });
        let residual = DVector::from_iterator(
            12,
            (0..12).map(|row| f64::from(input[row * storage_cols + 79])),
        );
        let factor = WhitenedFactorRowStack::new(state, landmark, residual).unwrap();
        let reduced = reduce_landmark_factors_f32(std::slice::from_ref(&factor), 75, 1e-10);

        let steps = [
            DVector::from_iterator(75, (0..75).map(|column| (column as f64 - 37.0) * 0.015625)),
            DVector::from_iterator(
                75,
                (0..75).map(|column| ((column as f64 % 9.0) - 4.0) * 0.03125),
            ),
            DVector::from_iterator(
                75,
                (0..75).map(|column| {
                    if column % 7 == 0 {
                        (column as f64 + 1.0) * 0.0078125
                    } else {
                        0.0
                    }
                }),
            ),
        ];
        let mut mismatches = 0;
        for (trial, step) in steps.iter().enumerate() {
            let full = model_cost_decrease_f32(std::slice::from_ref(&factor), step, 1e-10)
                .expect("full transformed-row model must be finite");
            let reduced_only = reduced_model_cost_decrease_f32(&reduced, step)
                .expect("reduced quadratic must be finite");
            let full_bits = (full as f32).to_bits();
            let reduced_bits = (reduced_only as f32).to_bits();
            println!(
                "m7 reduced model trial={trial} full={full:.9} full_f32={full_bits:08x} reduced={reduced_only:.9} reduced_f32={reduced_bits:08x}"
            );
            if full_bits != reduced_bits {
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, steps.len(), "Q1 rows must affect every witness");
    }

    #[test]
    fn m7_reduced_model_candidate_differs_on_multiple_frontier_trials() {
        // The integrated M7 frame-6 detail stream records the exact reduced
        // H/b and solved f32-cast step used by the LM loop.  Its model field
        // is emitted immediately after model_cost_decrease_f32.  Replaying
        // the reduced quadratic here therefore tests the candidate at the
        // real trial boundary while retaining production call-site
        // isolation.  The fixture is optional for ordinary source builds.
        let detail_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7im15_accum_diag_frame6_detail_20260828.jsonl");
        if !detail_path.exists() {
            return;
        }
        let detail = std::fs::read_to_string(&detail_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", detail_path.display()));
        let scalar_fma = |reduced: &ReducedNormalSystemF32, state_step: &DVector<f64>| {
            let step = as_f32_vector(state_step);
            let mut h_step = DVector::zeros(step.len());
            for row in 0..step.len() {
                let mut value = 0.0_f32;
                for column in 0..step.len() {
                    value = reduced.h[(row, column)].mul_add(step[column], value);
                }
                h_step[row] = value;
            }
            let mut linear = 0.0_f32;
            let mut quadratic = 0.0_f32;
            for index in 0..step.len() {
                linear = step[index].mul_add(reduced.b[index], linear);
                quadratic = step[index].mul_add(h_step[index], quadratic);
            }
            -(linear + 0.5_f32 * quadratic)
        };
        let ordered = |reduced: &ReducedNormalSystemF32, state_step: &DVector<f64>| {
            let step = as_f32_vector(state_step);
            let mut h_step = DVector::zeros(step.len());
            for row in 0..step.len() {
                let mut value = 0.0_f32;
                for column in 0..step.len() {
                    value =
                        add_f32_exact(value, mul_f32_exact(reduced.h[(row, column)], step[column]));
                }
                h_step[row] = value;
            }
            let mut linear = 0.0_f32;
            let mut quadratic = 0.0_f32;
            for index in 0..step.len() {
                linear = add_f32_exact(linear, mul_f32_exact(step[index], reduced.b[index]));
                quadratic = add_f32_exact(quadratic, mul_f32_exact(step[index], h_step[index]));
            }
            -add_f32_exact(linear, mul_f32_exact(0.5_f32, quadratic))
        };
        let mut checked = 0;
        for line in detail.lines() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            if value.get("phase").and_then(serde_json::Value::as_str) != Some("trial") {
                continue;
            }
            let h_rows = value["global"]["h"].as_array().unwrap();
            let state_dof = h_rows.len();
            assert!(state_dof > 0);
            let h = DMatrix::from_fn(state_dof, state_dof, |row, column| {
                h_rows[row][column].as_f64().unwrap() as f32
            });
            let b_values = value["global"]["b"].as_array().unwrap();
            assert_eq!(b_values.len(), state_dof);
            let b = DVector::from_iterator(
                state_dof,
                b_values.iter().map(|entry| entry.as_f64().unwrap() as f32),
            );
            let step_values = value["delta"].as_array().unwrap();
            assert_eq!(step_values.len(), state_dof);
            let step = DVector::from_iterator(
                state_dof,
                step_values.iter().map(|entry| entry.as_f64().unwrap()),
            );
            let reduced = ReducedNormalSystemF32 {
                h,
                b,
                back_substitution: Vec::new(),
                compact_back_substitution: None,
                model_decrease_payload: None,
                imu_diagnostic: None,
                diagnostic_stages: None,
            };
            let reduced_decrease = reduced_model_cost_decrease_f32(&reduced, &step)
                .expect("frontier reduced quadratic must be finite")
                as f32;
            let scalar_fma_decrease = scalar_fma(&reduced, &step);
            let ordered_decrease = ordered(&reduced, &step);
            let before = value["cost"]["before"].as_f64().unwrap() as f32;
            let expected_model = value["cost"]["model"].as_f64().unwrap() as f32;
            let candidate_model = before - reduced_decrease;
            println!(
                "m7 frontier frame={} iteration={} trial={} expected_model={:08x} candidate_model={:08x} packet_decrease={:08x} scalar_fma_decrease={:08x} ordered_decrease={:08x}",
                value["frame_id"].as_u64().unwrap_or_default(),
                value["iteration"].as_u64().unwrap_or_default(),
                value["trial"].as_u64().unwrap_or_default(),
                expected_model.to_bits(),
                candidate_model.to_bits(),
                reduced_decrease.to_bits(),
                scalar_fma_decrease.to_bits(),
                ordered_decrease.to_bits(),
            );
            assert_ne!(
                expected_model.to_bits(),
                candidate_model.to_bits(),
                "reduced H/b must not replace the complete transformed-row model"
            );
            checked += 1;
            if checked == 6 {
                break;
            }
        }
        assert_eq!(checked, 6, "expected six frame-6 trial records");
    }

    #[test]
    #[ignore = "requires pinned external frame-4 and frame-6 model captures"]
    fn m7_reduced_model_q1_constant_candidate_track1_and_frontier() {
        let q1_constant_variants = |entries: &[CompactLandmarkBackSubstitutionF32]| -> [f32; 4] {
            let mut fma_per_factor = 0.0_f32;
            let mut ordered_per_factor = 0.0_f32;
            let mut fma_norm_total = 0.0_f32;
            let mut ordered_norm_total = 0.0_f32;
            for entry in entries {
                assert!(entry.eligible);
                assert_eq!(entry.rank, entry.landmark_cols);
                let mut fma_norm = 0.0_f32;
                let mut ordered_norm = 0.0_f32;
                for row in 0..entry.q1_residual_len() {
                    let residual = entry.q1_residual_value(row);
                    fma_norm = residual.mul_add(residual, fma_norm);
                    ordered_norm = add_f32_exact(ordered_norm, mul_f32_exact(residual, residual));
                }
                fma_norm_total = add_f32_exact(fma_norm_total, fma_norm);
                ordered_norm_total = add_f32_exact(ordered_norm_total, ordered_norm);
                fma_per_factor = add_f32_exact(fma_per_factor, mul_f32_exact(0.5_f32, fma_norm));
                ordered_per_factor =
                    add_f32_exact(ordered_per_factor, mul_f32_exact(0.5_f32, ordered_norm));
            }
            [
                fma_per_factor,
                ordered_per_factor,
                mul_f32_exact(0.5_f32, fma_norm_total),
                mul_f32_exact(0.5_f32, ordered_norm_total),
            ]
        };

        // First check the identity on the pinned single visual block.  The
        // Q1 term is step-independent; any remaining difference is due to
        // the global Q2 H/b association/order rather than a missing constant.
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7_householder_track1_f4_i0.txt");
        let fixture_text = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
        let mut tokens = fixture_text.split_whitespace();
        let rows: usize = tokens.next().unwrap().parse().unwrap();
        let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!((rows, storage_cols), (15, 80));
        let input = (0..rows * storage_cols)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        let factor = WhitenedFactorRowStack::new(
            DMatrix::from_fn(12, 75, |row, column| {
                f64::from(input[row * storage_cols + column])
            }),
            DMatrix::from_fn(12, 3, |row, column| {
                f64::from(input[row * storage_cols + 76 + column])
            }),
            DVector::from_iterator(
                12,
                (0..12).map(|row| f64::from(input[row * storage_cols + 79])),
            ),
        )
        .unwrap();
        let reduced = reduce_landmark_factors_f32(std::slice::from_ref(&factor), 75, 1e-10);
        let q1 = compact_landmark_back_substitution_f32(&factor, 0, 1, 1e-10)
            .expect("track1 compact Q1 payload");
        let track_step =
            DVector::from_iterator(75, (0..75).map(|column| (column as f64 - 37.0) * 0.015625));
        let full = model_cost_decrease_f32(std::slice::from_ref(&factor), &track_step, 1e-10)
            .expect("track1 full model decrease");
        let corrected = reduced_model_cost_decrease_with_q1_constant_f32(
            &reduced,
            std::slice::from_ref(&q1),
            &track_step,
        )
        .expect("track1 Q1-corrected model decrease");
        let track_q1_variants = q1_constant_variants(std::slice::from_ref(&q1));
        println!(
            "m7 q1-constant track1 full={:08x} corrected={:08x} q1=({:08x},{:08x},{:08x}) variants=({:08x},{:08x},{:08x},{:08x})",
            (full as f32).to_bits(),
            (corrected as f32).to_bits(),
            q1.q1_residual_value(0).to_bits(),
            q1.q1_residual_value(1).to_bits(),
            q1.q1_residual_value(2).to_bits(),
            track_q1_variants[0].to_bits(),
            track_q1_variants[1].to_bits(),
            track_q1_variants[2].to_bits(),
            track_q1_variants[3].to_bits(),
        );
        assert!(corrected.is_finite());

        // The integrated detail stream contains the original visual factor
        // rows as well as the reduced global H/b.  Rebuild Q1 residuals from
        // those rows, preserving factor order, and test the same candidate on
        // six real frame-6 trial boundaries.
        let detail_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7im15_accum_diag_frame6_detail_20260828.jsonl");
        if !detail_path.exists() {
            return;
        }
        let detail = std::fs::read_to_string(&detail_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", detail_path.display()));
        let matrix_f32 = |value: &serde_json::Value| {
            let rows = value.as_array().expect("matrix rows");
            let row_count = rows.len();
            let column_count = rows
                .first()
                .map(|row| row.as_array().expect("matrix row").len())
                .unwrap_or(0);
            DMatrix::from_fn(row_count, column_count, |row, column| {
                rows[row][column].as_f64().expect("matrix value") as f32
            })
        };
        let vector_f32 = |value: &serde_json::Value| {
            let values = value.as_array().expect("vector");
            DVector::from_iterator(
                values.len(),
                values
                    .iter()
                    .map(|entry| entry.as_f64().expect("vector value") as f32),
            )
        };
        let base_variants = |reduced: &ReducedNormalSystemF32, state_step: &DVector<f64>| {
            let packet = reduced_model_cost_decrease_f32(reduced, state_step)
                .expect("packet reduced model decrease") as f32;
            let step = as_f32_vector(state_step);
            let mut h_step = DVector::zeros(step.len());
            for row in 0..step.len() {
                let mut value = 0.0_f32;
                for column in 0..step.len() {
                    value = reduced.h[(row, column)].mul_add(step[column], value);
                }
                h_step[row] = value;
            }
            let mut scalar_linear = 0.0_f32;
            let mut scalar_quadratic = 0.0_f32;
            for index in 0..step.len() {
                scalar_linear = step[index].mul_add(reduced.b[index], scalar_linear);
                scalar_quadratic = step[index].mul_add(h_step[index], scalar_quadratic);
            }
            let scalar_fma = -(scalar_linear + 0.5_f32 * scalar_quadratic);
            let mut ordered_h_step = DVector::zeros(step.len());
            for row in 0..step.len() {
                let mut value = 0.0_f32;
                for column in 0..step.len() {
                    value =
                        add_f32_exact(value, mul_f32_exact(reduced.h[(row, column)], step[column]));
                }
                ordered_h_step[row] = value;
            }
            let mut ordered_linear = 0.0_f32;
            let mut ordered_quadratic = 0.0_f32;
            for index in 0..step.len() {
                ordered_linear =
                    add_f32_exact(ordered_linear, mul_f32_exact(step[index], reduced.b[index]));
                ordered_quadratic = add_f32_exact(
                    ordered_quadratic,
                    mul_f32_exact(step[index], ordered_h_step[index]),
                );
            }
            let ordered = -add_f32_exact(ordered_linear, mul_f32_exact(0.5_f32, ordered_quadratic));
            [packet, scalar_fma, ordered]
        };
        let mut checked = 0;
        let mut mismatches = 0;
        for line in detail.lines() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            if value.get("phase").and_then(serde_json::Value::as_str) != Some("trial") {
                continue;
            }
            let h_rows = value["global"]["h"].as_array().unwrap();
            let state_dof = h_rows.len();
            let h = matrix_f32(&value["global"]["h"]);
            let b = vector_f32(&value["global"]["b"]);
            assert_eq!(h.nrows(), state_dof);
            let step_values = value["delta"].as_array().unwrap();
            let step = DVector::from_iterator(
                step_values.len(),
                step_values
                    .iter()
                    .map(|entry| entry.as_f64().expect("step value")),
            );
            let mut q1_entries = Vec::new();
            for (landmark_index, factor_value) in value["landmark_factors"]
                .as_array()
                .expect("landmark factors")
                .iter()
                .enumerate()
            {
                let state = matrix_f32(&factor_value["state_jacobian"]);
                let landmark = matrix_f32(&factor_value["landmark_jacobian"]);
                let residual = vector_f32(&factor_value["residual"]);
                let factor = WhitenedFactorRowStack::new(
                    state.map(f64::from),
                    landmark.map(f64::from),
                    residual.map(f64::from),
                )
                .expect("detail visual factor");
                let track_id = factor_value["track_id"].as_u64().expect("track id");
                let compact = compact_landmark_back_substitution_f32(
                    &factor,
                    landmark_index,
                    track_id,
                    1e-10,
                )
                .expect("full-rank detail visual factor");
                assert_eq!(
                    compact.rank,
                    factor_value["rank"].as_u64().expect("detail rank") as usize
                );
                q1_entries.push(compact);
            }
            let reduced = ReducedNormalSystemF32 {
                h,
                b,
                back_substitution: Vec::new(),
                compact_back_substitution: None,
                model_decrease_payload: None,
                imu_diagnostic: None,
                diagnostic_stages: None,
            };
            let candidate =
                reduced_model_cost_decrease_with_q1_constant_f32(&reduced, &q1_entries, &step)
                    .expect("Q1-corrected frontier model decrease") as f32;
            let bases = base_variants(&reduced, &step);
            let q1_variants = q1_constant_variants(&q1_entries);
            let before = value["cost"]["before"].as_f64().unwrap() as f32;
            let expected_model = value["cost"]["model"].as_f64().unwrap() as f32;
            let candidate_model = before - candidate;
            println!(
                "m7 q1-constant frame={} iteration={} expected_model={:08x} candidate_model={:08x} decrease={:08x}",
                value["frame_id"].as_u64().unwrap_or_default(),
                value["iteration"].as_u64().unwrap_or_default(),
                expected_model.to_bits(),
                candidate_model.to_bits(),
                candidate.to_bits(),
            );
            if expected_model.to_bits() != candidate_model.to_bits() {
                for (base_index, base) in bases.iter().enumerate() {
                    for (q1_index, q1_constant) in q1_variants.iter().enumerate() {
                        let decrease = add_f32_exact(*base, *q1_constant);
                        let model = before - decrease;
                        println!(
                            "m7 q1-constant mismatch-variant base={base_index} q1={q1_index} model={:08x} decrease={:08x}",
                            model.to_bits(),
                            decrease.to_bits(),
                        );
                    }
                }
                mismatches += 1;
            }
            checked += 1;
            if checked == 6 {
                break;
            }
        }
        assert_eq!(checked, 6, "expected six frame-6 trial records");
        println!("m7 q1-constant frontier mismatches={mismatches}/{checked}");
    }

    #[test]
    #[ignore = "requires pinned external frame-4 Householder capture"]
    fn m10_landmark_direct_pack_matches_materialized_track1_and_all61() {
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7_householder_track1_f4_i0.txt");
        let fixture_text = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
        let mut tokens = fixture_text.split_whitespace();
        let rows: usize = tokens.next().unwrap().parse().unwrap();
        let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!((rows, storage_cols), (15, 80));
        let input = (0..rows * storage_cols)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        let track1_state =
            DMatrix::from_fn(12, 75, |row, column| input[row * storage_cols + column]);
        let track1_landmark =
            DMatrix::from_fn(12, 3, |row, column| input[row * storage_cols + 76 + column]);
        let track1_residual =
            DVector::from_iterator(12, (0..12).map(|row| input[row * storage_cols + 79]));

        // The all61 capture has 61 live visual rows and three ABS_QR damping
        // rows.  Use representable f32 source values so this exercises the
        // same f64->f32 conversion and packet-aligned 75-column layout while
        // keeping the fixture self-contained.
        let all61_state = DMatrix::from_fn(61, 75, |row, column| {
            let value = ((row * 17 + column * 5) % 97) as f32 - 48.0;
            value * 0.00390625
        });
        let all61_landmark = DMatrix::from_fn(61, 3, |row, column| {
            let diagonal = (row == column) as u8 as f32;
            match column {
                0 => 0.75 + diagonal + row as f32 * 0.001953125,
                1 => -0.5 + diagonal * 0.875 - row as f32 * 0.00146484375,
                _ => 0.25 + diagonal * 1.125 + row as f32 * 0.00244140625,
            }
        });
        let all61_residual = DVector::from_fn(61, |row, _| (row as f32 - 30.0) * 0.015625);

        let cases = [
            ("track1", track1_state, track1_landmark, track1_residual),
            ("all61", all61_state, all61_landmark, all61_residual),
        ];
        for (label, state32, landmark32, residual32) in cases {
            let factor = WhitenedFactorRowStack::new(
                state32.clone().map(f64::from),
                landmark32.clone().map(f64::from),
                residual32.clone().map(f64::from),
            )
            .unwrap()
            .with_kind(FactorKind::Visual)
            .with_landmark_metadata(17, 120);

            let materialized =
                LandmarkHouseholderF32::factor(&state32, &landmark32, &residual32).unwrap();
            let (direct, direct_norm) =
                LandmarkHouseholderF32::factor_from_whitened(&factor).unwrap();
            assert_landmark_qr_bitwise_equal(&direct, &materialized, label);
            let materialized_norm = landmark32.norm();
            assert_eq!(
                direct_norm.to_bits(),
                materialized_norm.to_bits(),
                "{label} landmark norm"
            );

            let metadata = Some(LandmarkFactorMetadata {
                landmark_index: 17,
                track_id: 120,
            });
            let (materialized_q2, materialized_r, materialized_rank, materialized_compact) =
                landmark_nullspace_projection_f32_with_compact(&factor, 1e-10, metadata);
            let mut arena = Vec::new();
            let (direct_q2, direct_r, direct_rank, direct_compact) =
                landmark_nullspace_projection_f32_with_compact_into(
                    &factor, 1e-10, metadata, &mut arena,
                );
            assert_matrix_f32_bitwise_equal(&direct_q2, &materialized_q2, label);
            assert_vector_f32_bitwise_equal(&direct_r, &materialized_r, label);
            assert_eq!(direct_rank, materialized_rank, "{label} rank");
            match (materialized_compact, direct_compact) {
                (Some(materialized), Some(entry)) => {
                    assert_compact_payload_bitwise_equal(&materialized, &arena, &entry, label);
                }
                (None, None) => {}
                _ => panic!("{label} compact presence mismatch"),
            }
        }
    }

    #[test]
    fn m11_landmark_workspace_reuses_qr_scratch_without_changing_bits() {
        let state = DMatrix::from_fn(12, 75, |row, column| {
            ((row * 19 + column * 7) % 101) as f32 * 0.015625 - 0.75
        });
        let landmark = DMatrix::from_fn(12, 3, |row, column| {
            let diagonal = (row == column) as u8 as f32;
            (column as f32 - 1.0) * 0.25 + diagonal + row as f32 * 0.00390625
        });
        let residual = DVector::from_fn(12, |row, _| row as f32 * 0.03125 - 0.125);
        let factor = WhitenedFactorRowStack::new(
            state.map(f64::from),
            landmark.map(f64::from),
            residual.map(f64::from),
        )
        .unwrap()
        .with_kind(FactorKind::Visual)
        .with_landmark_metadata(17, 120);

        let metadata = Some(LandmarkFactorMetadata {
            landmark_index: 17,
            track_id: 120,
        });
        let mut qr_workspace = LandmarkHouseholderWorkspace::default();
        let mut arena = Vec::new();
        let first = landmark_nullspace_projection_f32_with_compact_into_with_workspace(
            &factor,
            1e-10,
            metadata,
            &mut arena,
            &mut qr_workspace,
        );
        let capacities = (
            qr_workspace.storage.capacity(),
            qr_workspace.pivots.capacity(),
            qr_workspace.tau.capacity(),
            qr_workspace.essential.capacity(),
            qr_workspace.gemv.capacity(),
        );
        assert!(capacities.0 > 0 && capacities.1 > 0 && capacities.3 > 0 && capacities.4 > 0);

        let mut second_arena = Vec::new();
        let second = landmark_nullspace_projection_f32_with_compact_into_with_workspace(
            &factor,
            1e-10,
            metadata,
            &mut second_arena,
            &mut qr_workspace,
        );
        assert_eq!(
            capacities,
            (
                qr_workspace.storage.capacity(),
                qr_workspace.pivots.capacity(),
                qr_workspace.tau.capacity(),
                qr_workspace.essential.capacity(),
                qr_workspace.gemv.capacity(),
            ),
            "same-sized visual factors must reuse QR/scratch capacities"
        );
        assert_matrix_f32_bitwise_equal(&first.0, &second.0, "workspace q2");
        assert_vector_f32_bitwise_equal(&first.1, &second.1, "workspace rhs");
        assert_eq!(first.2, second.2, "workspace rank");
        match (&first.3, &second.3) {
            (Some(first), Some(second)) => {
                assert_eq!(first.landmark_index, second.landmark_index);
                assert_eq!(first.track_id, second.track_id);
                assert_eq!(first.state_cols, second.state_cols);
                assert_eq!(first.landmark_cols, second.landmark_cols);
                assert_eq!(first.rank, second.rank);
                assert_eq!(first.eligible, second.eligible);
                let first_len = first.storage_len().expect("valid compact storage length");
                let second_len = second.storage_len().expect("valid compact storage length");
                assert_eq!(
                    &arena[first.storage_offset..first.storage_offset + first_len],
                    &second_arena[second.storage_offset..second.storage_offset + second_len]
                );
            }
            _ => panic!("workspace compact presence mismatch"),
        }
    }

    fn m11_workspace_fixture(
        observation_rows: usize,
        state_cols: usize,
        landmark_cols: usize,
        seed: f64,
        landmark_index: usize,
    ) -> WhitenedFactorRowStack {
        let state = DMatrix::from_fn(observation_rows, state_cols, |row, column| {
            seed + (row * 17 + column * 5) as f64 * 0.00390625
        });
        let landmark = DMatrix::from_fn(observation_rows, landmark_cols, |row, column| {
            let diagonal = (row == column) as u8 as f64;
            seed * 0.25 + (column as f64 + 1.0) * 0.125 + diagonal + row as f64 * 0.0078125
        });
        let residual = DVector::from_fn(observation_rows, |row, _| {
            seed * 0.5 + row as f64 * 0.015625 - 0.25
        });
        WhitenedFactorRowStack::new(state, landmark, residual)
            .unwrap()
            .with_kind(FactorKind::Visual)
            .with_landmark_metadata(landmark_index, 10_000 + landmark_index as u64)
    }

    fn m11_assert_workspace_matches_materialized(
        factor: &WhitenedFactorRowStack,
        label: &str,
        workspace: &mut LandmarkHouseholderWorkspace,
        arena: &mut Vec<f32>,
    ) {
        let metadata = factor.landmark_metadata;
        let expected = landmark_nullspace_projection_f32_with_compact(factor, 1e-10, metadata);
        let actual = landmark_nullspace_projection_f32_with_compact_into_with_workspace(
            factor, 1e-10, metadata, arena, workspace,
        );
        assert_matrix_f32_bitwise_equal(&actual.0, &expected.0, label);
        assert_vector_f32_bitwise_equal(&actual.1, &expected.1, label);
        assert_eq!(actual.2, expected.2, "{label} rank");
        match (&expected.3, &actual.3) {
            (Some(expected), Some(actual)) => {
                assert_compact_payload_bitwise_equal(expected, arena, actual, label);
            }
            (None, None) => {}
            _ => panic!("{label} compact presence mismatch"),
        }
    }

    fn m11_assert_workspace_rejects_without_mutation(
        factor: &WhitenedFactorRowStack,
        workspace: &mut LandmarkHouseholderWorkspace,
        arena: &mut Vec<f32>,
    ) {
        let arena_before = arena.clone();
        let capacities_before = (
            workspace.storage.capacity(),
            workspace.pivots.capacity(),
            workspace.tau.capacity(),
            workspace.essential.capacity(),
            workspace.gemv.capacity(),
        );
        let (q2, rhs, rank, compact) =
            landmark_nullspace_projection_f32_with_compact_into_with_workspace(
                factor,
                1e-10,
                factor.landmark_metadata,
                arena,
                workspace,
            );
        assert_eq!(q2.nrows(), 0, "malformed Q2 rows");
        assert_eq!(
            q2.ncols(),
            factor.state_jacobian.ncols(),
            "malformed Q2 cols"
        );
        assert_eq!(rhs.len(), 0, "malformed rhs");
        assert_eq!(rank, 0, "malformed rank");
        assert!(compact.is_none(), "malformed compact payload");
        assert_eq!(*arena, arena_before, "malformed input changed arena");
        assert_eq!(
            capacities_before,
            (
                workspace.storage.capacity(),
                workspace.pivots.capacity(),
                workspace.tau.capacity(),
                workspace.essential.capacity(),
                workspace.gemv.capacity(),
            ),
            "malformed input consumed reusable workspace"
        );
    }

    #[test]
    fn m11_landmark_workspace_mixed_shapes_and_rejection_keep_bits() {
        let large_three = m11_workspace_fixture(24, 31, 3, 0.25, 31);
        let small_one = m11_workspace_fixture(8, 7, 1, -0.5, 11);
        let small_two = m11_workspace_fixture(10, 7, 2, 1.0, 22);
        let mut workspace = LandmarkHouseholderWorkspace::default();
        let mut arena = Vec::new();

        // Exercise 3 -> 1 -> 2 columns and a large -> small transition in a
        // single reusable workspace.  Every result is compared with the
        // materialized, independent f32 QR wrapper bit-for-bit.
        m11_assert_workspace_matches_materialized(
            &large_three,
            "mixed 3-column large",
            &mut workspace,
            &mut arena,
        );
        m11_assert_workspace_matches_materialized(
            &small_one,
            "mixed 1-column small",
            &mut workspace,
            &mut arena,
        );
        m11_assert_workspace_matches_materialized(
            &small_two,
            "mixed 2-column small",
            &mut workspace,
            &mut arena,
        );

        // Once the workspace is warm, each malformed shape must fail before
        // taking a reusable vector or appending a compact arena entry.
        let mut wrong_rows = small_two.clone();
        wrong_rows.landmark_jacobian = DMatrix::zeros(9, 2);
        m11_assert_workspace_rejects_without_mutation(&wrong_rows, &mut workspace, &mut arena);

        let mut wrong_residual = small_two.clone();
        wrong_residual.residual = DVector::zeros(9);
        m11_assert_workspace_rejects_without_mutation(&wrong_residual, &mut workspace, &mut arena);

        let mut wrong_columns = small_two.clone();
        wrong_columns.landmark_jacobian = DMatrix::zeros(10, 4);
        m11_assert_workspace_rejects_without_mutation(&wrong_columns, &mut workspace, &mut arena);

        // Reuse after all three rejection paths must still produce the same
        // Q2/RHS/rank/compact bits as the materialized reference.
        m11_assert_workspace_matches_materialized(
            &large_three,
            "post-rejection 3-column",
            &mut workspace,
            &mut arena,
        );
    }

    #[test]
    fn m11_compact_lengths_fail_closed_without_panic() {
        assert!(checked_landmark_householder_layout(usize::MAX, 7, 3).is_none());
        assert!(checked_landmark_householder_layout(8, usize::MAX, 3).is_none());
        assert!(checked_compact_storage_len(3, usize::MAX).is_none());
        let fixture = m11_workspace_fixture(8, 7, 3, 0.5, 3);
        assert_eq!(
            checked_compact_storage_capacity(std::slice::from_ref(&fixture), 7),
            checked_compact_storage_len(3, 7)
        );
        assert!(
            checked_compact_storage_capacity(std::slice::from_ref(&fixture), usize::MAX).is_none()
        );

        let entry = CompactLandmarkBackSubstitutionEntryF32 {
            landmark_index: 0,
            track_id: 0,
            storage_offset: usize::MAX,
            state_cols: usize::MAX,
            landmark_cols: 3,
            rank: 3,
            eligible: true,
        };
        assert!(entry.storage_len().is_none());
        assert!(entry.view(&[]).is_none());
        assert!(
            back_substitute_landmark_compact_entry_f32(&entry, &[], &DVector::zeros(0), 1e-10,)
                .is_none()
        );

        let payload = CompactLandmarkBackSubstitutionF32 {
            landmark_index: 0,
            track_id: 0,
            storage: Vec::new(),
            state_cols: usize::MAX,
            landmark_cols: 3,
            rank: 3,
            eligible: true,
        };
        assert!(payload.storage_len().is_none());
        assert!(
            back_substitute_landmark_compact_f32(&payload, &DVector::zeros(0), 1e-10,).is_none()
        );
    }

    #[test]
    fn m10_landmark_direct_pack_rejects_malformed_shapes_without_mutating_arena() {
        let state = DMatrix::from_fn(8, 7, |row, column| (row * 7 + column) as f32 * 0.0625);
        let landmark = DMatrix::from_fn(8, 3, |row, column| {
            if row == column {
                1.0
            } else {
                (row + column + 1) as f32 * 0.03125
            }
        });
        let residual = DVector::from_fn(8, |row, _| row as f32 * 0.125 - 0.25);
        let valid = WhitenedFactorRowStack::new(
            state.clone().map(f64::from),
            landmark.clone().map(f64::from),
            residual.clone().map(f64::from),
        )
        .unwrap();
        assert!(LandmarkHouseholderF32::factor_from_whitened(&valid).is_some());

        let mut wrong_landmark_rows = valid.clone();
        wrong_landmark_rows.landmark_jacobian = DMatrix::zeros(7, 3);
        assert!(LandmarkHouseholderF32::factor_from_whitened(&wrong_landmark_rows).is_none());
        let mut arena = vec![f32::from_bits(0x3f80_0000)];
        let arena_before = arena.clone();
        let (q2, rhs, rank, compact) = landmark_nullspace_projection_f32_with_compact_into(
            &wrong_landmark_rows,
            1e-10,
            Some(LandmarkFactorMetadata {
                landmark_index: 0,
                track_id: 1,
            }),
            &mut arena,
        );
        assert_eq!(q2.nrows(), 0);
        assert_eq!(rhs.len(), 0);
        assert_eq!(rank, 0);
        assert!(compact.is_none());
        assert_eq!(arena, arena_before);

        let mut wrong_residual_len = valid.clone();
        wrong_residual_len.residual = DVector::zeros(7);
        assert!(LandmarkHouseholderF32::factor_from_whitened(&wrong_residual_len).is_none());

        let four_landmark = WhitenedFactorRowStack::new(
            state.map(f64::from),
            DMatrix::from_fn(8, 4, |row, column| {
                if row == column {
                    1.0
                } else {
                    (row + column + 1) as f64 * 0.03125
                }
            }),
            residual.map(f64::from),
        )
        .unwrap();
        assert!(LandmarkHouseholderF32::factor_from_whitened(&four_landmark).is_none());
        let (q2, rhs, rank, compact) = landmark_nullspace_projection_f32_with_compact_into(
            &four_landmark,
            1e-10,
            None,
            &mut arena,
        );
        assert_eq!(q2.nrows(), 0);
        assert_eq!(rhs.len(), 0);
        assert_eq!(rank, 0);
        assert!(compact.is_none());
        assert_eq!(arena, arena_before);
    }

    #[test]
    #[ignore = "requires pinned external frame-4 Householder capture"]
    fn m7_q2_model_reuse_track1_is_bitwise_exact_without_qr() {
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/m7_householder_track1_f4_i0.txt");
        let fixture_text = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
        let mut tokens = fixture_text.split_whitespace();
        let rows: usize = tokens.next().unwrap().parse().unwrap();
        let storage_cols: usize = tokens.next().unwrap().parse().unwrap();
        assert_eq!((rows, storage_cols), (15, 80));
        let input = (0..rows * storage_cols)
            .map(|_| tokens.next().unwrap().parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        let state32 = DMatrix::from_fn(12, 75, |row, column| input[row * storage_cols + column]);
        let landmark32 =
            DMatrix::from_fn(12, 3, |row, column| input[row * storage_cols + 76 + column]);
        let residual32 =
            DVector::from_iterator(12, (0..12).map(|row| input[row * storage_cols + 79]));
        let factor = WhitenedFactorRowStack::new(
            state32.clone().map(f64::from),
            landmark32.clone().map(f64::from),
            residual32.clone().map(f64::from),
        )
        .unwrap();

        // The only QR in this test is payload preparation.  The reuse helper
        // below receives Q1/Q2 values and never calls LandmarkHouseholderF32.
        let qr = LandmarkHouseholderF32::factor(&state32, &landmark32, &residual32).unwrap();
        let payload = QrModelReusePayloadF32 {
            q1: qr
                .compact_back_substitution(0, 1, 3, true)
                .expect("valid compact payload"),
            q2_state: qr.q2_state(),
            q2_residual: qr.q2_residual(),
        };
        let steps = [
            DVector::from_iterator(75, (0..75).map(|column| (column as f64 - 37.0) * 0.015625)),
            DVector::from_iterator(
                75,
                (0..75).map(|column| ((column as f64 % 9.0) - 4.0) * 0.03125),
            ),
            DVector::from_iterator(
                75,
                (0..75).map(|column| {
                    if column % 7 == 0 {
                        (column as f64 + 1.0) * 0.0078125
                    } else {
                        0.0
                    }
                }),
            ),
        ];
        for (index, step) in steps.iter().enumerate() {
            let expected = model_cost_decrease_f32(std::slice::from_ref(&factor), step, 1e-10)
                .expect("track1 full model decrease") as f32;
            let actual = model_cost_decrease_from_qr_payload_f32(&payload, step, 1e-10)
                .expect("track1 QR payload model decrease") as f32;
            println!(
                "m7 q2-reuse track1 step={index} full={:08x} reuse={:08x}",
                expected.to_bits(),
                actual.to_bits(),
            );
            assert_eq!(actual.to_bits(), expected.to_bits(), "track1 step {index}");
        }
    }

    #[test]
    fn m7_q2_model_reuse_preserves_mixed_factor_order_bitwise() {
        let state_cols = 7;
        let visual_state = DMatrix::from_fn(6, state_cols, |row, column| {
            (row as f32 - 2.0) * 0.1875 + (column as f32 - 3.0) * 0.03125
        });
        let visual_landmark = DMatrix::from_fn(6, 3, |row, column| match column {
            0 => [1.0_f32, 0.125, -0.25, 0.75, 0.5, -0.375][row],
            1 => [-0.5_f32, 1.25, 0.25, -0.625, 0.875, 0.3125][row],
            _ => [0.25_f32, -0.375, 1.5, 0.5, -0.75, 0.625][row],
        });
        let visual_residual = DVector::from_fn(6, |row, _| (row as f32 - 1.5) * 0.25);
        let visual_factor = WhitenedFactorRowStack::new(
            visual_state.clone().map(f64::from),
            visual_landmark.clone().map(f64::from),
            visual_residual.clone().map(f64::from),
        )
        .unwrap()
        .with_kind(FactorKind::Visual);
        let qr = LandmarkHouseholderF32::factor(&visual_state, &visual_landmark, &visual_residual)
            .unwrap();
        let visual_payload = QrModelReusePayloadF32 {
            q1: qr
                .compact_back_substitution(0, 700, 3, true)
                .expect("valid compact payload"),
            q2_state: qr.q2_state(),
            q2_residual: qr.q2_residual(),
        };

        let plain = |rows: usize, kind: FactorKind, seed: f32| {
            let state = DMatrix::from_fn(rows, state_cols, |row, column| {
                seed + (row * state_cols + column) as f32 * 0.0625
            });
            let residual =
                DVector::from_fn(rows, |row, _| seed * 0.25 - (row as f32 + 1.0) * 0.09375);
            let factor = WhitenedFactorRowStack::new(
                state.clone().map(f64::from),
                DMatrix::zeros(rows, 0),
                residual.clone().map(f64::from),
            )
            .unwrap()
            .with_kind(kind);
            (
                factor,
                ModelReusePayloadF32::Plain {
                    kind,
                    state,
                    residual,
                },
            )
        };
        let (prior, prior_payload) = plain(4, FactorKind::Prior, 0.5);
        let (imu, imu_payload) = plain(9, FactorKind::Imu, -0.75);
        let (bias, bias_payload) = plain(6, FactorKind::Bias, 1.25);
        // Keep a deliberately mixed order.  The model accumulator is a
        // per-factor f32 subtraction, so payload order is part of the test.
        let factors = vec![prior, visual_factor, imu, bias];
        let payloads = vec![
            prior_payload,
            ModelReusePayloadF32::Visual(visual_payload),
            imu_payload,
            bias_payload,
        ];
        let steps = [
            DVector::from_iterator(
                state_cols,
                (0..state_cols).map(|column| (column as f64 - 2.0) * 0.03125),
            ),
            DVector::from_iterator(
                state_cols,
                (0..state_cols).map(|column| (column as f64 + 1.0) * -0.046875),
            ),
        ];
        for (index, step) in steps.iter().enumerate() {
            let expected = model_cost_decrease_f32(&factors, step, 1e-10)
                .expect("mixed full model decrease") as f32;
            let actual = model_cost_decrease_from_payloads_f32(&payloads, &factors, step, 1e-10)
                .expect("mixed QR payload model decrease") as f32;
            println!(
                "m7 q2-reuse mixed step={index} full={:08x} reuse={:08x}",
                expected.to_bits(),
                actual.to_bits(),
            );
            assert_eq!(actual.to_bits(), expected.to_bits(), "mixed step {index}");
        }
    }

    #[test]
    fn m7_q2_model_reuse_reducer_rejects_malformed_shapes_without_panic() {
        let valid = WhitenedFactorRowStack::new(
            DMatrix::zeros(3, 4),
            DMatrix::zeros(3, 0),
            DVector::zeros(3),
        )
        .unwrap();
        let wrong_width = WhitenedFactorRowStack::new(
            DMatrix::zeros(3, 3),
            DMatrix::zeros(3, 0),
            DVector::zeros(3),
        )
        .unwrap();
        assert!(matches!(
            reduce_landmark_factors_f32_checked_with_compact_back_substitution(
                &[wrong_width],
                4,
                1e-10,
            ),
            Err(ImuReductionError::StateWidth {
                index: 0,
                expected: 4,
                actual: 3,
            })
        ));

        let mut wrong_rows = valid.clone();
        wrong_rows.state_jacobian = DMatrix::zeros(2, 4);
        assert!(matches!(
            reduce_landmark_factors_f32_checked_with_compact_back_substitution(
                &[wrong_rows],
                4,
                1e-10,
            ),
            Err(ImuReductionError::InvalidFactorShape {
                index: 0,
                state_rows: 2,
                landmark_rows: 3,
                residual_len: 3,
            })
        ));

        // A malformed producer count must not be truncated by `zip`; the
        // all-or-nothing sidecar contract returns None before inspecting any
        // factor identity, allowing the caller to use the full evaluator.
        let empty_compact = CompactLandmarkBackSubstitutionBatchF32 {
            storage: Vec::new(),
            entries: Vec::new(),
        };
        assert!(move_model_decrease_payload(&[valid], Vec::new(), Some(&empty_compact),).is_none());
    }

    #[test]
    fn m7_imu_bias_reduction_uses_one_fifteen_row_stack_and_tags_prior() {
        let state_dof = 40;
        let offsets = ImuLinkOffsets { start: 6, end: 21 };
        let local_imu_jacobian = DMatrix::from_fn(9, AOM_NAV_DOF * 2, |row, column| {
            let seed = (row * AOM_NAV_DOF * 2 + column) as f32;
            if column < AOM_NAV_DOF {
                10_000.0_f32 + seed * 0.25
            } else {
                -0.75_f32 + seed * 0.03125
            }
        });
        let mut imu_jacobian = DMatrix::<f32>::zeros(9, state_dof);
        for row in 0..9 {
            for column in 0..AOM_NAV_DOF * 2 {
                let block = column / AOM_NAV_DOF;
                let local_column = column % AOM_NAV_DOF;
                let global_offset = if block == 0 {
                    offsets.start
                } else {
                    offsets.end
                };
                imu_jacobian[(row, global_offset + local_column)] =
                    local_imu_jacobian[(row, column)];
            }
        }
        let imu_residual = DVector::from_fn(9, |row, _| 0.25_f32 + row as f32 * 0.125);
        let local_bias_jacobian = DMatrix::from_fn(6, AOM_NAV_DOF * 2, |row, column| {
            let seed = (row * AOM_NAV_DOF * 2 + column) as f32;
            if column < AOM_NAV_DOF {
                0.03125_f32 + seed * 0.0078125
            } else {
                2.0_f32 + seed * 0.0625
            }
        });
        let mut bias_jacobian = DMatrix::<f32>::zeros(6, state_dof);
        for row in 0..6 {
            for column in 0..AOM_NAV_DOF * 2 {
                let block = column / AOM_NAV_DOF;
                let local_column = column % AOM_NAV_DOF;
                let global_offset = if block == 0 {
                    offsets.start
                } else {
                    offsets.end
                };
                bias_jacobian[(row, global_offset + local_column)] =
                    local_bias_jacobian[(row, column)];
            }
        }
        let bias_residual = DVector::from_fn(6, |row, _| -0.5_f32 + row as f32 * 0.03125);

        let imu = WhitenedFactorRowStack::with_objective_cost_kind(
            imu_jacobian.clone().map(f64::from),
            DMatrix::zeros(9, 0),
            imu_residual.clone().map(f64::from),
            0.0,
            FactorKind::Imu,
        )
        .unwrap()
        .with_imu_link_offsets(offsets.start, offsets.end);
        let bias = WhitenedFactorRowStack::with_objective_cost_kind(
            bias_jacobian.clone().map(f64::from),
            DMatrix::zeros(6, 0),
            bias_residual.clone().map(f64::from),
            0.0,
            FactorKind::Bias,
        )
        .unwrap()
        .with_imu_link_offsets(offsets.start, offsets.end);

        // A nine-row prior is deliberately placed before the IMU pair. It
        // must remain a generic dynamic product despite sharing the IMU row
        // count. This also verifies that the explicit semantic tag, rather
        // than shape alone, controls the special schedule.
        let prior_jacobian = DMatrix::from_fn(9, state_dof, |row, column| {
            if (offsets.start..offsets.start + AOM_NAV_DOF).contains(&column)
                || (offsets.end..offsets.end + AOM_NAV_DOF).contains(&column)
            {
                0.5_f32 + (row * state_dof + column) as f32 * 0.09375
            } else {
                0.0
            }
        });
        let prior_residual = DVector::from_fn(9, |row, _| 1.0_f32 - row as f32 * 0.0625);
        let prior = WhitenedFactorRowStack::with_objective_cost_kind(
            prior_jacobian.clone().map(f64::from),
            DMatrix::zeros(9, 0),
            prior_residual.clone().map(f64::from),
            0.0,
            FactorKind::Prior,
        )
        .unwrap();

        let reduced = reduce_landmark_factors_f32(&[prior, imu, bias], state_dof, 1e-10);
        let (expected_local_h, expected_local_b) = local_imu_h_b_15x30(
            &DMatrix::from_fn(IMU_LOCAL_ROWS, IMU_LOCAL_COLS, |row, column| {
                if row < 9 {
                    local_imu_jacobian[(row, column)]
                } else {
                    local_bias_jacobian[(row - 9, column)]
                }
            }),
            &DVector::from_iterator(
                15,
                imu_residual
                    .iter()
                    .copied()
                    .chain(bias_residual.iter().copied()),
            ),
        );
        let mut expected_h = DMatrix::<f32>::zeros(state_dof, state_dof);
        let mut expected_b = DVector::<f32>::zeros(state_dof);
        scatter_local_imu_h_b_15x30(
            &mut expected_h,
            &mut expected_b,
            &expected_local_h,
            &expected_local_b,
            offsets,
        )
        .unwrap();
        let prior_h = prior_jacobian.transpose() * &prior_jacobian;
        let mut prior_b = DVector::<f32>::zeros(state_dof);
        accumulate_transpose_vector_f32_eigen(
            &mut prior_b,
            &prior_jacobian,
            &prior_residual,
            false,
        );
        expected_h += prior_h;
        expected_b += prior_b;
        assert_eq!(reduced.h, expected_h);
        assert_eq!(reduced.b, expected_b);
    }

    #[test]
    fn m7_checked_imu_reducer_rejects_missing_bias() {
        let state_dof = 36;
        let offsets = ImuLinkOffsets { start: 3, end: 18 };
        let (imu, _) = tagged_imu_bias_pair(state_dof, Some(offsets), None, None);
        assert!(matches!(
            reduce_landmark_factors_f32_checked(&[imu], state_dof, 1e-10),
            Err(ImuReductionError::MissingBias { .. })
        ));
    }

    #[test]
    fn m7_checked_imu_reducer_rejects_missing_offsets() {
        let state_dof = 36;
        let (imu, bias) = tagged_imu_bias_pair(state_dof, None, None, None);
        assert!(matches!(
            reduce_landmark_factors_f32_checked(&[imu, bias], state_dof, 1e-10),
            Err(ImuReductionError::MissingOffsets { .. })
        ));
    }

    #[test]
    fn m7_checked_imu_reducer_rejects_mismatched_offsets() {
        let state_dof = 36;
        let imu_offsets = ImuLinkOffsets { start: 3, end: 18 };
        let bias_offsets = ImuLinkOffsets { start: 4, end: 19 };
        let (imu, bias) =
            tagged_imu_bias_pair(state_dof, Some(imu_offsets), Some(bias_offsets), None);
        assert!(matches!(
            reduce_landmark_factors_f32_checked(&[imu, bias], state_dof, 1e-10),
            Err(ImuReductionError::MismatchedOffsets { .. })
        ));
    }

    #[test]
    fn m7_checked_imu_reducer_rejects_active_columns_outside_link_offsets() {
        let state_dof = 36;
        let offsets = ImuLinkOffsets { start: 3, end: 18 };
        let (imu, bias) = tagged_imu_bias_pair(state_dof, Some(offsets), Some(offsets), Some(35));
        assert!(matches!(
            reduce_landmark_factors_f32_checked(&[imu, bias], state_dof, 1e-10),
            Err(ImuReductionError::UnexpectedColumns { .. })
        ));
    }

    #[test]
    fn m7_three_state_imu_links_scatter_shared_state_in_source_order() {
        let state_dof = AOM_NAV_DOF * 3;
        let link0_offsets = ImuLinkOffsets {
            start: 0,
            end: AOM_NAV_DOF,
        };
        let link1_offsets = ImuLinkOffsets {
            start: AOM_NAV_DOF,
            end: AOM_NAV_DOF * 2,
        };

        let make_pair = |offsets: ImuLinkOffsets, scale: f32| {
            let local_imu = DMatrix::from_fn(9, AOM_NAV_DOF * 2, |row, column| {
                scale + (row * AOM_NAV_DOF * 2 + column) as f32 * 0.03125
            });
            let local_bias = DMatrix::from_fn(6, AOM_NAV_DOF * 2, |row, column| {
                -scale + (row * AOM_NAV_DOF * 2 + column) as f32 * 0.0625
            });
            let mut imu_jacobian = DMatrix::<f32>::zeros(9, state_dof);
            let mut bias_jacobian = DMatrix::<f32>::zeros(6, state_dof);
            for row in 0..9 {
                for column in 0..AOM_NAV_DOF * 2 {
                    let global_offset = if column < AOM_NAV_DOF {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    imu_jacobian[(row, global_offset + column % AOM_NAV_DOF)] =
                        local_imu[(row, column)];
                }
            }
            for row in 0..6 {
                for column in 0..AOM_NAV_DOF * 2 {
                    let global_offset = if column < AOM_NAV_DOF {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    bias_jacobian[(row, global_offset + column % AOM_NAV_DOF)] =
                        local_bias[(row, column)];
                }
            }
            let imu_residual = DVector::from_fn(9, |row, _| scale + row as f32 * 0.125);
            let bias_residual = DVector::from_fn(6, |row, _| -scale + row as f32 * 0.03125);
            let imu = WhitenedFactorRowStack::with_objective_cost_kind(
                imu_jacobian.map(f64::from),
                DMatrix::zeros(9, 0),
                imu_residual.map(f64::from),
                0.0,
                FactorKind::Imu,
            )
            .unwrap()
            .with_imu_link_offsets(offsets.start, offsets.end);
            let bias = WhitenedFactorRowStack::with_objective_cost_kind(
                bias_jacobian.map(f64::from),
                DMatrix::zeros(6, 0),
                bias_residual.map(f64::from),
                0.0,
                FactorKind::Bias,
            )
            .unwrap()
            .with_imu_link_offsets(offsets.start, offsets.end);
            (
                imu,
                bias,
                local_imu,
                local_bias,
                imu_residual,
                bias_residual,
            )
        };

        // The landmark row seeds the visual phase. Its second row survives
        // nullspace projection and gives the shared state-1 block a baseline
        // before the two chronological IMU links are folded in.
        let mut visual_state = DMatrix::<f64>::zeros(2, state_dof);
        for column in 0..state_dof {
            visual_state[(1, column)] = 32_768.0 + column as f64 * 0.5;
        }
        let visual = WhitenedFactorRowStack::with_objective_cost_kind(
            visual_state,
            DMatrix::from_row_slice(2, 1, &[1.0, 0.0]),
            DVector::from_row_slice(&[0.0, 1.0]),
            0.0,
            FactorKind::Visual,
        )
        .unwrap();
        let visual_baseline =
            reduce_landmark_factors_f32(std::slice::from_ref(&visual), state_dof, 1e-10);

        let (imu0, bias0, local_imu0, local_bias0, imu_r0, bias_r0) =
            make_pair(link0_offsets, 0.75);
        let (imu1, bias1, local_imu1, local_bias1, imu_r1, bias_r1) =
            make_pair(link1_offsets, 1.25);
        let (local_h0, local_b0) = local_imu_h_b_15x30(
            &DMatrix::from_fn(IMU_LOCAL_ROWS, IMU_LOCAL_COLS, |row, column| {
                if row < 9 {
                    local_imu0[(row, column)]
                } else {
                    local_bias0[(row - 9, column)]
                }
            }),
            &DVector::from_iterator(
                IMU_LOCAL_ROWS,
                imu_r0.iter().copied().chain(bias_r0.iter().copied()),
            ),
        );
        let (local_h1, local_b1) = local_imu_h_b_15x30(
            &DMatrix::from_fn(IMU_LOCAL_ROWS, IMU_LOCAL_COLS, |row, column| {
                if row < 9 {
                    local_imu1[(row, column)]
                } else {
                    local_bias1[(row - 9, column)]
                }
            }),
            &DVector::from_iterator(
                IMU_LOCAL_ROWS,
                imu_r1.iter().copied().chain(bias_r1.iter().copied()),
            ),
        );

        let reduced =
            reduce_landmark_factors_f32(&[visual, imu0, bias0, imu1, bias1], state_dof, 1e-10);
        let mut imu_h = DMatrix::<f32>::zeros(state_dof, state_dof);
        let mut imu_b = DVector::<f32>::zeros(state_dof);
        scatter_local_imu_h_b_15x30(&mut imu_h, &mut imu_b, &local_h0, &local_b0, link0_offsets)
            .unwrap();
        scatter_local_imu_h_b_15x30(&mut imu_h, &mut imu_b, &local_h1, &local_b1, link1_offsets)
            .unwrap();
        let mut expected_h = visual_baseline.h;
        let mut expected_b = visual_baseline.b;
        expected_h += imu_h.clone();
        expected_b += imu_b.clone();
        assert_eq!(reduced.h, expected_h);
        assert_eq!(reduced.b, expected_b);

        let state1 = AOM_NAV_DOF;
        let mut state1_h = DMatrix::<f32>::zeros(AOM_NAV_DOF, AOM_NAV_DOF);
        let mut state1_b = DVector::<f32>::zeros(AOM_NAV_DOF);
        for row in 0..AOM_NAV_DOF {
            for column in 0..AOM_NAV_DOF {
                state1_h[(row, column)] = local_h0[(AOM_NAV_DOF + row, AOM_NAV_DOF + column)];
            }
            state1_b[row] = local_b0[AOM_NAV_DOF + row];
        }
        state1_h += local_h1.view((0, 0), (AOM_NAV_DOF, AOM_NAV_DOF));
        state1_b += local_b1.rows(0, AOM_NAV_DOF);
        assert_eq!(
            imu_h.view((state1, state1), (AOM_NAV_DOF, AOM_NAV_DOF)),
            state1_h
        );
        assert_eq!(imu_b.rows(state1, AOM_NAV_DOF), state1_b);
    }

    #[test]
    fn model_decrease_includes_landmark_q1_rows() {
        // With no state Jacobian, the reduced camera H/b are both zero, but
        // ABS_QR still recovers a landmark increment and charges its Q1-row
        // model decrease.  This is the smallest fixture for the f4 LM
        // schedule divergence (the omitted term is not a damping tweak).
        let f = factor(&[0., 0.], &[1., 0.], &[1., 0.], 2, 1, 1);
        let reduced = reduce_landmark_factors(std::slice::from_ref(&f), 1, 1e-10);
        assert!(reduced.h[(0, 0)].abs() < 1e-12);
        assert!(reduced.b[0].abs() < 1e-12);
        let decrease = model_cost_decrease(&[f], &DVector::zeros(1), 1e-10).unwrap();
        assert!((decrease - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rank_deficient_landmark_is_reported_and_not_backsolved() {
        let f = factor(&[1., 0., 1., 0.], &[1., 1., 2., 2.], &[1., 2.], 2, 2, 2);
        let red = reduce_landmark_factors(&[f], 2, 1e-8);
        assert_eq!(red.back_substitution[0].rank, 1);
        assert!(
            back_substitute_landmark(&red.back_substitution[0], &DVector::zeros(2), 1e-8).is_none()
        );
    }
    #[test]
    fn row_stack_preserves_block_order_and_multiple_landmarks_accumulate() {
        let a = factor(&[1., 0., 1., 0.], &[1., 0., 0., 1.], &[1., 0.], 2, 2, 2);
        let b = factor(&[0., 1., 1., 1.], &[1., 0., 0., 1.], &[0., 2.], 2, 2, 2);
        let red = reduce_landmark_factors(&[a, b], 2, 1e-10);
        assert_eq!(red.back_substitution.len(), 2);
        assert_eq!(red.h.nrows(), 2);
        assert_eq!(red.b.len(), 2);
    }

    #[test]
    fn visual_double_sphere_factor_has_whitened_two_row_contract() {
        let cam = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
        let pose = SE3::identity();
        let point = Point3::new(0.1, -0.1, 2.0);
        let obs = cam.project(&point).unwrap() + Vector2::new(0.5, 0.0);
        let f =
            visual_reprojection_factor(&cam, &pose, point, obs, FactorConfig::default()).unwrap();
        assert_eq!(
            (
                f.rows(),
                f.state_jacobian.ncols(),
                f.landmark_jacobian.ncols()
            ),
            (2, 15, 3)
        );
        assert!((f.residual[0] - 1.0).abs() < 1e-10);
        assert!(f.state_jacobian[(0, 0)].is_finite());
    }

    fn basalt_pose_increment(pose: &SE3, column: usize, amount: f64) -> SE3 {
        let mut result = pose.clone();
        if column < 3 {
            result.translation[column] += amount;
        } else {
            let mut rotation = Vector3::zeros();
            rotation[column - 3] = amount;
            result.rotation = UnitQuaternion::from_scaled_axis(rotation) * result.rotation;
        }
        result
    }

    #[test]
    fn anchored_stereographic_factor_matches_upstream_finite_difference_contract() {
        let camera =
            DoubleSphereCamera::new(458.2, 457.4, 367.1, 248.2, 0.66, 0.78, 752, 480).unwrap();
        let anchor_pose = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.04, -0.02, 0.03)),
            Vector3::new(0.3, -0.1, 0.2),
        );
        let target_pose = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.03, 0.05, 0.02)),
            Vector3::new(0.55, -0.08, 0.24),
        );
        let anchor_extrinsic = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.01, 0.02, -0.01)),
            Vector3::new(0.04, -0.02, 0.01),
        );
        let target_extrinsic = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.02, 0.01, 0.015)),
            Vector3::new(-0.07, 0.015, 0.005),
        );
        let landmark = InverseDistanceLandmark {
            anchor_pose: 11,
            anchor_camera_id: 0,
            direction: StereographicDirection {
                xy: Point2::new(0.08, -0.045),
            },
            inverse_distance: 0.27,
        };
        let observation = Point2::new(380.0, 235.0);
        let config = FactorConfig {
            observation_stddev: 1.0,
            huber_delta: 0.0,
            outlier_threshold: f64::INFINITY,
        };
        let linearized = anchored_visual_reprojection_factor(
            &camera,
            &anchor_pose,
            &anchor_extrinsic,
            &target_pose,
            &target_extrinsic,
            &landmark,
            observation,
            false,
            config,
        )
        .unwrap();

        let epsilon = 1e-7;
        for column in 0..6 {
            let plus_anchor = basalt_pose_increment(&anchor_pose, column, epsilon);
            let minus_anchor = basalt_pose_increment(&anchor_pose, column, -epsilon);
            let plus = anchored_visual_reprojection_factor(
                &camera,
                &plus_anchor,
                &anchor_extrinsic,
                &target_pose,
                &target_extrinsic,
                &landmark,
                observation,
                false,
                config,
            )
            .unwrap()
            .residual;
            let minus = anchored_visual_reprojection_factor(
                &camera,
                &minus_anchor,
                &anchor_extrinsic,
                &target_pose,
                &target_extrinsic,
                &landmark,
                observation,
                false,
                config,
            )
            .unwrap()
            .residual;
            let numerical = (plus - minus) / (2.0 * epsilon);
            assert!(
                (numerical - linearized.anchor_pose_jacobian.column(column)).norm() < 2e-5,
                "anchor column {column}: numerical={numerical:?}, analytic={:?}",
                linearized.anchor_pose_jacobian.column(column)
            );

            let plus_target = basalt_pose_increment(&target_pose, column, epsilon);
            let minus_target = basalt_pose_increment(&target_pose, column, -epsilon);
            let plus = anchored_visual_reprojection_factor(
                &camera,
                &anchor_pose,
                &anchor_extrinsic,
                &plus_target,
                &target_extrinsic,
                &landmark,
                observation,
                false,
                config,
            )
            .unwrap()
            .residual;
            let minus = anchored_visual_reprojection_factor(
                &camera,
                &anchor_pose,
                &anchor_extrinsic,
                &minus_target,
                &target_extrinsic,
                &landmark,
                observation,
                false,
                config,
            )
            .unwrap()
            .residual;
            let numerical = (plus - minus) / (2.0 * epsilon);
            assert!(
                (numerical - linearized.target_pose_jacobian.column(column)).norm() < 2e-5,
                "target column {column}: numerical={numerical:?}, analytic={:?}",
                linearized.target_pose_jacobian.column(column)
            );
        }

        for column in 0..3 {
            let mut plus_landmark = landmark;
            let mut minus_landmark = landmark;
            if column < 2 {
                plus_landmark.direction.xy[column] += epsilon;
                minus_landmark.direction.xy[column] -= epsilon;
            } else {
                plus_landmark.inverse_distance += epsilon;
                minus_landmark.inverse_distance -= epsilon;
            }
            let plus = anchored_visual_reprojection_factor(
                &camera,
                &anchor_pose,
                &anchor_extrinsic,
                &target_pose,
                &target_extrinsic,
                &plus_landmark,
                observation,
                false,
                config,
            )
            .unwrap()
            .residual;
            let minus = anchored_visual_reprojection_factor(
                &camera,
                &anchor_pose,
                &anchor_extrinsic,
                &target_pose,
                &target_extrinsic,
                &minus_landmark,
                observation,
                false,
                config,
            )
            .unwrap()
            .residual;
            let numerical = (plus - minus) / (2.0 * epsilon);
            assert!(
                (numerical - linearized.landmark_jacobian.column(column)).norm() < 2e-5,
                "landmark column {column}: numerical={numerical:?}, analytic={:?}",
                linearized.landmark_jacobian.column(column)
            );
        }
    }

    #[test]
    fn huber_and_outlier_gates_are_explicit_and_deterministic() {
        let cam = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
        let p = Point3::new(0.0, 0.0, 2.0);
        let nominal = cam.project(&p).unwrap();
        let huber = visual_reprojection_factor(
            &cam,
            &SE3::identity(),
            p,
            nominal + Vector2::new(1.0, 0.0),
            FactorConfig::default(),
        )
        .unwrap();
        assert!((huber.residual[0] - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!(visual_reprojection_factor(
            &cam,
            &SE3::identity(),
            p,
            nominal + Vector2::new(10.0, 0.0),
            FactorConfig::default()
        )
        .is_none());
    }

    #[test]
    fn stereo_and_auxiliary_factor_row_ordering_is_fixed() {
        let cam = DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
        let p = Point3::new(0.1, 0.0, 2.0);
        let o = cam.project(&p).unwrap();
        let right_pose = SE3::new(
            nalgebra::UnitQuaternion::identity(),
            Vector3::new(0.2, 0.0, 0.0),
        );
        let right_o = cam
            .project(&right_pose.inverse().transform_point(&p))
            .unwrap();
        let s = stereo_reprojection_factor(
            &cam,
            &SE3::identity(),
            &cam,
            &right_pose,
            p,
            (o, right_o),
            FactorConfig::default(),
        )
        .unwrap();
        assert_eq!(
            (
                s.rows(),
                s.state_jacobian.ncols(),
                s.landmark_jacobian.ncols()
            ),
            (4, 15, 3)
        );
        let rw =
            bias_random_walk_factor(Vector3::new(1.0, 2.0, 3.0), Vector3::zeros(), 2.0).unwrap();
        assert_eq!(rw.rows(), 6);
    }

    struct Quadratic {
        target: f64,
        reject: bool,
    }
    impl LmProblem for Quadratic {
        fn linearize(&self, state: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
            let r = if self.reject {
                1.0
            } else {
                state[0] - self.target
            };
            let j = DMatrix::from_element(1, 1, 1.0);
            Ok(LmLinearization {
                factors: vec![WhitenedFactorRowStack::new(
                    j,
                    DMatrix::zeros(1, 0),
                    DVector::from_element(1, r),
                )
                .unwrap()],
                cost: r * r,
            })
        }
        fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
            if self.reject {
                Ok(1.0)
            } else {
                Ok((state[0] - self.target).powi(2))
            }
        }
    }

    struct UpstreamF32Quadratic {
        target: f64,
        diagnostic_events: usize,
        diagnostic_patches: Cell<usize>,
    }

    impl LmProblem for UpstreamF32Quadratic {
        fn linearize(&self, state: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
            let residual = state[0] - self.target;
            Ok(LmLinearization {
                factors: vec![WhitenedFactorRowStack::new(
                    DMatrix::from_element(1, 1, 1.0),
                    DMatrix::zeros(1, 0),
                    DVector::from_element(1, residual),
                )
                .unwrap()],
                cost: residual * residual,
            })
        }

        fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
            Ok((state[0] - self.target).powi(2))
        }

        fn scalar_mode(&self) -> ScalarMode {
            ScalarMode::UpstreamF32
        }

        fn diagnostic_lm_event(&mut self, _event: LmDiagnosticEvent<'_>) {
            self.diagnostic_events += 1;
        }

        fn diagnostic_patch_reduced_f32(
            &self,
            _iteration: usize,
            _h: &mut DMatrix<f32>,
            _b: &mut DVector<f32>,
        ) {
            self.diagnostic_patches
                .set(self.diagnostic_patches.get() + 1);
        }
    }

    #[test]
    fn lm_trial_preparation_default_hook_preserves_legacy_token_result() {
        let problem = Quadratic {
            target: 2.0,
            reject: false,
        };
        let state = DVector::from_element(1, 0.5);
        let step = DVector::from_element(1, 0.25);
        let trial = problem.apply_step(&state, &step);
        let preparation = LmTrialPreparation {
            landmark_steps: vec![LmPreparedLandmarkStep {
                landmark_index: 7,
                track_id: 99,
                step: None,
            }],
            tolerance_bits: 1e-10_f64.to_bits(),
            state_fingerprint: 0,
            step_fingerprint: 0,
        };
        let mut prepared_timing = TimingBreakdown::default();
        let prepared = problem
            .trial_cost_timed_with_preparation(
                &state,
                &step,
                &trial,
                Some(preparation),
                &mut prepared_timing,
            )
            .expect("default preparation hook");
        let mut legacy_timing = TimingBreakdown::default();
        let legacy = problem
            .trial_cost_timed_with_token(&state, &step, &trial, &mut legacy_timing)
            .expect("legacy token hook");
        assert_eq!(prepared.0, legacy.0);
        assert!(prepared.1.landmark_steps.is_none());
        assert!(legacy.1.landmark_steps.is_none());
    }

    struct MalformedUpstreamF32;

    impl LmProblem for MalformedUpstreamF32 {
        fn linearize(&self, _: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
            let factor = WhitenedFactorRowStack::with_objective_cost_kind(
                DMatrix::zeros(6, 1),
                DMatrix::zeros(6, 0),
                DVector::from_element(6, 1.0),
                1.0,
                FactorKind::Bias,
            )
            .unwrap();
            Ok(LmLinearization {
                factors: vec![factor],
                cost: 1.0,
            })
        }

        fn cost(&self, _: &DVector<f64>) -> Result<f64, LmFailure> {
            Ok(1.0)
        }

        fn scalar_mode(&self) -> ScalarMode {
            ScalarMode::UpstreamF32
        }
    }

    #[test]
    fn lm_upstream_f32_clean_matches_retained_diagnostics_exactly() {
        let mut retained = UpstreamF32Quadratic {
            target: 3.0,
            diagnostic_events: 0,
            diagnostic_patches: Cell::new(0),
        };
        let mut clean = UpstreamF32Quadratic {
            target: 3.0,
            diagnostic_events: 0,
            diagnostic_patches: Cell::new(0),
        };
        let mut retained_timing = TimingBreakdown::default();
        let mut clean_timing = TimingBreakdown::default();
        let initial = DVector::from_element(1, 0.0);
        let config = LmConfig::default();

        let retained_result = solve_lm_with_timing(
            &mut retained,
            initial.clone(),
            config,
            true,
            true,
            &mut retained_timing,
        )
        .unwrap();
        let clean_result =
            solve_lm_with_timing(&mut clean, initial, config, true, false, &mut clean_timing)
                .unwrap();

        assert_eq!(clean_result.state, retained_result.state);
        assert_eq!(clean_result.cost, retained_result.cost);
        assert_eq!(clean_result.lambda, retained_result.lambda);
        assert_eq!(clean_result.iterations, retained_result.iterations);
        assert_eq!(clean_result.trace, retained_result.trace);
        assert!(retained.diagnostic_events > 0);
        assert_eq!(clean.diagnostic_events, 0);
        assert!(retained.diagnostic_patches.get() > 0);
        assert_eq!(clean.diagnostic_patches.get(), 0);
        assert_eq!(clean_timing.lm_normal_system_prep.count, 0);
        assert_eq!(
            clean_timing.lm_landmark_reduction.count,
            retained_timing.lm_landmark_reduction.count
        );
    }

    #[test]
    fn lm_f32_small_model_decrease_survives_large_cost_offset() {
        // Synthetic trial oracle isolates the decision arithmetic: the model
        // decrease is below one cost ULP, but the actual trial decreases.
        struct SmallDecrease;
        impl LmProblem for SmallDecrease {
            fn scalar_mode(&self) -> ScalarMode {
                ScalarMode::UpstreamF32
            }
            fn linearize(&self, _: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
                Ok(LmLinearization {
                    factors: vec![WhitenedFactorRowStack::new(
                        DMatrix::from_element(1, 1, 1.0),
                        DMatrix::zeros(1, 0),
                        DVector::from_element(1, 0.01),
                    )
                    .unwrap()],
                    cost: -15000.0,
                })
            }
            fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
                Ok(if state[0] == 0.0 {
                    -15000.0
                } else {
                    -15000.0009765625
                })
            }
        }
        for diagnostics in [false, true] {
            let result = solve_lm_with_timing(
                &mut SmallDecrease,
                DVector::zeros(1),
                LmConfig {
                    max_iterations: 0,
                    convergence_step: 0.0,
                    ..LmConfig::default()
                },
                true,
                diagnostics,
                &mut TimingBreakdown::default(),
            )
            .unwrap();
            assert_eq!(result.trace.len(), 1);
            let entry = &result.trace[0];
            // The old cost-model reconstruction yields zero in this case.
            assert_eq!(entry.cost_before as f32, entry.model_cost as f32);
            assert_eq!(entry.decision, LmDecision::Accepted);
            assert!(result.state[0] < 0.0);
            assert!(entry.lambda_after < entry.lambda_before);
        }
    }

    #[test]
    fn lm_f32_refreshes_before_cost_from_each_linearization() {
        struct DistinctSchedules;
        impl LmProblem for DistinctSchedules {
            fn scalar_mode(&self) -> ScalarMode {
                ScalarMode::UpstreamF32
            }
            fn linearize(&self, _: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
                Ok(LmLinearization {
                    factors: vec![WhitenedFactorRowStack::new(
                        DMatrix::from_element(1, 1, 1.0),
                        DMatrix::zeros(1, 0),
                        DVector::from_element(1, 0.1),
                    )
                    .unwrap()],
                    cost: 101.0,
                })
            }
            fn cost(&self, _: &DVector<f64>) -> Result<f64, LmFailure> {
                Ok(100.0)
            }
        }
        for diagnostics in [false, true] {
            let result = solve_lm_with_timing(
                &mut DistinctSchedules,
                DVector::zeros(1),
                LmConfig {
                    max_iterations: 1,
                    convergence_step: 0.0,
                    ..LmConfig::default()
                },
                true,
                diagnostics,
                &mut TimingBreakdown::default(),
            )
            .unwrap();
            assert_eq!(result.trace.len(), 2);
            for entry in &result.trace {
                assert_eq!(entry.cost_before, 101.0);
                assert_eq!(entry.decision, LmDecision::Accepted);
            }
            assert_eq!(result.cost, 100.0);
            assert!(result.state[0] < -0.1);
        }
    }

    #[test]
    fn lm_upstream_f32_clean_preserves_malformed_and_config_errors() {
        let mut retained = MalformedUpstreamF32;
        let mut clean = MalformedUpstreamF32;
        let mut retained_timing = TimingBreakdown::default();
        let mut clean_timing = TimingBreakdown::default();
        let initial = DVector::zeros(1);

        assert_eq!(
            solve_lm_with_timing(
                &mut retained,
                initial.clone(),
                LmConfig::default(),
                true,
                true,
                &mut retained_timing,
            ),
            Err(LmFailure::LinearSolve)
        );
        assert_eq!(
            solve_lm_with_timing(
                &mut clean,
                initial,
                LmConfig::default(),
                true,
                false,
                &mut clean_timing,
            ),
            Err(LmFailure::LinearSolve)
        );

        let mut retained = UpstreamF32Quadratic {
            target: 0.0,
            diagnostic_events: 0,
            diagnostic_patches: Cell::new(0),
        };
        let mut clean = UpstreamF32Quadratic {
            target: 0.0,
            diagnostic_events: 0,
            diagnostic_patches: Cell::new(0),
        };
        let invalid = LmConfig {
            lambda_initial: 0.0,
            ..LmConfig::default()
        };
        assert_eq!(
            solve_lm_with_timing(
                &mut retained,
                DVector::zeros(1),
                invalid,
                true,
                true,
                &mut TimingBreakdown::default(),
            ),
            Err(LmFailure::NonFinite)
        );
        assert_eq!(
            solve_lm_with_timing(
                &mut clean,
                DVector::zeros(1),
                invalid,
                true,
                false,
                &mut TimingBreakdown::default(),
            ),
            Err(LmFailure::NonFinite)
        );
    }

    #[test]
    fn lm_quadratic_accepts_and_follows_lambda_trace() {
        let mut problem = Quadratic {
            target: 3.0,
            reject: false,
        };
        let result = solve_lm(
            &mut problem,
            DVector::from_element(1, 0.0),
            LmConfig::default(),
        )
        .unwrap();
        assert!((result.state[0] - 3.0).abs() < 1e-6);
        assert!(result.cost < 1e-10);
        assert_eq!(result.trace[0].lambda_before, 1e-4);
        assert!(result
            .trace
            .iter()
            .any(|x| x.decision == LmDecision::Accepted));
        assert!(result.trace.len() <= 8);
    }

    #[test]
    fn lm_trace_suppression_preserves_solution_exactly() {
        let mut retained_problem = Quadratic {
            target: 3.0,
            reject: false,
        };
        let mut lean_problem = Quadratic {
            target: 3.0,
            reject: false,
        };
        let retained = solve_lm(
            &mut retained_problem,
            DVector::from_element(1, 0.0),
            LmConfig::default(),
        )
        .unwrap();
        let lean = solve_lm_without_trace(
            &mut lean_problem,
            DVector::from_element(1, 0.0),
            LmConfig::default(),
        )
        .unwrap();

        assert_eq!(lean.state, retained.state);
        assert_eq!(lean.cost, retained.cost);
        assert_eq!(lean.lambda, retained.lambda);
        assert_eq!(lean.iterations, retained.iterations);
        assert!(lean.trace.is_empty());
        assert!(!retained.trace.is_empty());
    }

    #[test]
    fn lm_reject_restores_state_and_increases_lambda() {
        let mut problem = Quadratic {
            target: 3.0,
            reject: true,
        };
        let result = solve_lm(
            &mut problem,
            DVector::from_element(1, 2.0),
            LmConfig {
                max_iterations: 2,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.state[0], 2.0);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.trace[0].decision, LmDecision::Rejected);
        assert_eq!(result.trace[0].lambda_after, 2e-4);
        assert_eq!(result.trace.len(), 3);
        assert_eq!(result.trace[1].lambda_after, 8e-4);
        assert_eq!(result.trace[2].lambda_after, 6.4e-3);
    }
    struct NaNProblem;
    impl LmProblem for NaNProblem {
        fn linearize(&self, _: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
            Err(LmFailure::NonFinite)
        }
        fn cost(&self, _: &DVector<f64>) -> Result<f64, LmFailure> {
            Ok(f64::NAN)
        }
    }
    #[test]
    fn lm_rejects_nan_and_invalid_lambda_configuration() {
        let mut nan_problem = NaNProblem;
        assert_eq!(
            solve_lm(&mut nan_problem, DVector::zeros(1), LmConfig::default()),
            Err(LmFailure::NonFinite)
        );
        let mut quadratic = Quadratic {
            target: 0.0,
            reject: false,
        };
        assert_eq!(
            solve_lm(
                &mut quadratic,
                DVector::zeros(1),
                LmConfig {
                    lambda_initial: 0.0,
                    ..Default::default()
                }
            ),
            Err(LmFailure::NonFinite)
        );
    }

    #[test]
    fn m7_landmark_jacobian_intermediates_match_pinned_lanes() {
        let (rotation, translation, _, _) = m7_fixed_relative_point();
        let direction = StereographicDirection {
            xy: Point2::new(-0.39586612582206726, -0.1799013614654541),
        };
        let direction_jacobian = direction.bearing_jacobian_f32();
        assert_f32_bits(
            direction_jacobian.as_slice(),
            &[
                0x3f9e8bb7, 0xbe4e4fe2, 0x3f8f59cf, 0xbe4e4fe2, 0x3fcb92dc, 0x3f024aa3,
            ],
        );

        let rotated_direction_jacobian = eigen_quaternion_matrix_f32(rotation) * direction_jacobian;
        assert_f32_bits(
            rotated_direction_jacobian.as_slice(),
            &[
                0x3f9d9adf, 0xbe3b8c06, 0x3f90c8ab, 0xbe4fefe0, 0x3fccb288, 0x3ef5c0e4,
            ],
        );

        let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
        point_wrt_landmark
            .fixed_view_mut::<3, 2>(0, 0)
            .copy_from(&rotated_direction_jacobian);
        point_wrt_landmark.set_column(2, &translation);
        assert_f32_bits(
            point_wrt_landmark.as_slice(),
            &[
                0x3f9d9adf, 0xbe3b8c06, 0x3f90c8ab, 0xbe4fefe0, 0x3fccb288, 0x3ef5c0e4, 0xbde1e254,
                0xbaf1ecc0, 0xba884364,
            ],
        );

        let point = Vector3::new(
            f32::from_bits(0xbf2fc73d),
            f32::from_bits(0xbe94aaaa),
            f32::from_bits(0x3f2ec6b2),
        );
        let (_, projection_jacobian) =
            project_double_sphere_with_jacobian_f32(&m7_fixed_camera(), point).unwrap();
        let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
        camera_jacobian
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&projection_jacobian);
        let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
        homogeneous_point_wrt_landmark
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&point_wrt_landmark);
        let landmark_jacobian =
            eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark);
        assert_f32_bits(
            landmark_jacobian.as_slice(),
            &[
                0x44446eae, 0xc1e49388, 0xc20f4748, 0x445399f4, 0xc21720bd, 0x40df22e2,
            ],
        );
    }

    #[test]
    fn m7_same_timestamp_stereo_homogeneous_jacobian_matches_pinned_lanes() {
        // This is a camera-agnostic check of the exact fixed-size source
        // product.  The fourth column/row is homogeneous zero, but the native
        // 3x4 * 4x2 reduction still owns the f32 association.
        let transform_top_left = SMatrix::<f32, 3, 4>::from_column_slice(&[
            f32::from_bits(0x3f7fffd3),
            f32::from_bits(0xbb0b904f),
            f32::from_bits(0x3a6ef290),
            f32::from_bits(0x3b0c6c2f),
            f32::from_bits(0x3f7ff8d7),
            f32::from_bits(0xbc6fa4f8),
            f32::from_bits(0xba66c186),
            f32::from_bits(0x3c6facff),
            f32::from_bits(0x3f7ff8f6),
            0.0,
            0.0,
            0.0,
        ]);
        let source_jup = SMatrix::<f32, 4, 2>::from_column_slice(&[
            f32::from_bits(0x3f9e8bb7),
            f32::from_bits(0xbe4e4fe2),
            f32::from_bits(0x3f8f59cf),
            0.0,
            f32::from_bits(0xbe4e4fe2),
            f32::from_bits(0x3fcb92dc),
            f32::from_bits(0x3f024aa3),
            0.0,
        ]);
        let source_jpp = eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup);
        assert_f32_bits(
            source_jpp.as_slice(),
            &[
                0x3f9e5d28, 0xbe4036e0, 0x3f8fdb6e, 0xbe4b47dd, 0x3fcc8f31, 0x3ef88cf5,
            ],
        );

        // Exercise the production factor with a same-timestamp stereo row;
        // no track or camera-id-specific branch is involved here.
        let camera = m7_fixed_camera();
        let frame_pose = SE3::new(
            UnitQuaternion::new_unchecked(Quaternion::new(
                0.5944822430610657,
                -0.052778493613004684,
                -0.8023747801780701,
                0.0,
            )),
            Vector3::zeros(),
        );
        let anchor_extrinsic = SE3::new(
            UnitQuaternion::new_normalize(Quaternion::new(
                0.7123125505904486,
                -0.007239825785317818,
                0.007541278561558601,
                0.7017845426564943,
            )),
            Vector3::new(
                -0.016774788924641534,
                -0.068938940687127,
                0.005139123188382424,
            ),
        );
        let target_extrinsic = SE3::new(
            UnitQuaternion::new_normalize(Quaternion::new(
                0.7115930283929829,
                -0.0023360576185881625,
                0.013000769689092388,
                0.7024677108343111,
            )),
            Vector3::new(
                -0.01507436282032619,
                0.0412627204046637,
                0.00316287258752953,
            ),
        );
        let landmark = InverseDistanceLandmark {
            anchor_pose: 0,
            anchor_camera_id: 0,
            direction: StereographicDirection {
                xy: Point2::new(-0.39586612582206726, -0.1799013614654541),
            },
            inverse_distance: 0.14654140174388885,
        };
        let factor = anchored_visual_reprojection_factor_f32_with_time_cam(
            &camera,
            &frame_pose,
            &anchor_extrinsic,
            &frame_pose,
            &target_extrinsic,
            &landmark,
            Point2::new(29.425615310668945, 107.3565444946289),
            true,
            false,
            FactorConfig::default(),
        )
        .unwrap();
        assert_f32_bits(
            factor.projection.map(|value| value as f32).as_slice(),
            &[0x41eb7793, 0x42d69bf8],
        );
        assert_f32_bits(
            factor
                .landmark_jacobian
                .map(|value| value as f32)
                .as_slice(),
            &[
                0x44c477bc, 0xc279df70, 0xc2840b88, 0x44d35468, 0xc2978b08, 0x4180a392,
            ],
        );
        assert!(factor.anchor_pose_jacobian.norm() > 0.0);
        assert!(factor.target_pose_jacobian.norm() > 0.0);
    }

    #[test]
    fn sqrt_marginalization_matches_dense_factor_gram_and_gradient() {
        let j = DMatrix::from_row_slice(
            5,
            4,
            &[
                1., 0., 2., 0., 0., 1., 1., 1., 1., 1., 0., 2., 2., -1., 1., 0., 0.5, 2., 1., -1.,
            ],
        );
        let r = DVector::from_column_slice(&[1., 2., -1., 0.5, 3.]);
        let prior = sqrt_to_sqrt_marginalize(&j, &r, &[0, 2], &[1, 3], DVector::zeros(2)).unwrap();
        let h = &prior.jacobian.transpose() * &prior.jacobian;
        let b = &prior.jacobian.transpose() * &prior.rhs;
        let jk = j.select_columns(&[0, 2]);
        let jm = j.select_columns(&[1, 3]);
        let inv = (&jm.transpose() * &jm).try_inverse().unwrap();
        let expected_h = jk.transpose() * &jk - jk.transpose() * &jm * &inv * jm.transpose() * &jk;
        let expected_b = jk.transpose() * &r - jk.transpose() * &jm * &inv * jm.transpose() * &r;
        assert!((&h - expected_h).norm() < 1e-9);
        assert!((&b - expected_b).norm() < 1e-9);
    }

    #[test]
    fn mixed_pose_only_and_nav_columns_survive_two_window_shifts() {
        let j = DMatrix::from_fn(30, 21, |row, col| {
            (((row + 1) * (col + 2)) % 17) as f64 + 0.1
        });
        let r = DVector::from_element(30, 1.0);
        let p1 = sqrt_to_sqrt_marginalize(
            &j,
            &r,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            &(15..21).collect::<Vec<_>>(),
            DVector::zeros(15),
        )
        .unwrap();
        let p2 = sqrt_to_sqrt_marginalize(
            &p1.jacobian,
            &p1.rhs,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &(10..15).collect::<Vec<_>>(),
            DVector::zeros(10),
        )
        .unwrap();
        assert_eq!(p2.jacobian.ncols(), 10);
        assert_eq!(p2.fej_point.len(), 10);
        assert!(p2.jacobian.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn fej_re_reference_updates_rhs_without_changing_jacobian() {
        let j = DMatrix::from_row_slice(2, 2, &[2., 0., 0., 3.]);
        let r = DVector::from_column_slice(&[1., 2.]);
        let mut p = SqrtPrior {
            jacobian: j.clone(),
            rhs: r,
            fej_point: DVector::zeros(2),
        };
        p.re_reference(&DVector::from_column_slice(&[0.5, -1.0]))
            .unwrap();
        assert_eq!(p.jacobian, j);
        assert_eq!(p.rhs, DVector::from_column_slice(&[0., 5.]));
    }

    #[test]
    fn marginalization_reports_invalid_columns_and_gauge_rank() {
        let j = DMatrix::zeros(3, 2);
        let r = DVector::zeros(3);
        assert_eq!(
            sqrt_to_sqrt_marginalize(&j, &r, &[0], &[1], DVector::zeros(1)),
            Err(MarginalizationError::RankDeficient)
        );
        assert_eq!(
            sqrt_to_sqrt_marginalize(&j, &r, &[2], &[], DVector::zeros(1)),
            Err(MarginalizationError::InvalidColumns)
        );
    }

    #[test]
    #[ignore]
    fn m7_tmp_probe_clean_visual_reduction_trees() {
        // This is a one-shot source-order probe for the frame-4 clean oracle.
        // Keep it ignored: the large capture files are workspace artifacts,
        // not checked-in fixtures.  It is intentionally in this module so it
        // can reuse the exact f32 helpers used by the production reducer.
        fn bits(value: &serde_json::Value) -> u32 {
            value
                .as_str()
                .unwrap_or_else(|| panic!("expected f32 hex string, got {value}"))
                .parse::<u32>()
                .unwrap_or_else(|_| u32::from_str_radix(value.as_str().unwrap(), 16).unwrap())
        }
        fn bit_words(value: &serde_json::Value) -> Vec<u32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| u32::from_str_radix(entry.as_str().unwrap(), 16).unwrap())
                .collect()
        }
        fn actual_f32(value: &serde_json::Value) -> f32 {
            value.as_f64().unwrap() as f32
        }
        fn add_tree(parts: &[Vec<f32>], begin: usize, end: usize, leaf: usize) -> Vec<f32> {
            if end - begin <= leaf {
                let mut result = vec![0.0_f32; parts[0].len()];
                for part in &parts[begin..end] {
                    for (dst, src) in result.iter_mut().zip(part) {
                        *dst += *src;
                    }
                }
                return result;
            }
            let middle = begin + (end - begin) / 2;
            let left = add_tree(parts, begin, middle, leaf);
            let right = add_tree(parts, middle, end, leaf);
            left.into_iter()
                .zip(right)
                .map(|(left, right)| left + right)
                .collect()
        }
        fn count_equal(actual: &[f32], expected: &[u32]) -> usize {
            actual
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| actual.to_bits() == **expected)
                .count()
        }
        fn count_equal_bits(actual: &[f32], expected: &[f32]) -> usize {
            actual
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
                .count()
        }
        fn col_major_to_row_major(words: &[u32], width: usize) -> Vec<f32> {
            (0..width)
                .flat_map(|row| {
                    (0..width).map(move |column| f32::from_bits(words[column * width + row]))
                })
                .collect()
        }
        fn first_mismatch(actual: &[f32], expected: &[u32], width: usize) -> String {
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual.to_bits() != *expected {
                    return format!(
                        "idx={} row={} col={} actual={:08x} expected={expected:08x}",
                        index,
                        index / width,
                        index % width,
                        actual.to_bits(),
                    );
                }
            }
            "none".to_owned()
        }

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let detail_text = std::fs::read_to_string(
            root.join("../../target/m7im15_rust_detail_cleancompare_frame4_20260826.jsonl"),
        )
        .unwrap();
        let detail: serde_json::Value = detail_text
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|record| record["phase"] == "iteration_start" && record["iteration"] == 0)
            .expect("iteration_start detail record");
        let frontier_text = std::fs::read_to_string(
            root.join("../../target/m7im15_rust_frontier_cleancompare_frame4_20260826.jsonl"),
        )
        .unwrap();
        let frontier: serde_json::Value = frontier_text
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|record| record["phase"] == "iteration_start" && record["iteration"] == 0)
            .expect("iteration_start frontier record");
        let imu_text = std::fs::read_to_string(
            root.join("../../target/m7im15_rust_imu_cleancompare_frame4_20260826.jsonl"),
        )
        .unwrap();
        let imu: serde_json::Value = imu_text
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|record| record["iteration"] == 0)
            .expect("iteration-0 IMU record");
        let native: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                root.join("../../target/m7im15_native_step_clean_frame4_repeat_20260826.json"),
            )
            .unwrap(),
        )
        .unwrap();

        let mut factors = detail["landmark_factors"]
            .as_array()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        factors.sort_by_key(|factor| factor["row_span"][0].as_u64().unwrap());
        assert_eq!(factors.len(), 61);
        let mut h_parts = Vec::with_capacity(factors.len());
        let mut b_parts = Vec::with_capacity(factors.len());
        for factor in factors {
            let rows = factor["reduced_rows"].as_array().unwrap();
            let rhs = factor["reduced_rhs"].as_array().unwrap();
            let rows = rows
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(actual_f32)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let rhs = rhs.iter().map(actual_f32).collect::<Vec<_>>();
            let mut h = vec![0.0_f32; 75 * 75];
            for row in 0..75 {
                for column in 0..75 {
                    let mut value = 0.0_f32;
                    for depth in &rows {
                        value = depth[row].mul_add(depth[column], value);
                    }
                    h[row * 75 + column] = value;
                }
            }
            let mut b = vec![0.0_f32; 75];
            for column in 0..75 {
                for (row, depth) in rows.iter().enumerate() {
                    b[column] = depth[column].mul_add(rhs[row], b[column]);
                }
            }
            h_parts.push(h);
            b_parts.push(b);
        }

        let native_record = &native["records"][0];
        let native_h = bit_words(&native_record["dense"]["H"]["f32_bits"]);
        let native_b = bit_words(&native_record["dense"]["b_f32_bits"]);
        assert_eq!(native_h.len(), 75 * 75);
        assert_eq!(native_b.len(), 75);

        let old_visual_h_words =
            bit_words(&frontier["normal_system_stages"]["visual_accumulator"]["h"]["bits"]);
        let old_visual_b_words =
            bit_words(&frontier["normal_system_stages"]["visual_accumulator"]["b"]);
        let flat_h = add_tree(&h_parts, 0, h_parts.len(), h_parts.len());
        let flat_b = add_tree(&b_parts, 0, b_parts.len(), b_parts.len());
        let old_visual_h = col_major_to_row_major(&old_visual_h_words, 75);
        assert_eq!(count_equal_bits(&flat_h, &old_visual_h), 75 * 75);
        assert_eq!(count_equal(&flat_b, &old_visual_b_words), 75);

        let imu_h_words = bit_words(&imu["imu_accumulator"]["h"]["bits"]);
        let imu_b_words = bit_words(&imu["imu_accumulator"]["b"]["bits"]);
        let prior_h_words =
            bit_words(&frontier["normal_system_stages"]["marginal_prior"]["h"]["bits"]);
        let prior_b_words = bit_words(&frontier["normal_system_stages"]["marginal_prior"]["b"]);
        let imu_h = col_major_to_row_major(&imu_h_words, 75);
        let prior_h = col_major_to_row_major(&prior_h_words, 75);
        let capture_root = root.join("../../target/m7im15_gdb_clean_landmark_capture_20260826");
        let read_capture = |ordinal: usize, field: &str| {
            std::fs::read(capture_root.join(format!("{ordinal:03}.{field}_after.bin")))
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let mut cumulative_h = vec![0.0_f32; 75 * 75];
        let mut cumulative_b = vec![0.0_f32; 75];
        for ordinal in 0..h_parts.len() {
            for index in 0..cumulative_h.len() {
                cumulative_h[index] += h_parts[ordinal][index];
            }
            for index in 0..cumulative_b.len() {
                cumulative_b[index] += b_parts[ordinal][index];
            }
            let native_h_col_major = read_capture(ordinal, "h");
            let native_h = (0..75)
                .flat_map(|row| {
                    let value = &native_h_col_major;
                    (0..75).map(move |column| value[column * 75 + row])
                })
                .collect::<Vec<_>>();
            let native_b = read_capture(ordinal, "b");
            println!(
                "ordinal={ordinal} h={} b={} first_h={} first_b={}",
                count_equal(&cumulative_h, &native_h),
                count_equal(&cumulative_b, &native_b),
                first_mismatch(&cumulative_h, &native_h, 75),
                first_mismatch(&cumulative_b, &native_b, 75),
            );
            if ordinal == 0 {
                let row_counts = (0..75)
                    .map(|row| {
                        (0..75)
                            .filter(|column| {
                                cumulative_h[row * 75 + column].to_bits()
                                    != native_h[row * 75 + column]
                            })
                            .count()
                    })
                    .collect::<Vec<_>>();
                let col_counts = (0..75)
                    .map(|column| {
                        (0..75)
                            .filter(|row| {
                                cumulative_h[row * 75 + column].to_bits()
                                    != native_h[row * 75 + column]
                            })
                            .count()
                    })
                    .collect::<Vec<_>>();
                println!("ordinal=0 row_mismatches={row_counts:?}");
                println!("ordinal=0 col_mismatches={col_counts:?}");
            }
        }
        let leaves = [1_usize, 2, 4, 8, 16, 32, 64, 61];
        for &leaf in &leaves {
            let visual_h = add_tree(&h_parts, 0, h_parts.len(), leaf);
            let visual_b = add_tree(&b_parts, 0, b_parts.len(), leaf);
            let mut h = visual_h.clone();
            for index in 0..h.len() {
                h[index] += imu_h[index];
                h[index] += prior_h[index];
            }
            let mut b = visual_b.clone();
            for index in 0..b.len() {
                b[index] += f32::from_bits(imu_b_words[index]);
                b[index] += f32::from_bits(prior_b_words[index]);
            }
            println!(
                "leaf={leaf} visual_h={} visual_b={} final_h={} final_b={} first_h={} first_b={}",
                count_equal_bits(&visual_h, &old_visual_h),
                count_equal(&visual_b, &old_visual_b_words),
                count_equal(&h, &native_h),
                count_equal(&b, &native_b),
                first_mismatch(&h, &native_h, 75),
                first_mismatch(&b, &native_b, 75),
            );
        }
        let _ = bits; // Keep the parser helper available while probing captures.
    }

    #[test]
    fn m7_probe_relpose_candidate_vs_generic_all_iter0_pairs() {
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            let q = std::hint::black_box(q);
            let t = std::hint::black_box(t);
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }
        fn lanes(pose: F32Pose) -> [u32; 7] {
            [
                pose.rotation.i.to_bits(),
                pose.rotation.j.to_bits(),
                pose.rotation.k.to_bits(),
                pose.rotation.w.to_bits(),
                pose.translation.x.to_bits(),
                pose.translation.y.to_bits(),
                pose.translation.z.to_bits(),
            ]
        }
        let states = std::hint::black_box([
            pose(
                [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
                [0x00000000, 0x00000000, 0x00000000],
            ),
            pose(
                [0xbd5cdea6, 0xbf4d341f, 0xbb2aa78b, 0x3f186f67],
                [0x39c8bb60, 0xb8cebae8, 0xbb0527fc],
            ),
            pose(
                [0xbd5cf5f2, 0xbf4d97c0, 0xbbac4997, 0x3f17e7a6],
                [0x3af3e90a, 0xb9fd97e2, 0xbbfc3924],
            ),
            pose(
                [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
                [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
            ),
            pose(
                [0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719],
                [0x3c02c75a, 0xbb0c3bda, 0xbcdc2b7f],
            ),
        ]);
        let extrinsics = std::hint::black_box([
            pose(
                [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
                [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
            ),
            pose(
                [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
                [0xbc76fa76, 0x3d290319, 0x3b4f4832],
            ),
        ]);
        let host = std::hint::black_box(states[0]);
        for target_frame in 0..5 {
            for target_cam in 0..2 {
                if target_frame == 0 && target_cam == 0 {
                    continue;
                }
                let tmp2 = std::hint::black_box(extrinsics[target_cam].inverse());
                let target = std::hint::black_box(states[target_frame]);
                // OLD production FEJ chain, spelled locally as it was before
                // the out-of-line computeRelPose candidate was introduced.
                let target_inverse_rotation = sophus_so3_inverse(target.rotation);
                let relative_rotation =
                    sophus_quat_product_f32(target_inverse_rotation, host.rotation);
                let relative_translation = sophus_rotate_difference_f32(
                    target_inverse_rotation,
                    host.translation,
                    target.translation,
                );
                let generic = std::hint::black_box(F32Pose {
                    rotation: sophus_quat_product_f32(tmp2.rotation, relative_rotation),
                    translation: sophus_rotate_f32(tmp2.rotation, relative_translation)
                        + tmp2.translation,
                });
                let candidate = std::hint::black_box(sophus_compute_relpose_tmp_out_of_line_f32(
                    tmp2, target, host,
                ));
                let candidate_lanes = lanes(candidate);
                let generic_lanes = lanes(generic);
                if (target_frame == 1 && target_cam == 1)
                    || (target_frame == 4 && (target_cam == 0 || target_cam == 1))
                {
                    let candidate_raw = std::hint::black_box(
                        m7_relpose_packet_product_raw_for_test(tmp2.rotation, relative_rotation),
                    );
                    let generic_raw = std::hint::black_box(m7_generic_packet_product_raw_for_test(
                        tmp2.rotation,
                        relative_rotation,
                    ));
                    println!(
                        concat!(
                            "pair={}/{} raw_candidate=",
                            "{:08x},{:08x},{:08x},{:08x} raw_generic=",
                            "{:08x},{:08x},{:08x},{:08x} generic_q=",
                            "{:08x},{:08x},{:08x},{:08x}"
                        ),
                        target_frame,
                        target_cam,
                        candidate_raw[0].to_bits(),
                        candidate_raw[1].to_bits(),
                        candidate_raw[2].to_bits(),
                        candidate_raw[3].to_bits(),
                        generic_raw[0].to_bits(),
                        generic_raw[1].to_bits(),
                        generic_raw[2].to_bits(),
                        generic_raw[3].to_bits(),
                        generic.rotation.i.to_bits(),
                        generic.rotation.j.to_bits(),
                        generic.rotation.k.to_bits(),
                        generic.rotation.w.to_bits(),
                    );
                }
                let diffs = candidate_lanes
                    .iter()
                    .zip(generic_lanes.iter())
                    .enumerate()
                    .filter_map(|(index, (actual, expected))| {
                        (actual != expected)
                            .then_some(format!("{index}:{actual:08x}/{expected:08x}"))
                    })
                    .collect::<Vec<_>>();
                let d_candidate = eigen_adjoint_times_rotation_blocks_f32(
                    candidate,
                    eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                    1.0,
                );
                let d_generic = eigen_adjoint_times_rotation_blocks_f32(
                    generic,
                    eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                    1.0,
                );
                let d_diffs = d_candidate
                    .as_slice()
                    .iter()
                    .zip(d_generic.as_slice().iter())
                    .enumerate()
                    .filter_map(|(index, (actual, expected))| {
                        (actual.to_bits() != expected.to_bits()).then_some(format!(
                            "{index}:{:08x}/{:08x}",
                            actual.to_bits(),
                            expected.to_bits()
                        ))
                    })
                    .collect::<Vec<_>>();
                let drel_diff_count = |actual: F32Pose| {
                    let d_actual = eigen_adjoint_times_rotation_blocks_f32(
                        actual,
                        eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                        1.0,
                    );
                    d_actual
                        .as_slice()
                        .iter()
                        .zip(d_generic.as_slice().iter())
                        .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                        .count()
                };
                let candidate_rotation_only = F32Pose {
                    rotation: candidate.rotation,
                    translation: generic.translation,
                };
                let candidate_translation_only = F32Pose {
                    rotation: generic.rotation,
                    translation: candidate.translation,
                };
                if !d_diffs.is_empty() {
                    println!(
                        concat!(
                            "pair={}/{} intermediate ",
                            "cand_rot={:08x},{:08x},{:08x},{:08x} ",
                            "gen_rot={:08x},{:08x},{:08x},{:08x} ",
                            "cand_t={:08x},{:08x},{:08x} ",
                            "gen_t={:08x},{:08x},{:08x} ",
                            "drel_total={} drel_rot_only={} drel_trans_only={}"
                        ),
                        target_frame,
                        target_cam,
                        candidate.rotation.i.to_bits(),
                        candidate.rotation.j.to_bits(),
                        candidate.rotation.k.to_bits(),
                        candidate.rotation.w.to_bits(),
                        generic.rotation.i.to_bits(),
                        generic.rotation.j.to_bits(),
                        generic.rotation.k.to_bits(),
                        generic.rotation.w.to_bits(),
                        candidate.translation.x.to_bits(),
                        candidate.translation.y.to_bits(),
                        candidate.translation.z.to_bits(),
                        generic.translation.x.to_bits(),
                        generic.translation.y.to_bits(),
                        generic.translation.z.to_bits(),
                        d_diffs.len(),
                        drel_diff_count(candidate_rotation_only),
                        drel_diff_count(candidate_translation_only),
                    );
                }
                if !diffs.is_empty() || !d_diffs.is_empty() {
                    println!(
                        "pair={target_frame}/{target_cam} pose_diffs=[{}] drel_diffs=[{}]",
                        diffs.join(","),
                        d_diffs.join(",")
                    );
                }
            }
        }
    }

    #[test]
    fn m7im15_relpose_dual_pass_native_drel_and_intermediates() {
        // Direct native capture for the shared frame0/cam0 -> frame3/cam1
        // call at optimization passes 0 and 1.  Keep both passes in one
        // fixture because pass 0 is the guard against a schedule change that
        // merely fixes the later six cross lanes.
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }
        fn lanes(pose: F32Pose) -> [u32; 7] {
            [
                pose.rotation.i.to_bits(),
                pose.rotation.j.to_bits(),
                pose.rotation.k.to_bits(),
                pose.rotation.w.to_bits(),
                pose.translation.x.to_bits(),
                pose.translation.y.to_bits(),
                pose.translation.z.to_bits(),
            ]
        }
        fn format_lanes(values: [u32; 7]) -> String {
            values
                .iter()
                .map(|value| format!("{value:08x}"))
                .collect::<Vec<_>>()
                .join(",")
        }
        fn drel(pose: F32Pose, host: F32Pose) -> SMatrix<f32, 6, 6> {
            eigen_adjoint_times_rotation_blocks_f32(
                pose,
                eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
                1.0,
            )
        }
        fn exact(actual: &SMatrix<f32, 6, 6>, expected: &[u32; 36]) -> usize {
            actual
                .as_slice()
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| actual.to_bits() == **expected)
                .count()
        }
        let host = pose(
            [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
            [0x00000000, 0x00000000, 0x00000000],
        );
        let target_ext = pose(
            [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
            [0xbc76fa76, 0x3d290319, 0x3b4f4832],
        );
        let cases = [
            (
                pose(
                    [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
                    [0x3b8ebfa0, 0xba9428f4, 0xbc857642],
                ),
                [
                    0xba954ed4, 0xbbce8a7c, 0x3ad34473, 0x3f7ffe93, 0xbde1e309, 0xbc8b6eac,
                    0xbacb5e9a,
                ],
                [
                    0x3c31fe46, 0xbc520562, 0xbf33585a, 0x3f36a0a8, 0xbd27a81a, 0xbd05555f,
                    0xbb8ddf30,
                ],
                [
                    0x3de42768, 0x3e92d17a, 0xbf7395cf, 0x00000000, 0x00000000, 0x00000000,
                    0x3f7e3e31, 0xbd881872, 0x3dc51f14, 0x00000000, 0x00000000, 0x00000000,
                    0xbd11f0b3, 0xbf74a889, 0xbe9599d5, 0x00000000, 0x00000000, 0x00000000,
                    0x3d03f3e6, 0xbd21806f, 0xbc04e3d7, 0x3de42768, 0x3e92d17a, 0xbf7395cf,
                    0xbb6030cb, 0xb9bcd31b, 0x3d0f8f42, 0x3f7e3e31, 0xbd881872, 0x3dc51f14,
                    0x3bb0151d, 0xbc416c25, 0x3d1b7a6e, 0xbd11f0b3, 0xbf74a889, 0xbe9599d5,
                ],
            ),
            (
                pose(
                    [0xbd5e3af5, 0xbf4e66c7, 0xbbc31a02, 0x3f16cb91],
                    [0x3b2b6284, 0xbb51bffc, 0xbd4c7b94],
                ),
                [
                    0xba7446f7, 0xbbcdd358, 0x3adb39c1, 0x3f7ffe96, 0xbddfb9b6, 0xbd47e71b,
                    0xbc527981,
                ],
                [
                    0x3c342b3c, 0xbc4f602d, 0xbf3355a4, 0x3f36a35f, 0xbd235394, 0xbd83bd75,
                    0xbc801294,
                ],
                [
                    0x3de3fe8e, 0x3e9306e1, 0xbf738e5b, 0x00000000, 0x00000000, 0x00000000,
                    0x3f7e3f01, 0xbd87ecbc, 0x3dc4f985, 0x00000000, 0x00000000, 0x00000000,
                    0xbd118207, 0xbf74a0e5, 0xbe95cd73, 0x00000000, 0x00000000, 0x00000000,
                    0x3d8687db, 0xbd228425, 0xbb8c8d8c, 0x3de3fe8e, 0x3e9306e1, 0xbf738e5b,
                    0xbbecbb07, 0xbc3f8e51, 0x3d8841e7, 0x3f7e3f01, 0xbd87ecbc, 0x3dc4f985,
                    0x3b7e5e58, 0xbc360bff, 0x3d12b627, 0xbd118207, 0xbf74a0e5, 0xbe95cd73,
                ],
            ),
        ];
        for (pass, (target, native_relative, native_tmp, native_drel)) in
            cases.into_iter().enumerate()
        {
            let target_camera_from_imu = target_ext.inverse();
            let relative = sophus_relative_imu_f32(target, host);
            let generic = F32Pose {
                rotation: sophus_quat_product_f32(
                    target_camera_from_imu.rotation,
                    relative.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    relative.translation,
                ) + target_camera_from_imu.translation,
            };
            let candidate =
                sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
            let generic_rotated_translation =
                sophus_rotate_f32(target_camera_from_imu.rotation, relative.translation);
            let candidate_rotated_translation = sophus_rotate_relpose_out_of_line_f32(
                target_camera_from_imu.rotation,
                relative.translation,
            );
            let packet_rotated_translation = sophus_rotate_step_packet_f32(
                target_camera_from_imu.rotation,
                relative.translation,
            );
            let packet = F32Pose {
                rotation: sophus_quat_product_f32(
                    target_camera_from_imu.rotation,
                    relative.rotation,
                ),
                translation: packet_rotated_translation + target_camera_from_imu.translation,
            };
            let generic_drel = drel(generic, host);
            let candidate_drel = drel(candidate, host);
            let packet_drel = drel(packet, host);
            let relative_expected = pose(
                [
                    native_relative[0],
                    native_relative[1],
                    native_relative[2],
                    native_relative[3],
                ],
                [native_relative[4], native_relative[5], native_relative[6]],
            );
            let tmp_expected = pose(
                [native_tmp[0], native_tmp[1], native_tmp[2], native_tmp[3]],
                [native_tmp[4], native_tmp[5], native_tmp[6]],
            );
            let generic_exact = exact(&generic_drel, &native_drel);
            let candidate_exact = exact(&candidate_drel, &native_drel);
            let packet_exact = exact(&packet_drel, &native_drel);
            println!(
                "dual_pass={pass} relative={} relative_exact={} tmp2={} relative_t={} generic_rot={} candidate_rot={} packet_rot={} generic_tmp={} generic_tmp_exact={} candidate_tmp={} candidate_tmp_exact={} packet_tmp={} packet_tmp_exact={} generic_drel_exact={generic_exact}/36 candidate_drel_exact={candidate_exact}/36 packet_drel_exact={packet_exact}/36",
                format_lanes(lanes(relative)),
                lanes(relative) == lanes(relative_expected),
                format_lanes(lanes(target_camera_from_imu)),
                format_lanes([
                    relative.translation.x.to_bits(),
                    relative.translation.y.to_bits(),
                    relative.translation.z.to_bits(),
                    0,
                    0,
                    0,
                    0,
                ]),
                format_lanes(lanes(F32Pose { rotation: UnitQuaternion::identity(), translation: generic_rotated_translation })),
                format_lanes(lanes(F32Pose { rotation: UnitQuaternion::identity(), translation: candidate_rotated_translation })),
                format_lanes(lanes(F32Pose { rotation: UnitQuaternion::identity(), translation: packet_rotated_translation })),
                format_lanes(lanes(generic)),
                lanes(generic) == lanes(tmp_expected),
                format_lanes(lanes(candidate)),
                lanes(candidate) == lanes(tmp_expected),
                format_lanes(lanes(packet)),
                lanes(packet) == lanes(tmp_expected),
            );
            // The direct GDB `T_t_h_sophus_qt_u32` capture is retained in the
            // printout above, but this helper's packet schedule is not the
            // final `tmp` boundary by itself.  Do not turn that intermediate
            // observation into a production assertion: the composed generic
            // path is the contract being compared below.
            assert_eq!(
                generic_exact,
                if pass == 0 { 36 } else { 30 },
                "current generic d_rel_h exact count at pass {pass}"
            );
            assert_eq!(
                candidate_exact,
                if pass == 0 { 23 } else { 19 },
                "out-of-line candidate d_rel_h exact count at pass {pass}"
            );
            assert_eq!(
                lanes(packet),
                lanes(tmp_expected),
                "step-only packet SO3 action must reproduce native tmp at pass {pass}"
            );
            assert_eq!(
                packet_exact, 36,
                "step-only packet SO3 action must reproduce native d_rel_h at pass {pass}"
            );
        }
    }

    #[test]
    fn m7im15_current_relative_packet_pass0_3_exact() {
        // The stage-only native capture stops immediately before the current
        // camera suffix and records the normalized target inverse in xmm0,
        // the current host quaternion in rdi, and the normalized relative
        // result at 0x3055f6.  Feed those exact words to the isolated packet
        // helper so this test covers the quaternion product/normalization
        // boundary independently of translation and camera extrinsics.
        fn quaternion(words: [u32; 4]) -> UnitQuaternion<f32> {
            UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(words[3]),
                f32::from_bits(words[0]),
                f32::from_bits(words[1]),
                f32::from_bits(words[2]),
            ))
        }

        let target_inverse = [
            [0x3d5e760f, 0x3f4e5e85, 0x3bc2d95f, 0x3f16d68a],
            [0x3d5e3af6, 0x3f4e66c8, 0x3bc31a03, 0x3f16cb92],
            [0x3d5e2068, 0x3f4e76f4, 0x3bc519c6, 0x3f16b589],
            [0x3d5cd678, 0x3f4e9bf7, 0x3bd4932e, 0x3f168457],
        ];
        let host_current = [
            [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
            [0xbd587ff0, 0xbf4d69a0, 0x38d8e054, 0x3f182def],
            [0xbd589afe, 0xbf4d86a4, 0x38cfb660, 0x3f180696],
            [0xbd5782db, 0xbf4db536, 0xb9aa33e3, 0x3f17c91a],
        ];
        let expected = [
            [0x3bc353bb, 0x3bc9740e, 0x3b240734, 0x3f7ffd65],
            [0x3bc3e5d4, 0x3bceeae2, 0x3b2fc253, 0x3f7ffd49],
            [0x3bc3ffb8, 0x3bc40dbc, 0x3b33b475, 0x3f7ffd69],
            [0x3bc409b5, 0x3bbc4711, 0x3b3739d6, 0x3f7ffd7d],
        ];

        for pass in 0..4 {
            for _target_cam in 0..2 {
                let result = sophus_quat_product_current_packet_f32(
                    quaternion(target_inverse[pass]),
                    quaternion(host_current[pass]),
                );
                let q = result.quaternion();
                assert_f32_bits(&[q.i, q.j, q.k, q.w], &expected[pass]);
            }
        }
    }

    #[test]
    #[ignore = "requires pinned external target-frame3 relpose capture"]
    fn m7im15_current_relpose_targetframe3_cam0_cam1_schedule_probe() {
        // This is a diagnostic sidecar for the first iter3 visual factor.  It
        // uses the exact current/FEJ endpoint words from the native fixture,
        // then prints the value-chain schedule variants.  No production
        // branch is selected by this test; it exists to make the cam0 path
        // (which was not present in the original cam1-only fixture) directly
        // comparable before any call-site change.
        fn pose(words: &[u32]) -> F32Pose {
            assert_eq!(words.len(), 7);
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(words[3]),
                    f32::from_bits(words[0]),
                    f32::from_bits(words[1]),
                    f32::from_bits(words[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(words[4]),
                    f32::from_bits(words[5]),
                    f32::from_bits(words[6]),
                ),
            }
        }
        fn words(value: &serde_json::Value, key: &str) -> Vec<u32> {
            value[key]
                .as_array()
                .unwrap_or_else(|| panic!("missing array {key}"))
                .iter()
                .map(|word| {
                    let word = word.as_str().expect("fixture bit must be a string");
                    u32::from_str_radix(word, 16).expect("fixture bit must be hexadecimal")
                })
                .collect()
        }
        fn lanes(value: F32Pose) -> [u32; 7] {
            [
                value.rotation.i.to_bits(),
                value.rotation.j.to_bits(),
                value.rotation.k.to_bits(),
                value.rotation.w.to_bits(),
                value.translation.x.to_bits(),
                value.translation.y.to_bits(),
                value.translation.z.to_bits(),
            ]
        }
        fn exact_pose(actual: F32Pose, expected: &[u32]) -> usize {
            lanes(actual)
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| actual == expected)
                .count()
        }
        fn exact_matrix(actual: &SMatrix<f32, 6, 6>, expected: &[u32]) -> usize {
            actual
                .as_slice()
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| actual.to_bits() == **expected)
                .count()
        }
        fn fmt7(value: F32Pose) -> String {
            lanes(value)
                .iter()
                .map(|word| format!("{word:08x}"))
                .collect::<Vec<_>>()
                .join(" ")
        }

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let native_path =
            root.join("../../target/m7im15_relpose_targetframe3_pass0_3_20260827.json");
        let cam1_path = root.join("../../target/m7im15_step_relpose_inputs_gdb_20260827.json");
        let native: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&native_path)
                .expect("target-frame3 native capture is required"),
        )
        .expect("target-frame3 native capture must be valid JSON");
        let cam1: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&cam1_path).expect("cam1 native fixture is required"),
        )
        .expect("cam1 native fixture must be valid JSON");
        let native_records = native["records"].as_array().expect("native records array");
        let cam1_records = cam1["records"].as_array().expect("cam1 records array");

        let host_ext = pose(&[
            0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
        ]);
        let target_exts = [
            host_ext,
            pose(&[
                0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6, 0xbc76fa76, 0x3d290319, 0x3b4f4832,
            ]),
        ];
        let mut total = [[0usize; 5]; 2];
        let mut total_drel_h = [[0usize; 5]; 2];
        let mut total_drel_t = [[0usize; 5]; 2];
        for pass in 0..4 {
            let host_record = cam1_records
                .iter()
                .find(|record| record["optimization_linearization_pass_index"] == pass)
                .expect("cam1 fixture pass");
            let host_current = pose(&words(host_record, "host_pose_current_qt_u32"));
            let host_fej = pose(&words(host_record, "host_pose_lin_qt_u32"));
            let native_pass = native_records
                .iter()
                .filter(|record| record["pass"] == pass)
                .collect::<Vec<_>>();
            assert_eq!(native_pass.len(), 2);
            for native_record in native_pass {
                let target_cam = native_record["target_cam"].as_u64().unwrap() as usize;
                let target_current = pose(&words(native_record, "target_current_qt_u32"));
                let target_fej = pose(&words(native_record, "target_lin_qt_u32"));
                let expected = words(native_record, "T_t_h_sophus_qt_u32");
                let expected_h = words(native_record, "d_rel_d_h_column_major_u32");
                let expected_t = words(native_record, "d_rel_d_t_column_major_u32");
                let target_camera_from_imu = target_exts[target_cam].inverse();
                let target_inverse_rotation = sophus_so3_inverse(target_current.rotation);

                // Production current value chain: scalar relative action,
                // camera-prefix/suffix quaternion packets, generic SO3 action
                // at both camera composition points.
                let target_imu_from_host = sophus_relative_imu_f32(target_current, host_current);
                let tmp_generic = F32Pose {
                    rotation: sophus_quat_product_camera_prefix_f32(
                        target_camera_from_imu.rotation,
                        target_imu_from_host.rotation,
                    ),
                    translation: sophus_rotate_f32(
                        target_camera_from_imu.rotation,
                        target_imu_from_host.translation,
                    ) + target_camera_from_imu.translation,
                };
                let result_generic = F32Pose {
                    rotation: sophus_quat_product_camera_suffix_f32(
                        tmp_generic.rotation,
                        host_ext.rotation,
                    ),
                    translation: sophus_rotate_f32(tmp_generic.rotation, host_ext.translation)
                        + tmp_generic.translation,
                };

                // Isolate the current relative quaternion packet while
                // retaining the already-proven translation/camera schedule.
                let target_imu_from_host_packet = F32Pose {
                    rotation: sophus_quat_product_current_packet_f32(
                        target_inverse_rotation,
                        host_current.rotation,
                    ),
                    translation: sophus_rotate_difference_visual_f32(
                        target_inverse_rotation,
                        host_current.translation,
                        target_current.translation,
                    ),
                };
                let tmp_current_packet = F32Pose {
                    rotation: sophus_quat_product_camera_prefix_f32(
                        target_camera_from_imu.rotation,
                        target_imu_from_host_packet.rotation,
                    ),
                    translation: sophus_rotate_f32(
                        target_camera_from_imu.rotation,
                        target_imu_from_host_packet.translation,
                    ) + target_camera_from_imu.translation,
                };
                let result_current_packet = F32Pose {
                    rotation: sophus_quat_product_camera_suffix_f32(
                        tmp_current_packet.rotation,
                        host_ext.rotation,
                    ),
                    translation: sophus_rotate_f32(
                        tmp_current_packet.rotation,
                        host_ext.translation,
                    ) + tmp_current_packet.translation,
                };

                // The two candidate step-only actions are kept independent so
                // a match identifies the exact SO3*Vector call-site boundary.
                let mut variants = [result_generic; 5];
                for (variant, (tmp_action, suffix_action)) in
                    [(false, false), (true, false), (false, true), (true, true)]
                        .into_iter()
                        .enumerate()
                {
                    let tmp_translation = if tmp_action {
                        sophus_rotate_step_packet_f32(
                            target_camera_from_imu.rotation,
                            target_imu_from_host.translation,
                        )
                    } else {
                        sophus_rotate_f32(
                            target_camera_from_imu.rotation,
                            target_imu_from_host.translation,
                        )
                    } + target_camera_from_imu.translation;
                    let tmp = F32Pose {
                        rotation: tmp_generic.rotation,
                        translation: tmp_translation,
                    };
                    let suffix_translation = if suffix_action {
                        sophus_rotate_step_packet_f32(tmp.rotation, host_ext.translation)
                    } else {
                        sophus_rotate_f32(tmp.rotation, host_ext.translation)
                    };
                    variants[variant + 1] = F32Pose {
                        rotation: tmp.rotation,
                        translation: suffix_translation + tmp.translation,
                    };
                }

                // A packetized relative translation is also reported as a
                // separate family; it is not selected by production here.
                let relative_packet = F32Pose {
                    rotation: target_imu_from_host.rotation,
                    translation: sophus_rotate_difference_f32(
                        sophus_so3_inverse(target_current.rotation),
                        host_current.translation,
                        target_current.translation,
                    ),
                };
                let packet_tmp = F32Pose {
                    rotation: sophus_quat_product_camera_prefix_f32(
                        target_camera_from_imu.rotation,
                        relative_packet.rotation,
                    ),
                    translation: sophus_rotate_step_packet_f32(
                        target_camera_from_imu.rotation,
                        relative_packet.translation,
                    ) + target_camera_from_imu.translation,
                };
                variants[4] = F32Pose {
                    rotation: sophus_quat_product_camera_suffix_f32(
                        packet_tmp.rotation,
                        host_ext.rotation,
                    ),
                    translation: sophus_rotate_step_packet_f32(
                        packet_tmp.rotation,
                        host_ext.translation,
                    ) + packet_tmp.translation,
                };

                let target_inverse_fej = sophus_so3_inverse(target_fej.rotation);
                let relative_fej = sophus_relative_imu_f32(target_fej, host_fej);
                let tmp_fej = F32Pose {
                    rotation: sophus_quat_product_f32(
                        target_camera_from_imu.rotation,
                        relative_fej.rotation,
                    ),
                    translation: sophus_rotate_step_packet_f32(
                        target_camera_from_imu.rotation,
                        relative_fej.translation,
                    ) + target_camera_from_imu.translation,
                };
                let drel_h = eigen_adjoint_times_rotation_blocks_f32(
                    tmp_fej,
                    eigen_quaternion_matrix_f32(sophus_so3_inverse(host_fej.rotation)),
                    1.0,
                );
                let drel_t = eigen_adjoint_times_rotation_blocks_f32(
                    target_camera_from_imu,
                    eigen_quaternion_matrix_f32(target_inverse_fej),
                    -1.0,
                );
                println!(
                    "m7im15_current_relpose pass={pass} cam={target_cam} native={} generic={} packet={} exact={}/7 packet_exact={}/7 variants={:?} native_drel={}/{} rust_drel_h={}/36 rust_drel_t={}/36",
                    expected.iter().map(|word| format!("{word:08x}")).collect::<Vec<_>>().join(" "),
                    fmt7(result_generic),
                    fmt7(result_current_packet),
                    exact_pose(result_generic, &expected),
                    exact_pose(result_current_packet, &expected),
                    variants.iter().map(|value| exact_pose(*value, &expected)).collect::<Vec<_>>(),
                    exact_matrix(&drel_h, &expected_h),
                    exact_matrix(&drel_t, &expected_t),
                    exact_matrix(&drel_h, &expected_h),
                    exact_matrix(&drel_t, &expected_t),
                );
                for (index, value) in variants.iter().enumerate() {
                    total[target_cam][index] += exact_pose(*value, &expected);
                }
                total_drel_h[target_cam][0] += exact_matrix(&drel_h, &expected_h);
                total_drel_t[target_cam][0] += exact_matrix(&drel_t, &expected_t);
            }
        }
        println!("m7im15_current_relpose totals cam0={total:?} drel_h={total_drel_h:?} drel_t={total_drel_t:?}");
    }

    #[test]
    #[ignore = "requires pinned external target-frame3 transform captures"]
    fn m7im15_current_packet_transform_matrix_pass0_3_probe() {
        // The target-frame3 relpose fixture records the exact current endpoint
        // words, while the current-transform fixture records the independent
        // inlined linearizePoint matrix.  Compare the packet-relative-
        // quaternion variant against that matrix, keeping the existing
        // translation and camera prefix/suffix schedules unchanged.
        fn pose(words: &[u32]) -> F32Pose {
            assert_eq!(words.len(), 7);
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(words[3]),
                    f32::from_bits(words[0]),
                    f32::from_bits(words[1]),
                    f32::from_bits(words[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(words[4]),
                    f32::from_bits(words[5]),
                    f32::from_bits(words[6]),
                ),
            }
        }
        fn words(value: &serde_json::Value, key: &str) -> Vec<u32> {
            value[key]
                .as_array()
                .unwrap_or_else(|| panic!("missing array {key}"))
                .iter()
                .map(|word| {
                    u32::from_str_radix(word.as_str().expect("fixture bit must be a string"), 16)
                        .expect("fixture bit must be hexadecimal")
                })
                .collect()
        }
        fn matrix_bits(value: F32Pose) -> [u32; 16] {
            let rotation = eigen_quaternion_matrix_f32(value.rotation);
            [
                rotation[(0, 0)].to_bits(),
                rotation[(1, 0)].to_bits(),
                rotation[(2, 0)].to_bits(),
                0,
                rotation[(0, 1)].to_bits(),
                rotation[(1, 1)].to_bits(),
                rotation[(2, 1)].to_bits(),
                0,
                rotation[(0, 2)].to_bits(),
                rotation[(1, 2)].to_bits(),
                rotation[(2, 2)].to_bits(),
                0,
                value.translation.x.to_bits(),
                value.translation.y.to_bits(),
                value.translation.z.to_bits(),
                0x3f800000,
            ]
        }
        fn exact(actual: &[u32], expected: &[u32]) -> usize {
            actual
                .iter()
                .zip(expected)
                .filter(|(actual, expected)| actual == expected)
                .count()
        }
        fn pose_q_bits(value: F32Pose) -> String {
            [
                value.rotation.i.to_bits(),
                value.rotation.j.to_bits(),
                value.rotation.k.to_bits(),
                value.rotation.w.to_bits(),
            ]
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect::<Vec<_>>()
            .join(" ")
        }
        fn first_mismatch(actual: &[u32], expected: &[u32]) -> String {
            actual
                .iter()
                .zip(expected)
                .enumerate()
                .find_map(|(index, (actual, expected))| {
                    (actual != expected)
                        .then(|| format!("idx={index} native={expected:08x} rust={actual:08x}"))
                })
                .unwrap_or_else(|| "none".to_string())
        }

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let native_path =
            root.join("../../target/m7im15_relpose_targetframe3_pass0_3_20260827.json");
        let matrix_path =
            root.join("../../target/m7im15_current_transform_factor_fixture_20260827.json");
        let native: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&native_path)
                .expect("target-frame3 native capture is required"),
        )
        .expect("target-frame3 native capture must be valid JSON");
        let matrix_fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&matrix_path).expect("current-transform fixture is required"),
        )
        .expect("current-transform fixture must be valid JSON");
        let native_records = native["records"].as_array().expect("native records array");

        let host_ext = pose(&[
            0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc, 0x3ba86617,
        ]);
        let target_exts = [
            host_ext,
            pose(&[
                0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6, 0xbc76fa76, 0x3d290319, 0x3b4f4832,
            ]),
        ];

        for pass in 0..4 {
            let fixture_case = &matrix_fixture["current_transform_cases"][pass];
            for target_cam in 0..2 {
                let native_record = native_records
                    .iter()
                    .find(|record| record["pass"] == pass && record["target_cam"] == target_cam)
                    .expect("target-frame3 native pass/camera record");
                let fixture_observation = fixture_case["observations"]
                    .as_array()
                    .expect("fixture observations array")
                    .iter()
                    .find(|observation| observation["target_cam"] == target_cam)
                    .expect("current-transform camera observation");
                let expected_matrix = words(fixture_observation, "native_T_t_h_bits_column_major");
                assert_eq!(expected_matrix.len(), 16);
                // The native inlined current visual branch receives the
                // effective endpoint returned by PoseStateWithLin::getPose.
                // For this fixture target frame 3 is non-linearized, so that
                // endpoint is its raw pose_linearized payload.  The host
                // endpoint remains current; target translation is likewise
                // the value-side endpoint selected by getPose().
                let target_current = pose(&words(native_record, "target_lin_qt_u32"));
                let host_current = pose(&words(native_record, "host_current_qt_u32"));
                let target_camera_from_imu = target_exts[target_cam].inverse();
                let target_inverse_rotation =
                    sophus_quat_inverse_current_packet_f32(target_current.rotation);
                let relative_packet = F32Pose {
                    rotation: sophus_quat_product_current_packet_f32(
                        target_inverse_rotation,
                        host_current.rotation,
                    ),
                    translation: sophus_rotate_difference_visual_f32(
                        target_inverse_rotation,
                        host_current.translation,
                        target_current.translation,
                    ),
                };
                let tmp_packet = F32Pose {
                    rotation: sophus_quat_product_camera_prefix_f32(
                        target_camera_from_imu.rotation,
                        relative_packet.rotation,
                    ),
                    translation: sophus_rotate_f32(
                        target_camera_from_imu.rotation,
                        relative_packet.translation,
                    ) + target_camera_from_imu.translation,
                };
                let result_packet = F32Pose {
                    rotation: sophus_quat_product_camera_suffix_f32(
                        tmp_packet.rotation,
                        host_ext.rotation,
                    ),
                    translation: sophus_rotate_f32(tmp_packet.rotation, host_ext.translation)
                        + tmp_packet.translation,
                };
                let actual_packet = matrix_bits(result_packet);
                println!(
                    "m7im15_current_packet_matrix pass={pass} cam={target_cam} exact={}/16 first={} q={} t={} target_ext_inv={} target_pose_inv={} relative={} prefix={}",
                    exact(&actual_packet, &expected_matrix),
                    first_mismatch(&actual_packet, &expected_matrix),
                    pose_q_bits(result_packet),
                    [
                        result_packet.translation.x.to_bits(),
                        result_packet.translation.y.to_bits(),
                        result_packet.translation.z.to_bits(),
                    ]
                    .iter()
                    .map(|word| format!("{word:08x}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                    pose_q_bits(target_camera_from_imu),
                    pose_q_bits(F32Pose {
                        rotation: target_inverse_rotation,
                        translation: Vector3::zeros(),
                    }),
                    pose_q_bits(relative_packet),
                    pose_q_bits(tmp_packet),
                );
            }
        }
    }

    #[test]
    fn m7_probe_candidate_weighted_operands_track120_frame1_cam1() {
        fn pose(q: [u32; 4], t: [u32; 3]) -> F32Pose {
            let q = std::hint::black_box(q);
            let t = std::hint::black_box(t);
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q[3]),
                    f32::from_bits(q[0]),
                    f32::from_bits(q[1]),
                    f32::from_bits(q[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(t[0]),
                    f32::from_bits(t[1]),
                    f32::from_bits(t[2]),
                ),
            }
        }
        fn print_matrix<const R: usize, const C: usize>(name: &str, matrix: &SMatrix<f32, R, C>) {
            let bits = matrix
                .as_slice()
                .iter()
                .map(|value| format!("{:08x}", value.to_bits()))
                .collect::<Vec<_>>();
            println!("{name}_column_major={}", bits.join(","));
            let rows = (0..R)
                .flat_map(|row| {
                    (0..C).map(move |column| format!("{:08x}", matrix[(row, column)].to_bits()))
                })
                .collect::<Vec<_>>();
            println!("{name}_row_major={}", rows.join(","));
        }

        let host = pose(
            [0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd],
            [0x00000000, 0x00000000, 0x00000000],
        );
        let target = pose(
            [0xbd5cdea6, 0xbf4d341f, 0xbb2aa78b, 0x3f186f67],
            [0x39c8bb60, 0xb8cebae8, 0xbb0527fc],
        );
        let host_ext = pose(
            [0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e],
            [0xbc896b48, 0xbd8d2fdc, 0x3ba86617],
        );
        let target_ext = pose(
            [0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6],
            [0xbc76fa76, 0x3d290319, 0x3b4f4832],
        );
        let landmark = InverseDistanceLandmark {
            anchor_pose: 0,
            anchor_camera_id: 0,
            direction: StereographicDirection {
                xy: Point2::new(
                    f32::from_bits(0x3ea596dd) as f64,
                    f32::from_bits(0xbd8fc60e) as f64,
                ),
            },
            inverse_distance: f32::from_bits(0x3d804cdd) as f64,
        };
        let target_camera_from_imu = target_ext.inverse();
        let target_inverse_rotation = sophus_so3_inverse(target.rotation);
        let target_imu_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_f32(target_inverse_rotation, host.rotation),
            translation: sophus_rotate_difference_visual_f32(
                target_inverse_rotation,
                host.translation,
                target.translation,
            ),
        };
        let target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu.translation,
            ) + target_camera_from_imu.translation,
        };
        let target_camera_from_anchor_camera = F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                target_camera_from_anchor_imu.rotation,
                host_ext.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_anchor_imu.rotation,
                host_ext.translation,
            ) + target_camera_from_anchor_imu.translation,
        };
        let bearing = landmark.direction.bearing_f32();
        let point4_target = eigen_homogeneous_point_gemv_f32(
            target_camera_from_anchor_camera.rotation,
            target_camera_from_anchor_camera.translation,
            bearing,
            landmark.inverse_distance as f32,
        );
        let point_target = point4_target.fixed_rows::<3>(0).into_owned();
        let (_, projection_jacobian) =
            project_double_sphere_with_jacobian_f32(&m7_fixed_camera(), point_target).unwrap();
        let mut point_wrt_relative_pose = SMatrix::<f32, 3, 6>::zeros();
        point_wrt_relative_pose
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(SMatrix::<f32, 3, 3>::identity() * landmark.inverse_distance as f32));
        point_wrt_relative_pose
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-skew3_f32(point_target)));
        let residual_wrt_relative_pose =
            eigen_relative_pose_jacobian_f32(projection_jacobian, point_wrt_relative_pose);
        let mut weighted_relative = residual_wrt_relative_pose;
        for value in weighted_relative.as_mut_slice() {
            *value *= f32::from_bits(0x40000000);
        }
        // Keep the packet-generation witness separate from the current-value
        // chain above.  Native `computeRelPose` materializes the generic
        // Sophus product at this boundary; it is this pose (not the camera
        // prefix packet spelling used for the value path) that feeds drel.
        let generic_target_imu_from_anchor_imu = sophus_relative_imu_f32(target, host);
        let generic_target_camera_from_anchor_imu = F32Pose {
            rotation: sophus_quat_product_f32(
                target_camera_from_imu.rotation,
                generic_target_imu_from_anchor_imu.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                generic_target_imu_from_anchor_imu.translation,
            ) + target_camera_from_imu.translation,
        };
        let candidate_tmp =
            sophus_compute_relpose_tmp_out_of_line_f32(target_camera_from_imu, target, host);
        let baseline_drel = eigen_adjoint_times_rotation_blocks_f32(
            generic_target_camera_from_anchor_imu,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        );
        let drel = eigen_adjoint_times_rotation_blocks_f32(
            candidate_tmp,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(host.rotation)),
            1.0,
        );
        let drel_target = eigen_adjoint_times_rotation_blocks_f32(
            target_camera_from_imu,
            eigen_quaternion_matrix_f32(sophus_so3_inverse(target.rotation)),
            -1.0,
        );
        let anchor = eigen_weighted_pose_jacobian_f32(
            residual_wrt_relative_pose,
            drel,
            f32::from_bits(0x40000000),
        );
        // The generic visual-chain pose is the faithful diagnostic packet
        // input for this tied native call.  Keep the out-of-line candidate
        // above visible as a negative witness: it differs in 15/36 lanes.
        let expected_drel = [
            0x3dda79ad, 0x3e8b3905, 0xbf74d5e7, 0x00000000, 0x00000000, 0x00000000, 0x3f7e5131,
            0xbd8df5f8, 0x3dba92eb, 0x00000000, 0x00000000, 0x00000000, 0xbd2a12ca, 0xbf75b6c8,
            0xbe8e17f1, 0x00000000, 0x00000000, 0x00000000, 0x3c93c295, 0xbd226d6d, 0xbc17c30c,
            0x3dda79ad, 0x3e8b3905, 0xbf74d5e7, 0xbaf80701, 0xb9828894, 0x3ca77d72, 0x3f7e5131,
            0xbd8df5f8, 0x3dba92eb, 0x3a8bd267, 0xbc37c50e, 0x3d1e3cc6, 0xbd2a12ca, 0xbf75b6c8,
            0xbe8e17f1,
        ];
        assert_f32_bits(baseline_drel.as_slice(), &expected_drel);
        assert_eq!(
            drel.as_slice()
                .iter()
                .zip(expected_drel)
                .filter(|(actual, expected)| actual.to_bits() != *expected)
                .count(),
            15,
            "out-of-line diagnostic drel should remain the 15-lane negative witness"
        );
        print_matrix("weighted_relative", &weighted_relative);
        print_matrix("baseline_drel", &baseline_drel);
        print_matrix("drel", &drel);
        print_matrix("drel_target", &drel_target);
        print_matrix("anchor", &anchor);
        println!(
            "point_target={:08x},{:08x},{:08x} projection_jacobian={}",
            point_target.x.to_bits(),
            point_target.y.to_bits(),
            point_target.z.to_bits(),
            projection_jacobian
                .as_slice()
                .iter()
                .map(|value| format!("{:08x}", value.to_bits()))
                .collect::<Vec<_>>()
                .join(",")
        );
    }

    /// Keep the native `RowMajor` destination boundary separate from the
    /// production FEJ path.  The upstream expression is a fixed `2x6 * 6x6`
    /// product followed by `block +=`; this test-only wrapper makes that
    /// destination update explicit while retaining the scalar Eigen packet
    /// schedule in [`eigen_matrix_product_2x6_f32`].
    fn diagnostic_row_major_pose_block_add_assign(
        destination: &mut [f32; 12],
        left: SMatrix<f32, 2, 6>,
        right: SMatrix<f32, 6, 6>,
    ) {
        let product = eigen_matrix_product_2x6_f32(left, right);
        for row in 0..2 {
            for column in 0..6 {
                destination[row * 6 + column] += product[(row, column)];
            }
        }
    }

    #[test]
    fn m7_actual_eigen_site_track120_obs3_anchor_block_is_bit_exact() {
        // Captured from the pinned native Eigen site for frame timestamp
        // 1403636579963555584, iteration 0, track 120, observation 3
        // (target frame 1/cam 1).  The operands are already whitened and are
        // intentionally supplied in Eigen's column-major memory order.
        let left = SMatrix::<f32, 2, 6>::from_column_slice(&[
            f32::from_bits(0x423ee1b5),
            f32::from_bits(0x4032bc80),
            f32::from_bits(0x40334645),
            f32::from_bits(0x42746653),
            f32::from_bits(0xc2053c7b),
            f32::from_bits(0x40d2d4c3),
            f32::from_bits(0x41c095b6),
            f32::from_bits(0xc4480ee4),
            f32::from_bits(0x4465cd32),
            f32::from_bits(0xc1c001b4),
            f32::from_bits(0x42df9490),
            f32::from_bits(0x440c7238),
        ]);
        let right = SMatrix::<f32, 6, 6>::from_column_slice(&[
            f32::from_bits(0x3dda79ad),
            f32::from_bits(0x3e8b3905),
            f32::from_bits(0xbf74d5e7),
            f32::from_bits(0x00000000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x3f7e5131),
            f32::from_bits(0xbd8df5f8),
            f32::from_bits(0x3dba92eb),
            f32::from_bits(0x00000000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x00000000),
            f32::from_bits(0xbd2a12ca),
            f32::from_bits(0xbf75b6c8),
            f32::from_bits(0xbe8e17f1),
            f32::from_bits(0x00000000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x3c93c295),
            f32::from_bits(0xbd226d6d),
            f32::from_bits(0xbc17c30c),
            f32::from_bits(0x3dda79ad),
            f32::from_bits(0x3e8b3905),
            f32::from_bits(0xbf74d5e7),
            f32::from_bits(0xbaf80701),
            f32::from_bits(0xb9828894),
            f32::from_bits(0x3ca77d72),
            f32::from_bits(0x3f7e5131),
            f32::from_bits(0xbd8df5f8),
            f32::from_bits(0x3dba92eb),
            f32::from_bits(0x3a8bd267),
            f32::from_bits(0xbc37c50e),
            f32::from_bits(0x3d1e3cc6),
            f32::from_bits(0xbd2a12ca),
            f32::from_bits(0xbf75b6c8),
            f32::from_bits(0xbe8e17f1),
        ]);
        let mut destination = [0.0_f32; 12];
        let destination_before = destination;
        diagnostic_row_major_pose_block_add_assign(&mut destination, left, right);

        assert_f32_bits(&destination_before, &[0x00000000; 12]);
        assert_f32_bits(
            &destination,
            &[
                0x4216d5cf, 0x4230b65b, 0x40925ef6, 0x4312a950, 0xc1f31d9c, 0xc464e41e, 0x4129c6d0,
                0xbf5c5301, 0xc2725b87, 0xc41de71f, 0xc43980fe, 0xc2c8260a,
            ],
        );
    }

    #[test]
    fn visual_prefix_filter_defaults_to_all_events_and_selects_exact_context() {
        let all = visual_prefix_trace_filter_from_cached(None, None, None).unwrap();
        assert!(all.matches(None, None, 0));
        assert!(all.matches(Some(12), Some(0), 0));

        let selected =
            visual_prefix_trace_filter_from_cached(Some(Ok(12)), Some(Ok(0)), Some(Ok(0))).unwrap();
        assert!(selected.matches(Some(12), Some(0), 0));
        assert!(!selected.matches(Some(11), Some(0), 0));
        assert!(!selected.matches(Some(12), Some(1), 0));
        assert!(!selected.matches(Some(12), Some(0), 1));
        assert!(!selected.matches(None, Some(0), 0));
    }

    #[test]
    fn visual_prefix_filter_rejects_malformed_values() {
        for (frame_id, iteration, trial) in [
            (Some(Err(())), None, None),
            (Some(Err(())), None, None),
            (None, Some(Err(())), None),
            (None, None, Some(Err(()))),
        ] {
            let result = visual_prefix_trace_filter_from_cached(frame_id, iteration, trial);
            assert!(result.is_err(), "malformed selector must fail closed");
        }
    }

    #[test]
    fn visual_prefix_filter_non_target_skips_open_and_materialization() {
        let selected =
            visual_prefix_trace_filter_from_cached(Some(Ok(12)), Some(Ok(0)), Some(Ok(0))).unwrap();
        let factor = WhitenedFactorRowStack::new(
            DMatrix::from_row_slice(3, 2, &[1.0, 0.25, -0.5, 2.0, 0.75, -1.25]),
            DMatrix::from_row_slice(3, 1, &[0.5, -1.0, 2.0]),
            DVector::from_column_slice(&[0.25, -0.75, 1.5]),
        )
        .expect("visual sidecar fixture")
        .with_kind(FactorKind::Visual)
        .with_landmark_metadata(7, 701)
        .with_visual_observation_ids(vec![(0, 0), (2, 1)]);

        // An empty path would fail if the writer were opened.  The selector
        // must short-circuit before path validation, factor metadata copying,
        // or any sidecar materialization for a non-target event.
        let path = visual_prefix_test_path("non_target");
        let canonical = canonical_visual_prefix_trace_key(&path).expect("canonical test path");
        let lock_path = visual_prefix_trace_lock_path(&canonical).expect("lock test path");
        let result = visual_prefix_trace_writer_selected(
            selected,
            Some(11),
            Some(0),
            0,
            Some(&path),
            std::slice::from_ref(&factor),
            2,
        )
        .expect("non-target events are successful no-ops");
        assert!(result.is_none());
        assert!(!path.exists(), "non-target event must not create target");
        assert!(!lock_path.exists(), "non-target event must not create lock");
    }

    #[test]
    fn visual_prefix_trace_records_exact_prefix_and_prior_boundaries() {
        let factor = WhitenedFactorRowStack::new(
            DMatrix::from_row_slice(3, 2, &[1.0, 0.25, -0.5, 2.0, 0.75, -1.25]),
            DMatrix::from_row_slice(3, 1, &[0.5, -1.0, 2.0]),
            DVector::from_column_slice(&[0.25, -0.75, 1.5]),
        )
        .expect("visual sidecar fixture")
        .with_kind(FactorKind::Visual)
        .with_landmark_metadata(7, 701)
        .with_visual_observation_ids(vec![(0, 0), (2, 1)]);
        let factors = vec![factor];
        let path = std::env::temp_dir().join(format!(
            "visloc_visual_prefix_trace_{}_{}.jsonl",
            std::process::id(),
            VISUAL_PREFIX_TRACE_EVENT.fetch_add(1, Ordering::Relaxed)
        ));
        set_active_diagnostic_lm_frame(Some(42));
        set_active_diagnostic_lm_iteration(Some(3));
        let mut writer = VisualPrefixTraceWriter::open(path.clone(), &factors, 2)
            .expect("sidecar writer")
            .expect("sidecar enabled fixture");
        let (projected, projected_residual, rank) =
            landmark_nullspace_projection_f32(&factors[0], 1e-10);
        let mut visual_h = DMatrix::<f32>::zeros(2, 2);
        let mut visual_b = DVector::<f32>::zeros(2);
        let pending = writer
            .begin_visual_prefix(
                0,
                &factors[0],
                &projected,
                &projected_residual,
                rank,
                &visual_h,
                &visual_b,
            )
            .expect("visual prefix metadata");
        visual_h += projected.transpose() * &projected;
        accumulate_transpose_vector_f32_eigen(
            &mut visual_b,
            &projected,
            &projected_residual,
            false,
        );
        writer
            .finish_visual_prefix(pending, &visual_h, &visual_b)
            .expect("visual prefix record");
        writer
            .write_stage("visual_total", &visual_h, &visual_b)
            .expect("visual total record");
        let prior_h = DMatrix::<f32>::zeros(2, 2);
        let prior_b = DVector::<f32>::zeros(2);
        writer
            .write_prior_before(&visual_h, &visual_b, &prior_h, &prior_b)
            .expect("prior boundary record");
        writer
            .write_stage("prior_after", &visual_h, &visual_b)
            .expect("prior after record");
        writer
            .write_stage("final", &visual_h, &visual_b)
            .expect("final record");
        drop(writer);
        set_active_diagnostic_lm_frame(None);
        set_active_diagnostic_lm_iteration(None);

        let lines = std::fs::read_to_string(&path)
            .expect("sidecar output")
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSONL record"))
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 6);
        assert_eq!(lines[0]["record"], "header");
        assert_eq!(lines[1]["record"], "visual_prefix");
        assert_eq!(lines[1]["frame_id"], 42);
        assert_eq!(lines[1]["iteration"], 3);
        assert_eq!(lines[1]["visual_ordinal"], 0);
        assert_eq!(lines[1]["factor_index"], 0);
        assert_eq!(lines[1]["track_id"], 701);
        assert_eq!(lines[1]["observations"].as_array().unwrap().len(), 2);
        assert_eq!(lines[2]["stage"], "visual_total");
        assert_eq!(lines[3]["stage"], "prior_before");
        assert_eq!(lines[4]["stage"], "prior_after");
        assert_eq!(lines[5]["stage"], "final");
        assert_eq!(lines[1]["global_h_after"], diagnostic_f32_matrix(&visual_h));
        assert_eq!(lines[1]["global_b_after"], diagnostic_f32_vector(&visual_b));
        std::fs::remove_file(path).expect("remove sidecar fixture");
    }

    #[test]
    fn visual_prefix_trace_rejects_missing_observation_identity() {
        let factor = WhitenedFactorRowStack::new(
            DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]),
            DMatrix::from_row_slice(3, 1, &[1.0, 0.5, -0.25]),
            DVector::from_column_slice(&[0.25, -0.75, 1.5]),
        )
        .expect("visual sidecar fixture")
        .with_kind(FactorKind::Visual)
        .with_landmark_metadata(3, 300);
        let path = std::env::temp_dir().join(format!(
            "visloc_visual_prefix_trace_invalid_{}_{}.jsonl",
            std::process::id(),
            VISUAL_PREFIX_TRACE_EVENT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut writer = VisualPrefixTraceWriter::open(path.clone(), &[factor.clone()], 1)
            .expect("sidecar writer")
            .expect("sidecar enabled fixture");
        let (projected, residual, rank) = landmark_nullspace_projection_f32(&factor, 1e-10);
        let h = DMatrix::<f32>::zeros(1, 1);
        let b = DVector::<f32>::zeros(1);
        assert!(matches!(
            writer.begin_visual_prefix(0, &factor, &projected, &residual, rank, &h, &b),
            Err(ImuReductionError::VisualPrefixTraceInvalid { index: 0 })
        ));
        drop(writer);
        std::fs::remove_file(path).expect("remove invalid sidecar fixture");
    }

    #[test]
    fn visual_prefix_trace_rejects_empty_output_path() {
        let factor = WhitenedFactorRowStack::new(
            DMatrix::zeros(0, 0),
            DMatrix::zeros(0, 0),
            DVector::zeros(0),
        )
        .expect("empty fixture");
        assert!(matches!(
            VisualPrefixTraceWriter::open(std::path::PathBuf::new(), &[factor], 0),
            Err(ImuReductionError::VisualPrefixTraceIo)
        ));
    }

    fn visual_prefix_test_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "visloc_visual_prefix_{label}_{}_{}.jsonl",
            std::process::id(),
            VISUAL_PREFIX_TRACE_EVENT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn visual_prefix_disabled_reducer_child() {
        if std::env::var_os("VISLOC_VISUAL_PREFIX_DISABLED_CHILD").is_none() {
            return;
        }
        let factor = WhitenedFactorRowStack::new(
            DMatrix::from_row_slice(3, 1, &[1.0, -2.0, 3.0]),
            DMatrix::from_row_slice(3, 1, &[0.5, 1.0, -0.25]),
            DVector::from_column_slice(&[0.25, -0.75, 1.5]),
        )
        .expect("lean visual fixture")
        .with_kind(FactorKind::Visual)
        .with_landmark_metadata(1, 101)
        .with_visual_observation_ids(vec![(0, 0)]);
        let before = VISUAL_PREFIX_TRACE_OPEN_ATTEMPTS.load(Ordering::Relaxed);
        let reduced =
            reduce_landmark_factors_f32_checked_without_back_substitution(&[factor], 1, 1e-10)
                .expect("lean reducer");
        assert!(!crate::vio::window::diagnostic_env_active());
        assert_eq!(
            VISUAL_PREFIX_TRACE_OPEN_ATTEMPTS.load(Ordering::Relaxed),
            before,
            "disabled reducer must not open a visual-prefix writer"
        );
        assert!(reduced.diagnostic_stages.is_none());
        assert!(reduced.imu_diagnostic.is_none());
    }

    #[test]
    fn visual_prefix_disabled_reducer_has_no_sidecar_work() {
        let executable = std::env::current_exe().expect("test executable path");
        let mut command = std::process::Command::new(executable);
        for (key, _) in std::env::vars_os() {
            if key
                .to_string_lossy()
                .get(0.."VISLOC_BASALT_".len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("VISLOC_BASALT_"))
            {
                command.env_remove(key);
            }
        }
        let output = command
            .env("VISLOC_VISUAL_PREFIX_DISABLED_CHILD", "1")
            .args(["visual_prefix_disabled_reducer_child", "--nocapture"])
            .output()
            .expect("spawn clean reducer child");
        assert!(
            output.status.success(),
            "clean reducer child failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn visual_prefix_writer_cross_process_lock_child() {
        let Some(role) = std::env::var_os("VISLOC_VISUAL_PREFIX_LOCK_CHILD") else {
            return;
        };
        let target = std::path::PathBuf::from(
            std::env::var_os("VISLOC_VISUAL_PREFIX_LOCK_TARGET").expect("lock target"),
        );
        let ready = std::path::PathBuf::from(
            std::env::var_os("VISLOC_VISUAL_PREFIX_LOCK_READY").expect("lock ready"),
        );
        let release = std::path::PathBuf::from(
            std::env::var_os("VISLOC_VISUAL_PREFIX_LOCK_RELEASE").expect("lock release"),
        );
        match role.to_string_lossy().as_ref() {
            "holder" => {
                let writer = VisualPrefixTraceWriter::open(&target, &[], 0)
                    .expect("holder open")
                    .expect("holder owns cross-process lock");
                std::fs::write(&ready, b"holder-ready").expect("holder ready");
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while !release.exists() {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "holder release timeout"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                drop(writer);
            }
            "contender" => {
                let result = VisualPrefixTraceWriter::open(&target, &[], 0);
                assert!(
                    matches!(result, Err(ImuReductionError::VisualPrefixTraceIo)),
                    "existing cross-process lock must reject contender"
                );
                std::fs::write(&ready, b"contender-rejected").expect("contender ready");
            }
            _ => panic!("unknown lock child role"),
        }
    }

    #[test]
    fn visual_prefix_writer_cross_process_lock_and_stale_lock_rejection() {
        fn spawn_child(
            role: &str,
            target: &std::path::Path,
            ready: &std::path::Path,
            release: &std::path::Path,
        ) -> std::process::Child {
            let executable = std::env::current_exe().expect("test executable path");
            let mut command = std::process::Command::new(executable);
            for (key, _) in std::env::vars_os() {
                let key = key.to_string_lossy();
                if key
                    .get(0.."VISLOC_BASALT_".len())
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case("VISLOC_BASALT_"))
                {
                    command.env_remove(key.as_ref());
                }
            }
            command
                .env("VISLOC_VISUAL_PREFIX_LOCK_CHILD", role)
                .env("VISLOC_VISUAL_PREFIX_LOCK_TARGET", target)
                .env("VISLOC_VISUAL_PREFIX_LOCK_READY", ready)
                .env("VISLOC_VISUAL_PREFIX_LOCK_RELEASE", release)
                .args([
                    "visual_prefix_writer_cross_process_lock_child",
                    "--nocapture",
                ])
                .spawn()
                .expect("spawn lock child")
        }

        fn wait_for_file(path: &std::path::Path) -> bool {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !path.exists() && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            path.exists()
        }

        let target = visual_prefix_test_path("cross_process");
        let holder_ready = target.with_extension("holder.ready");
        let contender_ready = target.with_extension("contender.ready");
        let release = target.with_extension("release");
        let canonical = canonical_visual_prefix_trace_key(&target).expect("canonical target");
        let lock_path = visual_prefix_trace_lock_path(&canonical).expect("lock path");
        let mut holder = spawn_child("holder", &target, &holder_ready, &release);
        if !wait_for_file(&holder_ready) {
            let _ = holder.kill();
            let _ = holder.wait();
            panic!("holder did not acquire lock");
        }
        let payload = std::fs::read_to_string(&lock_path).expect("lock payload");
        let payload: serde_json::Value = serde_json::from_str(&payload).expect("lock JSON");
        assert_eq!(payload["schema"], "basalt.m11.visual_prefix_trace_lock.v1");
        assert_eq!(payload["run_id"].as_str().map(str::len), Some(32));
        assert!(payload["pid"].as_u64().is_some());
        assert_eq!(
            payload["target_path_hash"],
            format!("{:016x}", visual_prefix_path_hash(&canonical))
        );

        let mut contender = spawn_child("contender", &target, &contender_ready, &release);
        if !wait_for_file(&contender_ready) {
            let _ = contender.kill();
            let _ = contender.wait();
            let _ = std::fs::write(&release, b"release");
            let _ = holder.wait();
            panic!("contender did not report");
        }
        assert!(contender.wait().expect("wait contender").success());
        std::fs::write(&release, b"release").expect("release holder");
        assert!(holder.wait().expect("wait holder").success());
        assert!(!lock_path.exists(), "live lease must remove its lock");

        let writer = VisualPrefixTraceWriter::open(&target, &[], 0)
            .expect("reacquire after cleanup")
            .expect("reacquire writer");
        drop(writer);
        assert!(!lock_path.exists(), "reacquired lease must clean lock");

        std::fs::write(&lock_path, br#"{"schema":"stale"}"#).expect("stale lock");
        let rejected = VisualPrefixTraceWriter::open(&target, &[], 0);
        assert!(matches!(
            rejected,
            Err(ImuReductionError::VisualPrefixTraceIo)
        ));
        assert!(lock_path.exists(), "stale lock must not be auto-removed");

        for path in [target, holder_ready, contender_ready, release, lock_path] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn visual_prefix_writer_maps_write_failures_to_fail_closed_error() {
        let path = visual_prefix_test_path("write_failure");
        let mut writer = VisualPrefixTraceWriter::open(&path, &[], 0)
            .expect("writer open")
            .expect("writer enabled");
        let read_only = std::fs::OpenOptions::new()
            .read(true)
            .open(&path)
            .expect("read-only replacement handle");
        let writable = std::mem::replace(&mut writer.file, read_only);
        drop(writable);
        assert!(matches!(
            writer.write_record(&json!({"record": "write_failure"})),
            Err(ImuReductionError::VisualPrefixTraceIo)
        ));
        drop(writer);
        std::fs::remove_file(path).expect("remove writer failure fixture");
    }

    #[test]
    fn visual_prefix_writer_rejects_same_canonical_path_concurrently() {
        use std::sync::{mpsc, Arc, Barrier};

        let path = visual_prefix_test_path("same_path");
        let start = Arc::new(Barrier::new(3));
        let release = Arc::new(Barrier::new(3));
        let (sender, receiver) = mpsc::channel();
        let mut workers = Vec::new();
        for _ in 0..2 {
            let path = path.clone();
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let sender = sender.clone();
            workers.push(std::thread::spawn(move || {
                start.wait();
                let result = VisualPrefixTraceWriter::open(path, &[], 0);
                sender
                    .send(result.as_ref().is_ok_and(Option::is_some))
                    .expect("send writer result");
                release.wait();
                drop(result);
            }));
        }
        start.wait();
        let mut opened = [
            receiver.recv().expect("first writer result"),
            receiver.recv().expect("second writer result"),
        ];
        opened.sort_unstable();
        assert_eq!(opened, [false, true]);
        release.wait();
        for worker in workers {
            worker.join().expect("join same-path writer");
        }
        drop(sender);
        std::fs::remove_file(path).expect("remove same-path fixture");
    }

    #[test]
    fn visual_prefix_writer_allows_distinct_canonical_paths_concurrently() {
        use std::sync::{mpsc, Arc, Barrier};

        let paths = [
            visual_prefix_test_path("distinct_a"),
            visual_prefix_test_path("distinct_b"),
        ];
        let start = Arc::new(Barrier::new(3));
        let release = Arc::new(Barrier::new(3));
        let (sender, receiver) = mpsc::channel();
        let mut workers = Vec::new();
        for path in paths.iter().cloned() {
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let sender = sender.clone();
            workers.push(std::thread::spawn(move || {
                start.wait();
                let result = VisualPrefixTraceWriter::open(path, &[], 0);
                sender
                    .send(result.as_ref().is_ok_and(Option::is_some))
                    .expect("send writer result");
                release.wait();
                drop(result);
            }));
        }
        start.wait();
        assert!(receiver.recv().expect("first distinct writer result"));
        assert!(receiver.recv().expect("second distinct writer result"));
        release.wait();
        for worker in workers {
            worker.join().expect("join distinct-path writer");
        }
        drop(sender);
        for path in paths {
            std::fs::remove_file(path).expect("remove distinct-path fixture");
        }
    }

    #[test]
    fn visual_prefix_writer_binds_run_context_and_is_thread_local() {
        set_active_diagnostic_lm_frame(None);
        set_active_diagnostic_lm_iteration(None);
        let guard = begin_diagnostic_lm_run();
        let run_id = active_diagnostic_lm_run_id().expect("run id");
        set_active_diagnostic_lm_frame(Some(17));
        set_active_diagnostic_lm_iteration(Some(2));
        assert_eq!(active_diagnostic_lm_run_id(), Some(run_id));
        assert_eq!(active_diagnostic_lm_frame(), Some(17));
        assert_eq!(active_diagnostic_lm_iteration(), Some(2));
        std::thread::spawn(|| {
            assert_eq!(active_diagnostic_lm_run_id(), None);
            assert_eq!(active_diagnostic_lm_frame(), None);
            assert_eq!(active_diagnostic_lm_iteration(), None);
        })
        .join()
        .expect("join thread-local context check");
        drop(guard);
        assert_eq!(active_diagnostic_lm_run_id(), None);
        assert_eq!(active_diagnostic_lm_frame(), None);
        assert_eq!(active_diagnostic_lm_iteration(), None);
    }

    #[test]
    fn m11_full70_frame4_ordered_f32_legacy_compact_model_recovery_parity() {
        let state_dof = 75;
        let prior_jacobian = DMatrix::from_fn(15, state_dof, |row, column| {
            if column == row {
                1.0 + row as f64 * 0.015625
            } else if column < 15 {
                (row + column + 1) as f64 * 0.00390625
            } else {
                0.0
            }
        });
        let prior_residual = DVector::from_fn(15, |row, _| (row as f64 - 7.0) * 0.03125);
        let prior = WhitenedFactorRowStack::with_objective_cost_kind(
            prior_jacobian,
            DMatrix::zeros(15, 0),
            prior_residual.clone(),
            0.5 * prior_residual.norm_squared(),
            FactorKind::Prior,
        )
        .unwrap()
        .with_prior_state_columns((0..state_dof).collect());
        let mut factors = vec![prior];
        for landmark_index in 0..61 {
            let rows = if landmark_index < 9 { 20 } else { 19 };
            let state_jacobian = DMatrix::from_fn(rows, state_dof, |row, column| {
                if column < 15 {
                    ((row + 1) * (column + 3) % 17) as f64 * 0.015625
                } else {
                    0.0
                }
            });
            let landmark_jacobian = DMatrix::from_fn(rows, 3, |row, column| {
                if row % 3 == column {
                    1.0 + (row + column) as f64 * 0.0078125
                } else {
                    ((row + column + 1) % 7) as f64 * 0.03125
                }
            });
            let residual = DVector::from_fn(rows, |row, _| {
                ((row + landmark_index + 1) % 13) as f64 * 0.03125 - 0.125
            });
            let factor = WhitenedFactorRowStack::with_objective_cost_kind(
                state_jacobian,
                landmark_jacobian,
                residual.clone(),
                0.5 * residual.norm_squared(),
                FactorKind::Visual,
            )
            .unwrap()
            .with_landmark_metadata(landmark_index, 10_000 + landmark_index as u64)
            .with_visual_observation_ids(vec![(0, 0), (1, 1), (2, 0)]);
            factors.push(factor);
        }
        for link in 0..4 {
            let offsets = ImuLinkOffsets {
                start: link * AOM_NAV_DOF,
                end: (link + 1) * AOM_NAV_DOF,
            };
            let mut imu_jacobian = DMatrix::<f64>::zeros(9, state_dof);
            for row in 0..9 {
                for local_column in 0..AOM_NAV_DOF * 2 {
                    let offset = if local_column < AOM_NAV_DOF {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    imu_jacobian[(row, offset + local_column % AOM_NAV_DOF)] =
                        (1 + row + local_column + link) as f64 * 0.0078125;
                }
            }
            let imu_residual = DVector::from_fn(9, |row, _| (row + link + 1) as f64 * 0.03125);
            let imu = WhitenedFactorRowStack::with_objective_cost_kind(
                imu_jacobian,
                DMatrix::zeros(9, 0),
                imu_residual.clone(),
                0.5 * imu_residual.norm_squared(),
                FactorKind::Imu,
            )
            .unwrap()
            .with_imu_link_offsets(offsets.start, offsets.end);

            let mut bias_jacobian = DMatrix::<f64>::zeros(6, state_dof);
            for row in 0..6 {
                for local_column in 0..AOM_NAV_DOF * 2 {
                    let offset = if local_column < AOM_NAV_DOF {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    bias_jacobian[(row, offset + local_column % AOM_NAV_DOF)] =
                        (2 + row + local_column + link) as f64 * 0.00390625;
                }
            }
            let bias_residual =
                DVector::from_fn(6, |row, _| -0.0625 + (row + link) as f64 * 0.015625);
            let bias = WhitenedFactorRowStack::with_objective_cost_kind(
                bias_jacobian,
                DMatrix::zeros(6, 0),
                bias_residual.clone(),
                0.5 * bias_residual.norm_squared(),
                FactorKind::Bias,
            )
            .unwrap()
            .with_imu_link_offsets(offsets.start, offsets.end);
            factors.push(imu);
            factors.push(bias);
        }

        validate_full70_frame4_factors(&factors, state_dof).expect("strict frame-4 factor order");
        let legacy = reduce_landmark_factors_f32_checked(&factors, state_dof, 1e-10)
            .expect("legacy f32 frame-4 reduction");
        let compact = reduce_landmark_factors_f32_checked_with_compact_back_substitution(
            &factors, state_dof, 1e-10,
        )
        .expect("compact f32 frame-4 reduction");
        assert!(full70_matrix_bits_match(&legacy.h, &compact.h));
        assert!(full70_vector_bits_match(&legacy.b, &compact.b));
        assert_eq!(legacy.back_substitution.len(), 70);
        let compact_batch = compact
            .compact_back_substitution
            .as_ref()
            .expect("all 61 visual factors have compact entries");
        assert_eq!(compact_batch.entries.len(), 61);
        assert_eq!(compact_batch.entries[0].landmark_index, 0);
        assert_eq!(compact_batch.entries[60].landmark_index, 60);

        let state_step =
            DVector::from_fn(state_dof, |column, _| (column as f64 - 23.0) * 0.0078125);
        for (entry_index, entry) in compact_batch.entries.iter().enumerate() {
            let factor_index = entry_index + 1;
            let legacy_step = back_substitute_landmark_f32_with_track(
                &legacy.back_substitution[factor_index],
                &state_step,
                1e-10,
                Some(entry.track_id),
            )
            .expect("legacy visual recovery");
            let compact_step = back_substitute_landmark_compact_entry_f32(
                entry,
                &compact_batch.storage,
                &state_step,
                1e-10,
            )
            .expect("compact visual recovery");
            assert_eq!(legacy_step.len(), compact_step.len());
            for (lane, (legacy, compact)) in legacy_step.iter().zip(compact_step.iter()).enumerate()
            {
                assert_eq!(
                    (*legacy as f32).to_bits(),
                    (*compact as f32).to_bits(),
                    "frame-4 recovery factor {entry_index} lane {lane}"
                );
            }
        }

        // The model evaluator is exposed as a complete transformed-row path,
        // while the compact reducer exposes Q1/R recovery.  Build the exact
        // Q1/Q2 payloads from the same f32 factor boundary and verify the
        // already-computed compact payload's model contract for all 70 source
        // factors, preserving source factor order (including IMU/Bias pairs).
        let model_payloads = factors
            .iter()
            .map(|factor| {
                if factor.landmark_jacobian.ncols() == 0 {
                    ModelReusePayloadF32::Plain {
                        kind: factor.kind,
                        state: as_f32_matrix(&factor.state_jacobian),
                        residual: as_f32_vector(&factor.residual),
                    }
                } else {
                    let state = as_f32_matrix(&factor.state_jacobian);
                    let landmark = as_f32_matrix(&factor.landmark_jacobian);
                    let residual = as_f32_vector(&factor.residual);
                    let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual)
                        .expect("model payload QR");
                    let rank = (0..landmark.ncols())
                        .filter(|&index| qr.pivots[index].abs() > 1e-10_f32)
                        .count();
                    let metadata = factor.landmark_metadata.expect("visual metadata");
                    ModelReusePayloadF32::Visual(QrModelReusePayloadF32 {
                        q1: qr
                            .compact_back_substitution(
                                metadata.landmark_index,
                                metadata.track_id,
                                rank,
                                rank == landmark.ncols(),
                            )
                            .expect("model payload compact Q1"),
                        q2_state: qr.q2_state(),
                        q2_residual: qr.q2_residual(),
                    })
                }
            })
            .collect::<Vec<_>>();
        let full_model = model_cost_decrease_f32(&factors, &state_step, 1e-10)
            .expect("complete transformed-row model");
        let payload_model =
            model_cost_decrease_from_payloads_f32(&model_payloads, &factors, &state_step, 1e-10)
                .expect("compact Q1/Q2 model payload");
        assert_eq!(
            (full_model as f32).to_bits(),
            (payload_model as f32).to_bits()
        );

        let linearization = LmLinearization {
            factors: factors.clone(),
            cost: 0.0,
        };
        let state = DVector::zeros(state_dof);
        let trial_state = &state + &state_step;
        let damping = DVector::zeros(state_dof);
        let reduced = legacy.clone().as_f64();
        let event = LmDiagnosticEvent {
            iteration: 0,
            trial: 0,
            phase: "iteration_start",
            lambda: 1e-4,
            lambda_after: 1e-4,
            cost_before: 0.0,
            model_cost: Some(full_model),
            model_decrease: None,
            actual_cost: Some(full_model),
            step_norm: Some(state_step.norm()),
            decision: "pending",
            base_state: &state,
            state: &state,
            trial_state: Some(&trial_state),
            step: Some(&state_step),
            damping_diag: &damping,
            damped_h: None,
            linearization: &linearization,
            reduced: &reduced,
        };
        let payload = full70_oracle_payload(
            4,
            &event,
            &json!({ "state_blocks": [], "prior_source": { "source": "test" } }),
            &json!({ "status": "test_bound" }),
        )
        .expect("full frame-4 oracle payload");
        assert_eq!(
            payload["schema"].as_str(),
            Some("basalt.m11.full70_factor_oracle.v1")
        );
        assert_eq!(payload["factor_count"].as_u64(), Some(70));
        assert_eq!(payload["row_count"].as_u64(), Some(1243));
        assert_eq!(payload["visual_factor_count"].as_u64(), Some(61));
        assert_eq!(payload["reducer"]["h_bitwise_equal"].as_bool(), Some(true));
        assert_eq!(payload["reducer"]["b_bitwise_equal"].as_bool(), Some(true));
        assert_eq!(payload["factors"][0]["kind"].as_str(), Some("Prior"));
        assert_eq!(payload["factors"][1]["kind"].as_str(), Some("Visual"));
        assert_eq!(payload["factors"][62]["kind"].as_str(), Some("Imu"));
        assert_eq!(payload["factors"][63]["kind"].as_str(), Some("Bias"));
        assert_eq!(
            payload["factors"][62]["imu_local_15x30"]["shape"],
            json!([15, 30])
        );
        assert_eq!(
            payload["recovery"]["status"].as_str(),
            Some("captured_from_same_ordered_rows")
        );
        assert!(payload["recovery"]["landmark_steps"]
            .as_array()
            .is_some_and(|steps| steps.iter().all(|step| step["bitwise_equal"] == true)));
    }

    #[test]
    fn m11_full70_oracle_rejects_count_and_nonfinite_inputs() {
        assert!(matches!(
            validate_full70_frame4_factors(&[], 75),
            Err(ImuReductionError::Full70OracleInvalid { .. })
        ));
        assert!(matches!(
            full70_checked_f32_bits(f64::NAN, 3),
            Err(ImuReductionError::Full70OracleInvalid { index: 3 })
        ));
        assert!(!full70_hash_like("not-a-hash"));
        assert!(full70_hash_like(&"a".repeat(64)));
    }

    #[test]
    fn m11_retry11_native_operands_replay_current_chain_stagewise() {
        struct NativeRelPoseFixture {
            name: &'static str,
            state_h: [u32; 7],
            state_t: [u32; 7],
            extrinsic_h: [u32; 7],
            extrinsic_t: [u32; 7],
            after_inverse: [u32; 7],
            relative_q: [u32; 4],
            relative_t: [u32; 3],
            before_adj: [u32; 7],
            after_adj: [u32; 36],
            after_adj2: [u32; 36],
            returned: [u32; 7],
        }

        fn pose(words: [u32; 7]) -> F32Pose {
            F32Pose {
                rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(words[3]),
                    f32::from_bits(words[0]),
                    f32::from_bits(words[1]),
                    f32::from_bits(words[2]),
                )),
                translation: Vector3::new(
                    f32::from_bits(words[4]),
                    f32::from_bits(words[5]),
                    f32::from_bits(words[6]),
                ),
            }
        }

        fn pose_bits(value: F32Pose) -> [u32; 7] {
            let q = value.rotation.quaternion();
            [
                q.i.to_bits(),
                q.j.to_bits(),
                q.k.to_bits(),
                q.w.to_bits(),
                value.translation.x.to_bits(),
                value.translation.y.to_bits(),
                value.translation.z.to_bits(),
            ]
        }

        fn adjoint_bits(value: F32Pose) -> Vec<u32> {
            // This is the same fixed Eigen block construction used by the
            // production d_rel helper, without the right-hand rotation
            // product. The native retry11 dump is the standalone Adj() result.
            let rotation = eigen_quaternion_matrix_f32(value.rotation);
            let cross = eigen_matrix_product_3x3_f32(skew3_f32(value.translation), rotation);
            let mut adjoint = SMatrix::<f32, 6, 6>::zeros();
            adjoint.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
            adjoint.fixed_view_mut::<3, 3>(0, 3).copy_from(&cross);
            adjoint.fixed_view_mut::<3, 3>(3, 3).copy_from(&rotation);
            adjoint
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        }

        fn record_stage(
            fixture: &str,
            stage: &str,
            actual: &[u32],
            expected: &[u32],
            failures: &mut Vec<String>,
        ) {
            assert_eq!(actual.len(), expected.len(), "{fixture} {stage} length");
            let mismatches = actual
                .iter()
                .zip(expected)
                .enumerate()
                .filter(|(_, (actual, expected))| actual != expected)
                .collect::<Vec<_>>();
            println!(
                "M11_V7_STAGE fixture={fixture} stage={stage} exact={}/{}",
                actual.len() - mismatches.len(),
                actual.len()
            );
            if let Some((index, (actual, expected))) = mismatches.first() {
                failures.push(format!(
                    "{fixture} {stage} first lane {index}: got {actual:08x}, expected {expected:08x} ({} mismatches)",
                    mismatches.len()
                ));
            }
        }

        let fixtures = [
            NativeRelPoseFixture {
                name: "ordinal8_frame4_cam0_obs8",
                state_h: [
                    0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd, 0x00000000, 0x00000000,
                    0x00000000,
                ],
                state_t: [
                    0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719, 0x3c02c75a, 0xbb0c3bda,
                    0xbcdc2b7f,
                ],
                extrinsic_h: [
                    0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc,
                    0x3ba86617,
                ],
                extrinsic_t: [
                    0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc,
                    0x3ba86617,
                ],
                after_inverse: [
                    0x3bed3c0f, 0xbbf71cd4, 0xbf33a827, 0x3f365a1e, 0x3d8ddf2e, 0xbc810144,
                    0xbb71aa03,
                ],
                relative_q: [0x3bc5abb3, 0x3c4782fd, 0x3b130617, 0x3f7ff9c9],
                // retry12 moved the action breakpoint after the native
                // SO3 store.  The old rich3 value was captured one
                // instruction before that store and is intentionally not
                // used as the endpoint oracle.
                relative_t: [0x3ce63e7b, 0xb8d7afbc, 0xba563900],
                before_adj: [
                    0x3ca45f54, 0xbb4c3a84, 0xbf33324e, 0x3f36c007, 0x3d8e8d8c, 0xbd339e72,
                    0xbb932378,
                ],
                after_adj: [
                    0x3ca3fe80, 0xbf7fe08e, 0xbcc1ab36, 0x00000000, 0x00000000, 0x00000000,
                    0x3f7fd02a, 0x3c9d8ea0, 0x3d0735b3, 0x00000000, 0x00000000, 0x00000000,
                    0xbd05484e, 0xbcc6f0e0, 0x3f7fc9f5, 0x00000000, 0x00000000, 0x00000000,
                    0xbb623180, 0x3acbe7de, 0xbd8cafc7, 0x3ca3fe80, 0xbf7fe08e, 0xbcc1ab36,
                    0xbab26aa0, 0xbbde5285, 0x3d38f8a5, 0x3f7fd02a, 0x3c9d8ea0, 0x3d0735b3,
                    0xbd33eadf, 0xbd8e22d9, 0xbb4c4ba8, 0xbd05484e, 0xbcc6f0e0, 0x3f7fc9f5,
                ],
                after_adj2: [
                    0x3c73d880, 0xbf7ff8bc, 0x3a188a9d, 0x00000000, 0x00000000, 0x00000000,
                    0x3f7fea6c, 0x3c73fdc0, 0x3cab33d7, 0x00000000, 0x00000000, 0x00000000,
                    0xbcab4127, 0x398de86d, 0x3f7ff1ad, 0x00000000, 0x00000000, 0x00000000,
                    0xbb723ce4, 0xb8c7a1b5, 0xbd8d6046, 0x3c73d880, 0xbf7ff8bc, 0x3a188a9d,
                    0xb98fc173, 0xbba83b39, 0x3c8969dc, 0x3f7fea6c, 0x3c73fdc0, 0x3cab33d7,
                    0xbc80f7f4, 0xbd8daed3, 0xb9a2c4c3, 0xbcab4127, 0x398de86d, 0x3f7ff1ad,
                ],
                returned: [
                    0x3c48260a, 0xbbbfafc5, 0x3b23e6e5, 0x3f7ff9ca, 0x3960ac00, 0xbce9c4da,
                    0xbaa1d03e,
                ],
            },
            NativeRelPoseFixture {
                name: "ordinal9_frame4_cam1_obs9",
                state_h: [
                    0xbd582e43, 0xbf4d686f, 0x00000000, 0x3f182ffd, 0x00000000, 0x00000000,
                    0x00000000,
                ],
                state_t: [
                    0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719, 0x3c02c75a, 0xbb0c3bda,
                    0xbcdc2b7f,
                ],
                extrinsic_h: [
                    0xbbed3c0f, 0x3bf71cd4, 0x3f33a827, 0x3f365a1e, 0xbc896b48, 0xbd8d2fdc,
                    0x3ba86617,
                ],
                extrinsic_t: [
                    0xbb19188b, 0x3c55012e, 0x3f33d4ed, 0x3f362af6, 0xbc76fa76, 0x3d290319,
                    0x3b4f4832,
                ],
                after_inverse: [
                    0x3b19188b, 0xbc55012e, 0xbf33d4ed, 0x3f362af6, 0xbd27e3b1, 0xbc8044de,
                    0xbb7a8e6e,
                ],
                relative_q: [0x3bc5abb3, 0x3c4782fd, 0x3b130617, 0x3f7ff9c9],
                relative_t: [0x3ce63e7b, 0xb8d7afbc, 0xba563900],
                before_adj: [
                    0x3c784607, 0xbc0c871e, 0xbf3360ef, 0x3f369746, 0xbd26c55e, 0xbd334a15,
                    0xbb8a1a0a,
                ],
                after_adj: [
                    0x3c929ea0, 0xbf7ff2db, 0xbc1377bf, 0x00000000, 0x00000000, 0x00000000,
                    0x3f7fd0c9, 0x3c901020, 0x3d09c617, 0x00000000, 0x00000000, 0x00000000,
                    0xbd091909, 0xbc1d399c, 0x3f7fd843, 0x00000000, 0x00000000, 0x00000000,
                    0xbb7a540c, 0xb9e7aee4, 0x3d29f249, 0x3c929ea0, 0xbf7ff2db, 0xbc1377bf,
                    0xbab743d6, 0xbb3a4078, 0x3d303a38, 0x3f7fd0c9, 0x3c901020, 0x3d09c617,
                    0xbd3358a9, 0x3d273f66, 0xba8cd213, 0xbd091909, 0xbc1d399c, 0x3f7fd843,
                ],
                after_adj2: [
                    0x3c50bc00, 0xbf7ff318, 0x3c795f6c, 0x00000000, 0x00000000, 0x00000000,
                    0x3f7feb21, 0x3c561800, 0x3cb0dd46, 0x00000000, 0x00000000, 0x00000000,
                    0xbcb27575, 0x3c74c968, 0x3f7fe922, 0x00000000, 0x00000000, 0x00000000,
                    0xbb851013, 0x3a16c64f, 0x3d28ac66, 0x3c50bc00, 0xbf7ff318, 0x3c795f6c,
                    0xb9970b1d, 0xbb407b2d, 0x3c77ae51, 0x3f7feb21, 0x3c561800, 0x3cb0dd46,
                    0xbc7f833d, 0x3d282c07, 0xba79f3d7, 0xbcb27575, 0x3c74c968, 0x3f7fe922,
                ],
                returned: [
                    0x3ba066a6, 0xbbce2fcd, 0x3ac222e9, 0x3f7ffdda, 0xbde17018, 0xbce785da,
                    0xbaa35d96,
                ],
            },
        ];

        let mut failures = Vec::new();
        let mut candidate_failures = Vec::new();
        for fixture in fixtures {
            let host = pose(fixture.state_h);
            let target = pose(fixture.state_t);
            let anchor_camera = pose(fixture.extrinsic_h);
            let target_camera = pose(fixture.extrinsic_t);

            let target_camera_from_imu = target_camera.inverse();
            let target_inverse_rotation = sophus_so3_inverse(target.rotation);
            let target_imu_from_anchor_imu = F32Pose {
                rotation: sophus_quat_product_current_packet_f32(
                    target_inverse_rotation,
                    host.rotation,
                ),
                translation: sophus_rotate_difference_visual_f32(
                    target_inverse_rotation,
                    host.translation,
                    target.translation,
                ),
            };
            let target_camera_from_anchor_imu = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_anchor_imu.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    target_imu_from_anchor_imu.translation,
                ) + target_camera_from_imu.translation,
            };
            let returned = F32Pose {
                rotation: sophus_quat_product_camera_suffix_f32(
                    target_camera_from_anchor_imu.rotation,
                    anchor_camera.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_anchor_imu.rotation,
                    anchor_camera.translation,
                ) + target_camera_from_anchor_imu.translation,
            };

            record_stage(
                fixture.name,
                "after_inverse",
                &pose_bits(target_camera_from_imu),
                &fixture.after_inverse,
                &mut failures,
            );
            record_stage(
                fixture.name,
                "after_relative_world_norm_q",
                &[
                    target_imu_from_anchor_imu.rotation.i.to_bits(),
                    target_imu_from_anchor_imu.rotation.j.to_bits(),
                    target_imu_from_anchor_imu.rotation.k.to_bits(),
                    target_imu_from_anchor_imu.rotation.w.to_bits(),
                ],
                &fixture.relative_q,
                &mut failures,
            );
            record_stage(
                fixture.name,
                "after_so3_action_xyz",
                &[
                    target_imu_from_anchor_imu.translation.x.to_bits(),
                    target_imu_from_anchor_imu.translation.y.to_bits(),
                    target_imu_from_anchor_imu.translation.z.to_bits(),
                ],
                &fixture.relative_t,
                &mut failures,
            );
            record_stage(
                fixture.name,
                "before_adj_tmp",
                &pose_bits(target_camera_from_anchor_imu),
                &fixture.before_adj,
                &mut failures,
            );
            let adjoint = adjoint_bits(target_camera_from_anchor_imu);
            record_stage(
                fixture.name,
                "after_adj",
                &adjoint,
                &fixture.after_adj,
                &mut failures,
            );
            let adjoint2 = adjoint_bits(target_camera_from_imu);
            record_stage(
                fixture.name,
                "after_adj2",
                &adjoint2,
                &fixture.after_adj2,
                &mut failures,
            );
            record_stage(
                fixture.name,
                "returned",
                &pose_bits(returned),
                &fixture.returned,
                &mut failures,
            );

            // The clean native computeRelPose body uses its out-of-line
            // scalar-lane SO3 action after the packet relative quaternion.
            // Keep this route test-only until it has been checked against
            // both camera suffixes and the existing track fixtures.  The
            // current production value chain above intentionally remains the
            // diagnostic control.
            let candidate_relative_t = sophus_rotate_difference_f32(
                target_inverse_rotation,
                host.translation,
                target.translation,
            );
            let candidate_target_imu_from_anchor_imu = F32Pose {
                rotation: target_imu_from_anchor_imu.rotation,
                translation: candidate_relative_t,
            };
            let candidate_target_camera_from_anchor_imu = F32Pose {
                rotation: sophus_quat_product_camera_prefix_f32(
                    target_camera_from_imu.rotation,
                    candidate_target_imu_from_anchor_imu.rotation,
                ),
                translation: sophus_rotate_f32(
                    target_camera_from_imu.rotation,
                    candidate_target_imu_from_anchor_imu.translation,
                ) + target_camera_from_imu.translation,
            };
            let candidate_returned = F32Pose {
                rotation: sophus_quat_product_camera_suffix_f32(
                    candidate_target_camera_from_anchor_imu.rotation,
                    anchor_camera.rotation,
                ),
                translation: sophus_rotate_f32(
                    candidate_target_camera_from_anchor_imu.rotation,
                    anchor_camera.translation,
                ) + candidate_target_camera_from_anchor_imu.translation,
            };
            record_stage(
                fixture.name,
                "candidate_after_so3_action_post_store",
                &[
                    candidate_target_imu_from_anchor_imu.translation.x.to_bits(),
                    candidate_target_imu_from_anchor_imu.translation.y.to_bits(),
                    candidate_target_imu_from_anchor_imu.translation.z.to_bits(),
                ],
                &fixture.relative_t,
                &mut candidate_failures,
            );
            record_stage(
                fixture.name,
                "candidate_before_adj_tmp",
                &pose_bits(candidate_target_camera_from_anchor_imu),
                &fixture.before_adj,
                &mut candidate_failures,
            );
            record_stage(
                fixture.name,
                "candidate_after_adj",
                &adjoint_bits(candidate_target_camera_from_anchor_imu),
                &fixture.after_adj,
                &mut candidate_failures,
            );
            record_stage(
                fixture.name,
                "candidate_after_adj2",
                &adjoint2,
                &fixture.after_adj2,
                &mut candidate_failures,
            );
            record_stage(
                fixture.name,
                "candidate_returned",
                &pose_bits(candidate_returned),
                &fixture.returned,
                &mut candidate_failures,
            );
        }

        assert_eq!(
            failures.len(),
            8,
            "retry12 current-chain mismatch boundary changed:\n{}",
            failures.join("\n")
        );
        assert!(
            candidate_failures.is_empty(),
            "retry12 out-of-line candidate mismatches:\n{}",
            candidate_failures.join("\n")
        );
    }

    /// Replay the real ordered frame-4 visual row stack captured by the
    /// current full-path detail oracle.  The non-visual rows are intentionally
    /// not reconstructed here (the detail contract exposes their aggregate),
    /// but the visual reducer/recovery boundary is the only part that differs
    /// between the retained and clean LM paths.  This keeps the oracle tied to
    /// a real 61-factor frame rather than a synthetic one-landmark fixture.
    #[test]
    fn m11_frame4_visual_stack_compact_matches_legacy_each_iteration() {
        let Some(path) = std::env::var_os("VISLOC_M11_FRAME4_FACTOR_FIXTURE") else {
            eprintln!(
                "skipping current frame-4 compact/legacy oracle: \
                 VISLOC_M11_FRAME4_FACTOR_FIXTURE is not set"
            );
            return;
        };
        let path = std::path::PathBuf::from(path);
        let fixture_text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let fixture: serde_json::Value = serde_json::from_str(&fixture_text)
            .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
        assert_eq!(
            fixture["schema"].as_str(),
            Some("m11.frame4.visual-factor-stack.v2")
        );
        assert_eq!(fixture["frame_id"].as_u64(), Some(4));
        let state_dof = fixture["state_dof"].as_u64().expect("frame-4 state_dof") as usize;
        assert_eq!(state_dof, 75);

        let matrix = |value: &serde_json::Value, label: &str| -> DMatrix<f64> {
            let rows = value
                .as_array()
                .unwrap_or_else(|| panic!("{label} must be a row array"));
            let row_count = rows.len();
            let column_count = rows
                .first()
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            assert!(
                row_count != 0 && column_count != 0,
                "{label} must be non-empty"
            );
            let mut values = Vec::with_capacity(row_count * column_count);
            for (row, value) in rows.iter().enumerate() {
                let row_values = value
                    .as_array()
                    .unwrap_or_else(|| panic!("{label}[{row}] must be an array"));
                assert_eq!(row_values.len(), column_count, "{label} row {row} width");
                values.extend(row_values.iter().map(|value| {
                    value
                        .as_f64()
                        .unwrap_or_else(|| panic!("{label}[{row}] contains non-number"))
                }));
            }
            DMatrix::from_row_slice(row_count, column_count, &values)
        };
        let vector = |value: &serde_json::Value, label: &str| -> DVector<f64> {
            let values = value
                .as_array()
                .unwrap_or_else(|| panic!("{label} must be an array"))
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .unwrap_or_else(|| panic!("{label} contains non-number"))
                })
                .collect::<Vec<_>>();
            DVector::from_vec(values)
        };

        let iterations = fixture["iterations"]
            .as_array()
            .expect("frame-4 iterations array");
        assert!(!iterations.is_empty());
        for iteration in iterations {
            let iteration_id = iteration["iteration"].as_u64().expect("iteration id") as usize;
            let factor_values = iteration["factors"].as_array().expect("iteration factors");
            assert_eq!(
                factor_values.len(),
                iteration["visual_factor_count"]
                    .as_u64()
                    .expect("visual factor count") as usize
            );
            assert_eq!(factor_values.len(), 61, "frame-4 visual factor count");
            let factors = factor_values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let track_id = value["track_id"].as_u64().expect("track id");
                    let state_jacobian = matrix(
                        &value["state_jacobian"],
                        &format!("iteration {iteration_id} factor {index} state Jacobian"),
                    );
                    let landmark_jacobian = matrix(
                        &value["landmark_jacobian"],
                        &format!("iteration {iteration_id} factor {index} landmark Jacobian"),
                    );
                    let residual = vector(
                        &value["residual"],
                        &format!("iteration {iteration_id} factor {index} residual"),
                    );
                    assert_eq!(state_jacobian.ncols(), state_dof);
                    assert_eq!(landmark_jacobian.ncols(), 3);
                    WhitenedFactorRowStack::with_objective_cost_kind(
                        state_jacobian,
                        landmark_jacobian,
                        residual,
                        0.0,
                        FactorKind::Visual,
                    )
                    .expect("valid frame-4 visual factor")
                    .with_landmark_metadata(index, track_id)
                })
                .collect::<Vec<_>>();
            let state = DVector::zeros(state_dof);
            let state_step = vector(
                &iteration["trial_delta"],
                &format!("iteration {iteration_id} trial delta"),
            );
            assert_eq!(state_step.len(), state_dof);

            let legacy = reduce_landmark_factors_f32_checked(&factors, state_dof, 1e-10)
                .unwrap_or_else(|error| {
                    panic!("legacy reduction at iteration {iteration_id}: {error:?}")
                });
            let compact = reduce_landmark_factors_f32_checked_with_compact_back_substitution(
                &factors, state_dof, 1e-10,
            )
            .unwrap_or_else(|error| {
                panic!("compact reduction at iteration {iteration_id}: {error:?}")
            });
            assert_matrix_f32_bitwise_equal(
                &compact.h,
                &legacy.h,
                &format!("iteration {iteration_id} reduced H"),
            );
            assert_vector_f32_bitwise_equal(
                &compact.b,
                &legacy.b,
                &format!("iteration {iteration_id} reduced b"),
            );

            let compact_batch = compact
                .compact_back_substitution
                .as_ref()
                .expect("compact frame-4 payload");
            assert_eq!(compact_batch.entries.len(), factors.len());
            for (index, (factor, value)) in factors.iter().zip(factor_values).enumerate() {
                let entry = &compact_batch.entries[index];
                assert_eq!(entry.landmark_index, index);
                assert_eq!(entry.track_id, factor.landmark_metadata.unwrap().track_id);
                let compact_step = back_substitute_landmark_compact_entry_f32(
                    entry,
                    &compact_batch.storage,
                    &state_step,
                    1e-10,
                )
                .unwrap_or_else(|| {
                    panic!("compact recovery at iteration {iteration_id} factor {index}")
                });
                let legacy_data = LandmarkBackSubstitution {
                    state_jacobian: factor.state_jacobian.clone(),
                    landmark_jacobian: factor.landmark_jacobian.clone(),
                    residual: factor.residual.clone(),
                    rank: value["rank"].as_u64().unwrap() as usize,
                };
                let legacy_step = back_substitute_landmark_f32_with_track(
                    &legacy_data,
                    &state_step,
                    1e-10,
                    Some(value["track_id"].as_u64().unwrap()),
                )
                .unwrap_or_else(|| {
                    panic!("legacy recovery at iteration {iteration_id} factor {index}")
                });
                let public_legacy_step = back_substitute_landmark_upstream_f32_with_track(
                    factor,
                    &state_step,
                    1e-10,
                    Some(value["track_id"].as_u64().unwrap()),
                )
                .unwrap_or_else(|| {
                    panic!("public legacy recovery at iteration {iteration_id} factor {index}")
                });
                assert_eq!(compact_step.len(), legacy_step.len());
                for (lane, (&compact, &legacy)) in
                    compact_step.iter().zip(legacy_step.iter()).enumerate()
                {
                    assert_eq!(
                        compact.to_bits(),
                        legacy.to_bits(),
                        "iteration {iteration_id} factor {index} landmark step lane {lane}"
                    );
                }
                for (lane, (&public_legacy, &legacy)) in public_legacy_step
                    .iter()
                    .zip(legacy_step.iter())
                    .enumerate()
                {
                    assert_eq!(
                        public_legacy.to_bits(),
                        legacy.to_bits(),
                        "iteration {iteration_id} factor {index} public legacy step lane {lane}"
                    );
                }
            }

            let preparation = compact
                .into_trial_preparation(&state, &state_step, 1e-10)
                .expect("compact trial preparation");
            let (_, state_fingerprint, step_fingerprint, prepared) =
                preparation.take_landmark_steps();
            assert_ne!(state_fingerprint, 0);
            assert_ne!(step_fingerprint, 0);
            assert_eq!(prepared.len(), factors.len());
            for (index, (_, track_id, step)) in prepared.into_iter().enumerate() {
                assert_eq!(track_id, factors[index].landmark_metadata.unwrap().track_id);
                assert!(step.is_some(), "prepared step at factor {index}");
            }
        }
    }
}
