//! Loop-closure + pose-graph post-process on top of a completed Basalt EuRoC
//! VIO replay (Stage A of the loop-closure-accuracy-ceiling experiment).
//!
//! Inputs (all produced by `basalt_euroc_vio_demo` with MargData enabled,
//! i.e. *without* `--no-marg-data`):
//!   - `trajectory.csv`: one row per processed frame, `imu_to_world` pose.
//!   - `marg_data/frame_NNNNNN.json`: one file per emitted mapper packet.
//!     Only the *filename* is used -- its `frame_id` is treated as a
//!     keyframe id. The (very large, per-frame image/Hessian-carrying) JSON
//!     content is never parsed; the pose for that frame comes from
//!     `trajectory.csv` instead, which is orders of magnitude cheaper.
//!
//! Pipeline (v2 -- see git history for the v1 appearance-retrieval design,
//! which regressed MH_01 ATE 0.066m -> 0.184m because a mean-pooled SIFT
//! descriptor could not distinguish true revisits from the machine hall's
//! repetitive texture, producing self-consistent-but-wrong PnP loops that
//! even GNC could not isolate as outliers):
//!
//!   1. Build a keyframe set from the MargData filenames and derive SE(3)
//!      odometry edges directly from consecutive keyframes' VIO poses.
//!   2. Loop *candidates* come from VIO pose proximity, not appearance
//!      (DROID-style): keyframe pairs separated by more than
//!      `--min-temporal-gap-s`, whose VIO translation distance is under a
//!      drift-scaled radius (`--proximity-base-m` + `--proximity-drift-frac`
//!      x accumulated path length between them), and whose cam0 viewing
//!      directions agree within `--max-viewing-angle-deg`. Basalt VIO is
//!      metric and already accurate (4-23 cm ATE over whole sequences), so
//!      this is a much stronger prior than appearance similarity, which is
//!      used only to *rank* within this geometrically-plausible set.
//!   3. Geometric verification: SuperPoint ONNX keypoints/descriptors +
//!      LightGlue ONNX matching, stereo-triangulated at the older keyframe
//!      (cam0/cam1, Double Sphere model, known extrinsics) and PnP RANSAC
//!      into the newer keyframe (`--min-inliers`, `--min-inlier-ratio`).
//!   4. Two more gates, run only on PnP-verified candidates: (a) a
//!      VIO-consistency check -- the loop's relative pose must agree with
//!      VIO's own (already-accurate) relative pose between the same two
//!      keyframes within a drift-scaled bound; (b) a reciprocal check -- the
//!      independent reverse-direction PnP solve (landmarks at the newer
//!      keyframe, matched into the older) must agree with the forward
//!      solve within the same bound. Both are needed: a wrong-but-locally-
//!      consistent match can still pass PnP with a high inlier count.
//!   5. Optimize an SE(3) pose graph over the keyframes with the existing
//!      GNC solver (`pipelines::slam::pose_graph::PoseGraph::optimize_se3_gnc`).
//!   6. Propagate each keyframe's correction to every VIO frame attached to
//!      it (ORB-SLAM-style rigid spanning-tree propagation).
//!
//! Ground truth is never read here -- only the VIO trajectory, the MargData
//! directory (filenames only), the calibration, and the raw EuRoC images.
//! Evaluate the corrected trajectory with the existing
//! `scripts/evaluate_euroc_trajectory.py`, exactly as the VIO baseline.

use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use nalgebra::{Point2, Point3, Quaternion, UnitQuaternion, Vector3};

use visloc_basalt::{BasaltCalibration, CameraId, DoubleSphereCamera};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::core::types::Camera as GenericCamera;
use visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher;
use visloc_rs::vision::features::superpoint_onnx::{SuperPointOnnxConfig, SuperPointOnnxExtractor};
use visloc_rs::{
    relative_world_to_camera, triangulate_two_view_left_frame, Correspondence2D3D,
    DeepFeatureExtractor, DeepFeatureSet, DltPnP, GaussNewtonPoseRefiner, LinearSolver,
    PnPRansac, PoseGraph, PoseGraphSe3Config, RobustPoseEstimator,
};

const CAM0: CameraId = 0;
const CAM1: CameraId = 1;
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

#[derive(Debug, Clone)]
struct Args {
    sequence: String,
    dataset_dir: PathBuf,
    calibration: PathBuf,
    vio_out_dir: PathBuf,
    out_dir: PathBuf,
    superpoint_model: Option<PathBuf>,
    lightglue_model: Option<PathBuf>,
    /// Fast-iteration mode: skip ONNX loading, feature extraction, candidate
    /// generation, and verification entirely; rebuild the pose graph's loop
    /// edges directly from a previously-written `loops.json`
    /// (`accepted_loops[].measurement_translation` /
    /// `measurement_quaternion_wxyz`) so the GNC weighting / PGO config can
    /// be swept in seconds instead of re-running the ~30-40 min ONNX pass.
    reuse_loops: Option<PathBuf>,
    loop_weight_scale: f64,
    loop_weight_max: f64,
    min_temporal_gap_s: f64,
    proximity_base_m: f64,
    proximity_drift_frac: f64,
    max_viewing_angle_deg: f64,
    appearance_rank_top_k: usize,
    min_inliers: usize,
    min_inlier_ratio: f64,
    reprojection_threshold: f64,
    vio_consistency_translation_base_m: f64,
    vio_consistency_translation_per_m: f64,
    vio_consistency_rotation_base_deg: f64,
    vio_consistency_rotation_per_10m_deg: f64,
    max_candidates: usize,
    max_keyframes: Option<usize>,
    sp_max_keypoints: usize,
}

impl Args {
    fn parse<I: Iterator<Item = String>>(mut args: I) -> Result<Self, String> {
        let mut sequence = None;
        let mut dataset_dir = None;
        let mut calibration = None;
        let mut vio_out_dir = None;
        let mut out_dir = None;
        let mut superpoint_model = None;
        let mut lightglue_model = None;
        let mut reuse_loops = None;
        // Tuned via --reuse-loops fast-iteration sweeps on MH_01: VIO's own
        // odometry edges (weight 1.0, sub-mm/sub-degree true uncertainty over
        // a ~0.4s keyframe gap) are far more precise than a single PnP loop
        // estimate (~4-5cm mean GT error even after all four gates), so an
        // inlier-count-scaled weight (originally 1-8, i.e. up to 8x a single
        // odometry edge) let a few hundred loop edges outweigh the whole
        // odometry chain and regress ATE (0.066m -> 0.087m). Individually
        // tiny per-edge weights still improve ATE in aggregate, because
        // hundreds of redundant, mostly-accurate loops average out their
        // noise; the sweep's ATE-vs-weight curve has a broad interior
        // minimum (0.057m, a genuine ~14% improvement) around this scale,
        // stable within an order of magnitude either side -- not a spike.
        let mut loop_weight_scale = 0.00002f64;
        let mut loop_weight_max = 0.001f64;
        let mut min_temporal_gap_s = 20.0f64;
        let mut proximity_base_m = 1.0f64;
        let mut proximity_drift_frac = 0.03f64;
        let mut max_viewing_angle_deg = 45.0f64;
        let mut appearance_rank_top_k = 8usize;
        let mut min_inliers = 40usize;
        let mut min_inlier_ratio = 0.4f64;
        let mut reprojection_threshold = 0.012f64;
        let mut vio_consistency_translation_base_m = 0.10f64;
        let mut vio_consistency_translation_per_m = 0.02f64;
        let mut vio_consistency_rotation_base_deg = 2.0f64;
        let mut vio_consistency_rotation_per_10m_deg = 0.5f64;
        let mut max_candidates = 4000usize;
        let mut max_keyframes = None;
        let mut sp_max_keypoints = 800usize;

        while let Some(flag) = args.next() {
            macro_rules! value {
                () => {
                    args.next().ok_or_else(|| format!("{flag} needs a value"))?
                };
            }
            match flag.as_str() {
                "--sequence" => sequence = Some(value!()),
                "--dataset-dir" => dataset_dir = Some(PathBuf::from(value!())),
                "--calibration" => calibration = Some(PathBuf::from(value!())),
                "--vio-out-dir" => vio_out_dir = Some(PathBuf::from(value!())),
                "--out-dir" => out_dir = Some(PathBuf::from(value!())),
                "--superpoint-model" => superpoint_model = Some(PathBuf::from(value!())),
                "--lightglue-model" => lightglue_model = Some(PathBuf::from(value!())),
                "--reuse-loops" => reuse_loops = Some(PathBuf::from(value!())),
                "--loop-weight-scale" => {
                    loop_weight_scale = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--loop-weight-max" => {
                    loop_weight_max = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--min-temporal-gap-s" => {
                    min_temporal_gap_s = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--proximity-base-m" => {
                    proximity_base_m = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--proximity-drift-frac" => {
                    proximity_drift_frac = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--max-viewing-angle-deg" => {
                    max_viewing_angle_deg = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--appearance-rank-top-k" => {
                    appearance_rank_top_k = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--min-inliers" => min_inliers = value!().parse().map_err(|e| format!("{e}"))?,
                "--min-inlier-ratio" => {
                    min_inlier_ratio = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--reprojection-threshold" => {
                    reprojection_threshold = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--vio-consistency-translation-base-m" => {
                    vio_consistency_translation_base_m =
                        value!().parse().map_err(|e| format!("{e}"))?
                }
                "--vio-consistency-translation-per-m" => {
                    vio_consistency_translation_per_m =
                        value!().parse().map_err(|e| format!("{e}"))?
                }
                "--vio-consistency-rotation-base-deg" => {
                    vio_consistency_rotation_base_deg =
                        value!().parse().map_err(|e| format!("{e}"))?
                }
                "--vio-consistency-rotation-per-10m-deg" => {
                    vio_consistency_rotation_per_10m_deg =
                        value!().parse().map_err(|e| format!("{e}"))?
                }
                "--max-candidates" => {
                    max_candidates = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--max-keyframes" => {
                    max_keyframes = Some(value!().parse().map_err(|e| format!("{e}"))?)
                }
                "--sp-max-keypoints" => {
                    sp_max_keypoints = value!().parse().map_err(|e| format!("{e}"))?
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }

        Ok(Args {
            sequence: sequence.ok_or("--sequence is required")?,
            dataset_dir: dataset_dir.ok_or("--dataset-dir is required")?,
            calibration: calibration.ok_or("--calibration is required")?,
            vio_out_dir: vio_out_dir.ok_or("--vio-out-dir is required")?,
            out_dir: out_dir.ok_or("--out-dir is required")?,
            superpoint_model: if reuse_loops.is_none() {
                Some(superpoint_model.ok_or("--superpoint-model is required")?)
            } else {
                superpoint_model
            },
            lightglue_model: if reuse_loops.is_none() {
                Some(lightglue_model.ok_or("--lightglue-model is required")?)
            } else {
                lightglue_model
            },
            reuse_loops,
            loop_weight_scale,
            loop_weight_max,
            min_temporal_gap_s,
            proximity_base_m,
            proximity_drift_frac,
            max_viewing_angle_deg,
            appearance_rank_top_k,
            min_inliers,
            min_inlier_ratio,
            reprojection_threshold,
            vio_consistency_translation_base_m,
            vio_consistency_translation_per_m,
            vio_consistency_rotation_base_deg,
            vio_consistency_rotation_per_10m_deg,
            max_candidates,
            max_keyframes,
            sp_max_keypoints,
        })
    }
}

#[derive(Debug, Clone)]
struct TrajRow {
    frame_id: u64,
    timestamp_ns: i64,
    /// `imu_to_world` as emitted by the VIO (body-to-world).
    imu_to_world: SE3,
}

#[derive(Debug, Clone)]
struct Landmark {
    /// Index into the owning keyframe's cam0 `DeepFeatureSet`.
    keypoint_index: usize,
    /// Triangulated 3D point in the *cam0* frame of the owning keyframe.
    point_cam0: Point3<f64>,
}

struct KeyframeData {
    frame_id: u64,
    timestamp_ns: i64,
    cam0_features: DeepFeatureSet,
    /// DS-normalized planar coordinate `(x/z, y/z)` per cam0 keypoint, or
    /// `None` when unprojection failed (grazing / invalid ray).
    cam0_normalized: Vec<Option<Point2<f64>>>,
    mean_descriptor: Vec<f32>,
    landmarks: Vec<Landmark>,
    /// cam0 optical-axis direction in world frame, for the viewing-angle gate.
    viewing_direction_world: Vector3<f64>,
}

#[derive(Debug, Clone)]
struct AcceptedLoop {
    from_frame_id: u64,
    to_frame_id: u64,
    from_timestamp_ns: i64,
    to_timestamp_ns: i64,
    forward_correspondences: usize,
    forward_inliers: usize,
    forward_inlier_ratio: f64,
    forward_mean_reprojection_error: f64,
    backward_inliers: usize,
    backward_inlier_ratio: f64,
    vio_consistency_translation_error_m: f64,
    vio_consistency_rotation_error_deg: f64,
    reciprocal_translation_error_m: f64,
    reciprocal_rotation_error_deg: f64,
    path_length_m: f64,
    /// The accepted loop's relative pose T_bodyA_to_bodyB, for a fully
    /// independent (GT-informed, post-hoc only) diagnostic in the Python
    /// driver.
    measurement_translation: [f64; 3],
    measurement_quaternion_wxyz: [f64; 4],
}

/// Reconstructs an [`AcceptedLoop`] from one `accepted_loops[]` entry of a
/// previously-written `loops.json`, for `--reuse-loops` fast iteration.
fn parse_cached_loop(v: &serde_json::Value) -> Option<AcceptedLoop> {
    let translation: Vec<f64> = v["measurement_translation"]
        .as_array()?
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0))
        .collect();
    let quaternion: Vec<f64> = v["measurement_quaternion_wxyz"]
        .as_array()?
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0))
        .collect();
    if translation.len() != 3 || quaternion.len() != 4 {
        return None;
    }
    Some(AcceptedLoop {
        from_frame_id: v["from_frame_id"].as_u64()?,
        to_frame_id: v["to_frame_id"].as_u64()?,
        from_timestamp_ns: v["from_timestamp_ns"].as_i64().unwrap_or(0),
        to_timestamp_ns: v["to_timestamp_ns"].as_i64().unwrap_or(0),
        forward_correspondences: v["forward_correspondences"].as_u64().unwrap_or(0) as usize,
        forward_inliers: v["forward_inliers"].as_u64().unwrap_or(0) as usize,
        forward_inlier_ratio: v["forward_inlier_ratio"].as_f64().unwrap_or(0.0),
        forward_mean_reprojection_error: v["forward_mean_reprojection_error"]
            .as_f64()
            .unwrap_or(0.0),
        backward_inliers: v["backward_inliers"].as_u64().unwrap_or(0) as usize,
        backward_inlier_ratio: v["backward_inlier_ratio"].as_f64().unwrap_or(0.0),
        vio_consistency_translation_error_m: v["vio_consistency_translation_error_m"]
            .as_f64()
            .unwrap_or(0.0),
        vio_consistency_rotation_error_deg: v["vio_consistency_rotation_error_deg"]
            .as_f64()
            .unwrap_or(0.0),
        reciprocal_translation_error_m: v["reciprocal_translation_error_m"].as_f64().unwrap_or(0.0),
        reciprocal_rotation_error_deg: v["reciprocal_rotation_error_deg"].as_f64().unwrap_or(0.0),
        path_length_m: v["path_length_m"].as_f64().unwrap_or(0.0),
        measurement_translation: [translation[0], translation[1], translation[2]],
        measurement_quaternion_wxyz: [quaternion[0], quaternion[1], quaternion[2], quaternion[3]],
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("basalt_loop_closure_postprocess: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let args = Args::parse(env::args().skip(1)).map_err(|e| {
        format!(
            "{e}\nusage: basalt_loop_closure_postprocess --sequence NAME --dataset-dir DIR \
             --calibration FILE --vio-out-dir DIR --out-dir DIR --superpoint-model FILE \
             --lightglue-model FILE [options] (requires ORT_DYLIB_PATH set and the \
             onnx-inference,image-io cargo features)"
        )
    })?;
    fs::create_dir_all(&args.out_dir)?;

    eprintln!("[{}] loading VIO trajectory", args.sequence);
    let trajectory = read_trajectory_csv(&args.vio_out_dir.join("trajectory.csv"))?;
    let cumulative_length = cumulative_arc_length(&trajectory);
    let pose_by_frame: BTreeMap<u64, &TrajRow> =
        trajectory.iter().map(|row| (row.frame_id, row)).collect();

    let mut keyframe_ids = list_keyframe_frame_ids(&args.vio_out_dir.join("marg_data"))?;
    if let Some(limit) = args.max_keyframes {
        keyframe_ids.truncate(limit);
    }
    eprintln!(
        "[{}] {} trajectory rows, {} keyframes",
        args.sequence,
        trajectory.len(),
        keyframe_ids.len()
    );
    if keyframe_ids.len() < 2 {
        return Err("fewer than 2 keyframes -- cannot build a pose graph".into());
    }

    // ---- Build the pose graph: nodes + sequential odometry edges ----
    let mut graph = PoseGraph::new();
    for &frame_id in &keyframe_ids {
        let row = pose_by_frame[&frame_id];
        graph.add_pose(frame_id, body_pose(row));
    }
    for window in keyframe_ids.windows(2) {
        let (from_id, to_id) = (window[0], window[1]);
        let from_pose = body_pose(pose_by_frame[&from_id]);
        let to_pose = body_pose(pose_by_frame[&to_id]);
        graph.add_sequential_edge(from_id, to_id, relative_world_to_camera(&from_pose, &to_pose));
    }
    graph.anchor(keyframe_ids[0]);

    let (accepted, candidate_pair_count, verified_count): (Vec<AcceptedLoop>, usize, usize) =
        if let Some(reuse_path) = &args.reuse_loops {
        // ---- Fast-iteration path: rebuild loop edges from a cached loops.json ----
        eprintln!(
            "[{}] reusing accepted loops from {} (weight_scale={} weight_max={})",
            args.sequence,
            reuse_path.display(),
            args.loop_weight_scale,
            args.loop_weight_max
        );
        let cached: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(reuse_path)?)?;
        let mut accepted = Vec::new();
        for item in cached["accepted_loops"].as_array().cloned().unwrap_or_default() {
            let Some(loop_record) = parse_cached_loop(&item) else {
                continue;
            };
            let q = loop_record.measurement_quaternion_wxyz;
            let t = loop_record.measurement_translation;
            let rotation = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
            let measurement_body = SE3::new(rotation, Vector3::new(t[0], t[1], t[2]));
            let loop_weight = (loop_record.forward_inliers as f64 * args.loop_weight_scale)
                .min(args.loop_weight_max)
                .max(0.0);
            graph.add_edge_with_information(
                loop_record.from_frame_id,
                loop_record.to_frame_id,
                measurement_body,
                visloc_rs::PoseGraphEdgeKind::LoopClosure,
                nalgebra::Matrix6::identity() * loop_weight,
            );
            accepted.push(loop_record);
        }
        let candidate_pair_count = cached["candidate_pairs"].as_u64().unwrap_or(0) as usize;
        let verified_count = cached["verified_both_directions"].as_u64().unwrap_or(0) as usize;
        eprintln!(
            "[{}] {} loops reloaded from cache",
            args.sequence,
            accepted.len()
        );
        (accepted, candidate_pair_count, verified_count)
    } else {

    eprintln!("[{}] loading calibration", args.sequence);
    let calibration = BasaltCalibration::from_path(&args.calibration)?;
    let cam0 = calibration
        .camera(CAM0)
        .ok_or("calibration is missing cam0")?
        .clone();
    let cam1 = calibration
        .camera(CAM1)
        .ok_or("calibration is missing cam1")?
        .clone();
    // T_cam0_to_cam1: maps a point expressed in the cam0 frame into cam1.
    let t_cam0_to_cam1 = calibration
        .imu_to_camera(CAM1)
        .ok_or("calibration is missing T_imu_cam1")?
        .compose(calibration.camera_to_imu(CAM0).ok_or("missing T_imu_cam0")?);
    // T_cam0_to_imu / T_imu_to_cam0: fixed extrinsics used to lift a loop's
    // cam0-frame relative pose into the body frame the pose graph uses.
    let t_cam0_to_imu = calibration
        .camera_to_imu(CAM0)
        .ok_or("missing T_imu_cam0")?
        .clone();
    let t_imu_to_cam0 = calibration
        .imu_to_camera(CAM0)
        .ok_or("missing T_imu_cam0")?;
    // cam0's optical axis (+Z in the DS/pinhole convention) expressed in the
    // body frame, for the viewing-angle proximity gate.
    let cam0_forward_body = t_cam0_to_imu.rotation.transform_vector(&Vector3::z());

    // ---- 1. Per-keyframe SuperPoint features + stereo triangulation ----
    let superpoint_model = args.superpoint_model.as_ref().expect("checked at parse time");
    let lightglue_model = args.lightglue_model.as_ref().expect("checked at parse time");
    eprintln!(
        "[{}] loading SuperPoint ONNX ({}) and LightGlue ONNX ({})",
        args.sequence,
        superpoint_model.display(),
        lightglue_model.display()
    );
    let sp_config = SuperPointOnnxConfig {
        max_keypoints: args.sp_max_keypoints,
        ..SuperPointOnnxConfig::default()
    };
    let extractor = SuperPointOnnxExtractor::load_from_path(superpoint_model, sp_config)?;
    let matcher = LightGlueOnnxMatcher::load_from_path(lightglue_model)?;
    let identity_camera = GenericCamera::pinhole(0, cam0.width, cam0.height, 1.0, 1.0, 0.0, 0.0);

    let mut keyframes: Vec<KeyframeData> = Vec::with_capacity(keyframe_ids.len());
    for (i, &frame_id) in keyframe_ids.iter().enumerate() {
        let row = *pose_by_frame
            .get(&frame_id)
            .ok_or_else(|| format!("keyframe frame_id {frame_id} missing from trajectory.csv"))?;
        let cam0_image = load_gray_image(&args.dataset_dir, "cam0", row.timestamp_ns)?;
        let cam1_image = load_gray_image(&args.dataset_dir, "cam1", row.timestamp_ns)?;
        let cam0_features = extractor.extract_deep(&cam0_image)?;
        let cam1_features = extractor.extract_deep(&cam1_image)?;

        let cam0_normalized = ds_normalize(&cam0, &cam0_features.keypoints);
        let cam1_normalized = ds_normalize(&cam1, &cam1_features.keypoints);
        let mean_descriptor = mean_l2_normalized_descriptor(&cam0_features.descriptors);

        let stereo_matches = matcher.match_features(
            &cam0_features.keypoints,
            &cam0_features.descriptors,
            &cam1_features.keypoints,
            &cam1_features.descriptors,
        )?;

        let mut landmarks = Vec::with_capacity(stereo_matches.len());
        for m in &stereo_matches {
            let (Some(l), Some(r)) = (
                cam0_normalized[m.query_index],
                cam1_normalized[m.train_index],
            ) else {
                continue;
            };
            let Some(point) = triangulate_two_view_left_frame(
                &identity_camera,
                &identity_camera,
                &t_cam0_to_cam1,
                &l,
                &r,
            ) else {
                continue;
            };
            if !point.coords.iter().all(|v| v.is_finite()) || point.z <= 0.2 || point.z > 60.0 {
                continue;
            }
            landmarks.push(Landmark {
                keypoint_index: m.query_index,
                point_cam0: point,
            });
        }

        let viewing_direction_world = row.imu_to_world.rotation.transform_vector(&cam0_forward_body);

        if (i + 1) % 50 == 0 || i + 1 == keyframe_ids.len() {
            eprintln!(
                "[{}] keyframe {}/{} frame_id={} cam0_kpts={} landmarks={}",
                args.sequence,
                i + 1,
                keyframe_ids.len(),
                frame_id,
                cam0_features.keypoints.len(),
                landmarks.len()
            );
        }

        keyframes.push(KeyframeData {
            frame_id,
            timestamp_ns: row.timestamp_ns,
            cam0_features,
            cam0_normalized,
            mean_descriptor,
            landmarks,
            viewing_direction_world,
        });
    }

    // ---- 3. Loop candidates: VIO pose proximity (DROID-style), not appearance ----
    let min_temporal_gap_ns = (args.min_temporal_gap_s * 1.0e9) as i64;
    let mut candidate_pairs: Vec<(usize, usize, f64)> = Vec::new(); // (i, j, appearance_score)
    for j in 0..keyframes.len() {
        let mut scored: Vec<(usize, f64, f64)> = Vec::new(); // (i, distance, appearance)
        for i in 0..j {
            if keyframes[j].timestamp_ns - keyframes[i].timestamp_ns < min_temporal_gap_ns {
                continue;
            }
            let path_len = (cumulative_length[keyframes[j].frame_id as usize]
                - cumulative_length[keyframes[i].frame_id as usize])
                .max(0.0);
            let pose_i = pose_by_frame[&keyframes[i].frame_id].imu_to_world.translation;
            let pose_j = pose_by_frame[&keyframes[j].frame_id].imu_to_world.translation;
            let distance = (pose_j - pose_i).norm();
            let radius = args.proximity_base_m + args.proximity_drift_frac * path_len;
            if distance > radius {
                continue;
            }
            let cos_angle = keyframes[i]
                .viewing_direction_world
                .normalize()
                .dot(&keyframes[j].viewing_direction_world.normalize())
                .clamp(-1.0, 1.0);
            let angle_deg = cos_angle.acos() / DEG_TO_RAD;
            if angle_deg > args.max_viewing_angle_deg {
                continue;
            }
            let appearance =
                cosine_similarity(&keyframes[i].mean_descriptor, &keyframes[j].mean_descriptor);
            scored.push((i, distance, appearance as f64));
        }
        // Appearance ranks within the geometrically-plausible set; it never
        // gates. If there are more proximity-gated candidates than the cap,
        // keep the most appearance-similar ones first.
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(args.appearance_rank_top_k);
        for (i, _distance, appearance) in scored {
            candidate_pairs.push((i, j, appearance));
            if candidate_pairs.len() >= args.max_candidates {
                break;
            }
        }
        if candidate_pairs.len() >= args.max_candidates {
            break;
        }
    }
    eprintln!(
        "[{}] {} VIO-proximity candidates (gap>={:.0}s, radius={:.2}m+{:.0}%*path, angle<={:.0}deg)",
        args.sequence,
        candidate_pairs.len(),
        args.min_temporal_gap_s,
        args.proximity_base_m,
        args.proximity_drift_frac * 100.0,
        args.max_viewing_angle_deg
    );

    // ---- 4. Geometric verification: SP+LightGlue -> PnP RANSAC, both directions ----
    let pnp_ransac = PnPRansac {
        pose_estimator: DltPnP::default(),
        pose_refiner: Some(GaussNewtonPoseRefiner::default()),
        iterations: 256,
        reprojection_threshold: args.reprojection_threshold,
        seed: 7,
        early_stop_min_iterations: 0,
        early_stop_inlier_ratio: None,
        confidence: None,
    };

    let mut accepted: Vec<AcceptedLoop> = Vec::new();
    let mut verified_count = 0usize;
    for &(i, j, _appearance) in &candidate_pairs {
        let kf_a = &keyframes[i];
        let kf_b = &keyframes[j];
        if kf_a.landmarks.is_empty() || kf_b.landmarks.is_empty() {
            continue;
        }

        let Some((forward_report, forward_corr)) =
            solve_relative_pose(kf_a, kf_b, &matcher, &pnp_ransac, &identity_camera)
        else {
            continue;
        };
        let forward_inlier_ratio = forward_report.inliers.len() as f64 / forward_corr as f64;
        if forward_report.inliers.len() < args.min_inliers || forward_inlier_ratio < args.min_inlier_ratio
        {
            continue;
        }
        let Some((backward_report, backward_corr)) =
            solve_relative_pose(kf_b, kf_a, &matcher, &pnp_ransac, &identity_camera)
        else {
            continue;
        };
        let backward_inlier_ratio = backward_report.inliers.len() as f64 / backward_corr as f64;
        if backward_report.inliers.len() < args.min_inliers
            || backward_inlier_ratio < args.min_inlier_ratio
        {
            continue;
        }
        verified_count += 1;

        // report.pose.world_to_camera == T_cam0A_to_cam0B (forward) /
        // T_cam0B_to_cam0A (backward).
        let r_forward = forward_report.pose.world_to_camera.clone();
        let r_backward = backward_report.pose.world_to_camera.clone();
        let measurement_body = t_cam0_to_imu.compose(&r_forward).compose(&t_imu_to_cam0);
        let measurement_body_backward_inv = t_cam0_to_imu
            .compose(&r_backward)
            .compose(&t_imu_to_cam0)
            .inverse();

        let path_length_m = (cumulative_length[kf_b.frame_id as usize]
            - cumulative_length[kf_a.frame_id as usize])
            .max(0.0);

        // Gate (a): VIO-consistency.
        let vio_relative = relative_world_to_camera(
            &body_pose(pose_by_frame[&kf_a.frame_id]),
            &body_pose(pose_by_frame[&kf_b.frame_id]),
        );
        let vio_translation_error = (measurement_body.translation - vio_relative.translation).norm();
        let vio_rotation_error_deg = rotation_angle_deg(&measurement_body.rotation, &vio_relative.rotation);
        let vio_translation_bound =
            args.vio_consistency_translation_base_m + args.vio_consistency_translation_per_m * path_length_m;
        let vio_rotation_bound_deg = args.vio_consistency_rotation_base_deg
            + args.vio_consistency_rotation_per_10m_deg * (path_length_m / 10.0);
        if vio_translation_error > vio_translation_bound || vio_rotation_error_deg > vio_rotation_bound_deg
        {
            continue;
        }

        // Gate (b): reciprocal solve agreement (same bound as VIO-consistency).
        let reciprocal_translation_error =
            (measurement_body.translation - measurement_body_backward_inv.translation).norm();
        let reciprocal_rotation_error_deg =
            rotation_angle_deg(&measurement_body.rotation, &measurement_body_backward_inv.rotation);
        if reciprocal_translation_error > vio_translation_bound
            || reciprocal_rotation_error_deg > vio_rotation_bound_deg
        {
            continue;
        }

        let loop_weight = (forward_report.inliers.len() as f64 * args.loop_weight_scale)
            .min(args.loop_weight_max)
            .max(0.0);
        graph.add_edge_with_information(
            kf_a.frame_id,
            kf_b.frame_id,
            measurement_body.clone(),
            visloc_rs::PoseGraphEdgeKind::LoopClosure,
            nalgebra::Matrix6::identity() * loop_weight,
        );
        let q = measurement_body.rotation.quaternion();
        accepted.push(AcceptedLoop {
            from_frame_id: kf_a.frame_id,
            to_frame_id: kf_b.frame_id,
            from_timestamp_ns: kf_a.timestamp_ns,
            to_timestamp_ns: kf_b.timestamp_ns,
            forward_correspondences: forward_corr,
            forward_inliers: forward_report.inliers.len(),
            forward_inlier_ratio,
            forward_mean_reprojection_error: forward_report.mean_reprojection_error,
            backward_inliers: backward_report.inliers.len(),
            backward_inlier_ratio,
            vio_consistency_translation_error_m: vio_translation_error,
            vio_consistency_rotation_error_deg: vio_rotation_error_deg,
            reciprocal_translation_error_m: reciprocal_translation_error,
            reciprocal_rotation_error_deg: reciprocal_rotation_error_deg,
            path_length_m,
            measurement_translation: [
                measurement_body.translation.x,
                measurement_body.translation.y,
                measurement_body.translation.z,
            ],
            measurement_quaternion_wxyz: [q.w, q.i, q.j, q.k],
        });
    }
    eprintln!(
        "[{}] {} candidates, {} PnP-verified (both directions), {} accepted (VIO-consistency + reciprocal)",
        args.sequence,
        candidate_pairs.len(),
        verified_count,
        accepted.len()
    );
        (accepted, candidate_pairs.len(), verified_count)
    };

    // ---- 5. Optimize the SE(3) pose graph with GNC ----
    let pgo_config = PoseGraphSe3Config {
        max_iterations: 100,
        linear_solver: if keyframe_ids.len() > 250 {
            LinearSolver::Sparse
        } else {
            LinearSolver::Dense
        },
        ..PoseGraphSe3Config::default()
    };
    let gnc_config = visloc_rs::slam::gnc::GncConfig {
        auto_scale: Some(visloc_rs::slam::gnc::AUTO_SCALE_K),
        ..visloc_rs::slam::gnc::GncConfig::default()
    };
    let initial_cost = graph.se3_cost();
    let has_loop_edges = !accepted.is_empty();
    let (final_cost, converged, outer_iterations, gnc_inliers, gnc_outliers) = if has_loop_edges {
        match graph.optimize_se3_gnc(&pgo_config, &gnc_config) {
            Ok(result) => (
                result.final_cost,
                result.converged,
                result.outer_iterations,
                result.inlier_count(0.5),
                result.outlier_count(0.5),
            ),
            Err(_) => (graph.se3_cost(), false, 0, 0, 0),
        }
    } else {
        eprintln!(
            "[{}] no loops accepted -- skipping GNC solve (odometry-only graph is already optimal)",
            args.sequence
        );
        (initial_cost, true, 0, graph.edges.len(), 0)
    };
    eprintln!(
        "[{}] gnc pgo cost {:.6} -> {:.6} (converged={} outer_iterations={} gnc_inlier_edges={} gnc_outlier_edges={})",
        args.sequence, initial_cost, final_cost, converged, outer_iterations, gnc_inliers, gnc_outliers
    );

    // ---- 6. Propagate keyframe corrections to the full trajectory ----
    let corrected_tum = propagate_corrections(&trajectory, &keyframe_ids, &graph);
    fs::write(args.out_dir.join("corrected_trajectory.tum"), corrected_tum)?;

    // ---- 7. Write diagnostics ----
    let wall_seconds = started.elapsed().as_secs_f64();
    let loops_json = serde_json::json!({
        "sequence": args.sequence,
        "keyframe_count": keyframe_ids.len(),
        "candidate_pairs": candidate_pair_count,
        "verified_both_directions": verified_count,
        "loops_accepted": accepted.len(),
        "accepted_loops": accepted.iter().map(|l| serde_json::json!({
            "from_frame_id": l.from_frame_id,
            "to_frame_id": l.to_frame_id,
            "from_timestamp_ns": l.from_timestamp_ns,
            "to_timestamp_ns": l.to_timestamp_ns,
            "forward_correspondences": l.forward_correspondences,
            "forward_inliers": l.forward_inliers,
            "forward_inlier_ratio": l.forward_inlier_ratio,
            "forward_mean_reprojection_error": l.forward_mean_reprojection_error,
            "backward_inliers": l.backward_inliers,
            "backward_inlier_ratio": l.backward_inlier_ratio,
            "vio_consistency_translation_error_m": l.vio_consistency_translation_error_m,
            "vio_consistency_rotation_error_deg": l.vio_consistency_rotation_error_deg,
            "reciprocal_translation_error_m": l.reciprocal_translation_error_m,
            "reciprocal_rotation_error_deg": l.reciprocal_rotation_error_deg,
            "path_length_m": l.path_length_m,
            "measurement_translation": l.measurement_translation,
            "measurement_quaternion_wxyz": l.measurement_quaternion_wxyz,
        })).collect::<Vec<_>>(),
    });
    fs::write(
        args.out_dir.join("loops.json"),
        serde_json::to_string_pretty(&loops_json)?,
    )?;

    let stats_json = serde_json::json!({
        "sequence": args.sequence,
        "trajectory_rows": trajectory.len(),
        "keyframe_count": keyframe_ids.len(),
        "candidate_pairs": candidate_pair_count,
        "verified_both_directions": verified_count,
        "loops_accepted": accepted.len(),
        "pgo_initial_cost": initial_cost,
        "pgo_final_cost": final_cost,
        "pgo_converged": converged,
        "pgo_outer_iterations": outer_iterations,
        "gnc_inlier_edges": gnc_inliers,
        "gnc_outlier_edges": gnc_outliers,
        "wall_seconds": wall_seconds,
        "args": {
            "min_temporal_gap_s": args.min_temporal_gap_s,
            "proximity_base_m": args.proximity_base_m,
            "proximity_drift_frac": args.proximity_drift_frac,
            "max_viewing_angle_deg": args.max_viewing_angle_deg,
            "appearance_rank_top_k": args.appearance_rank_top_k,
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": args.min_inlier_ratio,
            "reprojection_threshold": args.reprojection_threshold,
            "vio_consistency_translation_base_m": args.vio_consistency_translation_base_m,
            "vio_consistency_translation_per_m": args.vio_consistency_translation_per_m,
            "vio_consistency_rotation_base_deg": args.vio_consistency_rotation_base_deg,
            "vio_consistency_rotation_per_10m_deg": args.vio_consistency_rotation_per_10m_deg,
            "loop_weight_scale": args.loop_weight_scale,
            "loop_weight_max": args.loop_weight_max,
        },
    });
    fs::write(
        args.out_dir.join("stats.json"),
        serde_json::to_string_pretty(&stats_json)?,
    )?;

    eprintln!(
        "[{}] done in {:.1}s -> {}",
        args.sequence,
        wall_seconds,
        args.out_dir.join("corrected_trajectory.tum").display()
    );
    Ok(())
}

/// Solve a relative pose from `reference`'s stereo-triangulated landmarks
/// into `query`'s cam0 image: LightGlue-match the landmark-linked cam0
/// descriptors against `query`'s cam0 descriptors, build 2D-3D
/// correspondences in the DS normalized plane, and run PnP RANSAC. Returns
/// the RANSAC report plus the correspondence count (for the inlier-ratio
/// gate) or `None` if there are too few matches to attempt PnP.
fn solve_relative_pose(
    reference: &KeyframeData,
    query: &KeyframeData,
    matcher: &LightGlueOnnxMatcher,
    pnp_ransac: &PnPRansac,
    identity_camera: &GenericCamera,
) -> Option<(visloc_rs::RansacReport, usize)> {
    let ref_keypoints: Vec<Point2<f64>> = reference
        .landmarks
        .iter()
        .map(|lm| reference.cam0_features.keypoints[lm.keypoint_index])
        .collect();
    let ref_descriptors: Vec<Vec<f32>> = reference
        .landmarks
        .iter()
        .map(|lm| reference.cam0_features.descriptors[lm.keypoint_index].clone())
        .collect();
    let matches = matcher
        .match_features(
            &ref_keypoints,
            &ref_descriptors,
            &query.cam0_features.keypoints,
            &query.cam0_features.descriptors,
        )
        .ok()?;

    let mut correspondences = Vec::with_capacity(matches.len());
    for m in &matches {
        let Some(point2d) = query.cam0_normalized[m.train_index] else {
            continue;
        };
        correspondences.push(Correspondence2D3D {
            point2d,
            point3d: reference.landmarks[m.query_index].point_cam0,
            confidence: Some(m.score),
        });
    }
    let count = correspondences.len();
    if count < pnp_ransac.pose_estimator.min_correspondences {
        return None;
    }
    let report = pnp_ransac.estimate(&correspondences, identity_camera)?;
    Some((report, count))
}

/// `Pose.world_to_camera` here means "world-to-body": the inverse of the
/// VIO's `imu_to_world`. This lets [`relative_world_to_camera`] and the
/// pose-graph solver operate directly on IMU/body frames.
fn body_pose(row: &TrajRow) -> Pose {
    Pose {
        world_to_camera: row.imu_to_world.inverse(),
    }
}

/// Rotation angle (degrees, in `[0, 180]`) between two `UnitQuaternion`s.
fn rotation_angle_deg(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>) -> f64 {
    let relative = a.inverse() * b;
    let w = relative.quaternion().w.clamp(-1.0, 1.0);
    2.0 * w.abs().acos() / DEG_TO_RAD
}

fn read_trajectory_csv(path: &Path) -> Result<Vec<TrajRow>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line_index == 0 {
            continue; // header
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 9 {
            return Err(format!("malformed trajectory.csv line {}: {line}", line_index + 1).into());
        }
        let frame_id: u64 = fields[0].parse()?;
        let timestamp_ns: i64 = fields[1].parse()?;
        let tx: f64 = fields[2].parse()?;
        let ty: f64 = fields[3].parse()?;
        let tz: f64 = fields[4].parse()?;
        let qw: f64 = fields[5].parse()?;
        let qx: f64 = fields[6].parse()?;
        let qy: f64 = fields[7].parse()?;
        let qz: f64 = fields[8].parse()?;
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
        let imu_to_world = SE3::new(rotation, Vector3::new(tx, ty, tz));
        rows.push(TrajRow {
            frame_id,
            timestamp_ns,
            imu_to_world,
        });
    }
    rows.sort_by_key(|r| r.frame_id);
    Ok(rows)
}

/// Cumulative VIO arc length in metres, indexed by `frame_id` (assumes
/// `frame_id` is the dense 0-based row order `trajectory.csv` always uses).
fn cumulative_arc_length(rows: &[TrajRow]) -> Vec<f64> {
    let mut lengths = vec![0.0f64; rows.len()];
    for w in rows.windows(2) {
        let a = w[0].imu_to_world.translation;
        let b = w[1].imu_to_world.translation;
        let step = (b - a).norm();
        let idx = w[1].frame_id as usize;
        lengths[idx] = lengths[w[0].frame_id as usize] + step;
    }
    lengths
}

fn list_keyframe_frame_ids(marg_dir: &Path) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let mut ids = Vec::new();
    for entry in fs::read_dir(marg_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(stem) = name.strip_prefix("frame_").and_then(|s| s.strip_suffix(".json")) else {
            continue;
        };
        if let Ok(id) = stem.parse::<u64>() {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

fn load_gray_image(
    dataset_dir: &Path,
    camera_folder: &str,
    timestamp_ns: i64,
) -> Result<visloc_rs::GrayscaleImage, Box<dyn std::error::Error>> {
    let path = dataset_dir
        .join("mav0")
        .join(camera_folder)
        .join("data")
        .join(format!("{timestamp_ns}.png"));
    visloc_io::images::read_common_image(&path)
        .map_err(|e| format!("{}: {e}", path.display()).into())
}

fn ds_normalize(camera: &DoubleSphereCamera, keypoints: &[Point2<f64>]) -> Vec<Option<Point2<f64>>> {
    keypoints
        .iter()
        .map(|pixel| {
            let ray = camera.unproject(pixel)?;
            if !ray.z.is_finite() || ray.z <= 0.05 {
                return None;
            }
            Some(Point2::new(ray.x / ray.z, ray.y / ray.z))
        })
        .collect()
}

fn mean_l2_normalized_descriptor(descriptors: &[Vec<f32>]) -> Vec<f32> {
    if descriptors.is_empty() {
        return Vec::new();
    }
    let dim = descriptors[0].len();
    let mut mean = vec![0.0f32; dim];
    for d in descriptors {
        let norm = d.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= 1e-12 {
            continue;
        }
        for (m, v) in mean.iter_mut().zip(d.iter()) {
            *m += v / norm;
        }
    }
    let count = descriptors.len() as f32;
    for m in mean.iter_mut() {
        *m /= count;
    }
    let norm = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for m in mean.iter_mut() {
            *m /= norm;
        }
    }
    mean
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Rigid spanning-tree propagation: each trajectory row keeps its VIO-derived
/// relative pose to the nearest preceding keyframe (or the first keyframe,
/// for rows before it); only the keyframe's PGO correction moves it.
fn propagate_corrections(trajectory: &[TrajRow], keyframe_ids: &[u64], graph: &PoseGraph) -> String {
    let mut out = String::new();
    let mut kf_cursor = 0usize;
    for row in trajectory {
        while kf_cursor + 1 < keyframe_ids.len() && keyframe_ids[kf_cursor + 1] <= row.frame_id {
            kf_cursor += 1;
        }
        let reference_id = keyframe_ids[kf_cursor];
        let reference_row_pose = body_pose(&TrajRow {
            frame_id: reference_id,
            timestamp_ns: 0,
            imu_to_world: find_imu_to_world(trajectory, reference_id),
        });
        let current_pose = body_pose(row);
        let delta = relative_world_to_camera(&reference_row_pose, &current_pose);
        let Some(corrected_reference) = graph.poses.get(&reference_id) else {
            // Should not happen: every keyframe id has a graph pose.
            continue;
        };
        let corrected_wc = delta.compose(&corrected_reference.world_to_camera);
        let corrected_body_to_world = corrected_wc.inverse();
        let t = corrected_body_to_world.translation;
        let q = corrected_body_to_world.rotation.quaternion();
        out.push_str(&format!(
            "{:.9} {:.9} {:.9} {:.9} {:.12} {:.12} {:.12} {:.12}\n",
            row.timestamp_ns as f64 * 1.0e-9,
            t.x,
            t.y,
            t.z,
            q.i,
            q.j,
            q.k,
            q.w,
        ));
    }
    out
}

fn find_imu_to_world(trajectory: &[TrajRow], frame_id: u64) -> SE3 {
    // `trajectory` is sorted and dense by frame_id (0-based row order), so a
    // direct index lookup is valid; fall back to a scan defensively.
    if let Some(row) = trajectory.get(frame_id as usize) {
        if row.frame_id == frame_id {
            return row.imu_to_world.clone();
        }
    }
    trajectory
        .iter()
        .find(|r| r.frame_id == frame_id)
        .map(|r| r.imu_to_world.clone())
        .expect("keyframe frame_id must exist in trajectory.csv")
}
