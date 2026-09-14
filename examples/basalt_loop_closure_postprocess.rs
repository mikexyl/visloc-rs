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
//! Pipeline:
//!   1. Build a keyframe set from the MargData filenames and derive SE(3)
//!      odometry edges directly from consecutive keyframes' VIO poses.
//!   2. Place recognition: a cheap mean-pooled, L2-normalized SIFT
//!      "appearance descriptor" per keyframe (no ONNX / network dependency),
//!      cosine-similarity retrieval restricted to temporally distant,
//!      spatially-plausible pairs (`--min-frame-gap`, `--min-path-length`).
//!   3. Geometric verification: stereo-triangulate SIFT keypoints at the
//!      older keyframe (cam0/cam1, Double Sphere model, known extrinsics),
//!      match them into the newer keyframe's cam0 image (mutual-softmax,
//!      LightGlue-style but training-free), and solve PnP RANSAC in the DS
//!      normalized-image plane for a metric relative pose. Accepted loops
//!      become loop-closure edges (weight = inlier count).
//!   4. Optimize an SE(3) pose graph over the keyframes
//!      (`pipelines::slam::pose_graph`).
//!   5. Propagate each keyframe's correction to every VIO frame attached to
//!      it (ORB-SLAM-style rigid spanning-tree propagation: the frame's pose
//!      relative to its nearest preceding keyframe is kept exactly as VIO
//!      produced it, only the keyframe's absolute correction moves it).
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
use visloc_rs::vision::features::sift::{extract_sift_features, GrayImage, SiftConfig};
use visloc_rs::{
    relative_world_to_camera, triangulate_two_view_left_frame, Correspondence2D3D, DltPnP,
    FeatureSet, GaussNewtonPoseRefiner, LinearSolver, LoopClosureConstraint, Matcher,
    MutualSoftmaxConfig, MutualSoftmaxMatcher, PnPRansac, PoseGraph, PoseGraphSe3Config,
    RobustPoseEstimator,
};

const CAM0: CameraId = 0;
const CAM1: CameraId = 1;

#[derive(Debug, Clone)]
struct Args {
    sequence: String,
    dataset_dir: PathBuf,
    calibration: PathBuf,
    vio_out_dir: PathBuf,
    out_dir: PathBuf,
    min_frame_gap: u64,
    min_path_length_m: f64,
    retrieval_top_k: usize,
    retrieval_min_similarity: f32,
    match_min_confidence: f32,
    min_correspondences: usize,
    min_inliers: usize,
    min_inlier_ratio: f64,
    reprojection_threshold: f64,
    max_candidates: usize,
    max_keyframes: Option<usize>,
    sift_max_keypoints: usize,
    only_frame_ids: Option<Vec<u64>>,
}

impl Args {
    fn parse<I: Iterator<Item = String>>(mut args: I) -> Result<Self, String> {
        let mut sequence = None;
        let mut dataset_dir = None;
        let mut calibration = None;
        let mut vio_out_dir = None;
        let mut out_dir = None;
        let mut min_frame_gap = 200u64;
        let mut min_path_length_m = 5.0f64;
        let mut retrieval_top_k = 3usize;
        let mut retrieval_min_similarity = 0.75f32;
        let mut match_min_confidence = 0.2f32;
        let mut min_correspondences = 12usize;
        let mut min_inliers = 20usize;
        let mut min_inlier_ratio = 0.35f64;
        let mut reprojection_threshold = 0.012f64;
        let mut max_candidates = 4000usize;
        let mut max_keyframes = None;
        let mut sift_max_keypoints = 800usize;
        let mut only_frame_ids = None;

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
                "--min-frame-gap" => {
                    min_frame_gap = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--min-path-length" => {
                    min_path_length_m = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--retrieval-top-k" => {
                    retrieval_top_k = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--retrieval-min-similarity" => {
                    retrieval_min_similarity = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--match-min-confidence" => {
                    match_min_confidence = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--min-correspondences" => {
                    min_correspondences = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--min-inliers" => min_inliers = value!().parse().map_err(|e| format!("{e}"))?,
                "--min-inlier-ratio" => {
                    min_inlier_ratio = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--reprojection-threshold" => {
                    reprojection_threshold = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--max-candidates" => {
                    max_candidates = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--max-keyframes" => {
                    max_keyframes = Some(value!().parse().map_err(|e| format!("{e}"))?)
                }
                "--sift-max-keypoints" => {
                    sift_max_keypoints = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--only-frame-ids" => {
                    let raw = value!();
                    only_frame_ids = Some(
                        raw.split(',')
                            .map(|s| s.parse::<u64>().map_err(|e| format!("{e}")))
                            .collect::<Result<Vec<u64>, String>>()?,
                    );
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
            min_frame_gap,
            min_path_length_m,
            retrieval_top_k,
            retrieval_min_similarity,
            match_min_confidence,
            min_correspondences,
            min_inliers,
            min_inlier_ratio,
            reprojection_threshold,
            max_candidates,
            max_keyframes,
            sift_max_keypoints,
            only_frame_ids,
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
    /// Index into the owning keyframe's cam0 `FeatureSet`.
    keypoint_index: usize,
    /// Triangulated 3D point in the *cam0* frame of the owning keyframe.
    point_cam0: Point3<f64>,
}

struct KeyframeData {
    frame_id: u64,
    timestamp_ns: i64,
    cam0_features: FeatureSet,
    /// DS-normalized planar coordinate `(x/z, y/z)` per cam0 keypoint, or
    /// `None` when unprojection failed (grazing / invalid ray).
    cam0_normalized: Vec<Option<Point2<f64>>>,
    mean_descriptor: Vec<f32>,
    landmarks: Vec<Landmark>,
}

#[derive(Debug, Clone)]
struct AcceptedLoop {
    from_frame_id: u64,
    to_frame_id: u64,
    from_timestamp_ns: i64,
    to_timestamp_ns: i64,
    correspondences: usize,
    inliers: usize,
    inlier_ratio: f64,
    mean_reprojection_error: f64,
    retrieval_similarity: f32,
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
             --calibration FILE --vio-out-dir DIR --out-dir DIR [options]"
        )
    })?;
    fs::create_dir_all(&args.out_dir)?;

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

    eprintln!("[{}] loading VIO trajectory", args.sequence);
    let trajectory = read_trajectory_csv(&args.vio_out_dir.join("trajectory.csv"))?;
    let cumulative_length = cumulative_arc_length(&trajectory);
    let pose_by_frame: BTreeMap<u64, &TrajRow> =
        trajectory.iter().map(|row| (row.frame_id, row)).collect();

    let mut keyframe_ids = list_keyframe_frame_ids(&args.vio_out_dir.join("marg_data"))?;
    if let Some(limit) = args.max_keyframes {
        keyframe_ids.truncate(limit);
    }
    if let Some(only) = &args.only_frame_ids {
        keyframe_ids.retain(|id| only.contains(id));
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

    // ---- 1. Per-keyframe SIFT features, stereo triangulation, appearance ----
    let sift_config = SiftConfig {
        max_keypoints: args.sift_max_keypoints,
        ..SiftConfig::default()
    };
    let identity_camera = GenericCamera::pinhole(0, cam0.width, cam0.height, 1.0, 1.0, 0.0, 0.0);
    let stereo_matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
        min_confidence: args.match_min_confidence,
        ..MutualSoftmaxConfig::default()
    });

    let mut keyframes: Vec<KeyframeData> = Vec::with_capacity(keyframe_ids.len());
    for (i, &frame_id) in keyframe_ids.iter().enumerate() {
        let row = *pose_by_frame
            .get(&frame_id)
            .ok_or_else(|| format!("keyframe frame_id {frame_id} missing from trajectory.csv"))?;
        let cam0_features =
            load_sift_features(&args.dataset_dir, "cam0", row.timestamp_ns, &sift_config)?;
        let cam1_features =
            load_sift_features(&args.dataset_dir, "cam1", row.timestamp_ns, &sift_config)?;

        let cam0_normalized = ds_normalize(&cam0, &cam0_features);
        let cam1_normalized = ds_normalize(&cam1, &cam1_features);
        let mean_descriptor = mean_l2_normalized_descriptor(&cam0_features);

        let query = l2_normalize_all(&cam0_features.descriptors);
        let train = l2_normalize_all(&cam1_features.descriptors);
        let stereo_matches = stereo_matcher.match_descriptors(&query, &train);

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

        if (i + 1) % 50 == 0 || i + 1 == keyframe_ids.len() {
            eprintln!(
                "[{}] keyframe {}/{} frame_id={} cam0_kpts={} landmarks={}",
                args.sequence,
                i + 1,
                keyframe_ids.len(),
                frame_id,
                cam0_features.len(),
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
        });
    }

    // ---- 2. Build the pose graph: nodes + sequential odometry edges ----
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

    // ---- 3. Retrieval: cheap appearance top-k over gap/path-gated pairs ----
    let mut candidate_pairs: Vec<(usize, usize, f32)> = Vec::new();
    for j in 0..keyframes.len() {
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for i in 0..j {
            let frame_gap = keyframes[j].frame_id.saturating_sub(keyframes[i].frame_id);
            if frame_gap < args.min_frame_gap {
                continue;
            }
            let path_len =
                cumulative_length[keyframes[j].frame_id as usize].max(0.0)
                    - cumulative_length[keyframes[i].frame_id as usize].max(0.0);
            if path_len < args.min_path_length_m {
                continue;
            }
            let sim = cosine_similarity(&keyframes[i].mean_descriptor, &keyframes[j].mean_descriptor);
            if sim >= args.retrieval_min_similarity {
                scored.push((i, sim));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(args.retrieval_top_k);
        for (i, sim) in scored {
            candidate_pairs.push((i, j, sim));
            if candidate_pairs.len() >= args.max_candidates {
                break;
            }
        }
        if candidate_pairs.len() >= args.max_candidates {
            break;
        }
    }
    eprintln!(
        "[{}] {} retrieval candidates (gap>={}, path>={:.1}m, sim>={:.2})",
        args.sequence,
        candidate_pairs.len(),
        args.min_frame_gap,
        args.min_path_length_m,
        args.retrieval_min_similarity
    );

    // ---- 4. Geometric verification: stereo landmarks -> PnP RANSAC ----
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
    let mono_matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
        min_confidence: args.match_min_confidence,
        ..MutualSoftmaxConfig::default()
    });

    let mut accepted: Vec<AcceptedLoop> = Vec::new();
    let mut evaluated = 0usize;
    for &(i, j, similarity) in &candidate_pairs {
        evaluated += 1;
        let kf_a = &keyframes[i];
        let kf_b = &keyframes[j];
        if kf_a.landmarks.is_empty() {
            continue;
        }
        let query: Vec<Vec<f32>> = kf_a
            .landmarks
            .iter()
            .map(|lm| kf_a.cam0_features.descriptors[lm.keypoint_index].clone())
            .collect();
        let query = l2_normalize_all(&query);
        let train = l2_normalize_all(&kf_b.cam0_features.descriptors);
        let matches = mono_matcher.match_descriptors(&query, &train);

        let mut correspondences = Vec::with_capacity(matches.len());
        for m in &matches {
            let Some(point2d) = kf_b.cam0_normalized[m.train_index] else {
                continue;
            };
            correspondences.push(Correspondence2D3D {
                point2d,
                point3d: kf_a.landmarks[m.query_index].point_cam0,
                confidence: m.confidence,
            });
        }
        if correspondences.len() < args.min_correspondences {
            continue;
        }

        let Some(report) = pnp_ransac.estimate(&correspondences, &identity_camera) else {
            continue;
        };
        let inlier_ratio = report.inliers.len() as f64 / correspondences.len() as f64;
        if report.inliers.len() < args.min_inliers || inlier_ratio < args.min_inlier_ratio {
            continue;
        }

        // report.pose.world_to_camera == T_cam0A_to_cam0B.
        let r_rel = report.pose.world_to_camera.clone();
        let measurement_body = t_cam0_to_imu.compose(&r_rel).compose(&t_imu_to_cam0);
        if std::env::var("LC_DEBUG_LOOPS").is_ok() {
            let vio_a = pose_by_frame[&kf_a.frame_id].imu_to_world.clone();
            let vio_b = pose_by_frame[&kf_b.frame_id].imu_to_world.clone();
            let world_to_body_a = vio_a.inverse();
            let predicted_world_to_body_b = measurement_body.compose(&world_to_body_a);
            let predicted_body_to_world_b = predicted_world_to_body_b.inverse();
            let predicted_p_b = predicted_body_to_world_b.translation;
            eprintln!(
                "[debug] loop {}->{} predicted_world_p_b=[{:.3},{:.3},{:.3}] \
                 vio_world_p_b=[{:.3},{:.3},{:.3}] vio_world_p_a=[{:.3},{:.3},{:.3}] \
                 |predicted-vio_b|={:.3} inliers={} corr={} reproj={:.5}",
                kf_a.frame_id,
                kf_b.frame_id,
                predicted_p_b.x,
                predicted_p_b.y,
                predicted_p_b.z,
                vio_b.translation.x,
                vio_b.translation.y,
                vio_b.translation.z,
                vio_a.translation.x,
                vio_a.translation.y,
                vio_a.translation.z,
                (predicted_p_b - vio_b.translation).norm(),
                report.inliers.len(),
                correspondences.len(),
                report.mean_reprojection_error,
            );
        }
        let constraint = LoopClosureConstraint {
            from_keyframe_id: kf_a.frame_id,
            to_keyframe_id: kf_b.frame_id,
            relative_pose: measurement_body,
            inlier_count: report.inliers.len(),
            inlier_ratio,
            mean_sampson_error: 0.0,
            score: inlier_ratio,
        };
        // A raw inlier count (routinely 20-150 here) used as an isotropic
        // edge weight would out-leverage every sequential odometry edge
        // (weight 1.0) by one to two orders of magnitude, letting a single
        // noisy PnP loop estimate wrench the whole chain. Bound it to a
        // modest, inlier-confidence-scaled range instead.
        let loop_weight = (report.inliers.len() as f64 * 0.1).clamp(1.0, 8.0);
        graph.add_loop_closure_constraint_with_weight(&constraint, loop_weight);
        accepted.push(AcceptedLoop {
            from_frame_id: kf_a.frame_id,
            to_frame_id: kf_b.frame_id,
            from_timestamp_ns: kf_a.timestamp_ns,
            to_timestamp_ns: kf_b.timestamp_ns,
            correspondences: correspondences.len(),
            inliers: report.inliers.len(),
            inlier_ratio,
            mean_reprojection_error: report.mean_reprojection_error,
            retrieval_similarity: similarity,
        });
    }
    eprintln!(
        "[{}] {} candidates evaluated, {} loops accepted",
        args.sequence,
        evaluated,
        accepted.len()
    );

    // ---- 5. Optimize the SE(3) pose graph with GNC ----
    // A plain Huber/Cauchy IRLS solve turned out not to be robust enough on
    // its own: even a "verified" loop (low reprojection error, decent inlier
    // count) can be a self-consistent-but-wrong PnP solve on a repetitive
    // machine-hall texture, and a single such loop is enough to pull a
    // non-GNC solve into a worse basin than plain VIO (measured: MH_01 ATE
    // 0.066m VIO -> 0.187-0.198m with Huber-only PGO). GNC anneals from a
    // convex surrogate that trusts every edge to the true robust cost that
    // rejects outliers, matching the loop-closure machinery this repo's own
    // vision-only benchmark (docs/euroc_loop_closure_benchmark.md) uses.
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
    let gnc_result = graph.optimize_se3_gnc(&pgo_config, &gnc_config);
    let (final_cost, converged, outer_iterations, gnc_inliers, gnc_outliers) = match &gnc_result {
        Ok(result) => (
            result.final_cost,
            result.converged,
            result.outer_iterations,
            result.inlier_count(0.5),
            result.outlier_count(0.5),
        ),
        Err(_) => (graph.se3_cost(), false, 0, 0, 0),
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
        "candidate_pairs_retrieved": candidate_pairs.len(),
        "candidates_evaluated": evaluated,
        "loops_accepted": accepted.len(),
        "accepted_loops": accepted.iter().map(|l| serde_json::json!({
            "from_frame_id": l.from_frame_id,
            "to_frame_id": l.to_frame_id,
            "from_timestamp_ns": l.from_timestamp_ns,
            "to_timestamp_ns": l.to_timestamp_ns,
            "correspondences": l.correspondences,
            "inliers": l.inliers,
            "inlier_ratio": l.inlier_ratio,
            "mean_reprojection_error": l.mean_reprojection_error,
            "retrieval_similarity": l.retrieval_similarity,
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
        "candidate_pairs_retrieved": candidate_pairs.len(),
        "candidates_evaluated": evaluated,
        "loops_accepted": accepted.len(),
        "pgo_initial_cost": initial_cost,
        "pgo_final_cost": final_cost,
        "pgo_converged": converged,
        "pgo_outer_iterations": outer_iterations,
        "gnc_inlier_edges": gnc_inliers,
        "gnc_outlier_edges": gnc_outliers,
        "wall_seconds": wall_seconds,
        "args": {
            "min_frame_gap": args.min_frame_gap,
            "min_path_length_m": args.min_path_length_m,
            "retrieval_top_k": args.retrieval_top_k,
            "retrieval_min_similarity": args.retrieval_min_similarity,
            "match_min_confidence": args.match_min_confidence,
            "min_correspondences": args.min_correspondences,
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": args.min_inlier_ratio,
            "reprojection_threshold": args.reprojection_threshold,
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

/// `Pose.world_to_camera` here means "world-to-body": the inverse of the
/// VIO's `imu_to_world`. This lets [`relative_world_to_camera`] and the
/// pose-graph solver operate directly on IMU/body frames.
fn body_pose(row: &TrajRow) -> Pose {
    Pose {
        world_to_camera: row.imu_to_world.inverse(),
    }
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

fn load_sift_features(
    dataset_dir: &Path,
    camera_folder: &str,
    timestamp_ns: i64,
    config: &SiftConfig,
) -> Result<FeatureSet, Box<dyn std::error::Error>> {
    let path = dataset_dir
        .join("mav0")
        .join(camera_folder)
        .join("data")
        .join(format!("{timestamp_ns}.png"));
    let image = visloc_io::images::read_common_image(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let pixels: Vec<f32> = image.pixels().to_vec();
    let gray = GrayImage::new(image.width(), image.height(), &pixels)?;
    Ok(extract_sift_features(&gray, config)?)
}

fn ds_normalize(camera: &DoubleSphereCamera, features: &FeatureSet) -> Vec<Option<Point2<f64>>> {
    features
        .keypoints
        .iter()
        .map(|kp| {
            let pixel = Point2::new(kp.x, kp.y);
            let ray = camera.unproject(&pixel)?;
            if !ray.z.is_finite() || ray.z <= 0.05 {
                return None;
            }
            Some(Point2::new(ray.x / ray.z, ray.y / ray.z))
        })
        .collect()
}

fn l2_normalize_all(descriptors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    descriptors
        .iter()
        .map(|d| {
            let norm = d.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-12 {
                d.iter().map(|v| v / norm).collect()
            } else {
                d.clone()
            }
        })
        .collect()
}

fn mean_l2_normalized_descriptor(features: &FeatureSet) -> Vec<f32> {
    if features.descriptors.is_empty() {
        return Vec::new();
    }
    let normalized = l2_normalize_all(&features.descriptors);
    let dim = normalized[0].len();
    let mut mean = vec![0.0f32; dim];
    for d in &normalized {
        for (m, v) in mean.iter_mut().zip(d.iter()) {
            *m += v;
        }
    }
    let count = normalized.len() as f32;
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
