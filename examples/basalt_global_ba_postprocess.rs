//! Stage B: global bundle adjustment on top of the Stage-A pose-graph result
//! (`examples/basalt_loop_closure_postprocess.rs`).
//!
//! Inputs: Stage A's `vio-out-dir` (trajectory.csv + marg_data/ keyframe set,
//! same as Stage A) and Stage A's `out-dir` (corrected_trajectory.tum for the
//! PG-corrected keyframe pose initialization, loops.json for the accepted
//! loop pairs to reuse as extra BA observations).
//!
//! Per-keyframe SuperPoint keypoints/descriptors and the stereo landmarks
//! they triangulate to are cached to `--cache-dir` (one binary file per
//! keyframe) so repeated Stage B runs -- BA config sweeps in particular --
//! skip the ~8 min ONNX inference pass entirely after the first run.
//!
//! Structure building (deliberately simple, stated honestly -- this is not a
//! full incremental-SfM track merger):
//!   - Each keyframe's stereo (cam0/cam1) landmarks get their own landmark id
//!     (`frame_id * 10_000 + stereo_index`); no cross-keyframe landmark
//!     fusion is attempted, so the same physical point re-triangulated at
//!     two keyframes becomes two BA landmarks. This is simpler than track
//!     fusion but still lets every extra observation pull on keyframe poses.
//!   - Temporal edges (k -> k+1, k -> k+2): LightGlue-match keyframe k's
//!     landmark-linked cam0 descriptors into keyframe k+offset's cam0
//!     descriptors; each match adds a monocular `BaObservation` at k+offset.
//!   - Loop edges: the same match-and-add-observation step, over Stage A's
//!     accepted loop pairs instead of temporal neighbors.
//!   - Stereo (cam0/cam1) observations use `BaGeneralStereoObservation`
//!     (EuRoC's cam0/cam1 extrinsic is rotational, not a clean rectified
//!     baseline) with the same "DS-unproject to a normalized plane, feed an
//!     identity pinhole `Camera`" trick Stage A uses for PnP/triangulation.
//!   - VIO odometry between consecutive keyframes is optionally kept as a
//!     `PairwisePoseFactor` weak prior (`--odometry-prior-weight`, 0
//!     disables it), converted from Stage A's body-frame relative pose into
//!     the BA's cam0-frame poses.
//!
//! Ground truth is never read here. Robust-kernel / threshold defaults are
//! chosen a priori (see the `Args` doc comments for the stated reasoning),
//! not swept against ATE.

use std::{
    collections::BTreeMap,
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use nalgebra::{Point2, Point3, Quaternion, UnitQuaternion, Vector3};

use visloc_basalt::{BasaltCalibration, CameraId, DoubleSphereCamera};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::core::types::Camera as GenericCamera;
use visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher;
use visloc_rs::vision::features::superpoint_onnx::{SuperPointOnnxConfig, SuperPointOnnxExtractor};
use visloc_rs::slam::BaGeneralStereoObservation;
use visloc_rs::{
    relative_world_to_camera, triangulate_two_view_left_frame, BaConfig, BaObservation,
    BundleAdjustment, DeepFeatureExtractor, LinearSolver, PairwisePoseFactor, RobustKernel,
};

const CAM0: CameraId = 0;
const CAM1: CameraId = 1;
const CACHE_MAGIC: u32 = 0x4241_4b31; // "BAK1"

#[derive(Debug, Clone)]
struct Args {
    sequence: String,
    dataset_dir: PathBuf,
    calibration: PathBuf,
    vio_out_dir: PathBuf,
    stage_a_out_dir: PathBuf,
    out_dir: PathBuf,
    cache_dir: PathBuf,
    superpoint_model: PathBuf,
    lightglue_model: PathBuf,
    temporal_window: usize,
    sp_max_keypoints: usize,
    /// Huber delta on the reprojection residual, in the DS *normalized*
    /// image plane (not pixels). Chosen a priori, mirroring Stage A's PnP
    /// `reprojection_threshold`: ~4 px at EuRoC's fx~350-460 px is
    /// `4/350 ~= 0.011`; 0.01 is the same order of magnitude, picked before
    /// looking at any ATE number.
    ba_huber_delta: f64,
    ba_max_iterations: usize,
    /// `PairwisePoseFactor` weight (= 1/sigma^2) tying each keyframe to its
    /// VIO-derived relative pose from the previous keyframe. 0 disables the
    /// prior entirely. A weak, a priori pick: the mixed metre/radian SE(3)
    /// tangent residual at a converged, VIO-consistent pose is near zero, so
    /// this mostly resists gross drift rather than pinning the solution --
    /// "weak prior" per the design brief, not tuned against ATE.
    odometry_prior_weight: f64,
    max_keyframes: Option<usize>,
}

impl Args {
    fn parse<I: Iterator<Item = String>>(mut args: I) -> Result<Self, String> {
        let mut sequence = None;
        let mut dataset_dir = None;
        let mut calibration = None;
        let mut vio_out_dir = None;
        let mut stage_a_out_dir = None;
        let mut out_dir = None;
        let mut cache_dir = None;
        let mut superpoint_model = None;
        let mut lightglue_model = None;
        let mut temporal_window = 2usize;
        let mut sp_max_keypoints = 512usize;
        let mut ba_huber_delta = 0.01f64;
        let mut ba_max_iterations = 30usize;
        let mut odometry_prior_weight = 100.0f64;
        let mut max_keyframes = None;

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
                "--stage-a-out-dir" => stage_a_out_dir = Some(PathBuf::from(value!())),
                "--out-dir" => out_dir = Some(PathBuf::from(value!())),
                "--cache-dir" => cache_dir = Some(PathBuf::from(value!())),
                "--superpoint-model" => superpoint_model = Some(PathBuf::from(value!())),
                "--lightglue-model" => lightglue_model = Some(PathBuf::from(value!())),
                "--temporal-window" => {
                    temporal_window = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--sp-max-keypoints" => {
                    sp_max_keypoints = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--ba-huber-delta" => {
                    ba_huber_delta = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--ba-max-iterations" => {
                    ba_max_iterations = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--odometry-prior-weight" => {
                    odometry_prior_weight = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--max-keyframes" => {
                    max_keyframes = Some(value!().parse().map_err(|e| format!("{e}"))?)
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }

        Ok(Args {
            sequence: sequence.ok_or("--sequence is required")?,
            dataset_dir: dataset_dir.ok_or("--dataset-dir is required")?,
            calibration: calibration.ok_or("--calibration is required")?,
            vio_out_dir: vio_out_dir.ok_or("--vio-out-dir is required")?,
            stage_a_out_dir: stage_a_out_dir.ok_or("--stage-a-out-dir is required")?,
            out_dir: out_dir.ok_or("--out-dir is required")?,
            cache_dir: cache_dir.ok_or("--cache-dir is required")?,
            superpoint_model: superpoint_model.ok_or("--superpoint-model is required")?,
            lightglue_model: lightglue_model.ok_or("--lightglue-model is required")?,
            temporal_window,
            sp_max_keypoints,
            ba_huber_delta,
            ba_max_iterations,
            odometry_prior_weight,
            max_keyframes,
        })
    }
}

#[derive(Debug, Clone)]
struct TrajRow {
    frame_id: u64,
    timestamp_ns: i64,
    imu_to_world: SE3,
}

#[derive(Debug, Clone)]
struct StereoLandmark {
    keypoint_index: usize,
    point_cam0: Point3<f64>,
}

struct KeyframeCache {
    cam0_keypoints: Vec<Point2<f64>>,
    cam0_descriptors: Vec<Vec<f32>>,
    cam1_keypoints: Vec<Point2<f64>>,
    cam1_descriptors: Vec<Vec<f32>>,
    landmarks: Vec<StereoLandmark>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("basalt_global_ba_postprocess: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let args = Args::parse(env::args().skip(1)).map_err(|e| {
        format!(
            "{e}\nusage: basalt_global_ba_postprocess --sequence NAME --dataset-dir DIR \
             --calibration FILE --vio-out-dir DIR --stage-a-out-dir DIR --out-dir DIR \
             --cache-dir DIR --superpoint-model FILE --lightglue-model FILE [options]"
        )
    })?;
    fs::create_dir_all(&args.out_dir)?;
    fs::create_dir_all(&args.cache_dir)?;

    let calibration = BasaltCalibration::from_path(&args.calibration)?;
    let cam0 = calibration.camera(CAM0).ok_or("missing cam0")?.clone();
    let cam1 = calibration.camera(CAM1).ok_or("missing cam1")?.clone();
    let t_cam0_to_cam1 = calibration
        .imu_to_camera(CAM1)
        .ok_or("missing T_imu_cam1")?
        .compose(calibration.camera_to_imu(CAM0).ok_or("missing T_imu_cam0")?);
    let t_cam0_to_imu = calibration
        .camera_to_imu(CAM0)
        .ok_or("missing T_imu_cam0")?
        .clone();
    let t_imu_to_cam0 = calibration
        .imu_to_camera(CAM0)
        .ok_or("missing T_imu_cam0")?;

    let trajectory = read_trajectory_csv(&args.vio_out_dir.join("trajectory.csv"))?;
    let pose_by_frame: BTreeMap<u64, &TrajRow> =
        trajectory.iter().map(|row| (row.frame_id, row)).collect();
    let corrected_pg = read_tum_by_row(&args.stage_a_out_dir.join("corrected_trajectory.tum"))?;

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
        return Err("fewer than 2 keyframes".into());
    }

    let identity_camera = GenericCamera::pinhole(0, cam0.width, cam0.height, 1.0, 1.0, 0.0, 0.0);

    // ---- 1. Per-keyframe SP features + stereo landmarks, cached ----
    let mut extractor_and_matcher: Option<(SuperPointOnnxExtractor, LightGlueOnnxMatcher)> = None;
    let t_load = Instant::now();
    let mut cache_hits = 0usize;
    let mut cache_misses = 0usize;
    let mut keyframes: BTreeMap<u64, KeyframeCache> = BTreeMap::new();
    for &frame_id in &keyframe_ids {
        let cache_path = args.cache_dir.join(format!("{frame_id}.kfcache"));
        if let Some(cached) = read_cache(&cache_path)? {
            cache_hits += 1;
            keyframes.insert(frame_id, cached);
            continue;
        }
        cache_misses += 1;
        let (extractor, matcher) = match &extractor_and_matcher {
            Some(pair) => pair,
            None => {
                eprintln!(
                    "[{}] cache miss -- loading SuperPoint/LightGlue ONNX",
                    args.sequence
                );
                let sp_config = SuperPointOnnxConfig {
                    max_keypoints: args.sp_max_keypoints,
                    ..SuperPointOnnxConfig::default()
                };
                let extractor = SuperPointOnnxExtractor::load_from_path(
                    &args.superpoint_model,
                    sp_config,
                )?;
                let matcher = LightGlueOnnxMatcher::load_from_path(&args.lightglue_model)?;
                extractor_and_matcher = Some((extractor, matcher));
                extractor_and_matcher.as_ref().unwrap()
            }
        };
        let row = *pose_by_frame
            .get(&frame_id)
            .ok_or_else(|| format!("keyframe {frame_id} missing from trajectory.csv"))?;
        let cam0_image = load_gray_image(&args.dataset_dir, "cam0", row.timestamp_ns)?;
        let cam1_image = load_gray_image(&args.dataset_dir, "cam1", row.timestamp_ns)?;
        let cam0_features = extractor.extract_deep(&cam0_image)?;
        let cam1_features = extractor.extract_deep(&cam1_image)?;
        let cam0_normalized = ds_normalize(&cam0, &cam0_features.keypoints);
        let cam1_normalized = ds_normalize(&cam1, &cam1_features.keypoints);
        let stereo_matches = matcher.match_features(
            &cam0_features.keypoints,
            &cam0_features.descriptors,
            &cam1_features.keypoints,
            &cam1_features.descriptors,
        )?;
        let mut landmarks = Vec::new();
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
            landmarks.push(StereoLandmark {
                keypoint_index: m.query_index,
                point_cam0: point,
            });
        }
        let cache = KeyframeCache {
            cam0_keypoints: cam0_features.keypoints,
            cam0_descriptors: cam0_features.descriptors,
            cam1_keypoints: cam1_features.keypoints,
            cam1_descriptors: cam1_features.descriptors,
            landmarks,
        };
        write_cache(&cache_path, &cache)?;
        keyframes.insert(frame_id, cache);
    }
    eprintln!(
        "[{}] features: {} cache hits, {} computed+cached in {:.1}s",
        args.sequence,
        cache_hits,
        cache_misses,
        t_load.elapsed().as_secs_f64()
    );

    // ---- 2. Build the BA problem ----
    let mut ba = BundleAdjustment::new(identity_camera.clone());
    let mut cam0_pose_by_frame: BTreeMap<u64, Pose> = BTreeMap::new();
    for &frame_id in &keyframe_ids {
        let body_pose = corrected_pg
            .get(frame_id as usize)
            .cloned()
            .ok_or_else(|| format!("no corrected pose row for keyframe {frame_id}"))?;
        let cam0_wc = t_imu_to_cam0.compose(&body_pose.world_to_camera);
        let pose = Pose {
            world_to_camera: cam0_wc,
        };
        ba.add_pose(frame_id, pose.clone());
        cam0_pose_by_frame.insert(frame_id, pose);
    }
    ba.fixed_poses.insert(keyframe_ids[0]);

    let mut landmark_count = 0usize;
    let mut stereo_observation_count = 0usize;
    for &frame_id in &keyframe_ids {
        let kf = &keyframes[&frame_id];
        let cam0_pose = &cam0_pose_by_frame[&frame_id];
        for lm in &kf.landmarks {
            let landmark_id = frame_id * 10_000 + lm.keypoint_index as u64;
            let point_world = cam0_pose.world_to_camera.inverse().transform_point(&lm.point_cam0);
            ba.add_landmark(landmark_id, point_world);
            landmark_count += 1;
            let kp = kf.cam0_keypoints[lm.keypoint_index];
            let Some(xy_left) = ds_normalize_one(&cam0, &kp) else {
                continue;
            };
            // Recover the matched cam1 keypoint's normalized coordinate by
            // re-projecting the triangulated point through the known
            // extrinsic -- avoids re-storing the cam1 index in the cache.
            let point_cam1 = t_cam0_to_cam1.transform_point(&lm.point_cam0);
            if point_cam1.z <= 0.05 {
                continue;
            }
            let xy_right = Point2::new(point_cam1.x / point_cam1.z, point_cam1.y / point_cam1.z);
            ba.add_general_stereo_observation(BaGeneralStereoObservation {
                keyframe_id: frame_id,
                landmark_id,
                xy_left,
                xy_right,
                right_camera: identity_camera.clone(),
                left_to_right: t_cam0_to_cam1.clone(),
            });
            stereo_observation_count += 1;
        }
    }
    eprintln!(
        "[{}] {} landmarks, {} stereo observations",
        args.sequence, landmark_count, stereo_observation_count
    );

    // ---- 3. Temporal tracks (k -> k+1, k -> k+2, ...) ----
    let (temp_matcher, temp_obs) = {
        let matcher = LightGlueOnnxMatcher::load_from_path(&args.lightglue_model)?;
        let mut extra_observations = 0usize;
        for (i, &frame_id) in keyframe_ids.iter().enumerate() {
            let kf = &keyframes[&frame_id];
            if kf.landmarks.is_empty() {
                continue;
            }
            let ref_keypoints: Vec<Point2<f64>> = kf
                .landmarks
                .iter()
                .map(|lm| kf.cam0_keypoints[lm.keypoint_index])
                .collect();
            let ref_descriptors: Vec<Vec<f32>> = kf
                .landmarks
                .iter()
                .map(|lm| kf.cam0_descriptors[lm.keypoint_index].clone())
                .collect();
            for offset in 1..=args.temporal_window {
                let Some(&target_id) = keyframe_ids.get(i + offset) else {
                    continue;
                };
                let target = &keyframes[&target_id];
                let matches = matcher.match_features(
                    &ref_keypoints,
                    &ref_descriptors,
                    &target.cam0_keypoints,
                    &target.cam0_descriptors,
                )?;
                for m in &matches {
                    let Some(xy) = ds_normalize_one(&cam0, &target.cam0_keypoints[m.train_index])
                    else {
                        continue;
                    };
                    let landmark_id =
                        frame_id * 10_000 + kf.landmarks[m.query_index].keypoint_index as u64;
                    ba.add_observation(BaObservation {
                        keyframe_id: target_id,
                        landmark_id,
                        xy,
                    });
                    extra_observations += 1;
                }
            }
        }
        (matcher, extra_observations)
    };
    eprintln!(
        "[{}] {} temporal-track observations (window={})",
        args.sequence, temp_obs, args.temporal_window
    );

    // ---- 4. Loop-pair observations (reusing Stage A's accepted loops) ----
    let loops_path = args.stage_a_out_dir.join("loops.json");
    let mut loop_obs = 0usize;
    let mut loop_pairs_used = 0usize;
    if let Ok(text) = fs::read_to_string(&loops_path) {
        let payload: serde_json::Value = serde_json::from_str(&text)?;
        for item in payload["accepted_loops"].as_array().cloned().unwrap_or_default() {
            let (Some(from_id), Some(to_id)) =
                (item["from_frame_id"].as_u64(), item["to_frame_id"].as_u64())
            else {
                continue;
            };
            let (Some(kf_a), Some(kf_b)) = (keyframes.get(&from_id), keyframes.get(&to_id))
            else {
                continue;
            };
            if kf_a.landmarks.is_empty() {
                continue;
            }
            loop_pairs_used += 1;
            let ref_keypoints: Vec<Point2<f64>> = kf_a
                .landmarks
                .iter()
                .map(|lm| kf_a.cam0_keypoints[lm.keypoint_index])
                .collect();
            let ref_descriptors: Vec<Vec<f32>> = kf_a
                .landmarks
                .iter()
                .map(|lm| kf_a.cam0_descriptors[lm.keypoint_index].clone())
                .collect();
            let matches = temp_matcher.match_features(
                &ref_keypoints,
                &ref_descriptors,
                &kf_b.cam0_keypoints,
                &kf_b.cam0_descriptors,
            )?;
            for m in &matches {
                let Some(xy) = ds_normalize_one(&cam0, &kf_b.cam0_keypoints[m.train_index]) else {
                    continue;
                };
                let landmark_id =
                    from_id * 10_000 + kf_a.landmarks[m.query_index].keypoint_index as u64;
                ba.add_observation(BaObservation {
                    keyframe_id: to_id,
                    landmark_id,
                    xy,
                });
                loop_obs += 1;
            }
        }
    }
    eprintln!(
        "[{}] {} loop-pair observations from {} loop pairs",
        args.sequence, loop_obs, loop_pairs_used
    );

    // ---- 5. VIO odometry priors between consecutive keyframes (optional) ----
    let mut odometry_factors = 0usize;
    if args.odometry_prior_weight > 0.0 {
        for window in keyframe_ids.windows(2) {
            let (from_id, to_id) = (window[0], window[1]);
            let from_body = body_pose(pose_by_frame[&from_id]);
            let to_body = body_pose(pose_by_frame[&to_id]);
            let vio_relative_body = relative_world_to_camera(&from_body, &to_body);
            let cam0_relative = t_imu_to_cam0
                .compose(&vio_relative_body)
                .compose(&t_cam0_to_imu);
            ba.pairwise_pose_factors.push(PairwisePoseFactor {
                keyframe_id_from: from_id,
                keyframe_id_to: to_id,
                measurement: Pose {
                    world_to_camera: cam0_relative,
                },
                weight: args.odometry_prior_weight,
            });
            odometry_factors += 1;
        }
    }
    eprintln!(
        "[{}] {} odometry pairwise-pose priors (weight={})",
        args.sequence, odometry_factors, args.odometry_prior_weight
    );

    let mean_reproj_before = mean_reprojection_error(&ba);

    // ---- 6. Optimize ----
    let ba_config = BaConfig {
        max_iterations: args.ba_max_iterations,
        robust_kernel: RobustKernel::Huber {
            delta: args.ba_huber_delta,
        },
        linear_solver: if keyframe_ids.len() > 250 {
            LinearSolver::Sparse
        } else {
            LinearSolver::Dense
        },
        ..BaConfig::default()
    };
    let t_ba = Instant::now();
    let ba_result = ba.optimize(&ba_config);
    let ba_wall_seconds = t_ba.elapsed().as_secs_f64();
    let (initial_cost, final_cost, converged, iterations) = match &ba_result {
        Ok(result) => (
            result.initial_cost,
            result.final_cost,
            result.converged,
            result.iterations.len(),
        ),
        Err(error) => {
            eprintln!("[{}] BA optimize failed: {error:?}", args.sequence);
            (0.0, 0.0, false, 0)
        }
    };
    let mean_reproj_after = mean_reprojection_error(&ba);
    eprintln!(
        "[{}] BA cost {:.6} -> {:.6} (converged={} iterations={} wall={:.1}s) mean_reproj {:.5} -> {:.5}",
        args.sequence,
        initial_cost,
        final_cost,
        converged,
        iterations,
        ba_wall_seconds,
        mean_reproj_before,
        mean_reproj_after
    );

    // ---- 7. Propagate corrected keyframe poses (BA, cam0->body) to full trajectory ----
    let mut corrected_body_by_frame: BTreeMap<u64, Pose> = BTreeMap::new();
    for &frame_id in &keyframe_ids {
        let cam0_wc = &ba.poses[&frame_id].world_to_camera;
        let body_wc = t_cam0_to_imu.compose(cam0_wc);
        corrected_body_by_frame.insert(frame_id, Pose { world_to_camera: body_wc });
    }
    let corrected_tum = propagate_corrections(&trajectory, &keyframe_ids, &corrected_body_by_frame);
    fs::write(args.out_dir.join("corrected_trajectory.tum"), corrected_tum)?;

    let wall_seconds = started.elapsed().as_secs_f64();
    let stats = serde_json::json!({
        "sequence": args.sequence,
        "keyframe_count": keyframe_ids.len(),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "landmark_count": landmark_count,
        "stereo_observation_count": stereo_observation_count,
        "temporal_observation_count": temp_obs,
        "loop_observation_count": loop_obs,
        "loop_pairs_used": loop_pairs_used,
        "odometry_factor_count": odometry_factors,
        "mean_reprojection_error_before": mean_reproj_before,
        "mean_reprojection_error_after": mean_reproj_after,
        "ba_initial_cost": initial_cost,
        "ba_final_cost": final_cost,
        "ba_converged": converged,
        "ba_iterations": iterations,
        "ba_wall_seconds": ba_wall_seconds,
        "wall_seconds": wall_seconds,
        "args": {
            "temporal_window": args.temporal_window,
            "sp_max_keypoints": args.sp_max_keypoints,
            "ba_huber_delta": args.ba_huber_delta,
            "ba_max_iterations": args.ba_max_iterations,
            "odometry_prior_weight": args.odometry_prior_weight,
        },
    });
    fs::write(
        args.out_dir.join("stats.json"),
        serde_json::to_string_pretty(&stats)?,
    )?;
    eprintln!(
        "[{}] done in {:.1}s -> {}",
        args.sequence,
        wall_seconds,
        args.out_dir.join("corrected_trajectory.tum").display()
    );
    Ok(())
}

fn mean_reprojection_error(ba: &BundleAdjustment) -> f64 {
    let mut total = 0.0f64;
    let mut count = 0usize;
    for obs in &ba.general_stereo_observations {
        let (Some(pose), Some(landmark)) = (
            ba.poses.get(&obs.keyframe_id),
            ba.landmarks.get(&obs.landmark_id),
        ) else {
            continue;
        };
        let p_cam = pose.world_to_camera.transform_point(landmark);
        if p_cam.z <= 1e-6 {
            continue;
        }
        let predicted_left = Point2::new(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
        total += (predicted_left - obs.xy_left).norm();
        count += 1;
        let p_right = obs.left_to_right.transform_point(&p_cam);
        if p_right.z > 1e-6 {
            let predicted_right = Point2::new(p_right.x / p_right.z, p_right.y / p_right.z);
            total += (predicted_right - obs.xy_right).norm();
            count += 1;
        }
    }
    for obs in &ba.observations {
        let (Some(pose), Some(landmark)) = (
            ba.poses.get(&obs.keyframe_id),
            ba.landmarks.get(&obs.landmark_id),
        ) else {
            continue;
        };
        let p_cam = pose.world_to_camera.transform_point(landmark);
        if p_cam.z <= 1e-6 {
            continue;
        }
        let predicted = Point2::new(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
        total += (predicted - obs.xy).norm();
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn body_pose(row: &TrajRow) -> Pose {
    Pose {
        world_to_camera: row.imu_to_world.inverse(),
    }
}

fn read_trajectory_csv(path: &Path) -> Result<Vec<TrajRow>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line_index == 0 || line.trim().is_empty() {
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
        rows.push(TrajRow {
            frame_id,
            timestamp_ns,
            imu_to_world: SE3::new(rotation, Vector3::new(tx, ty, tz)),
        });
    }
    rows.sort_by_key(|r| r.frame_id);
    Ok(rows)
}

/// Reads a TUM trajectory (`t tx ty tz qx qy qz qw`, no header) into a
/// `Vec<Pose>` indexed by row order, i.e. `frame_id` for a Stage-A
/// `corrected_trajectory.tum` (one row per original trajectory frame, in
/// order). `Pose.world_to_camera` is world-to-body (inverse of the stored
/// body-to-world tum row), matching Stage A's own convention.
fn read_tum_by_row(path: &Path) -> Result<Vec<Pose>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<f64> = line
            .split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect::<Result<_, _>>()?;
        if fields.len() < 8 {
            return Err(format!("malformed tum line: {line}").into());
        }
        let translation = Vector3::new(fields[1], fields[2], fields[3]);
        let rotation =
            UnitQuaternion::from_quaternion(Quaternion::new(fields[7], fields[4], fields[5], fields[6]));
        let imu_to_world = SE3::new(rotation, translation);
        rows.push(Pose {
            world_to_camera: imu_to_world.inverse(),
        });
    }
    Ok(rows)
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
    visloc_io::images::read_common_image(&path).map_err(|e| format!("{}: {e}", path.display()).into())
}

fn ds_normalize(camera: &DoubleSphereCamera, keypoints: &[Point2<f64>]) -> Vec<Option<Point2<f64>>> {
    keypoints.iter().map(|kp| ds_normalize_one(camera, kp)).collect()
}

fn ds_normalize_one(camera: &DoubleSphereCamera, pixel: &Point2<f64>) -> Option<Point2<f64>> {
    let ray = camera.unproject(pixel)?;
    if !ray.z.is_finite() || ray.z <= 0.05 {
        return None;
    }
    Some(Point2::new(ray.x / ray.z, ray.y / ray.z))
}

/// Rigid spanning-tree propagation, identical in spirit to Stage A's: each
/// trajectory row keeps its VIO-derived relative pose to the nearest
/// preceding keyframe; only the keyframe's BA correction moves it.
fn propagate_corrections(
    trajectory: &[TrajRow],
    keyframe_ids: &[u64],
    corrected_by_frame: &BTreeMap<u64, Pose>,
) -> String {
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
        let Some(corrected_reference) = corrected_by_frame.get(&reference_id) else {
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

// ---- Simple binary per-keyframe cache: keypoints + descriptors + landmarks ----

fn write_cache(path: &Path, cache: &KeyframeCache) -> std::io::Result<()> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&CACHE_MAGIC.to_le_bytes());
    write_keypoints_descriptors(&mut buf, &cache.cam0_keypoints, &cache.cam0_descriptors);
    write_keypoints_descriptors(&mut buf, &cache.cam1_keypoints, &cache.cam1_descriptors);
    buf.extend_from_slice(&(cache.landmarks.len() as u32).to_le_bytes());
    for lm in &cache.landmarks {
        buf.extend_from_slice(&(lm.keypoint_index as u32).to_le_bytes());
        buf.extend_from_slice(&lm.point_cam0.x.to_le_bytes());
        buf.extend_from_slice(&lm.point_cam0.y.to_le_bytes());
        buf.extend_from_slice(&lm.point_cam0.z.to_le_bytes());
    }
    let tmp_path = path.with_extension("kfcache.tmp");
    fs::File::create(&tmp_path)?.write_all(&buf)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

fn write_keypoints_descriptors(buf: &mut Vec<u8>, keypoints: &[Point2<f64>], descriptors: &[Vec<f32>]) {
    buf.extend_from_slice(&(keypoints.len() as u32).to_le_bytes());
    let dim = descriptors.first().map(|d| d.len()).unwrap_or(0);
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    for kp in keypoints {
        buf.extend_from_slice(&(kp.x as f32).to_le_bytes());
        buf.extend_from_slice(&(kp.y as f32).to_le_bytes());
    }
    for d in descriptors {
        for &v in d {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn read_cache(path: &Path) -> Result<Option<KeyframeCache>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?.read_to_end(&mut bytes)?;
    let mut cursor = 0usize;
    let magic = read_u32(&bytes, &mut cursor)?;
    if magic != CACHE_MAGIC {
        return Ok(None);
    }
    let (cam0_keypoints, cam0_descriptors) = read_keypoints_descriptors(&bytes, &mut cursor)?;
    let (cam1_keypoints, cam1_descriptors) = read_keypoints_descriptors(&bytes, &mut cursor)?;
    let landmark_count = read_u32(&bytes, &mut cursor)? as usize;
    let mut landmarks = Vec::with_capacity(landmark_count);
    for _ in 0..landmark_count {
        let keypoint_index = read_u32(&bytes, &mut cursor)? as usize;
        let x = read_f64(&bytes, &mut cursor)?;
        let y = read_f64(&bytes, &mut cursor)?;
        let z = read_f64(&bytes, &mut cursor)?;
        landmarks.push(StereoLandmark {
            keypoint_index,
            point_cam0: Point3::new(x, y, z),
        });
    }
    Ok(Some(KeyframeCache {
        cam0_keypoints,
        cam0_descriptors,
        cam1_keypoints,
        cam1_descriptors,
        landmarks,
    }))
}

fn read_keypoints_descriptors(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<(Vec<Point2<f64>>, Vec<Vec<f32>>), Box<dyn std::error::Error>> {
    let count = read_u32(bytes, cursor)? as usize;
    let dim = read_u32(bytes, cursor)? as usize;
    let mut keypoints = Vec::with_capacity(count);
    for _ in 0..count {
        let x = read_f32(bytes, cursor)? as f64;
        let y = read_f32(bytes, cursor)? as f64;
        keypoints.push(Point2::new(x, y));
    }
    let mut descriptors = Vec::with_capacity(count);
    for _ in 0..count {
        let mut d = Vec::with_capacity(dim);
        for _ in 0..dim {
            d.push(read_f32(bytes, cursor)?);
        }
        descriptors.push(d);
    }
    Ok((keypoints, descriptors))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, Box<dyn std::error::Error>> {
    let end = *cursor + 4;
    let value = u32::from_le_bytes(bytes.get(*cursor..end).ok_or("cache truncated")?.try_into()?);
    *cursor = end;
    Ok(value)
}

fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, Box<dyn std::error::Error>> {
    let end = *cursor + 4;
    let value = f32::from_le_bytes(bytes.get(*cursor..end).ok_or("cache truncated")?.try_into()?);
    *cursor = end;
    Ok(value)
}

fn read_f64(bytes: &[u8], cursor: &mut usize) -> Result<f64, Box<dyn std::error::Error>> {
    let end = *cursor + 8;
    let value = f64::from_le_bytes(bytes.get(*cursor..end).ok_or("cache truncated")?.try_into()?);
    *cursor = end;
    Ok(value)
}
