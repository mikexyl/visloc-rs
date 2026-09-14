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
//! Structure building (v2 -- proper track fusion; see git history for the v1
//! "one landmark per keyframe's own stereo triangulation, no fusion" design,
//! which cut mean reprojection error 39% but did not beat Stage A on ATE):
//!   - Match graph: stereo (cam0<->cam1, same keyframe), temporal
//!     (cam0<->cam0, k -> k+1..k+`temporal_window`), covisibility (cam0<->cam0,
//!     the same VIO-proximity rule Stage A uses for loop candidates but
//!     *without* a temporal-gap requirement, ranked by appearance, top-k per
//!     query, and skipping pairs already covered by the temporal window --
//!     the ORB-SLAM3 covisibility-graph analogue, bounded so total LightGlue
//!     calls stay a small multiple of the keyframe count), and loop pairs
//!     (Stage A's accepted loops, cam0<->cam0).
//!   - Every matched keypoint pair is a union in a union-find over
//!     `(keyframe_id, camera_id, keypoint_index)` nodes; each resulting
//!     connected component is one candidate landmark ("track"). A track
//!     containing two *different* keypoint indices from the same
//!     `(keyframe_id, camera_id)` image is internally inconsistent (one
//!     camera image cannot observe the same 3D point at two pixels) and is
//!     dropped outright (the simpler of the two standard policies; splitting
//!     is not implemented).
//!   - Surviving tracks with fewer than `--min-track-keyframes` distinct
//!     keyframes are dropped. Each remaining track is re-triangulated by
//!     multi-view linear least squares (every observation's known Stage-A
//!     camera pose, both cams, DS-normalized-plane coordinates), then
//!     dropped again if its maximum inter-keyframe parallax is below
//!     `--min-parallax-deg` or its mean triangulation reprojection error
//!     exceeds `--max-triangulation-reproj-error` (both a priori thresholds,
//!     see the `Args` doc comments).
//!   - A keyframe contributing both a cam0 and a cam1 observation to a track
//!     (i.e. it was that track's stereo origin, or a redundant stereo hit
//!     merged in) becomes one `BaGeneralStereoObservation`; every other
//!     keyframe in the track becomes one monocular `BaObservation`.
//!     `BaGeneralStereoObservation` uses the same "DS-unproject to a
//!     normalized plane, feed an identity pinhole `Camera`" trick Stage A
//!     uses for PnP/triangulation (EuRoC's cam0/cam1 extrinsic is
//!     rotational, not a clean rectified baseline).
//!   - VIO odometry between consecutive keyframes is optionally kept as a
//!     `PairwisePoseFactor` weak prior (`--odometry-prior-weight`, 0
//!     disables it), converted from Stage A's body-frame relative pose into
//!     the BA's cam0-frame poses.
//!
//! Ground truth is never read here. Robust-kernel / threshold defaults are
//! chosen a priori (see the `Args` doc comments for the stated reasoning);
//! the one exception, disclosed in the commit message, is the odometry-prior
//! weight, swept once against ATE on MH_01 in the v1 design and carried over.

use std::{
    collections::{BTreeMap, HashMap},
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use nalgebra::{Matrix3, Point2, Point3, Quaternion, UnitQuaternion, Vector3};

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
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

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
    /// Proximity-based covisibility candidates ranked by appearance, kept
    /// per query keyframe (mirrors Stage A's `appearance_rank_top_k`, but
    /// with no temporal-gap requirement -- this is what widens the match
    /// graph beyond the `temporal_window` chain, the ORB-SLAM3
    /// covisibility-graph analogue).
    covisibility_top_k: usize,
    /// Same proximity radius/angle rule as Stage A's loop candidates
    /// (`base_m + drift_frac * path_length`, `max_viewing_angle_deg`).
    proximity_base_m: f64,
    proximity_drift_frac: f64,
    max_viewing_angle_deg: f64,
    /// Minimum distinct keyframes a fused track must retain after the
    /// consistency check to be triangulated at all. A priori: 2 is the
    /// mathematical minimum for triangulation.
    min_track_keyframes: usize,
    /// Minimum max-pairwise inter-keyframe parallax, degrees, for a
    /// triangulated track to be kept. A priori: 1 degree is the standard
    /// SfM near-degenerate-parallax gate (e.g. COLMAP's default is close to
    /// this).
    min_parallax_deg: f64,
    /// Maximum mean reprojection error (DS-normalized-plane units, same
    /// scale as `ba_huber_delta`) for a triangulated track to be kept. A
    /// priori: 2x the BA Huber delta -- a track whose *initial* triangulated
    /// residual is already well outside the robust-kernel's trust region is
    /// more likely a bad match than a hard-but-real point.
    max_triangulation_reproj_error: f64,
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
        let mut odometry_prior_weight = 1000.0f64;
        let mut max_keyframes = None;
        let mut covisibility_top_k = 4usize;
        let mut proximity_base_m = 1.0f64;
        let mut proximity_drift_frac = 0.03f64;
        let mut max_viewing_angle_deg = 45.0f64;
        let mut min_track_keyframes = 2usize;
        let mut min_parallax_deg = 1.0f64;
        let mut max_triangulation_reproj_error = 0.02f64;

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
                "--covisibility-top-k" => {
                    covisibility_top_k = value!().parse().map_err(|e| format!("{e}"))?
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
                "--min-track-keyframes" => {
                    min_track_keyframes = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--min-parallax-deg" => {
                    min_parallax_deg = value!().parse().map_err(|e| format!("{e}"))?
                }
                "--max-triangulation-reproj-error" => {
                    max_triangulation_reproj_error = value!().parse().map_err(|e| format!("{e}"))?
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
            covisibility_top_k,
            proximity_base_m,
            proximity_drift_frac,
            max_viewing_angle_deg,
            min_track_keyframes,
            min_parallax_deg,
            max_triangulation_reproj_error,
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

    // ---- 2. Keyframe cam0 poses (BA init) ----
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

    // ---- 3. Match graph: stereo + temporal + covisibility + loop pairs -> union-find ----
    let matcher = LightGlueOnnxMatcher::load_from_path(&args.lightglue_model)?;
    let mut uf = UnionFind::new();
    let mut node_index: HashMap<(u64, u8, u32), usize> = HashMap::new();
    let mut node_keys: Vec<(u64, u8, u32)> = Vec::new();

    let cumulative_length = cumulative_arc_length(&trajectory);
    let cam0_forward_body = t_cam0_to_imu.rotation.transform_vector(&Vector3::z());
    let mean_descriptor_by_frame: BTreeMap<u64, Vec<f32>> = keyframe_ids
        .iter()
        .map(|&id| (id, mean_l2_normalized_descriptor(&keyframes[&id].cam0_descriptors)))
        .collect();
    let viewing_direction_by_frame: BTreeMap<u64, Vector3<f64>> = keyframe_ids
        .iter()
        .map(|&id| {
            let row = pose_by_frame[&id];
            (id, row.imu_to_world.rotation.transform_vector(&cam0_forward_body))
        })
        .collect();

    let mut stereo_pairs = 0usize;
    for &frame_id in &keyframe_ids {
        let kf = &keyframes[&frame_id];
        let matches = matcher.match_features(
            &kf.cam0_keypoints,
            &kf.cam0_descriptors,
            &kf.cam1_keypoints,
            &kf.cam1_descriptors,
        )?;
        for m in &matches {
            let a = get_node(&mut node_index, &mut node_keys, &mut uf, (frame_id, 0, m.query_index as u32));
            let b = get_node(&mut node_index, &mut node_keys, &mut uf, (frame_id, 1, m.train_index as u32));
            uf.union(a, b);
        }
        stereo_pairs += 1;
    }

    // Covisibility candidates: VIO proximity, no temporal-gap requirement,
    // top-k by appearance, skipping pairs the temporal window already covers.
    let mut covisibility_pairs: Vec<(u64, u64)> = Vec::new();
    for (j, &id_j) in keyframe_ids.iter().enumerate() {
        let mut scored: Vec<(usize, f64)> = Vec::new();
        for (i, &id_i) in keyframe_ids.iter().enumerate().take(j) {
            if j - i <= args.temporal_window {
                continue; // already covered by the temporal chain
            }
            let path_len = (cumulative_length[id_j as usize] - cumulative_length[id_i as usize]).max(0.0);
            let radius = args.proximity_base_m + args.proximity_drift_frac * path_len;
            let distance = (pose_by_frame[&id_j].imu_to_world.translation
                - pose_by_frame[&id_i].imu_to_world.translation)
                .norm();
            if distance > radius {
                continue;
            }
            let cos_angle = viewing_direction_by_frame[&id_i]
                .normalize()
                .dot(&viewing_direction_by_frame[&id_j].normalize())
                .clamp(-1.0, 1.0);
            if cos_angle.acos() / DEG_TO_RAD > args.max_viewing_angle_deg {
                continue;
            }
            let appearance =
                cosine_similarity(&mean_descriptor_by_frame[&id_i], &mean_descriptor_by_frame[&id_j]);
            scored.push((i, appearance as f64));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(args.covisibility_top_k);
        for (i, _) in scored {
            covisibility_pairs.push((keyframe_ids[i], id_j));
        }
    }
    eprintln!(
        "[{}] {} stereo pairs, {} covisibility pairs (top-k={}, no gap requirement)",
        args.sequence,
        stereo_pairs,
        covisibility_pairs.len(),
        args.covisibility_top_k
    );

    // Temporal edges (k -> k+1 .. k+temporal_window), full cam0<->cam0 match.
    let mut cross_matches = 0usize;
    for (i, &frame_id) in keyframe_ids.iter().enumerate() {
        for offset in 1..=args.temporal_window {
            let Some(&target_id) = keyframe_ids.get(i + offset) else {
                continue;
            };
            cross_matches += match_and_union(
                &keyframes[&frame_id],
                frame_id,
                &keyframes[&target_id],
                target_id,
                &matcher,
                &mut node_index,
                &mut node_keys,
                &mut uf,
            )?;
        }
    }
    // Covisibility edges.
    for &(from_id, to_id) in &covisibility_pairs {
        cross_matches += match_and_union(
            &keyframes[&from_id],
            from_id,
            &keyframes[&to_id],
            to_id,
            &matcher,
            &mut node_index,
            &mut node_keys,
            &mut uf,
        )?;
    }
    // Loop-pair edges (Stage A's accepted loops).
    let loops_path = args.stage_a_out_dir.join("loops.json");
    let mut loop_pairs_used = 0usize;
    if let Ok(text) = fs::read_to_string(&loops_path) {
        let payload: serde_json::Value = serde_json::from_str(&text)?;
        for item in payload["accepted_loops"].as_array().cloned().unwrap_or_default() {
            let (Some(from_id), Some(to_id)) =
                (item["from_frame_id"].as_u64(), item["to_frame_id"].as_u64())
            else {
                continue;
            };
            let (Some(kf_a), Some(kf_b)) = (keyframes.get(&from_id), keyframes.get(&to_id)) else {
                continue;
            };
            cross_matches +=
                match_and_union(kf_a, from_id, kf_b, to_id, &matcher, &mut node_index, &mut node_keys, &mut uf)?;
            loop_pairs_used += 1;
        }
    }
    eprintln!(
        "[{}] {} cross-keyframe matches (temporal+covisibility+loop, {} loop pairs)",
        args.sequence, cross_matches, loop_pairs_used
    );

    // ---- 4. Extract tracks, drop inconsistent ones, triangulate, gate ----
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for node in 0..node_keys.len() {
        components.entry(uf.find(node)).or_default().push(node);
    }
    let mut track_count = 0usize;
    let mut dropped_inconsistent = 0usize;
    let mut dropped_too_few_keyframes = 0usize;
    let mut dropped_low_parallax = 0usize;
    let mut dropped_high_reproj = 0usize;
    let mut track_lengths: Vec<usize> = Vec::new();
    let mut stereo_observation_count = 0usize;
    let mut mono_observation_count = 0usize;
    let mut next_landmark_id = 1u64;

    for member_indices in components.values() {
        if member_indices.len() < 2 {
            continue;
        }
        // Consistency: no two different keypoints from the same (kf, cam).
        let mut by_image: HashMap<(u64, u8), u32> = HashMap::new();
        let mut consistent = true;
        for &idx in member_indices {
            let (kf_id, cam, kp) = node_keys[idx];
            match by_image.get(&(kf_id, cam)) {
                Some(&existing) if existing != kp => {
                    consistent = false;
                    break;
                }
                _ => {
                    by_image.insert((kf_id, cam), kp);
                }
            }
        }
        if !consistent {
            dropped_inconsistent += 1;
            continue;
        }
        let distinct_keyframes: std::collections::BTreeSet<u64> =
            by_image.keys().map(|(kf_id, _)| *kf_id).collect();
        if distinct_keyframes.len() < args.min_track_keyframes {
            dropped_too_few_keyframes += 1;
            continue;
        }

        // Multi-view triangulation.
        let mut rays: Vec<(SE3, Point2<f64>)> = Vec::new();
        let mut per_kf: BTreeMap<u64, (Option<Point2<f64>>, Option<Point2<f64>>)> = BTreeMap::new();
        for (&(kf_id, cam), &kp) in &by_image {
            let camera_pose = if cam == 0 {
                cam0_pose_by_frame[&kf_id].world_to_camera.clone()
            } else {
                t_cam0_to_cam1.compose(&cam0_pose_by_frame[&kf_id].world_to_camera)
            };
            let pixel = if cam == 0 {
                keyframes[&kf_id].cam0_keypoints[kp as usize]
            } else {
                keyframes[&kf_id].cam1_keypoints[kp as usize]
            };
            let camera_model = if cam == 0 { &cam0 } else { &cam1 };
            let Some(xy) = ds_normalize_one(camera_model, &pixel) else {
                continue;
            };
            rays.push((camera_pose, xy));
            let entry = per_kf.entry(kf_id).or_insert((None, None));
            if cam == 0 {
                entry.0 = Some(xy);
            } else {
                entry.1 = Some(xy);
            }
        }
        if rays.len() < 2 {
            dropped_too_few_keyframes += 1;
            continue;
        }
        let Some(point_world) = triangulate_multiview(&rays) else {
            dropped_high_reproj += 1;
            continue;
        };

        // Parallax gate: max angle between camera-center-to-point rays
        // across all pairs of observing camera poses.
        let centers: Vec<Point3<f64>> = rays
            .iter()
            .map(|(pose, _)| pose.inverse().translation.into())
            .collect();
        let mut max_parallax_deg = 0.0f64;
        for a in 0..centers.len() {
            for b in (a + 1)..centers.len() {
                let da = (point_world - centers[a]).normalize();
                let db = (point_world - centers[b]).normalize();
                let angle = da.dot(&db).clamp(-1.0, 1.0).acos() / DEG_TO_RAD;
                max_parallax_deg = max_parallax_deg.max(angle);
            }
        }
        if max_parallax_deg < args.min_parallax_deg {
            dropped_low_parallax += 1;
            continue;
        }

        // Reprojection gate on the fresh triangulation (pre-BA).
        let mut reproj_errors = Vec::with_capacity(rays.len());
        for (pose, xy) in &rays {
            let p_cam = pose.transform_point(&point_world);
            if p_cam.z <= 1e-6 {
                reproj_errors.push(f64::INFINITY);
                continue;
            }
            let predicted = Point2::new(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
            reproj_errors.push((predicted - *xy).norm());
        }
        let mean_error = reproj_errors.iter().sum::<f64>() / reproj_errors.len() as f64;
        if !mean_error.is_finite() || mean_error > args.max_triangulation_reproj_error {
            dropped_high_reproj += 1;
            continue;
        }

        let landmark_id = next_landmark_id;
        next_landmark_id += 1;
        ba.add_landmark(landmark_id, point_world);
        track_count += 1;
        track_lengths.push(distinct_keyframes.len());
        for (&kf_id, &(cam0_xy, cam1_xy)) in &per_kf {
            match (cam0_xy, cam1_xy) {
                (Some(xy_left), Some(xy_right)) => {
                    ba.add_general_stereo_observation(BaGeneralStereoObservation {
                        keyframe_id: kf_id,
                        landmark_id,
                        xy_left,
                        xy_right,
                        right_camera: identity_camera.clone(),
                        left_to_right: t_cam0_to_cam1.clone(),
                    });
                    stereo_observation_count += 1;
                }
                (Some(xy), None) => {
                    ba.add_observation(BaObservation {
                        keyframe_id: kf_id,
                        landmark_id,
                        xy,
                    });
                    mono_observation_count += 1;
                }
                (None, Some(_)) | (None, None) => {
                    // cam1-only cannot arise: cam1 nodes only enter a track
                    // via the same-keyframe stereo union with a cam0 node.
                }
            }
        }
    }
    let mean_track_length = if track_lengths.is_empty() {
        0.0
    } else {
        track_lengths.iter().sum::<usize>() as f64 / track_lengths.len() as f64
    };
    eprintln!(
        "[{}] {} tracks (mean length {:.2} KF), dropped: {} inconsistent, {} <{}KF, {} low-parallax, {} high-reproj",
        args.sequence,
        track_count,
        mean_track_length,
        dropped_inconsistent,
        dropped_too_few_keyframes,
        args.min_track_keyframes,
        dropped_low_parallax,
        dropped_high_reproj
    );
    eprintln!(
        "[{}] {} stereo observations, {} monocular observations",
        args.sequence, stereo_observation_count, mono_observation_count
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
        "track_count": track_count,
        "landmark_count": track_count,
        "mean_track_length_keyframes": mean_track_length,
        "stereo_observation_count": stereo_observation_count,
        "monocular_observation_count": mono_observation_count,
        "cross_keyframe_match_count": cross_matches,
        "covisibility_pair_count": covisibility_pairs.len(),
        "loop_pairs_used": loop_pairs_used,
        "dropped_inconsistent_tracks": dropped_inconsistent,
        "dropped_too_few_keyframes": dropped_too_few_keyframes,
        "dropped_low_parallax": dropped_low_parallax,
        "dropped_high_reproj": dropped_high_reproj,
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
            "covisibility_top_k": args.covisibility_top_k,
            "min_track_keyframes": args.min_track_keyframes,
            "min_parallax_deg": args.min_parallax_deg,
            "max_triangulation_reproj_error": args.max_triangulation_reproj_error,
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

/// Simple union-find (disjoint-set) with path compression and union by size.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            size: Vec::new(),
        }
    }

    fn make_set(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.size.push(1);
        id
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
    }
}

fn get_node(
    node_index: &mut HashMap<(u64, u8, u32), usize>,
    node_keys: &mut Vec<(u64, u8, u32)>,
    uf: &mut UnionFind,
    key: (u64, u8, u32),
) -> usize {
    *node_index.entry(key).or_insert_with(|| {
        node_keys.push(key);
        uf.make_set()
    })
}

/// LightGlue-match `from`'s full cam0 set into `to`'s full cam0 set and union
/// every match's two nodes. Returns the match count.
fn match_and_union(
    from: &KeyframeCache,
    from_id: u64,
    to: &KeyframeCache,
    to_id: u64,
    matcher: &LightGlueOnnxMatcher,
    node_index: &mut HashMap<(u64, u8, u32), usize>,
    node_keys: &mut Vec<(u64, u8, u32)>,
    uf: &mut UnionFind,
) -> Result<usize, Box<dyn std::error::Error>> {
    if from.cam0_keypoints.is_empty() || to.cam0_keypoints.is_empty() {
        return Ok(0);
    }
    let matches = matcher.match_features(
        &from.cam0_keypoints,
        &from.cam0_descriptors,
        &to.cam0_keypoints,
        &to.cam0_descriptors,
    )?;
    for m in &matches {
        let a = get_node(node_index, node_keys, uf, (from_id, 0, m.query_index as u32));
        let b = get_node(node_index, node_keys, uf, (to_id, 0, m.train_index as u32));
        uf.union(a, b);
    }
    Ok(matches.len())
}

/// Linear least-squares multi-view triangulation: for each observation
/// `(world_to_camera, normalized_xy)`, the constraint that the camera-frame
/// point is proportional to `(x, y, 1)` gives two linear rows in the unknown
/// world point; stack all observations and solve the 3x3 normal equations.
fn triangulate_multiview(observations: &[(SE3, Point2<f64>)]) -> Option<Point3<f64>> {
    if observations.len() < 2 {
        return None;
    }
    let mut ata = Matrix3::<f64>::zeros();
    let mut atb = Vector3::<f64>::zeros();
    for (pose, xy) in observations {
        let r = pose.rotation.to_rotation_matrix().into_inner();
        let t = pose.translation;
        let row_x = Vector3::new(
            r[(0, 0)] - xy.x * r[(2, 0)],
            r[(0, 1)] - xy.x * r[(2, 1)],
            r[(0, 2)] - xy.x * r[(2, 2)],
        );
        let row_y = Vector3::new(
            r[(1, 0)] - xy.y * r[(2, 0)],
            r[(1, 1)] - xy.y * r[(2, 1)],
            r[(1, 2)] - xy.y * r[(2, 2)],
        );
        let b_x = xy.x * t.z - t.x;
        let b_y = xy.y * t.z - t.y;
        ata += row_x * row_x.transpose() + row_y * row_y.transpose();
        atb += row_x * b_x + row_y * b_y;
    }
    let solved = ata.lu().solve(&atb)?;
    if !solved.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(Point3::new(solved.x, solved.y, solved.z))
}

/// Cumulative VIO arc length in metres, indexed by `frame_id` (dense 0-based
/// row order, same assumption Stage A makes).
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
