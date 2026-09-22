use crate::{
    covisibility::CovisibilityGraph,
    models::{jist_image_tensor, Features, Models, SEQUENCE},
    *,
};
use nalgebra::Matrix6;
use std::{
    collections::{BTreeSet, VecDeque},
    fs::File,
    io::{BufWriter, Write},
    sync::{
        mpsc::{self, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};
use visloc_core::{
    geometry::{reproject, Pose},
    types::Camera,
};
use visloc_slam::{LinearSolver, PoseGraph, PoseGraphEdgeKind, PoseGraphSe3Config, RobustKernel};
use visloc_vision::{
    pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert},
    ransac::{PnPRansac, RobustPoseEstimator},
};

pub struct OnlineLoop {
    tx: SyncSender<Frame>,
    shared: Arc<Mutex<Snapshot>>,
    handle: thread::JoinHandle<Result<Snapshot>>,
    dropped: usize,
}
impl OnlineLoop {
    pub fn start(
        config: Config,
        camera: Camera,
        camera_to_body: SE3,
        output: PathBuf,
    ) -> Result<Self> {
        config.validate()?;
        let (tx, rx) = mpsc::sync_channel(config.queue_capacity);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let shared = Arc::new(Mutex::new(Snapshot::default()));
        let publish = shared.clone();
        let handle = thread::Builder::new()
            .name("jist-xfeat-lighterglue".into())
            .spawn(move || {
                let mut worker = match Worker::new(config, camera, camera_to_body, output) {
                    Ok(w) => {
                        let _ = ready_tx.send(Ok(()));
                        w
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.clone()));
                        return Err(e);
                    }
                };
                for frame in rx {
                    if let Err(e) = worker.process(frame) {
                        publish.lock().unwrap().error = Some(e.to_string());
                        return Err(e);
                    }
                    *publish.lock().unwrap() = worker.snapshot.clone();
                }
                Ok(worker.snapshot)
            })
            .map_err(|e| Error(e.to_string()))?;
        ready_rx.recv().map_err(|e| Error(e.to_string()))??;
        Ok(Self {
            tx,
            shared,
            handle,
            dropped: 0,
        })
    }
    /// Nonblocking. Call only when Basalt's EstimatorOutput::is_keyframe is true.
    pub fn submit(&mut self, frame: Frame) -> Result<()> {
        frame.validate()?;
        match self.tx.try_send(frame) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => {
                self.dropped += 1;
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => Err(Error("Loop worker disconnected".into())),
        }
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        let mut s = self.shared.lock().unwrap().clone();
        s.dropped_keyframes = self.dropped;
        if let Some(e) = &s.error {
            return Err(Error(e.clone()));
        }
        Ok(s)
    }
    /// Drain pending keyframes after the sensor stream ends.
    pub fn finish(self) -> Result<Snapshot> {
        drop(self.tx);
        let mut result = self
            .handle
            .join()
            .map_err(|_| Error("Loop worker panicked".into()))??;
        result.dropped_keyframes = self.dropped;
        Ok(result)
    }
}

struct Keyframe {
    id: u64,
    timestamp_ns: i64,
    raw_body_to_world: SE3,
    features: Features,
    points_camera: Vec<Option<Point3<f64>>>,
    descriptor: Vec<f32>,
}
struct SequenceEntry {
    indices: Vec<usize>,
    descriptor: Vec<f32>,
}
struct Verified {
    measurement: SE3,
    inliers: usize,
    error: f64,
    coverage: f64,
}
struct Pending {
    query: usize,
    candidate: usize,
    correction: SE3,
}
struct Worker {
    config: Config,
    camera: Camera,
    camera_to_body: SE3,
    models: Models,
    keyframes: Vec<Keyframe>,
    sequences: Vec<SequenceEntry>,
    images: VecDeque<Vec<f32>>,
    sequence_indices: VecDeque<usize>,
    last_keyframe_index: Option<u64>,
    graph: PoseGraph,
    covisibility: CovisibilityGraph,
    pending: Option<Pending>,
    last_loop_ns: Option<i64>,
    snapshot: Snapshot,
    events: BufWriter<File>,
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn pose(transform: SE3) -> Pose {
    Pose {
        world_to_camera: rigid_pose(transform),
    }
}
// Basalt's f32 unit quaternion can lose unit length when cast to f64. Restore
// the invariant on the loop worker's copy before any SE(3) inverse/composition.
fn rigid_pose(mut transform: SE3) -> SE3 {
    transform.rotation.renormalize();
    transform
}

fn graph_correction(raw: &SE3, optimized: &Pose) -> SE3 {
    rigid_pose(optimized.camera_to_world().compose(&raw.inverse()))
}
fn body_constraint(camera_to_body: &SE3, newer_camera_from_older_camera: &SE3) -> SE3 {
    camera_to_body
        .compose(newer_camera_from_older_camera)
        .compose(&camera_to_body.inverse())
}

impl Worker {
    fn new(config: Config, camera: Camera, camera_to_body: SE3, output: PathBuf) -> Result<Self> {
        let models = Models::new(&config)?;
        let events = BufWriter::new(
            File::options()
                .create_new(true)
                .write(true)
                .open(output.join("loop_events.jsonl"))
                .map_err(|e| Error(e.to_string()))?,
        );
        std::fs::write(
            output.join("loop_config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .map_err(|e| Error(e.to_string()))?;
        Ok(Self {
            config,
            camera,
            camera_to_body: rigid_pose(camera_to_body),
            models,
            keyframes: Vec::new(),
            sequences: Vec::new(),
            images: VecDeque::new(),
            sequence_indices: VecDeque::new(),
            last_keyframe_index: None,
            graph: PoseGraph::new(),
            covisibility: CovisibilityGraph::default(),
            pending: None,
            last_loop_ns: None,
            snapshot: Snapshot::default(),
            events,
        })
    }
    fn log(&mut self, event: serde_json::Value) -> Result<()> {
        writeln!(self.events, "{event}")
            .and_then(|_| self.events.flush())
            .map_err(|e| Error(e.to_string()))
    }
    fn process(&mut self, mut frame: Frame) -> Result<()> {
        let start = Instant::now();
        frame.body_to_world = rigid_pose(frame.body_to_world);
        if self.keyframes.len() >= self.config.max_keyframes {
            self.snapshot.capacity_skips += 1;
            return Ok(());
        }
        if let Some(last) = self.keyframes.last() {
            if frame.id <= last.id || frame.timestamp_ns <= last.timestamp_ns {
                return Err(Error(
                    "Keyframes must be submitted in strictly increasing time order".into(),
                ));
            }
        }
        if self
            .last_keyframe_index
            .is_some_and(|last| last + 1 != frame.keyframe_index)
        {
            self.images.clear();
            self.sequence_indices.clear();
            self.pending = None;
            self.log(serde_json::json!({"event":"sequence_reset_after_dropped_keyframe", "frame_id":frame.id}))?;
        }
        self.last_keyframe_index = Some(frame.keyframe_index);
        let index = self.keyframes.len();
        // Use the complete VIO landmark observations, before the fixed-size
        // XFeat selection, so matching budgets cannot erase local connectivity.
        let covisibility_edges = self.covisibility.insert(
            frame.keyframe_index,
            frame
                .observations
                .iter()
                .filter(|o| o.point_world.is_some())
                .map(|o| o.track_id),
            self.config.covisibility_min_shared,
        );
        // Retain measured positions and their exact landmark association. A
        // round-robin spatial selection prevents truncation from clustering
        // the fixed-size matcher input around old track IDs in one image area.
        frame.observations = select_observations(
            frame.observations,
            frame.width,
            frame.height,
            self.config.matcher_keypoints,
        );
        let features = self.models.features(&frame)?;
        let camera_from_world = frame.body_to_world.compose(&self.camera_to_body).inverse();
        let points_camera = frame
            .observations
            .iter()
            .map(|o| {
                o.point_world.and_then(|p| {
                    let point = camera_from_world.transform_point(&p);
                    let projection = reproject(&self.camera, &Pose::identity(), &point)?;
                    (point.z > 0.1 && point.z < 150.0 && (projection - o.pixel).norm() < 3.0)
                        .then_some(point)
                })
            })
            .collect::<Vec<_>>();
        self.log(serde_json::json!({"event":"keyframe", "frame_id":frame.id,
            "keyframe_index":frame.keyframe_index,"timestamp_ns":frame.timestamp_ns,
            "covisibility_edges":covisibility_edges.iter().map(|(&i,&count)|
                (self.keyframes[i].id, count)).collect::<Vec<_>>(),
            "features":features.pixels.len(),"metric_points":points_camera.iter().flatten().count()}))?;
        let correction = self.snapshot.correction_at(frame.timestamp_ns);
        self.graph.add_pose(
            frame.id,
            pose(correction.compose(&frame.body_to_world).inverse()),
        );
        if let Some(previous) = self.keyframes.last() {
            // Odometry measurements always use the uncorrected VIO poses.
            let measurement = frame
                .body_to_world
                .inverse()
                .compose(&previous.raw_body_to_world);
            self.graph.add_edge_with_information(
                previous.id,
                frame.id,
                measurement,
                PoseGraphEdgeKind::Sequential,
                information(0.15, 0.03),
            );
        } else {
            self.graph.anchor(frame.id);
        }
        self.images.push_back(jist_image_tensor(&frame));
        self.sequence_indices.push_back(index);
        self.keyframes.push(Keyframe {
            id: frame.id,
            timestamp_ns: frame.timestamp_ns,
            raw_body_to_world: frame.body_to_world,
            features,
            points_camera,
            descriptor: Vec::new(),
        });
        if self.images.len() > SEQUENCE {
            self.images.pop_front();
            self.sequence_indices.pop_front();
        }
        if self.images.len() == SEQUENCE {
            let (descriptor, frame_descriptors) =
                self.models.sequence(self.images.make_contiguous())?;
            for (&i, descriptor) in self.sequence_indices.iter().zip(frame_descriptors) {
                self.keyframes[i].descriptor = descriptor;
            }
            // Retrieve before inserting the query: never self-match or look ahead.
            self.retrieve(index, &descriptor)?;
            self.sequences.push(SequenceEntry {
                indices: self.sequence_indices.iter().copied().collect(),
                descriptor,
            });
        }
        self.update_snapshot();
        self.snapshot.worker_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    fn retrieve(&mut self, current: usize, descriptor: &[f32]) -> Result<()> {
        let t = self.keyframes[current].timestamp_ns;
        if self
            .last_loop_ns
            .is_some_and(|last| (t - last) as f64 * 1e-9 < self.config.loop_cooldown_s)
        {
            return Ok(());
        }
        let query_start = self.keyframes[*self.sequence_indices.front().unwrap()].timestamp_ns;
        let query: Vec<_> = self.sequence_indices.iter().copied().collect();
        let local = self
            .covisibility
            .neighborhood(&query, self.config.covisibility_hops);
        let mut ranked = Vec::new();
        let mut excluded_local = Vec::new();
        let mut excluded_time = 0;
        let mut compared_sequences = 0;
        self.snapshot.sequence_queries += 1;
        for (i, sequence) in self.sequences.iter().enumerate() {
            // Establish the search pool before computing similarity or taking
            // top-k. A local high score must never displace a valid remote one.
            if let Some(reason) = self
                .covisibility
                .local_reason(&query, &sequence.indices, &local)
            {
                self.snapshot.local_sequences_skipped += 1;
                excluded_local.push((self.keyframes[*sequence.indices.last().unwrap()].id, reason));
                continue;
            }
            let end = self.keyframes[*sequence.indices.last().unwrap()].timestamp_ns;
            if (query_start - end) as f64 * 1e-9 < self.config.temporal_exclusion_s {
                excluded_time += 1;
                continue;
            }
            compared_sequences += 1;
            let similarity = dot(descriptor, &sequence.descriptor);
            if similarity < self.config.min_similarity {
                continue;
            }
            ranked.push((i, similarity));
        }
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.log(
            serde_json::json!({"event":"retrieval","query":self.keyframes[current].id,
            "min_similarity":self.config.min_similarity,"eligible_sequences":ranked.len(),
            "compared_sequences":compared_sequences,
            "local_exclusions":excluded_local,"temporal_exclusions":excluded_time}),
        )?;
        let mut tested = Vec::new();
        let mut tested_groups: Vec<(Vec<usize>, BTreeSet<usize>)> = Vec::new();
        for (sequence, similarity) in ranked {
            let members = &self.sequences[sequence].indices;
            // Neighboring database windows describe the same local place.
            // Spend top-k on distinct covisibility neighborhoods per query.
            if tested_groups.iter().any(|(group, local)| {
                self.covisibility
                    .local_reason(group, members, local)
                    .is_some()
            }) {
                self.snapshot.redundant_sequences_skipped += 1;
                continue;
            }
            // JIST's per-frame output selects the geometrically useful member
            // of the retrieved five-keyframe sequence for the newest query KF.
            let candidate = *self.sequences[sequence]
                .indices
                .iter()
                .max_by(|&&a, &&b| {
                    dot(
                        &self.keyframes[a].descriptor,
                        &self.keyframes[current].descriptor,
                    )
                    .total_cmp(&dot(
                        &self.keyframes[b].descriptor,
                        &self.keyframes[current].descriptor,
                    ))
                })
                .unwrap();
            if tested.contains(&candidate) {
                continue;
            }
            if tested.is_empty() {
                self.snapshot.candidate_queries += 1;
            }
            tested_groups.push((
                members.clone(),
                self.covisibility
                    .neighborhood(members, self.config.covisibility_hops),
            ));
            tested.push(candidate);
            self.snapshot.candidates += 1;
            let matches = self.models.matches(
                &self.keyframes[candidate].features,
                &self.keyframes[current].features,
            )?;
            let (verified, geometry) = verify_geometry(
                &self.config,
                &self.camera,
                &self.camera_to_body,
                &self.keyframes[candidate],
                &self.keyframes[current],
                &matches,
            );
            let mut event = serde_json::json!({"event":"candidate","query":self.keyframes[current].id,
                "candidate":self.keyframes[candidate].id,"similarity":similarity,"matches":matches.len(),
                "query_sequence":self.sequence_indices.iter().map(|&i|self.keyframes[i].id).collect::<Vec<_>>(),
                "candidate_sequence":self.sequences[sequence].indices.iter().map(|&i|self.keyframes[i].id).collect::<Vec<_>>(),
                "geometry":geometry,"accepted":false,"reason":"geometry_rejected"});
            if let Some(v) = verified {
                event["inliers"] = v.inliers.into();
                event["reprojection_px"] = v.error.into();
                event["coverage"] = v.coverage.into();
                let old = &self.keyframes[candidate];
                let new = &self.keyframes[current];
                let corrected = self.graph.poses[&old.id]
                    .camera_to_world()
                    .compose(&v.measurement.inverse());
                let correction = corrected.compose(&new.raw_body_to_world.inverse());
                let consistent = self.pending.as_ref().is_some_and(|p| {
                    let query_dt =
                        (new.timestamp_ns - self.keyframes[p.query].timestamp_ns) as f64 * 1e-9;
                    let candidate_dt = (old.timestamp_ns - self.keyframes[p.candidate].timestamp_ns)
                        .abs() as f64
                        * 1e-9;
                    let delta = correction.compose(&p.correction.inverse());
                    query_dt > 0.0
                        && query_dt <= self.config.confirmation_window_s
                        && candidate_dt <= self.config.confirmation_window_s * 2.0
                        && delta.translation.norm() < 1.0
                        && delta.rotation.angle() < 0.1
                });
                if consistent {
                    // Optimize a copy and publish only a finite, non-increasing solution.
                    let mut trial = self.graph.clone();
                    trial.add_edge_with_information(
                        old.id,
                        new.id,
                        v.measurement,
                        PoseGraphEdgeKind::LoopClosure,
                        information(0.2, 0.04),
                    );
                    let optimized = trial
                        .optimize_se3_iterative(&PoseGraphSe3Config {
                            max_iterations: 20,
                            initial_lambda: Some(1e-4),
                            chordal_init: false,
                            robust_kernel: RobustKernel::Huber { delta: 3.0 },
                            linear_solver: LinearSolver::Sparse,
                            ..Default::default()
                        })
                        .map_err(|e| Error(format!("Pose graph: {e:?}")))?;
                    if optimized.final_cost.is_finite()
                        && optimized.final_cost <= optimized.initial_cost
                        && trial
                            .poses
                            .values()
                            .all(|p| p.matrix().iter().all(|v| v.is_finite()))
                    {
                        self.graph = trial;
                        self.snapshot.edges.push((old.id, new.id));
                        self.snapshot.accepted_loops += 1;
                        self.snapshot.revision += 1;
                        self.snapshot.corrections = self
                            .keyframes
                            .iter()
                            .map(|k| {
                                (
                                    k.timestamp_ns,
                                    graph_correction(
                                        &k.raw_body_to_world,
                                        &self.graph.poses[&k.id],
                                    ),
                                )
                            })
                            .collect();
                        self.last_loop_ns = Some(t);
                        self.pending = None;
                        event["accepted"] = true.into();
                        event["reason"] = "verified_and_confirmed".into();
                        event["graph_cost_before"] = optimized.initial_cost.into();
                        event["graph_cost_after"] = optimized.final_cost.into();
                    } else {
                        event["reason"] = "optimizer_rejected".into();
                    }
                } else {
                    self.pending = Some(Pending {
                        query: current,
                        candidate,
                        correction,
                    });
                    event["reason"] = "awaiting_temporal_confirmation".into();
                }
                self.log(event)?;
                // One verified candidate per query; don't replace its pending
                // confirmation with another candidate from the same query.
                return Ok(());
            }
            self.log(event)?;
            if tested.len() >= self.config.retrieval_top_k {
                break;
            }
        }
        Ok(())
    }
    fn update_snapshot(&mut self) {
        self.snapshot.keyframes = self.keyframes.len();
        // Only optimization changes corrections. Propagate the last correction
        // exactly between loops; repeatedly deriving it through inverse pose
        // round trips feeds numerical error back into subsequent graph poses.
        if self.snapshot.corrections.len() < self.keyframes.len() {
            let timestamp_ns = self.keyframes.last().unwrap().timestamp_ns;
            let correction = self.snapshot.correction_at(timestamp_ns);
            self.snapshot.corrections.push((timestamp_ns, correction));
        }
        self.snapshot.keyframe_positions = self
            .graph
            .poses
            .iter()
            .map(|(&id, p)| {
                let t = p.camera_to_world().translation;
                (id, [t.x, t.y, t.z])
            })
            .collect();
    }
}
fn verify_geometry(
    config: &Config,
    camera: &Camera,
    camera_to_body: &SE3,
    a: &Keyframe,
    b: &Keyframe,
    matches: &[(usize, usize, f32)],
) -> (Option<Verified>, serde_json::Value) {
    let mut diagnostics = serde_json::json!({"candidate_landmarks":{},"query_landmarks":{}});
    let forward = verify_direction(
        config,
        camera,
        camera_to_body,
        a,
        b,
        matches,
        &mut diagnostics["candidate_landmarks"],
    );
    if forward.is_some() {
        diagnostics["direction"] = "candidate_to_query".into();
        return (forward, diagnostics);
    }
    // Either keyframe may have more mature metric landmarks. Reversing PnP
    // uses the same observed matches, and preserves the graph's edge direction.
    let reverse_matches: Vec<_> = matches.iter().map(|&(i, j, score)| (j, i, score)).collect();
    let reverse = verify_direction(
        config,
        camera,
        camera_to_body,
        b,
        a,
        &reverse_matches,
        &mut diagnostics["query_landmarks"],
    )
    .map(|mut v| {
        v.measurement = v.measurement.inverse();
        v
    });
    if reverse.is_some() {
        diagnostics["direction"] = "query_to_candidate".into();
    }
    (reverse, diagnostics)
}

fn verify_direction(
    config: &Config,
    camera: &Camera,
    camera_to_body: &SE3,
    a: &Keyframe,
    b: &Keyframe,
    matches: &[(usize, usize, f32)],
    diagnostics: &mut serde_json::Value,
) -> Option<Verified> {
    let correspondences: Vec<_> = matches
        .iter()
        .filter_map(|&(i, j, score)| {
            a.points_camera
                .get(i)
                .copied()
                .flatten()
                .map(|point3d| Correspondence2D3D {
                    point2d: Point2::new(
                        b.features.pixels[j][0] as f64,
                        b.features.pixels[j][1] as f64,
                    ),
                    point3d,
                    confidence: Some(score),
                })
        })
        .collect();
    diagnostics["metric_matches"] = correspondences.len().into();
    if correspondences.len() < config.min_matches {
        diagnostics["reason"] = "insufficient_metric_matches".into();
        return None;
    }
    let weights: Vec<_> = correspondences
        .iter()
        .map(|c| c.confidence.unwrap())
        .collect();
    let estimator = PnPRansac {
        pose_estimator: P3PGrunert,
        pose_refiner: Some(GaussNewtonPoseRefiner::default()),
        iterations: 1000,
        reprojection_threshold: config.reprojection_threshold_px,
        seed: b.id.wrapping_mul(1337).wrapping_add(a.id),
        early_stop_min_iterations: 40,
        early_stop_inlier_ratio: Some(0.85),
        confidence: Some(0.999),
    };
    let Some(report) = estimator.estimate_with_weights(&correspondences, camera, &weights) else {
        diagnostics["reason"] = "pnp_failed".into();
        return None;
    };
    diagnostics["inliers"] = report.inliers.len().into();
    diagnostics["mean_reprojection_px"] = report.mean_reprojection_error.into();
    if report.inliers.len() < config.min_inliers
        || (report.inliers.len() as f64 / correspondences.len() as f64) < config.min_inlier_ratio
        || report.mean_reprojection_error > config.reprojection_threshold_px * 0.75
    {
        diagnostics["reason"] = "insufficient_geometric_support".into();
        return None;
    }
    let mut low = Point2::new(f64::INFINITY, f64::INFINITY);
    let mut high = Point2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut cells = [false; 16];
    for &i in &report.inliers {
        let p = correspondences[i].point2d;
        low.x = low.x.min(p.x);
        low.y = low.y.min(p.y);
        high.x = high.x.max(p.x);
        high.y = high.y.max(p.y);
        let x = (p.x / camera.width as f64 * 4.0).clamp(0.0, 3.0) as usize;
        let y = (p.y / camera.height as f64 * 4.0).clamp(0.0, 3.0) as usize;
        cells[y * 4 + x] = true;
    }
    let coverage =
        (high.x - low.x) * (high.y - low.y) / (camera.width as f64 * camera.height as f64);
    diagnostics["coverage"] = coverage.into();
    if coverage < config.min_image_coverage || cells.iter().filter(|&&x| x).count() < 4 {
        diagnostics["reason"] = "insufficient_spatial_coverage".into();
        return None;
    }
    diagnostics["reason"] = "verified".into();
    Some(Verified {
        measurement: body_constraint(camera_to_body, &report.pose.world_to_camera),
        inliers: report.inliers.len(),
        error: report.mean_reprojection_error,
        coverage,
    })
}

fn information(translation_sigma: f64, rotation_sigma: f64) -> Matrix6<f64> {
    let mut m = Matrix6::zeros();
    for i in 0..6 {
        m[(i, i)] = 1.0
            / if i < 3 {
                translation_sigma.powi(2)
            } else {
                rotation_sigma.powi(2)
            };
    }
    m
}

fn select_observations(
    observations: Vec<Observation>,
    width: usize,
    height: usize,
    count: usize,
) -> Vec<Observation> {
    let mut bins: Vec<VecDeque<Observation>> = (0..16).map(|_| VecDeque::new()).collect();
    for o in observations {
        let x = (o.pixel.x / width as f64 * 4.0).clamp(0.0, 3.0) as usize;
        let y = (o.pixel.y / height as f64 * 4.0).clamp(0.0, 3.0) as usize;
        if o.point_world.is_some() {
            bins[y * 4 + x].push_front(o);
        } else {
            bins[y * 4 + x].push_back(o);
        }
    }
    let mut selected = Vec::new();
    while selected.len() < count {
        let before = selected.len();
        for bin in &mut bins {
            if selected.len() == count {
                break;
            }
            if let Some(o) = bin.pop_front() {
                selected.push(o);
            }
        }
        if selected.len() == before {
            break;
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    #[test]
    fn camera_constraint_includes_lever_arm() {
        let extrinsic = SE3::new(
            UnitQuaternion::from_euler_angles(0.2, 0.1, 0.4),
            Vector3::new(0.4, 0.1, 0.2),
        );
        let older = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.2, 0.0),
            Vector3::new(1.0, 2.0, 3.0),
        );
        let newer = SE3::new(
            UnitQuaternion::from_euler_angles(0.1, 0.3, 0.1),
            Vector3::new(3.0, 1.0, 4.0),
        );
        let relative_camera = newer
            .compose(&extrinsic)
            .inverse()
            .compose(&older.compose(&extrinsic));
        let relative_body = body_constraint(&extrinsic, &relative_camera);
        assert!((relative_body.matrix() - newer.inverse().compose(&older).matrix()).norm() < 1e-10);
    }
    #[test]
    fn metric_pnp_recovers_loop_with_outliers_and_rejects_wrong_place() {
        let config = Config::default();
        let camera = Camera::pinhole(0, 800, 550, 450.0, 450.0, 400.0, 275.0);
        let relative = SE3::new(
            UnitQuaternion::from_euler_angles(0.02, -0.04, 0.03),
            Vector3::new(0.3, -0.1, 0.2),
        );
        let points: Vec<_> = (0..80)
            .map(|i| {
                Point3::new(
                    (i % 10) as f64 * 0.5 - 2.25,
                    (i / 10) as f64 * 0.45 - 1.6,
                    4.0 + ((i * 17) % 23) as f64 * 0.13,
                )
            })
            .collect();
        let features = |transform: &SE3| Features {
            pixels: points
                .iter()
                .map(|p| {
                    let q = reproject(&camera, &pose(transform.clone()), p).unwrap();
                    [q.x as f32, q.y as f32]
                })
                .collect(),
            descriptors: Vec::new(),
            image_size: [800.0, 550.0],
        };
        let a = Keyframe {
            id: 0,
            timestamp_ns: 0,
            raw_body_to_world: SE3::identity(),
            features: features(&SE3::identity()),
            points_camera: points.iter().copied().map(Some).collect(),
            descriptor: Vec::new(),
        };
        let b = Keyframe {
            id: 500,
            timestamp_ns: 30_000_000_000,
            raw_body_to_world: relative.inverse(),
            features: features(&relative),
            points_camera: Vec::new(),
            descriptor: Vec::new(),
        };
        let matches: Vec<_> = (0..80)
            .map(|i| (i, if i % 4 == 0 { (i + 19) % 80 } else { i }, 0.9))
            .collect();
        let verified = verify_geometry(&config, &camera, &SE3::identity(), &a, &b, &matches)
            .0
            .unwrap();
        assert!(verified.inliers >= 60);
        assert!((verified.measurement.matrix() - relative.matrix()).norm() < 1e-5);
        let wrong: Vec<_> = (0..80).map(|i| (i, (i * 31 + 7) % 80, 0.9)).collect();
        assert!(
            verify_geometry(&config, &camera, &SE3::identity(), &a, &b, &wrong)
                .0
                .is_none()
        );
        let too_few = &matches[..5];
        assert!(
            verify_geometry(&config, &camera, &SE3::identity(), &a, &b, too_few)
                .0
                .is_none()
        );
        let reverse_matches: Vec<_> = matches.iter().map(|&(i, j, s)| (j, i, s)).collect();
        let (reverse, diagnostics) =
            verify_geometry(&config, &camera, &SE3::identity(), &b, &a, &reverse_matches);
        assert_eq!(diagnostics["direction"], "query_to_candidate");
        assert!(
            (reverse.unwrap().measurement.matrix() - relative.inverse().matrix()).norm() < 1e-5
        );
    }

    #[test]
    fn float_vio_rotations_do_not_accumulate_map_drift() {
        let mut correction = SE3::identity();
        for i in 0..1000 {
            let rotation32 = UnitQuaternion::<f32>::from_euler_angles(0.1, 0.2, i as f32 * 0.01);
            let rotation64 = UnitQuaternion::new_unchecked(rotation32.into_inner().cast::<f64>());
            let raw = rigid_pose(SE3::new(
                rotation64,
                Vector3::new(i as f64 * 0.1, 15.0, 1.0),
            ));
            let optimized = pose(correction.compose(&raw).inverse());
            correction = graph_correction(&raw, &optimized);
        }
        assert!(
            correction.translation.norm() < 1e-8,
            "{:?}",
            correction.translation
        );
        assert!(correction.rotation.angle() < 1e-10);
        assert!((correction.rotation.norm() - 1.0).abs() < 1e-14);
    }
}
