//! Global visual BA inputs and conservative, deterministic track admission.
//! All measured camera views are retained; no descriptors or ground truth are
//! needed. Track IDs are scoped to robot/session and never merged by proximity.
use crate::*;
use nalgebra::{Matrix3, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use visloc_core::geometry::SE3;
use visloc_gtsam::ProjectionFactor;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendMode {
    PoseGraph,
    #[default]
    GlobalBundleAdjustment,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BundleConfig {
    pub pixel_sigma: f64,
    pub huber_delta: f64,
    pub initial_reprojection_gate_px: f64,
    pub min_parallax_deg: f64,
    pub min_keyframes: usize,
    pub max_landmarks: usize,
    pub max_observations: usize,
}
impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            pixel_sigma: 1.5,
            huber_delta: 3.,
            initial_reprojection_gate_px: 3.,
            min_parallax_deg: 1.,
            min_keyframes: 2,
            max_landmarks: 100_000,
            max_observations: 2_000_000,
        }
    }
}
impl BundleConfig {
    pub fn validate(&self) -> Result<()> {
        if [
            self.pixel_sigma,
            self.huber_delta,
            self.initial_reprojection_gate_px,
            self.min_parallax_deg,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.)
            || self.min_parallax_deg >= 90.
            || self.min_keyframes < 2
            || self.max_landmarks == 0
            || self.max_observations < 4
        {
            return Err(Error("invalid global BA configuration".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LandmarkObservation {
    pub track_id: u64,
    pub pixel: [f64; 2],
    pub point_camera: Option<[f64; 3]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraObservations {
    pub camera: CameraModel,
    pub observations: Vec<LandmarkObservation>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleFrame {
    pub key: Key,
    pub timestamp_ns: i64,
    pub views: Vec<CameraObservations>,
}
impl BundleFrame {
    pub fn validate(&self) -> Result<()> {
        self.key.validate()?;
        if self.views.len() > 8 {
            return Err(Error("too many BA camera views".into()));
        }
        for view in &self.views {
            view.camera.camera()?;
            let mut tracks = BTreeSet::new();
            if view.observations.len() > 10000 {
                return Err(Error("too many BA observations".into()));
            }
            for o in &view.observations {
                if !tracks.insert(o.track_id)
                    || !o.pixel.iter().all(|x| x.is_finite())
                    || o.pixel[0] < 0.
                    || o.pixel[1] < 0.
                    || o.pixel[0] >= view.camera.width as f64
                    || o.pixel[1] >= view.camera.height as f64
                    || o.point_camera
                        .is_some_and(|p| !p.iter().all(|x| x.is_finite()))
                {
                    return Err(Error(
                        "invalid BA observation or duplicate track in camera".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizedLandmark {
    pub key: Key,
    pub component: Key,
    pub position: [f64; 3],
    pub observations: usize,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct BundleDiagnostics {
    pub frames: usize,
    pub candidate_tracks: usize,
    pub accepted_tracks: usize,
    pub accepted_observations: usize,
    pub rejected_tracks: usize,
    pub rejected_observations: usize,
}
pub struct BundleProblem {
    pub points: BTreeMap<u64, Vector3<f64>>,
    pub projections: Vec<ProjectionFactor>,
    pub identities: BTreeMap<u64, (Key, usize)>,
    pub diagnostics: BundleDiagnostics,
}
struct View<'a> {
    frame: &'a BundleFrame,
    camera: &'a CameraModel,
    observation: &'a LandmarkObservation,
    camera_to_map: SE3,
    ray: Vector3<f64>,
}
fn error(view: &View<'_>, point: &Vector3<f64>) -> f64 {
    let inverse = view.camera_to_map.inverse();
    let p = inverse.rotation * point + inverse.translation;
    if p.z <= 0.1 {
        return f64::INFINITY;
    }
    let [fx, fy, cx, cy] = view.camera.intrinsics;
    (Vector2::new(fx * p.x / p.z + cx, fy * p.y / p.z + cy) - Vector2::from(view.observation.pixel))
        .norm()
}
fn triangulate(views: &[View<'_>], indices: &[usize]) -> Option<Vector3<f64>> {
    let mut a = Matrix3::zeros();
    let mut b = Vector3::zeros();
    for &i in indices {
        let v = &views[i];
        let m = Matrix3::identity() - v.ray * v.ray.transpose();
        a += m;
        b += m * v.camera_to_map.translation;
    }
    if a.symmetric_eigen().eigenvalues.min() <= 1e-8 {
        return None;
    }
    let p = a.lu().solve(&b)?;
    p.iter().all(|x| x.is_finite()).then_some(p)
}

fn supported(config: &BundleConfig, views: &[View<'_>], inliers: &[usize]) -> bool {
    let keyframes: BTreeSet<_> = inliers.iter().map(|&i| &views[i].frame.key).collect();
    let cosine = config.min_parallax_deg.to_radians().cos();
    keyframes.len() >= config.min_keyframes
        && inliers.iter().any(|&i| {
            inliers
                .iter()
                .any(|&j| views[i].ray.dot(&views[j].ray) <= cosine)
        })
}
fn projection(
    config: &BundleConfig,
    view: &View<'_>,
    ids: &BTreeMap<Key, u64>,
    landmark: u64,
) -> Result<ProjectionFactor> {
    Ok(ProjectionFactor {
        pose: ids[&view.frame.key],
        landmark,
        camera_to_body: view.camera.camera_to_body.se3()?,
        intrinsics: view.camera.intrinsics,
        pixel: view.observation.pixel,
        sigma: config.pixel_sigma,
        huber: config.huber_delta,
    })
}

pub fn assemble(
    config: &BundleConfig,
    frames: &BTreeMap<Key, BundleFrame>,
    poses: &BTreeMap<Key, SE3>,
    ids: &BTreeMap<Key, u64>,
    previous: &GraphSnapshot,
) -> Result<BundleProblem> {
    let mut tracks: BTreeMap<Key, Vec<View<'_>>> = BTreeMap::new();
    let mut diagnostics = BundleDiagnostics::default();
    let mut total = 0;
    for (key, body) in poses {
        let Some(frame) = frames.get(key) else {
            continue;
        };
        diagnostics.frames += 1;
        for view in &frame.views {
            let camera_to_map = body.compose(&view.camera.camera_to_body.se3()?);
            let [fx, fy, cx, cy] = view.camera.intrinsics;
            for o in &view.observations {
                let ray = camera_to_map.rotation
                    * Vector3::new((o.pixel[0] - cx) / fx, (o.pixel[1] - cy) / fy, 1.).normalize();
                tracks
                    .entry(Key {
                        id: o.track_id,
                        ..key.clone()
                    })
                    .or_default()
                    .push(View {
                        frame,
                        camera: &view.camera,
                        observation: o,
                        camera_to_map: camera_to_map.clone(),
                        ray,
                    });
                total += 1;
                if total > config.max_observations {
                    return Err(Error("global BA observation capacity exceeded".into()));
                }
            }
        }
    }
    let old_points: BTreeMap<_, _> = previous
        .landmarks
        .iter()
        .map(|p| ((&p.key, &p.component), p))
        .collect();
    let old_poses: BTreeMap<_, _> = previous.poses.iter().map(|p| (&p.key, p)).collect();
    let mut problem = BundleProblem {
        points: BTreeMap::new(),
        projections: vec![],
        identities: BTreeMap::new(),
        diagnostics: Default::default(),
    };
    diagnostics.candidate_tracks = tracks.len();
    for (key, views) in tracks {
        let all: Vec<_> = (0..views.len()).collect();
        let mut seeds = Vec::new();
        let mut seeded_components = BTreeSet::new();
        for v in &views {
            let Some(old_pose) = old_poses.get(&v.frame.key) else {
                continue;
            };
            if !seeded_components.insert(&old_pose.component) {
                continue;
            }
            if let Some(p) = old_points.get(&(&key, &old_pose.component)) {
                let alignment = poses[&v.frame.key].compose(&old_pose.body_to_map.se3()?.inverse());
                seeds.push(alignment.rotation * Vector3::from(p.position) + alignment.translation);
            }
        }
        // Direct joint BA: cheap ray triangulation and bounded metric VIO
        // hypotheses, retaining the original admission/initialization path.
        if let Some(p) = triangulate(&views, &all) {
            seeds.push(p);
        }
        for v in views.iter().step_by((views.len() / 16).max(1)).take(16) {
            if let Some(p) = v.observation.point_camera.filter(|p| p[2] > 0.1) {
                seeds.push(
                    v.camera_to_map.rotation * Vector3::from(p) + v.camera_to_map.translation,
                );
            }
        }
        let mut best = None;
        for p in seeds {
            let errors: Vec<_> = views.iter().map(|v| error(v, &p)).collect();
            let inliers: Vec<_> = all
                .iter()
                .copied()
                .filter(|&i| errors[i] <= config.initial_reprojection_gate_px)
                .collect();
            let cost = inliers.iter().map(|&i| errors[i] * errors[i]).sum();
            if best
                .as_ref()
                .is_none_or(|(_, ids, e): &(Vector3<f64>, Vec<usize>, f64)| {
                    inliers.len() > ids.len() || (inliers.len() == ids.len() && cost < *e)
                })
            {
                best = Some((p, inliers, cost));
            }
        }
        let Some((mut point, mut inliers, _)) = best else {
            continue;
        };
        if let Some(p) = triangulate(&views, &inliers) {
            inliers.retain(|&i| error(&views[i], &p) <= config.initial_reprojection_gate_px);
            point = p;
        }
        if !supported(config, &views, &inliers) {
            continue;
        }
        if problem.points.len() >= config.max_landmarks {
            return Err(Error("global BA landmark capacity exceeded".into()));
        }
        let id = problem.points.len() as u64;
        problem.points.insert(id, point);
        problem.identities.insert(id, (key, inliers.len()));
        for &i in &inliers {
            problem
                .projections
                .push(projection(config, &views[i], ids, id)?);
        }
    }
    diagnostics.accepted_tracks = problem.points.len();
    diagnostics.accepted_observations = problem.projections.len();
    diagnostics.rejected_tracks = diagnostics.candidate_tracks - diagnostics.accepted_tracks;
    diagnostics.rejected_observations = total - diagnostics.accepted_observations;
    problem.diagnostics = diagnostics;
    Ok(problem)
}
