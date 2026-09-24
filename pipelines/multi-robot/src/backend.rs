use crate::*;
use nalgebra::Matrix6;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::Instant,
};
use visloc_core::geometry::{Pose, SE3};
use visloc_slam::{LinearSolver, PoseGraph, PoseGraphEdgeKind, PoseGraphSe3Config, RobustKernel};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    pub odometry_translation_sigma: f64,
    pub odometry_rotation_sigma: f64,
    pub loop_translation_sigma: f64,
    pub loop_rotation_sigma: f64,
}
impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            odometry_translation_sigma: 0.15,
            odometry_rotation_sigma: 0.03,
            loop_translation_sigma: 0.2,
            loop_rotation_sigma: 0.04,
        }
    }
}

#[derive(Clone, Default)]
pub struct Backend {
    pub records: BTreeMap<Key, KeyframeRecord>,
    pub loops: BTreeMap<Pair, LoopConstraint>,
    pub input_revision: u64,
    ids: BTreeMap<Key, u64>,
    pub config: BackendConfig,
}
impl Backend {
    pub fn insert_keyframe(&mut self, record: KeyframeRecord) -> Result<bool> {
        record.key.validate()?;
        record.body_to_odom.se3()?;
        if let Some(previous) = &record.previous {
            if !previous.same_session(&record.key) || previous.id >= record.key.id {
                return Err(Error("invalid odometry predecessor".into()));
            }
        }
        if let Some(previous) = self.records.get(&record.key) {
            if serde_json::to_vec(previous).unwrap() != serde_json::to_vec(&record).unwrap() {
                return Err(Error("conflicting keyframe retransmission".into()));
            }
            return Ok(false);
        }
        self.ids.insert(record.key.clone(), self.ids.len() as u64);
        self.records.insert(record.key.clone(), record);
        self.input_revision += 1;
        Ok(true)
    }
    pub fn insert_loop(&mut self, edge: LoopConstraint) -> Result<bool> {
        edge.from.validate()?;
        edge.to.validate()?;
        edge.to_from.se3()?;
        checked_information(&edge.information)?;
        if edge.from == edge.to
            || edge.pair.0 >= edge.pair.1
            || !edge.from.same_session(&edge.pair.0)
            || !edge.to.same_session(&edge.pair.1)
            || !edge.similarity.is_finite()
            || edge.similarity < 0.8
            || edge.verification.pnp_inliers < 15
        {
            return Err(Error("invalid verified loop constraint".into()));
        }
        if let Some(old) = self.loops.get(&edge.pair) {
            if serde_json::to_vec(old).unwrap() != serde_json::to_vec(&edge).unwrap() {
                return Err(Error("conflicting loop retransmission".into()));
            }
            return Ok(false);
        }
        // Endpoints may arrive later on a different DDS topic; keep the edge
        // pending until both records are available, without inventing poses.
        self.loops.insert(edge.pair.clone(), edge);
        self.input_revision += 1;
        Ok(true)
    }
    pub fn solve(&self, previous: &GraphSnapshot) -> Result<GraphSnapshot> {
        let start = Instant::now();
        let sigmas = [
            self.config.odometry_translation_sigma,
            self.config.odometry_rotation_sigma,
            self.config.loop_translation_sigma,
            self.config.loop_rotation_sigma,
        ];
        if sigmas.iter().any(|x| !x.is_finite() || *x <= 0.) {
            return Err(Error("invalid graph noise configuration".into()));
        }
        let mut edges: Vec<(Key, Key, SE3, PoseGraphEdgeKind, Matrix6<f64>)> = Vec::new();
        for r in self.records.values() {
            if let Some(p) = r.previous.as_ref().and_then(|p| self.records.get(p)) {
                if p.timestamp_ns >= r.timestamp_ns {
                    return Err(Error("non-increasing odometry time".into()));
                }
                edges.push((
                    p.key.clone(),
                    r.key.clone(),
                    r.body_to_odom
                        .se3()?
                        .inverse()
                        .compose(&p.body_to_odom.se3()?),
                    PoseGraphEdgeKind::Sequential,
                    information(
                        self.config.odometry_translation_sigma,
                        self.config.odometry_rotation_sigma,
                    ),
                ));
            }
        }
        for e in self.loops.values() {
            if self.records.contains_key(&e.from) && self.records.contains_key(&e.to) {
                // Keep the received anisotropy, with an explicit graph-level
                // scale for the configurable loop uncertainty defaults.
                let mut scale = Matrix6::identity();
                for i in 0..3 {
                    scale[(i, i)] = 0.2 / self.config.loop_translation_sigma;
                    scale[(i + 3, i + 3)] = 0.04 / self.config.loop_rotation_sigma;
                }
                edges.push((
                    e.from.clone(),
                    e.to.clone(),
                    e.to_from.se3()?,
                    PoseGraphEdgeKind::LoopClosure,
                    scale * checked_information(&e.information)? * scale,
                ));
            }
        }
        // Deterministic spanning-tree initialization welds unknown local frames
        // using measured constraints, never ground truth or pose proximity.
        let mut adjacency: BTreeMap<Key, Vec<(Key, SE3)>> = BTreeMap::new();
        for (a, b, z, _, _) in &edges {
            adjacency
                .entry(a.clone())
                .or_default()
                .push((b.clone(), z.inverse()));
            adjacency
                .entry(b.clone())
                .or_default()
                .push((a.clone(), z.clone()));
        }
        let mut seen = BTreeSet::new();
        let mut result = GraphSnapshot {
            revision: previous.revision + 1,
            input_revision: self.input_revision,
            loops: self.loops.values().cloned().collect(),
            ..Default::default()
        };
        let warm: BTreeMap<_, _> = previous.poses.iter().map(|p| (p.key.clone(), p)).collect();
        for root in self.records.keys() {
            if seen.contains(root) {
                continue;
            }
            result.components += 1;
            let mut poses = BTreeMap::new();
            let mut queue = VecDeque::from([root.clone()]);
            poses.insert(root.clone(), self.records[root].body_to_odom.se3()?);
            seen.insert(root.clone());
            while let Some(a) = queue.pop_front() {
                for (b, a_from_b) in adjacency.get(&a).into_iter().flatten() {
                    if seen.insert(b.clone()) {
                        poses.insert(b.clone(), poses[&a].compose(a_from_b));
                        queue.push_back(b.clone());
                    }
                }
            }
            // Reuse each previous component rigidly aligned into the new
            // component. Its internal corrected geometry remains intact.
            let mut alignments = BTreeMap::<Key, SE3>::new();
            for (key, seed) in &poses {
                if let Some(old) = warm.get(key) {
                    if !alignments.contains_key(&old.component) {
                        alignments.insert(
                            old.component.clone(),
                            seed.compose(&old.body_to_map.se3()?.inverse()),
                        );
                    }
                }
            }
            for (key, pose) in &mut poses {
                if let Some(old) = warm.get(key) {
                    *pose = alignments[&old.component].compose(&old.body_to_map.se3()?);
                }
            }
            let mut graph = PoseGraph::new();
            for (key, t) in &poses {
                graph.add_pose(
                    self.ids[key],
                    Pose {
                        world_to_camera: t.inverse(),
                    },
                );
            }
            graph.anchor(self.ids[root]);
            let mut has_loop = false;
            for (a, b, z, kind, info) in &edges {
                if poses.contains_key(a) && poses.contains_key(b) {
                    graph.add_edge_with_information(
                        self.ids[a],
                        self.ids[b],
                        z.clone(),
                        *kind,
                        *info,
                    );
                    has_loop |= *kind == PoseGraphEdgeKind::LoopClosure;
                }
            }
            if has_loop && poses.len() > 1 {
                let report = graph
                    .optimize_se3_iterative(&PoseGraphSe3Config {
                        max_iterations: 20,
                        initial_lambda: Some(1e-4),
                        chordal_init: false,
                        robust_kernel: RobustKernel::Huber { delta: 3. },
                        linear_solver: LinearSolver::Sparse,
                        ..Default::default()
                    })
                    .map_err(|e| Error(format!("pose graph: {e:?}")))?;
                if !report.initial_cost.is_finite()
                    || !report.final_cost.is_finite()
                    || report.final_cost > report.initial_cost + 1e-9
                {
                    return Err(Error("optimizer increased robust objective".into()));
                }
                result.initial_cost += report.initial_cost;
                result.final_cost += report.final_cost;
            }
            for key in poses.keys() {
                let t = graph.poses[&self.ids[key]].camera_to_world();
                let pose = Transform::from(&t);
                pose.se3()?;
                let raw = self.records[key].body_to_odom.se3()?;
                result.poses.push(OptimizedPose {
                    key: key.clone(),
                    timestamp_ns: self.records[key].timestamp_ns,
                    component: root.clone(),
                    body_to_map: pose,
                    map_from_odom: Transform::from(&t.compose(&raw.inverse())),
                });
            }
        }
        result.poses.sort_by(|a, b| a.key.cmp(&b.key));
        result.solve_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok(result)
    }
}
