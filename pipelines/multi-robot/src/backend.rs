use crate::*;
use nalgebra::Matrix6;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::Instant,
};
use visloc_core::geometry::SE3;
use visloc_gtsam::PoseGraph;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BackendConfig {
    pub min_loop_similarity: f32,
    pub odometry_translation_sigma: f64,
    pub odometry_rotation_sigma: f64,
    pub loop_translation_sigma: f64,
    pub loop_rotation_sigma: f64,
    pub gps: GpsConfig,
}
impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            min_loop_similarity: 0.8,
            odometry_translation_sigma: 0.15,
            odometry_rotation_sigma: 0.03,
            loop_translation_sigma: 0.2,
            loop_rotation_sigma: 0.04,
            gps: GpsConfig::default(),
        }
    }
}

impl BackendConfig {
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.min_loop_similarity) {
            return Err(Error("invalid loop similarity threshold".into()));
        }
        self.gps.validate()?;
        let sigmas = [
            self.odometry_translation_sigma,
            self.odometry_rotation_sigma,
            self.loop_translation_sigma,
            self.loop_rotation_sigma,
        ];
        if sigmas.iter().any(|x| !x.is_finite() || *x <= 0.) {
            return Err(Error("invalid graph noise configuration".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct Backend {
    pub records: BTreeMap<Key, KeyframeRecord>,
    pub loops: BTreeMap<Pair, LoopConstraint>,
    pub input_revision: u64,
    pub gps_records: BTreeMap<Key, GpsRecord>,
    pub gps_datum: Option<GpsDatum>,
    ids: BTreeMap<Key, u64>,
    pub config: BackendConfig,
}
impl Backend {
    pub fn insert_gps(&mut self, record: GpsRecord) -> Result<bool> {
        self.config.gps.validate()?;
        record.validate()?;
        if !self.config.gps.enabled {
            return Ok(false);
        }
        if let Some(old) = self.gps_records.get(&record.key) {
            if old != &record {
                return Err(Error("conflicting GPS retransmission".into()));
            }
            return Ok(false);
        }
        if self.gps_datum.is_none() && record.status >= 0 && record.quality != Some(0) {
            self.gps_datum = self
                .config
                .gps
                .origin
                .or_else(|| record.lla.map(|p| GpsDatum::from_lla([p[0], p[1], 0.])));
        }
        self.gps_records.insert(record.key.clone(), record);
        self.input_revision += 1;
        Ok(true)
    }

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
        self.config.validate()?;
        edge.from.validate()?;
        edge.to.validate()?;
        edge.to_from.se3()?;
        checked_information(&edge.information)?;
        if edge.from == edge.to
            || edge.pair.0 >= edge.pair.1
            || !edge.from.same_session(&edge.pair.0)
            || !edge.to.same_session(&edge.pair.1)
            || !edge.similarity.is_finite()
            || edge.similarity < self.config.min_loop_similarity
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
        self.config.validate()?;
        let mut edges: Vec<(Key, Key, SE3, bool, Matrix6<f64>)> = Vec::new();
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
                    false,
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
                    true,
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
        if self.config.gps.enabled
            && self
                .config
                .gps
                .origin
                .zip(self.gps_datum)
                .is_some_and(|(a, b)| a != b)
        {
            return Err(Error(
                "configured GPS datum conflicts with persisted mission datum".into(),
            ));
        }
        let datum = self.gps_datum.or(self.config.gps.origin);
        if self.config.gps.enabled && previous.gps.datum.zip(datum).is_some_and(|(a, b)| a != b) {
            return Err(Error(
                "GPS datum changed since the previous solution".into(),
            ));
        }
        let associated = if self.config.gps.enabled {
            if let Some(datum) = datum {
                let (associated, diagnostics) = crate::gps::associate(
                    &self.config.gps,
                    datum,
                    &self.gps_records,
                    &self.records,
                );
                result.gps.datum = Some(datum);
                result.gps.diagnostics = diagnostics;
                associated
            } else {
                result.gps.diagnostics = self
                    .gps_records
                    .values()
                    .map(|r| {
                        let mut d = crate::gps::GpsDiagnostic::new(r);
                        d.reason = "no_fix".into();
                        d
                    })
                    .collect();
                Vec::new()
            }
        } else {
            Vec::new()
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
            let component_gps: Vec<_> = associated
                .iter()
                .filter(|g| poses.contains_key(&g.from) && poses.contains_key(&g.to))
                .collect();
            let mut georeferenced = false;
            if self.config.gps.enabled {
                if let Some((key, old)) = poses
                    .keys()
                    .filter_map(|key| warm.get(key).map(|old| (key, old)))
                    .find(|(_, old)| previous.gps.aligned_components.contains(&old.component))
                {
                    let alignment = old.body_to_map.se3()?.compose(&poses[key].inverse());
                    for pose in poses.values_mut() {
                        *pose = alignment.compose(pose);
                    }
                    georeferenced = true;
                }
            }
            let mut session_alignments = BTreeMap::new();
            for (key, pose) in &poses {
                session_alignments
                    .entry((key.robot.clone(), key.session.clone()))
                    .or_insert_with(|| {
                        pose.compose(&self.records[key].body_to_odom.se3().unwrap().inverse())
                    });
            }
            // Reuse each previous component rigidly aligned into the new
            // component. Its internal corrected geometry remains intact.
            let mut alignments = BTreeMap::<Key, SE3>::new();
            for (key, seed) in &poses {
                if let Some(old) = warm.get(key) {
                    if !alignments.contains_key(&old.component) {
                        alignments.insert(
                            old.component.clone(),
                            if georeferenced
                                && previous.gps.aligned_components.contains(&old.component)
                            {
                                SE3::identity()
                            } else {
                                seed.compose(&old.body_to_map.se3()?.inverse())
                            },
                        );
                    }
                }
            }
            for (key, pose) in &mut poses {
                if let Some(old) = warm.get(key) {
                    *pose = alignments[&old.component].compose(&old.body_to_map.se3()?);
                }
            }
            if !georeferenced && !component_gps.is_empty() {
                let samples: Vec<_> = component_gps
                    .iter()
                    .map(|g| {
                        let alignment =
                            &session_alignments[&(g.from.robot.clone(), g.from.session.clone())];
                        (
                            alignment.rotation * g.raw_position + alignment.translation,
                            g.measured,
                            g.sigma_horizontal,
                        )
                    })
                    .collect();
                if let Some(alignment) = crate::gps::initialize(&self.config.gps, &samples) {
                    for pose in poses.values_mut() {
                        *pose = alignment.compose(pose);
                    }
                    georeferenced = true;
                }
            }
            let mut graph = PoseGraph::new();
            for (key, t) in &poses {
                graph.add_pose(self.ids[key], t.clone());
            }
            graph.anchor(self.ids[root]);
            if georeferenced {
                result.gps.aligned_components.push(root.clone());
                if !component_gps.is_empty() {
                    graph.horizontal_anchor = true;
                    graph.gps_factors = component_gps
                        .iter()
                        .map(|g| g.factor(&self.ids, &self.config.gps))
                        .collect();
                }
            }
            let mut has_loop = false;
            for (a, b, z, is_loop, info) in &edges {
                if poses.contains_key(a) && poses.contains_key(b) {
                    graph.add_edge_with_information(self.ids[a], self.ids[b], z.clone(), *info);
                    has_loop |= *is_loop;
                }
            }
            if (has_loop || !graph.gps_factors.is_empty()) && poses.len() > 1 {
                let report = graph.optimize().map_err(|e| Error(e.to_string()))?;
                result.initial_cost += report.initial_cost;
                result.final_cost += report.final_cost;
                result.optimizer_reports.push(report);
            }
            if georeferenced {
                for (g, diagnostic) in component_gps.iter().zip(&graph.gps_diagnostics) {
                    let residual = diagnostic.residual;
                    let d2 = diagnostic.squared_mahalanobis;
                    let d = &mut result.gps.diagnostics[g.diagnostic];
                    d.reason = "active".into();
                    d.predicted_enu = Some((g.measured + residual).into());
                    d.residual_enu = Some(residual.into());
                    d.squared_mahalanobis = Some(d2);
                    d.robust_weight = Some(diagnostic.robust_weight);
                    d.optimization_weight = Some(diagnostic.optimization_weight);
                    d.within_tolerance = d2.sqrt() <= self.config.gps.residual_deadband_sigma;
                }
            }
            for key in poses.keys() {
                let t = &graph.poses[&self.ids[key]];
                let pose = Transform::from(t);
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
