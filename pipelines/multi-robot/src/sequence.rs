use crate::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub struct KeyframeMeta {
    pub record: KeyframeRecord,
    pub camera_position: [f64; 3],
    pub landmarks: BTreeSet<u64>,
}
#[derive(Default)]
pub struct Partition {
    pub retained: bool,
    pub reset: bool,
    pub complete: Option<Vec<KeyframeMeta>>,
}

/// One instance per robot session. All keyframes enter covisibility, including
/// those deliberately omitted from the ten-keyframe VPR blocks.
#[derive(Default)]
pub struct SequenceBuilder {
    pending: Vec<KeyframeMeta>,
    last_key: Option<Key>,
    last_retained: BTreeSet<u64>,
    observers: BTreeMap<u64, Vec<u64>>,
    neighbors: BTreeMap<u64, BTreeSet<u64>>,
}
impl SequenceBuilder {
    pub fn push(&mut self, meta: KeyframeMeta) -> Result<Partition> {
        meta.record.key.validate()?;
        meta.record.body_to_odom.se3()?;
        if !meta.camera_position.iter().all(|v| v.is_finite()) {
            return Err(Error("invalid camera center".into()));
        }
        let key = &meta.record.key;
        if let Some(last) = &self.last_key {
            if !last.same_session(key) || key.id <= last.id {
                return Err(Error("keyframes must increase within one session".into()));
            }
        }
        let reset = self
            .last_key
            .as_ref()
            .is_some_and(|last| last.id + 1 != key.id);
        if reset {
            self.pending.clear();
            self.last_retained.clear();
        }
        self.last_key = Some(key.clone());
        let mut shared = BTreeMap::<u64, usize>::new();
        for id in &meta.landmarks {
            let observers = self.observers.entry(*id).or_default();
            for other in observers.iter() {
                *shared.entry(*other).or_default() += 1;
            }
            observers.push(key.id);
        }
        self.neighbors.entry(key.id).or_default();
        for (other, count) in shared {
            if count >= 15 {
                self.neighbors.entry(other).or_default().insert(key.id);
                self.neighbors.entry(key.id).or_default().insert(other);
            }
        }
        let overlap = meta.landmarks.intersection(&self.last_retained).count();
        let skip =
            !meta.landmarks.is_empty() && overlap as f64 / meta.landmarks.len() as f64 > 0.95;
        if skip {
            return Ok(Partition {
                reset,
                ..Default::default()
            });
        }
        self.last_retained = meta.landmarks.clone();
        self.pending.push(meta);
        let complete = (self.pending.len() == 10).then(|| std::mem::take(&mut self.pending));
        Ok(Partition {
            retained: true,
            reset,
            complete,
        })
    }
    pub fn neighborhood(&self, members: &[Key]) -> Vec<u64> {
        let mut visited: BTreeSet<_> = members.iter().map(|k| k.id).collect();
        let mut frontier = visited.clone();
        for _ in 0..2 {
            let mut next = BTreeSet::new();
            for id in frontier {
                if let Some(neighbors) = self.neighbors.get(&id) {
                    for &neighbor in neighbors {
                        if visited.insert(neighbor) {
                            next.insert(neighbor);
                        }
                    }
                }
            }
            frontier = next;
        }
        visited.into_iter().collect()
    }
}

/// Exact objective from SB-SLAM III-B, with lexicographic deterministic ties.
pub fn select_five(positions: &[[f64; 3]]) -> Result<[usize; 5]> {
    if positions.len() != 10 || !positions.iter().flatten().all(|x| x.is_finite()) {
        return Err(Error("selection requires ten finite camera centers".into()));
    }
    let mut best = [0, 1, 2, 3, 9];
    let mut cost = f64::INFINITY;
    for a in 1..7 {
        for b in a + 1..8 {
            for c in b + 1..9 {
                let ids = [0, a, b, c, 9];
                let value = ids
                    .windows(2)
                    .map(|w| {
                        (0..3)
                            .map(|d| (positions[w[0]][d] - positions[w[1]][d]).powi(2))
                            .sum::<f64>()
                    })
                    .sum();
                if value < cost {
                    cost = value;
                    best = ids;
                }
            }
        }
    }
    Ok(best)
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Search all 25 combinations; ties choose the earliest (a,b).
pub fn refine(a: &FrameDescriptors, b: &FrameDescriptors) -> Result<(Key, Key, f32)> {
    a.validate()?;
    b.validate()?;
    let mut best = (0, 0, f32::NEG_INFINITY);
    for i in 0..5 {
        for j in 0..5 {
            let score = dot(
                &a.descriptors[i * 512..(i + 1) * 512],
                &b.descriptors[j * 512..(j + 1) * 512],
            );
            if score > best.2 {
                best = (i, j, score);
            }
        }
    }
    Ok((a.frames[best.0].clone(), b.frames[best.1].clone(), best.2))
}

fn local(a: &Sequence, b: &Sequence, temporal: bool) -> bool {
    if !a.key.same_session(&b.key) {
        return false;
    }
    a.key.id.abs_diff(b.key.id) <= 1
        || a.members
            .iter()
            .any(|x| b.members.iter().any(|y| x.id.abs_diff(y.id) <= 1))
        || b.members
            .iter()
            .any(|x| a.excluded_keyframes.contains(&x.id))
        || a.members
            .iter()
            .any(|x| b.excluded_keyframes.contains(&x.id))
        || (temporal
            && a.start_ns
                .max(b.start_ns)
                .saturating_sub(a.end_ns.min(b.end_ns))
                < 20_000_000_000)
}

#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct RetrievalStats {
    pub excluded_local: usize,
    pub compared: usize,
    pub above_threshold: usize,
    pub grouped: usize,
}
#[derive(Default)]
pub struct Retrieval {
    pub sequences: BTreeMap<Key, Sequence>,
}
impl Retrieval {
    pub fn insert(&mut self, seq: Sequence) -> Result<bool> {
        seq.validate()?;
        if let Some(previous) = self.sequences.get(&seq.key) {
            if serde_json::to_vec(previous).unwrap() != serde_json::to_vec(&seq).unwrap() {
                return Err(Error("conflicting sequence retransmission".into()));
            }
            return Ok(false);
        }
        self.sequences.insert(seq.key.clone(), seq);
        Ok(true)
    }
    pub fn candidates(&self, query: &Sequence, minimum: f32) -> (Vec<(Key, f32)>, RetrievalStats) {
        self.candidates_for(query, minimum, None)
    }
    pub fn candidates_for(
        &self,
        query: &Sequence,
        minimum: f32,
        owner: Option<&Key>,
    ) -> (Vec<(Key, f32)>, RetrievalStats) {
        let mut stats = RetrievalStats::default();
        let mut ranked = Vec::new();
        for seq in self.sequences.values() {
            if owner.is_some_and(|o| query.key.robot != o.robot && seq.key.robot != o.robot) {
                continue;
            }
            if seq.key == query.key || seq.model_id != query.model_id {
                continue;
            }
            if local(query, seq, true) {
                stats.excluded_local += 1;
                continue;
            }
            stats.compared += 1;
            let similarity = dot(&query.descriptor, &seq.descriptor);
            if similarity >= minimum {
                stats.above_threshold += 1;
                ranked.push((seq, similarity));
            }
        }
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.key.cmp(&b.0.key)));
        let mut selected: Vec<(&Sequence, f32)> = Vec::new();
        for item in ranked {
            if selected.iter().any(|s| local(s.0, item.0, false)) {
                stats.grouped += 1;
                continue;
            }
            selected.push(item);
            if selected.len() == 3 {
                break;
            }
        }
        (
            selected
                .into_iter()
                .map(|(s, v)| (s.key.clone(), v))
                .collect(),
            stats,
        )
    }
}
