//! Local connectivity from persistent, triangulated VIO landmarks. Descriptor
//! matches and pose proximity never create edges in this graph.
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Default)]
pub(crate) struct CovisibilityGraph {
    observers: HashMap<u64, Vec<usize>>,
    neighbors: Vec<BTreeMap<usize, usize>>,
    ordinals: Vec<u64>,
}

impl CovisibilityGraph {
    /// Insert one keyframe and return weighted edges to older keyframes.
    pub fn insert(
        &mut self,
        ordinal: u64,
        landmarks: impl IntoIterator<Item = u64>,
        min_shared: usize,
    ) -> BTreeMap<usize, usize> {
        let index = self.neighbors.len();
        let mut shared = BTreeMap::<usize, usize>::new();
        // Count landmark identities once even if an upstream source repeats an observation.
        for landmark in landmarks.into_iter().collect::<BTreeSet<_>>() {
            let observers = self.observers.entry(landmark).or_default();
            for &other in observers.iter() {
                *shared.entry(other).or_default() += 1;
            }
            observers.push(index);
        }
        shared.retain(|_, count| *count >= min_shared);
        for (&other, &count) in &shared {
            self.neighbors[other].insert(index, count);
        }
        self.neighbors.push(shared.clone());
        self.ordinals.push(ordinal);
        shared
    }

    /// Multi-source breadth-first search with a bounded hop count. Never use
    /// the whole connected component: an odometry traverse can connect it all.
    pub fn neighborhood(&self, seeds: &[usize], hops: usize) -> BTreeSet<usize> {
        let mut visited: BTreeSet<_> = seeds.iter().copied().collect();
        let mut frontier = visited.clone();
        for _ in 0..hops {
            let mut next = BTreeSet::new();
            for index in frontier {
                for &neighbor in self.neighbors[index].keys() {
                    if visited.insert(neighbor) {
                        next.insert(neighbor);
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        visited
    }

    pub fn local_reason(
        &self,
        query: &[usize],
        candidate: &[usize],
        neighborhood: &BTreeSet<usize>,
    ) -> Option<&'static str> {
        if candidate.iter().any(|i| query.contains(i)) {
            return Some("overlapping_keyframes");
        }
        // Consecutive keyframes remain local even when landmarks are immature
        // or tracking is lost. Use actual KF ordinals, never raw frame spacing.
        if candidate.iter().any(|&a| {
            query
                .iter()
                .any(|&b| self.ordinals[a].abs_diff(self.ordinals[b]) == 1)
        }) {
            return Some("consecutive_keyframes");
        }
        candidate
            .iter()
            .any(|i| neighborhood.contains(i))
            .then_some("covisible_neighborhood")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_and_consecutive_sequences_rejected_without_landmarks() {
        let mut graph = CovisibilityGraph::default();
        for i in 0..12 {
            graph.insert(i, [], 15);
        }
        let query = [7, 8, 9, 10, 11];
        let local = graph.neighborhood(&query, 2);
        assert_eq!(
            graph.local_reason(&query, &[6, 7, 8, 9, 10], &local),
            Some("overlapping_keyframes")
        );
        assert_eq!(
            graph.local_reason(&query, &[2, 3, 4, 5, 6], &local),
            Some("consecutive_keyframes")
        );
        assert_eq!(graph.local_reason(&query, &[0, 1, 2, 3, 4], &local), None);
    }

    #[test]
    fn covisibility_excludes_two_hops_but_preserves_distant_revisits() {
        let mut graph = CovisibilityGraph::default();
        graph.insert(0, [1, 2, 3], 3);
        graph.insert(20, [1, 2, 3, 4, 5, 6], 3);
        graph.insert(40, [4, 5, 6, 7, 8, 9], 3);
        graph.insert(60, [7, 8, 9], 3);
        graph.insert(80, [101, 102, 103], 3); // same place with new tracks is eligible
        let local = graph.neighborhood(&[0], 2);
        assert_eq!(
            graph.local_reason(&[0], &[2], &local),
            Some("covisible_neighborhood")
        );
        assert_eq!(graph.local_reason(&[0], &[3], &local), None);
        assert_eq!(graph.local_reason(&[0], &[4], &local), None);
        assert_eq!(
            graph.local_reason(&[4, 2], &[0], &graph.neighborhood(&[4, 2], 2)),
            Some("covisible_neighborhood")
        );
    }

    #[test]
    fn duplicate_landmarks_do_not_create_spurious_edges() {
        let mut graph = CovisibilityGraph::default();
        graph.insert(0, [1, 2, 3], 3);
        let edges = graph.insert(10, [1, 1, 1, 2], 3);
        assert!(edges.is_empty());
        assert!(!graph.neighborhood(&[0], 2).contains(&1));
        let edges = graph.insert(20, [1, 2, 3], 3);
        assert_eq!(edges.get(&0), Some(&3));
        assert!(!edges.contains_key(&1));
    }
}
