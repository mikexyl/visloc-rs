use nalgebra::{Point3, UnitQuaternion, Vector3};
use std::collections::BTreeSet;
use visloc_core::geometry::SE3;
use visloc_multi_robot::{backend::*, sequence::*, *};

fn t(x: f64, y: f64, z: f64) -> Transform {
    Transform::from(&SE3::new(UnitQuaternion::identity(), Vector3::new(x, y, z)))
}
fn key(robot: &str, id: u64) -> Key {
    Key::new(robot, "session", id)
}
fn record(robot: &str, id: u64, x: f64) -> KeyframeRecord {
    KeyframeRecord {
        key: key(robot, id),
        timestamp_ns: id as i64 * 1_000_000_000,
        body_to_odom: t(x, 0., 0.),
        previous: (id > 0).then(|| key(robot, id - 1)),
    }
}
fn meta(id: u64, landmarks: BTreeSet<u64>) -> KeyframeMeta {
    KeyframeMeta {
        record: record("a", id, id as f64),
        camera_position: [id as f64, 0., 0.],
        landmarks,
    }
}
fn descriptor(axis: usize) -> Vec<f32> {
    let mut v = vec![0.; 512];
    v[axis] = 1.;
    v
}
fn sequence(robot: &str, id: u64, start: u64) -> Sequence {
    Sequence {
        key: key(robot, id),
        members: (start..start + 10).map(|i| key(robot, i)).collect(),
        selected: [0, 2, 4, 6, 9].map(|i| key(robot, start + i)).to_vec(),
        start_ns: start as i64 * 1_000_000_000,
        end_ns: (start + 9) as i64 * 1_000_000_000,
        model_id: "model".into(),
        descriptor: descriptor(0),
        excluded_keyframes: vec![],
    }
}
fn edge(a: Key, b: Key, z: Transform) -> LoopConstraint {
    LoopConstraint {
        pair: Pair::new(Key { id: 0, ..a.clone() }, Key { id: 0, ..b.clone() }),
        from: a,
        to: b,
        to_from: z,
        information: matrix_values(&information(0.2, 0.04)),
        similarity: 0.9,
        verification: Verification {
            pnp_inliers: 30,
            ..Default::default()
        },
    }
}

#[test]
fn exact_selection_and_stationary_ties() {
    let positions: Vec<_> = (0..10).map(|i| [i as f64, 0., 0.]).collect();
    assert_eq!(select_five(&positions).unwrap(), [0, 2, 4, 6, 9]);
    assert_eq!(select_five(&[[0.; 3]; 10]).unwrap(), [0, 1, 2, 3, 9]);
    assert!(select_five(&[[0.; 3]; 9]).is_err());
}
#[test]
fn covisibility_skips_do_not_reset_blocks_but_missing_keyframes_do() {
    let mut b = SequenceBuilder::default();
    assert!(b.push(meta(0, (0..100).collect())).unwrap().retained);
    assert!(!b.push(meta(1, (0..100).collect())).unwrap().retained);
    assert!(b.push(meta(2, (5..105).collect())).unwrap().retained); // exactly .95
    let result = b.push(meta(4, BTreeSet::new())).unwrap();
    assert!(result.reset && result.retained);
    for id in 5..13 {
        assert!(b
            .push(meta(id, BTreeSet::new()))
            .unwrap()
            .complete
            .is_none());
    }
    let result = b.push(meta(13, BTreeSet::new())).unwrap();
    let block = result.complete.unwrap();
    assert_eq!(block[0].record.key.id, 4);
    assert_eq!(block[9].record.key.id, 13);
}
#[test]
fn ten_retained_frames_not_raw_ordinals() {
    let mut b = SequenceBuilder::default();
    let mut completed = None;
    for i in 0..20 {
        let p = b
            .push(meta(i, ((i / 2) * 100..(i / 2 + 1) * 100).collect()))
            .unwrap();
        if p.complete.is_some() {
            completed = p.complete;
        }
    }
    let block = completed.unwrap();
    assert_eq!(
        block.iter().map(|m| m.record.key.id).collect::<Vec<_>>(),
        (0..20).step_by(2).collect::<Vec<_>>()
    );
}
#[test]
fn refinement_searches_both_axes_instead_of_newest_frame() {
    let a = FrameDescriptors {
        sequence: key("a", 0),
        frames: (0..5).map(|i| key("a", i)).collect(),
        descriptors: (0..5).flat_map(descriptor).collect(),
    };
    let mut b = FrameDescriptors {
        sequence: key("b", 0),
        frames: (0..5).map(|i| key("b", i)).collect(),
        descriptors: (10..15).flat_map(descriptor).collect(),
    };
    b.descriptors[512..1024].copy_from_slice(&descriptor(2));
    let (ka, kb, s) = refine(&a, &b).unwrap();
    assert_eq!((ka.id, kb.id, s), (2, 1, 1.));
}
#[test]
fn local_candidates_excluded_before_top_k_and_cross_robot_ids_never_excluded() {
    let mut db = Retrieval::default();
    let mut query = sequence("a", 10, 200);
    query.excluded_keyframes = (0..10).collect();
    db.insert(sequence("a", 0, 0)).unwrap();
    db.insert(sequence("a", 9, 190)).unwrap();
    db.insert(sequence("b", 10, 200)).unwrap();
    db.insert(sequence("c", 10, 200)).unwrap();
    db.insert(sequence("d", 10, 200)).unwrap();
    let (c, stats) = db.candidates(&query, 0.8);
    assert_eq!(c.len(), 3);
    assert_eq!(stats.excluded_local, 2);
    assert_eq!(stats.compared, 3);
    assert!(c.iter().all(|(k, _)| k.robot != "a"));
}
#[test]
fn remote_queries_do_not_spend_slots_on_unowned_pairs() {
    let mut db = Retrieval::default();
    for r in ["a", "b", "c", "d"] {
        db.insert(sequence(r, 0, 0)).unwrap();
    }
    let (c, s) = db.candidates_for(&sequence("d", 9, 90), 0.8, Some(&key("a", 0)));
    assert_eq!(s.compared, 1);
    assert_eq!(c[0].0.robot, "a");
}
#[test]
fn identity_dedup_and_conflicting_retransmissions() {
    let mut b = Backend::default();
    assert!(b.insert_keyframe(record("a", 0, 0.)).unwrap());
    assert!(!b.insert_keyframe(record("a", 0, 0.)).unwrap());
    assert!(b.insert_keyframe(record("b", 0, 0.)).unwrap());
    assert!(b.insert_keyframe(record("a", 0, 1.)).is_err());
    assert_eq!(b.records.len(), 2);
}
#[test]
fn disconnected_components_remain_independent_until_measured_bridge() {
    let mut b = Backend::default();
    for r in ["a", "b"] {
        for i in 0..4 {
            b.insert_keyframe(record(r, i, i as f64)).unwrap();
        }
    }
    let before = b.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(before.components, 2);
    let bridge = edge(key("a", 1), key("b", 1), t(-10., 0., 0.));
    b.insert_loop(bridge.clone()).unwrap();
    assert!(!b.insert_loop(bridge).unwrap());
    let after = b.solve(&before).unwrap();
    assert_eq!(after.components, 1);
    assert!(after.final_cost <= after.initial_cost + 1e-9);
    for p in &after.poses {
        assert!(
            (p.body_to_map.translation[0]
                - (p.key.id as f64 + if p.key.robot == "b" { 10. } else { 0. }))
            .abs()
                < 1e-7
        );
    }
    assert_eq!(after.poses[0].body_to_map.translation, [0., 0., 0.]);
}
#[test]
fn out_of_order_constraints_wait_for_real_endpoints() {
    let mut b = Backend::default();
    b.insert_loop(edge(key("a", 0), key("b", 0), t(-3., 0., 0.)))
        .unwrap();
    assert_eq!(b.solve(&GraphSnapshot::default()).unwrap().poses.len(), 0);
    b.insert_keyframe(record("a", 0, 0.)).unwrap();
    b.insert_keyframe(record("b", 0, 0.)).unwrap();
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(s.components, 1);
    assert!((s.poses[1].body_to_map.translation[0] - 3.).abs() < 1e-7);
}
#[test]
fn no_loop_graph_preserves_raw_vio() {
    let mut b = Backend::default();
    for i in (0..20).rev() {
        b.insert_keyframe(record("a", i, i as f64 * 0.31)).unwrap();
    }
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    for p in s.poses {
        assert!(
            (p.body_to_map.translation[0] - b.records[&p.key].body_to_odom.translation[0]).abs()
                < 1e-10
        );
    }
}
#[test]
fn invalid_information_and_pose_are_rejected() {
    assert!(checked_information(&vec![0.; 36]).is_err());
    let mut transform = t(0., 0., 0.);
    transform.rotation_xyzw = [0.; 4];
    assert!(transform.se3().is_err());
}

fn geometry_fixture() -> (FeatureFrame, FeatureFrame, SE3, Vec<(usize, usize, f32)>) {
    let ca = CameraModel {
        width: 800,
        height: 600,
        intrinsics: [480., 470., 395., 302.],
        camera_to_body: Transform::from(&SE3::new(
            UnitQuaternion::from_euler_angles(0.1, 0.02, -0.03),
            Vector3::new(0.25, 0.05, 0.02),
        )),
    };
    let cb = CameraModel {
        width: 960,
        height: 720,
        intrinsics: [580., 590., 481., 355.],
        camera_to_body: Transform::from(&SE3::new(
            UnitQuaternion::from_euler_angles(-0.05, 0.03, 0.02),
            Vector3::new(-0.1, 0.2, 0.03),
        )),
    };
    let z = SE3::new(
        UnitQuaternion::from_euler_angles(0.02, -0.03, 0.04),
        Vector3::new(0.4, 0.05, 0.02),
    );
    let mut a = FeatureFrame {
        key: key("a", 0),
        timestamp_ns: 0,
        camera: ca.clone(),
        pixels: vec![],
        descriptors: vec![],
        track_ids: vec![],
        points_camera: vec![],
    };
    let mut b = FeatureFrame {
        key: key("b", 0),
        timestamp_ns: 0,
        camera: cb.clone(),
        pixels: vec![],
        descriptors: vec![],
        track_ids: vec![],
        points_camera: vec![],
    };
    for y in 0..8 {
        for x in 0..10 {
            let p = Point3::new(
                (x as f64 - 4.5) * 0.45,
                (y as f64 - 3.5) * 0.45,
                4. + ((x * 7 + y * 3) % 9) as f64 * 0.2,
            );
            let q = z.transform_point(&p);
            for (frame, cam, point) in [(&mut a, &ca, p), (&mut b, &cb, q)] {
                let [fx, fy, cx, cy] = cam.intrinsics;
                frame.pixels.push([
                    (fx * point.x / point.z + cx) as f32,
                    (fy * point.y / point.z + cy) as f32,
                ]);
                frame.points_camera.push(Some(point.coords.into()));
                frame.track_ids.push(frame.pixels.len() as u64);
                frame.descriptors.extend([1.; 64]);
            }
        }
    }
    let expected = cb
        .camera_to_body
        .se3()
        .unwrap()
        .compose(&z)
        .compose(&ca.camera_to_body.se3().unwrap().inverse());
    (a, b, expected, (0..80).map(|i| (i, i, 1.)).collect())
}
#[test]
fn different_rigs_and_reverse_pnp_preserve_body_constraint_direction() {
    let (mut a, b, expected, matches) = geometry_fixture();
    a.points_camera.fill(None); // forward impossible; reverse must succeed
    let (edge, diag) =
        geometry::verify(&a, &b, &matches, Pair::new(key("a", 0), key("b", 0)), 0.9).unwrap();
    let result = edge.expect("reverse PnP");
    assert!(diag.reverse);
    let error = result.to_from.se3().unwrap().compose(&expected.inverse());
    assert!(error.translation.norm() < 1e-4 && error.rotation.angle() < 1e-4);
}
#[test]
fn two_d_and_pnp_reject_scrambled_matches() {
    let (a, b, _, _) = geometry_fixture();
    let matches = (0..80)
        .map(|i| (i, (i * 37 + 13) % 80, 1.))
        .collect::<Vec<_>>();
    let (edge, _) =
        geometry::verify(&a, &b, &matches, Pair::new(key("a", 0), key("b", 0)), 0.9).unwrap();
    assert!(edge.is_none());
}

#[test]
fn neighborhood_is_exactly_two_hops_over_all_keyframes() {
    let mut builder = SequenceBuilder::default();
    builder.push(meta(0, (0..15).collect())).unwrap();
    builder.push(meta(1, (0..30).collect())).unwrap();
    builder.push(meta(2, (15..45).collect())).unwrap();
    builder.push(meta(3, (30..60).collect())).unwrap();
    let neighborhood = builder.neighborhood(&[key("a", 0)]);
    assert_eq!(neighborhood, vec![0, 1, 2]);
    // Intentional retention skips still enter the covisibility graph.
    assert!(!builder.push(meta(4, (30..60).collect())).unwrap().retained);
    assert!(builder.neighborhood(&[key("a", 2)]).contains(&4));
}

#[test]
fn restarted_sessions_do_not_collide_or_apply_temporal_exclusions() {
    let old = sequence("a", 0, 0);
    let mut new = old.clone();
    new.key.session = "restarted".into();
    for key in new.members.iter_mut().chain(&mut new.selected) {
        key.session = new.key.session.clone();
    }
    let mut retrieval = Retrieval::default();
    retrieval.insert(old).unwrap();
    assert_eq!(retrieval.candidates(&new, 0.8).0.len(), 1);
    let mut backend = Backend::default();
    backend.insert_keyframe(record("a", 0, 0.)).unwrap();
    let mut r = record("a", 0, 2.);
    r.key.session = "restarted".into();
    backend.insert_keyframe(r).unwrap();
    assert_eq!(
        backend.solve(&GraphSnapshot::default()).unwrap().components,
        2
    );
}

#[test]
fn bidirectional_verification_chooses_the_stronger_direction() {
    let (a, mut b, expected, matches) = geometry_fixture();
    for (i, point) in b.points_camera.iter_mut().enumerate() {
        if i % 4 != 0 {
            *point = None;
        }
    }
    let (edge, diag) =
        geometry::verify(&a, &b, &matches, Pair::new(key("a", 0), key("b", 0)), 0.9).unwrap();
    assert!(!diag.reverse);
    assert_eq!(diag.pnp_inliers, 80);
    let error = edge
        .unwrap()
        .to_from
        .se3()
        .unwrap()
        .compose(&expected.inverse());
    assert!(error.translation.norm() < 1e-4 && error.rotation.angle() < 1e-4);
}
