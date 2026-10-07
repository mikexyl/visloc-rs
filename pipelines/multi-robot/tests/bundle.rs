use nalgebra::{UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;
use visloc_multi_robot::{backend::*, *};
fn transform(x: f64, y: f64) -> Transform {
    Transform::from(&SE3::new(
        UnitQuaternion::identity(),
        Vector3::new(x, y, 0.),
    ))
}
fn populate(b: &mut Backend, robot: &str, session: &str) {
    for id in (0..4).rev() {
        let key = Key::new(robot, session, id);
        let frame = BundleFrame {
            key: key.clone(),
            timestamp_ns: id as i64,
            views: (0..2)
                .map(|c| {
                    let camera = CameraModel {
                        width: 640,
                        height: 480,
                        intrinsics: [400., 420., 320., 240.],
                        camera_to_body: transform(c as f64 * 0.1, 0.),
                    };
                    let observations = (0..12)
                        .map(|track_id| {
                            let p = [
                                track_id as f64 * 0.1 - id as f64 * 0.4 - c as f64 * 0.1,
                                0.2,
                                4.,
                            ];
                            LandmarkObservation {
                                track_id,
                                pixel: [400. * p[0] / p[2] + 320., 420. * p[1] / p[2] + 240.],
                                point_camera: Some(p),
                            }
                        })
                        .collect();
                    CameraObservations {
                        camera,
                        observations,
                    }
                })
                .collect(),
        };
        assert!(b.insert_bundle_frame(frame.clone()).unwrap());
        assert!(!b.insert_bundle_frame(frame).unwrap());
        b.insert_keyframe(KeyframeRecord {
            key,
            timestamp_ns: id as i64,
            body_to_odom: transform(id as f64 * 0.4, id as f64 * 0.006),
            previous: (id > 0).then(|| Key::new(robot, session, id - 1)),
        })
        .unwrap();
    }
}
#[test]
fn global_ba_scopes_tracks_and_reanchors_joining_components() {
    let mut b = Backend::default();
    populate(&mut b, "a", "one");
    populate(&mut b, "b", "one");
    populate(&mut b, "a", "two");
    let raw = serde_json::to_vec(&b.records.values().collect::<Vec<_>>()).unwrap();
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(s.components, 3);
    assert_eq!(s.landmarks.len(), 36);
    assert_eq!(s.optimizer_reports.len(), 3);
    assert!(s
        .optimizer_reports
        .iter()
        .all(|r| r.landmarks == 12 && r.reprojection_factors == 96));
    assert!(s.final_cost < s.initial_cost);
    assert_eq!(
        raw,
        serde_json::to_vec(&b.records.values().collect::<Vec<_>>()).unwrap()
    );
    for p in &s.poses {
        if p.key.id == 0 {
            assert_eq!(p.body_to_map.translation, [0., 0., 0.]);
        }
    }
    assert_eq!(b.solve(&s).unwrap().landmarks.len(), 36);
    let mut conflicting = b.bundle_frames[&Key::new("a", "one", 0)].clone();
    conflicting.timestamp_ns += 1;
    assert!(b.insert_bundle_frame(conflicting).is_err());
    let a = Key::new("a", "one", 0);
    let z = Key::new("b", "one", 0);
    b.insert_loop(LoopConstraint {
        pair: Pair::new(a.clone(), z.clone()),
        from: a,
        to: z,
        to_from: transform(-5., 0.),
        information: matrix_values(&information(0.2, 0.04)),
        similarity: 0.9,
        verification: Verification {
            pnp_inliers: 30,
            ..Default::default()
        },
    })
    .unwrap();
    let joined = b.solve(&s).unwrap();
    assert_eq!(joined.components, 2);
    assert_eq!(joined.landmarks.len(), 36);
    assert_eq!(joined.optimizer_reports.len(), 2);
    let root = joined
        .poses
        .iter()
        .find(|p| p.key.robot == "b" && p.key.id == 0)
        .unwrap();
    assert!((root.body_to_map.translation[0] - 5.).abs() < 1e-5);
    assert!(joined.landmarks.iter().all(|p| p.component.robot == "a"));
}
#[test]
fn invalid_or_unconstrained_tracks_are_filtered_before_gtsam() {
    let mut b = Backend::default();
    populate(&mut b, "a", "s");
    for f in b.bundle_frames.values_mut() {
        for v in &mut f.views {
            // Track 0 becomes a zero-parallax ray; track 1 has an isolated outlier.
            v.observations[0].pixel = [320., 240.];
            v.observations[0].point_camera = None;
            if f.key.id == 3 {
                v.observations[1].pixel = [1., 1.];
            }
        }
    }
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(s.landmarks.len(), 11);
    assert!(s.bundle_diagnostics[0].rejected_observations >= 10);
    b.config.bundle_adjustment.max_observations = 4;
    assert!(b.solve(&s).unwrap_err().to_string().contains("capacity"));
}
#[test]
fn explicit_pose_graph_does_not_consume_ba_observations() {
    let mut b = Backend::default();
    b.config.mode = BackendMode::PoseGraph;
    assert!(!b
        .insert_bundle_frame(BundleFrame {
            key: Key::new("a", "s", 0),
            timestamp_ns: 0,
            views: vec![]
        })
        .unwrap());
    assert_eq!(b.input_revision, 0);
}
#[test]
fn global_ba_gps_aligns_poses_and_landmarks_and_preserves_raw_vio() {
    let mut b = Backend::default();
    populate(&mut b, "a", "s");
    for r in b.records.values_mut() {
        r.timestamp_ns = (r.key.id as i64 + 1) * 1_000_000_000;
    }
    for f in b.bundle_frames.values_mut() {
        f.timestamp_ns = (f.key.id as i64 + 1) * 1_000_000_000;
    }
    b.config.gps.enabled = true;
    b.config.gps.origin = Some(GpsDatum::from_lla([0.; 3]));
    let lever = Vector3::new(0.11, -0.095, -0.08);
    b.config.gps.lever_arms_m.insert("a".into(), lever.into());
    b.config.gps.min_alignment_fixes = 4;
    b.config.gps.alignment_span_m = 0.8;
    b.config.gps.min_distance_m = 0.1;
    b.config.gps.residual_deadband_sigma = 0.;
    let enu_from_odom = SE3::new(
        UnitQuaternion::from_euler_angles(0., 0., 0.4),
        Vector3::new(10., 20., 0.),
    );
    let raw = serde_json::to_vec(&b.records.values().collect::<Vec<_>>()).unwrap();
    // Warm-start from a visual BA before GPS became available.
    let visual = b.solve(&GraphSnapshot::default()).unwrap();
    assert!(visual.gps.aligned_components.is_empty());
    for id in 0..4 {
        let p = enu_from_odom.rotation * (Vector3::new(id as f64 * 0.4, 0., 0.) + lever)
            + enu_from_odom.translation;
        b.insert_gps(GpsRecord {
            key: Key::new("a", "s", id),
            timestamp_ns: (id as i64 + 1) * 1_000_000_000,
            receipt_timestamp_ns: (id as i64 + 1) * 1_000_000_000 + 50_000_000,
            time_source: "synthetic_measurement_utc".into(),
            lla: Some([
                (p.y / 6_335_439.32729282).to_degrees(),
                (p.x / 6_378_137.).to_degrees(),
                9000.,
            ]),
            status: 0,
            quality: Some(1),
            hdop: Some(0.8),
            covariance_enu: Some([0.25, 0., 0., 0., 0.25, 0., 0., 0., 1e8]),
        })
        .unwrap();
    }
    let s = b.solve(&visual).unwrap();
    assert_eq!(s.gps.aligned_components, vec![Key::new("a", "s", 0)]);
    assert_eq!(s.landmarks.len(), 12);
    assert_eq!(s.optimizer_reports[0].gps_factors, 4);
    assert_eq!(s.optimizer_reports[0].reprojection_factors, 96);
    assert_eq!(s.optimizer_reports[0].solver, "gtsam_global_ba");
    assert!(s.gps.diagnostics.iter().all(|d| d.reason == "active"));
    for p in &s.poses {
        let expected = enu_from_odom.rotation * Vector3::new(p.key.id as f64 * 0.4, 0., 0.)
            + enu_from_odom.translation;
        assert!((Vector3::from(p.body_to_map.translation) - expected).norm() < 0.02);
    }
    for p in &s.landmarks {
        let expected = enu_from_odom.rotation * Vector3::new(p.key.id as f64 * 0.1, 0.2, 4.)
            + enu_from_odom.translation;
        assert!((Vector3::from(p.position) - expected).norm() < 0.02);
    }
    assert_eq!(
        raw,
        serde_json::to_vec(&b.records.values().collect::<Vec<_>>()).unwrap()
    );
    let warm = b.solve(&s).unwrap();
    assert_eq!(warm.landmarks.len(), 12);
    assert_eq!(warm.gps.aligned_components, s.gps.aligned_components);
    assert!((warm.initial_cost - s.final_cost).abs() < 1e-8);
}

#[test]
fn legacy_snapshots_remain_pose_graph() {
    let current = GraphSnapshot::default();
    assert_eq!(current.backend_mode, BackendMode::GlobalBundleAdjustment);
    let mut json = serde_json::to_value(current).unwrap();
    json.as_object_mut().unwrap().remove("backend_mode");
    let legacy: GraphSnapshot = serde_json::from_value(json).unwrap();
    assert_eq!(legacy.backend_mode, BackendMode::PoseGraph);
}
