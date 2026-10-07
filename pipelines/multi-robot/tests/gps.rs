use nalgebra::{UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;
use visloc_multi_robot::{backend::Backend, gps::*, *};

fn key(id: u64) -> Key {
    Key::new("robot", "session", id)
}
fn record(id: u64, x: f64) -> KeyframeRecord {
    KeyframeRecord {
        key: key(id),
        timestamp_ns: 1_000_000_000 + id as i64 * 1_000_000_000,
        body_to_odom: Transform::from(&SE3::new(
            UnitQuaternion::identity(),
            Vector3::new(x, 0., 1.),
        )),
        previous: if id > 0 { Some(key(id - 1)) } else { None },
    }
}
fn fix(id: u64, x: f64, y: f64) -> GpsRecord {
    GpsRecord {
        key: key(id),
        timestamp_ns: 1_000_000_000 + id as i64 * 1_000_000_000,
        receipt_timestamp_ns: 1_050_000_000 + id as i64 * 1_000_000_000,
        time_source: "gnss_utc".into(),
        lla: Some([
            y / 6_335_439.32729282 * 180. / std::f64::consts::PI,
            x / 6_378_137. * 180. / std::f64::consts::PI,
            0.,
        ]),
        status: 0,
        quality: Some(2),
        hdop: Some(0.6),
        covariance_enu: None,
    }
}
fn backend() -> Backend {
    let mut b = Backend::default();
    b.config.mode = BackendMode::PoseGraph;
    b.config.gps.enabled = true;
    b.config.gps.origin = Some(GpsDatum::from_lla([0., 0., 0.]));
    b.config
        .gps
        .lever_arms_m
        .insert("robot".into(), [0., 0., 0.]);
    b
}
#[test]
fn geodesy_known_axes_and_unknown_covariance() {
    assert!((ecef([0., 0., 0.]) - Vector3::new(6378137., 0., 0.)).norm() < 1e-9);
    assert!((ecef([90., 0., 0.]) - Vector3::new(0., 0., 6356752.314245179)).norm() < 1e-7);
    let datum = GpsDatum::from_lla([0., 0., 0.]);
    let p = datum.project(fix(0, 20., 30.).lla.unwrap());
    assert!((p.xy() - nalgebra::Vector2::new(20., 30.)).norm() < 1e-7);
    let config = GpsConfig::default();
    let r = fix(0, 0., 0.);
    let c = config.covariance(&r, datum);
    assert!((c[(0, 0)] - 25.0004).abs() < 1e-9);
    let mut bad = r.clone();
    bad.covariance_enu = Some([0.; 9]);
    assert!(bad.validate().is_err());
    bad.covariance_enu = None;
    bad.hdop = Some(4.);
    assert!((config.covariance(&bad, datum)[(0, 0)] - 400.0004).abs() < 1e-9);
    let mut known = r;
    known.lla = Some([30., 45., 100.]);
    known.covariance_enu = Some([1., 0.2, 0., 0.2, 2., 0., 0., 0., 9.]);
    let cov = config.covariance(&known, datum);
    let rotation = datum.rotation()
        * GpsDatum::from_lla(known.lla.unwrap())
            .rotation()
            .transpose();
    let horizontal = rotation.fixed_view::<2, 2>(0, 0).into_owned();
    let input = nalgebra::Matrix2::new(1., 0.2, 0.2, 2.);
    let expected =
        horizontal * input * horizontal.transpose() + nalgebra::Matrix2::identity() * 0.0004;
    assert!((cov.fixed_view::<2, 2>(0, 0) - expected).norm() < 1e-10);
    assert_eq!(cov[(2, 2)], 0.);
}
#[test]
fn gps_without_loops_releases_horizontal_gauge_and_preserves_height_gravity() {
    let mut b = backend();
    for i in 0..16 {
        b.insert_keyframe(record(i, 3. * i as f64)).unwrap();
        b.insert_gps(fix(i, 10., 20. + 3. * i as f64)).unwrap();
    }
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(s.gps.aligned_components, vec![key(0)]);
    assert!(s.final_cost <= s.initial_cost + 1e-9);
    for p in &s.poses {
        assert!((p.body_to_map.translation[0] - 10.).abs() < 0.02);
        assert!((p.body_to_map.translation[1] - (20. + 3. * p.key.id as f64)).abs() < 0.02);
        assert!((p.body_to_map.translation[2] - 1.).abs() < 1e-8);
        let up = p.body_to_map.se3().unwrap().rotation * Vector3::z();
        assert!((up - Vector3::z()).norm() < 1e-8);
    }
    let next = b.solve(&s).unwrap();
    assert!(
        (next.poses[0].body_to_map.se3().unwrap().translation
            - s.poses[0].body_to_map.se3().unwrap().translation)
            .norm()
            < 1e-6
    );
}
#[test]
fn gps_disabled_is_exact_and_duplicates_are_idempotent() {
    let mut off = backend();
    off.config = serde_json::from_value(serde_json::json!({
        "mode": "pose_graph", "gps": {"enabled": false}
    }))
    .unwrap();
    off.insert_keyframe(record(0, 0.)).unwrap();
    let revision = off.input_revision;
    assert!(!off.insert_gps(fix(0, 10., 20.)).unwrap());
    assert_eq!(off.input_revision, revision);
    let s = off.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(s.poses[0].body_to_map, record(0, 0.).body_to_odom);
    let mut b = backend();
    assert!(b.insert_gps(fix(0, 0., 0.)).unwrap());
    assert!(!b.insert_gps(fix(0, 0., 0.)).unwrap());
    assert!(b.insert_gps(fix(0, 1., 0.)).is_err());
    let mut other = fix(0, 0., 0.);
    other.key.session = "new_session".into();
    assert!(b.insert_gps(other).unwrap());
}
#[test]
fn pending_association_skips_and_short_motion() {
    let mut b = backend();
    for i in 0..12 {
        b.insert_keyframe(record(i, 0.1 * i as f64)).unwrap();
        b.insert_gps(fix(i, 0., i as f64)).unwrap();
    }
    b.insert_gps(fix(15, 0., 15.)).unwrap();
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    assert!(s.gps.aligned_components.is_empty());
    assert!(s
        .gps
        .diagnostics
        .iter()
        .any(|d| d.reason == "distance_skip"));
    assert!(s
        .gps
        .diagnostics
        .iter()
        .any(|d| d.reason == "pending_keyframes"));
    assert_eq!(s.poses[0].body_to_map, record(0, 0.).body_to_odom);
}
#[test]
fn gps_jump_is_downweighted_and_outage_reacquires_without_reset() {
    let mut b = backend();
    b.config.gps.residual_deadband_sigma = 0.;
    for i in 0..30 {
        b.insert_keyframe(record(i, 3. * i as f64)).unwrap();
        let mut f = fix(i, 3. * i as f64, 0.);
        if i == 15 {
            f = fix(i, 3. * i as f64, 200.);
        }
        b.insert_gps(f).unwrap();
    }
    let first = b.solve(&GraphSnapshot::default()).unwrap();
    let outlier = first
        .gps
        .diagnostics
        .iter()
        .find(|d| d.key.id == 15)
        .unwrap();
    assert!(outlier.robust_weight.unwrap() < 0.001);
    assert!(first
        .poses
        .iter()
        .all(|p| p.body_to_map.translation[1].abs() < 0.1));
    for i in 30..40 {
        b.insert_keyframe(record(i, 3. * i as f64)).unwrap();
    }
    let outage = b.solve(&first).unwrap();
    assert_eq!(outage.gps.aligned_components.len(), 1);
    b.insert_gps(fix(39, 117., 0.)).unwrap();
    let recovered = b.solve(&outage).unwrap();
    assert_eq!(recovered.gps.diagnostics.last().unwrap().reason, "active");
    assert_eq!(recovered.poses.len(), 40);
}
#[test]
fn arbitrary_receiver_heights_and_received_order_do_not_change_poses() {
    let mut a = backend();
    let mut b = a.clone();
    for i in 0..16 {
        a.insert_keyframe(record(i, 3. * i as f64)).unwrap();
        b.insert_keyframe(record(i, 3. * i as f64)).unwrap();
    }
    for i in 0..16 {
        let mut r = fix(i, 3. * i as f64, 0.);
        r.lla.as_mut().unwrap()[2] = 10.;
        a.insert_gps(r).unwrap();
    }
    for i in (0..16).rev() {
        let mut r = fix(i, 3. * i as f64, 0.);
        r.lla.as_mut().unwrap()[2] = -9000. + i as f64 * 2000.;
        b.insert_gps(r).unwrap();
    }
    let x = a.solve(&GraphSnapshot::default()).unwrap();
    let y = b.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(x.poses[0].body_to_map, y.poses[0].body_to_map);
    assert!((x.poses[0].body_to_map.translation[2] - 1.).abs() < 1e-10);
}

#[test]
fn gps_does_not_join_components_but_verified_bridge_preserves_enu() {
    for second_has_gps in [false, true] {
        let mut b = backend();
        b.config.gps.lever_arms_m.insert("second".into(), [0.; 3]);
        for i in 0..16 {
            b.insert_keyframe(record(i, 3. * i as f64)).unwrap();
            b.insert_gps(fix(i, 10. + 3. * i as f64, 20.)).unwrap();
            let mut r = record(i, 3. * i as f64);
            r.key.robot = "second".into();
            if let Some(p) = &mut r.previous {
                p.robot = "second".into();
            }
            b.insert_keyframe(r).unwrap();
            if second_has_gps {
                let mut g = fix(i, 30. + 3. * i as f64, 20.);
                g.key.robot = "second".into();
                b.insert_gps(g).unwrap();
            }
        }
        let before = b.solve(&GraphSnapshot::default()).unwrap();
        assert_eq!(before.components, 2);
        assert_eq!(
            before.gps.aligned_components.len(),
            if second_has_gps { 2 } else { 1 }
        );
        let a = key(0);
        let z = Key::new("second", "session", 0);
        b.insert_loop(LoopConstraint {
            pair: Pair::new(a.clone(), z.clone()),
            from: a.clone(),
            to: z,
            to_from: Transform::from(&SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(-20., 0., 0.),
            )),
            information: matrix_values(&information(0.2, 0.04)),
            similarity: 0.9,
            verification: Verification {
                pnp_inliers: 30,
                ..Default::default()
            },
        })
        .unwrap();
        let after = b.solve(&before).unwrap();
        assert_eq!(after.components, 1);
        assert_eq!(after.gps.aligned_components, vec![a]);
        for p in after.poses {
            let expected = if p.key.robot == "robot" { 10. } else { 30. } + 3. * p.key.id as f64;
            assert!((p.body_to_map.translation[0] - expected).abs() < 0.01);
            assert!((p.body_to_map.translation[1] - 20.).abs() < 0.01);
        }
    }
}

#[test]
fn time_brackets_are_deferred_bounded_and_session_local() {
    let mut b = backend();
    b.insert_keyframe(record(0, 0.)).unwrap();
    let mut g = fix(100, 1.5, 0.);
    g.timestamp_ns = 1_500_000_000;
    b.insert_gps(g.clone()).unwrap();
    assert_eq!(
        b.solve(&GraphSnapshot::default()).unwrap().gps.diagnostics[0].reason,
        "pending_keyframes"
    );
    b.insert_keyframe(record(1, 3.)).unwrap();
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    let d = &s.gps.diagnostics[0];
    assert_eq!(d.alpha, Some(0.5));
    assert_eq!(d.from, Some(key(0)));
    assert_eq!(d.to, Some(key(1)));
    g.key.session = "another".into();
    b.insert_gps(g).unwrap();
    let mut r = record(2, 9.);
    r.timestamp_ns = 5_000_000_000;
    b.insert_keyframe(r).unwrap();
    let mut gap = fix(3, 6., 0.);
    gap.timestamp_ns = 3_000_000_000;
    b.insert_gps(gap).unwrap();
    let mut early = fix(4, 0., 0.);
    early.timestamp_ns = 500_000_000;
    b.insert_gps(early).unwrap();
    let reasons: Vec<_> = b
        .solve(&s)
        .unwrap()
        .gps
        .diagnostics
        .into_iter()
        .map(|d| d.reason)
        .collect();
    assert!(reasons.contains(&"pending_keyframes".into()));
    assert!(reasons.contains(&"bracket_too_wide".into()));
    assert!(reasons.contains(&"before_first_keyframe".into()));
}

#[test]
fn no_fix_diagnostics_survive_without_a_datum_and_datum_cannot_change() {
    let mut b = backend();
    b.config.gps.origin = None;
    let mut r = fix(0, 0., 0.);
    r.status = -1;
    r.lla = None;
    b.insert_gps(r).unwrap();
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(s.gps.diagnostics[0].reason, "no_fix");
    assert!(s.gps.datum.is_none());
    b.insert_gps(fix(1, 0., 0.)).unwrap();
    let s = b.solve(&s).unwrap();
    assert!(s.gps.datum.is_some());
    b.config.gps.origin = Some(GpsDatum::from_lla([1., 0., 0.]));
    assert!(b.solve(&s).is_err());
}

#[test]
fn persistent_bad_segment_is_robust_but_constant_bias_is_not_observable() {
    let mut b = backend();
    b.config.gps.residual_deadband_sigma = 0.;
    for i in 0..40 {
        b.insert_keyframe(record(i, 3. * i as f64)).unwrap();
        b.insert_gps(fix(
            i,
            3. * i as f64,
            if (15..23).contains(&i) { 150. } else { 0. },
        ))
        .unwrap();
    }
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    assert_eq!(s.gps.aligned_components.len(), 1);
    assert!(s
        .poses
        .iter()
        .all(|p| p.body_to_map.translation[1].abs() < 0.2));
    assert!(s
        .gps
        .diagnostics
        .iter()
        .filter(|d| (15..23).contains(&d.key.id))
        .all(|d| d.robust_weight.unwrap() < 0.002));
    // A common GPS offset is indistinguishable from global translation: do not
    // promise robust kernels can reject this real-world failure mode.
    let mut biased = backend();
    for i in 0..16 {
        biased.insert_keyframe(record(i, 3. * i as f64)).unwrap();
        biased.insert_gps(fix(i, 3. * i as f64, 50.)).unwrap();
    }
    let s = biased.solve(&GraphSnapshot::default()).unwrap();
    assert!((s.poses[0].body_to_map.translation[1] - 50.).abs() < 0.01);
}

#[test]
fn tight_quality_gate_accepts_ordinary_gps_without_requiring_rtk() {
    let config = GpsConfig::default();
    let mut r = fix(0, 0., 0.);
    assert_eq!(config.quality_reason(&r), None);
    for quality in [Some(0), Some(6), Some(7), Some(8)] {
        r.quality = quality;
        assert!(config.quality_reason(&r).is_some());
    }
    for quality in [None, Some(1), Some(2), Some(4), Some(5)] {
        r.quality = quality;
        assert_eq!(config.quality_reason(&r), None);
    }
    r.quality = Some(1);
    r.hdop = None;
    assert_eq!(config.quality_reason(&r), Some("quality_hdop_unknown"));
    r.hdop = Some(1.6);
    assert_eq!(config.quality_reason(&r), Some("quality_hdop_exceeded"));
    r.hdop = Some(0.8);
    r.covariance_enu = Some([36., 0., 0., 0., 36., 0., 0., 0., 1e8]);
    assert_eq!(
        config.quality_reason(&r),
        Some("quality_covariance_exceeded")
    );
    r.covariance_enu = Some([0.01, 0., 0., 0., 0.01, 0., 0., 0., 1e8]);
    assert_eq!(config.quality_reason(&r), None);
    let mut b = backend();
    for i in 0..16 {
        b.insert_keyframe(record(i, 3. * i as f64)).unwrap();
        let mut g = fix(i, 0., 200.);
        g.hdop = Some(3.);
        b.insert_gps(g).unwrap();
    }
    let s = b.solve(&GraphSnapshot::default()).unwrap();
    assert!(s.gps.aligned_components.is_empty());
    assert!(s
        .gps
        .diagnostics
        .iter()
        .all(|d| d.reason == "quality_hdop_exceeded"));
    assert_eq!(s.poses[0].body_to_map, record(0, 0.).body_to_odom);
}
