use nalgebra::{Matrix3, Matrix6, UnitQuaternion, Vector3};
use std::collections::BTreeMap;
use visloc_core::geometry::SE3;
use visloc_gtsam::{GpsPositionFactor, GpsRobustKernel, PoseGraph, ProjectionFactor};

fn scene() -> (
    PoseGraph,
    BTreeMap<u64, Vector3<f64>>,
    Vec<ProjectionFactor>,
    Vec<SE3>,
) {
    let truth: Vec<_> = (0..5)
        .map(|i| {
            SE3::new(
                UnitQuaternion::from_euler_angles(
                    0.01 * i as f64,
                    -0.02 * i as f64,
                    0.03 * i as f64,
                ),
                Vector3::new(i as f64 * 0.4, 0., 0.),
            )
        })
        .collect();
    let mut g = PoseGraph::new();
    for (i, t) in truth.iter().enumerate() {
        let mut p = t.clone();
        if i > 0 {
            p.translation.y += 0.06 * i as f64;
        }
        g.add_pose(i as u64, p);
        if i > 0 {
            g.add_edge_with_information(
                i as u64 - 1,
                i as u64,
                t.inverse().compose(&truth[i - 1]),
                Matrix6::identity() * 100.,
            );
        }
    }
    g.anchor(0);
    let mut points = BTreeMap::new();
    let mut factors = vec![];
    for id in 0..30 {
        let p = Vector3::new(
            (id % 6) as f64 * 0.3 - 0.5,
            (id / 6) as f64 * 0.25 - 0.5,
            4. + id as f64 * 0.04,
        );
        points.insert(id, p + Vector3::new(0.02, -0.015, 0.1));
        for (i, pose) in truth.iter().enumerate() {
            for camera in 0..2 {
                let extrinsic = SE3::new(
                    UnitQuaternion::from_euler_angles(0.02, 0.01, -0.04),
                    Vector3::new(0.03 + camera as f64 * 0.095095, -0.02, 0.01),
                );
                let [fx, fy, cx, cy] = [420. + camera as f64 * 30., 415., 320., 240.];
                let local = pose.compose(&extrinsic).inverse();
                let q = local.rotation * p + local.translation;
                factors.push(ProjectionFactor {
                    pose: i as u64,
                    landmark: id,
                    camera_to_body: extrinsic,
                    intrinsics: [fx, fy, cx, cy],
                    pixel: [fx * q.x / q.z + cx, fy * q.y / q.z + cy],
                    sigma: 1.,
                    huber: 3.,
                });
            }
        }
    }
    (g, points, factors, truth)
}
#[test]
fn joint_stereo_ba_moves_poses_and_points_with_fixed_body_gauge() {
    let (mut graph, mut points, factors, truth) = scene();
    let before = points.clone();
    let root = graph.poses[&0].clone();
    let report = graph.optimize_bundle(&mut points, &factors).unwrap();
    assert_eq!(report.solver, "gtsam_global_ba");
    assert_eq!(report.landmarks, 30);
    assert_eq!(report.reprojection_factors, 300);
    assert!(report.final_cost < report.initial_cost * 1e-6);
    assert!(report.reprojection_rmse_after_px.unwrap() < 1e-4);
    assert_eq!(root, graph.poses[&0]);
    assert!((graph.poses[&4].translation - truth[4].translation).norm() < 1e-4);
    assert!((before[&0] - points[&0]).norm() > 0.05);
    // Landmark IDs deliberately collide with body IDs: separate native keys.
    let again = graph.optimize_bundle(&mut points, &factors).unwrap();
    assert!(again.final_cost <= again.initial_cost + 1e-10);
}
#[test]
fn bad_projection_retains_both_poses_and_landmarks_atomically() {
    let (mut graph, mut points, mut factors, _) = scene();
    let before_points = points.clone();
    let before_poses = graph.poses.clone();
    factors[0].pose = 999;
    assert!(graph.optimize_bundle(&mut points, &factors).is_err());
    assert_eq!(before_points, points);
    assert_eq!(before_poses, graph.poses);
    factors[0].pose = 0;
    points.get_mut(&0).unwrap().z = -1.;
    let before = points.clone();
    assert!(graph.optimize_bundle(&mut points, &factors).is_err());
    assert_eq!(before, points);
    assert_eq!(before_poses, graph.poses);
}
#[test]
fn robust_projection_loss_tolerates_an_outlier() {
    let (mut graph, mut points, mut factors, truth) = scene();
    factors[15].pixel[0] += 100.;
    let r = graph.optimize_bundle(&mut points, &factors).unwrap();
    assert!(r.final_cost < r.initial_cost);
    assert!((graph.poses[&4].translation - truth[4].translation).norm() < 0.04);
}

#[test]
fn gps_and_stereo_reprojections_jointly_optimize_with_partial_gauge() {
    for kernel in [
        GpsRobustKernel::Huber { delta: 3. },
        GpsRobustKernel::Switchable { lambda: 9. },
    ] {
        let (mut graph, mut points, projections, truth) = scene();
        let shift = SE3::new(
            UnitQuaternion::from_euler_angles(0., 0., 0.2),
            Vector3::new(1., -0.8, 0.),
        );
        for pose in graph.poses.values_mut() {
            *pose = shift.compose(pose);
        }
        for point in points.values_mut() {
            *point = shift.rotation * *point + shift.translation;
        }
        let root = graph.poses[&0].clone();
        graph.horizontal_anchor = true;
        let lever = Vector3::new(0.11, -0.095, -0.08);
        for i in 1..truth.len() {
            let alpha = 0.35;
            let a = &truth[i - 1];
            let b = &truth[i];
            let mut antenna = (1. - alpha) * a.translation
                + alpha * b.translation
                + a.rotation.slerp(&b.rotation, alpha) * lever;
            antenna.z = 1e6; // Horizontal GPS must ignore arbitrary receiver heights.
            graph.gps_factors.push(GpsPositionFactor {
                from: i as u64 - 1,
                to: i as u64,
                alpha,
                position: antenna,
                lever_arm: lever,
                information: Matrix3::from_diagonal(&Vector3::new(25., 25., 0.)),
                kernel,
                deadband_sigma: 0.,
            });
        }
        let report = graph.optimize_bundle(&mut points, &projections).unwrap();
        assert_eq!(report.solver, "gtsam_global_ba");
        assert_eq!(report.gps_factors, 4);
        assert_eq!(report.reprojection_factors, 300);
        assert_eq!(report.landmarks, 30);
        assert!(report.final_cost < report.initial_cost * 1e-6);
        assert!(report.reprojection_rmse_after_px.unwrap() < 1e-4);
        for (id, pose) in &graph.poses {
            assert!(truth[*id as usize].inverse().compose(pose).log().norm() < 1e-4);
        }
        assert!((root.translation - graph.poses[&0].translation).norm() > 1.);
        assert_eq!(root.translation.z, graph.poses[&0].translation.z);
        assert!(
            (root.rotation.inverse() * Vector3::z()
                - graph.poses[&0].rotation.inverse() * Vector3::z())
            .norm()
                < 1e-12
        );
        assert_eq!(graph.gps_diagnostics.len(), 4);
        for fix in &mut graph.gps_factors {
            fix.position.z = -1e10;
        }
        let warm = graph.optimize_bundle(&mut points, &projections).unwrap();
        assert!((warm.initial_cost - report.final_cost).abs() < 1e-9);
    }
}
