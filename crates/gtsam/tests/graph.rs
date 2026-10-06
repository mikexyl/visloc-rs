//! Native GTSAM ABI, pose convention, robustness and exception tests.
use nalgebra::{Matrix3, Matrix6, UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;
use visloc_gtsam::{GpsPositionFactor, GpsRobustKernel, PoseGraph};

fn graph(gps: bool) -> PoseGraph {
    let mut g = PoseGraph::new();
    for i in 0..12 {
        let p = SE3::new(
            UnitQuaternion::from_euler_angles(0.3, -0.2, i as f64 * 0.07),
            Vector3::new(i as f64 * 3., 0.2 * (i * i) as f64, 2.),
        );
        g.add_pose(i, p);
        if i > 0 {
            let z = g.poses[&i].inverse().compose(&g.poses[&(i - 1)]);
            let mut information = Matrix6::identity() * 20.;
            information[(0, 4)] = 0.7;
            information[(4, 0)] = 0.7;
            g.add_edge_with_information(i - 1, i, z, information);
        }
        if gps && i > 0 {
            g.gps_factors.push(GpsPositionFactor {
                from: i - 1,
                to: i,
                alpha: 0.3,
                position: Vector3::new(
                    (i as f64 - 0.7) * 3. + 3.,
                    0.2 * (i as f64 - 0.7).powi(2) - 2.,
                    1e7,
                ),
                lever_arm: Vector3::new(0.11, -0.095, -0.08),
                information: Matrix3::from_diagonal(&Vector3::new(1., 1., 0.)),
                kernel: GpsRobustKernel::Switchable { lambda: 9. },
                deadband_sigma: 1.,
            });
        }
    }
    let mut z = g.poses[&11].inverse().compose(&g.poses[&0]);
    z.translation.x += 0.6;
    g.add_edge_with_information(0, 11, z, Matrix6::identity() * 10.);
    g.anchor(0);
    if gps {
        g.horizontal_anchor = true;
    }
    g
}

fn independent_relative_cost(g: &PoseGraph) -> f64 {
    g.edges
        .iter()
        .map(|e| {
            let predicted = g.poses[&e.to].inverse().compose(&g.poses[&e.from]);
            let r = e.to_from.inverse().compose(&predicted).log();
            let d2 = r.dot(&(e.information * r));
            if d2 <= 9. {
                d2
            } else {
                6. * d2.sqrt() - 9.
            }
        })
        .sum()
}

#[test]
fn gtsam_relative_measurement_and_cross_information_match_rust() {
    let mut g = graph(false);
    let root = g.poses[&0].clone();
    let before = independent_relative_cost(&g);
    let report = g.optimize().unwrap();
    assert!((before - report.initial_cost).abs() < 1e-9);
    assert!((independent_relative_cost(&g) - report.final_cost).abs() < 1e-9);
    assert!(report.final_cost < report.initial_cost);
    assert_eq!(report.gps_factors, 0);
    assert!(root.inverse().compose(&g.poses[&0]).log().norm() < 1e-9);
}

#[test]
fn gtsam_both_gps_losses_preserve_partial_gauge_and_native_cost() {
    for kernel in [
        GpsRobustKernel::Switchable { lambda: 9. },
        GpsRobustKernel::Huber { delta: 3. },
    ] {
        let mut g = graph(true);
        for f in &mut g.gps_factors {
            f.kernel = kernel;
        }
        let root = g.poses[&0].clone();
        let report = g.optimize().unwrap();
        let explicit_cost =
            independent_relative_cost(&g) + g.gps_diagnostics.iter().map(|d| d.cost).sum::<f64>();
        assert!((explicit_cost - report.final_cost).abs() < 1e-9);
        assert!(report.final_cost < report.initial_cost * 0.2);
        let after = g.poses[&0].clone();
        assert!((root.translation.z - after.translation.z).abs() < 1e-12);
        assert!(
            (root.rotation.inverse() * Vector3::z() - after.rotation.inverse() * Vector3::z())
                .norm()
                < 1e-12
        );
        assert!((root.translation - after.translation).norm() > 0.1);
        // Repeat on the same native interface with a warm start and a large
        // altitude perturbation: Up can never contribute to the objective.
        for f in &mut g.gps_factors {
            f.position.z = -1e10;
        }
        let second = g.optimize().unwrap();
        assert!((second.initial_cost - report.final_cost).abs() < 1e-9);
    }
}

#[test]
fn native_exception_retains_graph_and_next_call_recovers() {
    let mut g = graph(true);
    g.gps_factors[0].information[(2, 2)] = 1e-20;
    let before = g.poses.clone();
    let error = g.optimize().unwrap_err();
    assert!(error.0.contains("Up information"));
    assert_eq!(before, g.poses);
    g.gps_factors[0].information[(2, 2)] = 0.;
    assert!(g.optimize().is_ok());
}

#[test]
fn malformed_native_graph_is_rejected_without_pose_changes() {
    let mut g = graph(false);
    let before = g.poses.clone();
    g.edges[0].information[(0, 0)] = -1.;
    assert!(g.optimize().unwrap_err().0.contains("non-positive edge"));
    assert_eq!(before, g.poses);
    g.edges[0].information[(0, 0)] = 20.;
    g.edges[0].to = 999;
    assert!(g.optimize().unwrap_err().0.contains("endpoints"));
    assert_eq!(before, g.poses);
}
