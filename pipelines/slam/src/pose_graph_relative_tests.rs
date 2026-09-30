//! Exact relative-pose derivatives and endpoint inversion invariance.
use super::*;

fn fixture() -> PoseGraph {
    let mut graph = PoseGraph::new();
    graph.anchor(0);
    for (id, x) in [
        Vector6::new(1., -2., 0.5, 0.1, -0.3, 0.2),
        Vector6::new(-0.8, 1.2, 2., 0.4, 0.2, -0.1),
        Vector6::new(0.3, 2., -1., -0.5, 0.6, 0.2),
    ]
    .iter()
    .enumerate()
    {
        graph.add_pose(
            id as u64,
            Pose {
                world_to_camera: SE3::exp(x),
            },
        );
    }
    let mut l = Matrix6::from_diagonal(&Vector6::new(5., 4., 6., 25., 20., 15.));
    l[(4, 0)] = 2.;
    let information = l * l.transpose();
    for (from, to, kind) in [
        (0, 1, PoseGraphEdgeKind::Sequential),
        (0, 2, PoseGraphEdgeKind::LoopClosure),
    ] {
        graph.add_edge_with_information(
            from,
            to,
            SE3::exp(&Vector6::new(-1., 0.3, 0.5, 0.1, -0.2, 0.3)),
            kind,
            information,
        );
    }
    graph
}

fn system(graph: &PoseGraph, kernel: RobustKernel) -> (DMatrix<f64>, DVector<f64>) {
    let layout = BTreeMap::from([(1, 0), (2, 1)]);
    let (h, g) = graph.assemble_se3_system(&layout, 12, &kernel, None, LinearSolver::Dense);
    let NormalEquations6::Dense(h) = h else {
        unreachable!()
    };
    (h, g)
}

#[test]
fn relative_factor_gradient_matches_finite_difference_of_robust_objective() {
    for scalar_edges in [false, true] {
        let mut graph = fixture();
        if scalar_edges {
            for edge in &mut graph.edges {
                edge.information = None;
                edge.weight = 2.5;
            }
        }
        for kernel in [
            RobustKernel::None,
            RobustKernel::Huber { delta: 3. },
            RobustKernel::Cauchy { c: 2. },
        ] {
            let (_, g) = system(&graph, kernel);
            let mut numerical = DVector::zeros(12);
            for k in 0..12 {
                let id = (k / 6 + 1) as u64;
                let mut step = Vector6::zeros();
                step[k % 6] = 1e-6;
                let mut plus = graph.clone();
                let mut minus = graph.clone();
                plus.poses.get_mut(&id).unwrap().world_to_camera =
                    graph.poses[&id].world_to_camera.compose(&SE3::exp(&step));
                minus.poses.get_mut(&id).unwrap().world_to_camera =
                    graph.poses[&id].world_to_camera.compose(&SE3::exp(&-step));
                numerical[k] =
                    (plus.robust_se3_cost(&kernel) - minus.robust_se3_cost(&kernel)) / 2e-6;
            }
            // Public cost is r^T Omega r; normal-equation g is half its gradient.
            let error = (&numerical - 2. * g).norm() / numerical.norm();
            assert!(
                error < 2e-7,
                "relative objective gradient error={error}, scalar={scalar_edges}, kernel={kernel:?}"
            );
        }
    }
}

#[test]
fn reversing_loop_endpoints_and_transporting_information_preserves_factor() {
    let graph = fixture();
    let mut reversed = graph.clone();
    for edge in &mut reversed.edges {
        // r_reverse = -Ad_Z r; Omega_reverse = Ad_Z^-T Omega Ad_Z^-1.
        let a = edge.measurement.inverse().adjoint();
        edge.information = edge.information.map(|omega| a.transpose() * omega * a);
        edge.measurement = edge.measurement.inverse();
        std::mem::swap(&mut edge.from, &mut edge.to);
    }
    for kernel in [RobustKernel::None, RobustKernel::Huber { delta: 3. }] {
        assert!((graph.robust_se3_cost(&kernel) - reversed.robust_se3_cost(&kernel)).abs() < 1e-8);
        let (h, g) = system(&graph, kernel);
        let (hr, gr) = system(&reversed, kernel);
        assert!((&h - hr).norm() / h.norm() < 1e-10);
        assert!((&g - gr).norm() / g.norm() < 1e-10);
    }
}
