#include "factors.h"
#include <chrono>
#include <cstring>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <set>

extern "C" int visloc_gtsam_solve(const VgPose *poses, size_t n,
                                  const VgBetween *edges, size_t ne,
                                  const VgGps *gps, size_t ng, uint64_t anchor,
                                  uint32_t horizontal, VgPose *output,
                                  VgGpsDiagnostic *diagnostics,
                                  VgReport *report, char *error,
                                  size_t error_capacity) {
  try {
    using namespace visloc_gtsam;
    auto start = std::chrono::steady_clock::now();
    if (!poses || !output || !report || (ne && !edges) ||
        (ng && (!gps || !diagnostics)) || n == 0 || horizontal > 1)
      throw std::invalid_argument("invalid native graph buffers");
    std::map<Key, Pose3> initial_poses;
    for (size_t i = 0; i < n; ++i)
      if (!initial_poses.emplace(poses[i].id, pose(poses[i])).second)
        throw std::invalid_argument("duplicate pose ID");
    if (!initial_poses.count(anchor))
      throw std::invalid_argument("missing anchor");
    Root root{anchor, initial_poses.at(anchor), horizontal != 0};
    Values initial;
    for (const auto &[id, p] : initial_poses) {
      if (id != anchor)
        initial.insert(id, p);
      else if (horizontal)
        initial.insert(id, Vector3(p.x(), p.y(), 0.));
    }
    NonlinearFactorGraph graph;
    for (size_t i = 0; i < ne; ++i) {
      const auto &e = edges[i];
      if (e.from == e.to || !initial_poses.count(e.from) ||
          !initial_poses.count(e.to))
        throw std::invalid_argument("invalid edge endpoints");
      Matrix6 input_info = information<6>(e.information), info;
      const int order[] = {3, 4, 5, 0, 1, 2};
      for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
          info(r, c) = input_info(order[r], order[c]);
      if (Eigen::LLT<Matrix6>(info).info() != Eigen::Success)
        throw std::invalid_argument("non-positive edge information");
      auto model =
          noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(3.),
                                     noiseModel::Gaussian::Information(info));
      if (e.from == anchor || e.to == anchor)
        graph.emplace_shared<RootBetween>(root, e.from, e.to,
                                          pose(e.measurement), model);
      else
        graph.emplace_shared<BetweenFactor<Pose3>>(e.to, e.from,
                                                   pose(e.measurement), model);
    }
    std::vector<std::shared_ptr<HorizontalGps>> gps_factors;
    for (size_t i = 0; i < ng; ++i) {
      if (!initial_poses.count(gps[i].from) || !initial_poses.count(gps[i].to))
        throw std::invalid_argument("invalid GPS endpoints");
      auto f = std::make_shared<HorizontalGps>(root, gps[i]);
      gps_factors.push_back(f);
      graph.push_back(f);
    }
    LevenbergMarquardtParams params;
    params.setMaxIterations(20);
    params.setlambdaInitial(1e-4);
    params.setRelativeErrorTol(1e-7);
    params.setAbsoluteErrorTol(1e-9);
    params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
    double before = 2. * graph.error(initial);
    Values result = initial;
    size_t iterations = 0;
    if (!initial.empty() && !graph.empty()) {
      LevenbergMarquardtOptimizer optimizer(graph, initial, params);
      result = optimizer.optimize();
      iterations = optimizer.iterations();
    }
    double after = 2. * graph.error(result);
    if (!std::isfinite(before) || !std::isfinite(after) ||
        after > before + 1e-9)
      throw std::runtime_error("non-finite or increasing GTSAM objective");
    for (const auto &[id, p] : initial_poses)
      if (!root.at(result, id).matrix().allFinite())
        throw std::runtime_error("non-finite GTSAM pose");
    Pose3 final_root = root.at(result, anchor);
    if ((horizontal && (std::abs(final_root.z() - root.initial.z()) > 1e-9 ||
                        (final_root.rotation().unrotate(Vector3::UnitZ()) -
                         root.initial.rotation().unrotate(Vector3::UnitZ()))
                                .norm() > 1e-9)) ||
        (!horizontal &&
         Pose3::Logmap(root.initial.between(final_root)).norm() > 1e-9))
      throw std::runtime_error("GTSAM changed fixed gauge");
    // No output is consumed by Rust unless this function returns success.
    for (size_t i = 0; i < n; ++i)
      output[i] = wire(poses[i].id, root.at(result, poses[i].id));
    for (size_t i = 0; i < ng; ++i)
      diagnostics[i] = gps_factors[i]->diagnostic(result);
    *report = {before, after,
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count(),
               iterations};
    return 0;
  } catch (const std::exception &e) {
    if (error && error_capacity) {
      std::strncpy(error, e.what(), error_capacity - 1);
      error[error_capacity - 1] = '\0';
    }
    return 1;
  } catch (...) {
    if (error && error_capacity) {
      std::strncpy(error, "unknown GTSAM exception", error_capacity - 1);
      error[error_capacity - 1] = '\0';
    }
    return 2;
  }
}
