#include "ba_factors.h"
#include <limits>
#include <gtsam/inference/Ordering.h>
#include <chrono>
#include <cstring>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <set>

static int solve_joint(const VgPose *poses, size_t n,
                                  const VgBetween *edges, size_t ne,
                                  const VgGps *gps, size_t ng, uint64_t anchor,
                                  uint32_t horizontal,
                                  const VgLandmark *landmarks, size_t nl,
                                  const VgProjection *projections, size_t np,
                                  VgLandmark *landmark_output, VgPose *output,
                                  VgGpsDiagnostic *diagnostics,
                                  VgReport *report, char *error,
                                  size_t error_capacity) {
  try {
    using namespace visloc_gtsam;
    auto start = std::chrono::steady_clock::now();
    if (!poses || !output || !report || (ne && !edges) ||
        (ng && (!gps || !diagnostics)) || n == 0 || horizontal > 1 ||
        (nl && (!landmarks || !landmark_output)) || (np && !projections))
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
    std::map<uint64_t, Key> point_keys;
    Key next = initial_poses.rbegin()->first;
    for (size_t i = 0; i < nl; ++i) {
      Point3 point = Eigen::Map<const Point3>(landmarks[i].position);
      if (!point.allFinite() || next == std::numeric_limits<Key>::max() ||
          !point_keys.emplace(landmarks[i].id, ++next).second)
        throw std::invalid_argument("invalid or duplicate landmark ID");
      initial.insert(next, point);
    }
    std::map<Key, std::set<Key>> observers;
    for (size_t i = 0; i < np; ++i) {
      const auto &p = projections[i];
      if (!initial_poses.count(p.pose) || !point_keys.count(p.landmark) ||
          !std::isfinite(p.sigma) || p.sigma <= 0 ||
          !std::isfinite(p.huber) || p.huber <= 0 ||
          !Eigen::Map<const Vector4>(p.intrinsics).allFinite() ||
          !Eigen::Map<const Vector2>(p.pixel).allFinite() ||
          p.intrinsics[0] <= 0 || p.intrinsics[1] <= 0)
        throw std::invalid_argument("invalid projection factor");
      const auto key = point_keys.at(p.landmark);
      auto camera = initial_poses.at(p.pose).compose(pose(p.camera_to_body));
      if (camera.transformTo(initial.at<Point3>(key)).z() <= 0.)
        throw std::invalid_argument("initial landmark is behind camera");
      observers[key].insert(p.pose);
      graph.emplace_shared<BodyProjection>(root, p, key);
    }
    for (const auto &[id, key] : point_keys)
      if (observers[key].size() < 2)
        throw std::invalid_argument("landmark requires two distinct keyframes");
    LevenbergMarquardtParams params;
    params.setMaxIterations(20);
    params.setlambdaInitial(1e-4);
    params.setRelativeErrorTol(1e-7);
    params.setAbsoluteErrorTol(1e-9);
    params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
    if (nl) {
      KeyVector points;
      for (const auto &[id, key] : point_keys) points.push_back(key);
      params.ordering = Ordering::ColamdConstrainedFirst(graph, points);
    }
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
    auto body_at = [&](Key id) { return root.at(result, id); };
    for (const auto &[id, p] : initial_poses)
      if (!body_at(id).matrix().allFinite())
        throw std::runtime_error("non-finite GTSAM pose");
    Pose3 final_root = body_at(anchor);
    if ((horizontal && (std::abs(final_root.z() - root.initial.z()) > 1e-9 ||
                        (final_root.rotation().unrotate(Vector3::UnitZ()) -
                         root.initial.rotation().unrotate(Vector3::UnitZ()))
                                .norm() > 1e-9)) ||
        (!horizontal &&
         Pose3::Logmap(root.initial.between(final_root)).norm() > 1e-9))
      throw std::runtime_error("GTSAM changed fixed gauge");
    for (size_t i = 0; i < np; ++i) {
      const auto &p = projections[i];
      auto camera = body_at(p.pose).compose(pose(p.camera_to_body));
      auto point = result.at<Point3>(point_keys.at(p.landmark));
      if (!point.allFinite() || camera.transformTo(point).z() <= 0.)
        throw std::runtime_error("invalid optimized landmark cheirality");
    }
    // No output is consumed by Rust unless this function returns success.
    for (size_t i = 0; i < n; ++i)
      output[i] = wire(poses[i].id, body_at(poses[i].id));
    for (size_t i = 0; i < nl; ++i) {
      landmark_output[i].id = landmarks[i].id;
      Eigen::Map<Point3>(landmark_output[i].position) =
          result.at<Point3>(point_keys.at(landmarks[i].id));
    }
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

extern "C" int visloc_gtsam_solve(const VgPose *p, size_t n,
    const VgBetween *e, size_t ne, const VgGps *g, size_t ng,
    uint64_t anchor, uint32_t horizontal, VgPose *out,
    VgGpsDiagnostic *diagnostics, VgReport *report, char *error, size_t capacity) {
  return solve_joint(p, n, e, ne, g, ng, anchor, horizontal,
                     nullptr, 0, nullptr, 0, nullptr, out,
                     diagnostics, report, error, capacity);
}
extern "C" int visloc_gtsam_solve_bundle(const VgPose *p, size_t n,
    const VgBetween *e, size_t ne, const VgGps *g, size_t ng,
    uint64_t anchor, uint32_t horizontal, const VgLandmark *landmarks, size_t nl,
    const VgProjection *projections, size_t np, VgPose *out, VgLandmark *points,
    VgGpsDiagnostic *diagnostics, VgReport *report, char *error, size_t capacity) {
  return solve_joint(p, n, e, ne, g, ng, anchor, horizontal,
                     landmarks, nl, projections, np, points, out,
                     diagnostics, report, error, capacity);
}
