#include "ba_factors.h"
#include <iostream>
#include <tuple>
#include <limits>

using namespace visloc_gtsam;
void require(bool yes, const char *message) {
  if (!yes)
    throw std::runtime_error(message);
}
VgGps fix(Key from, Key to, double alpha) {
  VgGps out{};
  out.from = from;
  out.to = to;
  out.alpha = alpha;
  double p[] = {12., 4., -1e6}, l[] = {.11, -.095, -.08},
         info[] = {.1, .02, 0., .02, .15, 0., 0., 0., 0.};
  std::copy(p, p + 3, out.position);
  std::copy(l, l + 3, out.lever);
  std::copy(info, info + 9, out.information);
  out.kernel = 1;
  out.parameter = 9.;
  out.deadband = 1.;
  return out;
}
void check_derivatives(const NoiseModelFactor &factor, const Root &root,
                       const Values &values, Key point = std::numeric_limits<Key>::max()) {
  std::vector<Matrix> h(factor.keys().size());
  factor.unwhitenedError(values, &h);
  for (size_t i = 0; i < factor.keys().size(); ++i) {
    Key k = factor.keys()[i];
    int n = (k == root.key || k == point) ? 3 : 6;
    Matrix numeric(h[i].rows(), n);
    for (int j = 0; j < n; ++j) {
      Values plus(values), minus(values);
      double eps = 1e-6;
      if (k == root.key || k == point) {
        Vector3 d = Vector3::Zero();
        d[j] = eps;
        plus.update(k, Vector3(values.at<Vector3>(k) + d));
        minus.update(k, Vector3(values.at<Vector3>(k) - d));
      } else {
        Vector6 d = Vector6::Zero();
        d[j] = eps;
        plus.update(k, values.at<Pose3>(k).retract(d));
        minus.update(k, values.at<Pose3>(k).retract(-d));
      }
      numeric.col(j) =
          (factor.unwhitenedError(plus) - factor.unwhitenedError(minus)) /
          (2. * eps);
    }
    if ((numeric - h[i]).norm() > 3e-6) {
      std::cerr << "key " << k << " analytic:\n"
                << h[i] << "\nnumeric:\n"
                << numeric << '\n';
      throw std::runtime_error("analytical endpoint Jacobian mismatch");
    }
  }
}
int main() {
  try {
    Root root{0, Pose3(Rot3::Ypr(.3, -.2, .25), Point3(0., 0., 2.)), true};
    Values values;
    values.insert(0, Vector3(.8, -.3, .27));
    values.insert(1, Pose3(Rot3::Ypr(.33, -.2, .25), Point3(3., .1, 2.3)));
    values.insert(2, Pose3(Rot3::Ypr(.36, -.2, .25), Point3(6., .4, 2.6)));
    size_t checks = 0;
    for (double angle : {0., .7, 3.13})
      for (auto [a, b, alpha] : {std::tuple<Key, Key, double>{1, 2, 0.},
                                 {1, 2, .35},
                                 {1, 2, 1.},
                                 {1, 1, .5},
                                 {0, 1, .4},
                                 {0, 0, 0.}}) {
        Values trial(values);
        if (a != b) {
          Pose3 pa = root.at(trial, a), pb = root.at(trial, b);
          Rot3 r = pa.rotation().compose(
              Rot3::Expmap(Vector3(.2, .7, -.3).normalized() * angle));
          trial.update(b, Pose3(r, pb.translation()));
        }
        HorizontalGps f(root, fix(a, b, alpha));
        check_derivatives(f, root, trial);
        ++checks;
      }
    for (uint32_t kernel : {0u, 1u})
      for (double deadband : {0., 1., 100.}) {
        auto input = fix(1, 2, .35);
        input.kernel = kernel;
        input.parameter = kernel ? 9. : 3.;
        input.deadband = deadband;
        HorizontalGps f(root, input);
        auto diagnostic = f.diagnostic(values);
        Pose3 pi =
            interpolate(root.at(values, 1), root.at(values, 2), input.alpha);
        Vector3 arm = Eigen::Map<const Vector3>(input.lever);
        Vector3 r =
            pi.transformFrom(arm) - Eigen::Map<const Vector3>(input.position);
        double d2 = r.dot(information<3>(input.information) * r);
        double u = std::max(0., std::sqrt(d2) - deadband),
               s = 9. / (9. + u * u);
        double cost = kernel ? s * s * u * u + 9. * (1. - s) * (1. - s)
                             : (u <= 3. ? u * u : 6. * u - 9.);
        require(std::abs(2. * f.error(values) - cost) < 1e-11,
                "robust cost differs from explicit switch/Huber");
        require(std::abs(diagnostic.cost - cost) < 1e-11,
                "GPS diagnostic cost mismatch");
        input.position[2] = 1e12;
        HorizontalGps changed(root, input);
        require(std::abs(2. * changed.error(values) - cost) < 1e-11,
                "GPS height affected objective");
        if (deadband == 100.) {
          std::vector<Matrix> h(f.keys().size());
          f.unwhitenedError(values, &h);
          for (const auto &block : h)
            require(block.norm() == 0., "force inside deadband");
        }
        ++checks;
      }
    // Root reparameterization of the native BetweenFactor with a nonzero
    // residual checks its finite-rotation Jacobian, not just the zero case.
    Pose3 z(Rot3::Ypr(.1, -.2, .05), Point3(-3.2, -.1, .2));
    RootBetween between(root, 0, 1, z, noiseModel::Unit::Create(6));
    check_derivatives(between, root, values);
    require((between.unwhitenedError(values) -
             Pose3::Logmap(
                 z.between(root.at(values, 1).between(root.at(values, 0)))))
                    .norm() < 1e-12,
            "relative direction");
    ++checks;
    auto bad = fix(1, 2, .5);
    bad.information[8] = 1e-20;
    bool rejected = false;
    try {
      HorizontalGps f(root, bad);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "GPS Up information accepted");
    ++checks;
    for (bool horizontal : {false, true}) {
      Root gauge = root;
      gauge.horizontal = horizontal;
      for (Key body : {Key(0), Key(1)}) {
        Values trial(values);
        trial.insert(100, Point3(1., 2., 9.));
        VgProjection projection{};
        projection.pose = body;
        projection.landmark = 100;
        projection.camera_to_body = wire(0, Pose3(Rot3::Ypr(.1, -.02, .03), Point3(.1, -.03, .02)));
        projection.intrinsics[0] = 400.; projection.intrinsics[1] = 420.;
        projection.intrinsics[2] = 320.; projection.intrinsics[3] = 240.;
        projection.pixel[0] = 110.; projection.pixel[1] = 230.;
        projection.sigma = 1.5; projection.huber = 3.;
        BodyProjection factor(gauge, projection, 100);
        check_derivatives(factor, gauge, trial, 100);
        ++checks;
      }
    }
    char error[256]{};
    VgReport report{};
    require(visloc_gtsam_solve(nullptr, 0, nullptr, 0, nullptr, 0, 0, 0,
                               nullptr, nullptr, &report, error,
                               sizeof(error)) != 0,
            "null ABI accepted");
    ++checks;
    std::cout << checks << " native GTSAM factor/ABI checks passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
