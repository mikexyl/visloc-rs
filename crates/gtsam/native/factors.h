#pragma once
#include "bridge.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <cmath>
#include <gtsam/config.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <map>
#include <stdexcept>

static_assert(GTSAM_VERSION_MAJOR == 4 && GTSAM_VERSION_MINOR == 3 &&
                  GTSAM_VERSION_PATCH == 0,
              "visloc pins GTSAM 4.3.0");
namespace visloc_gtsam {
using namespace gtsam;

inline Pose3 pose(const VgPose &p) {
  Eigen::Quaterniond q(p.quaternion_xyzw[3], p.quaternion_xyzw[0],
                       p.quaternion_xyzw[1], p.quaternion_xyzw[2]);
  Vector3 t = Eigen::Map<const Vector3>(p.translation);
  if (!q.coeffs().allFinite() || !t.allFinite() ||
      std::abs(q.norm() - 1.) > 1e-6)
    throw std::invalid_argument("non-finite or non-rigid pose");
  return Pose3(Rot3(q.normalized()), t);
}
inline VgPose wire(Key id, const Pose3 &p) {
  VgPose out{};
  out.id = id;
  Eigen::Map<Vector3>(out.translation) = p.translation();
  Eigen::Map<Vector4>(out.quaternion_xyzw) =
      p.rotation().toQuaternion().coeffs();
  return out;
}
template <int N>
inline Eigen::Matrix<double, N, N> information(const double *data) {
  Eigen::Matrix<double, N, N> m =
      Eigen::Map<const Eigen::Matrix<double, N, N, Eigen::RowMajor>>(data);
  if (!m.allFinite() || (m - m.transpose()).norm() > 1e-9)
    throw std::invalid_argument("invalid information matrix");
  return m;
}
inline Matrix3 skew(const Vector3 &v) { return skewSymmetric(v); }
inline Matrix3 left_jacobian(const Vector3 &v) {
  double t2 = v.squaredNorm(), a, b;
  if (t2 < 1e-8) {
    a = .5 - t2 / 24. + t2 * t2 / 720.;
    b = 1. / 6. - t2 / 120. + t2 * t2 / 5040.;
  } else {
    double t = std::sqrt(t2);
    a = (1. - std::cos(t)) / t2;
    b = (t - std::sin(t)) / (t2 * t);
  }
  Matrix3 k = skew(v);
  return Matrix3::Identity() + a * k + b * k * k;
}

// Exact root parameterization: a constant pose, or EN/yaw with height and
// gravity direction preserved. Every other variable is a native Pose3.
struct Root {
  Key key;
  Pose3 initial;
  bool horizontal;
  bool variable(Key k) const { return k != key || horizontal; }
  KeyVector active(Key a, Key b) const {
    KeyVector out;
    if (variable(a))
      out.push_back(a);
    if (b != a && variable(b))
      out.push_back(b);
    return out;
  }
  Pose3 at(const Values &v, Key k) const {
    if (k != key)
      return v.at<Pose3>(k);
    if (!horizontal)
      return initial;
    Vector3 x = v.at<Vector3>(key);
    return Pose3(Rot3::Rz(x.z()).compose(initial.rotation()),
                 Point3(x.x(), x.y(), initial.z()));
  }
  Matrix derivative(const Pose3 &p, Key k) const {
    if (k != key)
      return Matrix6::Identity();
    Matrix out = Matrix::Zero(6, 3);
    Matrix3 rt = p.rotation().matrix().transpose();
    out.block<3, 1>(0, 2) = rt.col(2);
    out.block<3, 2>(3, 0) = rt.leftCols<2>();
    return out;
  }
};

inline Pose3 interpolate(const Pose3 &a, const Pose3 &b, double alpha,
                         Matrix6 *ja = nullptr, Matrix6 *jb = nullptr) {
  Vector3 phi = Rot3::Logmap(a.rotation().between(b.rotation()));
  Rot3 ri = a.rotation().compose(Rot3::Expmap(alpha * phi));
  if (ja && jb) {
    Matrix3 ra = a.rotation().matrix(), rb = b.rotation().matrix(),
            rit = ri.matrix().transpose();
    Matrix3 blend = ra * left_jacobian(alpha * phi) * alpha *
                    left_jacobian(phi).inverse() * ra.transpose();
    ja->setZero();
    jb->setZero();
    ja->topLeftCorner<3, 3>() = rit * (Matrix3::Identity() - blend) * ra;
    jb->topLeftCorner<3, 3>() = rit * blend * rb;
    ja->bottomRightCorner<3, 3>() = (1. - alpha) * rit * ra;
    jb->bottomRightCorner<3, 3>() = alpha * rit * rb;
  }
  return Pose3(ri, (1. - alpha) * a.translation() + alpha * b.translation());
}

// Used only for an edge touching the root; all other edges are inserted as
// native BetweenFactor<Pose3>. Endpoint order preserves T_to_from directly.
class RootBetween final : public NoiseModelFactor {
  Root root_;
  Key to_, from_;
  BetweenFactor<Pose3> native_;

public:
  RootBetween(const Root &root, Key from, Key to, const Pose3 &z,
              const SharedNoiseModel &model)
      : NoiseModelFactor(model, root.active(to, from)), root_(root), to_(to),
        from_(from), native_(to, from, z, noiseModel::Unit::Create(6)) {}
  Vector unwhitenedError(const Values &values,
                         OptionalMatrixVecType h = nullptr) const override {
    Pose3 a = root_.at(values, to_), b = root_.at(values, from_);
    Matrix ja, jb;
    Vector r =
        native_.evaluateError(a, b, h ? &ja : nullptr, h ? &jb : nullptr);
    if (h)
      for (size_t i = 0; i < keys().size(); ++i) {
        Key k = keys()[i];
        (*h)[i] = (k == to_ ? ja : jb) * root_.derivative(k == to_ ? a : b, k);
      }
    return r;
  }
  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<RootBetween>(*this);
  }
};

// Adapts the stock GPSFactorArm to the existing horizontal, interpolated GPS
// contract. GTSAM supplies antenna geometry, Jacobian and robust losses; no
// custom normal equations, damping, factorization or optimizer live here.
class HorizontalGps final : public NoiseModelFactor {
  Root root_;
  VgGps input_;
  GPSFactorArm native_;
  Matrix2 whiten_;
  noiseModel::mEstimator::Base::shared_ptr loss_;
  static noiseModel::mEstimator::Base::shared_ptr loss(const VgGps &in) {
    if (!std::isfinite(in.parameter) || in.parameter <= 0.)
      throw std::invalid_argument("invalid GPS loss parameter");
    if (in.kernel == 0)
      return noiseModel::mEstimator::Huber::Create(in.parameter);
    if (in.kernel == 1)
      return noiseModel::mEstimator::GemanMcClure::Create(
          std::sqrt(in.parameter));
    throw std::invalid_argument("invalid GPS loss");
  }

public:
  HorizontalGps(const Root &root, const VgGps &in)
      : NoiseModelFactor(
            noiseModel::Robust::Create(loss(in), noiseModel::Unit::Create(2)),
            root.active(in.from, in.to)),
        root_(root), input_(in),
        native_(0, Eigen::Map<const Vector3>(in.position),
                Eigen::Map<const Vector3>(in.lever),
                noiseModel::Unit::Create(3)),
        loss_(loss(in)) {
    if (!std::isfinite(in.alpha) || in.alpha < 0. || in.alpha > 1. ||
        !std::isfinite(in.deadband) || in.deadband < 0. ||
        !Eigen::Map<const Vector3>(in.position).allFinite() ||
        !Eigen::Map<const Vector3>(in.lever).allFinite())
      throw std::invalid_argument("invalid GPS interpolation/measurement");
    Matrix3 info = information<3>(in.information);
    if (info.row(2).cwiseAbs().maxCoeff() != 0. ||
        info.col(2).cwiseAbs().maxCoeff() != 0.)
      throw std::invalid_argument("GPS Up information is forbidden");
    Eigen::LLT<Matrix2> llt(info.topLeftCorner<2, 2>());
    if (llt.info() != Eigen::Success)
      throw std::invalid_argument("non-positive GPS information");
    whiten_ = llt.matrixU();
  }
  Vector unwhitenedError(const Values &values,
                         OptionalMatrixVecType h = nullptr) const override {
    Pose3 a = root_.at(values, input_.from), b = root_.at(values, input_.to);
    Matrix6 ja, jb;
    Pose3 pi =
        interpolate(a, b, input_.alpha, h ? &ja : nullptr, h ? &jb : nullptr);
    Matrix hn;
    Vector native_r = native_.evaluateError(pi, h ? &hn : nullptr);
    Vector2 r = whiten_ * native_r.head<2>();
    double d = r.norm(), tau = input_.deadband;
    if (tau > 0. && d <= tau) {
      if (h)
        for (size_t i = 0; i < keys().size(); ++i)
          (*h)[i] = Matrix::Zero(2, keys()[i] == root_.key ? 3 : 6);
      return Vector2::Zero();
    }
    double scale = tau > 0. ? 1. - tau / d : 1.;
    Matrix2 compress = scale * Matrix2::Identity();
    if (tau > 0.)
      compress += tau * r * r.transpose() / (d * d * d);
    if (h) {
      Matrix left = compress * whiten_ * hn.topRows<2>();
      std::map<Key, Matrix> blocks;
      blocks[input_.from] = left * ja;
      if (input_.to == input_.from)
        blocks[input_.to] += left * jb;
      else
        blocks[input_.to] = left * jb;
      for (size_t i = 0; i < keys().size(); ++i) {
        Key k = keys()[i];
        (*h)[i] = blocks.at(k) * root_.derivative(root_.at(values, k), k);
      }
    }
    return scale * r;
  }
  VgGpsDiagnostic diagnostic(const Values &values) const {
    Pose3 pi = interpolate(root_.at(values, input_.from),
                           root_.at(values, input_.to), input_.alpha);
    Vector r = native_.evaluateError(pi);
    double d = (whiten_ * r.head<2>()).norm();
    double u = std::max(0., d - input_.deadband), weight = loss_->weight(u);
    VgGpsDiagnostic out{};
    Eigen::Map<Vector3>(out.residual) = r;
    out.squared_mahalanobis = d * d;
    out.robust_weight = weight;
    out.optimization_weight = d > input_.deadband
                                  ? weight * u / d
                                  : (input_.deadband == 0. ? 1. : 0.);
    out.cost = 2. * loss_->loss(u);
    return out;
  }
  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<HorizontalGps>(*this);
  }
};
} // namespace visloc_gtsam
