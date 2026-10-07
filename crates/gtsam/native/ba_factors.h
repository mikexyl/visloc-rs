#pragma once
#include "factors.h"
#include <gtsam/slam/ProjectionFactor.h>

namespace visloc_gtsam {
// Adapt the stock GTSAM projection factor to the existing exact gauge. Body
// poses remain body-to-map; the native factor composes the fixed body_P_camera.
class BodyProjection final : public NoiseModelFactor {
  Root root_;
  Key body_, landmark_;
  GenericProjectionFactor<Pose3, Point3, Cal3_S2> native_;

public:
  BodyProjection(const Root &root, const VgProjection &p, Key landmark)
      : NoiseModelFactor(
            noiseModel::Robust::Create(
                noiseModel::mEstimator::Huber::Create(p.huber),
                noiseModel::Isotropic::Sigma(2, p.sigma)),
            root.variable(p.pose) ? KeyVector{p.pose, landmark}
                                  : KeyVector{landmark}),
        root_(root), body_(p.pose), landmark_(landmark),
        native_(Point2(p.pixel[0], p.pixel[1]), noiseModel::Unit::Create(2),
                p.pose, landmark,
                std::make_shared<Cal3_S2>(p.intrinsics[0], p.intrinsics[1],
                                          0., p.intrinsics[2], p.intrinsics[3]),
                false, false, pose(p.camera_to_body)) {}
  Vector unwhitenedError(const Values &values,
                        OptionalMatrixVecType h = nullptr) const override {
    auto body = root_.at(values, body_);
    Matrix jp, jl;
    Vector r = native_.evaluateError(body, values.at<Point3>(landmark_),
                                     h ? &jp : nullptr, h ? &jl : nullptr);
    if (h) {
      if (root_.variable(body_)) {
        (*h)[0] = jp * root_.derivative(body, body_);
        (*h)[1] = jl;
      } else {
        (*h)[0] = jl;
      }
    }
    return r;
  }
  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<BodyProjection>(*this);
  }
};
} // namespace visloc_gtsam
