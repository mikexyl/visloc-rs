//! Residual-only objective for base and trial windows. Preserve factor order,
//! per-landmark Huber sums, and the constant-free marginal prior convention.

use super::*;
use crate::imu::{whitened_bias_random_walk_residual, whitened_preintegration_residual};
use crate::vio::aom::anchored_visual_cost_f64;

impl WindowProblem {
    pub(super) fn cost_f64(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
        if state.len() != self.state_dof() {
            return Err(LmFailure::LinearSolve);
        }
        let prior_cost = self.prior.as_ref().map_or_else(
            || self.prior_factors(state).first().map(|f| f.objective_cost),
            |prior| Some(self.marginal_prior_reduced_cost(state, prior)),
        );
        self.objective_f64(
            |index, visual| {
                if visual {
                    self.block_nav_visual_current(state, index)
                } else {
                    self.block_nav(state, index)
                }
            },
            |index| {
                let landmark = self.landmarks.get(index)?;
                Some(landmark.parameter(self.block_frame_id(landmark.anchor_state_index)?))
            },
            state,
            prior_cost,
        )
    }

    fn objective_f64(
        &self,
        nav: impl Fn(usize, bool) -> Option<BasaltNavState>,
        parameter: impl Fn(usize) -> Option<InverseDistanceLandmark> + Sync,
        chart: &DVector<f64>,
        prior_cost: Option<f64>,
    ) -> Result<f64, LmFailure> {
        // Decode each pose once per objective evaluation, not per observation.
        let visual_nav = (0..self.poses.len() + self.states.len())
            .map(|index| nav(index, true))
            .collect::<Vec<_>>();
        let landmark_costs = map_ordered(
            0..self.landmarks.len(),
            self.parallel_landmarks(),
            |index| {
                let landmark = &self.landmarks[index];
                let anchor = visual_nav.get(landmark.anchor_state_index)?.as_ref()?;
                let anchor_extrinsic = self.camera_to_imu(landmark.anchor_camera_id)?;
                let anchor_frame = self.block_frame_id(landmark.anchor_state_index)?;
                let point = parameter(index)?;
                let mut cost = 0.0;
                let mut count = 0;
                for observation in &landmark.observations {
                    let value = (|| {
                        let target = visual_nav.get(observation.state_index)?.as_ref()?;
                        anchored_visual_cost_f64(
                            self.camera_model(observation.camera_id)?,
                            &anchor.imu_to_world,
                            &anchor_extrinsic,
                            &target.imu_to_world,
                            &self.camera_to_imu(observation.camera_id)?,
                            &point,
                            observation.pixel,
                            anchor_frame == self.block_frame_id(observation.state_index)?
                                && landmark.anchor_camera_id == observation.camera_id,
                            FactorConfig::default(),
                        )
                    })();
                    if let Some(value) = value {
                        cost += value;
                        count += 1;
                    }
                }
                (count != 0 && cost.is_finite()).then_some(cost)
            },
        );
        let mut cost: f64 = landmark_costs.into_iter().flatten().sum();
        let layout = self.layout();
        for link in &self.imu_links {
            let imu = (|| {
                let from = nav(self.poses.len() + link.from_index, false)?;
                let to = nav(self.poses.len() + link.to_index, false)?;
                let mut delta = link.delta.clone();
                if delta.covariance.norm_squared() <= 1e-30 && delta.delta_time.is_finite() {
                    let dt = delta.delta_time.max(1e-9);
                    for axis in 0..3 {
                        delta.covariance[(axis, axis)] =
                            self.imu_noise.accel_density.powi(2) * dt.powi(3) / 3.0;
                        delta.covariance[(3 + axis, 3 + axis)] =
                            self.imu_noise.gyro_density.powi(2) * dt;
                        delta.covariance[(6 + axis, 6 + axis)] =
                            self.imu_noise.accel_density.powi(2) * dt;
                    }
                }
                whitened_preintegration_residual(&from, &to, &delta, self.gravity_world).ok()
            })();
            if let Some(residual) = imu {
                cost += 0.5 * residual.norm_squared();
            }
            // Bias rows use the compact chart, matching bias_walk_factor.
            let bias = (|| {
                let from_offset = layout.offset_for_state(link.from_index)?;
                let to_offset = layout.offset_for_state(link.to_index)?;
                let from = decode_state_with_mode(
                    chart.rows(from_offset, NAV_STATE_DOF),
                    self.scalar_mode,
                );
                let to =
                    decode_state_with_mode(chart.rows(to_offset, NAV_STATE_DOF), self.scalar_mode);
                whitened_bias_random_walk_residual(
                    &from,
                    &to,
                    link.delta.delta_time,
                    self.bias_walk_noise,
                )
                .ok()
            })();
            if let Some(residual) = bias.filter(|r| r.iter().all(|v| v.is_finite())) {
                cost += 0.5 * residual.norm_squared();
            }
        }
        if let Some(prior) = prior_cost {
            cost += prior;
        }
        if cost.is_finite() {
            Ok(cost)
        } else {
            Err(LmFailure::NonFinite)
        }
    }
}

impl WindowTrialView<'_, '_> {
    pub(super) fn cost_f64(&self) -> Result<f64, LmFailure> {
        let prior_cost = self.base.prior.as_ref().map_or_else(
            || {
                self.base
                    .prior_factors_view(self)
                    .first()
                    .map(|f| f.objective_cost)
            },
            |prior| Some(self.base.marginal_prior_reduced_cost_view(self, prior)),
        );
        self.base.objective_f64(
            |index, visual| {
                if visual {
                    self.block_nav_visual_current(index)
                } else {
                    self.block_nav(index)
                }
            },
            |index| self.landmark_parameter(index),
            self.chart,
            prior_cost,
        )
    }
}
