//! Per-linearization f64 QR storage. Trial recovery keeps the historical
//! operation order (Qᵀ applied to the already assembled raw RHS), rather than
//! substituting the algebraically equivalent QᵀJ * dx + Qᵀr.

use super::*;
use nalgebra::{linalg::QR, Dyn};

struct LandmarkQr {
    qr: QR<f64, Dyn, Dyn>,
    r: DMatrix<f64>,
    state: DMatrix<f64>,
    residual: DVector<f64>,
    rank: usize,
}

pub(super) struct PreparedFactors {
    landmarks: Vec<Option<LandmarkQr>>,
    tolerance: f64,
}

pub(super) fn reduce(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> (ReducedNormalSystem, PreparedFactors) {
    let mut reduced = ReducedNormalSystem {
        h: DMatrix::zeros(state_dof, state_dof),
        b: DVector::zeros(state_dof),
        back_substitution: Vec::with_capacity(factors.len()),
        diagnostic_stages: None,
    };
    let mut landmarks = Vec::with_capacity(factors.len());
    for batch in factors.chunks(32) {
        let contributions = map_ordered(0..batch.len(), true, |index| {
            let factor = &batch[index];
            assert_eq!(factor.state_jacobian.ncols(), state_dof);
            let n = factor.landmark_jacobian.ncols();
            let prepared = (n != 0).then(|| {
                let (mut state, landmark, residual) = augmented_landmark_rows(factor);
                let qr = landmark.qr();
                let r = qr.r();
                let rank = (0..r.nrows().min(r.ncols()))
                    .filter(|&i| r[(i, i)].abs() > tolerance)
                    .count();
                let mut transformed =
                    DMatrix::from_column_slice(residual.len(), 1, residual.as_slice());
                qr.q_tr_mul(&mut state);
                qr.q_tr_mul(&mut transformed);
                LandmarkQr {
                    qr,
                    r,
                    state,
                    residual: transformed.column(0).into_owned(),
                    rank,
                }
            });
            let (j, residual, rank) = match &prepared {
                Some(data) => (
                    data.state.rows(n, data.state.nrows() - n).into_owned(),
                    data.residual.rows(n, data.residual.len() - n).into_owned(),
                    data.rank,
                ),
                None => (factor.state_jacobian.clone(), factor.residual.clone(), 0),
            };
            (
                j.transpose() * &j,
                j.transpose() * &residual,
                LandmarkBackSubstitution {
                    state_jacobian: factor.state_jacobian.clone(),
                    landmark_jacobian: factor.landmark_jacobian.clone(),
                    residual: factor.residual.clone(),
                    rank,
                },
                prepared,
            )
        });
        for (h, b, back, prepared) in contributions {
            reduced.h += h;
            reduced.b += b;
            reduced.back_substitution.push(back);
            landmarks.push(prepared);
        }
    }
    (
        reduced,
        PreparedFactors {
            landmarks,
            tolerance,
        },
    )
}

impl PreparedFactors {
    pub(super) fn model_decrease(
        &self,
        factors: &[WhitenedFactorRowStack],
        step: &DVector<f64>,
    ) -> Option<f64> {
        let mut decrease = 0.0;
        for (factor, prepared) in factors.iter().zip(&self.landmarks) {
            if factor.state_jacobian.ncols() != step.len() {
                return None;
            }
            let Some(data) = prepared else {
                let inc = &factor.state_jacobian * step;
                decrease -= inc.dot(&(0.5 * &inc + &factor.residual));
                continue;
            };
            let n = factor.landmark_jacobian.ncols();
            if data.rank < n {
                let (state, _, residual) = augmented_landmark_rows(factor);
                let inc = state * step;
                decrease -= inc.dot(&(0.5 * &inc + residual));
                continue;
            }
            let mut inc = &data.state * step;
            let mut rhs = data.residual.rows(0, n).into_owned();
            for row in 0..n {
                rhs[row] += inc[row];
            }
            let landmark_inc = triangular_solve(&data.r, &(-rhs), self.tolerance)?;
            let q1_inc = &data.r * landmark_inc;
            for row in 0..n {
                inc[row] += q1_inc[row];
            }
            decrease -= inc.dot(&(0.5 * &inc + &data.residual));
        }
        decrease.is_finite().then_some(decrease)
    }

    fn recover(
        &self,
        index: usize,
        factor: &WhitenedFactorRowStack,
        step: &DVector<f64>,
    ) -> Option<DVector<f64>> {
        let data = self.landmarks[index].as_ref()?;
        let n = factor.landmark_jacobian.ncols();
        if data.rank < n || factor.landmark_jacobian.norm() <= self.tolerance {
            return None;
        }
        let rhs = -(&factor.residual + &factor.state_jacobian * step);
        let mut transformed = DMatrix::zeros(data.state.nrows(), 1);
        transformed.rows_mut(0, rhs.len()).copy_from(&rhs);
        data.qr.q_tr_mul(&mut transformed);
        triangular_solve(
            &data.r,
            &transformed.column(0).rows(0, n).into_owned(),
            self.tolerance,
        )
    }

    pub(super) fn trial_preparation(
        &self,
        factors: &[WhitenedFactorRowStack],
        state: &DVector<f64>,
        step: &DVector<f64>,
    ) -> Option<LmTrialPreparation> {
        // Generic LM problems need not attach WindowProblem identities.
        // Fall back to their own recovery hook if any landmark is unmapped.
        if factors
            .iter()
            .any(|f| f.landmark_jacobian.ncols() != 0 && f.landmark_metadata.is_none())
        {
            return None;
        }
        let landmark_steps = map_ordered(0..factors.len(), true, |index| {
            let factor = &factors[index];
            let metadata = factor.landmark_metadata?;
            let increment = self.recover(index, factor, step).and_then(|value| {
                (value.len() == 3 && value.iter().all(|v| v.is_finite()))
                    .then(|| Vector3::new(value[0], value[1], value[2]))
            });
            Some(LmPreparedLandmarkStep {
                landmark_index: metadata.landmark_index,
                track_id: metadata.track_id,
                step: increment,
            })
        })
        .into_iter()
        .flatten()
        .collect();
        Some(LmTrialPreparation {
            landmark_steps,
            tolerance_bits: self.tolerance.to_bits(),
            state_fingerprint: lm_trial_vector_fingerprint(state),
            step_fingerprint: lm_trial_vector_fingerprint(step),
        })
    }
}

fn triangular_solve(r: &DMatrix<f64>, rhs: &DVector<f64>, tolerance: f64) -> Option<DVector<f64>> {
    let n = rhs.len();
    let mut increment = DVector::zeros(n);
    for row in (0..n).rev() {
        let mut value = rhs[row];
        for column in (row + 1)..n {
            value -= r[(row, column)] * increment[column];
        }
        let diagonal = r[(row, row)];
        if diagonal.abs() <= tolerance {
            return None;
        }
        increment[row] = value / diagonal;
    }
    Some(increment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_matches_repeated_qr_including_rank_deficient_and_unmapped_factors() {
        let factors = (0..65)
            .map(|i| {
                let rows = 2 + i % 11;
                let js = DMatrix::from_fn(rows, 17, |r, c| ((i + 3 * r + c) as f64 * 0.37).sin());
                let jl = if i % 5 == 0 {
                    DMatrix::zeros(rows, 0)
                } else {
                    DMatrix::from_fn(rows, 3, |r, c| {
                        if i % 7 == 0 {
                            0.0
                        } else {
                            ((2 * i + r * (c + 1) + c * 7) as f64 * 0.43).cos()
                        }
                    })
                };
                let residual = DVector::from_fn(rows, |r, _| (r + i) as f64 * 0.017);
                WhitenedFactorRowStack::new(js, jl, residual).unwrap()
            })
            .collect::<Vec<_>>();
        let step = DVector::from_fn(17, |r, _| r as f64 * 0.003 - 0.01);
        let (reduced, prepared) = reduce(&factors, 17, 1e-10);
        let reference = reduce_landmark_factors(&factors, 17, 1e-10);
        assert_eq!(reduced.h, reference.h);
        assert_eq!(reduced.b, reference.b);
        assert_eq!(
            prepared.model_decrease(&factors, &step),
            model_cost_decrease(&factors, &step, 1e-10)
        );
        for (index, factor) in factors.iter().enumerate() {
            if factor.landmark_jacobian.ncols() != 0 {
                assert_eq!(
                    prepared.recover(index, factor, &step),
                    back_substitute_landmark(&reference.back_substitution[index], &step, 1e-10)
                );
            }
        }
        assert!(prepared.trial_preparation(&factors, &step, &step).is_none());
        assert!(reference
            .back_substitution
            .iter()
            .any(|data| data.rank == 3));
        let mapped = factors
            .into_iter()
            .enumerate()
            .map(|(index, factor)| {
                if factor.landmark_jacobian.ncols() == 0 {
                    factor
                } else {
                    factor.with_landmark_metadata(index, 1000 + index as u64)
                }
            })
            .collect::<Vec<_>>();
        let state = DVector::zeros(step.len());
        let (tolerance, state_id, step_id, recovered) = prepared
            .trial_preparation(&mapped, &state, &step)
            .unwrap()
            .take_landmark_steps();
        assert_eq!(tolerance, 1e-10_f64.to_bits());
        assert_eq!(state_id, lm_trial_vector_fingerprint(&state));
        assert_eq!(step_id, lm_trial_vector_fingerprint(&step));
        assert_eq!(
            recovered.len(),
            mapped
                .iter()
                .filter(|f| f.landmark_metadata.is_some())
                .count()
        );
        for (index, track_id, increment) in recovered {
            assert_eq!(track_id, 1000 + index as u64);
            let expected =
                back_substitute_landmark(&reference.back_substitution[index], &step, 1e-10)
                    .map(|v| Vector3::new(v[0], v[1], v[2]));
            assert_eq!(increment, expected);
        }
    }
}
