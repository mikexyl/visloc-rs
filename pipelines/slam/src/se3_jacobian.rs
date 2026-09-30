//! Exact right-log Jacobian shared by sequential and loop pose factors.
use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};
#[cfg(test)]
use visloc_core::geometry::SE3;

fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0., -v.z, v.y, v.z, 0., -v.x, -v.y, v.x, 0.)
}

/// Inverse right Jacobian using the convergent dexp series. The principal SE3
/// log has rotation magnitude <= pi; 48 terms cover that interval, including
/// large translational lever arms, without a small-angle-only approximation.
pub(crate) fn right_log_jacobian(x: &Vector6<f64>) -> Matrix6<f64> {
    let mut ad = Matrix6::zeros();
    let w = skew(x.fixed_rows::<3>(3).into_owned());
    ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&w);
    ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&w);
    ad.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&skew(x.fixed_rows::<3>(0).into_owned()));
    let mut term = Matrix6::identity();
    let mut jr = term;
    for n in 1..=48 {
        term = term * (-ad) / (n + 1) as f64;
        jr += term;
        if term.norm() < 1e-16 {
            break;
        }
    }
    // dexp is nonsingular on the principal log chart (rotation < 2*pi).
    jr.try_inverse()
        .unwrap_or_else(|| Matrix6::repeat(f64::NAN))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn log_jacobian_matches_central_differences() {
        for x in [
            Vector6::zeros(),
            Vector6::new(2., -3., 4., 0.4, -0.8, 1.1),
            Vector6::new(15., 9., -4., 3.0, 0.2, -0.1),
        ] {
            let a = right_log_jacobian(&x);
            let z = SE3::exp(&x);
            for k in 0..6 {
                let mut d = Vector6::zeros();
                d[k] = 1e-6;
                let n = (z.compose(&SE3::exp(&d)).log() - z.compose(&SE3::exp(&-d)).log()) / 2e-6;
                assert!((a.column(k) - n).norm() < 1e-7);
            }
        }
    }
}
