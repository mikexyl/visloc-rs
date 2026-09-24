use crate::*;
use nalgebra::{Point2, Point3};
use visloc_vision::{
    pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert},
    ransac::{PnPRansac, RobustPoseEstimator},
    two_view::{fundamental_ransac, FundamentalRansacConfig, TwoViewCorrespondence},
};

fn direction(
    a: &FeatureFrame,
    b: &FeatureFrame,
    matches: &[(usize, usize, f32)],
) -> Option<(visloc_core::geometry::SE3, Verification)> {
    let camera = b.camera.camera().ok()?;
    let correspondences: Vec<_> = matches
        .iter()
        .filter_map(|&(i, j, s)| {
            let p = a.points_camera[i]?;
            Some(Correspondence2D3D {
                point3d: Point3::from(p),
                point2d: Point2::new(b.pixels[j][0] as f64, b.pixels[j][1] as f64),
                confidence: Some(s),
            })
        })
        .collect();
    if correspondences.len() < 15 {
        return None;
    }
    let weights: Vec<_> = correspondences
        .iter()
        .map(|c| c.confidence.unwrap())
        .collect();
    let ransac = PnPRansac {
        pose_estimator: P3PGrunert,
        pose_refiner: Some(GaussNewtonPoseRefiner::default()),
        iterations: 1000,
        reprojection_threshold: 3.,
        seed: b.key.id.wrapping_mul(1337).wrapping_add(a.key.id),
        early_stop_min_iterations: 40,
        early_stop_inlier_ratio: Some(0.85),
        confidence: Some(0.999),
    };
    let report = ransac.estimate_with_weights(&correspondences, &camera, &weights)?;
    if !report.mean_reprojection_error.is_finite()
        || report.inliers.len() < 15
        || (report.inliers.len() as f64 / correspondences.len() as f64) < 0.5
        || report.mean_reprojection_error > 2.25
    {
        return None;
    }
    let mut low = [f64::INFINITY; 2];
    let mut high = [f64::NEG_INFINITY; 2];
    let mut cells = [false; 16];
    for &i in &report.inliers {
        let p = correspondences[i].point2d;
        for d in 0..2 {
            low[d] = low[d].min(p[d]);
            high[d] = high[d].max(p[d]);
        }
        let x = (p.x / camera.width as f64 * 4.).clamp(0., 3.) as usize;
        let y = (p.y / camera.height as f64 * 4.).clamp(0., 3.) as usize;
        cells[y * 4 + x] = true;
    }
    let coverage =
        (high[0] - low[0]) * (high[1] - low[1]) / (camera.width as f64 * camera.height as f64);
    if coverage < 0.02 || cells.iter().filter(|&&x| x).count() < 4 {
        return None;
    }
    let transform = b
        .camera
        .camera_to_body
        .se3()
        .ok()?
        .compose(&report.pose.world_to_camera)
        .compose(&a.camera.camera_to_body.se3().ok()?.inverse());
    Transform::from(&transform).se3().ok()?;
    Some((
        transform,
        Verification {
            pnp_inliers: report.inliers.len(),
            reprojection_px: report.mean_reprojection_error,
            coverage,
            ..Default::default()
        },
    ))
}

/// Match indices always refer to frozen VIO observations, never detector order.
pub fn verify(
    a: &FeatureFrame,
    b: &FeatureFrame,
    matches: &[(usize, usize, f32)],
    pair: Pair,
    similarity: f32,
) -> Result<(Option<LoopConstraint>, Verification)> {
    a.validate()?;
    b.validate()?;
    if matches
        .iter()
        .any(|&(i, j, s)| i >= a.pixels.len() || j >= b.pixels.len() || !s.is_finite())
    {
        return Err(Error("invalid match index/score".into()));
    }
    let mut diagnostic = Verification {
        matches: matches.len(),
        reason: "insufficient_matches".into(),
        ..Default::default()
    };
    if matches.len() < 15 {
        return Ok((None, diagnostic));
    }
    let correspondences: Vec<_> = matches
        .iter()
        .map(|&(i, j, _)| TwoViewCorrespondence {
            previous_xy: Point2::new(a.pixels[i][0] as f64, a.pixels[i][1] as f64),
            current_xy: Point2::new(b.pixels[j][0] as f64, b.pixels[j][1] as f64),
        })
        .collect();
    let Some(report) = fundamental_ransac(
        &correspondences,
        &FundamentalRansacConfig {
            iterations: 1000,
            max_error_px: 3.,
            seed: a.key.id.wrapping_mul(1337).wrapping_add(b.key.id),
        },
    ) else {
        diagnostic.reason = "two_d_ransac_failed".into();
        return Ok((None, diagnostic));
    };
    let filtered: Vec<_> = report.inliers.iter().map(|&i| matches[i]).collect();
    diagnostic.two_d_inliers = filtered.len();
    let forward = direction(a, b, &filtered);
    let reversed: Vec<_> = filtered.iter().map(|&(i, j, s)| (j, i, s)).collect();
    let reverse = direction(b, a, &reversed).map(|(t, mut v)| {
        v.reverse = true;
        (t.inverse(), v)
    });
    let best = match (forward, reverse) {
        (Some(a), Some(b)) => Some(
            if b.1.pnp_inliers > a.1.pnp_inliers
                || (b.1.pnp_inliers == a.1.pnp_inliers && b.1.reprojection_px < a.1.reprojection_px)
            {
                b
            } else {
                a
            },
        ),
        (a, b) => a.or(b),
    };
    let Some((transform, mut v)) = best else {
        diagnostic.reason = "pnp_geometric_support_failed".into();
        return Ok((None, diagnostic));
    };
    v.matches = matches.len();
    v.two_d_inliers = filtered.len();
    v.reason = "verified".into();
    let constraint = LoopConstraint {
        pair,
        from: a.key.clone(),
        to: b.key.clone(),
        to_from: Transform::from(&transform),
        information: matrix_values(&information(0.2, 0.04)),
        similarity,
        verification: v.clone(),
    };
    Ok((Some(constraint), v))
}
