//! Replay recorded retrieval pairs without VIO, retrieval, or PGO changes.
//! cargo run --release -p visloc-multi-robot --features native
//!   --example diagnose_verification -- MISSION LOOP_CONFIG OUTPUT_JSON
//!
//! The detailed gates mirror geometry.rs. Every replay is checked against the
//! production verifier and the original recorded decisions before being saved.
#[cfg(feature = "native")]
mod diagnostic {
    use nalgebra::{Point2, Point3};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, path::PathBuf};
    use visloc_multi_robot::*;
    use visloc_online_loop::{
        models::{Features, Models},
        Config,
    };
    use visloc_vision::{
        pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert},
        ransac::{PnPRansac, RobustPoseEstimator},
        two_view::{fundamental_ransac, FundamentalRansacConfig, TwoViewCorrespondence},
    };

    const STAGES: [&str; 13] = [
        "insufficient_matches",
        "fundamental_no_model",
        "insufficient_fundamental_inliers",
        "insufficient_metric_landmarks",
        "pnp_no_model",
        "nonfinite_reprojection",
        "pnp_inlier_count",
        "pnp_inlier_ratio",
        "pnp_reprojection",
        "coverage_area",
        "coverage_cells",
        "invalid_transform",
        "verified",
    ];

    fn finish(mut details: Value, stage: usize) -> (usize, Value) {
        details["first_failed_gate"] = json!(STAGES[stage]);
        (stage, details)
    }

    fn direction(
        a: &FeatureFrame,
        b: &FeatureFrame,
        matches: &[(usize, usize, f32)],
    ) -> Result<(usize, Value)> {
        let camera = b.camera.camera()?;
        let correspondences: Vec<_> = matches
            .iter()
            .filter_map(|&(i, j, s)| {
                Some(Correspondence2D3D {
                    point3d: Point3::from(a.points_camera[i]?),
                    point2d: Point2::new(b.pixels[j][0] as f64, b.pixels[j][1] as f64),
                    confidence: Some(s),
                })
            })
            .collect();
        let mut details = json!({
            "source": a.key, "target": b.key,
            "fundamental_inliers": matches.len(),
            "metric_correspondences": correspondences.len(),
            "missing_metric_landmarks": matches.len() - correspondences.len(),
        });
        // There is no explicit F-inlier-count gate in production: fewer than
        // 15 F inliers necessarily fails its subsequent 2D-3D-count gate.
        if matches.len() < 15 {
            return Ok(finish(details, 2));
        }
        if correspondences.len() < 15 {
            return Ok(finish(details, 3));
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
        let Some(report) = ransac.estimate_with_weights(&correspondences, &camera, &weights) else {
            return Ok(finish(details, 4));
        };
        let ratio = report.inliers.len() as f64 / correspondences.len() as f64;
        details["pnp_inliers"] = json!(report.inliers.len());
        details["pnp_inlier_ratio"] = json!(ratio);
        details["reprojection_px"] = json!(report.mean_reprojection_error);
        details["refinement_applied"] = json!(report.diagnostics.refinement_applied);
        details["pre_refinement_reprojection_px"] =
            json!(report.diagnostics.pre_refinement_mean_reprojection_error);
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
        let cell_count = cells.iter().filter(|&&x| x).count();
        details["coverage"] = json!(coverage);
        details["occupied_cells"] = json!(cell_count);
        let gates = [
            (5, !report.mean_reprojection_error.is_finite()),
            (6, report.inliers.len() < 15),
            (7, ratio < 0.5),
            (8, report.mean_reprojection_error > 2.25),
            (9, coverage < 0.02),
            (10, cell_count < 4),
        ];
        // Also retain overlapping failures; the exclusive count uses the first
        // failing gate in production order, not the number of failed predicates.
        details["failed_predicates"] = json!(gates
            .iter()
            .filter_map(|&(i, fails)| fails.then_some(STAGES[i]))
            .collect::<Vec<_>>());
        if let Some(&(stage, _)) = gates.iter().find(|(_, fails)| *fails) {
            return Ok(finish(details, stage));
        }
        let transform = b
            .camera
            .camera_to_body
            .se3()?
            .compose(&report.pose.world_to_camera)
            .compose(&a.camera.camera_to_body.se3()?.inverse());
        if Transform::from(&transform).se3().is_err() {
            return Ok(finish(details, 11));
        }
        Ok(finish(details, 12))
    }

    fn diagnose(
        a: &FeatureFrame,
        b: &FeatureFrame,
        matches: &[(usize, usize, f32)],
    ) -> Result<(usize, Value)> {
        a.validate()?;
        b.validate()?;
        let mut details = json!({"matches": matches.len(), "directions": []});
        if matches.len() < 15 {
            return Ok((0, details));
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
            return Ok((1, details));
        };
        let filtered: Vec<_> = report.inliers.iter().map(|&i| matches[i]).collect();
        details["fundamental_inliers"] = json!(filtered.len());
        let (forward_stage, forward) = direction(a, b, &filtered)?;
        let reversed: Vec<_> = filtered.iter().map(|&(i, j, s)| (j, i, s)).collect();
        let (reverse_stage, reverse) = direction(b, a, &reversed)?;
        details["directions"] = json!([forward, reverse]);
        // Either direction suffices. Attribute a rejected pair to the furthest
        // gate reached by either direction so each candidate is counted once.
        Ok((forward_stage.max(reverse_stage), details))
    }

    #[derive(serde::Deserialize)]
    struct Archive {
        descriptors: FrameDescriptors,
        features: Vec<FeatureFrame>,
    }

    pub fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let args: Vec<_> = std::env::args().collect();
        if args.len() != 4 {
            return Err("expected MISSION LOOP_CONFIG OUTPUT_JSON".into());
        }
        let root = PathBuf::from(&args[1]);
        let config = Config::from_path(&args[2])?;
        let mut matrices = BTreeMap::new();
        let mut frames = BTreeMap::new();
        let mut pairs = BTreeMap::new();
        let mut originals = BTreeMap::new();
        for entry in std::fs::read_dir(root.join("robots"))? {
            let path = entry?.path();
            for archive in std::fs::read_dir(path.join("sequences"))? {
                let archive: Archive = serde_json::from_slice(&std::fs::read(archive?.path())?)?;
                matrices.insert(archive.descriptors.sequence.clone(), archive.descriptors);
                for frame in archive.features {
                    frames.insert(frame.key.clone(), frame);
                }
            }
            for line in std::fs::read_to_string(path.join("retrieval.jsonl"))?.lines() {
                let event: Value = serde_json::from_str(line)?;
                if event["event"] == "refinement" {
                    let pair = serde_json::from_value::<Pair>(event["pair"].clone())?;
                    if pairs.insert(pair, event).is_some() {
                        return Err("duplicate refinement".into());
                    }
                }
            }
            for line in std::fs::read_to_string(path.join("events.jsonl"))?.lines() {
                let mut event: Value = serde_json::from_str(line)?;
                if event["event"] == "verification" {
                    let pair = serde_json::from_value::<Pair>(event["pair"].clone())?;
                    event["owner"] = json!(path.file_name().unwrap().to_str().unwrap());
                    if originals.insert(pair, event).is_some() {
                        return Err("duplicate verification".into());
                    }
                }
            }
        }
        if pairs.keys().ne(originals.keys()) {
            return Err("retrieval/verification pair mismatch".into());
        }
        let mut models = Models::new(&config)?;
        let f = |s: &FeatureFrame| Features {
            pixels: s.pixels.clone(),
            descriptors: s.descriptors.clone(),
            image_size: [s.camera.width as f32, s.camera.height as f32],
        };
        if let Some(frame) = frames.values().find(|frame| frame.pixels.len() == 128) {
            let _ = models.matches(&f(frame), &f(frame))?;
        }
        let mut results = Vec::new();
        let mut pair_counts = BTreeMap::from(STAGES.map(|s| (s, 0usize)));
        let mut direction_counts = BTreeMap::<String, usize>::new();
        for (pair, retrieval) in pairs {
            let similarity = retrieval["similarity"]
                .as_f64()
                .ok_or("missing similarity")? as f32;
            if similarity < config.min_similarity {
                return Err("pair below configured threshold".into());
            }
            let (ka, kb, _) = sequence::refine(&matrices[&pair.0], &matrices[&pair.1])?;
            if json!(ka) != retrieval["from"] || json!(kb) != retrieval["to"] {
                return Err("selected frame mismatch".into());
            }
            let (a, b) = (&frames[&ka], &frames[&kb]);
            let matches = models.matches(&f(a), &f(b))?;
            let (edge, verification) = geometry::verify(a, b, &matches, pair.clone(), similarity)?;
            let recorded: Verification =
                serde_json::from_value(originals[&pair]["diagnostic"].clone())?;
            if verification.reason != recorded.reason
                || verification.matches != recorded.matches
                || verification.two_d_inliers != recorded.two_d_inliers
                || verification.pnp_inliers != recorded.pnp_inliers
                || verification.reverse != recorded.reverse
                || (verification.reprojection_px - recorded.reprojection_px).abs() > 1e-6
                || (verification.coverage - recorded.coverage).abs() > 1e-6
            {
                return Err(format!(
                    "recorded verification mismatch for {pair:?}: {verification:?} vs {recorded:?}"
                )
                .into());
            }
            let (stage, mut details) = diagnose(a, b, &matches)?;
            if (stage == 12) != edge.is_some()
                || details["matches"] != json!(verification.matches)
                || details
                    .get("fundamental_inliers")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    != verification.two_d_inliers as u64
            {
                return Err(format!("detailed/production verifier mismatch for {pair:?}").into());
            }
            details["pair"] = json!(pair);
            details["from"] = json!(ka);
            details["to"] = json!(kb);
            details["owner"] = originals[&pair]["owner"].clone();
            details["similarity"] = json!(similarity);
            details["frame_similarity"] = retrieval["frame_similarity"].clone();
            details["pair_outcome"] = json!(STAGES[stage]);
            details["recorded_diagnostic"] = json!(recorded);
            details["matches_with_scores"] = json!(matches);
            details["source_features"] = json!([a.pixels.len(), b.pixels.len()]);
            details["source_metric_landmarks"] = json!([
                a.points_camera.iter().flatten().count(),
                b.points_camera.iter().flatten().count()
            ]);
            *pair_counts.get_mut(STAGES[stage]).unwrap() += 1;
            for d in details["directions"].as_array().unwrap() {
                *direction_counts
                    .entry(d["first_failed_gate"].as_str().unwrap().into())
                    .or_default() += 1;
            }
            results.push(details);
        }
        let output = json!({
            "mission": root, "threshold": config.min_similarity, "pairs": results,
            "total_pairs": results.len(), "pair_counts": pair_counts,
            "direction_counts": direction_counts, "all_recorded_outcomes_reproduced": true,
            "all_production_verifier_outcomes_reproduced": true,
            "pair_accounting": "Furthest gate reached by either direction; one count per pair",
            "direction_accounting": "First failed gate in production order; two directions per F model",
            "geometry_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("../src/geometry.rs"))),
        });
        let output_path = PathBuf::from(&args[3]);
        if output_path.exists() {
            return Err("output already exists; choose a new filename".into());
        }
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(output_path, serde_json::to_vec_pretty(&output)?)?;
        println!(
            "{}",
            json!({"pairs":output["total_pairs"], "pair_counts":output["pair_counts"],
            "direction_counts":output["direction_counts"], "all_recorded_outcomes_reproduced":true})
        );
        Ok(())
    }
}

#[cfg(feature = "native")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    diagnostic::run()
}
#[cfg(not(feature = "native"))]
fn main() {
    panic!("enable the native feature");
}
