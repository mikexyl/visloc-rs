//! Re-run only matching/geometry on identical archived retrieval pairs.
//! Usage: cargo run --release -p visloc-multi-robot --features native
//!   --example compare_refinement -- MISSION_DIRECTORY LOOP_CONFIG
#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{collections::BTreeMap, path::PathBuf};
    use visloc_multi_robot::{backend::Backend, sequence::refine, *};
    use visloc_online_loop::{
        models::{Features, Models},
        Config,
    };
    #[derive(serde::Deserialize)]
    struct Archive {
        descriptors: FrameDescriptors,
        features: Vec<FeatureFrame>,
    }
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected MISSION_DIRECTORY LOOP_CONFIG".into());
    }
    let root = PathBuf::from(&args[1]);
    let mut models = Models::new(&Config::from_path(&args[2])?)?;
    let mut matrices = BTreeMap::new();
    let mut frames = BTreeMap::new();
    let mut pairs = BTreeMap::new();
    for entry in std::fs::read_dir(root.join("robots"))? {
        let path = entry?.path();
        for archive in std::fs::read_dir(path.join("sequences"))
            .into_iter()
            .flatten()
        {
            let archive: Archive = serde_json::from_slice(&std::fs::read(archive?.path())?)?;
            matrices.insert(archive.descriptors.sequence.clone(), archive.descriptors);
            for frame in archive.features {
                frames.insert(frame.key.clone(), frame);
            }
        }
        for line in std::fs::read_to_string(path.join("retrieval.jsonl"))?.lines() {
            let event: serde_json::Value = serde_json::from_str(line)?;
            if event["event"] == "refinement" {
                let pair = serde_json::from_value::<Pair>(event["pair"].clone())?;
                let similarity = event["similarity"]
                    .as_f64()
                    .ok_or("missing retrieval score")? as f32;
                pairs.insert(pair, similarity);
            }
        }
    }
    let mut raw = Backend::default();
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("backend/config.json"))?)?;
    raw.config = serde_json::from_value(config["pgo"].clone())?;
    for line in std::fs::read_to_string(root.join("backend/graph.jsonl"))?.lines() {
        let event: serde_json::Value = serde_json::from_str(line)?;
        if let Some(record) = event.get("keyframe") {
            raw.insert_keyframe(serde_json::from_value(record.clone())?)?;
        }
    }
    let mut graphs = BTreeMap::from([("refined", raw.clone()), ("fixed_last", raw)]);
    // Warm the fixed-shape engine on real archived observations before either
    // timed mode. This self-match is not a loop-verification attempt.
    if let Some(frame) = frames.values().find(|frame| frame.pixels.len() == 128) {
        let warmup = Features {
            pixels: frame.pixels.clone(),
            descriptors: frame.descriptors.clone(),
            image_size: [frame.camera.width as f32, frame.camera.height as f32],
        };
        let _ = models.matches(&warmup, &warmup)?;
    }
    let mut results = Vec::new();
    for (pair, similarity) in pairs {
        let a = &matrices[&pair.0];
        let b = &matrices[&pair.1];
        let mut modes = serde_json::Map::new();
        for refine_pair in [false, true] {
            let (ka, kb, _) = if refine_pair {
                refine(a, b)?
            } else {
                (a.frames[4].clone(), b.frames[4].clone(), 0.)
            };
            let fa = &frames[&ka];
            let fb = &frames[&kb];
            let f = |s: &FeatureFrame| Features {
                pixels: s.pixels.clone(),
                descriptors: s.descriptors.clone(),
                image_size: [s.camera.width as f32, s.camera.height as f32],
            };
            let started = std::time::Instant::now();
            let matches = models.matches(&f(fa), &f(fb))?;
            let (edge, diagnostic) = geometry::verify(fa, fb, &matches, pair.clone(), similarity)?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
            let mode = if refine_pair { "refined" } else { "fixed_last" };
            if let Some(edge) = &edge {
                graphs.get_mut(mode).unwrap().insert_loop(edge.clone())?;
            }
            modes.insert(mode.into(),serde_json::json!({"from":ka,"to":kb,"accepted":edge.is_some(),"diagnostic":diagnostic,"matching_and_geometry_ms":elapsed_ms}));
        }
        results.push(serde_json::json!({"pair":pair,"modes":modes}));
    }
    let mut summary =
        serde_json::json!({"identical_retrieval_pairs":results.len(),"pairs":results});
    let comparison = root.join("refinement");
    std::fs::create_dir_all(&comparison)?;
    for mode in ["fixed_last", "refined"] {
        let r = summary["pairs"].as_array().unwrap();
        let accepted = r
            .iter()
            .filter(|r| r["modes"][mode]["accepted"] == true)
            .count();
        let total_inliers: u64 = r
            .iter()
            .map(|r| {
                r["modes"][mode]["diagnostic"]["pnp_inliers"]
                    .as_u64()
                    .unwrap()
            })
            .sum();
        let mean_ms = r
            .iter()
            .map(|r| {
                r["modes"][mode]["matching_and_geometry_ms"]
                    .as_f64()
                    .unwrap()
            })
            .sum::<f64>()
            / r.len().max(1) as f64;
        // Both variants use the same raw graph and batch solve, so their
        // accuracy comparison does not depend on asynchronous arrival timing.
        let graph = graphs[mode].solve(&GraphSnapshot::default())?;
        std::fs::write(
            comparison.join(format!("{mode}_graph.json")),
            serde_json::to_vec_pretty(&graph)?,
        )?;
        summary[mode] = serde_json::json!({"accepted":accepted,"total_pnp_inliers":total_inliers,"components":graph.components,"mean_matching_and_geometry_ms":mean_ms});
    }
    std::fs::write(
        root.join("refinement_comparison.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!(
        "{}",
        serde_json::json!({"pairs":summary["identical_retrieval_pairs"],"fixed_last":summary["fixed_last"],"refined":summary["refined"]})
    );
    Ok(())
}
#[cfg(not(feature = "native"))]
fn main() {
    panic!("enable the native feature");
}
