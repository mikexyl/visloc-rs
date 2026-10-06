//! Deterministic graph-only ablation on frozen VIO/loop records. No estimator.
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::PathBuf,
};
use visloc_multi_robot::{
    backend::{Backend, BackendConfig},
    *,
};
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 {
        return Err("usage: gps_backend_replay GRAPH.jsonl GPS.jsonl CONFIG.json OFFSET_NS KEEP_LOOPS OUTPUT_DIR".into());
    }
    let mut backend = Backend::default();
    backend.config = serde_json::from_slice::<BackendConfig>(&std::fs::read(&args[3])?)?;
    let offset: i64 = args[4].parse()?;
    let keep_loops: bool = args[5].parse()?;
    let output = PathBuf::from(&args[6]);
    if output.exists() {
        return Err("output already exists".into());
    }
    std::fs::create_dir_all(&output)?;
    let mut sessions: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for line in std::fs::read_to_string(&args[1])?.lines() {
        let row: serde_json::Value = serde_json::from_str(line)?;
        if let Some(r) = row.get("keyframe") {
            let r: KeyframeRecord = serde_json::from_value(r.clone())?;
            sessions
                .entry(r.key.robot.clone())
                .or_default()
                .insert(r.key.session.clone());
            backend.insert_keyframe(r)?;
        }
        if keep_loops {
            if let Some(r) = row.get("loop") {
                backend.insert_loop(serde_json::from_value(r.clone())?)?;
            }
        }
    }
    for line in std::fs::read_to_string(&args[2])?.lines() {
        let mut r: GpsRecord = serde_json::from_str(line)?;
        let names = sessions
            .get(&r.key.robot)
            .ok_or("GPS robot missing from graph")?;
        if names.len() != 1 {
            return Err("frozen replay requires an explicit single session per robot".into());
        }
        r.key.session = names.first().unwrap().clone();
        r.timestamp_ns += offset;
        r.receipt_timestamp_ns += offset;
        backend.insert_gps(r)?;
    }
    let snapshot = backend.solve(&GraphSnapshot::default())?;
    std::fs::write(
        output.join("graph_snapshot.json"),
        serde_json::to_vec_pretty(&snapshot)?,
    )?;
    std::fs::write(
        output.join("config.json"),
        serde_json::to_vec_pretty(&backend.config)?,
    )?;
    let mut tum = std::fs::File::create(output.join("keyframes.tum"))?;
    for p in &snapshot.poses {
        let t = p.body_to_map.translation;
        let q = p.body_to_map.rotation_xyzw;
        let ns = p.timestamp_ns - offset;
        writeln!(
            tum,
            "{}.{:09} {} {} {} {} {} {} {}",
            ns / 1_000_000_000,
            ns % 1_000_000_000,
            t[0],
            t[1],
            t[2],
            q[0],
            q[1],
            q[2],
            q[3]
        )?;
    }
    println!(
        "{}",
        serde_json::json!({"poses":snapshot.poses.len(),"gps_records":snapshot.gps.diagnostics.len(),"active":snapshot.gps.diagnostics.iter().filter(|d|d.reason=="active").count(),"aligned_components":snapshot.gps.aligned_components.len(),"initial_cost":snapshot.initial_cost,"final_cost":snapshot.final_cost,"solve_ms":snapshot.solve_ms})
    );
    Ok(())
}
