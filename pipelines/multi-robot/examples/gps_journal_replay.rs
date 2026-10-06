//! Reproduce recorded online optimization revisions from immutable backend input.
//! This audits alignment timing and diagnostics without any VIO or sensor access.
use std::{io::Write, path::PathBuf};
use visloc_multi_robot::{
    backend::{Backend, BackendConfig},
    *,
};
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err(
            "usage: gps_journal_replay GRAPH.jsonl REVISIONS.jsonl BACKEND_CONFIG.json OUTPUT"
                .into(),
        );
    }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[3])?)?;
    let mut backend = Backend::default();
    backend.config = serde_json::from_value::<BackendConfig>(config["pgo"].clone())?;
    let output = PathBuf::from(&a[4]);
    if output.exists() {
        return Err("output exists".into());
    }
    std::fs::create_dir_all(&output)?;
    let records: Vec<serde_json::Value> = std::fs::read_to_string(&a[1])?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let mut cursor = 0;
    let mut previous = GraphSnapshot::default();
    let mut log = std::fs::File::create(output.join("gps_revisions.jsonl"))?;
    for line in std::fs::read_to_string(&a[2])?.lines() {
        let revision: serde_json::Value = serde_json::from_str(line)?;
        let Some(target) = revision["input_revision"]
            .as_u64()
            .filter(|_| revision.get("event").is_none())
        else {
            continue;
        };
        while backend.input_revision < target {
            let row = records.get(cursor).ok_or("revision exceeds journal")?;
            cursor += 1;
            if let Some(r) = row.get("keyframe") {
                backend.insert_keyframe(serde_json::from_value(r.clone())?)?;
            }
            if let Some(r) = row.get("loop") {
                backend.insert_loop(serde_json::from_value(r.clone())?)?;
            }
            if let Some(d) = row.get("gps_datum").filter(|d| !d.is_null()) {
                backend.gps_datum = Some(serde_json::from_value(d.clone())?);
            }
            if let Some(r) = row.get("gps") {
                backend.insert_gps(serde_json::from_value(r.clone())?)?;
            }
        }
        previous = backend.solve(&previous)?;
        previous.revision = revision["revision"].as_u64().ok_or("missing revision")?;
        writeln!(
            log,
            "{}",
            serde_json::json!({"revision":previous.revision,"input_revision":previous.input_revision,"latest_keyframe_timestamp_ns":previous.poses.iter().map(|p|p.timestamp_ns).max(),"gps":previous.gps,"initial_cost":previous.initial_cost,"final_cost":previous.final_cost,"solve_ms":previous.solve_ms,"optimizer_reports":previous.optimizer_reports})
        )?;
    }
    std::fs::write(
        output.join("graph_snapshot.json"),
        serde_json::to_vec_pretty(&previous)?,
    )?;
    Ok(())
}
