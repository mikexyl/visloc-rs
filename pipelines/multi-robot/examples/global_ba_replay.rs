//! Replay immutable backend inputs without rerunning or modifying VIO.
//! cargo run --release -p visloc-multi-robot --example global_ba_replay -- MISSION_ROOT OUTPUT [MAX_KEYFRAMES]
use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
};
use visloc_multi_robot::{backend::Backend, *};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn lines(path: &Path) -> Result<Vec<serde_json::Value>> {
    BufReader::new(fs::File::open(path)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}
#[derive(serde::Deserialize)]
struct DisplayFeature {
    key: Key,
    timestamp_ns: i64,
    camera: CameraModel,
    pixels: Vec<[f64; 2]>,
    track_ids: Vec<u64>,
    points_camera: Vec<Option<[f64; 3]>>,
}
fn save(root: &Path, name: &str, s: &GraphSnapshot) -> Result<()> {
    fs::write(
        root.join(format!("{name}.json")),
        serde_json::to_vec_pretty(s)?,
    )?;
    let robots: BTreeSet<_> = s.poses.iter().map(|p| &p.key.robot).collect();
    for robot in robots {
        let mut f = fs::File::create(root.join(format!("{name}_{robot}.tum")))?;
        let mut poses: Vec<_> = s.poses.iter().filter(|p| &p.key.robot == robot).collect();
        poses.sort_by_key(|p| p.timestamp_ns);
        for p in poses {
            let [x, y, z] = p.body_to_map.translation;
            let [qx, qy, qz, qw] = p.body_to_map.rotation_xyzw;
            writeln!(
                f,
                "{}.{:09} {x:.9} {y:.9} {z:.9} {qx:.12} {qy:.12} {qz:.12} {qw:.12}",
                p.timestamp_ns.div_euclid(1_000_000_000),
                p.timestamp_ns.rem_euclid(1_000_000_000)
            )?;
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("expected MISSION_ROOT OUTPUT [MAX_KEYFRAMES]".into());
    }
    let root = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    let limit = args
        .get(3)
        .map(|arg| arg.parse())
        .transpose()?
        .unwrap_or(usize::MAX);
    fs::create_dir(output)?;
    let mut b = Backend::default();
    b.config.mode = BackendMode::GlobalBundleAdjustment;
    let backend_config = root.join("backend/config.json");
    if backend_config.exists() {
        let v: serde_json::Value = serde_json::from_slice(&fs::read(backend_config)?)?;
        if let Some(c) = v.get("pgo") {
            b.config = serde_json::from_value(c.clone())?;
        }
        b.config.mode = BackendMode::GlobalBundleAdjustment;
    }
    let mut packets = vec![];
    for v in lines(&root.join("backend/graph.jsonl"))? {
        if let Some(r) = v.get("keyframe") {
            let r: KeyframeRecord = serde_json::from_value(r.clone())?;
            if b.records.len() < limit || b.records.contains_key(&r.key) {
                b.insert_keyframe(r)?;
            }
        }
        if let Some(r) = v.get("loop") {
            b.insert_loop(serde_json::from_value(r.clone())?)?;
        }
        if let Some(r) = v.get("gps") {
            b.insert_gps(serde_json::from_value(r.clone())?)?;
        }
        if let Some(d) = v.get("gps_datum").filter(|d| !d.is_null()) {
            b.gps_datum = Some(serde_json::from_value(d.clone())?);
        }
        if let Some(r) = v.get("bundle_frame") {
            packets.push(serde_json::from_value::<BundleFrame>(r.clone())?);
        }
    }
    for frame in packets {
        if b.records.contains_key(&frame.key) {
            b.insert_bundle_frame(frame)?;
        }
    }
    let mut legacy = false;
    let mut sources = vec![root.join("backend/graph.jsonl")];
    for robot in fs::read_dir(root.join("robots"))? {
        let robot = robot?.path();
        let mut dirs = vec![robot.clone()];
        for d in fs::read_dir(&robot)? {
            let d = d?.path();
            if d.is_dir()
                && d.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("session-")
            {
                dirs.push(d);
            }
        }
        for dir in dirs {
            let native = dir.join("bundle_frames.jsonl");
            let display = dir.join("visualization.jsonl");
            if native.exists() {
                sources.push(native.clone());
                for v in lines(&native)? {
                    let frame: BundleFrame = serde_json::from_value(v)?;
                    if b.records.contains_key(&frame.key) {
                        b.insert_bundle_frame(frame)?;
                    }
                }
            } else if display.exists() {
                // Older runs only archived measured left-camera observations.
                legacy = true;
                sources.push(display.clone());
                for v in lines(&display)? {
                    let f: DisplayFeature = serde_json::from_value(v["feature"].clone())?;
                    if f.pixels.len() != f.track_ids.len()
                        || f.pixels.len() != f.points_camera.len()
                    {
                        return Err("misaligned display arrays".into());
                    }
                    let observations = f
                        .track_ids
                        .into_iter()
                        .zip(f.pixels)
                        .zip(f.points_camera)
                        .map(|((track_id, pixel), point_camera)| LandmarkObservation {
                            track_id,
                            pixel,
                            point_camera,
                        })
                        .collect();
                    // Direct insertion allows legacy journals from multiple robots.
                    if b.records.contains_key(&f.key) && !b.bundle_frames.contains_key(&f.key) {
                        b.insert_bundle_frame(BundleFrame {
                            key: f.key,
                            timestamp_ns: f.timestamp_ns,
                            views: vec![CameraObservations {
                                camera: f.camera,
                                observations,
                            }],
                        })?;
                    }
                }
            }
        }
    }
    fs::write(
        output.join("config.json"),
        serde_json::to_vec_pretty(&b.config)?,
    )?;
    fs::write(
        output.join("inputs.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"sources":sources,"legacy_left_camera_only":legacy,"keyframes":b.records.len(),"bundle_frames":b.bundle_frames.len(),"max_keyframes":limit}),
        )?,
    )?;
    let mut pgo = b.clone();
    pgo.config.mode = BackendMode::PoseGraph;
    let baseline = pgo.solve(&GraphSnapshot::default())?;
    save(output, "pgo", &baseline)?;
    let result = b.solve(&GraphSnapshot::default())?;
    save(output, "global_ba", &result)?;
    let report = serde_json::json!({"keyframes":result.poses.len(),"components":result.components,"loops":result.loops.len(),"landmarks":result.landmarks.len(),"legacy_left_camera_only":legacy,"pgo_ms":baseline.solve_ms,"global_ba_ms":result.solve_ms,"initial_cost":result.initial_cost,"final_cost":result.final_cost,"optimizer_reports":result.optimizer_reports,"bundle_diagnostics":result.bundle_diagnostics});
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
