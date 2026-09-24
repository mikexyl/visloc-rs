use crate::{
    robot::Peer,
    wire::{self, Wire},
    AnyResult,
};
use rclrs::*;
use std::{
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    sync::{mpsc, Arc, Mutex},
    time::{Duration, Instant},
};
use visloc_msgs::{msg as m, srv as s};
use visloc_multi_robot::{
    self as c,
    backend::{Backend, BackendConfig},
};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub peers: Vec<String>,
    pub output: PathBuf,
    #[serde(default)]
    pub pgo: BackendConfig,
}
enum Update {
    Keyframe(c::KeyframeRecord),
    Loop(c::LoopConstraint),
    Finish,
}
pub fn run(config: Config) -> AnyResult<()> {
    std::fs::create_dir_all(&config.output)?;
    std::fs::write(
        config.output.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    let finished = config.output.join("finished.json");
    if finished.exists() {
        std::fs::remove_file(finished)?;
    }
    let mut executor = Context::default_from_env()?.create_basic_executor();
    let node = executor.create_node("visloc_central_pgo")?;
    let (tx, rx) = mpsc::sync_channel(8192);
    let (solutions, results) = mpsc::sync_channel(2);
    let mut backend = Backend::default();
    backend.config = config.pgo.clone();
    let state = Arc::new(Mutex::new(backend));
    // Replay the persisted graph when the backend restarts. Updates are
    // immutable and idempotent; live peer histories recover any missing tail.
    let journal_path = config.output.join("graph.jsonl");
    if journal_path.exists() {
        // A process may stop during the final write. Recover the complete
        // prefix; peer history will resend any uncommitted tail record.
        let bytes = std::fs::read(&journal_path)?;
        if !bytes.is_empty() && bytes.last() != Some(&b'\n') {
            let length = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |p| p + 1);
            std::fs::OpenOptions::new()
                .write(true)
                .open(&journal_path)?
                .set_len(length as u64)?;
        }
        for line in std::fs::read_to_string(&journal_path)?.lines() {
            let v: serde_json::Value = serde_json::from_str(line)?;
            if let Some(r) = v.get("keyframe") {
                state
                    .lock()
                    .unwrap()
                    .insert_keyframe(serde_json::from_value(r.clone())?)?;
            }
            if let Some(r) = v.get("loop") {
                state
                    .lock()
                    .unwrap()
                    .insert_loop(serde_json::from_value(r.clone())?)?;
            }
        }
    }
    let worker_state = state.clone();
    let worker_config = config.clone();
    std::thread::Builder::new()
        .name("central-pgo".into())
        .spawn(move || {
            if let Err(e) = optimize(
                worker_config.clone(),
                worker_state,
                journal_path,
                rx,
                solutions,
            ) {
                eprintln!("central backend failed: {e}");
                let _ = std::fs::write(
                    worker_config.output.join("backend_error.txt"),
                    e.to_string(),
                );
            }
        })?;
    let mut subscriptions = Vec::new();
    let mut status_subscriptions = Vec::new();
    let active_sessions = Arc::new(Mutex::new(BTreeMap::<String, String>::new()));
    for robot in &config.peers {
        let active = active_sessions.clone();
        status_subscriptions.push(
            node.create_subscription(
                format!("/{robot}/slam/status")
                    .as_str()
                    .reliable()
                    .transient_local()
                    .keep_last(1),
                move |s: m::Status| {
                    if !s.session.is_empty() {
                        active.lock().unwrap().insert(s.robot, s.session);
                    }
                },
            )?,
        );
        let tx = tx.clone();
        subscriptions.push(
            node.create_subscription(
                format!("/{robot}/slam/keyframes")
                    .as_str()
                    .reliable()
                    .keep_last(256),
                move |r: m::Keyframe| {
                    if let Err(e) = tx.try_send(Update::Keyframe(c::KeyframeRecord::from_wire(r))) {
                        crate::traffic::dropped(crate::traffic::Queue::Graph);
                        eprintln!("keyframe queue: {e}");
                    }
                },
            )?,
        );
    }
    let loop_tx = tx.clone();
    let loop_sub = node.create_subscription(
        "/visloc/loops".reliable().keep_last(256),
        move |r: m::LoopConstraint| {
            if let Err(e) = loop_tx.try_send(Update::Loop(c::LoopConstraint::from_wire(r))) {
                crate::traffic::dropped(crate::traffic::Queue::Graph);
                eprintln!("loop queue: {e}");
            }
        },
    )?;
    let finish_tx = tx.clone();
    let finish = node.create_service::<s::Finish, _>(
        "/visloc/backend/finish",
        move |_: s::Finish_Request| s::Finish_Response {
            accepted: finish_tx.try_send(Update::Finish).is_ok(),
        },
    )?;
    {
        let node = node.clone();
        let config = config.clone();
        std::thread::Builder::new()
            .name("graph-history-recovery".into())
            .spawn(move || {
                let result = (|| -> AnyResult<()> {
                    let mut peers: Vec<_> = config
                        .peers
                        .iter()
                        .map(|r| Peer::new(&node, r, true))
                        .collect::<AnyResult<_>>()?;
                    loop {
                        for peer in &mut peers {
                            if !peer.history.service_is_ready()? {
                                continue;
                            }
                            match peer.page() {
                                Ok(page) => {
                                    for r in page.keyframes {
                                        tx.send(Update::Keyframe(c::KeyframeRecord::from_wire(r)))?;
                                    }
                                    for r in page.loops {
                                        tx.send(Update::Loop(c::LoopConstraint::from_wire(r)))?;
                                    }
                                }
                                Err(e) => eprintln!("graph history retry: {e}"),
                            }
                        }
                        std::thread::sleep(Duration::from_millis(250));
                    }
                })();
                if let Err(e) = result {
                    eprintln!("graph history failed: {e}");
                }
            })?;
    }
    let graph = node.create_publisher::<m::GraphSnapshot>(
        "/visloc/graph".reliable().transient_local().keep_last(1),
    )?;
    let transforms =
        node.create_publisher::<tf2_msgs::msg::TFMessage>("/tf".reliable().keep_last(64))?;
    let paths: BTreeMap<_, _> = config
        .peers
        .iter()
        .map(|r| {
            Ok((
                r.clone(),
                node.create_publisher::<nav_msgs::msg::Path>(
                    format!("/{r}/slam/path")
                        .as_str()
                        .reliable()
                        .transient_local()
                        .keep_last(1),
                )?,
            ))
        })
        .collect::<AnyResult<_>>()?;
    let pump = node.create_timer_repeating(Duration::from_millis(50), move || {
        for snapshot in results.try_iter() {
            let result = (|| -> AnyResult<()> {
                graph.publish(snapshot.wire())?;
                for (robot, publisher) in &paths {
                    let mut poses: Vec<_> = snapshot
                        .poses
                        .iter()
                        .filter(|p| {
                            p.key.robot == *robot
                                && active_sessions
                                    .lock()
                                    .unwrap()
                                    .get(robot)
                                    .is_none_or(|session| p.key.session == *session)
                        })
                        .collect();
                    poses.sort_by_key(|p| p.timestamp_ns);
                    if let Some(last) = poses.last() {
                        let frame = format!(
                            "map_{}_{}_{}",
                            last.component.robot, last.component.session, last.component.id
                        );
                        let header = std_msgs::msg::Header {
                            frame_id: frame.clone(),
                            stamp: wire::stamp(last.timestamp_ns),
                        };
                        publisher.publish(nav_msgs::msg::Path {
                            header: header.clone(),
                            poses: poses
                                .iter()
                                .filter(|p| p.component == last.component)
                                .map(|p| geometry_msgs::msg::PoseStamped {
                                    header: std_msgs::msg::Header {
                                        frame_id: frame.clone(),
                                        stamp: wire::stamp(p.timestamp_ns),
                                    },
                                    pose: wire::pose(&p.body_to_map),
                                })
                                .collect(),
                        })?;
                        transforms.publish(tf2_msgs::msg::TFMessage {
                            transforms: vec![wire::tf(
                                &last.map_from_odom,
                                &frame,
                                &format!("{robot}/odom"),
                                last.timestamp_ns,
                            )],
                        })?;
                    }
                }
                Ok(())
            })();
            if let Err(e) = result {
                eprintln!("graph publisher: {e}");
            }
        }
    })?;
    let _keep = (subscriptions, status_subscriptions, loop_sub, finish, pump);
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
/// Only the optimization worker owns updates to this snapshot. ROS callbacks
/// enqueue updates while a solve is in progress; they never lock or solve it.
fn optimize(
    config: Config,
    state: Arc<Mutex<Backend>>,
    journal_path: PathBuf,
    rx: mpsc::Receiver<Update>,
    solutions: mpsc::SyncSender<c::GraphSnapshot>,
) -> AnyResult<()> {
    let mut journal = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(journal_path)?;
    let mut revisions = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(config.output.join("revisions.jsonl"))?;
    let snapshot_path = config.output.join("graph_snapshot.json");
    let mut previous = if snapshot_path.exists() {
        serde_json::from_slice(&std::fs::read(&snapshot_path)?)?
    } else {
        c::GraphSnapshot::default()
    };
    let mut last_solve = Instant::now() - Duration::from_secs(2);
    let mut force = false;
    let mut publish_recovered = snapshot_path.exists();
    loop {
        let first = rx.recv_timeout(Duration::from_millis(50)).ok();
        for update in first.into_iter().chain(rx.try_iter()) {
            let mut state = state.lock().unwrap();
            match update {
                Update::Keyframe(r) => {
                    if state.insert_keyframe(r.clone())? {
                        writeln!(journal, "{}", serde_json::json!({"keyframe":r}))?;
                    }
                }
                Update::Loop(r) => {
                    if state.insert_loop(r.clone())? {
                        writeln!(journal, "{}", serde_json::json!({"loop":r}))?;
                    }
                }
                Update::Finish => force = true,
            }
        }
        let dirty =
            state.lock().unwrap().input_revision != previous.input_revision || publish_recovered;
        if (force || last_solve.elapsed() >= Duration::from_secs(1)) && (dirty || force) {
            let graph = state.lock().unwrap().clone();
            match graph.solve(&previous) {
                Ok(snapshot) => {
                    atomic_json(&snapshot_path, &snapshot)?;
                    writeln!(
                        revisions,
                        "{}",
                        serde_json::json!({"revision":snapshot.revision,"input_revision":snapshot.input_revision,"keyframes":snapshot.poses.len(),"loops":snapshot.loops.len(),"components":snapshot.components,"initial_cost":snapshot.initial_cost,"final_cost":snapshot.final_cost,"solve_ms":snapshot.solve_ms})
                    )?;
                    for robot in &config.peers {
                        let mut out = std::fs::File::create(
                            config.output.join(format!("{robot}_keyframes.tum")),
                        )?;
                        for p in snapshot.poses.iter().filter(|p| p.key.robot == *robot) {
                            let [x, y, z] = p.body_to_map.translation;
                            let [qx, qy, qz, qw] = p.body_to_map.rotation_xyzw;
                            writeln!(
                                out,
                                "{:.9} {x:.9} {y:.9} {z:.9} {qx:.12} {qy:.12} {qz:.12} {qw:.12}",
                                p.timestamp_ns as f64 * 1e-9
                            )?;
                        }
                    }
                    solutions.send(snapshot.clone())?;
                    previous = snapshot;
                    publish_recovered = false;
                    if force {
                        atomic_json(
                            &config.output.join("communication.json"),
                            &crate::traffic::snapshot(),
                        )?;
                        atomic_json(
                            &config.output.join("finished.json"),
                            &serde_json::json!({"revision":previous.revision,"input_revision":previous.input_revision}),
                        )?;
                    }
                }
                Err(e) => {
                    eprintln!("PGO rejected: {e}");
                    writeln!(
                        revisions,
                        "{}",
                        serde_json::json!({"event":"rejected_solution","input_revision":graph.input_revision,"error":e.to_string()})
                    )?;
                    std::fs::write(config.output.join("optimizer_error.txt"), e.to_string())?;
                }
            }
            journal.flush()?;
            revisions.flush()?;
            last_solve = Instant::now();
            force = false;
        }
    }
}
fn atomic_json(path: &std::path::Path, value: &impl serde::Serialize) -> AnyResult<()> {
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
