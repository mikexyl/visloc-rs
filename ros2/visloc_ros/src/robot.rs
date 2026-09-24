use crate::{
    sensor::{self, Input, PreprocessConfig},
    wire::{self, Wire},
    AnyResult,
};
use rclrs::*;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use visloc_msgs::{msg as m, srv as s};
use visloc_multi_robot::{
    self as c,
    native::Encoder,
    sequence::{refine, Retrieval},
};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotConfig {
    pub robot: String,
    pub peers: Vec<String>,
    pub calibration: PathBuf,
    pub vio_config: PathBuf,
    pub loop_config: PathBuf,
    pub output: PathBuf,
    pub preprocess: Option<PreprocessConfig>,
    #[serde(default = "default_imu_startup")]
    pub imu_startup: Option<visloc_basalt::startup::StationaryStartupConfig>,
    #[serde(default)]
    pub reliable_sensors: bool,
    #[serde(default)]
    pub fixed_last_frame: bool,
    #[serde(default = "enabled")]
    pub loop_enabled: bool,
    #[serde(default = "default_capacity")]
    pub max_keyframes: usize,
}
fn default_imu_startup() -> Option<visloc_basalt::startup::StationaryStartupConfig> {
    Some(Default::default())
}
fn default_capacity() -> usize {
    4000
}
fn enabled() -> bool {
    true
}

#[cfg(test)]
mod config_tests {
    use super::RobotConfig;

    fn required_config() -> serde_json::Value {
        serde_json::json!({
            "robot": "test", "peers": ["test"],
            "calibration": "calibration.json", "vio_config": "vio.json",
            "loop_config": "loop.json", "output": "output"
        })
    }

    #[test]
    fn omitted_startup_enables_stationary_gyro_only() {
        let config: RobotConfig = serde_json::from_value(required_config()).unwrap();
        let startup = config.imu_startup.as_ref().unwrap();
        assert!(!startup.average_gravity);
        assert!(!startup.wait_for_motion);
        assert_eq!(startup.window_ns, 1_000_000_000);
        let saved = serde_json::to_value(config).unwrap();
        assert!(saved.get("scalar_mode").is_none());
        assert!(saved["imu_startup"].is_object());
    }

    #[test]
    fn explicit_null_preserves_legacy_startup_override() {
        let mut value = required_config();
        value["imu_startup"] = serde_json::Value::Null;
        let config: RobotConfig = serde_json::from_value(value).unwrap();
        assert!(config.imu_startup.is_none());
    }

    #[test]
    fn obsolete_precision_setting_is_rejected() {
        for mode in ["f32", "f64"] {
            let mut value = required_config();
            value["scalar_mode"] = mode.into();
            let error = serde_json::from_value::<RobotConfig>(value).err().unwrap();
            assert!(error.to_string().contains("unknown field `scalar_mode`"));
        }
    }
}
pub enum LoopCommand {
    Frame(visloc_online_loop::Frame, c::KeyframeRecord),
    Verify(c::Pair, f32, c::FeatureFrame, c::FeatureFrame, Instant),
    Finish,
}
pub enum Event {
    Odometry {
        timestamp_ns: i64,
        pose: c::Transform,
        frame_id: u64,
        process_ms: f64,
    },
    Keyframe(c::KeyframeRecord),
    Sequence(c::Sequence, f64),
    Loop(c::LoopConstraint),
    Verified(c::Pair, c::Verification, f64, f64),
    VioFinished,
    Error(String),
}
enum Comm {
    Sequence(c::Sequence),
    Proposal(c::Pair, f32),
}

#[derive(Default)]
pub struct History {
    pub session: String,
    pub attempted_pairs: BTreeSet<c::Pair>,
    pub keyframes: Vec<c::KeyframeRecord>,
    pub sequences: Vec<c::Sequence>,
    pub loops: Vec<c::LoopConstraint>,
    pub descriptors: BTreeMap<c::Key, c::FrameDescriptors>,
    pub features: BTreeMap<c::Key, c::FeatureFrame>,
}
impl History {
    pub fn page(&self, r: s::GetHistory_Request) -> s::GetHistory_Response {
        let take = |len: usize, cursor: u64| {
            let start = (cursor as usize).min(len);
            (start, (start + 64).min(len))
        };
        let (a, b) = take(self.keyframes.len(), r.keyframe_cursor);
        let (d, e) = take(self.sequences.len(), r.sequence_cursor);
        let (f, g) = take(self.loops.len(), r.loop_cursor);
        s::GetHistory_Response {
            session: self.session.clone(),
            keyframes: if r.include_keyframes {
                self.keyframes[a..b].iter().map(Wire::wire).collect()
            } else {
                vec![]
            },
            sequences: self.sequences[d..e].iter().map(Wire::wire).collect(),
            loops: if r.include_loops {
                self.loops[f..g].iter().map(Wire::wire).collect()
            } else {
                vec![]
            },
            keyframe_cursor: if r.include_keyframes {
                b as u64
            } else {
                self.keyframes.len() as u64
            },
            sequence_cursor: e as u64,
            loop_cursor: if r.include_loops {
                g as u64
            } else {
                self.loops.len() as u64
            },
            more: (r.include_keyframes && b < self.keyframes.len())
                || e < self.sequences.len()
                || (r.include_loops && g < self.loops.len()),
        }
    }
}

/// A ROS async call with a bounded wait on the dedicated communication thread.
/// The executor, sensor thread and GPU thread never wait here.
pub fn request<S>(
    node: &Node,
    client: &mut Client<S>,
    request: S::Request,
) -> AnyResult<S::Response>
where
    S: rosidl_runtime_rs::Service,
    S::Request: Clone + crate::traffic::Payload,
    S::Response: Send + crate::traffic::Payload + 'static,
{
    let service = client.service_name();
    for attempt in 0..=3 {
        crate::traffic::sent(&request);
        let started = Instant::now();
        let result = match client.call::<_, S::Response>(request.clone()) {
            Ok(mut promise) => loop {
                match promise.try_recv() {
                    Ok(Some(response)) => break Ok(response),
                    Err(error) => break Err(error.to_string()),
                    Ok(None) if started.elapsed() >= Duration::from_secs(2) => {
                        break Err("service response exceeded two seconds".to_owned());
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(2)),
                }
            },
            Err(error) => Err(error.to_string()),
        };
        if let Ok(response) = result {
            crate::traffic::received(&response);
            return Ok(response);
        }
        // rclrs 0.7 keeps unanswered requests in its private request board,
        // even after the response Promise is dropped. Replacing the client
        // removes that waitable/board so prolonged outages remain bounded.
        *client = node.create_client(service.as_str())?;
        if attempt == 3 {
            return Err(result.err().unwrap().into());
        }
    }
    unreachable!()
}

pub struct Peer {
    node: Node,
    pub history: Client<s::GetHistory>,
    pub descriptors: Client<s::GetSequence>,
    pub features: Client<s::GetFeatures>,
    pub cursor: s::GetHistory_Request,
    session: String,
}
impl Peer {
    pub fn new(node: &Node, robot: &str, graph: bool) -> AnyResult<Self> {
        Ok(Self {
            node: node.clone(),
            history: node.create_client(format!("/{robot}/slam/history").as_str())?,
            descriptors: node.create_client(format!("/{robot}/slam/sequence_frames").as_str())?,
            features: node.create_client(format!("/{robot}/slam/features").as_str())?,
            cursor: s::GetHistory_Request {
                include_keyframes: graph,
                include_loops: graph,
                ..Default::default()
            },
            session: String::new(),
        })
    }
    pub fn page(&mut self) -> AnyResult<s::GetHistory_Response> {
        let mut result = request(&self.node, &mut self.history, self.cursor.clone())?;
        if self.session != result.session {
            self.session = result.session.clone();
            self.cursor.keyframe_cursor = 0;
            self.cursor.sequence_cursor = 0;
            self.cursor.loop_cursor = 0;
            result = request(&self.node, &mut self.history, self.cursor.clone())?;
        }
        self.cursor.keyframe_cursor = result.keyframe_cursor;
        self.cursor.sequence_cursor = result.sequence_cursor;
        self.cursor.loop_cursor = result.loop_cursor;
        Ok(result)
    }
}

fn communications(
    config: RobotConfig,
    owner: c::Key,
    node: Node,
    rx: Receiver<Comm>,
    tx: SyncSender<LoopCommand>,
    history: Arc<Mutex<History>>,
    status: Arc<Mutex<m::Status>>,
    pending: Arc<AtomicUsize>,
    vio_done: Arc<AtomicBool>,
    encoder_done: Arc<AtomicBool>,
) -> AnyResult<()> {
    let proposals =
        node.create_publisher::<m::Proposal>("/visloc/proposals".reliable().keep_last(256))?;
    let mut peers: BTreeMap<_, _> = config
        .peers
        .iter()
        .filter(|r| **r != owner.robot)
        .map(|r| Ok((r.clone(), Peer::new(&node, r, false)?)))
        .collect::<AnyResult<_>>()?;
    let mut retrieval = Retrieval::default();
    let mut seen = history.lock().unwrap().attempted_pairs.clone();
    let mut jobs = VecDeque::new();
    let mut deferred = VecDeque::new();
    let mut retry_after = BTreeMap::<c::Pair, Instant>::new();
    let mut last_history = Instant::now() - Duration::from_secs(2);
    let mut last_activity = Instant::now();
    let mut local_cursor = 0;
    let mut log = std::fs::File::create(config.output.join("retrieval.jsonl"))?;
    loop {
        let mut incoming = Vec::new();
        if let Ok(message) = rx.recv_timeout(Duration::from_millis(20)) {
            incoming.push(message);
        }
        incoming.extend(rx.try_iter());
        if last_history.elapsed() >= Duration::from_secs(1) {
            for (robot, peer) in &mut peers {
                if !peer.history.service_is_ready()? {
                    continue;
                }
                match peer.page() {
                    Ok(page) => incoming.extend(
                        page.sequences
                            .into_iter()
                            .map(|s| Comm::Sequence(c::Sequence::from_wire(s))),
                    ),
                    Err(e) => {
                        status.lock().unwrap().request_failures += 1;
                        writeln!(
                            log,
                            "{}",
                            serde_json::json!({"event":"history_request_failed","robot":robot,"error":e.to_string()})
                        )?;
                    }
                }
            }
            {
                let h = history.lock().unwrap();
                incoming.extend(
                    h.sequences[local_cursor..]
                        .iter()
                        .cloned()
                        .map(Comm::Sequence),
                );
                local_cursor = h.sequences.len();
            }
            last_history = Instant::now();
        }
        for message in incoming {
            match message {
                Comm::Sequence(sequence) => {
                    if sequence.key.robot != owner.robot
                        && !config.peers.contains(&sequence.key.robot)
                    {
                        continue;
                    }
                    if !retrieval.insert(sequence.clone())? {
                        continue;
                    }
                    last_activity = Instant::now();
                    status.lock().unwrap().finished = false;
                    if !config.loop_enabled {
                        continue;
                    }
                    let (candidates, stats) =
                        retrieval.candidates_for(&sequence, 0.8, Some(&owner));
                    writeln!(
                        log,
                        "{}",
                        serde_json::json!({"event":"retrieval","query":sequence.key,"stats":stats,"candidates":candidates})
                    )?;
                    for (candidate, similarity) in candidates {
                        let pair = c::Pair::new(sequence.key.clone(), candidate);
                        if pair.0.robot == owner.robot {
                            jobs.push_back((pair, similarity));
                        } else {
                            proposals.publish(m::Proposal {
                                sequence_from: pair.0.wire(),
                                sequence_to: pair.1.wire(),
                                similarity,
                            })?;
                        }
                    }
                }
                Comm::Proposal(pair, similarity) => {
                    if config.loop_enabled && pair.0.robot == owner.robot {
                        jobs.push_back((pair, similarity));
                    }
                }
            }
        }
        jobs.append(&mut deferred);
        if let Some((pair, _)) = jobs.pop_front() {
            if retry_after.get(&pair).is_some_and(|t| Instant::now() < *t) {
                deferred.push_back((pair, 0.));
            } else if !seen.contains(&pair) {
                if let (Some(a), Some(b)) = (
                    retrieval.sequences.get(&pair.0),
                    retrieval.sequences.get(&pair.1),
                ) {
                    let similarity = c::sequence::dot(&a.descriptor, &b.descriptor);
                    if a.model_id == b.model_id && similarity >= 0.8 {
                        status.lock().unwrap().finished = false;
                        last_activity = Instant::now();
                        // Resolve matrices first; feature service traffic starts
                        // only after the 5x5 refinement selects one pair.
                        let exchange_started = Instant::now();
                        let attempt = (|| -> AnyResult<()> {
                            let mut descriptors = |key: &c::Key| -> AnyResult<c::FrameDescriptors> {
                                if key.robot == owner.robot {
                                    return history
                                        .lock()
                                        .unwrap()
                                        .descriptors
                                        .get(key)
                                        .cloned()
                                        .ok_or_else(|| "missing local sequence matrix".into());
                                }
                                let response = request(
                                    &node,
                                    &mut peers
                                        .get_mut(&key.robot)
                                        .ok_or("unknown peer")?
                                        .descriptors,
                                    s::GetSequence_Request { key: key.wire() },
                                )?;
                                if !response.found {
                                    return Err("remote sequence matrix unavailable".into());
                                }
                                let result = c::FrameDescriptors::from_wire(response.descriptors);
                                result.validate()?;
                                if result.sequence != *key {
                                    return Err("sequence response identity mismatch".into());
                                }
                                Ok(result)
                            };
                            let da = descriptors(&pair.0)?;
                            let db = descriptors(&pair.1)?;
                            if da.frames != a.selected || db.frames != b.selected {
                                return Err("sequence membership mismatch".into());
                            }
                            let (ka, kb, frame_similarity) = if config.fixed_last_frame {
                                (
                                    da.frames[4].clone(),
                                    db.frames[4].clone(),
                                    c::sequence::dot(
                                        &da.descriptors[2048..],
                                        &db.descriptors[2048..],
                                    ),
                                )
                            } else {
                                refine(&da, &db)?
                            };
                            let mut feature = |key: &c::Key| -> AnyResult<c::FeatureFrame> {
                                if key.robot == owner.robot {
                                    return history
                                        .lock()
                                        .unwrap()
                                        .features
                                        .get(key)
                                        .cloned()
                                        .ok_or_else(|| "missing local features".into());
                                }
                                let response = request(
                                    &node,
                                    &mut peers.get_mut(&key.robot).ok_or("unknown peer")?.features,
                                    s::GetFeatures_Request { key: key.wire() },
                                )?;
                                if !response.found {
                                    return Err("remote feature packet unavailable".into());
                                }
                                let result = c::FeatureFrame::from_wire(response.frame);
                                result.validate()?;
                                if result.key != *key {
                                    return Err("feature response identity mismatch".into());
                                }
                                Ok(result)
                            };
                            let fa = feature(&ka)?;
                            let fb = feature(&kb)?;
                            writeln!(
                                log,
                                "{}",
                                serde_json::json!({"event":"refinement","pair":pair,"from":ka,"to":kb,"similarity":similarity,"frame_similarity":frame_similarity,"fixed_last_frame":config.fixed_last_frame})
                            )?;
                            pending.fetch_add(1, Ordering::SeqCst);
                            if tx
                                .send(LoopCommand::Verify(
                                    pair.clone(),
                                    similarity,
                                    fa,
                                    fb,
                                    exchange_started,
                                ))
                                .is_err()
                            {
                                pending.fetch_sub(1, Ordering::SeqCst);
                                return Err("loop worker disconnected".into());
                            }
                            Ok(())
                        })();
                        if let Err(e) = attempt {
                            status.lock().unwrap().request_failures += 1;
                            writeln!(
                                log,
                                "{}",
                                serde_json::json!({"event":"verification_request_failed","pair":pair,"error":e.to_string()})
                            )?;
                            retry_after
                                .insert(pair.clone(), Instant::now() + Duration::from_secs(5));
                            deferred.push_back((pair, 0.));
                            continue;
                        }
                    }
                    retry_after.remove(&pair);
                    seen.insert(pair);
                } else {
                    deferred.push_back((pair, 0.));
                }
            }
        }
        if jobs.len() + deferred.len() > 4096 {
            return Err("verification queue exceeded 4096 entries".into());
        }
        status.lock().unwrap().finished = vio_done.load(Ordering::SeqCst)
            && encoder_done.load(Ordering::SeqCst)
            && pending.load(Ordering::SeqCst) == 0
            && jobs.is_empty()
            && deferred.is_empty()
            && last_activity.elapsed() > Duration::from_secs(3);
    }
}

pub fn run(mut config: RobotConfig) -> AnyResult<()> {
    let owner = c::Key::new(&config.robot, &uuid::Uuid::new_v4().simple().to_string(), 0);
    owner.validate()?;
    let root = config.output.clone();
    if root.exists() && !root.join("config.json").exists() {
        return Err("existing output is not a visloc robot archive".into());
    }
    let restored =
        crate::archive::restore(&root, &owner.robot, &owner.session, config.max_keyframes)?;
    let archived_keyframes = restored.keyframes.len();
    if root.exists() {
        config.output = root.join(format!("session-{}", owner.session));
    }
    std::fs::create_dir_all(&config.output)?;
    std::fs::write(
        root.join("active_session.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"session":owner.session,"output":config.output}),
        )?,
    )?;
    std::fs::copy(
        &config.calibration,
        config.output.join("basalt_calibration.json"),
    )?;
    config.calibration = config.output.join("basalt_calibration.json");
    std::fs::copy(&config.vio_config, config.output.join("basalt_config.json"))?;
    config.vio_config = config.output.join("basalt_config.json");
    std::fs::copy(&config.loop_config, config.output.join("loop_config.json"))?;
    if let Some(parent) = config.loop_config.parent() {
        let manifest = parent.join("manifest.json");
        if manifest.exists() {
            std::fs::copy(manifest, config.output.join("model_manifest.json"))?;
        }
    }
    std::fs::write(
        config.output.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    std::fs::write(
        config.output.join("session.json"),
        serde_json::to_vec_pretty(&owner)?,
    )?;
    let mut executor = Context::default_from_env()?.create_basic_executor();
    let node = executor.create_node(format!("{}_slam", config.robot).as_str())?;
    let history = Arc::new(Mutex::new(restored));
    let status = Arc::new(Mutex::new(m::Status {
        robot: owner.robot.clone(),
        session: owner.session.clone(),
        ..Default::default()
    }));
    let pending = Arc::new(AtomicUsize::new(0));
    let vio_done = Arc::new(AtomicBool::new(false));
    let encoder_done = Arc::new(AtomicBool::new(false));
    let (input_tx, input_rx) = mpsc::sync_channel(8192);
    let (loop_tx, loop_rx) = mpsc::sync_channel(4);
    let (event_tx, event_rx) = mpsc::sync_channel(1024);
    let (comm_tx, comm_rx) = mpsc::sync_channel(512);
    let event_errors = event_tx.clone();
    {
        let config = config.clone();
        let owner = owner.clone();
        let events = event_tx.clone();
        let state = status.clone();
        let tx = loop_tx.clone();
        std::thread::Builder::new()
            .name("basalt-vio".into())
            .spawn(move || {
                if let Err(e) = sensor::run(config, owner, input_rx, tx, events.clone(), state) {
                    let _ = events.send(Event::Error(e.to_string()));
                }
            })?;
    }
    {
        let config = config.clone();
        let owner = owner.clone();
        let events = event_tx.clone();
        let history = history.clone();
        let state = status.clone();
        let done = encoder_done.clone();
        std::thread::Builder::new().name("jist-xfeat-refinement".into()).spawn(move || {
            let result=(||->AnyResult<()> {
                let mut attempts=std::fs::File::create(config.output.join("attempts.jsonl"))?;
                let calibration=visloc_basalt::BasaltCalibration::from_path(&config.calibration)?;let cam=&calibration.cameras[0];
                let (w,h)=calibration.resolutions[0];
                let camera=c::CameraModel {width:w,height:h,intrinsics:[cam.fx,cam.fy,cam.cx,cam.cy],camera_to_body:c::Transform::from(calibration.camera_to_imu(0).unwrap())};
                if cam.xi!=0. || cam.alpha!=0. {return Err("loop inference requires a rectified pinhole camera".into());}
                let mut encoder=if config.loop_enabled {Some(Encoder::new(visloc_online_loop::Config::from_path(&config.loop_config)?,camera,owner)?)}else{None};
                state.lock().unwrap().ready=true;
                for command in loop_rx {
                    match command {
                        LoopCommand::Frame(frame,record)=> {
                            let started=Instant::now();
                            if record.key.id as usize+archived_keyframes>=config.max_keyframes {return Err("configured keyframe capacity reached; VIO history remains on disk".into());}
                            if let Some(encoded)=if let Some(encoder)=&mut encoder {encoder.push(frame,record)?}else{None} {
                                let archive=config.output.join("sequences");std::fs::create_dir_all(&archive)?;
                                crate::archive::store_json(&archive.join(format!("{:06}.json",encoded.sequence.key.id)),&serde_json::json!({"sequence":encoded.sequence,"descriptors":encoded.descriptors,"features":encoded.features}))?;
                                {let mut h=history.lock().unwrap();h.descriptors.insert(encoded.sequence.key.clone(),encoded.descriptors);
                                for f in encoded.features {h.features.insert(f.key.clone(),f);}h.sequences.push(encoded.sequence.clone());}
                                events.send(Event::Sequence(encoded.sequence,started.elapsed().as_secs_f64()*1000.))?;
                            }
                        },
                        LoopCommand::Verify(pair,similarity,a,b,exchange_started)=> {
                            writeln!(attempts,"{}",serde_json::to_string(&pair)?)?;
                            history.lock().unwrap().attempted_pairs.insert(pair.clone());
                            let started=Instant::now();
                            let result=encoder.as_mut().ok_or("loop processing is disabled")?.verify(&a,&b,pair.clone(),similarity);
                            let (edge,diagnostic)=result?;
                            if let Some(edge)=edge {events.send(Event::Loop(edge))?;}
                            events.send(Event::Verified(pair,diagnostic,started.elapsed().as_secs_f64()*1000.,exchange_started.elapsed().as_secs_f64()*1000.))?;
                        },
                        LoopCommand::Finish=>done.store(true,Ordering::SeqCst),
                    }
                }Ok(())
            })();
            if let Err(e)=result {let _=events.send(Event::Error(e.to_string()));}
        })?;
    }
    let image_topic = format!("/{}/camera/image", config.robot);
    let imu_topic = format!("/{}/imu", config.robot);
    let image_options = if config.reliable_sensors {
        image_topic.as_str().reliable().keep_last(32)
    } else {
        image_topic.as_str().best_effort().keep_last(8)
    };
    let tx = input_tx.clone();
    let errors = event_errors.clone();
    let image_sub =
        node.create_subscription(image_options, move |image: sensor_msgs::msg::Image| {
            if let Err(e) = tx.try_send(Input::Image(image)) {
                crate::traffic::dropped(crate::traffic::Queue::Sensor);
                let _ = errors.try_send(Event::Error(format!("sensor queue overflow: {e}")));
            }
        })?;
    let imu_options = if config.reliable_sensors {
        imu_topic.as_str().reliable().keep_last(2048)
    } else {
        imu_topic.as_str().best_effort().keep_last(512)
    };
    let tx = input_tx.clone();
    let errors = event_errors.clone();
    let imu_sub = node.create_subscription(imu_options, move |imu: sensor_msgs::msg::Imu| {
        if let Err(e) = tx.try_send(Input::Imu(imu)) {
            crate::traffic::dropped(crate::traffic::Queue::Sensor);
            let _ = errors.try_send(Event::Error(format!("IMU queue overflow: {e}")));
        }
    })?;
    let h = history.clone();
    let history_service = node.create_service::<s::GetHistory, _>(
        format!("/{}/slam/history", config.robot).as_str(),
        move |r: s::GetHistory_Request| h.lock().unwrap().page(r),
    )?;
    let h = history.clone();
    let sequence_service = node.create_service::<s::GetSequence, _>(
        format!("/{}/slam/sequence_frames", config.robot).as_str(),
        move |r: s::GetSequence_Request| {
            let h = h.lock().unwrap();
            let d = h.descriptors.get(&c::Key::from_wire(r.key));
            s::GetSequence_Response {
                found: d.is_some(),
                descriptors: d.map(Wire::wire).unwrap_or_default(),
            }
        },
    )?;
    let h = history.clone();
    let feature_service = node.create_service::<s::GetFeatures, _>(
        format!("/{}/slam/features", config.robot).as_str(),
        move |r: s::GetFeatures_Request| {
            let h = h.lock().unwrap();
            let f = h.features.get(&c::Key::from_wire(r.key));
            s::GetFeatures_Response {
                found: f.is_some(),
                frame: f.map(Wire::wire).unwrap_or_default(),
            }
        },
    )?;
    let tx = input_tx;
    let finish_service = node.create_service::<s::Finish, _>(
        format!("/{}/slam/finish", config.robot).as_str(),
        move |r: s::Finish_Request| s::Finish_Response {
            accepted: tx
                .try_send(Input::Finish(r.last_image_timestamp_ns))
                .is_ok(),
        },
    )?;
    let tx = comm_tx.clone();
    let announcement_sub = node.create_subscription(
        "/visloc/sequences".reliable().keep_last(256),
        move |s: m::SequenceAnnouncement| {
            if tx
                .try_send(Comm::Sequence(c::Sequence::from_wire(s)))
                .is_err()
            {
                crate::traffic::dropped(crate::traffic::Queue::Communication);
            }
        },
    )?;
    let tx = comm_tx.clone();
    let proposal_sub = node.create_subscription(
        "/visloc/proposals".reliable().keep_last(256),
        move |p: m::Proposal| {
            if tx
                .try_send(Comm::Proposal(
                    c::Pair(
                        c::Key::from_wire(p.sequence_from),
                        c::Key::from_wire(p.sequence_to),
                    ),
                    p.similarity,
                ))
                .is_err()
            {
                crate::traffic::dropped(crate::traffic::Queue::Communication);
            }
        },
    )?;
    {
        let config = config.clone();
        let owner = owner.clone();
        let node = node.clone();
        let history = history.clone();
        let status = status.clone();
        let tx = loop_tx;
        let pending = pending.clone();
        let vio_done = vio_done.clone();
        let encoder_done = encoder_done.clone();
        let events = event_tx;
        std::thread::Builder::new()
            .name("sequence-exchange".into())
            .spawn(move || {
                if let Err(e) = communications(
                    config,
                    owner,
                    node,
                    comm_rx,
                    tx,
                    history,
                    status,
                    pending,
                    vio_done,
                    encoder_done,
                ) {
                    let _ = events.send(Event::Error(e.to_string()));
                }
            })?;
    }
    let odom = node.create_publisher::<nav_msgs::msg::Odometry>(
        format!("/{}/vio/odometry", config.robot)
            .as_str()
            .reliable()
            .keep_last(64),
    )?;
    let tf = node.create_publisher::<tf2_msgs::msg::TFMessage>("/tf".reliable().keep_last(64))?;
    let keyframes = node.create_publisher::<m::Keyframe>(
        format!("/{}/slam/keyframes", config.robot)
            .as_str()
            .reliable()
            .keep_last(256),
    )?;
    let sequences = node.create_publisher::<m::SequenceAnnouncement>(
        "/visloc/sequences".reliable().keep_last(256),
    )?;
    let loops =
        node.create_publisher::<m::LoopConstraint>("/visloc/loops".reliable().keep_last(256))?;
    let statuses = node.create_publisher::<m::Status>(
        format!("/{}/slam/status", config.robot)
            .as_str()
            .reliable()
            .transient_local()
            .keep_last(1),
    )?;
    let mut log = std::fs::File::create(config.output.join("events.jsonl"))?;
    let mut keyframe_log = std::fs::File::create(config.output.join("keyframes.jsonl"))?;
    let mut loop_log = std::fs::File::create(config.output.join("loops.jsonl"))?;
    let robot = config.robot.clone();
    let output = config.output.clone();
    let pump=node.create_timer_repeating(Duration::from_millis(20),move || {
        let result=(||->AnyResult<()> {
            for event in event_rx.try_iter() {
                match event {
                    Event::Odometry {timestamp_ns,pose,frame_id,process_ms}=> {
                        let parent=format!("{robot}/odom");let child=format!("{robot}/base_link");
                        let mut msg=nav_msgs::msg::Odometry::default();msg.header.frame_id=parent.clone();msg.header.stamp=wire::stamp(timestamp_ns);msg.child_frame_id=child.clone();msg.pose.pose=wire::pose(&pose);
                        odom.publish(msg)?;tf.publish(tf2_msgs::msg::TFMessage {transforms:vec![wire::tf(&pose,&parent,&child,timestamp_ns)]})?;
                        writeln!(log,"{}",serde_json::json!({"event":"vio","frame_id":frame_id,"timestamp_ns":timestamp_ns,"process_ms":process_ms}))?;
                    },
                    Event::Keyframe(record)=> {writeln!(keyframe_log,"{}",serde_json::to_string(&record)?)?;history.lock().unwrap().keyframes.push(record.clone());keyframes.publish(record.wire())?;},
                    Event::Sequence(sequence,encoding_ms)=> {status.lock().unwrap().sequences+=1;sequences.publish(sequence.wire())?;writeln!(log,"{}",serde_json::json!({"event":"sequence_encoding","sequence":sequence.key,"encoding_ms":encoding_ms}))?;if comm_tx.try_send(Comm::Sequence(sequence)).is_err() {crate::traffic::dropped(crate::traffic::Queue::Communication);}},
                    Event::Loop(edge)=> {writeln!(loop_log,"{}",serde_json::to_string(&edge)?)?;history.lock().unwrap().loops.push(edge.clone());status.lock().unwrap().loops+=1;loops.publish(edge.wire())?;writeln!(log,"{}",serde_json::json!({"event":"accepted_loop","constraint":edge}))?;},
                    Event::Verified(pair,v,matching_geometry_ms,exchange_through_verification_ms)=> {pending.fetch_sub(1,Ordering::SeqCst);writeln!(log,"{}",serde_json::json!({"event":"verification","pair":pair,"diagnostic":v,"matching_geometry_ms":matching_geometry_ms,"exchange_through_verification_ms":exchange_through_verification_ms}))?;},
                    Event::VioFinished=>vio_done.store(true,Ordering::SeqCst),
                    Event::Error(error)=>{eprintln!("{robot}: {error}");let mut s=status.lock().unwrap();s.error=error;s.ready=false;},
                }
            }
            let s=status.lock().unwrap().clone();statuses.publish(s.clone())?;
            if s.finished || !s.error.is_empty() {
                crate::archive::store_json(&output.join("communication.json"),&crate::traffic::snapshot())?;
                crate::archive::store_json(&output.join("status.json"),&serde_json::json!({"robot":s.robot,"session":s.session,"finished":s.finished,"frames":s.frames,"keyframes":s.keyframes,"sequences":s.sequences,"loops":s.loops,"dropped_images":s.dropped_images,"dropped_keyframes":s.dropped_keyframes,"request_failures":s.request_failures,"error":s.error}))?;}
            Ok(())
        })();if let Err(e)=result {eprintln!("ROS output error: {e}");}
    })?;
    let _keep = (
        image_sub,
        imu_sub,
        history_service,
        sequence_service,
        feature_service,
        finish_service,
        announcement_sub,
        proposal_sub,
        pump,
    );
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
