//! GPS ROS adapter; independent from sensor ingestion, Basalt and loop workers.
use crate::{wire::Wire, AnyResult};
use rclrs::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use visloc_msgs::{msg as m, srv as s};
use visloc_multi_robot::{GpsRecord, Key};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InputConfig {
    pub enabled: bool,
    /// Replay sends normalized, UTC-stamped records. Live input uses NavSatFix.
    pub normalized_input: bool,
    pub fix_topic: Option<String>,
}
impl Default for InputConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            normalized_input: false,
            fix_topic: None,
        }
    }
}
enum Input {
    Record(GpsRecord),
    Finish,
}
#[derive(Default)]
struct History {
    records: Vec<GpsRecord>,
    ids: BTreeMap<Key, usize>,
    finished: bool,
    rejected: u64,
    error: Option<String>,
}

// The robot stores each restarted estimator under session-<id>. History must
// cover every archived session, matching keyframe history semantics.
fn restore_history(root: &Path, robot: &str) -> AnyResult<History> {
    let mut paths = std::collections::BTreeSet::from([root.join("gps_records.jsonl")]);
    if root.exists() {
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && entry.file_name().to_string_lossy().starts_with("session-")
            {
                paths.insert(entry.path().join("gps_records.jsonl"));
            }
        }
    }
    let mut history = History::default();
    for path in paths.into_iter().filter(|p| p.exists()) {
        let bytes = std::fs::read(path)?;
        let end = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |p| p + 1);
        for line in std::str::from_utf8(&bytes[..end])?.lines() {
            let record: GpsRecord = serde_json::from_str(line)?;
            record.validate()?;
            if record.key.robot != robot {
                return Err("GPS archive belongs to another robot".into());
            }
            if let Some(&index) = history.ids.get(&record.key) {
                if history.records[index] != record {
                    return Err("conflicting archived GPS records".into());
                }
                continue;
            }
            history
                .ids
                .insert(record.key.clone(), history.records.len());
            history.records.push(record);
        }
    }
    Ok(history)
}

// All ingestion callbacks use the same nonblocking bounded queue.
fn enqueue(tx: &mpsc::SyncSender<Input>, drops: &AtomicU64, input: Input) {
    if tx.try_send(input).is_err() {
        drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restarted_robot_restores_all_gps_sessions_and_partial_journal_prefixes() {
        let root =
            std::env::temp_dir().join(format!("visloc-gps-history-{}", uuid::Uuid::new_v4()));
        let child = root.join("session-new");
        std::fs::create_dir_all(&child).unwrap();
        let record = |session: &str| GpsRecord {
            key: Key::new("robot", session, 0),
            timestamp_ns: 1,
            receipt_timestamp_ns: 2,
            time_source: "test".into(),
            lla: Some([0.; 3]),
            status: 0,
            quality: Some(1),
            hdop: Some(0.8),
            covariance_enu: None,
        };
        let old = record("old");
        let new = record("new");
        std::fs::write(
            root.join("gps_records.jsonl"),
            format!("{}\n{{partial", serde_json::to_string(&old).unwrap()),
        )
        .unwrap();
        std::fs::write(
            child.join("gps_records.jsonl"),
            format!(
                "{}\n{}\n",
                serde_json::to_string(&new).unwrap(),
                serde_json::to_string(&old).unwrap()
            ),
        )
        .unwrap();
        let h = restore_history(&root, "robot").unwrap();
        assert_eq!(h.records.len(), 2);
        assert!(h.ids.contains_key(&old.key));
        assert!(h.ids.contains_key(&new.key));
        assert!(restore_history(&root, "different").is_err());
        let mut conflict = old.clone();
        conflict.timestamp_ns += 1;
        std::fs::write(
            child.join("gps_records.jsonl"),
            format!("{}\n", serde_json::to_string(&conflict).unwrap()),
        )
        .unwrap();
        assert!(restore_history(&root, "robot").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn bounded_ingestion_reports_overflow_and_recovers_after_drain() {
        let (tx, rx) = mpsc::sync_channel(2);
        let drops = AtomicU64::new(0);
        for _ in 0..3 {
            enqueue(&tx, &drops, Input::Finish);
        }
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        rx.try_recv().unwrap();
        enqueue(&tx, &drops, Input::Finish);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(rx.try_iter().count(), 2);
    }
}

pub fn start(
    node: &Node,
    config: &InputConfig,
    owner: &Key,
    output: &Path,
    archive_root: &Path,
    reliable: bool,
) -> AnyResult<Option<Box<dyn std::any::Any>>> {
    if !config.enabled {
        return Ok(None);
    }
    let history = Arc::new(Mutex::new(restore_history(archive_root, &owner.robot)?));
    let drops = Arc::new(AtomicU64::new(0));
    let (tx, rx) = mpsc::sync_channel::<Input>(2048);
    let topic = config.fix_topic.clone().unwrap_or_else(|| {
        format!(
            "/{}/gps/{}",
            owner.robot,
            if config.normalized_input {
                "normalized"
            } else {
                "fix"
            }
        )
    });
    let publisher = node.create_publisher::<m::GpsFix>(
        format!("/{}/slam/gps", owner.robot)
            .as_str()
            .reliable()
            .keep_last(256),
    )?;
    let normalized = if config.normalized_input {
        let tx = tx.clone();
        let drops = drops.clone();
        let owner = owner.clone();
        Some(node.create_subscription(
            topic.as_str().reliable().keep_last(256),
            move |msg: m::GpsFix| {
                let mut record = GpsRecord::from_wire(msg);
                record.key.robot = owner.robot.clone();
                record.key.session = owner.session.clone();
                enqueue(&tx, &drops, Input::Record(record));
            },
        )?)
    } else {
        None
    };
    let raw = if !config.normalized_input {
        let tx = tx.clone();
        let drops = drops.clone();
        let owner = owner.clone();
        let options = if reliable {
            topic.as_str().reliable().keep_last(256)
        } else {
            topic.as_str().best_effort().keep_last(64)
        };
        Some(
            node.create_subscription(options, move |msg: sensor_msgs::msg::NavSatFix| {
                let t =
                    msg.header.stamp.sec as i64 * 1_000_000_000 + msg.header.stamp.nanosec as i64;
                let position = [msg.latitude, msg.longitude, msg.altitude];
                // NavSatFix zero covariance with UNKNOWN type is not a measurement.
                let covariance = (msg.position_covariance_type != 0
                    && msg.position_covariance.iter().any(|x| *x != 0.))
                .then_some(msg.position_covariance);
                let record = GpsRecord {
                    key: Key::new(&owner.robot, &owner.session, t.max(0) as u64),
                    timestamp_ns: t,
                    receipt_timestamp_ns: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as i64,
                    time_source: "navsatfix_header".into(),
                    lla: position.iter().all(|x| x.is_finite()).then_some(position),
                    status: msg.status.status,
                    quality: None,
                    hdop: None,
                    covariance_enu: covariance,
                };
                enqueue(&tx, &drops, Input::Record(record));
            })?,
        )
    } else {
        None
    };
    let h = history.clone();
    let session = owner.session.clone();
    let d = drops.clone();
    let service = node.create_service::<s::GetGpsHistory, _>(
        format!("/{}/slam/gps_history", owner.robot).as_str(),
        move |r: s::GetGpsHistory_Request| {
            let h = h.lock().unwrap();
            let begin = (r.cursor as usize).min(h.records.len());
            let end = (begin + 64).min(h.records.len());
            s::GetGpsHistory_Response {
                session: session.clone(),
                fixes: h.records[begin..end].iter().map(Wire::wire).collect(),
                cursor: end as u64,
                more: end < h.records.len(),
                finished: h.finished,
                dropped_inputs: d.load(Ordering::Relaxed),
                rejected_inputs: h.rejected,
            }
        },
    )?;
    let finish = node.create_service::<s::Finish, _>(
        format!("/{}/slam/gps_finish", owner.robot).as_str(),
        move |_: s::Finish_Request| s::Finish_Response {
            accepted: tx.try_send(Input::Finish).is_ok(),
        },
    )?;
    let file = output.join("gps_records.jsonl");
    let errors = output.join("gps_ingestion.jsonl");
    let h = history.clone();
    // Only the active append target is truncated; archived journals remain
    // read-only and their complete prefixes were restored above.
    if file.exists() {
        let bytes = std::fs::read(&file)?;
        let end = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |p| p + 1);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file)?
            .set_len(end as u64)?;
    }
    std::thread::Builder::new()
        .name("gps-ingestion".into())
        .spawn(move || {
            let result = (|| -> AnyResult<()> {
                let mut log = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(file)?;
                let mut rejected = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(errors)?;
                for input in rx {
                    match input {
                        Input::Finish => {
                            log.flush()?;
                            h.lock().unwrap().finished = true;
                        }
                        Input::Record(record) => {
                            let mut history = h.lock().unwrap();
                            let invalid =
                                record.validate().err().map(|e| e.to_string()).or_else(|| {
                                    history.ids.get(&record.key).and_then(|&i| {
                                        (history.records[i] != record)
                                            .then(|| "conflicting GPS retransmission".into())
                                    })
                                });
                            if let Some(reason) = invalid {
                                history.rejected += 1;
                                writeln!(
                                    rejected,
                                    "{}",
                                    serde_json::json!({"record":record,"reason":reason})
                                )?;
                                continue;
                            }
                            if history.ids.contains_key(&record.key) {
                                continue;
                            }
                            writeln!(log, "{}", serde_json::to_string(&record)?)?;
                            let index = history.records.len();
                            history.ids.insert(record.key.clone(), index);
                            history.records.push(record.clone());
                            history.finished = false;
                            drop(history);
                            publisher.publish(record.wire())?;
                        }
                    }
                }
                Ok(())
            })();
            if let Err(e) = result {
                eprintln!("GPS ingestion failed: {e}");
                h.lock().unwrap().error = Some(e.to_string());
            }
        })?;
    let path = output.join("gps_status.json");
    let timer=node.create_timer_repeating(Duration::from_millis(250),move|| {
        let h=history.lock().unwrap();
        let value=serde_json::json!({"records":h.records.len(),"finished":h.finished,"rejected_inputs":h.rejected,"dropped_inputs":drops.load(Ordering::Relaxed),"error":h.error});
        if let Err(e)=crate::archive::store_json(&path,&value){eprintln!("GPS status: {e}");}
    })?;
    Ok(Some(Box::new((normalized, raw, service, finish, timer))))
}
