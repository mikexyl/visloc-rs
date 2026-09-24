//! Durable robot records survive estimator restarts; only the active VIO state
//! resets. Archived features are indexed by the complete robot/session/local ID.
use crate::{robot::History, AnyResult};
use std::path::{Path, PathBuf};
use visloc_multi_robot as c;

#[derive(serde::Deserialize)]
struct SequenceArchive {
    sequence: c::Sequence,
    descriptors: c::FrameDescriptors,
    features: Vec<c::FeatureFrame>,
}
fn lines<T: serde::de::DeserializeOwned>(path: &Path) -> AnyResult<Vec<T>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let text = std::fs::read_to_string(path)?;
    let all: Vec<_> = text.lines().collect();
    let mut records = Vec::new();
    for (i, line) in all.iter().enumerate() {
        match serde_json::from_str(line) {
            Ok(record) => records.push(record),
            Err(_) if i + 1 == all.len() && !text.ends_with('\n') => eprintln!(
                "Ignoring incomplete final archive record in {}",
                path.display()
            ),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(records)
}
pub fn restore(root: &Path, robot: &str, session: &str, capacity: usize) -> AnyResult<History> {
    let mut history = History {
        session: session.into(),
        ..Default::default()
    };
    if !root.exists() {
        return Ok(history);
    }
    let mut directories: Vec<PathBuf> = vec![root.to_owned()];
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && entry.file_name().to_string_lossy().starts_with("session-")
        {
            directories.push(entry.path());
        }
    }
    directories.sort();
    for directory in directories {
        history
            .attempted_pairs
            .extend(lines::<c::Pair>(&directory.join("attempts.jsonl"))?);
        history.keyframes.extend(lines::<c::KeyframeRecord>(
            &directory.join("keyframes.jsonl"),
        )?);
        history
            .loops
            .extend(lines::<c::LoopConstraint>(&directory.join("loops.jsonl"))?);
        let sequences = directory.join("sequences");
        if !sequences.exists() {
            continue;
        }
        let mut files = std::fs::read_dir(sequences)?
            .map(|p| p.map(|p| p.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        files.sort();
        for file in files {
            if file.extension().is_none_or(|s| s != "json") {
                continue;
            }
            let archive: SequenceArchive = serde_json::from_slice(&std::fs::read(file)?)?;
            archive.sequence.validate()?;
            archive.descriptors.validate()?;
            if archive.sequence.key.robot != robot {
                return Err("robot archive identity mismatch".into());
            }
            history
                .descriptors
                .insert(archive.sequence.key.clone(), archive.descriptors);
            history.sequences.push(archive.sequence);
            for feature in archive.features {
                feature.validate()?;
                history.features.insert(feature.key.clone(), feature);
            }
        }
    }
    history
        .attempted_pairs
        .extend(history.loops.iter().map(|e| e.pair.clone()));
    if history.keyframes.len() >= capacity {
        return Err(
            "archived keyframe capacity reached; increase max_keyframes or use a new archive root"
                .into(),
        );
    }
    Ok(history)
}

pub fn store_json(path: &Path, value: &impl serde::Serialize) -> AnyResult<()> {
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_vec(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn restart_restores_both_sessions_without_local_id_collisions() {
        let root =
            std::env::temp_dir().join(format!("visloc-archive-test-{}", uuid::Uuid::new_v4()));
        for (session, directory) in [
            ("first", root.clone()),
            ("second", root.join("session-second")),
        ] {
            std::fs::create_dir_all(directory.join("sequences")).unwrap();
            let key = |id| c::Key::new("robot", session, id);
            let mut log = std::fs::File::create(directory.join("keyframes.jsonl")).unwrap();
            for id in 0..10 {
                writeln!(
                    log,
                    "{}",
                    serde_json::to_string(&c::KeyframeRecord {
                        key: key(id),
                        timestamp_ns: id as i64,
                        body_to_odom: c::Transform::default(),
                        previous: (id > 0).then(|| key(id - 1))
                    })
                    .unwrap()
                )
                .unwrap();
            }
            // Simulate interruption while writing the next journal record.
            write!(log, "{{\"key\":").unwrap();
            let descriptor = {
                let mut d = vec![0.; 512];
                d[0] = 1.;
                d
            };
            let selected = vec![key(0), key(2), key(4), key(6), key(9)];
            let sequence = c::Sequence {
                key: key(0),
                members: (0..10).map(key).collect(),
                selected: selected.clone(),
                start_ns: 0,
                end_ns: 9,
                model_id: "test".into(),
                descriptor: descriptor.clone(),
                excluded_keyframes: vec![],
            };
            let descriptors = c::FrameDescriptors {
                sequence: key(0),
                frames: selected.clone(),
                descriptors: descriptor.repeat(5),
            };
            let features: Vec<_> = selected
                .into_iter()
                .map(|key| c::FeatureFrame {
                    key,
                    timestamp_ns: 0,
                    camera: c::CameraModel {
                        width: 640,
                        height: 480,
                        intrinsics: [400., 400., 320., 240.],
                        camera_to_body: c::Transform::default(),
                    },
                    pixels: vec![[100., 100.]],
                    descriptors: vec![1.; 64],
                    track_ids: vec![42],
                    points_camera: vec![Some([0., 0., 3.])],
                })
                .collect();
            std::fs::write(directory.join("sequences/000000.json"),serde_json::to_vec(&serde_json::json!({"sequence":sequence,"descriptors":descriptors,"features":features})).unwrap()).unwrap();
        }
        let pair = c::Pair::new(
            c::Key::new("robot", "first", 0),
            c::Key::new("robot", "second", 0),
        );
        std::fs::write(
            root.join("attempts.jsonl"),
            format!("{}\n", serde_json::to_string(&pair).unwrap()),
        )
        .unwrap();
        let restored = restore(&root, "robot", "third", 100).unwrap();
        assert!(restored.attempted_pairs.contains(&pair));
        assert_eq!(restored.session, "third");
        assert_eq!(restored.keyframes.len(), 20);
        assert_eq!(restored.sequences.len(), 2);
        assert_eq!(restored.descriptors.len(), 2);
        assert_eq!(restored.features.len(), 10);
        assert!(restored
            .features
            .contains_key(&c::Key::new("robot", "first", 0)));
        assert!(restored
            .features
            .contains_key(&c::Key::new("robot", "second", 0)));
        assert!(restore(&root, "robot", "third", 20).is_err());
        assert!(restore(&root, "wrong_robot", "third", 100).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
