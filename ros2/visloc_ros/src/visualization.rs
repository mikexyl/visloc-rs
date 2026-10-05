//! A best-effort display journal, independent of loop inference and estimation.
//! A slow disk drops display packets, never sensor inputs. Image encoding runs
//! on its own bounded worker, independently of the keyframe/landmark journal.
use crate::AnyResult;
use std::{
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread::JoinHandle,
};
use visloc_multi_robot::{CameraModel, KeyframeRecord};
use visloc_online_loop::Frame;

#[derive(serde::Serialize)]
struct Feature {
    key: visloc_multi_robot::Key,
    timestamp_ns: i64,
    camera: CameraModel,
    pixels: Vec<[f64; 2]>,
    track_ids: Vec<u64>,
    points_camera: Vec<Option<[f64; 3]>>,
}
#[derive(serde::Serialize)]
struct Packet {
    feature: Feature,
    pose: KeyframeRecord,
}

pub struct Writer {
    tx: mpsc::SyncSender<Packet>,
    worker: JoinHandle<AnyResult<()>>,
    dropped: Arc<AtomicU64>,
}
impl Writer {
    pub fn new(output: &Path) -> AnyResult<Self> {
        let file = std::fs::File::create(output.join("visualization.jsonl"))?;
        let output = output.to_owned();
        let (tx, rx) = mpsc::sync_channel::<Packet>(4);
        let dropped = Arc::new(AtomicU64::new(0));
        let counter = dropped.clone();
        let worker = std::thread::Builder::new()
            .name("display-journal".into())
            .spawn(move || {
                let mut file = std::io::BufWriter::new(file);
                let mut written = 0u64;
                let save_status = |written| {
                    crate::archive::store_json(
                        &output.join("visualization_status.json"),
                        &serde_json::json!({"written_keyframes":written,
                    "dropped_keyframes":counter.load(Ordering::Relaxed),"queue_capacity":4}),
                    )
                };
                for packet in rx {
                    serde_json::to_writer(&mut file, &packet)?;
                    writeln!(file)?;
                    file.flush()?;
                    written += 1;
                    save_status(written)?;
                }
                save_status(written)
            })?;
        Ok(Self {
            tx,
            worker,
            dropped,
        })
    }

    pub fn submit(
        &self,
        frame: &Frame,
        pose: &KeyframeRecord,
        camera: &CameraModel,
    ) -> AnyResult<()> {
        let camera_from_world = frame
            .body_to_world
            .compose(&camera.camera_to_body.se3()?)
            .inverse();
        let mut feature = Feature {
            key: pose.key.clone(),
            timestamp_ns: frame.timestamp_ns,
            camera: camera.clone(),
            pixels: Vec::new(),
            track_ids: Vec::new(),
            points_camera: Vec::new(),
        };
        for o in &frame.observations {
            feature.pixels.push([o.pixel.x, o.pixel.y]);
            feature.track_ids.push(o.track_id);
            feature.points_camera.push(
                o.point_world
                    .map(|p| {
                        let p = camera_from_world.transform_point(&p);
                        [p.x, p.y, p.z]
                    })
                    .filter(|p| p.iter().all(|v| v.is_finite()) && p[2] > 0.),
            );
        }
        if self
            .tx
            .try_send(Packet {
                feature,
                pose: pose.clone(),
            })
            .is_err()
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Drain only at end of replay, never while accepting sensor frames.
    pub fn finish(self) -> AnyResult<()> {
        drop(self.tx);
        self.worker
            .join()
            .map_err(|_| "display journal worker panicked")?
    }
}

#[derive(serde::Serialize)]
struct CameraPacket {
    robot: String,
    session: String,
    frame_id: u64,
    timestamp_ns: i64,
    body_to_odom: visloc_multi_robot::Transform,
    images: Vec<String>,
    #[serde(skip)]
    gray: Vec<Vec<u8>>,
}

pub struct CameraWriter {
    tx: mpsc::SyncSender<CameraPacket>,
    worker: JoinHandle<AnyResult<()>>,
    dropped: Arc<AtomicU64>,
}
impl CameraWriter {
    pub fn new(output: &Path, cameras: Vec<CameraModel>) -> AnyResult<Self> {
        use opencv::{
            core::{Mat, Vector},
            imgcodecs,
            prelude::*,
        };
        for camera in &cameras {
            camera.camera()?;
        }
        let file = std::fs::File::create(output.join("camera_frames.jsonl"))?;
        std::fs::create_dir_all(output.join("camera_images"))?;
        crate::archive::store_json(&output.join("camera_calibration.json"), &cameras)?;
        let output = output.to_owned();
        let (tx, rx) = mpsc::sync_channel::<CameraPacket>(4);
        let dropped = Arc::new(AtomicU64::new(0));
        let counter = dropped.clone();
        let worker = std::thread::Builder::new()
            .name("display-cameras".into())
            .spawn(move || {
                let mut file = std::io::BufWriter::new(file);
                let mut written = 0u64;
                let save_status = |written| {
                    crate::archive::store_json(
                        &output.join("camera_status.json"),
                        &serde_json::json!({"written_frames":written,
                    "dropped_frames":counter.load(Ordering::Relaxed),"queue_capacity":4}),
                    )
                };
                for mut packet in rx {
                    if packet.gray.len() != cameras.len() {
                        return Err("display camera count mismatch".into());
                    }
                    for (i, (pixels, camera)) in packet.gray.iter().zip(&cameras).enumerate() {
                        if pixels.len() != camera.width as usize * camera.height as usize {
                            return Err("display image/calibration dimensions mismatch".into());
                        }
                        let mat = Mat::from_slice(pixels)?;
                        let mat = mat.reshape(1, camera.height as i32)?;
                        let mut jpeg = Vector::<u8>::new();
                        if !imgcodecs::imencode(
                            ".jpg",
                            &mat,
                            &mut jpeg,
                            &Vector::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, 90]),
                        )? {
                            return Err("display JPEG encoding failed".into());
                        }
                        let name = format!("camera_images/{:08}_cam{i}.jpg", packet.frame_id);
                        std::fs::write(output.join(&name), jpeg.as_slice())?;
                        packet.images.push(name);
                    }
                    // Publish only after both complete images are readable. The tailer
                    // ignores a partially written metadata line until its newline.
                    serde_json::to_writer(&mut file, &packet)?;
                    writeln!(file)?;
                    file.flush()?;
                    written += 1;
                    save_status(written)?;
                }
                save_status(written)
            })?;
        Ok(Self {
            tx,
            worker,
            dropped,
        })
    }

    pub fn submit(
        &self,
        owner: &visloc_multi_robot::Key,
        frame_id: u64,
        timestamp_ns: i64,
        pose: &visloc_multi_robot::Transform,
        gray: Vec<Vec<u8>>,
    ) {
        let packet = CameraPacket {
            robot: owner.robot.clone(),
            session: owner.session.clone(),
            frame_id,
            timestamp_ns,
            body_to_odom: pose.clone(),
            images: vec![],
            gray,
        };
        if self.tx.try_send(packet).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn finish(self) -> AnyResult<()> {
        drop(self.tx);
        self.worker
            .join()
            .map_err(|_| "display camera worker panicked")?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;
    use visloc_multi_robot::{Key, Transform};
    use visloc_online_loop::Observation;

    #[test]
    fn camera_journal_publishes_matching_stereo_images_and_body_pose() {
        use opencv::{imgcodecs, prelude::*};
        let root = std::env::temp_dir().join(format!("visloc-camera-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let camera = CameraModel {
            width: 64,
            height: 48,
            intrinsics: [40., 41., 32., 24.],
            camera_to_body: Transform::default(),
        };
        let mut right = camera.clone();
        right.camera_to_body.translation = [0.095, 0., 0.];
        let writer = CameraWriter::new(&root, vec![camera, right]).unwrap();
        let pose = Transform {
            translation: [1., 2., 3.],
            ..Default::default()
        };
        writer.submit(
            &Key::new("robot", "new-session", 0),
            17,
            1234567890,
            &pose,
            vec![vec![16; 64 * 48], vec![160; 64 * 48]],
        );
        writer.finish().unwrap();
        let packet: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("camera_frames.jsonl")).unwrap(),
        )
        .unwrap();
        assert_eq!(packet["frame_id"], 17);
        assert_eq!(packet["timestamp_ns"], 1234567890);
        assert_eq!(packet["session"], "new-session");
        assert_eq!(
            packet["body_to_odom"]["translation"],
            serde_json::json!([1., 2., 3.])
        );
        for (i, value) in [16, 160].into_iter().enumerate() {
            let path = root.join(packet["images"][i].as_str().unwrap());
            let image =
                imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_GRAYSCALE).unwrap();
            assert_eq!((image.cols(), image.rows()), (64, 48));
            assert!(image.data_bytes().unwrap().iter().all(|p| *p == value));
        }
        let status: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("camera_status.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(status["written_frames"], 1);
        assert_eq!(status["dropped_frames"], 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn display_export_uses_observing_camera_and_preserves_ids() {
        let root = std::env::temp_dir().join(format!("visloc-display-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let pose = SE3::new(
            UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
            Vector3::new(4., 2., 1.),
        );
        let extrinsic = SE3::new(
            UnitQuaternion::from_euler_angles(-0.1, 0.05, 0.2),
            Vector3::new(0.1, 0., 0.),
        );
        let point = Point3::new(0.3, 0.1, 5.);
        let camera = CameraModel {
            width: 640,
            height: 480,
            intrinsics: [400., 400., 320., 240.],
            camera_to_body: Transform::from(&extrinsic),
        };
        let record = KeyframeRecord {
            key: Key::new("robot", "session", 7),
            timestamp_ns: 123,
            body_to_odom: Transform::from(&pose),
            previous: None,
        };
        let frame = Frame {
            id: 30,
            keyframe_index: 7,
            timestamp_ns: 123,
            width: 640,
            height: 480,
            gray: vec![],
            body_to_world: pose.clone(),
            observations: vec![Observation {
                track_id: u64::MAX,
                pixel: Point2::new(344., 248.),
                point_world: Some(pose.compose(&extrinsic).transform_point(&point)),
            }],
        };
        let writer = Writer::new(&root).unwrap();
        writer.submit(&frame, &record, &camera).unwrap();
        writer.finish().unwrap();
        let packet: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("visualization.jsonl")).unwrap(),
        )
        .unwrap();
        assert_eq!(packet["feature"]["track_ids"][0].as_u64(), Some(u64::MAX));
        assert_eq!(packet["feature"]["key"]["id"], 7);
        for i in 0..3 {
            assert!(
                (packet["feature"]["points_camera"][0][i].as_f64().unwrap() - point[i]).abs()
                    < 1e-12
            );
        }
        // A deliberately unconsumed display queue cannot block the VIO producer.
        let (tx, _rx) = mpsc::sync_channel(0);
        let dropped = Arc::new(AtomicU64::new(0));
        let writer = Writer {
            tx,
            worker: std::thread::spawn(|| Ok(())),
            dropped: dropped.clone(),
        };
        writer.submit(&frame, &record, &camera).unwrap();
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        writer.finish().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
