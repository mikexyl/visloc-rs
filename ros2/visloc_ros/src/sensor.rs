use crate::{
    robot::{Event, LoopCommand, RobotConfig},
    AnyResult,
};
use nalgebra::Vector3;
use opencv::{
    calib3d,
    core::{self, Mat, Size},
    imgproc,
    prelude::*,
};
use sensor_msgs::msg::{Image, Imu};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::{BufWriter, Write},
    sync::{
        mpsc::{Receiver, SyncSender, TrySendError},
        Arc, Mutex,
    },
};
use visloc_basalt::{
    config::BasaltConfig, BasaltCalibration, BasaltVioEstimatorAdapter, EurocSensorFrame,
    ImuSample, RawU16Image,
};
use visloc_msgs::msg::Status;
use visloc_multi_robot::{Key, KeyframeRecord, Transform};
use visloc_online_loop::{Frame, Observation};

pub enum Input {
    Image(Image),
    RightImage(Image),
    Imu(Imu),
    Finish(i64),
}
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CameraMode {
    #[default]
    Mono,
    Stereo,
}

/// Exact-stamp pairing, independent of callback order between camera topics.
/// Each topic must be ordered; a later stamp proves a missing counterpart.
struct CameraQueue {
    stereo: bool,
    pending: BTreeMap<i64, [Option<Image>; 2]>,
    latest: [Option<i64>; 2],
    retired: Option<i64>,
    dropped: u64,
}
impl CameraQueue {
    fn new(stereo: bool) -> Self {
        Self {
            stereo,
            pending: BTreeMap::new(),
            latest: [None, None],
            retired: None,
            dropped: 0,
        }
    }
    fn push(&mut self, camera: usize, image: Image) -> AnyResult<()> {
        if camera > 1 || (!self.stereo && camera != 0) {
            return Err("unexpected right camera in monocular mode".into());
        }
        let t = timestamp(&image.header.stamp);
        if self.latest[camera].is_some_and(|previous| t <= previous) {
            return Err("non-monotonic camera timestamps".into());
        }
        self.latest[camera] = Some(t);
        // Count a discarded exposure once; its delayed second image must not
        // resurrect a pair already consumed or evicted for capacity.
        if self.retired.is_some_and(|retired| t <= retired) {
            return Ok(());
        }
        self.pending.entry(t).or_default()[camera] = Some(image);
        if self.pending.len() > 32 {
            self.retired = Some(self.pending.pop_first().unwrap().0);
            self.dropped += 1;
        }
        Ok(())
    }
    fn pop_ready(&mut self, imu_watermark: Option<i64>) -> Option<(Image, Option<Image>)> {
        loop {
            let (&t, pair) = self.pending.first_key_value()?;
            let required = if self.stereo { 2 } else { 1 };
            if (0..required)
                .any(|i| pair[i].is_none() && self.latest[i].is_some_and(|latest| latest > t))
            {
                self.retired = Some(self.pending.pop_first().unwrap().0);
                self.dropped += 1;
                continue;
            }
            if pair[0].is_none()
                || (self.stereo && pair[1].is_none())
                || !imu_watermark.is_some_and(|imu| imu >= t)
            {
                return None;
            }
            let (_, [left, right]) = self.pending.pop_first().unwrap();
            self.retired = Some(t);
            return Some((left.unwrap(), right));
        }
    }
}

#[cfg(test)]
mod camera_queue_tests {
    use super::*;
    fn image(t: i64) -> Image {
        let mut image = Image::default();
        image.header.stamp.sec = (t / 1_000_000_000) as i32;
        image.header.stamp.nanosec = (t % 1_000_000_000) as u32;
        image
    }
    #[test]
    fn stereo_waits_for_both_cameras_and_imu_in_either_callback_order() {
        for first in [0, 1] {
            let mut q = CameraQueue::new(true);
            q.push(first, image(100)).unwrap();
            assert!(q.pop_ready(Some(100)).is_none());
            q.push(1 - first, image(100)).unwrap();
            assert!(q.pop_ready(Some(99)).is_none());
            let (left, right) = q.pop_ready(Some(100)).unwrap();
            assert_eq!(timestamp(&left.header.stamp), 100);
            assert_eq!(timestamp(&right.unwrap().header.stamp), 100);
            assert_eq!(q.dropped, 0);
        }
    }
    #[test]
    fn missing_counterpart_is_counted_and_never_paired_with_next_exposure() {
        let mut q = CameraQueue::new(true);
        q.push(0, image(100)).unwrap();
        q.push(1, image(101)).unwrap();
        assert!(q.pop_ready(Some(200)).is_none());
        q.push(0, image(101)).unwrap();
        assert_eq!(
            timestamp(&q.pop_ready(Some(200)).unwrap().0.header.stamp),
            101
        );
        assert_eq!(q.dropped, 1);
    }
    #[test]
    fn capacity_eviction_does_not_resurrect_delayed_half_pairs() {
        let mut q = CameraQueue::new(true);
        for t in 0..48 {
            q.push(0, image(t)).unwrap();
        }
        for t in 0..48 {
            q.push(1, image(t)).unwrap();
        }
        assert_eq!(q.pending.len(), 32);
        assert_eq!(q.dropped, 16);
        for t in 16..48 {
            let (left, right) = q.pop_ready(Some(100)).unwrap();
            assert_eq!(timestamp(&left.header.stamp), t);
            assert_eq!(timestamp(&right.unwrap().header.stamp), t);
        }
        assert!(q.pop_ready(Some(100)).is_none());
    }
    #[test]
    fn mono_still_needs_no_right_camera_and_rejects_duplicate_stamps() {
        let mut q = CameraQueue::new(false);
        q.push(0, image(100)).unwrap();
        assert!(q.push(0, image(100)).is_err());
        assert!(q.push(1, image(100)).is_err());
        assert!(q.pop_ready(Some(100)).unwrap().1.is_none());
    }
}
pub fn timestamp(t: &builtin_interfaces::msg::Time) -> i64 {
    t.sec as i64 * 1_000_000_000 + t.nanosec as i64
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct PreprocessConfig {
    pub raw_width: i32,
    pub raw_height: i32,
    /// Intrinsics after area resizing, including the half-pixel center shift.
    pub intrinsics: [f64; 4],
    pub distortion: Vec<f64>,
}
struct Preprocessor {
    width: usize,
    height: usize,
    maps: Option<(Mat, Mat)>,
    raw_size: Option<(i32, i32)>,
}
impl Preprocessor {
    fn new(width: usize, height: usize, c: Option<&PreprocessConfig>) -> AnyResult<Self> {
        core::set_num_threads(2)?;
        let maps = if let Some(c) = c {
            let [fx, fy, cx, cy] = c.intrinsics;
            let k = Mat::from_slice_2d(&[[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])?;
            let dist = Mat::from_slice(&c.distortion)?;
            let mut x = Mat::default();
            let mut y = Mat::default();
            calib3d::init_undistort_rectify_map(
                &k,
                &dist,
                &Mat::eye(3, 3, core::CV_64F)?.to_mat()?,
                &k,
                Size::new(width as i32, height as i32),
                core::CV_32FC1,
                &mut x,
                &mut y,
            )?;
            Some((x, y))
        } else {
            None
        };
        Ok(Self {
            width,
            height,
            maps,
            raw_size: c.map(|c| (c.raw_width, c.raw_height)),
        })
    }
    fn image(&self, msg: &Image) -> AnyResult<Vec<u8>> {
        let channels = match msg.encoding.as_str() {
            "mono8" | "8UC1" => 1,
            "rgb8" | "bgr8" => 3,
            _ => return Err(format!("unsupported image encoding {}", msg.encoding).into()),
        };
        let row = msg.width as usize * channels;
        if msg.width == 0
            || msg.height == 0
            || (msg.step as usize) < row
            || msg.data.len() != msg.step as usize * msg.height as usize
        {
            return Err("invalid ROS image dimensions/stride".into());
        }
        if self
            .raw_size
            .is_some_and(|s| s != (msg.width as i32, msg.height as i32))
        {
            return Err("raw image/calibration resolution mismatch".into());
        }
        let packed: Vec<u8> = msg
            .data
            .chunks(msg.step as usize)
            .flat_map(|r| r[..row].iter().copied())
            .collect();
        let source = Mat::from_slice(&packed)?;
        let source = source.reshape(channels as i32, msg.height as i32)?;
        let mut gray = Mat::default();
        if channels == 3 {
            imgproc::cvt_color(
                &source,
                &mut gray,
                if msg.encoding == "rgb8" {
                    imgproc::COLOR_RGB2GRAY
                } else {
                    imgproc::COLOR_BGR2GRAY
                },
                0,
            )?;
        } else {
            source.copy_to(&mut gray)?;
        }
        if let Some((x, y)) = &self.maps {
            let mut resized = Mat::default();
            let mut rectified = Mat::default();
            imgproc::resize(
                &gray,
                &mut resized,
                Size::new(self.width as i32, self.height as i32),
                0.,
                0.,
                imgproc::INTER_AREA,
            )?;
            imgproc::remap(
                &resized,
                &mut rectified,
                x,
                y,
                imgproc::INTER_LINEAR,
                core::BORDER_CONSTANT,
                core::Scalar::default(),
            )?;
            Ok(rectified.data_bytes()?.to_vec())
        } else {
            if msg.width as usize != self.width || msg.height as usize != self.height {
                return Err("rectified image/calibration mismatch".into());
            }
            Ok(gray.data_bytes()?.to_vec())
        }
    }
}

pub fn run(
    config: RobotConfig,
    owner: Key,
    rx: Receiver<Input>,
    loop_tx: SyncSender<LoopCommand>,
    events: SyncSender<Event>,
    status: Arc<Mutex<Status>>,
) -> AnyResult<()> {
    let calibration = BasaltCalibration::from_path(&config.calibration)?;
    let stereo = config.camera_mode == CameraMode::Stereo;
    if stereo && calibration.cameras.len() < 2 {
        return Err("stereo mode requires two calibrated cameras".into());
    }
    if !stereo && config.preprocess_right.is_some() {
        return Err("right preprocessing requires stereo mode".into());
    }
    if stereo && config.preprocess.is_some() != config.preprocess_right.is_some() {
        return Err("stereo raw images require preprocessing for both cameras".into());
    }
    let vio_config = BasaltConfig::from_json(&std::fs::read_to_string(&config.vio_config)?)?;
    let mut adapter = BasaltVioEstimatorAdapter::from_config(&calibration, &vio_config)?;
    let mut startup = config
        .imu_startup
        .clone()
        .map(|c| visloc_basalt::startup::StationaryStartup::new(&calibration, &vio_config, c))
        .transpose()?;
    let mut startup_seeded = false;
    let mut startup_report_saved = false;
    let mut processed_frames = 0;
    status.lock().unwrap().initializing = startup.is_some();
    let (width, height) = calibration.resolutions[0];
    let cam = &calibration.cameras[0];
    let display_camera = visloc_multi_robot::CameraModel {
        width,
        height,
        intrinsics: [cam.fx, cam.fy, cam.cx, cam.cy],
        camera_to_body: Transform::from(calibration.camera_to_imu(0).unwrap()),
    };
    let display = config
        .visualization_enabled
        .then(|| crate::visualization::Writer::new(&config.output))
        .transpose()?;
    let bundle_cameras = if config.bundle_adjustment_enabled {
        (0..if stereo { 2 } else { 1 })
            .map(|i| {
                let cam = &calibration.cameras[i];
                if cam.xi != 0. || cam.alpha != 0. {
                    return Err("global BA requires rectified pinhole observations".into());
                }
                let (width, height) = calibration.resolutions[i];
                Ok(visloc_multi_robot::CameraModel {
                    width,
                    height,
                    intrinsics: [cam.fx, cam.fy, cam.cx, cam.cy],
                    camera_to_body: Transform::from(calibration.camera_to_imu(i as _).unwrap()),
                })
            })
            .collect::<AnyResult<Vec<_>>>()?
    } else {
        vec![]
    };
    let camera_display = if config.visualization_enabled {
        let cameras = (0..if stereo { 2 } else { 1 })
            .map(|i| {
                let cam = &calibration.cameras[i];
                let (width, height) = calibration.resolutions[i];
                visloc_multi_robot::CameraModel {
                    width,
                    height,
                    intrinsics: [cam.fx, cam.fy, cam.cx, cam.cy],
                    camera_to_body: Transform::from(calibration.camera_to_imu(i as _).unwrap()),
                }
            })
            .collect();
        Some(crate::visualization::CameraWriter::new(
            &config.output,
            cameras,
        )?)
    } else {
        None
    };
    let preprocessor =
        Preprocessor::new(width as usize, height as usize, config.preprocess.as_ref())?;
    let right_preprocessor = if stereo {
        let (w, h) = calibration.resolutions[1];
        Some(Preprocessor::new(
            w as usize,
            h as usize,
            config.preprocess_right.as_ref(),
        )?)
    } else {
        None
    };
    let mut imu = VecDeque::new();
    let mut images = CameraQueue::new(stereo);
    let mut last_imu = None;
    let mut first = true;
    let mut frame_id = 0;
    let mut kf_id = 0;
    let mut previous = None;
    let mut finish = None;
    let mut csv = BufWriter::new(File::create(config.output.join("trajectory.csv"))?);
    let mut tum = BufWriter::new(File::create(config.output.join("trajectory.tum"))?);
    let mut tracking = BufWriter::new(File::create(config.output.join("tracking.jsonl"))?);
    writeln!(csv, "frame_id,timestamp_ns,tx,ty,tz,qw,qx,qy,qz")?;
    loop {
        let event = rx.recv()?;
        match event {
            Input::Imu(m) => {
                let t = timestamp(&m.header.stamp);
                if last_imu.is_some_and(|v| t <= v) {
                    return Err("non-monotonic IMU timestamps".into());
                }
                let g = &m.angular_velocity;
                let a = &m.linear_acceleration;
                if ![g.x, g.y, g.z, a.x, a.y, a.z].iter().all(|x| x.is_finite()) {
                    return Err("nonfinite IMU".into());
                }
                imu.push_back(ImuSample::new(
                    t,
                    Vector3::new(g.x, g.y, g.z),
                    Vector3::new(a.x, a.y, a.z),
                ));
                last_imu = Some(t);
                if imu.len() > 100_000 {
                    return Err("IMU buffer exceeded 100000 samples without camera progress".into());
                }
            }
            Input::Image(m) => {
                images.push(0, m)?;
            }
            Input::RightImage(m) => images.push(1, m)?,
            Input::Finish(t) => finish = Some(t),
        }
        while let Some((message, right_message)) = images.pop_ready(last_imu) {
            let t = timestamp(&message.header.stamp);
            let initialization_imu = if first {
                imu.iter().find(|s| s.timestamp_ns >= t).cloned()
            } else {
                None
            };
            let n = imu.iter().take_while(|s| s.timestamp_ns <= t).count();
            let samples: Vec<_> = imu.drain(..n).collect();
            // The first image may precede the first IMU sample by a few ms.
            // Its explicit initialization sample sets gravity; there is no
            // preceding camera interval to preintegrate. Later gaps are errors.
            if samples.is_empty() && !first {
                return Err("camera interval has no IMU samples".into());
            }
            let gray = preprocessor.image(&message)?;
            let frame = EurocSensorFrame {
                frame_id,
                timestamp_ns: t,
                cam0: RawU16Image::new(
                    width as usize,
                    height as usize,
                    gray.iter().map(|&v| (v as u16) << 8).collect(),
                )?,
                cam1: match (right_message, &right_preprocessor) {
                    (Some(message), Some(p)) => Some(RawU16Image::new(
                        p.width,
                        p.height,
                        p.image(&message)?
                            .iter()
                            .map(|&v| (v as u16) << 8)
                            .collect(),
                    )?),
                    (None, None) => None,
                    _ => return Err("incomplete stereo frame".into()),
                },
                cam0_path: Default::default(),
                cam1_path: None,
                imu: samples,
                initialization_imu,
            };
            first = false;
            frame_id += 1;
            let ready = if let Some(gate) = &mut startup {
                gate.push(frame)?
            } else {
                vec![frame]
            };
            if !startup_seeded {
                if let Some(report) = startup.as_ref().and_then(|s| s.report.as_ref()) {
                    adapter.estimator.apply_stationary_startup(report)?;
                    startup_seeded = true;
                }
            }
            {
                let mut s = status.lock().unwrap();
                s.input_timestamp_ns = t;
                s.frames_ingested = frame_id;
                s.initializing = startup.as_ref().is_some_and(|s| !s.is_initialized());
                s.initialization_skipped_frames = startup
                    .as_ref()
                    .and_then(|s| s.report.as_ref())
                    .map_or(0, |r| r.discarded_stationary_frames as u64);
            }
            if !startup_report_saved && !ready.is_empty() {
                if let Some(report) = startup.as_ref().and_then(|s| s.report.as_ref()) {
                    std::fs::write(
                        config.output.join("imu_startup.json"),
                        serde_json::to_vec_pretty(report)?,
                    )?;
                    startup_report_saved = true;
                }
            }
            for frame in ready {
                let frame_id = frame.frame_id;
                let t = frame.timestamp_ns;
                let gray: Vec<u8> = frame.cam0.pixels().iter().map(|v| (v >> 8) as u8).collect();
                let camera_images = camera_display.as_ref().map(|_| {
                    let mut images = vec![gray.clone()];
                    if let Some(right) = &frame.cam1 {
                        images.push(right.pixels().iter().map(|v| (v >> 8) as u8).collect());
                    }
                    images
                });
                let start = std::time::Instant::now();
                let result = adapter.process_without_marg_data_no_trace(frame)?;
                let left_tracks = result
                    .tracks
                    .observations
                    .iter()
                    .filter(|o| o.camera_id == 0)
                    .count();
                let right_tracks = result
                    .tracks
                    .observations
                    .iter()
                    .filter(|o| o.camera_id == 1)
                    .count();
                writeln!(
                    tracking,
                    "{}",
                    serde_json::json!({"frame_id":frame_id,"timestamp_ns":t,
                    "stereo_input":stereo,"left_tracks":left_tracks,"right_tracks":right_tracks})
                )?;
                let pose = result.estimator.state.imu_to_world.clone();
                let transform = Transform::from(&pose);
                transform.se3()?;
                if let (Some(display), Some(images)) = (&camera_display, camera_images) {
                    display.submit(&owner, frame_id, t, &transform, images);
                }
                let [x, y, z] = transform.translation;
                let [qx, qy, qz, qw] = transform.rotation_xyzw;
                writeln!(csv, "{frame_id},{t},{x},{y},{z},{qw},{qx},{qy},{qz}")?;
                writeln!(
                    tum,
                    "{:.9} {x:.9} {y:.9} {z:.9} {qx:.12} {qy:.12} {qz:.12} {qw:.12}",
                    t as f64 * 1e-9
                )?;
                events.send(Event::Odometry {
                    timestamp_ns: t,
                    pose: transform.clone(),
                    frame_id,
                    process_ms: start.elapsed().as_secs_f64() * 1000.,
                })?;
                if result.estimator.is_keyframe {
                    let key = Key {
                        id: kf_id,
                        ..owner.clone()
                    };
                    let record = KeyframeRecord {
                        key: key.clone(),
                        timestamp_ns: t,
                        body_to_odom: transform,
                        previous: previous.clone(),
                    };
                    previous = Some(key);
                    kf_id += 1;
                    events.send(Event::Keyframe(record.clone()))?;
                    let map = adapter.estimator.map_points();
                    if config.bundle_adjustment_enabled {
                        if record.key.id as usize >= config.max_keyframes {
                            return Err("configured BA keyframe capacity reached".into());
                        }
                        let views = bundle_cameras
                            .iter()
                            .enumerate()
                            .map(|(i, camera)| {
                                let inverse = pose
                                    .compose(&camera.camera_to_body.se3().unwrap())
                                    .inverse();
                                visloc_multi_robot::CameraObservations {
                                    camera: camera.clone(),
                                    observations: result
                                        .tracks
                                        .observations
                                        .iter()
                                        .filter(|o| o.camera_id as usize == i)
                                        .map(|o| visloc_multi_robot::LandmarkObservation {
                                            track_id: o.track_id,
                                            pixel: o.pixel.into(),
                                            point_camera: map.get(&o.track_id).map(|p| {
                                                (inverse.rotation * p + inverse.translation).into()
                                            }),
                                        })
                                        .collect(),
                                }
                            })
                            .collect();
                        let frame = visloc_multi_robot::BundleFrame {
                            key: record.key.clone(),
                            timestamp_ns: t,
                            views,
                        };
                        frame.validate()?;
                        events.send(Event::BundleFrame(frame))?;
                    }
                    let frame = Frame {
                        id: frame_id,
                        keyframe_index: record.key.id,
                        timestamp_ns: t,
                        width: width as usize,
                        height: height as usize,
                        gray,
                        body_to_world: pose,
                        observations: result
                            .tracks
                            .observations
                            .iter()
                            .filter(|o| o.camera_id == 0)
                            .map(|o| Observation {
                                track_id: o.track_id,
                                pixel: o.pixel,
                                point_world: map.get(&o.track_id).copied(),
                            })
                            .collect(),
                    };
                    if let Some(display) = &display {
                        display.submit(&frame, &record, &display_camera)?;
                    }
                    match loop_tx.try_send(LoopCommand::Frame(frame, record)) {
                        Ok(()) => (),
                        Err(TrySendError::Full(_)) => status.lock().unwrap().dropped_keyframes += 1,
                        Err(TrySendError::Disconnected(_)) => {
                            return Err("loop worker disconnected".into())
                        }
                    }
                }
                processed_frames += 1;
                let mut s = status.lock().unwrap();
                s.frames = processed_frames;
                s.keyframes = kf_id;
                s.timestamp_ns = t;
            }
        }
        status.lock().unwrap().dropped_images = images.dropped;
        if finish.is_some_and(|t| images.latest[0].is_some_and(|i| i >= t))
            && images.pending.is_empty()
        {
            if let Some(gate) = &startup {
                gate.finish()?;
            }
        }
        if finish.is_some_and(|t| status.lock().unwrap().timestamp_ns >= t) {
            csv.flush()?;
            tum.flush()?;
            tracking.flush()?;
            loop_tx.send(LoopCommand::Finish)?;
            if let Some(display) = display {
                display.finish()?;
            }
            if let Some(display) = camera_display {
                display.finish()?;
            }
            events.send(Event::VioFinished)?;
            return Ok(());
        }
    }
}
