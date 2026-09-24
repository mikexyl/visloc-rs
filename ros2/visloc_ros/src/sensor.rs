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
    collections::VecDeque,
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
    Imu(Imu),
    Finish(i64),
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
    let preprocessor =
        Preprocessor::new(width as usize, height as usize, config.preprocess.as_ref())?;
    let mut imu = VecDeque::new();
    let mut images = VecDeque::new();
    let mut last_imu = None;
    let mut last_image = None;
    let mut first = true;
    let mut frame_id = 0;
    let mut kf_id = 0;
    let mut previous = None;
    let mut finish = None;
    let mut csv = BufWriter::new(File::create(config.output.join("trajectory.csv"))?);
    let mut tum = BufWriter::new(File::create(config.output.join("trajectory.tum"))?);
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
                let t = timestamp(&m.header.stamp);
                if last_image.is_some_and(|v| t <= v) {
                    return Err("non-monotonic camera timestamps".into());
                }
                last_image = Some(t);
                if images.len() >= 32 {
                    images.pop_front();
                    status.lock().unwrap().dropped_images += 1;
                }
                images.push_back(m);
            }
            Input::Finish(t) => finish = Some(t),
        }
        while images
            .front()
            .is_some_and(|m| last_imu.is_some_and(|t| t >= timestamp(&m.header.stamp)))
        {
            let message = images.pop_front().unwrap();
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
                cam1: None,
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
                let start = std::time::Instant::now();
                let result = adapter.process_without_marg_data_no_trace(frame)?;
                let pose = result.estimator.state.imu_to_world.clone();
                let transform = Transform::from(&pose);
                transform.se3()?;
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
        if finish.is_some_and(|t| last_image.is_some_and(|i| i >= t)) && images.is_empty() {
            if let Some(gate) = &startup {
                gate.finish()?;
            }
        }
        if finish.is_some_and(|t| status.lock().unwrap().timestamp_ns >= t) {
            csv.flush()?;
            tum.flush()?;
            loop_tx.send(LoopCommand::Finish)?;
            events.send(Event::VioFinished)?;
            return Ok(());
        }
    }
}
