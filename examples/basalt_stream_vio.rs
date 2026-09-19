//! Sensor-stream adapter: length-prefixed JSON metadata followed by two Y8 images.
//! stdin is binary, stdout is one JSON pose acknowledgement per processed frame.
use nalgebra::Vector3;
use serde_json::{json, Value};
use std::{
    env, fs,
    io::{self, BufWriter, Read, Write},
    path::PathBuf,
    time::Instant,
};
use visloc_basalt::config::BasaltConfig;
use visloc_basalt::vio::scalar::ScalarMode;
use visloc_basalt::{
    BasaltCalibration, BasaltVioEstimatorAdapter, EurocSensorFrame, ImuSample, RawU16Image,
};

fn sample(v: &Value) -> Result<ImuSample, Box<dyn std::error::Error>> {
    let a = v.as_array().ok_or("IMU must be an array")?;
    if a.len() != 7 {
        return Err("IMU must contain timestamp, gyro xyz, accel xyz".into());
    }
    let t = a[0].as_i64().ok_or("invalid IMU timestamp")?;
    let mut x = [0.0; 6];
    for i in 0..6 {
        x[i] = a[i + 1]
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or("nonfinite IMU")?;
    }
    Ok(ImuSample::new(
        t,
        Vector3::new(x[0], x[1], x[2]),
        Vector3::new(x[3], x[4], x[5]),
    ))
}
fn main() {
    if let Err(e) = run() {
        eprintln!("basalt_stream_vio: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || (args.len() - 4) % 2 != 0 {
        return Err(
            "usage: basalt_stream_vio CALIBRATION CONFIG OUTPUT_DIR [--scalar-mode f32|f64] [--camera-mode stereo|mono]".into(),
        );
    }
    let mut scalar_mode = ScalarMode::UpstreamF32;
    let mut monocular = false;
    for option in args[4..].chunks_exact(2) {
        match (option[0].as_str(), option[1].as_str()) {
            ("--scalar-mode", "f32") => scalar_mode = ScalarMode::UpstreamF32,
            ("--scalar-mode", "f64") => scalar_mode = ScalarMode::ExtendedF64,
            ("--camera-mode", "stereo") => monocular = false,
            ("--camera-mode", "mono") => monocular = true,
            _ => return Err(format!("invalid option: {} {}", option[0], option[1]).into()),
        }
    }
    let calibration = BasaltCalibration::from_path(&args[1])?;
    if calibration.cameras.len() != 2 {
        return Err("stereo calibration requires two cameras".into());
    }
    let config = BasaltConfig::from_json(&fs::read_to_string(&args[2])?)?;
    let mut adapter = BasaltVioEstimatorAdapter::from_config(&calibration, &config)?;
    adapter.estimator.config.scalar_mode = scalar_mode;
    let out = PathBuf::from(&args[3]);
    fs::create_dir_all(&out)?;
    let mut tum = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("trajectory.tum"))?,
    );
    let mut csv = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("trajectory.csv"))?,
    );
    writeln!(csv,"frame_id,timestamp_ns,tx,ty,tz,qw,qx,qy,qz,cam0_observations,cam1_observations,imu_samples")?;
    let mut states = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("inertial_state.csv"))?,
    );
    writeln!(states, "timestamp_ns,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz")?;
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut previous = None;
    let mut last_imu = None;
    let mut frame_id = 0u64;
    loop {
        let mut length = [0u8; 4];
        if input.read(&mut length[..1])? == 0 {
            break;
        }
        input.read_exact(&mut length[1..])?;
        let n = u32::from_le_bytes(length) as usize;
        if n > 1_000_000 {
            return Err("oversized metadata".into());
        }
        let mut metadata = vec![0u8; n];
        input.read_exact(&mut metadata)?;
        let h: Value = serde_json::from_slice(&metadata)?;
        let t = h["timestamp_ns"].as_i64().ok_or("missing timestamp")?;
        if previous.is_some_and(|p| t <= p) {
            return Err("non-monotonic image timestamp".into());
        }
        let width = h["width"].as_u64().ok_or("missing width")? as usize;
        let height = h["height"].as_u64().ok_or("missing height")? as usize;
        if calibration
            .resolutions
            .iter()
            .any(|r| *r != (width as u32, height as u32))
        {
            return Err("image/calibration resolution mismatch".into());
        }
        let count = width
            .checked_mul(height)
            .filter(|c| *c <= 16_000_000)
            .ok_or("invalid image size")?;
        let mut left = vec![0u8; count];
        let mut right = vec![0u8; count];
        input.read_exact(&mut left)?;
        input.read_exact(&mut right)?;
        let imu: Vec<ImuSample> = h["imu"]
            .as_array()
            .ok_or("missing imu")?
            .iter()
            .map(sample)
            .collect::<Result<_, _>>()?;
        if imu.is_empty() {
            return Err("empty IMU interval".into());
        }
        for s in &imu {
            if s.timestamp_ns > t || last_imu.is_some_and(|p| s.timestamp_ns <= p) {
                return Err("invalid IMU interval ordering".into());
            }
            last_imu = Some(s.timestamp_ns);
        }
        let initialization_imu = if frame_id == 0 {
            Some(sample(&h["initialization_imu"])?)
        } else {
            None
        };
        if initialization_imu.is_some_and(|s| s.timestamp_ns < t) {
            return Err("initialization IMU must bracket first image".into());
        }
        let frame = EurocSensorFrame {
            frame_id,
            timestamp_ns: t,
            cam0: RawU16Image::new(
                width,
                height,
                left.into_iter().map(|x| (x as u16) << 8).collect(),
            )?,
            cam1: (!monocular)
                .then(|| {
                    RawU16Image::new(
                        width,
                        height,
                        right.into_iter().map(|x| (x as u16) << 8).collect(),
                    )
                })
                .transpose()?,
            cam0_path: PathBuf::new(),
            cam1_path: None,
            imu,
            initialization_imu,
        };
        let started = Instant::now();
        let result = adapter.process_without_marg_data_no_trace(frame)?;
        let pose = &result.estimator.state.imu_to_world;
        let q = pose.rotation.quaternion();
        let p = pose.translation;
        let process_ms = started.elapsed().as_secs_f64() * 1000.0;
        if p.iter().chain(q.coords.iter()).any(|x| !x.is_finite()) {
            return Err("nonfinite VIO pose".into());
        }
        let c0 = result
            .tracks
            .observations
            .iter()
            .filter(|o| o.camera_id == 0)
            .count();
        let c1 = result
            .tracks
            .observations
            .iter()
            .filter(|o| o.camera_id == 1)
            .count();
        writeln!(
            tum,
            "{:.9} {:.9} {:.9} {:.9} {:.12} {:.12} {:.12} {:.12}",
            t as f64 * 1e-9,
            p.x,
            p.y,
            p.z,
            q.i,
            q.j,
            q.k,
            q.w
        )?;
        writeln!(
            csv,
            "{frame_id},{t},{},{},{},{},{},{},{},{c0},{c1},{}",
            p.x, p.y, p.z, q.w, q.i, q.j, q.k, result.imu_count
        )?;
        tum.flush()?;
        csv.flush()?;
        let state = &result.estimator.state;
        let v = state.velocity_world_m_s;
        let bg = state.gyro_bias_rad_s;
        let ba = state.accel_bias_m_s2;
        writeln!(
            states,
            "{t},{},{},{},{},{},{},{},{},{}",
            v.x, v.y, v.z, bg.x, bg.y, bg.z, ba.x, ba.y, ba.z
        )?;
        states.flush()?;
        let current_map = adapter.estimator.map_points();
        let map_points: Vec<_> = current_map
            .iter()
            // Sample the whole active ID range if the visualization budget is
            // exceeded; taking only the first IDs biases output to old tracks.
            .step_by(current_map.len().div_ceil(3000).max(1))
            .filter(|(_, point)| point.coords.iter().all(|v| v.is_finite()))
            .take(3000)
            .map(|(id, point)| json!([id, point.x, point.y, point.z]))
            .collect();
        writeln!(
            output,
            "{}",
            json!({"frame_id":frame_id,"timestamp_ns":t,"position":[p.x,p.y,p.z],"quaternion_xyzw":[q.i,q.j,q.k,q.w],"observations":[c0,c1],"imu_samples":result.imu_count,"process_ms":process_ms,"map_points":map_points})
        )?;
        output.flush()?;
        previous = Some(t);
        frame_id += 1;
    }
    if frame_id == 0 {
        return Err("sensor stream ended without frames".into());
    }
    fs::write(
        out.join("summary.json"),
        json!({"frames_processed":frame_id,"sensor_only":true,
            "camera_mode": if monocular { "mono" } else { "stereo" },
            "scalar_mode": if scalar_mode == ScalarMode::UpstreamF32 { "f32" } else { "f64" }
        })
        .to_string(),
    )?;
    Ok(())
}
