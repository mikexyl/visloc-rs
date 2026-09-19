//! Compare actual Rust/C++/TensorRT outputs to export_megaloc.py reference files.
use std::{path::Path, time::Instant};
use visloc_tensorrt::{Input, Session};
fn read_f32(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("Invalid FP32 file size".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: validate_megaloc engine.plan corpus_dir count".into());
    }
    let dir = Path::new(&args[2]);
    let count: usize = args[3].parse()?;
    if count < 2 {
        return Err("Need at least two images".into());
    }
    let mut session = Session::from_file(&args[1], 0)?;
    eprintln!(
        "TensorRT header version: {}; tensors: {:?}",
        visloc_tensorrt::header_version(),
        session.tensors()
    );
    let first = read_f32(&dir.join("input_0.bin"))?;
    for _ in 0..5 {
        session.run(&[Input::f32("images", &[1, 3, 322, 322], &first)], 0)?;
    }
    println!("image,cosine,max_abs_error,rmse,output_norm,host_to_host_ms");
    let mut failed = false;
    for i in 0..count {
        let input = read_f32(&dir.join(format!("input_{i}.bin")))?;
        let reference = read_f32(&dir.join(format!("reference_{i}.bin")))?;
        let mut times = Vec::new();
        let mut output = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let outputs = session.run(&[Input::f32("images", &[1, 3, 322, 322], &input)], 0)?;
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            if outputs.len() != 1 || outputs[0].info.shape != [1, 8448] {
                return Err("Unexpected output shape".into());
            }
            output = outputs[0].to_f32()?;
        }
        if output.len() != reference.len()
            || output.len() != 8448
            || output.iter().chain(&reference).any(|v| !v.is_finite())
        {
            return Err("Invalid/nonfinite descriptor".into());
        }
        let (mut dot, mut nr, mut no, mut squared, mut max_abs) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in output.iter().zip(&reference) {
            let (a, b) = (f64::from(a), f64::from(b));
            dot += a * b;
            no += a * a;
            nr += b * b;
            squared += (a - b).powi(2);
            max_abs = max_abs.max((a - b).abs());
        }
        let cosine = dot / (no * nr).sqrt();
        let rmse = (squared / output.len() as f64).sqrt();
        times.sort_by(f64::total_cmp);
        println!(
            "{i},{cosine:.10},{max_abs:.10},{rmse:.10},{:.10},{:.4}",
            no.sqrt(),
            (times[4] + times[5]) / 2.0
        );
        // Fixed acceptance criteria, chosen before observing inference results.
        failed |= !cosine.is_finite()
            || cosine < 0.9999
            || max_abs > 1e-3
            || (no.sqrt() - 1.0).abs() > 1e-4;
        let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("tensorrt_{i}.bin")), bytes)?;
    }
    if failed {
        return Err("MegaLoc parity thresholds failed".into());
    }
    Ok(())
}
