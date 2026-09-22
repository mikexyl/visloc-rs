//! Run a list of named FP32 input fixtures through one reusable GPU session.
use std::{fs, path::PathBuf, time::Instant};
use visloc_tensorrt::{Input, Session};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: infer_fixtures ENGINE FIXTURES_JSON OUTPUT_DIR".into());
    }
    let fixtures: serde_json::Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let output = PathBuf::from(&args[3]);
    fs::create_dir_all(&output)?;
    let mut session = Session::from_file(&args[1], 0)?;
    let mut reports = Vec::new();
    for (i, fixture) in fixtures
        .as_array()
        .ok_or("Expected fixture array")?
        .iter()
        .enumerate()
    {
        let mut names = Vec::new();
        let mut shapes = Vec::new();
        let mut values = Vec::new();
        for (name, input) in fixture.as_object().ok_or("Expected named inputs")? {
            names.push(name.clone());
            shapes.push(
                input["shape"]
                    .as_array()
                    .ok_or("shape")?
                    .iter()
                    .map(|x| x.as_i64().ok_or("dimension"))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let bytes = fs::read(input["path"].as_str().ok_or("path")?)?;
            if bytes.len() % 4 != 0 {
                return Err("Invalid f32 file size".into());
            }
            values.push(
                bytes
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            );
        }
        let inputs: Vec<_> = (0..names.len())
            .map(|j| Input::f32(&names[j], &shapes[j], &values[j]))
            .collect();
        for _ in 0..2 {
            session.run(&inputs, 0)?;
        }
        let started = Instant::now();
        let mut outputs = Vec::new();
        for _ in 0..5 {
            outputs = session.run(&inputs, 0)?;
        }
        let ms = started.elapsed().as_secs_f64() * 200.0;
        let mut metadata = serde_json::Map::new();
        for (j, out) in outputs.iter().enumerate() {
            let path = output.join(format!("{i}_{j}.bin"));
            fs::write(&path, &out.data)?;
            metadata.insert(
                out.info.name.clone(),
                serde_json::json!({"path":path,
                "shape":out.info.shape,"dtype":format!("{:?}",out.info.dtype)}),
            );
        }
        reports.push(serde_json::json!({"mean_ms":ms,"outputs":metadata}));
    }
    fs::write(
        output.join("outputs.json"),
        serde_json::to_vec_pretty(&reports)?,
    )?;
    Ok(())
}
