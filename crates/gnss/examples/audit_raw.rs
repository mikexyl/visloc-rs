use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
};
use visloc_gnss::{
    bootstrap::solve_epoch,
    navigation::Navigation,
    ubx::{Decoder, Message},
    Config,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: audit_raw recording.ubx output.jsonl".into());
    }
    let mut input = File::open(&args[1])?;
    let mut output = BufWriter::new(File::create(&args[2])?);
    let (mut decoder, mut navigation) = (Decoder::default(), Navigation::default());
    let mut week = 0;
    let mut seed = None;
    let mut hash = Sha256::new();
    let mut nsolutions = 0;
    loop {
        let mut data = [0u8; 4096];
        let n = input.read(&mut data)?;
        if n == 0 {
            break;
        }
        hash.update(&data[..n]);
        for msg in decoder.push(&data[..n], 0) {
            match msg {
                Message::Subframe(f) => {
                    if let Some(e) = navigation.push(&f, week) {
                        writeln!(output, "{}", serde_json::json!({"ephemeris":e}))?;
                    }
                }
                Message::Epoch(e) => {
                    week = e.week;
                    let solution = solve_epoch(&e, &navigation, &Config::default(), seed);
                    if let Some(s) = &solution {
                        seed = Some(s.ecef_m);
                        nsolutions += 1;
                    }
                    writeln!(
                        output,
                        "{}",
                        serde_json::json!({"epoch":e,"solution":solution})
                    )?;
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"sha256":format!("{:x}",hash.finalize()),"decoder":decoder.diagnostics,"satellites_with_ephemeris":navigation.ephemerides.len(),"rejected_pages":navigation.rejected_pages,"solutions":nsolutions})
    );
    Ok(())
}
