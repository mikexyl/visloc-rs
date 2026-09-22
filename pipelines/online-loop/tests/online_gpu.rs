//! Positive end-to-end regression with repeated textured keyframes and known
//! synthetic drift. This tests the correction path, not dataset accuracy.
#![cfg(feature = "native")]
use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use std::{
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};
use visloc_core::{geometry::SE3, types::Camera};
use visloc_online_loop::{Config, Frame, Observation, OnlineLoop};

#[test]
#[ignore = "requires GPU and VISLOC_LOOP_CONFIG pointing to a validated engine bundle"]
fn repeated_keyframes_confirm_loop_and_correct_known_drift() {
    run_case(false);
}

#[test]
#[ignore = "requires GPU and VISLOC_LOOP_CONFIG pointing to a validated engine bundle"]
fn persistent_covisibility_blocks_loops_even_after_temporal_exclusion() {
    run_case(true);
}

fn run_case(persistent_landmarks: bool) {
    // Cargo runs integration tests in the package directory. Interpret the
    // documented relative bundle path from the workspace root.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(std::env::var("VISLOC_LOOP_CONFIG").expect("VISLOC_LOOP_CONFIG"));
    let config = Config::from_path(path).unwrap();
    assert_eq!(config.matcher_keypoints, 128);
    let output = std::env::temp_dir().join(format!(
        "visloc-loop-gpu-{}-{persistent_landmarks}",
        std::process::id()
    ));
    std::fs::create_dir(&output).unwrap();
    let camera = Camera::pinhole(0, 800, 550, 450.0, 450.0, 400.0, 275.0);
    let mut worker = OnlineLoop::start(config, camera, SE3::identity(), output.clone()).unwrap();
    // Reproducible random texture with multiscale structure, identical at each
    // visit. Landmarks have known depth; the submitted VIO poses drift in x.
    let gray: Vec<u8> = (0..550)
        .flat_map(|y| {
            (0..800).map(move |x| {
                let hash = |n: u32| {
                    let n = n.wrapping_mul(747796405).wrapping_add(2891336453);
                    let v = ((n >> ((n >> 28) + 4)) ^ n).wrapping_mul(277803737);
                    ((v >> 22) ^ v) as u8
                };
                ((hash((y / 8 * 100 + x / 8) as u32) as u16 + hash((y * 800 + x) as u32) as u16)
                    / 2) as u8
            })
        })
        .collect();
    for i in 0..40 {
        let raw = SE3::new(
            UnitQuaternion::identity(),
            Vector3::new(i as f64 * 0.1, 0.0, 0.0),
        );
        let observations = (0..128)
            .map(|j| {
                let pixel =
                    Point2::new(45.0 + (j % 16) as f64 * 47.0, 40.0 + (j / 16) as f64 * 65.0);
                let depth = 4.0 + (j * 17 % 23) as f64 * 0.13;
                let local = Point3::new(
                    (pixel.x - 400.0) * depth / 450.0,
                    (pixel.y - 275.0) * depth / 450.0,
                    depth,
                );
                Observation {
                    // Track IDs persist within a visit and are newly assigned
                    // after leaving and returning to the same textured place.
                    track_id: if persistent_landmarks {
                        j
                    } else {
                        (i / 20) * 128 + j
                    },
                    pixel,
                    point_world: Some(raw.transform_point(&local)),
                }
            })
            .collect();
        worker
            .submit(Frame {
                id: i * 17,
                keyframe_index: i,
                timestamp_ns: i as i64 * 1_000_000_000,
                width: 800,
                height: 550,
                gray: gray.clone(),
                body_to_world: raw,
                observations,
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while worker.snapshot().unwrap().keyframes != i as usize + 1 {
            assert!(
                Instant::now() < deadline,
                "worker stalled; artifacts: {}",
                output.display()
            );
            thread::sleep(Duration::from_millis(10));
        }
        if i < 28 {
            assert_eq!(
                worker
                    .snapshot()
                    .unwrap()
                    .correction_at(i as i64 * 1_000_000_000),
                SE3::identity()
            );
        }
    }
    let result = worker.finish().unwrap();
    assert_eq!(result.keyframes, 40);
    assert_eq!(result.dropped_keyframes, 0);
    assert!(result.local_sequences_skipped > 0);
    if persistent_landmarks {
        assert_eq!(result.candidates, 0);
        assert_eq!(result.accepted_loops, 0);
        assert_eq!(result.correction_at(39_000_000_000), SE3::identity());
        let events = std::fs::read_to_string(output.join("loop_events.jsonl")).unwrap();
        for line in events.lines() {
            let event: serde_json::Value = serde_json::from_str(line).unwrap();
            if event["event"] == "retrieval" {
                assert_eq!(
                    event["compared_sequences"], 0,
                    "covisible descriptors must not enter ranking"
                );
                assert_eq!(event["eligible_sequences"], 0);
            }
        }
        eprintln!(
            "persistent landmarks: {} local exclusions, zero matcher calls; artifacts {}",
            result.local_sequences_skipped,
            output.display()
        );
        return;
    }
    assert!(
        result.accepted_loops >= 1,
        "no loop; artifacts: {}",
        output.display()
    );
    let correction = result.correction_at(39_000_000_000);
    assert!(correction.translation.x < -1.0, "{correction:?}");
    assert!((correction.rotation.norm() - 1.0).abs() < 1e-12);
    assert!(result.keyframe_positions.last().unwrap().1[0].abs() < 2.0);
    let events = std::fs::read_to_string(output.join("loop_events.jsonl")).unwrap();
    for line in events.lines() {
        let event: serde_json::Value = serde_json::from_str(line).unwrap();
        if event["event"] == "candidate" {
            let ids = event["query_sequence"].as_array().unwrap();
            assert_eq!(ids.len(), 5);
            assert!(ids
                .windows(2)
                .all(|w| w[1].as_u64().unwrap() - w[0].as_u64().unwrap() == 17));
            assert!(event["similarity"].as_f64().unwrap() >= 0.8);
        }
    }
    eprintln!(
        "accepted {} loops; correction {:?}; artifacts {}",
        result.accepted_loops,
        correction.translation,
        output.display()
    );
}
