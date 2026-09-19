#![cfg(feature = "native")]
use visloc_tensorrt::{DataType, Input, Session};

/// Build tests/fixtures/build_identity.cpp first; see README for exact commands.
#[test]
#[ignore = "requires GPU and TRT_TEST_PLAN pointing to the identity fixture"]
fn dynamic_multi_input_inference_and_recovery() {
    let path = std::env::var("TRT_TEST_PLAN").expect("Set TRT_TEST_PLAN to identity.plan");
    let mut session = Session::from_file(path, 0).unwrap();
    assert_eq!(session.tensors().len(), 4);
    assert!(session.tensors().iter().all(|t| t.shape == [-1, 3]));
    for (batch, profile) in [(1, 0), (4, 0), (5, 1), (2, 0)] {
        let shape = [batch, 3];
        let x: Vec<f32> = (0..batch * 3).map(|i| i as f32 * 0.25 - 1.0).collect();
        let z: Vec<f32> = x.iter().map(|v| -v).collect();
        // Deliberately reverse engine input order.
        let outputs = session
            .run(
                &[Input::f32("z", &shape, &z), Input::f32("x", &shape, &x)],
                profile,
            )
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            assert_eq!(output.info.shape, shape);
            assert_eq!(
                output.to_f32().unwrap(),
                if output.info.name == "y" { &x } else { &z }.clone()
            );
        }
    }
    let data = [2.0; 3];
    assert!(session.run(&[], 0).is_err());
    assert!(session
        .run(&[Input::f32("unknown", &[1, 3], &data)], 0)
        .is_err());
    assert!(session
        .run(
            &[
                Input::f32("x", &[1, 3], &data),
                Input::f32("x", &[1, 3], &data)
            ],
            0
        )
        .is_err());
    let mut wrong_type = Input::f32("x", &[1, 3], &data);
    wrong_type.dtype = DataType::I32;
    assert!(session
        .run(&[wrong_type, Input::f32("z", &[1, 3], &data)], 0)
        .is_err());
    assert!(session
        .run(
            &[
                Input::f32("x", &[1, 3], &data),
                Input::f32("z", &[1, 3], &data)
            ],
            10
        )
        .is_err());
    assert!(session
        .run(
            &[
                Input::f32("x", &[6, 3], &[0.; 18]),
                Input::f32("z", &[6, 3], &[0.; 18])
            ],
            0
        )
        .is_err());
    let outputs = session
        .run(
            &[
                Input::f32("x", &[1, 3], &data),
                Input::f32("z", &[1, 3], &data),
            ],
            0,
        )
        .unwrap();
    assert!(outputs.iter().all(|o| o.to_f32().unwrap() == data));
    assert!(Session::from_bytes(b"not a TensorRT plan", 0).is_err());
}
