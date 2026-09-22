# visloc-tensorrt

A small Rust inference API backed by a C++17/C ABI bridge to TensorRT 10/11.
Loads serialized `.plan`/`.engine` files, accepts named CPU tensors, and returns
owned CPU outputs. No Python, ONNX Runtime, Candle, bindgen, or libclang is needed.
The crate is a workspace member; it does not change the existing vision backend.

## Use

```toml
[dependencies]
visloc-tensorrt = { path = "path/to/visloc-rs/crates/tensorrt-runtime", features = ["native"] }
```

```rust,no_run
use visloc_tensorrt::{Input, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = Session::from_file("model.plan", 0)?;
    println!("{:?}", session.tensors());
    let pixels = vec![0.0f32; 3 * 224 * 224];
    let outputs = session.run(&[Input::f32("images", &[1, 3, 224, 224], &pixels)], 0)?;
    for output in outputs {
        println!("{} {:?}", output.info.name, output.info.shape);
        // For FP32 outputs: output.to_f32()?
    }
    Ok(())
}
```

Use the model's actual input names, types, shapes, and preprocessing. `Input`
also accepts raw native-endian bytes for FP16, BF16, FP8, INT8, UINT8, BOOL,
INT32, and INT64; the wrapper does not convert/quantize values. All required
inputs must be supplied exactly once; their order does not matter.

Build an engine from ONNX using the **installed SDK's** `trtexec`:

```sh
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.plan --skipInference
# For dynamic inputs add, for example:
# --minShapes=images:1x3x224x224 --optShapes=images:4x3x224x224 --maxShapes=images:8x3x224x224
cargo run -p visloc-tensorrt --features native --example infer -- model.plan images 1,3,224,224
```

The example supplies zero-valued FP32 data to a single-input model. Engine
building is intentionally external; this crate wraps inference, not the network
construction/ONNX parser API. TensorRT 11 uses strongly typed networks: set model
precision in the model/export workflow rather than assuming legacy `--fp16` or
implicit INT8 calibration flags work on every release.

## Build configuration

Linux x86_64 and aarch64 paths are supported. Install TensorRT development
headers/libraries, CUDA development headers/libraries, and a C++17 compiler.
Native linking uses `nvinfer`, `nvinfer_plugin`, and `cudart` (plus the C++ runtime).
No removed `nvparsers` dependency is used.

| Environment variable | Default |
| --- | --- |
| `TENSORRT_ROOT` | System installation |
| `TENSORRT_INCLUDE_DIR` | `$TENSORRT_ROOT/include` or `/usr/include/<arch>-linux-gnu` |
| `TENSORRT_LIB_DIR` | `$TENSORRT_ROOT/lib` or system linker search |
| `CUDA_HOME` | `/usr/local/cuda` |
| `CUDA_INCLUDE_DIR` | `$CUDA_HOME/include` |
| `CUDA_LIB_DIR` | `$CUDA_HOME/lib64` |

Explicit include/lib overrides take precedence over roots. For tarball SDKs,
also add the chosen library directories to `LD_LIBRARY_PATH` when running.
Build and runtime headers/libraries must refer to the same TensorRT installation.
Recompile this crate when changing SDK major versions; this is source
compatibility, not a promise of binary/engine-plan compatibility.

`native` is opt-in so ordinary workspace builds need no NVIDIA SDK. Without it,
only tensor metadata/conversion/validation types are available, not `Session`.

## Ownership and supported models

- `Session` owns runtime → engine → context, a private CUDA stream, and reusable
  device allocations. Dependencies are destroyed in the correct order.
- `run(&mut self, inputs, profile)` selects an optimization profile, sets all
  input shapes, infers output shapes, copies inputs, binds named addresses, calls
  `enqueueV3`, copies outputs, and synchronizes. Error paths also drain the stream.
- Returned output buffers own their data; they remain valid after another run or
  after the session is dropped. Sessions are neither `Send` nor `Sync`.
- Each native operation selects the session's CUDA device on the calling thread.
  The wrapper does not restore a previously selected device.
- Static and dynamic execution dimensions, multiple inputs/outputs, scalar and
  empty tensor storage, and profile selection are supported. Size calculations
  check negative/unresolved dimensions, rank, dtype, and overflow.
- Data-dependent outputs (such as NonZero match indices) use a reusable,
  aligned `IOutputAllocator`; actual dimensions are captured via `notifyShape`.
  Empty outputs are supported. `Output::to_i64()` reads INT64 match indices.
- Only contiguous LINEAR **device** I/O is supported. Host/shape-inference I/O,
  vectorized formats, packed
  INT4/FP4 I/O, and unknown future dtypes are rejected explicitly.
- NVIDIA's standard plugins are registered. Loading custom plugin shared
  libraries is not exposed. Engine deserialization reports missing plugins.
- This is synchronous CPU tensor I/O, not a zero-copy GPU pipeline. Input/output
  copies are expected; asynchronous Rust/device-buffer APIs are future work.
- Load only trusted engine plans. Build plans for a compatible TensorRT version
  and target GPU; arbitrary plans are not portable across SDKs/devices.

## Verification

From the repository root:

```sh
cargo test -p visloc-tensorrt
cargo test -p visloc-tensorrt --features native
cargo clippy -p visloc-tensorrt --features native --all-targets -- -D warnings

# Adjust include/library paths if using a non-system TensorRT installation.
c++ -std=c++17 crates/tensorrt-runtime/tests/fixtures/build_identity.cpp \
    -I/usr/local/cuda/include -lnvinfer -o /tmp/visloc-trt-build-identity
/tmp/visloc-trt-build-identity /tmp/visloc-trt-identity.plan
TRT_TEST_PLAN=/tmp/visloc-trt-identity.plan \
    cargo test -p visloc-tensorrt --features native --test gpu -- --ignored --nocapture
```

The GPU test is explicitly ignored by default and fails if invoked without a
fixture. It checks numerical results for two inputs/two outputs, batches 1/2/4/5,
profile switching, reversed input order, missing/duplicate/unknown inputs, dtype
mismatch, invalid profile, out-of-range shape, recovery, and invalid plans.
Expected TensorRT diagnostics appear for deliberately rejected inputs.

Validated on 2026-09-17:

- NVIDIA RTX 4070 Laptop GPU, driver 580.178.04, installed TensorRT **10.13.2**:
  native compilation/linking, 3 unit tests, and the GPU numerical test passed.
- SDK-free build: 2 unit tests passed. Rust formatting and Clippy passed.
- NVIDIA TensorRT **11.2.1** public headers at commit
  `505170043e0e2799bfcfe8fab777a78d2fb52854`: bridge and fixture compile-checked
  with C++17, `-Wall -Wextra -Werror`, SDK headers treated as system includes.
  **TensorRT 11 runtime execution has not been tested.**

See [research.md](research.md) for the open-source comparison and migration sources.

## MegaLoc model validation

The full pretrained MegaLoc model was also validated through this Rust API on
2026-09-17: eight EuRoC frames, FP32 TensorRT 10.13.2, descriptor cosine at least
0.999999999959 against PyTorch, maximum absolute error 4.06e-7, and nearest-neighbor
agreement 8/8. Median host-to-host latency was about 17.0 ms on the RTX 4070 Laptop
GPU. This is numerical parity on a small corpus, not a retrieval accuracy benchmark.
See [validation instructions](validation/README.md) and the
[run report](../../work/megaloc_tensorrt_20260917/REPORT.md).
