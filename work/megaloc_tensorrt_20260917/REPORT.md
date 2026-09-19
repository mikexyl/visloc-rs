# MegaLoc Rust/TensorRT validation — 2026-09-17

**PASS.** The pretrained MegaLoc model ran through `visloc-tensorrt`'s actual
Rust → C ABI → C++ → TensorRT inference path and matched PyTorch on eight real
EuRoC MH_03_medium cam0 frames sampled evenly across the sequence.

| Measurement | Result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| TensorRT | 10.13.2 |
| Reference | PyTorch 2.3.1 CUDA, FP32, TF32 disabled |
| Engine | FP32, TF32 disabled, ONNX opset 17 |
| Input | `images`, FP32 `[1, 3, 322, 322]` |
| Output | `descriptor`, FP32 `[1, 8448]` |
| Minimum descriptor cosine vs PyTorch | 0.9999999999590219 |
| Maximum absolute descriptor error | 4.05591e-7 |
| Descriptor RMSE | 7.92923e-8 |
| Maximum output L2 norm error | 6.11321e-8 |
| Maximum pairwise cosine difference | 9.31913e-7 |
| Nearest-neighbor agreement, excluding self | 8/8 |
| Median of per-image median host-to-host times | 17.0048 ms |
| Engine size | Approximately 875 MiB |

Timing uses five warmup calls followed by ten calls per image. It includes
host-to-device transfer, inference, device-to-host transfer, synchronization,
and construction of owned Rust outputs. It excludes preprocessing, model/engine
loading, and the subsequent FP32 conversion used by the validator. This is a
local latency observation under normal desktop use, not a controlled benchmark
or a PyTorch speedup claim.

Preprocessing follows the model checkout's README: convert images to RGB,
ToTensor, ImageNet normalization, and antialiased bilinear resize to 322×322.
EuRoC frames are grayscale and converted to three RGB channels. The exact same
preprocessed tensors feed PyTorch and Rust/TensorRT. The official model source
and safetensors checkpoint were used without edits; source, weights, exported
ONNX, and image hashes are in [manifest.json](manifest.json).

Acceptance criteria were fixed before inference: cosine >= 0.9999, maximum
absolute error <= 1e-3, L2 norm error <= 1e-4, correct output shape, and all finite
values. All passed. Eight-frame parity is not a VPR recall benchmark, validation
of other image resolutions/batches, or FP16/INT8 validation.

The export reports trace warnings for shape-dependent Python branches and an
ONNX mutation-removal warning. The exported shape is deliberately fixed. The
measured parity above checks the resulting graph on all eight inputs.

## Artifacts and reproduction

- [summary.json](summary.json): aggregate numerical/retrieval checks.
- [parity.csv](parity.csv): per-image metrics and median timings.
- [rust_inference.log](rust_inference.log): actual runtime tensor metadata.
- [build_fp32.log](build_fp32.log): complete successful TensorRT build log.
- [export.log](export.log): exporter diagnostics and manifest.
- `megaloc_fp32.onnx` and `megaloc_fp32.plan`: generated model and usable engine.
- `input_*.bin`, `reference_*.bin`, `tensorrt_*.bin`: little-endian FP32 tensors.

Large model/tensor binaries remain available locally and are Git-ignored.
Reproduction instructions and reusable scripts are in
[the validation directory](../../crates/tensorrt-runtime/validation/README.md).
The Python environment used here was:
`/home/mikexyl/workspaces/dpvo_cbs_ws/src/DPVO/.runtime/trt-update-env/bin/python`.

Run the existing engine again from the repository root:

```sh
cargo run --release -p visloc-tensorrt --features native --example validate_megaloc -- \
  work/megaloc_tensorrt_20260917/megaloc_fp32.plan \
  work/megaloc_tensorrt_20260917 8
```

No changes to the native bridge were needed for MegaLoc.
