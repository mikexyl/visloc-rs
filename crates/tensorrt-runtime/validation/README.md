# MegaLoc validation

Use an existing official MegaLoc checkout and pretrained `model.safetensors`.
The exporter loads weights strictly and does not modify the model or checkpoint.
Its Python environment needs torch, torchvision, safetensors, Pillow, numpy, and
onnx. The exact environment and hashes are captured in `manifest.json`.

From the visloc-rs repository root:

```sh
python crates/tensorrt-runtime/validation/export_megaloc.py \
  --repo /path/to/MegaLoc --weights /path/to/model.safetensors \
  --images /path/to/image/directory --out work/megaloc_validation --count 8
/usr/src/tensorrt/bin/trtexec \
  --onnx=work/megaloc_validation/megaloc_fp32.onnx \
  --saveEngine=work/megaloc_validation/megaloc_fp32.plan \
  --noTF32 --memPoolSize=workspace:2048 --skipInference
cargo run --release -p visloc-tensorrt --features native --example validate_megaloc -- \
  work/megaloc_validation/megaloc_fp32.plan work/megaloc_validation 8 \
  > work/megaloc_validation/parity.csv
python crates/tensorrt-runtime/validation/summarize_megaloc.py work/megaloc_validation
```

The input shape is fixed to `[1, 3, 322, 322]`. Preprocessing follows the model
README exactly: convert to RGB, ToTensor, ImageNet normalization, then bilinear
antialiased resize. The exporter records PyTorch GPU descriptors with TF32 off.
Export warnings about shape-dependent Python branches are expected for this
fixed shape; the graph is not advertised as accepting other sizes or batches.

The Rust example calls the actual C++ bridge, with 5 warmup calls followed by 10
runs per image. It writes outputs and per-image median host-to-host latency. It
fails on incorrect shapes, nonfinite values, descriptor cosine below 0.9999,
maximum absolute error above 1e-3, or output norm error above 1e-4. The summary
also compares pairwise similarities and nearest neighbors (excluding self).
This checks numerical parity, not dataset retrieval accuracy or general model
quality. Timing is a local observation, not a controlled performance benchmark.

The full model is exported and built in FP32 first; reduced precision needs its
own parity run and should not silently reuse these results. Large generated
ONNX/plan/tensor files should be ignored by Git.
