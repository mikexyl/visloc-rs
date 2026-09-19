# Rust / TensorRT options reviewed (2026-09-17)

This is a source/documentation review, not a comparative performance benchmark
or a complete safety audit of the projects below.

| Project | Documented approach | Fit for this task |
| --- | --- | --- |
| [mstallmo/tensorrt-rs](https://github.com/mstallmo/tensorrt-rs/blob/develop/tensorrt/README.md) | C++ wrapper, sys crate, Rust layer; advertised TensorRT 5/6/7 support | Useful historical architecture, but a significant API migration is required for current SDKs. |
| [LdDl/tensorrt-infer](https://github.com/LdDl/tensorrt-infer) | Serialized engines, explicit buffers/streams; sys crate advertises TensorRT 6–8 and 10+ | A direct alternative. Its example exposes raw buffer addresses and asks callers to manage resource drop order. |
| [trt-runner](https://docs.rs/trt-runner/latest/trt_runner/) | Small TensorRT 10 runtime wrapper, named tensors/enqueueV3, explicit buffers/streams and borrowed engine contexts | Closest alternative for lower-level integration. Its explicit unsafe buffer binding is appropriate for callers managing GPU lifetimes. |
| [yiso94/tensorrt-rs](https://github.com/yiso94/tensorrt-rs) | Small C ABI with Candle CUDA tensor I/O and TensorRT-LLM support | Useful if adopting Candle; introduces a tensor framework dependency for this use case. |
| [rustnn/trtx-rs](https://github.com/rustnn/trtx-rs) | TensorRT-RTX Rust bindings | Different NVIDIA SDK target; not a direct replacement for installed standard TensorRT. |

Decision: implement a small original runtime bridge in this workspace. This
keeps dependencies small, keeps all context/stream/buffer ownership together,
and exposes a synchronous safe host-I/O entry point. No upstream implementation
was copied or forked, and no upstream update was published. An existing low-level
crate may be a better fit when caller-owned CUDA streams and device tensors are
required. This implementation prioritizes a bounded API that can be numerically
verified against the installed SDK.

## Current API decisions

[NVIDIA's 8-to-10 C++ migration examples](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-8x-to-10x-c-api-patterns.html)
describe the switch from positional bindings/enqueueV2 to named tensors,
setTensorAddress, and enqueueV3, and the change to 64-bit dimensions. The bridge
uses these current interfaces and passes `int64_t` dimensions through its C ABI.

[NVIDIA's migration guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration-guide.html)
and [TensorRT OSS](https://github.com/NVIDIA/TensorRT) document TensorRT 11's move
to strongly typed networks, explicit quantization, and PluginV3. The runtime
bridge does not bind removed builder precision/calibration APIs or implement
PluginV2. The small engine fixture uses strong typing on 10 and the default
network mode on 11. ONNX compilation is delegated to the matching `trtexec`.

The implementation uses RAII `delete` via C++ smart pointers, owns the runtime
for the entire engine lifetime, and links only current inference/plugin/CUDA
libraries. It catches C++ exceptions at fallible C entry points. The Rust side
does not expose native handles or callable raw pointer APIs.

Compile-time major guards allow 10/11 and reject other majors pending review.
The concrete verification matrix and known limitations are in [README.md](README.md).
A successful compile against 11 headers does not establish runtime compatibility
with 11, or engine portability between SDK versions.
