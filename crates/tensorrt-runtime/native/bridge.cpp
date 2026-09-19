#include "bridge.h"
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#if NV_TENSORRT_MAJOR < 10 || NV_TENSORRT_MAJOR > 11
#error "This bridge targets TensorRT 10 and 11; review the API before enabling another major"
#endif
namespace {
thread_local char error_text[2048] = {};
struct Logger : nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) std::fprintf(stderr, "TensorRT: %s\n", message);
    }
};
// Process lifetime: plugin registry may retain this logger.
Logger logger;
std::once_flag plugins_once;
static_assert(static_cast<int>(nvinfer1::DataType::kFLOAT) == 0
    && static_cast<int>(nvinfer1::DataType::kHALF) == 1
    && static_cast<int>(nvinfer1::DataType::kINT8) == 2
    && static_cast<int>(nvinfer1::DataType::kINT32) == 3
    && static_cast<int>(nvinfer1::DataType::kBOOL) == 4
    && static_cast<int>(nvinfer1::DataType::kUINT8) == 5
    && static_cast<int>(nvinfer1::DataType::kFP8) == 6
    && static_cast<int>(nvinfer1::DataType::kBF16) == 7
    && static_cast<int>(nvinfer1::DataType::kINT64) == 8,
    "Update Rust DataType mapping for this SDK");
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void cuda_check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
void capture() noexcept {
    try { throw; }
    catch (const std::exception& e) { std::snprintf(error_text, sizeof(error_text), "%s", e.what()); }
    catch (...) { std::snprintf(error_text, sizeof(error_text), "Unknown native exception"); }
}
size_t element_size(nvinfer1::DataType type) {
    using D = nvinfer1::DataType;
    switch(type) {
    case D::kFLOAT: case D::kINT32: return 4;
    case D::kHALF: case D::kBF16: return 2;
    case D::kINT64: return 8;
    case D::kINT8: case D::kUINT8: case D::kBOOL: case D::kFP8: return 1;
    default: throw std::runtime_error("Unsupported I/O dtype (including packed INT4/FP4)");
    }
}
size_t byte_size(nvinfer1::Dims dims, nvinfer1::DataType type) {
    require(dims.nbDims >= 0 && dims.nbDims <= 8, "Invalid tensor rank");
    size_t n = element_size(type);
    for (int i = 0; i < dims.nbDims; ++i) {
        require(dims.d[i] >= 0, "Unresolved/data-dependent output shape is unsupported");
        auto d = static_cast<size_t>(dims.d[i]);
        require(d == 0 || n <= std::numeric_limits<size_t>::max() / d, "Tensor size overflow");
        n *= d;
    }
    return n;
}
struct Buffer {
    void* device = nullptr;
    size_t capacity = 0;
    std::vector<uint8_t> host;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : device(other.device), capacity(other.capacity), host(std::move(other.host)) { other.device = nullptr; }
    ~Buffer() { if (device) cudaFree(device); }
    void resize(size_t bytes) {
        if (capacity < std::max(size_t(1), bytes)) {
            void* next = nullptr;
            cuda_check(cudaMalloc(&next, std::max(size_t(1), bytes)));
            if (device) cudaFree(device);
            device = next;
            capacity = std::max(size_t(1), bytes);
        }
        host.resize(bytes);
    }
};
struct Synchronize {
    cudaStream_t stream;
    ~Synchronize() { cudaStreamSynchronize(stream); }
};
}
struct TrtSession {
    int device;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    std::vector<Buffer> buffers;
    bool ready = false;
    explicit TrtSession(int d) : device(d) {}
    ~TrtSession() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        context.reset();
        buffers.clear();
        if (stream) cudaStreamDestroy(stream);
    }
};
extern "C" const char* vt_error() { return error_text; }
extern "C" int32_t vt_version() { return NV_TENSORRT_MAJOR * 10000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH; }
extern "C" TrtSession* vt_open(const uint8_t* plan, size_t bytes, int32_t device) {
    try {
        require(bytes > 0, "Empty engine plan");
        cuda_check(cudaSetDevice(device));
        std::call_once(plugins_once, [] {
            require(initLibNvInferPlugins(&logger, ""), "Plugin registration failed");
        });
        auto s = std::make_unique<TrtSession>(device);
        s->runtime.reset(nvinfer1::createInferRuntime(logger));
        require(bool(s->runtime), "Cannot create TensorRT runtime");
        s->engine.reset(s->runtime->deserializeCudaEngine(plan, bytes));
        require(bool(s->engine), "Cannot deserialize engine; check SDK/GPU compatibility and plugins");
        s->context.reset(s->engine->createExecutionContext());
        require(bool(s->context), "Cannot create execution context");
        cuda_check(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking));
        s->buffers.resize(s->engine->getNbIOTensors());
        return s.release();
    } catch (...) { capture(); return nullptr; }
}
extern "C" void vt_close(TrtSession* s) { delete s; }
extern "C" int32_t vt_count(TrtSession* s) { return s->engine->getNbIOTensors(); }
static void info(TrtSession* s, int i, TrtInfo* out, bool resolved) {
    require(i >= 0 && i < vt_count(s), "Tensor index out of range");
    *out = {};
    out->name = s->engine->getIOTensorName(i);
    out->dtype = static_cast<int32_t>(s->engine->getTensorDataType(out->name));
    out->input = s->engine->getTensorIOMode(out->name) == nvinfer1::TensorIOMode::kINPUT;
    auto dims = resolved ? s->context->getTensorShape(out->name) : s->engine->getTensorShape(out->name);
    require(dims.nbDims >= 0 && dims.nbDims <= 8, "Invalid tensor rank");
    out->rank = dims.nbDims;
    std::copy_n(dims.d, dims.nbDims, out->dims);
}
extern "C" int32_t vt_info(TrtSession* s, int32_t i, TrtInfo* out) {
    try { info(s, i, out, false); return 0; } catch (...) { capture(); return -1; }
}
extern "C" int32_t vt_run(TrtSession* s, const TrtInput* inputs, size_t count, int32_t profile) {
    try {
        s->ready = false;
        cuda_check(cudaSetDevice(s->device));
        Synchronize guard{s->stream}; // Also drains queued work on every exceptional return.
        require(profile >= 0 && profile < s->engine->getNbOptimizationProfiles(), "Invalid optimization profile");
        require(s->context->setOptimizationProfileAsync(profile, s->stream), "Cannot select optimization profile");
        std::vector<const TrtInput*> by_index(vt_count(s), nullptr);
        for (size_t j = 0; j < count; ++j) {
            int index = -1;
            for (int i = 0; i < vt_count(s); ++i)
                if (std::strcmp(inputs[j].name, s->engine->getIOTensorName(i)) == 0) index = i;
            require(index >= 0, "Unknown input tensor");
            require(!by_index[index], "Duplicate input tensor");
            require(s->engine->getTensorIOMode(inputs[j].name) == nvinfer1::TensorIOMode::kINPUT, "Output supplied as input");
            by_index[index] = &inputs[j];
        }
        for (int i = 0; i < vt_count(s); ++i) {
            auto name = s->engine->getIOTensorName(i);
            require(s->engine->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE && !s->engine->isShapeInferenceIO(name), "Host/shape-inference I/O is unsupported");
            require(s->engine->getTensorFormat(name, profile) == nvinfer1::TensorFormat::kLINEAR, "Only LINEAR I/O is supported");
            if (s->engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
            const auto* input = by_index[i];
            require(input, "Missing input tensor");
            require(input->dtype == static_cast<int32_t>(s->engine->getTensorDataType(name)), "Input dtype mismatch");
            require(input->rank >= 0 && input->rank <= 8, "Invalid input rank");
            nvinfer1::Dims dims{};
            dims.nbDims = input->rank;
            std::copy_n(input->dims, input->rank, dims.d);
            require(s->context->setInputShape(name, dims), "Input shape incompatible with engine");
            require(byte_size(dims, s->engine->getTensorDataType(name)) == input->bytes, "Input byte count mismatch");
        }
        require(s->context->inferShapes(0, nullptr) == 0, "Shape inference failed or has unspecified inputs");
        for (int i = 0; i < vt_count(s); ++i) {
            auto name = s->engine->getIOTensorName(i);
            auto bytes = byte_size(s->context->getTensorShape(name), s->engine->getTensorDataType(name));
            auto& b = s->buffers[i];
            b.resize(bytes);
            require(s->context->setTensorAddress(name, b.device), "Cannot bind tensor address");
            if (by_index[i] && bytes) cuda_check(cudaMemcpyAsync(b.device, by_index[i]->data, bytes, cudaMemcpyHostToDevice, s->stream));
        }
        require(s->context->enqueueV3(s->stream), "TensorRT enqueueV3 failed");
        for (int i = 0; i < vt_count(s); ++i) {
            auto& b = s->buffers[i];
            if (!by_index[i] && !b.host.empty()) cuda_check(cudaMemcpyAsync(b.host.data(), b.device, b.host.size(), cudaMemcpyDeviceToHost, s->stream));
        }
        cuda_check(cudaStreamSynchronize(s->stream));
        s->ready = true;
        return 0;
    } catch (...) { capture(); return -1; }
}
extern "C" int32_t vt_output(TrtSession* s, int32_t i, TrtInfo* out, const uint8_t** data, size_t* bytes) {
    try {
        require(s->ready, "No successful inference outputs");
        info(s, i, out, true);
        require(!out->input, "Requested input as output");
        *data = s->buffers[i].host.data();
        *bytes = s->buffers[i].host.size();
        return 0;
    } catch (...) { capture(); return -1; }
}
