// Build a tiny dynamic, two-input/two-output plan on the target GPU.
#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
struct Logger : nvinfer1::ILogger {
    void log(Severity s, const char* m) noexcept override {
        if (s <= Severity::kWARNING) std::cerr << m << '\n';
    }
};
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) return 3;
    uint32_t flags = 0;
#if NV_TENSORRT_MAJOR < 11
    flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#endif
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!network || !config) return 4;
    nvinfer1::Dims dims{}; dims.nbDims = 2; dims.d[0] = -1; dims.d[1] = 3;
    for (auto name : {"x", "z"}) {
        auto* input = network->addInput(name, nvinfer1::DataType::kFLOAT, dims);
        auto* layer = input ? network->addIdentity(*input) : nullptr;
        if (!layer) return 5;
        layer->getOutput(0)->setName(name[0] == 'x' ? "y" : "w");
        network->markOutput(*layer->getOutput(0));
    }
    for (int i = 0; i < 2; ++i) {
        auto* profile = builder->createOptimizationProfile();
        if (!profile) return 6;
        for (auto name : {"x", "z"}) {
            dims.d[0] = 1;
            if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, dims)) return 7;
            dims.d[0] = 2 + i;
            if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, dims)) return 7;
            dims.d[0] = 4 + i;
            if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, dims)) return 7;
        }
        if (config->addOptimizationProfile(profile) < 0) return 8;
    }
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) return 9;
    std::ofstream file(argv[1], std::ios::binary);
    file.write(static_cast<const char*>(plan->data()), plan->size());
    return file ? 0 : 10;
}
