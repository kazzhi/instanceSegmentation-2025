#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::ostream& out = (severity == Severity::kINFO) ? std::cout : std::cerr;
            out << "[TensorRT] " << msg << '\n';
        }
    }
};

struct TRTDestroy
{
    template <typename T>
    void operator()(T* obj) const noexcept
    {
        if (!obj)
        {
            return;
        }
        callDestroy(obj, 0);
    }

private:
    template <typename T>
    static auto callDestroy(T* obj, int) -> decltype(obj->destroy(), void())
    {
        obj->destroy();
    }

    template <typename T>
    static void callDestroy(T* obj, long)
    {
        delete obj;
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

inline size_t volume(const nvinfer1::Dims& dims)
{
    size_t v = 1U;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        v *= dims.d[i];
    }
    return v;
}

void checkCuda(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << '\n';
        std::abort();
    }
}

class BinaryEntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    BinaryEntropyCalibrator(int batchSize,
        const nvinfer1::Dims& inputDims,
        std::vector<fs::path> samples,
        fs::path cacheFile)
        : mBatchSize(batchSize)
        , mInputDims(inputDims)
        , mSamples(std::move(samples))
        , mCacheFile(std::move(cacheFile))
    {
        mSampleSize = volume(mInputDims);
        checkCuda(cudaMalloc(&mDeviceInput, mSampleSize * mBatchSize * sizeof(float)));
    }

    ~BinaryEntropyCalibrator() override
    {
        if (mDeviceInput)
        {
            cudaFree(mDeviceInput);
        }
    }

    int getBatchSize() const noexcept override
    {
        return mBatchSize;
    }

    bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept override
    {
        (void) names;
        (void) nbBindings;
        if (mNextSample >= mSamples.size())
        {
            return false;
        }

        std::vector<float> hostBuffer(mSampleSize * mBatchSize, 0.0F);
        for (int b = 0; b < mBatchSize && mNextSample < mSamples.size(); ++b, ++mNextSample)
        {
            const auto& file = mSamples[mNextSample];
            std::ifstream in(file, std::ios::binary);
            if (!in)
            {
                std::cerr << "Failed to open calibration sample " << file << '\n';
                return false;
            }
            in.read(reinterpret_cast<char*>(hostBuffer.data() + b * mSampleSize), mSampleSize * sizeof(float));
            if (in.gcount() != static_cast<std::streamsize>(mSampleSize * sizeof(float)))
            {
                std::cerr << "Calibration sample " << file << " has unexpected size\n";
                return false;
            }
        }

        const cudaError_t status = cudaMemcpy(mDeviceInput, hostBuffer.data(), hostBuffer.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (status != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(status) << '\n';
            return false;
        }

        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        mCalibrationCache.clear();
        if (mCacheFile.empty() || !fs::exists(mCacheFile))
        {
            length = 0;
            return nullptr;
        }

        std::ifstream in(mCacheFile, std::ios::binary);
        if (!in)
        {
            length = 0;
            return nullptr;
        }

        mCalibrationCache.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        length = mCalibrationCache.size();
        return mCalibrationCache.data();
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        if (mCacheFile.empty() || cache == nullptr || length == 0)
        {
            return;
        }
        std::ofstream out(mCacheFile, std::ios::binary);
        out.write(static_cast<const char*>(cache), static_cast<std::streamsize>(length));
    }
private:
    int mBatchSize{};
    nvinfer1::Dims mInputDims{};
    size_t mSampleSize{};
    std::vector<fs::path> mSamples{};
    fs::path mCacheFile{};
    void* mDeviceInput{nullptr};
    size_t mNextSample{0};
    std::vector<char> mCalibrationCache{};
};

nvinfer1::ITensor* findTensorByName(nvinfer1::INetworkDefinition& network, const std::string& tensorName)
{
    for (int i = 0; i < network.getNbInputs(); ++i)
    {
        nvinfer1::ITensor* tensor = network.getInput(i);
        if (tensor && tensor->getName() && tensorName == tensor->getName())
        {
            return tensor;
        }
    }

    for (int i = 0; i < network.getNbLayers(); ++i)
    {
        nvinfer1::ILayer* layer = network.getLayer(i);
        for (int o = 0; o < layer->getNbOutputs(); ++o)
        {
            nvinfer1::ITensor* tensor = layer->getOutput(o);
            if (tensor && tensor->getName() && tensorName == tensor->getName())
            {
                return tensor;
            }
        }
    }

    for (int i = 0; i < network.getNbOutputs(); ++i)
    {
        nvinfer1::ITensor* tensor = network.getOutput(i);
        if (tensor && tensor->getName() && tensorName == tensor->getName())
        {
            return tensor;
        }
    }

    return nullptr;
}

nvinfer1::ILayer* findProducerLayer(nvinfer1::INetworkDefinition& network, const nvinfer1::ITensor& tensor)
{
    for (int i = 0; i < network.getNbLayers(); ++i)
    {
        nvinfer1::ILayer* layer = network.getLayer(i);
        for (int o = 0; o < layer->getNbOutputs(); ++o)
        {
            if (layer->getOutput(o) == &tensor)
            {
                return layer;
            }
        }
    }
    return nullptr;
}

void enforceHalfPrecisionForBranch(nvinfer1::INetworkDefinition& network, const std::vector<std::string>& branchSeeds)
{
    std::unordered_set<const nvinfer1::ITensor*> visitedTensors;
    std::queue<nvinfer1::ITensor*> pending;

    for (const std::string& seedName : branchSeeds)
    {
        if (nvinfer1::ITensor* tensor = findTensorByName(network, seedName))
        {
            pending.push(tensor);
            if (tensor->isNetworkOutput())
            {
                tensor->setType(nvinfer1::DataType::kHALF);
            }
        }
        else
        {
            std::cerr << "Warning: tensor \"" << seedName << "\" not found, skipping FP16 enforcement for it.\n";
        }
    }

    while (!pending.empty())
    {
        nvinfer1::ITensor* tensor = pending.front();
        pending.pop();

        if (tensor == nullptr || !visitedTensors.insert(tensor).second)
        {
            continue;
        }

        nvinfer1::ILayer* layer = findProducerLayer(network, *tensor);
        if (layer == nullptr)
        {
            continue;
        }

        layer->setPrecision(nvinfer1::DataType::kHALF);

        for (int o = 0; o < layer->getNbOutputs(); ++o)
        {
            nvinfer1::ITensor* output = layer->getOutput(o);
            if (output)
            {
                layer->setOutputType(o, nvinfer1::DataType::kHALF);
                if (output->isNetworkOutput())
                {
                    output->setType(nvinfer1::DataType::kHALF);
                }
            }
        }

        for (int i = 0; i < layer->getNbInputs(); ++i)
        {
            if (nvinfer1::ITensor* input = layer->getInput(i))
            {
                pending.push(input);
            }
        }
    }
}

struct ProgramOptions
{
    fs::path onnxPath;
    fs::path enginePath;
    std::vector<std::string> fp16Seeds;
    std::optional<fs::path> calibrationDir;
    std::optional<fs::path> calibrationCache;
    int calibrationBatch{1};
    bool obeyPrecision{true};
};

ProgramOptions parseArgs(int argc, char** argv)
{
    ProgramOptions opts;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        auto takeValue = [&](fs::path& target) {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value after " + arg);
            }
            target = fs::path(argv[++i]);
        };

        if (arg == "--onnx")
        {
            takeValue(opts.onnxPath);
        }
        else if (arg == "--engine")
        {
            takeValue(opts.enginePath);
        }
        else if (arg == "--fp16-tensor")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value after --fp16-tensor");
            }
            opts.fp16Seeds.emplace_back(argv[++i]);
        }
        else if (arg == "--calib-dir")
        {
            fs::path tmp;
            takeValue(tmp);
            opts.calibrationDir = tmp;
        }
        else if (arg == "--calib-cache")
        {
            fs::path tmp;
            takeValue(tmp);
            opts.calibrationCache = tmp;
        }
        else if (arg == "--calib-batch")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value after --calib-batch");
            }
            opts.calibrationBatch = std::stoi(argv[++i]);
        }
        else if (arg == "--no-obey-precision")
        {
            opts.obeyPrecision = false;
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " --onnx model.onnx --engine model.engine "
                      << "[--fp16-tensor tensor_name ...] [--calib-dir dir] [--calib-cache file] "
                      << "[--calib-batch N] [--no-obey-precision]\n"
                      << "Provide at least one --fp16-tensor matching the proto-mask branch output(s).\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.onnxPath.empty() || opts.enginePath.empty())
    {
        throw std::runtime_error("Both --onnx and --engine are required.");
    }

    if (opts.fp16Seeds.empty())
    {
        std::cerr << "Warning: no --fp16-tensor provided; no branch will be pinned to FP16.\n";
    }

    return opts;
}

std::vector<fs::path> listCalibrationSamples(const fs::path& dir)
{
    std::vector<fs::path> samples;
    for (const auto& entry : fs::directory_iterator(dir))
    {
        if (entry.is_regular_file())
        {
            const fs::path& file = entry.path();
            if (file.extension() == ".bin")
            {
                samples.push_back(file);
            }
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples;
}

int main(int argc, char** argv)
{
    TRTLogger logger;
    ProgramOptions options;
    try
    {
        options = parseArgs(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Argument error: " << e.what() << '\n';
        return 1;
    }

    TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        std::cerr << "Failed to create TensorRT builder\n";
        return 1;
    }

    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cerr << "Failed to create TensorRT network\n";
        return 1;
    }

    TRTUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser)
    {
        std::cerr << "Failed to create ONNX parser\n";
        return 1;
    }

    if (!parser->parseFromFile(options.onnxPath.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "Failed to parse ONNX model: " << options.onnxPath << '\n';
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            const auto* err = parser->getError(i);
            std::cerr << "Parser error " << i << ": " << err->desc() << '\n';
        }
        return 1;
    }

    enforceHalfPrecisionForBranch(*network, options.fp16Seeds);

    TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config)
    {
        std::cerr << "Failed to create builder config\n";
        return 1;
    }

    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (options.obeyPrecision)
    {
        config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
    }

    auto profile = builder->createOptimizationProfile();
    if (!profile)
    {
        std::cerr << "Failed to create optimization profile\n";
        return 1;
    }

    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        nvinfer1::ITensor* input = network->getInput(i);
        nvinfer1::Dims dims = input->getDimensions();
        if (dims.d[0] == -1)
        {
            dims.d[0] = options.calibrationBatch;
        }
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
    }
    if (!profile->isValid())
    {
        std::cerr << "Optimization profile is invalid\n";
        return 1;
    }
    config->addOptimizationProfile(profile);

    TRTUniquePtr<BinaryEntropyCalibrator> calibrator;
    if (options.calibrationDir)
    {
        std::vector<fs::path> samples = listCalibrationSamples(*options.calibrationDir);
        if (samples.empty())
        {
            std::cerr << "Calibration directory " << *options.calibrationDir << " does not contain .bin samples\n";
            return 1;
        }
        if (network->getNbInputs() != 1)
        {
            std::cerr << "Calibration helper only supports single-input networks\n";
            return 1;
        }

        calibrator.reset(new BinaryEntropyCalibrator(
            options.calibrationBatch,
            network->getInput(0)->getDimensions(),
            std::move(samples),
            options.calibrationCache.value_or(fs::path())));
        config->setInt8Calibrator(calibrator.get());
    }
    else if (options.calibrationCache)
    {
        calibrator.reset(new BinaryEntropyCalibrator(
            options.calibrationBatch,
            network->getInput(0)->getDimensions(),
            {},
            *options.calibrationCache));
        config->setInt8Calibrator(calibrator.get());
    }
    else
    {
        std::cerr << "Warning: no calibration data supplied. Ensure your network contains QAT scales or provide --calib-dir / --calib-cache.\n";
    }

    TRTUniquePtr<nvinfer1::IHostMemory> engineBlob(builder->buildSerializedNetwork(*network, *config));
    if (!engineBlob)
    {
        std::cerr << "Failed to build TensorRT engine\n";
        return 1;
    }

    if (!options.enginePath.empty())
    {
        std::ofstream out(options.enginePath, std::ios::binary);
        out.write(static_cast<const char*>(engineBlob->data()), static_cast<std::streamsize>(engineBlob->size()));
        if (!out)
        {
            std::cerr << "Failed to write engine file " << options.enginePath << '\n';
            return 1;
        }
        std::cout << "Engine written to " << options.enginePath << '\n';
    }

    std::cout << "FP16 branch enforcement complete.\n";
    return 0;
}
