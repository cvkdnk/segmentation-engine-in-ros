//
// Created by cvkdnk on 2023/2/16
//

#include "engine.h"
#include "spdlog/spdlog.h"
#include "NvInfer.h"
#include "logger.h"
#include <fstream>
#include <ctime>
#include <cuda_runtime_api.h>


bool CPGNetEngine::infer(const PointCloudData& pcd, vector<int32_t>& results)
{
    clock_t start = clock();

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        spdlog::error("CUDA stream creation failed.");
        return false;
    }

    // Set input
    auto inputIndex = mEngine->getBindingIndex("points");
    if (mEngine->getBindingDataType(inputIndex) != nvinfer1::DataType::kFLOAT){
        spdlog::error("Wrong input data type.");
        return false;
    }
    mInputDims = nvinfer1::Dims2{pcd.mPointsNum, extendedDims};
    mContext->setBindingDimensions(inputIndex, mInputDims);
    auto inputSize = getMemorySize(mInputDims, sizeof(float));

    // Set output
    auto outputIndex = mEngine->getBindingIndex("logits");
    if (mEngine->getBindingDataType(outputIndex) != nvinfer1::DataType::kINT32){
        spdlog::error("Wrong output data type.");
        return false;
    }
    mOutputDims = nvinfer1::Dims{};
    mOutputDims.nbDims = 1;
    mOutputDims.d[0] = pcd.mPointsNum;
    for (int32_t i{mOutputDims.nbDims}; i<nvinfer1::Dims::MAX_DIMS; i++)
        mOutputDims.d[i] = 0;
    auto outputSize = getMemorySize(mOutputDims, sizeof(int32_t));

    // Allocate CUDA memory for input and output bindings.
    void *inputMem{nullptr};
    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess) {
        spdlog::error("input CUDA memory allocation failed, size = {} bytes.", inputSize);
        return false;
    }
    void *outputMem{nullptr};
    if (cudaMalloc(&outputMem, outputSize) != cudaSuccess) {
        spdlog::error("output CUDA memory allocation failed, size = {} bytes", outputSize);
        return false;
    }
    spdlog::debug("Need to allocate CUDA memory: input {}B, output {}B; points num: {}", inputSize, outputSize, pcd.mPointsNum);

    // Copy image data to input binding memory
    if (cudaMemcpyAsync(inputMem, pcd.mDataBuffer.data(), inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        spdlog::error("CUDA memory copy of input failed, size = {} bytes", inputSize);
        return false;
    }

    clock_t end1 = clock();
    spdlog::debug("Set memory takes {:.4f}s", (double)(end1-start)/CLOCKS_PER_SEC);


    //Run TensorRT inference
    void* bindings[] = {inputMem, outputMem};
    if (!mContext->enqueueV2(bindings, stream, nullptr)) {
        spdlog::error("TensorRT inferrence failed.");
        return false;
    }

    clock_t end2 = clock();
    spdlog::debug("Infer takes {:.4f}s", (double)(end2-end1)/CLOCKS_PER_SEC);
    
    // Copy predictions from output binding memory.
    results.resize(outputSize/sizeof(int32_t));
    if (cudaMemcpyAsync(results.data(), outputMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess){
        spdlog::error("CUDA memory copy of output failed, size = {}B", outputSize);
        return false;
    }
    cudaStreamSynchronize(stream);
    clock_t end3 = clock();
    spdlog::debug("Copy predict result from cuda takes {:.4f}s", (double)(end3-end2)/CLOCKS_PER_SEC);

    cudaFree(inputMem);
    cudaFree(outputMem);

    spdlog::info("Inference \'{}\' takes {:.4f}s", pcd.mUniStr, (double)(end3-start)/CLOCKS_PER_SEC);
    return true;
}


bool CPGNetEngine::loadEngine(const string& enginePath)
{
    spdlog::info("Loading engine file: {}", enginePath);
    clock_t start = clock();
    ifstream file(enginePath, ios::binary);
    if (!file.good()) {
        spdlog::error("Can't open the engine file.");
        return false;
    }

    file.seekg(0, ifstream::end);
    auto fsize = file.tellg();
    file.seekg(0, ifstream::beg);
    spdlog::info("Engine file size: {}KB", fsize/1024);

    vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);

    UniquePtr <nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(cpgnet::gLogger)};                                                                                                                          
    // initLibNvInferPlugins(&cpgnet::gLogger, "");
    mEngine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);
    if (mEngine == nullptr){
        spdlog::error("Failing to load engine.");
        return false;
    }
    clock_t end = clock();
    spdlog::debug("Loading engine successfully, it takes {:.4f}s.", (double)(end-start)/CLOCKS_PER_SEC);
    return true;
}

