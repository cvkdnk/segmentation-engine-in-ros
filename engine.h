//
// Created by cvkdnk on 2023/2/16
//

#ifndef ROS_ENGINE_ENGINE_H
#define ROS_ENGINE_ENGINE_H

#include "NvInfer.h"
#include "common.h"
#include "spdlog/spdlog.h"
#include <memory>
#include <string>
#include <numeric>


struct InferDeleter {
    template<typename T>
    void operator()(T *obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template<typename T>
using UniquePtr = unique_ptr<T, InferDeleter>;


class CPGNetEngine
{
public:
    CPGNetEngine(string enginePath){
        loadEngine(enginePath);
        mContext = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!mContext) {
            spdlog::error("Context init failed.");
        }
    }
    ~CPGNetEngine(){delete mEngine;}

    bool infer(const PointCloudData& pcd, vector<int32_t>& results);
protected:
    UniquePtr<nvinfer1::IExecutionContext> mContext = nullptr;
    nvinfer1::ICudaEngine* mEngine = nullptr;
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;

    bool loadEngine(const string& enginePath);

    size_t getMemorySize(const nvinfer1::Dims &dims, const int32_t elem_size) {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
    }
};


#endif // ROS_ENGINE_ENGINE_H