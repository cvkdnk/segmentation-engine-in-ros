//
// Created by cvkdnk on 2023/2/14
//

#include "loader.h"
#include "spdlog/spdlog.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <unordered_map>

using namespace std;

bool PointCloudLoader::extendDims(PointCloudData& inputs)
{
    if (inputs.mPointsNum < mMinPointsNum || inputs.mPointsNum > mMaxPointsNum){
        spdlog::error("The number of points should be between 30,000 and 50,000, currently consisting of {} points.", 
            inputs.mPointsNum);
        return false;
    }

    vector<float> buffer(inputs.mPointsNum * extendedDims);
    float xy_range = mXYRange[1] - mXYRange[0]; 
    for (int pointIndex = 0; pointIndex < inputs.mPointsNum; pointIndex++)
    {
        float distance, u, v, u_center, v_center;
        for (int xyzr = 0; xyzr < 4; xyzr++) {
            buffer[pointIndex * 7 + xyzr] = inputs.mDataBuffer[pointIndex * 4 + xyzr];
        }
        distance = 0;
        for (int xyz = 0; xyz < 3; xyz++)
            distance += pow(inputs.mDataBuffer[pointIndex*4+xyz], 2);
        distance = sqrt(distance);
        buffer[pointIndex*7+4] = distance;
        u = (inputs.mDataBuffer[pointIndex*4]-mXYRange[0]) / xy_range * mBevImgShape[0];
        v = (inputs.mDataBuffer[pointIndex*4+1]-mXYRange[0]) / xy_range * mBevImgShape[1];
        buffer[pointIndex*7+5] = u - (floor(u)+0.5);
        buffer[pointIndex*7+6] = v - (floor(v)+0.5);
    }
    inputs.mDataBuffer = buffer;
    return true;
}


void PointCloudLoader::examinePCD(PointCloudData& pcd)
{
    spdlog::debug("pcd.mUniStr: {}", pcd.mUniStr);
    spdlog::debug("pcd.mFilePath: {}", pcd.mFilePath);
    spdlog::debug("pcd.mPointsNum: {}", pcd.mPointsNum);
    spdlog::debug("pcd.mKeepIndex.size(): {}", pcd.mKeepIndex.size());
    spdlog::debug("pcd.mRawMapSample.size(): {}", pcd.mRawMapSample.size());
    spdlog::debug("pcd.mDataBuffer.size(): {}", pcd.mDataBuffer.size());
}


PointCloudData* SemanticKITTILoader::readFromFile(const string& filePath)
{
    auto pcd = new PointCloudData{
        .mUniStr=getUniStr(filePath),
        .mFilePath=filePath
    };

    clock_t start = clock();

    if (!readBin(*pcd)) return nullptr;
    clock_t endRead = clock();
    spdlog::debug("Reading bin file takes {:.4f}s.", (double)(endRead-start)/CLOCKS_PER_SEC);

    if (!filterData(*pcd)) return nullptr;
    clock_t endFilter = clock();
    spdlog::debug("Filtering and downsampling point cloud takes {:.4f}s.", (double)(endFilter-endRead)/CLOCKS_PER_SEC);

    if (!extendDims(*pcd)) return nullptr;
    clock_t endExtendDims = clock();
    spdlog::debug("Extending channels takes {:.4f}s.", (double)(endExtendDims-endFilter)/CLOCKS_PER_SEC);

    examinePCD(*pcd);
    spdlog::info("Reading and processing point cloud takes {:.4f}s totally.", (double)(endExtendDims-start)/CLOCKS_PER_SEC);
    return pcd;
}

// file path example: /dataset/sequences/00/velodyne/000000.bin
string SemanticKITTILoader::getUniStr(const string& filePath)
{
    string uniStr(filePath, filePath.length()-22, 2);
    uniStr += "_";
    uniStr += string(filePath, filePath.length()-10, 6);
    return uniStr;
}


bool SemanticKITTILoader::readBin(PointCloudData& pcd)
{
    spdlog::debug("Load point cloud from bin file.");
    clock_t start = clock();

    std::ifstream infile(pcd.mFilePath, std::ifstream::binary);
    if (!infile.is_open()){
        spdlog::error("Cannot open point cloud bin file: {}", pcd.mFilePath);
        return false;
    }

    // getting file size.
    infile.seekg(0, std::ifstream::end);
    auto fsize = infile.tellg();
    infile.seekg(0, std::ifstream::beg);

    // loading data into mDataBuffer and getting mPointsNum
    pcd.mDataBuffer.resize(fsize / sizeof(float));
    infile.read((char*)pcd.mDataBuffer.data(), fsize);
    pcd.mPointsNum = (fsize/sizeof(float)) / rawDims; // Original number of channels, xyzr.

    // finish read bin file.
    spdlog::debug("Current point cloud has {} points.", pcd.mPointsNum);
    infile.close();

    return true;
}


bool SemanticKITTILoader::filterData(PointCloudData& pcd)
{
    if (pcd.mPointsNum < mMinPointsNum)
    {
        spdlog::warn("{} have not enough points to use engine. Skip this scene.", pcd.mUniStr);
        return false;
    }
    vector<float> newBuffer(mMaxPointsNum * rawDims);
    pcd.mKeepIndex.resize(mMaxPointsNum);
    pcd.mRawMapSample.resize(pcd.mPointsNum);
    int32_t newPointsNum = 0;
    unordered_map<int32_t, int32_t> hashtable;
    for (int32_t pointIndex = 0; pointIndex < pcd.mPointsNum; pointIndex++)
    {
        if ((pcd.mPointsNum - pointIndex) <= (mMinPointsNum - newPointsNum))
        {
            spdlog::debug("Stopping down sampling: Have not enough points.");
            while(pointIndex < pcd.mPointsNum)
            {
                for (int xyzr=0; xyzr<4; xyzr++)
                {
                    newBuffer[newPointsNum*rawDims+xyzr] = pcd.mDataBuffer[pointIndex*rawDims+xyzr];
                }
                pcd.mRawMapSample[pointIndex] = newPointsNum;
                pcd.mKeepIndex[newPointsNum++] = pointIndex++;
            }
            break;
        }

        if (newPointsNum >= mMaxPointsNum)
        {
            spdlog::debug("Stopping down sampling: Too much points, remove superfluous points");
            while(pointIndex < pcd.mPointsNum)
            {
                pcd.mRawMapSample[pointIndex++] = -1;
            }
            break;
        }

        // Examine the point whether in the valid xy range.
        if (abs(pcd.mDataBuffer[pointIndex*rawDims]<mXYRange[1] && abs(pcd.mDataBuffer[pointIndex*rawDims+1]<mXYRange[1])))
        {
            float distance = 0, pitch;
            for (int xyz = 0; xyz < 3; xyz++)
                distance += pow(pcd.mDataBuffer[pointIndex*rawDims + xyz], 2);
            distance = sqrt(distance);
            pitch = asin(pcd.mDataBuffer[pointIndex*rawDims+2] / distance);
            if (pitch >= mPitchRange[0] && pitch <= mPitchRange[1])
            {
                int32_t hashkey = getHashKey(pointIndex, pcd.mDataBuffer);
                auto findResult = hashtable.find(hashkey);
                if (findResult == hashtable.end()){
                    for (int xyzr=0; xyzr<4; xyzr++)
                    {
                        newBuffer[newPointsNum*rawDims+xyzr] = pcd.mDataBuffer[pointIndex*rawDims+xyzr];
                    }
                    pcd.mKeepIndex[newPointsNum] = pointIndex;
                    hashtable.insert({hashkey, newPointsNum});
                    pcd.mRawMapSample[pointIndex] = newPointsNum;
                    newPointsNum++;
                }
                else
                {
                    pcd.mRawMapSample[pointIndex] = findResult->second;
                }
            }
            else
            {
                pcd.mRawMapSample[pointIndex] = -1;
            }
        }
        else
        {
            pcd.mRawMapSample[pointIndex] = -1;
        }
    }
    spdlog::info("Grid sampling: {} --> {}", pcd.mPointsNum, newPointsNum);
    pcd.mDataBuffer = newBuffer;
    pcd.mPointsNum = newPointsNum;
    return true;
}


int32_t SemanticKITTILoader::getHashKey(const float x, const float y, const float z)
{
    int32_t hashkey = floor((x + mXYRange[1]) / mGridSampleSize[0]) +
                      floor((y + mXYRange[1]) / mGridSampleSize[1]) * yTimes +
                      floor((z + 10) / mGridSampleSize[2]) * zTimes;
    return hashkey;
}


int32_t SemanticKITTILoader::getHashKey(const int32_t pointIndex, vector<float>& buffer)
{
    int32_t hashkey = floor((buffer[pointIndex*4] + mXYRange[1]) / mGridSampleSize[0]) +
                      floor((buffer[pointIndex*4+1] + mXYRange[1]) / mGridSampleSize[1]) * yTimes +
                      floor((buffer[pointIndex*4+2] + 10) / mGridSampleSize[2]) * zTimes;
    return hashkey;
}