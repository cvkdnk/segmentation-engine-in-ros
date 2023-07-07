//
// Created by cvkdnk on 2023/2/9
//

#ifndef ROS_ENGINE_LOADER_H
#define ROS_ENGINE_LOADER_H

#include "common.h"
#include <vector>
#include <memory>
#include <string>
#include <cmath>

using namespace std;


//!
//! \class PointCloudLoader
//! \brief To read point clouds data from different dataset such as SemanticKITTI
//! and Nuscenes, this class sets out a series of guidelines about dataloading class.
//! This is a base class to load data. Every dataset needs its own class which should 
//! be inheriting the base class.
//!
class PointCloudLoader
{
public:
    PointCloudLoader(){};
    PointCloudLoader(int32_t MaxXY){
        mXYRange[0] = -MaxXY;
        mXYRange[1] = MaxXY;
    }; 
    virtual ~PointCloudLoader()=default;

    // 2 methods to read points cloud data.
    // return: An array about point cloud data, which dim is [pointsNum * 4]
    virtual PointCloudData* readFromFile(const string& filePath)=0;
    // virtual PointCloudData* readFromMsg()=0;

    void setGridSize(float girdSize){
        mGridSampleSize[0] = girdSize; 
        mGridSampleSize[1] = girdSize; 
        mGridSampleSize[2] = girdSize; 
    }

protected:
    // Raw Point Cloud's data may have 4 Dims including x, y, z, i. But CPGNet needs
    // inputs be 7 Dims. The extendDims() does this work.
    bool extendDims(PointCloudData& inputs);
    
    // TensorRT Engine limits the number of points in a fixed range. So the filterData()
    // the points cloud to fix the limitation.
    virtual bool filterData(PointCloudData& pcd)=0;

    // use to get unistr from filename
    virtual string getUniStr(const string& filePath)=0;

    // print PointCloudLoader's info
    void examinePCD(PointCloudData& pcd);

    // Engine limits the number of points.
    int32_t mMaxPointsNum{50000};
    int32_t mMinPointsNum{30000};

    // Engine limits the range of point clouds.
    int32_t mXYRange[2]{-50, 50};
    int32_t mPitchRange[2]{-25, 3};
    
    int32_t mBevImgShape[2]{600, 600};
    float mGridSampleSize[3]{0.1, 0.1, 0.1};
};


//!
//! \class SemanticKITTILoader
//! \brief A class to load Semantic KITTI Dataset.
//!
class SemanticKITTILoader: public PointCloudLoader
{
public:
    SemanticKITTILoader(): PointCloudLoader(){};
    virtual ~SemanticKITTILoader()=default;

    virtual PointCloudData* readFromFile(const string& filePath);
    // virtual PointCloudData* readFromMsg();
protected:
    int32_t yTimes = std::ceil(mXYRange[1] * 2 / mGridSampleSize[0]);
    int32_t zTimes = std::ceil(mXYRange[1] * 2 / mGridSampleSize[1]) * yTimes;

    bool readBin(PointCloudData& pcd);
    virtual bool filterData(PointCloudData& pcd);
    virtual string getUniStr(const string& filePath);
    int32_t getHashKey(const float x, const float y, const float z);
    int32_t getHashKey(const int32_t pointIndex, vector<float>& buffer);
};

#endif // ROS_ENGINE_LOADER_H