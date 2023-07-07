//
// Created by cvkdnk on 2023/2/14
//

#ifndef ROS_ENGINE_STRUCT_H
#define ROS_ENGINE_STRUCT_H

#include <vector>
#include <string>

using namespace std;

const int32_t rawDims = 4;
const int32_t extendedDims = 7;

typedef struct PointCloudData{
    string mUniStr; // When saving results, it is used as the file name.
    string mFilePath;
    int32_t mPointsNum;

    // After filtering and downsampling, it records keeped points' index.
    vector<int32_t> mKeepIndex; 

    // After filtering and downsampling, it records points' label mapping.
    // Those removed points' label should be same as corresponding point's label.
    vector<int32_t> mRawMapSample; 
    
    vector<float> mDataBuffer;
} PointCloudData;

#endif // ROS_ENGINE_STRUCT_H

