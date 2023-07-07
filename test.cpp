#include "spdlog/spdlog.h"
#include "engine.h"
#include "loader.h"
#include <fstream>
#include <ctime>
#include <dirent.h>
#include <unistd.h>
#include <algorithm>

using namespace std;

void getFileNames(string path, vector<string>& files);

template <typename T>
void saveVector(string path, vector<T>& vec);

int main()
{
    string enginePath = "/root/autodl-nas/scatter_net.engine";
    string dataDir = "/root/autodl-tmp/dataset/sequences/00/velodyne/";
    string saveDir = "/root/autodl-tmp/results/0217/";
    // string dataPath = "/root/autodl-tmp/dataset/sequences/00/velodyne/000000.bin";
    spdlog::set_level(spdlog::level::info);
    vector<string> files;
    getFileNames(dataDir, files);

    clock_t start = clock();
    unique_ptr<CPGNetEngine> engine{new CPGNetEngine(enginePath)};
    clock_t endInitEngine = clock();
    spdlog::debug("success init engine object, takes {:.4f}s", (double)(endInitEngine-start)/CLOCKS_PER_SEC);
    unique_ptr<PointCloudLoader> loader{new SemanticKITTILoader()};
    loader->setGridSize(0.08);
    clock_t endInitLoader = clock();
    spdlog::debug("success init loader object, takes {:.4f}s", (double)(endInitLoader-endInitEngine)/CLOCKS_PER_SEC);
    vector<int32_t> results;
    PointCloudData* pcd = nullptr;
    for(auto it = files.begin(); it != files.end(); it++){
        spdlog::info("=================================================================");
        pcd = loader->readFromFile(*it);
        
        if (engine->infer(*pcd, results)) {
            spdlog::debug("infer success");
        }
        else{
            spdlog::debug("infer fail");
        }

        saveVector(saveDir+"pred/"+pcd->mUniStr+".txt", results);
        saveVector(saveDir+"keep/"+pcd->mUniStr+".txt", pcd->mKeepIndex);
        saveVector(saveDir+"map/"+pcd->mUniStr+".txt", pcd->mRawMapSample);

        delete pcd;
    }
    spdlog::debug("Complete");

    return 0;
}

void getFileNames(string path, vector<string>& files)
{
	DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(path.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    char dot[3] = ".";
    char dotdot[6] = "..";

    while ((ptr=readdir(dir)) != NULL){
        if ( (strcmp(ptr->d_name, dot) != 0) && (strcmp(ptr->d_name, dotdot) != 0) ){
            files.push_back(path+ptr->d_name);
            spdlog::debug(path+ptr->d_name);
        }
    }
    sort(files.begin(), files.end());
    closedir(dir);
}

template <typename T>
void saveVector(string path, vector<T>& vec)
{
    ofstream f;
    f.open(path, ios_base::out);
    if (f.is_open()){
        for (auto it=vec.begin(); it!=vec.end(); it++)
        {
            f << (*it) << " ";
        }
    }
    else{
        spdlog::error("open wirte file fail, {}.", path);
    }
    f.close();
}
