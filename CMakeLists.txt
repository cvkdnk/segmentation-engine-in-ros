cmake_minimum_required(VERSION 3.16)

set(CUDA_HOME "/usr/local/cuda-11.6")
set(CPGNET_LIBS "/root/autodl-tmp/libscatter_max_tensorrt.so")

set(CMAKE_CXX_COMPILE "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc")

project(cpgnet
        LANGUAGES CXX CUDA
)

include_directories(
    "${CUDA_HOME}/include"
)

include_directories(/root/autodl-tmp/TensorRT-8.4.3.1//include)
include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(/root/autodl-tmp/TensorRT-8.4.3.1//lib)
find_library(SPDLOG_LIB spdlog)

message(STATUS "include ${CUDA_HOME}/include")

file(GLOB TRT_LIBS "/root/autodl-tmp/TensorRT-8.4.3.1/lib/*")

file(GLOB CUDA_LIBS "${CUDA_HOME}/lib64/*.so")
list(REMOVE_ITEM CUDA_LIBS "${CUDA_HOME}/lib64/libnvrtc.so")
list(APPEND CUDA_LIBS "${CUDA_HOME}/lib64/libnvrtc.so.11.6.55")


add_executable(
    test
    test.cpp
    common.h
    loader.h loader.cpp
    engine.h engine.cpp
    logging.h
    logger.h logger.cpp
)

target_link_libraries(test "${CUDA_LIBS}")
target_link_libraries(test "${TRT_LIBS}")
target_link_libraries(test ${CPGNET_LIBS})
target_link_libraries(test ${SPDLOG_LIB})
