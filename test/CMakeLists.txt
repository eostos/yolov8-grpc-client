cmake_minimum_required(VERSION 3.16)

project(triton_client_rtsp)
message(STATUS "Building project: ${PROJECT_NAME}")
message(STATUS "Using TritonClientBuild_DIR = $ENV{TritonClientBuild_DIR}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TritonClient_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonCommon")
# --- Find dependencies ---
find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)
find_package(CURL REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)

# --- Source file ---
set(SOURCES
    ${PROJECT_SOURCE_DIR}/main-ipc.cpp
)

# --- Executable ---
add_executable(${PROJECT_NAME} ${SOURCES})

# --- Include directories ---
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    /usr/local/cuda/include
    ${OpenCV_INCLUDE_DIRS}
    $ENV{TritonClientBuild_DIR}/include
    /workspace/client/src/c++/library
    ${PROJECT_SOURCE_DIR}
)

# --- Link directories ---
target_link_directories(${PROJECT_NAME}
    PRIVATE
    $ENV{TritonClientBuild_DIR}/lib
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
)

# --- Compiler definitions ---
#target_compile_definitions(${PROJECT_NAME} PRIVATE
#    TRITON_ENABLE_GPU=1
#    TRITON_NO_GPU=0
#)

# --- Libraries ---
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    grpcclient
    httpclient
    ${OpenCV_LIBS}
    CURL::libcurl
    jsoncpp
    tbb
    boost_system
    boost_filesystem
    ${CUDA_LIBRARIES}
    CUDA::cudart
    pthread
    ev
)

# --- Optional warning suppression ---
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wno-deprecated-enum-enum-conversion
)

message(STATUS "✅ Triton RTSP client configured successfully")
