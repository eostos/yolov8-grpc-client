cmake_minimum_required(VERSION 3.0)

project(computer-vision-triton-cpp-client)
message(STATUS ${PROJECT_NAME})
message(STATUS $ENV{TritonClientBuild_DIR})
set(CMAKE_CXX_STANDARD 20)

option(WITH_SHOW_FRAME "Option description" ON)
option(WITH_WRITE_FRAME "Option description" OFF)

if(WITH_SHOW_FRAME)
  add_definitions(-DSHOW_FRAME)
  message(STATUS "Show frame enabled")
endif()
if(WITH_WRITE_FRAME)
  add_definitions(-DWRITE_FRAME)
  message(STATUS "Write frame enabled")
endif()

set(TritonClient_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonCommon")


find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)
find_package(CURL REQUIRED)
find_package(RapidJSON REQUIRED)
###
find_package(hiredis REQUIRED)
find_package(PkgConfig REQUIRED)

# Set the source files
set(SOURCES
    ${PROJECT_SOURCE_DIR}/client.cpp
    ${PROJECT_SOURCE_DIR}/src/tasks/tracker/EventQueue.cpp
    ${PROJECT_SOURCE_DIR}/src/tasks/tracker/CentroidTracker.cpp
    ${PROJECT_SOURCE_DIR}/src/tasks/tracker/
    ${PROJECT_SOURCE_DIR}/src/tasks/tracker/Utils.cpp
    ${PROJECT_SOURCE_DIR}/src/tasks/tracker/poligonTrackerManager.cpp
    # Add any additional source files here
)

set(DETECTOR_SOURCES ${PROJECT_SOURCE_DIR} src/tasks/object_detection/src/Yolo.cpp src/tasks/object_detection/src/YoloNas.cpp)

# Set the source files
set(TRITON_SOURCES ${PROJECT_SOURCE_DIR}/src/triton/Triton.cpp)

# Add an executable target
add_executable(${PROJECT_NAME} ${SOURCES} ${DETECTOR_SOURCES} ${TRITON_SOURCES})

# Set include directories for the target
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    $ENV{TritonClientBuild_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/tasks/object_detection/include
    ${PROJECT_SOURCE_DIR}/src/tasks/object_detection/src
    ${PROJECT_SOURCE_DIR}/src/tasks/classification/include
    ${PROJECT_SOURCE_DIR}/src/tasks/classification/src    
    ${PROJECT_SOURCE_DIR}/src/triton
    ${PROJECT_SOURCE_DIR}/lib/redox/include
    ${PROJECT_SOURCE_DIR}/src/tasks/tracker
    ${hiredis_INCLUDE_DIRS}  # Add hiredis include directory
   
    # Add any additional include directories here
)

# Set link directories for the target
target_link_directories(${PROJECT_NAME}
    PRIVATE
    $ENV{TritonClientBuild_DIR}/lib
    ${hiredis_LIBRARY_DIRS}  # Add hiredis library directory
    ${LIBEV_LIBRARY_DIRS}    # Add libev library directory
    ${PROJECT_SOURCE_DIR}/lib/redox/build
    /usr/lib/x86_64-linux-gnu
    # Add any additional link directories here
)

# Link libraries for the target
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    grpcclient
    httpclient
    ${OpenCV_LIBS}
    CURL::libcurl
    redox  # Link Redox library
    ${hiredis_LIBRARIES}  # Link hiredis library
    ev 
    jsoncpp
    tbb
    boost_system
    boost_filesystem

)
