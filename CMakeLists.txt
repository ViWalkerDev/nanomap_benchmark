cmake_minimum_required(VERSION 3.15)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set (CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set (CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
Set(CUDA_SEPARABLE_COMPILATION ON)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
project(nanomap_benchmark)
set(nanomap_VERSION 0.9)


if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  # Allow the user to provide CMAKE_CUDA_ARCHITECTURES
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 53)
  endif()
endif()
enable_language(CUDA)

set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
  set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
endif()

set(CMAKE_CUDA_FLAGS "${NANOVDB_CUDA_EXTENDED_LAMBDA} -use_fast_math -lineinfo -rdc=true ${CMAKE_CUDA_FLAGS}")


set (CUDA_INCLUDE_DIR "/usr/local/cuda/include")

set(EIGEN3_INCLUDE_DIR "$ENV{HOME}/github/third-party/lib/eigen3/include/")
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
find_package(Eigen3 REQUIRED)
set(TBB_ROOT "$ENV{HOME}/github/third-party/lib/tbb/lib/cmake/tbb")
find_package(TBB REQUIRED)
find_package(CUDA REQUIRED)

## FIND AND ADD OVDB PACKAGE
set(Blosc_ROOT "$ENV{HOME}/github/third-party/lib/blosc")
set(OpenVDB_LIBRARYDIR "$ENV{HOME}/github/third-party/lib/openvdb/lib")
list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/github/third-party/lib/openvdb/lib/cmake/OpenVDB")
find_package(OpenVDB REQUIRED)

set (NanoVDB_INCLUDE_DIR "$ENV{HOME}/github/third-party/lib/openvdb/include")


find_package(OpenCV REQUIRED)
list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/github/third-party/lib/octomap/share/octomap")
find_package(octomap REQUIRED)

add_library(nanomap_benchmark
    INTERFACE
    )
target_link_libraries(nanomap_benchmark
  INTERFACE
    ${OpenVDB_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OCTOMAP_LIBRARIES}
    $ENV{HOME}/github/first-party/lib/nanomap/lib/libnanomap.a
    -ltbb
    -lHalf
    )
target_include_directories(nanomap_benchmark
  INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      ${EIGEN3_INCLUDE_DIR}
      ${OpenVDB_INCLUDE_DIR}
      ${NanoVDB_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIR}
      ${OCTOMAP_INCLUDE_DIRS}
      $ENV{HOME}/github/first-party/lib/nanomap/include
      )

find_package(catkin REQUIRED COMPONENTS
          rosbag
          rosfmt
          backward_ros
          sensor_msgs
          octomap_ros
          tf
          pcl_ros
          pcl_conversions
          tf2_ros
          tf_conversions
      )


catkin_package()


add_executable(benchmark src/benchmark.cpp)

target_link_libraries(benchmark PUBLIC nanomap_benchmark ${catkin_LIBRARIES})

set_property(TARGET benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(benchmark PUBLIC ${catkin_INCLUDE_DIRS})
