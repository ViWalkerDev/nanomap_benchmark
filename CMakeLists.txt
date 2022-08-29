cmake_minimum_required(VERSION 3.15)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Release)
set (CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set (CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
project(nanomap_benchmark)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  # Allow the user to provide CMAKE_CUDA_ARCHITECTURES
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
endif()
enable_language(CUDA)

set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
  set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
endif()

set(CMAKE_CUDA_FLAGS "${NANOVDB_CUDA_EXTENDED_LAMBDA} -use_fast_math -lineinfo ${CMAKE_CUDA_FLAGS}")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set (CUDA_INCLUDE_DIR "/usr/local/cuda/include")

find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDA REQUIRED)


set(Blosc_ROOT "$ENV{HOME}/github/third-party/lib/blosc")
set(OpenVDB_LIBRARYDIR "$ENV{HOME}/github/third-party/lib/openvdb/lib")
set(CMAKE_MODULE_PATH "$ENV{HOME}/github/third-party/lib/openvdb/lib/cmake/OpenVDB")
find_package(OpenVDB REQUIRED)
set(nanomap_DIR "$ENV{HOME}/github/first-party/lib/nanomap/lib/cmake/nanomap")
find_package(nanomap REQUIRED)
set(NANOMAP_INCLUDE_DIR "$ENV{HOME}/github/first-party/lib/nanomap/include")
set(octomap_DIR "$ENV{HOME}/github/third-party/lib/octomap/share/octomap")
list(APPEND CMAKE_MODULE_PATH "$ENV{HOME}/github/third-party/lib/octomap/share/octomap")
find_package(octomap REQUIRED)
set(dependencies "ament_index_cpp" "geometry_msgs" "rclcpp" "rosbag2_cpp" "tf2" "tf2_ros" "tf2_msgs" "std_msgs" "pcl_ros" "pcl_conversions" "octomap_ros")

add_executable(benchmark src/benchmark.cpp)

target_link_libraries(benchmark
    ${OpenVDB_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OCTOMAP_LIBRARIES}
    -ltbb
    -lHalf
    $ENV{HOME}/github/first-party/lib/nanomap/lib/libnanomap.a
    )

target_include_directories(benchmark
  PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      ${NANOMAP_INCLUDE_DIR}
      ${OpenVDB_INCLUDE_DIR}
      ${CUDA_INCLUDE_DIR}
      ${OCTOMAP_LIBRARIES}
      )
ament_target_dependencies(benchmark ${dependencies})

install(TARGETS benchmark
    DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  )

ament_package()
