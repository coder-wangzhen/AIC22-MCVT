cmake_minimum_required(VERSION 3.10)

project(city_tracker)
set(CMAKE_BUILD_TYPE "Release")
set(MAKESHARED FALSE CACHE BOOL "Generate py extension")
if(UNIX)
        set (CMAKE_CXX_STANDARD 11)
        include_directories(/usr/local/cuda/include)
        link_directories(/usr/local/cuda/lib64)
        link_directories(/usr/local/cuda/lib64/stubs)
endif(UNIX)
add_compile_options(-std=c++14 -fpermissive -fPIC)

enable_language(CUDA)

#opencv 
set(OpenCV_DIR /usr/local/opencv-4.5.3/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)
MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/rapidjson-1.1.0/include)

set (optical_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/OpticalFlow/inc)
set (optical_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/OpticalFlow/src)

set (Munkres_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/MunkresAssignment/inc)
set (Munkres_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/MunkresAssignment/src)

set (MOG_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/MOG/src)
set (MOG_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/MOG/inc)

set (KalmanFilter_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/KalmanFilter/inc)
set (KalmanFilter_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/KalmanFilter/src)



set (descriptor_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/DeepAppearanceDescriptor/inc)
set (descriptor_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/DeepAppearanceDescriptor/src)

set(common_inc_dir ${CMAKE_CURRENT_SOURCE_DIR}/common)

message(STATUS "Searching for optical_inc_dir ${optical_inc_dir}")
message(STATUS "Searching for Munkres_inc_dir ${Munkres_inc_dir}")
message(STATUS "Searching for MOG_inc_dir ${MOG_inc_dir}")
message(STATUS "Searching for KalmanFilter_inc_dir ${KalmanFilter_inc_dir}")
message(STATUS "Searching for eco_inc_dir ${eco_inc_dir}")
message(STATUS "Searching for descriptor_inc_dir ${descriptor_inc_dir}")
message(STATUS "Searching for common_inc_dir ${common_inc_dir}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eco/inc)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/eco/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd/eigen3)
include_directories(${optical_inc_dir})
include_directories(${Munkres_inc_dir})
include_directories(${MOG_inc_dir})
include_directories(${KalmanFilter_inc_dir})

include_directories(${descriptor_inc_dir})
include_directories(${common_inc_dir})
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(${optical_inc_dir})
add_subdirectory(${optical_src_dir})
add_subdirectory(${Munkres_inc_dir})
add_subdirectory(${Munkres_src_dir})
add_subdirectory(${MOG_inc_dir})
add_subdirectory(${MOG_src_dir})
add_subdirectory(${KalmanFilter_inc_dir})
add_subdirectory(${KalmanFilter_src_dir})

add_subdirectory(${descriptor_inc_dir})
add_subdirectory(${descriptor_src_dir})
add_subdirectory(${common_inc_dir})
#
add_executable(${PROJECT_NAME} ${DeepAppearanceDescriptor_SOURCES} ${KalmanFilter_SOURCES}  ${MunkresAssignment_SOURCES} ${OPTICAL_SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} pthread)
target_link_libraries(${PROJECT_NAME} eco)


#docker run -it --net=host -e DISPLAY -e QT_X11_NO_MITSHM=1 --privileged=true --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all  --name city_challenge -v /etc/localtime:/etc/localtime -v $(pwd):/workspace -e OMP_NUM_THREADS=20 10.252.97.209:80/aiteam/boe_engine:p2 /bin/bash

#echo "/tmp/core.%t.%e.%p" | sudo tee /proc/sys/kernel/core_pattern
