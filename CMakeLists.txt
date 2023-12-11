cmake_minimum_required(VERSION 3.0.0)

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(HailoRT EXACT REQUIRED)
find_package(OpenCV REQUIRED)
find_package (gflags REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wall -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wall -fopenmp")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g")

include_directories (${gflags_INCLUDE_DIR} ./include)

add_executable(hailort-yoloXP src/hailort_yoloXP.cpp src/common.cpp src/yoloXP.cpp src/utils.cpp src/config_parser.cpp src/colormap.cpp)
target_include_directories(hailort-yoloXP PRIVATE  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hailort-yoloXP PRIVATE HailoRT::libhailort Threads::Threads ${OpenCV_LIBS} gflags boost_system  boost_filesystem "stdc++fs" "stdc++")

if(WIN32)
    target_compile_options(hailort-yoloXP PRIVATE
        /DWIN32_LEAN_AND_MEAN
        /DNOMINMAX                  
        /wd4201 /wd4251
    )
endif()

set_target_properties(hailort-yoloXP PROPERTIES CXX_STANDARD 14)
