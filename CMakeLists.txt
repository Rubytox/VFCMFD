cmake_minimum_required(VERSION 3.0)

# Notre projet est étiqueté hello
project(copyMoveCheck)

#find_package(OpenCV REQUIRED PATHS /home/puech/Documents/CHRISTOPHE/opencv/install NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED)

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost REQUIRED COMPONENTS log_setup log)
include_directories(${Boost_INCLUDE_DIRS})
MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

SET(GXX_DEBUG_FLAG "-g")
SET(GXX_11 "-Wall -pedantic -std=c++1z -Wno-attributes")

SET(CMAKE_CSS_FLAGS "${CMAKE_CXX_FLAGS} -O0")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Crée des variables avec les fichiers à compiler
set(SRCS
    src/main.cpp
    #src/surf.cpp
    src/dbscan.cpp
    src/line.cpp
    src/InterestPoint.cpp
    src/ClusteredLine.cpp
    src/copyMoveDetector.cpp
    src/InterestPoints.cpp)

set(HEADERS
    include/surf.hpp
    #include/line.hpp
    include/dbscan.hpp
    include/copyMoveDetector.hpp
    include/InterestPoint.hpp
    include/ClusteredLine.hpp
    include/InterestPoints.hpp include/DetectorOptions.hpp)

add_definitions(${GXX_DEBUG_FLAG})
add_definitions(${GXX_11})

add_definitions(${GXX_DEBUG_FLAG})
add_definitions(${GXX_11})

# On indique que l'on veut un exécutable "hello" compilé à partir des fichiers décrits par les variables SRCS et HEADERS
add_executable(copyMoveCheck ${SRCS} ${HEADERS})

target_link_libraries(copyMoveCheck ${OpenCV_LIBS})
target_link_libraries(copyMoveCheck Threads::Threads)
target_link_libraries(copyMoveCheck ${Boost_LIBRARIES})
