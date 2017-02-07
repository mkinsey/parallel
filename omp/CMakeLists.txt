cmake_minimum_required(VERSION 3.6)
project(omp)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCE_FILES
        linked.c
        #original-linked.c
        #mandel.c
        #pad-integrate.c
        #serial-integrate.c
        #sync-integrate.c
        #workflow-integrate.c
        )

add_executable(omp ${SOURCE_FILES})