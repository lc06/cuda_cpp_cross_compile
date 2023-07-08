cmake_minimum_required(VERSION 3.1...3.15)

project(vectorAdd)

enable_language(CUDA)

file(GLOB_RECURSE CURRENT_HEADERS *.h *.cpp *.cuh)
file(GLOB CURRENT_SOURCES *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

add_library(vectorAdd STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})