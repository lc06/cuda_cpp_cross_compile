cmake_minimum_required(VERSION 3.1...3.15)

project(cudatest)
set(CXX_STANDARD_VERSION 17)
set(CMAKE_BUILD_TYPE Release)

enable_language(CUDA)
enable_language(RC)

include_directories(cuda)

# add_subdirectory(cuda)
# set (EXTRA_LIBS ${EXTRA_LIBS} vectorAdd)

add_executable(cudatest main.cpp cuda/vectorAdd.cu)
target_link_libraries(cudatest cuda cudart)