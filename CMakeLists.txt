cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin/nvcc.exe")
project(HelloCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(HelloCuda src/main.cu)

set_target_properties(
        HelloCuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)