# CMake minimum required version
cmake_minimum_required(VERSION 3.11)

# Project name and version
project(conjgrad VERSION 1.0 LANGUAGES CUDA CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CUDA_SEPARABLE_COMPILATION ON)

# Enable CUDA language support in your project
enable_language(CUDA)

# Define an executable target
add_executable(conjgrad
               src/main.cpp
               src/utils.cpp
               src/serial.cpp
               src/parallel.cu)

# Link against cuBLAS
target_link_libraries(conjgrad cublas)

# Add support for device kernel debugging
target_compile_options(conjgrad PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--device-debug>)

# Optionally, set the GPU architecture for compilation
set_target_properties(conjgrad PROPERTIES CUDA_ARCHITECTURES 60)
