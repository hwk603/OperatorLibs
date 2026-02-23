# CMakeLists.txt for LeakyReluOptimized Operator
# Optimized LeakyReLU operator for Ascend AI Algorithm Challenge

cmake_minimum_required(VERSION 3.16)
project(leaky_relu_optimized C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CANN packages
find_package(Ascend REQUIRED)
find_package(slog REQUIRED)
find_package(adump REQUIRED)

# Include directories
include_directories(
    ${CANN_HOME}/include
    ${CANN_HOME}/include/acl
    ${CANN_HOME}/include/aclnn
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

# Link directories
link_directories(
    ${CANN_HOME}/lib64
)

# ===================================================================
# Operator Kernel
# ===================================================================

set_source_files_properties(
    op_kernel/leaky_relu_optimized.cpp
    PROPERTIES
    LANGUAGE ASCEND
)

add_library(leaky_relu_optimized_kernel SHARED
    op_kernel/leaky_relu_optimized.cpp
)

target_link_libraries(leaky_relu_optimized_kernel
    ascendcl
)

# ===================================================================
# Operator Host (Tiling)
# ===================================================================

add_library(leaky_relu_optimized_host SHARED
    op_host/leaky_relu_optimized.cpp
)

target_link_libraries(leaky_relu_optimized_host
    ascendcl
    register
    ge_runner
)

target_compile_options(leaky_relu_optimized_host PRIVATE
    -fPIC
    -O2
)

# ===================================================================
# Test/Benchmark
# ===================================================================

add_executable(benchmark_leaky_relu
    test/benchmark_leaky_relu.cpp
)

target_link_libraries(benchmark_leaky_relu
    ascendcl
    acl_nn
    slog
)

target_compile_options(benchmark_leaky_relu PRIVATE
    -O2
    -Wall
)

# ===================================================================
# Installation
# ===================================================================

install(TARGETS leaky_relu_optimized_kernel leaky_relu_optimized_host benchmark_leaky_relu
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
