# CMakeLists.txt for MatMulOptimized Operator
# Optimized MatMul operator for Ascend AI Algorithm Challenge

cmake_minimum_required(VERSION 3.16)
project(matmul_optimized C CXX)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../catlass/include
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
    op_kernel/matmul_optimized.cpp
    PROPERTIES
    LANGUAGE ASCEND
)

add_library(matmul_optimized_kernel SHARED
    op_kernel/matmul_optimized.cpp
)

target_link_libraries(matmul_optimized_kernel
    acl_cblas
    ascendcl
)

# ===================================================================
# Operator Host (Tiling)
# ===================================================================

add_library(matmul_optimized_host SHARED
    op_host/matmul_optimized.cpp
)

target_link_libraries(matmul_optimized_host
    ascendcl
    register
    ge_runner
)

target_compile_options(matmul_optimized_host PRIVATE
    -fPIC
    -O2
)

# ===================================================================
# Test/Benchmark
# ===================================================================

add_executable(benchmark_matmul
    test/benchmark_matmul.cpp
)

target_link_libraries(benchmark_matmul
    ascendcl
    acl_nn
    slog
)

target_compile_options(benchmark_matmul PRIVATE
    -O2
    -Wall
)

# ===================================================================
# Installation
# ===================================================================

install(TARGETS matmul_optimized_kernel matmul_optimized_host benchmark_matmul
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
