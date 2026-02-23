# CMakeLists.txt for Optimized Reduce Operator
# Ascend AI Algorithm Challenge Submission

cmake_minimum_required(VERSION 3.16.0)
project(ReduceOptimized LANGUAGES CXX)

# Set build type
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SOC version (target hardware)
set(SOC_VERSION "Ascend910B" CACHE STRING "System on Chip version")

# CANN package path
if(DEFINED ENV{USER} AND "$ENV{USER}" STREQUAL "root")
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
else()
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "$ENV{HOME}/Ascend/ascend-toolkit/latest")
endif()

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN installation path" FORCE)
else()
    set(ASCEND_CANN_PACKAGE_PATH "${DEFAULT_ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "ASCEND CANN installation path")
endif()

# Find AscendC CMake directory
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "AscendC CMake files not found. Please check CANN installation.")
endif()

message(STATUS "SOC_VERSION: ${SOC_VERSION}")
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "ASCENDC_CMAKE_DIR: ${ASCENDC_CMAKE_DIR}")

# Include AscendC CMake utilities
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# Add operator kernel library
ascendc_library(reduce_optimized_kernel STATIC
    op_kernel/reduce_optimized.cpp
)

# Set output directories
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "Install path" FORCE)

# Print build configuration
message(STATUS "===============================================")
message(STATUS "Reduce Operator - Optimized Build Configuration")
message(STATUS "===============================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target SOC: ${SOC_VERSION}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===============================================")
