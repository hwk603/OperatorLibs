#!/bin/bash

# Build script for MatMulOptimized Operator
# Optimized MatMul operator for Ascend AI Algorithm Challenge

set -e

# ===================================================================
# Configuration
# ===================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/output"
SOC_VERSION="Ascend910B"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ===================================================================
# Functions
# ===================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v, --soc-version VERSION   Set SOC version (default: Ascend910B)"
    echo "  -c, --clean                 Clean build directory before building"
    echo "  -d, --debug                 Build in debug mode"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Supported SOC versions:"
    echo "  Ascend910A, Ascend910B, Ascend310P, Ascend310B, etc."
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_env() {
    if [ -z "${ASCEND_HOME_PATH}" ] && [ -z "${ASCEND_INSTALL_PATH}" ]; then
        log_error "CANN environment not set. Please source the CANN environment first."
        echo "Example: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        exit 1
    fi

    if [ -n "${ASCEND_HOME_PATH}" ]; then
        export CANN_HOME="${ASCEND_HOME_PATH}"
    else
        export CANN_HOME="${ASCEND_INSTALL_PATH}"
    fi

    log_info "CANN_HOME: ${CANN_HOME}"
}

clean_build() {
    log_info "Cleaning build directories..."
    rm -rf "${BUILD_DIR}"
    rm -rf "${OUTPUT_DIR}"
    log_info "Clean complete."
}

# ===================================================================
# Parse Arguments
# ===================================================================

CLEAN_BUILD=false
DEBUG_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--soc-version)
            SOC_VERSION="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -d|--debug)
            DEBUG_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# ===================================================================
# Main Build Process
# ===================================================================

log_info "================================================"
log_info "MatMul Optimized Operator Build Script"
log_info "================================================"
log_info "SOC Version: ${SOC_VERSION}"
log_info "Debug Build: ${DEBUG_BUILD}"
log_info "================================================"

# Check environment
check_env

# Clean if requested
if [ "${CLEAN_BUILD}" = true ]; then
    clean_build
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"/{bin,lib}

log_info "Build directory: ${BUILD_DIR}"
log_info "Output directory: ${OUTPUT_DIR}"

# Configure CMake
cd "${BUILD_DIR}"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=$( [ "${DEBUG_BUILD}" = true ] && echo "Debug" || echo "Release" )
    -DSOC_VERSION="${SOC_VERSION}"
    -DCANN_PACKAGE_PATH="${CANN_HOME}"
    -DCMAKE_INSTALL_PREFIX="${OUTPUT_DIR}"
)

log_info "Configuring CMake..."
cmake "${CMAKE_ARGS[@]}" ..

# Build
log_info "Building..."
NUM_CORES=$(nproc)
make -j${NUM_CORES}

# Install
log_info "Installing..."
make install

# Copy operator JSON if exists
if [ -f "${SCRIPT_DIR}/op_host/MatmulOptimized.json" ]; then
    cp "${SCRIPT_DIR}/op_host/MatmulOptimized.json" "${OUTPUT_DIR}/lib/"
fi

log_info "================================================"
log_info "Build completed successfully!"
log_info "================================================"
log_info "Output files:"
log_info "  Libraries: ${OUTPUT_DIR}/lib/"
log_info "  Binaries:  ${OUTPUT_DIR}/bin/"
log_info "================================================"
log_info ""
log_info "To run benchmark:"
log_info "  cd ${OUTPUT_DIR}/bin"
log_info "  ./benchmark_matmul <device_id>"
log_info ""
