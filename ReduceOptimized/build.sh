#!/bin/bash
#
# build.sh - Build script for optimized Reduce operator
# Ascend AI Algorithm Challenge
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default values
SOC_VERSION="Ascend910B"
BUILD_TYPE="Release"

# Parse command line arguments
SHORT=v:t:,h
LONG=soc-version:,build-type:,help
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
        -v | --soc-version)
            SOC_VERSION="$2"
            shift 2
            ;;
        -t | --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -h | --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --soc-version VERSION    Target SOC version (default: Ascend910B)"
            echo "  -t, --build-type TYPE       Build type: Debug/Release (default: Release)"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Build with default settings"
            echo "  $0 -v Ascend910B            # Build for Ascend910B"
            echo "  $0 -v Ascend310P3           # Build for Ascend310P3"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Error: Unexpected option: $1"
            exit 1
            ;;
    esac
done

# Detect CANN installation path
if [ -n "$ASCEND_HOME_PATH" ]; then
    ASCEND_CANN_PATH="$ASCEND_HOME_PATH"
elif [ -n "$ASCEND_INSTALL_PATH" ]; then
    ASCEND_CANN_PATH="$ASCEND_INSTALL_PATH"
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    ASCEND_CANN_PATH="$HOME/Ascend/ascend-toolkit/latest"
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    ASCEND_CANN_PATH="/usr/local/Ascend/ascend-toolkit/latest"
else
    echo "Error: Cannot find CANN installation. Please set ASCEND_HOME_PATH environment variable."
    exit 1
fi

echo "=============================================="
echo "Reduce Operator - Optimized Build"
echo "=============================================="
echo "Target SOC:         ${SOC_VERSION}"
echo "Build Type:         ${BUILD_TYPE}"
echo "CANN Path:          ${ASCEND_CANN_PATH}"
echo "=============================================="

# Source CANN environment
if [ -f "${ASCEND_CANN_PATH}/bin/setenv.bash" ]; then
    source "${ASCEND_CANN_PATH}/bin/setenv.bash"
else
    echo "Error: Cannot find setenv.bash in ${ASCEND_CANN_PATH}/bin/"
    exit 1
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf build out

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PATH}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

# Build
echo "Building..."
cmake --build . -j$(nproc)

# Install
echo "Installing..."
cmake --install .

echo "=============================================="
echo "Build completed successfully!"
echo "Output directory: ${SCRIPT_DIR}/out"
echo "=============================================="
