#!/bin/bash

# Test script for MatMulOptimized operator
# Automated build and test execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_ID=${1:-0}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo "================================================"
echo "  MatMul Optimized - Automated Test"
echo "================================================"
echo "Device ID: $DEVICE_ID"
echo "================================================"

# Check CANN environment
if [ -z "${ASCEND_HOME_PATH}" ] && [ -z "${ASCEND_INSTALL_PATH}" ]; then
    log_error "CANN environment not set!"
    echo "Please run: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi

log_info "Step 1: Building operator..."
cd "${SCRIPT_DIR}"
chmod +x build.sh
./build.sh -v Ascend910B

if [ ! -d "output/bin" ]; then
    log_error "Build failed - output directory not found"
    exit 1
fi

log_info "Step 2: Checking NPU device..."
npu-smi info || log_warn "Could not query NPU info"

log_info "Step 3: Running benchmark..."
cd output/bin

if [ ! -f "benchmark_matmul" ]; then
    log_error "benchmark_matmul executable not found"
    exit 1
fi

./benchmark_matmul $DEVICE_ID

log_info "================================================"
log_info "Test completed successfully!"
log_info "================================================"
