/**
 * @file matmul_optimized_tiling.h
 * Tiling data structure for optimized MatMul operator
 *
 * Copyright (C) 2025. Competition Submission.
 */

#ifndef MATMUL_OPTIMIZED_TILING_H
#define MATMUL_OPTIMIZED_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

/**
 * @brief Tiling data structure for MatMul optimization
 *
 * This structure contains all the parameters needed for
 * adaptive tiling strategy selection on the kernel side.
 */
struct MatmulOptimizedTilingData {
    // Matrix dimensions
    uint32_t m;              // M dimension (rows of A)
    uint32_t n;              // N dimension (columns of B)
    uint32_t k;              // K dimension (columns of A / rows of B)

    // Padding flags
    bool needPaddingA;       // Whether matrix A needs padding
    bool needPaddingB;       // Whether matrix B needs padding

    // Memory information
    uint64_t localMemSize;   // UB (Unified Buffer) size

    // Standard cube tiling data
    AscendC::tiling::TCubeTiling cubeTilingData;
};

#endif  // MATMUL_OPTIMIZED_TILING_H
