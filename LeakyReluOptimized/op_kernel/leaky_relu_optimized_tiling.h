/**
 * @file leaky_relu_optimized_tiling.h
 * Tiling data structure for optimized LeakyReLU operator
 *
 * Copyright (C) 2025. Competition Submission.
 */

#ifndef LEAKY_RELU_OPTIMIZED_TILING_H
#define LEAKY_RELU_OPTIMIZED_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

/**
 * @brief Tiling data structure for LeakyReLU optimization
 *
 * This structure contains all the parameters needed for
 * adaptive tiling strategy selection on the kernel side.
 */
struct LeakyReluOptimizedTilingData {
    // Input/output dimensions
    uint32_t totalLength;    // Total number of elements

    // LeakyReLU parameters
    float negativeSlope;     // Alpha value for negative inputs

    // Data type indicator
    uint32_t dataType;       // 0: float32, 1: float16

    // Multi-core configuration
    uint32_t coreId;         // Current core ID
    uint32_t coreNum;        // Total number of cores
};

#endif  // LEAKY_RELU_OPTIMIZED_TILING_H
