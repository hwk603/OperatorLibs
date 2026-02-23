/**
 * @file broadcast_optimized_tiling.h
 * Tiling data structure for optimized Broadcast operator
 *
 * Copyright (C) 2025. Competition Submission.
 */

#ifndef BROADCAST_OPTIMIZED_TILING_H
#define BROADCAST_OPTIMIZED_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

/**
 * @brief Tiling data structure for Broadcast optimization
 */
struct BroadcastOptimizedTilingData {
    // Input/output dimensions
    uint32_t totalLength;      // Input tensor length
    uint32_t broadcastDim;     // Broadcast dimension (1 or 2)
    uint32_t broadcastAxis;    // Axis to broadcast along (0 or 1)
    uint32_t broadcastFactor;  // Multiplication factor for broadcasting
    uint32_t baseLength;       // Base length before broadcast
    uint32_t dataType;         // 0: float32, 1: float16
    uint32_t coreId;           // Current core ID
    uint32_t coreNum;          // Total number of cores
};

#endif  // BROADCAST_OPTIMIZED_TILING_H
