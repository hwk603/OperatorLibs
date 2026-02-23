/**
 * @file broadcast_optimized.cpp
 * Optimized Broadcast Operator for Ascend AI Algorithm Challenge
 *
 * Broadcast implements element-wise operations with automatic tensor broadcasting:
 * - Small tensor is "broadcast" to match shape of larger tensor
 * - Supports 1D->ND and ND->ND broadcasting
 *
 * Key Optimizations:
 * 1. Adaptive multi-strategy tiling based on broadcast factor and data size
 * 2. Multi-core parallel processing for large broadcasts
 * 3. Double buffering pipeline (BUFFER_NUM = 2)
 * 4. Vectorized BroadCast instruction optimization
 * 5. Memory alignment optimization
 * 6. Graded strategies for small vs large broadcast factors
 *
 * Copyright (C) 2025. Competition Submission.
 */

#include "kernel_operator.h"
#include "broadcast_optimized_tiling.h"

// ===================================================================
// Tiling Strategy Keys
// ===================================================================

#define BROADCAST_TILING_SMALL_FACTOR 1   // Small broadcast factor (<8): optimized replication
#define BROADCAST_TILING_MEDIUM_FACTOR 2  // Medium factor (8-64): multi-core with tiles
#define BROADCAST_TILING_LARGE_FACTOR 3   // Large factor (>64): max throughput mode

// ===================================================================
// Constants Optimized for Ascend910B
// ===================================================================

constexpr uint32_t BUFFER_COUNT = 2;  // Double buffering for pipeline
constexpr uint32_t MAX_TILE_LENGTH = 16384;  // Max elements per tile

// Broadcast factor thresholds
constexpr uint32_t SMALL_FACTOR_THRESHOLD = 8;
constexpr uint32_t MEDIUM_FACTOR_THRESHOLD = 64;

// ===================================================================
// Optimized KernelBroadcast Class
// ===================================================================

template<typename DTYPE_X, typename DTYPE_Y>
class KernelBroadcastOptimized {
public:
    __aicore__ inline KernelBroadcastOptimized() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t broadcastDim, uint32_t broadcastAxis,
                                uint32_t broadcastFactor, uint32_t baseLength,
                                uint32_t coreId, uint32_t coreNum)
    {
        this->totalLength = totalLength;
        this->broadcastDim = broadcastDim;
        this->broadcastAxis = broadcastAxis;
        this->broadcastFactor = broadcastFactor;
        this->baseLength = baseLength;
        this->coreId = coreId;
        this->coreNum = coreNum;

        // Calculate per-core data split for multi-core mode
        uint32_t outputElements = totalLength * broadcastFactor;
        if (coreNum > 1 && outputElements > 65536) {  // Only use multi-core for large outputs
            uint32_t elementsPerCore = (outputElements + coreNum - 1) / coreNum;
            uint32_t startIdx = coreId * elementsPerCore;
            uint32_t endIdx = std::min(startIdx + elementsPerCore, outputElements);
            localOutputLength = endIdx - startIdx;
            localInputOffset = startIdx / broadcastFactor;

            // Adjust for core alignment
            if (localInputOffset >= totalLength) {
                localInputOffset = totalLength - 1;
            }

            xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + localInputOffset,
                               std::min(totalLength - localInputOffset, baseLength));
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + startIdx, localOutputLength);
        } else {
            localOutputLength = outputElements;
            localInputOffset = 0;
            xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, totalLength);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, localOutputLength);
        }

        // Initialize buffers with adaptive sizing
        uint32_t inputBufferElements = std::min(baseLength, static_cast<uint32_t>(8192));
        uint32_t outputBufferElements = std::min(inputBufferElements * broadcastFactor,
                                                  static_cast<uint32_t>(MAX_TILE_LENGTH));

        pipe.InitBuffer(inQueueX, BUFFER_COUNT, inputBufferElements * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_COUNT, outputBufferElements * sizeof(DTYPE_Y));

        // Temporary buffer for broadcast (if needed)
        if (broadcastFactor > 1) {
            pipe.InitBuffer(tmpBuffer, outputBufferElements * sizeof(DTYPE_Y));
        }

        currentInputTileLength = inputBufferElements;
        currentOutputTileLength = outputBufferElements;
    }

    template<uint32_t TilingKey>
    __aicore__ inline void Process()
    {
        if constexpr (TilingKey == BROADCAST_TILING_SMALL_FACTOR) {
            ProcessSmallFactor();
        } else if constexpr (TilingKey == BROADCAST_TILING_MEDIUM_FACTOR) {
            ProcessMediumFactor();
        } else if constexpr (TilingKey == BROADCAST_TILING_LARGE_FACTOR) {
            ProcessLargeFactor();
        }
    }

private:
    // ===================================================================
    // Strategy 1: Small Broadcast Factor (<8)
    // Optimized for minimal replication overhead
    // ===================================================================

    __aicore__ inline void ProcessSmallFactor()
    {
        uint32_t inputProcessed = 0;
        uint32_t outputProcessed = 0;

        while (outputProcessed < localOutputLength) {
            CopyIn(inputProcessed);
            ComputeSmallFactor();
            CopyOut(outputProcessed);

            inputProcessed += currentInputTileLength;
            outputProcessed += currentOutputTileLength;

            // Adjust for last tile
            uint32_t remainingInput = localInputOffset + currentInputTileLength;
            if (remainingInput >= totalLength) {
                currentInputTileLength = std::min(baseLength, totalLength - localInputOffset);
                currentOutputTileLength = currentInputTileLength * broadcastFactor;
            }
        }
    }

    __aicore__ inline void ComputeSmallFactor()
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // Use optimized broadCast for small factors
        if (broadcastDim == 1) {
            // 1D to ND broadcast
            const uint32_t srcShape[] = {currentInputTileLength};
            const uint32_t dstShape[] = {currentOutputTileLength};
            AscendC::BroadCast<DTYPE_X, 1, 0>(yLocal, xLocal, dstShape, srcShape);
        } else {
            // ND to ND broadcast
            if (broadcastAxis == 0) {
                const uint32_t srcShape[] = {baseLength, currentInputTileLength / baseLength};
                const uint32_t dstShape[] = {baseLength * broadcastFactor,
                                             currentInputTileLength / baseLength};
                AscendC::BroadCast<DTYPE_X, 2, 0>(yLocal, xLocal, dstShape, srcShape);
            } else {
                const uint32_t srcShape[] = {baseLength, currentInputTileLength / baseLength};
                const uint32_t dstShape[] = {baseLength,
                                             (currentInputTileLength / baseLength) * broadcastFactor};
                AscendC::BroadCast<DTYPE_X, 2, 1>(yLocal, xLocal, dstShape, srcShape);
            }
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // ===================================================================
    // Strategy 2: Medium Broadcast Factor (8-64)
    // Multi-core with optimized tile sizes
    // ===================================================================

    __aicore__ inline void ProcessMediumFactor()
    {
        uint32_t inputProcessed = 0;
        uint32_t outputProcessed = 0;

        while (outputProcessed < localOutputLength) {
            CopyIn(inputProcessed);
            ComputeMediumFactor();
            CopyOut(outputProcessed);

            inputProcessed += currentInputTileLength;
            outputProcessed += currentOutputTileLength;
        }
    }

    __aicore__ inline void ComputeMediumFactor()
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // Use temporary buffer for medium factors
        AscendC::LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

        if (broadcastDim == 1) {
            const uint32_t srcShape[] = {currentInputTileLength};
            const uint32_t dstShape[] = {currentOutputTileLength};
            AscendC::BroadCast<DTYPE_X, 1, 0>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
        } else {
            if (broadcastAxis == 0) {
                const uint32_t srcShape[] = {baseLength, currentInputTileLength / baseLength};
                const uint32_t dstShape[] = {baseLength * broadcastFactor,
                                             currentInputTileLength / baseLength};
                AscendC::BroadCast<DTYPE_X, 2, 0>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
            } else {
                const uint32_t srcShape[] = {baseLength, currentInputTileLength / baseLength};
                const uint32_t dstShape[] = {baseLength,
                                             (currentInputTileLength / baseLength) * broadcastFactor};
                AscendC::BroadCast<DTYPE_X, 2, 1>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
            }
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // ===================================================================
    // Strategy 3: Large Broadcast Factor (>64)
    // Maximum throughput with full pipeline
    // ===================================================================

    __aicore__ inline void ProcessLargeFactor()
    {
        // Fully pipelined processing for large broadcast factors
        int32_t loopCount = (localOutputLength + currentOutputTileLength - 1) / currentOutputTileLength;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i * currentInputTileLength);
            ComputeLargeFactor();
            CopyOut(i * currentOutputTileLength);
        }
    }

    __aicore__ inline void ComputeLargeFactor()
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // Use temporary buffer for large factors (may need more workspace)
        AscendC::LocalTensor<uint8_t> tmpTensor = tmpBuffer.Get<uint8_t>();

        // Vectorized broadcast for maximum throughput
        if (broadcastDim == 1) {
            const uint32_t srcShape[] = {currentInputTileLength};
            const uint32_t dstShape[] = {currentOutputTileLength};
            AscendC::BroadCast<DTYPE_X, 1, 0>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
        } else {
            if (broadcastAxis == 0) {
                const uint32_t srcShape[] = {baseLength, currentInputTileLength / baseLength};
                const uint32_t dstShape[] = {baseLength * broadcastFactor,
                                             currentInputTileLength / baseLength};
                AscendC::BroadCast<DTYPE_X, 2, 0>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
            } else {
                const uint32_t srcShape[] = {baseLength, currentInputTileLength / baseLength};
                const uint32_t dstShape[] = {baseLength,
                                             (currentInputTileLength / baseLength) * broadcastFactor};
                AscendC::BroadCast<DTYPE_X, 2, 1>(yLocal, xLocal, dstShape, srcShape, tmpTensor);
            }
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // ===================================================================
    // Data Transfer Functions with Pipeline Optimization
    // ===================================================================

    __aicore__ inline void CopyIn(uint32_t inputOffset)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();

        // Calculate actual input length considering boundaries
        uint32_t actualInputLength = std::min(currentInputTileLength,
                                                totalLength - localInputOffset);

        if (actualInputLength > 0) {
            AscendC::DataCopy(xLocal, xGm[inputOffset - localInputOffset], actualInputLength);
        }

        inQueueX.EnQue<DTYPE_X>(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t outputOffset)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        // Calculate actual output length
        uint32_t actualOutputLength = std::min(currentOutputTileLength,
                                                localOutputLength - outputOffset);

        if (actualOutputLength > 0) {
            AscendC::DataCopy(yGm[outputOffset], yLocal, actualOutputLength);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_COUNT> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_COUNT> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;

    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t totalLength;
    uint32_t localOutputLength;
    uint32_t localInputOffset;
    uint32_t broadcastDim;
    uint32_t broadcastAxis;
    uint32_t broadcastFactor;
    uint32_t baseLength;
    uint32_t coreId;
    uint32_t coreNum;

    uint32_t currentInputTileLength;
    uint32_t currentOutputTileLength;
};

// ===================================================================
// Kernel Entry Point with Type Selection
// ===================================================================

extern "C" __global__ __aicore__ void broadcast_optimized(GM_ADDR x, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    // Extract tiling parameters
    uint32_t totalLength = tiling_data.totalLength;
    uint32_t broadcastDim = tiling_data.broadcastDim;
    uint32_t broadcastAxis = tiling_data.broadcastAxis;
    uint32_t broadcastFactor = tiling_data.broadcastFactor;
    uint32_t baseLength = tiling_data.baseLength;
    uint32_t dataType = tiling_data.dataType;  // 0: float, 1: float16

    uint32_t coreId = GetBlockIdx();
    uint32_t coreNum = GetBlockNum();

    // Determine tiling strategy based on broadcast factor
    uint32_t tilingKey;
    if (broadcastFactor <= SMALL_FACTOR_THRESHOLD) {
        tilingKey = BROADCAST_TILING_SMALL_FACTOR;
    } else if (broadcastFactor <= MEDIUM_FACTOR_THRESHOLD) {
        tilingKey = BROADCAST_TILING_MEDIUM_FACTOR;
    } else {
        tilingKey = BROADCAST_TILING_LARGE_FACTOR;
    }

    // Execute with appropriate data types
    if (dataType == 1) {  // float16
        KernelBroadcastOptimized<half, half> op;
        op.Init(x, y, totalLength, broadcastDim, broadcastAxis, broadcastFactor,
               baseLength, coreId, coreNum);

        if (tilingKey == BROADCAST_TILING_SMALL_FACTOR) {
            op.Process<BROADCAST_TILING_SMALL_FACTOR>();
        } else if (tilingKey == BROADCAST_TILING_MEDIUM_FACTOR) {
            op.Process<BROADCAST_TILING_MEDIUM_FACTOR>();
        } else {
            op.Process<BROADCAST_TILING_LARGE_FACTOR>();
        }
    } else {  // float
        KernelBroadcastOptimized<float, float> op;
        op.Init(x, y, totalLength, broadcastDim, broadcastAxis, broadcastFactor,
               baseLength, coreId, coreNum);

        if (tilingKey == BROADCAST_TILING_SMALL_FACTOR) {
            op.Process<BROADCAST_TILING_SMALL_FACTOR>();
        } else if (tilingKey == BROADCAST_TILING_MEDIUM_FACTOR) {
            op.Process<BROADCAST_TILING_MEDIUM_FACTOR>();
        } else {
            op.Process<BROADCAST_TILING_LARGE_FACTOR>();
        }
    }
}

#ifndef ASCENDC_CPU_DEBUG
void broadcast_optimized_do(uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *x, uint8_t *y, uint8_t *workspace, uint8_t *tiling)
{
    broadcast_optimized<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
