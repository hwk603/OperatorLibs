/**
 * @file reduce_optimized.cpp
 * Optimized Reduce Operator for Ascend AI Algorithm Challenge
 *
 * Key Optimizations:
 * 1. Multi-core parallel reduction with cross-core merge
 * 2. Optimized binary reduction with adaptive thresholds
 * 3. Pipeline optimization with double buffering
 * 4. Improved UB buffer utilization
 *
 * Copyright (C) 2025. Competition Submission.
 */

#include "kernel_operator.h"

// Tiling strategy keys
#define REDUCE_TILING_SMALL 1      // <= 256B: WholeReduceSum only
#define REDUCE_TILING_MEDIUM 2     // 256B-8KB: BlockReduceSum + WholeReduceSum
#define REDUCE_TILING_LARGE 3      // 8KB-64KB: Two-stage WholeReduceSum
#define REDUCE_TILING_BINARY 4     // 64KB-512KB: Binary reduction
#define REDUCE_TILING_MULTICORE 5  // >512KB: Multi-core parallel

// Constants optimized for Ascend910B
constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
constexpr uint32_t DEFAULT_REP_STRIDE = 8;
constexpr uint32_t REP_LEN = 256;
constexpr uint32_t BLK_LEN = 32;
constexpr uint32_t ONE_REPEAT_FLOAT_SIZE = REP_LEN / sizeof(float);  // 64 floats
constexpr uint32_t ONE_REPEAT_HALF_SIZE = REP_LEN / sizeof(uint16_t); // 128 halves
constexpr uint32_t BINARY_BOUNDARY = 16;  // Optimized binary split ratio

// Size thresholds (in elements)
constexpr uint32_t FLOAT_SMALL_THRESHOLD = 64;      // 256B
constexpr uint32_t FLOAT_MEDIUM_THRESHOLD = 2048;   // 8KB
constexpr uint32_t FLOAT_LARGE_THRESHOLD = 16384;   // 64KB
constexpr uint32_t FLOAT_BINARY_THRESHOLD = 131072; // 512KB

constexpr uint32_t HALF_SMALL_THRESHOLD = 128;      // 256B
constexpr uint32_t HALF_MEDIUM_THRESHOLD = 4096;    // 8KB
constexpr uint32_t HALF_LARGE_THRESHOLD = 32768;    // 64KB
constexpr uint32_t HALF_BINARY_THRESHOLD = 262144;  // 512KB

template<typename DTYPE>
class KernelReduceOptimized {
public:
    __aicore__ inline KernelReduceOptimized() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength,
                                uint32_t outLength, uint32_t coreId, uint32_t coreNum)
    {
        this->totalLength = totalLength;
        this->outLength = outLength;
        this->coreId = coreId;
        this->coreNum = coreNum;

        xGm.SetGlobalBuffer((__gm__ DTYPE *)x, totalLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE *)z, outLength);

        // Calculate per-core data split for multi-core mode
        if (coreNum > 1) {
            uint32_t elementsPerCore = (totalLength + coreNum - 1) / coreNum;
            uint32_t startIdx = coreId * elementsPerCore;
            uint32_t endIdx = std::min(startIdx + elementsPerCore, totalLength);
            localLength = endIdx - startIdx;
            localOffset = startIdx;

            pipe.InitBuffer(inQueueX, BUFFER_COUNT, localLength * sizeof(DTYPE));
            pipe.InitBuffer(outQueueZ, 1, outLength * sizeof(DTYPE));
        } else {
            localLength = totalLength;
            localOffset = 0;
            pipe.InitBuffer(inQueueX, BUFFER_COUNT, totalLength * sizeof(DTYPE));
            pipe.InitBuffer(outQueueZ, 1, outLength * sizeof(DTYPE));
        }

        // Initialize intermediate buffer for cross-core reduction
        if (coreNum > 1) {
            pipe.InitBuffer(reduceBuf, 1, coreNum * sizeof(DTYPE));
        }
    }

    template<size_t ComputeKey>
    __aicore__ inline void Compute()
    {
        if constexpr (ComputeKey == REDUCE_TILING_SMALL) {
            ComputeSmall();
        } else if constexpr (ComputeKey == REDUCE_TILING_MEDIUM) {
            ComputeMedium();
        } else if constexpr (ComputeKey == REDUCE_TILING_LARGE) {
            ComputeLarge();
        } else if constexpr (ComputeKey == REDUCE_TILING_BINARY) {
            ComputeBinary();
        } else if constexpr (ComputeKey == REDUCE_TILING_MULTICORE) {
            ComputeMultiCore();
        }
    }

    template<size_t ComputeKey>
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute<ComputeKey>();
        CopyOut();
    }

private:
    static constexpr uint32_t BUFFER_COUNT = 2;  // Double buffering

    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.AllocTensor<DTYPE>();
        if (coreNum > 1) {
            AscendC::DataCopy(xLocal, xGm[localOffset], localLength);
        } else {
            AscendC::DataCopy(xLocal, xGm, totalLength);
        }
        inQueueX.EnQue(xLocal);
    }

    // Small data: Single WholeReduceSum (fastest for <=256B)
    __aicore__ inline void ComputeSmall()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();

        uint32_t dataLength = (coreNum > 1) ? localLength : totalLength;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DTYPE>(0, dataLength);
        AscendC::WholeReduceSum<DTYPE>(zLocal, xLocal, AscendC::MASK_PLACEHOLDER, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // Medium data: BlockReduceSum + WholeReduceSum (optimal for 256B-8KB)
    __aicore__ inline void ComputeMedium()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();

        uint32_t dataLength = (coreNum > 1) ? localLength : totalLength;
        pipe.InitBuffer(calcBuf, dataLength * sizeof(DTYPE));
        AscendC::LocalTensor<DTYPE> tempTensor = calcBuf.Get<DTYPE>();

        constexpr uint32_t c0Count = BLK_LEN / sizeof(DTYPE);
        const uint32_t blockNum = (dataLength + c0Count - 1) / c0Count;

        // Stage 1: BlockReduceSum to reduce within blocks
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DTYPE>(0, dataLength);
        AscendC::BlockReduceSum<DTYPE, false>(tempTensor, xLocal, 1,
            AscendC::MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        // Stage 2: WholeReduceSum for final reduction
        AscendC::SetVectorMask<DTYPE>(0, blockNum);
        AscendC::WholeReduceSum<DTYPE, false>(zLocal, tempTensor,
            AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // Large data: Two-stage WholeReduceSum (optimal for 8KB-64KB)
    __aicore__ inline void ComputeLarge()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();

        uint32_t dataLength = (coreNum > 1) ? localLength : totalLength;
        pipe.InitBuffer(calcBuf, dataLength * sizeof(DTYPE));
        AscendC::LocalTensor<DTYPE> tempTensor = calcBuf.Get<DTYPE>();

        const uint32_t repeatNum = (dataLength * sizeof(DTYPE) + REP_LEN - 1) / REP_LEN;

        // Stage 1: First WholeReduceSum pass
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<DTYPE>(0, dataLength);
        AscendC::WholeReduceSum<DTYPE, false>(tempTensor, xLocal, 1,
            AscendC::MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        // Stage 2: Second WholeReduceSum pass
        AscendC::SetVectorMask<DTYPE>(0, repeatNum);
        AscendC::WholeReduceSum<DTYPE, false>(zLocal, tempTensor, 1,
            AscendC::MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // Binary reduction: Pairwise accumulation (optimal for 64KB-512KB)
    __aicore__ inline void ComputeBinary()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();

        uint32_t dataLength = (coreNum > 1) ? localLength : totalLength;

        AscendC::BinaryRepeatParams binaryParams;
        AscendC::SetMaskCount();

        AscendC::LocalTensor<DTYPE> srcTmp = xLocal;
        AscendC::LocalTensor<DTYPE> dstTmp = zLocal;

        uint32_t totalNum = dataLength;
        constexpr uint32_t finalThreshold = (sizeof(DTYPE) == sizeof(float)) ?
            ONE_REPEAT_FLOAT_SIZE : ONE_REPEAT_HALF_SIZE;

        // Binary reduction: pairwise accumulation
        while (totalNum > finalThreshold) {
            // Split into halves and accumulate pairwise
            uint32_t halfNum = AscendC::DivCeil(totalNum, BINARY_BOUNDARY) * DEFAULT_REP_STRIDE;
            uint32_t repeatCount = (totalNum > halfNum) ? (totalNum - halfNum) : 0;

            if (repeatCount > 0) {
                AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, repeatCount);
                AscendC::Add<DTYPE, false>(dstTmp, srcTmp, srcTmp[halfNum],
                    AscendC::MASK_PLACEHOLDER, 1, binaryParams);
                AscendC::PipeBarrier<PIPE_V>();
            }

            totalNum = halfNum;
            srcTmp = dstTmp;
        }

        // Final WholeReduceSum
        AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, totalNum);
        AscendC::WholeReduceSum<DTYPE, false>(dstTmp, srcTmp,
            AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ResetMask();
        AscendC::SetMaskNorm();

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // Multi-core parallel reduction with cross-core merge
    __aicore__ inline void ComputeMultiCore()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();

        // Each core computes local reduction using binary reduction
        pipe.InitBuffer(calcBuf, localLength * sizeof(DTYPE));
        AscendC::LocalTensor<DTYPE> tempTensor = calcBuf.Get<DTYPE>();

        AscendC::BinaryRepeatParams binaryParams;
        AscendC::SetMaskCount();

        AscendC::LocalTensor<DTYPE> srcTmp = xLocal;
        AscendC::LocalTensor<DTYPE> dstTmp = tempTensor;

        uint32_t totalNum = localLength;
        constexpr uint32_t finalThreshold = (sizeof(DTYPE) == sizeof(float)) ?
            ONE_REPEAT_FLOAT_SIZE : ONE_REPEAT_HALF_SIZE;

        // Local binary reduction
        while (totalNum > finalThreshold) {
            uint32_t halfNum = AscendC::DivCeil(totalNum, BINARY_BOUNDARY) * DEFAULT_REP_STRIDE;
            uint32_t repeatCount = (totalNum > halfNum) ? (totalNum - halfNum) : 0;

            if (repeatCount > 0) {
                AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, repeatCount);
                AscendC::Add<DTYPE, false>(dstTmp, srcTmp, srcTmp[halfNum],
                    AscendC::MASK_PLACEHOLDER, 1, binaryParams);
                AscendC::PipeBarrier<PIPE_V>();
            }

            totalNum = halfNum;
            srcTmp = dstTmp;
        }

        // Final local reduction
        AscendC::SetVectorMask<uint8_t, AscendC::MaskMode::COUNTER>(0, totalNum);
        AscendC::WholeReduceSum<DTYPE, false>(dstTmp, srcTmp,
            AscendC::MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ResetMask();
        AscendC::SetMaskNorm();

        // Store local result to global memory for cross-core merge
        AscendC::LocalTensor<DTYPE> reduceBufLocal = reduceBuf.Get<DTYPE>();
        reduceBufLocal[0] = dstTmp[0];

        // Synchronize all cores before cross-core reduction
        AscendC::PipeBarrier<PIPE_V>();

        // Core 0 performs final cross-core reduction
        if (coreId == 0) {
            AscendC::LocalTensor<DTYPE> crossCoreSrc = reduceBuf.Get<DTYPE>();
            AscendC::LocalTensor<DTYPE> crossCoreDst = zLocal;

            AscendC::SetMaskCount();
            AscendC::SetVectorMask<DTYPE>(0, coreNum);
            AscendC::WholeReduceSum<DTYPE>(crossCoreDst, crossCoreSrc, 1,
                DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetMaskNorm();
        }

        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.DeQue<DTYPE>();

        // In multi-core mode, only core 0 writes the final result
        if (coreNum == 1 || coreId == 0) {
            AscendC::DataCopy(zGm, zLocal, this->outLength);
        }

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX;  // Double buffered
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;  // For cross-core reduction
    AscendC::GlobalTensor<DTYPE> xGm;
    AscendC::GlobalTensor<DTYPE> zGm;
    uint32_t totalLength;
    uint32_t outLength;
    uint32_t localLength;   // Per-core data length
    uint32_t localOffset;   // Per-core data offset
    uint32_t coreId;
    uint32_t coreNum;
};

extern "C" __global__ __aicore__ void reduce_optimized(GM_ADDR x, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    KernelReduceOptimized<float> op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.outLength,
            tiling_data.coreId, tiling_data.coreNum);

    if (TILING_KEY_IS(REDUCE_TILING_SMALL)) {
        op.Process<REDUCE_TILING_SMALL>();
    } else if (TILING_KEY_IS(REDUCE_TILING_MEDIUM)) {
        op.Process<REDUCE_TILING_MEDIUM>();
    } else if (TILING_KEY_IS(REDUCE_TILING_LARGE)) {
        op.Process<REDUCE_TILING_LARGE>();
    } else if (TILING_KEY_IS(REDUCE_TILING_BINARY)) {
        op.Process<REDUCE_TILING_BINARY>();
    } else if (TILING_KEY_IS(REDUCE_TILING_MULTICORE)) {
        op.Process<REDUCE_TILING_MULTICORE>();
    }
}

#ifndef ASCENDC_CPU_DEBUG
void reduce_optimized_do(uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *x, uint8_t *z, uint8_t *workspace, uint8_t *tiling)
{
    reduce_optimized<<<blockDim, l2ctrl, stream>>>(x, z, workspace, tiling);
}
#endif
