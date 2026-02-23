/**
 * @file matmul_optimized.cpp
 * Optimized MatMul Operator for Ascend AI Algorithm Challenge
 *
 * Key Optimizations:
 * 1. CATLASS-based template library for high-performance GEMM
 * 2. Adaptive tiling strategy based on matrix dimensions
 * 3. Padding optimization for memory alignment
 * 4. Multi-core parallel processing
 * 5. Ping-pong pipeline for data transfer overlap
 * 6. L1/L0 cache optimization
 *
 * Copyright (C) 2025. Competition Submission.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/gemm/kernel/optimized_matmul.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

using namespace Catlass;

// ===================================================================
// Optimization Configuration
// ===================================================================

constexpr uint32_t alignByByte = 512;
constexpr uint32_t alignByElement = alignByByte / sizeof(half);

// Architecture tag
using ArchTag = Arch::AtlasA2;

// Enable flags for optimization
constexpr bool ENABLE_UNIT_FLAG = true;
constexpr bool ENABLE_SHUFFLE_K = true;

// Data types
using ElementA = half;
using ElementB = half;
using ElementC = half;

// Layout configurations - optimized for row-major A and column-major B
using LayoutA = layout::RowMajor;
using LayoutB = layout::ColumnMajor;
using LayoutC = layout::RowMajor;

// GemmType definitions
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;

// Dispatch policy with preload and shuffle optimization
using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;

// ===================================================================
// Tile Shape Optimization
// ===================================================================

// Adaptive tile shapes based on matrix layout
// For ColumnMajor x ColumnMajor: different shape for better performance
using L1TileShape = std::conditional_t<
    std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
    GemmShape<256, 128, 256>,  // ColumnMajor x ColumnMajor
    GemmShape<128, 256, 256>>;  // RowMajor x ColumnMajor (default)

using L0TileShape = std::conditional_t<
    std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
    GemmShape<256, 128, 64>,   // ColumnMajor x ColumnMajor
    GemmShape<128, 256, 64>>;   // RowMajor x ColumnMajor (default)

// Block scheduler for optimized data access pattern
using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using BlockScheduler31 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

// No epilogue (post-processing) for basic MatMul
using BlockEpilogue = void;

// ===================================================================
// Tile Copy Optimization
// ===================================================================

template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class BiasType = void>
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;

    // Enable interval data copy for small matrices
    // When matrix A is row-major with rows < 16, use CopyGmToL1IntervalDataCopy
    // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

    using CopyGmToL1A = typename Base::CopyGmToL1A;
    using CopyGmToL1B = typename Base::CopyGmToL1B;
    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;
    using CopyL0CToGm = typename Base::CopyL0CToGm;
    using BiasTypeSelector = typename Base::BiasTypeSelector;
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;
};

// ===================================================================
// Padding Configuration
// ===================================================================

template <class Layout, class Element>
constexpr auto GetPaddingTag() {
    using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;
    if constexpr (std::is_same_v<Layout, layout::zN> || std::is_same_v<Layout, layout::nZ>) {
        return PaddingTag::NO_PADDING;
    } else {
        return PaddingTag::PADDING_NZ;
    }
}

constexpr uint32_t COMPUTE_LENGTH_A = 48 * 1024 / sizeof(ElementA);
constexpr uint32_t COMPUTE_LENGTH_B = 48 * 1024 / sizeof(ElementB);

template <class Layout, class Element, uint32_t ComputeLength>
using PaddingBuilder = Catlass::Gemm::Kernel::PaddingBuilder<
    GetPaddingTag<Layout, Element>(), ArchTag, Element, Layout, ComputeLength>;

using PaddingTagA = GetPaddingTag<LayoutA, ElementA>;
using PaddingTagB = GetPaddingTag<LayoutB, ElementB>;

using PaddingBuilderA = PaddingBuilder<LayoutA, ElementA, COMPUTE_LENGTH_A>;
using PaddingBuilderB = PaddingBuilder<LayoutB, ElementB, COMPUTE_LENGTH_B>;

using GlobalPaddingA = typename PaddingBuilderA::Padding;
using GlobalPaddingB = typename PaddingBuilderB::Padding;

// ===================================================================
// Kernel Implementation with Strategy Selection
// ===================================================================

template <bool UsePaddingA, bool UsePaddingB, class BlockScheduler>
struct SelectKernelImpl;

// Specialization: No padding
template <class BlockScheduler>
struct SelectKernelImpl<false, false, BlockScheduler> {
    using TileCopy = TileCopyOpt<ArchTag, AType, BType, CType>;
    using BlockMmadOpt = Gemm::Block::BlockMmad<
        DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
    using Kernel = Gemm::Kernel::OptimizedMatmul<
        void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler>;
};

// Specialization: Padding A only
template <class BlockScheduler>
struct SelectKernelImpl<true, false, BlockScheduler> {
    using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
    using ATypeMmad = Gemm::GemmType<ElementA, LayoutMmadA>;
    using TileCopy = TileCopyOpt<ArchTag, ATypeMmad, BType, CType>;
    using BlockMmadOpt = Gemm::Block::BlockMmad<
        DispatchPolicy, L1TileShape, L0TileShape, ATypeMmad, BType, CType, void, TileCopy>;
    using Kernel = Gemm::Kernel::OptimizedMatmul<
        GlobalPaddingA, void, BlockMmadOpt, BlockEpilogue, BlockScheduler>;
};

// Specialization: Padding B only
template <class BlockScheduler>
struct SelectKernelImpl<false, true, BlockScheduler> {
    using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
    using BTypeMmad = Gemm::GemmType<ElementB, LayoutMmadB>;
    using TileCopy = TileCopyOpt<ArchTag, AType, BTypeMmad, CType>;
    using BlockMmadOpt = Gemm::Block::BlockMmad<
        DispatchPolicy, L1TileShape, L0TileShape, AType, BTypeMmad, CType, void, TileCopy>;
    using Kernel = Gemm::Kernel::OptimizedMatmul<
        void, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler>;
};

// Specialization: Both padding
template <class BlockScheduler>
struct SelectKernelImpl<true, true, BlockScheduler> {
    using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
    using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
    using ATypeMmad = Gemm::GemmType<ElementA, LayoutMmadA>;
    using BTypeMmad = Gemm::GemmType<ElementB, LayoutMmadB>;
    using TileCopy = TileCopyOpt<ArchTag, ATypeMmad, BTypeMmad, CType>;
    using BlockMmadOpt = Gemm::Block::BlockMmad<
        DispatchPolicy, L1TileShape, L0TileShape, ATypeMmad, BTypeMmad, CType, void, TileCopy>;
    using Kernel = Gemm::Kernel::OptimizedMatmul<
        GlobalPaddingA, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler>;
};

// ===================================================================
// Main Kernel Entry Point
// ===================================================================

extern "C" __global__ __aicore__ void matmul_optimized(
    GM_ADDR a, GM_ADDR b, GM_ADDR c,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    // Extract tiling parameters
    uint32_t m = tiling_data.m;
    uint32_t n = tiling_data.n;
    uint32_t k = tiling_data.k;
    bool needPaddingA = tiling_data.needPaddingA;
    bool needPaddingB = tiling_data.needPaddingB;
    bool useMFirstStrategy = (m > n);

    // Create problem shape
    GemmCoord problemShape(m, n, k);

    // Select kernel based on padding requirements and strategy
    if (useMFirstStrategy) {
        if (needPaddingA && needPaddingB) {
            using Kernel = SelectKernelImpl<true, true, BlockScheduler30>::Kernel;
            typename Kernel::Arguments arguments{problemShape, a, b, c};
            using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
            MatmulAdapter matmulOp;

            // Get FFTS address and core num
            uint32_t fftsLen{0};
            uint64_t fftsAddr{0};
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

            auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RunAdapter(matmulOp, arguments, nullptr, aicCoreNum, fftsAddr);
        } else if (needPaddingA) {
            using Kernel = SelectKernelImpl<true, false, BlockScheduler30>::Kernel;
            typename Kernel::Arguments arguments{problemShape, a, b, c};
            using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
            MatmulAdapter matmulOp;

            uint32_t fftsLen{0};
            uint64_t fftsAddr{0};
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

            auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RunAdapter(matmulOp, arguments, nullptr, aicCoreNum, fftsAddr);
        } else if (needPaddingB) {
            using Kernel = SelectKernelImpl<false, true, BlockScheduler30>::Kernel;
            typename Kernel::Arguments arguments{problemShape, a, b, c};
            using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
            MatmulAdapter matmulOp;

            uint32_t fftsLen{0};
            uint64_t fftsAddr{0};
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

            auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RunAdapter(matmulOp, arguments, nullptr, aicCoreNum, fftsAddr);
        } else {
            using Kernel = SelectKernelImpl<false, false, BlockScheduler30>::Kernel;
            typename Kernel::Arguments arguments{problemShape, a, b, c};
            using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
            MatmulAdapter matmulOp;

            uint32_t fftsLen{0};
            uint64_t fftsAddr{0};
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

            auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RunAdapter(matmulOp, arguments, nullptr, aicCoreNum, fftsAddr);
        }
    } else {
        if (needPaddingA && needPaddingB) {
            using Kernel = SelectKernelImpl<true, true, BlockScheduler31>::Kernel;
            typename Kernel::Arguments arguments{problemShape, a, b, c};
            using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
            MatmulAdapter matmulOp;

            uint32_t fftsLen{0};
            uint64_t fftsAddr{0};
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

            auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RunAdapter(matmulOp, arguments, nullptr, aicCoreNum, fftsAddr);
        } else if (needPaddingA) {
            using Kernel = SelectKernelImpl<true, false, BlockScheduler31>::Kernel;
            typename Kernel::Arguments arguments{problemShape, a, b, c};
            using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
            MatmulAdapter matmulOp;

            uint32_t fftsLen{0};
            uint64_t fftsAddr{0};
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

            auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RunAdapter(matmulOp, arguments, nullptr, aicCoreNum, fftsAddr);
        } else if (needPaddingB) {
            using Kernel = SelectKernelImpl<false, true, BlockScheduler31>::Kernel;
            typename Kernel::Arguments arguments{problemShape, a, b, c};
            using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
            MatmulAdapter matmulOp;

            uint32_t fftsLen{0};
            uint64_t fftsAddr{0};
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

            auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RunAdapter(matmulOp, arguments, nullptr, aicCoreNum, fftsAddr);
        } else {
            using Kernel = SelectKernelImpl<false, false, BlockScheduler31>::Kernel;
            typename Kernel::Arguments arguments{problemShape, a, b, c};
            using MatmulAdapter = Gemm::Device::DeviceGemm<Kernel>;
            MatmulAdapter matmulOp;

            uint32_t fftsLen{0};
            uint64_t fftsAddr{0};
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

            auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
            RunAdapter(matmulOp, arguments, nullptr, aicCoreNum, fftsAddr);
        }
    }
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_optimized_do(uint32_t blockDim, void *l2ctrl, void *stream,
    uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *workspace, uint8_t *tiling)
{
    matmul_optimized<<<blockDim, l2ctrl, stream>>>(a, b, c, workspace, tiling);
}
#endif
