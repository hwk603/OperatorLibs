/**
 * @file matmul_optimized.cpp
 * Host-side tiling implementation for optimized MatMul operator
 *
 * Key Features:
 * 1. Adaptive tiling strategy based on matrix dimensions
 * 2. Padding detection and optimization
 * 3. Multi-core workspace allocation
 * 4. Architecture-specific configuration
 *
 * Copyright (C) 2025. Competition Submission.
 */

#include "../op_kernel/matmul_optimized_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace optiling;

namespace {

// ===================================================================
// Tiling Strategy Configuration
// ===================================================================

struct TilingConfig {
    int32_t baseM;
    int32_t baseN;
    int32_t singleCoreM;
    int32_t singleCoreN;

    // Configuration for different matrix sizes
    static TilingConfig GetConfig(int32_t m, int32_t n, int32_t k, platform_ascendc::SocVersion socVersion) {
        TilingConfig config;

        if (socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            // Ascend310P configuration
            config.baseM = 128;
            config.baseN = 128;
            config.singleCoreM = 512;
            config.singleCoreN = 640;
        } else if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
            // Ascend910B optimized configuration
            // Adaptive tile sizes based on matrix dimensions
            if (m * n * k <= 256 * 256 * 256) {
                // Small matrices: use smaller tiles for better utilization
                config.baseM = 64;
                config.baseN = 128;
                config.singleCoreM = 256;
                config.singleCoreN = 512;
            } else if (m * n * k <= 2048 * 2048 * 2048) {
                // Medium matrices: balanced configuration
                config.baseM = 128;
                config.baseN = 256;
                config.singleCoreM = 512;
                config.singleCoreN = 640;
            } else {
                // Large matrices: maximize single core workload
                config.baseM = 128;
                config.baseN = 128;
                config.singleCoreM = 1024;
                config.singleCoreN = 1024;
            }
        } else {
            // Default configuration for other platforms
            config.baseM = 128;
            config.baseN = 128;
            config.singleCoreM = 512;
            config.singleCoreN = 640;
        }

        return config;
    }
};

// ===================================================================
// Padding Detection
// ===================================================================

struct PaddingInfo {
    bool needPaddingA;
    bool needPaddingB;

    static PaddingInfo Detect(int32_t m, int32_t n, int32_t k) {
        constexpr uint32_t alignByElement = 512 / sizeof(half);

        PaddingInfo info;
        info.needPaddingA = ((m * k) % alignByElement) != 0;
        info.needPaddingB = ((k * n) % alignByElement) != 0;

        return info;
    }
};

} // anonymous namespace

namespace optiling {

/**
 * @brief Generate optimized MatMul tiling
 * @param context: Tiling kernel context
 * @retval Status of GetTiling (GRAPH_SUCCESS or GRAPH_FAILED)
 */
ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // Get platform information
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // Extract matrix dimensions
    auto shape_a = context->GetInputTensor(0)->GetOriginShape();
    auto shape_b = context->GetInputTensor(1)->GetOriginShape();
    int32_t M = shape_a.GetDim(0);
    int32_t K = shape_a.GetDim(1);
    int32_t N = shape_b.GetDim(1);

    // Validate dimensions
    if (M <= 0 || N <= 0 || K <= 0) {
        return ge::GRAPH_FAILED;
    }

    // Get tiling configuration
    auto socVersion = ascendcPlatform.GetSocVersion();
    TilingConfig config = TilingConfig::GetConfig(M, N, K, socVersion);

    // Detect padding requirements
    PaddingInfo paddingInfo = PaddingInfo::Detect(M, N, K);

    // Setup multi-core MatMul tiling
    auto coreNumAic = ascendcPlatform.GetCoreNumAic();
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(coreNumAic);

    // Configure data types
    cubeTiling.SetAType(matmul_tiling::TPosition::GM,
                        matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM,
                        matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM,
                        matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM,
                          matmul_tiling::CubeFormat::ND,
                          matmul_tiling::DataType::DT_FLOAT);

    // Set matrix shapes
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);

    // Apply platform-specific configurations
    if (socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        cubeTiling.SetSingleShape(config.singleCoreM, config.singleCoreN, -1);
        cubeTiling.SetFixSplit(config.baseM, config.baseN, -1);
    }

    // Disable bias (pure MatMul)
    cubeTiling.SetBias(false);

    // Set buffer space
    cubeTiling.SetBufferSpace(-1, -1, -1);

    // Generate tiling data
    MatmulOptimizedTilingData *tiling = context->GetTilingData<MatmulOptimizedTilingData>();
    if (cubeTiling.GetTiling(tiling->cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    // Set custom tiling parameters
    tiling->m = M;
    tiling->n = N;
    tiling->k = K;
    tiling->needPaddingA = paddingInfo.needPaddingA;
    tiling->needPaddingB = paddingInfo.needPaddingB;

    // Get UB memory size
    uint64_t localMemSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    tiling->localMemSize = localMemSize;

    // Set block dimension and tiling key
    if (socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        // Ascend310P uses vector cores
        context->SetBlockDim(2);
        context->SetTilingKey(2);
    } else {
        // Ascend910B uses cube cores (separated arch: AIC:AIV = 1:2)
        // Vector cores: 48, Cube cores: 24
        int cubeCoreNum = (socVersion == platform_ascendc::SocVersion::ASCEND910B) ? 24 : coreNumAic / 2;
        context->SetBlockDim(cubeCoreNum);

        // Tiling key based on padding configuration
        if (paddingInfo.needPaddingA && paddingInfo.needPaddingB) {
            context->SetTilingKey(4);  // Both padding
        } else if (paddingInfo.needPaddingA) {
            context->SetTilingKey(3);  // A padding only
        } else if (paddingInfo.needPaddingB) {
            context->SetTilingKey(2);  // B padding only
        } else {
            context->SetTilingKey(1);  // No padding
        }
    }

    // Calculate workspace size
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

/**
 * @brief Optimized MatMul operator definition
 */
class MatmulOptimized : public OpDef {
public:
    explicit MatmulOptimized(const char *name) : OpDef(name)
    {
        // Input A: M x K matrix, float16
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // Input B: K x N matrix, float16
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // Output C: M x N matrix, float16
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        // Configure AICore with tiling function
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910b");
    }
};

OP_ADD(MatmulOptimized);

} // namespace ops
