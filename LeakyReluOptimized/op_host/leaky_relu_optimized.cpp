/**
 * @file leaky_relu_optimized.cpp
 * Host-side tiling implementation for optimized LeakyReLU operator
 *
 * Key Features:
 * 1. Adaptive tiling strategy based on data size
 * 2. Multi-core configuration for large tensors
 * 3. Support for both float16 and float32
 * 4. Architecture-specific optimization
 *
 * Copyright (C) 2025. Competition Submission.
 */

#include "../op_kernel/leaky_relu_optimized_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace optiling;

namespace {

// ===================================================================
// Tiling Strategy Configuration
// ===================================================================

struct LeakyReluTilingConfig {
    uint32_t totalLength;
    uint32_t coreNum;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t strategy;  // 1: small, 2: medium, 3: large

    static LeakyReluTilingConfig GetConfig(uint32_t totalLength, uint32_t dataType,
                                          platform_ascendc::SocVersion socVersion)
    {
        LeakyReluTilingConfig config;

        // Size thresholds based on data type
        uint32_t smallThreshold = (dataType == 1) ? 4096 : 2048;    // 8KB
        uint32_t mediumThreshold = (dataType == 1) ? 131072 : 65536; // 256KB

        config.totalLength = totalLength;

        if (totalLength <= smallThreshold) {
            // Small data: single core, vectorized
            config.strategy = 1;
            config.coreNum = 1;
            config.tileNum = 1;
            config.tileLength = totalLength;
        } else if (totalLength <= mediumThreshold) {
            // Medium data: multi-core with optimized tiles
            config.strategy = 2;
            if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
                config.coreNum = std::min(static_cast<uint32_t>(6), totalLength / 256);
            } else {
                config.coreNum = std::min(static_cast<uint32_t>(2), totalLength / 256);
            }
            config.coreNum = std::max(config.coreNum, static_cast<uint32_t>(1));

            uint32_t elementsPerCore = (totalLength + config.coreNum - 1) / config.coreNum;
            config.tileLength = std::min(elementsPerCore, static_cast<uint32_t>(8192));
            config.tileNum = (elementsPerCore + config.tileLength - 1) / config.tileLength;
        } else {
            // Large data: max throughput
            config.strategy = 3;
            if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
                // Ascend910B: use more cores for large data
                config.coreNum = std::min(static_cast<uint32_t>(24), totalLength / 4096);
            } else {
                config.coreNum = std::min(static_cast<uint32_t>(2), totalLength / 4096);
            }
            config.coreNum = std::max(config.coreNum, static_cast<uint32_t>(1));

            uint32_t elementsPerCore = (totalLength + config.coreNum - 1) / config.coreNum;
            config.tileLength = std::min(elementsPerCore, static_cast<uint32_t>(8192));
            config.tileNum = (elementsPerCore + config.tileLength - 1) / config.tileLength;
        }

        return config;
    }
};

} // anonymous namespace

namespace optiling {

/**
 * @brief Generate optimized LeakyReLU tiling
 * @param context: Tiling kernel context
 * @retval Status of GetTiling (GRAPH_SUCCESS or GRAPH_FAILED)
 */
ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // Get platform information
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // Extract input tensor information
    auto shape_x = context->GetInputTensor(0)->GetOriginShape();
    auto dtype_x = context->GetInputTensor(0)->GetDataType();

    // Calculate total length
    uint32_t totalLength = 1;
    int32_t dimNum = shape_x.GetDimNum();
    for (int32_t i = 0; i < dimNum; ++i) {
        totalLength *= shape_x.GetDim(i);
    }

    // Get data type
    uint32_t dataType = 0;  // 0: float32, 1: float16
    if (dtype_x == ge::DT_FLOAT16) {
        dataType = 1;
    } else if (dtype_x == ge::DT_FLOAT) {
        dataType = 0;
    } else {
        // Unsupported data type
        return ge::GRAPH_FAILED;
    }

    // Get negative slope (alpha) parameter
    float negativeSlope = 0.01f;  // Default value
    // Try to get from attributes if available
    if (context->GetAttrs()->GetFloat("negative_slope", negativeSlope) != ge::GRAPH_SUCCESS) {
        // Use default value
        negativeSlope = 0.01f;
    }

    // Validate negative slope range
    if (negativeSlope < -10.0f || negativeSlope > 10.0f) {
        return ge::GRAPH_FAILED;
    }

    // Get tiling configuration
    auto socVersion = ascendcPlatform.GetSocVersion();
    LeakyReluTilingConfig config = LeakyReluTilingConfig::GetConfig(
        totalLength, dataType, socVersion);

    // Set tiling data
    LeakyReluOptimizedTilingData *tiling = context->GetTilingData<LeakyReluOptimizedTilingData>();
    tiling->totalLength = totalLength;
    tiling->negativeSlope = negativeSlope;
    tiling->dataType = dataType;
    tiling->coreId = 0;  // Will be set by each core
    tiling->coreNum = config.coreNum;

    // Set block dimension
    context->SetBlockDim(config.coreNum);

    // Set tiling key based on strategy
    context->SetTilingKey(config.strategy);

    // LeakyReLU doesn't require workspace
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops {

/**
 * @brief Optimized LeakyReLU operator definition
 */
class LeakyReluOptimized : public OpDef {
public:
    explicit LeakyReluOptimized(const char *name) : OpDef(name)
    {
        // Input X: 1D or 2D tensor
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // Output Y: same shape and type as input
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // Negative slope (alpha) attribute
        this->Attr("negative_slope")
            .Float()
            .SetDefault(0.01f);

        // Configure AICore with tiling function
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910b");
    }
};

OP_ADD(LeakyReluOptimized);

} // namespace ops
