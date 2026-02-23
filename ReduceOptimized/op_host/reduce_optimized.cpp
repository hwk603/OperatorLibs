/**
 * @file reduce_optimized_tiling.cpp
 * Optimized Tiling Strategy for Reduce Operator
 *
 * Key Features:
 * 1. Adaptive threshold selection based on data type and size
 * 2. Multi-core parallelization strategy for large tensors
 * 3. Smart block dimension calculation
 *
 * Copyright (C) 2025. Competition Submission.
 */

#include "reduce_optimized_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {

// Tiling strategy keys
constexpr uint32_t REDUCE_TILING_SMALL = 1;
constexpr uint32_t REDUCE_TILING_MEDIUM = 2;
constexpr uint32_t REDUCE_TILING_LARGE = 3;
constexpr uint32_t REDUCE_TILING_BINARY = 4;
constexpr uint32_t REDUCE_TILING_MULTICORE = 5;

// Optimized size thresholds (in elements)
constexpr uint32_t FLOAT_SMALL_THRESHOLD = 64;      // 256B
constexpr uint32_t FLOAT_MEDIUM_THRESHOLD = 2048;   // 8KB  (optimized from 512)
constexpr uint32_t FLOAT_LARGE_THRESHOLD = 16384;   // 64KB (optimized from 4096)
constexpr uint32_t FLOAT_BINARY_THRESHOLD = 131072; // 512KB (new threshold)

constexpr uint32_t HALF_SMALL_THRESHOLD = 128;      // 256B
constexpr uint32_t HALF_MEDIUM_THRESHOLD = 4096;    // 8KB
constexpr uint32_t HALF_LARGE_THRESHOLD = 32768;    // 64KB
constexpr uint32_t HALF_BINARY_THRESHOLD = 262144;  // 512KB

// Output shape (scalar reduction result)
constexpr uint32_t OUT_SHAPE = 1;

// Multi-core configuration
constexpr uint32_t MIN_MULTICORE_THRESHOLD = 512 * 1024;  // 512KB minimum for multi-core
constexpr uint32_t MAX_CORE_NUM = 8;  // Maximum cores to use

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;

    // Get input tensor information
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto inputDtype = context->GetInputTensor(0)->GetDataType();

    // Determine data size per element
    uint32_t elementSize = 4;  // float = 4 bytes
    if (inputDtype == ge::DT_FLOAT16) {
        elementSize = 2;
    }

    // Calculate data size in bytes
    uint64_t dataSizeBytes = static_cast<uint64_t>(totalLength) * elementSize;

    // Determine optimal tiling strategy and block dimension
    uint32_t tilingKey = REDUCE_TILING_SMALL;
    uint32_t blockDim = 1;

    // Adaptive threshold selection based on data type and size
    if (inputDtype == ge::DT_FLOAT16) {
        if (totalLength <= HALF_SMALL_THRESHOLD) {
            tilingKey = REDUCE_TILING_SMALL;
            blockDim = 1;
        } else if (totalLength <= HALF_MEDIUM_THRESHOLD) {
            tilingKey = REDUCE_TILING_MEDIUM;
            blockDim = 1;
        } else if (totalLength <= HALF_LARGE_THRESHOLD) {
            tilingKey = REDUCE_TILING_LARGE;
            blockDim = 1;
        } else if (totalLength <= HALF_BINARY_THRESHOLD) {
            tilingKey = REDUCE_TILING_BINARY;
            blockDim = 1;
        } else {
            // Multi-core for very large tensors
            tilingKey = REDUCE_TILING_MULTICORE;
            // Calculate optimal core count based on data size
            uint32_t optimalCores = std::min(
                static_cast<uint32_t>(std::sqrt(dataSizeBytes / MIN_MULTICORE_THRESHOLD)),
                MAX_CORE_NUM
            );
            // Ensure at least 2 cores for multi-core mode
            blockDim = std::max(2U, optimalCores);
        }
    } else {  // DT_FLOAT
        if (totalLength <= FLOAT_SMALL_THRESHOLD) {
            tilingKey = REDUCE_TILING_SMALL;
            blockDim = 1;
        } else if (totalLength <= FLOAT_MEDIUM_THRESHOLD) {
            tilingKey = REDUCE_TILING_MEDIUM;
            blockDim = 1;
        } else if (totalLength <= FLOAT_LARGE_THRESHOLD) {
            tilingKey = REDUCE_TILING_LARGE;
            blockDim = 1;
        } else if (totalLength <= FLOAT_BINARY_THRESHOLD) {
            tilingKey = REDUCE_TILING_BINARY;
            blockDim = 1;
        } else {
            // Multi-core for very large tensors
            tilingKey = REDUCE_TILING_MULTICORE;
            // Calculate optimal core count based on data size
            uint32_t optimalCores = std::min(
                static_cast<uint32_t>(std::sqrt(dataSizeBytes / MIN_MULTICORE_THRESHOLD)),
                MAX_CORE_NUM
            );
            // Ensure at least 2 cores for multi-core mode
            blockDim = std::max(2U, optimalCores);
        }
    }

    // Set tiling key and block dimension
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim);

    // Set tiling data
    tiling.set_totalLength(totalLength);
    tiling.set_outLength(OUT_SHAPE);
    tiling.set_coreId(0);  // Will be set per-block by runtime
    tiling.set_coreNum(blockDim);

    // Save tiling data
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // No workspace needed
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

// Shape inference
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = {optiling::OUT_SHAPE};
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

// Operator definition
namespace ops {
class ReduceOptimized : public OpDef {
public:
    explicit ReduceOptimized(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910")
            .AddConfig("ascend910b");
    }
};

OP_ADD(ReduceOptimized);
} // namespace ops
