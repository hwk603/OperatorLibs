/**
 * @file reduce_optimized_tiling.h
 * Tiling data structure for optimized Reduce operator
 */

#ifndef REDUCE_OPTIMIZED_TILING_H
#define REDUCE_OPTIMIZED_TILING_H

#include "graph/ascend_string.h"
#include "runtime/tiling_context.h"

namespace optiling {

BEGIN_TILING_DATA_CLASS(ReduceTilingData)
public:
    // Basic tiling parameters
    uint32_t totalLength;  // Total number of elements in input
    uint32_t outLength;    // Output size (scalar = 1)
    uint32_t coreId;       // Current AI core ID (0-based)
    uint32_t coreNum;      // Total number of AI cores being used

    // Registration function
    void GetMaxMinDim(int32_t& maxDim, int32_t& minDim) const
    {
        maxDim = totalLength;
        minDim = outLength;
    }

    // Serialization
    bool CheckSupport()
    {
        return true;
    }

END_TILING_DATA_CLASS

} // namespace optiling

#endif // REDUCE_OPTIMIZED_TILING_H
