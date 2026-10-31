// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#pragma once
#include <cstdint>

// Same arch35 tiling fields as the original QLI implementation. The CANN
// entry reads this structure from CANN tiling data.
struct QLITilingData {
    uint32_t bSize, tSize, n2Size, gSize, s1Size, s2Size, sparseCount;
    uint32_t keyStride0, keyDequantScaleStride0, usedCoreNum;
    uint32_t blockSize, maxBlockNumPerBatch, sparseMode;
};
