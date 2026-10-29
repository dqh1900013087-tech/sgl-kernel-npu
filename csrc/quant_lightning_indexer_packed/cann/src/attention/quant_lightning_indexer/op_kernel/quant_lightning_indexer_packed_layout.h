/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef QUANT_LIGHTNING_INDEXER_PACKED_LAYOUT_H
#define QUANT_LIGHTNING_INDEXER_PACKED_LAYOUT_H

#include <cstdint>
#include <limits>

namespace QLIPacked {
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t SCALE_BYTES = 4;
constexpr uint32_t TOKEN_BYTES = HEAD_DIM + SCALE_BYTES;

struct Layout {
    uint32_t pageSize;
    uint32_t keyStride;       // FP8 elements (bytes).
    uint32_t scaleStride;     // FP32 elements.
    uint32_t scaleOffset;     // Bytes from packed base to first page's scale region.
};

// Host-only validation shared by ACLNN, tiling and portable layout tests.
inline bool MakeLayout(int64_t pages, int64_t rowBytes, Layout& layout)
{
    if (pages <= 0 || pages > std::numeric_limits<int32_t>::max() || rowBytes <= 0 ||
        rowBytes % TOKEN_BYTES != 0) {
        return false;
    }
    const int64_t pageSize = rowBytes / TOKEN_BYTES;
    if (pageSize < 16 || pageSize > 1024 || pageSize % 16 != 0) {
        return false;
    }
    layout = {static_cast<uint32_t>(pageSize), static_cast<uint32_t>(rowBytes),
              static_cast<uint32_t>(rowBytes / SCALE_BYTES), static_cast<uint32_t>(pageSize * HEAD_DIM)};
    return true;
}
} // namespace QLIPacked
#endif
