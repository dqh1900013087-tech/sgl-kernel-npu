/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_lightning_indexer_template_tiling_key.h"
#include "quant_lightning_indexer_packed_layout.h"

#include "arch35/quant_lightning_indexer_kernel_arch35.h"

using namespace QLIKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<QLIType<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(QLITilingData, tiling_data_in, tiling); \
        const QLITilingData* __restrict tiling_data = &tiling_data_in; \
        if constexpr (DT_K == QLI_TPL_UINT8) { \
            /* Each page is [P * 128 FP8 K bytes][P FP32 scales]. */ \
            /* Cube uses P*132 byte stride, Vector uses P*33 FP32 stride. */ \
            keyScale = key + static_cast<uint64_t>(tiling_data->blockSize) * QLIPacked::HEAD_DIM; \
        } \
        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \
                sparseIndices, user, tiling_data, &tPipe); \
        op.Process(); \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void sgl_quant_lightning_indexer_packed(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights,
                                                   __gm__ uint8_t* queryScale, __gm__ uint8_t* keyScale,
                                                   __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengthsK,
                                                   __gm__ uint8_t* blocktable, __gm__ uint8_t* sparseIndices,
                                                   __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    TPipe tPipe;
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // This private operator is registered only for Ascend950 FP8 packed cache.
    INVOKE_LI_NO_KFC_OP_IMPL(SglQuantLightningIndexerPackedKernel,
        fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, PAGE_ATTENTION,
        LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
        bfloat16_t, float32_t, float32_t, uint16_t);
}
