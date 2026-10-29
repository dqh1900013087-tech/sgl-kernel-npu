// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
// Torch adapter for the Packed QLI implementation bundled in this wheel.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <aclnn_sgl_quant_lightning_indexer_packed.h>
#include <limits>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace sglang::npu_kernel {
namespace {
void libraryAnchor() {}
struct PackedApi {
    void* handle;
    decltype(&aclnnSglQuantLightningIndexerPackedGetWorkspaceSize) prepare;
    decltype(&aclnnSglQuantLightningIndexerPacked) execute;
    PackedApi() {
        // Resolve relative to this DSO, so wheels can be relocated and do not
        // depend on an externally installed qli_packed_transformer vendor.
        Dl_info self{};
        TORCH_CHECK(dladdr(reinterpret_cast<void*>(&libraryAnchor), &self) && self.dli_fname,
                    "Cannot locate sgl-kernel-npu shared library");
        auto path = std::filesystem::canonical(self.dli_fname).parent_path() /
            "qli_packed/vendors/sgl_qli_packed/op_api/lib/libcust_opapi.so";
        TORCH_CHECK(std::filesystem::is_regular_file(path), "Packed opapi not found: ", path.string());
        path = std::filesystem::canonical(path);
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(handle, "Cannot load Packed opapi ", path.string(), ": ", dlerror());
        prepare = reinterpret_cast<decltype(prepare)>(dlsym(handle, "aclnnSglQuantLightningIndexerPackedGetWorkspaceSize"));
        execute = reinterpret_cast<decltype(execute)>(dlsym(handle, "aclnnSglQuantLightningIndexerPacked"));
        TORCH_CHECK(prepare && execute, "Packed ACLNN symbols missing from ", path.string());
        // Intentionally retain the handle while queued calls may use the functions.
    }
};
PackedApi& api() { static PackedApi value; return value; }
aclDataType toAclDtype(at::ScalarType t) {
    switch (t) {
        case at::kFloat8_e4m3fn: return ACL_FLOAT8_E4M3FN;
        case at::kByte: return ACL_UINT8;
        case at::kBFloat16: return ACL_BF16;
        case at::kFloat: return ACL_FLOAT;
        case at::kInt: return ACL_INT32;
        default: TORCH_CHECK(false, "Unsupported Packed QLI tensor dtype");
    }
}
struct Descriptors {
    std::vector<aclTensor*> values;
    const aclTensor* add(const at::Tensor& tensor) {
        auto* descriptor = aclCreateTensor(tensor.sizes().data(), tensor.dim(), toAclDtype(tensor.scalar_type()),
            tensor.strides().data(), 0, ACL_FORMAT_ND, tensor.sizes().data(), tensor.dim(), tensor.data_ptr());
        TORCH_CHECK(descriptor, "aclCreateTensor failed");
        values.push_back(descriptor);
        return descriptor;
    }
    ~Descriptors() { for (auto* value : values) aclDestroyTensor(value); }
};
} // namespace

at::Tensor quant_lightning_indexer_packed(const at::Tensor& query, const at::Tensor& packed_cache,
    const at::Tensor& weights, const at::Tensor& query_scale, const at::Tensor& seq_q,
    const at::Tensor& seq_k, const at::Tensor& block_table, int64_t topk, int64_t sparse_mode)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on NPU");
    const c10::OptionalDeviceGuard guard(query.device());
    std::vector<at::Tensor> keep_alive{query, packed_cache, weights, query_scale, seq_q, seq_k, block_table};
    for (const auto& tensor : keep_alive) {
        TORCH_CHECK(tensor.device() == query.device(), "all inputs must be on the same NPU device");
        TORCH_CHECK(tensor.is_contiguous(), "all inputs must be contiguous");
        TORCH_CHECK(at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND, "all inputs must use ND format");
    }
    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn, "query must be float8_e4m3fn");
    TORCH_CHECK(query.dim() == 3 && query.size(2) == 128, "query must have TND shape [T, heads, 128]");
    TORCH_CHECK(packed_cache.scalar_type() == at::kByte && packed_cache.dim() == 2,
        "packed_cache must be uint8 [pages, P*132]");
    TORCH_CHECK(packed_cache.size(0) > 0 && packed_cache.size(1) % 132 == 0,
        "packed_cache must have positive pages and width P*132");
    auto page = packed_cache.size(1) / 132;
    TORCH_CHECK(page >= 16 && page <= 1024 && page % 16 == 0,
        "page size must be a multiple of 16 in [16, 1024]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed_cache.data_ptr()) % 4 == 0,
        "packed_cache must be FP32 aligned");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be float32");
    TORCH_CHECK(weights.dim() == 2 && query_scale.dim() == 2 &&
        weights.size(0) == query.size(0) && weights.size(1) == query.size(1) &&
        weights.sizes() == query_scale.sizes(), "weights/query_scale must have shape [T, heads]");
    TORCH_CHECK(seq_q.scalar_type() == at::kInt && seq_k.scalar_type() == at::kInt &&
        block_table.scalar_type() == at::kInt, "sequence lengths and block_table must be int32");
    TORCH_CHECK(seq_q.dim() == 1 && seq_k.dim() == 1 && seq_q.numel() == seq_k.numel(),
        "seq_q/seq_k must have shape [batch]");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == seq_k.numel(),
        "block_table must have shape [batch, max_pages]");
    TORCH_CHECK(topk > 0 && topk <= 2048, "topk must be in [1, 2048]");
    TORCH_CHECK(sparse_mode == 0 || sparse_mode == 3, "sparse_mode must be 0 or 3");
    auto output = at::empty({query.size(0), 1, topk}, query.options().dtype(at::kInt));
    if (query.size(0) == 0) return output;

    auto descriptors = std::make_shared<Descriptors>();
    const auto* q = descriptors->add(query);
    const auto* cache = descriptors->add(packed_cache);
    const auto* w = descriptors->add(weights);
    const auto* qs = descriptors->add(query_scale);
    const auto* sq = descriptors->add(seq_q);
    const auto* sk = descriptors->add(seq_k);
    const auto* table = descriptors->add(block_table);
    const auto* out = descriptors->add(output);
    uint64_t size = 0;
    aclOpExecutor* executor = nullptr;
    char layout[] = "TND";
    auto& library = api();
    char key_layout[] = "PA_BSND";
    auto result = library.prepare(q, cache, w, qs, nullptr, sq, sk, table,
        0, 0, layout, key_layout, topk, sparse_mode,
        std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
        -1, -1, out, &size, &executor);
    TORCH_CHECK(result == 0, "Packed GetWorkspaceSize failed: ", result, "; ", aclGetRecentErrMsg());
    auto workspace = at::empty({static_cast<int64_t>(size)}, query.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream(query.device().index());
    keep_alive.push_back(output);
    keep_alive.push_back(workspace);
    for (const auto& tensor : keep_alive) {
        if (tensor.numel()) c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    auto execute = library.execute;
    at_npu::native::OpCommand command;
    command.Name("SglQuantLightningIndexerPacked").SetCustomHandler(
        [descriptors, keep_alive, workspace, size, executor, execute, stream]() -> int {
            auto status = execute(size ? workspace.data_ptr() : nullptr, size, executor, stream.stream(false));
            TORCH_CHECK(status == 0, "Packed execute failed: ", status, "; ", aclGetRecentErrMsg());
            return 0;
        });
    command.Run();
    return output;
}
} // namespace sglang::npu_kernel
