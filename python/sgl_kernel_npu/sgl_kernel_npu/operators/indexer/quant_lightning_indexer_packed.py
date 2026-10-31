"""Public interface to the native Ascend950 Packed QLI implementation."""

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

if hasattr(torch.ops.npu, "sgl_quant_lightning_indexer_packed"):

    @torch.library.register_fake("npu::sgl_quant_lightning_indexer_packed")
    def _fake(
        query,
        packed_cache,
        weights,
        query_scale,
        seq_q,
        seq_k,
        block_table,
        topk=2048,
        sparse_mode=3,
    ):
        torch._check(query.dim() == 3)
        torch._check(query.shape[-1] == 128)
        torch._check(0 < topk <= 2048)
        return query.new_empty((query.shape[0], 1, topk), dtype=torch.int32)


@register_impl(
    "indexer.quant_lightning_indexer_packed",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "indexer.quant_lightning_indexer_packed",
    "ascendc",
    supported_generations={Generation.ARCH35},
)
def _ascendc_impl(
    query,
    packed_cache,
    weights,
    query_scale,
    seq_q,
    seq_k,
    block_table,
    topk,
    sparse_mode,
):
    if not hasattr(torch.ops.npu, "sgl_quant_lightning_indexer_packed"):
        raise RuntimeError(
            "Build with python3 setup.py --arch-version 3510 --ops quant_lightning_indexer_packed"
        )
    return torch.ops.npu.sgl_quant_lightning_indexer_packed(
        query,
        packed_cache,
        weights,
        query_scale,
        seq_q,
        seq_k,
        block_table,
        topk,
        sparse_mode,
    )


_torch_native_fn = None


@register_impl(
    "indexer.quant_lightning_indexer_packed",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_native_impl(*args):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.quant_lightning_indexer_packed import (
            quant_lightning_indexer_packed,
        )

        _torch_native_fn = quant_lightning_indexer_packed
    return _torch_native_fn(*args)


@register_operator("indexer.quant_lightning_indexer_packed")
def quant_lightning_indexer_packed(
    query,
    packed_cache,
    weights,
    query_scale,
    seq_q,
    seq_k,
    block_table,
    topk=2048,
    sparse_mode=3,
    *,
    impl="auto",
):
    """Return DSA Top-k indices; ``auto`` selects ``ascendc`` on arch35.

    All inputs are contiguous ND tensors on the same NPU. Inputs are not modified.
    Each cache page stores all FP8 keys followed by all FP32 key scales.

    Args:
        query (torch.Tensor): FP8 e4m3fn queries [T, H, 128].
        packed_cache (torch.Tensor): UINT8 cache [pages, P * 132], FP32-aligned;
            P is a multiple of 16 in [16, 1024].
        weights (torch.Tensor): BF16 per-query head weights [T, H].
        query_scale (torch.Tensor): FP32 query dequantization scales [T, H].
        seq_q (torch.Tensor): INT32 cumulative query lengths [B].
        seq_k (torch.Tensor): INT32 valid key lengths [B].
        block_table (torch.Tensor): INT32 physical page IDs [B, max_pages].
        topk (int): Number of indices per query, in [1, 2048]; default 2048.
        sparse_mode (int): 0 for unmasked or 3 for causal indexing; default 3.
        impl (str): "auto", "ascendc", or "torch_native"; auto selects ascendc.

    Returns:
        torch.Tensor: INT32 indices [T, 1, topk] on the query device.
    """
    impl_fn = get_impl_by_name("indexer.quant_lightning_indexer_packed", impl)
    return impl_fn(
        query,
        packed_cache,
        weights,
        query_scale,
        seq_q,
        seq_k,
        block_table,
        topk,
        sparse_mode,
    )
