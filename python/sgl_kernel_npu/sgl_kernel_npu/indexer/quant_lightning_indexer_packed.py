"""GLM DSA adapter for the externally installed Ascend 950 Packed QLI package."""
import torch

_available = hasattr(torch.ops.npu, "sgl_quant_lightning_indexer_packed")

if _available:
    @torch.library.register_fake("npu::sgl_quant_lightning_indexer_packed")
    def _fake(query, packed_cache, weights, query_scale, seq_q, seq_k, block_table,
              topk=2048, sparse_mode=3):
        torch._check(query.dim() == 3)
        torch._check(query.shape[-1] == 128)
        torch._check(0 < topk <= 2048)
        return query.new_empty((query.shape[0], 1, topk), dtype=torch.int32)


def quant_lightning_indexer_packed(query, packed_cache, weights, query_scale,
                                  seq_q, seq_k, block_table, topk=2048, sparse_mode=3):
    if not _available:
        raise RuntimeError("Build sgl-kernel-npu for Ascend950 with SGL_KERNEL_ENABLE_QLI_PACKED=ON")
    return torch.ops.npu.sgl_quant_lightning_indexer_packed(
        query, packed_cache, weights, query_scale, seq_q, seq_k, block_table, topk, sparse_mode)
