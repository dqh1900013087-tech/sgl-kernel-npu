"""Compatibility entry and FakeTensor registration for the bundled Packed QLI."""
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
                                  seq_q, seq_k, block_table, topk=2048, sparse_mode=3,
                                  *, impl="auto"):
    """Compatibility alias; prefer ``sgl_kernel_npu.operators.indexer``.

    ``auto`` selects the bundled AscendC implementation on arch35.
    """
    # Keep registry hardware discovery lazy for unrelated package users.
    from ..operators.indexer import quant_lightning_indexer_packed as public_op

    return public_op(
        query, packed_cache, weights, query_scale, seq_q, seq_k, block_table,
        topk, sparse_mode, impl=impl,
    )
