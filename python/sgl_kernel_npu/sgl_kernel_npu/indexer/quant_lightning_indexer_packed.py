"""Compatibility module; prefer sgl_kernel_npu.operators.indexer."""
from ..operators.indexer import quant_lightning_indexer_packed

__all__ = ["quant_lightning_indexer_packed"]
