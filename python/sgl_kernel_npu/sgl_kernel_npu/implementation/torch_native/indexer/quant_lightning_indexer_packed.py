"""Independent eager PyTorch reference for packed FP8 DSA indexing."""

import torch


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
):
    """Compute scores in FP32 and rank their BF16 values, as native QLI does.

    This reference reads sequence metadata on the host and is not graph-safe.
    It uses ordinary PyTorch operations, without calling a custom QLI operator.
    """
    tensors = (query, packed_cache, weights, query_scale, seq_q, seq_k, block_table)
    if any(x.device != query.device or not x.is_contiguous() for x in tensors):
        raise RuntimeError("all inputs must be contiguous and on the same device")
    if query.dtype != torch.float8_e4m3fn or query.ndim != 3 or query.shape[-1] != 128:
        raise RuntimeError("query must be float8_e4m3fn [T, heads, 128]")
    if query.shape[1] not in (8, 16, 24, 32, 64):
        raise RuntimeError("query head count must be 8, 16, 24, 32 or 64")
    if packed_cache.dtype != torch.uint8 or packed_cache.ndim != 2:
        raise RuntimeError("packed_cache must be uint8 [pages, P*132]")
    pages, width = packed_cache.shape
    page = width // 132
    if pages == 0 or width % 132 or not 16 <= page <= 1024 or page % 16:
        raise RuntimeError("page size must be a multiple of 16 in [16, 1024]")
    if packed_cache.data_ptr() % 4:
        raise RuntimeError("packed_cache must be FP32 aligned")
    if weights.dtype != torch.bfloat16 or query_scale.dtype != torch.float32:
        raise RuntimeError("weights must be bfloat16 and query_scale must be float32")
    if weights.shape != query.shape[:2] or query_scale.shape != weights.shape:
        raise RuntimeError("weights/query_scale must have shape [T, heads]")
    if any(x.dtype != torch.int32 for x in (seq_q, seq_k, block_table)):
        raise RuntimeError("sequence lengths and block_table must be int32")
    if seq_q.ndim != 1 or seq_k.shape != seq_q.shape or seq_q.numel() == 0:
        raise RuntimeError("seq_q/seq_k must have shape [batch] with positive batch")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != seq_q.numel()
        or block_table.shape[1] == 0
    ):
        raise RuntimeError("block_table must have shape [batch, max_pages]")
    if not 0 < topk <= 2048 or sparse_mode not in (0, 3):
        raise RuntimeError("topk must be in [1, 2048] and sparse_mode must be 0 or 3")
    output = torch.full(
        (query.shape[0], 1, topk), -1, dtype=torch.int32, device=query.device
    )
    if query.shape[0] == 0:
        return output
    ends, lengths = seq_q.tolist(), seq_k.tolist()
    if ends[-1] != query.shape[0] or any(a > b for a, b in zip([0] + ends, ends)):
        raise RuntimeError("seq_q must be cumulative query lengths ending at T")
    # Copy the sliced byte regions before dtype reinterpretation: their row
    # stride includes the other region of each packed page.
    keys = (
        packed_cache[:, : page * 128]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(pages, page, 128)
    )
    scales = packed_cache[:, page * 128 :].contiguous().view(torch.float32)
    start = 0
    for batch, (end, length) in enumerate(zip(ends, lengths)):
        if length < 0 or length > block_table.shape[1] * page:
            raise RuntimeError("seq_k exceeds block_table capacity")
        ids = block_table[batch, : (length + page - 1) // page].long()
        if ids.numel() and ((ids < 0).any().item() or (ids >= pages).any().item()):
            raise RuntimeError("block_table page ID is out of range")
        key = keys[ids].reshape(-1, 128)[:length]
        scale = scales[ids].reshape(-1)[:length]
        # One query at a time bounds temporary score storage to H * KV length.
        for token in range(start, end):
            valid = length if sparse_mode == 0 else max(0, length - (end - token) + 1)
            count = min(topk, valid)
            if count:
                dots = torch.matmul(query[token].float(), key[:valid].T).relu()
                weighted = weights[token].float() * query_scale[token]
                scores = (dots * weighted[:, None]).sum(dim=0) * scale[:valid]
                scores = scores.to(torch.bfloat16).float()
                indices = torch.argsort(scores, descending=True, stable=True)[:count]
                output[token, 0, :count] = indices.sort().values.to(torch.int32)
        start = end
    return output
