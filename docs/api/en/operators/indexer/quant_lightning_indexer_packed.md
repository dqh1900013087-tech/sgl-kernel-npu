# quant_lightning_indexer_packed

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| ascendc (also auto) | Unsupported | Ascend950DT, CANN 9.1.0 |
| torch_native | Not registered / not validated | Validated on Ascend950DT |

## Signature

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed

quant_lightning_indexer_packed(
    query, packed_cache, weights, query_scale, seq_q, seq_k, block_table,
    topk=2048, sparse_mode=3, *, impl="auto",
)
```

`auto` selects `ascendc`; there is no automatic fallback. Build with repository-root
`python3 setup.py --arch-version 3510`, or explicitly select
`--ops quant_lightning_indexer_packed`. The legacy `build.sh` does not include it.

## Description

Return DSA Top-k key indices for FP8 queries and paged FP8 keys with FP32
scales. For query token t and key position j, the mathematical score is
`sum_h weights[t,h] * relu(dot(query[t,h] * query_scale[t,h], key[j] * key_scale[j]))`.
The native QLI precision and tie-breaking behavior are retained; the math
expression does not imply bitwise agreement with a full-FP32 reference.
Causal mode masks keys after the query's position at the end of its KV sequence.

Each physical cache page contains all P*128 key bytes followed by P FP32 scales;
it is not an array of 132-byte token records. Inputs are not modified.

## Parameters

All tensor arguments are contiguous ND tensors on the same NPU.

| Parameter | Type, shape and constraints |
| --- | --- |
| query | FP8 e4m3fn `[T,H,128]`; H is 8,16,24,32 or 64. |
| packed_cache | UINT8 `[pages,P*132]`; pages > 0, P is a multiple of 16 in [16,1024]; address is FP32-aligned. |
| weights | BF16 `[T,H]`. |
| query_scale | FP32 `[T,H]`. |
| seq_q | INT32 `[B]`, cumulative query lengths; B > 0. |
| seq_k | INT32 `[B]`, valid KV lengths. |
| block_table | INT32 `[B,max_pages]`, physical page IDs; max_pages > 0. |
| topk | int, 1 through 2048; default 2048. |
| sparse_mode | int, 0 for unmasked or 3 for causal; default 3. |
| impl | str, `auto`, `ascendc`, or `torch_native`; default `auto`. |

Lengths and page IDs must describe valid entries in the supplied buffers;
the adapter validates metadata without reading these device values on the host.
Shape-derived tiling dimensions must fit uint32. Empty T returns an empty output.

## Return value

| Value | Type and shape |
| --- | --- |
| indices | INT32 `[T,1,topk]`, on the query device. |

## AscendC backend

CANN OpDef/inference/tiling and generated ACLNN functions are packaged by the
shared build. The per-operator Torch adapter uses the shared vendor resolver,
records input/output stream usage and allocates graph-compatible workspace.
FakeTensor registration lives in the public module. Tests cover asynchronous
execution and NPU graph replay; full torch.compile and model-level validation
are not claimed. The external separate-cache CANN QLI is the regression
reference; `auto` and `ascendc` are aliases, not independent implementations.

## PyTorch reference

Select `impl="torch_native"` for an independent implementation using ordinary
PyTorch operations: unpack paged FP8 keys and FP32 scales, compute FP32 dot
products, apply ReLU and head weights/scales, round scores to BF16, and select
Top-k. Equal scores prefer smaller logical key indices. Selected indices are
returned in ascending index order; missing entries are padded with -1.
AscendC preserves its native output order, so compare selected sets rather than
positions across backends. Floating-point accumulation near a BF16 rounding
boundary may also affect selection.

The reference reads sequence lengths/page validation values on the host and
synchronizes the device. It is intended for correctness checking, not production
performance, FakeTensor, torch.compile, or NPU Graph capture. `auto` still selects
AscendC without fallback. Tests compare both implementations across page sizes,
head counts, batch requests, mask modes, ties, empty input, and short sequences.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed

T, H, P = 1, 16, 128
q = torch.randn(T, H, 128, device="npu").to(torch.float8_e4m3fn)
k = torch.randn(1, P, 128, device="npu").to(torch.float8_e4m3fn)
ks = torch.ones(1, P, device="npu", dtype=torch.float32)
packed = torch.cat((k.view(torch.uint8).reshape(1, -1), ks.view(torch.uint8)), dim=1)
w = torch.ones(T, H, device="npu", dtype=torch.bfloat16)
qs = torch.ones(T, H, device="npu", dtype=torch.float32)
sq = torch.tensor([T], device="npu", dtype=torch.int32)
sk = torch.tensor([P], device="npu", dtype=torch.int32)
table = torch.tensor([[0]], device="npu", dtype=torch.int32)
indices = quant_lightning_indexer_packed(q, packed, w, qs, sq, sk, table, topk=64, impl="ascendc")
assert indices.shape == (T, 1, 64)
```
