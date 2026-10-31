# Packed Quant Lightning Indexer

Ascend950 FP8 DSA indexing with FP8 keys and FP32 key scales stored in the same
physical page. This operator follows the existing AscendC direct-launch build
and runtime path; it does not generate or load a separate CANN vendor package.

## Build

From the repository root, with the normal build dependencies prepared:

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh -a kernels Ascend950
```

Packed QLI is enabled with the existing A5-only operators. No additional build
flag is required. It is not built for A2/A3. The separate experimental root
`setup.py --arch-version` pipeline does not select this operator.

`cmake/legacy_kernels.cmake` compiles a dedicated `qli_packed_kernel` static
library, preserving the kernel's manual synchronization compiler options, and
links it with its C++ host into `libsgl_kernel_npu.so`. The normal wheel build
packages that shared library. There is no separate `.run`, `libcust_opapi.so`,
`dlopen`, or QLI-specific `ASCEND_CUSTOM_OPP_PATH` initialization.

## Interface

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed

indices = quant_lightning_indexer_packed(
    query, packed_cache, weights, query_scale,
    seq_q, seq_k, block_table, topk=2048, sparse_mode=3,
    impl="auto",
)
```

`auto` and `ascendc` select the same native implementation on arch35.
`sgl_kernel_npu.indexer.quant_lightning_indexer_packed` remains a compatibility
module. Import the function from one of these operator modules; the previous
branch-only top-level `sgl_kernel_npu.quant_lightning_indexer_packed` export has
been removed so the package root `__init__.py` remains identical to main.
FakeTensor registration happens when the public operator module is imported.

Inputs are contiguous ND tensors on one NPU:

- query: FP8 e4m3fn `[T, H, 128]`, H in 8, 16, 24, 32, 64.
- packed_cache: UINT8 `[pages, P * 132]`, FP32-aligned, P a multiple of 16 in
  `[16, 1024]`. Each page contains `[P * 128 FP8 key bytes][P FP32 scales]`.
- weights: BF16 `[T, H]`; query_scale: FP32 `[T, H]`.
- seq_q: INT32 cumulative query lengths `[B]`; seq_k: INT32 key lengths `[B]`.
- block_table: INT32 physical page IDs `[B, max_pages]`.
- topk: 1 through 2048; sparse_mode: 0 (unmasked) or 3 (causal).

Sequence lengths and page IDs must describe valid entries in the supplied
buffers. They are consumed on device, without host synchronization for values.
The output is INT32 `[T, 1, topk]`. Inputs are not modified.

## Execution

The Torch registration calls the C++ host, which validates metadata, determines
blockDim and workspace size, and launches through `EXEC_KERNEL_CMD`. The
Ascend950 FP8/TND/paged specialization is compiled directly. Shape-only tiling
fields are passed as scalar launch arguments and assembled locally in the
kernel, so no mutable global tiling cache or host-to-device tiling copy is
needed during NPU graph capture.

Input, output and workspace allocations are recorded on the current stream
before queueing. The existing launch helper captures raw pointers and scalar
values rather than owning Tensor objects in the task callback.

## Validation

```bash
python3 -m pytest tests/python/sgl_kernel_npu/test_quant_lightning_indexer_packed.py \
  tests/python/operators/indexer/test_quant_lightning_indexer_packed_api.py -v --tb=short
```

Tests compare exact indices with the separate-cache CANN QLI implementation,
exercise invalid metadata, public dispatch, FakeTensor, cross-stream temporary
inputs, and NPU graph replay with updated input values. They also check that
Packed QLI does not load a private vendor library. These standalone tests do
not validate full-model startup or distributed model performance.

## Source

The arch35 computation derives from ops-transformer commit
`aacba10c247cc454768c07704c7d0d026adc3d6d`. Original CANN notices and
`LICENSE.ops-transformer` are retained. `source-manifest.json` records the
adapted source hashes. The host and entry point specialize the original
implementation to the public FP8 packed-cache contract above.
