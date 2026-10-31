# Packed Quant Lightning Indexer

Ascend950 FP8 DSA indexing with FP8 keys and FP32 scales in one physical cache
page. Supports both the shared CANN/ACLNN build introduced by PR #880 and
the legacy direct-launch build on Ascend950.

## Build

```bash
source /usr/local/Ascend/cann/set_env.sh
# All migrated native operators supported on Ascend950 (includes Packed QLI).
python3 setup.py --arch-version 3510 -j 8
# Or select only this operator; clean when changing the operator selection.
python3 setup.py --arch-version 3510 --ops quant_lightning_indexer_packed --make_clean -j 8
```

The wheel is written to `output/`. Install it into the Python environment used
by the consumer. Selected wheels replace one another; they do not accumulate
operators. The new wheel includes migrated native operators only, not all
legacy native bindings. For SGLang with NEXTN, build the legacy wheel to retain
`build_tree_kernel_efficient` and the other existing native operators:

```bash
bash build.sh -a kernels Ascend950
python3 -m pip install --force-reinstall --no-deps \
  output/sgl_kernel_npu-2026.9.0-cp312-cp312-linux_aarch64.whl
```

Use the exact legacy wheel filename printed by the build (version/Python tags
may vary), not a wildcard that also selects a previous refactor wheel.
`build.sh` itself is unchanged. Explicit QLI selection on arch2201 is rejected;
the legacy build includes QLI only in its A5 operator group.

## Integration

- `csrc/indexer/CMakeLists.txt` discovers declarations in this domain.
- This directory's `CMakeLists.txt` declares ID, arch3510 support, sources and
  the kernel's manual synchronization compiler options.
- `op_host/` defines the CANN operator, output inference and tiling.
- `op_kernel/` contains the Ascend950 FP8/TND/paged specialization.
- `torch_adapter.cpp` validates Tensor metadata, invokes generated ACLNN APIs,
  and registers the Torch schema and NPU implementation locally.

The adapter reuses `csrc/aclnn_torch_adapter/aclnn_runtime.h` to resolve symbols
from the shared `_cann/vendors/sgl_kernel_npu` vendor. The existing package
initializer prepares that vendor before loading the unified Torch library.
There is no QLI-specific root import, environment variable or private vendor.

## Legacy integration

The legacy implementation lives separately in
`csrc/quant_lightning_indexer_packed/op_host` and `op_kernel`, following the
layout of the legacy `apply_token_bitmask` operator. It has its own computation
headers and does not include sources from this migrated directory. The legacy
CMake target and the existing A5 Torch registration blocks select that version.
The CANN/ACLNN build selects only this directory. Each wheel registers QLI once.
The legacy directory and its build/registration entries can be removed later
without modifying this migrated implementation.

## Public interface

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed
indices = quant_lightning_indexer_packed(
    query, packed_cache, weights, query_scale,
    seq_q, seq_k, block_table, topk=2048, sparse_mode=3, impl="auto",
)
```

`auto` and `ascendc` select the native implementation packaged in the wheel
on arch35 (ACLNN or direct launch). Neither the old
`indexer/` compatibility module nor a top-level function export is provided.
See the [English API](../../../docs/api/en/operators/indexer/quant_lightning_indexer_packed.md)
and [Chinese API](../../../docs/api/zh/operators/indexer/quant_lightning_indexer_packed.md)
for the supported contract.

The callback captures ACL descriptors and raw pointers, not Tensor owners.
Inputs and output are recorded on the execution stream; torch_npu's workspace
allocator handles workspace and graph lifetime. Validation targets CANN 9.1.0,
Torch 2.10, torch_npu 2.10.post6, Python 3.12 and Ascend950DT.

## Tests

```bash
python3 -m pytest tests/python/operators/indexer -v --tb=short
```

Tests compare exact indices against the separate-cache CANN QLI implementation,
including mask modes, heads, page sizes, invalid metadata, FakeTensor,
cross-stream temporary inputs and graph replay with updated inputs. An independent
`torch_native` implementation additionally checks selected Top-k sets using
PyTorch FP32 scoring and BF16 ranking, including ties and short sequences.
The separate-cache CANN reference shares upstream algorithm lineage. Full-model startup,
distributed execution and end-to-end performance are not covered.

## Source

The arch35 computation derives from ops-transformer commit
`aacba10c247cc454768c07704c7d0d026adc3d6d`. Original CANN copyright and license
notices remain in the source files. Host inference and tiling specialize the
original implementation to the public FP8 packed-cache contract.
