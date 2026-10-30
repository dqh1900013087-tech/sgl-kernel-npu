# Bundled Packed Quant Lightning Indexer

This operator implements Ascend950 FP8 DSA indexing with K and its FP32 scale
stored in the same physical page. The Python interface remains
`sgl_kernel_npu.quant_lightning_indexer_packed`.

The wheel contains the AscendC implementation, host tiling, generated ACLNN API,
and compiled device kernels. An external `qli_packed_transformer` installation
is **not required**. A compatible base CANN SDK/runtime, torch/torch_npu, and
Ascend950 hardware are still required. Validated with CANN 9.1.0, torch 2.10,
torch_npu 2.10.post6, Python 3.12, and Ascend950DT.

## Build and install

From the repository root, after preparing the repository's normal dependencies:

```bash
source /usr/local/Ascend/cann/set_env.sh
export SGL_KERNEL_ENABLE_QLI_PACKED=ON
bash build.sh -a kernels Ascend950

# Run this when you want to replace the installed package:
python3 -m pip install --force-reinstall --no-deps \
  output/sgl_kernel_npu-2026.9.0-cp312-cp312-linux_aarch64.whl
python3 -m pytest tests/python/sgl_kernel_npu/test_quant_lightning_indexer_packed.py -v --tb=short
```

The CMake option defaults to ON for Ascend950 and OFF for A2/A3. Explicitly
enabling it for A2/A3 fails configuration. This integration uses the existing
`build.sh` pipeline; the separate experimental root `setup.py --arch-version`
pipeline does not select Packed QLI.

No `LD_LIBRARY_PATH` entry pointing at `qli_packed_transformer` is needed.
`QLI_PACKED_OPAPI_LIBRARY` is no longer used. Importing `sgl_kernel_npu` adds its
private vendor to `ASCEND_CUSTOM_OPP_PATH`, preserving other existing entries.
The adapter resolves its opapi relative to its own shared library, so the wheel
can be installed at a different path from its build directory.

## Implementation

```text
op_host/                                Torch adapter, declaration, tiling and shape inference
op_host/quant_lightning_indexer_host_utils.h  Small shape diagnostic helper; SDK logging
op_kernel/                              Packed QLI entry and shared QLI types
op_kernel/arch35/                       Ascend950 implementation only
op_kernel/common/                       Required arch35 LightningIndexer/V2 helpers
CMakeLists.txt                          Generated API, host and kernel build
stage.py                                Stage the generated vendor into the wheel
```

The internal operator name is `SglQuantLightningIndexerPacked`, distinct from
the system `QuantLightningIndexer`. The build creates an intermediate CANN
package and extracts it under the build directory; it never installs into
system CANN. The wheel contains it under:

```text
sgl_kernel_npu/lib/qli_packed/vendors/sgl_qli_packed/
```

The implementation comes from the validated `ops-transformer` packed source at
commit `aacba10c247cc454768c07704c7d0d026adc3d6d`. See
`source-manifest.json` for the imported files and adapted-source hashes.
Original CANN copyright notices and the upstream license are retained.
The unused arch22 kernels and the copied opbase logging source tree were removed;
host logging uses the base CANN SDK headers. The existing repository LightningIndexer
is the A2/A3 implementation and does not supply these arch35 helper headers.
No build-time download or external ops-transformer source directory is needed.

## Interface and validation

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed

indices = quant_lightning_indexer_packed(
    query, packed_cache, weights, query_scale,
    seq_q, seq_k, block_table, topk, sparse_mode,
    impl="auto",
)
```

The public operator uses the unified registry and declares arch35 support.
`impl="auto"` and `impl="ascendc"` select the same bundled backend; no
`torch_native` implementation is registered. The existing
`sgl_kernel_npu.quant_lightning_indexer_packed(...)` entry remains compatible
and forwards to this public API, including the optional `impl` keyword.
Public dispatch tests are in
`tests/python/operators/indexer/test_quant_lightning_indexer_packed_api.py`.

The cache is UINT8 `[pages, P * 132]`. Each page consists of all `P * 128`
FP8 key bytes followed by all `P` FP32 scales. The format is not an array of
132-byte token records. Query is FP8 `[T, H, 128]`, weights are BF16 `[T, H]`,
query scale is FP32 `[T, H]`, and sequence metadata/block tables are INT32.
Page size is a multiple of 16 in `[16, 1024]`; top-k is in `[1, 2048]`.

Migration validation unpacked the wheel into a separate directory without
replacing the installed package. In a private mount namespace, the old
`qli_packed_transformer` / `qli_base_transformer` vendors, original
ops-transformer source, and this repository's source/build directories were
hidden. The operator's 19 tests and the modified SGLang's 8 standalone DSA
integration tests passed. The latter use small tensors, not a model server.
Late import after NPU initialization also passed. Full-model performance and
distributed SGLang execution are not covered by these checks.


## Task queue lifetime

The ACLNN callback must not capture `at::Tensor` objects. A model graph warmup
was observed releasing those captures while holding the torch_npu task-queue
mutex; Tensor destruction recorded an NPU event and tried to acquire the same
mutex again. The adapter now uses torch_npu's exported `GetWorkSpaceAddr`,
`InitExecSubTheadCtx`, and `OpCommand::RunOpApiV2` interfaces. Input/output
stream usage is recorded outside the callback; only ACL descriptors, raw
workspace address, executor and stream are captured.

Validation on the current torch_npu 2.10.post6 / Ascend950 environment: 38
standalone tests passed, including cross-stream temporary inputs and repeated
NPU graph replay with changed input data. These small tests also passed with
the old binary and do not reproduce the full-model deadlock. Full SGLang startup
has intentionally not been rerun after this change. Other torch_npu releases
must provide the same exported ACLNN helper interfaces.
