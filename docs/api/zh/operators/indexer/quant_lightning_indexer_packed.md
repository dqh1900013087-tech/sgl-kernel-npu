# quant_lightning_indexer_packed

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| ascendc（也是 auto） | 不支持 | Ascend950DT，CANN 9.1.0 |
| torch_native | 未注册、未验证 | 已在 Ascend950DT 验证 |

## 签名

```python
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed

quant_lightning_indexer_packed(
    query, packed_cache, weights, query_scale, seq_q, seq_k, block_table,
    topk=2048, sparse_mode=3, *, impl="auto",
)
```

`auto` 选择 `ascendc`，没有自动回退。使用仓库根目录的
`python3 setup.py --arch-version 3510` 构建，也可通过
`--ops quant_lightning_indexer_packed` 单独选择。旧 `build.sh` 不包含此算子。

## 描述

对 FP8 query 和分页 FP8 key 计算 DSA Top-k 索引，反量化 scale 为 FP32。
数学分数为
`sum_h weights[t,h] * relu(dot(query[t,h] * query_scale[t,h], key[j] * key_scale[j]))`。
保留原生 QLI 的计算精度及同分处理行为；该公式不表示与全 FP32 参考逐位相同。
因果模式将 query 视为对应 KV 序列末尾的 query 序列，并屏蔽未来位置。

每页先存 P*128 个 key 字节，再存 P 个 FP32 scale，不能按每 token 132 字节交错排列。
输入不会被修改。

## 参数

所有张量必须为同一 NPU 上连续的 ND 张量。

| 参数 | 类型、形状及约束 |
| --- | --- |
| query | FP8 e4m3fn `[T,H,128]`，H 为 8、16、24、32 或 64。 |
| packed_cache | UINT8 `[pages,P*132]`，pages > 0；P 在 [16,1024] 内且为 16 的倍数；地址按 FP32 对齐。 |
| weights | BF16 `[T,H]`。 |
| query_scale | FP32 `[T,H]`。 |
| seq_q | INT32 `[B]`，累积 query 长度，B > 0。 |
| seq_k | INT32 `[B]`，有效 KV 长度。 |
| block_table | INT32 `[B,max_pages]`，物理页号，max_pages > 0。 |
| topk | int，范围 [1,2048]，默认 2048。 |
| sparse_mode | int，0 为无掩码，3 为因果模式，默认 3。 |
| impl | str，`auto`、`ascendc` 或 `torch_native`，默认 `auto`。 |

长度及页号必须对应输入缓冲区内的有效条目。适配层检查元数据，不在 host 读取这些设备值。
形状推导出的 tiling 维度须可用 uint32 表示；T 为 0 时返回空输出。

## 返回值

| 返回值 | 类型及形状 |
| --- | --- |
| indices | INT32 `[T,1,topk]`，位于 query 所在设备。 |

## AscendC 后端

由统一构建流程生成并打包 CANN 算子定义、推导、tiling 和 ACLNN 接口。
算子目录内的 Torch adapter 复用公共 vendor 加载器，记录输入输出 stream 使用情况，
并使用支持 graph 的 workspace 分配方式。FakeTensor 注册位于公开接口模块。
测试覆盖异步执行与 NPU graph replay，不代表已验证完整 torch.compile 或模型运行。
精度回归参考是独立缓存版 CANN QLI；`auto` 与 `ascendc` 是同一实现的别名。

## PyTorch 参考实现

通过 `impl="torch_native"` 选择独立的 PyTorch 实现：解包分页 FP8 key 和 FP32
scale，使用 FP32 计算点积、ReLU、head 权重和反量化 scale，将分数转为 BF16 后
选取 Top-k。同分时优先选择较小的逻辑 key 索引，结果按索引升序输出，不足部分填 -1。
AscendC 保留原生输出顺序，因此跨实现比较选中集合，不要求排列相同；
BF16 舍入边界附近的浮点累加差异也可能影响选择。

参考实现读取序列长度及页号校验值时会同步到 host，用于正确性验证，
不用于生产性能、FakeTensor、torch.compile 或 NPU Graph capture。
`auto` 仍选择 AscendC，不自动回退。测试覆盖不同页大小、head 数、batch、
掩码模式、同分、空输入和短序列下两种实现的结果。

## 示例

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
