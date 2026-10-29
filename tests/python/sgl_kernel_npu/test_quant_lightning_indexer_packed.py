"""Ascend 950 FP8 integration checks; requires installed Packed QLI CANN vendor."""
import importlib.util
import pytest
import torch
import torch_npu

_original = torch_npu.npu_quant_lightning_indexer
import sgl_kernel_npu


def inputs(batch=1, kv=8192, heads=16, page=128, qlen=1, topk=2048, offset=0):
    g = torch.Generator().manual_seed(20261029)
    per_request = (kv + page - 1) // page
    pages = batch * per_request
    def fp8(shape):
        raw = torch.randint(16, 96, shape, generator=g, dtype=torch.uint8)
        raw |= torch.randint(0, 2, shape, generator=g, dtype=torch.uint8) * 128
        return raw.view(torch.float8_e4m3fn)
    q = fp8((batch * qlen, heads, 128))
    k = fp8((pages, page, 1, 128))
    scale = torch.rand((pages, page, 1), generator=g) * .1 + .01
    w = torch.rand((batch * qlen, heads), generator=g).to(torch.bfloat16)
    qs = torch.rand((batch * qlen, heads), generator=g) * .1 + .01
    sq = torch.arange(1, batch + 1, dtype=torch.int32) * qlen
    sk = torch.tensor([max(topk, kv - i * 17) for i in range(batch)], dtype=torch.int32)
    table = torch.randperm(pages, generator=g).reshape(batch, per_request).int()
    packed = torch.empty((pages, page * 132), dtype=torch.uint8)
    packed[:, :page * 128] = k.view(torch.uint8).reshape(pages, page * 128)
    packed[:, page * 128:] = scale.view(torch.uint8).reshape(pages, page * 4)
    q, k, scale, w, qs, sq, sk, table, packed = [x.npu() for x in (q, k, scale, w, qs, sq, sk, table, packed)]
    if offset:
        backing = torch.empty(packed.numel() + offset, dtype=torch.uint8, device=q.device)
        view = backing[offset:].view_as(packed)
        view.copy_(packed)
        packed = view
    return [q, packed, w, qs, sq, sk, table], k, scale


def compare(args, k, scale, topk):
    actual = sgl_kernel_npu.quant_lightning_indexer_packed(*args, topk, 3)
    q, _, w, qs, sq, sk, table = args
    expected = torch.ops.npu.npu_quant_lightning_indexer(
        q, k, w, qs, scale, 0, 0, actual_seq_lengths_query=sq,
        actual_seq_lengths_key=sk, block_table=table, layout_query="TND",
        layout_key="PA_BSND", sparse_count=topk, sparse_mode=3)
    if isinstance(expected, (tuple, list)):
        expected = expected[0]
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [
    {}, dict(batch=2, kv=511, topk=256), dict(batch=8, kv=1024, heads=64, topk=256),
    dict(page=16, kv=257, topk=256), dict(page=1024, kv=2049, topk=2048),
    dict(qlen=4), dict(offset=4),
])
def test_exact_topk(shape):
    args, k, scale = inputs(**shape)
    compare(args, k, scale, shape.get("topk", 2048))


def test_nondefault_stream():
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        args, k, scale = inputs(kv=511, topk=256)
        compare(args, k, scale, 256)


@pytest.mark.parametrize("bad", ["cache_dtype", "weight_dtype", "scale_dtype", "seq_dtype", "offset", "topk", "cpu_query", "noncontiguous"])
def test_rejected(bad):
    args, _, _ = inputs(kv=256, topk=256, offset=1 if bad == "offset" else 0)
    index_dtype = {"cache_dtype": (1, torch.int8), "weight_dtype": (2, torch.float32),
                   "scale_dtype": (3, torch.float16), "seq_dtype": (4, torch.int64)}
    if bad in index_dtype:
        index, dtype = index_dtype[bad]
        args[index] = args[index].to(dtype)
    if bad == "cpu_query":
        args[0] = args[0].cpu()
    if bad == "noncontiguous":
        args[0] = args[0][:, ::2, :]
    with pytest.raises(RuntimeError):
        sgl_kernel_npu.quant_lightning_indexer_packed(*args, 0 if bad == "topk" else 256, 3)


def test_empty_and_fake():
    args, _, _ = inputs(kv=256, topk=256)
    for index in (0, 2, 3):
        args[index] = args[index][:0]
    assert sgl_kernel_npu.quant_lightning_indexer_packed(*args, 256, 3).shape == (0, 1, 256)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(x) for x in args]
        result = torch.ops.npu.sgl_quant_lightning_indexer_packed(*fake, 256, 3)
        assert result.shape == (0, 1, 256) and result.dtype == torch.int32


def test_existing_registration():
    assert torch_npu.npu_quant_lightning_indexer is _original
    assert importlib.util.find_spec("qli_packed") is None
    assert hasattr(torch.ops.npu, "helloworld")
    x = torch.ones(1024, dtype=torch.float16, device="npu")
    torch.testing.assert_close(torch.ops.npu.helloworld(x, x).cpu(), torch.full((1024,), 2., dtype=torch.float16))


def test_vendor_library():
    from pathlib import Path
    args, k, scale = inputs(kv=256, topk=256)
    compare(args, k, scale, 256)
    assert 'qli_packed_transformer/op_api/lib/libcust_opapi.so' in Path('/proc/self/maps').read_text()
