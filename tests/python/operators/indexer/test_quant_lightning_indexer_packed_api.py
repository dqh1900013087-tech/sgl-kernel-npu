"""Public API dispatch and compatibility tests for the native packed indexer.

The separate-cache CANN operator is an additional regression baseline.
"""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from sgl_kernel_npu.operators import get_impl, list_impls
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed
from sgl_kernel_npu.operators.utils.registry import (
    ImplementationNotFoundError,
    get_op_name,
)


def generate_data(query_tokens):
    torch.npu.manual_seed(42)
    heads, page, pages = 16, 128, 2
    query, qs = torch_npu.npu_dynamic_quant(
        torch.randn(query_tokens * heads, 128, dtype=torch.bfloat16, device="npu"),
        dst_type=torch.float8_e4m3fn,
    )
    key, ks = torch_npu.npu_dynamic_quant(
        torch.randn(pages * page, 128, dtype=torch.bfloat16, device="npu"),
        dst_type=torch.float8_e4m3fn,
    )
    key = key.reshape(pages, page, 1, 128)
    ks = ks.float().reshape(pages, page, 1)
    packed = torch.cat(
        (
            key.view(torch.uint8).reshape(pages, -1),
            ks.view(torch.uint8).reshape(pages, -1),
        ),
        dim=1,
    )
    args = (
        query.reshape(query_tokens, heads, 128),
        packed,
        torch.rand(query_tokens, heads, dtype=torch.bfloat16, device="npu"),
        qs.float().reshape(query_tokens, heads),
        torch.tensor([query_tokens], dtype=torch.int32, device="npu"),
        torch.tensor([255], dtype=torch.int32, device="npu"),
        torch.tensor([[0, 1]], dtype=torch.int32, device="npu"),
    )
    return args, key, ks


@pytest.mark.parametrize("impl", ["auto", "ascendc", "torch_native"])
@pytest.mark.parametrize("query_tokens", [1, 4], ids=["decode", "prefill"])
def test_public_api(impl, query_tokens):

    args, key, ks = generate_data(query_tokens)
    q, _, w, qs, sq, sk, table = args
    actual = quant_lightning_indexer_packed(*args, topk=128, impl=impl)
    expected = torch.ops.npu.npu_quant_lightning_indexer(
        q,
        key,
        w,
        qs,
        ks,
        0,
        0,
        actual_seq_lengths_query=sq,
        actual_seq_lengths_key=sk,
        block_table=table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=128,
        sparse_mode=3,
    )
    if isinstance(expected, (tuple, list)):
        expected = expected[0]
    if impl == "torch_native":
        actual, expected = actual.sort(dim=-1).values, expected.sort(dim=-1).values
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


def test_registry():
    assert (
        get_op_name(quant_lightning_indexer_packed)
        == "indexer.quant_lightning_indexer_packed"
    )
    assert list_impls(quant_lightning_indexer_packed) == [
        "ascendc",
        "auto",
        "torch_native",
    ]
    assert get_impl(quant_lightning_indexer_packed, "auto") is get_impl(
        quant_lightning_indexer_packed, "ascendc"
    )


@pytest.mark.parametrize("impl", ["unknown"])
def test_unavailable_implementation(impl):
    # Dispatch must fail before trying to inspect tensors or launch a kernel.
    with pytest.raises(ImplementationNotFoundError):
        quant_lightning_indexer_packed(*([None] * 7), impl=impl)


def test_fake_public_api():
    from torch._subclasses.fake_tensor import FakeTensorMode

    args, _, _ = generate_data(1)
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(x) for x in args]
        result = quant_lightning_indexer_packed(*fake, topk=128, impl="ascendc")
        assert result.shape == (1, 1, 128)
        assert result.dtype == torch.int32


@pytest.mark.parametrize("capture", [False, True], ids=["async", "graph-replay"])
def test_transient_inputs_lifetime(capture):
    """Release cross-stream temporaries, then enqueue allocation-heavy ops.

    Tensor-owning task callbacks used to deadlock while the queue recycled a
    callback under its lock and Tensor destruction tried to record an event.
    Run this regression under an external timeout to detect that failure.
    """
    import gc

    args, _, _ = generate_data(4)
    producer = torch.npu.Stream()
    consumer = torch.npu.Stream()
    producer.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(producer):
        inputs = [x.clone() for x in args]
    consumer.wait_stream(producer)

    def run():
        # Many queued calls exercise callback reuse, including temporary Tensor
        # destruction. Quantization after each call matches the failing model path.
        for _ in range(24):
            result = quant_lightning_indexer_packed(
                *[x.clone() for x in inputs], topk=128
            )
            torch_npu.npu_dynamic_quant(inputs[2].float().to(torch.bfloat16))
        return result

    with torch.npu.stream(consumer):
        expected = quant_lightning_indexer_packed(*inputs, topk=128)
        run()
    consumer.synchronize()
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=consumer):
            actual = run()
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        # Replay must consume updated input values, not stale temporary buffers.
        inputs[2].zero_()
        updated = quant_lightning_indexer_packed(*inputs, topk=128)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), updated.cpu(), rtol=0, atol=0)
        del graph
    else:
        with torch.npu.stream(consumer):
            actual = run()
        consumer.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    del inputs, actual, expected
    gc.collect()
    torch.npu.synchronize()


def test_cross_stream_callback_release():
    """The callback must not own the last reference to foreign-stream tensors."""
    args, _, _ = generate_data(4)
    expected = quant_lightning_indexer_packed(*args, topk=128)
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    producer.wait_stream(torch.npu.current_stream())
    for _ in range(128):
        with torch.npu.stream(producer):
            temporary = [x.clone() for x in args]
        consumer.wait_stream(producer)
        with torch.npu.stream(consumer):
            actual = quant_lightning_indexer_packed(*temporary, topk=128)
            del temporary
            # Force callback queue reuse and release on subsequent dispatches.
            torch_npu.npu_dynamic_quant(args[2].clone())
        producer.wait_stream(consumer)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize(
    "heads,page", [(8, 16), (16, 128), (24, 16), (32, 128), (64, 1024)]
)
def test_torch_native_reference(heads, page, sparse_mode):
    from test_quant_lightning_indexer_packed import generate_data as generate_paged

    args, _, _ = generate_paged(
        batch=2, qlen=3, kv=511, heads=heads, page=page, topk=128
    )
    expected = quant_lightning_indexer_packed(
        *args, 128, sparse_mode, impl="torch_native"
    )
    actual = quant_lightning_indexer_packed(*args, 128, sparse_mode, impl="ascendc")
    torch.testing.assert_close(
        actual.cpu().sort(dim=-1).values,
        expected.cpu().sort(dim=-1).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("sparse_mode", [0, 3])
def test_torch_native_ties_padding_and_empty(sparse_mode):
    args, _, _ = generate_data(4)
    args[2].zero_()
    args[5].fill_(2)
    for impl in ("ascendc", "torch_native"):
        actual = quant_lightning_indexer_packed(*args, 8, sparse_mode, impl=impl).cpu()
        expected = torch.full((4, 1, 8), -1, dtype=torch.int32, device="cpu")
        for row in range(4):
            count = 2 if sparse_mode == 0 else max(0, row - 1)
            expected[row, 0, :count] = torch.arange(
                count, dtype=torch.int32, device="cpu"
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    args = list(args)
    for index in (0, 2, 3):
        args[index] = args[index][:0]
    assert quant_lightning_indexer_packed(*args, impl="torch_native").shape == (
        0,
        1,
        2048,
    )


def test_torch_native_does_not_call_custom_kernel(monkeypatch):
    args, _, _ = generate_data(1)

    def fail(*args, **kwargs):
        raise AssertionError("reference must not call custom QLI")

    monkeypatch.setattr(torch.ops.npu, "sgl_quant_lightning_indexer_packed", fail)
    monkeypatch.setattr(torch.ops.npu, "npu_quant_lightning_indexer", fail)
    assert quant_lightning_indexer_packed(*args, topk=8, impl="torch_native").shape == (
        1,
        1,
        8,
    )


@pytest.mark.parametrize("impl", ["ascendc", "torch_native"])
def test_hand_computed_scores(impl):
    # Logical keys [-2, 1, 3, 2] have scales [1, 4, 1, 2]. With
    # positive queries and unit head weights, the best scores tie at keys 1, 3.
    q = torch.ones((1, 8, 128), device="npu").to(torch.float8_e4m3fn)
    key = torch.zeros((1, 16, 128), device="npu")
    key[0, :4] = torch.tensor([-2.0, 1.0, 3.0, 2.0], device="npu")[:, None]
    key = key.to(torch.float8_e4m3fn)
    scale = torch.ones((1, 16), device="npu")
    scale[0, :4] = torch.tensor([1.0, 4.0, 1.0, 2.0], device="npu")
    packed = torch.cat(
        (key.view(torch.uint8).reshape(1, -1), scale.view(torch.uint8)), dim=1
    )
    args = (
        q,
        packed,
        torch.ones((1, 8), dtype=torch.bfloat16, device="npu"),
        torch.ones((1, 8), device="npu"),
        torch.tensor([1], dtype=torch.int32, device="npu"),
        torch.tensor([4], dtype=torch.int32, device="npu"),
        torch.tensor([[0]], dtype=torch.int32, device="npu"),
    )
    actual = (
        quant_lightning_indexer_packed(*args, topk=2, impl=impl)
        .cpu()
        .sort(dim=-1)
        .values
    )
    torch.testing.assert_close(
        actual, torch.tensor([[[1, 3]]], dtype=torch.int32, device="cpu")
    )
