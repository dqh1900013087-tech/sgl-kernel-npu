"""Public API dispatch and compatibility tests for the bundled packed indexer.

The separate-cache CANN operator is the regression baseline here. This does
not introduce a torch_native implementation or claim independent math coverage.
"""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

from sgl_kernel_npu.operators import get_impl, list_impls
from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_packed
from sgl_kernel_npu.operators.utils.registry import ImplementationNotFoundError, get_op_name


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
        (key.view(torch.uint8).reshape(pages, -1), ks.view(torch.uint8).reshape(pages, -1)),
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


@pytest.mark.parametrize("impl", ["auto", "ascendc"])
@pytest.mark.parametrize("query_tokens", [1, 4], ids=["decode", "prefill"])
def test_public_and_compatibility(impl, query_tokens):
    import sgl_kernel_npu

    args, key, ks = generate_data(query_tokens)
    q, _, w, qs, sq, sk, table = args
    actual = quant_lightning_indexer_packed(*args, topk=128, impl=impl)
    expected = torch.ops.npu.npu_quant_lightning_indexer(
        q, key, w, qs, ks, 0, 0,
        actual_seq_lengths_query=sq, actual_seq_lengths_key=sk,
        block_table=table, layout_query="TND", layout_key="PA_BSND",
        sparse_count=128, sparse_mode=3,
    )
    if isinstance(expected, (tuple, list)):
        expected = expected[0]
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    # Existing SGLang calls pass topk and sparse_mode positionally.
    compat = sgl_kernel_npu.quant_lightning_indexer_packed(*args, 128, 3)
    torch.testing.assert_close(compat.cpu(), actual.cpu(), rtol=0, atol=0)
    selected = sgl_kernel_npu.quant_lightning_indexer_packed(*args, 128, 3, impl=impl)
    torch.testing.assert_close(selected.cpu(), actual.cpu(), rtol=0, atol=0)


def test_registry():
    assert get_op_name(quant_lightning_indexer_packed) == "indexer.quant_lightning_indexer_packed"
    assert list_impls(quant_lightning_indexer_packed) == ["ascendc", "auto"]
    assert get_impl(quant_lightning_indexer_packed, "auto") is get_impl(
        quant_lightning_indexer_packed, "ascendc"
    )


@pytest.mark.parametrize("impl", ["torch_native", "unknown"])
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
