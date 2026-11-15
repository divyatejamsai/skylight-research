"""Tests for ModelAdapterHF's static-cache decode path (generation_kwargs["graph_decode"]).

A two-layer Qwen2 model with random weights and a word-level tokenizer stand in for a real
model, so the adapter's own code runs end to end. On CPU the static-cache steps run
eagerly; the CUDA tests replay them from a CUDA graph.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Phi3Config,
    Phi3ForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

import sparse_attention_hub.adapters.huggingface as hf_adapter
from sparse_attention_hub.adapters import ModelAdapterHF, Request, static_decode
from sparse_attention_hub.sparse_attention import (
    LocalMaskerConfig,
    ResearchAttentionConfig,
    TwoStageBlockSparseConfig,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 96
# A stop id that is not in the vocabulary: no step can generate it.
NO_STOP = -1


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"w{i}": i for i in range(VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="w0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="w0", eos_token="w1")


def _model(
    device: str = "cpu", dtype: torch.dtype = torch.float32, hidden_size: int = 64
) -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=hidden_size,  # 8 heads: a head dimension of hidden_size / 8
        intermediate_size=2 * hidden_size,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        initializer_range=0.5,  # varied greedy tokens from random weights
    )
    model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype).eval()
    model.generation_config.eos_token_id = NO_STOP
    return model


def _longrope_model(
    original_max: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    sliding_window: int = 262144,
) -> Phi3ForCausalLM:
    """A two-layer Phi3 model as the Phi configs make it (head dimension 32): LongRoPE,
    with one set of RoPE frequencies up to original_max tokens and another beyond,
    chosen at every call, and a sliding window in the config that is longer than any
    sequence, for which transformers builds sliding-window cache layers."""
    torch.manual_seed(0)
    config = Phi3Config(
        vocab_size=VOCAB,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        original_max_position_embeddings=original_max,
        sliding_window=sliding_window,
        rope_scaling={
            "type": "longrope",
            "short_factor": [1.0] * 16,
            "long_factor": [1.0 + 0.25 * i for i in range(16)],
        },
        initializer_range=0.5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Phi3ForCausalLM(config).to(device=device, dtype=dtype).eval()
    model.generation_config.eos_token_id = NO_STOP
    return model


def _adapter(
    model: Qwen2ForCausalLM,
    sparse_attention_config: Optional[Any] = None,
    hybrid: bool = False,
) -> ModelAdapterHF:
    with patch.object(hf_adapter, "ModelServerHF") as server:
        server.return_value.get_model.return_value = model
        server.return_value.get_tokenizer.return_value = _tokenizer()
        return ModelAdapterHF(
            "tiny-random",
            sparse_attention_config,
            device=str(model.device),
            hybrid=hybrid,
        )


def _words(n: int, seed: int) -> str:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, VOCAB - 1, (n,), generator=g).tolist()
    return " ".join(f"w{i}" for i in ids)


def _request(n_context: int = 40, n_question: int = 6) -> Request:
    return Request(
        context=_words(n_context, 1), questions=_words(n_question, 2), answer_prefix=""
    )


def _generate(
    adapter: ModelAdapterHF, request: Request, **generation_kwargs: Any
) -> str:
    return adapter.process_request(request, generation_kwargs, {}).responses


def _ids(text: str) -> List[int]:
    return [int(w[1:]) for w in text.split()]


@pytest.mark.unit
class TestGraphDecodeOnCPU:
    """Without CUDA the static-cache steps run eagerly: the same code but the graph."""

    def test_same_text_as_the_eager_loop(self) -> None:
        adapter = _adapter(_model())
        eager = _generate(adapter, _request(), max_new_tokens=24)
        with patch.object(
            adapter, "_answer_on_static_cache", wraps=adapter._answer_on_static_cache
        ) as path:
            static = _generate(
                adapter, _request(), max_new_tokens=24, graph_decode=True
            )
        path.assert_called_once()
        assert len(_ids(eager)) == 24
        assert len(set(_ids(eager))) > 4  # not one token repeated
        assert static == eager

    @pytest.mark.parametrize("max_new_tokens", [1, 2, 3])
    def test_short_generations(self, max_new_tokens: int) -> None:
        adapter = _adapter(_model())
        eager = _generate(adapter, _request(), max_new_tokens=max_new_tokens)
        static = _generate(
            adapter, _request(), max_new_tokens=max_new_tokens, graph_decode=True
        )
        assert static == eager
        assert len(_ids(static)) == max_new_tokens

    @pytest.mark.parametrize("stop_at", [1, 2, 7, 23])
    def test_stops_after_a_stop_token(self, stop_at: int) -> None:
        model = _model()
        adapter = _adapter(model)
        free = _ids(_generate(adapter, _request(), max_new_tokens=24))
        stop = free[stop_at]
        # The eager loop does not test the first token; neither does the static path.
        first_hit = next(i for i in range(1, 24) if free[i] == stop)
        model.generation_config.eos_token_id = [NO_STOP, stop]
        # skip_special_tokens drops nothing here: the stop token is an ordinary word.
        eager = _ids(_generate(adapter, _request(), max_new_tokens=24))
        static = _ids(
            _generate(adapter, _request(), max_new_tokens=24, graph_decode=True)
        )
        assert eager == free[: first_hit + 1]
        assert static == eager

    def test_model_without_a_stop_token(self) -> None:
        model = _model()
        model.generation_config.eos_token_id = None
        adapter = _adapter(model)
        eager = _generate(adapter, _request(), max_new_tokens=12)
        static = _generate(adapter, _request(), max_new_tokens=12, graph_decode=True)
        assert len(_ids(static)) == 12
        assert static == eager

    def test_several_questions(self) -> None:
        adapter = _adapter(_model())
        request = Request(
            context=_words(40, 1),
            questions=[_words(6, 2), _words(9, 3)],
            answer_prefix="",
        )
        eager = adapter.process_request(request, {"max_new_tokens": 12}, {}).responses
        static = adapter.process_request(
            request, {"max_new_tokens": 12, "graph_decode": True}, {}
        ).responses
        assert isinstance(static, list) and len(static) == 2
        assert static == eager and static[0] != static[1]

    def test_context_is_prefilled_once_for_all_questions(self) -> None:
        model = _model()
        adapter = _adapter(model)
        # A long answer first: the second question starts on a cache that still holds
        # the first question and its answer beyond the context.
        questions = [_words(9, 3), _words(4, 2), _words(6, 5)]
        request = Request(context=_words(40, 1), questions=questions, answer_prefix="")
        kwargs: Dict[str, Any] = {"max_new_tokens": 20, "graph_decode": True}
        with patch.object(
            adapter, "_prefill_static_cache", wraps=adapter._prefill_static_cache
        ) as prefill:
            together = adapter.process_request(request, kwargs, {}).responses
        prefill.assert_called_once()
        alone = [
            adapter.process_request(
                Request(context=_words(40, 1), questions=q, answer_prefix=""),
                kwargs,
                {},
            ).responses
            for q in questions
        ]
        assert together == alone
        eager = adapter.process_request(request, {"max_new_tokens": 20}, {}).responses
        assert together == eager and len(set(together)) == 3

    def test_attention_implementation_is_restored(self) -> None:
        model = _model()
        adapter = _adapter(model)
        before = model.config._attn_implementation
        _generate(adapter, _request(), max_new_tokens=6, graph_decode=True)
        assert model.config._attn_implementation == before

    def test_attention_implementation_is_restored_after_an_error(self) -> None:
        model = _model()
        before = model.config._attn_implementation
        with pytest.raises(RuntimeError, match="in the step"):
            with static_decode.attention_implementation(
                model, static_decode.DENSE_ATTENTION
            ):
                assert (
                    model.config._attn_implementation == static_decode.DENSE_ATTENTION
                )
                raise RuntimeError("in the step")
        assert model.config._attn_implementation == before

    def test_prefill_cache_is_moved_not_copied(self) -> None:
        adapter = _adapter(_model())
        ids = torch.randint(2, VOCAB - 1, (1, 30))
        with torch.no_grad():
            pkv = adapter.model(ids, use_cache=True).past_key_values
        keys = [layer.keys.clone() for layer in pkv.layers]
        cache = static_decode.move_to_static_cache(adapter.model.config, pkv, 30, 10)
        assert static_decode.capacity(cache) == 40
        for layer, src, k in zip(cache.layers, pkv.layers, keys):
            assert src.keys is None and src.values is None
            assert layer.keys.shape[2] == 40
            assert torch.equal(layer.keys[:, :, :30], k)
            assert not layer.keys[:, :, 30:].any()

    def test_two_stage_block_sparse_is_accepted(self) -> None:
        # On CPU TwoStageBlockSparse attends densely, so the text is the dense one: this
        # covers the adapter's sparse branch (sparse mode on, a new sparse_meta_data).
        model = _model()
        dense = _generate(_adapter(model), _request(), max_new_tokens=12)
        sparse_adapter = _adapter(model, TwoStageBlockSparseConfig())
        static = _generate(
            sparse_adapter, _request(), max_new_tokens=12, graph_decode=True
        )
        assert static == dense

    @pytest.mark.parametrize("max_new_tokens", [1, 6])
    def test_other_sparse_attention_is_refused(self, max_new_tokens: int) -> None:
        config = ResearchAttentionConfig(
            masker_configs=[LocalMaskerConfig(window_size=8)]
        )
        model = _model()
        adapter = _adapter(model, config)
        # Refused before the prefill, whatever the number of tokens asked for.
        with patch.object(model, "forward", wraps=model.forward) as forward:
            with pytest.raises(
                NotImplementedError, match="supports_static_cache_decode"
            ):
                _generate(
                    adapter,
                    _request(),
                    max_new_tokens=max_new_tokens,
                    graph_decode=True,
                )
        forward.assert_not_called()

    def test_hybrid_is_refused(self) -> None:
        model = _model()
        adapter = _adapter(model, hybrid=True)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            with pytest.raises(NotImplementedError, match="hybrid"):
                _generate(adapter, _request(), max_new_tokens=6, graph_decode=True)
        forward.assert_not_called()

    def test_through_the_benchmark_runner(self) -> None:
        # generation_kwargs reach process_request through Benchmark's request loop.
        from benchmark.mock_benchmark.mock_benchmark import MockBenchmark

        adapter = _adapter(_model())
        benchmark = MockBenchmark()
        rows = benchmark._load_datasets()
        # Columns the request loop reads that the mock rows do not carry.
        rows["answer_prefix"] = ""
        rows["max_new_tokens"] = 8
        eager = benchmark._process_all_requests(
            adapter, rows, {"max_new_tokens": 8}, {}
        )
        with patch.object(
            adapter, "_answer_on_static_cache", wraps=adapter._answer_on_static_cache
        ) as answer, patch.object(
            adapter, "_prefill_static_cache", wraps=adapter._prefill_static_cache
        ) as prefill:
            static = benchmark._process_all_requests(
                adapter, rows, {"max_new_tokens": 8, "graph_decode": True}, {}
            )
        assert answer.call_count == len(rows) == 5
        assert prefill.call_count == rows["context"].nunique()  # once per context
        assert list(static["predicted_answer"]) == list(eager["predicted_answer"])

    def test_static_cache_layers_never_slide(self) -> None:
        # A window longer than the cache never slides: plain layers, no token count.
        cache = static_decode.static_cache(_longrope_model(100).config, 300)
        assert all(type(layer).__name__ == "StaticLayer" for layer in cache.layers)
        assert all(layer.max_cache_len == 300 for layer in cache.layers)
        plain = static_decode.static_cache(_model().config, 300)
        assert all(type(layer).__name__ == "StaticLayer" for layer in plain.layers)
        # A window shorter than the cache would drop tokens: refused.
        short = _longrope_model(100, sliding_window=64)
        with pytest.raises(NotImplementedError, match="slides"):
            static_decode.static_cache(short.config, 300)

    def test_rope_blocks_graph(self) -> None:
        # LongRoPE: a graph is possible while the frequencies stay the same.
        model = _longrope_model(original_max=100)
        assert not static_decode.rope_blocks_graph(model, 41, 90)
        assert not static_decode.rope_blocks_graph(model, 101, 150)
        assert static_decode.rope_blocks_graph(model, 90, 110)
        assert not static_decode.rope_blocks_graph(_model(), 90, 110)  # plain RoPE

    @pytest.mark.parametrize("n_tokens", [40, 300])
    def test_frozen_rope_keeps_the_last_frequencies(self, n_tokens: int) -> None:
        model = _longrope_model(original_max=100)
        rope = model.model.rotary_emb
        x = torch.zeros(1, 1, 256)
        position_ids = torch.arange(n_tokens)[None]
        cos, sin = rope(x, position_ids)  # chooses the frequencies for n_tokens
        with static_decode.frozen_rope(model):
            assert "forward" in rope.__dict__
            frozen_cos, frozen_sin = rope(x, position_ids)
        assert "forward" not in rope.__dict__
        assert torch.equal(frozen_cos, cos) and torch.equal(frozen_sin, sin)
        # The two sets of frequencies differ, so the freeze is not a no-op.
        other = rope(x, torch.arange(340 - n_tokens)[None])[0]
        assert not torch.equal(other[:, :40], cos[:, :40])

    @pytest.mark.parametrize("original_max", [20, 60, 4096])
    def test_longrope_model_same_text_as_the_eager_loop(
        self, original_max: int
    ) -> None:
        # 46 prompt tokens, 24 new ones: above the limit, across it and below it.
        adapter = _adapter(_longrope_model(original_max))
        eager = _generate(adapter, _request(), max_new_tokens=24)
        static = _generate(adapter, _request(), max_new_tokens=24, graph_decode=True)
        assert len(set(_ids(eager))) > 4
        assert static == eager

    def test_default_is_the_eager_loop(self) -> None:
        adapter = _adapter(_model())
        with patch.object(adapter, "_answer_on_static_cache") as static:
            _generate(adapter, _request(), max_new_tokens=6)
            _generate(adapter, _request(), max_new_tokens=6, graph_decode=False)
        static.assert_not_called()


def _decode_tokens(
    adapter: ModelAdapterHF,
    ids: torch.Tensor,
    max_steps: int,
    stop: List[int],
    use_graph: bool,
) -> List[int]:
    """The engine's decode after a dense prefill of ids, with or without the graph."""
    sparse = adapter._sparse_attention_available
    n = ids.shape[1]
    with torch.no_grad():
        pre = adapter.model(ids, use_cache=True, logits_to_keep=1)
        first = pre.logits[0, -1].argmax()
        cache = static_decode.move_to_static_cache(
            adapter.model.config,
            pre.past_key_values,
            n,
            max_steps + static_decode.CACHE_MARGIN,
        )

        def run() -> torch.Tensor:
            return static_decode.decode_on_static_cache(
                adapter.model,
                cache,
                first,
                n,
                max_steps,
                attention=adapter.enable_sparse_mode() if sparse else None,
                sparse_meta_data={} if sparse else None,
                stop_token_ids=stop,
            ).tokens[0]

        out = _with_graph(run, use_graph)
    return [int(t) for t in out]


def _with_graph(run: Any, use_graph: bool) -> torch.Tensor:
    if use_graph:
        return run()
    # Without a graph the path runs every step eagerly.
    with patch.object(static_decode, "graph_available", lambda device: False):
        return run()


@cuda
@pytest.mark.unit
class TestGraphDecodeOnCUDA:
    """Replays of the captured step give the tokens of eager steps on the same cache."""

    N_CONTEXT = 3000  # 46 blocks of 64 tokens: the kernels select among them
    KERNEL_HIDDEN = 256  # a head dimension of 32; the kernels need at least 16

    def _ids(self) -> torch.Tensor:
        g = torch.Generator().manual_seed(5)
        return torch.randint(2, VOCAB - 1, (1, self.N_CONTEXT), generator=g).cuda()

    @pytest.mark.parametrize("carry_threshold", [0.0, 0.2])
    def test_sparse_replay_matches_eager_steps(self, carry_threshold: float) -> None:
        model = _model("cuda", torch.bfloat16, self.KERNEL_HIDDEN)
        config = TwoStageBlockSparseConfig(carry_threshold=carry_threshold)
        kwargs: Dict[str, Any] = dict(max_steps=50, stop=[NO_STOP])
        graph = _decode_tokens(
            _adapter(model, config), self._ids(), use_graph=True, **kwargs
        )
        eager = _decode_tokens(
            _adapter(model, config), self._ids(), use_graph=False, **kwargs
        )
        assert len(graph) == 50
        assert graph == eager

    def test_dense_replay_matches_eager_steps(self) -> None:
        model = _model("cuda", torch.float32)
        kwargs: Dict[str, Any] = dict(max_steps=50, stop=[NO_STOP])
        graph = _decode_tokens(_adapter(model), self._ids(), use_graph=True, **kwargs)
        eager = _decode_tokens(_adapter(model), self._ids(), use_graph=False, **kwargs)
        assert len(graph) == 50
        assert graph == eager

    @pytest.mark.parametrize("every", [1, 2, 7, 16])
    def test_stop_token_at_every_position(self, every: int) -> None:
        # Tokens are read back once in `every` steps and whatever was generated after
        # the stop token is dropped. Each token of a free run is made the stop token in
        # turn; over the intervals its first position falls on a check, right after one
        # and between two.
        model = _model("cuda", torch.float32)
        adapter = _adapter(model)
        free = _decode_tokens(adapter, self._ids(), 40, [NO_STOP], use_graph=True)
        firsts = sorted({free.index(token) for token in free})
        assert len(firsts) >= 8  # several places to stop at
        with patch.object(static_decode, "STOP_CHECK_EVERY", every):
            for hit in firsts:
                got = _decode_tokens(
                    adapter, self._ids(), 40, [NO_STOP, free[hit]], use_graph=True
                )
                assert got == free[: hit + 1], (every, hit)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
    def test_model_on_another_gpu(self) -> None:
        # The graph has to be recorded on the model's GPU, not on the current one.
        model = _model("cuda:1", torch.float32)
        ids = self._ids().to("cuda:1")
        kwargs: Dict[str, Any] = dict(max_steps=50, stop=[NO_STOP])
        assert torch.cuda.current_device() == 0
        graph = _decode_tokens(_adapter(model), ids, use_graph=True, **kwargs)
        eager = _decode_tokens(_adapter(model), ids, use_graph=False, **kwargs)
        assert torch.cuda.current_device() == 0
        assert len(graph) == 50 and len(set(graph)) > 4
        assert graph == eager

    @pytest.mark.parametrize("sparse", [False, True])
    def test_longrope_replay_matches_eager_steps(self, sparse: bool) -> None:
        # LongRoPE chooses its frequencies on the host at every call: the graphed steps
        # keep the choice of the eager first step.
        dtype = torch.bfloat16 if sparse else torch.float32
        model = _longrope_model(1024, "cuda", dtype)
        config = TwoStageBlockSparseConfig(carry_threshold=0.2) if sparse else None
        kwargs: Dict[str, Any] = dict(max_steps=50, stop=[NO_STOP])
        graph = _decode_tokens(
            _adapter(model, config), self._ids(), use_graph=True, **kwargs
        )
        eager = _decode_tokens(
            _adapter(model, config), self._ids(), use_graph=False, **kwargs
        )
        assert len(graph) == 50 and len(set(graph)) > 4
        assert graph == eager
        assert "forward" not in model.model.rotary_emb.__dict__

    def test_longrope_change_of_frequencies_is_not_graphed(self) -> None:
        # 3000 cached tokens and 50 steps cross the limit of 3020: no graph.
        model = _longrope_model(3020, "cuda", torch.float32)
        adapter = _adapter(model)
        with patch.object(torch.cuda, "CUDAGraph", wraps=torch.cuda.CUDAGraph) as graph:
            crossing = _decode_tokens(adapter, self._ids(), 50, [NO_STOP], True)
            graph.assert_not_called()
            _decode_tokens(adapter, self._ids(), 10, [NO_STOP], True)  # stays below
            graph.assert_called_once()
        eager = _decode_tokens(adapter, self._ids(), 50, [NO_STOP], False)
        assert len(crossing) == 50 and crossing == eager

    @pytest.mark.parametrize("sparse", [False, True])
    def test_same_text_as_the_eager_loop(self, sparse: bool) -> None:
        # The contract of graph_decode, on the GPU: the text of the loop it replaces.
        dtype = torch.bfloat16 if sparse else torch.float32
        model = _model("cuda", dtype, self.KERNEL_HIDDEN)
        config = TwoStageBlockSparseConfig(carry_threshold=0.2) if sparse else None
        adapter = _adapter(model, config)
        request = Request(
            context=_words(self.N_CONTEXT, 1), questions=_words(8, 2), answer_prefix=""
        )
        eager = _generate(adapter, request, max_new_tokens=40)
        static = _generate(adapter, request, max_new_tokens=40, graph_decode=True)
        # Decoding drops the two special words, so the text can be a little shorter.
        assert 30 <= len(_ids(eager)) <= 40 and len(set(_ids(eager))) > 4
        assert static == eager

    def test_min_cached_tokens_decodes_densely(self) -> None:
        # Below min_cached_tokens the sparse attention runs its dense decode step, which
        # is the computation of the dense path.
        model = _model("cuda", torch.float32, self.KERNEL_HIDDEN)
        config = TwoStageBlockSparseConfig(min_cached_tokens=10**6)
        kwargs: Dict[str, Any] = dict(max_steps=30, stop=[NO_STOP], use_graph=True)
        sparse = _decode_tokens(_adapter(model, config), self._ids(), **kwargs)
        dense = _decode_tokens(_adapter(model), self._ids(), **kwargs)
        assert sparse == dense

    def test_small_head_dimension_is_refused(self) -> None:
        model = _model("cuda", torch.bfloat16)  # a head dimension of 8
        adapter = _adapter(model, TwoStageBlockSparseConfig())
        with pytest.raises(NotImplementedError, match="head dimension"):
            _decode_tokens(adapter, self._ids(), 4, [NO_STOP], use_graph=False)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_context_is_prefilled_once_for_all_questions(self, sparse: bool) -> None:
        # Several questions on one prefilled cache give the answers of one request each.
        dtype = torch.bfloat16 if sparse else torch.float32
        model = _model("cuda", dtype, self.KERNEL_HIDDEN)
        config = TwoStageBlockSparseConfig(carry_threshold=0.2) if sparse else None
        adapter = _adapter(model, config)
        context = _words(self.N_CONTEXT, 1)
        questions = [_words(70, 3), _words(5, 2), _words(20, 5)]
        kwargs: Dict[str, Any] = {"max_new_tokens": 40, "graph_decode": True}
        with patch.object(
            adapter, "_prefill_static_cache", wraps=adapter._prefill_static_cache
        ) as prefill:
            together = adapter.process_request(
                Request(context=context, questions=questions, answer_prefix=""),
                kwargs,
                {},
            ).responses
        prefill.assert_called_once()
        alone = [
            adapter.process_request(
                Request(context=context, questions=q, answer_prefix=""), kwargs, {}
            ).responses
            for q in questions
        ]
        assert together == alone and len(set(together)) == 3

    def test_process_request_sparse(self) -> None:
        model = _model("cuda", torch.bfloat16, self.KERNEL_HIDDEN)
        adapter = _adapter(model, TwoStageBlockSparseConfig(carry_threshold=0.2))
        request = Request(
            context=_words(self.N_CONTEXT, 1), questions=_words(8, 2), answer_prefix=""
        )
        text = _generate(adapter, request, max_new_tokens=40, graph_decode=True)
        again = _generate(adapter, request, max_new_tokens=40, graph_decode=True)
        assert len(_ids(text)) == 40
        assert text == again
