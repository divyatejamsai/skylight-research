"""Greedy decoding on a preallocated cache, the decode step replayed from a CUDA graph.

HuggingFace's own loop launches every kernel of every layer from Python at each token.
At long context that launch cost bounds the step whatever the attention costs, so a
faster attention does not make it faster. Here the prefill cache is moved into a
``StaticCache``, one decode step runs eagerly, that step is recorded in one
``torch.cuda.CUDAGraph``, and the remaining tokens are replays of it.

Everything a caller needs is in this module, because each of these was found the hard
way and has to hold wherever a decode step is captured:

* The attention mask is a 4D boolean tensor updated in place. The mask HuggingFace would
  build for a custom attention copies a host scalar, which a graph cannot record.
* A graph records one device's stream and is replayed on the current device. The step is
  recorded and replayed with the model's device current, on a stream of that device
  (torch's default capture stream belongs to the first device a process captured on).
* LongRoPE (the Phi models) picks one of two sets of frequencies with a comparison on the
  host at every call: ``frozen_rope`` keeps the choice of the eager step, and
  ``rope_blocks_graph`` says when that is not valid.
* For a config that names a sliding window, transformers builds cache layers that count
  their tokens in a Python integer: ``static_cache`` replaces them by plain layers when
  the window can never slide.
* The tokens that end an answer are the model's own list, not only the tokenizer's eos:
  ``end_token_ids``.
* A sparse attention keeps per-sequence state in ``sparse_meta_data``: a new sequence on
  a new cache needs a new dictionary, and the attention must read the sequence length
  from ``position_ids`` (the cache is longer than the sequence).
"""

import functools
import inspect
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence

import torch
from transformers import StaticCache
from transformers.cache_utils import StaticLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Name the dense decode attention is registered under.
DENSE_ATTENTION: str = "skylight_dense_decode"
# Free slots a static cache keeps beyond the tokens that will be written to it.
CACHE_MARGIN: int = 8
# Under a graph the generated tokens are read back once in this many steps to look for an
# end token (a read is a host synchronisation).
STOP_CHECK_EVERY: int = 16


def _dense_attention(  # pylint: disable=unused-argument
    module: torch.nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float = 1.0,
    dropout: float = 0.0,
    **kwargs: Dict[str, Any],
) -> Any:
    """Dense attention of one decode step on a static cache.

    SDPA with native grouped-query attention: K/V are not repeated per query head, and
    attention_mask limits the preallocated cache to its live tokens.
    """
    out = torch.nn.functional.scaled_dot_product_attention(
        queries, keys, values, attn_mask=attention_mask, scale=scaling, enable_gqa=True
    )
    return out.transpose(1, 2), None


ALL_ATTENTION_FUNCTIONS.register(DENSE_ATTENTION, _dense_attention)


def graph_available(device: torch.device) -> bool:
    """Whether decode steps on this device can be recorded in a CUDA graph."""
    return bool(device.type == "cuda")


def end_token_ids(model: torch.nn.Module, tokenizer: Any = None) -> List[int]:
    """The tokens that end an answer: the model's own list, then the tokenizer's eos.

    Phi ends a turn with ``<|end|>``, which is in its generation config and is not its
    tokenizer's eos token.

    Args:
        model: The HuggingFace model.
        tokenizer: Its tokenizer, if its eos token should be included.

    Returns:
        The ids, without None and without repeats.
    """
    own: Any = model.generation_config.eos_token_id
    ids: List[Any] = list(own) if isinstance(own, (list, tuple)) else [own]
    if tokenizer is not None:
        ids.append(tokenizer.eos_token_id)
    return list(dict.fromkeys(int(i) for i in ids if i is not None))


def static_cache(config: Any, max_cache_len: int) -> StaticCache:
    """A StaticCache of max_cache_len tokens whose layers never slide.

    For a model whose config names a sliding window (the Phi models: 262,144 tokens),
    transformers builds sliding-window layers, which count their tokens in a Python
    integer. A CUDA graph replay does not advance that count, and a compiled layer is
    recompiled whenever it changes. A window that is not shorter than the cache never
    slides, so such layers are replaced by plain static ones, which do the same thing
    without the count. A window shorter than the cache is refused.

    Args:
        config: The model's config.
        max_cache_len: Number of tokens to preallocate.

    Returns:
        The cache, every layer a plain StaticLayer.
    """
    cache = StaticCache(config=config, max_cache_len=max_cache_len)
    for index, layer in enumerate(cache.layers):
        # The sliding-window layer is a subclass: only the exact class is plain.
        if type(layer) is StaticLayer:  # pylint: disable=unidiomatic-typecheck
            continue
        if getattr(layer, "is_sliding", False) and layer.max_cache_len == max_cache_len:
            cache.layers[index] = StaticLayer(max_cache_len=max_cache_len)
        else:
            raise NotImplementedError(
                "graph_decode does not support a cache that slides or chunks within "
                f"the sequence ({type(layer).__name__} of {layer.max_cache_len} tokens "
                f"for {max_cache_len})"
            )
    return cache


def _initialize_layer(layer: Any, keys: torch.Tensor, values: torch.Tensor) -> None:
    """Allocate a static layer for keys and values like these (empty tensors will do).

    transformers 4 takes the keys alone, transformers 5 the keys and the values.
    """
    if len(inspect.signature(layer.lazy_initialization).parameters) == 1:
        layer.lazy_initialization(keys)
    else:
        layer.lazy_initialization(keys, values)


def capacity(cache: StaticCache) -> int:
    """Number of token slots of a cache from ``static_cache``."""
    return int(cache.layers[0].max_cache_len)


def set_cached_length(cache: StaticCache, n_cached: int) -> None:
    """Tell a static cache that it holds n_cached tokens.

    transformers 5 keeps the number of tokens of a static layer in a tensor on the
    device and writes new tokens after it; transformers 4 writes at the positions it is
    given and keeps no count. The count is set in place, because a recorded graph and
    compiled code hold its address.

    Args:
        cache: A cache from ``static_cache`` / ``move_to_static_cache``.
        n_cached: Tokens it holds; the next ones are written after them.
    """
    for layer in cache.layers:
        length = getattr(layer, "cumulative_length", None)
        if isinstance(length, torch.Tensor):
            length.fill_(n_cached)


def move_to_static_cache(
    config: Any,
    past_key_values: Any,
    n_cached: int,
    room: int,
    batch: int = 1,
    device: Optional[torch.device] = None,
    keep_source: bool = False,
) -> StaticCache:
    """Put the first sequence of a prefill cache into a StaticCache with room to grow.

    The source is emptied layer by layer as it is copied, so the peak memory is one
    cache and one layer, not two caches (unless keep_source).

    Args:
        config: The model's config.
        past_key_values: The cache a prefill returned (per-layer keys and values).
        n_cached: Number of tokens it holds.
        room: Free slots to preallocate after them.
        batch: Rows of the new cache; each gets the source's first sequence.
        device: Device of the new cache (the source's by default; the source may wait
            on the CPU).
        keep_source: Leave the source as it is, for another copy.

    Returns:
        The static cache, holding n_cached tokens in every row.
    """
    cache = static_cache(config, n_cached + room)
    for dst, src in zip(cache.layers, past_key_values.layers):
        target = device if device is not None else src.keys.device
        _initialize_layer(
            dst,
            src.keys.new_empty(batch, src.keys.shape[1], 0, src.keys.shape[3]).to(
                target
            ),
            src.values.new_empty(batch, src.values.shape[1], 0, src.values.shape[3]).to(
                target
            ),
        )
        dst.keys[:, :, :n_cached] = src.keys[:1, :, :n_cached].to(target)
        dst.values[:, :, :n_cached] = src.values[:1, :, :n_cached].to(target)
        if not keep_source:
            src.keys = src.values = None
            if target.type == "cuda":
                # The freed layer is a little too small for the next one to reuse.
                torch.cuda.empty_cache()
    set_cached_length(cache, n_cached)
    return cache


def rope_blocks_graph(model: torch.nn.Module, first_len: int, last_len: int) -> bool:
    """Whether the model's RoPE rules out a CUDA graph over these sequence lengths.

    Some RoPE variants choose their frequencies from the sequence length at every call,
    with a comparison made on the host that a CUDA graph cannot record. LongRoPE (the
    Phi models) has two sets of frequencies, below and above the length the model was
    first trained at: a graph is possible while the choice stays the same (see
    ``frozen_rope``). Dynamic RoPE recomputes them as the sequence grows: no graph.

    Args:
        model: The HuggingFace model.
        first_len: Sequence length at the first decode step.
        last_len: Sequence length at the last one.

    Returns:
        True if the decode steps must run eagerly.
    """
    for module in model.modules():
        rope_type: Any = getattr(module, "rope_type", None)
        if not isinstance(rope_type, str):
            continue
        if "dynamic" in rope_type:
            return True
        if rope_type == "longrope":
            config: Any = module.config
            limit: int = getattr(
                config, "original_max_position_embeddings", None
            ) or getattr(config, "max_position_embeddings")
            if (first_len > limit) != (last_len > limit):
                return True
    return False


@contextmanager
def frozen_rope(model: torch.nn.Module) -> Generator[None, None, None]:
    """Keep a LongRoPE model's frequencies as its last call left them.

    Inside the context the RoPE modules skip the per-call choice of frequencies (a
    host-side comparison), so a decode step can be recorded in a CUDA graph. Use it after
    an eager step at the first sequence length, and only when ``rope_blocks_graph`` is
    False for the lengths that follow. Models with other RoPE types are not touched.

    Args:
        model: The HuggingFace model.

    Yields:
        None
    """
    frozen: List[torch.nn.Module] = []
    for module in model.modules():
        forward: Any = getattr(type(module), "forward", None)
        if getattr(module, "rope_type", None) == "longrope" and hasattr(
            forward, "__wrapped__"
        ):
            # The forward under its decorators: the frequency update is one of them.
            plain: Callable = torch.no_grad()(inspect.unwrap(forward))
            module.forward = functools.partial(plain, module)
            frozen.append(module)
    try:
        yield
    finally:
        for module in frozen:
            del module.forward


@contextmanager
def attention_implementation(
    model: torch.nn.Module, name: str
) -> Generator[None, None, None]:
    """Temporarily switch every module of the model to a registered attention.

    Args:
        model: The HuggingFace model.
        name: Key of the attention function in ALL_ATTENTION_FUNCTIONS.

    Yields:
        None
    """
    modules: List[Any] = [
        module
        for module in model.modules()
        if hasattr(getattr(module, "config", None), "_attn_implementation")
    ]
    original: List[str] = [m.config._attn_implementation for m in modules]
    try:
        for module in modules:
            module.config._attn_implementation = name
        yield
    finally:
        for module, implementation in zip(modules, original):
            module.config._attn_implementation = implementation


def compile_decoder_layers(
    model: torch.nn.Module, dynamic: Optional[bool] = False
) -> None:
    """torch.compile every decoder layer (fused norms, RoPE, activations, residuals).

    Call it after the prefill: the layers are compiled for the shapes of a decode step.
    The attention functions have to stay out of the compiled graphs: decode with
    ``opaque_attention=True``.

    Args:
        model: The HuggingFace model.
        dynamic: torch.compile's ``dynamic``. False compiles for the exact shapes, so a
            cache of another length is compiled again.
    """
    torch._dynamo.config.cache_size_limit = 256  # layer_idx specializes per layer
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    for layer in model.model.layers:
        layer.compile(mode="max-autotune-no-cudagraphs", dynamic=dynamic)


def keep_attention_out_of_compiled_graphs() -> None:
    """Make every registered attention function a graph break for torch.compile, so
    compiled layers run around it. Call it once the attention to be used is registered.
    """
    mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
    for name, fn in mapping.items():
        if not getattr(fn, "_opaque", False):
            mapping[name] = torch.compiler.disable(fn)
            mapping[name]._opaque = True


def forward_on_static_cache(
    model: torch.nn.Module,
    cache: StaticCache,
    tokens: torch.Tensor,
    n_cached: int,
    attention: Any = None,
    sparse_meta_data: Optional[Dict[Any, Any]] = None,
) -> torch.Tensor:
    """Run several tokens through the model on a static cache (eagerly).

    The tokens are written at positions n_cached, n_cached + 1, ... of the cache, which
    holds n_cached tokens; whatever the cache holds from there on (an earlier question
    and its answer) is not attended and is overwritten. This is how a question is added
    to a context that was prefilled once.

    Args:
        model: The HuggingFace model.
        cache: A cache from ``static_cache`` / ``move_to_static_cache`` (one sequence).
        tokens: [1, q] token ids.
        n_cached: Tokens the cache holds before them.
        attention: A context manager that selects the attention, as in
            ``decode_on_static_cache``. None: the dense decode attention.
        sparse_meta_data: Passed to the model when given.

    Returns:
        The logits after the last token, [vocab].
    """
    q = tokens.shape[1]
    assert tokens.shape[0] == 1 and q >= 1 and capacity(cache) >= n_cached + q
    set_cached_length(cache, n_cached)
    positions = torch.arange(n_cached, n_cached + q, device=tokens.device)
    slots = torch.arange(capacity(cache), device=tokens.device)
    # Token j attends the slots up to its own: a 4D mask is used as given.
    mask = (slots[None, :] <= positions[:, None])[None, None]
    model_kwargs: Dict[str, Any] = {
        "input_ids": tokens,
        "attention_mask": mask,
        "position_ids": positions[None],
        "cache_position": positions,
        "past_key_values": cache,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    if sparse_meta_data is not None:
        model_kwargs["sparse_meta_data"] = sparse_meta_data
    if attention is None:
        attention = attention_implementation(model, DENSE_ATTENTION)
    with torch.no_grad(), attention:
        logits: torch.Tensor = model(**model_kwargs).logits[0, -1].float()
    return logits


@dataclass
class DecodeOutput:
    """What ``decode_on_static_cache`` returns.

    Attributes:
        tokens: The top token of every step that ran, [batch, steps run]. With stop
            tokens (batch 1) it ends with the first of them.
        first_logits: The logits of the first step for the first sequence, [vocab].
        logits: The logits of every step for the first sequence, [steps, vocab], when
            asked for.
        steps: Steps that ran, the eager first one included.
        replay_seconds: Wall-clock time of the steps after the first (replays, or eager
            steps without a graph), the stop checks among them included.
        used_graph: Whether those steps were graph replays.
        stopped: Whether a stop token was generated.
    """

    tokens: torch.Tensor
    first_logits: torch.Tensor
    logits: Optional[torch.Tensor]
    steps: int
    replay_seconds: float
    used_graph: bool
    stopped: bool


def decode_on_static_cache(
    model: torch.nn.Module,
    cache: StaticCache,
    first_token: torch.Tensor,
    n_cached: int,
    steps: int,
    attention: Any = None,
    sparse_meta_data: Optional[Dict[Any, Any]] = None,
    batch: int = 1,
    stop_token_ids: Sequence[int] = (),
    stop_check_every: Optional[int] = None,
    forced_tokens: Optional[torch.Tensor] = None,
    record_logits: bool = False,
    opaque_attention: bool = False,
) -> DecodeOutput:
    """Greedy decoding on a static cache: one eager step, then replays of its graph.

    Step t is fed the top token of step t - 1 (first_token at t = 0), or
    forced_tokens[t - 1] when forced_tokens is given (teacher forcing: the step's own
    top token is still what is returned). Without CUDA, or when the model's RoPE rules
    a graph out, the same steps run eagerly.

    Args:
        model: The HuggingFace model.
        cache: A cache from ``static_cache`` / ``move_to_static_cache`` holding n_cached
            tokens in each of its batch rows, with room for steps more.
        first_token: The token the prefill produced, fed to the first step.
        n_cached: Tokens in the cache.
        steps: Number of decode steps.
        attention: A context manager entered around the steps that selects the
            attention (ModelAdapterHF.enable_sparse_mode(), or
            ``attention_implementation``). None: the dense decode attention.
        sparse_meta_data: Passed to the model when given: a new dictionary for a sparse
            attention's per-sequence state.
        batch: Sequences decoded together; every one is fed the same tokens.
        stop_token_ids: Decoding ends after the first of these (batch 1).
        stop_check_every: Steps between two looks for a stop token under a graph
            (STOP_CHECK_EVERY by default); without a graph every step is checked.
        forced_tokens: [steps] tokens to feed in place of the steps' own.
        record_logits: Keep the first sequence's logits of every step.
        opaque_attention: The decoder layers are torch.compile'd: keep the registered
            attention functions out of the compiled graphs.

    Returns:
        A DecodeOutput.
    """
    assert steps >= 1 and capacity(cache) >= n_cached + steps, (n_cached, steps)
    set_cached_length(cache, n_cached)
    device = first_token.device
    token = first_token.reshape(-1)[:1].view(1, 1).expand(batch, 1).clone()
    position = torch.full((1,), n_cached, device=device, dtype=torch.long)
    index = torch.zeros(1, device=device, dtype=torch.long)  # the step being run
    slots = torch.arange(capacity(cache), device=device)
    # A 4D mask is used as given (see the module docstring).
    mask = torch.zeros(batch, 1, 1, slots.numel(), device=device, dtype=torch.bool)
    tops = torch.zeros(steps, batch, device=device, dtype=torch.long)
    kept: Dict[str, torch.Tensor] = {}
    model_kwargs: Dict[str, Any] = {
        "input_ids": token,
        "attention_mask": mask,
        "position_ids": position.view(1, 1).expand(batch, 1),
        "cache_position": position,
        "past_key_values": cache,
        "use_cache": True,
        "logits_to_keep": 1,
    }
    if sparse_meta_data is not None:
        model_kwargs["sparse_meta_data"] = sparse_meta_data
    if attention is None:
        attention = attention_implementation(model, DENSE_ATTENTION)

    def step() -> None:
        # Every tensor is updated in place: a replay reads and writes the same ones.
        mask.copy_((slots <= position).expand_as(mask))
        logits = model(**model_kwargs).logits[:, -1].float()
        if "first" not in kept:  # the eager first step
            kept["first"] = logits[0].clone()
            if record_logits:
                kept["logits"] = torch.empty(steps, logits.shape[-1], device=device)
        if record_logits:
            kept["logits"].index_copy_(0, index, logits[:1])
        top = logits.argmax(-1)
        tops.index_copy_(0, index, top.view(1, batch))
        if forced_tokens is None:
            token.copy_(top.view(batch, 1))
        else:
            token.copy_(
                forced_tokens.index_select(0, index).view(1, 1).expand(batch, 1)
            )
        position.add_(1)
        index.add_(1)

    stop = torch.tensor(list(stop_token_ids), device=device, dtype=torch.long)
    cuda = device.type == "cuda"
    graph: Optional[torch.cuda.CUDAGraph] = None
    done, stopped, seconds = 1, False, 0.0
    on_device = torch.cuda.device(device) if cuda else nullcontext()
    with torch.no_grad(), attention, on_device:
        if opaque_attention:
            keep_attention_out_of_compiled_graphs()
        step()
        capture = (
            graph_available(device)
            and steps > 1
            and not rope_blocks_graph(model, n_cached + 1, n_cached + steps)
            and not bool(torch.isin(tops[0], stop).any())
        )
        with frozen_rope(model) if capture else nullcontext():
            if capture:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=torch.cuda.Stream(device)):
                    step()  # recorded, not run: the state is still the eager step's
            every = (stop_check_every or STOP_CHECK_EVERY) if graph is not None else 1
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            checked = 0
            while done < steps:
                if stop.numel() > 0 and done - checked >= every:
                    if bool(torch.isin(tops[checked:done, 0], stop).any()):
                        break
                    checked = done
                if graph is not None:
                    graph.replay()
                else:
                    step()
                done += 1
            if cuda:
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
    used_graph = graph is not None
    del graph
    tokens = tops[:done].t().clone()
    if stop.numel() > 0:
        hits = torch.isin(tokens[0], stop).nonzero()
        if hits.numel() > 0:
            tokens = tokens[:, : int(hits[0]) + 1]
            stopped = True
    return DecodeOutput(
        tokens=tokens,
        first_logits=kept["first"],
        logits=kept.get("logits"),
        steps=done,
        replay_seconds=seconds,
        used_graph=used_graph,
        stopped=stopped,
    )
