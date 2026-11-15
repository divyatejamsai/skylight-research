"""HuggingFace adapter implementation for sparse attention."""

import random
import string
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import StaticCache
from transformers.cache_utils import StaticLayer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..sparse_attention.base import SparseAttention, SparseAttentionConfig
from ..sparse_attention.efficient_attention.base import EfficientAttention
from ..sparse_attention.research_attention.base import ResearchAttention
from .base import ModelAdapter, Request, RequestResponse
from .model_servers.huggingface import ModelServerHF
from .utils.config import ModelServerConfig

INT_MAX = 2**31 - 1

# The static-cache decode path (generation_kwargs["graph_decode"]).
# Name its dense attention is registered under.
_DENSE_DECODE_ATTENTION: str = "skylight_dense_decode"
# Free slots the static cache keeps beyond the tokens that will be written to it.
_STATIC_CACHE_MARGIN: int = 8
# Under a CUDA graph the generated tokens are read back once in this many decode steps
# to look for a stop token (a read is a host synchronisation).
_STOP_CHECK_EVERY: int = 16


def _dense_decode_attention(
    module: torch.nn.Module,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float = 1.0,
    dropout: float = 0.0,
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, None]:
    """Dense attention of one decode step on a static cache.

    SDPA with native grouped-query attention: K/V are not repeated per query head, and
    attention_mask limits the preallocated cache to its live tokens.
    """
    out = torch.nn.functional.scaled_dot_product_attention(
        queries, keys, values, attn_mask=attention_mask, scale=scaling, enable_gqa=True
    )
    return out.transpose(1, 2), None


ALL_ATTENTION_FUNCTIONS.register(_DENSE_DECODE_ATTENTION, _dense_decode_attention)


def _graph_available(device: torch.device) -> bool:
    """Whether decode steps on this device can be captured in a CUDA graph."""
    return device.type == "cuda"


class ModelAdapterHF(ModelAdapter):
    """ModelAdapter for HuggingFace integration. Provides concrete implementations for huggingface's
    transformer library.
    """

    def __init__(
        self,
        model_name: str,
        sparse_attention_config: Optional[SparseAttentionConfig],
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        hybrid: Optional[bool] = None,
        revision: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HuggingFace adapter.

        Args:
            model_name: Name of the HuggingFace model to use
            sparse_attention_config: Configuration for sparse attention. If None, adapter runs in dense-only mode.
            model_kwargs: Additional keyword arguments for model creation
            device: Device to run the model on TODO: support dynamic and multipledevice placement
            tokenizer_kwargs: Additional keyword arguments for tokenizer creation
            revision: Optional HuggingFace revision (branch, tag or commit sha) to pin both
                the model weights and the tokenizer to, e.g. ``"stage1-step10000"``. It is
                merged into ``model_kwargs``/``tokenizer_kwargs``, which makes it part of the
                ModelServer cache key so that two revisions of the same model name do not
                collide on a single cached instance. An explicit ``revision`` already present
                in either dict takes precedence.
        """
        super().__init__(model_name, sparse_attention_config, **kwargs)
        self._registered_attention_name: Optional[str] = None
        self._custom_attention_fn: Optional[Callable] = None
        self.model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
        self.tokenizer_kwargs: Dict[str, Any] = dict(tokenizer_kwargs or {})

        self.revision: Optional[str] = revision
        if revision is not None:
            self.model_kwargs.setdefault("revision", revision)
            self.tokenizer_kwargs.setdefault("revision", revision)

        raw_registry_path: Any = kwargs.get("model_registry_path", "")
        self.model_registry_path: str = (
            raw_registry_path if isinstance(raw_registry_path, str) else ""
        )

        raw_allow_unregistered: Any = kwargs.get("allow_unregistered_models", True)
        self.allow_unregistered_models: bool = (
            raw_allow_unregistered if isinstance(raw_allow_unregistered, bool) else True
        )

        # more useful parameters to store
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_dtype = self.model_kwargs.get("torch_dtype", torch.float32)

        # Handle dense-only mode when sparse_attention_config is None
        self._sparse_attention_available: bool = sparse_attention_config is not None
        # Control token-by-token question processing (for hybrid models)
        self.hybrid = hybrid if hybrid is not None else False
        # Convert device string to GPU ID for ModelServer
        gpu_id: Optional[int] = None
        if self.device.startswith("cuda"):
            if self.device == "cuda":
                gpu_id = 0 if torch.cuda.is_available() else None
            else:
                # Extract GPU ID from "cuda:0", "cuda:1", etc.
                try:
                    gpu_id = int(self.device.split(":")[1])
                except (IndexError, ValueError):
                    gpu_id = None

        model_server = ModelServerHF(
            ModelServerConfig(
                model_registry_path=self.model_registry_path,
                allow_unregistered_models=self.allow_unregistered_models,
            )
        )
        self.model = model_server.get_model(self.model_name, gpu_id, self.model_kwargs)
        self.tokenizer = model_server.get_tokenizer(
            self.model_name, self.tokenizer_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.random_separator = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=100)
        )

    def __del__(self) -> None:
        """Clean up registered attention functions when the adapter is destroyed."""
        self._cleanup_attention_registration()

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> RequestResponse:
        """Processes request with optimized tokenization but independent question processing.
        Context is tokenized once but each question is processed independently to avoid KV cache contamination.

        Args:
            request: The request to process
            generation_kwargs: ``max_new_tokens``, and optionally ``graph_decode``. With
                ``graph_decode=True`` the tokens after the first are generated on a
                preallocated cache with the decode step captured in a CUDA graph (see
                ``_decode_on_static_cache``): the same greedy decoding, without the
                per-step launch cost of the eager loop. It needs dense attention or a
                sparse attention with ``supports_static_cache_decode``.

        Returns:
            response: The response to the request
        """
        max_context_length: int = request_kwargs.get("max_context_length", INT_MAX)
        max_new_tokens: int = generation_kwargs.get("max_new_tokens", INT_MAX)
        if generation_kwargs.get("graph_decode"):
            self._check_graph_decode()  # before the prefill, not after it
        print(
            " Processing request with max_context_length: ",
            max_context_length,
            " and max_new_tokens: ",
            max_new_tokens,
            flush=True,
        )

        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )
        context: str = request.context
        answer_prefix: str = request.answer_prefix

        context, questions = self._preprocess_context_and_questions(
            context, questions, answer_prefix
        )

        # `_preprocess_context_and_questions` may have already applied the chat
        # template, which emits the model's BOS itself. Several fast tokenizers
        # (Llama-3.x, Gemma) carry a `TemplateProcessing` post-processor that
        # prepends BOS on *every* `encode()` regardless of `add_bos_token`, so
        # letting `add_special_tokens` default to True duplicates it here and
        # splices a third one in mid-sequence at the question boundary below.
        context_tokens = self.tokenizer.encode(
            context,
            return_tensors="pt",
            add_special_tokens=self.tokenizer.chat_template is None,
        )
        context_tokens = context_tokens[
            :, :max_context_length
        ]  # truncate context to max_context_length

        input_device = (
            self.model.device if hasattr(self.model, "device") else self.device
        )
        if input_device is not None:
            context_tokens = context_tokens.to(input_device)
        print(f"Context tokens: {context_tokens.shape}")
        responses: List[str] = []
        # The prefill's logits are not used. The static-cache path asks for one position
        # only: at 128k tokens the full logits are tens of GB.
        prefill_kwargs: Dict[str, Any] = (
            {"logits_to_keep": 1} if generation_kwargs.get("graph_decode") else {}
        )

        self.model.eval()
        with torch.no_grad():
            for question in questions:
                sparse_meta_data: Dict[str, Any] = {}

                # The question continues the already-BOS-prefixed context; any
                # special token added here lands mid-sequence.
                question_tokens = self.tokenizer.encode(
                    question, return_tensors="pt", add_special_tokens=False
                )
                if input_device is not None:
                    question_tokens = question_tokens.to(input_device)

                if self._sparse_attention_available:
                    context_outputs = self.model(
                        context_tokens,
                        past_key_values=None,
                        use_cache=True,
                        sparse_meta_data=sparse_meta_data,
                        **prefill_kwargs,
                    )

                    with self.enable_sparse_mode():
                        response_text = self._generate_response(
                            question_tokens,
                            context_outputs,
                            sparse_meta_data,
                            generation_kwargs,
                            **kwargs,
                        )
                else:
                    context_outputs = self.model(
                        context_tokens,
                        past_key_values=None,
                        use_cache=True,
                        **prefill_kwargs,
                    )

                    response_text = self._generate_response(
                        question_tokens,
                        context_outputs,
                        sparse_meta_data,
                        generation_kwargs,
                        **kwargs,
                    )

                responses.append(response_text)

        if isinstance(request.questions, str):
            return RequestResponse(responses=responses[0])
        else:
            return RequestResponse(responses=responses)

    def _preprocess_context_and_questions(
        self, context: str, questions: List[str], answer_prefix: str
    ) -> Tuple[str, List[str]]:
        """Preprocess the context and questions -- apply chat template if needed

        Args:
            context: The context to preprocess
            questions: The questions to preprocess
        """
        context = context + self.random_separator
        if self.tokenizer.chat_template is not None:
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )
        new_context = context.split(self.random_separator)[0]
        new_questions = [
            question + context.split(self.random_separator)[1] + answer_prefix
            for question in questions
        ]
        return new_context, new_questions

    def get_custom_attention_function(
        self, sparse_attention: SparseAttention
    ) -> Callable:
        """Returns custom_attention_fn callable with the correct signature required for HuggingFace.

        Args:
            sparse_attention: The sparse attention instance

        Returns:
            custom_attention_fn: Callable with correct signature for HuggingFace
        """

        def custom_attention_callable(
            module: torch.nn.Module,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs: Dict[str, Any],
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Custom attention callable for HuggingFace integration."""
            if hasattr(module, "layer_idx"):
                layer_idx = getattr(module, "layer_idx", None)
                if layer_idx is not None:
                    kwargs["layer_idx"] = layer_idx

            if "sparse_meta_data" in kwargs:
                sparse_meta_data: Dict[Any, Any] = kwargs["sparse_meta_data"]
                kwargs.pop("sparse_meta_data", None)
            else:
                raise ValueError(
                    "sparse_meta_data must be provided while calling model.forward()"
                )

            return sparse_attention.custom_attention(
                module=module,
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=attention_mask,
                scaling=scaling,
                dropout=dropout,
                sparse_meta_data=sparse_meta_data,
                **kwargs,
            )

        return custom_attention_callable

    def _generate_unique_attention_name(self) -> str:
        """Generate a unique name not present in ALL_ATTENTION_FUNCTIONS."""
        base_name: str = "sparse_attention"
        existing_keys: List[str] = (
            ALL_ATTENTION_FUNCTIONS.valid_keys()
            + ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        )

        while True:
            suffix: str = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=8)
            )
            name: str = f"{base_name}_{suffix}"

            if name not in existing_keys:
                return name

    def _ensure_attention_registered(self) -> str:
        """Ensure custom attention function is registered and return the name.
        Caches the registration to avoid repeated registration overhead.

        Returns:
            The name of the registered attention function
        """
        if self._registered_attention_name is None:
            if not self._sparse_attention_available or self.sparse_attention is None:
                raise RuntimeError(
                    "Cannot register attention function: sparse attention is not available"
                )
            self._custom_attention_fn = self.get_custom_attention_function(
                self.sparse_attention
            )
            self._registered_attention_name = self._generate_unique_attention_name()

            from transformers.masking_utils import eager_mask

            ALL_ATTENTION_FUNCTIONS.register(
                self._registered_attention_name, self._custom_attention_fn
            )
            if isinstance(
                self.sparse_attention, (ResearchAttention, EfficientAttention)
            ):
                ALL_MASK_ATTENTION_FUNCTIONS.register(
                    self._registered_attention_name, eager_mask
                )
            else:
                raise NotImplementedError(
                    "Sparse attention is not supported for this model yet"
                )

        return self._registered_attention_name

    def _cleanup_attention_registration(self) -> None:
        """Clean up registered attention functions."""
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name in ALL_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(self._registered_attention_name)
        if (
            self._registered_attention_name is not None
            and self._registered_attention_name
            in ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
        ):
            ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(
                self._registered_attention_name
            )
        self._registered_attention_name = None
        self._custom_attention_fn = None

    @contextmanager
    def enable_sparse_mode(self) -> Generator[None, None, None]:
        """Context manager to temporarily enable sparse attention mode.

        Yields:
            None
        """
        # If sparse attention is not available, raise an error
        if not self._sparse_attention_available:
            raise RuntimeError(
                "Cannot enable sparse mode: sparse attention is not available"
            )

        # Store original implementations to restore later
        original_implementations: Dict[str, str] = {}

        # First, store the original implementations before registering custom attention
        for name, module in self.model.named_modules():
            if hasattr(module, "config") and hasattr(
                module.config, "_attn_implementation"
            ):
                original_implementations[name] = module.config._attn_implementation

        # Ensure custom attention function is registered (reuse if already registered)
        custom_attention_name: str = self._ensure_attention_registered()

        try:
            # Switch to sparse attention
            for name, module in self.model.named_modules():
                if hasattr(module, "config") and hasattr(
                    module.config, "_attn_implementation"
                ):
                    module.config._attn_implementation = custom_attention_name

            yield

        finally:
            # Restore original implementations
            for name, module in self.model.named_modules():
                if name in original_implementations:
                    module.config._attn_implementation = original_implementations[name]

    def _generate_response(
        self,
        question_tokens: torch.Tensor,
        context_outputs: Any,
        sparse_meta_data: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text response.

        For hybrid models, after context prefill, feed the question tokens one-by-one so the
        linear-attention cached path uses seq_len == 1. For dense models, process all question tokens at once.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        max_new_tokens: int = generation_kwargs.get("max_new_tokens", 50)

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        # Start from cached context state
        past_key_values = context_outputs.past_key_values

        last_outputs = None

        if self.hybrid:
            # TODO(sahil, P1): add regression coverage for hybrid models that
            # verifies question/prefix tokens are consumed one-by-one after
            # context prefill, while non-hybrid models keep the batched path.
            # ------------------------------------------------------------------
            # Step 1: consume the question/prefix ONE TOKEN AT A TIME
            # ------------------------------------------------------------------
            for idx in range(question_tokens.shape[1]):
                step_input = question_tokens[:, idx : idx + 1]

                model_kwargs = {
                    "input_ids": step_input,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "logits_to_keep": 1,
                }

                # Only pass sparse_meta_data when sparse mode is actually enabled
                if self._sparse_attention_available:
                    model_kwargs["sparse_meta_data"] = sparse_meta_data

                with torch.no_grad():
                    last_outputs = self.model(**model_kwargs)

                past_key_values = last_outputs.past_key_values
        else:
            # Dense models: process all question tokens at once
            model_kwargs = {
                "input_ids": question_tokens,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            if self._sparse_attention_available:
                model_kwargs["sparse_meta_data"] = sparse_meta_data
            with torch.no_grad():
                last_outputs = self.model(**model_kwargs)
            past_key_values = last_outputs.past_key_values

        if last_outputs is None:
            raise ValueError("Question tokens are empty; cannot generate response")

        generated_ids = [last_outputs.logits[0, -1].argmax()]

        if generation_kwargs.get("graph_decode"):
            # Step 2 on a preallocated cache, the decode step replayed from a CUDA graph.
            generated_ids += self._decode_on_static_cache(
                generated_ids[0],
                past_key_values,
                max_new_tokens - 1,
                should_stop_token_ids,
            )
            return self.tokenizer.decode(
                torch.stack(generated_ids), skip_special_tokens=True
            )

        # ------------------------------------------------------------------
        # Step 2: continue normal autoregressive generation, one token at a time
        # ------------------------------------------------------------------
        for _ in tqdm(range(max_new_tokens - 1), disable=(max_new_tokens < 1000)):
            model_kwargs = {
                "input_ids": generated_ids[-1].unsqueeze(0).unsqueeze(0),
                "past_key_values": past_key_values,
                "use_cache": True,
            }

            if self._sparse_attention_available:
                model_kwargs["sparse_meta_data"] = sparse_meta_data

            with torch.no_grad():
                outputs = self.model(**model_kwargs)

            past_key_values = outputs.past_key_values
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)

            if new_id.item() in should_stop_token_ids:
                break

        answer: str = self.tokenizer.decode(
            torch.stack(generated_ids), skip_special_tokens=True
        )

        return answer

    @contextmanager
    def _attention_implementation(self, name: str) -> Generator[None, None, None]:
        """Temporarily switch every module of the model to a registered attention.

        Args:
            name: Key of the attention function in ALL_ATTENTION_FUNCTIONS.

        Yields:
            None
        """
        modules: List[Any] = [
            module
            for module in self.model.modules()
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

    def _to_static_cache(
        self, past_key_values: Any, n_cached: int, room: int
    ) -> StaticCache:
        """Move a prefill cache into a StaticCache with room for more tokens.

        The source cache is emptied layer by layer as it is copied, so the peak memory is
        one cache and one layer, not two caches.

        Args:
            past_key_values: The cache the prefill returned (per-layer keys and values).
            n_cached: Number of tokens it holds.
            room: Free slots to preallocate after them.

        Returns:
            The static cache, holding the same n_cached tokens.
        """
        cache = StaticCache(config=self.model.config, max_cache_len=n_cached + room)
        if any(type(layer) is not StaticLayer for layer in cache.layers):
            # A sliding-window layer keeps fewer tokens than the sequence and counts them
            # on the host, which a replay would not advance.
            raise NotImplementedError(
                "graph_decode does not support models with sliding-window cache layers"
            )
        for dst, src in zip(cache.layers, past_key_values.layers):
            dst.lazy_initialization(src.keys[:, :, :0])
            dst.keys[:, :, :n_cached] = src.keys[:, :, :n_cached]
            dst.values[:, :, :n_cached] = src.values[:, :, :n_cached]
            src.keys = src.values = None
            if dst.keys.is_cuda:
                # The freed layer is a little too small for the next one to reuse.
                torch.cuda.empty_cache()
        return cache

    def _check_graph_decode(self) -> None:
        """Raise NotImplementedError if this adapter cannot decode with graph_decode."""
        if self.hybrid:
            raise NotImplementedError(
                "graph_decode does not support hybrid models (token-by-token prefill)"
            )
        if self._sparse_attention_available and not getattr(
            self.sparse_attention, "supports_static_cache_decode", False
        ):
            raise NotImplementedError(
                f"{type(self.sparse_attention).__name__} does not support graph_decode: "
                "it needs an attention with supports_static_cache_decode"
            )

    def _decode_on_static_cache(
        self,
        first_token: torch.Tensor,
        past_key_values: Any,
        max_steps: int,
        stop_token_ids: List[int],
    ) -> List[torch.Tensor]:
        """Greedy decoding on a preallocated cache, the step replayed from a CUDA graph.

        The eager loop of ``_generate_response`` launches every kernel of every layer
        from Python at each token, which bounds its speed whatever the attention costs.
        Here the prefill cache is moved into a StaticCache, the first step runs eagerly
        (it builds the sparse attention's per-layer state), and the remaining steps are
        replays of one captured CUDA graph. Generated tokens are read back once in
        ``_STOP_CHECK_EVERY`` steps to look for a stop token; what was generated after
        it is dropped, so the result is the eager loop's. Without CUDA the same steps
        run eagerly.

        Dense attention runs as SDPA with native grouped-query attention. A sparse
        attention must set ``supports_static_cache_decode``: it has to read the sequence
        length from position_ids (the cache is longer than the sequence) and to run
        without host-side reads after its first step.

        Args:
            first_token: The token the prefill produced (0-dim), fed to the first step.
            past_key_values: The prefill cache; it is consumed.
            max_steps: Largest number of tokens to generate after first_token.
            stop_token_ids: Generation ends after the first of these.

        Returns:
            The generated tokens after first_token (0-dim tensors), through the first
            stop token if there is one.
        """
        self._check_graph_decode()
        if max_steps <= 0:
            return []
        n_cached: int = int(past_key_values.get_seq_length())
        cache: StaticCache = self._to_static_cache(
            past_key_values, n_cached, max_steps + _STATIC_CACHE_MARGIN
        )
        device = first_token.device
        token: torch.Tensor = first_token.view(1, 1).clone()
        position: torch.Tensor = torch.full(
            (1,), n_cached, device=device, dtype=torch.long
        )
        slots: torch.Tensor = torch.arange(cache.max_cache_len, device=device)
        # A 4D mask is used as given. The mask HuggingFace would build for a custom
        # attention copies a host scalar, which a CUDA graph cannot capture.
        mask: torch.Tensor = torch.zeros(
            1, 1, 1, cache.max_cache_len, device=device, dtype=torch.bool
        )
        model_kwargs: Dict[str, Any] = {
            "input_ids": token,
            "attention_mask": mask,
            "position_ids": position.view(1, 1),
            "cache_position": position,
            "past_key_values": cache,
            "use_cache": True,
            "logits_to_keep": 1,
        }
        attention: Any
        if self._sparse_attention_available:
            # The caller has enabled sparse mode. A new sequence on a new cache needs a
            # new state.
            model_kwargs["sparse_meta_data"] = {}
            attention = nullcontext()
        else:
            attention = self._attention_implementation(_DENSE_DECODE_ATTENTION)

        def step() -> None:
            # Every tensor is updated in place: a replay reads and writes the same ones.
            mask.copy_((slots <= position).expand_as(mask))
            logits: torch.Tensor = self.model(**model_kwargs).logits[:, -1]
            token.copy_(logits.argmax(-1, keepdim=True))
            position.add_(1)

        # A model without a stop token has None here: nothing stops it but max_steps.
        stop: torch.Tensor = torch.tensor(
            [i for i in stop_token_ids if i is not None],
            device=device,
            dtype=torch.long,
        )
        tokens: List[torch.Tensor] = []
        # A CUDA graph records one device's stream and is replayed on the current
        # device: on another GPU the step would run instead of being recorded, and the
        # replays would do nothing.
        current_device: Any = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with attention, current_device:
            step()
            tokens.append(token.clone())
            checked: int = 0
            graph: Optional[torch.cuda.CUDAGraph] = None
            if _graph_available(device) and max_steps > 1:
                if not bool(torch.isin(tokens[0], stop).any()):
                    graph = torch.cuda.CUDAGraph()
                    # torch's default capture stream is the one of the device the
                    # process captured its first graph on: give it this device's.
                    with torch.cuda.graph(graph, stream=torch.cuda.Stream(device)):
                        step()  # recorded, not run: the state is still the eager step's
            every: int = _STOP_CHECK_EVERY if graph is not None else 1
            while len(tokens) < max_steps:
                if len(tokens) - checked >= every:
                    if bool(torch.isin(torch.cat(tokens[checked:], dim=1), stop).any()):
                        break
                    checked = len(tokens)
                if graph is not None:
                    graph.replay()
                else:
                    step()
                tokens.append(token.clone())
        del graph
        generated: torch.Tensor = torch.cat(tokens, dim=1)[0]
        hits: torch.Tensor = torch.isin(generated, stop).nonzero()
        if hits.numel() > 0:
            generated = generated[: int(hits[0]) + 1]
        return list(generated.unbind(0))
