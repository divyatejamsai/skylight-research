"""Decode speed of ModelAdapterHF.process_request, Skylight's own request loop.

scripts/hf_graph_decode_bench.py times a decode loop of its own. This script times the
loop every Skylight benchmark goes through, with and without
generation_kwargs["graph_decode"]:

  * eager_dense, eager_sparse: the loop process_request has always used (a growing
    cache, every step launched from Python);
  * graph_dense, graph_sparse: the static-cache path (the decode step replayed from a
    CUDA graph). Dense is an adapter without sparse attention; sparse is
    TwoStageBlockSparse with the --sparse overrides.

One request per mode: a wikitext context of --ctx tokens and a short question, with stop
tokens disabled so that every mode generates exactly --new_tokens tokens. With
--compile_layers the graph modes also pass generation_kwargs["compile_layers"], and the
decode steps run through torch.compile'd decoder layers.

--rounds N repeats the graph modes N times in one process, a round being one request per
mode, with the context --round_step tokens shorter in each round. That is a benchmark
loop in small: later requests come after several graph captures and at cache lengths
the compiled layers have not seen. results[mode]["rounds"] has every round;
results[mode] itself is the first.

What is timed:
  * graph modes: the replays, from the first one to the return of the decode. Also
    reported: the prefill, and the fixed cost before the first replay (moving the
    prefill into the static cache, the eager first step, the capture).
  * eager modes: three requests, of --new_tokens, --short_tokens and --new_tokens
    tokens. The per-token time is the slope between the last two, which cancels what a
    request costs before its first step. The first request is reported apart
    (first_ms_per_token): it is the first time the process sees these sequence lengths.

Reading the other fields:
  * prefill_s runs from the start of the request to the start of generation: the chat
    template, tokenising, the copy to the GPU and the context forward. Graph modes ask
    the prefill for one position of logits and eager modes keep all of them, so
    prefill_s and peak_gib differ between the two kinds by design.
  * setup_s (decode start to first replay) is measured once per mode in a fresh process:
    it includes what the process pays once, such as loading the Triton kernels. It is an
    upper bound on what a later request pays.
  * generated_tokens is the decoded text tokenised again, a check on its length.

A process captures one graph per graph mode and round. The fourth and later capture in
a process is slower (about 0.8 ms per step, 0.2 ms with compiled layers), so the first
round of a fresh process and a later round are two different figures; --order sets
which mode goes first. With --check the graph_sparse request is repeated with every
step eager, and the two texts must be equal.

Example:
    python scripts/hf_request_bench.py --ctx 131072 --modes graph_dense,graph_sparse \
        --sparse "carry_threshold=0.2" --out results/request_bench.json
"""

import gc
import json
import time
from typing import Any, Dict, List, Optional

import fire
import torch
from datasets import load_dataset
from hf_graph_decode_bench import _config

from sparse_attention_hub.adapters import ModelAdapterHF, Request, static_decode

QUESTION = "\n\nContinue the text above in the same style."


class _Clock:
    """Wall-clock marks around the adapter's work on a request: the prefill (with the
    move into the static cache), the answer, the engine's decode and its first replay.
    """

    def __init__(self) -> None:
        self.marks: Dict[str, float] = {}
        self.replays: int = 0

    @staticmethod
    def now() -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    def reset(self) -> None:
        self.marks, self.replays = {}, 0

    def _timed(self, fn: Any, name: str) -> Any:
        clock = self

        def timed(*args: Any, **kwargs: Any) -> Any:
            clock.marks[name + "_in"] = clock.now()
            out = fn(*args, **kwargs)
            clock.marks[name + "_out"] = clock.now()
            return out

        return timed

    def install(self, adapter: ModelAdapterHF) -> None:
        # graph_decode: the prefill and the answer are two methods of the adapter.
        adapter._prefill_static_cache = self._timed(  # type: ignore[method-assign]
            adapter._prefill_static_cache, "prefill"
        )
        adapter._answer_on_static_cache = self._timed(  # type: ignore[method-assign]
            adapter._answer_on_static_cache, "generate"
        )
        # The default loop: the prefill is inside process_request, before this.
        adapter._generate_response = self._timed(  # type: ignore[method-assign]
            adapter._generate_response, "generate"
        )


def _time_decode(clock: _Clock) -> None:
    """Mark the engine's decode, count its CUDAGraph replays and mark the first one."""
    static_decode.decode_on_static_cache = clock._timed(
        static_decode.decode_on_static_cache, "decode"
    )
    replay = torch.cuda.CUDAGraph.replay

    def timed_replay(graph: torch.cuda.CUDAGraph) -> None:
        if clock.replays == 0:
            clock.marks["first_replay"] = clock.now()
        clock.replays += 1
        replay(graph)

    torch.cuda.CUDAGraph.replay = timed_replay  # type: ignore[method-assign]


def _run(
    adapter: ModelAdapterHF,
    clock: _Clock,
    request: Request,
    ctx: int,
    new_tokens: int,
    graph: bool,
    compile_layers: bool = False,
) -> Dict[str, Any]:
    clock.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = clock.now()
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": new_tokens,
        "graph_decode": graph,
    }
    if graph and compile_layers:
        generation_kwargs["compile_layers"] = True
    text = adapter.process_request(
        request, generation_kwargs, {"max_context_length": ctx}
    ).responses
    t1 = clock.now()
    m = clock.marks
    res: Dict[str, Any] = dict(
        ctx=ctx,
        new_tokens=new_tokens,
        total_s=t1 - t0,
        prefill_s=m["generate_in"] - t0,
        generate_s=m["generate_out"] - m["generate_in"],
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        text=text,
    )
    if graph:
        res["replays"] = clock.replays
        # The context forward and the move into the static cache, without the chat
        # template and the tokenising that prefill_s also covers.
        res["prefill_forward_s"] = m["prefill_out"] - m["prefill_in"]
        res["setup_s"] = m["first_replay"] - m["decode_in"]
        res["replay_s"] = m["decode_out"] - m["first_replay"]
        res["ms_per_token"] = 1000 * res["replay_s"] / clock.replays
    return res


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    ctx: int = 131072,
    new_tokens: int = 512,
    short_tokens: int = 128,
    modes: str = "graph_dense,graph_sparse",
    sparse: str = "",
    order: str = "dense_first",
    check: bool = False,
    compile_layers: bool = False,
    rounds: int = 1,
    round_step: int = 0,
    out: Optional[str] = None,
) -> None:
    cfg = _config(sparse)
    model_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    adapters: Dict[str, ModelAdapterHF] = dict(
        dense=ModelAdapterHF(model, None, model_kwargs=model_kwargs, device="cuda"),
        sparse=ModelAdapterHF(model, cfg, model_kwargs=model_kwargs, device="cuda"),
    )
    assert (
        adapters["dense"].model is adapters["sparse"].model
    )  # one copy of the weights
    tokenizer = adapters["dense"].tokenizer
    # No stop token: every mode generates new_tokens tokens.
    adapters["dense"].model.generation_config.eos_token_id = -1
    text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    ids = tokenizer(text[: ctx * 8], add_special_tokens=False).input_ids[: ctx - 64]
    request = Request(
        context=tokenizer.decode(ids), questions=QUESTION, answer_prefix=""
    )
    # The prompt as process_request builds it: the chat template around the context,
    # cut at ctx tokens, then the question and the rest of the template.
    templated, questions = adapters["dense"]._preprocess_context_and_questions(
        request.context, [QUESTION], ""
    )
    n_context = min(
        ctx,
        len(
            tokenizer.encode(
                templated, add_special_tokens=tokenizer.chat_template is None
            )
        ),
    )
    n_question = len(tokenizer.encode(questions[0], add_special_tokens=False))
    clock = _Clock()
    _time_decode(clock)
    for adapter in adapters.values():
        clock.install(adapter)

    mode_list: List[str] = modes.split(",") if isinstance(modes, str) else list(modes)
    assert order in ("dense_first", "sparse_first"), order
    mode_list.sort(key=lambda mode: mode.endswith("dense") != (order == "dense_first"))
    results: Dict[str, Dict[str, Any]] = {}
    for mode in mode_list:
        kind, attention = mode.split("_")
        adapter = adapters[attention]
        if kind == "graph":
            res = _run(adapter, clock, request, ctx, new_tokens, True, compile_layers)
        else:
            first = _run(adapter, clock, request, ctx, new_tokens, graph=False)
            short = _run(adapter, clock, request, ctx, short_tokens, graph=False)
            long = _run(adapter, clock, request, ctx, new_tokens, graph=False)
            res = dict(long)
            res["short"] = {k: v for k, v in short.items() if k != "text"}
            res["first"] = {k: v for k, v in first.items() if k != "text"}
            res["ms_per_token"] = (
                1000
                * (long["generate_s"] - short["generate_s"])
                / (new_tokens - short_tokens)
            )
            # What a request costs before its first step, from the last two requests.
            fixed_s = short["generate_s"] - short_tokens * res["ms_per_token"] / 1000
            res["first_ms_per_token"] = (
                1000 * (first["generate_s"] - fixed_s) / new_tokens
            )
            res["same_text_first_and_last"] = first["text"] == long["text"]
        res["generated_tokens"] = len(
            tokenizer(res["text"], add_special_tokens=False).input_ids
        )
        results[mode] = res
        print(
            mode,
            {k: round(v, 4) for k, v in res.items() if isinstance(v, float)},
            flush=True,
        )

    # Later rounds of the graph modes: a shorter context each time.
    for mode in mode_list:
        if mode.startswith("graph"):
            first_round = {k: v for k, v in results[mode].items() if k != "text"}
            results[mode]["rounds"] = [first_round]
    for index in range(1, rounds):
        for mode in mode_list:
            if not mode.startswith("graph"):
                continue
            res = _run(
                adapters[mode.split("_")[1]],
                clock,
                request,
                ctx - index * round_step,
                new_tokens,
                True,
                compile_layers,
            )
            res.pop("text")
            results[mode]["rounds"].append(res)
            print(
                mode,
                f"round {index + 1}",
                {k: round(v, 4) for k, v in res.items() if isinstance(v, float)},
                flush=True,
            )

    if check:
        # Graph replays against eager steps on the same static-cache path.
        adapter = adapters["sparse"]
        kwargs = {"max_new_tokens": short_tokens, "graph_decode": True}
        if compile_layers:
            kwargs["compile_layers"] = True
        replayed = adapter.process_request(
            request, kwargs, {"max_context_length": ctx}
        ).responses
        available = static_decode.graph_available
        static_decode.graph_available = lambda device: False
        try:
            stepped = adapter.process_request(
                request, kwargs, {"max_context_length": ctx}
            ).responses
        finally:
            static_decode.graph_available = available
        results["check_sparse_replay_equals_eager_steps"] = dict(
            equal=replayed == stepped, tokens=short_tokens
        )
        print("check: replayed == stepped:", replayed == stepped, flush=True)

    summary = dict(
        model=model,
        ctx=ctx,
        prompt_tokens=n_context + n_question,
        order=order,
        compile_layers=compile_layers,
        rounds=rounds,
        round_step=round_step,
        sparse_config=vars(cfg),
        gpu=torch.cuda.get_device_name(),
        results=results,
    )
    for res in results.values():
        res.pop("text", None)
    print(json.dumps(summary, indent=2))
    if out:
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
