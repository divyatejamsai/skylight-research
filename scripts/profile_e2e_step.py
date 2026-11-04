"""Kernel-time breakdown of one CUDA-graphed decode step (see e2e_decode_bench.py).

Example:
    python scripts/profile_e2e_step.py --ctx 32768 --batch 4 --mode sparse
"""

import math

import fire
import torch
from e2e_decode_bench import DecodeLoop, layer_post, layer_pre
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    ctx: int = 32768,
    batch: int = 4,
    mode: str = "sparse",
    block_size: int = 64,
    rows: int = 25,
    cdf: float = 0.95,
    floor: float = 0.02,
    max_frac: float = 0.1,
    prefilter: float = 0.25,
) -> None:
    m = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    ids = torch.randint(0, 1000, (1, ctx), device="cuda")
    with torch.no_grad():
        pre = m(ids, use_cache=True, logits_to_keep=1)
    first = pre.logits[:, -1].argmax(-1)
    loop = DecodeLoop(
        m,
        pre.past_key_values,
        ctx,
        batch,
        16,
        block_size,
        mode,
        cdf,
        floor,
        max_frac,
        math.ceil(128 / block_size),
        math.ceil(128 / block_size),
        (layer_pre, layer_post),
    )
    loop.prefilter = prefilter
    del pre
    loop.reset(first)
    loop.capture()
    loop.reset(first)
    loop.run(3)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        loop.run(5)
        torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=rows, max_name_column_width=60
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
