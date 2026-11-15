"""loft_eval.py: scripts/hf_ruler_multi_eval.py on LOFT RAG at 128k.

--data loft128k (test split) or loft128k_dev; --subsets from nq, hotpotqa, musique,
qampari, quest. Everything else is hf_ruler_multi_eval.py: one dense prefill per prompt,
then one greedy decode per mode from it. Three things are swapped in:

  rows    the mirror Skylight's benchmark/loft reads (f20180301/loft-rag-<name>-128k).
          The prompt ends at the query, with no answer prefix, as that benchmark does.
  score   subspan exact match of benchmark/loft/calculate_metrics.py, times 100.
  decode  the runner's decode, stopped once an end token has appeared (checked every
          CHUNK steps). The runner cuts the text at the first end token, so stopping
          changes no prediction; it saves the steps after it. At most CAP new tokens.
"""

import fire
import hf_ruler_multi_eval as M
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from benchmark.loft.calculate_metrics import calculate_metrics
from sparse_attention_hub.adapters import static_decode

# 4096 by the registered rule: on the dev split 4 of 50 dense outputs (8%, above the 2%
# limit) reached the first cap of 1024. The dev pilot itself ran with 1024.
CAP = 4096
CHUNK = 16


def rows_loft(data: str, subset: str, n: int) -> pd.DataFrame:
    assert data in ("loft128k", "loft128k_dev"), data
    split = "dev" if data.endswith("_dev") else "test"
    path = hf_hub_download(
        f"f20180301/loft-rag-{subset}-128k",
        f"data/{split}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path).head(n)
    return pd.DataFrame(
        {
            "context": df["context"],
            "question": df["question"],
            "answer_prefix": "",
            "max_new_tokens": CAP,
            "answer": [[str(a) for a in answers] for answers in df["answers"]],
        }
    )


def score_loft(subset: str, preds, refs) -> float:
    df = pd.DataFrame(
        {
            "task": [f"{subset}_128k"],
            "answers": [list(refs[0])],
            "predicted_answer": [preds[0]],
            "answer_prefix": ["Final Answer: "],
        }
    )
    return 100.0 * calculate_metrics(df)["subspan_em"]


@torch.no_grad()
def decode_stop(adapter, cache, first, ctx, batch, new_tokens, mode):
    """hf_ruler_multi_eval._decode with one change: the decode ends at the first look
    for an end token that finds one (a look every CHUNK steps)."""
    assert batch == 1 and cache.max_cache_len >= ctx + new_tokens + 8
    m, t = adapter.model, adapter.tokenizer
    stop = static_decode.end_token_ids(m, t)
    if mode == "sparse":
        attention, meta = adapter.enable_sparse_mode(), {}
    else:
        attention, meta = M._attention(m, mode), None
    out = static_decode.decode_on_static_cache(
        m,
        cache,
        first,
        ctx,
        # An answer that the prefill's own token ends needs no more than the first step.
        2 if int(first) in stop else new_tokens + 1,
        attention=attention,
        sparse_meta_data=meta,
        stop_token_ids=stop,
        stop_check_every=CHUNK,
        opaque_attention=True,
    )
    ended = out.stopped or int(first) in stop
    # steps: replays run; capped: no end token within the cap
    print("  steps", mode, out.steps - 1, "capped" if not ended else "", flush=True)
    return out.first_logits, out.tokens[:, 1:], None


if __name__ == "__main__":
    import hf_graph_decode_bench
    import hf_ruler_eval

    import benchmark.loft
    import sparse_attention_hub

    for mod in (M, hf_ruler_eval, hf_graph_decode_bench, benchmark.loft):
        print("module", mod.__name__, "from", mod.__file__, flush=True)
    print("module sparse_attention_hub from", sparse_attention_hub.__file__, flush=True)
    M._rows, M._score, M._decode = rows_loft, score_loft, decode_stop
    fire.Fire(M.main)
