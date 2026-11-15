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
    """hf_ruler_multi_eval._decode with one change: the replay loop ends at the first
    multiple of CHUNK steps after an end token."""
    assert batch == 1 and cache.max_cache_len >= ctx + new_tokens + 8
    m, t = adapter.model, adapter.tokenizer
    stop = torch.tensor(
        [t.eos_token_id, t.convert_tokens_to_ids("<|im_end|>")], device=m.device
    )
    tok = first.view(1, 1).clone()
    pos = torch.full((1,), ctx, device=m.device, dtype=torch.long)
    slots = torch.arange(cache.max_cache_len, device=m.device)
    mask = torch.zeros(1, 1, 1, slots.numel(), device=m.device, dtype=torch.bool)
    kw = dict(
        input_ids=tok,
        attention_mask=mask,
        position_ids=pos.view(1, 1),
        cache_position=pos,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    if mode == "sparse":
        kw["sparse_meta_data"] = {}
        attention = adapter.enable_sparse_mode()
    else:
        attention = M._attention(m, mode)

    def step():
        mask.copy_((slots <= pos).expand_as(mask))
        logits = m(**kw).logits[:, -1].float()
        tok.copy_(logits.argmax(-1, keepdim=True))
        pos.add_(1)
        return logits

    with attention:
        M._opaque_attention()
        first_logits = step()[0].clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        head = torch.stack([first.view(()), first_logits.argmax()])
        ended = bool(torch.isin(head, stop).any())
        toks = []
        for i in range(new_tokens):
            graph.replay()
            toks.append(tok.clone())
            if ended:
                break
            if (i + 1) % CHUNK == 0:
                ended = bool(torch.isin(torch.cat(toks[-CHUNK:], dim=1), stop).any())
        if not ended:  # the steps after the last check
            ended = bool(torch.isin(torch.cat(toks, dim=1), stop).any())
        torch.cuda.synchronize()
    del graph
    # steps: replays run; capped: no end token within the cap
    print("  steps", mode, len(toks), "capped" if not ended else "", flush=True)
    return first_logits, torch.cat(toks, dim=1), None


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
