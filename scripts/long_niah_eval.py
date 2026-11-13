"""Needle retrieval at long context (default ~128k tokens) through the real decode kernels.

RULER-32k prompts span at most two of the prefilter's 512-block regions, so this builds
longer RULER-style prompts and runs them through ruler_kernel_eval.main (dense prefill,
then each mode's greedy decode loop, RULER string-match scoring):

  wiki_mk4  wikitext haystack with 4 "special magic number" needles; asks for one
  uuid_mk3  a haystack made only of uuid-key -> uuid-value lines (like niah_multikey_3);
            asks for one key's value

--tune "key=value,..." overrides kernels.TUNE; --target sets the prompt length in tokens.

Example:
    python scripts/long_niah_eval.py --model Qwen/Qwen3-4B-Instruct-2507 \\
        --subsets wiki_mk4,uuid_mk3 --n 30 --modes dense_sdpa,two_stage_qm@0.95@0.02@0.15@0.25
"""

import random
import uuid

import fire
import ruler_kernel_eval as R
from datasets import load_dataset
from transformers import AutoTokenizer

from sparse_attention_hub.kernels.block_sparse_decode import TUNE

WORDS = [
    "apple",
    "river",
    "candle",
    "violet",
    "harbor",
    "pepper",
    "falcon",
    "marble",
    "lantern",
    "meadow",
    "copper",
    "orchid",
]


class _Rows(list):
    def select(self, idx):
        return _Rows(self[i] for i in idx)


def _wiki_rows(tok, n, target):
    text = "\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    ids = tok(text, add_special_tokens=False).input_ids
    rows = []
    for i in range(n):
        rng = random.Random(1000 + i)
        start = rng.randrange(0, len(ids) - target)
        hay = ids[start : start + target - 300]
        keys = rng.sample(WORDS, 4)
        vals = [str(rng.randrange(10**6, 10**7)) for _ in keys]
        depths = sorted(rng.sample(range(1, 20), 4))
        pieces, prev = [], 0
        for d, k, v in zip(depths, keys, vals):
            cut = len(hay) * d // 20
            pieces += [
                tok.decode(hay[prev:cut]),
                f"\nOne of the special magic numbers for {k} is: {v}.\n",
            ]
            prev = cut
        pieces.append(tok.decode(hay[prev:]))
        j = rng.randrange(4)
        rows.append(
            dict(
                context="Some special magic numbers are hidden within the following "
                "text. Make sure to memorize it. I will quiz you about the numbers "
                "afterwards.\n" + "".join(pieces),
                question=f"\nWhat is the special magic number for {keys[j]} "
                "mentioned in the provided text?",
                answer_prefix=f" The special magic number for {keys[j]} mentioned in "
                "the provided text is",
                answer=[vals[j]],
                max_new_tokens=24,
            )
        )
    return rows


def _uuid_rows(tok, n, target):
    line = "One of the special magic uuids for {} is: {}.\n"
    per_line = len(
        tok(
            line.format(uuid.UUID(int=1), uuid.UUID(int=2)), add_special_tokens=False
        ).input_ids
    )
    rows = []
    for i in range(n):
        rng = random.Random(2000 + i)

        def u():
            return str(uuid.UUID(int=rng.getrandbits(128), version=4))

        pairs = [(u(), u()) for _ in range((target - 300) // per_line)]
        j = rng.randrange(len(pairs))
        rows.append(
            dict(
                context="Some special magic uuids are hidden within the following "
                "text. Make sure to memorize it. I will quiz you about the uuids "
                "afterwards.\n" + "".join(line.format(a, b) for a, b in pairs),
                question=f"\nWhat is the special magic uuid for {pairs[j][0]} "
                "mentioned in the provided text?",
                answer_prefix=f" The special magic uuid for {pairs[j][0]} mentioned "
                "in the provided text is",
                answer=[pairs[j][1]],
                max_new_tokens=40,
            )
        )
    return rows


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    subsets: str = "wiki_mk4,uuid_mk3",
    n: int = 30,
    modes: str = "dense_sdpa,two_stage_qm@0.95@0.02@0.15@0.25",
    target: int = 128000,
    tune: str = "",
    out: str = "",
) -> None:
    for kv in filter(None, str(tune).split(",")):
        key, val = kv.split("=")
        assert key in TUNE, key
        TUNE[key] = int(val)
    tok = AutoTokenizer.from_pretrained(model)
    build = dict(wiki_mk4=_wiki_rows, uuid_mk3=_uuid_rows)
    R.load_dataset = lambda name, subset, split=None: _Rows(
        build[subset](tok, n, target)
    )
    R.main(model=model, subsets=subsets, n=n, modes=modes, out=out)


if __name__ == "__main__":
    fire.Fire(main)
