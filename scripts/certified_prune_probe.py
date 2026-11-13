"""How many blocks can stage 1 drop with a proof that they hold almost no attention?

For one query head with true logits l_t, a block b of n_b tokens has
    sum_{t in b} exp(l_t) <= n_b * exp(B_b)          (B_b = Quest bound >= max_t l_t)
    sum_{t in b} exp(l_t) >= n_b * exp(q . mean_k_b)  (Jensen: mean of exp >= exp of mean)
so its true attention mass is at most
    U_b = n_b exp(B_b) / sum_b' n_b' exp(q . mean_k_b').
Dropping free blocks in ascending U order while their U sum stays <= delta loses at most
delta of that head's mass. For a candidate set shared by a GQA group, a block is ranked by
its largest U over the group's heads. This probe decodes greedily with dense attention and,
at every step and layer, reports how many free blocks are certified (dropped), how many of
the top-`frac` quest+mean candidates are among them, and the true mass actually dropped
(which must stay <= delta up to bf16 rounding of the bounds).

Example:
    python scripts/certified_prune_probe.py --model Qwen/Qwen3-4B-Instruct-2507 --n 4
"""

import json
import math
from collections import defaultdict

import fire
import torch
import torch.nn.functional as F
from datasets import load_dataset
from e2e_decode_bench import DecodeLoop, layer_post, layer_pre
from ruler_kernel_eval import _prompt_ids
from transformers import AutoModelForCausalLM, AutoTokenizer


class CertifiedProbe(DecodeLoop):
    def __init__(self, *args, deltas, frac, **kwargs):
        super().__init__(*args, **kwargs)
        self.deltas, self.frac = deltas, frac
        self.stats = defaultdict(list)

    def _attend(self, i, q, attn):
        qs = (q.squeeze(2) * attn.scaling).contiguous()
        n = int(self.seq_len)
        B, Hkv, Hq, bs = self.B, self.Hkv, self.Hq, self.bs
        G = Hq // Hkv
        cur_p = (n + bs - 1) // bs
        dev = q.device
        blocks = torch.arange(cur_p, device=dev)
        forced = (blocks < self.sink_blocks) | (blocks >= cur_p - self.local_blocks)
        free = ~forced
        n_free = int(free.sum())
        qg = qs.view(B, Hkv, G, -1).float()
        kk = self.k[i][:, :, :n].float()
        pad = cur_p * bs - n
        logits = torch.einsum("bhgd,bhnd->bhgn", qg, kk)
        true_mass = (
            F.pad(torch.softmax(logits, -1), (0, pad))
            .view(B, Hkv, G, cur_p, bs)
            .sum(-1)
        )
        kb = F.pad(kk, (0, 0, 0, pad)).view(B, Hkv, cur_p, bs, -1)
        valid_tok = (torch.arange(cur_p * bs, device=dev) < n).view(cur_p, bs, 1)
        n_tok = valid_tok.sum(1).squeeze(-1).float()  # [P]
        kmax = kb.masked_fill(~valid_tok, float("-inf")).amax(3)
        kmin = kb.masked_fill(~valid_tok, float("inf")).amin(3)
        kmean = (kb * valid_tok).sum(3) / n_tok[:, None]
        bound = qg.relu() @ kmax.transpose(-1, -2) + (qg - qg.relu()) @ kmin.transpose(
            -1, -2
        )
        s2 = qg @ kmean.transpose(-1, -2)
        log_n = n_tok.log()
        log_u = bound + log_n - torch.logsumexp(s2 + log_n, -1, keepdim=True)
        # Sanity: the certificate must upper-bound the true block mass.
        viol = (true_mass > log_u.exp() * (1 + 1e-3) + 1e-6).float().mean()
        self.stats["bound_violations"].append(viol.view(1).cpu())

        # Top-frac quest+mean candidates, as the kernels rank them (shared over the group).
        qm = 0.5 * torch.softmax(bound.masked_fill(~free, float("-inf")), -1)
        qm = qm + 0.5 * torch.softmax(s2.masked_fill(~free, float("-inf")), -1)
        n_keep = math.ceil(n_free * self.frac)
        top = torch.zeros(B, Hkv, cur_p, dtype=torch.bool, device=dev).scatter(
            -1, qm.mean(2).topk(n_keep, -1).indices, True
        )

        u_grp = log_u.amax(2).masked_fill(~free, float("inf"))  # [B, Hkv, P]
        order = u_grp.argsort(-1)
        u_sorted = u_grp.gather(-1, order).exp()
        csum = u_sorted.cumsum(-1)
        for delta in self.deltas:
            drop_sorted = csum <= delta
            drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
            lost = (true_mass * drop.unsqueeze(2)).sum(-1)  # [B, Hkv, G]
            k = f"d{delta}"
            self.stats[k + "|drop_free"].append(
                (drop.sum(-1).float() / n_free).flatten().cpu()
            )
            self.stats[k + "|drop_cand"].append(
                ((drop & top).sum(-1).float() / n_keep).flatten().cpu()
            )
            self.stats[k + "|lost"].append(lost.flatten().cpu())
        valid_k = (self.positions < self.seq_len).view(1, 1, 1, -1)
        return F.scaled_dot_product_attention(
            q,
            self.k[i],
            self.v[i],
            attn_mask=valid_k,
            scale=attn.scaling,
            enable_gqa=True,
        ).squeeze(2)


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    subsets: str = "niah_multikey_2,niah_multikey_3,cwe,qa_2",
    n: int = 4,
    new_tokens: int = 16,
    deltas=(1e-4, 1e-3, 1e-2),
    frac: float = 0.25,
    block_size: int = 64,
    out: str = "",
) -> None:
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    deltas = [float(d) for d in deltas]
    stats = defaultdict(list)
    subs = subsets.split(",") if isinstance(subsets, str) else list(subsets)
    for subset in subs:
        rows = load_dataset("xAlg-AI/att-hub-ruler-32k", subset, split=subset).select(
            range(n)
        )
        for row in rows:
            ids = _prompt_ids(tok, row).cuda()
            with torch.no_grad():
                pre = m(ids, use_cache=True, logits_to_keep=1)
            loop = CertifiedProbe(
                m,
                pre.past_key_values,
                ids.shape[1],
                1,
                new_tokens + 1,
                block_size,
                "two_stage",
                0.95,
                0.02,
                0.15,
                math.ceil(128 / block_size),
                math.ceil(128 / block_size),
                (layer_pre, layer_post),
                deltas=deltas,
                frac=frac,
            )
            loop.reset(pre.logits[:, -1].argmax(-1))
            for _ in range(new_tokens):
                loop.step()
            for key, vals in loop.stats.items():
                stats[key] += vals
            del loop, pre
            torch.cuda.empty_cache()
    res = {"bound_violations": float(torch.cat(stats["bound_violations"]).mean())}
    print(f"bound violations (fraction of head-blocks): {res['bound_violations']:.2e}")
    for delta in deltas:
        k = f"d{delta}"
        df = torch.cat(stats[k + "|drop_free"])
        dc = torch.cat(stats[k + "|drop_cand"])
        lost = torch.cat(stats[k + "|lost"])
        res[k] = dict(
            drop_free_mean=float(df.mean()),
            drop_free_p10=float(df.quantile(0.1)),
            drop_cand_mean=float(dc.mean()),
            drop_cand_p10=float(dc.quantile(0.1)),
            lost_max=float(lost.max()),
            lost_mean=float(lost.mean()),
        )
        r = res[k]
        print(
            f"delta {delta:g}: certified free blocks {r['drop_free_mean']:.3f} "
            f"(p10 {r['drop_free_p10']:.3f}), of top-{frac} candidates "
            f"{r['drop_cand_mean']:.3f} (p10 {r['drop_cand_p10']:.3f}), "
            f"true mass lost mean {r['lost_mean']:.2e} max {r['lost_max']:.2e}",
            flush=True,
        )
    if out:
        json.dump(
            dict(model=model, subsets=subs, frac=frac, results=res),
            open(out, "w"),
            indent=2,
        )


if __name__ == "__main__":
    fire.Fire(main)
