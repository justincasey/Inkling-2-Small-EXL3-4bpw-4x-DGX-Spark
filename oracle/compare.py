"""Compare an engine dump against an oracle fixture and apply the tolerance gates (docs/ORACLE-GAP.md, "Gates").

Both files use the fixture layout written by gen_fixtures.py: `tokens` I64 [N] (prompt + generated), `logits` F32
[N-1, V] (row t predicts token t+1), optional `hidden` BF16 [L+1, N-1, H], `h_final` BF16 [N-1, H], `router_idx`
I32 [L, N-1, k], `router_margin` F32 [L, N-1]; metadata `prompt_len`, and `emulate` on fixtures that reproduce the
engine's lossy numerics (gen_fixtures.py --emulate).

The gate set follows the oracle fixture's `emulate`: `engine` (the oracle emulates int8 global KV, bf16 P and fp16
expert inputs) is the correctness gate; `none` (the spec's numerics, also fixtures without the key) is the measured
drift gate. Partial emulations have no gate set; `--gates engine|none` forces one.

    python oracle/compare.py ORACLE.safetensors ENGINE.safetensors [--json report.json] [--gates engine|none]

Exit status 0 when every gate passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

import st

# A gate of None is reported but not enforced.
GATES_ENGINE = {
    "hidden_cos_min": 0.999,  # every layer, every token: cosine(P_engine, P_oracle)
    "hidden_cos_median_min": None,  # per layer, the median over tokens; the smallest layer
    "h_final_cos_min": 0.999,
    "h_final_cos_median_min": None,
    "kl_mean_max": 1e-3,  # nats, KL(oracle || engine) over the valid positions
    "kl_max": 1e-2,
    "top1_agree_min": 0.99,  # fraction of valid positions
    "tie_margin": 0.05,  # oracle top-1 minus top-2 logit below this: a disagreement there is excused
}
# Measured on the four default fixtures against the spec oracle (docs/ORACLE-GAP.md, "Gates"): int8 KV, bf16 P and
# fp16 expert inputs move single tokens' hidden states far (cosine down to 0.33 on raw-long-window) through routing
# flips, so the per-token minimum is not gated; the per-layer medians and the logits are, with about 2x headroom.
GATES_SPEC = {
    "hidden_cos_min": None,
    "hidden_cos_median_min": 0.95,
    "h_final_cos_min": None,
    "h_final_cos_median_min": 0.99,
    "kl_mean_max": 2.5e-2,
    "kl_max": 1.0,
    "top1_agree_min": 0.95,
    "tie_margin": 0.1,
}
GATE_SETS = {"engine": GATES_ENGINE, "none": GATES_SPEC}


def gates_for(emulate: str) -> dict:
    if emulate not in GATE_SETS:
        raise ValueError(f"no gate set for an oracle fixture with emulate={emulate!r} (engine or none; or --gates)")
    return GATE_SETS[emulate]


def load(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        meta = json.loads(f.read(n)).get("__metadata__", {})
    return st.load_all(path), meta


def _cos_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.to(torch.float64), b.to(torch.float64)
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(den > 0, num / den.clamp_min(1e-300), (a.norm(dim=-1) == b.norm(dim=-1)).to(torch.float64))


def _ge(x: float, gate: float | None) -> bool:
    return gate is None or x >= gate


def compare(oracle: dict[str, torch.Tensor], engine: dict[str, torch.Tensor], prompt_len: int,
            gates: dict | None = None) -> dict:
    g = dict(GATES_ENGINE, **(gates or {}))
    rep: dict = {"gates": g, "checks": {}}
    ot, et = oracle["tokens"], engine["tokens"]
    if not torch.equal(ot[:prompt_len], et[:prompt_len]):
        raise ValueError("the dumps were made from different prompts")
    ol = oracle["logits"].to(torch.float64)
    top2 = ol.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    # generated tokens: equal, or the first difference sits where the oracle's top two were within tie_margin
    n = min(ot.shape[0], et.shape[0])
    diff = [j for j in range(prompt_len, n) if ot[j] != et[j]]
    first = diff[0] if diff else n
    excused = bool(diff) and float(margin[first - 1]) < g["tie_margin"]
    rep["checks"]["tokens"] = {"generated": n - prompt_len, "first_divergence": first if diff else None,
                               "excused_tie": excused, "pass": (not diff) or excused}
    valid = first if diff else n - 1  # logits rows < first saw identical prefixes in both dumps
    el = engine["logits"][:valid].to(torch.float64)
    ol = ol[:valid]
    lp, lq = torch.log_softmax(ol, -1), torch.log_softmax(el, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    agree = ol.argmax(-1) == el.argmax(-1)
    strict = margin[:valid] >= g["tie_margin"]
    rep["checks"]["logits"] = {
        "positions": valid, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
        "top1_agree": float(agree.double().mean()), "top1_disagree_outside_ties": int((~agree & strict).sum()),
        "pass": float(kl.mean()) <= g["kl_mean_max"] and float(kl.max()) <= g["kl_max"]
        and float(agree.double().mean()) >= g["top1_agree_min"] and int((~agree & strict).sum()) == 0,
    }
    if "h_final" in engine:
        c = _cos_rows(oracle["h_final"][:valid], engine["h_final"][:valid])
        rep["checks"]["h_final"] = {"cos_min": float(c.min()), "cos_median": float(c.median()),
                                    "pass": _ge(float(c.min()), g["h_final_cos_min"])
                                    and _ge(float(c.median()), g["h_final_cos_median_min"])}
    if "hidden" in engine:
        c = _cos_rows(oracle["hidden"][:, :valid], engine["hidden"][:, :valid])  # [L+1, T]
        per_layer, med = c.min(-1).values, c.median(-1).values
        rep["checks"]["hidden"] = {"cos_min_per_layer": [round(float(x), 7) for x in per_layer],
                                   "cos_median_per_layer": [round(float(x), 7) for x in med],
                                   "worst_layer": int(per_layer.argmin()), "worst_median_layer": int(med.argmin()),
                                   "pass": _ge(float(per_layer.min()), g["hidden_cos_min"])
                                   and _ge(float(med.min()), g["hidden_cos_median_min"])}
    if "router_idx" in engine and "router_idx" in oracle:
        oi, ei = oracle["router_idx"][:, :valid].long(), engine["router_idx"][:, :valid].long()
        k = oi.shape[-1]
        same = (oi.sort(-1).values == ei.sort(-1).values).all(-1)  # [L, T]
        overlap = torch.stack([(oi[..., j:j + 1] == ei).any(-1) for j in range(k)], -1).double().mean(-1)
        info = {"identical_sets": float(same.double().mean()), "slot_overlap_mean": float(overlap.mean())}
        if "router_margin" in oracle:
            m = oracle["router_margin"][:, :valid].to(torch.float64)
            info["flips_at_margin_max"] = float(m[~same].max()) if (~same).any() else None
        rep["checks"]["router"] = info  # informative: oracle/teacher_force.py checks the router on identical inputs
    rep["pass"] = all(v.get("pass", True) for v in rep["checks"].values())
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("oracle")
    ap.add_argument("engine")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--gates", choices=sorted(GATE_SETS), help="gate set (default: the oracle fixture's emulate)")
    a = ap.parse_args(argv)
    o, meta = load(a.oracle)
    e, _ = load(a.engine)
    which = a.gates or meta.get("emulate", "none")
    rep = compare(o, e, int(meta["prompt_len"]), gates_for(which))
    rep["gate_set"] = which
    txt = json.dumps(rep, indent=1)
    if a.json:
        with open(a.json, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
