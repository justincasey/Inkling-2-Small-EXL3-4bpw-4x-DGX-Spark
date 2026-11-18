"""Generate DSpark drafter fixtures over a pack folder holding `mtp.safetensors` and the target's shared tensors.

    python oracle/gen_dspark_fixtures.py --pack DIR --fixtures DIR --out DIR [--threads 8] [--only CASE ...]

Cases (C):
  real_raw-counting_tp4, real_chat-sky-effort0.9_tp4, real_raw-counting_tp1
      features of the target oracle fixture FIXTURES/<name>.safetensors (gen_fixtures.py):
      F[t] = stack(hidden[5], hidden[11], ..., hidden[41], h_final)[t], tokens = its `tokens`;
  long_tp4
      N = 1100 seeded bf16 features (normal, per-tap scales LONG_SCALES), seeded random tokens < 200058.
Per case the reference commits the rows up to each anchor p (ascending) and proposes at p with anchor tokens[p]; the
step logits are those of its own drafts (teacher forcing on the reference's drafts equals the free-running chain).
Files OUT/C_*.npy and OUT/C_meta.json follow the drafter reference contract:
  feats u16 [N, 8, H] (bf16 bits), tokens i64 [N+1], anchors i64 [P], drafts i64 [P, 7], hd u16 [P, 8, H],
  frnn u16 [P, H], state / latent u16 [P, 7, R], logits f32 [P, 7, 200058], ctxkv u16 [5, N, 2, 8, 128].
For the real cases it prints how many leading drafts match the target's next tokens tokens[p+1 .. p+7].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import st
from dspark_ref import DSparkConfig, DSparkRef, bf16_bits
from gen_fixtures import oracle_source_sha, sha256_file
from pack import PackIndex, PackWeights
from synthetic import random_features

CASES = {
    "real_raw-counting_tp4": dict(fixture="raw-counting", tp=4, anchors=[1, 13, 25, 35, 38, 41, 44, 50]),
    "real_chat-sky-effort0.9_tp4": dict(fixture="chat-sky-effort0.9", tp=4, anchors=[1, 13, 25, 26, 30, 36, 42, 49]),
    "real_raw-counting_tp1": dict(fixture="raw-counting", tp=1, anchors=[1, 13, 25, 35, 38, 41, 44, 50]),
    "long_tp4": dict(fixture=None, tp=4, anchors=[1, 600, 1023, 1024, 1025, 1031, 1032, 1100]),
}
LONG_N = 1100
LONG_SEED = 20261117
LONG_SCALES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 1.0)


class DSparkPackWeights:
    """`mtp.safetensors` tensors by their unprefixed names, everything else (model.llm.embed / embed_norm / unembed)
    from the pack index; positional reads, nothing mmapped."""

    def __init__(self, pack: str):
        self.mtp = st.read_header(os.path.join(pack, "mtp.safetensors"))
        self.target = PackWeights(PackIndex(pack))

    def tensor(self, name: str) -> torch.Tensor:
        return st.read_tensor(self.mtp[name]) if name in self.mtp else self.target.tensor(name)

    def rows(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        if name not in self.mtp:
            return self.target.rows(name, ids)
        return torch.stack([st.read_rows(self.mtp[name], int(i), int(i) + 1)[0] for i in ids.tolist()])

    def row_range(self, name: str, start: int, stop: int) -> torch.Tensor:
        if name not in self.mtp:
            return self.target.row_range(name, start, stop)
        return st.read_rows(self.mtp[name], start, stop)


def real_features(cfg: DSparkConfig, path: str) -> tuple[torch.Tensor, np.ndarray]:
    hdr = st.read_header(path)
    hidden = st.read_tensor(hdr["hidden"])  # [L+1, T, H]
    h_final = st.read_tensor(hdr["h_final"])  # [T, H]
    feats = torch.stack([hidden[j] for j in cfg.target_layer_ids[:-1]] + [h_final], 1)
    tokens = st.read_tensor(hdr["tokens"]).numpy().astype(np.int64)
    if tokens.shape[0] != feats.shape[0] + 1:
        raise ValueError(f"{path}: {tokens.shape[0]} tokens for {feats.shape[0]} feature rows")
    return feats, tokens


def long_features(cfg: DSparkConfig) -> tuple[torch.Tensor, np.ndarray]:
    feats = random_features(cfg, LONG_N, seed=LONG_SEED, scales=LONG_SCALES)
    tokens = np.random.default_rng(LONG_SEED).integers(0, cfg.valid_vocab, LONG_N + 1, dtype=np.int64)
    return feats, tokens


def accepted(drafts: list[int], tokens: np.ndarray, p: int) -> tuple[int, int]:
    """(matching prefix length, number of comparable drafts) against tokens[p+1 .. p+7]."""
    truth = tokens[p + 1:p + 1 + len(drafts)].tolist()
    k = 0
    while k < len(truth) and drafts[k] == truth[k]:
        k += 1
    return k, len(truth)


def run_case(ref: DSparkRef, feats: torch.Tensor, tokens: np.ndarray, anchors: list[int]) -> dict:
    n = feats.shape[0]
    if sorted(anchors) != anchors or anchors[0] < 1 or anchors[-1] > n:
        raise ValueError(f"anchors {anchors} must ascend within [1, {n}]")
    state = ref.new_state()
    cur = 0
    out = {k: [] for k in ("drafts", "hd", "frnn", "state", "latent", "logits")}
    for p in anchors:
        if p > cur:
            ref.commit(state, feats[cur:p], cur)
            cur = p
        t0 = time.time()
        prop = ref.propose(state, int(tokens[p]), trace=True)
        tr = prop.trace
        out["drafts"].append(prop.drafts)
        out["hd"].append(bf16_bits(tr["hd"]))
        out["frnn"].append(bf16_bits(tr["f_rnn"]))
        out["state"].append(bf16_bits(tr["state"]))
        out["latent"].append(bf16_bits(tr["latent"]))
        out["logits"].append(tr["logits"].to(torch.float32).numpy())
        print(f"  p={p} anchor={int(tokens[p])} drafts={prop.drafts} ({time.time() - t0:.1f}s)", flush=True)
    if n > cur:
        ref.commit(state, feats[cur:n], cur)
    res = {k: np.stack(v) for k, v in out.items()}
    res["drafts"] = res["drafts"].astype(np.int64)
    res["ctxkv"] = bf16_bits(state.ctx_kv())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--fixtures", required=True, help="target oracle fixtures (gen_fixtures.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8, help="torch intra-op threads")
    ap.add_argument("--only", nargs="*", choices=sorted(CASES))
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    with open(os.path.join(a.pack, "config.json")) as f:
        cfg = DSparkConfig.from_hf(json.load(f))
    weights = DSparkPackWeights(a.pack)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    mtp_sha = sha256_file(os.path.join(a.pack, "mtp.safetensors"))
    src_sha = oracle_source_sha()
    base = DSparkRef(cfg, weights, tp=4)
    base.fold()
    print(f"mtp sha256 + fold: {time.time() - t0:.0f}s", flush=True)
    refs = {4: base, 1: base.with_tp(1)}
    for name in a.only or list(CASES):
        case = CASES[name]
        t0 = time.time()
        if case["fixture"]:
            path = os.path.join(a.fixtures, f"{case['fixture']}.safetensors")
            feats, tokens = real_features(cfg, path)
            ids = ", ".join(str(j) for j in cfg.target_layer_ids[:-1])
            source = (f"oracle fixture {case['fixture']}.safetensors (sha256 {sha256_file(path)}): "
                      f"stack(hidden[{ids}], h_final)")
        else:
            feats, tokens = long_features(cfg)
            source = f"synthetic: seed {LONG_SEED}, normal bf16 features with per-tap scales {list(LONG_SCALES)}"
        print(f"{name}: N={feats.shape[0]} tp={case['tp']} anchors={case['anchors']}", flush=True)
        res = run_case(refs[case["tp"]], feats, tokens, case["anchors"])
        secs = time.time() - t0
        arrays = {"feats": bf16_bits(feats), "tokens": tokens, "anchors": np.array(case["anchors"], dtype=np.int64),
                  **res}
        for k, v in arrays.items():
            np.save(os.path.join(a.out, f"{name}_{k}.npy"), v)
        meta = {"tp": case["tp"], "anchors": case["anchors"], "n_positions": int(feats.shape[0]),
                "feature_source": source, "mtp_sha256": mtp_sha, "oracle_source_sha256": src_sha,
                "seconds": round(secs, 1)}
        with open(os.path.join(a.out, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if case["fixture"]:
            acc = [accepted(d, tokens, p) for d, p in zip(res["drafts"].tolist(), case["anchors"])]
            print(f"{name}: acceptance (prefix/comparable) " + " ".join(f"p{p}:{k}/{m}" for p, (k, m)
                                                                         in zip(case["anchors"], acc)), flush=True)
        print(f"{name}: {secs:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
