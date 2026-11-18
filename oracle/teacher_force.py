"""Teacher-forced, op-by-op comparison of an engine dump against the oracle (docs/ORACLE-GAP.md).

The engine dump (bench/tools/merge_dump.py output of a boot with REXL3_INKLING2_DUMP_OPS=1) holds, per layer, the
partial block entering it and the outputs of its sub-ops. Every oracle op here is fed the ENGINE's inputs of that op,
so each comparison measures one op's numerics with nothing accumulated from earlier layers:

  attention  P and the completed blocks entering layer l (engine `hidden`) -> mix/norm -> attention core (rank 0's 12
             heads, vs `attn_core`) -> wo, all-reduce, attn_sconv (vs `attn`); once per attention emulation variant
  mix        the engine's `attn` added to its P -> mix + mlp_norm (vs `mlp_in`)
  router     the engine's `mlp_in` -> fp32 logits (vs `router_logits`) and the selection (vs `router_idx`/`router_w`)
  moe        the engine's `mlp_in`, picks and weights -> experts, rank partials, all-reduce (vs `moe`); with and
             without the fp16 GEMM-input emulation, from one decode of each expert
  residual   the engine's `moe` -> mlp_sconv + residual (vs `hidden[l + 1]`)

and two noise floors, oracle against oracle: the fully emulated attention sublayer and the fp16-input experts once with
float64 and once with fp32 accumulation of their products (torch / numpy float32 matmul: another fp32 summation order
than the engine's). An engine op whose difference from the emulating oracle is at that floor differs from it by
summation order only.

    python oracle/teacher_force.py --pack DIR --engine ENGINE.safetensors --out REPORT.json
                                   [--layers 0-47] [--workers N] [--threads N] [--no-moe]

Row statistics: `rel` = |o - e| / |e| per row (median and max over rows), `cos_min`, `diff_frac` = fraction of
elements that differ, `abs_max_rms` = the largest element difference over its row's RMS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

import exl3
import st
from inkling2_ref import (F64, Emulation, Inkling2Config, Inkling2Ref, Rounding, expert_ffn, rmsnorm_w, route,
                          static_conv)
from pack import PROJS, PackIndex, PackWeights, exit_with_parent, expert_prefix, load_exl3

ATTN_VARIANTS = {
    "none": Emulation(),
    "attn_bf16_p": Emulation(attn_bf16_p=True),
    "kv8": Emulation(kv8=True),
    "kv8+attn_bf16_p": Emulation(kv8=True, attn_bf16_p=True),
}
# (fp16 GEMM inputs, fp32 accumulation of the expert products)
MOE_VARIANTS = {"none": (False, False), "moe_fp16": (True, False), "moe_fp16_acc32": (True, True)}

_ix: PackIndex | None = None


class Fp32Accum(Inkling2Ref):
    """The oracle with its dense products accumulated in fp32 (torch's float32 matmul) and its AttnRes mix + norm
    computed in fp32: a second fp32 implementation of the same arithmetic, for the noise floor."""

    def matmul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float32) @ w.to(torch.float32).T).to(F64)

    def mix(self, blocks, partial, q, norms):
        f = torch.float32
        x = torch.stack(blocks + [partial]).to(f)
        rho = torch.rsqrt((x * x).mean(-1) + 1e-6)
        p = torch.softmax(torch.einsum("nth,qh->qnt", x, q.to(f)) * rho[None], dim=1)
        m = torch.einsum("qnt,nth->qth", p * rho[None], x)
        return [self.r.bf16((m[j] * torch.rsqrt((m[j] * m[j]).mean(-1, keepdim=True) + self.cfg.eps)
                             * norms[j].to(f)).to(F64)) for j in range(m.shape[0])]


def _init(root: str, worker: bool = True) -> None:
    global _ix
    if worker:
        exit_with_parent()
    _ix = PackIndex(root)


def _acc32(lin: exl3.Exl3Linear):
    """The EXL3 layer with its W_q product accumulated in fp32 (inputs and W_q values are exact in fp32)."""
    def f(x: np.ndarray, in_round=None) -> np.ndarray:
        xh = exl3.fwht_128(np.asarray(x, dtype=np.float64) * lin.suh[None, :], axis=1)
        if in_round is not None:
            xh = in_round(xh)
        y = (xh.astype(np.float32) @ lin.wq.astype(np.float32)).astype(np.float64)
        return exl3.fwht_128(y, axis=1) * lin.svh[None, :]
    return f


def _ffn_variants(layer: int, e: int, x: np.ndarray, gate_cap: float, up_cap: float,
                  variants: tuple[tuple[bool, bool], ...]) -> list[np.ndarray]:
    """One decode of expert e, its FFN once per (fp16 inputs, fp32 accumulation) setting."""
    lins = [load_exl3(_ix, expert_prefix(layer, e, p)) for p in PROJS]
    out = []
    for fp16, acc32 in variants:
        ls = [_acc32(lin) for lin in lins] if acc32 else lins
        out.append(expert_ffn(*ls, x, gate_cap, up_cap, Rounding(), fp16))
    return out


def row_stats(o: torch.Tensor, e: torch.Tensor) -> dict:
    o, e = o.to(F64).reshape(o.shape[0], -1), e.to(F64).reshape(e.shape[0], -1)
    d = o - e
    en = e.norm(dim=-1)
    rel = d.norm(dim=-1) / en.clamp_min(1e-300)
    cos = (o * e).sum(-1) / (o.norm(dim=-1) * en).clamp_min(1e-300)
    rms = (en / e.shape[1] ** 0.5).clamp_min(1e-300)
    return {"rel_med": float(rel.median()), "rel_max": float(rel.max()), "rel_max_row": int(rel.argmax()),
            "cos_min": float(cos.min()), "diff_frac": float((d != 0).double().mean()),
            "abs_max_rms": float((d.abs().amax(-1) / rms).max())}


def parse_layers(spec: str, n: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    if any(not 0 <= i < n for i in out):
        raise ValueError(f"layers out of range 0..{n - 1}: {spec}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--engine", required=True, help="merge_dump.py output with the sub-op planes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-moe", action="store_true", help="skip the expert comparison (the costly part)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    with open(os.path.join(a.pack, "config.json")) as f:
        cfg = Inkling2Config.from_hf(json.load(f))
    eng = st.load_all(a.engine)
    need = ("hidden", "attn_core", "attn", "mlp_in", "router_logits", "router_idx", "router_w", "moe")
    missing = [k for k in need if k not in eng]
    if missing:
        raise SystemExit(f"{a.engine}: no {missing} (boot with REXL3_INKLING2_DUMP_OPS=1 and merge_dump.py)")
    hid = eng["hidden"].to(F64)
    t = hid.shape[1]
    pos = torch.arange(t)
    ix = PackIndex(a.pack)
    model = Inkling2Ref(cfg, PackWeights(ix), experts=None, tp=a.tp)
    model32 = Fp32Accum(cfg, PackWeights(ix), experts=None, tp=a.tp, emulate=Emulation(kv8=True, attn_bf16_p=True))
    pool = None
    if not a.no_moe and a.workers > 0:
        pool = ProcessPoolExecutor(a.workers, mp_context=get_context("spawn"), initializer=_init, initargs=(ix.root,))
    elif not a.no_moe:
        _init(ix.root, worker=False)
    r = model.r
    qcols = cfg.num_heads * cfg.head_dim // a.tp
    report = {"engine": os.path.abspath(a.engine), "rows": t, "tp": a.tp, "layers": []}
    try:
        for i in parse_layers(a.layers, cfg.num_layers):
            t0 = time.time()
            glob = cfg.is_global(i)
            first = i % cfg.layers_per_block == 0
            ln = f"model.llm.layers.{i}."
            blocks = [hid[j] for j in range(0, i, cfg.layers_per_block)]
            p = hid[i]
            rep: dict = {"layer": i, "global": glob, "attn": {}}
            # attention sublayer from the engine's P and blocks, per emulation variant
            norms = ([model.w64(ln + f"{n}_input_norm.weight") for n in "qkv"] if glob
                     else [model.w64(ln + "attn_norm.weight")])
            cw = model.w64(ln + "attn_sconv.weight")[:, 0, :]

            def attn_out(m: Inkling2Ref):
                if i == 0:
                    xs = [rmsnorm_w(p, w, cfg.eps, r) for w in norms]
                else:
                    xs = m.mix(blocks, p, m.query(f"model.llm.attn_res.attn_queries.{i - 1}"), norms)
                xq, xk, xv = (xs[0], xs[1], xs[2]) if glob else (xs[0], xs[0], xs[0])
                m.taps = {}
                lc = m.new_cache().layers[i]
                a_raw = m.attention(i, xq, xk, xv, pos, lc)
                av, _ = static_conv(a_raw, cw, lc.conv["attn"], r)
                core = m.taps[i]["core"][:, :qcols]
                m.taps = None
                return core, av

            for name, em in ATTN_VARIANTS.items():
                model.em = em
                core, av = attn_out(model)
                rep["attn"][name] = {"core": row_stats(core, eng["attn_core"][i]), "a": row_stats(av, eng["attn"][i])}
            core32, av32 = attn_out(model32)  # noise floor: the same emulated sublayer in a second fp32 implementation
            rep["attn_noise"] = {"core": row_stats(core32, core), "a": row_stats(av32, av)}
            model.em = Emulation()
            # MoE input mix from the engine's attention output
            e_a = eng["attn"][i].to(F64)
            mblocks = blocks + [p] if first else blocks
            part = e_a if first else r.f32(p + e_a)
            mq, mn = model.query(f"model.llm.attn_res.mlp_queries.{i}"), [model.w64(ln + "mlp_norm.weight")]
            mlp_in = model.mix(mblocks, part, mq, mn)[0]
            rep["mix"] = row_stats(mlp_in, eng["mlp_in"][i])
            rep["mix_noise"] = row_stats(model32.mix(mblocks, part, mq, mn)[0], mlp_in)
            # router on the engine's MoE input
            x = eng["mlp_in"][i].to(F64)
            wg = model.w64(ln + "mlp.gate.weight")
            logits, idx, w, margin = model.router(i, x)
            e_log = eng["router_logits"][i].to(F64)
            sumabs = x.abs() @ wg.abs().T  # [T, E]: sum_k |x_k w_k|, the scale of an fp32 dot product's error
            dlog = (logits - e_log).abs()
            bias = model.w.tensor(ln + "mlp.gate.bias").to(F64)
            e_idx = eng["router_idx"][i].long()
            # the engine's picks against a float64 selection on the engine's own logits: the selection logic alone
            idx_from_e, w_from_e, _ = route(e_log, bias, cfg.top_k, cfg.route_scale, r)
            same = (idx.sort(-1).values == e_idx.sort(-1).values).all(-1)
            consistent = (idx_from_e.sort(-1).values == e_idx.sort(-1).values).all(-1)
            # score error bound of a row: twice the largest |sigmoid(oracle logit) - sigmoid(engine logit)|
            dsel = (torch.sigmoid(logits) - torch.sigmoid(e_log)).abs().amax(-1)
            flips = [{"row": int(j), "oracle_margin": float(margin[j]), "score_err_bound": float(2 * dsel[j]),
                      "within_bound": bool(margin[j] <= 2 * dsel[j])}
                     for j in torch.nonzero(~same).flatten().tolist()]
            rep["router"] = {
                "logit_err_max": float(dlog.max()),
                "logit_err_max_rel_sumabs": float((dlog / sumabs.clamp_min(1e-300)).max()),
                "picks_identical": float(same.double().mean()),
                "engine_select_consistent": bool(consistent.all()),
                "weights_err_max": float((w_from_e - eng["router_w"][i].to(F64)).abs().max()),
                "margin_min": float(margin.min()),
                "flips": flips,
            }
            # experts on the engine's input, picks and weights. The engine's `moe` plane is the f32 rank-order sum of
            # the bf16 rank partials; mlp_sconv reads it rounded to bf16, which is the oracle's all-reduce output.
            mo = r.bf16(eng["moe"][i].to(F64))
            if not a.no_moe:
                e_w = eng["router_w"][i].to(F64)
                jobs, rows = Inkling2Ref.expert_jobs(x, e_idx)
                variants = tuple(MOE_VARIANTS.values())
                args = [(i, e, xe, cfg.gate_softcap, cfg.up_softcap, variants) for e, xe in jobs]
                outs = list(pool.map(_ffn_variants, *zip(*args))) if pool else [_ffn_variants(*g) for g in args]
                ms = {name: model.combine(jobs, rows, [o[k] for o in outs], e_w, x.shape)
                      for k, name in enumerate(MOE_VARIANTS)}
                rep["moe"] = {name: row_stats(m, mo) for name, m in ms.items() if name != "moe_fp16_acc32"}
                rep["moe_noise"] = row_stats(ms["moe_fp16_acc32"], ms["moe_fp16"])
                rep["moe_experts"] = len(jobs)
            # mlp_sconv + residual from the engine's MoE output
            mc, _ = static_conv(mo, model.w64(ln + "mlp_sconv.weight")[:, 0, :],
                                torch.zeros(cfg.conv_kernel - 1, cfg.hidden_size, dtype=F64), r)
            p_out = r.bf16(r.f32(part + mc))
            rep["residual_mismatch"] = int((p_out != hid[i + 1]).sum())
            rep["seconds"] = round(time.time() - t0, 1)
            report["layers"].append(rep)
            at, an = rep["attn"], rep["attn_noise"]
            print(f"layer {i:2d}{'G' if glob else ' '} core rel {at['none']['core']['rel_med']:.2e}/"
                  f"{at['kv8+attn_bf16_p']['core']['rel_med']:.2e}/{an['core']['rel_med']:.2e}  "
                  f"a rel {at['none']['a']['rel_med']:.2e}/{at['kv8+attn_bf16_p']['a']['rel_med']:.2e}/"
                  f"{an['a']['rel_med']:.2e}  mix diff {rep['mix']['diff_frac']:.1e}  "
                  f"router same {rep['router']['picks_identical']:.3f} flips {len(flips)} "
                  f"(out of bound {sum(not f['within_bound'] for f in flips)})"
                  + (f"  moe rel {rep['moe']['none']['rel_med']:.2e}/{rep['moe']['moe_fp16']['rel_med']:.2e}/"
                     f"{rep['moe_noise']['rel_med']:.2e}" if "moe" in rep else "")
                  + f"  resid {rep['residual_mismatch']}  {rep['seconds']}s", flush=True)
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1)
    finally:
        if pool is not None:
            pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
