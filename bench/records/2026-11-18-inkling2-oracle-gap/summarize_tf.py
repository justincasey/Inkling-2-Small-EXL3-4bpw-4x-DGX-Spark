"""Summarize teacher_force.py reports into the tables of docs/ORACLE-GAP.md.

    python summarize_tf.py REPORT.json [REPORT.json ...]
"""

import json
import os
import statistics as s
import sys


def med(xs):
    return s.median(xs) if xs else float("nan")


def main(paths):
    print("| fixture | rows | sublayer | layers | spec oracle | + bf16 P | + int8 KV | emulating (both) | noise floor |")
    print("|---|---|---|---|---|---|---|---|---|")
    extra = []
    for p in paths:
        d = json.load(open(p))
        name = os.path.basename(p).removesuffix(".json")
        L = d["layers"]
        for kind in ("core", "a"):
            for glob in (False, True):
                ls = [x for x in L if x["global"] == glob]
                if not ls:
                    continue
                v = {n: med([x["attn"][n][kind]["rel_med"] for x in ls]) for n in
                     ("none", "attn_bf16_p", "kv8", "kv8+attn_bf16_p")}
                nf = med([x["attn_noise"][kind]["rel_med"] for x in ls])
                print(f"| {name} | {d['rows']} | {'attention core (rank 0)' if kind == 'core' else 'attention a'} "
                      f"| {'global' if glob else 'local'} {len(ls)} | {v['none']:.2e} | {v['attn_bf16_p']:.2e} | "
                      f"{v['kv8']:.2e} | {v['kv8+attn_bf16_p']:.2e} | {nf:.2e} |")
        r = [x["router"] for x in L]
        flips = [f for x in r for f in x["flips"]]
        row = {
            "fixture": name, "rows": d["rows"], "layers": len(L),
            "router_row_layers": d["rows"] * len(L),
            "router_identical_min": min(x["picks_identical"] for x in r),
            "router_flips": len(flips),
            "router_flips_out_of_bound": sum(not f["within_bound"] for f in flips),
            "router_select_consistent": all(x["engine_select_consistent"] for x in r),
            "router_margin_min": min(x["margin_min"] for x in r),
            "router_logit_err_max": max(x["logit_err_max"] for x in r),
            "router_logit_err_rel_sumabs_max": max(x["logit_err_max_rel_sumabs"] for x in r),
            "router_weights_err_max": max(x["weights_err_max"] for x in r),
            "mix_rel_med": med([x["mix"]["rel_med"] for x in L]),
            "mix_rel_max": max(x["mix"]["rel_max"] for x in L),
            "mix_diff_frac_max": max(x["mix"]["diff_frac"] for x in L),
            "mix_noise_rel_max": max(x["mix_noise"]["rel_max"] for x in L) if "mix_noise" in L[0] else None,
            "residual_mismatch_total": sum(x["residual_mismatch"] for x in L),
            "residual_elements": d["rows"] * len(L) * 4608,
        }
        if "moe" in L[0]:
            row.update({
                "moe_rel_med_spec": med([x["moe"]["none"]["rel_med"] for x in L]),
                "moe_rel_med_fp16": med([x["moe"]["moe_fp16"]["rel_med"] for x in L]),
                "moe_rel_med_noise": med([x["moe_noise"]["rel_med"] for x in L]),
                "moe_rel_max_fp16": max(x["moe"]["moe_fp16"]["rel_max"] for x in L),
                "moe_rel_max_noise": max(x["moe_noise"]["rel_max"] for x in L),
                "moe_cos_min_fp16": min(x["moe"]["moe_fp16"]["cos_min"] for x in L),
            })
        extra.append(row)
    print()
    print(json.dumps(extra, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
