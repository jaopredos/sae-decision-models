"""Rotation behaviour of several models side by side (PROTOCOLO.md, emenda 8).

Usage: python run_rotation_compare.py --dirs multilingual=data/activations/rotation/test base_en=... ft_en=... [--out results/sae_features/rotation_compare]

For each directory (a rotation extraction of the test split) it reports, by question type, the flip rate, the accuracy averaged over
rotations and the position preference (`behaviour` of run_rotation_analysis), and the paired case-bootstrap difference of the flip rate
between every other model and the first one. Models are different networks: differences are descriptive, not causal.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_rotation_analysis import behaviour, case_boot, load_rot


def flips(rows: pd.DataFrame) -> pd.DataFrame:
    dec = rows["case_id"] + "|" + rows["question_id"]
    g = rows.assign(dec=dec).groupby("dec").agg(case_id=("case_id", "first"), qtype=("qtype", "first"), n=("pred_idx", "nunique"))
    g["flip"] = (g["n"] > 1).astype(float)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="name=rotation_split_dir (each holds rows.parquet)")
    ap.add_argument("--out", default="results/sae_features/rotation_compare")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    named = [d.split("=", 1) for d in args.dirs]
    rows = {n: pd.read_parquet(Path(p) / "rows.parquet") for n, p in named}
    res = {"behaviour": {n: behaviour(r.assign(grp=r["workflow"] + "|" + r["question_id"])) for n, r in rows.items()}, "flip": {}, "flip_diff_vs_first": {}}
    fl = {n: flips(r) for n, r in rows.items()}
    for n, g in fl.items():
        res["flip"][n] = {t: float(g[g.qtype == t].flip.mean()) for t in ("score", "choice", "noul")} | {"n_decisions": int(len(g))}
    first = named[0][0]
    for n, _ in named[1:]:
        d = {}
        both = fl[first].add_suffix("_a").join(fl[n].add_suffix("_b"), how="inner")
        for t in ("score", "choice", "noul"):
            x = both[both.qtype_a == t]
            m, ci = case_boot(x["case_id_a"].to_numpy(), (x["flip_b"] - x["flip_a"]).to_numpy(), np.ones(len(x)))
            d[t] = {"diff_pp": 100 * m, "ci95_pp": [100 * c for c in ci], "n": int(len(x))}
        res["flip_diff_vs_first"][f"{n}-{first}"] = d
    (out / "compare.json").write_text(json.dumps(res, indent=1, default=float))
    for n in fl:
        print(n, {k: round(v, 3) for k, v in res["flip"][n].items() if k != "n_decisions"}, {t: round(res["behaviour"][n]["accuracy"][t]["mean_over_rotations"], 3) for t in ("score", "choice", "noul")})
    for k, d in res["flip_diff_vs_first"].items():
        print(k, {t: (round(v["diff_pp"], 1), [round(c, 1) for c in v["ci95_pp"]]) for t, v in d.items()})


if __name__ == "__main__":
    main()
