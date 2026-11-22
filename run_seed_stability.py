"""PROTOCOLO.md, emenda 6, parte B: is the feature ranking stable across SAE seeds?

Usage: python run_seed_stability.py --layer enc_08 --a ml_enc08_k32 --b ml_enc08_k32_s1 [--out results/sae_features/seed_stability]

(ii) fraction of seed-0 features matched to a seed-1 feature by mutual nearest decoder neighbour (cosine >= 0.7);
(iii) of the 20 neg-list features of seed 0 (variant `pred`, selected on discovery only), the fraction whose matched seed-1 feature is also
      among the 100 most negative mean-`a_pred` features of seed 1 (same rule and support as the ablation selection).
(iv) is run by the ablation job (results/sae_features/ablation/<b>/); it is only compared here when its summary exists.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run_ablation import select_features
from run_part_c_compare import PAIR_COS, pair_by_decoder
from src.sae import load_sae

REPO_ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True)
    ap.add_argument("--a", required=True, help="seed-0 SAE name")
    ap.add_argument("--b", required=True, help="seed-1 SAE name")
    ap.add_argument("--sae-dir", default=str(REPO_ROOT / "checkpoints" / "sae"))
    ap.add_argument("--stats-dir", default=str(REPO_ROOT / "results" / "sae_features"))
    ap.add_argument("--ablation-dir", default=str(REPO_ROOT / "results" / "sae_features" / "ablation"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sae_features" / "seed_stability"))
    args = ap.parse_args()
    sa, sb = (load_sae(str(Path(args.sae_dir) / n / "sae.pt"), "cpu") for n in (args.a, args.b))
    perm, cos = pair_by_decoder(sa, sb, PAIR_COS)
    res = {"layer": args.layer, "seed0": args.a, "seed1": args.b, "n_features": int(len(perm)), "matched": int((perm >= 0).sum()),
           "matched_fraction": float((perm >= 0).mean()), "median_cosine_of_nearest": float(np.median(cos))}
    sel0 = json.loads((Path(args.ablation_dir) / args.a / "selected_features.json").read_text())["pred"]
    sel1 = select_features(Path(args.stats_dir) / args.b)["pred"]
    top1 = set(sel1["neg"][:100])
    neg20 = sel0["neg"][:20]
    mapped = [int(perm[f]) for f in neg20]
    res["list_stability"] = {"neg20_seed0": neg20, "matched_in_seed1": mapped, "n_matched": int(sum(m >= 0 for m in mapped)),
                             "n_matched_and_in_seed1_top100": int(sum(m in top1 for m in mapped)),
                             "fraction_of_20": float(sum(m in top1 for m in mapped) / 20),
                             "fraction_of_matched": float(sum(m in top1 for m in mapped) / max(1, sum(m >= 0 for m in mapped))),
                             "overlap_without_matching": int(len(set(neg20) & top1))}
    for name, tag in ((args.a, "seed0"), (args.b, "seed1")):
        f = Path(args.ablation_dir) / name / "dev" / "summary.json"
        if f.exists():
            h = json.loads(f.read_text())["headline"]["neg_K20_pred"]
            res[f"ablation_dev_neg_K20_pred_{tag}"] = {k: h[k] for k in ("dm_err", "ci95", "fixed", "broken", "acc", "null_p95", "p_empirical", "dm_over_mean_abs_margin")}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.layer}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "list_stability"}, indent=1), json.dumps(res["list_stability"]))


if __name__ == "__main__":
    main()
