"""Stage 5: gradient attribution of Laya's wrong decisions to SAE features (see src/attribution.py).

Usage: python run_feature_attribution.py --sae checkpoints/sae/ml_enc17_k32/sae.pt --layer enc_17
           [--splits train,test] [--limit N] [--device cuda] [--out results/sae_features/<name>/attribution]
           [--repo R --subfolder S --revision V]

One forward+backward per case in fp32 (no autocast), the official bf16 `predict` only defines which
decisions are wrong. Writes attribution_<split>.npz (case_id, question_id, delta, a_pred, a_corr, a_state,
a_other as float16 [n_wrong, F], ...) and attribution_summary.json (how often the fp32 argmax differs from the
official one, the pooled fraction of h.g carried by the SAE features, markers that are massive).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER, load_agent
from src.attribution import attribute_case
from src.datasets import load_typed_cases
from src.models import set_all_seeds
from src.sae import load_sae

REPO_ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=LAYA_REPO)
    ap.add_argument("--subfolder", default=LAYA_SUBFOLDER)
    ap.add_argument("--revision", default=LAYA_REVISION)
    args = ap.parse_args()
    sub = None if args.subfolder in ("", "none", "None") else args.subfolder
    set_all_seeds()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agent = load_agent(args.device, args.repo, sub, args.revision)
    sae = load_sae(args.sae, str(agent.device)).eval()
    summary, t0 = {}, time.time()
    for split in args.splits.split(","):
        cases = load_typed_cases(split)
        if args.limit:
            cases = cases[: args.limit]
        recs = []
        for n, case in enumerate(cases, 1):
            recs.extend(attribute_case(agent, sae, args.layer, case))
            if n % 100 == 0 or n == len(cases):
                print(f"[attr] {split} {n}/{len(cases)} cases, {len(recs)} wrong decisions, {time.time() - t0:.0f}s", flush=True)
        arrays = {k: np.stack([r[k] for r in recs]) for k in ("a_pred", "a_corr", "a_state", "a_other")}
        meta = {k: np.array([r[k] for r in recs]) for k in ("case_id", "question_id", "pred_idx", "gold_idx", "delta",
                                                            "fp32_pred_idx", "total_attr", "total_hg", "marker_pred_massive", "marker_corr_massive")}
        np.savez_compressed(out / f"attribution_{split}.npz", **arrays, **meta)
        summary[split] = {"n_cases": len(cases), "n_wrong": len(recs),
                          "frac_fp32_argmax_differs_from_official": float((meta["fp32_pred_idx"] != meta["pred_idx"]).mean()),
                          "mean_delta": float(meta["delta"].mean()),
                          "pooled_explained_fraction": float(meta["total_attr"].sum() / meta["total_hg"].sum()),
                          "markers_massive_pred": int(meta["marker_pred_massive"].sum()), "markers_massive_corr": int(meta["marker_corr_massive"].sum())}
        print(f"[attr] {split}: {json.dumps(summary[split])}", flush=True)
    (out / "attribution_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
