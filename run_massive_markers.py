"""Are the option markers that fall outside the SAE ("massive", emenda 1) linked to Laya's choice?

Usage: python run_massive_markers.py --sae checkpoints/sae/en_base_enc17_k32/sae.pt --layer enc_17
           --tokens-dir data/activations/partc_tokens/base --stage3-dir data/activations/partc/base_common
           --out results/sae_features/part_c/massive_markers_base.json

A marker is massive when its activation norm at the layer is above the SAE's `massive_threshold` (the same rule
that excludes it from the SAE). Features cannot see these markers (they are zero), so if being massive goes with
being the option Laya picks, part of the mechanism is invisible to the SAE analysis. For every split this
reports P(massive) by role of the option (picked / correct / neither), by whether the decision is right, by
option position and question type, and the mean probability Laya gives to massive vs non-massive markers.
No test-based selection: it only describes; train is the discovery set, test the replication.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.activations import ROLE_MARKER
from src.sae import load_sae
from src.sae_data import load_tokens

REPO_ROOT = Path(__file__).resolve().parent


def marker_table(tokens_root: Path, stage3_root: Path, layer: str, thr: float, chunk=200_000) -> pd.DataFrame:
    acts, ti, rows = load_tokens(tokens_root, layer)
    dec = pd.read_parquet(stage3_root / "decisions.parquet")
    key = ["case_id", "question_id"]
    meta = rows[key + ["decision_idx", "is_dev"]].merge(
        dec[key + ["qtype", "option_ids", "gold_argmax", "model_probs", "model_pred", "n_options"]], on=key, how="left").sort_values("decision_idx")
    mk = ti[ti["role"] == ROLE_MARKER]
    norms = np.concatenate([np.linalg.norm(np.asarray(acts[mk.index.to_numpy()[a:a + chunk]]).astype(np.float32), axis=1)
                            for a in range(0, len(mk), chunk)])
    t = pd.DataFrame({"decision_idx": mk["decision_idx"].to_numpy(), "option_index": mk["option_index"].to_numpy(), "norm": norms})
    t = t.merge(meta, on="decision_idx")
    opts = [list(o) for o in t["option_ids"]]
    t["is_pred"] = [o[i] == p for o, i, p in zip(opts, t["option_index"], t["model_pred"])]
    t["is_gold"] = [o[i] == g for o, i, g in zip(opts, t["option_index"], t["gold_argmax"])]
    t["prob"] = [float(p[i]) for p, i in zip(t["model_probs"], t["option_index"])]
    t["correct"] = t["model_pred"] == t["gold_argmax"]
    t["massive"] = t["norm"] > thr
    return t


def summarize(t: pd.DataFrame) -> dict:
    g = lambda m: float(t.loc[m, "massive"].mean()) if m.sum() else float("nan")
    out = {"n_markers": int(len(t)), "p_massive": float(t["massive"].mean()),
           "p_massive_picked": g(t.is_pred), "p_massive_not_picked": g(~t.is_pred),
           "p_massive_correct_option": g(t.is_gold), "p_massive_not_correct_option": g(~t.is_gold),
           "p_massive_picked_when_decision_right": g(t.is_pred & t.correct), "p_massive_picked_when_decision_wrong": g(t.is_pred & ~t.correct),
           "p_massive_correct_option_when_decision_wrong": g(t.is_gold & ~t.correct),
           "p_massive_wrong_option_not_picked": g(~t.is_gold & ~t.is_pred),
           "mean_prob_massive": float(t.loc[t.massive, "prob"].mean()) if t.massive.any() else float("nan"),
           "mean_prob_non_massive": float(t.loc[~t.massive, "prob"].mean()),
           "p_picked_given_massive": float(t.loc[t.massive, "is_pred"].mean()) if t.massive.any() else float("nan"),
           "p_picked_given_non_massive": float(t.loc[~t.massive, "is_pred"].mean()),
           "p_decision_right_given_picked_massive": float(t.loc[t.is_pred & t.massive, "correct"].mean()) if (t.is_pred & t.massive).any() else float("nan"),
           "p_decision_right_given_picked_non_massive": float(t.loc[t.is_pred & ~t.massive, "correct"].mean()),
           "by_option_position": t.groupby("option_index")["massive"].mean().round(4).to_dict(),
           "by_qtype": t.groupby("qtype")["massive"].mean().round(4).to_dict()}
    # share of decisions with a massive picked / correct marker, and with any massive marker
    d = t.groupby("decision_idx").agg(any_massive=("massive", "any"), correct=("correct", "first"))
    out["decisions"] = int(len(d)); out["p_decision_has_massive_marker"] = float(d.any_massive.mean())
    out["acc_when_decision_has_massive_marker"] = float(d.loc[d.any_massive, "correct"].mean()) if d.any_massive.any() else float("nan")
    out["acc_when_no_massive_marker"] = float(d.loc[~d.any_massive, "correct"].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--stage3-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.0, help="override the SAE's massive_threshold (e.g. to read the fine-tuned model at the base's value)")
    args = ap.parse_args()
    thr = args.threshold or float(load_sae(args.sae, "cpu").massive_threshold)
    res = {"layer": args.layer, "threshold": thr, "sae": args.sae}
    for split in ("train", "test"):
        t = marker_table(Path(args.tokens_dir) / split, Path(args.stage3_dir) / split, args.layer, thr)
        res[split] = summarize(t)
        print(f"[massive] {split}: P(massive)={res[split]['p_massive']:.3f} picked={res[split]['p_massive_picked']:.3f} "
              f"correct-option={res[split]['p_massive_correct_option']:.3f} neither={res[split]['p_massive_wrong_option_not_picked']:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
