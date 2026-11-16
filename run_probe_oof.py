"""Stage 5, J3: out-of-fold predictions of the stage-3 linear probe, so that "the probe is right and
Laya is wrong" can be defined on train/dev decisions without ever using the test split.

Usage: python run_probe_oof.py --layers enc_17:1e-5,enc_08:1e-3 [--acts data/activations/typed_decisions]
           [--out results/sae_features] [--folds 5] [--device cuda|cpu]

For each layer: 5 case-grouped folds (fold = sha256(case_id) % folds, over all train cases, dev
included); each fold is predicted by a probe fitted on the other folds (features standardised with
that fit set's statistics, weight decay given per layer from stage 3, seed 0; the objective is convex
so the seed only moves the initialisation). The test split is predicted by a probe fitted on the whole
train split, and its agreement with the stage-3 `per_decision.parquet` (enc_17) is reported.

Writes <out>/probe_oof_<layer>.parquet with one line per decision of train and test:
split, case_id, question_id, option_ids, probe_pred, probe_probs, fold (-1 for test).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import probe as P

REPO_ROOT = Path(__file__).resolve().parent


def fold_of(case_id: str, folds: int) -> int:
    return int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % folds


def fit_predict(train: P.SplitData, xtr, fit_rows, target_x, target_mask, wd, device, seed=0):
    std = P.standardizer(xtr[fit_rows], train.mask[fit_rows])
    xf = torch.from_numpy(std(xtr[fit_rows], train.mask[fit_rows])).to(device)
    mf = torch.from_numpy(train.mask[fit_rows]).to(device)
    gf = torch.from_numpy(train.gold[fit_rows]).to(device)
    params = P.fit_probe(xf, mf, gf, wd, seed)
    xt = torch.from_numpy(std(target_x, target_mask)).to(device)
    return P.predict_probe(params, xt, torch.from_numpy(target_mask).to(device))


def records(split: str, sd: P.SplitData, probs: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    d = sd.dec
    pr = [probs[i, :n].tolist() for i, n in enumerate(d["n_options"])]
    return pd.DataFrame({"split": split, "case_id": d["case_id"], "question_id": d["question_id"],
                         "option_ids": [list(o) for o in d["option_ids"]],
                         "probe_pred": [list(o)[int(np.argmax(p))] for o, p in zip(d["option_ids"], pr)],
                         "probe_probs": pr, "fold": folds})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True, help="label:wd,label:wd (wd from stage 3)")
    ap.add_argument("--acts", default=str(REPO_ROOT / "data" / "activations" / "typed_decisions"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sae_features"))
    ap.add_argument("--stage3-per-decision", default=str(REPO_ROOT / "results" / "probe_typed" / "per_decision.parquet"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acts = Path(args.acts)
    train, test = P.load_split(acts / "train"), P.load_split(acts / "test")
    fold = np.array([fold_of(c, args.folds) for c in train.dec["case_id"]])
    summary = {}
    for item in args.layers.split(","):
        label, wd = item.split(":")
        wd = float(wd)
        stem = next(s for s in train.layers if s.split("_", 1)[1] == label)
        xtr, xte = train.layer(stem), test.layer(stem)
        probs_tr = np.zeros(train.gold.shape, dtype=np.float64)
        for f in range(args.folds):
            tgt = fold == f
            probs_tr[tgt] = fit_predict(train, xtr, ~tgt, xtr[tgt], train.mask[tgt], wd, args.device)
        probs_te = fit_predict(train, xtr, np.ones(len(fold), bool), xte, test.mask, wd, args.device)
        df = pd.concat([records("train", train, probs_tr, fold),
                        records("test", test, probs_te, np.full(len(test.dec), -1))], ignore_index=True)
        df.to_parquet(out / f"probe_oof_{label}.parquet")
        acc_oof = float((probs_tr.argmax(1) == train.gold_idx).mean())
        acc_te = float((probs_te.argmax(1) == test.gold_idx).mean())
        dev = train.dec["is_dev"].to_numpy()
        info = {"wd": wd, "oof_acc_train": acc_oof, "oof_acc_dev": float((probs_tr.argmax(1) == train.gold_idx)[dev].mean()),
                "test_acc_fit_on_all_train": acc_te, "folds": args.folds}
        s3 = Path(args.stage3_per_decision)
        if s3.exists():
            ref = pd.read_parquet(s3)
            m = ref.merge(df[df.split == "test"][["case_id", "question_id", "probe_pred"]],
                          on=["case_id", "question_id"], suffixes=("_stage3", "_oof"))
            info["stage3_per_decision_layer"] = str(ref["probe_layer"].iloc[0])
            if str(ref["probe_layer"].iloc[0]) == label:
                info["test_agreement_with_stage3"] = float((m["probe_pred_stage3"] == m["probe_pred_oof"]).mean())
        summary[label] = info
        print(f"[oof] {label}: {json.dumps(info)}", flush=True)
    (out / "probe_oof_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
