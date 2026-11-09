"""Stage A: per-layer activations at Laya multilingual's option markers on typed-decisions.

Usage: python run_extract.py --split {train,test} [--limit N] [--device cuda|mps|cpu]
                             [--out data/activations/typed_decisions]
Writes, under <out>/<split>/:
  L{idx:02d}_{label}.npy   fp16 [n_markers, hidden], one file per layer, rows = markers.parquet
  markers.parquet          one row per option marker (case_id, question_id, option_index, option_id,
                           decision_idx)
  decisions.parquet        one row per decision (gold, Laya's predict output, truncation, is_dev)
  parity.json, truncation.json, meta.json
Exits non-zero, before writing activations, if the parity check fails (see src/activations.py).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.activations import LAYA_REVISION, extract_case, layer_modules, load_agent
from src.datasets import TYPED_DECISIONS_REVISION, load_typed_cases, typed_dev_ids
from src.models import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parent

# |Δp| between the recomputed and the official probabilities. predict() rounds to 4 decimals,
# so up to 5e-5 is pure rounding; 1e-4 leaves room for one ulp of the softmax on top.
PROB_TOL = 1e-4


def summarize_parity(parity: list) -> dict:
    df = pd.DataFrame(parity)
    cols = [c for c in df.columns if c.startswith(("logit_", "prob_"))]
    out = {c: {"max": float(df[c].max()), "mean": float(df[c].mean())} for c in cols}
    bad_argmax = df[~df["argmax_equal"] & ~df["near_tie"]]
    out["n_decisions"] = len(df)
    out["argmax_mismatch_excluding_ties"] = int(len(bad_argmax))
    out["prob_tol"] = PROB_TOL
    out["passed"] = bool(
        df["prob_absdiff_recomputed_vs_hooked"].max() <= PROB_TOL
        and df["prob_absdiff_recomputed_vs_ref"].max() <= PROB_TOL
        and len(bad_argmax) == 0)
    return out


def summarize_truncation(dec: pd.DataFrame) -> dict:
    per_case = dec.groupby("case_id")["state_truncated"].any()
    return {
        "n_cases": int(dec["case_id"].nunique()),
        "n_decisions": int(len(dec)),
        "cases_with_state_truncated": int(per_case.sum()),
        "decisions_with_state_truncated": int(dec["state_truncated"].sum()),
        "state_tokens_dropped_total": int(dec["state_tokens_dropped"].sum()),
        "state_tokens_max": int(dec["state_tokens"].max()),
        "seq_len_max": int(dec["seq_len"].max()),
        "options_cut_at_48_tokens": int(dec["options_over_48_tokens"].sum()),
        "decisions_with_head_budget_cap": int(dec["tokens_per_option_cap"].notna().sum()),
        "decisions_with_collapsed_options": int(dec["options_collapsed"].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (0 = all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "activations" / "typed_decisions"))
    args = ap.parse_args()

    set_all_seeds()
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_typed_cases(args.split)
    if args.limit:
        cases = cases[: args.limit]
    dev_ids = typed_dev_ids() if args.split == "train" else set()

    agent = load_agent(args.device)
    modules = layer_modules(agent.model)
    cfg = agent.model.encoder.config
    print(f"[extract] split={args.split} cases={len(cases)} device={agent.device} "
          f"autocast={agent.dtype_for(5)} max_len={agent.cfg.get('max_len')} "
          f"head_max_len={agent.cfg.get('head_max_len')} hidden={cfg.hidden_size} "
          f"layers={[m[0] for m in modules][:2]}..{[m[0] for m in modules][-3:]} ({len(modules)})",
          flush=True)

    decisions, markers, parity = [], [], []
    vectors = {label: [] for label, _ in modules}
    t0 = time.time()
    for n, case in enumerate(cases, 1):
        res = extract_case(agent, modules, case)
        for d in res["decisions"]:
            d["is_dev"] = case["case_id"] in dev_ids
            d["decision_idx"] = len(decisions)
            decisions.append(d)
        base = len(decisions) - len(res["decisions"])
        for m in res["markers"]:
            # decisions of one case are contiguous and in question order
            qpos = [d["question_id"] for d in res["decisions"]].index(m["question_id"])
            m["decision_idx"] = base + qpos
            markers.append(m)
        parity.extend(res["parity"])
        for label in vectors:
            vectors[label].append(res["vectors"][label])
        if n % 100 == 0 or n == len(cases):
            print(f"[extract] {n}/{len(cases)} cases, {time.time() - t0:.0f}s", flush=True)

    par = summarize_parity(parity)
    pd.DataFrame(parity).to_parquet(out_dir / "parity_per_decision.parquet")
    (out_dir / "parity.json").write_text(json.dumps(par, indent=2))
    print("[extract] parity:", json.dumps(par, indent=2), flush=True)
    if not par["passed"]:
        worst = pd.DataFrame(parity).sort_values("prob_absdiff_recomputed_vs_ref", ascending=False)
        print("[extract] PARITY FAILED - worst decisions:\n", worst.head(10).to_string(), flush=True)
        sys.exit(2)

    dec = pd.DataFrame(decisions)
    mk = pd.DataFrame(markers)
    trunc = summarize_truncation(dec)
    print("[extract] truncation:", json.dumps(trunc), flush=True)

    for i, (label, _) in enumerate(modules):
        arr = np.concatenate(vectors[label])
        assert arr.shape == (len(mk), cfg.hidden_size), (label, arr.shape)
        np.save(out_dir / f"L{i:02d}_{label}.npy", arr)
    dec.to_parquet(out_dir / "decisions.parquet")
    mk.to_parquet(out_dir / "markers.parquet")
    (out_dir / "truncation.json").write_text(json.dumps(trunc, indent=2))

    import laya
    import transformers
    meta = {
        "split": args.split, "n_cases": len(cases), "limit": args.limit,
        "n_decisions": len(dec), "n_markers": len(mk), "hidden_size": cfg.hidden_size,
        "layers": [label for label, _ in modules],
        "laya_revision": LAYA_REVISION, "dataset_revision": TYPED_DECISIONS_REVISION,
        "device": str(agent.device), "autocast_dtype_5_rows": str(agent.dtype_for(5)),
        "max_len": agent.cfg.get("max_len"), "head_max_len": agent.cfg.get("head_max_len"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "laya": laya.__version__, "transformers": transformers.__version__,
                     "cuda": torch.version.cuda},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[extract] wrote {len(modules)} layer files + parquet to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
