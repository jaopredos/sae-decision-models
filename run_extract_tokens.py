"""Stage 4, step 1: activations of ALL non-padding tokens of the typed-decisions prompts, per layer.

Usage: python run_extract_tokens.py --split {train,test} --layers enc_17,enc_08 [--limit N]
           [--device cuda|mps|cpu] [--out data/activations/tokens] [--gate-dir data/activations/typed_decisions]
           [--repo R --subfolder S --revision V --max-len N --head-max-len N] [--name multilingual]

Writes <out>/<name>/<split>/:
  tokens_<layer>.npy   float16 [n_tokens, hidden]: every non-padding token of every row (one row per
                       question), rows in decision order, tokens in sequence order
  token_index.parquet  one line per token: decision_idx, position, token_id, role, option_index
                       (role codes: src.activations.ROLE_*; option_index is -1 off the markers)
  rows.parquet         one line per decision: case_id, question_id, decision_idx, n_options, offset,
                       length, n_state_tokens_dropped, is_dev
  meta.json            versions, parity numbers and the gate against the stage-3 marker vectors

The inputs are exactly those of `agent.predict` (same internal formatter), so the marker vectors are
the ones of stage 3. Gate (results/sae_features/PROTOCOLO.md): relative L2 error per marker vector
<= 1e-3 against the stage-3 files, and |dp| <= 1e-4 between a hooked and an un-hooked predict.
Exits non-zero before writing the activations' final state if a gate fails.
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

from src.activations import (ROLE_MARKER, extract_tokens_case, layer_modules, load_agent, LAYA_REPO,
                             LAYA_REVISION, LAYA_SUBFOLDER)
from src.datasets import TYPED_DECISIONS_REVISION, load_typed_cases, typed_dev_ids
from src.models import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parent
PROB_TOL = 1e-4
GATE_REL_L2 = 1e-3


def count_tokens(agent, cases, pk) -> int:
    """Number of non-padding tokens the run will see; uses the same encoder as `predict`."""
    n = 0
    for case in cases:
        ids = list(case["questions"].keys())
        internal = {qid: agent._to_internal(case["questions"][qid]) for qid in ids}
        items = agent._encode_state(case["state"], ids, internal, **pk)
        n += sum(len(it["ids"]) for it in items)
    return n


def gate_against_stage3(out_dir: Path, gate_dir: Path, labels, token_index: pd.DataFrame) -> dict:
    """Compare marker vectors with the stage-3 marker files (same order by construction)."""
    mk3 = pd.read_parquet(gate_dir / "markers.parquet")
    mk = token_index[token_index["role"] == ROLE_MARKER]
    if len(mk3) != len(mk) or not (mk3["decision_idx"].to_numpy() == mk["decision_idx"].to_numpy()).all() \
            or not (mk3["option_index"].to_numpy() == mk["option_index"].to_numpy()).all():
        return {"passed": False, "reason": "marker order differs from stage 3"}
    rows = mk.index.to_numpy()
    res = {"passed": True, "layers": {}}
    for label in labels:
        f3 = next(gate_dir.glob(f"L*_{label}.npy"))
        a = np.load(out_dir / f"tokens_{label}.npy", mmap_mode="r")[rows].astype(np.float32)
        b = np.load(f3).astype(np.float32)
        rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1).clip(1e-8)
        ok = bool(rel.max() <= GATE_REL_L2)
        res["layers"][label] = {"n_markers": len(rel), "max_rel_l2": float(rel.max()),
                                "mean_rel_l2": float(rel.mean()), "n_above_tol": int((rel > GATE_REL_L2).sum()),
                                "passed": ok}
        res["passed"] &= ok
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--layers", required=True, help="comma list of labels, e.g. enc_17,enc_08")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "activations" / "tokens"))
    ap.add_argument("--name", default="multilingual")
    ap.add_argument("--gate-dir", default=None, help="stage-3 dir with L*_<label>.npy and markers.parquet")
    ap.add_argument("--repo", default=LAYA_REPO)
    ap.add_argument("--subfolder", default=LAYA_SUBFOLDER)
    ap.add_argument("--revision", default=LAYA_REVISION)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--head-max-len", type=int, default=0)
    args = ap.parse_args()
    subfolder = None if args.subfolder in ("", "none", "None") else args.subfolder
    pk = {}
    if args.max_len:
        pk["max_len"] = args.max_len
    if args.head_max_len:
        pk["head_max_len"] = args.head_max_len

    set_all_seeds()
    labels = args.layers.split(",")
    out_dir = Path(args.out) / args.name / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_typed_cases(args.split)
    if args.limit:
        cases = cases[: args.limit]
    dev_ids = typed_dev_ids() if args.split == "train" else set()

    agent = load_agent(args.device, args.repo, subfolder, args.revision)
    all_modules = dict(layer_modules(agent.model))
    modules = [(lab, all_modules[lab]) for lab in labels]
    hidden = agent.model.encoder.config.hidden_size
    n_tok = count_tokens(agent, cases, pk)
    print(f"[tokens] split={args.split} cases={len(cases)} device={agent.device} layers={labels} hidden={hidden} "
          f"tokens={n_tok:,} predict_kwargs={pk}", flush=True)

    mm = {lab: np.lib.format.open_memmap(out_dir / f"tokens_{lab}.npy", mode="w+", dtype=np.float16,
                                         shape=(n_tok, hidden)) for lab in labels}
    idx_cols = {k: [] for k in ("decision_idx", "position", "token_id", "role", "option_index")}
    rows_out, parity_max, off, t0 = [], 0.0, 0, time.time()
    for n, case in enumerate(cases, 1):
        res = extract_tokens_case(agent, modules, case, pk)
        parity_max = max(parity_max, max(res["parity"]))
        for r in res["rows"]:
            d, length = len(rows_out), len(r["token_ids"])
            opt_idx = np.full(length, -1, dtype=np.int8)
            opt_idx[r["markers"]] = np.arange(len(r["markers"]), dtype=np.int8)
            idx_cols["decision_idx"].append(np.full(length, d, dtype=np.int32))
            idx_cols["position"].append(np.arange(length, dtype=np.int16))
            idx_cols["token_id"].append(r["token_ids"])
            idx_cols["role"].append(r["roles"])
            idx_cols["option_index"].append(opt_idx)
            rows_out.append({"case_id": r["case_id"], "question_id": r["question_id"], "decision_idx": d,
                             "n_options": r["n_options"], "offset": off, "length": length,
                             "n_state_tokens_dropped": r["n_state_tokens_dropped"],
                             "is_dev": r["case_id"] in dev_ids})
            off += length
        start = off - sum(len(r["token_ids"]) for r in res["rows"])
        for lab in labels:
            mm[lab][start:off] = res["acts"][lab]
        if n % 100 == 0 or n == len(cases):
            print(f"[tokens] {n}/{len(cases)} cases, {time.time() - t0:.0f}s", flush=True)
    assert off == n_tok, (off, n_tok)
    for lab in labels:
        mm[lab].flush()
    del mm

    token_index = pd.DataFrame({k: np.concatenate(v) for k, v in idx_cols.items()})
    token_index.to_parquet(out_dir / "token_index.parquet")
    pd.DataFrame(rows_out).to_parquet(out_dir / "rows.parquet")

    gate = {"passed": True, "skipped": True}
    if args.gate_dir and Path(args.gate_dir).exists():
        gate = gate_against_stage3(out_dir, Path(args.gate_dir) / args.split, labels, token_index)
    parity_ok = parity_max <= PROB_TOL
    import laya
    import transformers
    meta = {"split": args.split, "n_cases": len(cases), "n_decisions": len(rows_out), "n_tokens": n_tok,
            "layers": labels, "hidden_size": hidden, "predict_kwargs": pk, "device": str(agent.device),
            "autocast_dtype_5_rows": str(agent.dtype_for(5)),
            "repo": args.repo, "subfolder": subfolder, "revision": args.revision,
            "dataset_revision": TYPED_DECISIONS_REVISION,
            "parity_hooked_vs_unhooked_max_abs": parity_max, "parity_passed": parity_ok, "gate_stage3": gate,
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "laya": laya.__version__,
                         "transformers": transformers.__version__, "cuda": torch.version.cuda},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "elapsed_s": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[tokens] parity:", parity_max, "passed" if parity_ok else "FAILED", flush=True)
    print("[tokens] gate vs stage 3:", json.dumps(gate), flush=True)
    if not parity_ok or not gate["passed"]:
        print("[tokens] GATE FAILED - activations written for inspection, do not use them.", flush=True)
        sys.exit(2)
    print(f"[tokens] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
