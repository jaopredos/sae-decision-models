"""Stage 7 (PROTOCOLO.md, emenda 6, parte A): Laya under cyclic rotations of the option order.

Usage: python run_rotation_extract.py --split {train,test} [--limit N] [--device cuda|mps|cpu] [--out data/activations/rotation]

For every decision with k options and every r in 0..k-1, the question is shown with `option_order[j] = (j + r) % k`
(slot j shows option order[j]; r = 0 is the original order). One `predict` per (case, r) with the questions that have
more than r options; the marker vectors of all 26 layers come from the forward inside that `predict` (same capture as
stage 3), and the official probabilities from the same call. The gold label does not change.

Writes <out>/<split>/:
  rows.parquet     one line per (decision, rotation): case_id, question_id, workflow, qtype, k, rot, is_dev, option_ids (canonical
                   order), order (slot -> canonical option index), probs (canonical order), pred_idx (canonical), pred_slot,
                   gold_idx (canonical), gold_slot
  markers.parquet  one line per marker: row_idx, slot, opt_idx (canonical option index)
  L??_<label>.npy  float16 [n_markers, hidden] per layer, rows = markers.parquet
  meta.json
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER, capture, layer_modules, load_agent, option_ids
from src.datasets import load_typed_cases, typed_dev_ids
from src.models import normalize_answers, set_all_seeds

REPO_ROOT = Path(__file__).resolve().parent


def rotation(k: int, r: int) -> list:
    return [(j + r) % k for j in range(k)]


@torch.no_grad()
def rotated_call(agent, modules, case, r):
    """Run one (case, r); returns per-decision records and marker vectors per layer, or None if no question has > r options."""
    questions = {}
    for qid, q in case["questions"].items():
        k = len(option_ids(agent._to_internal(q)))
        if k > r:
            questions[qid] = {**q, "option_order": rotation(k, r)}
    if not questions:
        return None
    with capture(agent, modules) as cap:
        ans = agent.predict(case["state"], questions)["answers"]
    assert len(cap.forward_calls) == 1 and len(cap.encode_calls) == 1
    ids, internal, items = cap.encode_calls[0]
    marker_pos = cap.forward_calls[0][0][2]
    recs, vec = [], {label: [] for label, _ in modules}
    gathered = {}
    for label, _ in modules:
        h = cap.outputs[label]
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        gathered[label] = torch.gather(h, 1, idx).float().cpu()
    for j, qid in enumerate(ids):
        opts = option_ids(internal[qid])
        k = len(opts)
        order = internal[qid]["option_order"]
        probs = normalize_answers({qid: ans[qid]})[qid]["probabilities"]
        p = np.array([probs[o] for o in opts], dtype=np.float64)
        gold_idx = opts.index(str(case["gold"][qid]["label"]))
        pred_idx = int(p.argmax())
        recs.append({"case_id": case["case_id"], "question_id": qid, "workflow": case["workflow"], "qtype": internal[qid]["t"], "k": k, "rot": r,
                     "option_ids": opts, "order": order, "probs": p.tolist(), "pred_idx": pred_idx, "pred_slot": order.index(pred_idx),
                     "gold_idx": gold_idx, "gold_slot": order.index(gold_idx)})
        assert len(items[j]["markers"]) == k
        for label in vec:
            vec[label].append(gathered[label][j, :k].numpy().astype(np.float16))
    return recs, {label: np.concatenate(v) for label, v in vec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "activations" / "rotation"))
    args = ap.parse_args()
    set_all_seeds()
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    cases = load_typed_cases(args.split)
    if args.limit:
        cases = cases[: args.limit]
    dev_ids = typed_dev_ids() if args.split == "train" else set()
    agent = load_agent(args.device, LAYA_REPO, LAYA_SUBFOLDER, LAYA_REVISION)
    modules = layer_modules(agent.model)
    rows, markers, vecs, t0 = [], [], {label: [] for label, _ in modules}, time.time()
    for n, case in enumerate(cases, 1):
        for r in range(5):
            res = rotated_call(agent, modules, case, r)
            if res is None:
                continue
            recs, v = res
            for rec in recs:
                rec["is_dev"] = case["case_id"] in dev_ids
                row_idx = len(rows)
                rows.append(rec)
                for slot, opt in enumerate(rec["order"]):
                    markers.append((row_idx, slot, opt))
            for label in vecs:
                vecs[label].append(v[label])
        if n % 100 == 0 or n == len(cases):
            print(f"[rot] {n}/{len(cases)} cases, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "rows.parquet")
    mk = pd.DataFrame(markers, columns=["row_idx", "slot", "opt_idx"])
    mk.to_parquet(out / "markers.parquet")
    for i, (label, _) in enumerate(modules):
        arr = np.concatenate(vecs[label])
        assert arr.shape[0] == len(mk), (label, arr.shape, len(mk))
        np.save(out / f"L{i:02d}_{label}.npy", arr)
    import laya
    import transformers
    meta = {"split": args.split, "n_cases": len(cases), "n_rows": len(df), "n_markers": len(mk), "layers": [m[0] for m in modules],
            "rows_by_qtype": df.groupby("qtype").size().to_dict(),
            "acc_rot0": float((df[df.rot == 0].pred_idx == df[df.rot == 0].gold_idx).mean()),
            "acc_all_rotations": float((df.pred_idx == df.gold_idx).mean()),
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "laya": laya.__version__, "transformers": transformers.__version__},
            "device": str(agent.device), "elapsed_s": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[rot] done", json.dumps({k: meta[k] for k in ("n_rows", "n_markers", "acc_rot0", "acc_all_rotations", "elapsed_s")}), flush=True)


if __name__ == "__main__":
    main()
