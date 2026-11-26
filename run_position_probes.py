"""Stage 8, probes (PROTOCOLO.md, emenda 7): what the marker vectors encode about position, content and the answer, layer by layer.

Usage: python run_position_probes.py [--rot-dir data/activations/rotation] [--out results/sae_features/position_probes] [--device cuda]

Reuses the rotation extraction (no new model pass). Trained on the discovery rows (is_dev False) of the rotation train split,
read on dev (the is_dev rows) and on the rotation test split.
  slot     linear classifier predicting the slot (0..k-1) from ONE marker vector, one classifier per question group
           (workflow x question), chance 1/k.
  content  same, predicting the option index (the option's identity in that question), chance 1/k.
  answer   the stage-3 scorer (src/probe.py, softmax over the options, soft gold) trained on ALL rotations, so position is
           decorrelated from the answer; accuracy is the mean over rotations of each decision, decision-weighted.
Everything is reported by question type; the layer is chosen on dev, never on test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_rotation_analysis import case_boot
from src import probe as P
from src.datasets import load_typed_cases

REPO_ROOT = Path(__file__).resolve().parent
WD_CLASS, WD_ANSWER = 1e-2, 1e-3
WIDTH = 5


def load_split(rot_dir: Path, split: str):
    rows, mk = pd.read_parquet(rot_dir / split / "rows.parquet"), pd.read_parquet(rot_dir / split / "markers.parquet")
    rows["grp"] = rows["workflow"] + "|" + rows["question_id"]
    mk["grp"] = rows["grp"].to_numpy()[mk["row_idx"].to_numpy()]
    mk["qtype"] = rows["qtype"].to_numpy()[mk["row_idx"].to_numpy()]
    mk["k"] = rows["k"].to_numpy()[mk["row_idx"].to_numpy()]
    mk["is_dev"] = rows["is_dev"].to_numpy()[mk["row_idx"].to_numpy()]
    return rows, mk


def soft_gold(rows: pd.DataFrame, split: str) -> np.ndarray:
    """[n, 5] gold distribution in SLOT order for every (decision, rotation) row."""
    cases = {c["case_id"]: c for c in load_typed_cases("test" if split == "test" else "train")}
    out = np.zeros((len(rows), WIDTH), dtype=np.float32)
    for i, (cid, q, opts, order) in enumerate(zip(rows["case_id"], rows["question_id"], rows["option_ids"], rows["order"])):
        pr = cases[cid]["gold"][q]["probabilities"]
        for j, o in enumerate(order):
            out[i, j] = float(pr.get(str(opts[o]), 0.0))
    return out


def fit_class(x, y, n_cls, wd=WD_CLASS, iters=200):
    w = torch.zeros(x.shape[1], n_cls, device=x.device, requires_grad=True)
    b = torch.zeros(n_cls, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w + b, y) + wd * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def decode(tr, te, col, device):
    """Per-group classifier of `col` (slot or opt_idx); returns {qtype: (accuracy, chance)} on `te`."""
    xtr, xte = tr[0], te[0]
    correct, chance, qt = [], [], []
    for g in np.unique(tr[1]["grp"]):
        itr, ite = np.flatnonzero(tr[1]["grp"].to_numpy() == g), np.flatnonzero(te[1]["grp"].to_numpy() == g)
        if len(ite) == 0:
            continue
        k = int(tr[1]["k"].to_numpy()[itr][0])
        mu, sd = xtr[itr].mean(0), xtr[itr].std(0) + 1e-6
        w, b = fit_class(torch.from_numpy((xtr[itr] - mu) / sd).to(device), torch.from_numpy(tr[1][col].to_numpy()[itr]).long().to(device), k)
        pred = (torch.from_numpy((xte[ite] - mu) / sd).to(device) @ w + b).argmax(1).cpu().numpy()
        correct.append(pred == te[1][col].to_numpy()[ite]); chance.append(np.full(len(ite), 1 / k)); qt.append(te[1]["qtype"].to_numpy()[ite])
    if not correct:
        return {}
    correct, chance, qt = np.concatenate(correct), np.concatenate(chance), np.concatenate(qt)
    return {t: (float(correct[qt == t].mean()), float(chance[qt == t].mean())) for t in ("score", "choice", "noul")} | {"all": (float(correct.mean()), float(chance.mean()))}


def dense(x, mk_idx, n_rows):
    """Marker vectors [m, d] -> X [n_rows, 5, d] and mask [n_rows, 5] by (row_idx, slot)."""
    X = np.zeros((n_rows, WIDTH, x.shape[1]), dtype=np.float32)
    mask = np.zeros((n_rows, WIDTH), dtype=bool)
    X[mk_idx["row_idx"].to_numpy(), mk_idx["slot"].to_numpy()] = x
    mask[mk_idx["row_idx"].to_numpy(), mk_idx["slot"].to_numpy()] = True
    return X, mask


def answer_acc(probs, rows):
    ok = (probs.argmax(1) == rows["gold_slot"].to_numpy()).astype(float)
    dec = pd.factorize(rows["case_id"] + "|" + rows["question_id"])[0]
    acc = np.bincount(dec, weights=ok) / np.bincount(dec)
    first = np.unique(dec, return_index=True)[1]
    qt = rows["qtype"].to_numpy()[first]
    return {t: float(acc[qt == t].mean()) for t in ("score", "choice", "noul")} | {"all": float(acc.mean())}, acc, rows["case_id"].to_numpy()[first], qt


def figure(df: pd.DataFrame, path: Path, laya_acc: dict):
    """Slot / content decodability and rotation-trained answer accuracy by layer (test split), English labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#e4e3df"
    t = df[df.split == "test"]
    layers = list(dict.fromkeys(t.layer))
    x = np.arange(len(layers))
    lab = [l.replace("enc_", "").replace("final", "norm").replace("head_", "h") for l in layers]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    ax = axes[0]
    for task, col, name in (("slot", blue, "slot of the option"), ("content", orange, "identity of the option")):
        d = t[(t.task == task) & (t.qtype == "score")].set_index("layer").acc.reindex(layers)
        ax.plot(x, d, color=col, marker="o", ms=3, lw=2, label=name)
    ax.axhline(0.25, color=ink, lw=1, ls="--"); ax.text(len(x) - 1, 0.27, "chance", ha="right", fontsize=8)
    ax.set_title("What a linear classifier reads from one marker vector (score)", fontsize=9)
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05)
    ax = axes[1]
    for qt, col, name in (("all", blue, "all question types"), ("score", orange, "score only")):
        d = t[(t.task == "answer") & (t.qtype == qt)].set_index("layer").acc.reindex(layers)
        ax.plot(x, d, color=col, marker="o", ms=3, lw=2, label=name)
        ax.axhline(laya_acc[qt], color=col, lw=1, ls=":")
    ax.text(0, laya_acc["all"] + 0.01, "Laya, balanced order", fontsize=8)
    ax.set_title("Answer probe trained on all rotations", fontsize=9); ax.set_ylim(0.2, 0.65)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.5); ax.set_xlabel("layer (backbone 01-22, final norm, decision head h1-h2)", fontsize=8)
        ax.grid(axis="y", color=grid); ax.legend(frameon=False, fontsize=8, loc="lower left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot-dir", default=str(REPO_ROOT / "data" / "activations" / "rotation"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sae_features" / "position_probes"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layers", default="", help="comma list of layer labels; default all")
    args = ap.parse_args()
    rot, out, dev = Path(args.rot_dir), Path(args.out), torch.device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    rows_tr, mk_tr = load_split(rot, "train")
    rows_te, mk_te = load_split(rot, "test")
    labels = json.loads((rot / "test" / "meta.json").read_text())["layers"]
    labels = [l for l in labels if l != "emb" and (not args.layers or l in args.layers.split(","))]
    g_tr, g_te = soft_gold(rows_tr, "train"), soft_gold(rows_te, "test")
    disc_rows = ~rows_tr["is_dev"].to_numpy()
    res, test_dec = [], {}
    for label in labels:
        x_tr = np.load(next((rot / "train").glob(f"L*_{label}.npy"))).astype(np.float32)
        x_te = np.load(next((rot / "test").glob(f"L*_{label}.npy"))).astype(np.float32)
        d_m = ~mk_tr["is_dev"].to_numpy()
        for task, col in (("slot", "slot"), ("content", "opt_idx")):
            tr = (x_tr[d_m], mk_tr[d_m].reset_index(drop=True))
            for split, te in (("dev", (x_tr[~d_m], mk_tr[~d_m].reset_index(drop=True))), ("test", (x_te, mk_te))):
                for t, (a, c) in decode(tr, te, col, dev).items():
                    res.append({"layer": label, "task": task, "split": split, "qtype": t, "acc": a, "chance": c})
        Xtr, mtr = dense(x_tr, mk_tr, len(rows_tr))
        Xte, mte = dense(x_te, mk_te, len(rows_te))
        std = P.standardizer(Xtr[disc_rows], mtr[disc_rows])
        params = P.fit_probe(torch.from_numpy(std(Xtr[disc_rows], mtr[disc_rows])).to(dev), torch.from_numpy(mtr[disc_rows]).to(dev),
                             torch.from_numpy(g_tr[disc_rows]).to(dev), WD_ANSWER, 0)
        for split, X, m, r in (("dev", Xtr[~disc_rows], mtr[~disc_rows], rows_tr[~disc_rows]), ("test", Xte, mte, rows_te)):
            if len(r) == 0:
                continue
            probs = P.predict_probe(params, torch.from_numpy(std(X, m)).to(dev), torch.from_numpy(m).to(dev))
            accs, acc_dec, cases_dec, qt_dec = answer_acc(probs, r)
            for t, a in accs.items():
                res.append({"layer": label, "task": "answer", "split": split, "qtype": t, "acc": a, "chance": float("nan")})
            if split == "test":
                test_dec[label] = acc_dec
        print(f"[probes] {label} done", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(out / "probes_by_layer.csv", index=False)
    figure(df, out / "probes_by_layer.png", {"all": 0.334, "score": 0.276})  # Laya mean over rotations, test (REPORT.md sec. 8)
    best = {}
    for t in ("score", "all"):
        d = df[(df.task == "answer") & (df.split == "dev") & (df.qtype == t)]
        best[t] = d.loc[d.acc.idxmax(), "layer"]
    ci = {}
    for t in best:
        sel = (qt_dec == t) if t != "all" else np.ones(len(qt_dec), bool)
        m, c = case_boot(cases_dec[sel], test_dec[best[t]][sel], np.ones(sel.sum()))
        ci[t] = {"acc": m, "ci95": c}
    (out / "summary.json").write_text(json.dumps({"best_answer_layer_on_dev": best, "test_ci_at_best": ci,
        "wd": {"slot_content": WD_CLASS, "answer": WD_ANSWER}, "test_at_best": {t: float(df[(df.task == "answer") & (df.split == "test") & (df.qtype == t) & (df.layer == best[t])].acc.iloc[0]) for t in best}}, indent=1))
    print("[probes] done; best answer layer on dev:", best, flush=True)


if __name__ == "__main__":
    main()
