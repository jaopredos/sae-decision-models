"""Figures of the position-sensitivity results (English labels), from the saved summaries and conditions.

Usage: python run_position_figures.py [--base results/sae_features] [--out results/sae_features/position_causal]
  causal_by_layer.png    change of the score flip rate when the position subspace is removed (dev, grey = range of 20 random subspaces; test, only cells that qualified on dev)
  flip_by_model.png      flip rate by question type and first-option preference of score, three models
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BLUE, ORANGE, INK, GRID, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#e4e3df", "#b9b8b2"


def cells(base: Path, run: str):
    """label -> dict(dev, null_lo, null_hi, test, qualifies) in percentage points of the score flip rate."""
    out = {}
    dev = json.loads((base / run / "dev" / "summary.json").read_text())
    cond = pd.read_csv(base / run / "dev" / "conditions.csv", keep_default_na=False)  # the kind "null" would otherwise read as NaN
    test = {(str(c["layer"]), c["mode"]): c for c in json.loads((base / run / "test" / "summary.json").read_text())["cells"]}
    b = dev["baseline"]["flip_score"]
    for c in dev["cells"]:
        key = (str(c["layer"]), c["mode"])
        nl = 100 * (cond[(cond.layer.astype(str) == key[0]) & (cond["mode"] == key[1]) & (cond.kind == "null")].flip_score.to_numpy() - b)
        out[key] = {"dev": c["dflip_score_pp"], "lo": float(nl.min()), "hi": float(nl.max()), "test": test[key]["dflip_score_pp"] if key in test else None, "ok": c["qualifies"]}
    return out


def fig_causal(base: Path, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    single, tail = cells(base, "subspace_single"), cells(base, "subspace_tail_sets")
    items = [(f"{k[0]}", v) for k, v in sorted(single.items(), key=lambda kv: int(kv[0][0]))]
    items.append(("norm", tail[("23", "single")]))
    items += [(f"{k[0]}", tail[(k[0], "set")]) for k in (("14+22", 0), ("17+22", 0), ("14+17+22", 0))]
    fig, ax = plt.subplots(figsize=(10, 3.9), dpi=150)
    x = np.arange(len(items))
    for i, (lab, v) in enumerate(items):
        ax.fill_between([i - 0.3, i + 0.3], v["lo"], v["hi"], color=GREY, alpha=0.8, lw=0)
        ax.scatter(i - 0.1, v["dev"], s=38, color=BLUE, zorder=3, facecolors=BLUE if v["ok"] else "white", edgecolors=BLUE, linewidths=1.4)
        if v["test"] is not None:
            ax.scatter(i + 0.1, v["test"], s=38, marker="D", color=ORANGE, zorder=3)
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(len(single) - 0.5, color=GRID, lw=1)
    ax.axvline(len(single) + 0.5, color=GRID, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([l for l, _ in items], fontsize=7)
    ax.set_xlabel("layer where the position subspace is removed at the option markers (single layers 1-22, final norm, then sets of layers)", fontsize=8)
    ax.set_ylabel("change in score flip rate (p.p.)")
    ax.scatter([], [], s=38, color=BLUE, label="dev, qualifies"); ax.scatter([], [], s=38, facecolors="white", edgecolors=BLUE, label="dev, does not qualify")
    ax.scatter([], [], s=38, marker="D", color=ORANGE, label="test (cells that qualified on dev)"); ax.fill_between([], [], [], color=GREY, label="20 random subspaces (min-max)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(axis="y", color=GRID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "causal_by_layer.png"); plt.close(fig)


def fig_models(base: Path, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads((base / "rotation_compare" / "compare.json").read_text())
    names = [("multilingual", "Laya multilingual"), ("base_en", "Laya base (English)"), ("ft_en", "Laya fine-tuned (English)")]
    cols = [BLUE, ORANGE, "#3a8f5f"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    for j, ((k, lab), c) in enumerate(zip(names, cols)):
        ax.bar(np.arange(3) + (j - 1) * 0.26, [d["flip"][k][t] for t in ("score", "choice", "noul")], 0.26, color=c, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["score", "choice", "noul (yes/no)"]); ax.set_ylabel("flip rate (choice changes with option order)", fontsize=9)
    ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 1)
    ax = axes[1]
    for j, ((k, lab), c) in enumerate(zip(names, cols)):
        s = d["behaviour"][k]["preference"]["score_k4"]["slots"]["0"]
        ax.bar(j, s["dev_from_uniform_pp"], 0.6, color=c)
    ax.axhline(0, color=INK, lw=1); ax.set_xticks(range(3)); ax.set_xticklabels(["multi-\nlingual", "base\n(EN)", "fine-tuned\n(EN)"], fontsize=8)
    ax.set_ylabel("first listed option, score:\nshare of picks minus 25% (p.p.)", fontsize=8)
    for a in axes:
        a.grid(axis="y", color=GRID)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "flip_by_model.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/sae_features")
    ap.add_argument("--out", default="results/sae_features/position_causal")
    a = ap.parse_args()
    fig_causal(Path(a.base) / "position_causal", Path(a.out)); fig_models(Path(a.base), Path(a.out))
    print("figures written")
