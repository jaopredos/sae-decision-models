"""Phase A: quick sanity check (A1) + flip-rate/accuracy-by-position (A2).

Usage: python3 -m run_phaseA [--n 300] [--skip-sanity] [--only-dataset banking20]
Writes one JSONL log per (model, dataset) under logs/, then results/phaseA_table.csv and
figures/acc_by_gold_position.png, figures/flip_rate_by_dataset.png.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.datasets import load_banking20, load_newsgroups, load_typed_decisions
from src.perturbations import generate_orders
from src.runner import canonical_criteria, predict_record, JsonlWriter, read_jsonl
from src.models import LayaModel, JuliaModel
from src import metrics

REPO_ROOT = Path(__file__).resolve().parent
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

DATASET_LOADERS = {
    "banking20": load_banking20,
    "newsgroups": load_newsgroups,
    "typed_decisions": load_typed_decisions,
}

PAPER_RUNOFF_NATIVE = {  # Table 9, Appendix D of arXiv:2610.18233 (see NOTES.md)
    "banking20": {"acc": 60.3, "flip": 29.3},
    "newsgroups": {"acc": 48.0, "flip": 27.7},
    "typed_decisions": {"acc": 31.0, "flip": 26.7},
}


def log_path(model_name: str, dataset: str) -> Path:
    return LOGS_DIR / f"phaseA_{model_name}_{dataset}.jsonl"


def run_sweep(model, examples, path: Path, progress_every: int = 50):
    t0 = time.monotonic()
    with JsonlWriter(path, mode="w") as w:
        for i, ex in enumerate(examples, 1):
            criteria = canonical_criteria(ex)
            ids = list(criteria.keys())
            if ex["qtype"] == "noul":
                rec = predict_record(model, ex, ids, criteria, "orig")
                w.write(rec)
            else:
                for cond, order in generate_orders(ids):
                    rec = predict_record(model, ex, order, criteria, cond)
                    w.write(rec)
            if i % progress_every == 0:
                elapsed = time.monotonic() - t0
                print(f"  [{model.name}] {i}/{len(examples)} examples, {elapsed:.1f}s elapsed",
                      flush=True)
    print(f"  [{model.name}] done: {path} ({time.monotonic()-t0:.1f}s)", flush=True)


def aggregate(records):
    by_example = defaultdict(list)
    for r in records:
        by_example[r["example_id"]].append(r)
    orig = [r for r in records if r["condition"] in ("orig", "rot_0", "perm_0")]
    # for qtype != noul, "orig" condition is named rot_0/perm_0 by generate_orders; noul uses "orig"
    acc_point, acc_lo, acc_hi = metrics.bootstrap_ci(orig, metrics.accuracy)
    non_noul_by_example = {k: v for k, v in by_example.items() if v[0]["qtype"] != "noul"}
    flip_point, flip_lo, flip_hi = metrics.bootstrap_ci_by_example(non_noul_by_example, metrics.flip_rate)
    return {
        "acc": acc_point * 100, "acc_lo": acc_lo * 100, "acc_hi": acc_hi * 100,
        "flip": flip_point * 100, "flip_lo": flip_lo * 100, "flip_hi": flip_hi * 100,
        "n_orig_records": len(orig), "n_examples": len(by_example),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--only-dataset", default=None)
    ap.add_argument("--skip-sanity", action="store_true", help="skip A1 Laya-English run")
    args = ap.parse_args()

    datasets = {args.only_dataset: DATASET_LOADERS[args.only_dataset]} if args.only_dataset \
        else DATASET_LOADERS

    print("Loading datasets...", flush=True)
    examples = {name: loader(n=args.n) for name, loader in datasets.items()}
    for name, exs in examples.items():
        print(f"  {name}: {len(exs)} decisions")

    # ---- A1: sanity (Laya English + multilingual, original + rotations) ----
    if not args.skip_sanity:
        print("\n=== A1 sanity: Laya English ===", flush=True)
        laya_en = LayaModel(subfolder=None)
        for name, exs in examples.items():
            run_sweep(laya_en, exs, log_path("laya_english", name))
        del laya_en

    print("\n=== A1/A2: Laya multilingual ===", flush=True)
    laya_ml = LayaModel(subfolder="multilingual")
    for name, exs in examples.items():
        run_sweep(laya_ml, exs, log_path("laya_multilingual", name))
    del laya_ml

    print("\n=== A2: Julia-1 ===", flush=True)
    with JuliaModel() as julia:
        for name, exs in examples.items():
            run_sweep(julia, exs, log_path("julia1", name))

    # ---- Aggregate ----
    print("\n=== Aggregating ===", flush=True)
    rows = []
    models = (["laya_english"] if not args.skip_sanity else []) + ["laya_multilingual", "julia1"]
    for model_name in models:
        for name in datasets:
            p = log_path(model_name, name)
            if not p.exists():
                continue
            records = read_jsonl(p)
            agg = aggregate(records)
            paper = PAPER_RUNOFF_NATIVE.get(name, {})
            rows.append({"model": model_name, "dataset": name, **agg,
                        "paper_laya_native_acc": paper.get("acc"),
                        "paper_laya_native_flip": paper.get("flip")})

    RESULTS_DIR.mkdir(exist_ok=True)
    import csv
    csv_path = RESULTS_DIR / "phaseA_table.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    make_figures(models, datasets)


def make_figures(models, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES_DIR.mkdir(exist_ok=True)

    # acc_by_gold_position.png: one subplot per dataset, bars per model, x = gold position
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    for ax, name in zip(axes[0], datasets):
        for model_name in models:
            p = log_path(model_name, name)
            if not p.exists():
                continue
            records = [r for r in read_jsonl(p) if r["qtype"] != "noul"]
            by_pos = metrics.accuracy_by_position(records)
            if not by_pos:
                continue
            xs = sorted(by_pos)
            ys = [by_pos[x] * 100 for x in xs]
            ax.plot(xs, ys, marker="o", label=model_name)
        ax.set_title(name)
        ax.set_xlabel("posição da opção correta")
        ax.set_ylabel("acurácia (%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "acc_by_gold_position.png", dpi=150)
    plt.close(fig)

    # flip_rate_by_dataset.png: grouped bar chart, dataset x model
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8 / max(len(models), 1)
    names = list(datasets)
    for mi, model_name in enumerate(models):
        vals = []
        for name in names:
            p = log_path(model_name, name)
            if not p.exists():
                vals.append(0)
                continue
            records = [r for r in read_jsonl(p) if r["qtype"] != "noul"]
            by_example = defaultdict(list)
            for r in records:
                by_example[r["example_id"]].append(r)
            vals.append(metrics.flip_rate(by_example) * 100)
        xs = [i + mi * width for i in range(len(names))]
        ax.bar(xs, vals, width=width, label=model_name)
    ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(names))])
    ax.set_xticklabels(names)
    ax.set_ylabel("flip rate (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "flip_rate_by_dataset.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'acc_by_gold_position.png'} and flip_rate_by_dataset.png")


if __name__ == "__main__":
    main()
