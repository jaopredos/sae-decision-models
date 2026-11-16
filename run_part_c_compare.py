"""Part C: base x fine-tuned (English pair) with one SAE trained on the base model.

Usage: python run_part_c_compare.py --sae-base ckpt.pt --layer enc_NN --out results/sae_features/part_c
           --tokens-base DIR --tokens-ft DIR --markers-base-common DIR --markers-ft DIR
           [--markers-base-default DIR] [--sae-ft ckpt.pt] [--attr-base DIR --attr-ft DIR] [--device cuda]

Inputs (all produced by run_extract.py / run_extract_tokens.py with the SAME common input configuration,
max_len=1024 and head_max_len=256, except --markers-base-default which uses the base's own 512/192):
  --tokens-*    all-token activations of the SAE layer (run_extract_tokens.py, rows.parquet/token_index.parquet)
  --markers-*   stage-3 style dirs (decisions.parquet with each model's answers) for gold and predictions

Steps and outputs (results/sae_features/PROTOCOLO.md, part C):
  1. identical inputs: token ids of base and fine-tuned compared row by row (both splits);
  2. accuracy of base (own defaults and common config) and fine-tuned, by split;
  3. FVE of the base SAE on base vs fine-tuned activations (all non-massive tokens and option markers, dev);
     if FVE_ft < FVE_base - 0.05 and --sae-ft is given, features are paired by decoder cosine (mutual nearest
     neighbours with cos >= 0.7; see the amendment in PROTOCOLO.md) and the fine-tuned side uses SAE-ft;
  4. per feature, base vs fine-tuned: activation at the correct-option marker minus the mean over the wrong
     options ("contrast"), its change with a paired bootstrap over cases (1,000 resamples), error enrichment
     of the feature at the predicted-option marker, and gradient attribution means when available.
Rankings come from the discovery set (train minus dev); dev and test are replication columns. Accuracy and
"association with being right" of the fine-tuned model are only reported on test: it was fine-tuned on the
benchmark's workflows (NOT VERIFIED on which split).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import features as F
from src.sae import load_sae
from src.sae_data import load_tokens, massive_threshold, token_is_dev
from run_feature_stats import VIEWS, build_views
from run_sae_quality import group_stats

REPO_ROOT = Path(__file__).resolve().parent
N_BOOT = 1000
FVE_DROP = 0.05
PAIR_COS = 0.7


def paired_inputs(tokens_a: Path, tokens_b: Path) -> dict:
    out = {}
    for split in ("train", "test"):
        ia = pd.read_parquet(tokens_a / split / "token_index.parquet")
        ib = pd.read_parquet(tokens_b / split / "token_index.parquet")
        same = len(ia) == len(ib) and bool((ia["token_id"].to_numpy() == ib["token_id"].to_numpy()).all()) \
            and bool((ia["role"].to_numpy() == ib["role"].to_numpy()).all())
        out[split] = {"tokens_a": len(ia), "tokens_b": len(ib), "identical_token_ids_and_roles": same}
    return out


def accuracy_table(dirs: dict) -> pd.DataFrame:
    rows = []
    for name, d in dirs.items():
        if not d:
            continue
        for split in ("train", "test"):
            f = Path(d) / split / "decisions.parquet"
            if f.exists():
                t = pd.read_parquet(f)
                dev = t["is_dev"].to_numpy() if "is_dev" in t else np.zeros(len(t), bool)
                ok = (t["model_pred"] == t["gold_argmax"]).to_numpy()
                rows.append({"model_config": name, "split": split, "subset": "all", "n": len(t), "accuracy": float(ok.mean())})
                if split == "train":
                    rows.append({"model_config": name, "split": split, "subset": "dev", "n": int(dev.sum()), "accuracy": float(ok[dev].mean())})
    return pd.DataFrame(rows)


@torch.no_grad()
def sae_fve(sae, tokens_dir: Path, layer: str, device) -> dict:
    acts, ti, rows = load_tokens(tokens_dir / "train", layer)
    thr, _ = massive_threshold(acts, ti, rows, 5.0)
    idx = np.flatnonzero(token_is_dev(ti, rows))
    x = torch.from_numpy(np.concatenate([np.asarray(acts[idx[a:a + 200_000]]) for a in range(0, len(idx), 200_000)])).to(device)
    ok = x.float().norm(dim=1) <= thr
    role = torch.from_numpy(ti["role"].to_numpy()[idx]).to(device)
    out = {}
    for g, m in (("all_non_massive", ok), ("markers", ok & (role == 2))):
        if int(m.sum()) >= 10:
            out[g] = group_stats(sae, x[m])[0]
    return out


def pair_by_decoder(sae_a, sae_b, cos_min: float, chunk: int = 2048):
    """map[i] = j: feature i of A to feature j of B, mutual nearest neighbours by decoder cosine >= cos_min, else -1."""
    da = sae_a.W_dec.detach().float(); db = sae_b.W_dec.detach().float()
    da = da / da.norm(dim=1, keepdim=True); db = db / db.norm(dim=1, keepdim=True)
    best_ab = torch.empty(len(da), dtype=torch.long); cos_ab = torch.empty(len(da))
    for a in range(0, len(da), chunk):
        c = da[a:a + chunk] @ db.t(); v, i = c.max(1); best_ab[a:a + chunk] = i.cpu(); cos_ab[a:a + chunk] = v.cpu()
    best_ba = torch.empty(len(db), dtype=torch.long)
    for a in range(0, len(db), chunk):
        best_ba[a:a + chunk] = (db[a:a + chunk] @ da.t()).argmax(1).cpu()
    mutual = best_ba[best_ab] == torch.arange(len(da))
    ok = mutual & (cos_ab >= cos_min)
    return torch.where(ok, best_ab, torch.full_like(best_ab, -1)).numpy(), cos_ab.numpy()


def contrast(Zm: torch.Tensor, marker_row: np.ndarray, gold_idx: np.ndarray, n_options: np.ndarray, device):
    """[N, F]: activation at the correct-option marker minus the mean over the wrong options' markers."""
    n = len(gold_idx)
    rows = torch.from_numpy(marker_row).to(device)
    corr = Zm[rows[np.arange(n), gold_idx]].float()
    tot = torch.zeros_like(corr); cnt = torch.zeros(n, 1, device=device)
    for j in range(marker_row.shape[1]):
        valid = torch.from_numpy((marker_row[:, j] >= 0) & (gold_idx != j)).to(device)
        if valid.any():
            z = Zm[rows[valid.nonzero().squeeze(1), j]].float()
            tot[valid] += z; cnt[valid] += 1
    return corr - tot / cnt.clamp_min(1), corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-base", required=True); ap.add_argument("--sae-ft", default=None)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--tokens-base", required=True); ap.add_argument("--tokens-ft", required=True)
    ap.add_argument("--markers-base-common", required=True); ap.add_argument("--markers-ft", required=True)
    ap.add_argument("--markers-base-default", default=None)
    ap.add_argument("--attr-base", default=None); ap.add_argument("--attr-ft", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {}

    report["identical_inputs"] = paired_inputs(Path(args.tokens_base), Path(args.tokens_ft))
    print("[C] identical inputs:", json.dumps(report["identical_inputs"]), flush=True)
    acc = accuracy_table({"base_default_512/192": args.markers_base_default, "base_common_1024/256": args.markers_base_common,
                          "finetuned_1024/256": args.markers_ft})
    acc.to_csv(out / "accuracy_by_config.csv", index=False)
    print(acc.to_string(index=False), flush=True)

    sae_b = load_sae(args.sae_base, args.device).eval()
    fve_base = sae_fve(sae_b, Path(args.tokens_base), args.layer, device)
    fve_ft = sae_fve(sae_b, Path(args.tokens_ft), args.layer, device)
    report["fve_base_sae"] = {"on_base": fve_base, "on_finetuned": fve_ft}
    drop = fve_base["all_non_massive"] - fve_ft["all_non_massive"]
    sae_f, perm = sae_b, np.arange(sae_b.n_latents)
    report["fve_drop"] = drop
    if drop > FVE_DROP and args.sae_ft:
        sae_f = load_sae(args.sae_ft, args.device).eval()
        perm, cos = pair_by_decoder(sae_b, sae_f, PAIR_COS)
        report["fve_ft_with_sae_ft"] = sae_fve(sae_f, Path(args.tokens_ft), args.layer, device)
        report["paired_features"] = {"paired": int((perm >= 0).sum()), "total": len(perm), "cos_min": PAIR_COS,
                                     "median_cos_of_nearest": float(np.median(cos))}
    elif drop > FVE_DROP:
        report["note"] = "FVE drop > 0.05 but no --sae-ft given: the base SAE is used on the fine-tuned model unchanged"
    print("[C] FVE:", json.dumps(report["fve_base_sae"]), "drop", round(drop, 4), "| paired:", report.get("paired_features"), flush=True)

    # ---- per-feature contrast, base vs fine-tuned, paired by decision --------------------------------
    parts = {}
    for tag, sae, tdir, mdir in (("base", sae_b, args.tokens_base, args.markers_base_common), ("ft", sae_f, args.tokens_ft, args.markers_ft)):
        per_split = {}
        for split in ("train", "test"):
            meta, views, Zm, mrow, _ = build_views(sae, Path(tdir) / split, Path(mdir) / split, args.layer, device)
            c, corr = contrast(Zm, mrow, meta["gold_idx"].to_numpy(), meta["n_options"].to_numpy(), device)
            per_split[split] = {"meta": meta, "contrast": c, "pred": views["pred"], "err": meta["err"].to_numpy()}
        parts[tag] = per_split
    keep = torch.from_numpy(perm >= 0).to(device); pj = torch.from_numpy(np.where(perm >= 0, perm, 0)).to(device)
    rows_out = []
    rng = np.random.default_rng(0)
    for split, subs in (("train", ("discovery", "dev")), ("test", ("test",))):
        mb, mf = parts["base"][split]["meta"], parts["ft"][split]["meta"]
        assert (mb["case_id"] == mf["case_id"]).all() and (mb["question_id"] == mf["question_id"]).all()
        cb = parts["base"][split]["contrast"]; cf = parts["ft"][split]["contrast"][:, pj] * keep
        for sub in subs:
            sel = (mb["subset"] == sub).to_numpy() if "subset" in mb else None
            if sel is None:
                from src.datasets import typed_dev_ids
                dev = mb["is_dev"].to_numpy() if split == "train" else np.zeros(len(mb), bool)
                sel = (dev if sub == "dev" else ~dev) if split == "train" else np.ones(len(mb), bool)
            idx = torch.from_numpy(np.flatnonzero(sel)).to(device)
            d = (cf[idx] - cb[idx])
            case_codes, inv = np.unique(mb["case_id"].to_numpy()[sel], return_inverse=True)
            S = torch.zeros(len(case_codes), d.shape[1], device=device).index_add_(0, torch.from_numpy(inv).to(device), d)
            cnt = torch.from_numpy(np.bincount(inv)).float().to(device)
            W = torch.from_numpy(rng.multinomial(len(case_codes), np.ones(len(case_codes)) / len(case_codes), size=N_BOOT)).float().to(device)
            boot = (W @ S) / (W @ cnt)[:, None]
            lo, hi = torch.quantile(boot, 0.025, dim=0), torch.quantile(boot, 0.975, dim=0)
            row = pd.DataFrame({"feature": np.arange(d.shape[1]), "split": sub, "n_decisions": int(sel.sum()),
                                "contrast_base": cb[idx].mean(0).cpu().numpy(), "contrast_ft": cf[idx].mean(0).cpu().numpy(),
                                "delta": d.mean(0).cpu().numpy(), "ci_lo": lo.cpu().numpy(), "ci_hi": hi.cpu().numpy(),
                                "paired": (perm >= 0)})
            row["ci_excludes_zero"] = (row["ci_lo"] > 0) | (row["ci_hi"] < 0)
            for tag in ("base", "ft"):
                e = parts[tag][split]["err"][sel]; a = parts[tag][split]["pred"][idx]
                t = F.enrichment_table(a if tag == "base" else a[:, pj] * keep, torch.from_numpy(e).to(device), args.min_support)
                row[f"lift_error_{tag}"] = t["lift"].to_numpy(); row[f"q_error_{tag}"] = t["q_bh"].to_numpy()
                row[f"n_active_{tag}"] = t["n_active"].to_numpy()
            rows_out.append(row)
    res = pd.concat(rows_out, ignore_index=True)
    for tag, adir in (("base", args.attr_base), ("ft", args.attr_ft)):
        f = Path(adir) / "feature_stats_attribution.parquet" if adir else None
        if f and f.exists():
            t = pd.read_parquet(f); t = t[(t.group == "pred") & (t.stratum == "all")][["feature", "split", "mean_attr"]]
            if tag == "ft":
                t = t.copy(); inv = {int(p): i for i, p in enumerate(perm) if p >= 0}; t["feature"] = t["feature"].map(inv)
                t = t.dropna(subset=["feature"]).astype({"feature": int})
            res = res.merge(t.rename(columns={"mean_attr": f"mean_attr_{tag}"}), on=["feature", "split"], how="left")
    res.to_parquet(out / "feature_stats_base_vs_ft.parquet")
    disc = res[(res.split == "discovery") & res.paired & res.ci_excludes_zero]
    top = disc.reindex(disc["delta"].abs().sort_values(ascending=False).index).head(30)
    top.to_csv(out / "top_changed_features_discovery.csv", index=False)
    (out / "top_changed_features.json").write_text(json.dumps([int(f) for f in top["feature"]]))
    report["n_features_ci_excludes_zero_discovery"] = int(len(disc))
    (out / "part_c_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"[C] {len(disc)} features change (95% CI excludes 0, discovery); top-30 written to {out}", flush=True)


if __name__ == "__main__":
    main()
