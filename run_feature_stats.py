"""Stage 5: which SAE features go with Laya's errors on typed-decisions (correlational).

Usage: python run_feature_stats.py --sae checkpoints/sae/ml_enc17_k32/sae.pt --layer enc_17 --name ml_enc17_k32
           [--tokens-dir data/activations/tokens/multilingual] [--stage3-dir data/activations/typed_decisions]
           [--oof results/sae_features/probe_oof_enc_17.parquet] [--attr-dir DIR] [--out results/sae_features]
           [--min-support 20] [--device cuda]

Discovery set = train minus dev (the rankings come from here), dev = a check, test = replication only
(results/sae_features/PROTOCOLO.md). For every decision the SAE latents are read in three views: at the
marker of the option Laya predicted (`pred`), at the marker of the correct option (`corr`), and the maximum
over the state tokens (`state`). Massive tokens (emenda 1) are outside the SAE and contribute zeros.

Writes, under <out>/<name>/:
  feature_stats_<view>_error.parquet                  outcome = Laya is wrong, all decisions
  feature_stats_<view>_probe_right_laya_wrong.parquet pool = decisions where the (out-of-fold) probe is right;
                                                      outcome = Laya is wrong ("the information was there")
  feature_stats_<view>_confusion.parquet              most over-represented (question, predicted -> correct)
                                                      class per feature, on the discovery set
  feature_stats_prior_bias.parquet, prior_bias_summary.json
  feature_stats_attribution.parquet                   if --attr-dir has the gradient-attribution arrays
  decision_views.parquet                              per-decision metadata used (pred/gold/err/probe/subset)
Each stats file has one line per (feature, split, stratum); strata are all, qtype=*, workflow=*.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import features as F
from src.activations import ROLE_MARKER, ROLE_STATE
from src.sae import load_sae
from src.sae_data import load_tokens

REPO_ROOT = Path(__file__).resolve().parent
VIEWS = ("pred", "corr", "state")


def build_views(sae, tokens_root: Path, stage3_root: Path, label: str, device, chunk=100_000):
    """Decision table and the three [N_dec, F] activation views of one split (fp16 on `device`)."""
    acts, ti, rows = load_tokens(tokens_root, label)
    dec3 = pd.read_parquet(stage3_root / "decisions.parquet")
    key = ["case_id", "question_id"]
    meta = rows[key + ["decision_idx", "n_options", "is_dev"]].merge(
        dec3[key + ["workflow", "qtype", "option_ids", "gold_argmax", "gold_dist", "model_probs", "model_pred"]], on=key, how="left")
    assert len(meta) == len(rows) and meta["workflow"].notna().all(), "stage-3 decisions do not match the token rows"
    meta = meta.sort_values("decision_idx").reset_index(drop=True)
    n_dec, n_lat = len(meta), sae.n_latents
    role = ti["role"].to_numpy(); dec_of = ti["decision_idx"].to_numpy(); opt_of = ti["option_index"].to_numpy()

    state_max = torch.zeros(n_dec, n_lat, device=device)
    zm_rows, n_mass_marker = [], 0
    with torch.no_grad():
        for a in range(0, len(ti), chunk):
            x = torch.from_numpy(np.asarray(acts[a:a + chunk])).to(device).float()
            r, d = role[a:a + chunk], dec_of[a:a + chunk]
            massive = sae.is_massive(x)
            sel = np.flatnonzero(np.isin(r, [ROLE_MARKER, ROLE_STATE]))
            if len(sel) == 0:
                continue
            xs = x[torch.from_numpy(sel).to(device)]
            z = sae.encode(xs).masked_fill(massive[torch.from_numpy(sel).to(device)][:, None], 0.0)
            is_state = torch.from_numpy(r[sel] == ROLE_STATE).to(device)
            if is_state.any():
                idx = torch.from_numpy(d[sel]).to(device)[is_state]
                state_max.index_reduce_(0, idx, z[is_state], "amax", include_self=True)
            is_marker = ~is_state
            zm_rows.append(z[is_marker].half())
            n_mass_marker += int(massive[torch.from_numpy(sel).to(device)][is_marker].sum())
    Zm = torch.cat(zm_rows)
    mk = np.flatnonzero(role == ROLE_MARKER)
    marker_row = np.full((n_dec, int(meta["n_options"].max())), -1, dtype=np.int64)
    marker_row[dec_of[mk], opt_of[mk]] = np.arange(len(mk))
    assert len(mk) == len(Zm)

    opt_lists = [list(o) for o in meta["option_ids"]]
    meta["pred_idx"] = [o.index(p) for o, p in zip(opt_lists, meta["model_pred"])]
    meta["gold_idx"] = [o.index(g) for o, g in zip(opt_lists, meta["gold_argmax"])]
    meta["err"] = meta["pred_idx"] != meta["gold_idx"]
    pick = lambda col: Zm[torch.from_numpy(marker_row[np.arange(n_dec), meta[col].to_numpy()]).to(device)]
    views = {"pred": pick("pred_idx"), "corr": pick("gold_idx"), "state": state_max.half()}
    return meta, views, Zm, marker_row, {"markers_massive": n_mass_marker, "markers_total": len(mk)}


def strata_of(meta: pd.DataFrame):
    yield "all", np.ones(len(meta), bool)
    for q in sorted(meta["qtype"].unique()):
        yield f"qtype={q}", (meta["qtype"] == q).to_numpy()
    for w in sorted(meta["workflow"].unique()):
        yield f"workflow={w}", (meta["workflow"] == w).to_numpy()


def stats_for(views, meta, pool: np.ndarray, subset_masks, min_support, device):
    """Concatenated enrichment tables over views x subsets x strata, outcome = Laya wrong, restricted to `pool`."""
    out = {v: [] for v in VIEWS}
    y_all = torch.from_numpy(meta["err"].to_numpy()).to(device)
    for sub_name, sub in subset_masks.items():
        for s_name, s in strata_of(meta):
            m = pool & sub & s
            if m.sum() < 2 * min_support:
                continue
            idx = torch.from_numpy(np.flatnonzero(m)).to(device)
            for v in VIEWS:
                t = F.enrichment_table(views[v][idx], y_all[idx], min_support)
                t.insert(0, "stratum", s_name); t.insert(0, "split", sub_name)
                out[v].append(t)
    return {v: pd.concat(t, ignore_index=True) for v, t in out.items() if t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens-dir", default=str(REPO_ROOT / "data" / "activations" / "tokens" / "multilingual"))
    ap.add_argument("--stage3-dir", default=str(REPO_ROOT / "data" / "activations" / "typed_decisions"))
    ap.add_argument("--oof", default=None)
    ap.add_argument("--attr-dir", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sae_features"))
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    sae = load_sae(args.sae, args.device).eval()

    parts, views_by, zm_by, mrow_by, info = [], {}, {}, {}, {}
    for split in ("train", "test"):
        meta, views, Zm, marker_row, inf = build_views(sae, Path(args.tokens_dir) / split, Path(args.stage3_dir) / split,
                                                       args.layer, device)
        meta["split_src"] = split
        meta["subset"] = np.where(split == "test", "test", np.where(meta["is_dev"], "dev", "discovery"))
        parts.append(meta); views_by[split], zm_by[split], mrow_by[split], info[split] = views, Zm, marker_row, inf
        print(f"[stats] {split}: {len(meta)} decisions, markers massive {inf['markers_massive']}/{inf['markers_total']}", flush=True)
    meta = pd.concat(parts, ignore_index=True)
    views = {v: torch.cat([views_by["train"][v], views_by["test"][v]]) for v in VIEWS}

    if args.oof and Path(args.oof).exists():
        oof = pd.read_parquet(args.oof)[["split", "case_id", "question_id", "probe_pred"]]
        oof["split_src"] = oof.pop("split")
        meta = meta.merge(oof, on=["split_src", "case_id", "question_id"], how="left")
        meta["probe_ok"] = (meta["probe_pred"] == meta["gold_argmax"]).where(meta["probe_pred"].notna(), other=np.nan)
    else:
        meta["probe_ok"] = np.nan
    meta.drop(columns=["option_ids", "gold_dist", "model_probs"]).to_parquet(out / "decision_views.parquet")

    subsets = {s: (meta["subset"] == s).to_numpy() for s in ("discovery", "dev", "test")}
    everyone = np.ones(len(meta), bool)
    res = stats_for(views, meta, everyone, subsets, args.min_support, device)
    for v, t in res.items():
        t.to_parquet(out / f"feature_stats_{v}_error.parquet")
    pool = (meta["probe_ok"] == True).to_numpy()  # noqa: E712
    if pool.sum() > 4 * args.min_support:
        res2 = stats_for(views, meta, pool, subsets, args.min_support, device)
        for v, t in res2.items():
            t.to_parquet(out / f"feature_stats_{v}_probe_right_laya_wrong.parquet")
    print("[stats] error / probe-right tables written", flush=True)

    # most over-represented confusion per feature (discovery set, all strata pooled)
    disc = np.flatnonzero(subsets["discovery"] & meta["err"].to_numpy())
    if len(disc):
        keys = (meta["question_id"] + "|" + meta["pred_idx"].astype(str) + ">" + meta["gold_idx"].astype(str))
        classes = {k: i for i, k in enumerate(sorted(keys[meta["err"]].unique()))}
        cls = np.where(meta["err"], keys.map(classes).fillna(-1).astype(int), -1)
        dmask = subsets["discovery"]
        for v in VIEWS:
            t = F.confusion_enrichment(views[v][torch.from_numpy(np.flatnonzero(dmask)).to(device)],
                                       torch.from_numpy(cls[dmask]).to(device), len(classes), args.min_support)
            inv = {i: k for k, i in classes.items()}
            t["top_confusion"] = t["top_confusion_class"].map(inv)
            t.to_parquet(out / f"feature_stats_{v}_confusion.parquet")

    # prior bias: which option does Laya over-pick, and which features sit at the over-picked option's marker
    bias_rows, bias_tables = [], []
    disc_meta = meta[subsets["discovery"]]
    majority = disc_meta.groupby(["workflow", "question_id"])["gold_idx"].agg(lambda s: int(s.value_counts().index[0])).to_dict()
    hyps = {"score_highest_level": lambda r: r.n_options - 1 if r.qtype == "score" else None,
            "choice_majority_label": lambda r: majority.get((r.workflow, r.question_id)) if r.qtype == "choice" else None,
            "noul_true": lambda r: 1 if r.qtype == "noul" else None,
            "noul_false": lambda r: 0 if r.qtype == "noul" else None}
    for hname, fn in hyps.items():
        opt = np.array([-1 if (o := fn(r)) is None else o for r in meta.itertuples()])
        has = opt >= 0
        for sub, sm in subsets.items():
            m = has & sm
            if m.sum() < 10:
                continue
            picked = (meta["pred_idx"].to_numpy() == opt)[m]; gold_is = (meta["gold_idx"].to_numpy() == opt)[m]
            lo, hi = F.wilson(picked.sum(), m.sum())
            bias_rows.append({"hypothesis": hname, "subset": sub, "n": int(m.sum()), "p_laya_picks": float(picked.mean()),
                              "p_gold_is": float(gold_is.mean()), "diff": float(picked.mean() - gold_is.mean()),
                              "laya_picks_wilson_lo": float(lo), "laya_picks_wilson_hi": float(hi)})
        # features at that option's marker: Laya picks it although gold says it is not the answer (g = 0)
        for sub in ("discovery", "dev", "test"):
            m = np.flatnonzero(has & subsets[sub] & (meta["gold_idx"].to_numpy() != opt))
            if len(m) < 4 * args.min_support:
                continue
            Zrows, ys = [], []
            for i in m:
                src = meta.at[i, "split_src"]; off = 0 if src == "train" else (meta["split_src"] == "train").sum()
                Zrows.append(zm_by[src][int(mrow_by[src][i - off, opt[i]])] if mrow_by[src][i - off, opt[i]] >= 0 else None)
                ys.append(meta.at[i, "pred_idx"] == opt[i])
            ok = [k for k, z in enumerate(Zrows) if z is not None]
            if len(ok) < 4 * args.min_support:
                continue
            t = F.enrichment_table(torch.stack([Zrows[k] for k in ok]), torch.tensor([ys[k] for k in ok], device=device), args.min_support)
            t.insert(0, "hypothesis", hname); t.insert(0, "subset", sub)
            bias_tables.append(t)
    if bias_tables:
        pd.concat(bias_tables, ignore_index=True).to_parquet(out / "feature_stats_prior_bias.parquet")
    (out / "prior_bias_summary.json").write_text(json.dumps(bias_rows, indent=2))

    # gradient attribution (needs the arrays written by run_feature_attribution.py)
    if args.attr_dir and (Path(args.attr_dir) / "attribution_train.npz").exists():
        tabs = []
        for split in ("train", "test"):
            z = np.load(Path(args.attr_dir) / f"attribution_{split}.npz", allow_pickle=True)
            keyed = pd.DataFrame({"split_src": split, "case_id": z["case_id"], "question_id": z["question_id"]})
            keyed = keyed.merge(meta[["split_src", "case_id", "question_id", "subset", "qtype", "workflow"]], on=["split_src", "case_id", "question_id"], how="left")
            for grp in ("pred", "corr", "state", "other"):
                A = torch.from_numpy(z[f"a_{grp}"].astype(np.float32)).to(device)
                for sub in ("discovery", "dev", "test"):
                    for s_name, s in strata_of(keyed):
                        m = ((keyed["subset"] == sub).to_numpy() & s)
                        if m.sum() < args.min_support:
                            continue
                        a = A[torch.from_numpy(np.flatnonzero(m)).to(device)]
                        tabs.append(pd.DataFrame({"feature": np.arange(a.shape[1]), "group": grp, "split": sub, "stratum": s_name,
                                                  "n_errors": int(m.sum()), "mean_attr": a.mean(0).cpu().numpy(),
                                                  "sd_attr": a.std(0).cpu().numpy(), "frac_negative": (a < 0).float().mean(0).cpu().numpy(),
                                                  "frac_positive": (a > 0).float().mean(0).cpu().numpy(), "sum_attr": a.sum(0).cpu().numpy()}))
        if tabs:
            pd.concat(tabs, ignore_index=True).to_parquet(out / "feature_stats_attribution.parquet")
            print("[stats] attribution table written", flush=True)
    (out / "stats_meta.json").write_text(json.dumps({"sae": args.sae, "layer": args.layer, "min_support": args.min_support,
                                                     "decisions": {s: int((meta["subset"] == s).sum()) for s in ("discovery", "dev", "test")},
                                                     "marker_massive": info, "errors": {s: int(meta[meta["subset"] == s]["err"].sum()) for s in ("discovery", "dev", "test")}}, indent=2))
    print("[stats] done ->", out, flush=True)


if __name__ == "__main__":
    main()
