"""Stage 7 analysis (PROTOCOLO.md, emenda 6, parte A): position vs content under cyclic rotations of the options.

Usage: python run_rotation_analysis.py --rot-dir data/activations/rotation --stage3-train data/activations/typed_decisions/train
           [--out results/sae_features/rotation] [--device cuda] [--sae-dir checkpoints/sae] [--ablation-dir results/sae_features/ablation]

A1  behaviour: positional preference P(slot j) against 1/k (balanced design: every option visits every slot once per decision),
    flip rate, content-/position-consistency, accuracy by rotation and by the slot the correct option sits in.
A2  probe stability: the stage-3 probe (trained on the original order only) applied to the rotated test markers vs Laya itself.
A3  eta^2 of position and of content (workflow x question x option) of the marker vector, per layer (two-way ANOVA, Latin square).
A4  eta^2 of position / content of each SAE feature at the markers, and of the ablation lists against matched random sets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ablation import matched_null
from src import probe as P
from src.sae import load_sae

REPO_ROOT = Path(__file__).resolve().parent
N_BOOT = 1000


def load_rot(d: Path):
    rows, mk = pd.read_parquet(d / "rows.parquet"), pd.read_parquet(d / "markers.parquet")
    rows["grp"] = rows["workflow"] + "|" + rows["question_id"]
    mk["grp"] = rows["grp"].to_numpy()[mk["row_idx"].to_numpy()]
    return rows, mk


def layer_array(d: Path, label: str) -> np.ndarray:
    return np.load(next(d.glob(f"L*_{label}.npy")))


def case_boot(cases, num, den, seed=0):
    """Mean and 95% CI of sum(num)/sum(den) resampling cases."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(cases, return_inverse=True)
    n_, d_ = np.bincount(inv, weights=num), np.bincount(inv, weights=den)
    est = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        est.append((w * n_).sum() / (w * d_).sum())
    return float(n_.sum() / d_.sum()), [float(x) for x in np.percentile(est, [2.5, 97.5])]


# ---- A1 ------------------------------------------------------------------------------------------
def behaviour(rows: pd.DataFrame) -> dict:
    dec_id = pd.factorize(rows["case_id"] + "|" + rows["question_id"])[0]
    n_dec = dec_id.max() + 1
    F = np.zeros((n_dec, 5))
    np.add.at(F, (dec_id, rows["pred_slot"].to_numpy()), 1.0 / rows["k"].to_numpy())
    first = rows.groupby(dec_id).first()
    nopt, nslot = rows.groupby(dec_id)["pred_idx"].nunique(), rows.groupby(dec_id)["pred_slot"].nunique()
    out = {"preference": {}, "consistency": {}, "accuracy": {}}
    for (qt, k), g in first.groupby(["qtype", "k"]):
        ids = g.index.to_numpy()
        res = {}
        for j in range(k):
            m, ci = case_boot(g["case_id"].to_numpy(), F[ids, j], np.ones(len(ids)))
            res[str(j)] = {"p": m, "dev_from_uniform_pp": 100 * (m - 1 / k), "ci_dev_pp": [100 * (c - 1 / k) for c in ci]}
        out["preference"][f"{qt}_k{k}"] = {"n_decisions": len(ids), "slots": res}
        out["consistency"][f"{qt}_k{k}"] = {"choice_changes_with_rotation": float((nopt[ids] > 1).mean()), "same_option_always (content)": float((nopt[ids] == 1).mean()),
                                            "same_slot_always (position)": float((nslot[ids] == 1).mean())}
    rows = rows.assign(ok=(rows.pred_idx == rows.gold_idx).astype(float))
    for qt, g in rows.groupby("qtype"):
        by_rot = g.groupby("rot")["ok"].mean().round(4).to_dict()
        by_gold = g.groupby("gold_slot")["ok"].mean().round(4).to_dict()
        m, ci = case_boot(g["case_id"].to_numpy(), g["ok"].to_numpy(), np.ones(len(g)))
        m0, ci0 = case_boot(g[g.rot == 0]["case_id"].to_numpy(), g[g.rot == 0]["ok"].to_numpy(), np.ones((g.rot == 0).sum()))
        out["accuracy"][qt] = {"mean_over_rotations": m, "ci": ci, "rot0": m0, "rot0_ci": ci0, "by_rotation": by_rot, "by_gold_slot": by_gold}
    return out


# ---- A2 ------------------------------------------------------------------------------------------
def probe_stability(stage3_train: Path, rot_test: Path, layers, device) -> dict:
    train = P.load_split(stage3_train)
    rows, mk = load_rot(rot_test)
    n = len(rows)
    out = {}
    for label, wd in layers:
        stem = next(s for s in train.layers if s.split("_", 1)[1] == label)
        xtr = train.layer(stem)
        std = P.standardizer(xtr, train.mask)
        params = P.fit_probe(torch.from_numpy(std(xtr, train.mask)).to(device), torch.from_numpy(train.mask).to(device),
                             torch.from_numpy(train.gold).to(device), wd, 0)
        arr = layer_array(rot_test, label).astype(np.float32)
        X = np.zeros((n, 5, arr.shape[1]), dtype=np.float32)
        X[mk["row_idx"].to_numpy(), mk["slot"].to_numpy()] = arr
        mask = np.zeros((n, 5), dtype=bool)
        mask[mk["row_idx"].to_numpy(), mk["slot"].to_numpy()] = True
        probs = P.predict_probe(params, torch.from_numpy(std(X, mask)).to(device), torch.from_numpy(mask).to(device))
        pslot = probs.argmax(1)
        order = np.stack([np.pad(o, (0, 5 - len(o)), constant_values=-1) for o in rows["order"]])
        pred = order[np.arange(n), pslot]
        rows = rows.assign(**{f"ok_probe_{label}": (pred == rows.gold_idx.to_numpy()).astype(float)})
    rows["ok_laya"] = (rows.pred_idx == rows.gold_idx).astype(float)
    big = rows[(rows.k >= 4) & (rows.rot <= 3)]
    for label, _ in layers:
        res = {"by_rotation": {"probe": big.groupby("rot")[f"ok_probe_{label}"].mean().round(4).to_dict(), "laya": big.groupby("rot")["ok_laya"].mean().round(4).to_dict()}}
        cases, inv = np.unique(big["case_id"].to_numpy(), return_inverse=True)
        def counts(col):
            return np.stack([np.bincount(inv, weights=(big[col].to_numpy() * (big.rot.to_numpy() == r))) for r in range(4)]), \
                   np.stack([np.bincount(inv, weights=(big.rot.to_numpy() == r).astype(float)) for r in range(4)])
        (cp, np_), (cl, nl) = counts(f"ok_probe_{label}"), counts("ok_laya")
        rng = np.random.default_rng(0)
        diffs, amp_p0, amp_l0 = [], None, None
        for b in range(N_BOOT + 1):
            w = np.ones(len(cases)) if b == 0 else np.bincount(rng.integers(0, len(cases), len(cases)), minlength=len(cases))
            ap = (cp @ w) / (np_ @ w); al = (cl @ w) / (nl @ w)
            amp_p, amp_l = ap.max() - ap.min(), al.max() - al.min()
            if b == 0:
                amp_p0, amp_l0 = amp_p, amp_l
            else:
                diffs.append(amp_l - amp_p)
        res.update({"amplitude_probe": float(amp_p0), "amplitude_laya": float(amp_l0), "amplitude_diff_laya_minus_probe": float(amp_l0 - amp_p0),
                    "ci95_diff": [float(x) for x in np.percentile(diffs, [2.5, 97.5])], "n_decisions_rows": int(len(big)), "wd": dict(layers)[label]})
        out[label] = res
    nl = rows[rows.k == 2]
    out["noul_k2_by_rotation"] = {"laya": nl.groupby("rot")["ok_laya"].mean().round(4).to_dict(),
                                  **{f"probe_{label}": nl.groupby("rot")[f"ok_probe_{label}"].mean().round(4).to_dict() for label, _ in layers}}
    return out


# ---- A3 ------------------------------------------------------------------------------------------
def eta_sums(X, grp, slot, opt):
    """Pooled SS_total, SS_position, SS_content of vectors X [n, d] (any trailing shape), per question group."""
    tot = pos = cont = 0.0
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        x = X[idx].astype(np.float32)
        mu = x.mean(0)
        tot += float(((x - mu) ** 2).sum())
        for f, name in ((slot, "pos"), (opt, "cont")):
            s = 0.0
            for v in np.unique(f[idx]):
                m = f[idx] == v
                s += m.sum() * float(((x[m].mean(0) - mu) ** 2).sum())
            if name == "pos":
                pos += s
            else:
                cont += s
    return tot, pos, cont


def eta_layers(rot_dir: Path, layers) -> pd.DataFrame:
    rows, mk = load_rot(rot_dir)
    qt = rows["qtype"].to_numpy()[mk["row_idx"].to_numpy()]
    out = []
    for label in layers:
        X = layer_array(rot_dir, label)
        for scope, sel in [("all", np.ones(len(mk), bool))] + [(t, qt == t) for t in ("choice", "score", "noul")]:
            tot, pos, cont = eta_sums(X[sel], mk["grp"].to_numpy()[sel], mk["slot"].to_numpy()[sel], mk["opt_idx"].to_numpy()[sel])
            out.append({"layer": label, "qtype": scope, "eta2_position": pos / tot, "eta2_content": cont / tot, "residual": 1 - (pos + cont) / tot})
    return pd.DataFrame(out)


# ---- A4 ------------------------------------------------------------------------------------------
@torch.no_grad()
def feature_eta(rot_dir: Path, label: str, sae, device) -> dict:
    rows, mk = load_rot(rot_dir)
    X = layer_array(rot_dir, label)
    F_ = sae.n_latents
    tot = torch.zeros(F_, device=device, dtype=torch.float64)
    pos, cont = tot.clone(), tot.clone()
    for g in np.unique(mk["grp"].to_numpy()):
        idx = np.flatnonzero(mk["grp"].to_numpy() == g)
        z = sae.encode(torch.from_numpy(X[idx].astype(np.float32)).to(device)).double()
        mu = z.mean(0)
        tot += ((z - mu) ** 2).sum(0)
        for f, acc in ((mk["slot"].to_numpy()[idx], pos), (mk["opt_idx"].to_numpy()[idx], cont)):
            for v in np.unique(f):
                m = torch.from_numpy(f == v).to(device)
                acc += m.sum() * (z[m].mean(0) - mu) ** 2
    ok = tot > 0
    return {"pos": (pos / tot.where(ok, torch.ones_like(tot))).where(ok, torch.full_like(tot, float("nan"))).cpu().numpy(),
            "cont": (cont / tot.where(ok, torch.ones_like(tot))).where(ok, torch.full_like(tot, float("nan"))).cpu().numpy()}


def list_vs_null(eta: dict, sel_v: dict, ks=(20, 100), n_null=100, seed=0) -> dict:
    n_act = {int(f): x for f, x in sel_v["n_active"].items()}
    edges = np.unique(np.quantile(np.array(list(n_act.values())), np.linspace(0, 1, 6)))
    bins_of = {f: int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)) for f, x in n_act.items()}
    pool_bins = {b: [f for f, bb in bins_of.items() if bb == b] for b in range(len(edges) - 1)}
    exclude = set(sel_v["neg"]) | set(sel_v["pos"])
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        res = {}
        for name in ("pos", "cont"):
            obs = float(np.nanmean(eta[name][sel_v["neg"][:k]]))
            null = np.array([np.nanmean(eta[name][matched_null(sel_v, k, rng, pool_bins, bins_of, exclude)]) for _ in range(n_null)])
            res[f"eta2_{'position' if name == 'pos' else 'content'}"] = {"neg_list": obs, "null_mean": float(null.mean()), "null_p5": float(np.percentile(null, 5)),
                                                                        "null_p95": float(np.percentile(null, 95)), "above_p95": bool(obs > np.percentile(null, 95))}
        out[f"K{k}"] = res
    sup = np.array(sorted(n_act))
    out["supported_features"] = {"n": int(len(sup)), "median_eta2_position": float(np.nanmedian(eta["pos"][sup])), "median_eta2_content": float(np.nanmedian(eta["cont"][sup])),
                                 "share_position_gt_content": float(np.nanmean(eta["pos"][sup] > eta["cont"][sup]))}
    return out


def figures(beh: dict, eta: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    a = eta[eta.qtype == "all"].reset_index(drop=True)
    ax.plot(range(len(a)), a.eta2_position, color=blue, marker="o", ms=3, lw=2, label="position of the option")
    ax.plot(range(len(a)), a.eta2_content, color=orange, marker="o", ms=3, lw=2, label="content of the option")
    ax.set_xticks(range(len(a))); ax.set_xticklabels([l.replace("enc_", "").replace("final", "norm").replace("head_", "h") for l in a.layer], fontsize=7)
    ax.set_xlabel("layer (emb, backbone 01-22, final norm, decision head h1-h2)"); ax.set_ylabel("share of marker-vector variance (eta$^2$)")
    ax.grid(axis="y", color=grid); ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "eta_by_layer.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), dpi=150, sharey=True)
    for ax, key in zip(axes, ("choice_k4", "choice_k5")):
        if key not in beh["preference"]:
            continue
        s = beh["preference"][key]["slots"]; k = len(s)
        ax.bar(range(k), [s[str(j)]["p"] for j in range(k)], color=blue)
        ax.axhline(1 / k, color=ink, lw=1, ls="--")
        ax.set_title(f"choice, {k} options", fontsize=9); ax.set_xlabel("slot the model picks (0 = first listed)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("share of picks (dashed = uniform)")
    fig.tight_layout(); fig.savefig(out / "position_preference_choice.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot-dir", default=str(REPO_ROOT / "data" / "activations" / "rotation"))
    ap.add_argument("--stage3-train", default=str(REPO_ROOT / "data" / "activations" / "typed_decisions" / "train"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sae_features" / "rotation"))
    ap.add_argument("--sae-dir", default=str(REPO_ROOT / "checkpoints" / "sae"))
    ap.add_argument("--ablation-dir", default=str(REPO_ROOT / "results" / "sae_features" / "ablation"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-only-smoke", action="store_true", help="use the train rotation dir as the test one too (code check)")
    args = ap.parse_args()
    out, rot = Path(args.out), Path(args.rot_dir)
    out.mkdir(parents=True, exist_ok=True)
    rtrain, rtest = rot / "train", rot / ("train" if args.train_only_smoke else "test")
    layers = json.loads((rtest / "meta.json").read_text())["layers"]

    beh = {s: behaviour(load_rot(d)[0]) for s, d in (("train", rtrain), ("test", rtest))}
    (out / "behaviour.json").write_text(json.dumps(beh, indent=1))
    print("[rot] A1 done", flush=True)
    probe = probe_stability(Path(args.stage3_train), rtest, [("enc_08", 1e-3), ("enc_17", 1e-5)], args.device)
    (out / "probe_stability.json").write_text(json.dumps(probe, indent=1))
    print("[rot] A2 done", {k: (v["amplitude_probe"], v["amplitude_laya"]) for k, v in probe.items() if k.startswith("enc")}, flush=True)
    eta = pd.concat([eta_layers(rtrain, layers).assign(split="train"), eta_layers(rtest, layers).assign(split="test")])
    eta.to_csv(out / "eta_by_layer.csv", index=False)
    print("[rot] A3 done", flush=True)
    feats = {}
    for label, name in (("enc_08", "ml_enc08_k32"), ("enc_17", "ml_enc17_k32")):
        sae = load_sae(str(Path(args.sae_dir) / name / "sae.pt"), args.device).eval()
        sel = json.loads((Path(args.ablation_dir) / name / "selected_features.json").read_text())
        res = {}
        for split, d in (("train", rtrain), ("test", rtest)):
            e = feature_eta(d, label, sae, torch.device(args.device))
            res[split] = list_vs_null(e, sel["pred"])
            if split == "train":
                np.savez_compressed(out / f"feature_eta_{label}_train.npz", pos=e["pos"], cont=e["cont"])
        feats[label] = res
    (out / "feature_eta.json").write_text(json.dumps(feats, indent=1))
    figures(beh["test"], eta[(eta.split == "test")], out)
    print("[rot] A4 done; figures written", flush=True)


if __name__ == "__main__":
    main()
