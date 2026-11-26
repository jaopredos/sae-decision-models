"""Stage 8 (PROTOCOLO.md, emenda 7): causal tests of position sensitivity under cyclic rotations of the options.

Usage:
  python run_position_causal.py --exp subspace --subset dev  [--layers 1,2,...] [--n-null 20] [--device cuda]
  python run_position_causal.py --exp subspace --subset test          (re-runs only the cells that qualified on dev)
  python run_position_causal.py --exp features --sae ml_enc08_k32 --layer enc_08 --subset test [--ks 5,20,100]

Sensitivity = the choice changes when only the order of the options changes. For every decision the model is run on all k
cyclic rotations of the options; `flip` is the fraction of decisions whose chosen option is not the same in all rotations.
  subspace  remove, at the option markers of layer L (alone, or from L onward), the span of the slot-class directions
            (position), of the option-class directions (content, control) or of a random subspace of the same size (null).
  features  zero, at all markers, the SAE features with the largest eta^2 of position (or of content, control), against
            50-frequency-matched random sets.
Directions and feature lists come from the discovery rotations only (data/activations/rotation/train, rows with is_dev False).
Baseline parity: the batched argmax is compared with the official single-case `predict` rows of run_rotation_extract.
"""
import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from run_ablation import matched_null
from run_rotation_analysis import case_boot
from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER, layer_modules, load_agent, option_ids
from src.datasets import load_typed_cases, typed_dev_ids
from src.models import set_all_seeds
from src.position_edit import ProjectOut, between_class_directions, random_subspace
from src.sae import load_sae
from src.sae_intervention import SAEIntervention

REPO_ROOT = Path(__file__).resolve().parent
WIDTH = 5
TYPES = ("score", "choice", "noul")
SLOT_KEYS = (("score", 4), ("choice", 4), ("noul", 2))
DEFAULT_LAYERS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 22)
MIN_SUPPORT = 20


# ---- running Laya on all rotations ----------------------------------------------------------------
class RotationRunner:
    def __init__(self, agent, cases):
        self.agent, self.cur = agent, {}
        agent.model.register_forward_pre_hook(lambda m, a: self.cur.__setitem__("marker_pos", a[2]))
        groups = {}
        for c in cases:
            groups.setdefault(json.dumps(c["questions"], sort_keys=True), []).append(c)
        self.plan, meta = [], []
        for g in groups.values():
            info = {}
            for q, qd in g[0]["questions"].items():
                internal = agent._to_internal(qd)
                info[q] = (internal["t"], option_ids(internal))
            for r in range(WIDTH):
                qs = {q: {**g[0]["questions"][q], "option_order": [(j + r) % len(o) for j in range(len(o))]} for q, (_, o) in info.items() if len(o) > r}
                if not qs:
                    continue
                self.plan.append((g, qs))
                for c in g:
                    for q in qs:
                        t, opts = info[q]
                        meta.append((c["case_id"], q, t, len(opts), r, qs[q]["option_order"], opts.index(str(c["gold"][q]["label"]))))
        df = pd.DataFrame(meta, columns=["case_id", "question_id", "qtype", "k", "rot", "order", "gold_idx"])
        self.df = df
        self.order = np.stack([np.pad(o, (0, WIDTH - len(o)), constant_values=-1) for o in df["order"]])
        self.dec = pd.factorize(df["case_id"] + "|" + df["question_id"])[0]
        self.n_dec = int(self.dec.max()) + 1
        first = np.unique(self.dec, return_index=True)[1]
        self.dec_case, self.dec_type, self.dec_k = (df[c].to_numpy()[first] for c in ("case_id", "qtype", "k"))
        self.valid = np.arange(WIDTH)[None, :] < df["k"].to_numpy()[:, None]
        self.rows_per_dec = np.bincount(self.dec)

    def marker_mask(self, h):
        pos = self.cur["marker_pos"].to(h.device)
        m = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
        b, j = (pos >= 0).nonzero(as_tuple=True)
        m[b, pos[b, j]] = True
        return m

    @torch.no_grad()
    def logits(self, contexts=()) -> np.ndarray:
        outs = []
        hook = self.agent.model.register_forward_hook(lambda m, i, o: outs.append(o[0].float().cpu()))
        try:
            with ExitStack() as st:
                for c in contexts:
                    st.enter_context(c)
                for g, qs in self.plan:
                    self.agent.predict_batch([c["state"] for c in g], qs, batch_size=40)
        finally:
            hook.remove()
        L = torch.cat([F.pad(o, (0, WIDTH - o.shape[1]), value=-1e4) for o in outs]).numpy()
        assert len(L) == len(self.df), (len(L), len(self.df))
        return L

    def summarize(self, L: np.ndarray, base: np.ndarray):
        """Scalars per type plus the per-decision arrays used for bootstrap."""
        slot = L.argmax(1)
        pred = self.order[np.arange(len(L)), slot]
        ok = (pred == self.df["gold_idx"].to_numpy()).astype(float)
        counts = np.zeros((self.n_dec, WIDTH))
        np.add.at(counts, (self.dec, pred), 1)
        flip = ((counts > 0).sum(1) > 1).astype(float)
        acc = np.bincount(self.dec, weights=ok) / self.rows_per_dec
        sp = np.zeros((self.n_dec, WIDTH))
        np.add.at(sp, (self.dec, slot), 1)
        sp /= self.rows_per_dec[:, None]
        z = np.where(self.valid, L, -np.inf)
        zb = np.where(self.valid, base, -np.inf)
        lp, lq = z - np.logaddexp.reduce(z, 1, keepdims=True), zb - np.logaddexp.reduce(zb, 1, keepdims=True)
        kl = np.where(self.valid, np.exp(lq) * (lq - lp), 0).sum(1)
        out = {}
        for t in TYPES:
            d, rows = self.dec_type == t, self.df["qtype"].to_numpy() == t
            out[f"flip_{t}"], out[f"acc_{t}"], out[f"kl_{t}"] = float(flip[d].mean()), float(acc[d].mean()), float(kl[rows].mean())
        for t, k in SLOT_KEYS:
            d = (self.dec_type == t) & (self.dec_k == k)
            p = sp[d][:, :k].mean(0)
            out[f"first_{t}"], out[f"last_{t}"] = float(100 * (p[0] - 1 / k)), float(100 * (p[k - 1] - 1 / k))
        return out, {"flip": flip, "acc": acc}


def pick_cases(subset: str, limit: int):
    cases = load_typed_cases("test" if subset == "test" else "train")
    if subset == "dev":
        dev = typed_dev_ids()
        cases = [c for c in cases if c["case_id"] in dev]
    return cases[:limit] if limit else cases


def parity(runner, base, rot_dir: Path, subset: str):
    rows = pd.read_parquet(rot_dir / ("test" if subset == "test" else "train") / "rows.parquet")[["case_id", "question_id", "rot", "pred_idx"]]
    mine = runner.df.assign(pred=runner.order[np.arange(len(base)), base.argmax(1)])
    j = mine.merge(rows, on=["case_id", "question_id", "rot"])
    return float((j["pred"] == j["pred_idx"]).mean()), len(j)


# ---- directions and feature lists from the discovery rotations ------------------------------------
def discovery_markers(rot_dir: Path):
    rows, mk = pd.read_parquet(rot_dir / "train" / "rows.parquet"), pd.read_parquet(rot_dir / "train" / "markers.parquet")
    keep = ~rows["is_dev"].to_numpy()[mk["row_idx"].to_numpy()]
    grp = (rows["workflow"] + "|" + rows["question_id"]).to_numpy()[mk["row_idx"].to_numpy()]
    return keep, grp, mk["slot"].to_numpy(), mk["opt_idx"].to_numpy()


def layer_vectors(rot_dir: Path, label: str):
    return np.load(next((rot_dir / "train").glob(f"L*_{label}.npy")))


def lab(l: int) -> str:
    """Layer index -> label: 1-22 backbone, 23 final norm, 24-25 decision-head layers."""
    return f"enc_{l:02d}" if l <= 22 else {23: "enc_final", 24: "head_1", 25: "head_2"}[l]


def span_of(layer, mode) -> list:
    """Layers edited by a cell: `single` the layer itself, `onward` L..22, `tail` L..25, `set` an explicit 'a+b+c' list."""
    if mode == "set":
        return [int(x) for x in str(layer).split("+")]
    L = int(layer)
    return list(range(L, 23)) if mode == "onward" else list(range(L, 26)) if mode == "tail" else [L]


def estimate_subspaces(rot_dir: Path, layers, energy: float, out: Path):
    f = out / "subspaces.npz"
    res = {k: v for k, v in np.load(f).items()} if f.exists() else {}
    layers = [L for L in layers if f"pos_{L}" not in res]
    if not layers:
        return res
    keep, grp, slot, opt = discovery_markers(rot_dir)
    for L in layers:
        X = layer_vectors(rot_dir, lab(L))[keep]
        for name, cls in (("pos", slot), ("cont", opt)):
            U, r, expl = between_class_directions(X, grp[keep], cls[keep], energy)
            res[f"{name}_{L}"], res[f"expl_{name}_{L}"] = U, expl
        print(f"[pos] {lab(L)}: position dims {res[f'pos_{L}'].shape[1]} content dims {res[f'cont_{L}'].shape[1]}", flush=True)
    np.savez_compressed(f, **res)
    return res


@torch.no_grad()
def feature_stats(rot_dir: Path, label: str, sae, device):
    """Per feature, on discovery marker vectors: n_active, eta^2 of position and of option (within question group)."""
    keep, grp, slot, opt = discovery_markers(rot_dir)
    X = layer_vectors(rot_dir, label)
    n = sae.n_latents
    dt = torch.float32 if device.type == "mps" else torch.float64  # MPS has no float64
    tot, pos, cont, act = (torch.zeros(n, dtype=dt, device=device) for _ in range(4))
    for g in np.unique(grp[keep]):
        idx = np.flatnonzero(keep & (grp == g))
        z = sae.encode(torch.from_numpy(X[idx].astype(np.float32)).to(device)).to(dt)
        mu = z.mean(0)
        tot += ((z - mu) ** 2).sum(0)
        act += (z > 0).sum(0)
        for cls, acc in ((slot[idx], pos), (opt[idx], cont)):
            for v in np.unique(cls):
                m = torch.from_numpy(cls == v).to(device)
                acc += m.sum() * (z[m].mean(0) - mu) ** 2
    ok = tot > 0
    safe = tot.where(ok, torch.ones_like(tot))
    return {"n_active": act.cpu().numpy(), "eta_pos": (pos / safe).where(ok, torch.zeros_like(tot)).cpu().numpy(),
            "eta_cont": (cont / safe).where(ok, torch.zeros_like(tot)).cpu().numpy()}


# ---- reading the conditions -----------------------------------------------------------------------
def score_delta_ci(runner, store, key, base_dec):
    """Delta of the score flip rate vs baseline with a case-bootstrap 95% CI, and the same for accuracy."""
    d = runner.dec_type == "score"
    res = {}
    for name in ("flip", "acc"):
        delta = (store[key][name] - base_dec[name])[d]
        m, ci = case_boot(runner.dec_case[d], delta, np.ones(d.sum()))
        res[name] = (100 * m, [100 * c for c in ci])
    return res


def judge(df: pd.DataFrame, base: dict, key_cols, store, runner, base_dec):
    """Apply the emenda-7 criterion to each cell (key_cols) with pos/cont/null kinds."""
    out = []
    for cell, g in df.groupby(key_cols):
        cell = cell if isinstance(cell, tuple) else (cell,)
        pos, cont, null = g[g.kind == "pos"], g[g.kind == "cont"], g[g.kind == "null"]
        if pos.empty:
            continue
        dflip = lambda x: 100 * (float(x["flip_score"].iloc[0]) - base["flip_score"])
        ci = score_delta_ci(runner, store, cell + ("pos",), base_dec)
        nl = 100 * (null["flip_score"].to_numpy() - base["flip_score"])
        p_emp = float((1 + (nl <= dflip(pos)).sum()) / (1 + len(nl))) if len(nl) else float("nan")
        d_acc = 100 * (float(pos["acc_score"].iloc[0]) - base["acc_score"])
        d_cont = dflip(cont) if not cont.empty else float("nan")
        row = dict(zip(key_cols if isinstance(key_cols, list) else [key_cols], cell))
        row.update({"dflip_score_pp": dflip(pos), "dflip_ci": ci["flip"][1], "dacc_score_pp": d_acc, "dacc_ci": ci["acc"][1], "null_p5": float(np.percentile(nl, 5)) if len(nl) else None,
                    "null_mean": float(nl.mean()) if len(nl) else None, "p_empirical": p_emp, "dflip_content_pp": d_cont,
                    "dflip_choice_pp": 100 * (float(pos["flip_choice"].iloc[0]) - base["flip_choice"]), "dflip_noul_pp": 100 * (float(pos["flip_noul"].iloc[0]) - base["flip_noul"]),
                    "first_score_shift_pp": float(pos["first_score"].iloc[0]) - base["first_score"], "kl_score": float(pos["kl_score"].iloc[0])})
        row["qualifies"] = bool(row["dflip_score_pp"] <= -3 and row["dflip_ci"][1] < 0 and row["p_empirical"] <= 0.05 and row["dacc_score_pp"] >= -2
                                and (np.isnan(d_cont) or abs(d_cont) < 0.5 * abs(row["dflip_score_pp"])))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["subspace", "features"], required=True)
    ap.add_argument("--subset", choices=["dev", "test", "smoke"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--modes", default="single,onward")
    ap.add_argument("--cells", default="", help="explicit cells 'layer:mode,...' (mode single|onward|tail|set; a set is 'a+b+c:set'); default = layers x modes")
    ap.add_argument("--tag", default="", help="output folder name; default derives from --modes")
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--energy", type=float, default=0.9)
    ap.add_argument("--sae", default=None)
    ap.add_argument("--layer", default=None, help="features: encoder layer label, e.g. enc_08")
    ap.add_argument("--ks", default="5,20,100")
    ap.add_argument("--rot-dir", default=str(REPO_ROOT / "data" / "activations" / "rotation"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sae_features" / "position_causal"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_all_seeds()
    rot_dir = Path(args.rot_dir)
    name = args.tag or (f"subspace_{args.modes.replace(',', '_')}" if args.exp == "subspace" else f"{args.exp}_{args.sae}")
    out = Path(args.out) / name
    (out / args.subset).mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    agent = load_agent(args.device, LAYA_REPO, LAYA_SUBFOLDER, LAYA_REVISION)
    mods = dict(layer_modules(agent.model))
    dev = str(agent.device)
    runner = RotationRunner(agent, pick_cases(args.subset, args.limit))
    t0 = time.time()
    base_L = runner.logits()
    base, base_dec = runner.summarize(base_L, base_L)
    agree, n_agree = parity(runner, base_L, rot_dir, args.subset)
    print(f"[pos] {args.subset}: {runner.n_dec} decisions, {len(runner.df)} rows; baseline flip score {base['flip_score']:.3f} choice {base['flip_choice']:.3f} noul {base['flip_noul']:.3f}; "
          f"argmax agreement with the official rotated predict {agree:.4f} ({n_agree} rows); {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    conds, store = [], {}

    def run(cell, kind, contexts, keep=False):
        L = runner.logits(contexts)
        s, d = runner.summarize(L, base_L)
        conds.append({"experiment": args.exp, "subset": args.subset, **dict(zip(cell_cols, cell)), "kind": kind, **s})
        if keep:
            store[cell + (kind,)] = d

    if args.exp == "subspace":
        cell_cols = ["layer", "mode"]
        if args.cells:  # layers kept as text so a set like '14+22' and plain layers sort together
            cells = [tuple(spec.split(":")) for spec in args.cells.split(",")]
        else:
            cells = [(L, m) for L in layers for m in args.modes.split(",")]
        if args.subset == "test":  # no new selection: only the cells that qualified on dev
            dev_sum = json.loads((out / "dev" / "summary.json").read_text())
            cells = [(c["layer"], c["mode"]) for c in dev_sum["cells"] if c["qualifies"]]
            print(f"[pos] test re-runs the {len(cells)} cells that qualified on dev: {cells}", flush=True)
        sub = estimate_subspaces(rot_dir, sorted({l for L, m in cells for l in span_of(L, m)}), args.energy, out)
        for n, (L, mode) in enumerate(cells, 1):
            span = span_of(L, mode)

            def ctx(Us):
                return [ProjectOut(mods[lab(l)], Us[l], runner.marker_mask) for l in span]
            run((L, mode), "pos", ctx({l: sub[f"pos_{l}"] for l in span}), keep=True)
            run((L, mode), "cont", ctx({l: sub[f"cont_{l}"] for l in span}))
            for i in range(args.n_null):
                run((L, mode), "null", ctx({l: random_subspace(sub[f"pos_{l}"].shape[0], sub[f"pos_{l}"].shape[1], rng) for l in span}))
            print(f"[pos] {n}/{len(cells)} cells, {time.time() - t0:.0f}s", flush=True)
        key_cols = ["layer", "mode"]
    else:
        cell_cols = ["K"]
        sae = load_sae(str(REPO_ROOT / "checkpoints" / "sae" / args.sae / "sae.pt"), dev).eval()
        fs = feature_stats(rot_dir, args.layer, sae, torch.device(dev))
        ok = np.flatnonzero(fs["n_active"] >= MIN_SUPPORT)
        by_pos, by_cont = (ok[np.argsort(-fs[k][ok])] for k in ("eta_pos", "eta_cont"))
        lists = {"pos": [int(f) for f in by_pos[:100]], "cont": [int(f) for f in by_cont[:100]]}
        (out / "selected_features.json").write_text(json.dumps({**lists, "eta_pos": {str(f): float(fs["eta_pos"][f]) for f in lists["pos"]},
                                                                "eta_cont": {str(f): float(fs["eta_cont"][f]) for f in lists["cont"]}, "n_supported": int(len(ok))}, indent=1))
        edges = np.unique(np.quantile(fs["n_active"][ok], np.linspace(0, 1, 6)))
        bins_of = {int(f): int(np.clip(np.searchsorted(edges, fs["n_active"][f], side="right") - 1, 0, len(edges) - 2)) for f in ok}
        pool = {b: [f for f, bb in bins_of.items() if bb == b] for b in range(len(edges) - 1)}
        exclude = set(lists["pos"]) | set(lists["cont"])
        print(f"[pos] {len(ok)} supported features; top position eta^2 {fs['eta_pos'][lists['pos'][0]]:.3f}, top content eta^2 {fs['eta_cont'][lists['cont'][0]]:.3f}", flush=True)

        def ctx(feats):
            return [SAEIntervention(sae, mods[args.layer], mode="edit", edits={f: ("zero",) for f in feats}, position_mask=runner.marker_mask)]
        for K in [int(k) for k in args.ks.split(",")]:
            run((K,), "pos", ctx(lists["pos"][:K]), keep=True)
            run((K,), "cont", ctx(lists["cont"][:K]))
            for i in range(args.n_null):
                run((K,), "null", ctx(matched_null({"neg": lists["pos"]}, K, rng, pool, bins_of, exclude)))
            print(f"[pos] K={K} done, {time.time() - t0:.0f}s", flush=True)
        key_cols = ["K"]

    df = pd.DataFrame(conds)
    if df.empty:  # test re-run with no cell that qualified on dev
        (out / args.subset / "summary.json").write_text(json.dumps({"experiment": args.exp, "subset": args.subset, "baseline": base, "argmax_agreement_official": agree,
                                                                      "cells": [], "note": "nenhuma célula qualificou no dev; nada a replicar no teste"}, indent=1))
        print("[pos] nothing to run: no cell qualified on dev", flush=True)
        return
    df.to_csv(out / args.subset / "conditions.csv", index=False)
    np.savez_compressed(out / args.subset / "dec_arrays.npz", **{"_".join(map(str, k)) + "_" + n: v for k, d in store.items() for n, v in d.items()},
                        dec_case=runner.dec_case, dec_type=runner.dec_type)
    cells_out = judge(df, base, key_cols, store, runner, base_dec)
    summ = {"experiment": args.exp, "subset": args.subset, "n_decisions": runner.n_dec, "baseline": base, "argmax_agreement_official": agree, "n_null": args.n_null,
            "cells": cells_out, "elapsed_s": round(time.time() - t0, 1)}
    (out / args.subset / "summary.json").write_text(json.dumps(summ, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    q = [c for c in cells_out if c["qualifies"]]
    print(f"[pos] done in {time.time() - t0:.0f}s | cells qualifying (flip down >=3 pp, CI<0, p<=0.05, acc >= -2 pp, content control < half): {len(q)}/{len(cells_out)}", flush=True)
    for c in cells_out:
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k in (*key_cols, "dflip_score_pp", "dacc_score_pp", "p_empirical", "dflip_content_pp", "qualifies")}, flush=True)


if __name__ == "__main__":
    main()
