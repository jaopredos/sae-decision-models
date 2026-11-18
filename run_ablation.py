"""Stage 6: causal ablation of SAE features on Laya's choices (results/sae_features/PROTOCOLO.md, emendas 4 and 5).

Usage: python run_ablation.py --sae checkpoints/sae/ml_enc08_k32/sae.pt --layer enc_08 --name ml_enc08_k32
           --subset {dev,test,discovery} [--limit N] [--out DIR] [--n-null 100] [--n-single 25] [--device cuda]

Two variants, each with its own list selected on the discovery set only and written once to
<out>/selected_features.json (later runs, dev then test, reuse it):
  pred  zero the features only at the [MASK] of the option Laya picks; list ranked by mean `a_pred` over errors.
  all   zero them at every non-massive token; list ranked by mean total attribution (a_pred+a_corr+a_state+a_other).
Each condition uses `SAEIntervention(mode="edit")` (reconstruction error kept), runs Laya on the eval cases in
batches by workflow and compares raw logits with the un-intervened run. Margin m = logit[gold] - logit[pred0],
pred0 = baseline argmax. The test subset only runs the K = 20, 100 lists and nulls (replication).
Writes <out>/<subset>/{conditions.csv, summary.json, headline_dm.npz}.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER, layer_modules, load_agent, option_ids
from src.datasets import load_typed_cases, typed_dev_ids
from src.models import set_all_seeds
from src.sae import load_sae
from src.sae_intervention import SAEIntervention

REPO_ROOT = Path(__file__).resolve().parent
KS = (1, 5, 20, 100)
VARIANTS = ('pred', 'all')
MIN_SUPPORT = 20
WIDTH = 5  # widest question (options)


# ---- selection (discovery only) -------------------------------------------------------------------
def load_attr(stats: Path, split_src: str):
    z = np.load(stats / "attribution" / f"attribution_{split_src}.npz", allow_pickle=True)
    return z


def select_features(stats: Path, n_top: int = 100) -> dict:
    z = load_attr(stats, "train")
    dv = pd.read_parquet(stats / "decision_views.parquet")
    key = pd.DataFrame({"split_src": "train", "case_id": z["case_id"], "question_id": z["question_id"]})
    key = key.merge(dv[["split_src", "case_id", "question_id", "subset"]], on=["split_src", "case_id", "question_id"], how="left")
    disc = (key["subset"] == "discovery").to_numpy()
    f32 = lambda name: z[name][disc].astype(np.float32)
    ranks = {"pred": f32("a_pred").mean(0),
             "all": (f32("a_pred") + f32("a_corr") + f32("a_state") + f32("a_other")).mean(0)}
    view = {"pred": "pred", "all": "state"}  # which activation view supports each variant
    sel = {"n_discovery_errors": int(disc.sum())}
    for v in VARIANTS:
        st = pd.read_parquet(stats / f"feature_stats_{view[v]}_error.parquet")
        st = st[(st.split == "discovery") & (st.stratum == "all")].set_index("feature")
        n_active = st["n_active"].reindex(range(len(ranks[v]))).fillna(0).to_numpy()
        ok = n_active >= MIN_SUPPORT
        order = np.argsort(ranks[v])
        neg = [int(f) for f in order if ok[f]][:n_top]
        pos = [int(f) for f in order[::-1] if ok[f]][:n_top]
        sel[v] = {"neg": neg, "pos": pos, "mean_attr_discovery": {str(f): float(ranks[v][f]) for f in neg + pos},
                  "n_active": {str(f): int(n_active[f]) for f in range(len(n_active)) if ok[f]}}
    return sel


NULL_FALLBACKS = {"n": 0, "draws": 0}


def matched_null(sel_v: dict, k: int, rng, pool_bins, bins_of, exclude) -> list:
    """One random set of k features, each drawn from the pool in the same activation-frequency quintile as neg[i];
    if that quintile has nothing left (the top one is small), from the nearest quintile that does (counted)."""
    out = []
    for f in sel_v["neg"][:k]:
        b = bins_of[f]
        for d in range(len(pool_bins)):
            cand = [g for bb in {b - d, b + d} if bb in pool_bins for g in pool_bins[bb] if g not in exclude and g not in out]
            if cand:
                break
        NULL_FALLBACKS["draws"] += 1
        NULL_FALLBACKS["n"] += int(d > 0)
        out.append(int(rng.choice(cand)))
    return out


# ---- running Laya under an intervention -----------------------------------------------------------
class Runner:
    def __init__(self, agent, sae, layer_mod, cases):
        self.agent, self.sae, self.layer = agent, sae, layer_mod
        self.mask_id = agent.tok.mask_token_id
        self.cur, self.count, self.pred0 = {}, 0, None
        agent.model.register_forward_pre_hook(self._pre)
        groups = {}
        for c in cases:
            groups.setdefault(json.dumps(c["questions"], sort_keys=True), []).append(c)
        self.groups = list(groups.values())
        self.rows, self.gold, self.k = [], [], []
        for g in self.groups:
            qids = list(g[0]["questions"])
            ks = [len(option_ids(agent._to_internal(g[0]["questions"][q]))) for q in qids]
            for c in g:
                for q, k in zip(qids, ks):
                    self.rows.append((c["case_id"], q))
                    self.gold.append(option_ids(agent._to_internal(c["questions"][q])).index(str(c["gold"][q]["label"])))
                    self.k.append(k)
        self.gold = np.array(self.gold)
        self.case_ids = np.array([r[0] for r in self.rows])

    def _pre(self, _m, a):
        rows = a[0].shape[0]
        self.cur.update(ids=a[0], marker_pos=a[2], offset=self.count)
        self.count += rows

    def pred_marker(self, h):
        """True at the [MASK] of the option the un-intervened model picked (row order = baseline row order)."""
        mp, off = self.cur["marker_pos"], self.cur["offset"]
        rows = mp.shape[0]
        pos = mp.gather(1, self.pred0[off:off + rows, None].to(mp.device)).squeeze(1)
        m = torch.zeros(h.shape[:2], dtype=torch.bool, device=h.device)
        m[torch.arange(rows, device=h.device), pos.to(h.device)] = True
        return m

    @torch.no_grad()
    def logits(self, edits=None, variant="pred") -> np.ndarray:
        outs, self.count = [], 0
        h = self.agent.model.register_forward_hook(lambda m, i, o: outs.append(o[0].float().cpu()))
        try:
            ctx = (SAEIntervention(self.sae, self.layer, mode="edit", edits={f: ("zero",) for f in edits},
                                   position_mask=self.pred_marker if variant == "pred" else None)
                   if edits else None)
            if ctx:
                ctx.__enter__()
            try:
                for g in self.groups:
                    self.agent.predict_batch([c["state"] for c in g], g[0]["questions"], batch_size=40)
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)
        finally:
            h.remove()
        L = torch.cat([F.pad(o, (0, WIDTH - o.shape[1]), value=-1e4) for o in outs]).numpy()
        assert len(L) == len(self.rows), (len(L), len(self.rows))
        return L


def margin(L, pred0, gold):
    n = np.arange(len(L))
    return L[n, gold] - L[n, pred0]


def effect(L, base, gold):
    pred0 = base.argmax(1)
    err0 = pred0 != gold
    dm = margin(L, pred0, gold) - margin(base, pred0, gold)
    pred = L.argmax(1)
    return {"dm_err": float(dm[err0].mean()), "acc": float((pred == gold).mean()),
            "fixed": float((pred[err0] == gold[err0]).mean()), "broken": float((pred[~err0] != gold[~err0]).mean())}, dm, err0


def boot_ci(dm, err0, cases, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(cases, return_inverse=True)
    s = np.bincount(inv, weights=np.where(err0, dm, 0.0))
    n = np.bincount(inv, weights=err0.astype(float))
    est = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        est.append((w * s).sum() / (w * n).sum())
    return [float(x) for x in np.percentile(est, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", choices=["dev", "test", "discovery"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--n-single", type=int, default=25)
    ap.add_argument("--stats-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_all_seeds()
    stats = Path(args.stats_dir or REPO_ROOT / "results" / "sae_features" / args.name)
    out = Path(args.out or REPO_ROOT / "results" / "sae_features" / "ablation" / args.name)
    (out / args.subset).mkdir(parents=True, exist_ok=True)

    sel_path = out / "selected_features.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        print("[abl] selection reused from", sel_path, flush=True)
    else:
        sel = select_features(stats)
        sel_path.write_text(json.dumps(sel, indent=1))
        print("[abl] selection written:", sel_path, {v: (sel[v]["neg"][:4], sel[v]["pos"][:4]) for v in VARIANTS}, flush=True)

    # matched null pool per variant: supported features binned by n_active quintiles
    pools = {}
    for v in VARIANTS:
        n_act = {int(f): x for f, x in sel[v]["n_active"].items()}
        edges = np.unique(np.quantile(np.array(list(n_act.values())), np.linspace(0, 1, 6)))
        bins_of = {f: int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)) for f, x in n_act.items()}
        pools[v] = ({b: [f for f, bb in bins_of.items() if bb == b] for b in range(len(edges) - 1)}, bins_of,
                    set(sel[v]["neg"]) | set(sel[v]["pos"]))

    split = "test" if args.subset == "test" else "train"
    cases = load_typed_cases(split)
    if args.subset != "test":
        dev = typed_dev_ids()
        cases = [c for c in cases if (c["case_id"] in dev) == (args.subset == "dev")]
    if args.limit:
        cases = cases[: args.limit]
    agent = load_agent(args.device, LAYA_REPO, LAYA_SUBFOLDER, LAYA_REVISION)
    sae = load_sae(args.sae, str(agent.device)).eval()
    layer_mod = dict(layer_modules(agent.model))[args.layer]
    R = Runner(agent, sae, layer_mod, cases)
    t0 = time.time()
    base = R.logits()
    gold, pred0 = R.gold, base.argmax(1)
    R.pred0 = torch.from_numpy(pred0)
    dv = pd.read_parquet(stats / "decision_views.parquet")
    ref = dv[dv.split_src == split].set_index(["case_id", "question_id"])
    official = np.array([ref.at[r, "pred_idx"] for r in R.rows])
    agree = float((pred0 == official).mean())
    print(f"[abl] {args.subset}: {len(cases)} cases, {len(R.rows)} decisions, {len(R.groups)} workflow groups; baseline acc "
          f"{(pred0 == gold).mean():.3f}; argmax agreement with the official predict {agree:.4f}; baseline pass {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    replication = args.subset == "test"  # the test only replicates the K = 20, 100 lists and nulls
    ks = (20, 100) if replication else KS
    conds = []  # (family, K, variant, features)
    for variant in VARIANTS:
        for fam in ("neg", "pos"):
            for k in ks:
                conds.append((fam, k, variant, sel[variant][fam][:k]))
        for k in ks:
            for i in range(args.n_null):
                pb, bo, ex = pools[variant]
                conds.append(("null", k, variant, matched_null(sel[variant], k, rng, pb, bo, ex)))
        if not replication:
            for fam in ("neg", "pos"):
                for f in sel[variant][fam][: args.n_single]:
                    conds.append((f"single_{fam}", 1, variant, [f]))
    rows, headline = [], {}
    for n, (fam, k, variant, feats) in enumerate(conds, 1):
        L = R.logits(feats, variant)
        m, dm, err0 = effect(L, base, gold)
        rows.append({"family": fam, "K": k, "variant": variant, "features": " ".join(map(str, feats)), **m})
        if fam in ("neg", "pos"):
            headline[f"{fam}_{k}_{variant}"] = dm
        if n % 50 == 0:
            print(f"[abl] {n}/{len(conds)} conditions, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / args.subset / "conditions.csv", index=False)
    np.savez_compressed(out / args.subset / "headline_dm.npz", err0=pred0 != gold, case_ids=R.case_ids, **headline)

    # ---- summary ------------------------------------------------------------------------------------
    err0 = pred0 != gold
    m0 = margin(base, pred0, gold)
    summ = {"subset": args.subset, "layer": args.layer, "sae": args.name, "n_cases": len(cases), "n_decisions": len(R.rows),
            "n_baseline_errors": int(err0.sum()), "baseline_acc": float((pred0 == gold).mean()), "argmax_agreement_official": agree,
            "mean_abs_margin_on_errors": float(np.abs(m0[err0]).mean()), "null_draws_from_neighbour_quintile": dict(NULL_FALLBACKS), "headline": {}}
    for variant in VARIANTS:
        for k in ks:
            null = df[(df.family == "null") & (df.K == k) & (df.variant == variant)].dm_err.to_numpy()
            for fam in ("neg", "pos"):
                dm = headline[f"{fam}_{k}_{variant}"]
                obs = float(dm[err0].mean())
                p = float((1 + (null >= obs).sum()) / (1 + len(null))) if fam == "neg" else float((1 + (null <= obs).sum()) / (1 + len(null)))
                r = df[(df.family == fam) & (df.K == k) & (df.variant == variant)].iloc[0]
                summ["headline"][f"{fam}_K{k}_{variant}"] = {
                    "dm_err": obs, "dm_over_mean_abs_margin": obs / summ["mean_abs_margin_on_errors"], "ci95": boot_ci(dm, err0, R.case_ids),
                    "fixed": float(r.fixed), "broken": float(r.broken), "acc": float(r.acc), "null_mean": float(null.mean()),
                    "null_p5": float(np.percentile(null, 5)), "null_p95": float(np.percentile(null, 95)), "p_empirical": p, "n_null": int(len(null))}
    # calibration of the exact first-order prediction: single features, vs -mean attribution on the same errors
    if not replication:
        z = load_attr(stats, "train")
        keyz = pd.DataFrame({"case_id": z["case_id"], "question_id": z["question_id"]})
        mine = pd.DataFrame(R.rows, columns=["case_id", "question_id"])
        mine["row"] = np.arange(len(mine))
        j = keyz.reset_index().merge(mine, on=["case_id", "question_id"])
        j = j[err0[j.row.to_numpy()]]
        idx = j["index"].to_numpy()
        g32 = lambda name: z[name][idx].astype(np.float32)
        pred_mean = g32("a_pred").mean(0)
        all_mean = (g32("a_pred") + g32("a_corr") + g32("a_state") + g32("a_other")).mean(0)
        cal = []
        for variant, vec in (("pred", pred_mean), ("all", all_mean)):
            for fam in ("neg", "pos"):
                for f in sel[variant][fam][: args.n_single]:
                    r = df[(df.family == f"single_{fam}") & (df.variant == variant) & (df.features == str(f))].iloc[0]
                    cal.append({"variant": variant, "feature": f, "list": fam, "measured_dm": float(r.dm_err), "predicted_dm": float(-vec[f])})
    if not replication:
        cal = pd.DataFrame(cal)
        summ["calibration"] = {"n_decisions_used": int(len(j))}
        for variant in VARIANTS:
            c = cal[cal.variant == variant]
            summ["calibration"][variant] = {"spearman": float(c.measured_dm.corr(c.predicted_dm, method="spearman")),
                                            "sign_agreement": float((np.sign(c.measured_dm) == np.sign(c.predicted_dm)).mean()),
                                            "abs_measured_over_abs_predicted": float(c.measured_dm.abs().sum() / max(c.predicted_dm.abs().sum(), 1e-9))}
        cal.to_csv(out / args.subset / "calibration.csv", index=False)
    (out / args.subset / "summary.json").write_text(json.dumps(summ, indent=1))
    for v in VARIANTS:
        h = summ["headline"][f"neg_K20_{v}"]
        c = summ.get("calibration", {}).get(v, {}).get("spearman", float("nan"))
        print(f"[abl] done in {time.time() - t0:.0f}s | {v} neg K=20: dm_err {h['dm_err']:+.4f} ({h['dm_over_mean_abs_margin']:+.3f} of |margin|) CI "
              f"[{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}] fixed {h['fixed']:.3f} broken {h['broken']:.3f} | null p95 {h['null_p95']:+.4f} p {h['p_empirical']:.3f} | "
              f"calibration spearman {c:.2f}", flush=True)

if __name__ == "__main__":
    main()
