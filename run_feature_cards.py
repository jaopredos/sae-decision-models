"""Stage 5: feature cards (markdown) for the top-ranked SAE features of each criterion.

Usage: python run_feature_cards.py --sae checkpoints/sae/ml_enc17_k32/sae.pt --layer enc_17 --name ml_enc17_k32
           [--stats-dir results/sae_features/<name>] [--tokens-dir data/activations/tokens/multilingual]
           [--stage3-dir data/activations/typed_decisions] [--top 30] [--out results/sae_features/feature_cards]
           [--device cuda]

Criteria (rankings come from the discovery set only; the test appears as a replication column):
  pred_error, corr_error, state_error                  error-rate enrichment per view (q <= 0.05, by lift)
  <view>_error_stratified, <view>_probe_right_laya_wrong_stratified  same, ranked by the CMH z controlling for the
                                                        question and the option Laya picked
  pred_probe_right_laya_wrong, corr_probe_right_laya_wrong
                                                        same, among decisions the out-of-fold probe gets right
  attribution_toward_wrong, attribution_toward_correct most negative / positive mean gradient attribution at the
                                                        predicted-option marker over the errors
Each card shows: the top-activating tokens and a band of MEDIAN activations, each with the whole instance
(question, options, Laya's answer, gold) and the token marked; the feature's confusion table; the curve
activation vs. the predicted probability of that option; discovery/dev/test numbers; and an empty
"Interpretação" field for you. Nothing is labelled by me.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER, ROLE_MARKER
from src.sae import load_sae
from src.sae_data import load_tokens, token_is_dev

REPO_ROOT = Path(__file__).resolve().parent
ROLE_NAMES = ["CLS", "pergunta", "marcador", "opção", "SEP-cabeça", "estado", "SEP-fim"]


def pick_features(stats_dir: Path, top: int) -> dict:
    """{criterion: [feature ids]} from the discovery-set rankings."""
    crit = {}
    for kind in ("error", "probe_right_laya_wrong"):
        for view in ("pred", "corr", "state"):
            f = stats_dir / f"feature_stats_{view}_{kind}.parquet"
            if kind == "probe_right_laya_wrong" and view == "state":
                continue
            if not f.exists():
                continue
            t = pd.read_parquet(f)
            t = t[(t.split == "discovery") & (t.stratum == "all") & t.supported]
            sig = t[t.q_bh <= 0.05]
            ranked = (sig if len(sig) >= top else t).sort_values(["q_bh", "p_two_sided"] if len(sig) < top else "lift",
                                                                 ascending=[True, True] if len(sig) < top else False)
            crit[f"{view}_{kind}"] = ranked.head(top)["feature"].tolist()
            if "cmh_q" in t:  # controlling for the question and the option Laya picked
                st = t[(t.cmh_q <= 0.05) & (t.cmh_z > 0)].sort_values("cmh_z", ascending=False)
                crit[f"{view}_{kind}_stratified"] = st.head(top)["feature"].tolist()
    f = stats_dir / "feature_stats_attribution.parquet"
    if f.exists():
        t = pd.read_parquet(f)
        t = t[(t.split == "discovery") & (t.stratum == "all") & (t.group == "pred")]
        crit["attribution_toward_wrong"] = t.sort_values("mean_attr").head(top)["feature"].tolist()
        crit["attribution_toward_correct"] = t.sort_values("mean_attr", ascending=False).head(top)["feature"].tolist()
    return crit


@torch.no_grad()
def token_activations(sae, acts, mask_idx, feats, device, chunk=100_000):
    """float16 [len(mask_idx), len(feats)] latent activations at the given token rows (massive -> 0)."""
    cols = torch.tensor(feats, device=device)
    out = np.zeros((len(mask_idx), len(feats)), dtype=np.float16)
    for a in range(0, len(mask_idx), chunk):
        x = torch.from_numpy(np.asarray(acts[mask_idx[a:a + chunk]])).to(device).float()
        z = sae.encode(x).masked_fill(sae.is_massive(x)[:, None], 0.0)
        out[a:a + chunk] = z[:, cols].half().cpu().numpy()
    return out


def render_instance(tok, ids, role, pos, meta_row, act) -> str:
    """Whole row decoded, with the activating token marked as ⟦…⟧ and the instance's answer line."""
    pieces = []
    for i, (t, r) in enumerate(zip(ids, role)):
        s = tok.decode([int(t)])
        if i == pos:
            s = f"⟦{s}⟧"
        if r == ROLE_MARKER:
            s = "\n▸ " + s
        elif r in (4, 5) and (i == 0 or role[i - 1] != r):
            s = "\n\n" + s
        pieces.append(s)
    opts = list(meta_row["option_ids"])
    probs = np.round(np.asarray(meta_row["model_probs"], float), 3).tolist()
    head = (f"**{meta_row['case_id']} / {meta_row['question_id']}** ({meta_row['qtype']}, {meta_row['workflow']}) | "
            f"ativação {act:.3f} em `{ROLE_NAMES[role[pos]]}` | opções {opts} | Laya: **{meta_row['model_pred']}** {probs} | "
            f"gabarito: **{meta_row['gold_argmax']}**")
    return head + "\n\n```\n" + "".join(pieces) + "\n```\n"


def curve_png(path, x, p_pred, p_gold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#e4e3df"
    on = x > 0
    qs = np.unique(np.quantile(x[on], np.linspace(0, 1, 6))) if on.sum() >= 10 else np.array([])
    bins = [("inactive", ~on)] + ([(f"{a:.2g}-{b:.2g}", on & (x >= a) & (x <= b)) for a, b in zip(qs[:-1], qs[1:])] if len(qs) > 1 else [])
    names = [n for n, m in bins if m.sum() > 0]
    mp = [p_pred[m].mean() for n, m in bins if m.sum() > 0]; mg = [p_gold[m].mean() for n, m in bins if m.sum() > 0]
    cnt = [int(m.sum()) for n, m in bins if m.sum() > 0]
    fig, ax = plt.subplots(figsize=(6, 3), dpi=110)
    ax.plot(range(len(names)), mp, color=blue, marker="o", lw=2, label="Laya predicted probability")
    ax.plot(range(len(names)), mg, color=orange, marker="o", lw=2, label="gold probability")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, cnt)], fontsize=7)
    ax.set_xlabel("feature activation at the option marker", fontsize=8); ax.set_ylabel("mean probability of that option", fontsize=8)
    ax.set_ylim(0, 1); ax.grid(axis="y", color=grid)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True); ap.add_argument("--layer", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--stats-dir", default=None)
    ap.add_argument("--tokens-dir", default=str(REPO_ROOT / "data" / "activations" / "tokens" / "multilingual"))
    ap.add_argument("--stage3-dir", default=str(REPO_ROOT / "data" / "activations" / "typed_decisions"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sae_features" / "feature_cards"))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--features-file", default=None, help="JSON list of feature ids to card instead of the rankings")
    ap.add_argument("--criterion-name", default="given")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repo", default=LAYA_REPO); ap.add_argument("--subfolder", default=LAYA_SUBFOLDER); ap.add_argument("--revision", default=LAYA_REVISION)
    args = ap.parse_args()
    stats_dir = Path(args.stats_dir or REPO_ROOT / "results" / "sae_features" / args.name)
    out = Path(args.out) / args.name
    (out / "img").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    sae = load_sae(args.sae, args.device).eval()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    sub = None if args.subfolder in ("", "none", "None") else args.subfolder
    base = snapshot_download(args.repo, revision=args.revision, allow_patterns=[f"{sub + '/' if sub else ''}tokenizer/*"])
    tok = AutoTokenizer.from_pretrained(str(Path(base) / (sub or "") / "tokenizer"))

    if args.features_file:
        import json
        crit = {args.criterion_name: [int(f) for f in json.loads(Path(args.features_file).read_text())][: args.top]}
    else:
        crit = pick_features(stats_dir, args.top)
    feats = sorted({f for fl in crit.values() for f in fl})
    which = {f: [c for c, fl in crit.items() if f in fl] for f in feats}
    print(f"[cards] {len(feats)} features in {len(crit)} criteria", flush=True)

    acts, ti, rows = load_tokens(Path(args.tokens_dir) / "train", args.layer)
    disc_tok = np.flatnonzero(~token_is_dev(ti, rows))
    A = token_activations(sae, acts, disc_tok, feats, device)
    ti_d = ti.iloc[disc_tok].reset_index(drop=True)
    dv = pd.read_parquet(stats_dir / "decision_views.parquet")
    dec3 = pd.read_parquet(Path(args.stage3_dir) / "train" / "decisions.parquet")[["case_id", "question_id", "option_ids", "model_probs", "gold_dist"]]
    meta_tr = rows.merge(dec3, on=["case_id", "question_id"]).merge(dv[dv.split_src == "train"][["case_id", "question_id", "workflow", "qtype", "model_pred", "gold_argmax", "pred_idx", "gold_idx", "err"]], on=["case_id", "question_id"]).sort_values("decision_idx").reset_index(drop=True)
    rng = np.random.default_rng(0)
    stats_cache = {}
    for kind in ("error", "probe_right_laya_wrong"):
        for view in ("pred", "corr", "state"):
            f = stats_dir / f"feature_stats_{view}_{kind}.parquet"
            if f.exists():
                stats_cache[(view, kind)] = pd.read_parquet(f)
    conf = {v: pd.read_parquet(stats_dir / f"feature_stats_{v}_confusion.parquet") for v in ("pred", "corr", "state") if (stats_dir / f"feature_stats_{v}_confusion.parquet").exists()}

    marker_rows = np.flatnonzero(ti_d["role"].to_numpy() == ROLE_MARKER)
    for k, f in enumerate(feats):
        a = A[:, k].astype(np.float32)
        nz = np.flatnonzero((a > 0) & (ti_d["role"].to_numpy() >= 0))
        lines = [f"# Feature {f}\n", f"Critérios em que aparece: {', '.join(which[f])}\n", "**Interpretação (preencher):** \n\n---\n"]
        if len(nz):
            top_idx = nz[np.argsort(a[nz])[::-1][:args.examples]]
            med = np.median(a[nz]); band = nz[(a[nz] >= np.quantile(a[nz], 0.4)) & (a[nz] <= np.quantile(a[nz], 0.6))]
            mid_idx = rng.choice(band, size=min(args.examples, len(band)), replace=False) if len(band) else np.array([], int)
            for title, idx in (("Maiores ativações", top_idx), (f"Ativações medianas (faixa 40-60%, mediana {med:.3f})", mid_idx)):
                lines.append(f"## {title}\n")
                for i in idx:
                    d = int(ti_d.at[i, "decision_idx"]); r = rows.iloc[d]
                    seq = ti_d[ti_d.decision_idx == d]
                    lines.append(render_instance(tok, seq["token_id"].to_numpy(), seq["role"].to_numpy(), int(ti_d.at[i, "position"]), meta_tr.iloc[d], float(a[i])))
        lines.append("## Estatísticas (conjunto de descoberta, dev e teste)\n")
        for (view, kind), t in stats_cache.items():
            s = t[(t.feature == f) & (t.stratum == "all")].set_index("split")
            if s.empty:
                continue
            lines.append(f"**{view} / {kind}**\n\n| split | decisões | ativas | ativas e Laya errou | taxa | lift | q (BH) | OR (CMH) | q (CMH) |\n|---|---|---|---|---|---|---|---|---|")
            for sp in ("discovery", "dev", "test"):
                if sp in s.index:
                    r = s.loc[sp]
                    lines.append(f"| {sp} | {int(r.n_decisions)} | {int(r.n_active)} | {int(r.n_active_y)} | {r.rate_active:.3f} | {r.lift:.2f} | {r.q_bh:.3g} | {r.get('cmh_or', float('nan')):.2f} | {r.get('cmh_q', float('nan')):.3g} |")
            lines.append("")
        for v, c in conf.items():
            r = c[c.feature == f]
            if len(r) and r.iloc[0]["top_confusion_count"] > 0:
                r = r.iloc[0]
                lines.append(f"Confusão mais sobre-representada ({v}): `{r['top_confusion']}` (n={int(r.top_confusion_count)}, lift {r.top_confusion_lift:.2f}, p={r.top_confusion_p:.3g}). Formato `pergunta|índice previsto>índice correto`.\n")
        # curve: activation at option markers vs. predicted / gold probability of that option
        m_in = np.isin(ti_d["decision_idx"].to_numpy()[marker_rows], np.arange(len(meta_tr)))
        d_of = ti_d["decision_idx"].to_numpy()[marker_rows]; o_of = ti_d["option_index"].to_numpy()[marker_rows]
        xm = a[marker_rows]
        pp = np.array([meta_tr.at[d, "model_probs"][o] for d, o in zip(d_of, o_of)], float)
        pg = np.array([meta_tr.at[d, "gold_dist"][o] for d, o in zip(d_of, o_of)], float)
        curve_png(out / "img" / f"f{f:05d}.png", xm, pp, pg)
        lines.append(f"\n![ativação × probabilidade](img/f{f:05d}.png)\n")
        (out / f"f{f:05d}.md").write_text("\n".join(lines))
    for c, fl in crit.items():
        (out / f"index_{c}.md").write_text(f"# Critério: {c}\n\n" + "\n".join(f"- [feature {f}](f{f:05d}.md)" for f in fl) + "\n")
    print(f"[cards] wrote {len(feats)} cards to {out}", flush=True)


if __name__ == "__main__":
    main()
