"""Stage B: linear option-scorer probes per layer on the stage-A activations.

Usage: python run_probes.py [--acts data/activations/typed_decisions] [--out results/probe_typed]
                            [--smoke] [--device cuda|cpu] [--job-ids "extract-smoke=1,..."]

Nothing is chosen on test. Per layer, weight decay is picked on the dev split (15% of train cases
by hash, see src/datasets.typed_dev_ids) from fits on train-minus-dev, by mean dev accuracy over
the seeds (ties: lower dev KL); the best layer is picked the same way. Each layer is then refit on
the whole train split with its chosen wd and evaluated on test, alongside a control fitted on
train labels shuffled within (workflow, question_id).

Writes under <out>/: probe_by_layer.csv, wd_selection.csv, summary_table.csv,
per_decision.parquet, acc_by_layer.png, results.json, REPORT.md.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import probe as P

REPO_ROOT = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
WD_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
N_BOOT = 2000


def layer_label(stem: str) -> str:
    return stem.split("_", 1)[1]


def tensors(x, mask, gold, device):
    return (torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device),
            torch.from_numpy(gold.astype(np.float32)).to(device))


def boot_diff(a_correct: np.ndarray, b_correct: np.ndarray, cases: np.ndarray, seed: int = 0):
    """Paired bootstrap over test *cases* (a case's 5 decisions move together): mean and 95% CI of
    acc(a) - acc(b)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(cases, return_inverse=True)
    diff = a_correct.astype(float) - b_correct.astype(float)
    per_case = np.bincount(inv, weights=diff)
    n_case = np.bincount(inv)
    stats = []
    for _ in range(N_BOOT):
        s = rng.integers(0, len(uniq), len(uniq))
        stats.append(per_case[s].sum() / n_case[s].sum())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def plot(by_layer: pd.DataFrame, refs: dict, best_stem: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference palette (dataviz skill): slot 1 blue, slot 2 orange; validated light, ΔE 24.7 CVD
    blue, orange = "#2a78d6", "#eb6834"
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    agg = (by_layer[by_layer["kind"].isin(["probe", "control"])]
           .groupby(["layer_idx", "layer", "kind"])["test_acc"].agg(["mean", "std"]).reset_index())
    labels = agg[agg["kind"] == "probe"].sort_values("layer_idx")["layer"].tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for kind, color, name in (("probe", blue, "Probe linear"), ("control", orange, "Controle (rótulos embaralhados)")):
        d = agg[agg["kind"] == kind].sort_values("layer_idx")
        ax.fill_between(x, d["mean"] - d["std"].fillna(0), d["mean"] + d["std"].fillna(0),
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(x, d["mean"], color=color, linewidth=2, marker="o", markersize=4, label=name)
    for (name, val), ls in zip(refs.items(), ("--", "-.", ":")):
        ax.axhline(val, color=ink2, linewidth=1.2, linestyle=ls, zorder=0)
    # direct labels, nudged apart when two reference values (nearly) coincide
    lo, hi = ax.get_ylim()
    gap, placed = 0.045 * (hi - lo), []
    for name, val in sorted(refs.items(), key=lambda kv: kv[1]):
        y = max(val, placed[-1] + gap) if placed else val
        placed.append(y)
        ax.text(len(x) - 0.4, y, f" {name} {val:.3f}", color=ink2, va="center", ha="left", fontsize=9,
                bbox=dict(facecolor=surface, edgecolor="none", pad=1.5))
    bi = labels.index(layer_label(best_stem))
    bm = agg[(agg["kind"] == "probe")].sort_values("layer_idx")["mean"].iloc[bi]
    ax.plot([bi], [bm], marker="o", markersize=9, markerfacecolor="none", markeredgecolor=ink, markeredgewidth=1.5)
    ax.annotate("melhor camada (dev)", (bi, bm), xytext=(-6 if bi == 0 else 0, 12), textcoords="offset points",
                ha="left" if bi == 0 else "center", fontsize=9, color=ink, bbox=dict(facecolor=surface, edgecolor="none", pad=1.5))
    head_start = labels.index("enc_final") + 0.5
    ax.axvline(head_start, color=grid, linewidth=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("enc_", "").replace("final", "norm").replace("head_", "h") for l in labels],
                       fontsize=8, color=ink2)
    ax.set_xlabel("camada  (emb · backbone 01–22 · final_norm · cabeça h1–h2)", color=ink2)
    ax.set_ylabel("acurácia no teste (vs. argmax do gabarito)", color=ink2)
    ax.set_title("Probe linear por camada — Laya multilingual, typed-decisions (média ± dp, 3 seeds)",
                 color=ink, fontsize=11, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2)
    ax.set_xlim(-0.5, len(x) + 2.2)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=str(REPO_ROOT / "data" / "activations" / "typed_decisions"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "probe_typed"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny stage-A output; exercises the code, numbers are meaningless")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--job-ids", default=os.environ.get("PROBE_JOB_IDS", ""))
    args = ap.parse_args()
    if args.smoke:
        args.acts = str(REPO_ROOT / "data" / "activations" / "smoke")
        args.out = str(REPO_ROOT / "results" / "probe_typed" / "smoke")
    acts, out = Path(args.acts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train, test = P.load_split(acts / "train"), P.load_split(acts / "test")
    assert train.layers == test.layers and train.layers, (train.layers, test.layers)
    dev = train.dec["is_dev"].to_numpy()
    if not dev.any():
        if not args.smoke:
            raise SystemExit("no dev cases in the train activations")
        # smoke only: the first N train cases may hold no hash-dev case
        dev = train.dec["case_id"].isin(train.dec["case_id"].unique()[-5:]).to_numpy()
        print("[probes] SMOKE: no hash-dev case in the subset; using the last 5 cases as dev")
    fit = ~dev
    print(f"[probes] train={len(train.dec)} decisions (dev {dev.sum()}), test={len(test.dec)}, "
          f"layers={len(train.layers)}, device={args.device}", flush=True)

    sel_rows, rows, test_probs = [], [], {}
    for li, stem in enumerate(train.layers):
        xtr, xte = train.layer(stem), test.layer(stem)

        # 1) wd selection: fit on train-minus-dev, score on dev
        std = P.standardizer(xtr[fit], train.mask[fit])
        xf, mf, gf = tensors(std(xtr[fit], train.mask[fit]), train.mask[fit], train.gold[fit], args.device)
        xd, md, _ = tensors(std(xtr[dev], train.mask[dev]), train.mask[dev], train.gold[dev], args.device)
        for wd in WD_GRID:
            for seed in SEEDS:
                params = P.fit_probe(xf, mf, gf, wd, seed)
                m = P.metrics(P.predict_probe(params, xd, md), train.gold[dev], train.gold_idx[dev], train.mask[dev])
                sel_rows.append({"layer_idx": li, "layer": layer_label(stem), "wd": wd, "seed": seed,
                                 **{f"dev_{k}": v for k, v in m.items()}})
        sel = pd.DataFrame([r for r in sel_rows if r["layer_idx"] == li])
        g = sel.groupby("wd")[["dev_acc", "dev_kl"]].mean().reset_index()
        g = g.sort_values(["dev_acc", "dev_kl"], ascending=[False, True])
        wd = float(g["wd"].iloc[0])
        dev_at_wd = sel[sel["wd"] == wd].set_index("seed")

        # 2) refit on all of train with the chosen wd; probe and shuffled-label control on test
        std = P.standardizer(xtr, train.mask)
        xa, ma, ga = tensors(std(xtr, train.mask), train.mask, train.gold, args.device)
        xt, mt, _ = tensors(std(xte, test.mask), test.mask, test.gold, args.device)
        probs_seeds = []
        for seed in SEEDS:
            params = P.fit_probe(xa, ma, ga, wd, seed)
            pt = P.predict_probe(params, xt, mt)
            probs_seeds.append(pt)
            m = P.metrics(pt, test.gold, test.gold_idx, test.mask, test.dec)
            d = dev_at_wd.loc[seed]
            rows.append({"layer_idx": li, "layer": layer_label(stem), "kind": "probe", "seed": seed, "wd": wd,
                         "dev_acc": d["dev_acc"], "dev_soft_acc": d["dev_soft_acc"], "dev_kl": d["dev_kl"],
                         **{f"test_{k}": v for k, v in m.items()}})
            gshuf = P.shuffle_within_question(train.gold, train.dec, seed)
            params = P.fit_probe(xa, ma, torch.from_numpy(gshuf).to(args.device), wd, seed)
            m = P.metrics(P.predict_probe(params, xt, mt), test.gold, test.gold_idx, test.mask, test.dec)
            rows.append({"layer_idx": li, "layer": layer_label(stem), "kind": "control", "seed": seed, "wd": wd,
                         **{f"test_{k}": v for k, v in m.items()}})
        test_probs[stem] = np.mean(probs_seeds, 0)
        cur = pd.DataFrame([r for r in rows if r["layer_idx"] == li])
        print(f"[probes] {stem:14s} wd={wd:g} dev_acc={cur[cur.kind == 'probe'].dev_acc.mean():.3f} "
              f"test_acc={cur[cur.kind == 'probe'].test_acc.mean():.3f} "
              f"control={cur[cur.kind == 'control'].test_acc.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)

    by_layer = pd.DataFrame(rows)
    pd.DataFrame(sel_rows).to_csv(out / "wd_selection.csv", index=False)
    by_layer.to_csv(out / "probe_by_layer.csv", index=False)

    # best layer: chosen on dev only
    dev_rank = (by_layer[by_layer.kind == "probe"].groupby(["layer_idx", "layer"])[["dev_acc", "dev_kl"]]
                .mean().reset_index().sort_values(["dev_acc", "dev_kl"], ascending=[False, True]))
    best_idx = int(dev_rank["layer_idx"].iloc[0])
    best_stem = train.layers[best_idx]

    # baselines on test
    model_p = P.padded(test.dec["model_probs"], test.gold.shape)
    base = {
        "Laya multilingual": model_p,
        "prior (treino)": P.prior_probs(train, test),
        "aleatório": P.uniform_probs(test),
    }
    summary = []
    for kind in ("probe", "control"):
        d = by_layer[(by_layer.layer_idx == best_idx) & (by_layer.kind == kind)]
        cols = [c for c in d.columns if c.startswith("test_")]
        name = f"probe {layer_label(best_stem)}" if kind == "probe" else f"controle {layer_label(best_stem)}"
        summary.append({"model": name, **{c[5:]: d[c].mean() for c in cols},
                        **{c[5:] + "_sd": d[c].std() for c in ("test_acc", "test_soft_acc", "test_kl")}})
    for name, pr in base.items():
        summary.append({"model": name, **P.metrics(pr, test.gold, test.gold_idx, test.mask, test.dec)})
    # random accuracy is an expectation (mean of 1/k), not the argmax of a uniform vector
    rnd = next(s for s in summary if s["model"] == "aleatório")
    k = test.mask.sum(1)
    rnd["acc"] = rnd["soft_acc"] = float((1.0 / k).mean())
    for key, idx in list(test.dec.groupby("qtype").indices.items()) + list(test.dec.groupby("workflow").indices.items()):
        rnd[f"acc_{key}"] = float((1.0 / k[idx]).mean())
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary_table.csv", index=False)

    # per-decision output for the behavioural cross-analysis (column names fixed)
    pb = test_probs[best_stem]
    per = test.dec[["case_id", "question_id", "workflow", "qtype", "n_options", "gold_argmax",
                    "gold_dist", "model_pred", "model_probs", "option_ids"]].copy()
    per["probe_layer"] = layer_label(best_stem)
    per["probe_probs"] = [pb[i, :n].tolist() for i, n in enumerate(per["n_options"])]
    per["probe_pred"] = [list(o)[int(np.argmax(p))] for o, p in zip(per["option_ids"], per["probe_probs"])]
    per.to_parquet(out / "per_decision.parquet")

    # significance-ish: paired bootstrap over test cases for the best layer
    cases = test.dec["case_id"].to_numpy()
    probe_ok = pb.argmax(1) == test.gold_idx
    cmp = {name: boot_diff(probe_ok, np.where(test.mask, pr, 0).argmax(1) == test.gold_idx, cases)
           for name, pr in base.items() if name != "aleatório"}

    refs = {n: float(summary.loc[summary.model == n, "acc"].iloc[0]) for n in base}
    plot(by_layer, refs, best_stem, out / "acc_by_layer.png")

    results = {
        "best_layer": layer_label(best_stem), "best_layer_stem": best_stem,
        "best_layer_selected_on": "dev accuracy (mean over seeds), ties -> lower dev KL",
        "summary": summary.to_dict(orient="records"),
        "best_minus_baseline_acc_bootstrap95": {k: {"diff": v[0], "lo": v[1], "hi": v[2]} for k, v in cmp.items()},
        "seeds": list(SEEDS), "wd_grid": list(WD_GRID),
        "chosen_wd": by_layer[by_layer.kind == "probe"].groupby("layer")["wd"].first().to_dict(),
        "n_test_gold_ties": int(((test.gold == test.gold.max(1, keepdims=True)) & test.mask).sum(1).__gt__(1).sum()),
        "n_train_fit": int(fit.sum()), "n_dev": int(dev.sum()), "n_test": len(test.dec),
        "smoke": args.smoke, "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_report(out, acts, results, summary, by_layer, args.job_ids)
    print(f"[probes] done in {time.time() - t0:.0f}s -> {out}", flush=True)


def _rel(p: Path) -> Path:
    try:
        return p.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return p


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(out: Path, acts: Path, res: dict, summary: pd.DataFrame, by_layer: pd.DataFrame, job_ids: str):
    s = summary.set_index("model")
    best = res["best_layer"]
    pr, ct = s.loc[f"probe {best}"], s.loc[f"controle {best}"]
    laya, prior, rnd = s.loc["Laya multilingual"], s.loc["prior (treino)"], s.loc["aleatório"]
    boot = res["best_minus_baseline_acc_bootstrap95"]
    acts, out = _rel(acts), _rel(out)
    tr = {sp: json.loads((acts / sp / "truncation.json").read_text()) for sp in ("train", "test")}
    par = {sp: json.loads((acts / sp / "parity.json").read_text()) for sp in ("train", "test")}
    meta = {sp: json.loads((acts / sp / "meta.json").read_text()) for sp in ("train", "test")}
    sel_gap = pr["acc"] - ct["acc"]

    def verdict(name):
        b = boot[name]
        if b["lo"] > 0:
            return f"acima, por {b['diff']*100:+.1f} p.p. (IC95% bootstrap por caso: {b['lo']*100:+.1f} a {b['hi']*100:+.1f})"
        if b["hi"] < 0:
            return f"**abaixo**, por {b['diff']*100:+.1f} p.p. (IC95%: {b['lo']*100:+.1f} a {b['hi']*100:+.1f})"
        return f"sem diferença distinguível: {b['diff']*100:+.1f} p.p. (IC95%: {b['lo']*100:+.1f} a {b['hi']*100:+.1f}, inclui 0)"

    cols = ["acc", "soft_acc", "kl", "acc_choice", "acc_score", "acc_noul"]
    wfs = sorted(c for c in summary.columns
                 if c.startswith("acc_") and c != "acc_sd" and c[4:] not in ("choice", "score", "noul"))
    lines = [
        "# Probe linear por camada — Laya multilingual × typed-decisions",
        "",
        "> " + ("**SMOKE** — subconjunto mínimo só para exercitar o código; os números não significam nada."
                if res["smoke"] else "Execução completa (1.200 casos de treino, 400 de teste)."),
        "",
        "## Resultado",
        "",
        f"- **Melhor camada (escolhida no dev, nunca no teste): `{best}`.** Acurácia de teste do probe "
        f"{_fmt(pr['acc'])} ± {_fmt(pr['acc_sd'])} (3 seeds), contra {_fmt(laya['acc'])} do próprio Laya, "
        f"{_fmt(prior['acc'])} do prior e {_fmt(rnd['acc'])} do aleatório.",
        f"- Probe vs. Laya: {verdict('Laya multilingual')}.",
        f"- Probe vs. prior: {verdict('prior (treino)')}.",
        f"- **Controle** (mesmo probe e wd, gabarito embaralhado dentro de cada (workflow, pergunta)): "
        f"{_fmt(ct['acc'])} ± {_fmt(ct['acc_sd'])}. Seletividade (probe − controle) = {sel_gap*100:+.1f} p.p. "
        "O controle preserva a marginal de cada pergunta, então ele mede o que se acerta só com a identidade da "
        "opção (≈ prior); a seletividade é o que vem da informação específica de cada caso.",
        "",
        "| modelo | acc | acc suave | KL(gold‖p) | choice | score | noul | " + " | ".join(w[4:] for w in wfs) + " |",
        "|---|" + "---|" * (len(cols) + len(wfs)),
    ]
    for name, r in s.iterrows():
        acc = _fmt(r["acc"]) + (f" ± {_fmt(r['acc_sd'])}" if not np.isnan(r.get("acc_sd", np.nan)) else "")
        lines.append(f"| {name} | {acc} | {_fmt(r['soft_acc'])} | {_fmt(r['kl'])} | "
                     + " | ".join(_fmt(r[c]) for c in cols[3:]) + " | "
                     + " | ".join(_fmt(r[w]) for w in wfs) + " |")
    lines += [
        "",
        "Acurácia = argmax da predição igual ao `label` do gabarito (o `label` está sempre entre os argmax da "
        f"distribuição; {res['n_test_gold_ties']} decisões de teste têm empate no gabarito e contam pelo `label`). Acurácia suave = média de "
        "`gold_dist[pred]`. KL = KL(gabarito ‖ predição), média por decisão. Aleatório: acurácia esperada (média de 1/k). "
        "Prior: frequência do `label` por (workflow, pergunta) no treino, suavizada com +0,5 (só afeta a KL).",
        "",
        "![acurácia por camada](acc_by_layer.png)",
        "",
        "## Método (resumo)",
        "",
        "- Ativações no `[MASK]` de cada opção, capturadas por hooks **dentro** de `agent.predict` (entrada montada "
        "pelo próprio `laya.common.build_sequence`, `max_len`/`head_max_len` do checkpoint: "
        f"{meta['test']['max_len']}/{meta['test']['head_max_len']}; autocast `{meta['test']['autocast_dtype_5_rows']}` em "
        f"`{meta['test']['device']}`). Camadas: {', '.join(meta['test']['layers'])}.",
        "- Probe: `s = w·h + b` por marcador, softmax entre as opções da pergunta, CE contra o gabarito suave + "
        "wd·‖w‖². O `b` se cancela no softmax. Features padronizadas com estatísticas do conjunto de ajuste. "
        "L-BFGS em batch cheio; o problema é convexo, então a variação entre seeds do probe real é ~0 por construção "
        "(a seed só muda a inicialização); no controle a seed também muda a permutação.",
        "- `emb` é um piso de sanidade, não uma camada informativa: na saída dos embeddings todo marcador é o mesmo "
        "vetor de `[MASK]` (o ModernBERT não tem embedding posicional; a posição entra via RoPE na atenção), então o "
        "probe só pode dar distribuição uniforme e o argmax cai na primeira opção.",
        f"- Seleção só no dev ({res['n_dev']} decisões = 15% dos casos de treino por hash do `case_id`): "
        f"wd ∈ {list(res['wd_grid'])} por camada e melhor camada, ambos por acurácia média no dev (empate → menor KL). "
        f"Depois, reajuste no treino inteiro ({res['n_train_fit'] + res['n_dev']} decisões) e avaliação no teste.",
        "",
        "## Paridade (checagem obrigatória da etapa A)",
        "",
    ]
    for sp in ("train", "test"):
        p = par[sp]
        lines.append(f"- {sp}: {p['n_decisions']} decisões, passou={p['passed']}; "
                     f"máx |Δlogit| recomputado vs forward = {p['logit_absdiff_recomputed_vs_forward']['max']:.2e}; "
                     f"máx |Δp| recomputado vs predict sem hooks = {p['prob_absdiff_recomputed_vs_ref']['max']:.2e} "
                     f"(predict arredonda para 4 casas; tolerância {p['prob_tol']}); "
                     f"máx |Δlogit| a partir dos vetores salvos em fp16 = {p['logit_absdiff_fp16saved_vs_forward']['max']:.2e}.")
    lines += ["", "## Truncamento", ""]
    for sp in ("train", "test"):
        t = tr[sp]
        lines.append(f"- {sp}: {t['cases_with_state_truncated']}/{t['n_cases']} casos com o estado cortado "
                     f"({t['state_tokens_dropped_total']} tokens descartados; maior estado {t['state_tokens_max']} tokens, "
                     f"maior sequência {t['seq_len_max']}); opções cortadas no limite de 48 tokens: "
                     f"{t['options_cut_at_48_tokens']}; decisões com orçamento por opção reduzido: "
                     f"{t['decisions_with_head_budget_cap']}; opções colapsadas: {t['decisions_with_collapsed_options']}.")
    v = meta["test"]["versions"]
    lines += [
        "", "## Reprodutibilidade", "",
        f"- Versões: Python {v['python']}, torch {v['torch']} (CUDA {v['cuda']}), laya {v['laya']}, "
        f"transformers {v['transformers']}; GPU: {meta['test']['gpu']}. Ambiente completo: `uv.lock` (raiz do repo) "
        "e `pip_freeze.txt` ao lado deste relatório.",
        f"- Laya `{meta['test']['laya_revision']}`; dataset `LocalLLaMA/typed-decisions@{meta['test']['dataset_revision']}`.",
        f"- Seeds: {res['seeds']} (probe/controle); bootstrap seed 0, {N_BOOT} reamostras por caso.",
        f"- Jobs: {job_ids or '(preencher com os IDs do sauron)'}.",
        "- Artefatos (relativos à raiz do repo): "
        f"ativações `{acts}/<split>/L??_*.npy` (+ `markers.parquet`, `decisions.parquet`, `parity*.`, `truncation.json`, `meta.json`); "
        f"`{out}/per_decision.parquet`, `probe_by_layer.csv`, `wd_selection.csv`, `summary_table.csv`, `results.json`, `acc_by_layer.png`.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
