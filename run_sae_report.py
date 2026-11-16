"""Stage 4/5: assemble results/sae_features/REPORT.md tables from the artifacts of the runs.

Usage: python run_sae_report.py --names ml_enc17_k32,ml_enc08_k32 [--literal ml_enc17_k32_literal]
           [--part-c results/sae_features/part_c] [--part-c-name partc_base_encNN]
           [--jobs "3603=extract-tokens-smoke,3605=extract-tokens,..."] [--dir results/sae_features]
           [--out results/sae_features/REPORT_tables.md]

It only writes numbers and tables (no interpretation, no feature labels). The prose of REPORT.md is written by
hand on top of these tables, so every claim in it can be traced to a table here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    if df is None or df.empty:
        return "_(sem dados)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def quality_section(d: Path, names) -> str:
    f = d / "sae_quality.csv"
    if not f.exists():
        return "_(sae_quality.csv ausente)_\n"
    q = pd.read_csv(f)
    s = ""
    for name in names:
        t = q[q["sae"] == name]
        if t.empty:
            continue
        s += f"\n### {name}\n\n**FVE e L0 (dev e texto genérico retido)**\n\n"
        a = t[t.metric.isin(["fve", "l0", "n_tokens", "frac_massive"]) & t.section.isin(["domain_dev", "generic_val"])]
        s += md(a.pivot_table(index=["section", "group"], columns="metric", values="value").reset_index())
        s += "\n**Latentes mortas**\n\n" + md(t[t.section == "dead"][["group", "metric", "value"]])
        dn = t[t.section == "downstream"].pivot_table(index="group", columns="metric", values="value").reset_index()
        s += "\n**Fidelidade a jusante (dev)**\n\n" + md(dn[["group"] + [c for c in ("accuracy", "delta_accuracy_pp", "kl_nat", "argmax_flip_rate", "n_decisions") if c in dn]])
        fl = d / f"sae_quality_flags_{name}.json"
        if fl.exists():
            flags = json.loads(fl.read_text())["flags"]
            s += "\n**Sinalizadores do protocolo:** " + ("; ".join(flags) if flags else "nenhum") + "\n"
    return s


def top_table(d: Path, name: str, view: str, kind: str, k: int = 10) -> str:
    f = d / name / f"feature_stats_{view}_{kind}.parquet"
    if not f.exists():
        return ""
    t = pd.read_parquet(f)
    t = t[t.stratum == "all"]
    disc = t[(t.split == "discovery") & t.supported]
    sig = disc[disc.q_bh <= 0.05]
    top = (sig.sort_values("lift", ascending=False) if len(sig) else disc.sort_values("p_two_sided")).head(k)
    rows = []
    for _, r in top.iterrows():
        rep = {sp: t[(t.split == sp) & (t.feature == r.feature)] for sp in ("dev", "test")}
        row = {"feature": int(r.feature), "ativas": int(r.n_active), "taxa de erro": r.rate_active, "base": r.base_rate, "lift": r.lift, "q (BH)": r.q_bh}
        for sp in ("dev", "test"):
            x = rep[sp]
            row[f"lift {sp}"] = float(x["lift"].iloc[0]) if len(x) else np.nan
            row[f"q {sp}"] = float(x["q_bh"].iloc[0]) if len(x) else np.nan
        rows.append(row)
    n_sig = int((disc.q_bh <= 0.05).sum())
    return f"\n**{view} / {kind}**: {n_sig} features com q <= 0,05 de {len(disc)} com suporte; top {k} por lift (descoberta) com replicação:\n\n" + md(pd.DataFrame(rows))


def attribution_section(d: Path, name: str, k: int = 10) -> str:
    f = d / name / "feature_stats_attribution.parquet"
    if not f.exists():
        return ""
    t = pd.read_parquet(f)
    t = t[(t.stratum == "all") & (t.group == "pred")]
    s = ""
    for title, asc in (("empurram para a resposta errada (atribuição média mais negativa)", True), ("empurram para a correta (mais positiva)", False)):
        disc = t[t.split == "discovery"].sort_values("mean_attr", ascending=asc).head(k)
        rows = []
        for _, r in disc.iterrows():
            row = {"feature": int(r.feature), "erros": int(r.n_errors), "atribuição média": r.mean_attr, "desvio": r.sd_attr, "fração negativa": r.frac_negative}
            for sp in ("dev", "test"):
                x = t[(t.split == sp) & (t.feature == r.feature)]
                row[f"média {sp}"] = float(x["mean_attr"].iloc[0]) if len(x) else np.nan
            rows.append(row)
        s += f"\n**Atribuição no marcador previsto: features que {title}** (conjunto de descoberta, com dev e teste ao lado)\n\n" + md(pd.DataFrame(rows), "{:.4f}")
    return s


def bias_section(d: Path, name: str) -> str:
    f = d / name / "prior_bias_summary.json"
    if not f.exists():
        return ""
    return "\n**Viés de prior (Laya escolhe a opção da hipótese mais do que o gabarito?)**\n\n" + md(pd.DataFrame(json.loads(f.read_text())))


def part_c_section(c: Path, top_k: int = 10) -> str:
    if not c.exists():
        return "_(Parte C ainda não rodou)_\n"
    s = ""
    f = c / "part_c_report.json"
    if f.exists():
        r = json.loads(f.read_text())
        s += "**Entradas idênticas:** " + json.dumps(r["identical_inputs"]) + "\n\n"
        fve = r["fve_base_sae"]
        s += "**FVE do SAE do base** (dev): no base " + json.dumps(fve["on_base"]) + "; no fine-tuned " + json.dumps(fve["on_finetuned"]) + f"; queda {r['fve_drop']:.4f}.\n\n"
        for k in ("fve_ft_with_sae_ft", "paired_features", "note"):
            if k in r:
                s += f"{k}: {json.dumps(r[k])}\n\n"
        s += f"Features com IC 95% do delta fora de zero (descoberta): {r.get('n_features_ci_excludes_zero_discovery')}\n\n"
    a = c / "accuracy_by_config.csv"
    if a.exists():
        s += "**Acurácia por configuração**\n\n" + md(pd.read_csv(a), "{:.4f}") + "\n"
    t = c / "top_changed_features_discovery.csv"
    if t.exists():
        top = pd.read_csv(t).head(top_k)
        cols = [x for x in ("feature", "contrast_base", "contrast_ft", "delta", "ci_lo", "ci_hi", "lift_error_base", "lift_error_ft", "mean_attr_base", "mean_attr_ft") if x in top]
        s += f"**Features que mais mudam (|delta| do contraste correta - erradas, IC 95% fora de zero, descoberta), top {top_k}**\n\n" + md(top[cols], "{:.4f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO_ROOT / "results" / "sae_features"))
    ap.add_argument("--names", required=True)
    ap.add_argument("--literal", default="")
    ap.add_argument("--part-c", default=None)
    ap.add_argument("--jobs", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.dir)
    names = args.names.split(",")
    s = ["# Tabelas geradas dos artefatos (sem interpretação)\n"]
    if args.jobs:
        s.append("## Jobs\n\n" + md(pd.DataFrame([x.split("=", 1) for x in args.jobs.split(",")], columns=["job", "o que"])))
    s.append("## Qualidade dos SAEs\n" + quality_section(d, names + ([args.literal] if args.literal else [])))
    for name in names:
        s.append(f"\n## Features de {name}\n")
        for kind in ("error", "probe_right_laya_wrong"):
            for view in ("pred", "corr", "state"):
                s.append(top_table(d, name, view, kind))
        s.append(attribution_section(d, name))
        s.append(bias_section(d, name))
    s.append("\n## Parte C: base x fine-tuned\n\n" + part_c_section(Path(args.part_c)) if args.part_c else "")
    out = Path(args.out or d / "REPORT_tables.md")
    out.write_text("\n".join(s))
    print(f"[report] tables -> {out}")


if __name__ == "__main__":
    main()
