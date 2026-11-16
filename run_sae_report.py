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


def counts_and_replication(d: Path, name: str) -> str:
    """Naive vs CMH counts per view/criterion, and discovery -> dev/test reproducibility that does not depend on the
    selection: Spearman of z over the features with support on both sides, and sign agreement of the top-k by |z|."""
    rows_c, rows_r = [], []
    for kind in ("error", "probe_right_laya_wrong"):
        for view in ("pred", "corr", "state"):
            f = d / name / f"feature_stats_{view}_{kind}.parquet"
            if not f.exists():
                continue
            t = pd.read_parquet(f)
            t = t[t.stratum == "all"]
            disc = t[(t.split == "discovery") & t.supported]
            g = lambda col, z, sgn: int(((disc[col] <= 0.05) & (disc[z] * sgn > 0)).sum())
            rows_c.append({"visão": view, "critério": kind, "features com suporte": len(disc), "ingênuo q<=0,05": int((disc.q_bh <= 0.05).sum()),
                           "CMH (estrato da visão) mais erro": g("cmh_q", "cmh_z", 1), "CMH menos erro": g("cmh_q", "cmh_z", -1),
                           "CMH 1ª passada mais erro": g("cmh_qp_q", "cmh_qp_z", 1), "CMH 1ª passada menos erro": g("cmh_qp_q", "cmh_qp_z", -1)})
            a = disc.set_index("feature")
            for sp in ("dev", "test"):
                b = t[(t.split == sp) & t.supported].set_index("feature")
                j = a[["cmh_z", "lift"]].join(b[["cmh_z", "lift"]], how="inner", rsuffix="_b").dropna()
                if len(j) < 10:
                    continue
                row = {"visão": view, "critério": kind, "split": sp, "features nos dois": len(j),
                       "Spearman z": j.cmh_z.corr(j.cmh_z_b, method="spearman"), "Spearman lift ingênuo": j.lift.corr(j.lift_b, method="spearman")}
                for k in (20, 100):
                    top = j.reindex(j.cmh_z.abs().sort_values(ascending=False).index).head(k)
                    row[f"sinal igual no top-{k}"] = float((np.sign(top.cmh_z) == np.sign(top.cmh_z_b)).mean())
                rows_r.append(row)
    return ("\n**Contagens: teste ingênuo × Cochran-Mantel-Haenszel** (conjunto de descoberta; PROTOCOLO emenda 3)\n\n" + md(pd.DataFrame(rows_c), "{:.0f}")
            + "\n**Reprodutibilidade descoberta -> dev/teste** (todas as features com suporte nos dois lados, sem depender da seleção)\n\n" + md(pd.DataFrame(rows_r)))


def top_table(d: Path, name: str, view: str, kind: str, k: int = 10) -> str:
    f = d / name / f"feature_stats_{view}_{kind}.parquet"
    if not f.exists():
        return ""
    t = pd.read_parquet(f)
    t = t[t.stratum == "all"]
    disc = t[(t.split == "discovery") & t.supported]
    sig = disc[(disc.cmh_q <= 0.05) & (disc.cmh_z > 0)]
    top = (sig if len(sig) else disc.dropna(subset=["cmh_p"])).sort_values("cmh_z", ascending=False).head(k)
    rows = []
    for _, r in top.iterrows():
        row = {"feature": int(r.feature), "ativas": int(r.n_active), "taxa de erro": r.rate_active, "lift ingênuo": r.lift,
               "OR CMH": r.cmh_or, "z CMH": r.cmh_z, "q CMH": r.cmh_q}
        for sp in ("dev", "test"):
            x = t[(t.split == sp) & (t.feature == r.feature)]
            row[f"z CMH {sp}"] = float(x["cmh_z"].iloc[0]) if len(x) else np.nan
            row[f"ativas {sp}"] = float(x["n_active"].iloc[0]) if len(x) else np.nan
        rows.append(row)
    return (f"\n**{view} / {kind}**: {len(sig)} features com q(CMH) <= 0,05 e mais erro (de {len(disc)} com suporte); top {k} por z do CMH "
            "(descoberta) com os mesmos números em dev e teste:\n\n" + md(pd.DataFrame(rows)))


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
        s.append(counts_and_replication(d, name))
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
