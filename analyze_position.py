"""Offline analysis of completed order experiments. No model inference or GPU use."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.position import orders


def cluster_interval(frame, column, n_boot=2000, seed=0):
    """Equal weight per decision; bootstrap whole cases including all their questions."""
    units = frame.groupby("case_id")[column].agg(["sum", "count"])
    sums, counts = units["sum"].to_numpy(), units["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(units), (n_boot, len(units)))
    samples = sums[draws].sum(1) / counts[draws].sum(1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def projection_interval(frame, n_boot=2000, seed=0):
    units = frame.groupby("case_id")[["projection_numerator", "contrast_sqnorm"]].sum()
    numer, denom = units.to_numpy().T
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(units), (n_boot, len(units)))
    estimates = numer[draws].sum(1) / denom[draws].sum(1)
    return [float(v) for v in np.quantile(estimates, [0.025, 0.975])]


def paired_layer_contrast(frame, later="enc_18", earlier="enc_17", n_boot=2000, seed=0):
    target = frame[frame.intervention_group == "all_markers"]
    columns = ["projection_numerator", "contrast_sqnorm"]
    a = target[target.layer == later].groupby("case_id")[columns].sum().sort_index()
    b = target[target.layer == earlier].groupby("case_id")[columns].sum().sort_index()
    if not a.index.equals(b.index):
        raise ValueError("paired layer contrast has different case coverage")
    an, ad = a.to_numpy().T
    bn, bd = b.to_numpy().T
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), (n_boot, len(a)))
    samples = an[draws].sum(1)/ad[draws].sum(1) - bn[draws].sum(1)/bd[draws].sum(1)
    return {"later": later, "earlier": earlier, "cases": len(a),
            "difference": float(an.sum()/ad.sum()-bn.sum()/bd.sum()),
            "ci": [float(v) for v in np.quantile(samples, [.025, .975])],
            "max_case_contrast_share": float(np.max(ad)/ad.sum())}


def behavior_metrics(records, gold_index):
    logits = np.array([r["logits"] for r in records], dtype=float)
    p = np.array([r["probabilities"] for r in records], dtype=float)
    p /= p.sum(1, keepdims=True)
    pred = logits.argmax(1)
    centered = logits - logits.mean(1, keepdims=True)
    mean_p = p.mean(0)
    k = p.shape[1]
    log_p = np.log(np.clip(p, 1e-15, 1))
    slot_counts = np.bincount([r["order"].index(int(i)) for r,i in zip(records,pred)], minlength=k)
    return {"n_orders": len(records), "k": k, "flip_any": float(len(set(pred)) > 1),
            "flip_vs_canonical": float(np.mean(pred != pred[0])),
            "distinct_choices": len(set(pred)), "canonical_accuracy": float(pred[0] == gold_index),
            "mean_order_accuracy": float(np.mean(pred == gold_index)),
            "tv_to_mean": float(np.abs(p-mean_p).sum(1).mean()/2),
            "max_probability_range": float(np.ptp(p, axis=0).max()),
            "centered_logit_rms": float(np.sqrt(np.mean((centered-centered.mean(0))**2))),
            "normalized_entropy": float(-(p*log_p).sum(1).mean()/np.log(k)),
            "official_raw_argmax_disagreements": sum(r["pred_index"] != r["official_pred_index"] for r in records),
            **{f"slot_{i}": float(slot_counts[i]/len(records)) for i in range(k)}}


def compare_conditions(original, compared):
    source = {tuple(r["order"]): r for r in original}
    changes, distances = [], []
    for row in compared:
        ref = source[tuple(row["order"])]
        p, q = (np.array(r["probabilities"], dtype=float) for r in (ref, row))
        p, q = p/p.sum(), q/q.sum()
        changes.append(int(np.argmax(ref["logits"]) != np.argmax(row["logits"])))
        distances.append(float(np.abs(p-q).sum()/2))
    return {"mean_choice_change_vs_original": float(np.mean(changes)),
            "mean_tv_vs_original": float(np.mean(distances)),
            "canonical_choice_change_vs_original": changes[0],
            "canonical_tv_vs_original": distances[0]}


def load_collection(root):
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((root/"COMPLETE.json").read_text(encoding="utf-8"))
    runtime = json.loads((root/"runtime.json").read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()
    if actual_hash != complete["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    files = sorted((root/"decisions").glob("*.json"))
    if len(files) != complete["decisions"]:
        raise ValueError("decision count mismatch")
    behavior, patches = [], []
    decisions_seen, cases_seen, patched_cases, patch_decisions = set(), set(), set(), 0
    counts, parity, noop_error, final_error = {}, [], [], []
    for file in files:
        d = json.loads(file.read_text(encoding="utf-8"))
        key = (d["case_id"], d["qid"])
        if key in decisions_seen:
            raise ValueError(f"duplicate decision {key}")
        decisions_seen.add(key)
        cases_seen.add(d["case_id"])
        meta = {"case_id": d["case_id"], "qid": d["qid"], "workflow": d["workflow"]}
        gold_index = d["semantic_ids"].index(d["gold_id"])
        variants = {}
        for row in d["behavior"]:
            variants.setdefault(row["condition"], []).append(row)
            if "parity_max_probability_error" in row:
                parity.append(row["parity_max_probability_error"])
        if set(variants) != {"original", "empty", "swapped_state", "renamed_ids"}:
            raise ValueError("missing behavioral condition")
        for condition, rows in variants.items():
            rows.sort(key=lambda r: tuple(r["order"]))
            # collection generation puts canonical first; lexicographic sort also does so.
            if rows[0]["order"] != list(range(len(rows[0]["ids"]))):
                raise ValueError("canonical reference missing")
            if len({tuple(r["order"]) for r in rows}) != len(rows):
                raise ValueError("duplicate order")
            if {tuple(r["order"]) for r in rows} != set(map(tuple, orders(len(rows[0]["ids"])))):
                raise ValueError("permutation coverage differs from protocol")
            stats = behavior_metrics(rows, gold_index)
            if condition in ("empty", "swapped_state"):
                # The recipient's gold is not ground truth for an empty or donor state.
                stats["canonical_accuracy"] = stats["mean_order_accuracy"] = np.nan
            behavior.append({**meta, "condition": condition, **stats,
                             **compare_conditions(variants["original"], rows)})
            counts[condition] = counts.get(condition, 0) + len(rows)
        if d["patches"]:
            patch_decisions += 1
            patched_cases.add(d["case_id"])
            if not d.get("activations") or not (root/d["activations"]).is_file():
                raise ValueError("missing selected activation file")
            expected_rows = len(runtime["layers"])*2*(4+manifest["args"]["n_null"])
            if len(d["patches"]) != expected_rows:
                raise ValueError("incomplete intervention coverage")
        clean_orders = {tuple(r["order"]): r for r in variants["original"]}
        seen = set()
        for row in d["patches"]:
            identity = (row["layer"], row["intervention"], tuple(row["donor_order"]))
            if identity in seen:
                raise ValueError("duplicate intervention")
            seen.add(identity)
            p, r, donor = (np.array(row[k]) for k in ("patched_logits", "recipient_logits", "donor_logits"))
            if row["donor_logits"] != clean_orders[tuple(row["donor_order"])]["logits"] or \
                    row["recipient_logits"] != clean_orders[tuple(row["recipient_order"])]["logits"]:
                raise ValueError("intervention reference differs from clean order")
            if row["intervention"] == "same_order_noop":
                noop_error.append(float(np.max(np.abs(p-r))))
            if row["layer"] == "head_2" and row["intervention"] == "all_markers":
                final_error.append(float(np.max(np.abs(p-donor))))
            pc, rc, dc = (v-v.mean() for v in (p,r,donor))
            denominator = float(np.sum((dc-rc)**2))
            numerator = float(np.dot(pc-rc, dc-rc))
            if not np.isclose(denominator, row["contrast_sqnorm"], rtol=1e-7, atol=1e-9):
                raise ValueError("stored denominator differs")
            if not np.isclose(numerator, row["projection_numerator"], rtol=1e-7, atol=1e-9):
                raise ValueError("stored numerator differs")
            name = row["intervention"]
            patches.append({**meta, **row, "intervention_group": "norm_matched_noise" if name.startswith("norm_matched_noise_") else name,
                            "pair_flip": int(donor.argmax() != r.argmax()),
                            "donor_choice_recovered": int(p.argmax() == donor.argmax()),
                            "recipient_choice_changed": int(p.argmax() != r.argmax()),
                            "distance_improvement": float(np.linalg.norm(dc-rc)-np.linalg.norm(pc-dc))})
    if cases_seen != set(manifest["case_ids"]) or len(cases_seen) != complete["cases"]:
        raise ValueError("case coverage differs from manifest")
    if patched_cases != set(manifest["case_ids"][:manifest["args"]["patch_cases"]]):
        raise ValueError("patched case selection differs from protocol hash ordering")
    audit = {"cases": len(cases_seen), "decisions": len(files), "behavior_forward_rows": counts,
             "patched_cases": len(patched_cases), "patched_decisions": patch_decisions,
             "patch_forward_rows": len(patches), "parity_max_probability_error": max(parity),
             "noop_max_logit_error": max(noop_error), "final_patch_max_logit_error": max(final_error),
             "manifest_sha256": actual_hash}
    return pd.DataFrame(behavior), pd.DataFrame(patches), audit


def summarize(behavior, patches, out, n_boot):
    results = []
    for condition, group in behavior.groupby("condition"):
        row = {"condition": condition, "cases": group.case_id.nunique(), "decisions": len(group)}
        for metric in ("flip_any", "flip_vs_canonical", "tv_to_mean", "max_probability_range",
                       "centered_logit_rms", "normalized_entropy", "canonical_accuracy", "mean_order_accuracy",
                       "mean_choice_change_vs_original", "mean_tv_vs_original",
                       "canonical_choice_change_vs_original", "canonical_tv_vs_original"):
            row[metric] = float(group[metric].mean()) if group[metric].notna().any() else None
            # Recipient/donor dependencies invalidate an independent-case interval for
            # cross-condition swap effects. Keep those contrasts descriptive only.
            row[metric+"_ci"] = (None if row[metric] is None or
                                     (condition == "swapped_state" and "vs_original" in metric)
                                     else cluster_interval(group, metric, n_boot))
        results.append(row)
    summaries = pd.DataFrame(results)
    summaries.to_json(out/"behavior_summary.json", orient="records", indent=2)
    subgroup = behavior.groupby(["condition", "workflow", "qid", "k"]).mean(numeric_only=True).reset_index()
    subgroup.to_csv(out/"behavior_by_question.csv", index=False)
    contrasts = []
    original = behavior[behavior.condition == "original"].set_index(["case_id", "qid"])
    for condition in ("empty", "swapped_state", "renamed_ids"):
        compared = behavior[behavior.condition == condition].set_index(["case_id", "qid"])
        for metric in ("flip_any", "tv_to_mean", "centered_logit_rms", "normalized_entropy"):
            delta = (compared[metric]-original[metric]).rename("delta").reset_index()
            contrasts.append({"condition": condition, "metric": metric, "difference": float(delta.delta.mean()),
                              "ci": None if condition == "swapped_state" else cluster_interval(delta, "delta", n_boot),
                              "note": ("Aggregate difference is structurally zero: complete permutation of states within question bundles."
                                       if condition == "swapped_state" else "paired by case and question")})
    (out/"behavior_paired_contrasts.json").write_text(json.dumps(contrasts, indent=2), encoding="utf-8")
    patch_summary = []
    # Null directions are averaged within each decision/direction/layer before aggregation.
    keys = ["case_id", "qid", "workflow", "layer", "intervention_group"]
    patches["direction"] = patches.donor_order.map(lambda x: "canonical_to_rotated" if x[0] == 0 else "rotated_to_canonical")
    keys += ["direction"]
    numeric = ["projection_numerator", "contrast_sqnorm", "transfer_ratio", "change_norm", "orthogonal_norm",
               "distance_improvement", "pair_flip", "donor_choice_recovered", "recipient_choice_changed"]
    averaged = patches.groupby(keys, dropna=False)[numeric].mean().reset_index()
    contrast = paired_layer_contrast(averaged, n_boot=n_boot)
    (out/"enc18_minus_enc17.json").write_text(json.dumps(contrast, indent=2), encoding="utf-8")
    workflow_table = averaged.groupby(["workflow", "layer", "intervention_group"]).agg(
        numerator=("projection_numerator", "sum"), denominator=("contrast_sqnorm", "sum"),
        cases=("case_id", "nunique"), rows=("qid", "size"), median_ratio=("transfer_ratio", "median"),
        change_norm=("change_norm", "mean"), orthogonal_norm=("orthogonal_norm", "mean")).reset_index()
    workflow_table["projection"] = workflow_table.numerator/workflow_table.denominator
    workflow_table.to_csv(out/"patch_by_workflow.csv", index=False)
    for (layer, intervention), group in averaged.groupby(["layer", "intervention_group"]):
        row = {"layer": layer, "intervention": intervention, "cases": group.case_id.nunique(),
               "decisions": group[["case_id", "qid"]].drop_duplicates().shape[0],
               "projection": float(group.projection_numerator.sum()/group.contrast_sqnorm.sum()),
               "projection_ci": projection_interval(group, n_boot),
               "median_ratio": float(group.transfer_ratio.median()),
               "ratio_q05": float(group.transfer_ratio.quantile(.05)),
               "ratio_q95": float(group.transfer_ratio.quantile(.95)),
               "change_norm": float(group.change_norm.mean()),
               "distance_improvement": float(group.distance_improvement.mean()),
               "denominator_min": float(group.contrast_sqnorm.min()),
               "denominator_median": float(group.contrast_sqnorm.median()),
               "ratio_undefined": int(group.transfer_ratio.isna().sum()),
               "donor_choice_recovered": float(group.donor_choice_recovered.mean())}
        flipped = group[group.pair_flip == 1]
        row["flip_pairs"] = len(flipped)
        row["donor_choice_recovered_on_flips"] = float(flipped.donor_choice_recovered.mean()) if len(flipped) else None
        patch_summary.append(row)
    (out/"patch_summary.json").write_text(json.dumps(patch_summary, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(patch_summary).to_csv(out/"patch_summary.csv", index=False)
    averaged.to_csv(out/"patch_by_decision.csv", index=False)
    return results, patch_summary


def plots(results, patch_summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    labels = {"original": "Original", "empty": "Sem estado", "swapped_state": "Estado trocado", "renamed_ids": "IDs renomeados"}
    # Swapping states is a full within-question permutation, so its aggregate order
    # sensitivity is identical by construction; do not plot it as independent evidence.
    results = sorted([r for r in results if r["condition"] != "swapped_state"],
                     key=lambda r: ["original", "empty", "renamed_ids"].index(r["condition"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    for ax, metric, title in zip(axes, ["flip_any", "tv_to_mean"],
                               ["Decisões cuja escolha muda entre ordens", "Variação das probabilidades entre ordens"]):
        values = np.array([r[metric] for r in results])*100
        bounds = np.array([r[metric+"_ci"] for r in results])*100
        ax.bar(range(len(results)), values, color=["#236B8E", "#A3BDC7", "#D89458"])
        ax.errorbar(range(len(results)), values, yerr=[values-bounds[:,0], bounds[:,1]-values], fmt="none", color="#222222", capsize=4)
        ax.set_xticks(range(len(results)), [labels[r["condition"]] for r in results], rotation=15)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("%" if metric == "flip_any" else "Distância TV média × 100")
    fig.suptitle("Laya multilíngue · descoberta · 1.020 casos", fontsize=14)
    fig.savefig(out/"behavior.png", dpi=180)
    plt.close(fig)
    table = pd.DataFrame(patch_summary)
    layers = sorted(table.layer.unique(), key=lambda s: (0,int(s[4:])) if s.startswith("enc_") and s[4:].isdigit()
                    else ((1,0) if s == "enc_final" else (2,int(s[5:]))))
    fig, ax = plt.subplots(figsize=(11.8, 5.1), layout="constrained")
    display = {"all_markers": ("Todos os marcadores", "#236B8E"), "single_marker": ("Um marcador", "#D89458"),
               "norm_matched_noise": ("Ruído de mesma norma", "#777777")}
    for intervention, (label,color) in display.items():
        rows = table[table.intervention == intervention].set_index("layer").loc[layers]
        x = np.arange(len(layers))
        ax.plot(x, rows.projection, label=label, color=color, marker="o", markersize=3)
        bounds = np.array(rows.projection_ci.tolist())
        ax.fill_between(x, bounds[:,0], bounds[:,1], color=color, alpha=.15)
    ax.axhline(0, color="#bbbbbb", linewidth=1)
    ax.axhline(1, color="#bbbbbb", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(layers)), layers, rotation=60, fontsize=8)
    ax.set_ylabel("Projeção agregada na mudança da execução doadora")
    ax.set_title("Transferência causal do efeito da ordem por camada\nIC95% por caso · camada final é controle positivo, não origem do efeito", fontsize=12)
    ax.legend(loc="best")
    fig.savefig(out/"patching.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    behavior, patches, audit = load_collection(args.input)
    (args.out/"audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    behavior.to_csv(args.out/"behavior_by_decision.csv", index=False)
    results, patch_summary = summarize(behavior, patches, args.out, args.bootstrap)
    plots(results, patch_summary, args.out)
    print(json.dumps(audit, indent=2))
    print(pd.DataFrame(results)[["condition", "flip_any", "flip_vs_canonical", "tv_to_mean", "canonical_accuracy", "mean_order_accuracy"]].to_string(index=False))
    table = pd.DataFrame(patch_summary)
    print(table[(table.intervention == "all_markers") | ((table.intervention == "norm_matched_noise") & table.layer.isin(["enc_08", "enc_17", "head_1"]))][["layer", "intervention", "projection", "projection_ci", "median_ratio", "donor_choice_recovered_on_flips"]].to_string(index=False))


if __name__ == "__main__":
    main()
