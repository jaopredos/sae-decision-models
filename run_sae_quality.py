"""Stage 4, step 3: quality of a trained SAE, on the dev split and on held-out generic text.

Usage: python run_sae_quality.py --sae checkpoints/sae/ml_enc17_k32/sae.pt --layer enc_17 --name ml_enc17_k32
           [--device cuda] [--domain-dir data/activations/tokens/multilingual]
           [--corpus data/corpus/text.parquet] [--out results/sae_features] [--max-dev-cases N]

Measures (results/sae_features/PROTOCOLO.md, including emenda 1):
  * FVE and L0 on the typed-decisions dev prompts by token group (all non-massive; option markers;
    option text; question; state; and, as diagnostics, special tokens, massive tokens, and every
    non-padding token = the literal protocol) and on held-out generic text by source;
  * dead latents: from the training counter (no firing in the last 5M tokens) and on the evaluation tokens;
  * downstream fidelity on the dev decisions: replace the layer's activation by the SAE reconstruction
    (all non-massive tokens; only option markers; and, as a diagnostic, also the massive ones), or by the
    mean activation / zeros (scale), and compare Laya's answers: accuracy against the gold label, change in
    accuracy, argmax flips, KL(original || modified).
Appends rows (sae, section, group, metric, value) to <out>/sae_quality.csv and prints the flags of the
protocol (FVE < 0.80; markers more than 0.10 below; dead > 20%; accuracy drop > 3 p.p.; KL > 0.10 nat).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import corpus as C
from src.activations import (ROLE_CLS, ROLE_END_SEP, ROLE_HEAD_SEP, ROLE_MARKER, ROLE_OPTION, ROLE_QUESTION,
                             ROLE_STATE, layer_modules, layer_output_fn, load_agent, option_ids)
from src.datasets import load_typed_cases, typed_dev_ids
from src.models import normalize_answers
from src.sae import load_sae, pooled_fve
from src.sae_data import SPECIAL_ROLES, load_tokens, massive_threshold, token_is_dev
from src.sae_intervention import SAEIntervention

REPO_ROOT = Path(__file__).resolve().parent
ROLE_GROUPS = {"markers": [ROLE_MARKER], "option_text": [ROLE_OPTION], "question": [ROLE_QUESTION],
               "state": [ROLE_STATE], "special": list(SPECIAL_ROLES)}


@torch.no_grad()
def group_stats(sae, x, chunk=50_000):
    parts = [sae.fve_stats(x[a:a + chunk]) for a in range(0, len(x), chunk)]
    return pooled_fve(parts), float(np.mean([p["l0"] for p in parts]))


@torch.no_grad()
def fired(sae, x, chunk=50_000):
    f = torch.zeros(sae.n_latents, dtype=torch.bool, device=x.device)
    for a in range(0, len(x), chunk):
        f |= (sae.encode(x[a:a + chunk]) > 0).any(0)
    return f


def predict_probs(agent, cases, qorder):
    """{(case_id, qid): np.array of probabilities in option order} from agent.predict."""
    out = {}
    for case in cases:
        ans = normalize_answers(agent.predict(case["state"], case["questions"])["answers"])
        for qid, a in ans.items():
            out[(case["case_id"], qid)] = np.array([a["probabilities"][o] for o in qorder[(case["case_id"], qid)]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--domain-dir", default=str(REPO_ROOT / "data" / "activations" / "tokens" / "multilingual"))
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "corpus" / "text.parquet"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "sae_features"))
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--max-dev-cases", type=int, default=0)
    ap.add_argument("--rows-per-fwd", type=int, default=64)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--subfolder", default="__default__")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER
    repo = args.repo or LAYA_REPO
    sub = LAYA_SUBFOLDER if args.subfolder == "__default__" else (None if args.subfolder in ("", "none") else args.subfolder)
    agent = load_agent(args.device, repo, sub, args.revision or LAYA_REVISION)
    device = agent.device
    sae = load_sae(args.sae, str(device)).eval()
    rows_out = []

    def add(section, group, metric, value):
        rows_out.append({"sae": args.name, "layer": args.layer, "section": section, "group": group,
                         "metric": metric, "value": float(value)})

    thr_sae = float(sae.massive_threshold)

    # ---- 1. FVE / L0 on dev prompts, by token group ------------------------------------------------
    acts, ti, rows = load_tokens(Path(args.domain_dir) / "train", args.layer)
    dev = token_is_dev(ti, rows)
    diag_thr, med = massive_threshold(acts, ti, rows, 5.0)  # the rule's threshold, also for the literal SAE
    idx = np.flatnonzero(dev)
    x = torch.from_numpy(np.concatenate([np.asarray(acts[idx[a:a + 200_000]]) for a in range(0, len(idx), 200_000)])).to(device)
    role = ti["role"].to_numpy()[idx]
    norm = x.float().norm(dim=1).cpu().numpy()
    is_mass = norm > diag_thr
    groups = {"all_non_massive": ~is_mass, "all_tokens_literal": np.ones(len(x), bool), "massive": is_mass}
    for g, roles in ROLE_GROUPS.items():
        groups[g] = np.isin(role, roles) & ~is_mass if g != "special" else np.isin(role, roles)
    all_fired = torch.zeros(sae.n_latents, dtype=torch.bool, device=device)
    for g, m in groups.items():
        if m.sum() < 10:
            continue
        xg = x[torch.from_numpy(m).to(device)]
        fve, l0 = group_stats(sae, xg)
        add("domain_dev", g, "fve", fve); add("domain_dev", g, "l0", l0); add("domain_dev", g, "n_tokens", int(m.sum()))
        if g == "all_non_massive":
            all_fired |= fired(sae, xg)
    add("domain_dev", "markers", "frac_massive", float(is_mass[role == ROLE_MARKER].mean()) if (role == ROLE_MARKER).any() else 0.0)
    for r, name in ((ROLE_CLS, "cls"), (ROLE_QUESTION, "question"), (ROLE_MARKER, "marker"), (ROLE_OPTION, "option_text"),
                    (ROLE_HEAD_SEP, "head_sep"), (ROLE_STATE, "state"), (ROLE_END_SEP, "end_sep")):
        if (role == r).any():
            add("domain_dev", name, "frac_massive", float(is_mass[role == r].mean()))

    # ---- 2. held-out generic text, by source -------------------------------------------------------
    df = pd.read_parquet(args.corpus)
    val_store = C.build_store(agent.tok, df, {}, val=True, seed=0)
    fwd = layer_output_fn(agent, args.layer)
    it = C.batches(val_store, args.rows_per_fwd, seed=1)
    xs, ss, n = [], [], 0
    while n < args.val_tokens:
        ids, mask, source = it.__next__()
        h = fwd(torch.from_numpy(ids).long(), torch.from_numpy(mask))
        m = torch.from_numpy(mask).to(h.device)
        xs.append(h[m].to(torch.float16)); ss.append(torch.from_numpy(np.repeat(source, mask.sum(1))).to(h.device))
        n += xs[-1].shape[0]
    xg_all, sg = torch.cat(xs)[: args.val_tokens].float(), torch.cat(ss)[: args.val_tokens]
    mass_g = xg_all.norm(dim=1) > diag_thr
    add("generic_val", "all_tokens_literal", "frac_massive", mass_g.float().mean().item())
    for label, sel in [("all_non_massive", ~mass_g)] + [(s, (sg == i) & ~mass_g) for i, s in enumerate(val_store.sources)]:
        if int(sel.sum()) < 10:
            continue
        fve, l0 = group_stats(sae, xg_all[sel])
        add("generic_val", label, "fve", fve); add("generic_val", label, "l0", l0); add("generic_val", label, "n_tokens", int(sel.sum()))
        if label == "all_non_massive":
            all_fired |= fired(sae, xg_all[sel])
    del xs, ss, xg_all

    # ---- 3. dead latents ---------------------------------------------------------------------------
    add("dead", "training_counter", "frac_dead", (sae.tokens_since_fired > 5_000_000).float().mean().item())
    add("dead", "eval_tokens", "frac_never_fired", 1.0 - all_fired.float().mean().item())

    # ---- 4. downstream fidelity on the dev decisions ----------------------------------------------
    dev_ids = typed_dev_ids()
    cases = [c for c in load_typed_cases("train") if c["case_id"] in dev_ids]
    if args.max_dev_cases:
        cases = cases[: args.max_dev_cases]
    qorder = {}
    for case in cases:
        for qid, q in case["questions"].items():
            qorder[(case["case_id"], qid)] = option_ids(agent._to_internal(q))
    gold = {(c["case_id"], qid): str(c["gold"][qid]["label"]) for c in cases for qid in c["questions"]}
    layer_mod = dict(layer_modules(agent.model))[args.layer]
    mean_vec = x[torch.from_numpy(~is_mass).to(device)].float().mean(0)
    mask_id = agent.tok.mask_token_id
    cur_ids = {}
    pre = agent.model.register_forward_pre_hook(lambda m, a: cur_ids.__setitem__("ids", a[0]))
    markers_only = lambda h: cur_ids["ids"].to(h.device) == mask_id

    base = predict_probs(agent, cases, qorder)
    keys = list(base)
    acc0 = float(np.mean([qorder[k][int(np.argmax(base[k]))] == gold[k] for k in keys]))
    add("downstream", "original", "accuracy", acc0); add("downstream", "original", "n_decisions", len(keys))
    variants = {
        "replace_non_massive_all": dict(mode="replace", position_mask=None, skip_massive=True),
        "replace_non_massive_markers": dict(mode="replace", position_mask=markers_only, skip_massive=True),
        "replace_all_incl_massive": dict(mode="replace", position_mask=None, skip_massive=False),
        "mean_non_massive_all": dict(mode="mean", position_mask=None, skip_massive=True, replacement=mean_vec),
        "mean_non_massive_markers": dict(mode="mean", position_mask=markers_only, skip_massive=True, replacement=mean_vec),
        "zero_non_massive_all": dict(mode="zero", position_mask=None, skip_massive=True),
        "zero_non_massive_markers": dict(mode="zero", position_mask=markers_only, skip_massive=True),
    }
    if not np.isfinite(thr_sae):
        # literal SAE: its own "non massive" is everything; use the rule's threshold for the massive-skipping variants
        sae.massive_threshold.fill_(diag_thr)
    for name, kw in variants.items():
        if name == "replace_all_incl_massive" and not np.isfinite(thr_sae):
            sae.massive_threshold.fill_(float("inf"))
        with SAEIntervention(sae, layer_mod, **kw):
            mod = predict_probs(agent, cases, qorder)
        if name == "replace_all_incl_massive" and not np.isfinite(thr_sae):
            sae.massive_threshold.fill_(diag_thr)
        acc = float(np.mean([qorder[k][int(np.argmax(mod[k]))] == gold[k] for k in keys]))
        kl = float(np.mean([np.sum(base[k] * (np.log(np.clip(base[k], 1e-9, None)) - np.log(np.clip(mod[k], 1e-9, None)))) for k in keys]))
        flips = float(np.mean([np.argmax(base[k]) != np.argmax(mod[k]) for k in keys]))
        add("downstream", name, "accuracy", acc); add("downstream", name, "delta_accuracy_pp", 100 * (acc - acc0))
        add("downstream", name, "kl_nat", kl); add("downstream", name, "argmax_flip_rate", flips)
        print(f"[quality] {name}: acc {acc:.3f} (delta {100 * (acc - acc0):+.1f} pp) KL {kl:.4f} flips {flips:.3f}", flush=True)
    pre.remove()

    # ---- output and flags --------------------------------------------------------------------------
    q = pd.DataFrame(rows_out)
    csv = Path(args.out) / "sae_quality.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_csv(csv), q]).to_csv(csv, index=False) if csv.exists() else q.to_csv(csv, index=False)
    get = lambda s, g, m: float(q[(q.section == s) & (q.group == g) & (q.metric == m)]["value"].iloc[0])
    flags = []
    fve_all = get("domain_dev", "all_non_massive", "fve")
    if fve_all < 0.80:
        flags.append(f"FVE (dev, tokens não massivos) {fve_all:.3f} < 0,80")
    if (q.group == "markers").any() and fve_all - get("domain_dev", "markers", "fve") > 0.10:
        flags.append(f"FVE nos marcadores {get('domain_dev', 'markers', 'fve'):.3f} está mais de 0,10 abaixo da geral {fve_all:.3f}")
    if get("domain_dev", "markers", "frac_massive") > 0.01:
        flags.append(f"{100 * get('domain_dev', 'markers', 'frac_massive'):.1f}% dos marcadores são massivos (> 1%)")
    if get("dead", "training_counter", "frac_dead") > 0.20:
        flags.append(f"latentes mortas {100 * get('dead', 'training_counter', 'frac_dead'):.1f}% > 20%")
    for v in ("replace_non_massive_all", "replace_non_massive_markers"):
        if get("downstream", v, "delta_accuracy_pp") < -3.0:
            flags.append(f"{v}: acurácia cai {-get('downstream', v, 'delta_accuracy_pp'):.1f} p.p. (> 3)")
        if get("downstream", v, "kl_nat") > 0.10:
            flags.append(f"{v}: KL {get('downstream', v, 'kl_nat'):.3f} nat > 0,10")
    print("[quality] SINALIZADORES:", flags if flags else "nenhum", flush=True)
    (Path(args.out) / f"sae_quality_flags_{args.name}.json").write_text(json.dumps({"sae": args.name, "flags": flags}, indent=2))


if __name__ == "__main__":
    main()
