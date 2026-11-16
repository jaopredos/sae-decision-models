"""Gradient attribution of Laya's errors to SAE features (stage 5, first order).

For a wrong decision let  delta = logit[correct] - logit[predicted]  (raw logits, before temperature). At
token t of the layer the SAE reads h_t, finds latents z_t and writes decode(z_t) = (z_t W_dec + b_dec) / scale
back in model units, so feature i moves the layer output along d_i = W_dec[i] / scale. First order:

    a[t, i] = z[t, i] * (d_i . g_t),      g_t = d delta / d h_t      (gradient w.r.t. the layer output)

a > 0: the feature pushes toward the correct answer, a < 0: toward the wrong one. Summing over i gives
(decode(z_t) - b_dec/scale) . g_t, so the rest of h_t . g_t is the reconstruction error and the bias.
Massive tokens are outside the SAE (see src/sae.py) and get no attribution. This is correlational evidence
about direction and size of a linear effect, not a causal ablation, and ignores the SAE error term.

`feature_attribution` is a pure function of (h, g); `attribute_case` runs it on a real Laya forward in fp32.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from src.activations import ROLE_MARKER, ROLE_STATE, capture, layer_modules, option_ids, token_roles
from src.sae import TopKSAE


@torch.no_grad()
def feature_attribution(sae: TopKSAE, h: torch.Tensor, g: torch.Tensor) -> Dict[str, torch.Tensor]:
    """h, g [T, H] (layer output and its gradient). Returns a [T, F] (zero at massive tokens), the mask of
    massive tokens, and the scalars total_attr = sum(a) and total_hg = sum over non-massive tokens of h.g."""
    h, g = h.float(), g.float()
    massive = sae.is_massive(h)
    z = sae.encode(h)
    z = z.masked_fill(massive[:, None], 0.0)
    proj = (g @ sae.W_dec.t()) / sae.scale                      # d_i . g_t for every i
    a = z * proj
    hg = (h * g).sum(-1).masked_fill(massive, 0.0).sum()
    return {"a": a, "massive": massive, "total_attr": a.sum(), "total_hg": hg}


def attribute_case(agent, sae: TopKSAE, label: str, case: Dict) -> List[Dict]:
    """Attribution of every WRONG decision of one case (wrong = official bf16 `predict` != gold label).

    Per wrong decision returns float16 vectors [F]: at the predicted-option marker, at the correct-option
    marker, summed over the state tokens, summed over every other non-massive token, plus delta, the fp32
    prediction (to count disagreements with the official one) and the explained fraction total_attr/total_hg."""
    state, questions = case["state"], case["questions"]
    layer = dict(layer_modules(agent.model))[label]
    with capture(agent, []) as cap:
        ref = agent.predict(state, questions)["answers"]
    ids, internal, items = cap.encode_calls[0]
    args = cap.forward_calls[0][0]
    input_ids, attention_mask, marker_pos, marker_mask, qtype = [a.to(agent.device) for a in args[:5]]

    wrong, info = [], {}
    for j, qid in enumerate(ids):
        opts = option_ids(internal[qid])
        a = ref[qid]
        probs = (a["probabilities"] if "probabilities" in a else {"false": 1 - a["noul"], "true": a["noul"]})
        pred = max(probs, key=probs.get)
        gold = str(case["gold"][qid]["label"])
        if pred != gold:
            wrong.append(j)
            info[j] = (opts.index(pred), opts.index(gold), qid)
    if not wrong:
        return []

    holder = {}

    def keep(_m, _i, output):
        out = output[0] if isinstance(output, tuple) else output
        out.retain_grad()
        holder["h"] = out

    handle = layer.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            logits, _ = agent.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            delta = torch.stack([logits[j, info[j][1]] - logits[j, info[j][0]] for j in wrong])
            delta.sum().backward()
    finally:
        handle.remove()
    h_all, g_all = holder["h"].detach(), holder["h"].grad
    cls_id, sep_id = agent.tok.cls_token_id, agent.tok.sep_token_id
    out = []
    for k, j in enumerate(wrong):
        n = len(items[j]["ids"])
        roles = token_roles(items[j]["ids"], items[j]["markers"], cls_id, sep_id)
        res = feature_attribution(sae, h_all[j, :n], g_all[j, :n])
        a = res["a"]
        pi, gi, qid = info[j]
        mk = items[j]["markers"]
        other = torch.from_numpy(~np.isin(np.arange(n), [mk[pi], mk[gi]]) & (roles != ROLE_STATE)).to(a.device)
        state = torch.from_numpy(roles == ROLE_STATE).to(a.device)
        fp32_pred = int(logits[j, : len(mk)].argmax())
        out.append({"case_id": case["case_id"], "question_id": qid, "pred_idx": pi, "gold_idx": gi,
                    "delta": float(delta[k].detach()), "fp32_pred_idx": fp32_pred,
                    "total_attr": float(res["total_attr"]), "total_hg": float(res["total_hg"]),
                    "a_pred": a[mk[pi]].half().cpu().numpy(), "a_corr": a[mk[gi]].half().cpu().numpy(),
                    "a_state": a[state].sum(0).half().cpu().numpy(), "a_other": a[other].sum(0).half().cpu().numpy(),
                    "marker_pred_massive": bool(res["massive"][mk[pi]]), "marker_corr_massive": bool(res["massive"][mk[gi]])})
    return out
