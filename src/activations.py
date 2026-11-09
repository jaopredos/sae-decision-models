"""Per-layer activations at Laya's option markers, captured from inside `agent.predict()`.

Nothing about the input is rebuilt here. Each case goes through `agent.predict(state, questions)`
twice: once untouched (the reference answer) and once with forward hooks on every layer plus a
spy on `Agent._encode_state` and on the `DecisionModel` forward. The hooks see the exact tensors
the official forward computes; the spy hands back the sequence layout (`markers`, option and
truncation stats) that `laya.common.build_sequence` produced, and the `marker_pos` the forward was
given. Vectors are then read at those marker positions.

Parity, checked on every case:
  * logits recomputed as `scorer(head_2[markers])` (same autocast) vs. the logits the forward
    returned;
  * those logits pushed through the agent's own `_decode_answers` (temperature, softmax, rounding)
    vs. the answers of the hooked `predict` and of the un-hooked one. Hooks switch off laya's CPU
    fast paths (`_cpu_window_encoder`, `_cpu_head`), so the un-hooked comparison is what shows that
    the captured path is the one `predict` normally runs.

Layer labels: `emb` (encoder.embeddings), `enc_01..enc_NN` (each encoder layer, before
`final_norm`), `enc_final` (encoder.final_norm, what the head reads), `head_1..head_M` (the
decision head's TransformerEncoder layers).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.models import normalize_answers

LAYA_REPO = "convaiinnovations/laya"
LAYA_SUBFOLDER = "multilingual"
LAYA_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"

OPTION_TOKEN_CAP = 48  # laya.common.build_head truncates every rendered option to this many tokens


def load_agent(device: str | None = None):
    import laya

    agent = laya.load(LAYA_REPO, subfolder=LAYA_SUBFOLDER, revision=LAYA_REVISION, device=device)
    agent.model.eval()
    return agent


def layer_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    """(label, module) for every layer whose output is recorded, read from the loaded model."""
    enc = model.encoder
    n_layers = enc.config.num_hidden_layers
    assert len(enc.layers) == n_layers, (len(enc.layers), n_layers)
    assert model.head is not None, "checkpoint has no decision head layers"
    out = [("emb", enc.embeddings)]
    out += [(f"enc_{i + 1:02d}", layer) for i, layer in enumerate(enc.layers)]
    out.append(("enc_final", enc.final_norm))
    out += [(f"head_{i + 1}", layer) for i, layer in enumerate(model.head.layers)]
    return out


def option_ids(q_internal: Dict) -> List[str]:
    """Option ids in marker order (laya.common.render_options order, no option_order given)."""
    t, crit = q_internal["t"], q_internal["crit"]
    if t == "choice":
        return [str(k) for k in crit.keys()]
    if t == "score":
        return [str(i) for i in range(len(crit))]
    return ["false", "true"]


class _Capture:
    def __init__(self):
        self.outputs: Dict[str, torch.Tensor] = {}
        self.forward_calls: List[Tuple[tuple, Any]] = []
        self.encode_calls: List[Tuple[List[str], Dict, List[Dict]]] = []


@contextmanager
def capture(agent, modules: List[Tuple[str, torch.nn.Module]]):
    cap = _Capture()
    handles = []

    def keep(label):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            cap.outputs[label] = out.detach()
        return hook

    for label, module in modules:
        handles.append(module.register_forward_hook(keep(label)))
    handles.append(agent.model.register_forward_hook(
        lambda _m, args, output: cap.forward_calls.append((args, output))))

    orig_encode = agent._encode_state

    def encode_spy(state, ids, internal, *args, **kwargs):
        items = orig_encode(state, ids, internal, *args, **kwargs)
        cap.encode_calls.append((list(ids), internal, items))
        return items

    agent._encode_state = encode_spy
    try:
        yield cap
    finally:
        del agent._encode_state  # back to the class method
        for h in handles:
            h.remove()


def _probs_vector(answers: Dict[str, Any], qid: str, opts: List[str]) -> np.ndarray:
    probs = normalize_answers({qid: answers[qid]})[qid]["probabilities"]
    assert set(probs) == set(opts), (qid, sorted(probs), opts)
    return np.array([probs[o] for o in opts], dtype=np.float64)


def _options_over_cap(agent, q_internal: Dict) -> int:
    from laya.common import encode_text, render_options

    mask = agent.tok.mask_token
    return sum(
        len(encode_text(agent.tok, " " + o.replace(mask, " "), add_special_tokens=False)["input_ids"])
        > OPTION_TOKEN_CAP
        for o in render_options(q_internal)
    )


@torch.no_grad()
def extract_case(agent, modules, case: Dict) -> Dict[str, Any]:
    """Run one case; returns per-decision records, per-layer marker vectors and parity numbers."""
    from laya.agent import _amp_context

    state, questions = case["state"], case["questions"]
    ref = agent.predict(state, questions)["answers"]
    with capture(agent, modules) as cap:
        hooked = agent.predict(state, questions)["answers"]

    assert len(cap.forward_calls) == 1 and len(cap.encode_calls) == 1, \
        (len(cap.forward_calls), len(cap.encode_calls))
    ids, internal, items = cap.encode_calls[0]
    (input_ids, attention_mask, marker_pos, marker_mask, qtype), (logits_fwd, act_fwd) = \
        cap.forward_calls[0][0][:5], cap.forward_calls[0][1]
    rows = input_ids.shape[0]
    assert rows == len(ids) == len(items)
    for label, _ in modules:
        out = cap.outputs[label]
        assert out.ndim == 3 and out.shape[:2] == input_ids.shape, (label, tuple(out.shape))

    def at_markers(h):
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        return torch.gather(h, 1, idx)

    head_last = modules[-1][0]
    amp_on = agent._amp_enabled_for(rows)
    with _amp_context(agent.device, agent.dtype, amp_on):
        logits_re = agent.model.scorer(at_markers(cap.outputs[head_last])).squeeze(-1).float()
    logits_re = logits_re.masked_fill(~marker_mask, -1e4)
    act_np = torch.softmax(act_fwd.float(), -1).cpu().numpy()
    answers_re = agent._decode_answers(logits_re.cpu().numpy(), act_np, items, ids, internal, 0)

    vectors: Dict[str, List[np.ndarray]] = {label: [] for label, _ in modules}
    gathered = {label: at_markers(cap.outputs[label]).float().cpu() for label, _ in modules}
    decisions, markers, parity = [], [], []
    for j, qid in enumerate(ids):
        q = internal[qid]
        opts = option_ids(q)
        k = len(items[j]["markers"])
        assert k == len(opts) == int(marker_mask[j].sum()), (qid, k, opts)
        for label, _ in modules:
            vec = gathered[label][j, :k].numpy().astype(np.float16)
            if not np.isfinite(vec).all():
                raise FloatingPointError(f"non-finite fp16 activation: {case['case_id']} {qid} {label}")
            vectors[label].append(vec)

        p_ref = _probs_vector(ref, qid, opts)
        p_hooked = _probs_vector(hooked, qid, opts)
        p_re = _probs_vector(answers_re, qid, opts)
        lf, lr = logits_fwd[j, :k].float().cpu(), logits_re[j, :k].cpu()
        # recomputed from the fp16 vectors that are actually saved (informational)
        with _amp_context(agent.device, agent.dtype, amp_on):
            v16 = torch.from_numpy(vectors[head_last][-1]).to(agent.device, torch.float32)
            l16 = agent.model.scorer(v16).squeeze(-1).float().cpu()
        top = np.sort(p_re)[::-1]
        parity.append({
            "case_id": case["case_id"], "question_id": qid,
            "logit_absdiff_recomputed_vs_forward": float((lr - lf).abs().max()),
            "logit_absdiff_fp16saved_vs_forward": float((l16 - lf).abs().max()),
            "prob_absdiff_recomputed_vs_hooked": float(np.abs(p_re - p_hooked).max()),
            "prob_absdiff_recomputed_vs_ref": float(np.abs(p_re - p_ref).max()),
            "prob_absdiff_hooked_vs_ref": float(np.abs(p_hooked - p_ref).max()),
            "argmax_equal": bool(p_re.argmax() == p_ref.argmax()),
            "near_tie": bool(len(top) > 1 and top[0] - top[1] <= 1e-4),
        })

        gold = case["gold"][qid]
        assert set(gold["probabilities"]) == set(opts), (qid, sorted(gold["probabilities"]), opts)
        st, ost = items[j]["state_stats"], items[j]["options"]
        decisions.append({
            "case_id": case["case_id"], "question_id": qid, "workflow": case["workflow"],
            "qtype": q["t"], "n_options": k, "option_ids": opts,
            "gold_dist": [float(gold["probabilities"][o]) for o in opts],
            "gold_argmax": str(gold["label"]),
            "model_probs": p_ref.tolist(), "model_pred": opts[int(p_ref.argmax())],
            "seq_len": len(items[j]["ids"]),
            "state_tokens": st["state_tokens"], "state_tokens_dropped": st["state_tokens_dropped"],
            "state_truncated": bool(st["truncated"]),
            "options_over_48_tokens": _options_over_cap(agent, q),
            "tokens_per_option_cap": ost["tokens_per_option"],
            "options_collapsed": ost["options_distinct"] < ost["options"],
        })
        for i, o in enumerate(opts):
            markers.append({"case_id": case["case_id"], "question_id": qid,
                            "option_index": i, "option_id": o})

    return {"decisions": decisions, "markers": markers, "parity": parity,
            "vectors": {label: np.concatenate(v) for label, v in vectors.items()}}
