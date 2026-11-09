"""Parity check: does `agent.predict()` (official laya convenience path) match an
independent `transformers`-only re-run with direct hidden-state access?

Method: monkeypatch `agent.model.forward` to capture the exact (input_ids, attention_mask,
marker_pos, marker_mask, qtype) tensors and the official (logits, act) output for each of
~50 real calls to `agent.predict()`. Then, for each captured call, load the encoder a
*second* time as a fresh `transformers.AutoModel.from_pretrained(encoder_dir)` (independent
weights load, not the one inside `agent.model`), run it on the captured input_ids/
attention_mask to get `last_hidden_state`, and manually replay the rest of the head
(type embedding -> head transformer layers -> marker gather -> scorer -> masked softmax),
reusing the already-loaded `agent.model.type_emb` / `.head` / `.scorer` submodules (same
weights; this is composition, not reimplementation of the architecture). No windowing/CPU
fast-path shortcuts are used in the manual replay, so it also checks that those shortcuts
(`_cpu_window_encoder`, `_cpu_head`) do not change the result.

Writes results/parity_check.json: max abs probability diff and argmax agreement over ~50
banking20 examples, run with `torch.no_grad()` only (no weights touched).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.datasets import load_banking20
from src.models import LayaModel

REPO_ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def manual_head_forward(agent_model, encoder_fresh, input_ids, attention_mask, marker_pos,
                         marker_mask, qtype):
    h = encoder_fresh(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h = h + agent_model.type_emb(qtype)[:, None, :]
    pad = ~attention_mask.bool()
    for layer in agent_model.head.layers:
        h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    m = torch.gather(h, 1, idx)
    logits = agent_model.scorer(m).squeeze(-1).float()
    logits = logits.masked_fill(~marker_mask, -1e4)
    return logits


def main(n_examples: int = 50):
    from transformers import AutoConfig, AutoModel
    from safetensors.torch import load_file

    examples = load_banking20(n=n_examples)
    laya = LayaModel(subfolder="multilingual")
    agent = laya.agent

    # Agent doesn't keep the resolved local directory on self; re-resolve it the same way
    # __init__ did (hits the local HF cache, so this is a no-op download).
    from huggingface_hub import snapshot_download
    prefix = "multilingual/" if agent.subfolder else ""
    base = snapshot_download(
        "convaiinnovations/laya",
        allow_patterns=[prefix + name for name in (
            "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")],
    )
    model_dir = str(Path(base) / agent.subfolder) if agent.subfolder else base
    encoder_dir = str(Path(model_dir) / "encoder")

    # Weights for the whole checkpoint (encoder + head + scorer + type_emb + act_head) live
    # in one combined model.safetensors with "encoder."-prefixed keys for the encoder; the
    # encoder/ subfolder only holds its HF config (this is what laya.common.build_model does
    # too: AutoConfig+AutoModel.from_config for the architecture, state dict loaded separately).
    econfig = AutoConfig.from_pretrained(encoder_dir)
    fresh_encoder = AutoModel.from_config(econfig, attn_implementation="sdpa")
    all_weights = load_file(str(Path(model_dir) / "model.safetensors"))
    encoder_weights = {k[len("encoder."):]: v for k, v in all_weights.items()
                       if k.startswith("encoder.")}
    missing, unexpected = fresh_encoder.load_state_dict(encoder_weights, strict=True)
    fresh_encoder.eval()
    fresh_encoder.to(agent.device)

    captured = []
    orig_forward = agent.model.forward

    def spy(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        captured.append((args, kwargs, out))
        return out

    agent.model.forward = spy
    try:
        for ex in examples:
            q = {"q": {"type": ex["qtype"], "instructions": ex["instructions"],
                       "criteria": ex["criteria"]}}
            agent.predict(ex["state"], q)
    finally:
        agent.model.forward = orig_forward

    max_abs_diff = 0.0
    argmax_matches = 0
    total = 0
    for args, kwargs, out in captured:
        input_ids, attention_mask, marker_pos, marker_mask, qtype = args[:5]
        official_logits, _ = out
        manual_logits = manual_head_forward(agent.model, fresh_encoder, input_ids,
                                             attention_mask, marker_pos, marker_mask, qtype)
        p_official = F.softmax(official_logits.float(), dim=-1)
        p_manual = F.softmax(manual_logits.float(), dim=-1)
        diff = (p_official - p_manual).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        argmax_matches += int(torch.equal(p_official.argmax(-1), p_manual.argmax(-1)))
        total += p_official.shape[0]

    result = {
        "model": "laya_multilingual",
        "n_examples": n_examples,
        "n_rows_compared": total,
        "max_abs_probability_diff": max_abs_diff,
        "argmax_agreement_rate": argmax_matches / len(captured) if captured else None,
        "method": "fresh AutoModel.from_pretrained(encoder_dir) + reused head/scorer/type_emb "
                  "submodules vs. captured agent.model.forward output, same input tensors",
    }
    out_path = REPO_ROOT / "results" / "parity_check_laya.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
