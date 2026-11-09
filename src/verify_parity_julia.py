"""Parity check for Julia-1, analogous to verify_parity.py for Laya.

Must run under the dedicated `.venv_julia` interpreter:
    .venv_julia/bin/python3 -I src/verify_parity_julia.py --checkpoint data/julia1_repo

Same method as verify_parity.py: monkeypatch `engine.model.forward` to capture the exact
input tensors and official (logits, actions) output for ~50 synthetic typed-choice calls,
then independently build the encoder a second time via `AutoConfig.from_pretrained` +
`AutoModel.from_config` (fresh architecture) and load its weights from the "encoder."
prefix of the single combined model.safetensors (exactly as julia/model.py's own
`from_pretrained` does), run it on the captured input_ids/attention_mask, and replay the
head (type_emb -> head layers -> marker gather -> scorer -> masked softmax) using the
already-loaded model's head/scorer/type_emb submodules. No synthetic/random weights are
used; only forward passes, no gradients.
"""
import argparse
import json
import random
import string
from pathlib import Path

import torch
import torch.nn.functional as F


def synthetic_examples(n=50, seed=0):
    rng = random.Random(seed)
    topics = ["billing", "shipping", "returns", "account_access", "technical_support",
              "warranty", "cancellation", "loyalty_program"]
    out = []
    for i in range(n):
        k = rng.randint(2, 5)
        opts = rng.sample(topics, k)
        gold = rng.choice(opts)
        words = " ".join(rng.choices(string.ascii_lowercase, k=rng.randint(10, 40)))
        out.append({
            "state": f"customer message about {gold}: {words}",
            "criteria": {o: o.replace("_", " ") for o in opts},
            "gold": gold,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, args.checkpoint)
    from transformers import AutoConfig, AutoModel
    from safetensors.torch import load_file
    from julia import load_model

    engine = load_model(args.checkpoint, device="cpu", max_length=1024, head_length=256,
                        strict_encoding=True, backend="torch")
    model = engine.model
    model.eval()

    root = Path(args.checkpoint)
    econfig = AutoConfig.from_pretrained(root / "encoder", trust_remote_code=False)
    econfig.reference_compile = False
    fresh_encoder = AutoModel.from_config(econfig, attn_implementation="sdpa",
                                          trust_remote_code=False)
    all_weights = load_file(str(root / "model.safetensors"))
    encoder_weights = {k[len("encoder."):]: v for k, v in all_weights.items()
                       if k.startswith("encoder.")}
    fresh_encoder.load_state_dict(encoder_weights, strict=True)
    fresh_encoder.eval()

    captured = []
    orig_forward = model.forward

    def spy(*fargs, **fkwargs):
        out = orig_forward(*fargs, **fkwargs)
        captured.append((fargs, fkwargs, out))
        return out

    model.forward = spy
    try:
        for ex in synthetic_examples(args.n):
            row = dict(state=ex["state"], question="Which category applies?",
                      type="choice", options=list(ex["criteria"].values()))
            engine.logits([row])
    finally:
        model.forward = orig_forward

    @torch.no_grad()
    def manual_forward(input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = fresh_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + model.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos[:, :, None].expand(-1, -1, h.size(-1))
        m = h.gather(1, idx)
        logits = model.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        return logits

    max_abs_diff = 0.0
    argmax_matches = 0
    for fargs, fkwargs, out in captured:
        keys = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
        values = list(fargs) + [fkwargs[k] for k in keys[len(fargs):]]
        input_ids, attention_mask, marker_pos, marker_mask, qtype = values[:5]
        official_logits = out if not isinstance(out, tuple) else out[0]
        manual_logits = manual_forward(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        p_official = F.softmax(official_logits.float(), dim=-1)
        p_manual = F.softmax(manual_logits.float(), dim=-1)
        diff = (p_official - p_manual).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        argmax_matches += int(torch.equal(p_official.argmax(-1), p_manual.argmax(-1)))

    result = {
        "model": "julia1",
        "n_examples": args.n,
        "n_calls_compared": len(captured),
        "max_abs_probability_diff": max_abs_diff,
        "argmax_agreement_rate": argmax_matches / len(captured) if captured else None,
        "method": "fresh AutoModel.from_config(encoder_config) + weights loaded from the "
                  "'encoder.' prefix of model.safetensors, head/scorer/type_emb reused, vs. "
                  "captured engine.model.forward output, same input tensors",
    }
    out_path = Path(__file__).resolve().parents[1] / "results" / "parity_check_julia.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
