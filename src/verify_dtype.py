"""BF16/FP16 autocast vs. FP32 comparison for Laya multilingual on ~100 examples.

Laya only autocasts on MPS/CUDA when a batch has >= `mps_amp_min_rows` (default 5) states
(agent.py `_amp_enabled_for`; see NOTES.md); a single `predict()` call is always FP32. So the
only way to actually exercise the reduced-precision path is `predict_batch` with a large
batch, which is what this script does, toggling `agent.amp_enabled` to force FP32 for the
control run. Julia-1 is not covered here: on CPU it never leaves FP32 (see NOTES.md), so
there is nothing to compare for it.
"""
import json
from pathlib import Path

from src.datasets import load_banking20
from src.models import LayaModel

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(n: int = 100):
    examples = load_banking20(n=n)
    criteria = examples[0]["criteria"]  # shared across all banking20 examples by construction
    questions = {"q": {"type": "choice", "instructions": examples[0]["instructions"],
                       "criteria": criteria}}
    states = [ex["state"] for ex in examples]

    laya = LayaModel(subfolder="multilingual")
    agent = laya.agent
    print("agent.device", agent.device, "native dtype", agent.dtype,
          "amp_enabled", agent.amp_enabled, "mps_amp_min_rows", agent.mps_amp_min_rows)

    reduced = agent.predict_batch(states, questions, batch_size=len(states))

    agent.amp_enabled = False
    fp32 = agent.predict_batch(states, questions, batch_size=len(states))

    diffs = []
    flips = 0
    for a, f in zip(reduced, fp32):
        pa, pf = a["answers"]["q"]["probabilities"], f["answers"]["q"]["probabilities"]
        diffs.append(max(abs(pa[k] - pf[k]) for k in pa))
        if a["answers"]["q"]["choice"] != f["answers"]["q"]["choice"]:
            flips += 1

    result = {
        "model": "laya_multilingual",
        "n_examples": n,
        "device": str(agent.device),
        "reduced_dtype": str(agent.dtype),
        "max_abs_prob_diff": max(diffs),
        "mean_abs_prob_diff": sum(diffs) / len(diffs),
        "n_choice_flips_reduced_vs_fp32": flips,
        "flip_rate": flips / n,
    }
    out_path = REPO_ROOT / "results" / "dtype_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
