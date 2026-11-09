"""Common wrapper over Laya and Julia-1, giving both the same predict() contract.

Laya runs in-process (this interpreter has `laya` + transformers>=5.17).
Julia-1 runs in a dedicated subprocess under .venv_julia (transformers==5.0.0 pinned,
see NOTES.md "Ambiente de execução" for why the two cannot share a process).

Contract: DecisionModel.predict(state, questions) -> {qid: {"type", "probabilities": {id: p},
"selected": id}}. `questions` is {qid: {"type": "choice"|"score"|"noul", "instructions": str,
"criteria": dict[id,str] (choice/noul) | list[str] (score)}}.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
JULIA_CHECKPOINT = REPO_ROOT / "data" / "julia1_repo"
JULIA_VENV_PYTHON = REPO_ROOT / ".venv_julia" / "bin" / "python3"
JULIA_WORKER = REPO_ROOT / "src" / "julia_worker.py"

SEED = 0


def set_all_seeds(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    """Make Laya's and Julia-1's native `answers` dicts uniform.

    Laya's `noul` answers carry only `noul` (= P(true)), no `probabilities` key
    (agent.py `_decode_answers`); Julia-1's `noul` answers always carry `probabilities`.
    We synthesize {"false": 1-p, "true": p} for Laya so both models expose the same shape.
    """
    out = {}
    for qid, ans in answers.items():
        qtype = ans["type"]
        if qtype == "noul":
            probs = ans.get("probabilities")
            if probs is None:
                p_true = float(ans["noul"])
                probs = {"false": 1.0 - p_true, "true": p_true}
        else:
            probs = ans["probabilities"]
        probs = {k: float(v) for k, v in probs.items()}
        selected = max(probs, key=probs.get)
        out[qid] = {"type": qtype, "probabilities": probs, "selected": selected}
    return out


class DecisionModel:
    name: str

    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LayaModel(DecisionModel):
    """Wraps `laya.load(..., subfolder=...)` + `agent.predict(...)`, always eval/no_grad."""

    def __init__(self, subfolder: str | None = "multilingual", device: str | None = None):
        import laya

        set_all_seeds()
        self.name = "laya_multilingual" if subfolder == "multilingual" else "laya_english"
        self.subfolder = subfolder
        self.agent = laya.load("convaiinnovations/laya", subfolder=subfolder, device=device)
        self.agent.model.eval()

    @torch.no_grad()
    def predict(self, state, questions):
        set_all_seeds()
        result = self.agent.predict(state, questions)
        return normalize_answers(result["answers"])


class JuliaModel(DecisionModel):
    """Wraps Julia-1's `load_model(...).predict(state=, questions=)` via a persistent
    subprocess running under the dedicated `.venv_julia` interpreter (see NOTES.md)."""

    name = "julia1"

    def __init__(self, checkpoint: Path = JULIA_CHECKPOINT, max_length: int = 2048,
                 head_length: int = 512, device: str = "cpu"):
        self.checkpoint = str(checkpoint)
        self.proc = subprocess.Popen(
            [str(JULIA_VENV_PYTHON), "-I", str(JULIA_WORKER),
             "--checkpoint", self.checkpoint, "--device", device,
             "--max-length", str(max_length), "--head-length", str(head_length),
             "--seed", str(SEED)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(REPO_ROOT),
        )
        ready_line = self.proc.stdout.readline()
        if not ready_line or "ready" not in ready_line:
            err = self.proc.stderr.read()
            raise RuntimeError(f"Julia worker failed to start: {err[-4000:]}")

    def predict(self, state, questions):
        if self.proc.poll() is not None:
            err = self.proc.stderr.read()
            raise RuntimeError(f"Julia worker died: {err[-4000:]}")
        request = {"state": state, "questions": questions}
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()
            raise RuntimeError(f"Julia worker closed stdout: {err[-4000:]}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"Julia worker error: {response.get('error')}")
        return normalize_answers(response["answers"])

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


def load_model(model_key: str) -> DecisionModel:
    if model_key == "laya_multilingual":
        return LayaModel(subfolder="multilingual")
    if model_key == "laya_english":
        return LayaModel(subfolder=None)
    if model_key == "julia1":
        return JuliaModel()
    raise ValueError(f"Unknown model_key: {model_key}")
