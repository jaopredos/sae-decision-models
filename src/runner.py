"""Glue between datasets.py / perturbations.py / models.py and the JSONL log schema.

Log schema (one row per decision x condition), matching the prompt's "Log de tudo":
model, dataset, example_id, condition, presented_order, gold_id, gold_position,
chosen_id, chosen_position, probabilities (full vector), plus workflow/qtype/seed for
bookkeeping. `example_id` is stable across conditions of the same underlying example, so
rows can be joined back into contrastive pairs for later SAE use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

SEED = 0


def canonical_criteria(example: dict) -> Dict[str, str]:
    """Return the example's criteria as an id->description dict regardless of qtype.
    For `score`, the logical id of rubric level i is str(i) (its position in the original,
    un-perturbed rubric) -- see module docstring in perturbations.py."""
    c = example["criteria"]
    if isinstance(c, list):
        return {str(i): desc for i, desc in enumerate(c)}
    return dict(c)


def _as_model_criteria(example: dict, ordered_criteria: Dict[str, str]):
    """Shape the ordered id->desc dict the way the model package expects it for this qtype:
    choice/noul keep the dict; score needs a plain ordered list of descriptions."""
    if example["qtype"] == "score":
        return list(ordered_criteria.values())
    return ordered_criteria


def predict_record(model, example: dict, order: List[str], criteria_map: Dict[str, str],
                    condition: str, gold_id: Optional[str] = None,
                    state_override=None, seed: int = SEED) -> dict:
    """Run one model call for `example` under a single (order, criteria_map, state) choice
    and return the logged record. `order` is the list of logical ids in presentation order;
    `criteria_map` gives the description shown for each id (lets B2 decouple id<->description
    while keeping `order` meaning "presentation order"). `gold_id` overrides the id considered
    correct under this condition (needed for decouple/rename; defaults to example['gold_id'])."""
    qtype = example["qtype"]
    gold_id = example["gold_id"] if gold_id is None else gold_id
    state = example["state"] if state_override is None else state_override

    ordered_desc = {i: criteria_map[i] for i in order}
    model_criteria = _as_model_criteria(example, ordered_desc)
    qspec = {"type": qtype, "instructions": example["instructions"], "criteria": model_criteria}

    answer = model.predict(state, {"q": qspec})["q"]

    if qtype == "score":
        # model keys probabilities by *presented position*, not by our logical id
        probabilities = {order[int(pos)]: p for pos, p in answer["probabilities"].items()}
        chosen_id = order[int(answer["selected"])]
    else:
        probabilities = answer["probabilities"]
        chosen_id = answer["selected"]

    gold_position = order.index(gold_id) if gold_id in order else -1
    chosen_position = order.index(chosen_id) if chosen_id in order else -1

    return {
        "model": model.name,
        "dataset": example["dataset"],
        "workflow": example.get("workflow"),
        "qtype": qtype,
        "example_id": example["example_id"],
        "condition": condition,
        "seed": seed,
        "presented_order": order,
        "criteria_presented": ordered_desc,
        "gold_id": gold_id,
        "gold_position": gold_position,
        "chosen_id": chosen_id,
        "chosen_position": chosen_position,
        "probabilities": probabilities,
    }


class JsonlWriter:
    def __init__(self, path: Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open(mode)

    def write(self, record: dict):
        self._fh.write(json.dumps(record) + "\n")

    def flush(self):
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path: Path) -> List[dict]:
    out = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
