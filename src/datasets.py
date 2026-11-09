"""Loaders for the three E0 benchmarks: Banking20, Newsgroups, Typed Decisions.

Canonical example schema (one per *decision*, i.e. one per question):
{
  "example_id": str,          # stable id, shared across contrastive pairs (same example,
                               # different perturbation) by construction (see perturbations.py)
  "dataset": "banking20" | "newsgroups" | "typed_decisions",
  "workflow": str | None,     # only for typed_decisions
  "qtype": "choice" | "score" | "noul",
  "state": str,                # text passed as `state` to DecisionModel.predict
  "instructions": str,
  "criteria": dict[str,str] | list[str],   # canonical (gold) order; id->desc for choice/noul,
                                            # ordered rubric for score
  "gold_id": str,              # correct id (choice/noul) or correct index as str (score)
}

Banking20 and Newsgroups are documented approximations of the paper's benchmarks (see
NOTES.md: the paper gives no class list/split for either) — we fix the approximation with a
seed so it is reproducible, and flag it in REPORT.md.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

BANKING20_SEED = 42
NEWSGROUPS_SEED = 42
TYPED_DECISIONS_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"

MAX_STATE_CHARS = 3000  # truncate long raw texts (newsgroups posts) to stay within context


def _humanize(label: str) -> str:
    return label.replace("_", " ").replace(".", " ").strip()


def _stratified_sample(items: List[dict], strata_key, n: int, seed: int) -> List[dict]:
    """Sample up to n items from `items`, balanced across the groups given by strata_key(item),
    deterministic given seed. If a stratum has fewer than its quota, the shortfall is
    redistributed to the other strata (in a fixed, seeded random order)."""
    rng = random.Random(seed)
    groups: Dict[Any, List[dict]] = {}
    for it in items:
        groups.setdefault(strata_key(it), []).append(it)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups.keys(), key=str)
    rng.shuffle(keys)
    quota = {k: 0 for k in keys}
    remaining = n
    active = list(keys)
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for k in list(active):
            available = len(groups[k]) - quota[k]
            if available <= 0:
                active.remove(k)
                continue
            take = min(share, available, remaining)
            if take <= 0:
                continue
            quota[k] += take
            remaining -= take
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    out = []
    for k in keys:
        out.extend(groups[k][: quota[k]])
    rng.shuffle(out)
    return out[:n]


def load_banking20(n: int = 500, seed: int = BANKING20_SEED) -> List[dict]:
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77")
    label_names = ds["train"].features["label"].names
    rng = random.Random(seed)
    chosen_classes = sorted(rng.sample(range(len(label_names)), 20))
    chosen_names = [label_names[i] for i in chosen_classes]
    criteria = {label_names[i]: _humanize(label_names[i]) for i in chosen_classes}

    pool = [
        {"text": row["text"], "label": label_names[row["label"]]}
        for row in ds["test"]
        if label_names[row["label"]] in chosen_names
    ]
    sampled = _stratified_sample(pool, lambda r: r["label"], n, seed)

    examples = []
    for i, row in enumerate(sampled):
        examples.append({
            "example_id": f"banking20_{i:04d}",
            "dataset": "banking20",
            "workflow": None,
            "qtype": "choice",
            "state": row["text"][:MAX_STATE_CHARS],
            "instructions": "Which banking intent category best matches this customer message?",
            "criteria": dict(criteria),
            "gold_id": row["label"],
        })
    return examples


_NEWSGROUPS_DESCRIPTIONS = {
    "alt.atheism": "discussion about atheism",
    "comp.graphics": "computer graphics",
    "comp.os.ms-windows.misc": "microsoft windows operating system",
    "comp.sys.ibm.pc.hardware": "ibm pc hardware",
    "comp.sys.mac.hardware": "mac hardware",
    "comp.windows.x": "x window system",
    "misc.forsale": "items for sale",
    "rec.autos": "automobiles",
    "rec.motorcycles": "motorcycles",
    "rec.sport.baseball": "baseball",
    "rec.sport.hockey": "hockey",
    "sci.crypt": "cryptography",
    "sci.electronics": "electronics",
    "sci.med": "medicine",
    "sci.space": "space and astronomy",
    "soc.religion.christian": "christian religion",
    "talk.politics.guns": "politics of guns",
    "talk.politics.mideast": "politics of the middle east",
    "talk.politics.misc": "general politics",
    "talk.religion.misc": "general religion",
}


def load_newsgroups(n: int = 500, seed: int = NEWSGROUPS_SEED) -> List[dict]:
    from sklearn.datasets import fetch_20newsgroups

    data_home = str(DATA_DIR / "sklearn_data")
    ds = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"),
                             data_home=data_home)
    names = ds.target_names
    criteria = {name: _NEWSGROUPS_DESCRIPTIONS.get(name, _humanize(name)) for name in names}

    pool = [{"text": text, "label": names[label]} for text, label in zip(ds.data, ds.target)
            if text.strip()]
    sampled = _stratified_sample(pool, lambda r: r["label"], n, seed)

    examples = []
    for i, row in enumerate(sampled):
        examples.append({
            "example_id": f"newsgroups_{i:04d}",
            "dataset": "newsgroups",
            "workflow": None,
            "qtype": "choice",
            "state": row["text"][:MAX_STATE_CHARS],
            "instructions": "Which newsgroup topic category best matches this post?",
            "criteria": dict(criteria),
            "gold_id": row["label"],
        })
    return examples


def _download_typed_decisions_parquet() -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="LocalLLaMA/typed-decisions", repo_type="dataset",
        filename="all/test-00000-of-00001.parquet", revision=TYPED_DECISIONS_REVISION,
        local_dir=str(DATA_DIR / "typed_decisions"),
    )
    return Path(path)


def load_typed_decisions(n: int = 500, seed: int = 42) -> List[dict]:
    """Parses the pinned test parquet exactly as julia1_repo/scripts/reproduce_typed.py does
    (same field layout: state/questions/gold JSON strings per case)."""
    import pyarrow.parquet as pq

    parquet_path = _download_typed_decisions_parquet()
    cases = pq.read_table(parquet_path).to_pylist()
    assert len(cases) == 400, f"expected 400 typed-decisions test cases, got {len(cases)}"

    decisions = []
    for case in cases:
        state = json.loads(case["state"])
        gold = json.loads(case["gold"])
        questions = json.loads(case["questions"])
        workflow = case["workflow"]
        for qid, q in questions.items():
            kind = q["type"]
            criteria = q.get("criteria")
            if criteria is None and kind == "noul":
                criteria = {"false": "false", "true": "true"}
            if kind == "score":
                # ordered rubric (list); keep as list for the model, keys are "0".."k-1"
                assert isinstance(criteria, list)
                gold_id = str(gold[qid]["label"])
            else:
                if isinstance(criteria, list):
                    criteria = {str(i): v for i, v in enumerate(criteria)}
                assert isinstance(criteria, dict)
                gold_id = str(gold[qid]["label"])
            decisions.append({
                "example_id": f"typed_{case['id']}_{qid}",
                "dataset": "typed_decisions",
                "workflow": workflow,
                "qtype": kind,
                "state": state,
                "instructions": q["instructions"],
                "criteria": criteria,
                "gold_id": gold_id,
            })

    counts = {}
    for d in decisions:
        counts[d["qtype"]] = counts.get(d["qtype"], 0) + 1
    assert counts == {"choice": 600, "score": 800, "noul": 600}, counts

    sampled = _stratified_sample(decisions, lambda d: (d["workflow"], d["qtype"]), n, seed)
    return sampled


LOADERS = {
    "banking20": load_banking20,
    "newsgroups": load_newsgroups,
    "typed_decisions": load_typed_decisions,
}
