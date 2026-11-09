"""Order/label/content perturbations used across Phase A and Phase B.

All functions are pure and take/return plain dicts/lists so they can be logged verbatim.
`criteria` here is always the canonical dict (id -> description) for `choice`/`noul`
questions; `score` questions use an ordered list and are handled by the few functions that
accept `score=True` explicitly (score rubrics are not renamed/decoupled: the rubric *is*
its own order, so B2 does not apply to it, matching the datasets.py design where only
typed_decisions carries `score` questions).
"""
from __future__ import annotations

import itertools
import random
import string
from typing import Dict, List, Tuple

CONTENT_FREE_STATES = {
    "empty": "",
    "na": "N/A",
    "mask": "[MASK]",
}


def cyclic_rotations(ids: List[str]) -> List[Tuple[str, List[str]]]:
    k = len(ids)
    return [(f"rot_{r}", ids[r:] + ids[:r]) for r in range(k)]


def all_permutations(ids: List[str]) -> List[Tuple[str, List[str]]]:
    return [(f"perm_{i}", list(p)) for i, p in enumerate(itertools.permutations(ids))]


def generate_orders(ids: List[str], max_full_perm_k: int = 4) -> List[Tuple[str, List[str]]]:
    """All k cyclic rotations; additionally all k! permutations when k <= max_full_perm_k
    (rotations are a subset of permutations, so the result has no duplicate orders)."""
    if len(ids) <= max_full_perm_k:
        return all_permutations(ids)
    return cyclic_rotations(ids)


def reorder_criteria(criteria: Dict[str, str], order: List[str]) -> Dict[str, str]:
    return {i: criteria[i] for i in order}


def rename_opt_n(criteria_in_order: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """ids -> opt_1..opt_k, preserving presentation order. Returns (renamed_criteria, id_map)
    where id_map maps new_id -> original_id."""
    new_to_old = {f"opt_{i+1}": old for i, old in enumerate(criteria_in_order)}
    renamed = {new: criteria_in_order[old] for new, old in new_to_old.items()}
    return renamed, new_to_old


def rename_alpha(criteria_in_order: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """ids -> A, B, C, ... preserving presentation order."""
    letters = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase
                                               for b in string.ascii_uppercase]
    new_to_old = {letters[i]: old for i, old in enumerate(criteria_in_order)}
    renamed = {new: criteria_in_order[old] for new, old in new_to_old.items()}
    return renamed, new_to_old


def decouple_descriptions(criteria_in_order: Dict[str, str], gold_id: str, seed: int
                           ) -> Tuple[Dict[str, str], str]:
    """Keep ids and presentation order fixed; shuffle which description goes with which id.
    The "correct" id under this condition is whichever id ends up holding the gold
    description (returned as the new gold_id)."""
    ids = list(criteria_in_order)
    descs = [criteria_in_order[i] for i in ids]
    rng = random.Random(seed)
    shuffled_descs = descs[:]
    # Guarantee an actual derangement relative to the original id->desc pairing when k>1,
    # otherwise decoupling could silently no-op for this example.
    for _ in range(50):
        rng.shuffle(shuffled_descs)
        if len(ids) <= 1 or shuffled_descs != descs:
            break
    new_criteria = {i: d for i, d in zip(ids, shuffled_descs)}
    gold_desc = criteria_in_order[gold_id]
    new_gold_id = next(i for i, d in new_criteria.items() if d == gold_desc)
    return new_criteria, new_gold_id


def subsample_k(criteria: Dict[str, str], gold_id: str, k: int, seed: int) -> Dict[str, str]:
    """Gold id + (k-1) distractors drawn from `criteria` without replacement, seeded."""
    others = [i for i in criteria if i != gold_id]
    rng = random.Random(seed)
    k = min(k, len(criteria))
    distractors = rng.sample(others, k - 1)
    chosen = [gold_id] + distractors
    rng.shuffle(chosen)
    return {i: criteria[i] for i in chosen}


def random_token_state(seed: int, n_tokens: int = 20) -> str:
    rng = random.Random(seed)
    vocab = string.ascii_lowercase
    return " ".join("".join(rng.choices(vocab, k=rng.randint(3, 9))) for _ in range(n_tokens))


def noul_literal_criteria() -> Dict[str, str]:
    return {"false": "false", "true": "true"}
