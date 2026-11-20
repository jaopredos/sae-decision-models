"""Pure helpers for identity-aligned order experiments (no model dependencies)."""
import hashlib
import itertools
import math
import random


def stable_seed(*parts):
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:16], 16)


def orders(k):
    if k < 2:
        raise ValueError("at least two options required")
    if k <= 4:
        return [list(p) for p in itertools.permutations(range(k))]
    base = list(range(k))
    candidates = [base[r:] + base[:r] for r in range(k)]
    candidates += [list(reversed(p)) for p in candidates.copy()]
    return [list(p) for p in dict.fromkeys(tuple(p) for p in candidates)]


def align_slots(values, order):
    if sorted(order) != list(range(len(values))):
        raise ValueError("order must permute option indices")
    aligned = [None] * len(order)
    for slot, identity in enumerate(order):
        aligned[identity] = values[slot]
    return aligned


def center(values):
    mean = sum(values) / len(values)
    return [float(v - mean) for v in values]


def effect_metrics(donor, recipient, patched, eps=1e-8):
    d, r, p = map(center, (donor, recipient, patched))
    contrast = [a - b for a, b in zip(d, r)]
    change = [a - b for a, b in zip(p, r)]
    denom = sum(x*x for x in contrast)
    dot = sum(a*b for a, b in zip(contrast, change))
    ratio = dot / denom if denom > eps else None
    orthogonal = ([b - ratio*a for a, b in zip(contrast, change)]
                  if ratio is not None else change)
    return {"contrast_sqnorm": denom, "projection_numerator": dot,
            "transfer_ratio": ratio, "change_norm": math.sqrt(sum(x*x for x in change)),
            "orthogonal_norm": math.sqrt(sum(x*x for x in orthogonal)),
            "distance_to_donor": math.sqrt(sum((a-b)**2 for a,b in zip(p,d)))}


def conditions(question, state, swapped_state, case_id, qid):
    """Each variant retains a stable semantic index; ID changes are a separate treatment."""
    import copy
    q = copy.deepcopy(question)
    if q["type"] != "choice":
        raise ValueError("primary protocol is choice only")
    if isinstance(q["criteria"], list):
        q["criteria"] = {str(i): v for i, v in enumerate(q["criteria"])}
    q.pop("option_order", None)
    keys = list(q["criteria"])
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate IDs")
    yield "original", q, state, keys
    yield "empty", copy.deepcopy(q), "", keys
    yield "swapped_state", copy.deepcopy(q), swapped_state, keys
    renamed = copy.deepcopy(q)
    renamed["criteria"] = {f"opt_{i+1}": q["criteria"][key] for i, key in enumerate(keys)}
    yield "renamed_ids", renamed, state, list(renamed["criteria"])


def choose_cases(cases, subset, dev_ids, limit=0):
    selected = [c for c in cases if subset == "test" or
                ((c["case_id"] in dev_ids) == (subset == "dev"))]
    selected.sort(key=lambda c: stable_seed("case-order-v1", c["case_id"]))
    return selected[:limit] if limit else selected


def donor_state_map(cases):
    """Deranged state donors with exactly identical question specifications."""
    import json
    groups = {}
    for c in cases:
        key = json.dumps(c["questions"], sort_keys=True)
        groups.setdefault(key, []).append(c)
    result = {}
    for group in groups.values():
        group = sorted(group, key=lambda c: stable_seed("state-donor", c["case_id"]))
        if len(group) < 2:
            raise ValueError("state-swap control requires two cases per question group")
        for i, c in enumerate(group):
            result[c["case_id"]] = group[(i+1) % len(group)]
    return result
