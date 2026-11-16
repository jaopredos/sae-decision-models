"""Readers for the all-token activation dumps of `run_extract_tokens.py`."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_tokens(root: Path, label: str):
    """(acts memmap [n_tokens, hidden] fp16, token_index df, rows df) of one split directory."""
    root = Path(root)
    acts = np.load(root / f"tokens_{label}.npy", mmap_mode="r")
    ti = pd.read_parquet(root / "token_index.parquet")
    rows = pd.read_parquet(root / "rows.parquet")
    assert len(ti) == len(acts), (len(ti), len(acts))
    return acts, ti, rows


def token_is_dev(ti: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Bool per token: does it belong to a dev decision? (rows.is_dev indexed by decision_idx)"""
    return rows["is_dev"].to_numpy()[ti["decision_idx"].to_numpy()]


SPECIAL_ROLES = (0, 4, 6)  # src.activations: ROLE_CLS, ROLE_HEAD_SEP, ROLE_END_SEP


def massive_threshold(acts: np.ndarray, ti: pd.DataFrame, rows: pd.DataFrame, mult: float = 5.0,
                      chunk: int = 200_000) -> Tuple[float, float]:
    """(threshold, median): `mult` x the median activation norm of the non-special, non-dev domain
    tokens. Fixed once per layer from the train-minus-dev prompts (no labels, no test)."""
    keep = ~token_is_dev(ti, rows) & ~np.isin(ti["role"].to_numpy(), SPECIAL_ROLES)
    idx = np.flatnonzero(keep)
    norms = np.concatenate([np.linalg.norm(np.asarray(acts[idx[a:a + chunk]]).astype(np.float32), axis=1)
                            for a in range(0, len(idx), chunk)])
    med = float(np.median(norms))
    return mult * med, med
