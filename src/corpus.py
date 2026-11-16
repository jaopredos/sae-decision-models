"""Generic-text corpus for the SAE (stage 4): download once as raw text, tokenize per model.

Sources (results/sae_features/PROTOCOLO.md): English from `HuggingFaceFW/fineweb` (sample-10BT) and
12 languages from `HuggingFaceFW/fineweb-2` (it has no English). Shards are ~5 GB, so they are read
by row group through `HfFileSystem` + pyarrow, one stream at a time, and stop at a token quota.
Token quotas are counted with a reference tokenizer; every training job re-tokenizes with its own.

`TokenStore` turns the text into fixed-format sequences `[CLS] <=510 tokens [SEP]`, split into
train and held-out documents, with a per-sequence source tag so quality can be reported by source.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

FINEWEB_REPO = "datasets/HuggingFaceFW/fineweb"
FINEWEB_EN_PATH = "sample/10BT/000_00000.parquet"
FINEWEB2_REPO = "datasets/HuggingFaceFW/fineweb-2"
FINEWEB2_LANGS = ["por_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "ita_Latn", "rus_Cyrl", "cmn_Hani",
                  "jpn_Jpan", "arb_Arab", "hin_Deva", "tur_Latn", "kor_Hang"]
MAX_DOC_CHARS = 20_000
CHUNK_TOKENS = 510  # + [CLS] + [SEP] = 512
MIN_CHUNK_TOKENS = 16
VAL_EVERY = 50  # one document in 50 is held out (by hash of its position in the source)


def stream_texts(repo: str, path: str, batch_rows: int = 256) -> Iterator[str]:
    """Texts of one parquet shard, by row group. One stream at a time keeps host RAM flat."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"{repo}/{path}", "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
            for t in batch.column("text").to_pylist():
                if t and t.strip():
                    yield t[:MAX_DOC_CHARS]


def count_tokens(tok, texts: List[str]) -> List[int]:
    ids = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
    return [len(i) for i in ids]


def collect(repo: str, path: str, tok, token_quota: int, source: str) -> pd.DataFrame:
    """First documents of a shard until `token_quota` tokens (by `tok`) are reached."""
    rows, total, buf = [], 0, []
    for t in stream_texts(repo, path):
        buf.append(t)
        if len(buf) == 256:
            for text, n in zip(buf, count_tokens(tok, buf)):
                rows.append((text, n)); total += n
            buf = []
            if total >= token_quota:
                break
    if buf and total < token_quota:
        for text, n in zip(buf, count_tokens(tok, buf)):
            rows.append((text, n)); total += n
    df = pd.DataFrame(rows, columns=["text", "n_tokens_ref"])
    df["source"] = source
    df["doc_idx"] = np.arange(len(df))
    df["is_val"] = [int(hashlib.sha256(f"{source}:{i}".encode()).hexdigest(), 16) % VAL_EVERY == 0
                    for i in range(len(df))]
    return df


# ---- token store -----------------------------------------------------------------------------
@dataclass
class TokenStore:
    ids: np.ndarray          # int32 [n_chunks, 512], 0-padded
    lengths: np.ndarray      # int32 [n_chunks], including [CLS] and [SEP]
    source: np.ndarray       # int16 [n_chunks], index into `sources`
    sources: List[str]

    @property
    def n_tokens(self) -> int:
        return int(self.lengths.sum())


def build_store(tok, df: pd.DataFrame, budgets: Dict[str, int], val: bool, seed: int = 0) -> TokenStore:
    """Chunk the documents of `df` (columns text/source/is_val), up to `budgets[source]` tokens each.

    `val=False` keeps training documents, `val=True` the held-out ones (budget = 1/VAL_EVERY-ish: the
    held-out documents are used whole). Documents are taken in order; a source that runs out is
    cycled and the caller is told through `TokenStore.n_tokens`."""
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    chunks, lens, src = [], [], []
    sources = sorted(df["source"].unique())
    for s_idx, s in enumerate(sources):
        d = df[(df["source"] == s) & (df["is_val"] == val)]
        budget = budgets.get(s, 0) if not val else None
        got = 0
        texts = d["text"].tolist()
        if not texts:
            continue
        cycle = 0
        while budget is None or got < budget:
            for i in range(0, len(texts), 256):
                for ids in tok(texts[i:i + 256], add_special_tokens=False, truncation=False)["input_ids"]:
                    for a in range(0, len(ids), CHUNK_TOKENS):
                        c = ids[a:a + CHUNK_TOKENS]
                        if len(c) < MIN_CHUNK_TOKENS:
                            continue
                        row = np.zeros(CHUNK_TOKENS + 2, dtype=np.int32)
                        row[0], row[1:len(c) + 1], row[len(c) + 1] = cls_id, c, sep_id
                        chunks.append(row); lens.append(len(c) + 2); src.append(s_idx)
                        got += len(c) + 2
                    if budget is not None and got >= budget:
                        break
                if budget is not None and got >= budget:
                    break
            cycle += 1
            if budget is None or cycle > 20:
                break
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chunks))
    return TokenStore(ids=np.stack(chunks)[order], lengths=np.asarray(lens, dtype=np.int32)[order],
                      source=np.asarray(src, dtype=np.int16)[order], sources=sources)


def batches(store: TokenStore, rows_per_batch: int, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Endless (input_ids, attention_mask, source) batches; reshuffles after each pass."""
    rng = np.random.default_rng(seed)
    n = len(store.lengths)
    while True:
        order = rng.permutation(n)
        for a in range(0, n - rows_per_batch + 1, rows_per_batch):
            idx = order[a:a + rows_per_batch]
            width = int(store.lengths[idx].max())
            ids = store.ids[idx, :width]
            mask = (np.arange(width)[None, :] < store.lengths[idx][:, None])
            yield ids, mask, store.source[idx]
