import numpy as np

from src.corpus import TokenStore, batches


def _store(n_chunks):
    ids = np.arange(n_chunks * 8, dtype=np.int32).reshape(n_chunks, 8)
    return TokenStore(ids=ids, lengths=np.full(n_chunks, 8, np.int32), source=np.zeros(n_chunks, np.int16), sources=["a"])


def test_batches_with_fewer_chunks_than_rows_does_not_hang():
    it = batches(_store(3), rows_per_batch=64)
    ids, mask, src = next(it)
    assert ids.shape == (3, 8) and mask.all()


def test_batches_cover_every_chunk_once_per_pass():
    it = batches(_store(10), rows_per_batch=5, seed=3)
    seen = np.concatenate([next(it)[0][:, 0] for _ in range(2)])
    assert sorted(seen.tolist()) == [i * 8 for i in range(10)]
