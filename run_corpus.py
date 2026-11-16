"""Stage 4, step 0: download the generic text of the SAE corpus once (raw text, no model).

Usage: python run_corpus.py [--out data/corpus] [--en-tokens N] [--lang-tokens N] [--limit-langs L]

Quotas are in tokens of the reference tokenizer (Laya multilingual). English is sized for the
English-large SAE of part C too (a different tokenizer, so a margin); each training job takes only
the budget it needs. Writes <out>/text.parquet (text, source, n_tokens_ref, doc_idx, is_val) and
<out>/corpus_stats.json. One shard is streamed at a time; RSS is logged.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER
from src import corpus as C

REPO_ROOT = Path(__file__).resolve().parent


def rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if r > 1e8 else r / 1e6  # macOS reports bytes, Linux kilobytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "corpus"))
    ap.add_argument("--en-tokens", type=int, default=52_000_000)
    ap.add_argument("--lang-tokens", type=int, default=2_000_000)
    ap.add_argument("--limit-langs", type=int, default=0, help="only the first N FineWeb-2 languages (smoke)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = snapshot_download(LAYA_REPO, revision=LAYA_REVISION, allow_patterns=[f"{LAYA_SUBFOLDER}/tokenizer/*"])
    tok = AutoTokenizer.from_pretrained(str(Path(base) / LAYA_SUBFOLDER / "tokenizer"))

    parts, stats, t0 = [], {}, time.time()
    plan = [("en", C.FINEWEB_REPO, C.FINEWEB_EN_PATH, args.en_tokens)]
    langs = C.FINEWEB2_LANGS[: args.limit_langs] if args.limit_langs else C.FINEWEB2_LANGS
    plan += [(lang, C.FINEWEB2_REPO, f"data/{lang}/train/000_00000.parquet", args.lang_tokens) for lang in langs]
    for source, repo, path, quota in plan:
        df = C.collect(repo, path, tok, quota, source)
        parts.append(df)
        stats[source] = {"docs": len(df), "tokens_ref": int(df["n_tokens_ref"].sum()), "quota": quota,
                         "val_docs": int(df["is_val"].sum()), "chars": int(df["text"].str.len().sum()),
                         "repo": repo, "path": path}
        print(f"[corpus] {source:9s} {stats[source]['docs']:7d} docs {stats[source]['tokens_ref']/1e6:6.2f}M tok "
              f"(quota {quota/1e6:.1f}M) | rss {rss_gb():.2f} GB | {time.time() - t0:.0f}s", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(out / "text.parquet")
    (out / "corpus_stats.json").write_text(json.dumps(
        {"reference_tokenizer": f"{LAYA_REPO}@{LAYA_REVISION}/{LAYA_SUBFOLDER}", "sources": stats,
         "total_tokens_ref": sum(s["tokens_ref"] for s in stats.values())}, indent=2))
    print(f"[corpus] wrote {out / 'text.parquet'}", flush=True)


if __name__ == "__main__":
    main()
