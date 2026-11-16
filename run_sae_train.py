"""Stage 4, step 2: train a TopK SAE on one layer of a Laya encoder.

Usage: python run_sae_train.py --layer enc_17 --name ml_enc17_k32 [--total-tokens 50000000] [--domain-frac 0.1]
           [--k 32] [--width-mult 16] [--device cuda] [--corpus data/corpus/text.parquet]
           [--domain-dir data/activations/tokens/multilingual] [--out checkpoints/sae]
           [--repo R --subfolder S --revision V] [--resume]

Training data per batch (results/sae_features/PROTOCOLO.md): `domain-frac` of the tokens come from the
stored all-token activations of the typed-decisions train prompts without the dev cases (drawn at
random, so ~3 passes over them in 50M tokens), the rest from generic text that is pushed through the
encoder while training (truncated at the SAE layer, same autocast as `predict`) into a GPU shuffle
buffer that is half-refreshed and reshuffled whenever half of it has been used. Padding is never
used. The dev cases and the held-out generic documents are only used for evaluation.
"""
import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import corpus as C
from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER, layer_output_fn, load_agent
from src.sae import TopKSAE, pooled_fve, save_sae
from src.sae_data import load_tokens, massive_threshold, token_is_dev

REPO_ROOT = Path(__file__).resolve().parent


def domain_tensors(domain_dir: Path, label: str, device, smoke: bool, mult: float):
    """fp16 GPU tensors (fit tokens, dev tokens) of the typed-decisions train prompts, without the
    massive tokens (norm > threshold), plus (threshold, median norm, stats)."""
    acts, ti, rows = load_tokens(domain_dir / "train", label)
    dev = token_is_dev(ti, rows)
    if smoke and not dev.any():  # tiny subsets may hold no hash-dev case
        dev = rows["decision_idx"].to_numpy()[ti["decision_idx"].to_numpy()] >= len(rows) - 15
    thr, med = (massive_threshold(acts, ti, rows, mult) if np.isfinite(mult) else (float("inf"), float("nan")))
    stats = {"threshold": thr, "median_norm": med}

    def take(mask, name):
        idx = np.flatnonzero(mask)
        parts = []
        for a in range(0, len(idx), 200_000):
            x = torch.from_numpy(np.asarray(acts[idx[a:a + 200_000]]))
            parts.append(x[x.float().norm(dim=1) <= thr])
        out = torch.cat(parts)
        stats[f"{name}_tokens_total"], stats[f"{name}_tokens_kept"] = len(idx), len(out)
        return out.to(device)
    return take(~dev, "fit"), take(dev, "dev"), thr, stats


class GenericBuffer:
    """GPU shuffle buffer of generic-text activations, half-refreshed when half is used."""

    def __init__(self, fwd, store, rows_per_fwd, size, hidden, device, seed, threshold=float("inf")):
        self.fwd, self.device, self.threshold = fwd, device, threshold
        self.tokens_massive = 0
        self.it = C.batches(store, rows_per_fwd, seed)
        self.buf = torch.empty(size, hidden, dtype=torch.float16, device=device)
        self.carry = torch.empty(0, hidden, dtype=torch.float16, device=device)
        self.size, self.used, self.gen = size, 0, torch.Generator(device="cpu").manual_seed(seed)
        self.tokens_forwarded = 0
        self._fill(0, size)
        self._shuffle()

    def _fresh(self, n):
        parts, have = [self.carry], self.carry.shape[0]
        while have < n:
            ids, mask, _ = next(self.it)
            ids_t, mask_t = torch.from_numpy(ids).long(), torch.from_numpy(mask)
            h = self.fwd(ids_t, mask_t).to(self.device)
            h = h[mask_t.to(self.device)].to(torch.float16)
            self.tokens_forwarded += h.shape[0]
            ok = h.float().norm(dim=1) <= self.threshold
            self.tokens_massive += int((~ok).sum())
            h = h[ok]
            parts.append(h); have += h.shape[0]
        cat = torch.cat(parts)
        self.carry = cat[n:]
        return cat[:n]

    def _fill(self, a, b):
        self.buf[a:b] = self._fresh(b - a)

    def _shuffle(self):
        self.buf = self.buf[torch.randperm(self.size, generator=self.gen).to(self.device)]
        self.used = 0

    def take(self, n):
        out = self.buf[self.used:self.used + n]
        self.used += n
        if self.used >= self.size // 2:
            self._fill(0, self.used)
            self._shuffle()
        return out


@torch.no_grad()
def val_activations(fwd, store, rows_per_fwd, device, max_tokens, threshold=float("inf")):
    it = C.batches(store, rows_per_fwd, seed=1)
    out, n, src = [], 0, []
    while n < max_tokens:
        ids, mask, source = next(it)
        h = fwd(torch.from_numpy(ids).long(), torch.from_numpy(mask))
        m = torch.from_numpy(mask).to(h.device)
        h = h[m].to(torch.float16).to(device)
        s_tok = torch.from_numpy(np.repeat(source, mask.sum(1))).to(device)
        ok = h.float().norm(dim=1) <= threshold
        out.append(h[ok]); src.append(s_tok[ok])
        n += out[-1].shape[0]
    return torch.cat(out)[:max_tokens], torch.cat(src)[:max_tokens]


@torch.no_grad()
def eval_fve(sae, x, chunk=50_000):
    return pooled_fve([sae.fve_stats(x[a:a + chunk]) for a in range(0, len(x), chunk)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--total-tokens", type=int, default=50_000_000)
    ap.add_argument("--domain-frac", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k-aux", type=int, default=512)
    ap.add_argument("--width-mult", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dead-window", type=int, default=5_000_000)
    ap.add_argument("--massive-mult", type=float, default=5.0,
                    help="exclude tokens with norm > mult x median (PROTOCOLO emenda 1); inf = literal all-token run")
    ap.add_argument("--buffer", type=int, default=2_000_000)
    ap.add_argument("--rows-per-fwd", type=int, default=64)
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--corpus", default=str(REPO_ROOT / "data" / "corpus" / "text.parquet"))
    ap.add_argument("--domain-dir", default=str(REPO_ROOT / "data" / "activations" / "tokens" / "multilingual"))
    ap.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "sae"))
    ap.add_argument("--repo", default=LAYA_REPO)
    ap.add_argument("--subfolder", default=LAYA_SUBFOLDER)
    ap.add_argument("--revision", default=LAYA_REVISION)
    ap.add_argument("--sources", default="", help="comma list of corpus sources to use (default: all), e.g. en")
    ap.add_argument("--init-sae", default=None, help="warm start from this SAE (keeps its scale; the massive threshold is recomputed for this model)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    subfolder = None if args.subfolder in ("", "none", "None") else args.subfolder

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = load_agent(args.device, args.repo, subfolder, args.revision)
    device = agent.device
    fwd = layer_output_fn(agent, args.layer)
    hidden = agent.model.encoder.config.hidden_size
    n_latents = args.width_mult * hidden

    dom_fit, dom_dev, thr, dom_stats = domain_tensors(Path(args.domain_dir), args.layer, device, args.smoke,
                                                      args.massive_mult)
    df = pd.read_parquet(args.corpus)
    if args.sources:
        df = df[df["source"].isin(args.sources.split(","))]
    n_dom = int(round(args.batch * args.domain_frac))
    n_gen = args.batch - n_dom
    total_steps = math.ceil(args.total_tokens / args.batch)
    generic_tokens = int(total_steps * n_gen)
    srcs = sorted(df["source"].unique())
    langs = [s for s in srcs if s != "en"]
    budgets = {"en": generic_tokens // 2 if langs else generic_tokens}
    budgets.update({s: generic_tokens // 2 // max(1, len(langs)) for s in langs})
    train_store = C.build_store(agent.tok, df, budgets, val=False, seed=args.seed)
    val_store = C.build_store(agent.tok, df, {}, val=True, seed=args.seed)
    print(f"[sae] layer={args.layer} hidden={hidden} latents={n_latents} k={args.k} steps={total_steps} "
          f"batch={args.batch} (domain {n_dom}/generic {n_gen}) device={device}", flush=True)
    print(f"[sae] domain fit tokens {len(dom_fit):,} dev tokens {len(dom_dev):,} | generic store "
          f"{train_store.n_tokens:,} tokens in {len(train_store.lengths):,} chunks, val {val_store.n_tokens:,}", flush=True)

    print(f"[sae] massive threshold {thr:.1f} (median norm {dom_stats['median_norm']:.1f}); domain tokens kept "
          f"fit {dom_stats['fit_tokens_kept']:,}/{dom_stats['fit_tokens_total']:,} dev "
          f"{dom_stats['dev_tokens_kept']:,}/{dom_stats['dev_tokens_total']:,}", flush=True)
    val_gen, val_src = val_activations(fwd, val_store, args.rows_per_fwd, device, args.val_tokens, thr)
    buf = GenericBuffer(fwd, train_store, args.rows_per_fwd, args.buffer, hidden, device, args.seed, thr)

    sae = TopKSAE(hidden, n_latents, k=args.k, k_aux=args.k_aux, seed=args.seed).to(device)
    sae.massive_threshold.fill_(thr)
    start_step = 0
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    ck_path = out_dir / "checkpoint.pt"
    if args.resume and ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        sae.load_state_dict(ck["sae"]); opt.load_state_dict(ck["opt"]); start_step = ck["step"]
        print(f"[sae] resumed at step {start_step} (data stream restarts; not bit-identical)", flush=True)
    elif args.init_sae:
        from src.sae import load_sae
        init = load_sae(args.init_sae, str(device))
        assert init.d_in == hidden and init.n_latents == n_latents
        sae.load_state_dict(init.state_dict())
        sae.massive_threshold.fill_(thr)  # the rule is per model: threshold from THIS model's domain tokens
        print(f"[sae] warm start from {args.init_sae} (keeps its scale {float(sae.scale):.5f}; threshold set to {thr:.1f})", flush=True)
    else:
        first = torch.cat([dom_fit[torch.randint(0, len(dom_fit), (20_000,), device=device)],
                           buf.buf[:80_000]]).float()
        print(f"[sae] scale={sae.fit_scale(first):.5f}", flush=True)
        sae.init_b_dec(first)

    gdom = torch.Generator(device="cpu").manual_seed(args.seed + start_step)
    log_path = out_dir / "train_log.csv"
    log_f = open(log_path, "a" if start_step else "w", newline="")
    log = csv.writer(log_f)
    if not start_step:
        log.writerow(["step", "tokens", "main", "aux", "n_dead", "fve_domain_dev", "fve_generic_val", "tok_per_s"])
    t0, t_log = time.time(), time.time()
    for step in range(start_step, total_steps):
        parts = [buf.take(n_gen)]
        if n_dom:
            parts.append(dom_fit[torch.randint(0, len(dom_fit), (n_dom,), generator=gdom).to(device)])
        x = torch.cat(parts).float()
        out = sae.training_loss(x, args.dead_window)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()
        done = step + 1
        if done % 200 == 0 or done == total_steps:
            rate = 200 * args.batch / max(1e-9, time.time() - t_log); t_log = time.time()
            ev = (eval_fve(sae, dom_dev), eval_fve(sae, val_gen)) if (done % args.eval_every == 0 or done == total_steps) else ("", "")
            log.writerow([done, done * args.batch, f"{out['main']:.5f}", f"{out['aux']:.5f}", int(out["n_dead"]), *ev, f"{rate:.0f}"])
            log_f.flush()
            print(f"[sae] step {done}/{total_steps} main={out['main']:.4f} aux={out['aux']:.3f} dead={int(out['n_dead'])} "
                  f"fve_dev={ev[0] if ev[0] == '' else f'{ev[0]:.4f}'} fve_gen={ev[1] if ev[1] == '' else f'{ev[1]:.4f}'} "
                  f"{rate:,.0f} tok/s | {time.time() - t0:.0f}s", flush=True)
        if done % args.ckpt_every == 0:
            torch.save({"sae": sae.state_dict(), "opt": opt.state_dict(), "step": done}, ck_path)
    log_f.close()

    import laya
    import transformers
    meta = {"args": vars(args), "hidden": hidden, "n_latents": n_latents, "scale": float(sae.scale),
            "steps": total_steps, "generic_tokens_forwarded": buf.tokens_forwarded,
            "generic_store_tokens": train_store.n_tokens,
            "generic_tokens_by_source": {s: int(train_store.lengths[train_store.source == i].sum())
                                         for i, s in enumerate(train_store.sources)},
            "domain_fit_tokens": len(dom_fit), "domain_dev_tokens": len(dom_dev),
            "final_fve_domain_dev": eval_fve(sae, dom_dev), "final_fve_generic_val": eval_fve(sae, val_gen),
            "versions": {"python": platform.python_version(), "torch": torch.__version__, "laya": laya.__version__,
                         "transformers": transformers.__version__},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "elapsed_s": round(time.time() - t0, 1)}
    save_sae(sae, str(out_dir / "sae.pt"), extra=meta)
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[sae] saved {out_dir / 'sae.pt'} | FVE domain-dev {meta['final_fve_domain_dev']:.4f} "
          f"generic-val {meta['final_fve_generic_val']:.4f}", flush=True)


if __name__ == "__main__":
    main()
