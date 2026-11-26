"""Do the top position-eta^2 features of the seed-0 SAE reappear in the seed-1 SAE? (PROTOCOLO.md, emenda 8)

Usage: python run_position_seed.py --layer enc_17 --a ml_enc17_k32 --b ml_enc17_k32_s1 [--ks 5,20]
For the K features of seed 0 with the largest position eta^2 it counts how many have a decoder-matched feature (mutual nearest
neighbour, cosine >= 0.7) that also belongs to the top-100 position list of seed 1.
"""
import argparse
import json
from pathlib import Path

from run_part_c_compare import PAIR_COS, pair_by_decoder
from src.sae import load_sae

REPO_ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ks", default="5,20")
    ap.add_argument("--base", default=str(REPO_ROOT / "results" / "sae_features" / "position_causal"))
    ap.add_argument("--sae-dir", default=str(REPO_ROOT / "checkpoints" / "sae"))
    args = ap.parse_args()
    la = json.loads((Path(args.base) / f"features_{args.a}" / "selected_features.json").read_text())["pos"]
    lb = json.loads((Path(args.base) / f"features_{args.b}" / "selected_features.json").read_text())["pos"]
    perm, _ = pair_by_decoder(load_sae(str(Path(args.sae_dir) / args.a / "sae.pt"), "cpu"), load_sae(str(Path(args.sae_dir) / args.b / "sae.pt"), "cpu"), PAIR_COS)
    top_b, res = set(lb[:100]), {"layer": args.layer, "seed0": args.a, "seed1": args.b}
    for k in [int(x) for x in args.ks.split(",")]:
        m = [int(perm[f]) for f in la[:k]]
        res[f"K{k}"] = {"matched": int(sum(x >= 0 for x in m)), "matched_and_in_seed1_top100_position": int(sum(x in top_b for x in m)), "fraction": sum(x in top_b for x in m) / k, "same_index_in_top100": int(len(set(la[:k]) & top_b))}
    Path(args.base, f"seed_overlap_{args.layer}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
