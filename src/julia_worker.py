"""Persistent worker that loads Julia-1 once and serves predictions over stdin/stdout.

Must run under the dedicated `.venv_julia` interpreter (transformers==5.0.0), e.g.:
    .venv_julia/bin/python3 -I src/julia_worker.py --checkpoint data/julia1_repo

Protocol: one JSON object per line on stdin -> one JSON object per line on stdout.
Request:  {"state": ..., "questions": {...}}
Response: {"ok": true, "answers": {...}}  or  {"ok": false, "error": "..."}
After loading the model, the worker writes a single line {"ready": true} to stdout.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--head-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    sys.path.insert(0, args.checkpoint)
    from julia import load_model

    engine = load_model(
        args.checkpoint,
        device=args.device,
        max_length=args.max_length,
        head_length=args.head_length,
        strict_encoding=True,
        backend="torch",
    )

    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("op") == "ping":
                print(json.dumps({"ok": True, "pong": True}), flush=True)
                continue
            result = engine.predict(state=request["state"], questions=request["questions"])
            print(json.dumps({"ok": True, "answers": result["answers"]}), flush=True)
        except Exception as exc:  # report and keep serving
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()
