"""Identity-aligned order diagnostics and causal marker patching, without SAEs.

Run ONLY inside a GPU job. See docs/position-mechanism/PROTOCOL.md.
Each decision is committed atomically, so interrupted jobs can resume safely.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from src.position import (align_slots, choose_cases, conditions, donor_state_map,
                          effect_metrics, orders, stable_seed)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def fingerprint():
    paths = [Path(__file__), Path("src/position.py"), Path("src/activations.py"),
             Path("src/datasets.py"), Path("pyproject.toml"), Path("uv.lock")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Runner:
    def __init__(self, agent):
        import torch
        from src.activations import layer_modules
        self.torch = torch
        self.agent = agent
        # CUDA can autocast even one question. Do not rely on the historical MPS rule.
        self.agent.amp_enabled = False
        self.agent.dtype = torch.float32
        self.agent.model.float()
        if self.agent.dtype_for(1) != torch.float32 or self.agent._amp_enabled_for(1):
            raise ValueError("FP32 precision policy was not applied")
        self.modules = dict(layer_modules(agent.model))
        self.layers = [name for name in self.modules if name != "emb"]
        # A protocol change is required if the checkpoint already uses parallel options.
        if getattr(agent, "parallel_options", False):
            raise ValueError("expected sequential checkpoint; found parallel option layout")

    def run(self, state, qid, question, order, keep=False, edit=None):
        from src.activations import capture, _options_over_cap
        torch = self.torch
        q = copy.deepcopy(question)
        q["option_order"] = list(order)
        handles = []
        if edit is not None:
            layer, positions, replacement = edit

            def patch(_module, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.ndim != 3 or h.shape[0] != 1:
                    raise ValueError("expected one question row and full token states")
                out = h.clone()
                for identity, pos in enumerate(positions):
                    out[0, pos] = replacement[identity].to(h)
                return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

            handles.append(self.modules[layer].register_forward_hook(patch))
        try:
            with torch.inference_mode(), capture(self.agent, list(self.modules.items()) if keep else []) as cap:
                answers = self.agent.predict(state, {qid: q}, max_len=1024, head_max_len=256)["answers"][qid]
            if len(cap.forward_calls) != 1 or len(cap.encode_calls) != 1:
                raise ValueError("unexpected batching or multiple forwards")
            args, output = cap.forward_calls[0]
            _, internal, items = cap.encode_calls[0]
            item = items[0]
            k = len(order)
            positions_slot = [int(v) for v in args[2][0, :k].tolist()]
            if positions_slot != item["markers"] or int(args[3][0].sum()) != k:
                raise ValueError("marker alignment failed")
            positions = align_slots(positions_slot, order)
            logits = align_slots(output[0][0, :k].float().cpu().tolist(), order)
            ids = list(q["criteria"])
            probabilities = [float(answers["probabilities"][key]) for key in ids]
            if any(not __import__("math").isfinite(v) for v in logits + probabilities):
                raise ValueError("non-finite model output")
            if item["state_stats"]["truncated"] or item["options"]["tokens_per_option"] is not None:
                raise ValueError("truncated input; stop rather than confound order with missing content")
            if item["options"]["options_distinct"] != k or _options_over_cap(self.agent, internal[qid]):
                raise ValueError("option truncation/collapse; protocol requires intact alternatives")
            # Token spans include each option marker and its own description, never its neighbour.
            tokens = list(item["ids"])
            last_end = next(p for p in range(positions_slot[-1]+1, len(tokens))
                            if tokens[p] == self.agent.tok.sep_token_id)
            ends = positions_slot[1:] + [last_end]
            spans = align_slots([tokens[start:end] for start, end in zip(positions_slot, ends)], order)
            result = {"order": list(order), "ids": ids, "positions": positions,
                      "logits": logits, "probabilities": probabilities,
                      "pred_index": max(range(k), key=lambda i: logits[i]),
                      "official_pred_index": ids.index(answers["choice"]),
                      "seq_len": len(tokens), "option_spans": spans,
                      "question_tokens": tokens[:positions_slot[0]],
                      "state_tokens": tokens[last_end+1:],
                      "option_lengths": [len(s) for s in spans]}
            if edit is not None and positions != list(edit[1]):
                raise ValueError("patch recipient marker positions changed")
            if keep:
                result["hidden"] = {
                    name: h[0, positions].float().cpu().clone() for name, h in cap.outputs.items()
                }
            return result
        finally:
            for h in handles:
                h.remove()

    def parity(self, state, qid, q, snapshot):
        # Compare unhooked official predict with the hooked reference (same FP32 single row).
        with self.torch.inference_mode():
            ans = self.agent.predict(state, {qid: q}, max_len=1024, head_max_len=256)["answers"][qid]
        probs = [float(ans["probabilities"][key]) for key in snapshot["ids"]]
        error = max(abs(a-b) for a,b in zip(probs, snapshot["probabilities"]))
        if error > 0.00011 or ans["choice"] != snapshot["ids"][snapshot["official_pred_index"]]:
            raise ValueError(f"official/hooked parity failed: {error}")
        return error


def behavioral(runner, case, qid, swapped):
    records, originals = [], {}
    for name, q, state, keys in conditions(case["questions"][qid], case["state"], swapped["state"],
                                         case["case_id"], qid):
        reference = None
        for order in orders(len(keys)):
            snapshot = runner.run(state, qid, q, order, keep=name == "original" and
                                  order in (list(range(len(keys))), list(range(1, len(keys)))+[0]))
            if reference is None:
                reference = snapshot
                snapshot["parity_max_probability_error"] = runner.parity(state, qid, q, snapshot)
            else:
                for field in ("option_spans", "question_tokens", "state_tokens"):
                    if snapshot[field] != reference[field]:
                        raise ValueError(f"reordering altered semantic token spans: {field}")
            if "hidden" in snapshot:
                originals[tuple(order)] = snapshot
            record = {k: v for k,v in snapshot.items() if k != "hidden"}
            record.update(condition=name, state_donor_case_id=swapped["case_id"] if name == "swapped_state" else None)
            records.append(record)
    return records, originals


def patching(runner, case, qid, originals, n_null):
    torch = runner.torch
    original_q = next(conditions(case["questions"][qid], case["state"], case["state"], case["case_id"], qid))[1]
    k = len(original_q["criteria"])
    base = tuple(range(k))
    rotated = tuple(list(range(1, k)) + [0])
    target = stable_seed("target", case["case_id"], qid) % k
    records = []
    for donor_key, recipient_key in ((base, rotated), (rotated, base)):
        donor, recipient = originals[donor_key], originals[recipient_key]
        for layer in runner.layers:
            source = donor["hidden"][layer]
            clean = recipient["hidden"][layer]
            delta = source - clean
            edits = [("same_order_noop", clean), ("all_markers", source)]
            single = clean.clone()
            single[target] = source[target]
            edits.append(("single_marker", single))
            edits.append(("wrong_identity_all", source.roll(1, dims=0)))
            for repeat in range(n_null):
                generator = torch.Generator().manual_seed(stable_seed(case["case_id"], qid, donor_key,
                                                                       layer, repeat) % (2**63-1))
                noise = torch.randn(delta.shape, generator=generator)
                noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                noise *= delta.norm(dim=-1, keepdim=True)
                edits.append((f"norm_matched_noise_{repeat}", clean + noise))
            for intervention, replacement in edits:
                patched = runner.run(case["state"], qid, original_q, recipient_key,
                                     edit=(layer, recipient["positions"], replacement))
                error = max(abs(a-b) for a,b in zip(patched["logits"], recipient["logits"]))
                if intervention == "same_order_noop" and error > 1e-5:
                    raise ValueError(f"same-order patch changes logits: {layer} {error}")
                # Final marker patch must transfer all scores; this is a positive plumbing control,
                # not evidence of where positional effects originate.
                if layer == runner.layers[-1] and intervention == "all_markers":
                    if max(abs(a-b) for a,b in zip(patched["logits"], donor["logits"])) > 1e-4:
                        raise ValueError("final-layer all-marker positive control failed")
                records.append({"layer": layer, "intervention": intervention,
                                "donor_order": list(donor_key), "recipient_order": list(recipient_key),
                                "target_identity_index": target,
                                "donor_logits": donor["logits"], "recipient_logits": recipient["logits"],
                                "patched_logits": patched["logits"],
                                "patched_probabilities": patched["probabilities"],
                                "delta_norm_per_marker": (replacement-clean).norm(dim=-1).tolist(),
                                **effect_metrics(donor["logits"], recipient["logits"], patched["logits"])})
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", choices=["discovery", "dev", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="case limit; 0 means all")
    ap.add_argument("--patch-cases", type=int, default=64)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confirmatory-protocol", help="required frozen follow-up protocol for test")
    args = ap.parse_args()
    if args.subset == "test":
        if not args.confirmatory_protocol or not Path(args.confirmatory_protocol).is_file():
            ap.error("test is gated until a frozen follow-up protocol exists")
    if min(args.limit, args.patch_cases, args.n_null) < 0:
        ap.error("limits must be non-negative")
    import torch
    from src.activations import LAYA_REPO, LAYA_REVISION, LAYA_SUBFOLDER, load_agent
    from src.datasets import TYPED_DECISIONS_REVISION, load_typed_cases, typed_dev_ids
    from src.models import set_all_seeds
    if not torch.cuda.is_available():
        raise RuntimeError("model execution is restricted to a Sauron GPU job")
    set_all_seeds(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Hooks must not silently switch TransformerEncoderLayer between fused and unfused paths.
    torch.backends.mha.set_fastpath_enabled(False)
    cases = load_typed_cases("test" if args.subset == "test" else "train")
    dev_ids = typed_dev_ids()
    # State donors never cross discovery/dev/test boundaries, even for limited pilot runs.
    population = choose_cases(cases, args.subset, dev_ids)
    donors = donor_state_map(population)
    selected = population[:args.limit] if args.limit else population
    out = Path(args.out) / args.subset
    manifest = {"schema": 1, "args": vars(args), "code": fingerprint(),
                "model": {"repo": LAYA_REPO, "revision": LAYA_REVISION, "subfolder": LAYA_SUBFOLDER},
                "dataset_revision": TYPED_DECISIONS_REVISION,
                "case_ids": [c["case_id"] for c in selected],
                "confirmatory_protocol_hash": (hashlib.sha256(Path(args.confirmatory_protocol).read_bytes()).hexdigest()
                                                if args.confirmatory_protocol else None)}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("resume manifest differs; use a new output directory")
    else:
        atomic_json(manifest_path, manifest)
    # Avoid overwriting a completed run, or loading the model when everything is already done.
    tasks = [(i,c,qid) for i,c in enumerate(selected) for qid,q in c["questions"].items() if q["type"] == "choice"]
    def task_path(case, qid):
        digest = hashlib.sha256(json.dumps([case["case_id"], qid]).encode()).hexdigest()[:24]
        return out / "decisions" / f"{digest}.json"
    todo = [(i,c,q) for i,c,q in tasks if not task_path(c,q).exists()]
    print(f"[position] subset={args.subset} cases={len(selected)} decisions={len(tasks)} remaining={len(todo)}", flush=True)
    if todo:
        os.environ["LAYA_CUDA_AMP"] = "fp32"
        runner = Runner(load_agent(device="cuda"))
        atomic_json(out / "runtime.json", {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "layers": runner.layers, "agent_config": runner.agent.cfg,
                    "precision": "single-question FP32; TF32 disabled", "seed": 0})
        for index,(case_index,case,qid) in enumerate(todo):
            started = time.monotonic()
            print(f"[position] START case={case['case_id']} q={qid} patch={case_index < args.patch_cases}", flush=True)
            observations, snapshots = behavioral(runner, case, qid, donors[case["case_id"]])
            patches = patching(runner, case, qid, snapshots, args.n_null) if case_index < args.patch_cases else []
            activation_path = None
            if case_index < args.patch_cases:
                destination = out / "activations" / task_path(case, qid).with_suffix(".pt").name
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".pt.tmp")
                with temporary.open("wb") as stream:
                    torch.save({"case_id": case["case_id"], "qid": qid,
                                "orders": [{"order": list(order), "positions": snapshot["positions"],
                                            "hidden": snapshot["hidden"]}
                                           for order,snapshot in snapshots.items()]}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                activation_path = str(destination.relative_to(out))
            atomic_json(task_path(case,qid), {"case_id": case["case_id"], "workflow": case["workflow"],
                        "qid": qid, "gold_id": str(case["gold"][qid]["label"]),
                        "semantic_ids": list(next(conditions(case["questions"][qid], case["state"],
                                                              case["state"], case["case_id"], qid))[1]["criteria"]),
                        "behavior": observations, "patches": patches, "activations": activation_path,
                        "elapsed_s": time.monotonic()-started})
            print(f"[position] {index+1}/{len(todo)} case={case['case_id']} q={qid} "
                  f"orders={len(observations)} patches={len(patches)} seconds={time.monotonic()-started:.1f}", flush=True)
    missing = [str(task_path(c,q)) for _,c,q in tasks if not task_path(c,q).exists()]
    if missing:
        raise RuntimeError(f"incomplete collection: {missing[:3]}")
    atomic_json(out / "COMPLETE.json", {"decisions": len(tasks), "cases": len(selected),
                                       "subset": args.subset, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
    print(f"[position] COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
