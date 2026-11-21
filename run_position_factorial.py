"""Discovery-only full-token factorial interventions after marker replication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from run_position import Runner, atomic_json
from src.position import choose_cases, conditions, effect_metrics, stable_seed
from src.position_factorial import token_alignment


LAYERS = ("enc_17", "enc_18", "enc_final")


def full_snapshot(runner, case, qid, question, order):
    handles, states = [], {}
    for name in LAYERS:
        def save(_m, _args, output, key=name):
            h = output[0] if isinstance(output, tuple) else output
            states[key] = h[0].float().cpu().clone()
        handles.append(runner.modules[name].register_forward_hook(save))
    try:
        reference = runner.run(case["state"], qid, question, order)
    finally:
        for handle in handles:
            handle.remove()
    alignment = token_alignment(reference, runner.agent.tok.sep_token_id)
    reference["alignment"] = alignment
    reference["states"] = {name: h[alignment["indices"]].clone() for name,h in states.items()}
    return reference


def full_patch(runner, case, qid, question, recipient, layer, replacement):
    def edit(_m, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        new = h.clone()
        new[0, recipient["alignment"]["indices"]] = replacement.to(h)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
    handle = runner.modules[layer].register_forward_hook(edit)
    try:
        return runner.run(case["state"], qid, question, recipient["order"])
    finally:
        handle.remove()


def collect_decision(runner, case, qid, n_null):
    torch = runner.torch
    question = next(conditions(case["questions"][qid], case["state"], case["state"], case["case_id"], qid))[1]
    k = len(question["criteria"])
    original, rotated = list(range(k)), list(range(1,k))+[0]
    with torch.inference_mode():
        a = full_snapshot(runner, case, qid, question, original)
        b = full_snapshot(runner, case, qid, question, rotated)
        runner.parity(case["state"], qid, question, a)
        for field in ("tokens", "groups"):
            if a["alignment"][field] != b["alignment"][field]:
                raise ValueError(f"semantic token alignment differs: {field}")
        records = []
        for donor, recipient in ((a,b),(b,a)):
            for layer in LAYERS:
                clean = recipient["states"][layer]
                source = donor["states"][layer]
                delta = source-clean
                # enc_final all-token patch tests permutation-equivariance of the head.
                groups = recipient["alignment"]["groups"]
                variants = [("self_patch", "all_tokens", clean)]
                for group, selected in groups.items():
                    replacement = clean.clone()
                    replacement[selected] = source[selected]
                    variants.append(("aligned", group, replacement))
                    if layer != "enc_final":
                        for repeat in range(n_null):
                            generator = torch.Generator().manual_seed(stable_seed(
                                "factorial-v1",case["case_id"],qid,layer,donor["order"],group,repeat) % (2**63-1))
                            noise = torch.randn(delta[selected].shape, generator=generator)
                            noise *= delta[selected].norm(dim=-1,keepdim=True)/noise.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                            random_replacement = clean.clone()
                            random_replacement[selected] += noise
                            variants.append((f"noise_{repeat}", group, random_replacement))
                for intervention, group, replacement in variants:
                    result = full_patch(runner, case, qid, question, recipient, layer, replacement)
                    if intervention == "self_patch":
                        if max(abs(x-y) for x,y in zip(result["logits"],recipient["logits"])) > 1e-5:
                            raise ValueError("factorial self patch failed")
                    if layer == "enc_final" and group == "all_tokens" and intervention == "aligned":
                        if max(abs(x-y) for x,y in zip(result["logits"],donor["logits"])) > 1e-4:
                            raise ValueError("full-token alignment positive control failed")
                    records.append({"layer":layer, "group":group, "intervention":intervention,
                                    "donor_order":donor["order"], "recipient_order":recipient["order"],
                                    "donor_logits":donor["logits"], "recipient_logits":recipient["logits"],
                                    "patched_logits":result["logits"], "patched_probabilities":result["probabilities"],
                                    "tokens_edited":len(groups[group]),
                                    "total_delta_norm":float((replacement-clean).norm()),
                                    **effect_metrics(donor["logits"],recipient["logits"],result["logits"])})
        return {"case_id":case["case_id"], "qid":qid, "workflow":case["workflow"],
                "gold_id":str(case["gold"][qid]["label"]), "semantic_ids":list(question["criteria"]),
                "token_count":a["seq_len"], "marker_count":k,
                "reference": [{key:value for key,value in snapshot.items() if key not in ("states",)} for snapshot in (a,b)],
                "patches":records}


def validate_marker_reference(payload, root):
    digest=hashlib.sha256(json.dumps([payload["case_id"],payload["qid"]]).encode()).hexdigest()[:24]
    original=json.loads((root/"decisions"/f"{digest}.json").read_text())
    marker_rows={(r["layer"],tuple(r["donor_order"])):r for r in original["patches"]
                 if r["intervention"]=="all_markers"}
    for row in payload["patches"]:
        if row["layer"] in ("enc_17","enc_18") and row["intervention"]=="aligned" and row["group"]=="markers":
            old=marker_rows[(row["layer"],tuple(row["donor_order"]))]
            for field in ("donor_logits","recipient_logits","patched_logits"):
                if max(abs(a-b) for a,b in zip(old[field],row[field])) > 1e-4:
                    raise ValueError(f"factorial markers do not reproduce original collection: {field}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.limit < 1 or args.n_null < 0:
        ap.error("positive case limit and non-negative n-null required")
    import torch
    from src.activations import load_agent, LAYA_REVISION
    from src.datasets import load_typed_cases, typed_dev_ids, TYPED_DECISIONS_REVISION
    from src.models import set_all_seeds
    if not torch.cuda.is_available():
        raise RuntimeError("model inference must run as a Sauron GPU job")
    os.environ["LAYA_CUDA_AMP"]="fp32"
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    set_all_seeds(0)
    cases = choose_cases(load_typed_cases("train"),"discovery",typed_dev_ids(),args.limit)
    out=Path(args.out)
    protocol=Path("docs/position-mechanism/FACTORIAL_PROTOCOL.md")
    marker_root=Path("results/position_mechanism/collection_v1/discovery")
    code_files=[Path(__file__),Path("run_position.py"),Path("src/position.py"),Path("src/position_factorial.py"),
                Path("src/activations.py"),Path("src/datasets.py"),Path("uv.lock"),protocol]
    manifest={"args":vars(args),"case_ids":[c["case_id"] for c in cases],"model_revision":LAYA_REVISION,
              "dataset_revision":TYPED_DECISIONS_REVISION,
              "marker_collection_manifest_sha256":hashlib.sha256((marker_root/"manifest.json").read_bytes()).hexdigest(),
              "files":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}}
    if (out/"manifest.json").exists():
        if json.loads((out/"manifest.json").read_text()) != manifest:
            raise ValueError("resume manifest changed")
    else:
        atomic_json(out/"manifest.json",manifest)
    runner=Runner(load_agent(device="cuda"))
    atomic_json(out/"runtime.json",{"torch":torch.__version__,"cuda":torch.version.cuda,
                "gpu":torch.cuda.get_device_name(0),"precision":"FP32, TF32 disabled",
                "layers":list(LAYERS),"model_revision":LAYA_REVISION})
    tasks=[(c,qid) for c in cases for qid,q in c["questions"].items() if q["type"]=="choice"]
    for i,(case,qid) in enumerate(tasks):
        digest=hashlib.sha256(json.dumps([case["case_id"],qid]).encode()).hexdigest()[:24]
        destination=out/"decisions"/f"{digest}.json"
        if destination.exists():
            continue
        print(f"[factorial] START {i+1}/{len(tasks)} case={case['case_id']} q={qid}",flush=True)
        started=time.monotonic()
        payload=collect_decision(runner,case,qid,args.n_null)
        validate_marker_reference(payload,marker_root)
        payload["elapsed_s"]=time.monotonic()-started
        atomic_json(destination,payload)
        print(f"[factorial] DONE {i+1}/{len(tasks)} seconds={payload['elapsed_s']:.1f}",flush=True)
    atomic_json(out/"COMPLETE.json",{"cases":len(cases),"decisions":len(tasks),
                "manifest_sha256":hashlib.sha256((out/"manifest.json").read_bytes()).hexdigest()})
    print(f"[factorial] COMPLETE {out}",flush=True)


if __name__=="__main__":
    main()
