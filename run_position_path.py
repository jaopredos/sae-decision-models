"""Discovery path test: nonmarkers at enc17 to markers at enc18."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from run_position import Runner, atomic_json
from run_position_factorial import full_snapshot
from src.position import choose_cases, conditions, effect_metrics, stable_seed


def branch_snapshot(runner,case,qid,question,order):
    block=runner.modules["enc_18"]
    saved,handles={},[]
    for name,module in (("attention",block.attn),("mlp",block.mlp)):
        def save(_m,_a,output,key=name):
            h=output[0] if isinstance(output,tuple) else output
            saved[key]=h[0].float().cpu().clone()
        handles.append(module.register_forward_hook(save))
    try:
        snapshot=full_snapshot(runner,case,qid,question,order)
    finally:
        for handle in handles: handle.remove()
    snapshot["branches"]={name:h[snapshot["alignment"]["indices"]] for name,h in saved.items()}
    return snapshot


def patch_hook(positions,values,selected):
    def edit(_m,_a,output):
        h=output[0] if isinstance(output,tuple) else output
        result=h.clone()
        physical=[positions[i] for i in selected]
        result[0,physical]=values[selected].to(h)
        return (result,)+tuple(output[1:]) if isinstance(output,tuple) else result
    return edit


def collect_path(runner,case,qid,n_null):
    torch=runner.torch
    q=next(conditions(case["questions"][qid],case["state"],case["state"],case["case_id"],qid))[1]
    k=len(q["criteria"])
    with torch.inference_mode():
        a=branch_snapshot(runner,case,qid,q,list(range(k)))
        b=branch_snapshot(runner,case,qid,q,list(range(1,k))+[0])
        runner.parity(case["state"],qid,q,a)
        if a["alignment"]["tokens"]!=b["alignment"]["tokens"] or a["alignment"]["groups"]!=b["alignment"]["groups"]:
            raise ValueError("path semantic alignment differs")
        records=[]
        for donor,recipient in ((a,b),(b,a)):
            positions=recipient["alignment"]["indices"]
            groups=recipient["alignment"]["groups"]
            clean=recipient["states"]["enc_17"]
            changed=clean.clone()
            changed[groups["nonmarkers"]]=donor["states"]["enc_17"][groups["nonmarkers"]]
            variants=[("self_patch",clean,None,None),
                      ("nonmarkers17",changed,None,None),
                      ("to_markers18",changed,"nonmarkers",None),
                      ("to_nonmarkers18",changed,"markers",None),
                      ("reset_all18",changed,"all_tokens",None),
                      ("to_markers18_clamp_attention",changed,"nonmarkers","attention"),
                      ("to_markers18_clamp_mlp",changed,"nonmarkers","mlp")]
            selected=groups["nonmarkers"]
            delta=changed[selected]-clean[selected]
            for repeat in range(n_null):
                g=torch.Generator().manual_seed(stable_seed("path-v1",case["case_id"],qid,donor["order"],repeat)%(2**63-1))
                noise=torch.randn(delta.shape,generator=g)
                noise*=delta.norm(dim=-1,keepdim=True)/noise.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                null=clean.clone();null[selected]+=noise
                variants.append((f"noise_to_markers18_{repeat}",null,"nonmarkers",None))
            for name,replacement,restore,branch in variants:
                handles=[runner.modules["enc_17"].register_forward_hook(patch_hook(positions,replacement,groups["all_tokens"]))]
                if branch:
                    module=runner.modules["enc_18"].attn if branch=="attention" else runner.modules["enc_18"].mlp
                    handles.append(module.register_forward_hook(patch_hook(positions,recipient["branches"][branch],groups["markers"])))
                if restore:
                    handles.append(runner.modules["enc_18"].register_forward_hook(patch_hook(positions,recipient["states"]["enc_18"],groups[restore])))
                try:
                    result=runner.run(case["state"],qid,q,recipient["order"])
                finally:
                    for handle in handles:handle.remove()
                error=max(abs(x-y) for x,y in zip(result["logits"],recipient["logits"]))
                if name in ("self_patch","reset_all18") and error>1e-5:
                    raise ValueError(f"path implementation control failed: {name}: {error}")
                records.append({"condition":name,"donor_order":donor["order"],"recipient_order":recipient["order"],
                                "donor_logits":donor["logits"],"recipient_logits":recipient["logits"],
                                "patched_logits":result["logits"],"patched_probabilities":result["probabilities"],
                                **effect_metrics(donor["logits"],recipient["logits"],result["logits"])})
        return {"case_id":case["case_id"],"qid":qid,"workflow":case["workflow"],"patches":records}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit",type=int,default=64)
    ap.add_argument("--n-null",type=int,default=4)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    if args.limit<1 or args.n_null<0:ap.error("invalid limits")
    import torch
    from src.activations import load_agent, LAYA_REVISION
    from src.datasets import load_typed_cases, typed_dev_ids, TYPED_DECISIONS_REVISION
    from src.models import set_all_seeds
    if not torch.cuda.is_available():raise RuntimeError("GPU job required")
    os.environ["LAYA_CUDA_AMP"]="fp32"
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    set_all_seeds(0)
    cases=choose_cases(load_typed_cases("train"),"discovery",typed_dev_ids(),args.limit)
    previous=Path("results/position_mechanism/factorial_v1")
    files=[Path(__file__),Path("run_position.py"),Path("run_position_factorial.py"),Path("src/position.py"),
           Path("src/position_factorial.py"),Path("src/activations.py"),Path("src/datasets.py"),Path("uv.lock"),
           Path("docs/position-mechanism/PATH_PROTOCOL.md")]
    manifest={"args":vars(args),"case_ids":[c["case_id"] for c in cases],"model_revision":LAYA_REVISION,
              "dataset_revision":TYPED_DECISIONS_REVISION,
              "factorial_manifest_sha256":hashlib.sha256((previous/"manifest.json").read_bytes()).hexdigest(),
              "files":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    out=Path(args.out)
    if (out/"manifest.json").exists():
        if json.loads((out/"manifest.json").read_text())!=manifest:raise ValueError("resume manifest changed")
    else:atomic_json(out/"manifest.json",manifest)
    runner=Runner(load_agent(device="cuda"))
    if runner.modules["enc_18"] is not runner.agent.model.encoder.layers[17]:
        raise ValueError("enc_18 does not map to encoder.layers[17]")
    atomic_json(out/"runtime.json",{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"precision":"FP32"})
    tasks=[(c,qid) for c in cases for qid,q in c["questions"].items() if q["type"]=="choice"]
    for i,(case,qid) in enumerate(tasks):
        digest=hashlib.sha256(json.dumps([case["case_id"],qid]).encode()).hexdigest()[:24]
        destination=out/"decisions"/f"{digest}.json"
        if destination.exists():continue
        print(f"[path] START {i+1}/{len(tasks)} case={case['case_id']} q={qid}",flush=True)
        started=time.monotonic()
        payload=collect_path(runner,case,qid,args.n_null)
        prior=json.loads((previous/"decisions"/destination.name).read_text())
        refs={tuple(r["donor_order"]):r for r in prior["patches"] if r["layer"]=="enc_17" and r["group"]=="nonmarkers" and r["intervention"]=="aligned"}
        for row in payload["patches"]:
            if row["condition"]=="nonmarkers17":
                ref=refs[tuple(row["donor_order"])]
                if max(abs(a-b) for field in ("donor_logits","recipient_logits","patched_logits") for a,b in zip(row[field],ref[field]))>1e-4:
                    raise ValueError("nonmarker transfer failed to reproduce factorial reference")
        payload["elapsed_s"]=time.monotonic()-started
        atomic_json(destination,payload)
        print(f"[path] DONE {i+1}/{len(tasks)} seconds={payload['elapsed_s']:.1f}",flush=True)
    atomic_json(out/"COMPLETE.json",{"cases":len(cases),"decisions":len(tasks),"manifest_sha256":hashlib.sha256((out/"manifest.json").read_bytes()).hexdigest()})
    print(f"[path] COMPLETE {out}",flush=True)


if __name__=="__main__":main()
