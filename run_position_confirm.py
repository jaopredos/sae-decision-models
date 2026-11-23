"""Prospective confirmation of the frozen path contrast; test accessed only as a GPU job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from run_position import Runner,atomic_json
from run_position_path import collect_path
from src.position import choose_cases


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset",choices=["discovery","test"],required=True)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--n-null",type=int,default=4)
    ap.add_argument("--protocol",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.limit<0 or args.n_null<0:ap.error("invalid limits")
    if not args.protocol.is_file():ap.error("frozen protocol missing")
    if args.subset=="test" and (args.limit!=0 or args.n_null!=4):
        ap.error("frozen test requires all cases and four null directions")
    import torch
    from src.activations import load_agent,LAYA_REVISION
    from src.datasets import load_typed_cases,typed_dev_ids,TYPED_DECISIONS_REVISION
    from src.models import set_all_seeds
    if not torch.cuda.is_available():raise RuntimeError("GPU job required")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    os.environ["LAYA_CUDA_AMP"]="fp32"
    set_all_seeds(0)
    population=load_typed_cases("test" if args.subset=="test" else "train")
    cases=choose_cases(population,args.subset,typed_dev_ids() if args.subset=="discovery" else set(),args.limit)
    tasks=[(c,qid) for c in cases for qid,q in c["questions"].items() if q["type"]=="choice"]
    if args.subset=="test" and (len(cases)!=400 or len(tasks)!=600):raise ValueError("test coverage differs from frozen protocol")
    paths=[Path(__file__),Path("run_position_path.py"),Path("run_position_factorial.py"),Path("run_position.py"),
           Path("src/position.py"),Path("src/position_factorial.py"),Path("src/activations.py"),Path("src/datasets.py"),Path("uv.lock"),args.protocol]
    manifest={"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "case_ids":[c["case_id"] for c in cases],"model_revision":LAYA_REVISION,"dataset_revision":TYPED_DECISIONS_REVISION,
              "files":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "protocol_sha256":hashlib.sha256(args.protocol.read_bytes()).hexdigest()}
    out=args.out
    if (out/"manifest.json").exists():
        if json.loads((out/"manifest.json").read_text())!=manifest:raise ValueError("resume manifest changed")
    else:atomic_json(out/"manifest.json",manifest)
    runner=Runner(load_agent(device="cuda"))
    if runner.modules["enc_18"] is not runner.agent.model.encoder.layers[17]:raise ValueError("block mapping differs")
    atomic_json(out/"runtime.json",{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
                "precision":"FP32; TF32 disabled","cpu_threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads()})
    for i,(case,qid) in enumerate(tasks):
        digest=hashlib.sha256(json.dumps([case["case_id"],qid]).encode()).hexdigest()[:24]
        target=out/"decisions"/f"{digest}.json"
        if target.exists():continue
        started=time.monotonic()
        print(f"[confirm] START {i+1}/{len(tasks)} case={case['case_id']} q={qid}",flush=True)
        payload=collect_path(runner,case,qid,args.n_null,validate_final=True)
        payload["elapsed_s"]=time.monotonic()-started
        if args.subset=="discovery":
            # Validate the new runtime and final-alignment control on discovery only.
            previous=json.loads((Path("results/position_mechanism/path_v1/decisions")/target.name).read_text())
            old={(r["condition"],tuple(r["donor_order"])):r for r in previous["patches"]}
            for row in payload["patches"]:
                if row["condition"].startswith("noise_"):continue
                ref=old[(row["condition"],tuple(row["donor_order"]))]
                if max(abs(a-b) for field in ("donor_logits","recipient_logits","patched_logits") for a,b in zip(row[field],ref[field]))>1e-4:
                    raise ValueError("new runtime does not reproduce discovery path")
        atomic_json(target,payload)
        print(f"[confirm] DONE {i+1}/{len(tasks)} seconds={payload['elapsed_s']:.1f}",flush=True)
    if len(list((out/"decisions").glob("*.json")))!=len(tasks):raise ValueError("incomplete confirmation")
    atomic_json(out/"COMPLETE.json",{"cases":len(cases),"decisions":len(tasks),"manifest_sha256":hashlib.sha256((out/"manifest.json").read_bytes()).hexdigest()})
    print(f"[confirm] COMPLETE {out}",flush=True)


if __name__=="__main__":main()
