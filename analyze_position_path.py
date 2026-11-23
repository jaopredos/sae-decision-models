"""Offline analysis of the nonmarker17 -> marker18 path experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_position import projection_interval
from src.position import effect_metrics


def load_path(root,factorial=None):
    manifest=json.loads((root/"manifest.json").read_text())
    complete=json.loads((root/"COMPLETE.json").read_text())
    digest=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()
    if digest!=complete["manifest_sha256"]:raise ValueError("manifest hash mismatch")
    if factorial and hashlib.sha256((factorial/"manifest.json").read_bytes()).hexdigest()!=manifest["factorial_manifest_sha256"]:
        raise ValueError("factorial reference differs")
    files=sorted((root/"decisions").glob("*.json"))
    if len(files)!=complete["decisions"]:raise ValueError("incomplete decisions")
    rows=[];cases=set();decisions=set();errors={"self_patch":[],"reset_all18":[],"factorial_reference":[],"final_alignment":[]}
    conditions={"self_patch","nonmarkers17","to_markers18","to_nonmarkers18","reset_all18",
                "to_markers18_clamp_attention","to_markers18_clamp_mlp"}
    conditions.update(f"noise_to_markers18_{i}" for i in range(manifest["args"]["n_null"]))
    for file in files:
        data=json.loads(file.read_text());key=(data["case_id"],data["qid"])
        if key in decisions:raise ValueError("duplicate decision")
        cases.add(key[0]);decisions.add(key)
        if manifest["args"].get("subset")=="test" and data.get("final_alignment_control_max_error") is None:
            raise ValueError("missing test alignment control")
        if data.get("final_alignment_control_max_error") is not None:
            errors["final_alignment"].append(data["final_alignment_control_max_error"])
        expected={}
        if factorial:
            old=json.loads((factorial/"decisions"/file.name).read_text())
            expected={tuple(r["donor_order"]):r for r in old["patches"] if r["layer"]=="enc_17" and r["group"]=="nonmarkers" and r["intervention"]=="aligned"}
        seen=set();directions=set()
        for r in data["patches"]:
            donor=tuple(r["donor_order"]);directions.add(donor)
            cell=(r["condition"],donor)
            if cell in seen:raise ValueError("duplicate condition")
            seen.add(cell)
            measured=effect_metrics(r["donor_logits"],r["recipient_logits"],r["patched_logits"])
            for field in ("projection_numerator","contrast_sqnorm","change_norm","orthogonal_norm"):
                if not np.isclose(r[field],measured[field],rtol=1e-7,atol=1e-9):raise ValueError("stored effect differs")
            if r["condition"] in ("self_patch","reset_all18"):
                errors[r["condition"]].append(max(abs(a-b) for a,b in zip(r["patched_logits"],r["recipient_logits"])))
            if factorial and r["condition"]=="nonmarkers17":
                errors["factorial_reference"].append(max(abs(a-b) for field in ("donor_logits","recipient_logits","patched_logits") for a,b in zip(r[field],expected[donor][field])))
            p,d,base=map(np.asarray,(r["patched_logits"],r["donor_logits"],r["recipient_logits"]))
            rows.append({"case_id":key[0],"qid":key[1],"workflow":data["workflow"],
                         "direction":"canonical_to_rotated" if donor[0]==0 else "rotated_to_canonical",
                         "condition":"noise_to_markers18" if r["condition"].startswith("noise_to_markers18_") else r["condition"],
                         **measured,"pair_flip":int(d.argmax()!=base.argmax()),
                         "donor_choice_recovered":int(p.argmax()==d.argmax())})
        if len(directions)!=2 or seen!={(c,d) for c in conditions for d in directions}:
            raise ValueError("condition coverage differs")
    if cases!=set(manifest["case_ids"]) or len(cases)!=complete["cases"]:raise ValueError("case coverage differs")
    errors={k:max(v) for k,v in errors.items() if v}
    if errors["self_patch"]>1e-5 or errors["reset_all18"]>1e-5 or max(errors.get("factorial_reference",0),errors.get("final_alignment",0))>1e-4:
        raise ValueError("controls exceed tolerance")
    audit={"cases":len(cases),"decisions":len(decisions),"forward_rows":len(rows),"manifest_sha256":digest,"max_logit_errors":errors,
           "frozen_protocol_sha256":manifest.get("protocol_sha256")}
    return pd.DataFrame(rows),audit


def paired_contrast(frame,left,right,n_boot):
    keys=["case_id","qid","workflow","direction"]
    a=frame[frame.condition==left].set_index(keys).sort_index()
    b=frame[frame.condition==right].set_index(keys).sort_index()
    if not a.index.equals(b.index):raise ValueError("paired condition coverage differs")
    delta=a.copy();delta["projection_numerator"]=a.projection_numerator-b.projection_numerator
    delta=delta.reset_index()
    return {"left":left,"right":right,"estimate":float(delta.projection_numerator.sum()/delta.contrast_sqnorm.sum()),
            "ci":projection_interval(delta,n_boot)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--factorial-reference",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=2000)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data,audit=load_path(args.input,args.factorial_reference)
    numeric=["projection_numerator","contrast_sqnorm","transfer_ratio","change_norm","orthogonal_norm","pair_flip","donor_choice_recovered"]
    frame=data.groupby(["case_id","qid","workflow","direction","condition"])[numeric].mean().reset_index()
    summary=[]
    for name,part in frame.groupby("condition"):
        flips=part[part.pair_flip==1]
        summary.append({"condition":name,"projection":float(part.projection_numerator.sum()/part.contrast_sqnorm.sum()),
                        "ci":projection_interval(part,args.bootstrap),"median_ratio":float(part.transfer_ratio.median()),
                        "change_norm":float(part.change_norm.mean()),"orthogonal_norm":float(part.orthogonal_norm.mean()),
                        "donor_choice_on_flips":float(flips.donor_choice_recovered.mean()) if len(flips) else None})
    pairs=[("to_markers18","noise_to_markers18"),("to_markers18","to_markers18_clamp_mlp"),
           ("to_markers18","to_markers18_clamp_attention"),("nonmarkers17","to_markers18")]
    contrasts=[paired_contrast(frame,*pair,args.bootstrap) for pair in pairs]
    for name,value in (("audit",audit),("summary",summary),("contrasts",contrasts)):
        (args.out/f"{name}.json").write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    pd.DataFrame(summary).to_csv(args.out/"summary.csv",index=False)
    frame.to_csv(args.out/"by_decision.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels={"nonmarkers17":"Sem restauração", "to_markers18":"Só via marcadores 18",
            "to_nonmarkers18":"Só via demais tokens 18","to_markers18_clamp_mlp":"Via marcadores; MLP restaurada",
            "to_markers18_clamp_attention":"Via marcadores; atenção restaurada",
            "noise_to_markers18":"Ruído via marcadores"}
    table=pd.DataFrame(summary).set_index("condition").loc[list(labels)]
    values=table.projection.to_numpy();bounds=np.array(table.ci.tolist())
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    ax.barh(range(len(table)),values,color=["#236B8E","#579D80","#D89458","#8B79A8","#aaaaaa","#aaaaaa"])
    ax.errorbar(values,range(len(table)),xerr=[values-bounds[:,0],bounds[:,1]-values],fmt="none",color="#222",capsize=4)
    ax.set_yticks(range(len(table)),list(labels.values()));ax.invert_yaxis();ax.axvline(0,color="#aaa",linewidth=1)
    ax.set_xlabel("Projeção na mudança da execução doadora")
    ax.set_title(f"Caminho dos demais tokens 17 aos marcadores 18\n{audit['cases']} casos · IC95% por caso · atenção restaurada é controle estrutural")
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(args.out/"path.png",dpi=180);plt.close(fig)
    print(json.dumps(audit,indent=2));print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(contrasts,indent=2))


if __name__=="__main__":main()
