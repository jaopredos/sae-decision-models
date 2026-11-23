"""Offline, case-clustered analysis of the token-group factorial collection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_position import projection_interval
from src.position import effect_metrics


def interaction_metrics(donor, recipient, markers, nonmarkers, all_tokens):
    r=np.asarray(recipient,dtype=float)
    interaction=np.asarray(all_tokens)-np.asarray(markers)-np.asarray(nonmarkers)+r
    return effect_metrics(donor,recipient,(r+interaction).tolist())


def load_factorial(root, marker_root):
    manifest=json.loads((root/"manifest.json").read_text())
    complete=json.loads((root/"COMPLETE.json").read_text())
    digest=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()
    if digest!=complete["manifest_sha256"]:
        raise ValueError("factorial manifest hash differs")
    if hashlib.sha256((marker_root/"manifest.json").read_bytes()).hexdigest()!=manifest["marker_collection_manifest_sha256"]:
        raise ValueError("marker reference manifest differs")
    paths=sorted((root/"decisions").glob("*.json"))
    if len(paths)!=complete["decisions"]:
        raise ValueError("incomplete decision collection")
    rows=[]
    cases,keys=set(),set()
    errors={"self_patch":[],"final_all_tokens":[],"marker_replication":[]}
    expected_rows=2*(2*(1+3*(1+manifest["args"]["n_null"]))+4)
    for path in paths:
        data=json.loads(path.read_text())
        key=(data["case_id"],data["qid"])
        if key in keys:
            raise ValueError("duplicate decision")
        keys.add(key);cases.add(key[0])
        if len(data["patches"])!=expected_rows:
            raise ValueError("incomplete factorial cells")
        original=json.loads((marker_root/"decisions"/path.name).read_text())
        old={(r["layer"],tuple(r["donor_order"])):r for r in original["patches"] if r["intervention"]=="all_markers"}
        refs={tuple(r["order"]):r for r in data["reference"]}
        actual={}
        for record in data["patches"]:
            row=dict(record)
            direction="canonical_to_rotated" if row["donor_order"][0]==0 else "rotated_to_canonical"
            meta={"case_id":key[0],"qid":key[1],"workflow":data["workflow"],"direction":direction}
            cell=(row["layer"],row["group"],row["intervention"],tuple(row["donor_order"]))
            if cell in actual:
                raise ValueError("duplicate factorial cell")
            actual[cell]=row
            if row["recipient_logits"]!=refs[tuple(row["recipient_order"])]["logits"] or row["donor_logits"]!=refs[tuple(row["donor_order"])]["logits"]:
                raise ValueError("reference logits differ")
            recalculated=effect_metrics(row["donor_logits"],row["recipient_logits"],row["patched_logits"])
            for field in ("contrast_sqnorm","projection_numerator","change_norm","orthogonal_norm"):
                if not np.isclose(row[field],recalculated[field],rtol=1e-7,atol=1e-9):
                    raise ValueError(f"stored metric differs: {field}")
            if row["intervention"]=="self_patch":
                errors["self_patch"].append(max(abs(a-b) for a,b in zip(row["patched_logits"],row["recipient_logits"])))
            if row["intervention"]=="aligned" and row["group"]=="all_tokens" and row["layer"]=="enc_final":
                errors["final_all_tokens"].append(max(abs(a-b) for a,b in zip(row["patched_logits"],row["donor_logits"])))
            if row["intervention"]=="aligned" and row["group"]=="markers" and row["layer"]!="enc_final":
                previous=old[(row["layer"],tuple(row["donor_order"]))]
                errors["marker_replication"].append(max(abs(a-b) for field in ("donor_logits","recipient_logits","patched_logits") for a,b in zip(row[field],previous[field])))
            donor,recipient,patched=map(np.asarray,(row["donor_logits"],row["recipient_logits"],row["patched_logits"]))
            row.update(meta,treatment="noise" if row["intervention"].startswith("noise_") else row["intervention"],
                       pair_flip=int(donor.argmax()!=recipient.argmax()),
                       donor_choice_recovered=int(patched.argmax()==donor.argmax()))
            rows.append(row)
        for layer in ("enc_17","enc_18","enc_final"):
            for direction in refs:
                groups={g:actual[(layer,g,"aligned",direction)] for g in ("markers","nonmarkers","all_tokens")}
                base=groups["markers"]
                metrics=interaction_metrics(base["donor_logits"],base["recipient_logits"],
                                            *[groups[g]["patched_logits"] for g in ("markers","nonmarkers","all_tokens")])
                rows.append({"case_id":key[0],"qid":key[1],"workflow":data["workflow"],
                             "direction":"canonical_to_rotated" if direction[0]==0 else "rotated_to_canonical",
                             "layer":layer,"group":"interaction","treatment":"derived",**metrics})
        expected=set()
        for direction in refs:
            for layer in ("enc_17","enc_18","enc_final"):
                expected.add((layer,"all_tokens","self_patch",direction))
                for group in ("markers","nonmarkers","all_tokens"):
                    expected.add((layer,group,"aligned",direction))
                    if layer!="enc_final":
                        expected.update((layer,group,f"noise_{i}",direction) for i in range(manifest["args"]["n_null"]))
        if set(actual)!=expected:
            raise ValueError("factorial treatment grid differs from protocol")
    if cases!=set(manifest["case_ids"]) or len(cases)!=complete["cases"]:
        raise ValueError("case coverage differs")
    maxima={name:max(values) for name,values in errors.items()}
    if maxima["self_patch"]>1e-5 or max(maxima["final_all_tokens"],maxima["marker_replication"])>1e-4:
        raise ValueError("intervention controls exceed tolerances")
    audit={"cases":len(cases),"decisions":len(paths),"patch_forward_rows":expected_rows*len(paths),
           "derived_interactions":6*len(paths),"manifest_sha256":digest,"max_logit_errors":maxima}
    return pd.DataFrame(rows),audit


def analyze(frame,out,n_boot):
    keys=["case_id","qid","workflow","direction","layer","group","treatment"]
    measures=["projection_numerator","contrast_sqnorm","transfer_ratio","change_norm","orthogonal_norm",
              "tokens_edited","total_delta_norm","pair_flip","donor_choice_recovered"]
    averaged=frame.groupby(keys)[measures].mean().reset_index()
    summary=[]
    for (layer,group,treatment),part in averaged.groupby(["layer","group","treatment"]):
        flipped=part[part.pair_flip==1]
        summary.append({"layer":layer,"group":group,"treatment":treatment,
                        "projection":float(part.projection_numerator.sum()/part.contrast_sqnorm.sum()),
                        "ci":projection_interval(part,n_boot),"median_ratio":float(part.transfer_ratio.median()),
                        "change_norm":float(part.change_norm.mean()),"orthogonal_norm":float(part.orthogonal_norm.mean()),
                        "tokens_edited":float(part.tokens_edited.mean()) if part.tokens_edited.notna().any() else None,
                        "total_delta_norm":float(part.total_delta_norm.mean()) if part.total_delta_norm.notna().any() else None,
                        "donor_choice_on_flips":float(flipped.donor_choice_recovered.mean()) if len(flipped) else None})
    aligned=averaged[averaged.treatment=="aligned"]
    contrasts=[]
    basekeys=["case_id","qid","workflow","direction","layer"]
    for name,left,right in (("all_minus_markers","all_tokens","markers"),
                            ("all_minus_nonmarkers","all_tokens","nonmarkers")):
        a=aligned[aligned.group==left].set_index(basekeys)
        b=aligned[aligned.group==right].set_index(basekeys)
        if not a.index.equals(b.index):
            raise ValueError("contrast cells differ")
        difference=a.copy()
        difference["projection_numerator"]=a.projection_numerator-b.projection_numerator
        difference=difference.reset_index()
        for layer,part in difference.groupby("layer"):
            contrasts.append({"contrast":name,"layer":layer,
                              "estimate":float(part.projection_numerator.sum()/part.contrast_sqnorm.sum()),
                              "ci":projection_interval(part,n_boot)})
    for group in ("markers","nonmarkers","all_tokens"):
        a=aligned[(aligned.group==group)&(aligned.layer=="enc_18")].set_index(basekeys[:-1])
        b=aligned[(aligned.group==group)&(aligned.layer=="enc_17")].set_index(basekeys[:-1])
        if not a.index.equals(b.index):
            raise ValueError("cross-layer contrast has different coverage")
        difference=a.copy()
        difference["projection_numerator"]=a.projection_numerator-b.projection_numerator
        difference=difference.reset_index()
        contrasts.append({"contrast":f"enc18_minus_enc17_{group}","layer":"paired",
                          "estimate":float(difference.projection_numerator.sum()/difference.contrast_sqnorm.sum()),
                          "ci":projection_interval(difference,n_boot)})
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    (out/"contrasts.json").write_text(json.dumps(contrasts,indent=2),encoding="utf-8")
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    averaged.to_csv(out/"by_decision.csv",index=False)
    return pd.DataFrame(summary),contrasts


def plot(summary,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels={"markers":"Marcadores","nonmarkers":"Demais tokens","all_tokens":"Ambos"}
    colors=["#236B8E","#D89458","#579D80"]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout="constrained",sharey=True)
    for ax,layer in zip(axes,("enc_17","enc_18")):
        for i,(group,label) in enumerate(labels.items()):
            actual=summary[(summary.layer==layer)&(summary.group==group)&(summary.treatment=="aligned")].iloc[0]
            noise=summary[(summary.layer==layer)&(summary.group==group)&(summary.treatment=="noise")].iloc[0]
            ax.bar(i,actual.projection,color=colors[i],width=.65)
            ax.errorbar(i,actual.projection,yerr=[[actual.projection-actual.ci[0]],[actual.ci[1]-actual.projection]],fmt="none",color="#222",capsize=4)
            ax.scatter(i,noise.projection,marker="x",color="#222",zorder=5)
        ax.axhline(0,color="#aaa",linewidth=1)
        ax.axhline(1,color="#aaa",linewidth=1,linestyle="--")
        ax.set_xticks(range(3),list(labels.values()))
        ax.set_title(layer)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Projeção na mudança da execução doadora")
    fig.suptitle("Contribuição dos grupos de tokens · 64 casos de descoberta\nBarras: transferência alinhada · ×: ruído de mesma norma · IC95% por caso",fontsize=12)
    fig.savefig(out/"factorial.png",dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--marker-reference",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=2000)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    frame,audit=load_factorial(args.input,args.marker_reference)
    summary,contrasts=analyze(frame,args.out,args.bootstrap)
    plot(summary,args.out)
    (args.out/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(contrasts,indent=2))


if __name__=="__main__":
    main()
