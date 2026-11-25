"""Freeze an order-delta feature rule and cross-fitted matched controls; no labels/logits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.position import stable_seed
from run_position import atomic_json
from analyze_position_sae import load,digest


def summarize(records):
    cases={}
    for d in records:
        f=next(f for f in d['features'] if f['group']=='nonmarkers')
        entry=cases.setdefault(d['case_id'],[])
        count=f['token_count'];active=np.asarray(f['active_count'],float)
        entry.append((np.asarray(f['sum_abs_delta'],float)/max(count,1),active/max(count,1),active>0))
    magnitude=[];frequency=[];support=[]
    for parts in cases.values():
        magnitude.append(np.mean([p[0] for p in parts],axis=0))
        frequency.append(np.mean([p[1] for p in parts],axis=0))
        support.append(np.any([p[2] for p in parts],axis=0))
    m=np.mean(magnitude,axis=0);f=np.mean(frequency,axis=0);support=np.sum(support,axis=0)
    return m,f,support


def rank_and_match(records,k,n_controls,min_cases,seed):
    m,f,s=summarize(records)
    pool=np.where(s>=min_cases)[0]
    if len(pool)<k*(n_controls+1):raise ValueError('insufficient supported features for disjoint controls')
    selected=sorted(pool,key=lambda i:(-m[i],int(i)))[:k]
    # Frequency and mean change per active token: magnitude matching does not use logits.
    values=np.column_stack((np.log(f.clip(1e-12)),np.log((m/f.clip(1e-12)).clip(1e-12))))
    bins=np.zeros((len(m),2),int)
    for j in range(2):bins[:,j]=np.searchsorted(np.quantile(values[pool,j],[.2,.4,.6,.8]),values[:,j],side='right')
    scale=values[pool].std(0).clip(.1)
    rng=np.random.default_rng(seed);available=set(pool)-set(selected);controls=[];balance=[]
    for repeat in range(n_controls):
        chosen=[]
        for target in rng.permutation(selected):
            candidates=np.array(sorted(available));distance=np.linalg.norm((values[candidates]-values[target])/scale,axis=1)
            matched=(bins[candidates]==bins[target]).all(1)
            candidates_good=np.where(matched)[0]
            fallback=len(candidates_good)==0
            choices=candidates_good if not fallback else np.arange(len(candidates))
            # Random among the 8 closest available matches; frozen seed.
            nearest=choices[np.argsort(distance[choices],kind='stable')[:8]]
            pick=int(rng.choice(nearest));feature=int(candidates[pick]);available.remove(feature);chosen.append(feature)
            balance.append(dict(repeat=repeat,selected=int(target),control=feature,same_quintiles=not fallback,
                standardized_log_distance=float(distance[pick]),selected_frequency=float(f[target]),control_frequency=float(f[feature]),
                selected_mean_abs_delta=float(m[target]),control_mean_abs_delta=float(m[feature])))
        controls.append(chosen)
    return dict(selected=[int(i) for i in selected],controls=controls,training_cases=len(set(d['case_id'] for d in records)),
        supported_features=len(pool),balance=balance),pd.DataFrame(dict(feature=np.arange(len(m)),mean_abs_delta_per_token=m,
        frequency=f,case_support=s,rank=pd.Series(-m).rank(method='first').astype(int)))


def build(root,out,k=32,n_controls=8,min_cases=20):
    manifest,data=load(root);data=[d for d in data if d['layer']=='enc_17']
    byworkflow={}
    for d in data:byworkflow.setdefault(d['workflow'],set()).add(d['case_id'])
    folds={}
    for workflow,ids in sorted(byworkflow.items()):
        for i,cid in enumerate(sorted(ids,key=lambda c:stable_seed('sae-feature-fold-v1',c))):folds[cid]=i%4
    selections={};ranks=[];balances=[]
    for fold in range(4):
        training=[d for d in data if folds[d['case_id']]!=fold]
        sel,table=rank_and_match(training,k,n_controls,min_cases,stable_seed('sae-matched-v1',fold))
        sel['evaluation_case_ids']=sorted(c for c,f in folds.items() if f==fold)
        selections[str(fold)]=sel;table['fold']=fold;ranks.append(table)
        balances.extend(dict(fold=fold,**b) for b in sel['balance'])
    final,table=rank_and_match(data,k,n_controls,min_cases,stable_seed('sae-matched-v1','all'))
    table['fold']='all';ranks.append(table)
    balances.extend(dict(fold='all',**b) for b in final['balance'])
    output=dict(protocol='docs/position-sae-mechanism/FEATURE_PROTOCOL.md',k=k,n_controls=n_controls,min_cases=min_cases,
        source_manifest_sha256=digest(root/'manifest.json'),source_complete_sha256=digest(root/'COMPLETE.json'),
        criterion='equal-case mean per-eligible-token absolute latent change; no gold, errors, flips or logits',
        fold_rule='four folds, stratified by workflow; stable_seed sae-feature-fold-v1',fold_by_case=folds,selections=selections,
        final_discovery_selection=final,selection_code_sha256=digest(Path(__file__)))
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'selection.json',output)
    pd.concat(ranks).to_csv(out/'feature_ranks.csv',index=False);pd.DataFrame(balances).to_csv(out/'control_balance.csv',index=False)
    overlaps=[]
    sets=[set(s['selected']) for s in selections.values()]
    for i in range(4):
        for j in range(i):overlaps.append(dict(fold_a=i,fold_b=j,jaccard=len(sets[i]&sets[j])/len(sets[i]|sets[j])))
    atomic_json(out/'selection_audit.json',dict(k=k,fold_counts=pd.Series(folds).value_counts().sort_index().tolist(),
        fallback_matches=sum(not b['same_quintiles'] for b in balances),matches=len(balances),fold_overlap=overlaps,
        note='Matching quality must be reported; candidate selection does not establish position selectivity.'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();build(a.input,a.out)

if __name__=='__main__':main()