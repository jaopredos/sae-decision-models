"""Additional paired diagnostics from the already audited SAE collection."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_position_sae import load,estimate
from run_position import atomic_json
from src.position import effect_metrics


def paired_difference(records,left,right):
    by={r['condition']:r for r in records}
    def vector(label):
        if label=='noise_mean':
            null=[r['patched_logits'] for r in records if r['condition'].startswith('noise_reconstructed_')]
            if not null:raise ValueError('noise conditions missing')
            return np.mean(null,axis=0)
        return np.asarray(by[label]['patched_logits'])
    ref=by['full'];r=np.asarray(ref['recipient_logits'])
    return effect_metrics(ref['donor_logits'],ref['recipient_logits'],r+vector(left)-vector(right))


def diagnostics(root,out,n_boot=2000):
    manifest,data=load(root);paired=[];coverage=[];delta=[];controls={};behavior=[]
    for d in data:
        base=dict(layer=d['layer'],case_id=d['case_id'],qid=d['qid'],workflow=d['workflow'])
        for c in d['controls']:
            controls[c['control']]=max(controls.get(c['control'],0),c['error'])
        for f in d['features']:
            delta.append(dict(**base,group=f['group'],**{k:f[k] for k in ('delta_sse','delta_sqnorm','delta_dot','reconstruction_delta_sqnorm')},eligible_tokens=f['token_count']))
        by={}
        for row in d['patches']:
            key=(row['group'],row['mode'],tuple(row['donor_order']))
            by.setdefault(key,[]).append(row)
        for (group,mode,order),records in by.items():
            for left,right in [('reconstructed','eligible'),('reconstructed','noise_mean'),('full','eligible'),('full','reconstructed'),('residual','self'),('outside','self')]:
                paired.append(dict(**base,group=group,mode=mode,donor_order=order,contrast=f'{left}_minus_{right}',**paired_difference(records,left,right)))
            bycondition={r['condition']:r for r in records}
            for label,terms in [('eligible_minus_reconstructed_minus_residual',{'eligible':1,'reconstructed':-1,'residual':-1}),('full_minus_eligible_minus_outside',{'full':1,'eligible':-1,'outside':-1})]:
                ref=bycondition['full'];rlog=np.asarray(ref['recipient_logits']);change=np.zeros_like(rlog)
                for name,coef in terms.items():change+=coef*(np.asarray(bycondition[name]['patched_logits'])-rlog)
                paired.append(dict(**base,group=group,mode=mode,donor_order=order,contrast=label,
                    **effect_metrics(ref['donor_logits'],ref['recipient_logits'],rlog+change)))
            if order==tuple(d['order_pair'][0]) and mode=='direct':
                direct=next(r for r in records if r['condition']=='full')
                o=next(r for r in records if r['condition']=='outside')
                coverage.append(dict(**base,group=group,tokens=direct['tokens'],eligible=direct['eligible'],outside=direct['outside'],
                    full_delta_sqnorm=direct['delta_norm']**2,outside_delta_sqnorm=o['delta_norm']**2))
            for r in records:
                dlog,rlog,plog=[np.asarray(r[k]) for k in ('donor_logits','recipient_logits','patched_logits')]
                flip=int(dlog.argmax()!=rlog.argmax())
                behavior.append(dict(**base,group=group,mode=mode,condition=r['condition'],direction='canonical_donor' if order==tuple(d['order_pair'][0]) else 'rotated_donor',flip=flip,
                    donor_recovered=int(plog.argmax()==dlog.argmax()),recipient_preserved=int(plog.argmax()==rlog.argmax()),
                    **{k:r[k] for k in ('projection_numerator','contrast_sqnorm','transfer_ratio','change_norm','orthogonal_norm')}))
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for key,f in pd.DataFrame(paired).groupby(['layer','group','mode','contrast']):
        summaries.append(dict(zip(['layer','group','mode','contrast'],key),**estimate(f,n_boot)))
    pd.DataFrame(summaries).to_csv(out/'paired_contrasts.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage_per_decision.csv',index=False)
    cov=[]
    for key,f in pd.DataFrame(coverage).groupby(['layer','group']):
        cov.append(dict(zip(['layer','group'],key),tokens=int(f.tokens.sum()),eligible=int(f.eligible.sum()),outside=int(f.outside.sum()),
            fraction_outside=float(f.outside.sum()/f.tokens.sum()),outside_delta_energy_fraction=float(f.outside_delta_sqnorm.sum()/f.full_delta_sqnorm.sum())))
    pd.DataFrame(cov).to_csv(out/'coverage.csv',index=False)
    delta_summary=[]
    for key,f in pd.DataFrame(delta).groupby(['layer','group']):
        v=f.delta_sqnorm.sum();rv=f.reconstruction_delta_sqnorm.sum();dot=f.delta_dot.sum()
        delta_summary.append(dict(zip(['layer','group'],key),tokens=int(f.eligible_tokens.sum()),
            relative_delta_l2_error=float(np.sqrt(f.delta_sse.sum()/v)) if v>1e-8 else None,
            delta_projection=float(dot/v) if v>1e-8 else None,
            delta_cosine=float(dot/np.sqrt(v*rv)) if v*rv>1e-16 else None,
            delta_sqnorm=float(v),reconstructed_delta_sqnorm=float(rv)))
    pd.DataFrame(delta_summary).to_csv(out/'activation_delta_fidelity.csv',index=False)
    b=pd.DataFrame(behavior)
    strat=[]
    for key,f in b.groupby(['layer','group','mode','condition','flip']):
        strat.append(dict(zip(['layer','group','mode','condition','flip'],key),**estimate(f,n_boot),
            decisions=len(f[['case_id','qid']].drop_duplicates()),donor_recovery=float(f.donor_recovered.mean()),recipient_preservation=float(f.recipient_preserved.mean())))
    pd.DataFrame(strat).to_csv(out/'flip_strata.csv',index=False)
    direction=[]
    for key,f in b.groupby(['layer','group','mode','condition','workflow']):
        direction.append(dict(zip(['layer','group','mode','condition','workflow'],key),**estimate(f,n_boot)))
    pd.DataFrame(direction).to_csv(out/'workflow_strata.csv',index=False)
    sensitive=[]
    for key,f in b.groupby(['layer','group','mode','condition']):
        for direction,g in f.groupby('direction'):
            sensitive.append(dict(zip(['layer','group','mode','condition'],key),analysis=direction,**estimate(g,n_boot)))
        for workflow in sorted(f.workflow.unique()):
            sensitive.append(dict(zip(['layer','group','mode','condition'],key),analysis=f'without_{workflow}',**estimate(f[f.workflow!=workflow],n_boot)))
        largest=f.groupby('case_id').contrast_sqnorm.sum().idxmax()
        sensitive.append(dict(zip(['layer','group','mode','condition'],key),analysis=f'without_max_weight_case:{largest}',**estimate(f[f.case_id!=largest],n_boot)))
    pd.DataFrame(sensitive).to_csv(out/'sensitivity.csv',index=False)
    atomic_json(out/'diagnostic_audit.json',dict(cases=len(manifest['case_ids']),decisions=len(manifest['tasks']),
        controls=controls,bootstrap_unit='case; questions, directions, conditions kept together',
        noise='mean of four null logits before paired contrasts; same norm as reconstructed delta',
        coverage=cov,activation_delta_fidelity=delta_summary))
    return summaries


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();diagnostics(a.input,a.out)

if __name__=='__main__':main()