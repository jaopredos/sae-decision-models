"""Offline integrity and fidelity audit; no checkpoint loading or inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_position import atomic_json
from src.position import effect_metrics
from analyze_position import projection_interval


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root):
    manifest=json.loads((root/'manifest.json').read_text())
    complete=json.loads((root/'COMPLETE.json').read_text())
    if digest(root/'manifest.json')!=complete['manifest_sha256']:raise ValueError('manifest hash differs')
    expected={(layer,c,q) for layer in manifest['saes'] for c,q in manifest['tasks']}
    seen=set();payloads=[]
    actual={p.relative_to(root).as_posix() for p in (root/'decisions').glob('*/*.json')}
    if actual!=set(complete['decision_sha256']):raise ValueError('file coverage differs')
    for rel,h in complete['decision_sha256'].items():
        path=root/rel
        if digest(path)!=h:raise ValueError('decision hash differs')
        d=json.loads(path.read_text());key=(d['layer'],d['case_id'],d['qid'])
        if key in seen:raise ValueError('duplicate decision')
        seen.add(key)
        orders=d['order_pair'];n_null=manifest['args']['n_null']
        variants={'self','full','eligible','reconstructed','residual','outside','recombined'}|{f'noise_reconstructed_{i}' for i in range(n_null)}
        grid={(g,m,c,tuple(order)) for g in ('markers','nonmarkers','all_tokens') for m in (['direct','path'] if d['layer']=='enc_17' and g=='nonmarkers' else ['direct']) for c in variants|({'reset_all18'} if m=='path' else set()) for order in orders}
        found=set()
        for r in d['patches']:
            rowkey=(r['group'],r['mode'],r['condition'],tuple(r['donor_order']))
            if rowkey in found:raise ValueError('duplicate condition')
            found.add(rowkey)
            metrics=effect_metrics(r['donor_logits'],r['recipient_logits'],r['patched_logits'])
            for k,v in metrics.items():
                if v is None:
                    if r[k] is not None:raise ValueError('undefined metric differs')
                elif not np.isclose(v,r[k],atol=1e-9,rtol=1e-8):raise ValueError('metric differs')
        if found!=grid:raise ValueError('condition coverage differs')
        reconstruction_groups=set()
        for r in d['reconstruction_contrasts']:
            if r['group'] in reconstruction_groups:raise ValueError('duplicate reconstruction group')
            reconstruction_groups.add(r['group'])
            clean=np.subtract(*r['original_logits']);rec=np.subtract(*r['reconstructed_logits'])
            if not np.allclose(clean,r['clean_contrast'],atol=1e-9) or not np.allclose(rec,r['reconstructed_contrast'],atol=1e-9):
                raise ValueError('reconstruction contrast differs from saved logits')
            metrics=effect_metrics(clean,[0.]*len(clean),rec)
            for k,v in metrics.items():
                if v is None and r[k] is not None:raise ValueError('undefined reconstruction metric differs')
                if v is not None and not np.isclose(v,r[k],atol=1e-9,rtol=1e-8):raise ValueError('reconstruction metric differs')
        if reconstruction_groups!={'markers','nonmarkers','all_tokens'}:raise ValueError('reconstruction coverage differs')
        clean_by_order={}
        for r in d['patches']:
            for role in ('donor','recipient','patched'):
                probs=np.asarray(r[f'{role}_probabilities'])
                if len(probs)!=len(d['option_ids']) or not np.isfinite(probs).all() or (probs<0).any() or abs(probs.sum()-1)>.001:
                    raise ValueError('invalid saved probabilities')
                if role!='patched':
                    key=tuple(r[f'{role}_order']);value=r[f'{role}_logits']
                    if key in clean_by_order and value!=clean_by_order[key]:raise ValueError('clean logits vary across conditions')
                    clean_by_order[key]=value
        payloads.append(d)
    if seen!=expected or complete['layer_decisions']!=len(seen):raise ValueError('decision coverage differs')
    return manifest,payloads


def estimate(frame,n_boot):
    denom=float(frame.contrast_sqnorm.sum())
    ratios=frame.transfer_ratio.dropna().to_numpy()
    weights=frame.groupby('case_id').contrast_sqnorm.sum()
    return dict(rows=len(frame),cases=frame.case_id.nunique(),denominator=denom,
        projection=float(frame.projection_numerator.sum()/denom) if denom>1e-8 else None,
        ci=projection_interval(frame,n_boot) if denom>1e-8 else None,
        undefined=int(frame.transfer_ratio.isna().sum()),
        ratio_quantiles=np.quantile(ratios,[.05,.5,.95]).tolist() if len(ratios) else None,
        mean_change_norm=float(frame.change_norm.mean()),mean_orthogonal_norm=float(frame.orthogonal_norm.mean()),
        max_case_weight=float(weights.max()/denom) if denom>1e-8 else None)


def probability_stats(logits,probabilities,gold):
    p=np.asarray(probabilities,dtype=float);p/=p.sum()
    l=np.asarray(logits);order=np.sort(l)
    return dict(accuracy=float(l.argmax()==gold),normalized_entropy=float(-(p*np.log(p.clip(1e-15))).sum()/np.log(len(p))),
        top2_margin=float(order[-1]-order[-2]),centered_logit_norm=float(np.linalg.norm(l-l.mean())))


def analyze(root,out,n_boot):
    manifest,payloads=load(root);patches=[];contrasts=[];fidelity={};side=[]
    for d in payloads:
        base=dict(layer=d['layer'],case_id=d['case_id'],qid=d['qid'])
        gold=d['option_ids'].index(str(d['gold']['label']))
        for r in d['patches']:
            patches.append(dict(**base,**r))
            initial=probability_stats(r['recipient_logits'],r['recipient_probabilities'],gold)
            edited=probability_stats(r['patched_logits'],r['patched_probabilities'],gold)
            p=np.asarray(r['recipient_probabilities']);p/=p.sum()
            q=np.asarray(r['patched_probabilities']);q/=q.sum()
            side.append(dict(**base,group=r['group'],mode=r['mode'],condition=r['condition'],
                kl=float((p*(np.log(p.clip(1e-15))-np.log(q.clip(1e-15)))).sum()),
                **{f'original_{k}':v for k,v in initial.items()},**{f'edited_{k}':v for k,v in edited.items()},
                **{f'delta_{k}':edited[k]-v for k,v in initial.items()}))
        for r in d['reconstruction_contrasts']:
            contrasts.append(dict(**base,**r))
            for original,reconstructed,po,pr in zip(r['original_logits'],r['reconstructed_logits'],r['original_probabilities'],r['reconstructed_probabilities']):
                initial=probability_stats(original,po,gold);edited=probability_stats(reconstructed,pr,gold)
                p=np.asarray(po);p/=p.sum();q=np.asarray(pr);q/=q.sum()
                side.append(dict(**base,group=r['group'],mode='replace',condition='reconstruction',
                    kl=float((p*(np.log(p.clip(1e-15))-np.log(q.clip(1e-15)))).sum()),
                    **{f'original_{k}':v for k,v in initial.items()},**{f'edited_{k}':v for k,v in edited.items()},
                    **{f'delta_{k}':edited[k]-v for k,v in initial.items()}))
        for r in d['fidelity']:
            key=(d['layer'],r['group'],r['stratum'])
            f=fidelity.setdefault(key,dict(n=0,sse=0.,sumsq=0.,sum=np.zeros(len(r['sum']))))
            for k in ('n','sse','sumsq'):f[k]+=r[k]
            f['sum']+=r['sum']
    frame=pd.DataFrame(patches);summary=[]
    for key,g in frame.groupby(['layer','group','mode','condition']):
        summary.append(dict(zip(['layer','group','mode','condition'],key),**estimate(g,n_boot)))
    replace=[]
    for key,g in pd.DataFrame(contrasts).groupby(['layer','group']):
        result=estimate(g,n_boot)
        clean=np.concatenate([np.asarray(v)-np.mean(v) for v in g.clean_contrast])
        rec=np.concatenate([np.asarray(v)-np.mean(v) for v in g.reconstructed_contrast])
        error=float(np.linalg.norm(rec-clean)/np.linalg.norm(clean)) if np.linalg.norm(clean)>1e-8 else None
        replace.append(dict(zip(['layer','group'],key),**result,relative_contrast_l2_error=error,
            fidelity_flag=(abs(result['projection']-1)>.20 or error>.50) if error is not None else None))
    fv=[]
    for key,f in fidelity.items():
        variance=f['sumsq']-float(f['sum']@f['sum'])/f['n'] if f['n'] else 0
        fv.append(dict(zip(['layer','group','stratum'],key),n=f['n'],sse=f['sse'],variance=variance,
                       fve=1-f['sse']/variance if variance>1e-8 else None))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary).to_csv(out/'patch_summary.csv',index=False)
    pd.DataFrame(replace).to_csv(out/'contrast_fidelity.csv',index=False)
    pd.DataFrame(fv).to_csv(out/'fve.csv',index=False)
    pd.DataFrame(side).groupby(['layer','group','mode','condition']).mean(numeric_only=True).to_csv(out/'side_effects.csv')
    atomic_json(out/'audit.json',dict(cases=len(manifest['case_ids']),layer_decisions=len(payloads),
        max_control_error=max(c['error'] for d in payloads for c in d['controls']),
        manifest_sha256=digest(root/'manifest.json'),n_boot=n_boot,seed=0,
        caution=('Pilot intervals are not evidence of population stability. ' if len(manifest['case_ids'])<10 else 'Exploratory discovery analysis. ')+'Ratios are not mediation percentages.'))
    atomic_json(out/'summary.json',dict(patches=summary,contrast_fidelity=replace,fve=fv))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target={r['condition']:r for r in summary if r['layer']=='enc_17' and r['group']=='nonmarkers' and r['mode']=='path'}
    order=['full','eligible','reconstructed','residual','outside']
    shown=[target[k] for k in order]
    labels=['Ativações\ncompletas','Ativações\nelegíveis','Delta\nreconstruído','Delta do\nresidual','Fora do\ndomínio']
    values=[r['projection'] for r in shown]
    errors=np.array([[max(0,r['projection']-r['ci'][0]),max(0,r['ci'][1]-r['projection'])] for r in shown]).T
    fig,ax=plt.subplots(figsize=(8,4.4));ax.bar(labels,values,yerr=errors,capsize=4,color=['#667d8d','#667d8d','#277da8','#c78036','#8d8d8d'])
    ax.axhline(0,color='black',linewidth=.6);ax.set_ylabel('Projeção no contraste limpo de ordem')
    ax.set_title(f'Caminho enc17 → marcadores18 — {len(manifest["case_ids"])} casos de descoberta')
    ax.text(.01,.98,'IC95% por caso; coeficientes não são porcentagens de mediação',transform=ax.transAxes,va='top',fontsize=8)
    ax.set_ylim(-.015,max(r['ci'][1] for r in shown)*1.16)
    fig.tight_layout();fig.savefig(out/'path.png',dpi=170);plt.close(fig)
    return replace


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--n-boot',type=int,default=2000)
    a=ap.parse_args();print(json.dumps(analyze(a.input,a.out,a.n_boot),indent=2))

if __name__=='__main__':main()