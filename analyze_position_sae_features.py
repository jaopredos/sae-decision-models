"""Offline audit and paired case-bootstrap of frozen cross-fitted feature transfers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_position import atomic_json
from analyze_position_sae import digest,estimate,probability_stats
from src.position import effect_metrics


def analyze(root,out,n_boot=2000):
    manifest=json.loads((root/'manifest.json').read_text());complete=json.loads((root/'COMPLETE.json').read_text())
    if digest(root/'manifest.json')!=complete['manifest_sha256']:raise ValueError('manifest differs')
    files={p.relative_to(root).as_posix() for p in (root/'decisions').glob('*.json')}
    if files!=set(complete['decision_sha256']):raise ValueError('file coverage differs')
    seen=set();rows=[];contrasts=[];side=[];controls={};doses=[]
    for rel,sha in complete['decision_sha256'].items():
        path=root/rel
        if digest(path)!=sha:raise ValueError('decision hash differs')
        d=json.loads(path.read_text());key=(d['case_id'],d['qid'])
        if key in seen:raise ValueError('duplicate decision')
        seen.add(key);base=dict(case_id=d['case_id'],qid=d['qid'],workflow=d['workflow'],fold=d['fold'])
        labels={'self','all_features','eligible_activation','only_selected','drop_selected','reset_all18'}
        labels|={f'{mode}_matched_{i}' for mode in ('only','drop') for i in range(len(d['matched_sets']))}
        labels|={f'noise_selected_{i}' for i in range(manifest['args']['n_null'])}
        orders={tuple(r['donor_order']) for r in d['patches']}
        if len(orders)!=2:raise ValueError('direction coverage differs')
        found=set()
        for r in d['patches']:
            rowkey=(r['condition'],tuple(r['donor_order']))
            if rowkey in found:raise ValueError('duplicate condition')
            found.add(rowkey)
            metric=effect_metrics(r['donor_logits'],r['recipient_logits'],r['patched_logits'])
            for name,value in metric.items():
                if value is None and r[name] is not None:raise ValueError('undefined metric differs')
                if value is not None and not np.isclose(value,r[name],atol=1e-9,rtol=1e-8):raise ValueError('metric differs')
            rows.append(dict(**base,**r))
            gold=d['option_ids'].index(str(d['gold']['label']))
            a=probability_stats(r['recipient_logits'],r['recipient_probabilities'],gold)
            b=probability_stats(r['patched_logits'],r['patched_probabilities'],gold)
            p=np.array(r['recipient_probabilities']);p/=p.sum();q=np.array(r['patched_probabilities']);q/=q.sum()
            side.append(dict(**base,condition=r['condition'],kl=float((p*(np.log(p.clip(1e-15))-np.log(q.clip(1e-15)))).sum()),
                **{f'original_{k}':v for k,v in a.items()},**{f'edited_{k}':v for k,v in b.items()},**{f'delta_{k}':b[k]-v for k,v in a.items()}))
        if found!={(label,order) for label in labels for order in orders}:raise ValueError('condition grid differs')
        for c in d['controls']:controls[c['control']]=max(controls.get(c['control'],0),c['error'])
        for order in orders:
            by={r['condition']:r for r in d['patches'] if tuple(r['donor_order'])==order};ref=by['all_features'];rlog=np.array(ref['recipient_logits'])
            def get(label):
                prefix={'matched_only_mean':'only_matched_','matched_drop_mean':'drop_matched_','noise_mean':'noise_selected_'}.get(label)
                if prefix:return np.mean([r['patched_logits'] for name,r in by.items() if name.startswith(prefix)],axis=0)
                return np.array(by[label]['patched_logits'])
            for name,left,right in [('selected_minus_matched','only_selected','matched_only_mean'),
                ('drop_matched_minus_drop_selected','matched_drop_mean','drop_selected'),
                ('selected_minus_noise','only_selected','noise_mean'),('loss_selected','all_features','drop_selected'),
                ('selected_minus_all','only_selected','all_features')]:
                contrasts.append(dict(**base,contrast=name,donor_order=order,
                    **effect_metrics(ref['donor_logits'],ref['recipient_logits'],rlog+get(left)-get(right))))
            for group,prefix in [('matched_only_mean','only_matched_'),('matched_drop_mean','drop_matched_'),('noise_mean','noise_selected_')]:
                temp=dict(**base,condition=group,donor_order=order,
                    predicted_numerator=float(np.mean([r['predicted_numerator'] for name,r in by.items() if name.startswith(prefix)])),
                    delta_norm=float(np.mean([r['delta_norm'] for name,r in by.items() if name.startswith(prefix)])),
                    **effect_metrics(ref['donor_logits'],ref['recipient_logits'],get(group)))
                rows.append(temp)
            matched=[r['delta_norm'] for name,r in by.items() if name.startswith('only_matched_')]
            doses.append(dict(**base,donor_order=order,selected_norm=by['only_selected']['delta_norm'],matched_norm_mean=float(np.mean(matched)),
                matched_norm_min=min(matched),matched_norm_max=max(matched)))
    expected={tuple(t) for t in manifest['tasks']}
    if seen!=expected or len(seen)!=complete['decisions']:raise ValueError('task coverage differs')
    out.mkdir(parents=True,exist_ok=True);summ=[];cs=[];sens=[]
    f=pd.DataFrame(rows)
    for condition,g in f.groupby('condition'):
        summ.append(dict(condition=condition,**estimate(g,n_boot)))
        for workflow in sorted(g.workflow.unique()):
            h=g[g.workflow!=workflow]
            if len(h):sens.append(dict(condition=condition,analysis=f'without_{workflow}',**estimate(h,n_boot)))
        for order,h in g.groupby(g.donor_order.map(tuple)):
            sens.append(dict(condition=condition,analysis=str(order),**estimate(h,n_boot)))
    for name,g in pd.DataFrame(contrasts).groupby('contrast'):cs.append(dict(contrast=name,**estimate(g,n_boot)))
    pd.DataFrame(summ).to_csv(out/'summary.csv',index=False);pd.DataFrame(cs).to_csv(out/'contrasts.csv',index=False)
    pd.DataFrame(sens).to_csv(out/'sensitivity.csv',index=False);pd.DataFrame(doses).to_csv(out/'dose_balance.csv',index=False)
    pd.DataFrame(side).groupby('condition').mean(numeric_only=True).to_csv(out/'side_effects.csv')
    calibration=[]
    for condition,g in f.groupby('condition'):
        valid=g[['predicted_numerator','projection_numerator']].to_numpy()
        corr=float(np.corrcoef(pd.Series(valid[:,0]).rank().to_numpy(),pd.Series(valid[:,1]).rank().to_numpy())[0,1]) if np.std(valid[:,0])>1e-12 and np.std(valid[:,1])>1e-12 else None
        calibration.append(dict(condition=condition,spearman=corr,mean_absolute_numerator_error=float(np.abs(valid[:,0]-valid[:,1]).mean()),
            predicted_projection=float(g.predicted_numerator.sum()/g.contrast_sqnorm.sum()) if g.contrast_sqnorm.sum()>1e-8 else None))
    pd.DataFrame(calibration).to_csv(out/'first_order_calibration.csv',index=False)
    atomic_json(out/'audit.json',dict(cases=len(manifest['case_ids']),decisions=len(seen),condition_rows=sum(1 for row in rows if row['condition'] not in ('matched_only_mean','matched_drop_mean','noise_mean')),
        controls=controls,manifest_sha256=digest(root/'manifest.json'),selection_sha256=manifest['selection_sha256'],
        n_boot=n_boot,seed=0,interpretation='Exploratory cross-fitted discovery; no position selectivity or mediation percentage.'))
    atomic_json(out/'summary.json',dict(conditions=summ,contrasts=cs,calibration=calibration))
    print(json.dumps(cs,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();analyze(a.input,a.out)

if __name__=='__main__':main()