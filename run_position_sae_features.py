"""Cross-fitted enc17 SAE feature transplants on discovery; GPU jobs only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from run_position import Runner,atomic_json
from run_position_sae import checked_sae,snapshot,sha
from run_position_factorial import full_patch
from run_position_path import patch_hook
from src.position import choose_cases,conditions,effect_metrics,stable_seed


def collect(runner,sae,case,qid,selection,reference,n_null=4):
    import torch
    from src.position_sae import decompose
    from src.position_sae_features import feature_delta,order_gradient
    q=next(conditions(case['questions'][qid],case['state'],case['state'],case['case_id'],qid))[1]
    k=len(q['criteria'])
    with torch.inference_mode():
        a,b=[snapshot(runner,case,qid,q,o,'enc_17') for o in (list(range(k)),list(range(1,k))+[0])]
        runner.parity(case['state'],qid,q,a)
        if a['alignment']['tokens']!=b['alignment']['tokens'] or a['alignment']['groups']!=b['alignment']['groups']:raise ValueError('alignment differs')
        parts=decompose(sae,a['states']['enc_17'],b['states']['enc_17'])
    e=parts['eligible'];groups=a['alignment']['groups'];mask=e.clone();mask[groups['markers']]=False
    local={i:j for j,i in enumerate(torch.where(e)[0].tolist())}
    physical_eligible=torch.where(mask)[0].tolist();latent_rows=[local[i] for i in physical_eligible]
    sets={'selected':selection['selected'],**{f'matched_{i}':v for i,v in enumerate(selection['controls'])}}
    records=[];checks=[];predictions=[];examples=[]
    for direction,(donor,recipient) in enumerate(((a,b),(b,a))):
        sign=1 if direction==0 else -1
        zd,zr=(parts['donor_latents'],parts['recipient_latents']) if direction==0 else (parts['recipient_latents'],parts['donor_latents'])
        gradient,replay_error=order_gradient(runner,case,qid,q,donor,recipient)
        checks.append(dict(control='gradient_replay',error=replay_error))
        with torch.inference_mode():
            dz=(zd[latent_rows]-zr[latent_rows]).to(sae.W_dec.device)
            first_order=(dz*((gradient[physical_eligible].to(sae.W_dec.device)@sae.W_dec.t())/sae.scale)).sum(0).cpu()
            all_delta=parts['reconstructed_delta']*sign;all_delta[~mask]=0
            eligible_delta=torch.zeros_like(all_delta);eligible_delta[mask]=(donor['states']['enc_17']-recipient['states']['enc_17'])[mask]
            variants={'self':torch.zeros_like(all_delta),'all_features':all_delta,'eligible_activation':eligible_delta}
            expected={'self':0.,'all_features':float(first_order.sum()),'eligible_activation':float((gradient*eligible_delta).sum())}
            set_doses={}
            for label,indices in sets.items():
                delta=torch.zeros_like(all_delta)
                delta[physical_eligible]=feature_delta(sae,zd[latent_rows],zr[latent_rows],indices)
                set_doses[label]=float(delta.norm())
                variants[f'only_{label}']=delta;variants[f'drop_{label}']=all_delta-delta
                predicted=float(first_order[indices].sum())
                expected[f'only_{label}']=predicted;expected[f'drop_{label}']=float(first_order.sum())-predicted
            for i in range(n_null):
                g=torch.Generator().manual_seed(stable_seed('sae-selected-noise-v1',case['case_id'],qid,donor['order'],i)%(2**63-1))
                noise=torch.randn(all_delta.shape,generator=g)
                noise*=variants['only_selected'].norm(dim=-1,keepdim=True)/noise.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                variants[f'noise_selected_{i}']=noise;expected[f'noise_selected_{i}']=float((gradient*noise).sum())
            for label,delta in list(variants.items())+[('reset_all18',all_delta)]:
                restore=groups['all_tokens'] if label=='reset_all18' else groups['nonmarkers']
                handle=runner.modules['enc_18'].register_forward_hook(patch_hook(recipient['alignment']['indices'],recipient['states']['enc_18'],restore))
                try:result=full_patch(runner,case,qid,q,recipient,'enc_17',recipient['states']['enc_17']+delta)
                finally:handle.remove()
                if label in ('self','reset_all18'):
                    error=max(abs(x-y) for x,y in zip(result['logits'],recipient['logits']))
                    if error>1e-5:raise ValueError('feature identity control failed')
                    checks.append(dict(control=label,error=error))
                if label in ('all_features','eligible_activation'):
                    refcondition='reconstructed' if label=='all_features' else 'eligible'
                    ref=next(r for r in reference['patches'] if r['mode']=='path' and r['group']=='nonmarkers' and r['condition']==refcondition and r['donor_order']==donor['order'])
                    error=max(abs(x-y) for x,y in zip(result['logits'],ref['patched_logits']))
                    if error>1e-4:raise ValueError('feature reference reproduction failed')
                    checks.append(dict(control=f'reference_{label}',error=error))
                records.append(dict(condition=label,donor_order=donor['order'],recipient_order=recipient['order'],
                    donor_logits=donor['logits'],recipient_logits=recipient['logits'],patched_logits=result['logits'],
                    recipient_probabilities=recipient['probabilities'],patched_probabilities=result['probabilities'],
                    delta_norm=float(delta.norm()),eligible_tokens=int(mask.sum()),selected_set_dose=set_doses.get('selected'),matched_set_dose=set_doses.get(label.replace('only_','').replace('drop_','')),
                    predicted_numerator=expected.get(label,0.),**effect_metrics(donor['logits'],recipient['logits'],result['logits'])))
            union=sorted(set(i for indices in sets.values() for i in indices))
            predictions.append(dict(donor_order=donor['order'],features=union,predicted_numerator=first_order[union].tolist()))
            # Token examples are descriptive; selection was frozen without these or gradients.
            physical_tokens=[0]*len(mask)
            for token,pos in zip(recipient['alignment']['tokens'],recipient['alignment']['indices']):physical_tokens[pos]=token
            for feature in sets['selected']:
                changes=dz[:,feature].abs().cpu();top=torch.topk(changes,min(2,len(changes))).indices.tolist()
                for index in top:
                    canonical=physical_eligible[index];position=recipient['alignment']['indices'][canonical]
                    examples.append(dict(feature=feature,recipient_order=recipient['order'],absolute_position=position,
                        canonical_token_index=canonical,token_id=physical_tokens[position],
                        text=runner.agent.tok.decode(physical_tokens[max(0,position-4):position+5]),
                        donor_value=float(zd[latent_rows[index],feature]),recipient_value=float(zr[latent_rows[index],feature])))
    return dict(case_id=case['case_id'],qid=qid,workflow=case['workflow'],option_ids=a['ids'],gold=case['gold'][qid],
        controls=checks,patches=records,feature_predictions=predictions,examples=examples,
        selected=selection['selected'],matched_sets=selection['controls'],set_doses=set_doses)


def main():
    import resource
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--limit',type=int,default=2);ap.add_argument('--n-null',type=int,default=4)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.limit<1 or args.n_null!=4:ap.error('positive limit and four nulls required')
    import torch
    from src.activations import load_agent,LAYA_REVISION
    from src.datasets import load_typed_cases,typed_dev_ids,TYPED_DECISIONS_REVISION
    from src.models import set_all_seeds
    if not torch.cuda.is_available():raise RuntimeError('GPU job required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);os.environ['LAYA_CUDA_AMP']='fp32'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False);set_all_seeds(0)
    selection=json.loads(args.selection.read_text())
    if sha(args.reference/'manifest.json')!=selection['source_manifest_sha256'] or sha(args.reference/'COMPLETE.json')!=selection['source_complete_sha256']:raise ValueError('selection source differs')
    sae,provenance=checked_sae(args.weights,'ml_enc17_k32','cuda')
    cases=choose_cases(load_typed_cases('train'),'discovery',typed_dev_ids(),args.limit)
    tasks=[(c,qid) for c in cases for qid,q in c['questions'].items() if q['type']=='choice']
    paths=['run_position_sae_features.py','src/position_sae_features.py','run_position_sae.py','src/position_sae.py','run_position.py',
        'run_position_factorial.py','run_position_path.py','src/position.py','src/position_factorial.py','src/activations.py','src/sae.py',
        'src/datasets.py','src/models.py','uv.lock','docs/position-sae-mechanism/FEATURE_PROTOCOL.md',str(args.selection)]
    manifest=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},case_ids=[c['case_id'] for c in cases],
        tasks=[[c['case_id'],qid] for c,qid in tasks],files={p:sha(p) for p in paths},model_revision=LAYA_REVISION,dataset_revision=TYPED_DECISIONS_REVISION,
        sae=provenance,selection_sha256=sha(args.selection),selection_source_manifest_sha256=selection['source_manifest_sha256'])
    out=args.out
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('resume manifest differs')
    else:atomic_json(out/'manifest.json',manifest)
    runner=Runner(load_agent(device='cuda'));runner.agent.model.requires_grad_(False)
    atomic_json(out/'runtime.json',dict(versions={n:importlib.metadata.version(n) for n in ('torch','laya','transformers')},
        cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),precision='FP32; AMP/TF32/MHA fastpath off',gpu=torch.cuda.get_device_name(),seed=0))
    for case,qid in tasks:
        name=hashlib.sha256(json.dumps([case['case_id'],qid]).encode()).hexdigest()[:24]+'.json';target=out/'decisions'/name
        if target.exists():
            old=json.loads(target.read_text())
            if (old['case_id'],old['qid'])!=(case['case_id'],qid):raise ValueError('resume identity differs')
            continue
        start=time.monotonic();print(f'[sae-features] START {case["case_id"]} {qid}',flush=True)
        fold=selection['fold_by_case'][case['case_id']];selected=selection['selections'][str(fold)]
        if case['case_id'] not in selected['evaluation_case_ids']:raise ValueError('crossfit fold differs')
        ref=json.loads((args.reference/'decisions'/'enc_17'/name).read_text())
        payload=collect(runner,sae,case,qid,selected,ref,args.n_null);payload['fold']=fold;payload['elapsed_s']=time.monotonic()-start
        atomic_json(target,payload);print(f'[sae-features] DONE seconds={payload["elapsed_s"]:.2f}',flush=True)
    if len(list((out/'decisions').glob('*.json')))!=len(tasks):raise ValueError('coverage differs')
    atomic_json(out/'resources.json',dict(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    atomic_json(out/'COMPLETE.json',dict(cases=len(cases),decisions=len(tasks),manifest_sha256=sha(out/'manifest.json'),
        decision_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted((out/'decisions').glob('*.json'))}))
    print(f'[sae-features] COMPLETE {out}',flush=True)

if __name__=='__main__':main()