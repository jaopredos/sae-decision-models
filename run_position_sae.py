"""Discovery-only SAE compatibility and order-contrast pilot. GPU jobs only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from run_position import Runner, atomic_json
from run_position_factorial import full_patch
from run_position_path import patch_hook
from src.position import choose_cases, conditions, effect_metrics, stable_seed
from src.position_factorial import token_alignment


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def snapshot(runner,case,qid,q,order,layer):
    saved={}; handles=[]
    for label in (layer,'enc_18','enc_final'):
        def keep(_m,_a,out,key=label):
            h=out[0] if isinstance(out,tuple) else out
            saved[key]=h[0].float().cpu().clone()
        handles.append(runner.modules[label].register_forward_hook(keep))
    try: result=runner.run(case['state'],qid,q,order)
    finally:
        for h in handles:h.remove()
    result['alignment']=token_alignment(result,runner.agent.tok.sep_token_id)
    result['states']={k:v[result['alignment']['indices']] for k,v in saved.items()}
    return result


def checked_sae(root,name,device):
    import torch
    from src.sae import TopKSAE
    from src.activations import LAYA_REVISION
    path=root/name/'sae.pt'
    checks={line.split()[1]:line.split()[0] for line in (root/'SHA256SUMS.txt').read_text().splitlines()}
    digest=sha(path)
    if digest != checks[f'{name}/sae.pt']:raise ValueError('checkpoint hash differs')
    meta=json.loads((path.parent/'train_meta.json').read_text())
    # Training metadata contains torch.__version__ as TorchVersion, a str subclass.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck=torch.load(path,map_location='cpu',weights_only=True)
    cfg=ck['config']; args=meta['args']
    expected_layer={'ml_enc08_k32':'enc_08','ml_enc17_k32':'enc_17'}[name]
    if (args['repo'],args['subfolder'],args['revision'],args['layer'],args['seed']) != ('convaiinnovations/laya','multilingual',LAYA_REVISION,expected_layer,0):
        raise ValueError('training provenance incompatible')
    if (cfg['d_in'],cfg['n_latents'],cfg['k']) != (768,12288,32):raise ValueError('SAE config incompatible')
    sae=TopKSAE(**cfg);sae.load_state_dict(ck['state_dict'],strict=True);sae.eval()
    if abs(float(sae.scale)-meta['scale'])>1e-7 or not 0<float(sae.scale):raise ValueError('scale differs')
    if not all(torch.isfinite(v).all() for k,v in sae.state_dict().items() if k!='massive_threshold'):
        raise ValueError('non-finite SAE weights')
    threshold=float(sae.massive_threshold)
    if threshold<=0:raise ValueError('invalid threshold')
    report=dict(name=name,layer=expected_layer,config=cfg,scale=float(sae.scale),
                massive_threshold=threshold if threshold<float('inf') else 'inf',
                checkpoint_sha256=digest,metadata_sha256=sha(path.parent/'train_meta.json'),
                checkpoint_extra=ck.get('extra',{}),training_metadata=meta,
                capture='encoder block output before final_norm',normalization='global scalar only')
    return sae.to(device),report


def collect(runner,sae,layer,case,qid,n_null,reference):
    import torch
    from src.position_sae import decompose,moments
    q=next(conditions(case['questions'][qid],case['state'],case['state'],case['case_id'],qid))[1]
    k=len(q['criteria']); orders=[list(range(k)),list(range(1,k))+[0]]
    a,b=[snapshot(runner,case,qid,q,o,layer) for o in orders]
    runner.parity(case['state'],qid,q,a)
    if a['alignment']['tokens']!=b['alignment']['tokens'] or a['alignment']['groups']!=b['alignment']['groups']:
        raise ValueError('semantic token alignment differs')
    if a['states'][layer].shape[1]!=sae.d_in:raise ValueError('hidden dimension differs')
    p=decompose(sae,a['states'][layer],b['states'][layer]); e=p['eligible']
    groups=a['alignment']['groups']; n=len(e)
    valid_indices=torch.where(e)[0].tolist()
    local={idx:j for j,idx in enumerate(valid_indices)}
    fidelity=[]; features=[]; recon_runs={}
    for group,idx in groups.items():
        good=[i for i in idx if e[i]]; loc=[local[i] for i in good]
        delta=a['states'][layer][good]-b['states'][layer][good]
        dr=p['reconstructed_delta'][good]
        for order,snap,recon in ((a['order'],a,p['donor_reconstruction']),(b['order'],b,p['recipient_reconstruction'])):
            # Position strata are physical ordinal marker positions, or all nonmarkers.
            subsets={'pooled':list(range(len(good)))}
            if group=='markers':
                for j,i in enumerate(good):
                    identity=idx.index(i); ordinal=order.index(identity)
                    subsets.setdefault(f'ordinal_{ordinal}',[]).append(j)
            for stratum,sub in subsets.items():
                token_idx=[good[j] for j in sub]; latent_idx=[loc[j] for j in sub]
                fidelity.append(dict(group=group,order=order,stratum=stratum,total=len(idx),eligible=len(good),
                    **moments(snap['states'][layer][token_idx],recon[latent_idx])))
        dz=p['donor_latents'][loc]-p['recipient_latents'][loc]
        features.append(dict(group=group,eligible_mask=e.tolist(),delta_latents=dz.tolist(),token_count=len(good),
            sum_abs_delta=dz.abs().sum(0).tolist(),sum_delta=dz.sum(0).tolist(),
            active_count=((p['donor_latents'][loc]>0)|(p['recipient_latents'][loc]>0)).sum(0).tolist(),
            delta_sse=float((delta-dr).square().sum()),delta_sqnorm=float(delta.square().sum()),
            delta_dot=float((delta*dr).sum()),reconstruction_delta_sqnorm=float(dr.square().sum())))
        # Replace both clean orders using the same symmetric eligible domain.
        outputs=[]
        for snap,recon in ((a,p['donor_reconstruction']),(b,p['recipient_reconstruction'])):
            h=snap['states'][layer].clone(); h[good]=recon[loc]
            out=full_patch(runner,case,qid,q,snap,layer,h);outputs.append(out)
        recon_runs[group]=outputs
    rows=[]; controls=[]
    for direction,(d,r) in enumerate(((a,b),(b,a))):
        sign=1 if direction==0 else -1
        clean=r['states'][layer]
        positions=r['alignment']['indices']
        final=full_patch(runner,case,qid,q,r,'enc_final',d['states']['enc_final'])
        err=max(abs(x-y) for x,y in zip(final['logits'],d['logits']))
        if err>1e-4:raise ValueError('final alignment failed')
        controls.append(dict(control='final_alignment',error=err))
        for group in ('markers','nonmarkers','all_tokens'):
            idx=groups[group]; mask=torch.zeros(n,dtype=torch.bool); mask[idx]=True
            good=mask&e; outside=mask&~e
            delta=d['states'][layer]-clean
            direct=torch.zeros_like(clean);direct[good]=delta[good]
            outdelta=torch.zeros_like(clean);outdelta[outside]=delta[outside]
            rec=p['reconstructed_delta']*sign*mask[:,None]
            res=p['residual_delta']*sign*mask[:,None]
            whole=torch.zeros_like(clean);whole[idx]=delta[idx]
            variants={'self':torch.zeros_like(clean),'full':whole,'eligible':direct,
                      'reconstructed':rec,'residual':res,'outside':outdelta,
                      'recombined':rec+res+outdelta}
            for repeat in range(n_null):
                gen=torch.Generator().manual_seed(stable_seed('sae-audit-v1',case['case_id'],qid,layer,group,d['order'],repeat)%(2**63-1))
                noise=torch.randn(clean.shape,generator=gen)
                noise*=rec.norm(dim=-1,keepdim=True)/noise.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                variants[f'noise_reconstructed_{repeat}']=noise
            modes=['direct']+(['path'] if layer=='enc_17' and group=='nonmarkers' else [])
            for mode in modes:
                outputs={}
                for condition,change in (list(variants.items())+([('reset_all18',whole)] if mode=='path' else [])):
                    handles=[]
                    if mode=='path':
                        handles.append(runner.modules['enc_18'].register_forward_hook(patch_hook(positions,r['states']['enc_18'],groups['all_tokens'] if condition=='reset_all18' else groups['nonmarkers'])))
                    try: result=full_patch(runner,case,qid,q,r,layer,clean+change)
                    finally:
                        for handle in handles:handle.remove()
                    outputs[condition]=result['logits']
                    if condition in ('self','reset_all18'):
                        error=max(abs(x-y) for x,y in zip(result['logits'],r['logits']))
                        if error>1e-5:raise ValueError('self patch failed')
                        controls.append(dict(control='self',error=error))
                    if reference and layer=='enc_17' and group=='nonmarkers' and condition=='full':
                        key='to_markers18' if mode=='path' else 'nonmarkers17'
                        ref=next(x for x in reference['patches'] if x['condition']==key and x['donor_order']==d['order'])
                        error=max(abs(x-y) for field,now in (('donor_logits',d['logits']),('recipient_logits',r['logits']),('patched_logits',result['logits'])) for x,y in zip(now,ref[field]))
                        if error>1e-4:raise ValueError(f'previous path reference differs: {error}')
                        controls.append(dict(control=f'reference_{mode}',error=error))
                    rows.append(dict(group=group,mode=mode,condition=condition,donor_order=d['order'],recipient_order=r['order'],
                        tokens=len(idx),eligible=int(good.sum()),outside=int(outside.sum()),delta_norm=float(change.norm()),
                        donor_logits=d['logits'],recipient_logits=r['logits'],patched_logits=result['logits'],
                        donor_probabilities=d['probabilities'],recipient_probabilities=r['probabilities'],patched_probabilities=result['probabilities'],
                        **effect_metrics(d['logits'],r['logits'],result['logits'])))
                error=max(abs(x-y) for x,y in zip(outputs['full'],outputs['recombined']))
                if error>1e-4:raise ValueError(f'recombination logits differ: {error}')
                controls.append(dict(control='recombination',error=error))
    # Compare contrast from replacements against the clean contrast, without renormalizing.
    replacement=[]
    for group,outs in recon_runs.items():
        contrast=[x-y for x,y in zip(outs[0]['logits'],outs[1]['logits'])]
        original=[x-y for x,y in zip(a['logits'],b['logits'])]
        replacement.append(dict(group=group,clean_contrast=original,reconstructed_contrast=contrast,
            original_logits=[a['logits'],b['logits']],reconstructed_logits=[o['logits'] for o in outs],
            original_probabilities=[a['probabilities'],b['probabilities']],reconstructed_probabilities=[o['probabilities'] for o in outs],
            **effect_metrics(original,[0.]*k,contrast)))
    return dict(case_id=case['case_id'],qid=qid,workflow=case['workflow'],layer=layer,order_pair=orders,
        option_ids=a['ids'],gold=case['gold'][qid],seq_len=n,controls=controls,patches=rows,
        reconstruction_contrasts=replacement,fidelity=fidelity,features=features,
        algebra_error=p['algebra_error'],geometry=dict(positions=[a['positions'],b['positions']],
        option_lengths=a['option_lengths'],tokens=a['alignment']['tokens']))


def main():
    import resource
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=2)
    ap.add_argument('--n-null',type=int,default=1)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.limit<1 or args.n_null<1:ap.error('positive limits required')
    import torch
    from src.activations import load_agent,LAYA_REVISION
    from src.datasets import load_typed_cases,typed_dev_ids,TYPED_DECISIONS_REVISION
    from src.models import set_all_seeds
    if not torch.cuda.is_available():raise RuntimeError('GPU job required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.environ['LAYA_CUDA_AMP']='fp32'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False);set_all_seeds(0)
    cases=choose_cases(load_typed_cases('train'),'discovery',typed_dev_ids(),args.limit)
    tasks=[(c,qid) for c in cases for qid,q in c['questions'].items() if q['type']=='choice']
    saes={}; provenance={}
    for name in ('ml_enc17_k32','ml_enc08_k32'):
        sae,report=checked_sae(args.weights,name,'cuda');saes[report['layer']]=sae;provenance[report['layer']]=report
    paths=['run_position_sae.py','src/position_sae.py','run_position.py','run_position_factorial.py','run_position_path.py',
           'src/position.py','src/position_factorial.py','src/activations.py','src/datasets.py','src/sae.py','src/models.py',
           'uv.lock','pyproject.toml','docs/position-sae-mechanism/PROTOCOL.md']
    manifest=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        case_ids=[c['case_id'] for c in cases],tasks=[[c['case_id'],qid] for c,qid in tasks],
        model_revision=LAYA_REVISION,dataset_revision=TYPED_DECISIONS_REVISION,
        files={p:sha(p) for p in paths},saes=provenance,reference_manifest_sha256=sha(args.reference/'manifest.json'))
    out=args.out
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('resume manifest changed')
    else:atomic_json(out/'manifest.json',manifest)
    runner=Runner(load_agent(device='cuda'))
    runtime=dict(python=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ('torch','laya','transformers','numpy')},
        gpu=torch.cuda.get_device_name(),cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        precision='FP32; AMP/TF32/MHA fastpath disabled',cuda=torch.version.cuda,seed=0)
    atomic_json(out/'runtime.json',runtime)
    for layer,sae in saes.items():
        for case,qid in tasks:
            digest=hashlib.sha256(json.dumps([case['case_id'],qid]).encode()).hexdigest()[:24]
            target=out/'decisions'/layer/f'{digest}.json'
            if target.exists():
                old=json.loads(target.read_text())
                if (old['case_id'],old['qid'],old['layer'])!=(case['case_id'],qid,layer):raise ValueError('resume identity differs')
                continue
            print(f'[sae-audit] START {layer} {case["case_id"]} {qid}',flush=True)
            start=time.monotonic()
            ref=json.loads((args.reference/'decisions'/f'{digest}.json').read_text())
            with torch.inference_mode():payload=collect(runner,sae,layer,case,qid,args.n_null,ref)
            payload['elapsed_s']=time.monotonic()-start
            atomic_json(target,payload)
            print(f'[sae-audit] DONE {layer} seconds={payload["elapsed_s"]:.2f}',flush=True)
    expected=len(tasks)*len(saes)
    if len(list((out/'decisions').glob('*/*.json')))!=expected:raise ValueError('coverage differs')
    atomic_json(out/'resources.json',dict(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    atomic_json(out/'COMPLETE.json',dict(cases=len(cases),decisions=len(tasks),layer_decisions=expected,manifest_sha256=sha(out/'manifest.json'),
        decision_sha256={str(p.relative_to(out)):sha(p) for p in sorted((out/'decisions').glob('*/*.json'))}))
    print(f'[sae-audit] COMPLETE {out}',flush=True)

if __name__=='__main__':main()