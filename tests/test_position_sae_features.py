import unittest
from unittest.mock import patch
import torch
from src.sae import TopKSAE
from src.position_sae_features import feature_delta,order_gradient


class FeatureTests(unittest.TestCase):
    def test_residual_preservation_units_and_complement(self):
        sae=TopKSAE(8,32,k=4);sae.scale.fill_(0.23)
        r,d=torch.randn(6,8),torch.randn(6,8)
        zr,zd=sae.encode(r),sae.encode(d);features=[1,3,7]
        edited=zr.clone();edited[:,features]=zd[:,features]
        delta=feature_delta(sae,zd,zr,features)
        torch.testing.assert_close(delta,sae.decode(edited)-sae.decode(zr),atol=1e-5,rtol=1e-5)
        comp=[i for i in range(32) if i not in features]
        total=feature_delta(sae,zd,zr,list(range(32)))
        torch.testing.assert_close(total,delta+feature_delta(sae,zd,zr,comp),atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.equal(feature_delta(sae,zr,zr,features),torch.zeros_like(r)))

    def test_gradient_replay_matches_finite_difference_in_path(self):
        from run_position import Runner
        from run_position_sae import snapshot
        from run_position_factorial import full_patch
        from run_position_path import patch_hook
        from test_position_runner import SyntheticAgent
        from test_position_factorial import FactorialModel
        from test_position_path import ToyBlock
        agent=SyntheticAgent();agent.device="cpu";agent.model=FactorialModel();agent.model.head=ToyBlock()
        modules=[('emb',agent.model.emb),('enc_17',agent.model.layer),('enc_18',agent.model.head),('enc_final',agent.model.final)]
        q=dict(type='choice',instructions='pick',criteria={'a':'red','b':'blue'})
        case=dict(case_id='toy',state='case',workflow='toy',questions={'q':q})
        with patch('src.activations.layer_modules',return_value=modules),patch('src.activations._options_over_cap',return_value=0):
            runner=Runner(agent)
            with torch.inference_mode():
                d=snapshot(runner,case,'q',q,[0,1],'enc_17');r=snapshot(runner,case,'q',q,[1,0],'enc_17')
            g,error=order_gradient(runner,case,'q',q,d,r)
            self.assertLess(error,1e-4)
            delta=torch.zeros_like(g);delta[r['alignment']['groups']['nonmarkers']]=0.25
            c=torch.tensor(d['logits'])-torch.tensor(r['logits']);c-=c.mean()
            results=[]
            for sign in (-1,1):
                handle=runner.modules['enc_18'].register_forward_hook(patch_hook(r['alignment']['indices'],r['states']['enc_18'],r['alignment']['groups']['nonmarkers']))
                try:result=full_patch(runner,case,'q',q,r,'enc_17',r['states']['enc_17']+sign*delta)
                finally:handle.remove()
                results.append(torch.dot(torch.tensor(result['logits']),c))
            numeric=(results[1]-results[0])/2
            torch.testing.assert_close(numeric,(g*delta).sum(),atol=1e-3,rtol=.03)

class FeaturePipelineTests(unittest.TestCase):
    def test_collection_and_offline_audit_synthetic(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from run_position import Runner,atomic_json
        from run_position_sae import collect as collect_reference
        from run_position_sae_features import collect
        from analyze_position_sae_features import analyze
        from test_position_runner import SyntheticAgent
        from test_position_factorial import FactorialModel
        from test_position_path import ToyBlock
        agent=SyntheticAgent();agent.device='cpu';agent.model=FactorialModel();agent.model.head=ToyBlock()
        agent.tok.decode=lambda ids:' '.join(map(str,ids))
        modules=[('emb',agent.model.emb),('enc_17',agent.model.layer),('enc_18',agent.model.head),('enc_final',agent.model.final)]
        q=dict(type='choice',instructions='pick',criteria={'a':'red','b':'blue'})
        case=dict(case_id='toy',state='case',workflow='toy',questions={'q':q},gold={'q':{'label':'a'}})
        sae=TopKSAE(2,16,k=3);sae.scale.fill_(0.02)
        with patch('src.activations.layer_modules',return_value=modules),patch('src.activations._options_over_cap',return_value=0):
            runner=Runner(agent)
            with torch.inference_mode():ref=collect_reference(runner,sae,'enc_17',case,'q',1,None)
            data=collect(runner,sae,case,'q',dict(selected=[1,2],controls=[[3,4],[5,6]]),ref,4)
        self.assertEqual(len(data['patches']),28)
        self.assertTrue(all(c['error']<1e-4 for c in data['controls']))
        data['fold']=0
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'analysis';dest=root/'decisions'/'toy.json'
            atomic_json(dest,data)
            atomic_json(root/'manifest.json',dict(tasks=[['toy','q']],case_ids=['toy'],args={'n_null':4},selection_sha256='synthetic'))
            atomic_json(root/'COMPLETE.json',dict(decisions=1,manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                decision_sha256={'decisions/toy.json':hashlib.sha256(dest.read_bytes()).hexdigest()}))
            analyze(root,out,n_boot=20)
            audit=json.loads((out/'audit.json').read_text())
            self.assertEqual(audit['condition_rows'],28)
            # A saved metric mutation must fail even if the file hash is refreshed.
            data['patches'][0]['projection_numerator']+=1
            atomic_json(dest,data)
            complete=json.loads((root/'COMPLETE.json').read_text());complete['decision_sha256']['decisions/toy.json']=hashlib.sha256(dest.read_bytes()).hexdigest()
            atomic_json(root/'COMPLETE.json',complete)
            with self.assertRaisesRegex(ValueError,'metric differs'):analyze(root,out,n_boot=20)

if __name__=='__main__':unittest.main()