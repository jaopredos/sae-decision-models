import unittest
import torch
from src.sae import TopKSAE
from src.position_sae import decompose


class DecompositionTests(unittest.TestCase):
    def test_units_domain_and_recombination(self):
        sae = TopKSAE(8, 32, k=4)
        sae.scale.fill_(0.3)
        sae.massive_threshold.fill_(20)
        r, d = torch.randn(7,8), torch.randn(7,8)
        d[0] *= 100; r[1] *= 100
        p = decompose(sae,d,r)
        self.assertFalse(bool(p['eligible'][0] or p['eligible'][1]))
        total = p['reconstructed_delta']+p['residual_delta']+p['outside_delta']
        torch.testing.assert_close(total,d-r,atol=1e-4,rtol=1e-5)
        e=p['eligible']
        expected=(p['donor_latents']-p['recipient_latents']) @ sae.W_dec / sae.scale
        torch.testing.assert_close(p['reconstructed_delta'][e],expected,atol=1e-5,rtol=1e-5)
        rev=decompose(sae,r,d)
        self.assertTrue(torch.equal(e,rev['eligible']))
        torch.testing.assert_close(rev['reconstructed_delta'],-p['reconstructed_delta'])

    def test_self_is_exact_identity(self):
        sae=TopKSAE(8,32,k=4); sae.scale.fill_(0.17)
        h=torch.randn(7,8)
        p=decompose(sae,h,h)
        self.assertTrue(torch.equal(p['reconstructed_delta'],torch.zeros_like(h)))
        self.assertTrue(torch.equal(p['residual_delta'],torch.zeros_like(h)))

class RunnerIntegrationTests(unittest.TestCase):
    def test_bidirectional_domain_and_path_controls(self):
        from unittest.mock import patch
        from run_position import Runner
        from run_position_sae import collect
        from test_position_runner import SyntheticAgent
        from test_position_factorial import FactorialModel
        from test_position_path import ToyBlock
        agent=SyntheticAgent();agent.model=FactorialModel();agent.model.head=ToyBlock()
        modules=[('emb',agent.model.emb),('enc_17',agent.model.layer),
                 ('enc_18',agent.model.head),('enc_final',agent.model.final)]
        case=dict(case_id='toy',state='case',workflow='toy',gold={'q':{'label':'a'}},
            questions={'q':dict(type='choice',instructions='pick',criteria={'a':'red','b':'blue'})})
        sae=TopKSAE(2,16,k=3);sae.scale.fill_(0.02)
        with patch('src.activations.layer_modules',return_value=modules),patch('src.activations._options_over_cap',return_value=0):
            with torch.inference_mode():
                payload=collect(Runner(agent),sae,'enc_17',case,'q',1,None)
        self.assertEqual(len(payload['patches']),66)
        self.assertEqual(len(payload['reconstruction_contrasts']),3)
        self.assertTrue(all(c['error']<1e-4 for c in payload['controls']))
        self.assertEqual({tuple(r['donor_order']) for r in payload['patches']},{(0,1),(1,0)})

if __name__=='__main__': unittest.main()