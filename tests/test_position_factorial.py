import types
import unittest
from unittest.mock import patch

import torch
from run_position import Runner
from run_position_factorial import collect_decision
from src.position_factorial import token_alignment
from test_position_runner import SyntheticAgent, SyntheticModel


class FactorialModel(SyntheticModel):
    def __init__(self):
        super().__init__()
        self.final=torch.nn.Identity()

    def forward(self, ids, mask, positions, marker_mask, qtype):
        pos=torch.arange(ids.shape[1]).float()[None,:,None]
        h=torch.stack((ids.float(),ids.float()*.1),dim=-1)+pos
        h=self.final(self.head(self.layer(self.emb(h))))
        logits=h.gather(1,positions[...,None].expand(-1,-1,2))[...,0]
        return logits,torch.zeros(1,2)


class FactorialTests(unittest.TestCase):
    def test_repeated_tokens_variable_lengths_are_aligned_by_roles(self):
        a={"question_tokens":[101,5,102],"option_spans":[[103,7,7],[103,7]],
           "positions":[3,6],"state_tokens":[9],"seq_len":10}
        b={**a,"positions":[5,3]}
        aa,bb=token_alignment(a,102),token_alignment(b,102)
        self.assertEqual(aa["tokens"],bb["tokens"])
        self.assertEqual(bb["indices"],[0,1,2,5,6,7,3,4,8,9])
        self.assertEqual(bb["groups"]["markers"],[3,6])
        self.assertEqual(len(bb["groups"]["nonmarkers"]),8)
        with self.assertRaises(ValueError):
            token_alignment({**a,"positions":[3,3]},102)

    def test_full_token_transplant_and_factorial_coverages(self):
        agent=SyntheticAgent()
        agent.model=FactorialModel()
        modules=[("emb",agent.model.emb),("enc_17",agent.model.layer),
                 ("enc_18",agent.model.head),("enc_final",agent.model.final)]
        case={"case_id":"synthetic","workflow":"synthetic","state":"real",
              "questions":{"q":{"type":"choice","instructions":"pick","criteria":{"a":"red","b":"blue"}}},
              "gold":{"q":{"label":"a"}}}
        with patch("src.activations.layer_modules",return_value=modules), \
             patch("src.activations._options_over_cap",return_value=0):
            data=collect_decision(Runner(agent),case,"q",n_null=1)
        self.assertEqual(len(data["patches"]),36)
        for row in data["patches"]:
            if row["intervention"]=="self_patch":
                self.assertEqual(row["change_norm"],0)
            if row["layer"]=="enc_final" and row["group"]=="all_tokens" and row["intervention"]=="aligned":
                self.assertAlmostEqual(row["transfer_ratio"],1)


if __name__=="__main__":
    unittest.main()
