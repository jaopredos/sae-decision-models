import unittest
from unittest.mock import patch
import torch
from run_position import Runner
from run_position_path import collect_path
from test_position_factorial import FactorialModel
from test_position_runner import SyntheticAgent


class ToyAttention(torch.nn.Module):
    def forward(self,h):
        weight=torch.arange(1,h.shape[1]+1,dtype=h.dtype)[None,:,None]
        context=(h*weight).mean(1,keepdim=True)
        return h*context/10000,None


class ToyMLP(torch.nn.Module):
    def forward(self,h):return h.square()*.0001


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__();self.attn=ToyAttention();self.mlp=ToyMLP()
    def forward(self,h):
        a,_=self.attn(h);h=h+a
        return h+self.mlp(h)


class PathTests(unittest.TestCase):
    def test_path_resets_and_attention_clamp_preserve_recipient(self):
        agent=SyntheticAgent();agent.model=FactorialModel();agent.model.head=ToyBlock()
        modules=[("emb",agent.model.emb),("enc_17",agent.model.layer),
                 ("enc_18",agent.model.head),("enc_final",agent.model.final)]
        case={"case_id":"toy","state":"case","workflow":"toy",
              "questions":{"q":{"type":"choice","instructions":"pick","criteria":{"a":"red","b":"blue"}}}}
        with patch("src.activations.layer_modules",return_value=modules),patch("src.activations._options_over_cap",return_value=0):
            data=collect_path(Runner(agent),case,"q",n_null=1)
        self.assertEqual(len(data["patches"]),16)
        for row in data["patches"]:
            if row["condition"] in ("self_patch","reset_all18","to_markers18_clamp_attention"):
                self.assertLess(max(abs(a-b) for a,b in zip(row["patched_logits"],row["recipient_logits"])),1e-5)


if __name__=="__main__":unittest.main()
