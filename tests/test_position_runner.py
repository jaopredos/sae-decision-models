"""Synthetic tensors only: exercise hooks/mapping without downloading a model."""
import types
import unittest
from unittest.mock import patch

import torch
from run_position import Runner, behavioral, patching
from src.position import align_slots


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Identity()
        self.layer = torch.nn.Identity()
        self.head = torch.nn.Identity()

    def forward(self, ids, mask, positions, marker_mask, qtype):
        # Deliberately order-sensitive synthetic states; no pretrained model inference.
        length = ids.shape[1]
        pos = torch.arange(length).float()[None, :, None]
        h = torch.stack((ids.float(), ids.float() * 0.1), dim=-1) + pos
        h = self.head(self.layer(self.emb(h)))
        logits = h.gather(1, positions[..., None].expand(-1, -1, 2))[..., 0]
        return logits, torch.zeros(1, 2)


class SyntheticAgent:
    tok = types.SimpleNamespace(sep_token_id=102)
    parallel_options = False

    def __init__(self):
        self.model = SyntheticModel()

    def dtype_for(self, rows):
        return self.dtype if self.amp_enabled else torch.float32

    def _amp_enabled_for(self, rows):
        return self.amp_enabled

    def _encode_state(self, state, ids, internal, **kwargs):
        q = internal[ids[0]]
        order = q["option_order"]
        tokens, markers = [101, 30, 102], []
        for identity in order:
            markers.append(len(tokens))
            tokens += [103, 200 + identity]
        tokens += [102, 300 + len(str(state)), 102]
        return [{"ids": tokens, "markers": markers, "state_stats": {"truncated": False},
                 "options": {"options_distinct": len(order), "tokens_per_option": None}}]

    def predict(self, state, questions, **kwargs):
        qid = next(iter(questions))
        q = dict(questions[qid])
        keys = list(q["criteria"])
        q.setdefault("option_order", list(range(len(keys))))
        item = self._encode_state(state, [qid], {qid: q}, **kwargs)[0]
        ids = torch.tensor([item["ids"]])
        marker_pos = torch.tensor([item["markers"]])
        logits, _ = self.model(ids, torch.ones_like(ids), marker_pos,
                               torch.ones_like(marker_pos, dtype=torch.bool), torch.zeros(1))
        canonical = align_slots(logits[0].tolist(), q["option_order"])
        probabilities = torch.softmax(torch.tensor(canonical), dim=-1).tolist()
        answer = {"choice": keys[max(range(len(keys)), key=lambda i: canonical[i])],
                  "probabilities": dict(zip(keys, probabilities))}
        return {"answers": {qid: answer}}


class RunnerTests(unittest.TestCase):
    def test_behavior_and_bidirectional_patching(self):
        agent = SyntheticAgent()
        modules = [("emb", agent.model.emb), ("enc_01", agent.model.layer), ("head_1", agent.model.head)]
        with patch("src.activations.layer_modules", return_value=modules), \
             patch("src.activations._options_over_cap", return_value=0):
            runner = Runner(agent)
            case = {"case_id": "synthetic", "state": "real", "questions": {
                "q": {"type": "choice", "instructions": "pick", "criteria": {"a": "red", "b": "blue"}}}}
            observations, originals = behavioral(runner, case, "q", {"case_id": "donor", "state": "other"})
            self.assertEqual(len(observations), 8)
            self.assertEqual(set(originals), {(0, 1), (1, 0)})
            effects = patching(runner, case, "q", originals, n_null=2)
            final = [e for e in effects if e["layer"] == "head_1" and e["intervention"] == "all_markers"]
            self.assertEqual(len(final), 2)
            for e in final:
                self.assertAlmostEqual(e["transfer_ratio"], 1)
            for e in effects:
                if e["intervention"] == "same_order_noop":
                    self.assertEqual(e["change_norm"], 0)


if __name__ == "__main__":
    unittest.main()
