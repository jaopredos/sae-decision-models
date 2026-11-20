import unittest
from src.position import align_slots, center, conditions, donor_state_map, effect_metrics, orders


class PositionTests(unittest.TestCase):
    def test_identity_alignment(self):
        self.assertEqual(align_slots([30, 10, 20], [2, 0, 1]), [10, 20, 30])
        with self.assertRaises(ValueError):
            align_slots([1, 2], [0, 0])

    def test_orders_balanced(self):
        for k in (2, 3, 4, 5, 7):
            all_orders = orders(k)
            self.assertEqual(len(all_orders), len(set(map(tuple, all_orders))))
            for identity in range(k):
                counts = [sum(p.index(identity) == s for p in all_orders) for s in range(k)]
                self.assertEqual(len(set(counts)), 1)

    def test_projection_and_degenerate_contrast(self):
        self.assertAlmostEqual(effect_metrics([2, 0], [0, 2], [2, 0])["transfer_ratio"], 1)
        self.assertAlmostEqual(effect_metrics([2, 0], [0, 2], [0, 2])["transfer_ratio"], 0)
        self.assertIsNone(effect_metrics([2, 0], [12, 10], [0, 2])["transfer_ratio"])
        self.assertEqual(center([2, 0]), center([12, 10]))

    def test_controls_preserve_semantic_indices_and_inputs(self):
        q = {"type": "choice", "instructions": "pick", "criteria": {"B": "blue", "R": "red"}}
        variants = list(conditions(q, "real", "other", "c", "q"))
        self.assertEqual(list(variants[-1][1]["criteria"].values()), ["blue", "red"])
        self.assertEqual(variants[-1][3], ["opt_1", "opt_2"])
        self.assertEqual(q["criteria"], {"B": "blue", "R": "red"})

    def test_state_swap_stays_in_question_group_and_has_no_self_donor(self):
        cases = [{"case_id": str(i), "questions": {"x": i//2}, "state": str(i)} for i in range(4)]
        mapping = donor_state_map(cases)
        for c in cases:
            self.assertNotEqual(c["case_id"], mapping[c["case_id"]]["case_id"])
            self.assertEqual(c["questions"], mapping[c["case_id"]]["questions"])


if __name__ == "__main__":
    unittest.main()
