import unittest
from analyze_position_factorial import interaction_metrics


class FactorialAnalysisTests(unittest.TestCase):
    def test_additive_effects_have_zero_interaction_with_nonzero_baseline(self):
        result=interaction_metrics([5,1],[3,1],[4,1],[4,1],[5,1])
        self.assertEqual(result["change_norm"],0)
        self.assertEqual(result["projection_numerator"],0)

    def test_interaction_preserves_original_reference_term(self):
        result=interaction_metrics([5,1],[3,1],[4,1],[4,1],[6,0])
        self.assertAlmostEqual(result["transfer_ratio"],1)


if __name__=="__main__":
    unittest.main()
