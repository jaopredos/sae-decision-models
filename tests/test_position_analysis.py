import unittest
import pandas as pd
from analyze_position import behavior_metrics, cluster_interval, compare_conditions, paired_layer_contrast, projection_interval


class AnalysisTests(unittest.TestCase):
    def test_semantic_choices_and_physical_slots_are_distinct(self):
        rows = [{"order": [0, 1], "logits": [3, 1], "probabilities": [.88, .12],
                 "pred_index": 0, "official_pred_index": 0},
                {"order": [1, 0], "logits": [0, 2], "probabilities": [.12, .88],
                 "pred_index": 1, "official_pred_index": 1}]
        stats = behavior_metrics(rows, 0)
        self.assertEqual(stats["flip_any"], 1)
        self.assertEqual(stats["mean_order_accuracy"], .5)
        self.assertEqual(stats["slot_0"], 1)
        self.assertEqual(stats["slot_1"], 0)

    def test_cluster_bootstrap_is_invariant_to_duplicate_rows(self):
        data = pd.DataFrame({"case_id": ["a", "a", "b", "c"], "x": [1., 1., 0., .5]})
        duplicated = pd.concat([data, data], ignore_index=True)
        self.assertEqual(cluster_interval(data, "x", 200), cluster_interval(duplicated, "x", 200))

    def test_projection_interval_exact_transfer(self):
        data = pd.DataFrame({"case_id": ["a", "b"], "projection_numerator": [2., 9.],
                             "contrast_sqnorm": [2., 9.]})
        self.assertEqual(projection_interval(data, 200), [1., 1.])

    def test_state_swap_pairing_detects_changes_even_if_marginals_match(self):
        a = [{"order": [0, 1], "logits": [2., 0.], "probabilities": [.9, .1]}]
        b = [{"order": [0, 1], "logits": [0., 2.], "probabilities": [.1, .9]}]
        self.assertEqual(compare_conditions(a, b)["mean_choice_change_vs_original"], 1)
        self.assertAlmostEqual(compare_conditions(a, b)["mean_tv_vs_original"], .8)

    def test_paired_layer_bootstrap_uses_same_cases(self):
        rows = []
        for case, weight in [("a", 1.), ("b", 10.)]:
            for layer, factor in [("enc_17", .2), ("enc_18", .7)]:
                rows.append({"case_id": case, "layer": layer, "intervention_group": "all_markers",
                             "projection_numerator": weight*factor, "contrast_sqnorm": weight})
        result = paired_layer_contrast(pd.DataFrame(rows), n_boot=200)
        self.assertAlmostEqual(result["difference"], .5)
        for endpoint in result["ci"]:
            self.assertAlmostEqual(endpoint, .5)


if __name__ == "__main__":
    unittest.main()
