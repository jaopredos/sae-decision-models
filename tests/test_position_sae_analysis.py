import unittest
import pandas as pd
from analyze_position_sae import estimate


class EstimateTests(unittest.TestCase):
    def test_clustered_projection_uses_sums_not_mean_ratios(self):
        f=pd.DataFrame(dict(case_id=['a','a','b','b'],contrast_sqnorm=[1.,1.,9.,9.],
            projection_numerator=[1.,1.,0.,0.],transfer_ratio=[1.,1.,0.,0.],change_norm=[1.,1.,0.,0.],orthogonal_norm=[0.]*4))
        result=estimate(f,100)
        self.assertAlmostEqual(result['projection'],0.1)
        self.assertEqual(result['cases'],2)
        self.assertAlmostEqual(result['max_case_weight'],0.9)
        duplicated=estimate(pd.concat([f,f]),100)
        self.assertEqual(result['ci'],duplicated['ci'])

    def test_zero_contrast_is_explicitly_undefined(self):
        f=pd.DataFrame(dict(case_id=['a'],contrast_sqnorm=[0.],projection_numerator=[0.],
            transfer_ratio=[None],change_norm=[1.],orthogonal_norm=[1.]))
        result=estimate(f,10)
        self.assertIsNone(result['projection']);self.assertIsNone(result['ci'])
        self.assertEqual(result['undefined'],1)

class PairedDifferenceTests(unittest.TestCase):
    def test_noise_is_averaged_within_direction_before_contrast(self):
        from analyze_position_sae_diagnostics import paired_difference
        ref=dict(donor_logits=[2.,0.],recipient_logits=[0.,0.])
        rows=[dict(**ref,condition='full',patched_logits=[2.,0.]),
              dict(**ref,condition='reconstructed',patched_logits=[1.6,0.]),
              dict(**ref,condition='noise_reconstructed_0',patched_logits=[0.2,0.]),
              dict(**ref,condition='noise_reconstructed_1',patched_logits=[0.6,0.])]
        result=paired_difference(rows,'reconstructed','noise_mean')
        self.assertAlmostEqual(result['transfer_ratio'],0.6)

if __name__=='__main__':unittest.main()