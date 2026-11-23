import unittest
import pandas as pd
from analyze_position_path import paired_contrast


class PathAnalysisTests(unittest.TestCase):
    def test_primary_contrast_is_difference_of_numerators_over_common_denominator(self):
        rows=[]
        for case,weight in (("a",1.),("b",10.)):
            for condition,coefficient in (("path",.4),("noise",.1)):
                rows.append({"case_id":case,"qid":"q","workflow":"w","direction":"forward",
                             "condition":condition,"projection_numerator":weight*coefficient,
                             "contrast_sqnorm":weight})
        result=paired_contrast(pd.DataFrame(rows),"path","noise",200)
        self.assertAlmostEqual(result["estimate"],.3)
        for bound in result["ci"]:self.assertAlmostEqual(bound,.3)


if __name__=="__main__":unittest.main()
