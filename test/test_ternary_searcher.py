import unittest
from src.model_calculations.ternary_searcher import TernarySearcher
import math

class TestTernarySearch(unittest.TestCase):
    def test_basic(self):
        expected_min=-2.0
        error=0.000000001
        def test_func(x):
            return (x-expected_min)**2
        ternary_searcher=TernarySearcher(test_func,error)
        self.assertAlmostEqual(expected_min,ternary_searcher.minimalize())