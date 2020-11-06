from random import randint
import unittest
from src.data_processing.dataset import Dataset
from src.data_processing.data_normalizer import DataNormalizer

import math
import random

class TestDataset(unittest.TestCase):
    def test_normalize_simple(self):
        def distribution():
            return randint(1,20)
        self.any_test(10,10,distribution)

    def any_test(self,x,y,dist):
        lines=[]
        for i in range(x):
            line=[]
            for j in range(y):
                line.append(dist())
            lines.append(line)
        dataset=Dataset(lines)
        norm_dataset=DataNormalizer().normalize(dataset)
        mean=norm_dataset.columnwise_mean()
        std=norm_dataset.columnwise_std()
        self.assertEqual(len(mean),y)
        self.assertEqual(len(std),y)
        for i in range(y):
            self.assertAlmostEqual(mean[i],0)

        for i in range(y):
            self.assertAlmostEqual(std[i],1)
