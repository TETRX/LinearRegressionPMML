import unittest
from src.data_processing.dataset import Dataset
import math

class TestDataset(unittest.TestCase):
    def test_basic(self):
        lines=[[1,2,3,4,5,6],[7,8,9,10,11,12]]
        dataset=Dataset(lines)
        self.assertEqual(lines,dataset.data_lines())

    def test_apply_funcs(self):
        lines=[[1,2,3,4,5,6],[7,8,9,10,11,12]]
        dataset=Dataset(lines)
        def powers(i):
            return lambda x: x**i
        funcs=[powers(i+1) for i in range(6)]
        expected_lines=[[1,4,27,256,3125,46656],[7,64,729,10000,161051,2985984]]
        new_dataset=dataset.apply_functions(funcs)
        self.assertEqual(expected_lines,new_dataset.data_lines())

    def test_mean(self):
        lines=[[1,3],[5,2],[3,4],[1,2]]
        dataset=Dataset(lines)
        expected_mean=[10/4,11/4]
        self.assertEqual(expected_mean,dataset.columnwise_mean())

    def test_std(self):
        lines=[[1,3],[5,2],[3,4],[1,1]]
        dataset=Dataset(lines)
        sum=[0,0]
        std=[0,0]
        for i in range(len(lines[0])):
            for line in lines:
                sum[i]+=line[i]
            sum[i]/=len(lines)
            for line in lines:
                std[i]+=(line[i]-sum[i])**2
            std[i]/=len(lines)
            std[i]=math.sqrt(std[i])
        self.assertEqual(std,dataset.columnwise_std())