import unittest
from src.data_processing.dataset import Dataset
from src.data_processing.data_divider import DataDivider


class TestDataDivider(unittest.TestCase):
    def test_basic(self):
        data_lines=[[i,i] for i in range(100)]
        dataset=Dataset(data_lines)
        data_divider=DataDivider()
        divided_sets=data_divider.divide([8,1,1],dataset)
        self.assertEqual(len(divided_sets[0].data_lines()),80)
        self.assertEqual(len(divided_sets[1].data_lines()),10)
        self.assertEqual(len(divided_sets[2].data_lines()),10)