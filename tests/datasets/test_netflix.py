from unittest import TestCase
import unittest
from aprec.datasets.datasets_register import DatasetsRegister


class TestNetflixDataset(TestCase):
    @unittest.skip
    def test_netflix(self):
        dataset = DatasetsRegister()["netflix"]()
        print(len(dataset))

if __name__ == "__main__":
    unittest.main()    
