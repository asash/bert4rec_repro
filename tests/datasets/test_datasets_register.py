from aprec.datasets.datasets_register import DatasetsRegister
import unittest

class TestDatasetsRegister(unittest.TestCase):
    def test_register(self):
        register = DatasetsRegister()
        dataset = register["ml-100k"]()
        self.assertEquals(len(dataset), 100000)

if __name__ == "__main__":
    unittest.main()