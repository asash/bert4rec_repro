import unittest

class TestItemId(unittest.TestCase):
    def test_get_id(self):
        from aprec.utils.item_id import ItemId
        items_dict = ItemId()
        self.assertEqual(items_dict.get_id("aaa"), 0)
        self.assertEqual(items_dict.get_id("bbb"), 1)
        self.assertEqual(items_dict.get_id("ccc"), 2)
        self.assertEqual(items_dict.get_id("ddd"), 3)
        self.assertEqual(items_dict.get_id("aaa"), 0)
        self.assertEqual(items_dict.get_id("ccc"), 2)
        self.assertEqual(items_dict.reverse_id(2), "ccc")
        self.assertTrue(items_dict.has_id(2))
        self.assertFalse(items_dict.has_id(4))
        self.assertTrue(items_dict.has_item("aaa"))
        self.assertFalse(items_dict.has_item("fff"))
        self.assertRaises(KeyError, items_dict.reverse_id, 4)
        

if __name__ == "__main__":
    unittest.main()
