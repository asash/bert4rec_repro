import unittest

class TestAddMaskHistoryVectorizer(unittest.TestCase):
    def test_add_mask(self):
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        seq = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        vectorizer = AddMaskHistoryVectorizer()
        vectorizer.set_sequence_len(4)
        vectorizer.set_padding_value(7)
        vectorized = vectorizer(seq)
        self.assertEqual(len(vectorized), 4)
        self.assertEqual(list(vectorized), [3, 4, 5, 8])

        seq = [(3, 3), (4, 4), (5, 5)]
        vectorizer = AddMaskHistoryVectorizer()
        vectorizer.set_sequence_len(4)
        vectorizer.set_padding_value(7)
        vectorized = vectorizer(seq)
        self.assertEqual(len(vectorized), 4)
        self.assertEqual(list(vectorized), [3, 4, 5, 8])


        seq = [(4, 4), (5, 5)]
        vectorizer = AddMaskHistoryVectorizer()
        vectorizer.set_sequence_len(4)
        vectorizer.set_padding_value(7)
        vectorized = vectorizer(seq)
        self.assertEqual(len(vectorized), 4)
        self.assertEqual(list(vectorized), [7, 4, 5, 8])

if __name__ == "__main__":
    unittest.main()
    