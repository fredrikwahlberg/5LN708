import unittest
from exercise0 import CleverDecisionTree
import numpy as np

class TestCleverDecisionTree(unittest.TestCase):

    def setUp(self):
        from sklearn.datasets import make_blobs
        self.X, self.y = make_blobs(n_samples=200, centers=3, cluster_std=.7, n_features=2, random_state=0)
        self.clf = CleverDecisionTree()

    def test_instantiation(self):
        self.assertTrue(hasattr(self.clf, 'predict'), "You need a predict method for making predictions.")
        self.assertTrue(hasattr(self.clf, 'score'), "You need a score method for evaluating the classifier.")

    def test_several_predictions(self):
        pred = self.clf.predict(self.X)
        self.assertIsInstance(pred, np.ndarray, "predict should return a numpy array")
        self.assertEqual(pred.ndim, 1, "predict() should return a flat vector (try using .ravel() )")

    def test_score(self):
        score = self.clf.score(self.X, self.y)
        self.assertIsInstance(score, float, "score should return the accuarcy as a float between 0 and 1.")
        self.assertGreater(score, .90, "The accuracy is not high enough.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
