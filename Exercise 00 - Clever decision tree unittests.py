import unittest
from exercise0 import CleverDecisionTree
import numpy as np

class TestCleverDecisionTree(unittest.TestCase):

    def setUp(self):
        self.X = np.tile([[0.9615202, 4.43376825], [1.99127366, 0.80487522], [-1.6656056, 2.88982984]], (70, 1))
        np.random.seed(0)
        self.X += np.random.normal(0, .7, size=self.X.shape)
        self.y = np.tile([0, 1, 2], 70)

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
        self.assertIsInstance(score, float, "score should return the accuracy as a float between 0 and 1.")
        self.assertGreater(score, .90, "The accuracy is not high enough.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
