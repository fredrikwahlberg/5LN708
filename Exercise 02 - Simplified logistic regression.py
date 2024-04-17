import unittest
from exercise2 import SimplifiedLogisticRegression
import numpy as np

class TestSimplifiedLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.X1 = np.asarray([[1, 1], [-1, -1]])
        self.y1 = np.asarray([0, 1])
        n_samples = 100
        self.X2 = np.concatenate((np.ones((n_samples//2, 2)), -np.ones((n_samples//2, 2))))
        rng = np.random.default_rng(seed=0)
        self.X2 += rng.normal(0, .4, size=self.X2.shape)
        self.y2 = np.concatenate((np.zeros(n_samples//2), np.ones(n_samples//2)))
        self.model = SimplifiedLogisticRegression()

    def test_instantiation(self):
        self.assertTrue(hasattr(self.model, 'loss'), "Can't find the list for the loss")
        self.assertIsInstance(self.model.loss, list)
        self.assertTrue(hasattr(self.model, '_theta'), "Can't find the parameter vector _theta")
        self.assertTrue(self.model._theta is not None, "The parameter vector _theta does not get initialized.")
        self.assertEqual(len(self.model._theta), 3, "Wrong number of parameters for the model")
        self.assertTrue(hasattr(self.model, 'n_max_iterations'), "Can't find the n_max_iterations parameter")
        self.assertIsInstance(self.model.n_max_iterations, int, "n_max_iterations should be an int")
        for method_name in ['predict', 'score', 'fit', '_fit', '_soft_predict', '_loss', '_sigmoid']:
            self.assertTrue(hasattr(self.model, 'loss'), "Can't find the method %s" % method_name)

    def test_sigmoid(self):
        self.assertAlmostEqual(self.model._sigmoid(0), .5, places=3, msg='Sigmoid function seems off')
        self.assertAlmostEqual(self.model._sigmoid(-1e2), 0, places=3, msg='Sigmoid function seems off')
        self.assertAlmostEqual(self.model._sigmoid(1e2), 1, places=3, msg='Sigmoid function seems off')

    def test_predict(self):
        # API compliance
        pred = self.model.predict(self.X2)
        self.assertIsInstance(pred, np.ndarray, msg="The prediction output should be a numpy ndarray")
        self.assertEqual(pred.ndim, 1, msg="Prediction output should be a flat vector")
        # Specifics
        self.model._theta = np.asarray([0, 0, 0])
        pred = self.model.predict(np.vstack([1, 1]).T)
        self.assertEqual(pred, 0, msg="predict() returned %s" % pred)
        self.model._theta = np.asarray([0, 1, 0])
        pred = self.model.predict(np.vstack([1, 1]).T)
        self.assertEqual(pred, 1, msg="predict() returned %s" % pred)
        self.model._theta = np.asarray([0, 0, 1])
        pred = self.model.predict(np.vstack([1, 1]).T)
        self.assertEqual(pred, 1, msg="predict() returned %s" % pred)

    def test_predict2(self):
        # Force 'trained' values
        self.model._theta = np.asarray([0, -1, -1])
        # Larger data
        pred = self.model.predict(self.X2)
        self.assertEqual(pred[0], self.y2[0], msg="predict() returned %s" % pred)
        self.assertEqual(pred[-1], self.y2[-1], msg="predict() returned %s" % pred)
        # Flipped data
        pred = self.model.predict(self.X2[::-1, :])
        self.assertEqual(pred[-1], self.y1[0], msg="predict() returned %s" % pred)
        self.assertEqual(pred[0], self.y1[-1], msg="predict() returned %s" % pred)

    def test_fit(self):
        # API compliance
        model = self.model.fit(self.X2, self.y2)
        self.assertIsInstance(model, SimplifiedLogisticRegression, 
                              msg="The fit function should end with 'return self'")
        # Specifics
        self.model._theta[:] = 0
        self.model.fit(self.X1, self.y1)
        for i in [1, 2]:
            self.assertTrue(self.model._theta[i] < 0, 
                            msg="Both 'slopes' should be negative on the given test data")

    def test_score(self):
        # API compliance
        score = self.model.score(self.X1, self.y1)
        self.assertIsInstance(score, float, msg="The score should be a single value")
        # Specifics
        self.assertTrue(score>=0, msg="Accuracy is a value between 0 and 1")
        self.assertTrue(score<=1, msg="Accuracy is a value between 0 and 1")


if __name__ == '__main__':
    unittest.main(verbosity=2)
