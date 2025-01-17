import unittest
from exercise1 import LinearRegression
import numpy as np

class TestLinearRegressionFirstOrder(unittest.TestCase):

    def setUp(self):
        self.linreg = LinearRegression(n_order=1)

    def test_instantiation(self):
        self.assertTrue(hasattr(self.linreg, 'loss'), "Can't find the vector or list for the loss")
        self.assertIsInstance(self.linreg.loss, list)
        self.assertTrue(hasattr(self.linreg, 'theta'), "Can't find the parameter vector")
        self.assertTrue(self.linreg.theta is not None)
        self.assertEqual(len(self.linreg.theta), 2, "Wrong number of parameters for the given order")
        self.assertIsInstance(self.linreg.n_max_iterations, int)

    def test_single_prediction(self):
        X = np.vstack([3])
        self.linreg.theta[0] = 2
        self.linreg.theta[1] = 1
        pred = self.linreg.predict(X)
        self.assertEqual(pred[0], 5, "predict() returned %s" % pred)

    def test_several_predictions(self):
        X = np.vstack([1, 2, 3])
        self.linreg.theta[0] = 2
        self.linreg.theta[1] = 1
        pred = self.linreg.predict(X)
        self.assertEqual(pred[0], 3, "predict() returned %s" % pred)
        self.assertEqual(pred[1], 4, "predict() returned %s" % pred)
        self.assertEqual(pred[2], 5, "predict() returned %s" % pred)

    def test_fit(self):
        X = np.vstack(np.linspace(-5, 5, num=11))
        assert X.ndim == 2
        y = (X*3 - 2).ravel()
        assert y.ndim == 1
        old_score = self.linreg.score(X, y)
        self.assertTrue(len(self.linreg.loss) == 0, "List of loss scores should be empty at start")
        self.linreg.fit(X, y)
        self.assertTrue(len(self.linreg.loss) > 0, "No loss values recorded")
        self.assertGreater(old_score, self.linreg.score(X, y), 'A trained model should have a lower loss than an untrained.')

    def test_score(self):
        X = np.vstack(np.linspace(-5, 5, num=11))
        y = (X*3 - 2).ravel()
        self.linreg.theta[0] = -2
        self.linreg.theta[1] = 3
        self.assertAlmostEqual(self.linreg.score(X, y), 0, places=3, msg='The loss should be zero here.')
        self.linreg.theta[0] = -1
        self.assertAlmostEqual(self.linreg.score(X, y), 11, places=3, msg='There should be some loss here.')
        self.linreg.theta[0] = 0
        self.assertAlmostEqual(self.linreg.score(X, y), 11*2**2, places=3, msg='Are you using square loss?')

class TestLinearRegressionSecondOrder(unittest.TestCase):

    def setUp(self):
        self.linreg = LinearRegression(n_order=2)

    def test_instantiation(self):
        self.assertIsInstance(self.linreg.loss, list)
        self.assertTrue(self.linreg.theta is not None)
        self.assertEqual(len(self.linreg.theta), 3, "Wrong number of parameters for the given order")
        self.assertIsInstance(self.linreg.n_max_iterations, int)

    def test_single_prediction(self):
        X = np.vstack([3])
        self.linreg.theta[0] = 2
        self.linreg.theta[1] = 1
        self.linreg.theta[2] = 0
        pred = self.linreg.predict(X)
        self.assertEqual(pred[0], 5, "predict() returned %s" % pred)
        X = np.vstack([4])
        self.linreg.theta[0] = 2
        self.linreg.theta[1] = 1
        self.linreg.theta[2] = 3
        pred = self.linreg.predict(X)
        self.assertEqual(pred[0], 2+1*4+3*4**2, "predict() returned %s" % pred)

    def test_several_predictions(self):
        X = np.vstack([1, 2, 3])
        self.linreg.theta[0] = 2
        self.linreg.theta[1] = 1
        self.linreg.theta[2] = 0
        pred = self.linreg.predict(X)
        self.assertEqual(pred[0], 3, "predict() returned %s" % pred)
        self.assertEqual(pred[1], 4, "predict() returned %s" % pred)
        self.assertEqual(pred[2], 5, "predict() returned %s" % pred)

    def test_fit(self):
        X = np.vstack(np.linspace(-5, 5, num=11))
        y = (X*3 - 2).ravel()
        self.linreg.theta[:] = 0
        old_score = self.linreg.score(X, y)
        self.assertTrue(len(self.linreg.loss) == 0, "List of loss scores should be empty at start")
        self.linreg.fit(X, y)
        self.assertTrue(len(self.linreg.loss) > 0, "No loss values recorded")
        self.assertGreater(old_score, self.linreg.score(X, y), 'A trained model should have a lower loss than an untrained.')

    def test_score(self):
        X = np.vstack(np.linspace(-5, 5, num=11))
        y = (X*3 - 2).ravel()
        self.linreg.theta[0] = -2
        self.linreg.theta[1] = 3
        self.linreg.theta[2] = 0
        self.assertAlmostEqual(self.linreg.score(X, y), 0, places=3, msg='The loss should be zero here.')
        self.linreg.theta[0] = -1
        self.assertAlmostEqual(self.linreg.score(X, y), 11, places=3, msg='There should be some loss here.')
        self.linreg.theta[0] = 0
        self.assertAlmostEqual(self.linreg.score(X, y), 11*2**2, places=3, msg='Are you using square loss?')


if __name__ == '__main__':
    unittest.main(verbosity=2)
