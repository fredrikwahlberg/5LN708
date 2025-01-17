import unittest
from exercise2 import LinearRegression
import numpy as np

class TestLinearRegressionFirstOrder(unittest.TestCase):

    def setUp(self):
        self.model = LinearRegression(n_order=1, n_max_iter=5000)

    def test_instantiation(self):
        self.assertTrue(hasattr(self.model, 'loss_'), 
                        msg="Can't find the list for the loss")
        self.assertIsInstance(self.model.loss_, list,
                              msg="loss_ should be a list")
        self.assertTrue(hasattr(self.model, '_theta'), 
                        msg="Can't find the parameter vector")
        self.assertIsInstance(self.model._theta, np.ndarray)
        self.assertEqual(len(self.model._theta), 2, 
                         msg="%i is the wrong number of parameters for the given order." % len(self.model._theta))
        self.assertIsInstance(self.model.n_max_iter, int)

    def test_single_prediction(self):
        X = np.vstack([3])
        self.model._theta[0] = 2
        self.model._theta[1] = 1
        prediction = self.model.predict(X)
        self.assertEqual(prediction.ndim, 1, 
                         msg="predict() should return an array with the shape (n,)")
        self.assertEqual(prediction.shape[0], 1, 
                         msg="predict() returns the wrong number of items")
        self.assertEqual(prediction[0], 5,
                         msg="predict() returned %s, not [5]" % prediction)

    def test_several_predictions(self):
        X = np.vstack([1, 2, 3])
        self.model._theta[0] = 2
        self.model._theta[1] = 1
        prediction = self.model.predict(X)
        self.assertEqual(prediction.ndim, 1, 
                         msg="predict() should return an array with the shape (n,)")
        self.assertEqual(prediction.shape[0], 3, 
                         msg="predict() returns the wrong number of items")
        self.assertEqual(prediction[0], 3, 
                         msg="predict() returned %s, not [3, 4, 5]" % prediction)
        self.assertEqual(prediction[1], 4, 
                         msg="predict() returned %s, not [3, 4, 5]" % prediction)
        self.assertEqual(prediction[2], 5, 
                         msg="predict() returned %s, not [3, 4, 5]" % prediction)

    def test_fit(self):
        X = np.vstack(np.linspace(-5, 5, num=11))
        assert X.ndim == 2
        y = (X*3 - 2).ravel()
        assert y.ndim == 1
        old_score = self.model.score(X, y)
        self.assertTrue(len(self.model.loss_) == 0, 
                        msg="List of loss scores should be empty at start")
        m = self.model.fit(X, y)
        self.assertIsInstance(m, LinearRegression, 
                             msg="fit() should return self")
        self.assertTrue(len(self.model.loss_) > 0, 
                        msg="No loss values recorded")
        self.assertGreater(old_score, self.model.score(X, y), 
                           msg="A trained model should have a lower loss than an untrained.")

    def test_score(self):
        X = np.vstack(np.linspace(-5, 5, num=11))
        y = (X*3 - 2).ravel()
        self.model._theta[0] = -2
        self.model._theta[1] = 3
        self.assertAlmostEqual(self.model.score(X, y), 0, places=3, 
                               msg='The loss should be zero here.')
        self.model._theta[0] = -1
        self.assertAlmostEqual(self.model.score(X, y), 11, places=3, 
                               msg='There should be some loss here.')
        self.model._theta[0] = 0
        self.assertAlmostEqual(self.model.score(X, y), 11*(2**2), places=3, 
                               msg='Are you using square loss?')

class TestLinearRegressionSecondOrder(unittest.TestCase):

    def setUp(self):
        self.model = LinearRegression(n_order=2)

    def test_instantiation(self):
        self.assertEqual(len(self.model._theta), 3, 
                         msg="Wrong number of parameters for the given order")

    def test_single_prediction(self):
        X = np.vstack([3])
        self.model._theta[0] = 2
        self.model._theta[1] = 1
        self.model._theta[2] = 0
        pred = self.model.predict(X)
        self.assertEqual(pred[0], 5, 
                         msg="predict() returned %s" % pred)
        X = np.vstack([4])
        self.model._theta[0] = 2
        self.model._theta[1] = 1
        self.model._theta[2] = 3
        pred = self.model.predict(X)
        self.assertEqual(pred[0], 2+1*4+3*4**2, 
                         msg="predict() returned %s" % pred)

    def test_several_predictions(self):
        X = np.vstack([1, 2, 3])
        self.model._theta[0] = 2
        self.model._theta[1] = 1
        self.model._theta[2] = 0
        pred = self.model.predict(X)
        self.assertEqual(pred[0], 3, "predict() returned %s" % pred)
        self.assertEqual(pred[1], 4, "predict() returned %s" % pred)
        self.assertEqual(pred[2], 5, "predict() returned %s" % pred)


if __name__ == '__main__':
    unittest.main(verbosity=2)
