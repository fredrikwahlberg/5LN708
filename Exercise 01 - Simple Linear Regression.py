import unittest
from exercise1 import SimpleLinearRegression
import numpy as np

class TestSimpleLinearRegressionInstatiation(unittest.TestCase):
    def test_instantiation(self):
        try:
            SimpleLinearRegression()
        except NotImplementedError:
            print("Remember to remove exceptions.")


class TestSimpleLinearRegression(unittest.TestCase):

    def setUp(self):
        self.model = SimpleLinearRegression()

    def test_instantiation(self):
        self.assertTrue(hasattr(self.model, '_theta'), 
                        msg="Can't find the parameter vector.")
        self.assertTrue(self.model._theta is not None, 
                        msg="The parameter vector is not initialized.")
        self.assertIsInstance(self.model._theta, np.ndarray)
        self.assertEqual(len(self.model._theta), 2, 
                         msg="%i is the wrong number of parameters for the given order." % len(self.model._theta))

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
