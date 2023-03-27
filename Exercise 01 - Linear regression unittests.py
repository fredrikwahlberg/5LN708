import unittest
from exercise1 import LinearRegression
import numpy as np

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        self.linreg = LinearRegression()

    def test_single_prediction(self):
        X = np.asarray([3])
        self.linreg.slope = 2
        self.linreg.intercept = 1
        pred = self.linreg.predict(X)
        self.assertEqual(pred, 7, "predict() returned %s" % pred)

    # def test_several_predictions(self):
    #     pass

    # def test_fit(self):
        # self.assertTrue(len(self.linreg.loss_)>0)
    #     self.assertEqual(self.calculation.get_difference(), 6, 'The difference is wrong.')


if __name__ == '__main__':
    unittest.main()