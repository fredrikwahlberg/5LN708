import unittest
from exercise4 import FeedForwardNeuralNetwork
import numpy as np

class TestFeedForwardNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.X1 = np.asarray([[1, 1], [-1, -1]])
        self.y1 = np.asarray([0, 1])
        n_samples = 100
        self.X2 = np.concatenate((np.ones((n_samples//2, 2)), -np.ones((n_samples//2, 2))))
        rng = np.random.default_rng(seed=0)
        self.X2 += rng.normal(0, .4, size=self.X2.shape)
        self.y2 = np.concatenate((np.zeros(n_samples//2), np.ones(n_samples//2)))
        self.model = FeedForwardNeuralNetwork(n_hidden=3)

    def test_instantiation(self):
        # Sklearn API compliance
        for method_name in ['predict', 'score', 'fit']:
            self.assertTrue(hasattr(self.model, 'loss'), "Can't find the method %s" % method_name)
        # Specifics
        for method_name in ['_fit', '_soft_predict', '_loss', '_sigmoid', 'theta_']:
            self.assertTrue(hasattr(self.model, 'loss'), "Can't find the method %s" % method_name)
        self.assertTrue(hasattr(self.model, 'loss'), "Can't find the list for the loss")
        self.assertIsInstance(self.model.loss, list)
        self.assertTrue(hasattr(self.model, '_theta1'), "Can't find the parameter matrix _theta1")
        self.assertTrue(hasattr(self.model, '_theta2'), "Can't find the parameter matrix _theta2")
        self.assertTrue(hasattr(self.model, 'n_max_iterations'), "Can't find the n_max_iterations parameter")
        self.assertIsInstance(self.model.n_max_iterations, int, "n_max_iterations should be an int")

    def test_setters_getters(self):
        m1 = FeedForwardNeuralNetwork(n_hidden=3)
        m2 = FeedForwardNeuralNetwork(n_hidden=4)
        self.assertEqual(len(m1.theta_), 12, "Wrong number of parameters for the model")
        self.assertEqual(len(m2.theta_), 16, "Wrong number of parameters for the model")
        m1.theta_ = np.arange(len(m1.theta_))
        m2.theta_ = np.arange(len(m2.theta_))
        msg = "Parameter matrix positions don't seem to line up correctly"
        self.assertEqual(m1._theta1[0, 0], 0, msg)
        self.assertEqual(m1._theta1[-1, -1], 8, msg)
        self.assertEqual(m1._theta2[0, 0], 9, msg)
        self.assertEqual(m1._theta2[-1, -1], 11, msg)
       
    def test_sigmoid(self):
        self.assertAlmostEqual(self.model._sigmoid(0), .5, places=3, msg='Sigmoid function seems off')
        self.assertAlmostEqual(self.model._sigmoid(-1e2), 0, places=3, msg='Sigmoid function seems off')
        self.assertAlmostEqual(self.model._sigmoid(1e2), 1, places=3, msg='Sigmoid function seems off')

    def test_predict(self):
        # Sklearn API compliance
        pred = self.model.predict(self.X2)
        self.assertIsInstance(pred, np.ndarray, msg="The prediction output should be a numpy ndarray")
        self.assertEqual(pred.ndim, 1, msg="Prediction output should be a flat vector")
    #     # Specifics
    #     self.model._theta = np.asarray([0, 0, 0])
    #     pred = self.model.predict(np.vstack([1, 1]).T)
    #     self.assertEqual(pred, 0, msg="predict() returned %s" % pred)
    #     self.model._theta = np.asarray([0, 1, 0])
    #     pred = self.model.predict(np.vstack([1, 1]).T)
    #     self.assertEqual(pred, 1, msg="predict() returned %s" % pred)
    #     self.model._theta = np.asarray([0, 0, 1])
    #     pred = self.model.predict(np.vstack([1, 1]).T)
    #     self.assertEqual(pred, 1, msg="predict() returned %s" % pred)


    def test_fit(self):
        # API compliance
        self.assertIsInstance(self.model.fit(self.X2, self.y2), 
                              FeedForwardNeuralNetwork, 
                              msg="The fit function should end with 'return self'")
         # Specifics
        self.model.theta_ = np.zeros(self.model.theta_.shape)
        self.model.fit(self.X1, self.y1)
        self.assertEqual(self.model.predict(np.vstack([1, 1]).T), 0,
                            msg="Test prediction seems off. Does the model train?")
        self.assertEqual(self.model.predict(np.vstack([-1, -1]).T), 1,
                            msg="Test prediction seems off. Does the model train?")

    def test_score(self):
        # API compliance
        score = self.model.score(self.X1, self.y1)
        self.assertIsInstance(score, float, msg="The score should be a single value")
        # Specifics
        self.assertTrue(score>=0, msg="Accuracy is a value between 0 and 1")
        self.assertTrue(score<=1, msg="Accuracy is a value between 0 and 1")


if __name__ == '__main__':
    unittest.main(verbosity=2)
