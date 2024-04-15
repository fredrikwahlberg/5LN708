import unittest
from exercise2 import NaiveKMeans
import numpy as np

class TestNaiveKMeans(unittest.TestCase):

    def setUp(self):
        self.kmeans = NaiveKMeans(n_clusters=3, n_max_iterations=1)
        self.data = np.random.normal(0, .01, size=(30, 2))
        self.data[:10, :] += 1
        self.data[:, 10:20] += 1
                        
    def test_methods(self):
        self.assertTrue(hasattr(self.kmeans, 'fit'), "fit(X) method missing")        
        self.assertTrue(hasattr(self.kmeans, '_E'), "_E(X) method missing")        
        self.assertTrue(hasattr(self.kmeans, '_M'), "_M(X) method missing")        
        self.assertTrue(hasattr(self.kmeans, 'predict'), "predict(X) method missing")        
        
    def test_instantiation(self):
        self.assertTrue(hasattr(self.kmeans, 'n_clusters'), "Number of clusters should be stored here")
        self.assertIsInstance(self.kmeans.n_clusters, int, "n_clusters should be a natural number")
        self.assertTrue(hasattr(self.kmeans, 'n_max_iterations'), "The maximum number of iterations should be stored here")
        self.assertIsInstance(self.kmeans.n_max_iterations, int, "n_max_iterations should be a natural number")

    def test_fit(self):
        self.kmeans.fit(self.data)
        self.assertTrue(hasattr(self.kmeans, 'attributions_'), "The cluster memberships should be stored here")
        self.assertTrue(hasattr(self.kmeans, 'centers_'), "The cluster centers should be stored here")        


class TestNaiveKMeansTraining(unittest.TestCase):

    def setUp(self):
        self.kmeans = NaiveKMeans(n_clusters=3)
        self.data = np.random.normal(0, .01, size=(30, 2))
        self.data[10:20, 0] += 1
        self.data[20:30, 1] += 1
        # plt.scatter(data[:, 0], data[:, 1])
        
    def test_e_step(self):
        self.kmeans.centers_ = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=float)
        self.kmeans._E(self.data)
        self.assertEqual(self.kmeans.attributions_[0], 0, "")
        self.assertEqual(self.kmeans.attributions_[10], 1, "")
        self.assertEqual(self.kmeans.attributions_[20], 2, "")

    def test_m_step(self):
        self.kmeans.centers_ = np.zeros((3, 2), dtype=float)
        self.kmeans.attributions_ = np.zeros(self.data.shape[0])
        self.kmeans.attributions_[10:20] = 1
        self.kmeans.attributions_[20:30] = 2
        self.kmeans._M(self.data)
        # print(self.kmeans.centers_)
        self.assertTrue(np.all(np.isclose(self.kmeans.centers_[0, :], 0, atol=.3)), "")
        self.assertTrue(np.isclose(self.kmeans.centers_[1, 0], 1, atol=.3), "")
        self.assertTrue(np.isclose(self.kmeans.centers_[2, 1], 1, atol=.3), "")

    def test_fit(self):
        # self.kmeans.centers_ = np.asarray([[-1, -1], [2, 0], [0, 2]], dtype=float)
        self.kmeans.fit(self.data)
        c = self.kmeans.centers_.copy()
        print(c)
        c = c[np.argsort(c[:, 0].ravel()), :]
        print(c)
        self.assertTrue(np.isclose(c[0, 0], 0, atol=.3), "")
        self.assertTrue(np.isclose(c[1, 0], 0, atol=.3), "")
        # self.assertTrue(np.isclose(c[2, 0], 1, atol=.3), "")
        
    def test_predict(self):
        self.kmeans.centers_ = np.asarray([[0, 0], [1, 0], [0, 1]])
        X = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=float)
        X += np.random.normal(0, .01, size=X.shape)
        self.assertTrue(np.all(np.isclose(self.kmeans.predict(X), np.asarray([0, 1, 2]))), "Predictions are wrong")


if __name__ == '__main__':
    unittest.main(verbosity=2)
