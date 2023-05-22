import numpy as np # Imports for the class here

class CleverDecisionTree:
    def __init__(self):
        """A simple decision tree with manually tuned decision boundaries. 
        Note that this classifier will only work for the given data set."""
        raise NotImplementedError

    def predict(self, X):
        """Predicts labels/classes for the inputs in X. X should have one data vector per row.
        Returns a flat vector with label/class (natural) numbers."""
        raise NotImplementedError

    def score(self, X, y): 
        """Returns the accuracy of the predictions made by the model, given inputs (X) and outputs (y).
        Returns a float between 0 and 1."""
        raise NotImplementedError

if __name__ == '__main__':
    from sklearn.datasets import make_blobs # Imports only used in the test code here
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=.7, n_features=2, random_state=0)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.axis('square')
    plt.show()
