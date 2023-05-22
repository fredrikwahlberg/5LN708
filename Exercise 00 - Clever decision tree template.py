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
    # Imports only used in the test code here
    import matplotlib.pyplot as plt

    X = np.tile([[0.9615202, 4.43376825], [1.99127366, 0.80487522], [-1.6656056, 2.88982984]], (70, 1))
    np.random.seed(0)
    X += np.random.normal(0, .7, size=X.shape)
    y = np.tile([0, 1, 2], 70)

    # clf = CleverDecisionTree()
    # print("Accuracy: %.1f%%" % (100*clf.score(X, y)))
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    # plt.plot([1, 1], [-1, 6])
    # plt.plot([-3, 3], [1, 1])
    # plt.scatter(X[:, 0], X[:, 1], c=clf.predict(X), cmap='coolwarm')    
    plt.axis('square')
    plt.show()
    