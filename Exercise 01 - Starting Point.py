import numpy as np

class SimpleLinearRegression: 
    def __init__(self):
        """Linear regression using a first order polynomial model for 2D data."""
        # TODO Your code here ... (and remember to remove the exception)
        raise NotImplementedError

    def predict(self, X):
        """Predicts outputs y from some input X. For a first order polynomial, 
        this is the line \hat y_i = \theta_0 + \theta_1 \cdot x_i . Note that 
        the shape of the input X is (n, 1), and the shape of your prediction 
        should be (n, ).""" 
        # TODO Your code here ... (and remember to remove the exception)
        raise NotImplementedError

    def fit(self, X, y):
        """This method finds the slope and intercept for your data."""
        # Leave this as is. You will reimplement this in the next exercise.
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2, "X should be a 2D array in sklearn"
        assert isinstance(y, np.ndarray)
        assert y.ndim == 1
        slope, intercept = np.polyfit(X.ravel(), y, 1)
        self._theta[0] = intercept
        self._theta[1] = slope

    def score(self, X, y): 
        """Returns the sum of squares residual for the given inputs and outputs."""
        # TODO Your code here ... (and remember to remove the exception)
        raise NotImplementedError

if __name__ == '__main__':
   
    # Example data
    X = np.vstack(np.linspace(0, 10, num=10))
    y = X.ravel()*2 + 3 + np.random.normal(0, 1, size=len(X))

    # Plot the data
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(X.ravel(), y, 'o', label="Data")
    model = SimpleLinearRegression()
    plt.plot(X.ravel(), model.predict(X), '-', label="Untrained model")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(X.ravel(), y, 'o', label="Data")
    model.fit(X, y)
    plt.plot(X.ravel(), model.predict(X), '-', label="Trained model")
    plt.legend()
    plt.show()
