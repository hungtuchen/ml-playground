from base_regression import BasicRegression
from utils.metrics import mean_squared_error

import numpy as np

class LinearRegressionGD(BasicRegression):
    """Linear regression with gradient descent optimizer."""

    def _loss(self, theta):
        X, y = self.X, self.y
        loss = (1.0 / 2) * mean_squared_error(y, X.dot(theta))
        grad = (1.0 / self.n_samples) * (X.T.dot(X.dot(theta) - y))
        return loss, grad

class LinearRegression(BasicRegression):
    """ Linear regression with closed form solution."""

    def _train(self):
        X, y = self.X, self.y
        pseudo_inverse = np.matrix(X.T.dot(X)).I.dot(X.T)
        theta = pseudo_inverse.dot(y)

        return np.squeeze(np.asarray(theta)), []
