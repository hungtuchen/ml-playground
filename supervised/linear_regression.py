from utils.base_regression import BasicRegression
from utils.metrics import mean_squared_error

import numpy as np

class LinearRegressionGD(BasicRegression):
    """Linear regression with gradient descent optimizer."""

    def _loss(self, w):
        X, y = self.X, self.y
        loss = (1.0 / 2) * mean_squared_error(y, X.dot(w))
        grad = (1.0 / self.n_samples) * (X.T.dot(X.dot(w) - y))
        return loss, grad

    def _train(self):
        theta = self.theta
        loss_history = []
        for i in xrange(self.max_iters):
            # evaluate loss and gradient
            loss, grad = self._loss(theta)
            loss_history.append(loss)

            # perform parameter update
            # Update the weights using the gradient and the learning rate
            theta -= self.lr * grad

            if self.verbose and (i + 1) % 100 == 0:
                print 'Iteration %s, loss %s' % (i + 1, loss_history[i])

        return theta, loss_history

class LinearRegression(BasicRegression):
    """ Linear regression with closed form solution."""

    def _train(self):
        X, y = self.X, self.y
        pseudo_inverse = np.matrix(X.T.dot(X)).I.dot(X.T)
        theta = pseudo_inverse.dot(y)

        return np.squeeze(np.asarray(theta)), []
