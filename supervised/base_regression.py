from utils.base_estimator import BaseEstimator

import numpy as np

class BasicRegression(BaseEstimator):
    def __init__(self, lr=0.001, max_iters=1000, C=0, verbose=False):
        """Basic class for implementing continuous regression estimators which
        are trained with gradient descent optimization on their particular loss function.
        Parameters
        ----------
        lr : float, default 0.001
            Learning rate.
        max_iters : int, default 1000
            The maximum number of iterations.
        C : float, default 0 (no regularization)
            The l2-regularization coefficient.
            Since l1-norm is not differentiable, We don't support l1 here for simplicity.
            If you want to implement it, you can check https://github.com/HIPS/autograd
            or there are mamy algorithms out there.
        verbose: boolean, default False
            If True, print progress during optimization.
        """
        self.lr = lr
        self.max_iters = max_iters
        self.C = C
        self.verbose = verbose
        self.loss_history = []
        self.theta = []
        self.n_samples, self.n_features = None, None

    def _loss(self, w):
        raise NotImplementedError()

    def _train(self):
        # default: gradient descent optimization
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

    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def init_weights(self):
        return np.random.normal(size=(self.n_features + 1), scale=0.5)

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.n_samples, self.n_features = X.shape

        # Initialize weights + bias term
        self.theta = self.init_weights()

        # Add an intercept column
        self.X = self._add_intercept(self.X)

        self.theta, self.loss_history = self._train()

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return X.dot(self.theta)
