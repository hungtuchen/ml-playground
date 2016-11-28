from base_estimator import BaseEstimator

import numpy as np

class BasicRegression(BaseEstimator):
    def __init__(self, lr=0.001, max_iters=1000, stochastic=False, verbose=False):
        """Basic class for implementing continuous regression estimators which
        are trained with (stochastic) gradient descent optimization
        on their particular loss function.
        Parameters
        ----------
        lr : float, default 0.001
            Learning rate.
        max_iters : int, default 1000
            The maximum number of iterations.
        stochastic: boolean, default False (which uses full batch)
            number of training examples to use at each step.
        verbose: boolean, default False
            If True, print progress during optimization.
        """
        self.lr = lr
        self.max_iters = max_iters
        self.verbose = verbose
        self.loss_history = []
        self.theta = []
        self.n_samples, self.n_features = None, None

    def _loss(self, w):
        raise NotImplementedError()

    def _train(self):
        raise NotImplementedError()

    @staticmethod
    def _add_intercept(X):
        b = np.ones([X.shape[0], 1])
        return np.concatenate([b, X], axis=1)

    def fit(self, X, y=None):
        self._setup_input(X, y)
        self.n_samples, self.n_features = X.shape

        # Initialize weights + bias term
        self.theta = np.random.normal(size=(self.n_features + 1), scale=0.5)

        # Add an intercept column
        self.X = self._add_intercept(self.X)

        self.theta, self.loss_history = self._train()

    def _predict(self, X=None):
        X = self._add_intercept(X)
        return X.dot(self.theta)
