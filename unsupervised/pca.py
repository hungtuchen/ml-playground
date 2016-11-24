from utils.base_estimator import BaseEstimator

import numpy as np

np.random.seed(2046)


class PCA(BaseEstimator):
    """
    In general, PCA aims to find the directions of maximum variance
    in high-dimensional data and projects it onto a new subspace
    with equal or fewer dimensions that the original one.
    Also, PCA directions are highly sensitive to data scaling, and we need to
    standardize the features prior to PCA if the features were measured on
    different scales and we want to assign equal importance to all features
    """

    def __init__(self, n_components, solver='svd'):
        """Principal Component Analysis (PCA) implementation.
        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components.
        The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.
        Parameters
        ----------
        n_components : int
        solver : str, default 'svd'
            {'svd', 'eigen'}
        """
        self.solver = solver
        self.n_components = n_components
        self.components = None
        self.variance_ratio = None
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        # standardize input then decompose
        X_std = (X - self.mean) / self.std
        self._decompose(X_std)

        return self

    def _decompose(self, X):
        if self.solver == 'svd':
            _, s, V = np.linalg.svd(X, full_matrices=True)
            s = (s ** 2) / len(X)
            V = V.T
        elif self.solver == 'eigen':
            X_cov = X.T.dot(X) / (len(X) - 1)
            s, V = np.linalg.eig(X_cov)

        # sort eigenvalues (and eigenvectors accordingly) descending
        sorted_idx = np.argsort(s)[::-1]
        s = s[sorted_idx]
        V = V[:, sorted_idx]

        variance_ratio = s / s.sum()
        self.variance_ratio = variance_ratio
        print 'Explained variance ratio: %s' % variance_ratio[0:self.n_components]

        self.components = V[:, 0:self.n_components]

    def transform(self, X):
        X_std = (X - self.mean) / self.std
        return np.dot(X_std, self.components)

    def _predict(self, X):
        return self.transform(X)
