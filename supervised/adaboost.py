from utils.base_estimator import BaseEstimator

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost(BaseEstimator):
    """
    An AdaBoost classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    Parameters
    ----------
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)

    def fit(self, X, y=None):
        """Build a boosted classifier from the training set (X, y)."""
        self._setup_input(X, y)
        # Initialize weights to 1 / n_samples
        sample_weight = np.zeros(self.n_samples)
        sample_weight[:] = 1.0 / self.n_samples

        for i in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight = self._boost(sample_weight)
            self.estimator_weights_[i] = estimator_weight

    def _boost(self, sample_weight):
        """Implement a single boost.
        sample_weight : array-like of shape = [n_samples]
            The current sample weights.
        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
        estimator_weight : float
            The weight for the current boost.
        """
        X, y = self.X, self.y
        estimator = DecisionTreeClassifier(max_depth=1)
        # TODO: replace with custom decision tree
        estimator.fit(X, y, sample_weight=sample_weight)
        self.estimators_.append(estimator)

        y_predict = estimator.predict(X)
        # Instances incorrectly and correctly classified
        incorrect = y != y_predict
        correct = y == y_predict

        # Error fraction
        estimator_error = np.average(incorrect, weights=sample_weight)

        scaling_factor = np.sqrt((1 - estimator_error) / estimator_error)
        # Boosting by weighting up incorrect samples and weighting down correct ones
        sample_weight[incorrect] *= scaling_factor
        sample_weight[correct] /= scaling_factor

        # Normalize sample_weight
        sample_weight /= sample_weight.sum()

        estimator_weight = np.log(scaling_factor)

        return sample_weight, estimator_weight

    def _predict(self, X):
        score = self.decision_function(X)
        # expect y to be 1 or 0
        return np.where(score >= 0.5, 1.0, 0.0)

    def decision_function(self, X):
        """Compute the decision function of ``X``."""
        pred = np.sum([estimator.predict(X) * w for estimator, w in
                       zip(self.estimators_, self.estimator_weights_)], axis=0)
        pred /= self.estimator_weights_.sum()
        return pred
