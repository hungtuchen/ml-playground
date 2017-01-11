from utils.base_estimator import BaseEstimator

import numpy as np

def softmax(z):
    # Avoid numerical overflow by removing max
    # See: http://cs231n.github.io/linear-classify/
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

class GaussianNB(BaseEstimator):
    """Gaussian Naive Bayes."""
    # Binary problem.
    n_classes = 2

    def fit(self, X, y=None):
        self._setup_input(X, y)
        # Check target labels
        assert list(np.unique(y)) == [0, 1]

        # Prepare mean, std, prior for each class
        self._mean = np.zeros((self.n_classes, self.n_features))
        self._var = np.zeros((self.n_classes, self.n_features))
        self._prior = np.zeros((self.n_classes))

        for c in range(self.n_classes):
            # Filter X only in current class
            X_per_class = X[y == c]

            self._mean[c, :] = np.mean(X_per_class, axis=0)
            self._var[c, :] = np.var(X_per_class, axis=0)
            self._prior[c] = len(X_per_class) / float(len(X))

    def _predict(self, X=None):
        """
        Return the class that each row in X most likely belongs to.
        Perform classification on an array of test vectors X.
        """
        return np.argmax(self._joint_log_likelihood(X), axis=1)

    def predict_proba(self, X):
        """Predict probability for each class for each row."""
        return softmax(self._joint_log_likelihood(X))

    def _joint_log_likelihood(self, X):
        """
        The conditional probabilities for each class given an feature value are small.
        When they are multiplied together they result in very small values,
        which can lead to floating point underflow.
        A common fix for this is to apply log function on joint probabilities.
        """
        joint_log_likelihood = []
        for c in range(self.n_classes):
            prior = np.log(self._prior[c])
            # log of denominator part of gaussian distribution
            posterior = - 0.5 * np.sum(np.log(2. * np.pi * self._var[c, :]))
            # log of numerator part of gaussian distribution
            posterior -= 0.5 * np.sum(((X - self._mean[c, :]) ** 2) /
                                 (self._var[c, :]), axis=1)
            joint_log_likelihood.append(prior + posterior)
        return np.array(joint_log_likelihood).T
