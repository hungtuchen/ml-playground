from base_regression import BasicRegression

import numpy as np

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def softmax(z):
    # Avoid numerical overflow by removing max
    # See: http://cs231n.github.io/linear-classify/
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

class LogisticRegression(BasicRegression):
    """Binary logistic regression with gradient descent optimizer."""
    def _loss(self, theta):
        X, y, C = self.X, self.y, self.C

        predict = sigmoid(X.dot(theta))
        # prevent overflow or underflow
        predict = np.clip(predict, 1e-15, 1 - 1e-15)
        loss = - np.mean(np.sum(y * np.log(predict) + (1 - y) * np.log(1 - predict)))
        grad = (1.0 / self.n_samples) * (X.T.dot(predict - y))

        # Add regularization to the loss.
        loss += 0.5 * C * np.sum(theta * theta)
        grad += C * theta

        return loss, grad

    def _predict(self, X=None, threshold=0.5):
        """Predict class labels for samples in X."""
        X = self._add_intercept(X)
        predict = sigmoid(X.dot(self.theta))
        return np.where(predict > threshold , 1, 0)

class Softmax(BasicRegression):
    """Multi-class logistic regression with gradient descent optimizer."""
    def init_weights(self):
        """Mutli-class weights"""
        self.num_classes = np.max(self.y) + 1 # assume y takes values 0...K-1 where K is number of classes
        return np.random.normal(size=(self.n_features + 1, self.num_classes), scale=0.5)

    def _loss(self, theta):
        """
        Naive implementation for softmax loss due to its simplicity and readibility
        For vectorized version: https://github.com/transedward/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py#L62
        """
        X, y, C, n_samples, n_features = self.X, self.y, self.C, self.n_samples, self.n_features
        loss = 0.0
        grad = np.zeros_like(theta)
        for i in range(n_samples):
            scores = X[i].dot(theta)
            probability = softmax(scores)
            loss += -np.log(probability[y[i]])
            # gradient for softmax loss
            # http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
            for j in range(self.num_classes):
                grad[:, j] += (probability[j] - (j == y[i])) * X[i, :]


        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by n_samples.
        loss /= n_samples
        grad /= n_samples

        # Add regularization to the loss.
        loss += 0.5 * C * np.sum(theta * theta)
        grad += C * theta

        return loss, grad

    def _predict(self, X=None):
        """Predict class labels for samples in X."""
        X = self._add_intercept(X)
        scores = X.dot(self.theta)
        predict = np.argmax(scores, axis=1)
        return predict
