from base_regression import BasicRegression

import numpy as np

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

class LogisticRegression(BasicRegression):
    """Binary logistic regression with gradient descent optimizer."""
    def _loss(self, w):
        X, y, C = self.X, self.y, self.C

        predict = sigmoid(X.dot(w))
        # prevent overflow or underflow
        predict = np.clip(predict, 1e-15, 1 - 1e-15)
        loss = - np.mean(np.sum(y * np.log(predict) + (1 - y) * np.log(1 - predict)))
        grad = (1.0 / self.n_samples) * (X.T.dot(predict - y))

        # Add regularization to the loss.
        loss += 0.5 * C * w.T.dot(w)
        grad += C * w

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

    def _predict(self, X=None, threshold=0.5):
        """Predict class labels for samples in X."""
        X = self._add_intercept(X)
        predict = sigmoid(X.dot(self.theta))
        return np.where(predict > threshold , 1, 0)
