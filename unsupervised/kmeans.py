from utils.base_estimator import BaseEstimator

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(2046)

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

class KMeans(BaseEstimator):
    """Partition a dataset into K clusters.
    Finds clusters by repeatedly assigning each data point to the cluster with
    the nearest centroid and iterating until the assignments converge (meaning
    they don't change during an iteration) or the maximum number of iterations
    is reached.
    Init centroids by randomly select k values from the dataset
    For better method to improve convergence rates and avoid degenerate cases.
    See: Arthur, D. and Vassilvitskii, S.
         "k-means++: the advantages of careful seeding". ACM-SIAM symposium
         on Discrete algorithms. 2007
    Parameters
    ----------
    K : int, default 8
        The number of clusters into which the dataset is partitioned.
    max_iters: int, default 300
        The maximum iterations of assigning points to the nearest cluster.
        Short-circuited by the assignments converging on their own.
    """

    def __init__(self, K=8, max_iters=300):
        self.K = K
        self.max_iters = max_iters
        # an array of cluster that each data point belongs to
        self.labels = []
        # an array of center value of cluster
        self.centroids = []

    def _init_cetroids(self):
        """Set the initial centroids."""
        indices = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = self.X[indices]

    def _dist_from_centers(self):
        return np.array([min([euclidean_distance(x, c) for c in self.centroids]) for x in self.X])

    def fit(self, X=None):
        """Perform the clustering on the given dataset."""
        self._setup_input(X, y_required=False)

        self._init_cetroids()

        for i in range(self.max_iters):
            new_centroids = []
            # update clusters base on new centroids
            new_labels = np.apply_along_axis(self._closest_cluster, 1, self.X)
            # update centroids base on new clusters
            for k in xrange(self.K):
                centroid = np.mean(self.X[new_labels == k], axis=0)
                new_centroids.append(centroid)

            if self._is_converged(self.centroids, new_centroids):
                print 'Converged on iteration %s' % (i + 1)
                break

            # not converged yet, update centroids / labels to new centroids / labels
            self.labels = new_labels
            self.centroids = new_centroids

    def _predict(self, X=None):
        return np.apply_along_axis(self._closest_cluster, 1, X)

    def _closest_cluster(self, data_point):
        """ Return the closest cluster index and distance given data point"""

        closest_index = 0
        closest_distance = float("inf")

        for cluster_i, centroid in enumerate(self.centroids):
            distance = euclidean_distance(data_point, centroid)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = cluster_i

        return closest_index

    def _is_converged(self, centroids, new_centroids):
        return True if sum([euclidean_distance(centroids[i], new_centroids[i]) for i in range(self.K)]) == 0 else False

    def plot(self, data=None):
        if data is None:
            data = self.X

        for k in xrange(self.K):
            points = data[self.labels == k].T
            plt.scatter(*points, c=sns.color_palette("hls", self.K + 1)[k])

        for point in self.centroids:
            plt.scatter(*point, marker='x', linewidths=10)
