import numpy as np

class KMeans:
    def __init__(self, K=2, max_iters=100):
        self.K = K
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        random_idx = np.random.choice(n_samples, self.K, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            clusters = self._create_clusters(X)
            new_centroids = self._calculate_centroids(X, clusters)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.K)]
        for idx, point in enumerate(X):
            closest_centroid = np.argmin(np.linalg.norm(point - self.centroids, axis=1))
            clusters[closest_centroid].append(idx)
        return clusters

    def _calculate_centroids(self, X, clusters):
        centroids = np.zeros((self.K, X.shape[1]))
        for cluster_idx, cluster in enumerate(clusters):
            if cluster:
                cluster_points = X[cluster]
                centroids[cluster_idx] = np.mean(cluster_points, axis=0)
        return centroids

    def predict(self, X):
        cluster_labels = [np.argmin(np.linalg.norm(x - self.centroids, axis=1)) for x in X]
        return cluster_labels

X = np.array([[1, 2], [2, 3], [3, 4], [8, 9], [9, 10]])
kmeans = KMeans(K=2)
kmeans.fit(X)
predictions = kmeans.predict(X)
print("Cluster assignments:", predictions)
