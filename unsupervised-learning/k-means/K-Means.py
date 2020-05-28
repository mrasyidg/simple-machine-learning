from numpy import np


class Kmeans:
    def __init__(self, n_clusters, seed=None):
        self.n_clusters = n_clusters
        self.seed = seed

    def initialize_centroids(self, X):
        if(self.seed):
            np.random.seed(self.seed)

        random_idx = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        centroids = X[random_idx, :]
        self.initial_centroids = centroids 
        return centroids

    def euclidean_distance(self, x1, x2):
        distance = 0

        # YOUR CODE HERE
        raise NotImplementedError()

        # return distance

    def compute_distance_matrix(self, X, rep):
        r = len(X)
        c = len(rep)
        S = np.empty((r, c))
        
        raise NotImplementedError()

    def assign_label(self, S):

        raise NotImplementedError()

    def update_centroids(self, X, labels, old_centroids):
        new_centroid = np.zeros(old_centroids.shape)

        raise NotImplementedError()

        # return np.round(new_centroid, 3)

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        old_centroids = None

        # the algorithm stops when the centroid cluster is 
        # the same as the centroid cluster in the previous iteration
        while(not np.array_equal(old_centroids, self.centroids)):
            
            raise NotImplementedError()

    def predict(self, X):
        labels = []

        raise NotImplementedError()

        # return labels