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
        n = len(x1)
        inside = 0
        for i in range(n):
            inside += ((x1[i]-x2[i])**2)
        distance = math.sqrt(inside)
        return distance

    def compute_distance_matrix(self, X, rep):
        r = len(X)
        c = len(rep)
        S = np.empty((r, c))
        for i in range(r):
            for j in range(c):
                S[i][j] = self.euclidean_distance(X[i], rep[j])
        return S

    def assign_label(self, S):
        r = len(S)
        res = np.empty(r)
        for i in range(r):
            res[i] = int(np.argmin(S[i]))
        return res.astype(int)

    def update_centroids(self, X, labels, old_centroids):
        new_centroid = np.zeros(old_centroids.shape)
        c_count = len(self.centroids)
        X_count = len(X)
        colony = []
        for i in range(c_count):
            for j in range(X_count):
                if labels[j] == i:
                    colony.append(X[j])
            colony = np.array(colony)
            colony = np.transpose(colony)
            colony_r = colony.shape[0]
            for k in range(colony_r):
                new_centroid[i][k] = np.mean(colony[k])
            colony = []
        return np.round(new_centroid, 3)

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        old_centroids = None
        # the algorithm stops when the centroid cluster is 
        # the same as the centroid cluster in the previous iteration
        while(not np.array_equal(old_centroids, self.centroids)):
            dist_mtx = self.compute_distance_matrix(X, self.centroids)
            self.labels = self.assign_label(dist_mtx)
            temp_centroids = self.centroids
            self.centroids = self.update_centroids(X, self.labels, old_centroids)
            old_centroids = temp_centroids  
        return self

    def predict(self, X):
        labels = []
        dist_mtx = self.compute_distance_matrix(X, self.centroids)
        labels = self.assign_label(dist_mtx)
        print(dist_mtx)
        return labels