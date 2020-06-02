import numpy as np


class Kmedoids:
    def __init__(self, n_clusters, seed=None):
        self.n_clusters = n_clusters
        self.seed = seed

    def initialize_medoids(self, X):
        if(self.seed):
            np.random.seed(self.seed)

        random_idx = np.random.choice(
            X.shape[0], size=self.n_clusters, replace=False)
        medoids = X[random_idx, :]
        return medoids

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

    def calculate_cost(self, S):

        res = []
        for i in range(len(S)):
            res.append(min(S[i]))
        return np.sum(res)

    def update_medoids(self, X, cost):
        best_medoids = self.initial_medoids
        lowest_cost = cost
        for i in range(100):

            # iterate until we get the medoid with the lowest cost
            random_idx = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
            medoids = X[random_idx, :]

            S = compute_distance_matrix(X, medoids)
            labels = assign_label(S)
            cost = calculate_cost(S)

            if lowest_cost > cost:
                lowest_cost = cost
                best_medoids = medoids
            else:
                continue

        print('Best medoids: ', best_medoids)
        print('Lowest cost: ', lowest_cost)
        return best_medoids, lowest_cost

    def fit(self, X):
        self.initial_medoids = self.initialize_medoids(X)

        dist_mtx = self.compute_distance_matrix(X, self.initial_medoids)
        cost = self.calculate_cost(dist_mtx)
        self.best_medoids, self.lowest_cost = self.update_medoids(X, cost)

        self.labels = assign_label(self.compute_distance_matrix(X, self.best_medoids))
        return self

    def predict(self, X):
        labels = []
        
        dist_mtx = self.compute_distance_matrix(X, self.best_medoids)
        labels = self.assign_label(dist_mtx)
        print(dist_mtx)
        print(labels)
        return labels