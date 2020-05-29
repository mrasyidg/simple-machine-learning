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

        raise NotImplementedError()

        # return distance

    def compute_distance_matrix(self, X, rep):
        r = len(X)
        c = len(rep)
        S = np.empty((r, c))
        
        raise NotImplementedError()

    def assign_label(self, S):

        raise NotImplementedError()

    def calculate_cost(self, S):

        raise NotImplementedError()

    def update_medoids(self, X, cost):
        best_medoids = self.initial_medoids
        lowest_cost = cost
        while True:
            # iterate until we get the medoid with the lowest cost

            # raise NotImplementedError()
            
            if lowest_cost < cost:
                cost = lowest_cost
                medoids = best_medoids
            else:
                break

        print('Best medoids: ', best_medoids)
        print('Lowest cost: ', lowest_cost)

        return best_medoids, lowest_cost

    def fit(self, X):
        self.initial_medoids = self.initialize_medoids(X)
        
        raise NotImplementedError()
        
        self.labels = assign_label(self.compute_distance_matrix(X, self.best_medoids))

    def predict(self, X):
        labels = []

        raise NotImplementedError()

        # return labels

def assign_label(S):
    
    raise NotImplementedError()