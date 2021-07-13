# Importing the library

import numpy as np

class Kmeans:
    def __init__(self,n_clusters,max_iters=100):
        '''
        Input:
        n_clusters -> Number of clusters
        max_iters -> Maximum number of iterations
        '''
        self.n_cluster = n_clusters
        self.max_iters = max_iters

    def initialise_centroids(self,data):
        '''
        Initializes the centroids randomly.
        Input:
        data -> data on which clustering is to be done
        Output:
        centroid-> Randomly initialized centroids 
        '''
        random_idx = np.random.permutation(data.shape[0])
        centroids = data[random_idx[:self.n_cluster]]
        return centroids

    def compute_centroids(self,data,labels):
        '''
        Computes the mean of the cluster to find new centroids.
        Input:
        data-> data on which clustering is to be performed.
        labels-> clusters to which each data point belongs to.
        '''
        centroids = np.zeros((self.n_cluster,data.shape[1]))
        for k in range(self.n_cluster):
            centroids[k,:] = np.mean(data[k == labels, : ],axis=0)
        return centroids

    def calculate_distance(self,data,centroids):
        '''
        Calculates the distance between each datapoint and centroid and stores it in distance matrix.
        Input:
        data -> data on which clustering is to be done.
        centroids -> centroids for the clusters.
        Output:
        distance -> an array(data.shape[0],n_clusters) with each data point and their respective distance from each centroid.
        '''
        
        distance = np.zeros((data.shape[0],self.n_cluster))
        for k in range(self.n_cluster):
            row_norm = np.linalg.norm(data - centroids[k,:],axis=1)
            distance[:,k] = np.square(row_norm)
        
        return distance
    def find_closest_cluster(self,distance):
        '''
        Finds closest cluster according to minimum distance from centroid
        Input: 
        distance-> an array with distacnes of data points with centroids.
        Output:
        Returns the index of minnimum distance
        '''
        return np.argmin(distance,axis=1)

    def calculate_inertia(self,data,labels,centroids):
        '''
        Caluclates inertia or sum of sqaured distance which we will use in Elbow method to find optimum number of cluster.
        Input:
        data -> data on which clustering is to be performed.
        labels -> cluster to which each data point belongs to.
        centroids -> centroids of each cluster.
        '''
        distance = np.zeros(data.shape[0])
        for k in range(self.n_cluster):
            distance[labels==k] = np.linalg.norm(data[labels==k] - centroids[k],axis=1)
        return np.sum(np.square(distance))
    

    def fit(self,data):
        '''
        Fit the data on the algorithm.
        Input:
        data-> data on which clustering is to be done.
        '''
        self.centroids = self.initialise_centroids(data)
        for i in range(self.max_iters):
            previous_centroids = self.centroids
            distance = self.calculate_distance(data,previous_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(data,self.labels)
            if np.all(self.centroids == previous_centroids):
                break   

        self.inertia_ = self.calculate_inertia(data,self.labels,self.centroids)

    def predict(self,data):
        '''
        To predict for an unknown data point to which cluster it belongs to.
        Input:
        data-> datapoint to cluster
        Output:
        cluster number it belongs to.
        '''
        distance = self.calculate_distance(data,self.centroids)
        return self.find_closest_cluster(distance)