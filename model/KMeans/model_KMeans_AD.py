import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class KMeans_AD:
    def __init__(self, args):
        super(KMeans_AD, self).__init__()
        self.kmeans = KMeans(args.clusters_per_label * args.labels_num)
        
        self.train_src = []
        self.train_distances = None
        self.train_labels = None
    
    def add_train_data(self, src):
        self.train_src.extend(src)

    def clear_train_data(self):
        self.train_src = []
    
    def train(self):
        src = np.array(self.train_src)
        self.kmeans.fit(src)
        labels, distances = pairwise_distances_argmin_min(src, self.kmeans.cluster_centers_, axis=1)
        self.train_distances = distances
        self.train_labels = labels
        return distances
    
    def test(self, src):
        src = np.array(src)
        labels, distances = pairwise_distances_argmin_min(src, self.kmeans.cluster_centers_, axis=1)
        return distances