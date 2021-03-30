# File name: K_means.py

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class KMeans(object):
    """ K-Means Clustering: unsupervised machine learning model - clustering algorithms"""

    """
        The k-means algorithm searches for a pre-determined number of clusters within an unlabeled 
        multidimensional dataset. It accomplishes this using a simple conception of what the optimal 
        clustering looks like:
            1. The "cluster center" is the arithmetic mean of all the points belonging to the cluster.
            2. Each point is closer to its own cluster center than to other cluster centers.
        Those two assumptions are the basis of the k-means model.
        
        In other words, algorithm works by a two-step process called expectation-maximization:
            - The expectation step assigns each data point to its nearest centroid. 
            - Then, the maximization step computes the mean of all the points for 
                each cluster and sets the new centroid. 
    """

    def __init__(self, k=5, max_iter=100, plot_flag=False):
        # Specify a k of clusters to assign
        self.K = k
        self.max_iter = max_iter
        self.plot_flag = plot_flag

        # A List to store each pair of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # The centers (arithmetic mean) for each cluster
        self.centroids = []
        # Group of clusters and group of centroids for plotting purpose
        self.clusters_group = []
        self.centroids_group = []

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        # Randomly initialize k centroids - Guess some cluster centers
        self.centroids = self.initialize_centroids(X)
        # Append the centroids to the group list
        self.centroids_group.append(self.centroids)

        # Repeat until converged
        for i in range(self.max_iter):
            # Expectation step -------------------
            # Create clusters - assign points to the nearest closest centroids
            self.clusters = self.create_clusters(X, self.centroids)
            # Append the new clusters to the group list
            self.clusters_group.append(self.clusters)

            # Keep the the centroids as a reference
            old_centroids = self.centroids

            # Maximization step -------------------
            # Calculate new centroids (arithmetic mean) from the clusters
            self.centroids = self.update_centroids(X, self.clusters)
            # Append the centroids to the group list
            self.centroids_group.append(self.centroids)

            # Check for convergence
            # Check if centroids positions have not changed, if yes then break
            if np.all(old_centroids == self.centroids):
                break

        # Plot k-means if plot flag is True
        if self.plot_flag:
            self.plot(X)

        # Classify samples as the index of their clusters
        labels = self.get_labels(self.clusters)

        return labels

    def initialize_centroids(self, X):
        # Initialize the centers randomly
        random_indices = np.random.choice(self.n_samples, self.K, replace=False)
        centroids = [X[idx] for idx in random_indices]
        return centroids

    def create_clusters(self, X, centroids):
        # Create clusters from the closest centroid
        _clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(X):
            centroids_index = self.closest_centroids(sample, centroids)
            _clusters[centroids_index].append(idx)
        return _clusters

    def closest_centroids(self, sample, centroids):
        # Distance of the current sample to each centroids
        euclidean_distance = [np.sqrt(np.sum((sample - point) ** 2)) for point in centroids]
        # Choose the min distance from the center
        closest_index = np.argmin(euclidean_distance)
        return closest_index

    def update_centroids(self, X, clusters):
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def get_labels(self, clusters):
        # Each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = idx
        return labels

    def predict(self, X):
        clusters_ = self.create_clusters(X, self.centroids)
        labels = self.get_labels(clusters_)
        return labels

    def plot(self, X):
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(8, 6))

        # plt.get_current_fig_manager().window.wm_geometry("+<x-pos>+<y-pos>")
        plt.get_current_fig_manager().window.wm_geometry("-1000-300")

        colors = ['#FFBF00', '#eb5408', '#427630', '#127adf', '#9FE2BF', '#FF7F50', '#FFBF00', '#6495ED']
        markers = ['d', 'o', 'H', 'D', '.', '*', '^', '>']

        for ind, (cluster, centroid) in enumerate(zip(self.clusters_group, self.centroids_group)):
            plt.cla()
            # print(cluster)
            for s, index in enumerate(cluster):
                point = X[index].T
                ax.scatter(*point
                           , marker=markers[s]
                           , color=colors[s]
                           , edgecolors='#9FE2BF'
                           , s=100)

            plt.cm.get_cmap('gray')
            # print(centroid)
            for point in centroid:
                ax.scatter(*point
                           , marker='+'
                           , color='black'
                           , s=100
                           , linewidth=3,
                           label='Center')

            left, right = plt.xlim()
            bottom, top = plt.ylim()

            plt.text(x=right - 0.1
                     , y=top - 0.1
                     , s=f'Iteration: {str(ind)}'
                     , fontsize=12
                     , fontstyle='italic'
                     , fontweight='demibold'
                     , alpha=0.8
                     , ha='right'
                     , va='top'
                     , bbox=dict(boxstyle="round"
                                 , alpha=0.7
                                 , ec=(1., 0.5, 0.5)
                                 , fc=(1., 0.8, 0.8))
                     , color='navy')

            plt.xlabel('X1', color='darkred'
                       , fontweight='bold')
            plt.ylabel('X2', color='darkred'
                       , fontweight='bold')
            plt.title('Visualization of clustered data'
                      , color='navy'
                      , fontweight='bold')
            plt.tight_layout()

            plt.pause(0.5)

        plt.show()


def feature_scaling(x):
    from sklearn.preprocessing import StandardScaler
    """ Scaling or standardizing our training and test data """
    """
        -- Data standardization is the process of rescaling the attributes so that they have 
            mean as 0 and variance as 1.
        -- The ultimate goal to perform standardization is to bring down all the features to 
            a common scale without distorting the differences in the range of the values.
        -- In sklearn.preprocessing.StandardScaler(), centering and scaling happens independently 
            on each feature.
    """
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    """
        The "fit method" is calculating the mean and variance of each of the features present in our data. 
        The "transform" method is transforming all the features using the respective mean and variance.
    """
    scaled_x = scaler.fit_transform(x)

    return scaled_x


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    np.random.seed(50)
    features, target = make_blobs(centers=4
                                  , n_samples=200
                                  , n_features=2
                                  #, cluster_std=1.03
                                  , shuffle=True
                                  , random_state=123)
    # features scaling
    scaled_feature = feature_scaling(features)
    # Set k
    n_clusters = len(np.unique(target))
    # Create K-means object
    k_means = KMeans(k=n_clusters
                     , max_iter=100
                     , plot_flag=True)
    # Fit
    predictions_fit = k_means.fit(scaled_feature)
    # Predict
    predictions_pre = k_means.predict(scaled_feature)

    from sklearn.metrics.cluster import adjusted_mutual_info_score \
        , completeness_score, adjusted_rand_score, calinski_harabasz_score \
        , davies_bouldin_score, contingency_matrix, silhouette_score

    print('adjusted_mutual_info_score:', adjusted_mutual_info_score(target, predictions_pre))
    print('completeness_score:', completeness_score(target, predictions_pre))
    print('adjusted_rand_score:', adjusted_rand_score(target, predictions_pre))
    print('calinski_harabasz_score:', calinski_harabasz_score(features, target))
    print('davies_bouldin_score:', davies_bouldin_score(features, target))
    print('contingency_matrix:\n', contingency_matrix(target, predictions_pre))
    print('silhouette_score:', silhouette_score(features, target))
