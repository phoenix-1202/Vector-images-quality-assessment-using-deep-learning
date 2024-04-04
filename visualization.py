from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def visualize(labels, matrix_similar, path):
    plt.clf()
    plt.scatter(matrix_similar[:, 0], matrix_similar[:, 1], c=labels, cmap='viridis')
    plt.title('Distribution')
    plt.savefig(path)


def pca_visualization(X, y, path):
    plt.clf()
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(14, 12))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
                edgecolor='none', alpha=0.7, s=40,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.colorbar()
    plt.savefig(path)


def visualize_distribution(list_width, list_height, path):
    plt.scatter(list_width, list_height)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('График распределения высоты и ширины изображения')
    plt.savefig(path)


def get_elbow_method(distortions, n_clusters, path_result):
    """Метод локтя"""
    plt.figure(figsize=(16, 8))
    plt.plot(n_clusters, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Метод локтя для определения оптимального количества кластеров')
    plt.savefig(path_result + 'kmeans/elbow_method.png')


def get_eps(k, matrix, path_result):
    nbrs = NearestNeighbors(n_neighbors=k).fit(matrix)
    distances, indices = nbrs.kneighbors(matrix)
    k_distances = distances[:, -1]
    k_distances.sort()
    plt.plot(list(range(1, len(matrix) + 1)), k_distances)
    plt.xlabel('Values of k')
    plt.ylabel('Distortion')
    plt.savefig(path_result + 'hdbscan/graphic_of_eps.png')
