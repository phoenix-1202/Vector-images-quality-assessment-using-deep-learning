from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, HDBSCAN
import numpy as np
from tqdm import tqdm


class ModelHDBSCAN(object):
    def __init__(self, matrix_similar):
        self.matrix_similar = matrix_similar
        self.best_model = HDBSCAN()
        self.best_score = -1

    def one_hdbcan(self, min_cluster, min_sample) -> None:
        hdb = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_sample)
        result = hdb.fit_predict(self.matrix_similar)
        try:
            sil_score = silhouette_score(self.matrix_similar, result)
        except ValueError:
            sil_score = 0
        if sil_score > self.best_score:
            self.best_score = sil_score
            self.best_model = hdb

    def hdbscan_result(self) -> None:
        # минимальное количество точек в окрестности для того, чтобы точка считалась основной
        # ака определение плотности
        min_cluster_list = np.arange(start=3, stop=20, step=1)
        # определяет размера кластера
        min_sample_list = np.arange(start=4, stop=25, step=2)

        for min_cluster in tqdm(min_cluster_list, desc="Запускаем hdbscan на разных кластерах"):
            for min_sample in min_sample_list:
                self.one_hdbcan(min_cluster, min_sample)


def run_hdbscan(matrix_similar):
    class_hdbscan = ModelHDBSCAN(matrix_similar)
    class_hdbscan.hdbscan_result()
    return class_hdbscan


class ModelKMEANS(object):
    def __init__(self, matrix_similar):
        self.matrix_similar = matrix_similar
        self.best_model = KMeans()
        self.best_score = -1
        self.count_clusters_list = []
        self.distortions = []

    def one_kmeans(self, count_clusters) -> None:
        kmeans = KMeans(n_clusters=count_clusters, init='k-means++')
        result = kmeans.fit_predict(self.matrix_similar)
        sil_score = silhouette_score(self.matrix_similar, result)
        self.distortions.append(kmeans.inertia_)

        if self.best_score < sil_score:
            self.best_score = sil_score
            self.best_model = kmeans

    def kmeans_result(self) -> None:
        self.count_clusters_list = np.arange(start=5, stop=50, step=5)
        for k in tqdm(self.count_clusters_list, desc="Запускаем kmeans на разных кластерах"):
            self.one_kmeans(k)


def run_kmeans(matrix_similar):
    class_kmeans = ModelKMEANS(matrix_similar)
    class_kmeans.kmeans_result()
    return class_kmeans
