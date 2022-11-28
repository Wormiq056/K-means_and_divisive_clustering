import random as rd
from typing import List

from matplotlib import pyplot as plot

from helpers.consts import RANDOM_SEED, K_MEANS_ITERATIONS
from helpers.measurements import distance, get_dist_calculator


class KMeans:
    final_clusters = []

    def __init__(self, clusters: List[List[int]], num_of_clusters: int, cluster_center: str):
        self.clusters = clusters
        self.cluster_center = cluster_center
        self.k = num_of_clusters
        rd.seed(RANDOM_SEED)
        self.center_calculator = get_dist_calculator(cluster_center)

    def _assign_points_to_init_clusters(self, init_clusters):
        init_dict = {tuple(cluster): [] for cluster in init_clusters}
        copied_clusters = self.clusters.copy()

        for cluster in copied_clusters:
            distances = [distance(cluster, init_cluster) for init_cluster in init_clusters]
            min_index = distances.index(min(distances))
            init_dict[tuple(init_clusters[min_index])].append(cluster)

        return init_dict

    def _choose_init_clusters(self):
        return rd.sample(self.clusters, self.k)

    def _calculate_center_points(self, clusters: dict):
        center_points = []
        for values in clusters.values():
            center_points.append(self.center_calculator(values))
        return center_points

    def _assign_points_to_recalculated_centers(self, centers):
        center_dict = {center: [] for center in centers}
        for cluster in self.clusters:
            distances = [distance(cluster, center) for center in centers]
            min_index = distances.index(min(distances))
            center_dict[tuple(centers[min_index])].append(cluster)
        return center_dict

    def _select_best_variance(self):
        variances = []
        for final_cluster in self.final_clusters:
            variance = 0
            keys = [key for key in final_cluster.keys()]
            i = 0
            for values in final_cluster.values():
                distance_from_middle = 0
                for value in values:
                    distance_from_middle += distance(value, keys[i])
                variance += distance_from_middle / len(values)
                i += 1
            variances.append(variance)
        min_variance_index = variances.index(min(variances))
        return self.final_clusters[min_variance_index]

    def run(self):
        for i in range(K_MEANS_ITERATIONS):
            init_clusters = self._choose_init_clusters()
            assigned_points = self._assign_points_to_init_clusters(init_clusters)
            calculated_center_points = self._calculate_center_points(assigned_points)
            recalculated_clusters = self._assign_points_to_recalculated_centers(calculated_center_points)
            self.final_clusters.append(recalculated_clusters)

        # counter = 1
        # for variance in self.final_clusters:
        #     for values in variance.values():
        #         plot.scatter([x[0] for x in values],[x[1] for x in values])
        #     plot.text(0,0, counter)
        #     plot.show()
        #     counter += 1
        best_variance = self._select_best_variance()
        for values in best_variance.values():
            plot.scatter([x[0] for x in values], [x[1] for x in values])
        plot.show()
