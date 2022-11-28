from typing import List
from helpers.measurements import get_dist_calculator, distance
from helpers.consts import RANDOM_SEED, DIVISIVE_ITERATIONS
import random as rd
import matplotlib.pyplot as plot

class DivisiveClustering:
    final_clusters = []

    def __init__(self, created_points: List[List[int]], num_of_clusters: int, center_calculation: str) -> None:
        self.clusters = [created_points]
        self.k = num_of_clusters
        self.center_calculation = get_dist_calculator(center_calculation)
        rd.seed(RANDOM_SEED)
        self.current_num_of_clusters = 1

    def _choose_2_points_as_clusters(self, cluster: List[List[int]]):
        return rd.sample(cluster, 2)

    def _assign_points_to_clusters(self, points, cluster):
        cluster_dict = {tuple(point): [] for point in points}
        for value in cluster:
            distance_a = distance(points[0], value)
            distance_b = distance(points[1], value)
            if distance_a <= distance_b:
                cluster_dict[tuple(points[0])].append(value)
            else:
                cluster_dict[tuple(points[1])].append(value)
        return cluster_dict

    def _calculate_new_center_points(self, clusters):
        center_points = []
        for values in clusters.values():
            center_points.append(self.center_calculation(values))
        return center_points

    def _reassign_points_to_center(self, clusters, centers):
        center_dict = {center: [] for center in centers}
        for cluster in clusters:
            distances = [distance(cluster, center) for center in centers]
            min_index = distances.index(min(distances))
            center_dict[tuple(centers[min_index])].append(cluster)
        return center_dict

    def _top_down_k_means(self, cluster):
        if self.current_num_of_clusters == self.k:
            return
        chosen_points = self._choose_2_points_as_clusters(cluster)
        assigned_points = self._assign_points_to_clusters(chosen_points, cluster)
        recalculated_center_points = self._calculate_new_center_points(assigned_points)
        reassigned_points = self._reassign_points_to_center(cluster,recalculated_center_points)
        self.current_num_of_clusters += 1
        return reassigned_points.values()

    def _handle_odd_k(self, clusters):
        cluster_lengths = [len(cluster) for cluster in clusters]
        index = cluster_lengths.index(max(cluster_lengths))
        chosen_points = self._choose_2_points_as_clusters(clusters[index])
        assigned_points = self._assign_points_to_clusters(chosen_points, clusters[index])
        recalculated_center_points = self._calculate_new_center_points(assigned_points)
        reassigned_points = self._reassign_points_to_center(clusters[index], recalculated_center_points)
        final_clusters = clusters
        del final_clusters[index]
        for cluster in reassigned_points.values():
            final_clusters.append(cluster)
        self.final_clusters.append(final_clusters)

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
        for i in range(DIVISIVE_ITERATIONS):
            self.current_num_of_clusters = 1
            current_clusters = self.clusters
            while self.current_num_of_clusters <= self.k:
                new_clusters = []
                for cluster in current_clusters:
                    created_clusters = self._top_down_k_means(cluster)
                    if not created_clusters:
                        if len(current_clusters) != self.k:
                            self._handle_odd_k(current_clusters)
                        else:
                            self.final_clusters.append(current_clusters)
                        self.current_num_of_clusters += 1
                        break
                    for created_cluster in created_clusters:
                            new_clusters.append(created_cluster)
                current_clusters = new_clusters

        counter = 1
        for variance in self.final_clusters:
            for values in variance:
                plot.scatter([x[0] for x in values],[x[1] for x in values])
            plot.text(0,0, counter)
            plot.show()
            counter += 1
