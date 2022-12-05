import random as rd
import timeit
from typing import List

import matplotlib.pyplot as plot

from helpers.consts import RANDOM_SEED, DIVISIVE_ITERATIONS
from helpers.measurements import get_dist_calculator, distance


class DivisiveClustering:
    """
    class that implements divisive reverse k-means algorithm. It follows the top down approach and every iteration it
    splits current clusters into two by implementing k-means logic. Meaning recursively we call method that selects 2
    random points from cluster and assigns closest points to them. We do this until we reach out wanted number of k
    clusters
    """
    final_clusters = []
    start_time: float
    stop_time: float

    def __init__(self, created_points: List[List[int]], num_of_clusters: int, center_calculation: str) -> None:
        """
        init method that also gets center_calculation even though i set divisive to only have centroid calculation
        """
        self.start_time = timeit.default_timer()
        self.clusters = [created_points]
        self.center_char = center_calculation
        self.k = num_of_clusters
        self.center_calculation = get_dist_calculator(center_calculation)
        rd.seed(RANDOM_SEED)  # setting seed for random for reproducibility
        self.current_num_of_clusters = 1
        self.final_cluster_success_rate = 0

    @staticmethod
    def _choose_2_points_as_clusters(cluster: List[List[int]]) -> List[List[int]] or None:
        """
        static method that returns 2 randomly selected points from given cluster
        :param cluster: from which we want to select random points
        :return: selected points
        """
        return rd.sample(cluster, 2)

    @staticmethod
    def _assign_points_to_clusters(points: List[List[int]], cluster: List[List[int]]) -> dict:
        """
        static method that assigned points from clusters to the closest center point that was selected
        :param points: selected center points of clusters
        :param cluster: from which we want to assign points to
        :return: dict with assigned points
        """

        cluster_dict = {tuple(point): [] for point in points}
        for value in cluster:
            distance_a = distance(points[0], value)
            distance_b = distance(points[1], value)
            if distance_a <= distance_b:
                cluster_dict[tuple(points[0])].append(value)
            else:
                cluster_dict[tuple(points[1])].append(value)
        return cluster_dict

    def _calculate_new_center_points(self, clusters: dict) -> List[List[int]]:
        """
        method that calculates new center points from given clusters
        :param clusters: for which we want to calculate new center points
        :return: recalculated center points
        """
        center_points = []
        for values in clusters.values():
            center_points.append(self.center_calculation(values))
        return center_points

    @staticmethod
    def _reassign_points_to_center(cluster: List[List[int]], centers: List[List[int]]) -> dict:
        """
        method that reassigns points from already assigned clusters to new recalculated centers
        :param cluster: with assigned points
        :param centers: newly calculated centers
        :return: dict with reassigned points
        """
        center_dict = {center: [] for center in centers}
        for point in cluster:
            distances = [distance(point, center) for center in centers]
            min_index = distances.index(min(distances))
            center_dict[tuple(centers[min_index])].append(point)
        return center_dict

    def _top_down_k_means(self, cluster: List[List[int]]) -> List[List[int]] or None:
        """
        method that implements top down k-means meaning it selects 2 random points from given cluster and continues
        k- means algorithm like when we want to have 2 final clusters
        :param cluster: cluster we want to split
        :return: list of 2 new clusters
        """
        created_center_points = {}
        if self.current_num_of_clusters == self.k:
            return

        chosen_points = self._choose_2_points_as_clusters(cluster)
        assigned_points = self._assign_points_to_clusters(chosen_points, cluster)
        while True:
            recalculated_center_points = self._calculate_new_center_points(assigned_points)
            assigned_points = self._reassign_points_to_center(cluster, recalculated_center_points)
            if created_center_points.get(tuple(recalculated_center_points)):
                break
            created_center_points[tuple(recalculated_center_points)] = True

        self.current_num_of_clusters += 1
        return assigned_points.values()

    def _select_best_variance(self) -> List[List[List[int]]]:
        """
        method that selects best variance out of generated clusters, we repeat the whole k-means process n times (in my
        case 5 times) and then based on variance we select the best one. Best variance means that generated k clusters
        have the most evenly split number of points in all clusters
        :return: k clusters with the best variance
        """
        variances = []
        total_length = len(self.clusters[0])
        for final_cluster in self.final_clusters:
            variance = 1
            for values in final_cluster:
                variance = variance * (len(values) / total_length)
            variances.append(variance)
        max_variance_index = variances.index(max(variances))
        return self.final_clusters[max_variance_index]

    def _calculate_success_rate(self, best_variance: List[List[List[int]]]) -> None:
        """
        method that calculates final cluster success rate for output, if average distance from cluster center is greater
        than 5 that cluster is marked as unsuccessful
        :param best_variance: final cluster
        """
        good_clusters = 0
        for cluster in best_variance:
            center = self.center_calculation(cluster)
            sum_of_distances = 0
            for points in cluster:
                sum_of_distances += distance(points, center)
            if sum_of_distances / len(cluster) <= 500:
                good_clusters += 1
        self.final_cluster_success_rate = good_clusters / len(best_variance) * 100

    def _console_print(self) -> None:
        """
        method that prints info to console
        """
        print(f"Divisive clustering with {self.k} clusters")
        if self.center_char == "m":
            print(f"Center calculation: medoid")
        else:
            print(f"Center calculation: centroid")
        print(f"Time to calculate clusters : {self.stop_time - self.start_time} seconds")
        print(f"Cluster success rate {self.final_cluster_success_rate} %")

    def _finish_iteration(self, old_clusters: List[List[int]], newly_created_clusters: List[List[int]], index: int) \
            -> None:
        """
        method that is called at the end of one divisive k-means iteration, it creates one final cluster for old
        clusters that have not been split yet and new clusters that have been already split
        :param old_clusters: one loop before splitting
        :param newly_created_clusters: clusters that have been already split
        :param index: at what index in old clusters did algorithm stop
        """
        final_clusters = []
        for i in range(index, len(old_clusters), 1):
            final_clusters.append(old_clusters[i])
        for cluster in newly_created_clusters:
            final_clusters.append(cluster)
        self.final_clusters.append(final_clusters)
        self.current_num_of_clusters = self.k + 1

    def run(self) -> None:
        """
        main method that implements divisive reverse k-means algorithm
        it repeats cluster splitting until we have created wanted k clusters
        at the end of method it also generates graphic plot for created clusters
        """
        for i in range(DIVISIVE_ITERATIONS):
            self.current_num_of_clusters = 1
            current_clusters = self.clusters
            while self.current_num_of_clusters <= self.k:
                new_clusters = []
                counter = 0
                for cluster in current_clusters:
                    created_clusters = self._top_down_k_means(cluster)
                    if not created_clusters:
                        self._finish_iteration(current_clusters, new_clusters, counter)
                        break
                    for created_cluster in created_clusters:
                        new_clusters.append(created_cluster)
                    counter += 1
                current_clusters = new_clusters

        best_variance = self._select_best_variance()
        self._calculate_success_rate(best_variance)
        self.stop_time = timeit.default_timer()
        self._console_print()

        for values in best_variance:
            plot.scatter([x[0] for x in values], [x[1] for x in values])
        plot.show()
