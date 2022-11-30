import timeit
import random as rd
from typing import List

from matplotlib import pyplot as plot

from helpers.consts import RANDOM_SEED, K_MEANS_ITERATIONS, COLORS
from helpers.measurements import distance, get_dist_calculator


class KMeans:
    """
    class that implements k-means clustering algorithm
    """
    final_clusters = []
    start_time: float
    stop_time: float

    def __init__(self, clusters: List[List[int]], num_of_clusters: int, cluster_center: str) -> None:
        """
        in init we also get center_calculation which returns either medoid or centroid calculation based on our argument
        """
        self.start_time = timeit.default_timer()
        self.clusters = clusters
        self.cluster_center = cluster_center
        self.k = num_of_clusters
        rd.seed(RANDOM_SEED)  # setting random seed fo reproducibility
        self.center_calculator = get_dist_calculator(cluster_center)

    def _assign_points_to_init_clusters(self, init_clusters: List[List[int]]) -> dict:
        """
        this method assign points to the closet init cluster which is selected randomly
        :param init_clusters: clusters that act as base points
        :return: dictionary that contains assigned points to selected init clusters
        """
        init_dict = {tuple(cluster): [] for cluster in init_clusters}

        for cluster in self.clusters:
            distances = [distance(cluster, init_cluster) for init_cluster in init_clusters]
            min_index = distances.index(min(distances))
            init_dict[tuple(init_clusters[min_index])].append(cluster)

        return init_dict

    def _choose_init_clusters(self) -> List[List[int]]:
        """
        method that returns k randomly selected clusters based on how many clusters we want to end up with
        :return: k randomly selected clusters
        """
        return rd.sample(self.clusters, self.k)

    def _calculate_center_points(self, clusters: dict) -> List[List[int]]:
        """
        method that calculates center points for k clusters that have generated in the first part of k-means algorithm
        center calculation can be centroid or medoid based on initial program argument
        :param clusters: for which we want to calculate new center points
        :return: created center points
        """
        center_points = []
        for values in clusters.values():
            center_points.append(self.center_calculator(values))
        return center_points

    def _assign_points_to_recalculated_centers(self, centers: List[List[int]]) -> dict:
        """
        2nd part of k-means algorithm when we want to assign points again based on recalculated init points
        :param centers: new calculated centers
        :return: reassigned points to new clusters
        """
        center_dict = {center: [] for center in centers}
        for cluster in self.clusters:
            distances = [distance(cluster, center) for center in centers]
            min_index = distances.index(min(distances))
            center_dict[tuple(centers[min_index])].append(cluster)
        return center_dict

    def _select_best_variance(self) -> dict:
        """
        method that selects best variance out of generated clusters, we repeat the whole k-means process n times (in my
        case 5 times) and then based on variance we select the best one. Best variance means that generated k clusters
        have the most evenly split number of points in all clusters
        :return: best variance of k-means algorithm
        """
        variances = []
        total_length = len(self.clusters)
        for final_cluster in self.final_clusters:
            variance = 1
            for values in final_cluster.values():
                variance = variance * (len(values) / total_length)
            variances.append(variance)
        max_variance_index = variances.index(max(variances))
        return self.final_clusters[max_variance_index]

    def _statistics(self, best_variance: dict) -> None:
        """
        method that outputs statistics for generated outcome, how much time it took to generate n k-means algorithms and
        select the best one. Also outputs selected variance success rate. Success rate is calculated by % and if cluster
        has an average distance from middle under 500 points it's classified as a successful cluster
        :param best_variance:
        :return:
        """
        good_clusters = 0
        for cluster in best_variance.values():
            center = self.center_calculator(cluster)
            sum_of_distances = 0
            for points in cluster:
                sum_of_distances += distance(points, center)
            if sum_of_distances / len(cluster) <= 500:
                good_clusters += 1
        cluster_success_rate = good_clusters / len(best_variance.values()) * 100
        print(f"Time to calculate clusters : {self.stop_time - self.start_time} seconds")
        print(f"Cluster success rate {cluster_success_rate} %")

    def run(self) -> None:
        """
        main method of k-means algorithm, it contains the basic logic of k-means pseudo code and at the end it also
        generates graphic plot for best selected k-means iteration
        """
        for i in range(K_MEANS_ITERATIONS):
            init_clusters = self._choose_init_clusters()
            assigned_points = self._assign_points_to_init_clusters(init_clusters)
            calculated_center_points = self._calculate_center_points(assigned_points)
            recalculated_clusters = self._assign_points_to_recalculated_centers(calculated_center_points)
            self.final_clusters.append(recalculated_clusters)

        best_variance = self._select_best_variance()
        self.stop_time = timeit.default_timer()

        self._statistics(best_variance)

        color_counter = 0
        for values in best_variance.values():
            if color_counter == len(COLORS):
                color_counter = 0
            plot.scatter([x[0] for x in values], [x[1] for x in values], color=COLORS[color_counter])
            color_counter += 1
        plot.show()
