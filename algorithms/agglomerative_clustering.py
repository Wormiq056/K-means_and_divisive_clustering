import timeit
from typing import List

import matplotlib.pyplot as plot

from helpers.consts import COLORS
from helpers.measurements import distance, get_dist_calculator


class AgglomerativeClustering:
    heap: List
    start_time: float
    stop_time: float

    def __init__(self, clusters: List[List[int]], k_wanted_clusters: int, center_calculator: str) -> None:
        self.start_time = timeit.default_timer()
        self.clusters = self._prepare_clusters(clusters)
        self.k = k_wanted_clusters
        self.center_calculator = get_dist_calculator(center_calculator)
        self.cluster_id = len(self.clusters)
        self.cluster_centers_by_index = {}

    @staticmethod
    def _prepare_clusters(created_clusters: List[List[int]]) -> List[List[List[int]]]:
        """
        method that prepares clusters to be processed, meaning every created points will now be a standalone cluster
        :param created_clusters: generated points
        :return: points as a single cluster
        """
        init_clusters = []
        for cluster in created_clusters:
            init_clusters.append([cluster])
        return init_clusters

    def _find_closest_clusters(self) -> tuple[int, int]:
        """
        method that returns the index of min value from heap of distances
        :return: index of min value
        """
        smallest_distance = float("inf")
        row = None
        column = None

        for row_index, distance_list in enumerate(self.heap):
            column_index = distance_list.index(min(distance_list))
            if self.heap[row_index][column_index] < smallest_distance:
                smallest_distance = self.heap[row_index][column_index]
                row = row_index
                column = column_index
        return row, column

    def _create_distance_heap(self) -> None:
        """
        method that creates distance heap from init clusters, distance heap stores all distances from cluster centers to
        all other cluster centers
        """
        distance_list = []

        for i in range(len(self.clusters)):
            self.cluster_centers_by_index[i] = self.clusters[i][0]
            cluster_list = []
            for j in range(len(self.clusters)):
                if i == j:
                    cluster_list.append(float("inf"))
                    continue
                cluster_list.append(distance(self.clusters[i][0], self.clusters[j][0]))
            distance_list.append(cluster_list)
        self.heap = distance_list

    def _remove_points_from_heap_and_dict(self, a, b) -> None:
        """
        method that removes given indexes from distance heap and dictionary used to store cluster centers by heap index
        :param a: index to remove
        :param b: index to remove
        """
        if a < b:
            del self.heap[b]
            del self.heap[a]
            del self.clusters[b]
            del self.clusters[a]
            for i in range(len(self.clusters)):
                del self.heap[i][b]
                del self.heap[i][a]
        else:
            del self.heap[a]
            del self.heap[b]
            del self.clusters[a]
            del self.clusters[b]
            for i in range(len(self.clusters)):
                del self.heap[i][a]
                del self.heap[i][b]
        self.cluster_centers_by_index.pop(a)
        self.cluster_centers_by_index.pop(b)

    def _merge_closest_clusters(self, index: tuple[int, int]) -> None:
        """
        method that merges clusters by index and handles heap logic
        :param index: 2 indexes of closest clusters
        """
        cluster1 = self.clusters[index[0]]
        cluster2 = self.clusters[index[1]]
        new_cluster = []
        for cluster in cluster1:
            new_cluster.append(cluster)
        for cluster in cluster2:
            new_cluster.append(cluster)
        center_point = self.center_calculator(new_cluster)
        self._remove_points_from_heap_and_dict(index[0], index[1])
        self._recalculate_distances_in_heap(center_point)
        self.cluster_centers_by_index[len(self.clusters)] = center_point
        self.clusters.append(new_cluster)

    def _recalculate_distances_in_heap(self, new_center: List[int, int]) -> None:
        """
        method that recalculates distances from newly created cluster center and adds them to distance heap
        :param new_center: newly created cluster center
        """
        add_to_heap = []
        i = 0
        new_center_dict = {}
        for center in self.cluster_centers_by_index.values():
            center_distances = distance(center, new_center)
            self.heap[i].append(center_distances)
            add_to_heap.append(center_distances)
            new_center_dict[i] = center
            i += 1
        add_to_heap.append(float("inf"))
        self.heap.append(add_to_heap)
        self.cluster_centers_by_index = new_center_dict
        self.cluster_centers_by_index[len(self.clusters)] = new_center

    def _statistics(self) -> None:
        """
        method that outputs statistics for generated outcome, how much time it took to generate clusters and
        Also outputs selected clusters success rate. Success rate is calculated by % and if cluster
        has an average distance from middle under 500 points it's classified as a successful cluster
        :return:
        """
        good_clusters = 0
        for cluster in self.clusters:
            center = self.center_calculator(cluster)
            sum_of_distances = 0
            for points in cluster:
                sum_of_distances += distance(points, center)
            if sum_of_distances / len(cluster) <= 500:
                good_clusters += 1
        cluster_success_rate = good_clusters / len(self.clusters) * 100
        print(f"Time to calculate clusters : {self.stop_time - self.start_time} seconds")
        print(f"Cluster success rate {cluster_success_rate} %")

    def run(self) -> None:
        """
        main method that runs agglomerative algorithm and outputs graphic plot for generated clusters
        """
        self._create_distance_heap()
        while len(self.clusters) != self.k:
            index = self._find_closest_clusters()
            self._merge_closest_clusters(index)
            print(len(self.heap))

        self.stop_time = timeit.default_timer()
        self._statistics()

        color_counter = 0
        for values in self.clusters:
            if color_counter == len(COLORS):
                color_counter = 0
            plot.scatter([x[0] for x in values], [x[1] for x in values], color=COLORS[color_counter])
            color_counter += 1
        plot.show()
