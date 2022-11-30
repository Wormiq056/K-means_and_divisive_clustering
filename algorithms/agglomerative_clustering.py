from typing import List
from helpers.measurements import distance, get_dist_calculator
import matplotlib.pyplot as plot


class AgglomerativeClustering:
    heap: List

    def __init__(self, clusters: List[List[int]], k_wanted_clusters: int, center_calculator: str) -> None:
        self.clusters = clusters
        self.k = k_wanted_clusters
        self.distance_calculator = get_dist_calculator(center_calculator)

    def _find_closest_clusters(self):
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

    def _create_distance_heap(self):
        distance_list = []

        for i in range(len(self.clusters)):
            cluster_list = []
            for j in range(len(self.clusters)):
                if i == j:
                    cluster_list.append(float("inf"))
                cluster_list.append(distance(self.clusters[i], self.clusters[j]))

        self.heap = distance_list

    def _remove_and_recalculate_points_from_heap(self, a, b):
        if a < b:
            del self.heap[b]
            del self.heap[a]
        else:
            del self.heap[a]
            del self.heap[b]

    def _process_new_cluster(self, new_cluster):
        new_distances = []
        for cluster in self.clusters:
            new_distances.append(distance(new_cluster.center, cluster.center))
        self.heap.insert(0, new_distances)
        self.clusters.insert(0, new_cluster)

    def _merge_closest_clusters(self, index):
        cluster1 = self.clusters[index[0]]
        cluster2 = self.clusters[index[0] + index[1]]
        
        new_cluster = Cluster(cluster1.points + cluster2.points, self.distance_calculator)
        self._process_new_cluster(new_cluster)

    def run(self):
        self._create_distance_heap()
        while len(self.clusters) != self.k:
            index = self._find_closest_clusters()
            self._merge_closest_clusters(index)
        for cluster in self.clusters:
            plot.scatter([x[0] for x in cluster.points], [x[1] for x in cluster.points])
        plot.show()