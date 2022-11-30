from typing import List
from helpers.measurements import distance, get_dist_calculator
import matplotlib.pyplot as plot


class AgglomerativeClustering:
    heap: List

    def __init__(self, clusters: List[List[int]], k_wanted_clusters: int, center_calculator: str) -> None:
        self.clusters = clusters
        self.k = k_wanted_clusters
        self.center_calculator = get_dist_calculator(center_calculator)
        self.cluster_id = len(self.clusters)
        self.cluster_centers_by_id = {}

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
            self.cluster_centers_by_id[i] = self.clusters[i]
            cluster_list = []
            for j in range(len(self.clusters)):
                if i == j:
                    cluster_list.append(float("inf"))
                cluster_list.append(distance(self.clusters[i], self.clusters[j]))

        self.heap = distance_list

    def _remove_points_from_heap_and_dict(self, a, b):
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
        self.cluster_centers_by_id.pop(a)
        self.cluster_centers_by_id.pop(b)
        self.cluster_id -= 1

    def _process_new_cluster(self, new_cluster):
        new_distances = []
        for cluster in self.clusters:
            new_distances.append(distance(new_cluster.center, cluster.center))
        self.heap.insert(0, new_distances)
        self.clusters.insert(0, new_cluster)

    def _merge_closest_clusters(self, index):
        cluster1 = self.clusters[index[0]]
        cluster2 = self.clusters[index[0] + index[1]]

        new_cluster = cluster1 +cluster2
        center_point = self.center_calculator(new_cluster)
        self.cluster_centers_by_id[self.cluster_id] = new_cluster
        self._recalculate_distances_in_heap(center_point)

    def _recalculate_distances_in_heap(self,new_center):
        for i in range(len(self.clusters)):
            cluster_center = self.cluster_centers_by_id.get(i)
            center_distances = distance(cluster_center,new_center)
            self.heap[i].append(center_distances)
            

    def run(self):
        self._create_distance_heap()
        while len(self.clusters) != self.k:
            index = self._find_closest_clusters()
            self._merge_closest_clusters(index)
        for cluster in self.clusters:
            plot.scatter([x[0] for x in cluster.points], [x[1] for x in cluster.points])
        plot.show()