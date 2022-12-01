from typing import List
from helpers.measurements import distance, get_dist_calculator
import matplotlib.pyplot as plot


class AgglomerativeClustering:
    heap: List

    def __init__(self, clusters: List[List[int]], k_wanted_clusters: int, center_calculator: str) -> None:
        self.clusters = self._prepare_clusters(clusters)
        self.k = k_wanted_clusters
        self.center_calculator = get_dist_calculator(center_calculator)
        self.cluster_id = len(self.clusters)
        self.cluster_centers_by_index = {}

    @staticmethod
    def _prepare_clusters(created_clusters):
        init_clusters = []
        for cluster in created_clusters:
            init_clusters.append([cluster])
        return init_clusters


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
            self.cluster_centers_by_index[i] = self.clusters[i][0]
            cluster_list = []
            for j in range(len(self.clusters)):
                if i == j:
                    cluster_list.append(float("inf"))
                cluster_list.append(distance(self.clusters[i][0], self.clusters[j][0]))
            distance_list.append(cluster_list)
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
        self.cluster_centers_by_index.pop(a)
        self.cluster_centers_by_index.pop(b)

    def _process_new_cluster(self, new_cluster):
        new_distances = []
        for cluster in self.clusters:
            new_distances.append(distance(new_cluster.center, cluster.center))
        self.heap.insert(0, new_distances)
        self.clusters.insert(0, new_cluster)

    def _merge_closest_clusters(self, index):
        cluster1 = self.clusters[index[0]]
        cluster2 = self.clusters[index[1]]
        new_cluster = []
        for cluster in cluster1:
            new_cluster.append(cluster)
        for cluster in cluster2:
            new_cluster.append(cluster)
        center_point = self.center_calculator(new_cluster)
        self._remove_points_from_heap_and_dict(index[0],index[1])
        self._recalculate_distances_in_heap(center_point)
        self.cluster_centers_by_index[len(self.clusters)] = center_point
        self.clusters.append(new_cluster)

    def _recalculate_distances_in_heap(self, new_center):
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
        self.cluster_centers_by_index[len(self.clusters)] = new_center
        self.cluster_centers_by_index = new_center_dict





    def run(self):
        self._create_distance_heap()
        print('test')
        while len(self.clusters) != self.k:
            index = self._find_closest_clusters()
            self._merge_closest_clusters(index)
            print(len(self.heap))
        print('test')

        # for cluster in self.clusters:
        #     plot.scatter([x[0] for x in cluster.points], [x[1] for x in cluster.points])
        # plot.show()
