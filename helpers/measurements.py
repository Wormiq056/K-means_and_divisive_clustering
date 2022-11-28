import math
from typing import List

import numpy as np


def distance(a, b):
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def centroid_calculation(cluster: List[List[int]]) -> tuple[int, int]:
    """
    function to calculate centroid location in cluster
    :param cluster: cluster you want to calculate for
    :return: centroid location
    """
    x = 0
    y = 0
    for point in cluster:
        x += point[0]
        y += point[1]
    return int(x / len(cluster)), int(y / len(cluster))


def medoid_calculation(cluster: List[List[int]]) -> tuple[int, int]:
    centroid = np.mean(cluster)
    medoid = cluster[np.argmin([sum((x - centroid) ** 2) for x in cluster])]
    return medoid[0], medoid[1]


def get_dist_calculator(algorithm: str):
    if algorithm == "c" or algorithm == "d":
        return centroid_calculation
    else:
        return medoid_calculation
