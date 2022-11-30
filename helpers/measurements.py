import math
from typing import List

import numpy as np


def distance(a: List[int], b: List[int]) -> int:
    """
    helper function that calculates euclidean distance for 2 given points
    :param a: first point coordinates
    :param b: second point coordinates
    :return: distance between points
    """
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def centroid_calculation(cluster: List[List[int]]) -> tuple[int, int]:
    """
    function to calculate centroid location in cluster
    :param cluster: cluster you want to calculate for
    :return: calculated centroid
    """
    x = 0
    y = 0
    for point in cluster:
        x += point[0]
        y += point[1]
    return int(x / len(cluster)), int(y / len(cluster))


def medoid_calculation(cluster: List[List[int]]) -> tuple[int, int]:
    """
    helper function that calculates medoid coordinates from given cluster and returns its location
    :param cluster: we want to calculate medoid for
    :return: calculated medoid position
    """
    centroid = np.mean(cluster)
    medoid = cluster[np.argmin([sum((x - centroid) ** 2) for x in cluster])]
    return medoid[0], medoid[1]


def get_dist_calculator(algorithm: str) -> centroid_calculation or medoid_calculation:
    """
    function that return method from this file based on which center calculation we want to use in algorithms
    :param algorithm: m = medoid, c,d = centroid (d -means divisive algorithm)
    :return: function for center calculation
    """
    if algorithm == "c" or algorithm == "d" or algorithm == 'a':
        return centroid_calculation
    else:
        return medoid_calculation
