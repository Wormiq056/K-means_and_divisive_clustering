import random as rd
from typing import List

from helpers.consts import RANDOM_SEED, COORDINATES_START, COORDINATES_END, OFFSET_START, OFFSET_END, NUM_OF_POINTS


class Generator:
    """
    class that generates random unique points
    """
    coordinates_created = {}
    created_clusters = []
    cluster_counter = 0

    def __init__(self) -> None:
        rd.seed(RANDOM_SEED)

    def _generate_first_20(self) -> None:
        """
        method that generated first random 20 points
        """
        while len(self.created_clusters) != 20:
            x = rd.randint(COORDINATES_START, COORDINATES_END)
            y = rd.randint(COORDINATES_START, COORDINATES_END)
            if not self.coordinates_created.get(tuple([x, y])):
                self.coordinates_created[tuple([x, y])] = True
                self.created_clusters.append([x, y])

    def _generate_rest(self) -> None:
        """
        method that creates rest of the unique points based on offset from randomly selected point
        """

        while len(self.created_clusters) != NUM_OF_POINTS:
            random_cluster = rd.choice(self.created_clusters)
            offset_x = rd.randint(OFFSET_START, OFFSET_END)
            offset_y = rd.randint(OFFSET_START, OFFSET_END)
            new_x = random_cluster[0] + offset_x
            new_y = random_cluster[1] + offset_y

            if new_x > COORDINATES_END or new_x < COORDINATES_START:
                continue
            elif new_y > COORDINATES_END or new_y < COORDINATES_START:
                continue
            elif self.coordinates_created.get(tuple([new_x, new_y])):
                continue
            self.created_clusters.append([new_x, new_y])
            self.coordinates_created[tuple([new_x, new_y])] = True

    def generate_points(self) -> List[List[int]]:
        """
        method that starts generating random points
        :return: randomly generated unique points
        """
        self._generate_first_20()
        self._generate_rest()
        return self.created_clusters
