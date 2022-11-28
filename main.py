import argparse

from algorithms.divisive_clustering import DivisiveClustering
from algorithms.k_means import KMeans
from helpers.generator import Generator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algorithm", required=True,
                    help="which algorithm to use (centroid k-means = c, medoid k-means = m, divisive = d)")
    ap.add_argument("-k", "--clusters", required=True, help="number of clusters (0 <x> points generated")
    args = vars(ap.parse_args())

    created_points = Generator().generate_points()
    if args['algorithm'] == 'c' or args['algorithm'] == 'm':
        KMeans(created_points, int(args["clusters"]), args["algorithm"]).run()
    else:
        DivisiveClustering(created_points, int(args["clusters"]), args["algorithm"]).run()


if __name__ == '__main__':
    main()
