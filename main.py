import argparse

from helpers.generator import Generator
from algorithms.k_means import KMeans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algorithm", required=True,
                    help="which algorithm to use (c = centroid, medoid = m, divizive =d)")
    ap.add_argument("-k", "--clusters", required=True, help="number of clusters (num)")
    args = vars(ap.parse_args())

    created_points = Generator().generate_points()
    if args['algorithm'] == 'c' or args['algorithm'] == 'm':
        KMeans(created_points, int(args["clusters"]), args["algorithm"]).run()


if __name__ == '__main__':
    main()
