import argparse

from algorithms.divisive_clustering import DivisiveClustering
from algorithms.k_means import KMeans
from helpers.consts import NUM_OF_POINTS
from helpers.generator import Generator


def main() -> None:
    """
    main function that checks if arguments are valid and runs the program
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--algorithm", required=True,
                    help="which algorithm to use (centroid k-means = c, medoid k-means = m, divisive = d)")
    ap.add_argument("-k", "--clusters", required=True, help="number of clusters (0 <x> points generated")
    args = vars(ap.parse_args())
    if args["algorithm"] not in ['c', 'm', 'd']:
        print("Select valid argument -a (centroid k-means = c, medoid k-means = m, divisive = d)")
        return
    try:
        int(args['clusters'])
    except ValueError:
        print("Argument -k must be a number")
        return
    if int(args['clusters']) < 1 or int(args['clusters']) > NUM_OF_POINTS:
        print(f"Argument -k must be 0<k>{NUM_OF_POINTS}")
        return

    created_points = Generator().generate_points()
    if args['algorithm'] == 'c' or args['algorithm'] == 'm':
        KMeans(created_points, int(args["clusters"]), args["algorithm"]).run()
    else:
        DivisiveClustering(created_points, int(args["clusters"]), args["algorithm"]).run()


if __name__ == '__main__':
    main()
