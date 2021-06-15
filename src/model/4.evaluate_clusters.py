
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pickle
import time
import argparse
import json
from item.clustering.evaluate import (
    get_score_pickle,
    evaluate_results_pickle,
)
from item.clustering.utils import (
    load_clustering_results_hive_table
)
from item.item_representation import (
    load_items_embeddings
)


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metric", required=True,
                        help="metric for cluster analysis results.")
    parser.add_argument("-d", "--distance", required=True,
                        help="distance.")
    parser.add_argument("-r", "--results", default=True,
                        help="results files directory.")
    parser.add_argument("-e", "--embeddings", default=True,
                        help="embeddings file.")
    parser.add_argument("-s", "--sample", type=str, default=0.2,
                        help="embeddings file.")
    p.add_argument("-p", "--password", default="", help="connection password.")
    p.add_argument("-i", "--hive", default=False, help="load table from hive and.")

    return parser.parse_args()


def main():

    args = parse_args()

    metric = args.metric
    distance = args.distance
    results_dir = args.results
    embeddings_file = args.embeddings
    sample_size = None if args.sample == 'None' else float(args.sample)

    if args.hive:
        results, outliers = load_clustering_results_hive_table('f03_grupos_sem_outliers',
                                                                'f03_grupos_outliers',
                                                                args.password)
    else:
        results_train, outliers_train = load_clustering_results_pickle(results_dir)
    embeddings = load_items_embeddings(embeddings_file)

    if metric == 'silhouette':
        score = get_score_pickle(results, embeddings, score='silhouette',
                                 metric=distance, sample_size=sample_size)
    elif metric == 'davies':
        score = get_score_pickle(results, embeddings, score='davies')
    elif metric == 'calinski':
        score = get_score_pickle(results, embeddings, score='calinski')
    elif metric == 'intra-cluster-dist':
        intracluster_distance = evaluate_results_pickle(results, embeddings,
                                                n_threads=20, metric=distance)

        # write to json file
        with open(results_dir + "intra-cluster-dist.json", "w") as JFile:
            json.dump(intracluster_distance, JFile)

        distances = []
        for group, distance in intracluster_distance.items():
            distances.append(distance['mean'])
        score = np.mean(distances)

    print(score)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
