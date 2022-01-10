
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import multiprocessing
import collections
from item.item_list import (
    ItemList,
    Item
)
from nlp.grouping import (
    groups_frequency_sort
)
from nlp.utils import (
    read_json_file,
    plot_histogram,
    get_completetext,
    plot_wordcloud,
    print_statistics)
from utils.hive_access import (
    dataframe_to_hive_table,
    hive_table_to_dataframe
)

from nlp.pos_tagging import *
from nlp.word_embeddings import *
from item.clustering.evaluate import *
from item.clustering.utils import *
from item.clustering.item_representation import *
from item.clustering.clustering import *
from item.pricing.utils import *
from item.pricing.pricing import *


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--clusters', required=True,
                   help='folder containing the clutering results.')
    p.add_argument('--outpath', required=True,
                   help='directory where the results will be saved.')
    p.add_argument('--items', type=str, default='f03_items_preprocessed_complete',
                  help='train dataset.')
    p.add_argument("-v", "--version", required=False, help="execution version.")
    p.add_argument("-s", "--sample", default=None, type=float,
                   help="percentage of items to be sampled.")
    p.add_argument("-i", "--hive", default=False, help="load table from hive and \
                    save the results on hive.")

    parsed = p.parse_args()

    return parsed


def main():

    args = parse_args()

    results_train, outliers_train = load_clustering_results_pickle(args.clusters)

    print(time.asctime()," Getting the descriptions processed:")

    # It gets the descriptions processed [TRAINING]:
    itemlist_train = ItemList()

    if args.hive:
        itemlist_train.load_items_from_hive_table(args.items)
    else:
        itemlist_train.load_items_from_file(args.items, sample=args.sample)

    print(time.asctime()," Getting the statistics for each cluster found in the training set:")

    # 1) PRICING: get the statistics for each cluster finded in the training set

    clusters_df = get_clusters_dataframe(results_train, outliers_train, baseline=True)
    cluster_prices = get_clusters_prices(itemlist_train, results_train)

    if args.hive:
        version = args.version
        dataframe_to_hive_table(itemlist_train.items_df, "f03_itens", version)

    print(time.asctime()," Pricing the items of the training set:")
    cluster_prices_statistics, cluster_prices_statistics_year, \
    items_clusters_wo_outliers = pricing(itemlist_train,
                                        results_train,
                                        cluster_prices,
                                        remove_outliers=False,
                                        threshold=0.5,
                                        dsc_unidade=True,
                                        year=True)

    # 3) Save result tables
    if args.hive:
        version = args.version
        dataframe_to_hive_table(clusters_df, "f03_grupos", version)
        dataframe_to_hive_table(cluster_prices_statistics, "f03_banco_precos_grupos",
                                version)
        dataframe_to_hive_table(items_clusters_wo_outliers, "f03_banco_precos_itens",
                                version)
    else:
        clusters_df.to_csv(args.outpath + "clusters.csv.zip", sep=';', index=False,
                           compression='zip')
        items_clusters_wo_outliers.to_csv(args.outpath + "items_clusters_train_wo_out.csv.zip",
                                          sep=';', index=False, compression='zip')
        cluster_prices_statistics.to_csv(args.outpath + "cluster_prices_statistics.csv.zip",
                                         sep=';', index=False, compression='zip')
        cluster_prices_statistics_year.to_csv(args.outpath + "cluster_prices_statistics_year.csv.zip",
                                 sep=';', index=False, compression='zip')


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
