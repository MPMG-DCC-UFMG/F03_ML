import argparse
import time
import pandas as pd
import numpy as np
import json
import collections
import copy
import random
import math
import matplotlib.pyplot as plt
from nlp.preprocessing import preprocess_items
from nlp.utils import (
    read_json_file)
from utils.read_files import (
    get_items
)
from item.item_list import (
    ItemList,
    Item
)
from item.clustering.utils import *
import io

from utils.hive_access import *
from pyhive import hive


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    p.add_argument('--input', type = str, default = 'f03_items_preprocessed_complete'
                  ,help='file containing the items dataset')
    parser.add_argument("-r", "--results", default=True,
                        help="results files directory.")
    parser.add_argument("-v", "--version", required=True,
                        help="execution version.")
    parser.add_argument("-p", "--n_process", default=20, type=int,
                    help="number of process in multiprocessing.")

    return parser.parse_args()


def main():

    args = parse_args()

    # Load dataframes

    # It gets the descriptions processed [TRAINING]:
    itemlist = ItemList()
    itemlist.load_items_from_file(args.train)

    clusters_df = pd.read_csv(args.results + "clusters.csv.zip", sep=';',
                              low_memory=False)
    cluster_prices_statistics = pd.read_csv(args.results + "cluster_prices_statistics.csv.zip",
                                            sep=';', low_memory=False)
    items_clusters_wo_outliers = pd.read_csv(args.results + "items_clusters_train_wo_out.csv.zip",
                                             sep=';', low_memory=False)

    # Save tables to HIVE

    version = args.version
    num_process = args.n_process

    dataframe_to_hive_table(itemlist.items_df, "f03_itens", version)
    dataframe_to_hive_table(clusters_df, "f03_grupos", version)
    dataframe_to_hive_table(cluster_prices_statistics, "f03_grupos_estatisticas", version)
    dataframe_to_hive_table(items_clusters_wo_outliers, "f03_itens_precificacao", version)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
