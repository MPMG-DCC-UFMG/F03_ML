
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

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
from nlp.utils import (
    read_json_file)
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
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='f03_items_preprocessed_complete',
                    help='file containing the items dataset')
    p.add_argument("-r", "--results", required=True, help="results files directory.")
    p.add_argument("-s", "--suffix", default="", help="table name suffix.")
    p.add_argument("-v", "--version", default=1, help="execution version.")
    p.add_argument("-n", "--n_process", default=16, type=int,
                   help="number of process in multiprocessing.")

    parsed = p.parse_args()

    return parsed


def main():

    args = parse_args()

    # Load dataframes

    print(time.asctime()," Getting the descriptions processed:")

    # It gets the descriptions processed [TRAINING]:
    # itemlist = ItemList()
    # itemlist.load_items_from_file(args.input)

    clusters_df = pd.read_csv(args.results + "clusters.csv.zip", sep=';',
                              low_memory=False)
    cluster_prices_statistics = pd.read_csv(args.results + "cluster_prices_statistics.csv.zip",
                                            sep=';', low_memory=False)
#     cluster_prices_statistics_year = pd.read_csv(args.results + "cluster_prices_statistics_year.csv.zip",
#                                             sep=';', low_memory=False)
#     items_clusters_wo_outliers = pd.read_csv(args.results + "items_clusters_train_wo_out.csv.zip",
#                                              sep=';', low_memory=False)

    # Save tables to HIVE

    version = args.version
    num_process = args.n_process

    print(time.asctime()," Saving dataframe to a HIVE table:")

    dataframe_to_hive_table(clusters_df, "f03_grupos" + args.suffix, version,
                            num_process=num_process)
    dataframe_to_hive_table(cluster_prices_statistics, "f03_banco_precos_grupos" + args.suffix,
                            version, num_process=num_process)
#     dataframe_to_hive_table(cluster_prices_statistics, "f03_banco_precos_grupos_ano" + args.suffix,
#                             version, num_process=num_process)
#     dataframe_to_hive_table(itemlist.items_df, "f03_itens" + args.suffix, version,
#                             num_process=num_process)
#     dataframe_to_hive_table(items_clusters_wo_outliers, "f03_itens_precificacao" + args.suffix,
#                             version, num_process=num_process)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
