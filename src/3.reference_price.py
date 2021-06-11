# imports

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

    p.add_argument('--models', required=False,
                   help='folder containing the clutering and reducer models.')
    p.add_argument('--clusters', required=True,
                   help='folder containing the clutering results.')
    p.add_argument('--outpath', required=True,
                   help='directory where the results will be saved.')
    p.add_argument('--test', type=str, default='f03_items_preprocessed_complete_test',
                   help='test dataset.')
    p.add_argument('--train', type=str, default='f03_items_preprocessed_complete_train',
                  help='train dataset.')
    p.add_argument('--embeddings_path',type=str,default='../../../embeddings/word2vec/cbow_s50.txt',
        help='Path to the file containing the embeddings to be used in the representation')
    p.add_argument("-t", "--run_test", default=False, help="get prices for the items in the test set.")
    p.add_argument("-v", "--version", required=False, help="execution version.")
    p.add_argument("-i", "--hive", default=False, help="load table from hive and \
                    save the results on hive.")
    p.add_argument('--operation', type = str, default = 'tcu'
                  ,help='operation used to build the items embeddings')
    p.add_argument('--class2use', nargs = '*',
                   help='The list of syntatic classes that will be used to construct the '\
                   'embeddings. When none is set all the description will be used '\
                   'Options are: N, MED, A, ADJ....')
    p.add_argument('--categories', nargs = '*',
                   help='The list of categories that will be used to construct the '\
                  'embeddings. When none is set all the description will be used '\
                  'Options are: unidades_medida, números, tamanho,....')

    parsed = p.parse_args()

    return parsed


def main():

    args = parse_args()

    if args.hive:
        results_train, outliers_train = load_clustering_results_hive_table('f03_grupos_sem_outliers', \
                                                                           'f03_grupos_outliers')
    else:
        results_train, outliers_train = load_clustering_results_pickle(args.clusters)

    print(time.asctime()," Getting the descriptions processed:")

    # It gets the descriptions processed [TRAINING]:
    itemlist_train = ItemList()

    if args.hive:
        itemlist_train.load_items_from_hive_table(args.train)
    else:
        itemlist_train.load_items_from_file(args.train)

    print(time.asctime()," Getting the statistics for each cluster finded in the training set:")

    # 1) PRICING: get the statistics for each cluster finded in the training set

    clusters_items = get_clusters_items(results_train, outliers_train)
    clusters_df = get_clusters_dataframe(clusters_items, baseline=True)
    cluster_prices = get_clusters_prices(itemlist_train, results_train)

    print(time.asctime()," Pricing the items of the training set:")
    cluster_prices_statistics, cluster_prices_statistics_year, \
    items_clusters_wo_outliers = pricing(itemlist_train,
                                        results_train,
                                        cluster_prices,
                                        remove_outliers=True,
                                        threshold=0.5,
                                        dsc_unidade=True,
                                        year=True)

    # 2) Get reference prices for the items in the test set
    if args.run_test:

        print(time.asctime()," Getting the models:")
        clustering_model, reducer_model = load_models_pickle(args.models)

        print(time.asctime()," Getting the descriptions processed:")
        # It gets the descriptions processed:
        itemlist = ItemList()

        if args.hive:
            itemlist.load_items_from_hive_table(args.test)
        else:
            itemlist.load_items_from_file(args.test)

        print(time.asctime()," Loading word embeddings files")
        # word embeddings file, each line contains a word embedding
        word_embeddings_file = args.embeddings_path

        # read word embeddings from file and store them in a map
        word_embeddings = load_word_embeddings(word_embeddings_file, itemlist.unique_words)

        print(time.asctime()," Getting the tags of tokens descriptions")
        # Get the tags of tokens descriptions
        if args.class2use:
            embedding_type = args.class2use
            print(time.asctime()," Using only ",embedding_type," words")
            # Get the tags of tokens descriptions
            word_class = get_tokens_tags(itemlist.unique_words)
        else:
            print(time.asctime()," Using all words")
            embedding_type = None
            word_class = None

        if args.categories:
            categories = args.categories
        else:
            categories = None

        operation = args.operation

        results = predict_items_clusters(itemlist, word_embeddings, word_class, \
                                     reducer_model, clustering_model, categories=categories, \
                                     embedding_type=embedding_type, operation=operation, \
                                     n_process=6)
        items_test_df = get_reference_prices(results, cluster_prices_statistics,
                                             dsc_unidade=True)


    # 3) Save result tables
    if args.hive:
        version = args.version
        dataframe_to_hive_table(itemlist_train.items_df, "f03_itens", version)
        dataframe_to_hive_table(clusters_df, "f03_grupos", version)
        dataframe_to_hive_table(cluster_prices_statistics, "f03_grupos_estatisticas", version)
        dataframe_to_hive_table(items_clusters_wo_outliers, "f03_itens_precificacao", version)
    else:
        clusters_df.to_csv(args.outpath + "clusters.csv.zip", sep=';', index=False,
                           compression='zip')
        items_clusters_wo_outliers.to_csv(args.outpath + "items_clusters_train_wo_out.csv.zip",
                                          sep=';', index=False, compression='zip')
        cluster_prices_statistics.to_csv(args.outpath + "cluster_prices_statistics.csv.zip",
                                         sep=';', index=False, compression='zip')
        cluster_prices_statistics_year.to_csv(args.outpath + "cluster_prices_statistics_year.csv.zip",
                                 sep=';', index=False, compression='zip')

    if args.hive and args.run_test:
        dataframe_to_hive_table(items_test_df, "f03_itens_teste", version)
    elif args.run_test:
        items_test_df.to_csv(args.outpath + "items_clusters_test.csv.zip",
                             sep=';', index=False, compression='zip')


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
