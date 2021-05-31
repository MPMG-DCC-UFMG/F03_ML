# imports

import pandas as pd
import argparse
import numpy as np
import collections
import copy
import random
import matplotlib.pyplot as plt
import time
import multiprocessing
import nltk
import pickle
import json
from item.item_list import (
    ItemList
)
from nlp.word_embeddings import (
    load_word_embeddings
)
from nlp.pos_tagging import (
    get_tokens_tags
)
from item.clustering.item_representation import *
from item.clustering.utils import *
from item.clustering.clustering import run_baseline_clustering


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_path',type=str,default='../../../embeddings/word2vec/cbow_s50.txt',
        help='Path to the file containing the embeddings to be used in the representation')
    p.add_argument('--outpath',type=str,default='./results/tcu/',
        help='path to the write the outputs')
    p.add_argument('--out_emb_file',type=str,default='embeddings.json',
        help='embeddings filename to the output data')
    p.add_argument('--input', type = str, default = 'f03_items_preprocessed_complete'
                  ,help='file containing the items dataset')
    p.add_argument('--operation', type = str, default = 'tcu'
                  ,help='operation used to build the items embeddings')
    p.add_argument('--algorithm', type = str, default = 'hdbscan'
                  ,help='clustering algorithm to use')
    p.add_argument('--n_process', type=int, default=10
                  ,help='number of process to use on clustering')
    p.add_argument("-v", "--version", default="", help="execution version.")
    p.add_argument("-i", "--hive", default=False, help="load table from hive and \
                    save the results on hive.")
    p.add_argument('--class2use', nargs = '*'
                  ,help='The list of syntatic classes that will be used to construct the '\
                   'embeddings. When none is set all the description will be used '\
                   'Options are: N, MED, A, ADJ....')
    p.add_argument('--categories', nargs = '*'
              ,help='The list of categories that will be used to construct the '\
               'embeddings. When none is set all the description will be used '\
               'Options are: unidades_medida, números, tamanho,....')

    parsed = p.parse_args()

    return parsed


def main():

    args = parse_args()

    print(time.asctime()," Getting the descriptions processed:")

    # It gets the descpitons processed:
    itemlist = ItemList()

    if args.hive:
        itemlist.load_items_from_hive_table(args.input)
    else:
        itemlist.load_items_from_file(args.input)

    num_items = len(itemlist.items_list)

    print(time.asctime()," Loading word embeddings files")

    #  word embeddings file, each line contains an embedding
    word_embeddings_file = args.embeddings_path
    # read word embeddings from file and store them in a map
    word_embeddings = load_word_embeddings(word_embeddings_file,
                                           words_set=itemlist.unique_words)

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
    algorithm = args.algorithm
    n_process = args.n_process

    clusters, outliers, items_vec, clustering_model, \
    reducer_model = run_baseline_clustering(itemlist, word_embeddings, word_class,
                                        algorithm='hdbscan', categories=categories,
                                        embedding_type=embedding_type,
                                        operation=operation, n_process=n_process)

    if args.hive:
        version = args.version
        save_clustering_results_hive_table(clusters, outliers, 'f03_grupos_hdbscan', \
                                           'f03_grupos_hdbscan_outliers', version)
    else:
        save_clustering_results_pickle(clusters, outliers, args.outpath)

    save_models_pickle(clustering_model, reducer_model, args.outpath)
    save_items_embeddings(items_vec, args.outpath + args.out_emb_file)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
