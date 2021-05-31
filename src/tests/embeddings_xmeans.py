import pandas as pd
import json
import random
import numpy as np
import collections
import time
import multiprocessing
import nltk
import pickle
from nlp.pos_tagging import (
    get_tokens_tags
)
from nlp.word_embeddings import (
    load_word_embeddings,
    get_items_embeddings,
    cosine_distance
)
from utils.read_files import (
    get_items)
from item.item_list import (
    ItemList,
    Item
)

#Import xmeans through pyclustering library:
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer


def main():
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    n_threads = 10

    # It gets the descriptions processed:
    itemlist = ItemList()
    itemlist.load_items_from_file('../dados/items_preprocessed.zip',
                                  just_words=True)

    # word embeddings file, each line contains a word embedding
    word_embeddings_file = '../../../embeddings/cbow_s50.txt'

    # read word embeddings from file and store them in a map
    word_embeddings = load_word_embeddings(word_embeddings_file)

    # Get the tags of tokens descriptions
    word_class = get_tokens_tags()

    # Build the vector representation for an item using the word embeddings
    items_embeddings = get_items_embeddings(itemlist.items_list, word_embeddings,
                                          word_class, embedding_type=['N', 'MED'],
                                          type='list')
    del itemlist
    del word_embeddings
    del word_class

    # X-means clustering:

    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.
    amount_initial_centers = 1000
    initial_centers = kmeans_plusplus_initializer(items_embeddings, amount_initial_centers).initialize()

    # Create instance of X-Means algorithm. The algorithm will start analysis from
    # 2 clusters, the maximum number of clusters that can be allocated is 20.
    xmeans_instance = xmeans(items_embeddings, initial_centers, kmax=6000,
                             ccore=False)
    xmeans_instance.process()

    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    group_len = len(clusters)

    print(group_len)
    print(clusters)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
