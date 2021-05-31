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
from item.clustering.item_representation import (
    get_group_embeddings_matrix,
    get_group_embeddings_from_dict,
    save_items_embeddings,
    load_items_embeddings,
    normalize,
    get_items_embeddings
)
from item.clustering.utils import (
    save_clustering_results_pickle,
    save_models_pickle
)

# Import HDBSCAN
import hdbscan

# Import UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction)
import umap


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--sample_frac',type=float,default=1.0,
        help = 'Fraction of the original dataset that will be used')
    p.add_argument('--embeddings_path',type=str,default='../../../embeddings/word2vec/cbow_s50.txt',
        help='Path to the file containing the embeddings to be used in the representation')
    p.add_argument('--min_cluster_size',type=int,default=30,
        help='The minimum size of a cluster')
    p.add_argument('--reduc_dim',type=int,default=15,
        help='Indicates the number of dimentions that will be used in the dimentionality '\
                   'reduction. When none is set the dataset will be used in its original '\
                   'dimentionality')
    p.add_argument('--outpath',type=str,default='./results/tcu/',
        help='path to the write the outputs')
    p.add_argument('--out_emb_file',type=str,default='embeddings.pkl',
        help='embeddings filename to the output data')
    p.add_argument('--input', type = str, default = '../dados/items_preprocessed_v3.zip'
                  ,help='file containing the items dataset')
    p.add_argument('--operation', type = str, default = 'tcu'
                  ,help='operation used to build the items embeddings')
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

    t0 = time.time()
    args = parse_args()

    print(time.asctime()," Getting the descriptions processed:")
    # It gets the descpitons processed:
    itemlist = ItemList()
    itemlist.load_items_from_file(args.input)
    num_items = len(itemlist.items_list)

    print(time.asctime()," Loading word embeddings files")
    #  word embeddings file, each line contains an embedding
    word_embeddings_file = args.embeddings_path
    # read word embeddings from file and store them in a map
    word_embeddings = load_word_embeddings(word_embeddings_file)

    print(time.asctime()," Getting the tags of tokens descriptions")
    # Get the tags of tokens descriptions
    if args.class2use:
        embedding_type=args.class2use
        print(time.asctime()," Using only ",embedding_type," words")
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

    if args.sample_frac < 1.0:
        sample_size = int(len(itemlist.items_list)*args.sample_frac)
        print(time.asctime()," Using a sample of ", sample_size, " items")
        used_ids = random.sample(list(range(0,len(itemlist.items_list))),sample_size)
        sampled_items = [itemlist.items_list[x] for x in used_ids]
        used_item_descriptions = sampled_items
        print(time.asctime()," Constructing the items embeddings vectors [SAMPLE]")
        # Build the vector representation for the items using the word embeddings
        items_embeddings_sample = get_items_embeddings(used_item_descriptions,
                                            word_embeddings, word_class,
                                            categories=categories,
                                            embedding_type=embedding_type,
                                            type='list', operation=operation)
        items_embeddings_sample = normalize(items_embeddings_sample)

    print(time.asctime()," Constructing the items embeddings vectors")
    items_embeddings = get_items_embeddings(itemlist.items_list, word_embeddings,
                                    word_class, categories=categories,
                                    embedding_type=embedding_type,
                                    type='list', operation=operation)

    del itemlist
    del word_embeddings

    print(time.asctime()," L2 Normalizing data in order to approximate euclidean distance to arc-cos")
    items_embeddings = normalize(items_embeddings)

    print(time.asctime()," Start dimentionality reduction")
    if operation == 'tcu':
        umap_redux = umap.UMAP(n_components=args.reduc_dim, random_state=999,
                               metric='cosine', verbose=True)
    else:
        umap_redux = umap.UMAP(n_components=args.reduc_dim, random_state=999,
                               metric='euclidean', verbose=True)

    if args.sample_frac < 1.0:
        umap_redux.fit(X=items_embeddings_sample)
    else:
        umap_redux.fit(X=items_embeddings)

    items_embeddings = umap_redux.transform(X=items_embeddings)

    print(time.asctime()," Running HDBSCAN")
    t1 = time.time()

    # HDBSCAN clustering
    if operation == 'tcu':
        hdb_clusterer = hdbscan.HDBSCAN(metric='l2', min_cluster_size=args.min_cluster_size,
                                    min_samples=1, prediction_data=True, core_dist_n_jobs=8)
    else:
        hdb_clusterer = hdbscan.HDBSCAN(metric='l2', min_cluster_size=args.min_cluster_size,
                                        prediction_data=True, core_dist_n_jobs=8)

    hdb_clusterer.fit_predict(items_embeddings)
    quantile_outliers_hdbscan = 0.95

    threshold = pd.Series(hdb_clusterer.outlier_scores_).quantile(quantile_outliers_hdbscan)
    outliers = set(np.where(hdb_clusterer.outlier_scores_ > threshold)[0])
    clusters_embeddings = hdb_clusterer.labels_
    print(time.asctime()," HDBSCAN time: ",time.time()-t1)
    print(time.asctime()," Results")

    cluster_items = collections.defaultdict(list)
    cluster_items_outliers = collections.defaultdict(list)
    for desc_id in range(num_items):
        cluster = str(clusters_embeddings[desc_id])
        if desc_id in outliers:
            cluster_items_outliers[cluster].append(desc_id)
        else:
            cluster_items[cluster].append(desc_id)

    items_embeddings = list(items_embeddings)
    items_vec = {}
    for i in range(len(items_embeddings)):
        items_vec[str(i)] = [float(x) for x in items_embeddings[i]]

    save_clustering_results_pickle(cluster_items, cluster_items_outliers,
                                   args.outpath)
    save_models_pickle(hdb_clusterer, umap_redux, args.outpath)
    save_items_embeddings(items_vec, args.outpath + args.out_emb_file)

    print(time.asctime()," Total time: ",time.time()-t0)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
