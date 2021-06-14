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

# clustering evaluation
from item.clustering.evaluate import (
    get_score_pickle,
    evaluate_results_pickle,
    evaluate_results,
    number_of_outliers_dict,
    get_score_baseline_pickle
)

import mlflow


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_path',type=str,
        default='../data//embeddings/models/fasttext/sg/output/items_embeddings.vec',
        help='path to the file containing the embeddings to be used in the representation')
    p.add_argument('--outpath',type=str,default='./results/',
        help='path to the write the outputs')
    p.add_argument('--input', type = str, default = 'f03_items_preprocessed_complete'
                  ,help='file containing the items dataset')
    p.add_argument('--operation', type = str, default = 'concatenate'
                  ,help='operation used to build the items embeddings')
    p.add_argument('--algorithm', type = str, default = 'hdbscan'
                  ,help='clustering algorithm to use')
    p.add_argument('--n_process', type=int, default=10
                  ,help='number of process to use on clustering')
    p.add_argument("-p", "--password", default="", help="connection password.")
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


def evaluate_clustering_results(clusters, outliers, embeddings, metrics):

    outliers_items, outliers_groups, total = number_of_outliers_dict(clusters,
                                                                     outliers,
                                                                     baseline=True,
                                                                     total_cov=False)
    perc_outliers = 100*(outliers_items/total)

    outliers_items, outliers_groups, total = number_of_outliers_dict(clusters,
                                                                     outliers,
                                                                     baseline=True,
                                                                     total_cov=True)
    perc_excluded = 100*(outliers_items/total)

    avg_calinski = get_score_baseline_pickle(clusters, embeddings, score='calinski',
                                             sample_size=None, norm=False)
    avg_davies = get_score_baseline_pickle(clusters, embeddings, score='davies',
                                           sample_size=None, norm=False)
    avg_silhouette_euclidean = get_score_baseline_pickle(clusters, embeddings,
                                                         score='silhouette',
                                                         metric='euclidean',
                                                         sample_size=None,
                                                         norm=False)
    avg_silhouette_cosine = get_score_baseline_pickle(clusters, embeddings,
                                                      score='silhouette',
                                                      metric='cosine',
                                                      sample_size=None,
                                                      norm=False)

    metrics['n_groups'] = len(clusters)
    metrics['outliear'] = perc_outliers
    metrics['excluded'] = perc_excluded
    metrics['avg_calinski'] = avg_calinski
    metrics['avg_davies-bouldin'] = avg_davies
    metrics['avg_silhouette-euclidean'] = avg_silhouette_euclidean
    metrics['avg_silhouette-cosine'] = avg_silhouette_cosine


def run_training(itemlist, word_embeddings, word_class, parameters,
                 artifacts, metrics, input, algorithm):

    categories = parameters['categories']
    embedding_type = parameters['tags']
    operation = parameters['operation']

    run_name = algorithm + '_' + operation
    mlflow.set_experiment(experiment_name='f03_clustering_training')

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.set_tag("algorithm") = algorithm
        mlflow.set_tag("input") = input

        mlflow.log_params(parameters)

        clusters, outliers, items_vec, clustering_model, \
        reducer_model = run_baseline_clustering(itemlist, word_embeddings,
                                                word_class, algorithm=algorithm,
                                                categories=categories,
                                                embedding_type=embedding_type,
                                                operation=operation,
                                                n_process=n_process)

        evaluate_clustering_results(clusters, outliers, embeddings, metrics)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(artifacts)

    return clusters, outliers, items_vec, clustering_model, reducer_model


def main():

    args = parse_args()

    print(time.asctime()," Getting the descriptions processed:")

    # It gets the descpitons processed:
    itemlist = ItemList()

    if args.hive:
        itemlist.load_items_from_hive_table(args.input, args.password)
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

    parameters = {
        'embeddings' : word_embeddings_file,
        'categories' : categories,
        'tags' : embedding_type,
        'operation' : operation
    }

    metrics = {
        'n_groups' : None,
        'outliers' : None,
        'excluded' : None,
        'avg_calinski' : None,
        'avg_davies-bouldin' : None,
        'avg_silhouette-euclidean' : None,
        'avg_silhouette-cosine' : None
    }

    artifacts = {
        'clusters' : None,
        'outliers' : None,
        'items_vec' : None,
        'clustering_model' : None,
        'reducer_model' : None
    }

    if args.hive:
        artifacts['clusters'] = 'f03_grupos_hdbscan'
        artifacts['outliers'] = 'f03_grupos_hdbscan_outliers'
    else:
        artifacts['clusters'] = args.outpath + "results.pkl"
        artifacts['outliers'] = args.outpath + "outliers.pkl"

    artifacts['items_vec'] = args.outpath + "embeddings.json"
    artifacts['clustering_model'] = args.outpath + "clustering_model.pkl"
    artifacts['reducer_model'] = args.outpath + "dimred_model.pkl"

    clusters, outliers, items_vec, clustering_model, \
    reducer_model = run_training(itemlist, word_embeddings, word_class,
                                 parameters, artifacts, metrics, args.input,
                                 algorithm)

    if args.hive:
        version = args.version
        save_clustering_results_hive_table(clusters, outliers, 'f03_grupos_hdbscan',
                                           'f03_grupos_hdbscan_outliers', version,
                                           args.password)
    else:
        save_clustering_results_pickle(clusters, outliers, args.outpath)

    save_models_pickle(clustering_model, reducer_model, args.outpath)
    save_items_embeddings(items_vec, args.outpath + "embeddings.json")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
