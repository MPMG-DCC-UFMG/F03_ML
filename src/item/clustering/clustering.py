# imports

import pandas as pd
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
import shutil
from .utils import *
from .item_representation import *

# Import xmeans through pyclustering library:
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer;
from pyclustering.cluster.xmeans import xmeans

# Import agglomerative clustering through scikit learn library:
from sklearn.cluster import AgglomerativeClustering

# Import agglomerative clustering through pyclustering library:
from pyclustering.cluster.agglomerative import agglomerative

# Import OPTICS through pyclustering library:
from pyclustering.cluster.optics import optics

# Import DBSCAN through pyclustering library:
from pyclustering.cluster.dbscan import dbscan

# Import HDBSCAN
import hdbscan
from hdbscan.prediction import approximate_predict

# Import UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction)
import umap

# Import PCA (Principal component analysis)
from sklearn.decomposition import PCA


def get_hdbscan_outliers(clustering, quantile_outliers_hdbscan=0.95):

    threshold = pd.Series(clustering.outlier_scores_).quantile(quantile_outliers_hdbscan)
    outliers = set(np.where(clustering.outlier_scores_ > threshold)[0])
    clusters_embeddings = clustering.labels_

    return clusters_embeddings, outliers


def get_hdbscan_clusters(group_descriptions, clusters_embeddings, outliers):

    cluster_items = collections.defaultdict(list)
    cluster_items_outliers = collections.defaultdict(list)

    for it, desc_id in enumerate(group_descriptions):
        cluster = clusters_embeddings[it]
        if it in outliers:
            cluster_items_outliers[cluster].append(it)
        else:
            cluster_items[cluster].append(it)

    return cluster_items, cluster_items_outliers


def get_items_for_process(itemlist, groups, limits):

    items_ids = []
    groups_names = list(groups.keys())
    group_descriptions = list(groups.values())

    lower = limits[0]
    upper = limits[1]

    sample_groups_names = []
    sample_groups_items = []

    # conferir limites
    for i in range(lower, upper + 1):
        sample_groups_names.append(groups_names[i])
        sample_groups_items.append(group_descriptions[i])
        items_ids += group_descriptions[i]

    sample_items = itemlist.items_df.loc[items_ids]
    itemlist.items_df.drop(items_ids, inplace=True)

    return sample_items, sample_groups_names, sample_groups_items


def get_items_for_processes(itemlist, n_process, process_ranges, groups):

    # get items sample for each process
    process_items = {}  # process -> (items, groups_names, groups_items)

    for i in range(n_process):
        lower = process_ranges[0][i]
        upper = process_ranges[1][i]
        limits = (lower, upper)
        items_data, groups_names, groups_items = get_items_for_process(itemlist,
                                                        groups, limits)
        process_items[i] = (items_data, groups_names, groups_items)

    return process_items


def shuffle_groups(groups):

    group_len = len(groups)
    groups_new = {}
    keys_ft = list(groups.keys())
    random.shuffle(keys_ft)
    random.shuffle(keys_ft)

    for k in keys_ft:
        groups_new[k] = groups[k]
    del groups

    return groups_new


def merge_multiprocessing_results(results_process, outliers_process,
                                  output_path, n_process):

    clusters = {}
    outliers = {}
    items_vec = {}
    clustering_model = {}
    reducer_model = {}

    for i in range(n_process):
        clusters.update(results_process[i])
        outliers.update(outliers_process[i])

    for group_name, item_ids in clusters.items():
        items_vec[group_name] = get_cluster_embeddings(output_path, group_name)[group_name]
        clustering_model[group_name] = get_model(output_path, "clustering_model", group_name)[group_name]
        reducer_model[group_name] = get_model(output_path, "dimred_model", group_name)[group_name]

    # remove directory
    shutil.rmtree(os.path.join(output_path, "clustering_model"))
    shutil.rmtree(os.path.join(output_path, "dimred_model"))
    shutil.rmtree(os.path.join(output_path, "embeddings"))

    return clusters, outliers, items_vec, clustering_model, reducer_model


def cluster_embeddings_by(embeddings_matrix, algorithm, kmax=None):
    '''
        It runs xmeans on the embeddings matrix and returns clusters.

        embeddings_matrix (list): list of item embeddings.
        algorithm (str): clustering algorithm to be used ['xmeans', 'agglomerative',
                         'optics', 'dbscan'].
        kmax (int): maximum number of clusters that can be allocated.
    '''

    if algorithm == 'xmeans':
        xmeans_instance = xmeans(embeddings_matrix, kmax=kmax, ccore=False)
        xmeans_instance.process()
        clusterer = xmeans_instance
    elif algorithm == 'agglomerative':
        agglomerative_instance = agglomerative(embeddings_matrix, number_clusters=kmax,
                                               ccore=False)
        agglomerative_instance.process()
        clusterer = agglomerative_instance
    elif algorithm == 'optics':
        radius = 0.4
        optics_instance = optics(embeddings_matrix, eps=radius, minpts=30, ccore=False)
        optics_instance.process()
        clusterer = optics_instance
    elif algorithm == 'dbscan':
        radius = 0.4
        dbscan_instance = dbscan(embeddings_matrix, eps=radius, neighbors=30, ccore=False)
        dbscan_instance.process()
        clusterer = dbscan_instance

    return clusterer


def baseline_plus_clustering(items_data, groups, word_embeddings, word_class,
                            categories, embedding_type, it_process, algorithm,
                            operation, result_dict, output_path,
                            items_embeddings=None, dim_reduction=True, norm=True,
                            sample_size=0.2):
    '''
        It runs a clustering algorithm for each group generated by the first token
        grouping.

        itemlist (ItemList object): contains the item and informations about the
                                    dataset.
        first_token_groups (dict): groups generated by the first token grouping.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
        it_process (int): process number.
        algorithm (str): clustering algorithm to be used ['xmeans', 'agglomerative',
                         'optics', 'dbscan'].
        operation (str): operation to be used to build the item embeddings/vectors.
        items_embeddings (dict): item embeddings.
        dim_reduction (bool): if the dimensionality reduction should be performed.
        norm (bool): if the item vectors should be normalized.
    '''

    print(it_process)

    # It creates a list of the the keys of these groups:
    groups_names = groups[0]

    # It gets the values of each group (i.e., the ids of the descriptions into that group):
    group_descriptions = groups[1]

    # It defines the dictionary that will have the clustering with first token
    # together with x-means considering a the item embeddings
    # grouped by the first token approach:
    first_token_plus_emb = {}

    for ft_it in range(len(groups_names)):
        if(len(group_descriptions[ft_it]) > 30):
            if items_embeddings == None:
                embeddings_matrix = get_group_embeddings_matrix(group_descriptions[ft_it],
                                                                items_data,
                                                                word_embeddings,
                                                                word_class, categories,
                                                                embedding_type,
                                                                norm, operation)
                if dim_reduction:
                    reducer = run_dim_reduction(embeddings_matrix,
                                                sample_size=sample_size,
                                                algorithm='UMAP')
                    embeddings_matrix = reducer.transform(embeddings_matrix)
                    dimred_model = {}
                    dimred_model[groups_names[ft_it]] = reducer
                    save_model(output_path, "dimred_model", groups_names[ft_it],
                               dimred_model)

                items_vec = {}
                for _id, desc_id in enumerate(group_descriptions[ft_it]):
                    items_vec[desc_id] = [float(x) for x in embeddings_matrix[_id]]

                save_cluster_embeddings(output_path, groups_names[ft_it], items_vec)
            else:
                embeddings_matrix = get_group_embeddings_from_dict(group_descriptions[ft_it],
                                                                  items_embeddings,
                                                                  norm=norm)

            kmax = len(group_descriptions[ft_it])/30
            #It applies the clusters on the embeddings matrix:
            clustering = cluster_embeddings_by(embeddings_matrix, algorithm, kmax)
            cluster_model = {}
            cluster_model[groups_names[ft_it]] = clustering
            clusters_embeddings = clustering.get_clusters()

            save_model(output_path, "clustering_model", groups_names[ft_it],
                       cluster_model)

            it = 0
            for c in clusters_embeddings:
                # It translates ids from x-means to actual descriptions (new groups):
                desc_ids = translate_id_to_descriptions(c, group_descriptions[ft_it])
                # It defines the key of the map:
                new_key = groups_names[ft_it] + '_' + str(it)
                # It sets the maps:
                first_token_plus_emb[new_key] = desc_ids
                it = it + 1
        else:
            first_token_plus_emb[groups_names[ft_it]] = group_descriptions[ft_it]

    result_dict[it_process] = first_token_plus_emb


def cluster_embeddings_by_hdbscan(embeddings_matrix, min_samples=None):
    '''
        It runs the HDBSCAN algorithm on the embeddings matrix and returns clusters.

        embeddings_matrix (list): list of item embeddings.
        min_samples (int): the number of samples in a neighbourhood for a point to be
                           considered a core point.
    '''

    hdb_clusterer = hdbscan.HDBSCAN(metric='l2', min_cluster_size=30,
                                    min_samples=min_samples,
                                    prediction_data=True)
    clusters = hdb_clusterer.fit_predict(embeddings_matrix)

    return hdb_clusterer


def run_dim_reduction(embeddings_matrix, sample_size=None, algorithm='UMAP'):
    '''
        Reduce the dimensionality of the input vectors.

        embeddings_matrix (list): list of item embeddings. Input vectors.
        sample_size (float or None): If float, should be between 0.0 and 1.0 and
                                    represent the proportion of the input vectors.
                                    If None, all input vectors are considered.
        algorithm (str): dimensionality reduction algorithm that should be used.
    '''

    if algorithm == 'UMAP':
        reducer = umap.UMAP(n_components=15, metric='euclidean',
                            random_state=999, low_memory=True,
                            verbose=False)
    elif algorithm == 'PCA':
        reducer = PCA(n_components=15, random_state=999)

    num_items = len(embeddings_matrix)
    if sample_size is None:
        reducer.fit(embeddings_matrix)
    else:
        # sample = max(min(num_items, 100), int(num_items*sample_size))
        sample = min(1000, num_items)
        reducer.fit(np.array(random.sample(list(embeddings_matrix),
                    sample)))

    return reducer


def baseline_plus_hdbscan(items_data, groups, word_embeddings,
                          word_class, categories, embedding_type, it_process,
                          operation, result_dict, outliers_dict, output_path,
                          items_embeddings=None, dim_reduction=True, norm=True,
                          sample_size=0.2):
    '''
        It runs the HDBSCAN algorithm for each group in a range.

        itemlist (ItemList object): contains the items and informations about the
                                    dataset.
        first_token_groups (dict): groups generated by the first token grouping.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
        it_process (int): process number.
        algorithm (str): clustering algorithm to be used ['xmeans', 'agglomerative',
                         'optics', 'dbscan'].
        operation (str): operation to be used to build the item embeddings/vectors.
        items_embeddings (dict): item embeddings.
        dim_reduction (bool): if the dimensionality reduction should be performed.
        norm (bool): if the item vectors should be normalized.
    '''

    print(it_process)

    # It creates a list of the the keys of these groups:
    groups_names = groups[0]

    # It gets the values of each group (i.e., the ids of the descriptions into that group):
    group_descriptions = groups[1]

    # It defines the dictionary that will have the clustering with first token
    # together with HDBSCAN considering a the item embeddings
    # grouped by the first token approach:
    first_token_plus_emb = {}
    first_token_plus_emb_outliers = {}

    for ft_it in range(len(groups_names)):
        if(len(group_descriptions[ft_it]) > 30):
            if items_embeddings == None:
                embeddings_matrix = get_group_embeddings_matrix(group_descriptions[ft_it],
                                                                items_data,
                                                                word_embeddings,
                                                                word_class, categories,
                                                                embedding_type,
                                                                norm, operation)
                if dim_reduction:
                    reducer = run_dim_reduction(embeddings_matrix,
                                                sample_size=sample_size,
                                                algorithm='UMAP')
                    embeddings_matrix = reducer.transform(embeddings_matrix)
                    dimred_model = {}
                    dimred_model[groups_names[ft_it]] = reducer
                    save_model(output_path, "dimred_model", groups_names[ft_it],
                              dimred_model)

                items_vec = {}
                for _id, desc_id in enumerate(group_descriptions[ft_it]):
                    items_vec[desc_id] = [float(x) for x in embeddings_matrix[_id]]
                save_cluster_embeddings(output_path, groups_names[ft_it], items_vec)
            else:
                embeddings_matrix = get_group_embeddings_from_dict(group_descriptions[ft_it],
                                                                  items_embeddings,
                                                                  norm=norm)

            #It applies the clusters on the embeddings matrix:
            clustering = cluster_embeddings_by_hdbscan(embeddings_matrix)
            del embeddings_matrix
            hdbscan_model = {}
            hdbscan_model[groups_names[ft_it]] = clustering
            clusters_embeddings, outliers = get_hdbscan_outliers(clustering)
            cluster_items, cluster_items_outliers = get_hdbscan_clusters(group_descriptions[ft_it],
                                                                         clusters_embeddings,
                                                                         outliers)
            save_model(output_path, "clustering_model", groups_names[ft_it], hdbscan_model)

            for cluster, items in cluster_items.items():
                # It translates ids from HDBSCAN to actual descriptions (new groups):
                desc_ids = translate_id_to_descriptions(items, group_descriptions[ft_it])
                desc_ids_outliers = translate_id_to_descriptions(cluster_items_outliers[cluster],
                                                                 group_descriptions[ft_it])
                # It defines the key of the map:
                new_key = groups_names[ft_it] + '_' + str(cluster)
                # It sets the maps:
                first_token_plus_emb[new_key] = desc_ids
                first_token_plus_emb_outliers[new_key] = desc_ids_outliers
        else:
            first_token_plus_emb[groups_names[ft_it]] = group_descriptions[ft_it]
            first_token_plus_emb_outliers[groups_names[ft_it]] = []

    result_dict[it_process] = first_token_plus_emb
    outliers_dict[it_process] = first_token_plus_emb_outliers


def run_baseline_clustering(itemlist, word_embeddings, word_class, output_path,
                            algorithm='hdbscan', items_embeddings=None,
                            dim_reduction=True, norm=True, n_process=10,
                            categories=None, embedding_type=None, operation='mean'):
    '''
        It runs the HDBSCAN algorithm for each group generated by the first token
        grouping.

        itemlist (ItemList object): contains the items and informations about the
                                    dataset.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        algorithm (str): clustering algorithm to be used ['xmeans', 'agglomerative',
                         'optics', 'dbscan'].
        items_embeddings (dict): item embeddings.
        dim_reduction (bool): if the dimensionality reduction should be performed.
        norm (bool): if the item vectors should be normalized.
        n_process (int): number of process.
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
        operation (str): operation to be used to build the item vectors.
    '''

    itemlist.items_df = itemlist.items_df[['palavras', 'unidades_medida', 'numeros', \
                                        'cores', 'materiais', 'tamanho', 'quantidade', \
                                        'item_id', 'original_prep']]

    manager = multiprocessing.Manager()
    results_process = manager.dict()
    outliers_process = manager.dict()
    items_vec_process = manager.dict()
    clustering_model_process = manager.dict()
    reducer_model_process = manager.dict()
    jobs = []

    # It gets the first tokens of each description and groups
    # based on this approach:
    groups = itemlist.get_first_token_groups()
    groups_new = shuffle_groups(groups)
    group_len = len(groups_new)

    # It defines the ranges (of the groups) the process will work on:
    process_ranges = get_ranges(group_len, n_process)
    print('Read ranges')
    print(process_ranges)

    if items_embeddings != None:
        dim_reduction = False

    process_items = get_items_for_processes(itemlist, n_process, process_ranges,
                                            groups_new)
    del itemlist

    if algorithm == 'xmeans':
        for i in range(n_process):
            items_data = process_items[i][0]
            groups_names = process_items[i][1]
            groups_items = process_items[i][2]

            p = multiprocessing.Process(target=baseline_plus_clustering,
            args = (items_data, (groups_names, groups_items), word_embeddings, \
                    word_class, categories, embedding_type, i, algorithm, \
                    operation, results_process, output_path, \
                    items_embeddings, dim_reduction, norm))
            jobs.append(p)
            p.start()
    elif algorithm == 'hdbscan':
        for i in range(n_process):
            items_data = process_items[i][0]
            groups_names = process_items[i][1]
            groups_items = process_items[i][2]

            p = multiprocessing.Process(target=baseline_plus_hdbscan,
            args = (items_data, (groups_names, groups_items), word_embeddings, \
                    word_class, categories, embedding_type, i, operation, \
                    results_process, outliers_process, output_path, \
                    items_embeddings, dim_reduction, norm))
            jobs.append(p)
            p.start()

    for proc in jobs:
        proc.join()
        proc.close()

    # merge multiprocessing results
    clusters, outliers, items_vec, clustering_model, \
    reducer_model = merge_multiprocessing_results(results_process, outliers_process,
                                                  output_path, n_process)

    return clusters, outliers, items_vec, clustering_model, reducer_model


def get_items_clusters(items_data, groups, word_embeddings, word_class,
                       reducer_model, clustering_model, categories, embedding_type,
                       operation, it_process, lower, upper, results_process):
    '''
        Predict labels for new unseen items for each group generated by the first
        token grouping.

        itemlist (ItemList object): contains the items and informations about the
                                    dataset.
        first_token_groups (dict): groups generated by the first token grouping.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        reducer_model (dict): dimensionality reduction models for each group generated
                             by the first token grouping.
        clustering_model (dict): HDBSCAN models for each group generated by the first
                                 token grouping.
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
        operation (str): operation to be used to build the item embeddings/vectors.
        it_process (int): process number.
        results_process (dict): results for each process.
    '''

    print(it_process)

    # It creates a list of the the keys of these groups:
    groups_names = groups[0]

    # It gets the values of each group (i.e., the ids of the descriptions into that group):
    group_descriptions = groups[1]

    results = []

    for ft_it in range(len(groups_names)):
        group = groups_names[ft_it]
        items = group_descriptions[ft_it]
        if group not in reducer_model:
            for item_id in items:
                cluster_id = '-2'
                cluster_prob = 0
                item_dict = items_data.loc[item_id].to_dict()
                dsc_unidade_medida = item_dict['dsc_unidade_medida']
                price = item_dict['preco']
                licitacao = item_dict['licitacao']
                description = item_dict['original_prep']
                original = item_dict['original']
                areas = item_dict['funcao']
                item_result = {'item_id': item_id, 'seq_dim_licitacao': licitacao,
                               'outlier': 1, 'cluster': cluster_id,
                               'cluster_prob': cluster_prob,
                               'dsc_unidade_medida': dsc_unidade_medida,
                               'description': description, 'areas': areas,
                               'price': price}
                results.append(item_result)
        else:
            embeddings_matrix = get_group_embeddings_matrix(items, items_data,
                                                        word_embeddings, word_class,
                                                        categories=categories,
                                                        embedding_type=embedding_type,
                                                        norm=True, operation=operation)
            # It gets the reduced vector for the item
            embeddings_matrix = reducer_model[group].transform(embeddings_matrix)
            # It gets the item cluster
            clusters = approximate_predict(clustering_model[group], embeddings_matrix)
            # ([4, 5, 6, 7, 8], [0.94, ...])
            for _id, item_id in enumerate(items):
                cluster_id = group + '_' + str(clusters[0][_id])
                outlier = 1 if str(clusters[0][_id]) == '-1' else 0
                cluster_prob = clusters[1][_id]
                item_dict = items_data.loc[item_id].to_dict()
                dsc_unidade_medida = item_dict['dsc_unidade_medida']
                price = item_dict['preco']
                licitacao = item_dict['licitacao']
                description = item_dict['original_prep']
                original = item_dict['original']
                areas = item_dict['funcao']
                item_result = {'item_id': item_id, 'seq_dim_licitacao': licitacao,
                               'outlier': outlier, 'cluster': cluster_id,
                               'cluster_prob': cluster_prob,
                               'dsc_unidade_medida': dsc_unidade_medida,
                               'description': description, 'areas': areas,
                               'price': price}
                results.append(item_result)

    results_process[it_process] = results


def predict_items_clusters(itemlist, word_embeddings, word_class, reducer_model,
                           clustering_model, categories=None, embedding_type=None,
                           operation='mean', n_process=10):
    '''
        Predict labels for new unseen items. It runs the HDBSCAN saved model for each
        group generated by the first token grouping.

        itemlist (ItemList object): contains the items and informations about the
                                    dataset.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        reducer_model (dict): dimensionality reduction models for each group generated
                             by the first token grouping.
        clustering_model (dict): HDBSCAN models for each group generated by the first
                                 token grouping.
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
        operation (str): operation to be used to build the item vectors.
        n_process (int): number of process.
    '''

    # It gets the first tokens of each description and groups
    # based on this approach:
    groups = itemlist.get_first_token_groups()
    groups_new = shuffle_groups(groups)
    group_len = len(groups_new)

    manager = multiprocessing.Manager()
    results_process = manager.dict()
    jobs = []

    # It defines the ranges (of the groups) the process will work on:
    process_ranges = get_ranges(group_len, n_process)
    print('Read ranges')
    print(process_ranges)

    process_items = get_items_for_processes(itemlist, n_process, process_ranges,
                                            groups_new)

    for i in range(n_process):
        items_data = process_items[i][0]
        groups_names = process_items[i][1]
        groups_items = process_items[i][2]

        p = multiprocessing.Process(target=get_items_clusters, \
                args = (items_data, (groups_names, groups_items), word_embeddings, \
                        word_class, reducer_model, clustering_model, categories, \
                        embedding_type, operation, i, results_process))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        proc.close()

    results = []
    for i in range(n_process):
        for inst in results_process[i]:
            results.append(inst)

    return results
