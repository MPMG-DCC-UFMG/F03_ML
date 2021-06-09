import numpy as np
import collections
import copy
import random
import pandas as pd
import multiprocessing
import math
from sklearn import preprocessing
from .utils import (
    get_items_vec,
    get_ranges
)
from .metrics import (
    cosine_distance,
    calc_distance
)
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score
)


def number_of_outliers(items_clusters):
    '''
        It gets the number of groups classified as outliers.

        items_clusters (dataframe): clusterings results.
    '''

    outliers = len(items_clusters[items_clusters.id_cluster == '-1'])
    total = len(items_clusters)

    return outliers, 100*float(outliers)/total


def number_of_outliers_dict(results, outliers, baseline=True, total_cov=True):
    '''
        It gets the number of items classified as outliers.

        results (dict): clusterings results (cluster id -> list of items).
        outliers (dict): clusters outliers (cluster id -> list of items classified
                         as outliers)
        baseline (bool): if the baseline was used.
        total_cov (bool): if the groups with less than 31 items should be considered
                          as outliers.
    '''

    if baseline:
        outliers_items = 0
        total = 0
        outliers_groups = 0

        for cluster, items in results.items():
            if total_cov and (cluster[-2:] == '-1' or '_' not in cluster):
                outliers_items += len(items)
                outliers_groups += 1
            elif cluster[-2:] == '-1':
                outliers_items += len(items)
                outliers_groups += 1
            outliers_items += len(outliers[cluster])
            total += (len(items) + len(outliers[cluster]))
    else:
        outliers_items = 0
        total = 0
        outliers_groups = 0

        for cluster, items in results.items():
            if cluster == '-1':
                outliers_items += len(items)
                outliers_groups += 1
            outliers_items += len(outliers[cluster])
            total += (len(items) + len(outliers[cluster]))

    return outliers_items, outliers_groups, total


def normalize(embeddings):
    '''
        It normalizes item embeddings/vectors.
    '''

    embeddings_normalized = preprocessing.normalize(embeddings, norm='l2')
    return embeddings_normalized


def get_items_distances(items, item_embedding, distance='euclidean'):
    '''
        Get distance of all item pairs in a list.

        items (list): item ids.
        items_embeddings (dict): item embeddings (item id -> item embedding)
        distance (str): the metric to use when calculating distance between instances
                        in a feature array.
                        'euclidean' -> Euclidean distance
                        'cosine' -> Cosine distance
    '''

    items_distances = collections.defaultdict(list)

    j = 0
    for item_id in items:
        itemA = item_id
        embeddingA = np.array(item_embedding[itemA])
        for i in range(j, len(items)):
            itemB = items[i]
            if itemB != itemA:
                embeddingB = np.array(item_embedding[itemB])
                items_distance = calc_distance(embeddingA, embeddingB, distance)
                items_distances[itemA].append((itemB, items_distance))
                items_distances[itemB].append((itemA, items_distance))
        j += 1

    return items_distances


def get_intraclusters_distances(item_embedding, groups, it_thread, lower, upper,
                               Result, distance):
    '''
        Computes the average distance for each cluster.
    '''

    print(it_thread)

    # It creates a list of the the keys of these groups:
    group_name = list(groups.keys())
    # It gets the values of each group (i.e., the ids of the descriptions into that group):
    group_descriptions = list(groups.values())

    # Iterator of the first token groups:
    ft_it = lower

    intra_cluster_distance = {}

    while ft_it <= upper:

        if len(group_descriptions[ft_it]) > 30:
            distances = get_items_distances(group_descriptions[ft_it], item_embedding, distance)
            distance_values = []
            for item, distance_list in distances.items():
                for s in distance_list:
                    distance_values.append(s[1])

            distance_values = [x for x in distance_values if x != None and math.isnan(x) == False]
            if len(distance_values) > 1:
                intra_cluster_distance[group_name[ft_it]] = {'mean': np.mean(distance_values), \
                                                             'max': np.max(distance_values)}
            else:
                intra_cluster_distance[group_name[ft_it]] = {'mean': 1.0, 'max': 1.0}

        ft_it = ft_it + 1

    Result[it_thread] = intra_cluster_distance


def evaluate_results(results, remove_outliers=True, n_threads=10,
                     metric='euclidean', norm=False):
    '''
        Computes the average distance for each cluster.

        results (dataframe): clusterings results.
        remove_outliers (bool): if the outliers should be removed for evaluation.
        n_threads (int): threads to use.
        metric (string): The metric to use when calculating distance between instances
                         in a feature array.
                         'euclidean' -> Euclidean distance
                         'cosine' -> Cosine distance
        norm (bool): if the item embeddings/vectors should be normalized.
    '''

    if remove_outliers:
        results = results[results.id_cluster != -1]

    X = get_items_vec(results)
    items_ids = list(results['id_item'])
    labels = list(results['id_cluster'])
    first_tokens = list(results['primeiro_token'])
    groups = collections.defaultdict(list)
    cluster_name = dict()
    name_cluster = dict()

    if first_tokens.count('-999') == 0:
        cluster_id = 0
        _id = 0
        for first_token, id_cluster in results[['primeiro_token', 'id_cluster']].values.tolist():
            cluster = first_token + '_' + str(id_cluster)
            if cluster not in cluster_name:
                cluster_name[cluster] = cluster_id
                name_cluster[cluster_id] = cluster
                cluster_id += 1
            groups[cluster_name[cluster]].append(_id)
            _id += 1
    else:
        cluster_id = 0
        _id = 0
        for id_cluster, id_item in list(results['id_cluster']):
            if cluster not in cluster_name:
                cluster_name[cluster] = cluster_id
                name_cluster[cluster_id] = cluster
                cluster_id += 1
            groups[cluster_name[cluster]].append(_id)
            _id += 1

    group_len = len(groups)
    print('Groups:', group_len)

    groups_new = {}
    keys_ft = list(groups.keys())
    random.shuffle(keys_ft)
    random.shuffle(keys_ft)

    for k in keys_ft:
        groups_new[k] = groups[k]

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    # It defines the ranges (of the groups) the threads will work on:
    thread_ranges = get_ranges(group_len, n_threads)
    print('Read ranges')
    print(thread_ranges)

    # Normalize item embeddings
    if norm:
        X = normalize(X)

    for i in range(n_threads):
        p = multiprocessing.Process(target=get_intraclusters_distances, \
        args = (X, groups_new, i, thread_ranges[0][i], thread_ranges[1][i], \
                return_dict, metric))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    dictionary_clusters = {}
    for i in range(n_threads):
        group_distances = {}
        for group, distance in return_dict[i].items():
            group_distances[name_cluster[group]] = distance
        dictionary_clusters.update(group_distances)

    return dictionary_clusters


def get_score(results, remove_outliers=True, score='davies', sample_size=0.2,
              metric='euclidean', norm=False):
    '''
        Computes the given score (Davies-Bouldin, Calinski and Harabasz, Silhouette
                                  Coefficient).

        results (dataframe): clusterings results.
        remove_outliers (bool): if the outliers should be removed for evaluation.
        score (string): score to compute.
                        'davies' -> Davies-Bouldin
                        'calinski' -> Calinski and Harabasz
                        'silhouette' -> Silhouette Coefficient
        sample_size (float): The size of the sample to use when computing the
                             Silhouette Coefficient on a random subset of the data.
        metric (string): The metric to use when calculating distance between instances
                         in a feature array.
                         'euclidean' -> Euclidean distance
                         'cosine' -> Cosine distance
        norm (bool): if the item embeddings/vectors should be normalized.
    '''

    random.seed(a=999)

    if remove_outliers:
        results = results[results.id_cluster != -1]

    X = get_items_vec(results)
    first_tokens = list(results['primeiro_token'])

    if first_tokens.count('-999') == 0:
        labels = []
        for first_token, id_cluster in results[['primeiro_token', 'id_cluster']].values.tolist():
            cluster = first_token + '_' + str(id_cluster)
            labels.append(cluster)
    else:
        labels = list(results['id_cluster'])

    # Normalize item embeddings
    if norm:
        X = normalize(X)

    if score == 'davies':
        return davies_bouldin_score(X, labels)
    elif score == 'calinski':
        return calinski_harabasz_score(X, labels)
    elif score == 'silhouette':
        sample = int(sample_size*len(X)) if sample_size != None else None
        return silhouette_score(X, labels, sample_size=sample, random_state=999,
                                metric=metric)


def evaluate_results_pickle(results, embeddings, baseline=True,
                            remove_outliers=True, n_threads=10, metric='euclidean',
                            norm=False):
    '''
        Computes the average distance for each cluster.

        results (dict): clusterings results (cluster_id -> list of items).
        embeddings (dict): item embeddings/vectors (item_id -> item vector).
        baseline (bool): if the baseline was used.
        remove_outliers (bool): if the outliers should be removed for evaluation.
        n_threads (int): threads to use.
        metric (string): The metric to use when calculating distance between instances
                         in a feature array.
                        'euclidean' -> Euclidean distance
                        'cosine' -> Cosine distance
        norm (bool): if the item embeddings/vectors should be normalized.
    '''


    X = []
    groups = collections.defaultdict(list)
    cluster_name = dict()
    name_cluster = dict()

    if baseline:
        cluster_id = 0
        _id = 0
        for group, items in results.items():
            if remove_outliers and (group[-2:] == '-1' or '_' not in group):
                continue
            name_cluster[cluster_id] = group
            for item in items:
                if isinstance(list(embeddings.keys())[0], int):
                    X.append(embeddings[item])
                else:
                    X.append(embeddings[str(item)])
                groups[cluster_id].append(_id)
                _id += 1
            cluster_id += 1
    else:
        cluster_id = 0
        _id = 0
        for group, items in results.items():
            if int(group) == -1:
                continue
            name_cluster[cluster_id] = group
            for item in items:
                if isinstance(list(embeddings.keys())[0], int):
                    X.append(embeddings[item])
                else:
                    X.append(embeddings[str(item)])
                groups[cluster_id].append(_id)
                _id += 1
            cluster_id += 1

    group_len = len(groups)
    print('Groups:', group_len)

    groups_new = {}
    keys_ft = list(groups.keys())
    random.shuffle(keys_ft)
    random.shuffle(keys_ft)

    for k in keys_ft:
        groups_new[k] = groups[k]

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    # It defines the ranges (of the groups) the threads will work on:
    thread_ranges = get_ranges(group_len, n_threads)
    print('Read ranges')
    print(thread_ranges)

    # Normalize item embeddings
    if norm:
        X = normalize(X)

    for i in range(n_threads):
        p = multiprocessing.Process(target=get_intraclusters_distances, \
        args = (X, groups_new, i, thread_ranges[0][i], thread_ranges[1][i], \
                return_dict, metric))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    dictionary_clusters = {}
    for i in range(n_threads):
        group_distances = {}
        for group, distance in return_dict[i].items():
            group_distances[name_cluster[group]] = distance
        dictionary_clusters.update(group_distances)

    return dictionary_clusters


def get_score_pickle(results, embeddings, baseline=True, remove_outliers=True,
                     score='davies', sample_group=None, sample_size=0.2,
                     metric='euclidean', norm=False):
    '''
        Computes the given score (Davies-Bouldin, Calinski and Harabasz, Silhouette
                                  Coefficient).

        results (dict): clusterings results (cluster_id -> list of items).
        embeddings (dict): item embeddings/vectors (item_id -> item vector).
        baseline (bool): if the baseline was used.
        remove_outliers (bool): if the outliers should be removed for evaluation.
        score (string): score to compute.
                        'davies' -> Davies-Bouldin
                        'calinski' -> Calinski and Harabasz
                        'silhouette' -> Silhouette Coefficient
        sample_group (float): The size of the sample for each group when computing the
                              Silhouette Coefficient on a random subset of the data.
        sample_size (float): The size of the sample to use when computing the
                             Silhouette Coefficient on a random subset of the data.
        metric (string): The metric to use when calculating distance between instances
                         in a feature array.
                         'euclidean' -> Euclidean distance
                         'cosine' -> Cosine distance
        norm (bool): if the item embeddings/vectors should be normalized.
    '''

    random.seed(a=999)

    X = []
    labels = []

    if baseline:
        for group, items in results.items():
            if remove_outliers and (group[-2:] == '-1' or '_' not in group):
                continue
            if sample_group != None:
                items = random.sample(items, int(len(items)*sample_group))
            for item in items:
                if isinstance(list(embeddings.keys())[0], int):
                    X.append(embeddings[item])
                else:
                    X.append(embeddings[str(item)])
                labels.append(group)
    else:
        for group, items in results.items():
            if int(group) == -1:
                continue
            if sample_group != None:
                items = random.sample(items, len(items)*sample_group)
            for item in items:
                if isinstance(list(embeddings.keys())[0], int):
                    X.append(embeddings[item])
                else:
                    X.append(embeddings[str(item)])
                labels.append(group)

    # Normalize items embeddings
    if norm:
        X = normalize(X)

    if score == 'davies':
        return davies_bouldin_score(X, labels)
    elif score == 'calinski':
        return calinski_harabasz_score(X, labels)
    elif score == 'silhouette':
        sample = int(sample_size*len(X)) if sample_size != None else None
        return silhouette_score(X, labels, sample_size=sample, random_state=999,
                                metric=metric)


def get_score_baseline(results, remove_outliers=True, score='davies',
                       sample_group=None, sample_size=None,
                       metric='euclidean', norm=False):
    '''
        Computes the given score (Davies-Bouldin, Calinski and Harabasz, Silhouette
        Coefficient) for each baseline group.

        results (dataframe): clusterings results.
        remove_outliers (bool): if the outliers should removed for evaluation.
        score (string): score to compute.
                        'davies' -> Davies-Bouldin
                        'calinski' -> Calinski and Harabasz
                        'silhouette' -> Silhouette Coefficient
        sample_group (float): The size of the sample for each group when computing the
                              Silhouette Coefficient on a random subset of the data.
        sample_size (float): The size of the sample to use when computing the
                             Silhouette Coefficient on a random subset of the data.
        metric (string): The metric to use when calculating distance between instances
                         in a feature array.
                         'euclidean' -> Euclidean distance
                         'cosine' -> Cosine distance
        norm (bool): if the item embeddings/vectors should normalized.
    '''

    random.seed(a=999)

    if remove_outliers:
        results = results[results.id_cluster != -1]

    first_token_results = collections.defaultdict(list)
    first_token_embeddings = collections.defaultdict(list)
    embeddings_df = results.drop(['primeiro_token', 'id_cluster',
                                  'desc_usada_no_agrupamento', 'desc_original'])
    item_vec = embeddings_df.set_index('id_item').to_dict('index')

    i = 0
    for first_token, id_item, id_cluster in results[['primeiro_token', 'id_item', \
                                                    'id_cluster']].values.tolist():
        group = first_token + '_' + str(id_cluster)
        first_toke_results[first_token].append(int(id_cluster))
        first_token_embeddings[first_token].append(item_vec[id_item])

    scores_results = []

    for group, labels in first_token_results.items():
        X = first_token_embeddings[group]
        if norm:
            X = normalize(X)
        if score == 'davies':
            r = davies_bouldin_score(X, labels)
        elif score == 'calinski':
            r = calinski_harabasz_score(X, labels)
        elif score == 'silhouette':
            sample = int(sample_size*len(X)) if sample_size != None else None
            r = silhouette_score(X, labels, sample_size=sample, random_state=999,
                                    metric=metric)
        scores_results.append(r)

    return scores_results


def get_score_baseline_pickle(results, embeddings, remove_outliers=True,
                     score='davies', sample_group=None, sample_size=None,
                     metric='euclidean', norm=False):
    '''
        Computes the given score (Davies-Bouldin, Calinski and Harabasz, Silhouette
        Coefficient) for each baseline group.

        results (dict): clusterings results (cluster id -> list of items).
        embeddings (dict): item embeddings/vectors (item id -> item vector).
        remove_outliers (bool): if the outliers should be removed for evaluation.
        score (string): score to compute.
                        'davies' -> Davies-Bouldin
                        'calinski' -> Calinski and Harabasz
                        'silhouette' -> Silhouette Coefficient
        sample_group (float): The size of the sample for each group when computing the
                              Silhouette Coefficient on a random subset of the data.
        sample_size (float): The size of the sample to use when computing the
                             Silhouette Coefficient on a random subset of the data.
        metric (string): The metric to use when calculating distance between instances
                         in a feature array.
                         'euclidean' -> Euclidean distance
                         'cosine' -> Cosine distance
        norm (bool): if the item embeddings/vectors should be normalized.
    '''

    random.seed(a=999)

    first_token_results = collections.defaultdict(list)
    first_token_embeddings = collections.defaultdict(list)

    for group, items in results.items():
        if remove_outliers and (group[-2:] == '-1' or '_' not in group):
            continue
        if sample_group != None:
            items = random.sample(items, int(len(items)*sample_group))

        cluster = group.split('_')
        first_token = cluster[0]
        id_cluster = int(cluster[1])

        X = []
        labels = []
        for item in items:
            if isinstance(list(embeddings.keys())[0], int):
                X.append(embeddings[item])
            else:
                X.append(embeddings[str(item)])
            labels.append(id_cluster)

        first_token_results[first_token] += labels
        for item_vec in X:
            first_token_embeddings[first_token].append(item_vec)

    scores_results = []

    for group, labels in first_token_results.items():
        X = first_token_embeddings[group]
        # Normalize item embeddings
        if norm:
            X = normalize(X)
        if score == 'davies':
            r = davies_bouldin_score(X, labels)
        elif score == 'calinski':
            r = calinski_harabasz_score(X, labels)
        elif score == 'silhouette':
            sample = int(sample_size*len(X)) if sample_size != None else None
            r = silhouette_score(X, labels, sample_size=sample, random_state=999,
                                    metric=metric)
        scores_results.append(r)

    return scores_results
