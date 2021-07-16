import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '...'))

import numpy as np
import collections
import copy
import random
import pandas as pd
import pickle
import json
from utils.hive_access import (
    dataframe_to_hive_table,
    hive_table_to_dataframe
)


def get_clusters_items(clusters_items, outliers):
    '''
        It gets the complete clustering result (without outlier removal).

        results (dict): clusters (cluster id -> list of items).
        outliers (dict): clusters outliers (cluster id -> list of items).
    '''
    result = {}
    for group, items_list in clusters_items.items():
        result[group] = items_list + outliers[group]

    return result


def add_original_description(data):
    '''
        Add "original_dsc" column to dataframe.
    '''

    data['original_dsc'] = data['original_prep'].apply(lambda x: ' '.join(eval(x)))
    data.drop(columns=['original_prep'], inplace=True)

    return data


def add_group_id_column(data):
    '''
        Add "grupo_id" column to dataframe.
    '''

    data['grupo_id'] = list(range(len(data)))
    return data


def add_first_token_column(data):
    '''
        Add "primeiro_termo" column to dataframe.
    '''

    data['primeiro_termo'] = data['grupo'].str.split('_').str[0]
    return data


def add_outlier_column(data):
    '''
        Add "ruido" column to dataframe.
    '''

    data['ruido'] = data['grupo'].str.split('_').str[1]
    data['ruido'].fillna(1, inplace=True)
    data['ruido'].replace({'-1': 1}, inplace=True)

    return data


def get_items_dataframe(items_clusters_df, cluster_items_df, left_on='item_id',
                        right_on='item_id'):
    '''
        Get the items dataframe. The dataframe should have the following columns:
        ['item_id', 'seq_dim_licitacao', 'ruido', 'grupo', 'dsc_unidade_medida',
         'ano', 'descricao', 'descricao_original', 'areas', 'vlr_unitario', 'mes',
         'data', 'cidade', 'orgao'].

        items_clusters_df (dataframe): contains the items.
        cluster_items_df (dataframe): clusters dataframe.
    '''

    items_clusters_df = pd.merge(left=cluster_items_df, right=items_clusters_df,
                                 left_on=left_on, right_on=right_on)
    items_clusters_df = add_first_token_column(items_clusters_df)

    return items_clusters_df


def get_clusters_dataframe(cluster_items, outliers=None, baseline=True):
    '''
        Get the clusters dataframe. The dataframe should have the following columns:
        ['item_id', 'grupo', 'grupo_ruido', 'item_ruido'].

        cluster_items (dict): dictionary of groups (group_id -> list of item ids).
        outliers (dict): clusters outliers (cluster id -> list of items).
        baseline (bool): if the first token grouping was used to generete the
                         groups.
    '''

    lines = []
    for cluster, items in cluster_items.items():
        for item in items:
            if baseline and ('_' not in cluster or cluster[-2:] == '-1'):
                outlier = 1
            elif not baseline and cluster == '-1':
                outlier = 1
            else:
                outlier = 0
            line = (item, cluster, outlier, outlier)
            lines.append(line)

    if outliers is not None:
        for cluster, items in outliers.items():
            for item in items:
                if baseline and ('_' not in cluster or cluster[-2:] == '-1'):
                    outlier = 1
                elif not baseline and cluster == '-1':
                    outlier = 1
                else:
                    outlier = 0
                line = (item, cluster, outlier, 1)
                lines.append(line)

    columns = ('item_id', 'grupo', 'grupo_ruido', 'item_ruido')
    clusters_df = pd.DataFrame(lines, columns=columns)

    return clusters_df


def save_clustering_results(results, desc_original, clustering_descriptions,
                            file, items_vec=None, first_token=True):
    '''
        It saves the clustering results in a dataframe.

        results (dict): clusters (cluster_id -> list of items).
        desc_original (dict): original descriptions of the items (item_id -> desc).
        clustering_descriptions (dict): descriptions used for clustering
                                        (item_id -> desc).
        file (string): output file.
        items_vec (dict): items vectors (item_id -> vector).
        first_token (bool): if the baseline was used.
    '''

    if items_vec != None:
        embedding_size = len(list(items_vec.values())[0])
    else:
        embedding_size = 15

    tuples = []
    for cluster, items in results.items():
        for item_id in items:
            if first_token:
                group = cluster.split('_')
                token = group[0]
                if len(group) == 2:
                    id_cluster = int(group[1])
                else:
                    id_cluster = -1
                items_info = (group[0], item_id, id_cluster,
                              clustering_descriptions[item_id],
                              desc_original[item_id])
            else:
                items_info = ('-999', item_id, int(cluster),
                              clustering_descriptions[item_id],
                              desc_original[item])

            if items_vec != None and item_id in items_vec:
                items_info = items_info + tuple(items_vec[item_id])
            else:
                items_info = items_info + tuple([-1] * embedding_size)
            tuples.append(items_info)

    embedding_columns = []
    for i in range(embedding_size):
        embedding_columns.append('dim_' + str(i))

    columns = ('primeiro_token', 'id_item', 'id_cluster', 'desc_usada_no_agrupamento',
               'desc_original') + embedding_columns
    items_clusters_df = pd.DataFrame(tuples, columns=columns)
    items_clusters_df.to_pickle(file)

    return items_clusters_df


def load_clustering_results(file):
    '''
        It loads the clustering results in a dataframe.
    '''

    items_clusters_df = pd.read_pickle(file)
    return items_clusters_df


def save_clustering_results_hive_table(results, outliers, results_table,
                                       outliers_table, version):
    '''
        It saves the clustering results in Hive tables.

        results (dict): clusters (cluster id -> list of items).
        outliers (dict): clusters outliers (cluster id -> list of items).
        results_table (str): Hive table to save clusters results.
        outliers_table (str): Hive table to save clusters outliers.
        version (str): execution version.
        password (str): connection password.
    '''
    dicts = [results, outliers]
    tables = [results_table, outliers_table]
    for d, t in zip (dicts, tables):

        keys = d.keys()
        data = []
        for k in keys:
            values = d[k]
            if (values == None) or (len(values)==0):
                data.append([k, None])
            else:
                for v in values:
                    data.append([k, v])
        dataframe = pd.DataFrame(np.asarray(data), columns = ['cluster_id', 'item_id'])
        dataframe_to_hive_table(dataframe, t, version)


def load_clustering_results_hive_table(results_table, outliers_table, password):
    '''
        It loads the clustering results in dictionaries from Hive tables.

        results_table (str): Hive table to load clusters results.
        outliers_table (str): Hive table to load clusters outliers.
        password (str): connection password.
    '''
    results = {}
    outliers = {}
    for dict_return, table in zip([results, outliers], [results_table, outliers_table]):
        dataframe = hive_table_to_dataframe(table, password)
        clusters = dataframe['cluster_id'].unique()
        for c in clusters:
            filtered = dataframe[dataframe['cluster_id'] == c]
            filtered = filtered['item_id'].tolist()
            if ((len(filtered)==1) and (filtered[0] == None)):
                dict_return[c] = []
            else:
                dict_return[c] = filtered

    return results, outliers


def save_clustering_results_pickle(results, outliers, out_dir):
    '''
        It saves the clustering results in pickle files.

        results (dict): clusters (cluster id -> list of items).
        outliers (dict): clusters outliers (cluster id -> list of items).
        out_dir (str): folder which the results should be saved on.
    '''

    with open(out_dir + "results.pkl", "wb") as PFile:
        pickle.dump(results, PFile)
    PFile.close()

    with open(out_dir + "outliers.pkl", "wb") as PFile:
        pickle.dump(outliers, PFile)
    PFile.close()


def load_clustering_results_pickle(dir):
    '''
        It loads the clustering results in dictionaries.

        dir (str): folder which the results were saved on.
    '''

    with open(dir + "results.pkl", "rb") as PFile:
        results = pickle.load(PFile)
    PFile.close()

    with open(dir + "outliers.pkl", "rb") as PFile:
        outliers = pickle.load(PFile)
    PFile.close()

    return results, outliers


def save_models_pickle(clustering_model, reducer_model, out_dir):
    '''
        It saves the Clustering models and the Dimension Reduction models.

        clustering_model (dict): clustering model (cluster_id -> model).
        reducer_model (dict): dim. reduction model (cluster_id -> model).
        out_dir (str): folder which the models should be saved on.
    '''

    with open(out_dir + "clustering_model.pkl", "wb") as PFile:
        pickle.dump(clustering_model, PFile)
    PFile.close()

    with open(out_dir + "dimred_model.pkl", "wb") as PFile:
        pickle.dump(reducer_model, PFile)
    PFile.close()


def load_models_pickle(folder):
    '''
        It loads the Clustering models and the Dimension Reduction models.

        folder (str): folder which the results were saved on.
    '''

    with open(folder + "clustering_model.pkl", "rb") as PFile:
        clustering_model = pickle.load(PFile)
    PFile.close()

    with open(folder + "dimred_model.pkl", "rb") as PFile:
        reducer_model = pickle.load(PFile)
    PFile.close()

    return clustering_model, reducer_model


def get_items_vec(results):
    '''
        It gets the items embedding from a dataframe.

        results (DataFrame): clustering results.
    '''

    X = results[results.columns.difference(['primeiro_token', 'id_item', 'id_cluster',
        'desc_usada_no_agrupamento', 'desc_original'])].values.tolist()

    return X


def translate_id_to_descriptions(ids, descriptions_ids):
    '''
        It translates the clustering results (ids -> items ids).

        ids (list): clustering ids.
        descriptions_ids (int): items ids.
    '''

    arr = []

    for i in ids:
        arr.append(descriptions_ids[i])

    return arr


def get_tokens_set(file):
    '''
        It gets tokens from a file.
    '''

    tokens = open(file, 'r').readlines()
    tokens = set([token.replace('\n', '') for token in tokens])

    return tokens


def get_ranges(group_len, n_process):
    '''
        It gets the ranges of the clusters generated by the First Token grouping.
        This is done in order to the processes work on.

        group_len (int): number of first token's groups.
        n_process (int): number of process.
    '''

    if(n_process == 1):
        return [0], [(group_len - 1)]

    total_len = group_len
    num_process = n_process
    lower = []
    upper = []
    step = int(total_len/num_process)

    for k in range(num_process):
        lower.append(0)
        upper.append(0)

    lower[0] = 0
    upper[0] = step

    i = 1
    j = 0
    while (i < num_process):
        upper[i]  = upper[j] + step
        lower[i]  = upper[j] +  1
        if (i%2 != 0):
            upper[i] = upper[i] + 1

        i = i + 1
        j = j + 1

    upper[n_process - 1] = group_len - 1

    return lower, upper
