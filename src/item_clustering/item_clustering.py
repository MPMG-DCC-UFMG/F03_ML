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
from nlp.word_embeddings import (
    load_word_embeddings
)
from nlp.pos_tagging import (
    get_tokens_tags
)
from item.item_list import (
    ItemList,
    Item
)
from item.clustering.item_representation import *
from item.clustering.utils import *
from item.clustering.clustering import run_baseline_clustering
from nlp.preprocessing import PreprocessingText
from item.clustering.item_representation import get_item_vec
from nlp.preprocess_units import group_dsc_unidade_medida
from .config import Config

# clustering evaluation
from item.clustering.evaluate import (
    get_score_pickle,
    evaluate_results_pickle,
    evaluate_results,
    number_of_outliers_dict,
    get_score_baseline_pickle
)


class ItemClustering(object):

    def __init__(self, config=None):

        if config is not None:
            self.config = Config(**config)
        else:
            self.config = Config()

        self.preprocessing = PreprocessingText()
        self.itemlist = None        # items description

        # read word embeddings from file and store them in a map
        # word embeddings used to build items' representation
        self.word_embeddings = None

        # Get the tags of tokens descriptions
        # pos-tags used to build items' representation
        self.word_class = get_tokens_tags()

        # artifacts
        self.clusters = None
        self.outliers = None
        self.items_vec = None
        self.clustering_model = None
        self.reducer_model = None
        self.predict = None


    def save_model(self):

        self.config.save_config(self.config.artifacts_path)
        save_clustering_results_pickle(self.clusters, self.outliers, self.config.artifacts_path)
        save_models_pickle(self.clustering_model, self.reducer_model, self.config.artifacts_path)
        save_items_embeddings(self.items_vec, self.config.artifacts_path + 'items_vec.json')


    def load_model(self, path):

        self.config = Config()
        self.config.load_config(path)

        self.word_embeddings = load_word_embeddings(self.config.word_embeddings_path)
        self.clustering_model, self.reducer_model = load_models_pickle(self.config.artifacts_path)
        self.clusters, self.outliers, prices = load_clustering_results_pickle(self.config.artifacts_path)
        self.items_vec = load_items_embeddings(self.config.artifacts_path + 'items_vec.json')


    def preprocess_items(self, items, table_name, save_dataframe=True):

        # Preprocessing items descriptions
        items_descriptions = self.preprocessing.preprocess_items(items, n_process=10)

        itemlist = ItemList()
        itemlist.structure_items(items_descriptions)

        if save_dataframe:
            items_df = itemlist.to_dataframe()
            group_dsc_unidade_medida(items_df)
            itemlist.save_items_in_dataframe(self.config.artifacts_path + table_name,
                                             items_df)
            del items_df

        del itemlist


    def get_input(self, input_table, dataframe=True, password=None):

        # It gets the descriptions processed:

        self.itemlist = ItemList()

        if dataframe:
            self.itemlist.load_items_from_file(input_table)
        else:
            self.itemlist.load_items_from_hive_table(input_table, password)

        return self.itemlist


    def fit(self, items):

        self.preprocess_items(items, 'f03_items.csv.zip')
        del self.preprocessing

        self.get_input(self.config.artifacts_path + 'f03_items.csv.zip')

        if self.word_embeddings is None:
            print(time.asctime(), "Loading word embeddings")
            self.word_embeddings = load_word_embeddings(self.config.word_embeddings_path)

        clusters, outliers, items_vec, clustering_model, \
        reducer_model = run_baseline_clustering(self.itemlist,
                                                self.word_embeddings,
                                                self.word_class,
                                                algorithm=self.config.algorithm,
                                                categories=self.config.categories,
                                                embedding_type=self.config.tags,
                                                operation=self.config.operation,
                                                n_process=self.config.n_process)

        self.clusters = clusters
        self.outliers = outliers
        self.items_vec = items_vec
        self.clustering_model = clustering_model
        self.reducer_model = reducer_model


    def evaluate(self):

        self.n_groups = len(self.clusters)
        outliers_items, outliers_groups, total = number_of_outliers_dict(self.clusters,
                                                                        self.outliers,
                                                                        baseline=True,
                                                                        total_cov=False)
        self.perc_outliers = 100*(outliers_items/total)

        outliers_items, outliers_groups, total = number_of_outliers_dict(self.clusters,
                                                                         self.outliers,
                                                                         baseline=True,
                                                                         total_cov=True)
        self.perc_excluded = 100*(outliers_items/total)

        self.avg_calinski = get_score_baseline_pickle(self.clusters, self.items_vec,
                                                     score='calinski', sample_size=None,
                                                     norm=False)
        self.avg_calinski = np.mean(self.avg_calinski)

        self.avg_davies = get_score_baseline_pickle(self.clusters, self.items_vec,
                                                score='davies', sample_size=None,
                                                norm=False)
        self.avg_davies = np.mean(self.avg_davies)

        self.avg_silhouette_euclidean = get_score_baseline_pickle(self.clusters,
                                                             self.items_vec,
                                                             score='silhouette',
                                                             metric='euclidean',
                                                             sample_size=None,
                                                             norm=False)
        self.avg_silhouette_euclidean = np.mean(self.avg_silhouette_euclidean)

        self.avg_silhouette_cosine = get_score_baseline_pickle(self.clusters,
                                                          self.items_vec,
                                                          score='silhouette',
                                                          metric='cosine',
                                                          sample_size=None,
                                                          norm=False)
        self.avg_silhouette_cosine = np.mean(self.avg_silhouette_cosine)

        metrics = {}
        metrics['n_groups'] = self.n_groups
        metrics['outlier'] = self.perc_outliers
        metrics['excluded'] = self.perc_excluded
        metrics['avg_calinski'] = self.avg_calinski
        metrics['avg_davies-bouldin'] = self.avg_davies
        metrics['avg_silhouette-euclidean'] = self.avg_silhouette_euclidean
        metrics['avg_silhouette-cosine'] = self.avg_silhouette_cosine

        return metrics


    def predict(self, items):

        if self.clustering_model is None:
            return None

        self.preprocess_items(items, 'f03_items_test.csv.zip')
        items = self.get_input(self.config.artifacts_path + 'f03_items_test.csv.zip')

        results = predict_items_clusters(items, self.word_embeddings,
                                         self.word_class, self.reducer_model,
                                         self.clustering_model,
                                         categories=self.config.categories,
                                         embedding_type=self.config.tags,
                                         operation=self.config.operation,
                                         n_process=self.config.n_process)
        self.predict = results

        return results


    def predict_cluster(self, item):

        if self.clustering_model is None:
            return None

        # 1) TEXT CLEANING
        # Preprocessing
        doc = self.preprocessing.preprocess_document(item)

        # Categorization
        item = Item()
        itemslist = ItemList()
        item.extract_entities(doc, None, None, None, None, description, None, None,
                             itemslist.set_unit_metrics, itemslist.set_colors,
                             itemslist.set_materials, itemslist.set_sizes,
                             itemslist.set_quantities, itemslist.set_qualifiers,
                             itemslist.set_numbers)

        # 2) TEXT REPRESENTATION
        embedding_size = len(list(self.word_embeddings.values())[0])
        item_emb = get_item_vec(item, self.word_embeddings, self.word_class,
                                categories=self.config.categories,
                                embedding_type=self.config.tags,
                                operation=self.config.operation)

        # 3) CLUSTERING

        item_dict = item.get_item_dict()
        group = item_dict['palavras'][0]

        # It gets the reduced vector for the item
        item_emb_red = self.reducer_model[group].transform(item_emb)
        # It gets the item cluster
        cluster = approximate_predict(self.clustering_model[group], item_emb_red)
        cluster = group + '_' + str(cluster[0][0])

        return cluster
