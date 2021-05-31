import pandas as pd
import numpy as np
import collections
import copy
import random
import matplotlib.pyplot as plt
from datetime import datetime
#Importing the multiprocessing library:
import multiprocessing
#Importing the libraries to save the final resutls and making it possible to load them:
import nltk
import pickle

from sklearn import preprocessing

from item.clustering.item_representation import (
    get_group_embeddings_matrix,
    get_group_embeddings_from_dict,
    save_items_embeddings,
    load_items_embeddings,
    normalize,
    get_items_embeddings
)



#Importing text preprocessing methods:
from nlp.preprocessing import (
    clean_text,
    preprocess,
    tokenize,
    preprocess_document,
    tokenize_document,
    get_stopwords, 
    lemmatization_document,
    get_canonical_words)
from textpp_ptbr.preprocessing import TextPreProcessing as tpp
from gensim.parsing.preprocessing import (
    strip_multiple_whitespaces,
    strip_non_alphanum,
    strip_punctuation2,
    strip_short)

#Importing libraries to check spelling:
from item.spellcheckeropt import SpellcheckerOpt
from item.utils import get_tokens_set


#Importing text analysis:
from nlp.utils import (
    plot_histogram,
    get_completetext,
    plot_wordcloud,
    print_statistics,
    groups_frequency_sort)

#Importing text statistics:
from nlp.text_statistics import (
    count_tokens,
    unique_tokens
)

#Importing baseline approaches for clustering:
from nlp.grouping import (
    get_groups,
    get_groups_size,
    get_unigram_groups,
    get_two_tokens_groups,
    get_first_token_groups,
    get_bigram_groups,
    get_first_two_groups,
    groups_frequency_sort
)

#Importing the stucture of the descriptions:
from utils.read_files import (
    get_items)
from item.item_list import (
    ItemList,
    Item
)

#Importing xmeans through pyclustering library:
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer;
from pyclustering.cluster.xmeans import xmeans

#Importing clustering evaluation measures and preprocessing approaches:
from sklearn.preprocessing import LabelEncoder


from item.clustering.utils import(
    get_items_vec
)


from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score
)

from item.clustering.evaluate import(
    get_score,
    evaluate_results,
    get_intraclusters_distances,
    get_items_distances
)

#Importing the HDBSCAN stand-alone method:
import hdbscan
#Importing UMAP dimensionality reduction method:
import umap





def get_avg_score_baseline(results, remove_outliers=True, score='davies', sample_group=None, sample_size=0.2, metric='euclidean', norm=True):

    #Group by the cell type
    results_by_ft = results.groupby('primeiro_token')
    sum_metric = 0.0
    count = 0
    for (ft, ft_cluster_df) in results_by_ft:
        shape_df = ft_cluster_df.shape

        if(shape_df[0] > 30):
            count += 1
            if remove_outliers:
                ft_cluster_df = ft_cluster_df[ft_cluster_df.id_cluster != -1]

            X = get_items_vec(ft_cluster_df)
            labels = list(ft_cluster_df['id_cluster'])
            # Normalize items embeddings
            if norm:
                X = normalize(X)

            mylist = list(set(labels))
            if(len(mylist) >= 2):                
                if score == 'davies':
                    sum_metric += davies_bouldin_score(X, labels)
                elif score == 'calinski':
                    sum_metric += calinski_harabasz_score(X, labels)
                elif score == 'silhouette':
                    sample = int(sample_size*len(X)) if sample_size != None else None
                    sum_metric += silhouette_score(X, labels, sample_size=sample, random_state=999, metric=metric)

    return sum_metric/count



#Main method for this code.
def main():
    #It measures the start time.
    now = datetime.now()
    print('Start Time = ' + str(now))
    #Loading the csv file in a dataframe:
    df = pd.read_csv('15_out_rep_without_outliers.pkl.csv', sep = ';', header=0, dtype='object')
    
    dbs = get_avg_score_baseline(df, score='davies')
    print('davies_bouldin_score: ' + str(dbs))
    chs = get_avg_score_baseline(df, score='calinski')
    print('calinski: ' + str(chs))
    sse = get_avg_score_baseline(df, score='silhouette', metric='euclidean')
    print('silhouette_score 20% - euclidean: ' + str(sse))
    ssc = get_avg_score_baseline(df, score='silhouette', metric='cosine')
    print('silhouette_score 20% - cosine: ' + str(ssc))

    #It measures the end time.
    now = datetime.now()
    print('\nEnd Time = ' + str(now))
    print('\n \n')  


if __name__ == '__main__':
    main()
