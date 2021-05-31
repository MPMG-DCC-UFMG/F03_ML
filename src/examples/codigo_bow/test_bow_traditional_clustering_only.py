#Importing common libraries
import pandas as pd
import numpy as np
import collections
import copy
import random
import matplotlib.pyplot as plt

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

#Importing the HDBSCAN stand-alone method:
import hdbscan

#Importing the multiprocessing library:
import multiprocessing

#Importing the libraries to save the final resutls and making it possible to load them:
import nltk
import pickle


#Get the list of words (medicines and nouns) from the list of descriptions
def get_list_of_words(itemlist, medicines, canonical_form, word_class):
    list_words = list()
    list_of_items = itemlist.items_list
    count = 1
    total_len = len(list_of_items)
    for item in list_of_items:
        count = count + 1
        words = item.get_item_dict()['palavras']
        for p in words:
            if((p not in list_words)): 
                if ((p in medicines) or ((p in word_class) and (word_class[p] == 'N'))):
                    list_words.append(p)
                
    list_words.sort()
    
    return list_words

#Define a zero matrix based on the list of items 
#Only medicines and nouns are accepted as tokens for now
def define_zero_matrix(itemlist, medicines, canonical_form, word_class):
    list_words = get_list_of_words(itemlist, medicines, canonical_form, word_class)
    rows = len(itemlist.items_list)
    columns = len(list_words)
    matrix_bow = np.zeros((rows, columns), dtype='uint8')
    print('rows: ' + str(rows))
    print('columns: ' + str(columns))

    
    
    return matrix_bow, list_words, rows, columns

# Define the bag-of-words matrix.
def define_description_bow(itemlist, medicines, canonical_form, word_class):
    matrix_list = define_zero_matrix(itemlist, medicines, canonical_form, word_class)
    zeros = matrix_list[0]
    list_words = matrix_list[1]
    rows = matrix_list[2]
    columns = matrix_list[3]   
    i = 0
    list_of_items = itemlist.items_list

    for item in list_of_items:

        words =  item.get_item_dict()['palavras']          
        for w in words:
            if(w in list_words):
                k = list_words.index(w)
                zeros[i, k]  = 1
        i = i + 1
    
    print('bow defined!')
    return zeros, rows, columns

#It applies x-means on the bag of words.
def cluster_by_xmeans(bow, number_of_descriptions):
    cluster_size_limit = round(number_of_descriptions/50)
    xmeans_instance = xmeans(bow, kmax=cluster_size_limit, ccore=False)
    xmeans_instance.process();
    clusters = xmeans_instance.get_clusters();
    
    return clusters

#It just transfors the sklearn output to the pyclustering output
#as they differ in terms of representation.
def transform_sklearn_to_pyclustering(output):
    output_dict = {}
    i = 0
    
    while i < len(output):     
        if(output[i] not in output_dict):
            aux_arr = []
            aux_arr.append(i)
            output_dict[output[i]] = aux_arr
        else:
            aux_arr = output_dict[output[i]]
            aux_arr.append(i)
            output_dict[output[i]] = aux_arr       
        
        i = i + 1
        
    output_arr = []
    negative_key = -1
    
    j = 0
    for key in output_dict:
        if(key == -1):
            #print('found it at ' + str(j))
            negative_key = j

        j = j + 1
        output_arr.append(output_dict[key])
    
    return output_arr, negative_key

#It applies hdbscan on the bag of words.
def cluster_by_hdbscan(bow, employed_metric, n_threads):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric=employed_metric, 
                                core_dist_n_jobs=n_threads, min_sample=1)
    cluster_labels = clusterer.fit_predict(bow)

    clusters = transform_sklearn_to_pyclustering(cluster_labels)

    return clusters


#It calls the specific method depending on 'cluster_option' parameter.
def general_clustering(bow, number_of_descriptions, 
                       cluster_option, n_threads):
    clusters = None
    
    #cluster_option = 0, it employs x_means with the Euclidean distance:
    if(cluster_option == 0):
        clusters = cluster_by_xmeans(bow, number_of_descriptions)
        return clusters, None
    #cluster_option = 1, it employs  hdbscan with the Euclidean distance (normalized by l2):
    elif(cluster_option == 1):
        clusters = cluster_by_hdbscan(bow, 'l2', n_threads)
        return clusters[0], clusters[1]
    #cluster_option = 1, it employs hdbscan with the Hamming distance:
    elif(cluster_option == 2):
        clusters = cluster_by_hdbscan(bow, 'hamming', n_threads)
        return clusters[0], clusters[1]
    #otherwise,  it employs x_means with the Euclidean distance:
    else:
        clusters = cluster_by_xmeans(bow, number_of_descriptions)
        return clusters, None


#It clusters using X-Means and HDBScan with specific characteristics.
def cluster_on_bow(itemlist, medicines, canonical_form, 
                   word_class, cluster_option, n_threads):
    #Bag of words:
    bow = define_description_bow(itemlist, medicines, 
                                 canonical_form, word_class)
    print('bow: \n' + str(bow[0]))
    number_of_descriptions = len(itemlist.items_list)
    
    #Applies the clustering method on the bag of words
    print('Start to train the algorithm!')
    clusters_bow_result = general_clustering(bow[0], number_of_descriptions, 
                                             cluster_option, n_threads)
    clusters_bow = clusters_bow_result[0]
    negative_index = clusters_bow_result[1]
    print('clustering completed!')
    
    return clusters_bow, negative_index




def main():
    print('Starting...')

    from datetime import datetime
    now = datetime.now()
    print("Current Time =" + str(now))

    #Parameters of the code: saving dictionary directory (through '-s')
    #                        cluster option (algorithm and distance, through '-o')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store', dest='saving_dict')
    parser.add_argument('-o', action='store', dest='cluster_option')  
    parser.add_argument('-n', action='store', dest='n_threads')    
    args_results = parser.parse_args()
    print(str(args_results))

    saving_dict = args_results.saving_dict
    cluster_option = args_results.cluster_option
    n_threads = args_results.n_threads
    print('Read parameters') 

    #It loads the medical terms (medicines, drugs, etc):
    medicines = get_tokens_set('../dados/palavras/medications.txt')
    #It loads the canonical forms and their classes
    canonical_form, word_class = get_canonical_words()

    print('Read canonical terms') 
    
    #It gets the descpitons processed:
    itemlist = ItemList()
    itemlist.load_items_from_file('../dados/items_preprocessed.zip')    
    #It gets the list of preprocessed descriptions:
            
    print('Read data preprocessed')    

    cluster_result = cluster_on_bow(itemlist, medicines, canonical_form, word_class, cluster_option, n_threads)

    print('cluster done!!')
    #It gets all the results of the processes by accessing the Return_dict of each process:
    dictionary_clusters = {}    

    key = 0
    employed_key = 0
    for group in cluster_result[0]:
        employed_key = key
        if(key == cluster_result[1]):
            print('######Negative index is: '+str(key))
            employed_key = -1
            
        dictionary_clusters[employed_key] = group
        key = key + 1
    
     
    #It saves the dictionary in a file, which is possible to reconstruct the final dictionary:
    a_file = open(saving_dict, "wb")
    pickle.dump(dictionary_clusters, a_file)
    a_file.close()    

    print('Saved results!')

    now = datetime.now()
    print("Current Time =" + str(now))



if __name__ == '__main__':    
    main()

