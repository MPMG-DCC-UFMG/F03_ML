#importing common libraries
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
#Importing UMAP dimensionality reduction method:
import umap

#Importing the multiprocessing library:
import multiprocessing

#Importing the libraries to save the final resutls and making it possible to load them:
import nltk
import pickle

from datetime import datetime


#Get the list of words (medicines and nouns) from the list of descriptions
#in a specific group.
def get_list_of_tokens_med_n(group_desc, itemlist, medicines, canonical_form, word_class):
    list_words = list()
   
    for desc_id in group_desc:
        words = itemlist.items_list[desc_id].get_item_dict()['palavras']
        for p in words:
            if((p not in list_words)): 
                if ((p in medicines) or ((p in word_class) and (word_class[p] == 'N'))):
                    list_words.append(p)
                
    list_words.sort()
    
    return list_words

#Get the list of words (medicines, nouns, verbs, adjectives and numerals) from the list of descriptions
#in a specific group.
def get_list_of_tokens_med_n_v_a_num(group_desc, itemlist, medicines, canonical_form, word_class):
    list_words = list()
   
    for desc_id in group_desc:
        words = itemlist.items_list[desc_id].get_item_dict()['palavras']
        for p in words:
            if((p not in list_words)): 
                if ((p in medicines) or ((p in word_class) and ((word_class[p] == 'N') or (word_class[p] == 'A') or (word_class[p] == 'V') or (word_class[p].isnumeric())))):
                    list_words.append(p)
                
    list_words.sort()
    
    return list_words
    
    
#Get the list of words (medicines, nouns, verbs, adjectives and numerals) from the list of descriptions
#in a specific group.
def get_list_of_tokens_med_n_v_a_num_adv(group_desc, itemlist, medicines, canonical_form, word_class):
    list_words = list()
   
    for desc_id in group_desc:
        words = itemlist.items_list[desc_id].get_item_dict()['palavras']
        for p in words:
            if((p not in list_words)): 
                if ((p in medicines) or ((p in word_class) and ((word_class[p] == 'N') or (word_class[p] == 'A') or (word_class[p] == 'V') or (word_class[p] == 'DET+Num') or (word_class[p] == 'ADV') or (word_class[p].isnumeric())))):
                    list_words.append(p)
                
    list_words.sort()
    
    return list_words    
    
#Get all list of words from the list of descriptions
def get_all_tokens(group_desc, itemlist, medicines, canonical_form, word_class):
    list_words = list()
   
    for desc_id in group_desc:
        words = itemlist.items_list[desc_id].get_item_dict()['palavras']
        for p in words:
        	list_words.append(p)
                
    list_words.sort()
    
    return list_words

#Define a zero matrix based on the size of the number 
#of descriptions in that group (row) and the number of 
#words (only medicines and nouns) from all descriptions
#in that group.
def define_zero_matrix(group_desc, itemlist, medicines, canonical_form, word_class, tokens):
    if(tokens == 0):
        print('BOW composed only by nouns and medicines/medical terms of the descriptions!')		
        list_words = get_list_of_tokens_med_n(group_desc, itemlist, medicines, canonical_form, word_class)
    elif(tokens == 1):
        print('BOW composed by all words of the descriptions!')
        list_words = get_all_list_of_words(group_desc, itemlist, medicines, canonical_form, word_class)
    elif(tokens == 2):
        print('BOW composed only by nouns, adjectives, numerals, verbs and medicines/medical terms of the descriptions!')
        list_words = get_list_of_tokens_med_n_v_a_num(group_desc, itemlist, medicines, canonical_form, word_class)
    elif(tokens == 3):
        print('BOW composed only by nouns, adjectives, numerals, adverbs, verbs and medicines/medical terms of the descriptions!')
        list_words = get_list_of_tokens_med_n_v_a_num_adv(group_desc, itemlist, medicines, canonical_form, word_class)               
    else:
        print('Option not available! Using all words of the descriptions to make the bag...')
        list_words = get_all_tokens(group_desc, itemlist, medicines, canonical_form, word_class)


    rows = len(group_desc)
    columns = len(list_words)
    matrix_bow = np.zeros((rows, columns))
    print('Rows = ' + str(rows))    
    print('Columns = ' + str(columns))
    return matrix_bow, list_words, rows, columns

# Define the bag-of-words matrix.
def define_description_bow(group_desc, itemlist, medicines, canonical_form, word_class, tokens):
    matrix_list = define_zero_matrix(group_desc, itemlist, medicines, canonical_form, word_class, tokens)
    zeros = matrix_list[0]
    list_words = matrix_list[1]
    rows = matrix_list[2]
    columns = matrix_list[3]   
    i = 0
    for desc_id in group_desc:
        words = itemlist.items_list[desc_id].get_item_dict()['palavras']           
        for w in words:
            if(w in list_words):
                k = list_words.index(w)
                zeros[i, k]  = 1.0
        i = i + 1
    return zeros, rows, columns

#It applies x-means on the bag of words.
def cluster_by_xmeans(bow, number_of_descriptions):
    cluster_size_limit = round(number_of_descriptions/30)
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
def cluster_by_hdbscan(bow, employed_metric, groups_ft):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric=employed_metric, min_samples=1, cluster_selection_method='leaf')
    cluster_labels = clusterer.fit_predict(bow)

    clusters = transform_sklearn_to_pyclustering(cluster_labels)

    return clusters


#It calls the specific method depending on 'cluster_option' parameter.
#groups_ft is used when we have outliers, so we can separate these outliers
#considering the groups (from the First Token approach) they actually represent.
def general_clustering(bow, groups_ft, number_of_descriptions, cluster_option):
    clusters = None
    
    #cluster_option = 0, it employs x_means with the Euclidean distance:
    if(cluster_option == 0):
        print('Clustering by X-Means.')
        clusters = cluster_by_xmeans(bow, number_of_descriptions)
        return clusters, None
    #cluster_option = 1, it employs  hdbscan with the Euclidean distance (normalized by l2):
    elif(cluster_option == 1):
        print('Clustering by HDBSCAN with Euclidean distance normalized by l2.')
        clusters = cluster_by_hdbscan(bow, 'l2', groups_ft)
        return clusters[0], clusters[1]
    #cluster_option = 1, it employs hdbscan with the Hamming distance:
    elif(cluster_option == 2):
        print('Clustering by HDBSCAN with Hamming distance.')
        clusters = cluster_by_hdbscan(bow, 'hamming', groups_ft)
        return clusters[0], clusters[1]
    #otherwise,  it employs x_means with the Euclidean distance:
    else:
        print('Option not available. Clustering by HDBSCAN with Hamming distance.')
        clusters = cluster_by_hdbscan(bow, 'hamming', groups_ft)
        return clusters[0], clusters[1]




#Translate the generated ids of the clustering approach to actual description ids.
def translate_id_to_descriptions(ids, descriptions_ids):
    arr = []
    
    for i in ids:
        arr.append(descriptions_ids[i])
    return arr

#Dimensionality reduction to the bag-of-words.
def dimensionality_reduction(bow, dim_red):
    bow_reduced = bow[0]
    flag = False    
 
    if(dim_red == 1):
    	if(bow[1] > 0 and bow[2] > 0):
            try:
                bow_reduced = umap.UMAP(n_components=15).fit_transform(bow[0])
                print('UMAP employed')                
            except:
                print('#####Exception occurred')
                bow_reduced = bow[0]
                flag = True
                
    if(flag):
    	bow_reduced = bow[0]  
    	              	            
    rows, cols = bow_reduced.shape
        
    return bow_reduced, rows, cols
        


#It clusters again the groups generated by the first token approach. For now, this method only accepts X-Means and HDBScan with specific characteristics.
def cluster_on_first_token_groups_bow(first_token_groups, itemlist, it_thread, lower, upper, medicines, canonical_form, word_class, cluster_option, tokens, dim_red, Return_dict):

    print(it_thread)
    #It creates a list of the the keys of these groups:
    groups = list(first_token_groups.keys())
    #It gets the values of each group (i.e., the id of the descriptions into that group):
    group_descriptions = list(first_token_groups.values())
    #It defines the dictionary that will have the clustering with first token
    #together with traditional clustering methods considering a bag-of-words of the descriptions 
    #grouped by the first token approach:
    first_token_plus_bow_traditional_clustering = {}
    #Iterator of the first token groups:
    ft_it = lower
    start_it = lower
    while ft_it <= upper:
        print(str(it_thread) + ': ' + str(start_it) + '/' + str(ft_it) + '/' + str(upper))
        #It only considers to cluster again if the number of descritptions of that group has more than 30 descriptions
        if(len(group_descriptions[ft_it]) >= 30):
            
            #Bag of words for the group 0:
            bow_raw = define_description_bow(group_descriptions[ft_it], itemlist, medicines, canonical_form, word_class, tokens)
            bow_w_dr = dimensionality_reduction(bow_raw, dim_red)
            
            #It only applies the traditional clustering methods if the number of rows and columns of the bow are greater than zero:
            if(bow_w_dr[1] > 0 and bow_w_dr[2] > 0):        
                #It applies the clusters on the bow of the descriptions - group 0:
                try:
                    clusters_bow_result = general_clustering(bow_w_dr[0], groups[ft_it], len(group_descriptions[ft_it]), cluster_option)
                    clusters_bow = clusters_bow_result[0]
                    negative_index = clusters_bow_result[1]
                    it = 0
                    for c in clusters_bow:
                        #It translates ids from x-means to actual descriptions (new groups):
                        desc_ids = translate_id_to_descriptions(c, group_descriptions[ft_it])
                        #It defines the key of the map:
                        if(it != negative_index):
                            new_key = groups[ft_it] + '_' + str(it)
                        else:
                            new_key = groups[ft_it] + '_-1'
                        #It sets the maps:
                        first_token_plus_bow_traditional_clustering[new_key] = desc_ids                   
                        it = it + 1
                    
                    
                except:
                    print('@ERROR!')
                    first_token_plus_bow_traditional_clustering[groups[ft_it]] = group_descriptions[ft_it]
     
            else:
                first_token_plus_bow_traditional_clustering[groups[ft_it]] = group_descriptions[ft_it]
        else:
            first_token_plus_bow_traditional_clustering[groups[ft_it]] = group_descriptions[ft_it]
        ft_it = ft_it + 1

    Return_dict[it_thread] = first_token_plus_bow_traditional_clustering


#It gets the ranges of the clusters generated by the First Token approach
#This is done in order to the processes work on.
def get_ranges(group_len, n_threads):
    if(n_threads == 1):
        return 0, (group_len - 1)

    total_len = group_len
    num_threads = n_threads
    lower = []
    upper = []
    step = int(total_len/num_threads)

    for k in range(num_threads):
        lower.append(0)
        upper.append(0)

    lower[0] = 0
    upper[0] = step
  
    i = 1
    j = 0
    while (i < num_threads):    
        upper[i]  = upper[j] + step
        lower[i]  = upper[j] +  1
        if(i%2 != 0):
            upper[i] = upper[i] + 1
        
        i = i + 1
        j = j + 1
    
    #Please, check if the final cluster range ends with 18,034 clusters
    #(i.e., the number of clusters generated by First Token).
    #Depending of the number of processes, you may have to change this "-1"
    #for something else.
    upper[n_threads - 1] = upper[n_threads - 1] - 1 
    return lower, upper


def main(saving_dict, cluster_option, number_threads, tokens, dim_red):


    now = datetime.now()
    print("Start Time = " + str(now))
    
    manager = multiprocessing.Manager()
    Return_dict = manager.dict()
    jobs = []
    n_threads = number_threads

    #It loads the medical terms (medicines, drugs, etc):
    medicines = get_tokens_set('../dados/palavras/medications.txt')
    #It loads the canonical forms and their classes
    canonical_form, word_class = get_canonical_words()
    print("Read Canonical terms.")
        
    #It loads the items from the list:
    itemlist = ItemList()
    itemlist.load_items_from_file('../dados/items_preprocessed.zip')
                
    print('Read data preprocessed')    
    #It gets the first tokens of each description and groups
    #based on this approach:
    first_token_groups = itemlist.get_first_token_groups()
    group_len = len(first_token_groups)
    first_token_groups_new = {}

    #It shuffles the itens based on their keys:
    keys_ft = list(first_token_groups.keys())
    random.shuffle(keys_ft)
    random.shuffle(keys_ft)
    
    #It fills another dictionary with the shuffled keys:
    for k in keys_ft:
        first_token_groups_new[k] = first_token_groups[k]
    
   
    #It defines the ranges (of the groups) the processes will work on:
    thread_ranges = get_ranges(group_len, n_threads)
    print('Read ranges')
    print(thread_ranges) 
    
    #It creates the processes (balanced by shuffling the keys of the dictionary:
    for i in range(n_threads):
        p = multiprocessing.Process(target=cluster_on_first_token_groups_bow, args=(first_token_groups_new, itemlist, i, thread_ranges[0][i], thread_ranges[1][i], medicines, canonical_form, word_class, cluster_option, tokens, dim_red, Return_dict))
        jobs.append(p)
        p.start()
        
    #It joins the results
    for i in range(n_threads):   
        jobs[i].join()
        
    dictionary_clusters = {}   
    #It joins the results
    for k in Return_dict:
        print('key: '+str(k))
        dictionary_clusters.update(Return_dict[k])

     
    #It saves the dictionary in a file, which is possible to reconstruct the final dictionary:
    a_file = open(saving_dict, "wb")
    pickle.dump(dictionary_clusters, a_file)
    a_file.close()
    
    now = datetime.now()
    print("End Time = " + str(now))    

if __name__ == "__main__":
    import argparse
    #Parameters of the code: saving dictionary directory (through '-s')
    #                        cluster option (algorithm and distance, through '-o')
    #                        n_threads (through '-n')
    #                        considered_tokens (through '-t')
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store', dest='saving_dict')
    parser.add_argument('-o', action='store', dest='cluster_option')  
    parser.add_argument('-n', action='store', dest='n_threads')
    parser.add_argument('-t', action='store', dest='considered_tokens')
    parser.add_argument('-d', action='store', dest='dim_red')        
    args_results = parser.parse_args()
    #print(args_results)
    main(args_results.saving_dict, int(args_results.cluster_option), int(args_results.n_threads), int(args_results.considered_tokens), int(args_results.dim_red))
