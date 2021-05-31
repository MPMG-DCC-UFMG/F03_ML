import pandas as pd
import numpy as np
import collections
import copy
import random
import matplotlib.pyplot as plt
from nlp.preprocessing import (
    clean_text,
    preprocess,
    tokenize,
    preprocess_document,
    tokenize_document,
    get_stopwords, 
    lemmatization_document,
    get_canonical_words)
from nlp.utils import (
    plot_histogram,
    get_completetext,
    plot_wordcloud,
    print_statistics,
    groups_frequency_sort)
from nlp.text_statistics import (
    count_tokens,
    unique_tokens
)
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
from utils.read_files import (
    get_items)
from item.item_list import (
    ItemList,
    Item
)
from item.spellcheckeropt import SpellcheckerOpt
from item.utils import get_tokens_set
from textpp_ptbr.preprocessing import TextPreProcessing as tpp
from gensim.parsing.preprocessing import (
    strip_multiple_whitespaces,
    strip_non_alphanum,
    strip_punctuation2,
    strip_short)

#Import xmeans through pyclustering library:
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer;
from pyclustering.cluster.xmeans import xmeans


import multiprocessing

import nltk
import pickle

def get_list_of_words(group_desc, itemlist, medicines, canonical_form, word_class):
    list_words = list()
   
    for desc_id in group_desc:
        words = itemlist.items_list[desc_id].get_item_dict()['palavras']
        for p in words:
            if((p not in list_words)): 
                if ((p in medicines) or ((p in word_class) and (word_class[p] == 'N'))):
                    list_words.append(p)
                
    list_words.sort()
    
    return list_words


def define_zero_matrix(group_desc, itemlist, medicines, canonical_form, word_class):
    list_words = get_list_of_words(group_desc, itemlist, medicines, canonical_form, word_class)
    rows = len(group_desc)
    columns = len(list_words)
    matrix_bow = np.zeros((rows, columns))
    
    return matrix_bow, list_words


def define_description_bow(group_desc, itemlist, medicines, canonical_form, word_class):
    matrix_list = define_zero_matrix(group_desc, itemlist, medicines, canonical_form, word_class)
    zeros = matrix_list[0]
    list_words = matrix_list[1]    
    i = 0
    for desc_id in group_desc:
        words = itemlist.items_list[desc_id].get_item_dict()['palavras']           
        for w in words:
            if(w in list_words):
                k = list_words.index(w)
                zeros[i, k]  = 1.0
        i = i + 1
    return zeros


def cluster_by_xmeans(bow, number_of_descriptions):
    cluster_size_limit = round(number_of_descriptions/30)
    xmeans_instance = xmeans(bow, kmax=cluster_size_limit, ccore=False)
    print(xmeans_instance)
    xmeans_instance.process();
    clusters = xmeans_instance.get_clusters();
    
    return clusters


def translate_id_to_descriptions(ids, descriptions_ids):
    arr = []
    
    for i in ids:
        arr.append(descriptions_ids[i])
    return arr



def cluster_on_first_token_groups_bow(itemlist, it_thread, lower, upper, medicines, canonical_form, word_class, Return_dict):
    # It emplys the first token approach to group the descriptions:
    first_token_groups = itemlist.get_first_token_groups()
    # It creates a list of the the keys of these groups:
    groups = list(first_token_groups.keys())
    # It gets the values of each group (i.e., the id of the descriptions into that group):
    group_descriptions = list(first_token_groups.values())
    # It defines the dictionary that will have the clustering with first token
    # together with x-means considering a bag-of-words of the descriptions 
    # grouped by the first token approach:
    first_token_plus_bow_xmeans = {}
    # Iterator of the first token groups:
    ft_it = lower
    sum_cols = 0
    sum_rows = 0
    max_cols = -1
    max_rows = -1
    min_cols = 92233720368547
    min_rows = 92233720368547
    iterat = 0

    while ft_it <= upper:
        if(len(group_descriptions[ft_it]) >= 30):
            #print(str(it_thread) + ': ' + str(ft_it) + '/' + str(upper))
            # Bag of words for the group 0:
            bow = define_description_bow(group_descriptions[ft_it], itemlist, medicines, canonical_form, word_class)
            bow_shape = bow.shape
            
            row = bow_shape[0]
            col = bow_shape[1]
            
            sum_cols = sum_cols + col
            sum_rows = sum_rows + row

            if(row < min_rows):
                min_rows = row

            if(row > max_rows):
                max_rows = row

            if(col < min_cols):
                min_cols = col

            if(col > max_cols):
                max_rows = col
            
            iterat = iterat + 1

        ft_it = ft_it + 1

    avg_rows = sum_rows/iterat
    avg_cols = sum_cols/iterat

    print(str(lower) + '; ' + str(upper) + '; ' + str(min_rows) + '; ' + str(max_rows) + '; ' + str(avg_rows) + '; ' + str(min_cols) + '; ' + str(max_cols) + '; ' + str(avg_cols))



        
def get_ranges(group_len, n_threads):
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
        
    upper[n_threads - 1] = upper[n_threads - 1] - 6 
    return lower, upper


def main():
    manager = multiprocessing.Manager()
    Return_dict = manager.dict()
    jobs = []
    n_threads = 19

    medicines = get_tokens_set('../dados/palavras/medications.txt')
    canonical_form, word_class = get_canonical_words()
    
    # It gets the descpitons processed:
    itemlist = ItemList()
    itemlist.load_items_from_file('../dados/items_preprocessed.zip')    
    #It gets the list of preprocessed descriptions:
            
    #print('Read data preprocessed')    
    # It gets the first tokens of each description and groups
    # based on this approach:
    first_token_groups = itemlist.get_first_token_groups()
    group_len = len(first_token_groups)
    
    # It defines the ranges (of the groups) the threads will work on:
    thread_ranges = get_ranges(group_len, n_threads)
    #print('Read ranges')
    #print(thread_ranges) 

    i = 0
    
    for tr in thread_ranges:
        cluster_on_first_token_groups_bow(itemlist, i, thread_ranges[0][i], thread_ranges[1][i], medicines, canonical_form, word_class, Return_dict)
        i = i + 1    



if __name__ == "__main__":
    main()

