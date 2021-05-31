#Importing common libraries:
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
                if ((p in medicines) or ((p in word_class) and ((word_class[p] == 'N') or 
                    (word_class[p] == 'A') or (word_class[p] == 'V') or (word_class[p].isnumeric())))):
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
                if ((p in medicines) or ((p in word_class) and ((word_class[p] == 'N') or 
                    (word_class[p] == 'A') or (word_class[p] == 'V') or (word_class[p] == 'DET+Num') or 
                    (word_class[p] == 'ADV') or (word_class[p].isnumeric())))):
                    list_words.append(p)
                
    list_words.sort()
    
    return list_words



    
#Get all list of words in 'palavras' from the list of descriptions from the first token groups.
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
        #print('BOW composed only by nouns and medicines/medical terms of the descriptions!')        
        list_words = get_list_of_tokens_med_n(group_desc, itemlist, medicines, canonical_form, word_class)
    elif(tokens == 1):
        #print('BOW composed by all words of the descriptions!')
        list_words = get_all_list_of_words(group_desc, itemlist, medicines, canonical_form, word_class)
    elif(tokens == 2):
        #print('BOW composed only by nouns, adjectives, numerals, verbs and medicines/medical terms of the descriptions!')
        list_words = get_list_of_tokens_med_n_v_a_num(group_desc, itemlist, medicines, canonical_form, word_class)
    elif(tokens == 3):
        #print('BOW composed only by nouns, adjectives, numerals, adverbs, verbs and medicines/medical terms of the descriptions!')
        list_words = get_list_of_tokens_med_n_v_a_num_adv(group_desc, itemlist, medicines, canonical_form, word_class)               
    else:
        #print('Option not available! Using all words of the descriptions to make the bag...')
        list_words = get_all_tokens(group_desc, itemlist, medicines, canonical_form, word_class)


    rows = len(group_desc)
    columns = len(list_words)
    matrix_bow = np.zeros((rows, columns))
    #print('Rows = ' + str(rows))    
    #print('Columns = ' + str(columns))
    return matrix_bow, list_words, rows, columns




# Define the bag-of-words matrix.
def define_description_bow(group_desc, itemlist, medicines, canonical_form, word_class, tokens):
    matrix_list = define_zero_matrix(group_desc, itemlist, medicines, canonical_form, word_class, tokens)
    #initializes bow with the matrix of zeros:
    bow_matrix = matrix_list[0]
    list_words = matrix_list[1]
    rows = matrix_list[2]
    columns = matrix_list[3]
    preproc_descs = np.empty((rows, 1), dtype='object')
    original_descs = np.empty((rows, 1), dtype='object')
    ids_descs = np.empty((rows, 1), dtype='object')
    
    i = 0
    for desc_id in group_desc:

        #This is to return the original and the preprocessed description to build the file of results:    
        desc_original = itemlist.items_list[desc_id].get_item_dict()['original']
        desc_prep = itemlist.items_list[desc_id].get_item_dict()['original_prep']
        desc_prep_rep = str(desc_prep).replace('\'', '').replace('[', '').replace(']', '').replace(',', '')
        
        original_descs[i, 0] = desc_original
        preproc_descs[i, 0 ] = desc_prep_rep
        ids_descs[i, 0 ] = desc_id

        #For now, we only look at the array of 'palavras' to build the BOW:
        words = itemlist.items_list[desc_id].get_item_dict()['palavras']   
        for w in words:
            if(w in list_words):
                k = list_words.index(w)
                #it updates the bow matrix with elements of that group (defined by the first token)
                bow_matrix[i, k]  = 1.0
        i = i + 1       
    
    #Concatanate the original descriptions with the preprocesses ones.
    #and, also, adds the ids of the descriptions.
    result_descs = np.concatenate((original_descs, preproc_descs), axis=1)
    result_descs_w_ids = np.concatenate((ids_descs, result_descs), axis=1)
    
    return bow_matrix, rows, columns, result_descs_w_ids

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
def cluster_by_hdbscan(bow, employed_metric, groups_ft, 
    cluster_alg_min_samples, cluster_alg_select_method, cluster_alg_allow_single_cluster):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric=employed_metric, min_samples= cluster_alg_min_samples,
        cluster_selection_method=cluster_alg_select_method, allow_single_cluster=cluster_alg_allow_single_cluster)
    clusters_sklearn = clusterer.fit_predict(bow)

    clusters_pyclustering = transform_sklearn_to_pyclustering(clusters_sklearn)

    return clusters_pyclustering, clusters_sklearn
    

#Check if the metrics defined by the user is available for HDBSCAN. If not, it uses the 'l2'.
def check_hdbscan_metrics(metric):
    available_metrics = ['braycurtis','canberra','chebyshev','cityblock','dice','euclidean',
    'hamming','haversine','infinity','jaccard','kulsinski','l1','l2','mahalanobis','manhattan',
    'matching','minkowski','p','pyfunc','rogerstanimoto','russellrao','seuclidean','sokalmichener',
    'sokalsneath','wminkowski']

    employed_metric = metric

    #It hecks if the employed metric is available for HDBSCAN;
    #If it is not available, it employs l2
    if(employed_metric not in available_metrics):
        print('Measure not available. We defined the l2 measure for the HDBSCAN for you then.')
        employed_metric = 'l2'

    return employed_metric


#It calls the specific method depending on 'cluster_alg' parameter.
#groups_ft is used when we have outliers, so we can separate these outliers
#considering the groups (from the First Token approach) they actually represent.
def general_clustering(bow, groups_ft, number_of_descriptions, cluster_alg, cluster_alg_metric, 
    cluster_alg_min_samples, cluster_alg_select_method, cluster_alg_allow_single_cluster):
    clusters = None
    employed_metric = check_hdbscan_metrics(cluster_alg_metric)
    
    if(cluster_alg == 'xmeans'):
        #print('Clustering by X-Means.')
        clusters = cluster_by_xmeans(bow, number_of_descriptions)
        return clusters, None, None
    #cluster_alg = 1, it employs  HDBSCAN with the Euclidean distance (normalized by l2):
    elif(cluster_alg == 'hdbscan'):
        #print('Clustering by HDBSCAN with ' + employed_metric +  ' metric.')
        clusters = cluster_by_hdbscan(bow, employed_metric, groups_ft, 
            cluster_alg_min_samples, cluster_alg_select_method, cluster_alg_allow_single_cluster)
        return clusters[0][0], clusters[0][1], clusters[1]
    #otherwise,  it employs HDBSCAN with an Euclidean distance (normalized by l2):
    else:
        #print('Option not available. Employing HDBSCAN clustering algorithm with l2 metric.')
        clusters = cluster_by_hdbscan(bow, 'l2', groups_ft, None, 'eom', False)
        return clusters[0][0], clusters[0][1], clusters[1]




#Translate the generated ids of the clustering approach to actual description ids.
def translate_id_to_descriptions(ids, descriptions_ids):
    arr = []
    
    for i in ids:
        arr.append(descriptions_ids[i])
    return arr

#Check if the metrics defined by the user is available for UMAP. If not, it uses cosine.
def check_umap_metrics(metric):
    available_metrics = ['euclidean','manhattan','chebyshev','minkowski','canberra','braycurtis',
    'mahalanobis','wminkowski','seuclidean','cosine','correlation','haversine','hamming',
    'jaccard','dice','russelrao','kulsinski','ll_dirichlet','hellinger','rogerstanimoto',
    'sokalmichener','sokalsneath','yule']

    employed_metric = metric

    #It hecks if the employed metric is available for UMap;
    #If it is not available, it employs cosine
    if(employed_metric not in available_metrics):
        employed_metric = 'cosine'

    return employed_metric

#Dimensionality reduction to the bag-of-words.
def dimensionality_reduction(bow, dr_alg, dr_n_comp, dr_metric):
    bow_reduced = None
    flag = False    

    if(dr_alg == 'umap'):
        if(bow[1] > 0 and bow[2] > 0):
            try:
                employed_metric = check_umap_metrics(dr_metric)
                bow_reduced = umap.UMAP(n_components=dr_n_comp, metric=employed_metric, low_memory = True, random_state = 999).fit_transform(bow[0])
                #print('UMAP employed')                
            except:
                print('#####Exception occurred when we applied UMAP!')
                flag = True
    else:
        #print('No dimensionality reduction employed. Using the traditional bag of words.')
        bow_reduced = bow[0]


    #it uses the bow if there is an error in the execution of the dimensionality reduction method
    if((bow_reduced is None) or flag):
        bow_reduced = bow[0]        

    rows, cols =  bow_reduced.shape

        
    return bow_reduced, rows, cols
        


#It clusters again the groups generated by the first token approach. For now, this method only accepts X-Means and HDBScan with specific characteristics.
def cluster_on_first_token_groups_bow(first_token_groups, itemlist, it_thread, lower, upper, 
                                      medicines, canonical_form, word_class, tokens,
                                      cluster_alg, cluster_alg_metric, 
                                      cluster_alg_min_samples, cluster_alg_select_method, cluster_alg_allow_single_cluster,
                                      dr_alg, dr_n_comp, dr_metric, remove_outliers,
                                      Return_Dict, Return_Repres_Comp, 
                                      Return_Count_Not_Covered, Return_Count_Covered):

    #print(it_thread)
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
    bow_w_dr = None
    df_results = None
    count_not_covered = 0
    count_covered = 0

    print(str(it_thread) + ': ' + str(start_it) + '/' + str(ft_it) + '/' + str(upper))
    while ft_it <= upper:
        
        #It only considers to cluster again if the number of descritptions of that group has more than 30 descriptions
        if(len(group_descriptions[ft_it]) > 30):
            
            #Bag of words for the group 0:
            bow_raw = define_description_bow(group_descriptions[ft_it], itemlist, medicines, canonical_form, word_class, tokens)
            bow_w_dr = dimensionality_reduction(bow_raw, dr_alg, dr_n_comp, dr_metric)

            #It only applies the traditional clustering methods if the number of rows and columns of the bow are greater than zero:
            if(bow_w_dr[1] > 0 and bow_w_dr[2] > 0):
                #It applies the clusters on the bow of the descriptions:
                clusters_bow_result = general_clustering(bow_w_dr[0], groups[ft_it], len(group_descriptions[ft_it]), 
                    cluster_alg, cluster_alg_metric, cluster_alg_min_samples, cluster_alg_select_method, cluster_alg_allow_single_cluster)
                clusters_bow = clusters_bow_result[0]
                negative_index = clusters_bow_result[1]

                if(bow_w_dr[2]==dr_n_comp):
                    first_token = np.full((bow_w_dr[1], 1), groups[ft_it], dtype='object')
                    result_descs = np.concatenate((first_token, bow_raw[3]), axis=1)
                    result_descs_w_dr = np.concatenate((result_descs, bow_w_dr[0]), axis=1)
                    ft_plus_clusters = np.concatenate((first_token, np.c_[clusters_bow_result[2]]), axis=1)
                    ft_plus_clusters_merged = np.c_[["".join(i) for i in ft_plus_clusters[:,0:].astype(str)]]                    
                    result_descs_w_clusters = np.concatenate((result_descs_w_dr, ft_plus_clusters_merged), axis=1)
                    #print(result_descs_w_clusters)
                    if(df_results is None):
                        df_results = result_descs_w_clusters
                    else:
                        df_results = np.concatenate((df_results, result_descs_w_clusters), axis=0)

                it = 0
                desc_ids_outlier = []
                for c in clusters_bow:
                    #It translates ids from traditional clustering to actual descriptions (new groups):
                    desc_ids = translate_id_to_descriptions(c, group_descriptions[ft_it])
                    #It defines the key of the map:
                    if(it != negative_index):
                        new_key = groups[ft_it] + '_' + str(it)
                    else:
                        #if you choose to remove the outliers, every outlier is set as '-1'
                        if(remove_outliers):
                            new_key = -1
                        else:
                            new_key = groups[ft_it] + '_-1'
                            
                    #It sets the maps:
                    if(new_key == -1):
                        #only merges all the outliers in one unique array and updates the dictionary
                        desc_ids_outlier.extend(desc_ids)
                        first_token_plus_bow_traditional_clustering[new_key] = desc_ids_outlier
                    else:
                        first_token_plus_bow_traditional_clustering[new_key] = desc_ids                   
                    it = it + 1

                count_covered = count_covered + len(group_descriptions[ft_it])    
            else:
                #It returns the first token groups if it is not possible to apply the traditional clustering.
                first_token_plus_bow_traditional_clustering[groups[ft_it]] = group_descriptions[ft_it]
                count_not_covered = count_not_covered + len(group_descriptions[ft_it])
        else:
            #It returns the first token groups if it is not possible to apply the traditional clustering.
            first_token_plus_bow_traditional_clustering[groups[ft_it]] = group_descriptions[ft_it]
            count_not_covered = count_not_covered + len(group_descriptions[ft_it])
        

        ft_it = ft_it + 1
    
    print(str(it_thread) + ': ' + str(start_it) + '/' + str(ft_it-1) + '/' + str(upper))
    #Returning dictionaries for this process
    Return_Repres_Comp[it_thread] = df_results        
    Return_Dict[it_thread] = first_token_plus_bow_traditional_clustering
    Return_Count_Not_Covered[it_thread] = count_not_covered
    Return_Count_Covered[it_thread] = count_covered


#It gets the ranges of the clusters generated by the First Token approach
#This is done in order to the processes work on.
def get_ranges(group_len, n_processes):
    if(n_processes == 1):
        return 0, (group_len - 1)

    total_len = group_len
    num_processes = n_processes
    lower = []
    upper = []
    step = int(total_len/num_processes)

    for k in range(num_processes):
        lower.append(0)
        upper.append(0)

    lower[0] = 0
    upper[0] = step
    i = 1
    j = 0
    while (i < num_processes):    
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
    upper[n_processes - 1] = upper[n_processes - 1] - 2 
    return lower, upper


#It calculates the the percentage of outliers, detected by HDBSCAN.
#It can be used with or without correction. The correction takes into account the coverage of items
#when using HDBSCAN (i.e., thouse First-Token's clusters with more than 30 descriptions)
def check_outliers(dictionary_clusters, apply_correction, count_covered, count_not_covered):
    count_minus = 0
    total = 0
    for key in dictionary_clusters:
        total += len(dictionary_clusters[key])
        if(key.endswith('_-1')):
            count_minus += len(dictionary_clusters[key])

    perc_minus = count_minus/total

    if(apply_correction):
        perc_not_covered = count_not_covered/total
        perc_covered = count_covered/total
        print('total: '+str(total))
        print('perc. not covered: '+str(perc_not_covered))
        print('perc. covered: '+str(perc_covered))
        perc_minus = (((perc_covered * total)*perc_minus) + (perc_not_covered*total))/total
        

    return perc_minus*100


#Main method for this code.
def main(saving_clusters, saving_representation, 
    cluster_alg, cluster_alg_metric, 
    cluster_alg_min_samples, cluster_alg_select_method, cluster_alg_allow_single_cluster,
    dr_alg, dr_n_comp, dr_metric,
    number_processes, tokens, remove_outliers):

    #It measures the start time.
    now = datetime.now()
    print("Start Time = " + str(now))
    
    #4 process managers to deal with the results
    manager_results = multiprocessing.Manager()
    Return_Dict = manager_results.dict()

    manager_repres = multiprocessing.Manager()
    Return_Repres_Comp = manager_repres.dict()

    manager_count_not_covered = multiprocessing.Manager()
    Return_Count_Not_Covered = manager_count_not_covered.dict()

    manager_count_covered = multiprocessing.Manager()
    Return_Count_Covered = manager_count_covered.dict()     

    jobs = []
    n_processes = number_processes

    #It loads the medical terms (medicines, drugs, etc):
    medicines = get_tokens_set('../dados/palavras/medications.txt')
    #It loads the canonical forms and their classes
    canonical_form, word_class = get_canonical_words()
    print("Read Canonical terms.")
        
    #It loads the items from the list:
    itemlist = ItemList()
    itemlist.load_items_from_file('../dados/items_preprocessed_sp0_sc1.zip', original=True)
                
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
    thread_ranges = get_ranges(group_len, n_processes)
    print('Read ranges')
    print(thread_ranges) 
    
    #It creates the processes (balanced by shuffling the keys of the dictionary:
    for i in range(n_processes):
        p = multiprocessing.Process(target=cluster_on_first_token_groups_bow, 
                                    args=(first_token_groups_new, itemlist, i, thread_ranges[0][i], 
                                          thread_ranges[1][i], medicines, canonical_form, word_class, 
                                          tokens, cluster_alg, cluster_alg_metric, 
                                          cluster_alg_min_samples, cluster_alg_select_method, cluster_alg_allow_single_cluster,
                                          dr_alg, dr_n_comp, dr_metric, remove_outliers,
                                          Return_Dict, Return_Repres_Comp, 
                                          Return_Count_Not_Covered, Return_Count_Covered))
        jobs.append(p)
        p.start()
        
    #It joins the results
    for i in range(n_processes):   
        jobs[i].join()
        
    dictionary_clusters = {}   
    

    #Starting to generate some statistics:
    #It joins the results
    for k in Return_Dict:
        print('key: '+str(k))
        dictionary_clusters.update(Return_Dict[k])

    print('Total of clusters is:  ' + str(len(dictionary_clusters)))

    

    #Calculates the number of notcovered descriptions:
    total_count_not_covered = 0
    for k in Return_Count_Not_Covered:
        #print('key: '+str(k))
        total_count_not_covered = total_count_not_covered + Return_Count_Not_Covered[k]

    #Calculates the number of covered descriptions:
    total_count_covered = 0
    for k in Return_Count_Covered:
        #print('key: '+str(k))
        total_count_covered = total_count_covered + Return_Count_Covered[k]

    print('Total count of not covered is: ' + str(total_count_not_covered))
    print('Total count of covered is: ' + str(total_count_covered))

    perc_minus = check_outliers(dictionary_clusters, False, total_count_covered, total_count_not_covered)
    perc_minus_correct = check_outliers(dictionary_clusters, True, total_count_covered, total_count_not_covered)

    print('Percentage of outliers is: ' + str(perc_minus))
    print('Corrected percentage of outliers is: ' + str(perc_minus_correct))

    #It saves the dictionary in a file, which is possible to reconstruct the final dictionary:
    a_file = open(saving_clusters, "wb")
    pickle.dump(dictionary_clusters, a_file)
    a_file.close()        
    print('\n')

    df_results = None
    for k in Return_Repres_Comp:
        #print('key: '+str(k))

        if(df_results is None):
            df_results = Return_Repres_Comp[k]
        else:
            df_results = np.concatenate((df_results, Return_Repres_Comp[k]), axis=0)

    print('\n')


    #Its saves the representation together with the results of the cluster and some ids of the item/description.
    all_results = pd.DataFrame(df_results, columns=['primeiro_token', 'id_item', 'desc_usada_no_agrupamento', 'desc_original', 'dim_0', 'dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5', 'dim_6', 'dim_7', 'dim_8', 'dim_9', 'dim_10', 'dim_11', 'dim_12', 'dim_13', 'dim_14', 'id_cluster'])    
    #itens_with_original_label = all_results[['primeiro_token', 'id_item', 'id_cluster']]
    #Saving the partial representation:
    #itens_with_original_label.to_pickle(saving_representation[0:2] + '_orig_labels_' + saving_representation)
    #itens_with_original_label.to_csv(saving_representation[0:2] + '_orig_label_' + saving_representation + '.csv', index=False, header=True, sep=';')

    #all_results['id_cluster'] = all_results[['id_cluster']].apply(LabelEncoder().fit_transform)
    
    #Saving the whole representation:
    all_results.to_pickle(saving_representation)
    all_results.to_csv(saving_representation+'.csv', index=False, header=True, sep=';')

    dbs = get_score(all_results, score='davies')
    print('davies_bouldin_score: ' + str(dbs))
    chs = get_score(all_results, score='calinski')
    print('calinski_harabasz_score: ' + str(chs))
    sse = get_score(all_results, score='silhouette', metric='euclidean')
    print('silhouette_score 20% - euclidean: ' + str(sse))
    ssc = get_score(all_results, score='silhouette', metric='cosine')
    print('silhouette_score 20% - cosine: ' + str(ssc))

    intracluster_distance  = evaluate_results(all_results)
 

    try:
        distances = []
        for group, distance in intracluster_distance.items():
            distances.append(distance['mean'])
        mean_distances = np.mean(distances)
        print('Mean distances is: '+str(mean_distances))
    except:
        print('Error on measuring the average distance.')


    
    #It measures the end time.
    now = datetime.now()
    print("\nEnd Time = " + str(now))      

if __name__ == "__main__":
    import argparse
    #Parameters of the code: saving clustering results dictionary directory (through '-sc')
    #                        saving clustering representation dictionary directory (through '-sr')    

    #                        cluster algorithm (through '-ca')    
    #                        cluster algorithm: distance metric (only for HDBSCAN algorithm, through '-cam')
    #                        cluster algorithm: min samples (only for HDBSCAN algorithm, through '-cams')
    #                        cluster algorithm: cluster selection method (only for HDBSCAN algorithm, through '-casm' [options: 'eom' or 'leaf'])
    #                        cluster algorithm: allow single cluster (only for HDBSCAN algorithm, through '-caas' [options: 0 or 1])

    #                        dimensionality reduction algorithm (through '-dra')
    #                        dimensionality reduction algorithm: number of components (only for umap algorithm, through '-drnc')
    #                        dimensionality reduction algorithm: distance metric (only for umap algorithm, through '-drm')     

    #                        n_processes (through '-n')
    #                        considered_tokens (through '-t') 
    parser = argparse.ArgumentParser()
    parser.add_argument('-sc', action='store', dest='saving_clusters')
    parser.add_argument('-sr', action='store', dest='saving_representation')

    parser.add_argument('-ca', action='store', dest='cluster_alg')
    parser.add_argument('-cam', action='store', dest='cluster_alg_metric')
    parser.add_argument('-cams', action='store', dest='cluster_alg_min_samples')
    parser.add_argument('-casm', action='store', dest='cluster_alg_select_method')
    parser.add_argument('-caas', action='store', dest='cluster_alg_allow_single_cluster')    

    parser.add_argument('-dra', action='store', dest='dr_alg') 
    parser.add_argument('-drnc', action='store', dest='dr_n_comp') 
    parser.add_argument('-drm', action='store', dest='dr_metric')

    parser.add_argument('-n', action='store', dest='n_processes')
    parser.add_argument('-t', action='store', dest='considered_tokens')

    parser.add_argument('-rmo', action='store', dest='remove_outliers')
           
    args_results = parser.parse_args()

    cluster_alg_min_samples = None
    if(args_results.cluster_alg_min_samples != 'None'):
        cluster_alg_min_samples = int(args_results.cluster_alg_min_samples)

    #print(args_results)
    main(args_results.saving_clusters, args_results.saving_representation,
        args_results.cluster_alg, args_results.cluster_alg_metric, 
        cluster_alg_min_samples, args_results.cluster_alg_select_method, 
        bool(int(args_results.cluster_alg_allow_single_cluster)),
        args_results.dr_alg, int(args_results.dr_n_comp), args_results.dr_metric,
        int(args_results.n_processes), int(args_results.considered_tokens), 
        bool(int(args_results.remove_outliers))
    )
