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
import hdbscan
import argparse
import pdb
import os
import umap 

from item.item_list import (
    ItemList,
    Item
)
from nlp.word_embeddings import (
    load_word_embeddings,
    get_item_embedding,
    get_items_embeddings
)

from nlp.preprocessing import (
    clean_text,
    preprocess,
    tokenize,
    preprocess_document,
    tokenize_document,
    get_stopwords, 
    lemmatization_document,
    get_canonical_words)

from nlp.pos_tagging import (
    get_tokens_tags
)
from item.spellcheckeropt import SpellcheckerOpt
from item.utils import get_tokens_set
from textpp_ptbr.preprocessing import TextPreProcessing as tpp

# Import xmeans through pyclustering library:
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer;
from pyclustering.cluster.xmeans import xmeans

# Import agglomerative clustering through scikit learn library:
from sklearn.cluster import AgglomerativeClustering

#Import normalize package 
from sklearn.preprocessing import normalize

# Import agglomerative clustering through pyclustering library:
from pyclustering.cluster.agglomerative import agglomerative

# Import OPTICS through pyclustering library:
from pyclustering.cluster.optics import optics

# Import DBSCAN through pyclustering library:
from pyclustering.cluster.dbscan import dbscan




def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--sample_frac',type=float,default=1.0,
        help = 'Fraction of the original dataset that will be used')
    p.add_argument('--embeddings_path',type=str,default='../../../embeddings/word2vec/cbow_s50.txt',
        help='Path to the file containing the embeddings to be used in the representation')
    #p.add_argument('--dim_reduction', action='store_true',
    #              help='If set the script will perform dimentionality reduction in the dataset')    
    p.add_argument('--min_cluster_size',type=int,default=30,
        help='The minimum size of a cluster')    
    
    p.add_argument('--reduc_dim',type=int,default=-1,
        help='Indicates the number of dimentions that will be used in the dimentionality '\
                   'reduction. When none is set the dataset will be used in its original '\
                   'dimentionality')
    
    p.add_argument('--outpath',type=str,default='./results/tcu/',
        help='path to the write the outputs')
    p.add_argument('--outname',type=str,default='saida_hdbscan.csv',
        help='filename to the output data')
    
    p.add_argument('--input', type = str, default = '../dados/items_preprocessed.zip'
                  ,help='file containing the items dataset')

    p.add_argument('--just_words',action='store_true'
                          ,help='When this paremeter is set just the section of '\
                          'items dataset that contains the words will be loaded ')

    p.add_argument('--re_cluster',action='store_true'
                   ,help='Indicates that the process will re-cluster '\
                   'a set of descriptions previouly described as outliers. '\
                   'this version receives a dataframe containings the items IDs and the '\
                   'words already processed')
    
    p.add_argument('--complete_desc',action='store_true'
                          ,help='When this paremeter is set the entire  '\
                          'items descriptions will be used ')

    p.add_argument('--class2use', nargs = '*' 
                  ,help='The list of syntatic classes that will be used to construct the '\
                   'embeddings. When none is set all the description will be used '\
                   'Options are: N, MED, A, ADJ....')
                   

    parsed = p.parse_args()
    #pdb.set_trace()         
    return parsed


# Build the vector representation for an item using the word embeddings
def get_item_embedding_weighted(document, word_embeddings, word_class, embedding_type=None,
                       embedding_size=50):

    item_embedding = np.zeros(embedding_size)
    num_tokens = len(document)
    peso_acum = 0
    
    for pos,token in enumerate(document):
        if token in word_embeddings:
            if embedding_type == None:
                #media ponderada pela posicao
                #decresce linearmente                
                peso = 1/(pos+1)
                
                if token.isdigit():
                    #segundo a abordagem do tcu eles deixam os pesos de numeros em 
                    #3/4 da faixa de pesos
                    peso_digito = (1+(1/(len(document))))*(1/4)
                    item_embedding += peso_digito*np.array(word_embeddings[token])
                    peso_acum += peso_digito
                else:                
                    item_embedding += peso*np.array(word_embeddings[token])
                    peso_acum += peso
                    
            elif token in word_class and word_class[token] in set(embedding_type):
                
                peso = 1/(pos+1)
                
                if token.isdigit():
                    #segundo a abordagem do tcu eles deixam os pesos de numeros em 
                    #3/4 da faixa de pesos
                    peso_digito = (1+(1/(len(document))))*(1/4)
                    item_embedding += peso_digito*np.array(word_embeddings[token])
                    peso_acum += peso_digito
                else:  
                    item_embedding += peso*np.array(word_embeddings[token])
                    peso_acum += peso
                    
    if peso_acum != 0:
        item_embedding /= peso_acum

    return item_embedding


def get_items_embeddings_weighted(documents, word_embeddings, word_class, embedding_type=None,
                         embedding_size=50, type='list'):

    if type == 'list':
        documents_embeddings = []
    elif type == 'dict':
        documents_embeddings = {}

    id = 0
    for doc in documents:
        if type == 'list':
            documents_embeddings.append(list(get_item_embedding_weighted(doc, word_embeddings,
                                                    word_class, embedding_type
                                                    ,embedding_size)))
        elif type == 'dict':
            documents_embeddings[id] = list(get_item_embedding_weighted(doc, word_embeddings,
                                                    word_class, embedding_type
                                                    ,embedding_size))
        id += 1

    return documents_embeddings



if __name__ == '__main__':
    
    t0 = time.time()    
    args = parse_args()
    SAMPLE_SIZE_TO_DIM_REDUC = 0.2
    #pdb.set_trace()
    print(time.asctime()," Getting the descpitons processed:")
    #TODO parametrize
    if args.re_cluster:
        df_recluster = pd.read_csv(args.input,sep=';')
        #pdb.set_trace()
        items_words_complete = list(\
            map(lambda x: x.replace('[','')
                            .replace(']','').split(','),df_recluster.desc.values))
        
    
    else:
        itemlist = ItemList()
        itemlist.load_items_from_file(args.input, just_words=args.just_words)
        # Get all tokens 

        if args.complete_desc:
            #TODO pedir para adicionar a descricao completa pre processada e tokenizada
            #alterar o nome da coluna que vai conter essa informacao na linha abaixo
            items_words_complete = [x.original_preprocessed for x in itemlist.items_list]
        else:
            items_words_complete = [x.words for x in itemlist.items_list]

        #del itemlist
    pdb.set_trace()
    unique_tokens = set([x for desc in  items_words_complete for x in desc])
    
    
    print(time.asctime()," Loading word embeddings files")
    #TODO parametrize
    word_embeddings_file = args.embeddings_path
    word_embeddings = load_word_embeddings(word_embeddings_file,words_set=unique_tokens)
    #pdb.set_trace()
    
    
    print(time.asctime()," Getting the tags of tokens descriptions")    
    #pdb.set_trace()    
                   
    if args.class2use:
        embedding_type=args.class2use
        word_class = get_tokens_tags(words_set=unique_tokens)
        print(time.asctime()," Using only ",embedding_type," words")
    else:
        print(time.asctime()," Using all words")
        embedding_type=None
        word_class = None
                   
    if args.sample_frac < 1.0:                
        sample_size = int(len(items_words_complete)*args.sample_frac)
        
        print(time.asctime()," Using a sample of ", sample_size, " items")        
        used_ids = random.sample(list(range(0,len(items_words_complete))),sample_size)        
        sampled_items = [items_words_complete[x] for x in used_ids]
        used_item_descriptions = sampled_items
        #db.set_trace()
        print(time.asctime()," Constructing the weigthed embeddings vectors")
        items_embeddings = get_items_embeddings_weighted(sampled_items, word_embeddings, word_class,embedding_type=embedding_type)
    else:
        #maintain the ids used in the first clustering step
        if args.re_cluster:
            used_ids = df_recluster.item_id.values
            used_item_descriptions = items_words_complete
        else:
            used_ids = list(range(len(items_words_complete)))

            used_item_descriptions = items_words_complete
        print(time.asctime()," Constructing the weigthed embeddings vectors")
        #pdb.set_trace()
        items_embeddings = get_items_embeddings_weighted(items_words_complete, word_embeddings, word_class, embedding_type=embedding_type)
              
    del word_embeddings           
                   
    #pdb.set_trace()

    print(time.asctime()," L2 Normalizing data in order to approximate euclidean distance to arc-cos")
    #normed_items_embeddings = normalize(items_embeddings, norm='l2')
    #normalizando no proprio hdbscan
    normed_items_embeddings = np.array(items_embeddings)
    
    if args.reduc_dim != -1:                 
        
        print(time.asctime()," Start dimentionality reduction")
        umap_redux = umap.UMAP(n_components=args.reduc_dim, random_state=999, metric='cosine',verbose=True)
        
        #no caso de usar uma amostragem do conjunto na clusterizacao
        #a reducao de dimensionalidade está sendo executada com o sample inteiro
        if args.sample_frac < 1.0:
            umap_redux.fit(normed_items_embeddings) 
        #caso nao usemos amostragem a reducao de dimensionalidade eh treinada 
        #em uma amostra e posteriomente eh aplicada ao conjunto todo
        else:
            num_items = len(normed_items_embeddings)
            sample_ids = random.sample(list(range(0,num_items)), int(SAMPLE_SIZE_TO_DIM_REDUC*num_items))            
            umap_redux.fit(normed_items_embeddings[sample_ids])
            
        sampled_items_emb_norm = umap_redux.transform(X=normed_items_embeddings)      
        
        
        reduc_filename = os.path.join(args.outpath, 'reduced_dim_'+
                                    os.path.basename(args.input).split('.')[0])
        
        pkl_file = open(reduc_filename, 'wb')
        pickle.dump(umap_redux, pkl_file)
        pkl_file.close()
        
        
    #TODO REMOVER    
    #sample_ids = random.sample(list(range(0,len(items_embeddings))),1000)            
    #sampled_items_emb = [items_embeddings[x] for x in sample_ids]
    #sampled_items_emb_norm = normalize(sampled_items_emb, norm='l2')

    
    print(time.asctime()," Running HDBSCAN")
    t1 = time.time()
    hdb_clusterer = hdbscan.HDBSCAN(metric='l2', min_cluster_size=args.min_cluster_size, min_samples=1, prediction_data=True, core_dist_n_jobs=8)
    #TODO ALTERAR ENTRADA
    hdb_clusterer.fit(normed_items_embeddings)
    print(time.asctime()," HDBSCAN time: ",time.time()-t1)
    print(time.asctime()," Results")

    #samp_desc = [items_words_complete[x] for x in items_words_complete]
    primeiro_token_id = [None for _ in used_ids]
    original_desc = [itemlist.items_list[x].original_preprocessed for x in used_ids]
    
    df_out = pd.DataFrame(data={'first_token':primeiro_token_id
                                , 'cluster_id':hdb_clusterer.labels_
                                , 'item_id':used_ids
                                , 'description':used_item_descriptions
                               ,'original_description':original_desc})
    
    #save_clustering_results(hdb_clusterer.labels_, items_list, used_item_descriptions
    #                        ,df_out.to_csv(os.path.join(args.outpath, args.outname)
    #                        ,first_toke=False)
    print(time.asctime()," ",len(df_out.cluster_id.unique()),' Clusters')
    print(time.asctime()," ",df_out[df_out.cluster_id == -1].shape[0]/df_out.shape[0],'% Outliers')
    df_out.to_pickle(os.path.join(args.outpath, args.outname))
    
    #df_out.to_csv(os.path.join(args.outpath, args.outname),index=False,sep=';')
    
    print(time.asctime()," Total time: ",time.time()-t0)