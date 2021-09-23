# imports

import pandas as pd
import numpy as np
import collections
from scipy import spatial
from sklearn import preprocessing
import pickle
import json
import sys
import os
from utils.read_files import *


def get_item_vec(_item, word_embeddings, word_class, categories=None,
                 embedding_type=None, norm=True, operation='mean'):
    '''
        Build the vector representation for an item using the word embeddings.

        _items (item): item object.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
        norm (bool): if the item embeddings/vectors should be normalized.
        operation (str): operation to be used to build the item embeddings/vectors.
    '''

    if operation == 'mean':
        item_vec = get_item_embedding(_item.get_item_dict(), word_embeddings, word_class, \
                    categories=categories, embedding_type=embedding_type)
    elif operation == 'weighted':
        item_vec = get_item_embedding_weighted(_item.get_item_dict(), word_embeddings, word_class, \
                    categories=categories, embedding_type=embedding_type)
    elif operation == 'concatenate':
        item_vec = get_words_plus_categories_embeddings(_item.get_item_dict(), word_embeddings, word_class, \
                    categories=categories, embedding_type=embedding_type)

    if norm:
        item_vec = normalize(item_vec.reshape(1, -1))

    return item_vec

def select_columns(items_df):
    '''
        It selects the columns ['palavras', 'unidades_medida', 'numeros', 'cores',
        'materiais', 'tamanho', 'quantidade', 'original_prep'] of a dataframe.
    '''

    return items_df[['palavras', 'unidades_medida', 'numeros', 'cores', \
                     'materiais', 'tamanho', 'quantidade', 'original_prep']]


def define_zero_matrix(group_desc, itemlist, word_class, tags, categories):
    '''
        Define a zero matrix based on the size of the number of descriptions in that
        group (row) and the number of words (only medicines and nouns) from all
        descriptions in that group.

        group_desc (list): item ids.
        itemlist (ItemList object): contains the item and informations about the
                                    dataset.
        word_class (dict): part of a speech tags (word -> tag).
        tags (list): word tags to be used.
        categories (list): word categories to be used.
    '''

    list_words = set()

    for desc_id in group_desc:
        item_dict = itemlist.items_df.loc[desc_id].to_dict()
        if isinstance(item_dict['palavras'], str):
            words = eval(item_dict['palavras'])
        else:
            words = item_dict['palavras']
        for token in words:
            if token in word_class and word_class[token] in tags:
                list_words.append(token)

        for category in categories:
            if isinstance(item_dict[category], str):
                tokens = eval(item_dict[category])
            else:
                tokens = item_dict[category]
            for token in tokens:
                list_words.append(token)

    list_words = list(list_words)
    list_words.sort()

    rows = len(group_desc)
    columns = len(list_words)
    matrix_bow = np.zeros((rows, columns))

    return matrix_bow, list_words, rows, columns


def define_description_bow(group_desc, itemlist, word_class, tags=None,
                           categories=None):
    '''
        Define the bag-of-words matrix.

        group_desc (list): item ids.
        itemlist (ItemList object): contains the item and informations about the
                                    dataset.
        word_class (dict): part of a speech tags (word -> tag).
        tags (list): word tags to be used.
        categories (list): word categories to be used.
    '''

    if categories == None:
        categories = {'unidades_medida', 'numeros', 'materiais', 'tamanho',
                      'quantidade'}
        categories = list(categories)

    tags = set(tags)

    matrix_list = define_zero_matrix(group_desc, itemlist, word_class, tags)

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

        item_dict = itemlist.items_df.loc[desc_id].to_dict()

        #This is to return the original and the preprocessed description to build the file of results:
        if isinstance(item_dict['original'], str):
            desc_original = eval(item_dict['original'])
            desc_prep = eval(item_dict['original_prep'])
            words = eval(item_dict['palavras'])
        else:
            desc_original = item_dict['original']
            desc_prep = item_dict['original_prep']
            words = item_dict['palavras']

        # desc_prep_rep = str(desc_prep).replace('\'', '').replace('[', '').replace(']', '').replace(',', '')

        original_descs[i, 0] = desc_original
        preproc_descs[i, 0 ] = desc_prep
        ids_descs[i, 0 ] = desc_id

        # For now, we only look at the array of 'palavras' to build the BOW:
        for w in words:
            if w in list_words:
                k = list_words.index(w)
                #it updates the bow matrix with elements of that group (defined by the first token)
                bow_matrix[i, k]  = 1.0

        # other categories of words
        for category in categories:
            if isinstance(item_dict[category], str):
                tokens = eval(item_dict[category])
            else:
                tokens = item_dict[category]
            for w in tokens:
                if w in list_words:
                    k = list_words.index(w)
                    bow_matrix[i, k]  = 1.0
        i = i + 1

    #Concatanate the original descriptions with the preprocesses ones.
    #and, also, adds the ids of the descriptions.
    result_descs = np.concatenate((original_descs, preproc_descs), axis=1)
    result_descs_w_ids = np.concatenate((ids_descs, result_descs), axis=1)

    return bow_matrix


def get_item_embedding_tcu(_item, word_embeddings, word_class, embedding_type):
    '''
        Build the vector representation for an item using the word embeddings
        using the 'tcu' operation..

        _item (dict): item object.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        embedding_type (list): word tags to be used.
    '''

    embedding_size = len(list(word_embeddings.values())[0])
    item_embedding = np.zeros(embedding_size)
    item_dict = _item

    if isinstance(item_dict['original_prep'], str):
        document = eval(item_dict['original_prep'])
    else:
        document = item_dict['original_prep']

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


def get_item_embedding_weighted(_item, word_embeddings, word_class, categories=None,
                                embedding_type=None):
    '''
        Build the vector representation for an item using the word embeddings
        using the 'weighted' operation.

        _item (dict): item object.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
    '''

    embedding_size = len(list(word_embeddings.values())[0])
    item_embedding = np.zeros(embedding_size)

    if embedding_type != None:
        tags = set(embedding_type)

    item_dict = _item
    weight_cum = 0

    if isinstance(item_dict['palavras'], str):
        words = set(eval(item_dict['palavras']))
        tokens = eval(item_dict['original_prep'])
    else:
        words = set(item_dict['palavras'])
        tokens = item_dict['original_prep']

    for pos, token in enumerate(tokens):
        if token in word_embeddings and token in words:
            if embedding_type == None:
                # position-weighted average
                # decreases linearly
                weight = 1.0/(pos + 1)
                item_embedding += weight * np.array(word_embeddings[token])
                weight_cum += weight
            elif token in word_class and word_class[token] in tags:
                weight = 1.0/(pos + 1)
                item_embedding += weight * np.array(word_embeddings[token])
                weight_cum += weight

    if categories == None:
        categories = {'unidades_medida', 'numeros', 'materiais', 'tamanho',
                      'quantidade'}
        categories = list(categories)

    num_tokens = len(tokens)

    for category in categories:
        if isinstance(item_dict[category], str):
            tokens = eval(item_dict[category])
        else:
            tokens = item_dict[category]
        for token in tokens:
            if token in word_embeddings:
                weight = (1+(1/num_tokens))*(1/4)
                item_embedding += weight * np.array(word_embeddings[token])
                weight_cum += weight

    if weight_cum != 0:
        item_embedding /= weight_cum

    return item_embedding


def get_item_embedding(_item, word_embeddings, word_class, categories=None,
                       embedding_type=None):
    '''
        Build the vector representation for an item using the word embeddings
        using the 'mean' operation.

        _item (dict): item object.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
    '''

    embedding_size = len(list(word_embeddings.values())[0])
    item_embedding = np.zeros(embedding_size)

    if embedding_type != None:
        tags = set(embedding_type)

    item_dict = _item
    num_tokens = 0

    if isinstance(item_dict['palavras'], str):
        words = eval(item_dict['palavras'])
    else:
        words = item_dict['palavras']

    for token in words:
        if token in word_embeddings:
            if embedding_type == None:
                item_embedding += np.array(word_embeddings[token])
                num_tokens += 1
            elif token in word_class and word_class[token] in tags:
                item_embedding += np.array(word_embeddings[token])
                num_tokens += 1

    if categories == None:
        categories = {'unidades_medida', 'numeros', 'materiais', 'tamanho',
                      'quantidade'}
        categories = list(categories)

    for category in categories:
        if isinstance(item_dict[category], str):
            tokens = eval(item_dict[category])
        else:
            tokens = item_dict[category]
        for token in tokens:
            if token in word_embeddings:
                item_embedding += np.array(word_embeddings[token])
                num_tokens += 1

    if num_tokens != 0:
        item_embedding /= num_tokens

    return item_embedding


def get_words_plus_categories_embeddings(_item, word_embeddings, word_class,
                                        categories=None, embedding_type=None):
    '''
        Build the vector representation for an item using the word embeddings
        using the 'concatenate' operation.

        _item (dict): item object.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
    '''

    embedding_size = len(list(word_embeddings.values())[0])
    item_embedding = np.zeros(embedding_size)

    if embedding_type != None:
        tags = set(embedding_type)

    item_dict = _item
    num_tokens = 0

    if isinstance(item_dict['palavras'], str):
        words = set(eval(item_dict['palavras']))
        tokens = eval(item_dict['original_prep'])
    else:
        words = set(item_dict['palavras'])
        tokens = item_dict['original_prep']

    for token in tokens:
        if token in word_embeddings and token in words:
            if embedding_type == None:
                item_embedding += np.array(word_embeddings[token])
                num_tokens += 1
            elif token in word_class and word_class[token] in tags:
                item_embedding += np.array(word_embeddings[token])
                num_tokens += 1

    if num_tokens != 0:
        item_embedding /= num_tokens

    item_embedding_cat = np.zeros(embedding_size)
    num_tokens = 0

    if categories == None:
        categories = {'unidades_medida', 'numeros', 'materiais', 'tamanho',
                      'quantidade'}
        categories = list(categories)

    for category in categories:
        if isinstance(item_dict[category], str):
            tokens = eval(item_dict[category])
        else:
            tokens = item_dict[category]
        for token in tokens:
            if token in word_embeddings:
                item_embedding_cat += np.array(word_embeddings[token])
                num_tokens += 1

    if num_tokens != 0:
        item_embedding_cat /= num_tokens

    item_embedding = np.concatenate((item_embedding, item_embedding_cat))

    return item_embedding


def normalize(embeddings):
    '''
        It normalizes item embeddings.
    '''

    embeddings_normalized = preprocessing.normalize(embeddings, norm='l2')
    return embeddings_normalized


def get_items_embeddings(items_list, word_embeddings, word_class, categories=None,
                        embedding_type=None, type='list', operation='weighted'):
    '''
        Build the vector representation for an item using the word embeddings.

        items_list (list): list of items.
        word_embeddings (dict): pre-trained word embeddings (word -> embedding).
        word_class (dict): part of a speech tags (word -> tag).
        categories (list): word categories to be used.
        embedding_type (list): word tags to be used.
        type (str): data type which the embedding should be stored.
        operation (str): operation to be used to build the item embeddings/vectors.
    '''

    embedding_size = len(list(word_embeddings.values())[0])

    if type == 'list':
        items_embs = []
    elif type == 'dict':
        items_embs = {}

    id = 0
    for _item in items_list:
        item_dict = _item.get_item_dict()
        if operation == 'weighted':
            item_emb = list(get_item_embedding_weighted(item_dict, word_embeddings, \
                            word_class, categories=categories,
                            embedding_type=embedding_type))
        elif operation == 'mean':
            item_emb = list(get_item_embedding(item_dict, word_embeddings, \
                            word_class, categories=categories,
                            embedding_type=embedding_type))
        elif operation == 'concatenate':
            item_emb = list(get_words_plus_categories_embeddings(item_dict,
                                                    word_embeddings, word_class,
                                                    categories=categories,
                                                    embedding_type=embedding_type))
        elif operation == 'tcu':
            item_emb = list(get_item_embedding_tcu(item_dict, word_embeddings,
                                                   word_class,
                                                   embedding_type=embedding_type))

        if type == 'list':
            items_embs.append(np.array(item_emb))
        elif type == 'dict':
            items_embs[id] = np.array(item_emb)
        id += 1

    return items_embs


def save_items_embeddings_pickle(items_embeddings, file):
    '''
        It saves item embeddings in a pickle file.
    '''

    items_embs = []

    if type(items_embeddings) == dict:
        items_embeddings = items_embeddings.items()
    elif type(items_embeddings) == list:
        items_embeddings = enumerate(items_embeddings)

    for item_id, embedding in items_embeddings:
        items_embs.append(tuple([item_id]) + tuple(embedding))

    items_embs_df = pd.DataFrame(items_embs)
    items_embs_df.to_pickle(file)

    return items_embs_df


def save_items_embeddings(items_embeddings, file):
    '''
        It saves item embeddings in a json file.
    '''

    # write to json file
    with open(file, "w") as JFile:
        json.dump(items_embeddings, JFile)
    JFile.close()


def load_items_embeddings(file):
    '''
        It loads item embeddings from a json file.
    '''

    return read_json_file(file)


def get_group_embeddings_matrix(group_desc, items_list, word_embeddings, word_class,
                                categories=None, embedding_type=None, norm=True,
                                operation='weighted'):
    '''
        Define the embedding matrix for a group.
    '''

    items_list = select_columns(items_list)
    embedding_size = len(list(word_embeddings.values())[0])
    embeddings_matrix = []

    for desc_id in group_desc:
        item_dict = items_list.loc[desc_id].to_dict()
        if operation == 'weighted':
            item_emb = get_item_embedding_weighted(item_dict, word_embeddings,
                                          word_class, categories=categories,
                                          embedding_type=embedding_type)
        elif operation == 'mean':
            item_emb = get_item_embedding(item_dict, word_embeddings,
                                          word_class, categories=categories,
                                          embedding_type=embedding_type)
        elif operation == 'concatenate':
            item_emb = get_words_plus_categories_embeddings(item_dict,
                                        word_embeddings, word_class,
                                        categories=categories,
                                        embedding_type=embedding_type)
        elif operation == 'tcu':
            item_emb = get_item_embedding_tcu(item_dict, word_embeddings,
                                              word_class,
                                              embedding_type=embedding_type)

        embeddings_matrix.append(np.array(item_emb))

    if norm:
        embeddings_matrix = normalize(embeddings_matrix)

    return embeddings_matrix


def get_group_embeddings_from_dict(group_desc, items_embeddings, norm=False):
    '''
        It loads item embeddings for a group from a json file.
    '''

    embeddings_matrix = []

    for desc_id in group_desc:
        item_emb = items_embeddings[str(desc_id)]
        embeddings_matrix.append(np.array(item_emb))

    if norm:
        embeddings_matrix = normalize(embeddings_matrix)

    return embeddings_matrix
