# imports

import pandas as pd
import numpy as np
import json
import collections
from .preprocessing_portuguese import TextPreProcessing as tpp
from scipy import spatial
from sklearn import preprocessing


def load_word_embeddings(file, words_set=None):
    '''
        Read word embedding from a file and store them in a map.
    '''

    if words_set != None:
        words_set = set(words_set)
    word_embeddings = {}

    with open(file, 'r') as data:

        data.readline()
        lines = data.readlines()

        for line in lines:
            line = line.strip('\n')
            line = line.split(' ', maxsplit=1)
            token = line[0]
            token_preprocess = tpp.remove_accents(token.lower())
            embedding = line[1].split(' ')
            embedding_nums = []
            for num in embedding:
                try:
                    embedding_nums.append(float(num))
                except:
                    None
            if words_set == None:
                word_embeddings[token_preprocess] = embedding_nums
            elif token_preprocess in words_set:
                word_embeddings[token_preprocess] = embedding_nums

    return word_embeddings


def get_item_embedding(document, word_embeddings, word_class, embedding_type=None,
                       embedding_size=50):
    '''
        Build the vector representation for an item using the word embeddings.
    '''

    item_embedding = np.zeros(embedding_size)

    if embedding_type != None:
        tags = set(embedding_type)

    num_tokens = 0

    for token in document:
        if token in word_embeddings:
            if embedding_type == None:
                item_embedding += np.array(word_embeddings[token])
                num_tokens += 1
            elif token in word_class and word_class[token] in tags:
                item_embedding += np.array(word_embeddings[token])
                num_tokens += 1

    if num_tokens != 0:
        item_embedding /= num_tokens

    return item_embedding


def get_item_embedding_weighted(document, word_embeddings, word_class,
                                embedding_type=None, embedding_size=50):
    '''
        Build the vector representation for an item using the word embeddings.
    '''

    item_embedding = np.zeros(embedding_size)

    if embedding_type != None:
        tags = set(embedding_type)

    weight_cum = 0

    for pos, token in enumerate(document):
        if token in word_embeddings:
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

    if weight_cum != 0:
        item_embedding /= weight_cum

    return item_embedding


def get_items_embeddings(documents, word_embeddings, word_class,
                         embedding_type=None, embedding_size=50, type='list',
                         operation='mean'):

    if type == 'list':
        documents_embeddings = []
    elif type == 'dict':
        documents_embeddings = {}

    id = 0
    for doc in documents:
        if operation == 'mean':
            item_emb = list(get_item_embedding(doc, word_embeddings, word_class, \
                                               embedding_type))
        elif operation == 'weighted':
            item_emb = list(get_item_embedding_weighted(doc, word_embeddings, \
                                                        word_class, embedding_type))

        if type == 'list':
            documents_embeddings.append(item_emb)
        elif type == 'dict':
            documents_embeddings[id] = item_emb
        id += 1

    return documents_embeddings


def cosine_distance(arrayA, arrayB):
    '''
        Get the cosine distance between two vectors.
    '''

    arrayA = arrayA.reshape(1, -1)
    arrayB = arrayB.reshape(1, -1)
    return spatial.distance.cosine(arrayA, arrayB)


def cosine_similarity(arrayA, arrayB):
    '''
        Get the cosine similarity between two vectors.
    '''

    return 1 - cosine_distance(arrayA, arrayB)


def calc_distance(arrayA, arrayB, distance='cosine'):
    '''
        Get the distance between two vectors.
    '''
    if distance == 'cosine':
        value = cosine_distance(arrayA, arrayB)

    return value


def zero_vector(embedding):
    '''
        Check if an item embedding is a zero vector.
    '''

    zero_vector = (embedding == np.zeros(len(embedding)))
    if zero_vector.all():
        return True

    return False


def normalize(embeddings):
    '''
        It normalizes items embeddings.
    '''

    embeddings_normalized = preprocessing.normalize(embeddings, norm='l2')
    return embeddings_normalized


def get_similarities_between_items(itemA, items, item_embedding):
    '''
        Get similarity between itemA and all items in a list.
    '''

    items_similarities = []
    embeddingA = np.array(item_embedding[itemA])

    for i in range(0, len(items)):
        itemB = items[i]
        if itemB != itemA:
            embeddingB = np.array(item_embedding[itemB])
            similarity = cosine_similarity(embeddingA, embeddingB)
            items_similarities.append((itemB, similarity))


'''
    Get similarity of all items pairs in a list.
'''
def get_items_similarities(items, item_embedding):

    items_similarities = collections.defaultdict(list)

    j = 0
    for item_id in items:
        itemA = item_id
        embeddingA = np.array(item_embedding[itemA])
        for i in range(j, len(items)):
            itemB = items[i]
            if itemB != itemA:
                embeddingB = np.array(item_embedding[itemB])
                similarity = cosine_similarity(embeddingA, embeddingB)
                items_similarities[itemA].append((itemB, similarity))
                items_similarities[itemB].append((itemA, similarity))
        j += 1

    return items_similarities


'''
    It saves items embeddings in a pickle file.
'''
def save_items_embeddings(items_embeddings, file):

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


'''
    It saves items embeddings in a json file.
'''
def save_items_embeddings_json(items_embeddings, file):

    # write to json file
    with open(file, "w") as JFile:
        json.dump(items_embeddings, JFile)


'''
    It loads items embeddings from a json file.
'''
def load_items_embeddings_json(file):

    with open(file, "r") as JFile:
        items_embs = json.load(JFile)
    JFile.close()

    return items_embs


'''
    Define the embedding matrix for a group.
'''
def get_group_embeddings_matrix(group_desc, items_list, word_embeddings, word_class,
                                embedding_type=None, embedding_size=50, norm=False,
                                operation='mean'):

    embeddings_matrix = []

    for desc_id in group_desc:
        if operation == 'mean':
            item_embedding = get_item_embedding(items_list[desc_id], word_embeddings,
                                                word_class, embedding_type=embedding_type,
                                                embedding_size=embedding_size)
        elif operation == 'weighted':
            item_embedding = get_item_embedding_weighted(items_list[desc_id], word_embeddings,
                                                word_class, embedding_type=embedding_type,
                                                embedding_size=embedding_size)
        embeddings_matrix.append(list(item_embedding))

    if norm:
        embeddings_matrix = normalize(embeddings_matrix)

    return embeddings_matrix
