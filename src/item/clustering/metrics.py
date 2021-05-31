import numpy as np
from scipy import spatial


def cosine_distance(arrayA, arrayB):
    '''
        Get the cosine distance between two vectors.
    '''

    return spatial.distance.cosine(arrayA, arrayB)


def cosine_similarity(arrayA, arrayB):
    '''
        Get the cosine similarity between two vectors.
    '''

    return 1 - cosine_distance(arrayA, arrayB)


def euclidean_distance(arrayA, arrayB):
    '''
        Get the euclidean distance between two vectors.
    '''

    return spatial.distance.euclidean(arrayA, arrayB)


def calc_distance(arrayA, arrayB, distance='euclidean'):
    '''
        Get the distance between two vectors.
    '''

    if distance == 'euclidean':
        return euclidean_distance(arrayA, arrayB)
    elif distance == 'cosine':
        return cosine_distance(arrayA, arrayB)
