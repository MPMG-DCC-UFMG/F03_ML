# imports

import collections


def get_tokens_set(file):

    tokens = open(file, 'r').readlines()
    tokens = set([token.replace('\n', '') for token in tokens])

    return tokens


def count_tokens(documents):

    token_count = collections.defaultdict(int)

    for doc in documents:
        for token in doc:
            token_count[token] += 1

    return token_count


def translate_id_to_descriptions(ids, descriptions_ids):
    arr = []

    for i in ids:
        arr.append(descriptions_ids[i])
    return arr
