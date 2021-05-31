# imports

import numpy as np
import collections
from .preprocessing_portuguese import TextPreProcessing as tpp

def count_tokens(documents):

    token_count = collections.defaultdict(int)

    for doc in documents:
        for token in doc:
            token_count[token] += 1

    return token_count


def sort_frequency_tokens(token_count):

    token_name_count = []

    for token, count in token_count.items():
        token_name_count.append((token, count))

    token_name_count.sort(key=lambda x:x[1], reverse=True)

    return token_name_count


def count_bigrams(documents):

    bigrams_count = collections.defaultdict(int)

    for doc in documents:
        if len(doc) == 2:
            bigrams_count[(doc[0], doc[1])] += 1
        elif len(doc) > 2:
            size = len(doc)
            for i in range(1, size):
                bigrams_count[(doc[i - 1], doc[i])] += 1

    return bigrams_count


def number_tokens(documents):

    doc_lengths = []

    for doc in documents:
        doc_lengths.append(len(doc))

    return doc_lengths


def tokens_length(documents, remove_duplicates=False):

    tokens = []
    for doc in documents:
        for token in doc:
            tokens.append(token.lower())

    if remove_duplicates:
        tokens = list(set(tokens))

    tok_lengths = []
    for token in tokens:
        tok_lengths.append(len(token))

    return tok_lengths


def unique_tokens(documents):

    tokens = []

    for doc in documents:
        for token in doc:
            tokens.append(token.lower())

    return list(set(tokens))


def count_numbers(documents):

    num_numbers = []
    for doc in documents:
        count = 0
        for token in doc:
            if token.isnumeric():
                count += 1
        num_numbers.append(count)

    return num_numbers


def number_stopwords(documents, stopwords):

    stopwords_ = []
    for word in stopwords:
        w = word.lower()
        stopwords_.append(w)
        w = tpp.remove_accents(w)
        stopwords_.append(w)

    stopwords_ = set(stopwords_)

    num_stopwords = []
    for doc in documents:
        count = 0
        for token in doc:
            t = token.lower()
            t = tpp.remove_accents(t)
            if t in stopwords_:
                count += 1
        num_stopwords.append(count)

    return num_stopwords


def print_statistics(numbers):

    print('Mean:', np.mean(numbers))
    print('First quartile:', np.percentile(numbers, 25, interpolation='midpoint'))
    print('Median:', np.median(numbers))
    print('Third quartile:', np.percentile(numbers, 75, interpolation='midpoint'))
    print('Std:', np.std(numbers))
    print('Var:', np.var(numbers))
    print('Max:', np.max(numbers))
    print('Min:', float(np.min(numbers)))
