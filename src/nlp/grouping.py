# imports

import copy
import collections

def get_groups(instances):

    groups = collections.defaultdict(int)

    for instance in instances:
        groups[instance] += 1

    return groups


def get_groups_size(groups):

    groups_size = []

    for group, count in groups.items():
        groups_size.append(count)

    return groups_size


def get_unigram_groups(documents, token_count):

    unigrams = []

    for doc in documents:
        maxi = 0
        for token in doc:
            if token_count[token] > maxi:
                maxi = token_count[token]
        tokens_sorted = copy.deepcopy(doc)
        tokens_sorted.sort()
        for token in tokens_sorted:
            if token_count[token] == maxi:
                unigrams.append(token)
                break

    return unigrams


def get_two_tokens_groups(documents, token_count):

    bigrams = []

    for doc in documents:
        tokens_ = []
        for token in doc:
            tokens_.append((token, token_count[token]))

        tokens_.sort(key=lambda x:(-x[1], x[0]))
        if len(tokens_) == 1:
            first = tokens_[0][0]
            bigrams.append(first)
        elif len(tokens_) > 1:
            first = tokens_[0][0]
            second = tokens_[1][0]
            bigram = [first, second]
            bigram.sort()
            bigrams.append((bigram[0], bigram[1]))

    return bigrams


def get_first_token_groups(documents):

    first_tokens = []

    for doc in documents:
        if len(doc) != 0:
            first_tokens.append(doc[0])

    return first_tokens


def get_bigram_groups(documents, bigrams_count):

    bigrams = []

    for doc in documents:
        if len(doc) == 1:
            bigrams.append(doc[0])
        elif len(doc) == 2:
            bigrams.append((doc[0], doc[1]))
        elif len(doc) > 2:
            size = len(doc)
            bigram_freq = []
            for i in range(1, size):
                prev_bigram = (doc[i - 1], doc[i])
                bigram_freq.append((prev_bigram, bigrams_count[prev_bigram]))
            bigram_freq.sort(key=lambda x : (-x[1], x[0]))
            bigrams.append(bigram_freq[0][0])

    return bigrams


def get_first_two_groups(documents):

    two_tokens = []

    for doc in documents:
        if len(doc) == 1:
            two_tokens.append(doc[0])
        elif len(doc) >= 2:
            two_tokens.append((doc[0], doc[1]))

    return two_tokens


def groups_frequency_sort(groups):

    groups_names_size = []

    for name, size in groups.items():
        groups_names_size.append((name, size))

    groups_names_size.sort(key=lambda x:x[1], reverse=True)
    return groups_names_size
