import pandas
import numpy
from collections import defaultdict, Counter
from .utils import *


def group_descriptions(items, results, cluster_name):

    descriptions_ids = results[cluster_name]
    cluster_items = list(items.loc[descriptions_ids]['original_desc'])

    return cluster_items


def descriptions_frequency(items, results, cluster_name, total_items=5,
                           verbose=False):

    cluster_items = group_descriptions(items, results, cluster_name)

    num_items = len(cluster_items)
    descriptions = dict(Counter(cluster_items))
    top_descriptions = list(descriptions.items())
    top_descriptions.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        for desc, freq in top_descriptions[:total_items]:
            print("{} ({:.2f}%)".format(desc, (freq/num_items)*100))

    most_freq_description = top_descriptions[0][0]
    tokens = most_freq_description.split()
    tokens.sort()
    tokens = ' '.join(tokens)
    frequency_perc = (top_descriptions[0][1]/num_items)*100

    return tokens, frequency_perc


def desc_most_frequent(groups, results, items_df, min_size=0):

    desc_most_freq = []

    for group_items in groups:
        group = group_items[0]
        num_items = group_items[1]
        if "_" not in group or "-1" in group:
            continue
        if num_items >= min_size:
            desc, frequency = descriptions_frequency(items_df, results, group)
            desc_most_freq.append((group, desc, frequency, num_items))

    description_count = defaultdict(list)

    for desc in desc_most_freq:
        description_count[desc[1]].append(desc[0])

    desc_canon_groups = regrouping(description_count)

    return desc_canon_groups


def tokens_desc_frequency(items, results, cluster_name, num_tokens=10, min_freq=50):

    cluster_items = group_descriptions(items, results, cluster_name)
    num_items = len(cluster_items)

    tokens_freq = defaultdict(int)
    for desc in cluster_items:
        tokens = set(desc.split())
        for token in tokens:
            tokens_freq[token] += 1

    top_tokens = list(tokens_freq.items())
    top_tokens.sort(key=lambda x:x[1], reverse=True)
    top_tokens = [(token, 100*(freq/num_items)) for token, freq in top_tokens if 100*(freq/num_items) >= min_freq]

    tokens_description = [token for token, freq in top_tokens[:num_tokens]]
    tokens_description.sort()
    description = ' '.join(tokens_description)

    return top_tokens[:num_tokens], description


def desc_tokens_most_frequent(groups, results, items_df, min_freq=30,
                              num_tokens=5, min_size=0):

    most_freq_tokens = []
    tokens_frequencies = []
    descriptions_tok = []

    for group_items in groups:
        group = group_items[0]
        num_items = group_items[1]
        if num_items >= min_size:
            top_tokens, description = tokens_desc_frequency(items_df, results,
                                                            group, num_tokens=num_tokens,
                                                            min_freq=min_freq)
            descriptions_tok.append((group, description, num_items))
            for token, freq in top_tokens:
                most_freq_tokens.append(token)
                tokens_frequencies.append(freq)

    description_tok_count = defaultdict(list)

    for desc in descriptions_tok:
        description_tok_count[desc[1]].append(desc[0])

    desctok_canon_groups = regrouping(description_tok_count)

    return desctok_canon_groups
