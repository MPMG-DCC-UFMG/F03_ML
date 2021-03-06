import pandas as pd
import numpy as np
from collections import Counter
import collections
from .utils import *


def calc_token_freq(items_df, results, cluster_name, k=None):

    freq = Counter()
    items = group_descriptions(items_df, results, cluster_name)

    for item_description in items:
        tokens = item_description.split()
        if k is not None:
            tokens = tokens[:k]
        for token in tokens:
            freq[token] += 1

    N = max(freq.values())
    for f in freq:
        freq[f] /= N

    return freq


def calc_token_pos(items_df, results, cluster_name, k=None):

    items = items = group_descriptions(items_df, results, cluster_name)
    token_pos = collections.defaultdict(list)

    for item_description in items:
        tokens = item_description.split()
        if k is not None:
            tokens = tokens[:k]
        for i, token in enumerate(tokens):
            token_pos[token].append(i)

    return token_pos


def scores(groups, results, items_df, metric="median"):

    token_pos_by_group, token_freq_by_group = {}, {}
    for group_items in groups:
        group = group_items[0]
        num_items = group_items[1]
        token_pos_by_group[group] = calc_token_pos(items_df, results, group)
        token_freq_by_group[group] = calc_token_freq(items_df, results, group)

    groups_scores = {}
    for group_items in groups:
        group = group_items[0]
        num_items = group_items[1]
        tokens = list(token_pos_by_group[group].keys())
        token_pos = token_pos_by_group[group]
        token_freq = token_freq_by_group[group]

        tokens_score = {}
        for token in tokens:
            pos = token_pos[token] # list
            freq = token_freq[token] # float

            if metric.lower() == "median":
                tokens_score[token] = \
                    freq * np.median(2 ** -np.array(pos, dtype=float))
            elif metric.lower() == "sum":
                tokens_score[token] = \
                    freq * np.sum(2 ** -np.array(pos, dtype=float))
            elif metric.lower() == "log":
                tokens_score[token] = \
                    freq * -np.log2(
                        np.sum(2 ** -np.array(pos, dtype=float)))

        groups_scores[group] = tokens_score

    return groups_scores


def get_canonical_descriptions(groups, results, items_df, metric="median",
                               num_words=5):

    group_scores = scores(groups, results, items_df, metric)
    description_count = collections.defaultdict(list)

    for group, token_scores in group_scores.items():
        tokens_ranking = sorted(token_scores.items(), key=lambda v:v[1],
                                reverse=True)

        top_n = tokens_ranking[:num_words]
        top_n_tokens = " ".join(sorted(t[0] for t in top_n))
        description_count[top_n_tokens].append(group)

    return description_count


def heuristic_regrouping(groups, results, items_df, metric="median", num_words=5):

    description_count = get_canonical_descriptions(groups, results, items_df,
                                                   metric, num_words)

    desctok_canon_groups = regrouping(description_count)

    return desctok_canon_groups
