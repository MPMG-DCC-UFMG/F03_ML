from collections import Counter
from datetime import datetime, timedelta
from multiprocessing import Process, Lock
import argparse
import json
import math
import numpy as np
import pandas as pd
import pickle
import pickle
import seaborn as sns

class Regrouping:
    def __init__(self, *, items_filename: str, groups_filename: str):
        self._token_pos_buf = None
        self._token_freq_buf = None

        self._df = pd.read_csv(items_filename, sep=";")

        with open(groups_filename, "rb") as f:
            self._groups = pickle.load(f)

        self._join_items_and_groups()
        self._transform()


    def _join_items_and_groups(self):
        inverse = {}
        group_prefix = {}
        for group_id, items in self._groups.items():
            if "_" in group_id:
                parts = group_id.split("_")
                group_name = parts[0]
                cod = int(parts[1])
            else:
                group_name = "outlier"
                cod = -1

            for item in items:
                inverse[item] = "outlier" if cod == -1 else group_id
                group_prefix[item] = group_name

        self._df["group"] = self._df.index
        self._df.group = self._df.group\
                                 .apply(lambda i: inverse.get(i, "outlier"))

        self._df["group_prefix"] = self._df.index
        self._df.group_prefix = self._df.group_prefix\
                            .apply(lambda i: group_prefix.get(i, "outlier"))

        self._df = self._df.loc[self._df.group != "outlier"]\
                           .reset_index(drop=True)

        print(self._df)


    def _transform(self):
        parse = lambda arg: eval(arg)
        self._df.original_prep = self._df.original_prep.apply(parse)

    def _get_groups(self):
        groups = self._df.group.drop_duplicates().to_list()
        return groups

    def _calc_token_freq(self, group: str):
        freq = Counter()

        df = self._df.loc[self._df.group == group]

        for item_description in df.original_prep:
            for token in item_description:
                freq[token] += 1

        N = max(freq.values())
        for f in freq:
            freq[f] /= N

        return freq

    def _calc_token_pos(self, group: str):
        df = self._df.loc[self._df.group == group]

        token_pos = dict()
        for k, row in df.iterrows():
            for i, token in enumerate(row.original_prep):
                if token not in token_pos:
                    token_pos[token] = []
                token_pos[token].append(i)

        return token_pos


    def scores(self, metric, use_buffer):
        all_groups = self._get_groups()

        if use_buffer and self._token_pos_buf and self._token_freq_buf:
            token_pos_by_group = self._token_pos_buf
            token_freq_by_group = self._token_freq_buf
        else:
            token_pos_by_group, token_freq_by_group = {}, {}
            for group in all_groups:
                token_pos_by_group[group] = self._calc_token_pos(group)
                token_freq_by_group[group] = self._calc_token_freq(group)
            if use_buffer:
                self._token_pos_buf = token_pos_by_group
                self._token_freq_buf = token_freq_by_group

        groups_scores = {}
        for group in sorted(all_groups):
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

    def get_canonical_descriptions(self, metric, n, use_buffer=True):
        group_scores = self.scores(metric, use_buffer)

        canonical_description_by_group = {}
        for group, token_scores in group_scores.items():
            tokens_ranking = sorted(token_scores.items(), key=lambda v:v[1],
                                    reverse=True)

            top_n = tokens_ranking[:n]
            top_n_tokens = " ".join(sorted(t[0] for t in top_n))

            canonical_description_by_group[group] = top_n_tokens

        return canonical_description_by_group

    def suggested_groups(self, metric, n, use_buffer=True):
        canonical_description_by_group = \
            self.get_canonical_descriptions(metric, n, use_buffer)

        description_df = pd.DataFrame(canonical_description_by_group.items())\
                           .rename(columns={
                               0: "group", 1:
                               "canonical_description"
                           })
        print(description_df)

        joined_df = self._df.merge(description_df,
                                   how="left",
                                   on="group")

        print(joined_df)

        grouped_df = joined_df.groupby(
            by=["canonical_description", "group_prefix"])

        print(grouped_df)

        suggestions = {}
        for key, group_df in grouped_df:
            if not len(group_df): # no group formed
                continue

            _groups = group_df.group.drop_duplicates().to_list()
            if len(_groups) == 1:
                continue

            trailing_numbers = sorted([g.split("_")[1] for g in _groups],
                                      key=lambda v: int(v))

            group_name = _groups[0].split("_")[0]
            new_group = group_name + "_" + "_".join(trailing_numbers)
            suggestions[new_group] = _groups

        return suggestions




if __name__ == "__main__":
    aparse = argparse.ArgumentParser()
    aparse.add_argument("items_filename",
                        type=str,
                        nargs=1,
                        help="Path to a .csv file containing the items")

    aparse.add_argument("items_group",
                        type=str,
                        nargs=1,
                        help="Path to a .pkl file containing the groups of " +
                             "each item")

    aparse.add_argument("-m", "--metric",
                        nargs=1,
                        required=True,
                        choices=["median", "sum", "log"],
                        help="Metric to rank the tokens")

    aparse.add_argument("-n", "--n-tokens",
                        nargs=1,
                        required=True,
                        type=int,
                        help="How many tokens to consider")

    args = aparse.parse_args()

    items_filename = args.items_filename[0]
    items_group = args.items_group[0]
    metric = args.metric[0]
    n_tokens = args.n_tokens

    print(args)

    g = Regrouping(items_filename=items_filename,
                   groups_filename=items_group)

    print(g.suggested_groups(metric, n_tokens))
