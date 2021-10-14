import pandas as pd
from collections import Counter
import math
import numpy as np
import json
import re
from itertools import chain
import argparse

class NormalizeStandardizeUM:
    def __init__(self, standarization_json, normalization_json):
        self._standarization = self._load_standarization(standarization_json)
        self._normalization = self._load_normalization(normalization_json)


    def _load_standarization(self, json_file):
        if json_file == None:
            return None
        
        loaded = json.load(open(json_file))
        standarization = {}
        for right, typos in loaded.items():
            for typo in typos:
                standarization[typo] = right

        return standarization

    def _load_normalization(self, json_file):
        if json_file == None:
            return None
        
        loaded = json.load(open(json_file))
        normalization = {}
        for std, variations in loaded.items():
            for variation, constant in variations:
                normalization[variation] = (std, constant)

        return normalization

    def standardize(self, tokens):
        if self._standarization == None:
            return tokens

        new_tokens = list(tokens)
        for i, tk in enumerate(new_tokens):
            if tk in self._standarization:
                new_tokens[i] = self._standarization[tk]

        return new_tokens

    def normalize(self, tokens):
        if self._normalization == None:
            return tokens
        
        number_ptn = re.compile("^[0-9\.]+$")
        unit_of_measurement_ptn = re.compile("^[a-z]+[0-9]*$")

        new_tokens = list(tokens)

        for i, tk in enumerate(new_tokens[:-1]):
            is_number = (type(tk) == float or number_ptn.match(tk))
            followed_by_text = (new_tokens[i+1] in self._normalization)

            if is_number and followed_by_text:
                u_m, constant = self._normalization[new_tokens[i+1]]

                new_tokens[i] = float(new_tokens[i]) * constant
                new_tokens[i + 1] = u_m

        return new_tokens

                
    def apply_both(self, tokens):
        new_tokens = self.standardize(tokens)
        return self.normalize(new_tokens)


def test():
    aparse = argparse.ArgumentParser()
    aparse.add_argument("items", nargs=1, type=str)
    aparse.add_argument("standarization_json", nargs=1, type=str)
    aparse.add_argument("normalization_json", nargs=1, type=str)

    args = aparse.parse_args()
    items = args.items[0]
    normalization = args.normalization_json[0]
    standarization = args.standarization_json[0]
    df = pd.read_csv(items, delimiter=";")
    print(df.columns)
    df.nom_item = df.nom_item.apply(lambda v: v.lower().split())
    ns = NormalizeStandardizeUM(standarization, normalization)
    for i, row in df.iterrows():
        new_tokens = ns.apply_both(row.nom_item)
        if new_tokens != list(row.nom_item):
            print(new_tokens)
            print(row.nom_item)


if __name__ == "__main__":
    test()
