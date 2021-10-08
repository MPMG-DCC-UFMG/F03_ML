import pandas as pd
from collections import Counter
import math
import pickle
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import pickle
import json
import re
from itertools import chain
import argparse

class NormalizeStandardizeUM:
    def __init__(self, df, column_name, standarization_json, normalization_json):
        self._df = df
        self._column_name = column_name
        self._standarization = self._load_standarization(standarization_json)
        self._normalization = self._load_normalization(normalization_json)

        self._evaluate()


    def _load_standarization(self, json_file):
        standarization = {}
        
        loaded = json.load(open(json_file))
        for right, typos in loaded.items():
            for typo in typos:
                standarization[typo] = right

        return standarization

    def _load_normalization(self, json_file):
        normalization = {}
        loaded = json.load(open(json_file))

        for std, variations in loaded.items():
            for variation, constant in variations:
                normalization[variation] = (std, constant)

        return normalization

    def _evaluate(self):
        self._df[self._column_name] = \
            self._df[self._column_name] \
                .apply(lambda v: eval(v) if type(v) == str else v)

    def run(self):
        number_ptn = re.compile("^[0-9\.]+$")
        unit_of_measurement_ptn = re.compile("^[a-z]+[0-9]*$")

        for _, desc_prep in self._df[self._column_name].iteritems():
            l = len(desc_prep)

            for i, tk in enumerate(desc_prep):
                if i == (l - 1):
                    break
                
                is_number = (type(tk) == float or number_ptn.match(tk))
                followed_by_text = (desc_prep[i+1] in self._normalization)

                if is_number and followed_by_text:
                    unit_of_measurement, constant = \
                        self._normalization[desc_prep[i+1]]

                    desc_prep[i] = float(desc_prep[i]) * constant
                    desc_prep[i + 1] = unit_of_measurement


def test():
    aparse = argparse.ArgumentParser()
    aparse.add_argument("items", nargs=1, type=str)
    aparse.add_argument("standarization_json", nargs=1, type=str)
    aparse.add_argument("normalization_json", nargs=1, type=str)

    args = aparse.parse_args()
    items = args.items[0]
    normalization = args.normalization_json[0]
    standarization = args.standarization_json[0]
    df = pd.read_csv(items)
    ns = NormalizeStandardizeUM(df, "original_prep", standarization,
                                normalization)


    ns.run()
    print(df.original_prep)


if __name__ == "__main__":
    test()
