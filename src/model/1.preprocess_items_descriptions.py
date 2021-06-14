# imports

import argparse
import time
import pandas as pd
import numpy as np
import json
import collections
import copy
import random
import math
import matplotlib.pyplot as plt
from nlp.preprocessing import preprocess_items
from nlp.utils import (
    read_json_file)
from utils.get_items_hive import (
    get_items_hive
)
from utils.read_files import (
    get_items
)
from item.item_list import (
    ItemList,
    Item
)
from nlp.preprocess_units import group_dsc_unidade_medida


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='f03_dataset_item_licitacao', help="input table.")
    parser.add_argument("-v", "--version", required=True, help="execution version.")
    p.add_argument("-i", "--hive", default=False, help="load table from hive.")

    return parser.parse_args()


def main():

    args = parse_args()

    table = args.input
    version = args.version

    if args.hive:
        items = get_items_hive(table, recurso_limit=0.0, area=None)
    else:
        items = get_items(file, recurso_limit=0.0, area=None)

    # Preprocessing items descriptions
    items_descriptions = preprocess_items(items, n_process=20)

    itemlist = ItemList()
    itemlist.structure_items(items_descriptions)

    items_df = itemlist.to_dataframe()
    group_dsc_unidade_medida(items_df)
    itemlist.save_items_in_dataframe('../data/items_preprocessed_complete.csv.zip', items_df)

    # Split products-services
    products, services = itemlist.products_services_split('../data/f03_items_preprocessed_complete', \
                                                          '../data/f03_items_preprocessed_complete_servicos', \
                                                          version, save_in_hive=args.hive)

    itemlist.items_list = products

    # Split train-test set
    train, test = itemlist.train_test_split('../data/f03_items_preprocessed_complete_train', \
                                            '../data/f03_items_preprocessed_complete_test', \
                                            version, save_in_hive=args.hive)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
