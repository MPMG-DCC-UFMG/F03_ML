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
from utils.read_files import (
    get_items
)
from item.item_list import (
    ItemList,
    Item
)
from item.clustering.utils import *
import io

from utils.hive_access import *
from pyhive import hive


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataframe", required=True,
                        help="daframe file.")
    parser.add_argument("-v", "--version", required=True,
                        help="execution version.")
    parser.add_argument("-n", "--name", required=True,
                        help="name of the table on hive.")
    parser.add_argument("-p", "--n_process", default=20, type=int,
                    help="number of process in multiprocessing.")

    return parser.parse_args()


def main():

    args = parse_args()

    # Load dataframe
    dataframe = pd.read_csv(args.dataframe, sep=';', low_memory=False)

    print(dataframe.info())

    # Save tables to HIVE

    version = args.version
    table_name = args.name
    num_process = args.n_process
    dataframe_to_hive_table(dataframe, table_name, version, num_process=num_process)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
