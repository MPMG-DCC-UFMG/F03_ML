
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
import math
import json
import multiprocessing
import json
from item.item_list import (
    ItemList,
    Item
)
from nlp.utils import (
    read_json_file,
    get_tokens_set
)
from nlp.word_embeddings import load_word_embeddings
from nlp.spellcheckeropt import SpellcheckerOpt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_path', type=str, default='../../../embeddings/word2vec/cbow_s50.txt',
        help='Path to the file containing the embeddings to be used in the representation')
    p.add_argument('--out_file',type=str, default='../dados/palavras/right_words.json',
        help='corrected words file.')
    p.add_argument('--input', type = str, default = '../dados/items_preprocessed_v3.csv.zip'
                  ,help='file containing the items dataset')
    p.add_argument('--n_threads', type=int, default=10,
                   help='number of threads to use on clustering')

    parsed = p.parse_args()

    return parsed

'''
    It gets the ranges of the words. This is done in order to the processes work
    on.
'''
def get_ranges(num_words, n_threads):

    if(n_threads == 1):
        return 0, (num_words - 1)

    total_len = num_words
    num_threads = n_threads
    lower = []
    upper = []
    step = int(total_len/num_threads)

    for k in range(num_threads):
        lower.append(0)
        upper.append(0)

    lower[0] = 0
    upper[0] = step

    i = 1
    j = 0
    while (i < num_threads):
        upper[i]  = upper[j] + step
        lower[i]  = upper[j] +  1
        if(i%2 != 0):
            upper[i] = upper[i] + 1

        i = i + 1
        j = j + 1

    upper[n_threads - 1] = num_words - 1

    return lower, upper


def run_spellchecker_thread(spellchecker, tokens_woembedding, distance, it_thread,
                            lower, upper, results_threads):

    token_woembedding_similar = {}

    for token in tokens_woembedding[lower:upper]:
        words_list = spellchecker.search(token, distance)
        if len(words_list) > 0:
            words_list.sort(key=lambda x:(x[1], x[0]))
            token_woembedding_similar[token] = words_list[0][0]

    results_threads[it_thread] = token_woembedding_similar



def run_spellchecker(words_set, tokens_woembedding, distance=2, n_threads=10):

    spellchecker = SpellcheckerOpt()
    spellchecker.load_words(list(words_set))

    # It defines the ranges (of the items) the threads will work on:
    thread_ranges = get_ranges(len(tokens_woembedding), n_threads)
    print('Read ranges')
    print(thread_ranges)

    manager = multiprocessing.Manager()
    results_threads = manager.dict()
    jobs = []

    for i in range(n_threads):
        p = multiprocessing.Process(target=run_spellchecker_thread,
        args = (spellchecker, tokens_woembedding, distance, i, thread_ranges[0][i], \
                thread_ranges[1][i], results_threads))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    token_woembedding_similar = {}
    for i in range(n_threads):
        token_woembedding_similar.update(results_threads[i])

    return token_woembedding_similar


def main():

    args = parse_args()

    # It gets the descriptions preprocessed:
    itemlist = ItemList()
    itemlist.load_items_from_file(args.input)
    num_items = len(itemlist.items_list)

    # Loading word embeddings
    #  word embeddings file, each line contains an embedding
    word_embeddings_file = args.embeddings_path

    # read word embeddings from file and store them in a map
    word_embeddings = load_word_embeddings(word_embeddings_file,
                                           words_set=itemlist.unique_words)

    # Spellchecker
    unique_words = itemlist.unique_words
    n_threads = args.n_threads
    del itemlist
    words_set_file = '../dados/palavras/words_nilc_preprocessed.json'
    words_set = read_json_file(words_set_file)
    medical = list(get_tokens_set('../dados/palavras/medications.txt'))

    words_set = words_set + medical + list(word_embeddings.keys())
    words_set = set(words_set)
    tokens_woembedding = set()

    for token in unique_words:
        if token not in words_set:
            if len(token) > 2 and not token.isnumeric():
                tokens_woembedding.add(token)

    tokens_woembedding = list(tokens_woembedding)
    print("Words that need to be corrected:", len(tokens_woembedding))

    token_woembedding_similar = run_spellchecker(words_set, tokens_woembedding,
                                                n_threads=n_threads)

    with open(args.out_file, "w") as JFile:
        json.dump(token_woembedding_similar, JFile)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
