
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import pickle
import sys
import time
import pandas as pd
import random
import math
import json
import multiprocessing
import json
import collections
from item.item_list import (
    ItemList,
    Item
)
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.test.utils import datapath
from nlp.utils import (
    isfloat,
    get_scientific_notation
)


def parse_args():
    """
        Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", default='item',
                        help="Use items descriptions to train language model.")
    parser.add_argument("-i", "--input",
                        default='items_preprocessed_complete.csv.zip',
                        help="File containing items descriptions.")
    parser.add_argument("-p", "--pretrained",
                        default=None,
                        help="Pretrained language model.")
    parser.add_argument("-o", "--output", required=True,
                        help="Trained model file.")
    parser.add_argument("-e", "--num_epochs", default=5,
                        help="number of epochs to train the model.")
    parser.add_argument("-w", "--num_workers", default=4,
                        help="number of worker threads to train the model.")
    parser.add_argument("-s", "--sci_notation", default=True,
                        help="convert numbers to scientific notation.")

    return parser.parse_args()


def main():

    args = parse_args()

    # It gets the descpitons processed:
    itemlist = ItemList()
    itemlist.load_items_from_file(args.input)

    if args.pretrained != None:
        print('Loading pretrained model...')
        model = load_facebook_model(args.pretrained)
    else:
        print('Initializing model...')
        model = FastText(size=300, window=10, batch_words=1000, sg=1, workers=3,
                        iter=20, min_count=0, word_ngrams=1)

    model.workers = int(args.num_workers)

    if args.type == 'item':
        items_list = list(itemlist.items_df['original_prep'])
        items_list = [eval(item) for item in items_list]
        del itemlist
    elif args.type == 'licitacao':

        licitacao_items = itemlist.items_df[['licitacao', 'original_prep']].values.tolist()
        del itemlist

        licitacao = collections.defaultdict(list)
        for licitacao, item in licitacao_items:
            licitacao[licitacao].append(eval(item))

        items_list = []

        for licitacao, items_list in licitacao.items():
            licitacao_items_list = []
            for item in items_list:
                licitacao_items_list += item
            items_list.append(licitacao_items_list)

    if args.sci_notation:
        samples = []
        for item in items_list:
            sample = []
            for token in item:
                if isfloat(token):
                    token = get_scientific_notation(token)
                sample.append(token)
            samples.append(sample)
        items_list = samples

    print('Training model...')
    if args.pretrained != None:
        model.build_vocab(sentences=items_list, update=True)
    else:
        model.build_vocab(sentences=items_list)
    total_examples = model.corpus_count
    model.train(sentences=items_list, total_examples=total_examples,
                epochs=int(args.num_epochs))

    print('Saving model...')
    model.save(args.output)
    model.wv.save_word2vec_format(args.output + '_embeddings.vec', binary=False)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
