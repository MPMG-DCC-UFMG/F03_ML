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


    if args.type == 'item':
        items_list = list(itemlist.items_df['original_prep'])
        del itemlist
    elif args.type == 'licitacao':

        licitacao_items = itemlist.items_df[['licitacao', 'original_prep']].values.tolist()
        del itemlist

        licitacao = collections.defaultdict(list)
        for licitacao, items in licitacao_items:
            licitcao[licitacao].append(items)

        items_list = []

        for licitacao, items_list in licitacao.items():
            licitacao_items_list = []
            for item in items_list:
                licitacao_items_list += item
            items_list.append(licitacao_items_list)

    print('Training model...')
    if args.pretrained != None:
        model.build_vocab(sentences=items_list, update=True)
    else:
        model.build_vocab(sentences=items_list)
    total_examples = model.corpus_count
    model.train(sentences=items_list, total_examples=total_examples,
                epochs=5)

    print('Saving model...')
    model.save(args.output)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("--- %s minutes ---" % ((end - start)/60))
