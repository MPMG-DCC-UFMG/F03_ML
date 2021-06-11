import collections
import json
import re


def remove_new_tokens_from_embeddings(tokens_set1, tokens_set2, word_embeddings):
    '''
        Remove tokens from the set of embeddings. A token that is in 'tokens_set2'
        and is not in 'tokens_set1' is removed from the set of embeddings.

        tokens_set1 (set): set of tokens.
        tokens_set2 (set): set of tokens.
        word_embeddings (dict): dictionary of word embeddings (word -> embedding).
    '''

    new_tokens = tokens_set2 - tokens_set1
    for token in new_tokens:
        word_embeddings.pop(token, None)

    return word_embeddings


def get_first_token(item_dict):
    '''
        Get the first token of an item description.
    '''

    if len(eval(item_dict['palavras'])) == 0:
        return ''

    return item_dict['palavras'][0]


def get_price(item_dict):
    '''
        Get the price of an item.
    '''
    return item_dict['preco']
