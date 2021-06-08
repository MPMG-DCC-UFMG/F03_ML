# imports

import collections
from .utils import (
    get_tokens_set
)
from .preprocessing import (
    PreprocessingText
)


def get_tokens_tags(words_set=None, medications=True):
    '''
        Get the classes of all words in the portuguese dictionary.
    '''

    if words_set != None:
        words_set = set(words_set)

    medical = get_tokens_set('../data/palavras/medications.txt')
    preprocessing = PreprocessingText()
    canonical_word, word_class = preprocessing.canonical_word, preprocessing.word_class

    del canonical_word
    del preprocessing

    if medications:
        for tok in medical:
            if tok not in word_class:
                word_class[tok] = 'MED'

    return word_class


'''
    Performs pos-tagging on a set of items. Get the tags of tokens descriptions.
    If a word doesn't match with any tag, it is mapped to 'UNTAGGED'.
'''
def pos_tagging(documents):

    word_class = get_tokens_tags()

    word_tags = []
    not_tagged = 0
    tag_count = collections.defaultdict(int)

    for doc in documents:
        for tok in doc:
            if tok in word_class:
                word_tags.append((tok, word_class[tok]))
                tag_count[word_class[tok]] += 1
            else:
                word_tags.append((tok, 'UNTAGGED'))
                not_tagged += 1

    return word_tags, tag_count, not_tagged
