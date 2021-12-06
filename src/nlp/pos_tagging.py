# imports

import collections
from .utils import (
    get_tokens_set
)
from .preprocessing import (
    PreprocessingText
)


def get_canonical_words(words_set=None):

    if words_set != None:
        words_set = set(words_set)

    dictionary_file = '../data/dicionario/delaf.dic.zip'
    canonical_forms = collections.defaultdict(list)
    word_class = {}
    tags = {'A', 'ADV', 'CONJ', 'DET', 'INTERJ', 'N', 'PF', 'PREP', 'PRO', 'V',
            'SIGL', 'ABREV'}

    with zipfile.ZipFile(dictionary_file, 'r') as zipped:
        with zipped.open('delaf.dic', 'r') as data:
            lines = data.readlines()
            for line in lines:
                line = line.decode('utf-8')
                line = line.strip('\n')
                word_canonical = re.split(r'[,.+:X\s]\s*', line)
                word = tpp.remove_accents(word_canonical[0].lower())
                canonical = tpp.remove_accents(word_canonical[1].lower())
                wclass = word_canonical[2]
                if wclass not in tags:
                    continue

                if words_set != None and word in words_set:
                    canonical_forms[word].append(canonical)
                else:
                    canonical_forms[word].append(canonical)

                if canonical in word_class and wclass == 'N':
                    word_class[canonical] = wclass
                elif canonical not in word_class:
                    word_class[canonical] = wclass

                if word in word_class and wclass == 'N':
                    if words_set != None and word in words_set:
                        word_class[word] = wclass
                    else:
                        word_class[word] = wclass
                elif word not in word_class:
                    if words_set != None and word in words_set:
                        word_class[word] = wclass
                    else:
                        word_class[word] = wclass

    canonical_form = {}

    for word, canonical in canonical_forms.items():
        if len(canonical) == 1:
            canonical_form[word] = canonical[0]
        else:
            noun = ''
            for c in canonical:
                if word_class[c] != 'V':
                    noun = c
                    break
            if noun == '':
                canonical_form[word] = canonical[0]
            else:
                canonical_form[word] = noun

    return canonical_form, word_class


def get_tokens_tags(words_set=None, medications=True):
    '''
        Get the classes of all words in the portuguese dictionary.
    '''

    medical = get_tokens_set('../data/palavras/medications.txt')
    canonical_word, word_class = get_canonical_words(words_set=words_set)

    if medications:
        for tok in medical:
            if tok not in word_class:
                word_class[tok] = 'MED'

    return canonical_word, word_class


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
