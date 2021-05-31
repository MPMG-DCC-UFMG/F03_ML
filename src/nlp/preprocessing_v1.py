# imports

import re
import math
import numpy as np
import collections
import json
import gensim
import nltk
import zipfile
import multiprocessing
from nltk import corpus
from nltk.tokenize import word_tokenize
from gensim.utils import tokenize
from gensim.parsing.preprocessing import (
    strip_multiple_whitespaces,
    strip_non_alphanum,
    strip_punctuation2,
    strip_short)
from .preprocessing_portuguese import TextPreProcessing as tpp


def has_numbers(string):
    return any(char.isdigit() for char in string)


def clean_text(text):

    text = text.rjust(1 + len(text))
    text += " "    # insert a space in end of the string

    # insert a space between sequence of digits and measurement units
    text = re.sub(r'([0-9]+)x([0-9]+)([a-z]+)', r' \1 x \2 \3 ', text, flags=re.I)

    # insert a space between sequence of digits and measurement units
    text = re.sub(r'([0-9]+)([a-z]+)x([0-9]+)([a-z]+)', r' \1 \2 x \3 \4 ', text,
                  flags=re.I)

    # insert a space between sequence of digits and measurement units
    text = re.sub(r'([0-9]+)x([0-9]+)', r' \1 x \2 ', text, flags=re.I)

    # insert a space between sequence of digits and sequence of letters
    text = re.sub(r' (\d+)([a-z]+) ', r' \1 \2 ', text, flags=re.I)
    text = re.sub(r' ([a-z]+)([0-9]+)([a-z]+) ', r' \1 \2 \3 ', text, flags=re.I)

    # insert a space between sequence of letters and sequence of digits
    # text = re.sub(r' ([a-z]+)(\d+)', r' \1 \2 ', text, flags=re.I)
    # text = re.sub(r' ([0-9]+)([a-z]+)([0-9]+)', r' \1 \2 \3 ', text, flags=re.I)

    # remove 0 in front of the number
    text = re.sub(r' (0+)([1-9]+)', r' \2', text)

    # remove big numbers
    text = re.sub(r' \d{5,} ', r' ', text)

    return text


def get_stopwords():

    stopwords_list = list(set(tpp.get_stopwords() + \
                        corpus.stopwords.words('portuguese')))
    stopwords_ = []
    for word in stopwords_list:
        w = word.lower()
        stopwords_.append(w)
        w = tpp.remove_accents(w)
        stopwords_.append(w)
    stopwords_ = set(stopwords_)

    return stopwords_


def tokenize_document(document, remove_tokens_with_digits=False):

    tokens = word_tokenize(document, 'portuguese')
    tokens_ = []
    strings = set()

    # remove duplicate tokens
    for t in tokens:
        if remove_tokens_with_digits and t not in strings:
            if t.isnumeric():
                tokens_.append(t)
            elif not has_numbers(t):
                tokens_.append(t)
        elif t not in strings:
            tokens_.append(t)
        strings.add(t)

    return tokens_


def preprocess_document(document, remove_numbers, stopwords=None,
                        remove_punctuation=True):

    item = document
    # lowercase letters
    description = item.lower()
    if remove_punctuation:
        # swaps punctuation with spaces
        description = strip_punctuation2(description)
        perc = True if '%' in description else False
        # remove non alphanumeric tokens
        description = strip_non_alphanum(description)
        description = tpp.remove_special_characters(description)
        # add '%' at the end of the description
        if perc:
            description += '%'

    # brazilian portuguese preprocessing
    description = tpp.remove_accents(description)
    if stopwords != None:
        rm_stopwords = re.compile(r'(^|\b)(' + r'|'.join(list(stopwords)) + r')($|\b)')
        description = rm_stopwords.sub(' ', description)
    # description = tpp.remove_hour(description)      # REMOVER

    # numbers preprocessing
    if remove_numbers:
        description = clean_text(description)
        description = tpp.remove_numbers(description)
        description = tpp.remove_numbers_in_full(description)
        description = strip_short(description, minsize=3)  # remove short tokens
    else:
        description = clean_text(description)

    description = re.sub(r'\w{21,}', r'', description)  # remove long tokens
    description = strip_multiple_whitespaces(description)  # strip whitespaces

    return description


def spellcheck_document(document, right_word):

    tokens_ = []
    for tok in document:
        if tok in right_word:
            correct_tok = right_word[tok]
            if ' ' in correct_tok:
                for t in correct_tok.split(' '):
                    tokens_.append(t)
            else:
                tokens_.append(correct_tok)
        else:
            tokens_.append(tok)

    return tokens_


def get_canonical_words(words_set=None):

    if words_set != None:
        words_set = set(words_set)

    dictionary_file = '../dados/dicionario/delaf.dic.zip'
    canonical_forms = collections.defaultdict(list)
    word_class = {}
    tags = {'A', 'ADV', 'CONJ','DET', 'INTERJ', 'N', 'PF', 'PREP', 'PRO', 'V',
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
                if word_class[c] == 'N':
                    noun = c
                    break
            if noun == '':
                canonical_form[word] = canonical[0]
            else:
                canonical_form[word] = noun

    return canonical_form, word_class


def lemmatization_document(document, canonical_form):

    doc_lemmatized = []
    for tok in document:
        if tok in canonical_form:
            doc_lemmatized.append(canonical_form[tok])
        else:
            doc_lemmatized.append(tok)

    return doc_lemmatized


def lemmatization(items):

    canonical_form = get_canonical_words()

    items = [lemmatization_document(doc, canonical_form) for doc in items]

    return items


def preprocess(items, remove_numbers=False, remove_duplicates=False,
               remove_stopwords=True, lemmatize=True, spellcheck=True):

    if remove_stopwords:
        stopwords_ = get_stopwords()
        relevant_stopwords = {'para', 'com', 'nao', 'mais', 'muito', 'so', 'sem', \
                              'mesmo', 'mesma', 'ha', 'haja', 'hajam', 'houver', \
                              'houvera', 'seja', 'sejam', 'fosse', 'fossem', 'forem', \
                              'sera', 'serao', 'seria', 'seriam', 'tem', 'tinha', \
                              'teve', 'tinham', 'tenha', 'tiver', 'tiverem', 'tera', \
                              'terao', 'teria', 'teriam', 'uma', 'mais', 'entre', \
                              'te'}
        stopwords_ = stopwords_ - relevant_stopwords
    else:
        stopwords_ = None

    items = [preprocess_document(item, remove_numbers, stopwords_) for item in items]

    # remove duplicates descriptions
    if remove_duplicates:
        items = list(set(items))

    # tokenize items descriptions
    items = [tokenize_document(item) for item in items]

    if lemmatize:
        items = lemmatization(items)
    if spellcheck:
        items = spell_check(items)

    return items


def tokenize(items, remove_duplicates=True):

    # remove duplicates
    if remove_duplicates:
        items = list(set(items))

    stopwords_ = get_stopwords()

    # tokenize items descriptions
    items = [tokenize_document(item) for item in items]

    return items


def spell_check(items):

    with open('../dados/palavras/right_words_nilc.json', "r") as jfile:
        right_word = json.load(jfile)
    jfile.close()

    items = [spellcheck_document(item, right_word) for item in items]

    return items


'''
    It gets the ranges of the items. This is done in order to the processes work
    on.
'''
def get_ranges(num_items, n_process):

    if(n_process == 1):
        return 0, (num_items - 1)

    total_len = num_items
    num_process = n_process
    lower = []
    upper = []
    step = int(total_len/num_process)

    for k in range(num_process):
        lower.append(0)
        upper.append(0)

    lower[0] = 0
    upper[0] = step

    i = 1
    j = 0
    while (i < num_process):
        upper[i]  = upper[j] + step
        lower[i]  = upper[j] +  1
        if(i%2 != 0):
            upper[i] = upper[i] + 1

        i = i + 1
        j = j + 1

    upper[n_process - 1] = num_items - 1

    return lower, upper


def check_first_token(doc, stopwords_):

    if len(doc) <= 1:
        return doc

    first_token = doc[0]

    if first_token in stopwords_ or has_numbers(first_token) or \
        first_token.isnumeric():
        doc.remove(first_token)
        doc.append(first_token)

    return doc


def preprocess_items_process(items, stopwords_, right_word, canonical_form,
                             it_process, lemmatize, spellcheck, results_process):

    items_descriptions = []

    for item in items:
        description = item[0]
        item_id = item[1]
        licitacao = item[2]
        price = item[3]
        dsc_unidade = item[4]
        # funcao = str(item[5])
        funcao = None
        ano = item[5]
        if type(dsc_unidade) == str:
            dsc_unidade = tpp.remove_accents(dsc_unidade.lower())
        elif math.isnan(dsc_unidade):
            dsc_unidade = ""
        if funcao == 'nan':
            funcao = 'Vazio'
        doc = preprocess_document(description, remove_numbers=False,
                                  stopwords=stopwords_)
        doc = tokenize_document(doc)
        if spellcheck:
            doc = spellcheck_document(doc, right_word)
        if lemmatize:
            doc = lemmatization_document(doc, canonical_form)
        doc = check_first_token(doc, stopwords_)
        items_descriptions.append((doc, item_id, licitacao, price, dsc_unidade, \
                                  description, funcao, ano))

    results_process[it_process] = items_descriptions


def preprocess_items(items, n_process=10, lemmatize=True, spellcheck=True):

    items_descriptions = []
    stopwords_ = get_stopwords()
    relevant_stopwords = {'para', 'com', 'nao', 'mais', 'muito', 'so', 'sem', \
                          'mesmo', 'mesma', 'ha', 'haja', 'hajam', 'houver', \
                          'houvera', 'seja', 'sejam', 'fosse', 'fossem', 'forem', \
                          'sera', 'serao', 'seria', 'seriam', 'tem', 'tinha', \
                          'teve', 'tinham', 'tenha', 'tiver', 'tiverem', 'tera', \
                          'terao', 'teria', 'teriam', 'uma', 'mais', 'entre', \
                          'te'}
    stopwords_ = stopwords_ - relevant_stopwords

    if lemmatize:
        canonical_form, word_class = get_canonical_words()
    else:
        canonical_form = None

    with open('../dados/palavras/right_words_nilc.json', "r") as jfile:
        right_word = json.load(jfile)
    jfile.close()

    # It defines the ranges (of the items) the process will work on:
    process_ranges = get_ranges(len(items), n_process)
    print('Read ranges')
    print(process_ranges)

    manager = multiprocessing.Manager()
    results_process = manager.dict()
    jobs = []

    for i in range(n_process):
        lower = process_ranges[0][i]
        upper = process_ranges[1][i]
        items_process = items[lower:(upper + 1)]
        p = multiprocessing.Process(target=preprocess_items_process,
        args = (items_process, stopwords_, right_word, canonical_form, i, \
                lemmatize, spellcheck, results_process))
        jobs.append(p)
        p.start()

    del items

    for proc in jobs:
        proc.join()
        proc.close()

    items_descriptions = []
    for i in range(n_process):
        items_descriptions += results_process[i]

    return items_descriptions
