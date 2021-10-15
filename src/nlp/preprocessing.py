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
from .std_norm_unit_of_measurement import NormalizeStandardizeUM
from .utils import *


class PreprocessingText:

    def __init__(self, language='pt', remove_stopwords=True, remove_numbers=False,
                 lemmatize=True, remove_duplicates=True,
                 remove_tokens_with_digits=False, remove_punctuation=True,
                 spellcheck="../data/dicionario/replacement_licitacao.json",
                 standarization="../data/palavras/standarization.json",
                 normalization="../data/palavras/normalization.json"):

        self.language = language
        self.right_word = None
        self.stopwords = None
        self.canonical_word = None
        self.word_class = None

        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        self.spellcheck = spellcheck
        self.remove_duplicates = remove_duplicates
        self.remove_tokens_with_digits = remove_tokens_with_digits
        self.remove_punctuation = remove_punctuation

        if standarization or normalization:
            self.ns_um = NormalizeStandardizeUM(standarization, normalization)
        else:
            self.ns_um = None

        if self.remove_stopwords:
            self.stopwords, self.relevant_stopwords  = get_stopwords(language)

        if self.spellcheck is not None:
            self.right_word = get_right_words(self.spellcheck, self.language)

        if self.language == 'pt' and self.lemmatize:
            self.canonical_word, self.word_class = self.get_canonical_words()


    def clean_text(self, text):

        text = text.rjust(1 + len(text))
        text += " "    # insert a space in end of the string

        # remove item and subitem indicator, eg, 1.2.3[.4[.5[...]]]
        text = re.sub(r'^\d+\.\d+(?:.\d+)+', ' ', text)

        # insert a space between sequence of digits and measurement units
        text = re.sub(r'([0-9]+)x([0-9]+)([a-z]+)', r' \1 x \2 \3 ', text, flags=re.I)

        # insert a space between sequence of digits and measurement units
        text = re.sub(r'([0-9]+)([a-z]+)x([0-9]+)([a-z]+)', r' \1 \2 x \3 \4 ', text,
                      flags=re.I)

        # insert a space between sequence of digits and measurement units
        text = re.sub(r'([0-9]+)x([0-9]+)', r' \1 x \2 ', text, flags=re.I)

        # insert a space between sequence of digits and sequence of letters
        text = re.sub(r' (\d+)([a-z]+)', r' \1 \2 ', text, flags=re.I)
        text = re.sub(r' ([a-z]+)([0-9]+)([a-z]+)', r' \1 \2 \3 ', text, flags=re.I)

        # remove 0 in front of the number
        text = re.sub(r' (0+)([1-9]+)', r' \2', text)

        # remove big numbers
        text = re.sub(r' \d{5,}', r' ', text)

        return text


    def tokenize_document(self, document):

        if self.language == 'pt':
            tokens = word_tokenize(document, 'portuguese')
        elif self.language == 'en':
            tokens = word_tokenize(document, 'english')

        tokens_ = []
        strings = set()

        # remove duplicate tokens
        for t in tokens:
            if self.remove_tokens_with_digits and t not in strings:
                if t.isnumeric() or isfloat(t):
                    tokens_.append(t)
                elif not has_numbers(t):
                    tokens_.append(t)
            elif t not in strings:
                tokens_.append(t)
            strings.add(t)

        return tokens_


    def preprocess_document_portuguese(self, document):

        item = document
        item = item.replace(',', '.')

        # lowercase letters
        description = item.lower()

        # remove non alphanumeric tokens
        description = remove_special_characters(description)

        # brazilian portuguese preprocessing
        description = tpp.remove_accents(description)
        if self.stopwords != None:
            rm_stopwords = re.compile(r'(^|\b)(' + r'|'.join(list(self.stopwords)) + r')($|\b)')
            description = rm_stopwords.sub(' ', description)

        description = self.clean_text(description)

        if self.remove_punctuation:
            description = remove_dots(description)

        # numbers preprocessing
        if self.remove_numbers:
            description = tpp.remove_numbers(description)
            description = tpp.remove_numbers_in_full(description)
            description = strip_short(description, minsize=3)  # remove short tokens

        description = re.sub(r'\w{21,}', r'', description)  # remove long tokens
        description = strip_multiple_whitespaces(description)  # strip whitespaces

        return description


    def preprocess_document_english(self, document):

        item = document

        # lowercase letters
        description = item.lower()
        if self.remove_punctuation:
            # swaps punctuation with spaces
            description = strip_punctuation2(description)
            # remove non alphanumeric tokens
            description = strip_non_alphanum(description)

        if self.stopwords != None:
            rm_stopwords = re.compile(r'(^|\b)(' + r'|'.join(list(self.stopwords)) + r')($|\b)')
            description = rm_stopwords.sub(' ', description)

        # numbers preprocesssing
        if self.remove_numbers:
            # TODO: implementa para o ingles
            description = tpp.remove_numbers(description)

        description = self.clean_text(description)

        # description = strip_short(description, minsize=3)  # remove short tokens
        description = re.sub(r'\w{21,}', r'', description)  # remove long tokens
        description = strip_multiple_whitespaces(description)  # strip whitespaces

        return description

    def standarize_normalize(self, tokens):
        return self.ns_um.apply_both(tokens)

    def preprocess_document(self, document):

        if self.language == 'pt':
            description = self.preprocess_document_portuguese(document)
            doc = self.tokenize_document(description)
            if self.spellcheck is not None:
                doc = self.spellcheck_document(doc)
            if self.ns_um:
                doc = self.standarize_normalize(doc)
            if self.lemmatize:
                doc = self.lemmatization_document(doc)
        elif self.language == 'en':
            description = self.preprocess_document_english(document)
            doc = self.tokenize_document(description)
            if self.spellcheck is not None:
                doc = self.spellcheck_document(doc)
            if self.lemmatize:
                doc = self.lemmatization_document(doc)

        return doc


    def spellcheck_document(self, document):

        tokens_ = []
        for tok in document:
            if tok in self.right_word:
                correct_tok = self.right_word[tok]
                if ' ' in correct_tok:
                    for t in correct_tok.split(' '):
                        tokens_.append(t)
                else:
                    tokens_.append(correct_tok)
            else:
                tokens_.append(tok)

        return tokens_


    def get_canonical_words(self, words_set=None):

        if words_set != None:
            words_set = set(words_set)

        dictionary_file = '../data/dicionario/delaf.dic.zip'
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
                    if word_class[c] != 'V':
                        noun = c
                        break
                if noun == '':
                    canonical_form[word] = canonical[0]
                else:
                    canonical_form[word] = noun

        return canonical_form, word_class


    def lemmatization_document(self, document):

        doc_lemmatized = []
        for tok in document:
            if tok in self.canonical_word:
                doc_lemmatized.append(self.canonical_word[tok])
            else:
                doc_lemmatized.append(tok)

        return doc_lemmatized


    def lemmatization(self, items):
        return [self.lemmatization_document(doc) for doc in items]


    def preprocess(self, items):

        items = [self.preprocess_document(item) for item in items]

        # remove duplicates descriptions
        if self.remove_duplicates:
            items = list(set(items))

        # tokenize items descriptions
        items = [self.tokenize_document(item) for item in items]

        if self.lemmatize:
            items = self.lemmatization(items)
        if self.spellcheck is not None:
            items = self.spell_check(items)

        return items


    def tokenize(self, items):

        # remove duplicates
        if self.remove_duplicates:
            items = list(set(items))

        # tokenize items descriptions
        items = [self.tokenize_document(item) for item in items]

        return items


    def spell_check(self, items):
        return [self.spellcheck_document(item) for item in items]


    '''
        It gets the ranges of the items. This is done in order to the processes work
        on.
    '''
    def get_ranges(self, num_items, n_process):

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


    def check_first_token(self, doc):

        if len(doc) <= 1:
            return doc

        first_token = doc[0]

        if first_token in self.stopwords or first_token in self.relevant_stopwords or \
           has_numbers(first_token) or first_token.isnumeric() or isfloat(first_token):
            doc.remove(first_token)
            doc.append(first_token)

        return doc


    def preprocess_items_process(self, items, it_process, results_process):

        items_descriptions = []

        for item in items:
            description = item[0]
            # if type(dsc_unidade) is not str and math.isnan(dsc_unidade):
            #     dsc_unidade = ""
            if isinstance(description, str) and description != "":
                doc = self.preprocess_document(description)
                doc = self.check_first_token(doc)
                items_descriptions.append((doc, description) + tuple(item[1:]))

        results_process[it_process] = items_descriptions


    def preprocess_items(self, items, n_process=10):

        items_descriptions = []

        # It defines the ranges (of the items) the process will work on:
        process_ranges = self.get_ranges(len(items), n_process)
        if n_process == 1:
            process_ranges = [[process_ranges[0]], [process_ranges[1]]]

        print('Read ranges')
        print(process_ranges)

        manager = multiprocessing.Manager()
        results_process = manager.dict()
        jobs = []

        for i in range(n_process):
            lower = process_ranges[0][i]
            upper = process_ranges[1][i]
            items_process = items[lower:(upper + 1)]
            p = multiprocessing.Process(target=self.preprocess_items_process,
            args = (items_process, i, results_process))
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
