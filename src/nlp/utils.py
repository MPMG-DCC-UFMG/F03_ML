# imports

import os
import sys
import numpy as np
import json
import collections
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from .preprocessing_portuguese import TextPreProcessing as tpp
from nltk import corpus


def get_scientific_notation(token):
    return "{:.2e}".format(float(token))


def has_numbers(string):
    return any(char.isdigit() for char in string)


def isfloat(value):
    value_ = value.replace(',','.')
    try:
        float(value_)
        return True
    except ValueError:
        return False


def remove_special_characters(text):
    lista = '-#@%?º°ª:/;~^`[{]}\\|!$"\'&*()=+><\t\r\n…_'
    result = text
    for i in range(0, len(lista)):
        result = result.replace(lista[i], ' ')
    return result


def remove_dots_commas(text, punctuations='.,'):
    lista = punctuations
    result = text
    for i in range(0, len(lista)):
        result = result.replace(lista[i], ' ')
    return result


def remove_dots(text):

    tokens = text.split(' ')
    tokens_ = []

    for token in tokens:
        if not isfloat(token):
            token = token.replace('.', ' ')
        tokens_.append(token)

    return ' '.join(tokens_)


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def get_stopwords(language='pt'):

    if language == 'en':
        stopwords_list = list(set(corpus.stopwords.words('english')))
    elif language == 'pt':
        stopwords_list = set(set(tpp.get_stopwords() + \
                            corpus.stopwords.words('portuguese')))
        relevant_stopwords = {'para', 'com', 'nao', 'mais', 'muito', 'so', 'sem', \
                              'mesmo', 'mesma', 'ha', 'haja', 'hajam', 'houver', \
                              'houvera', 'seja', 'sejam', 'fosse', 'fossem', 'forem', \
                              'sera', 'serao', 'seria', 'seriam', 'tem', 'tinha', \
                              'teve', 'tinham', 'tenha', 'tiver', 'tiverem', 'tera', \
                              'terao', 'teria', 'teriam', 'uma', 'mais', 'entre', \
                              'te'}
        stopwords_list = list(stopwords_list - relevant_stopwords)


    stopwords_ = []
    for word in stopwords_list:
        w = word.lower()
        stopwords_.append(w)
        w = tpp.remove_accents(w)
        stopwords_.append(w)
    stopwords_ = set(stopwords_)

    if language == 'pt':
        return stopwords_, relevant_stopwords

    return stopwords_


def get_right_words(file, language='pt'):

    if language == 'en':
        right_word = {}
    elif language == 'pt':
        with open(file, "r") as jfile:
            right_word = json.load(jfile)
        jfile.close()

    return right_word

def get_tokens_set(file):

    tokens = open(file, 'r').readlines()
    tokens = set([token.replace('\n', '') for token in tokens])

    return tokens


def read_json_file(file):

    with open(file, "r") as jfile:
        file_dict = json.load(jfile)
    jfile.close()

    return file_dict


def plot_histogram(x_axis, bins, x_label, y_label, figname=None, title=None,
                  log=False, histtype='bar'):

    fig, (axis1) = plt.subplots(figsize=(10,8))

    plt.hist(x_axis, histtype=histtype, align='mid', orientation='vertical',
            color='royalblue', edgecolor='black', linewidth=0.4, bins=bins,
            log=log, lw=0.5)

    if log:
        axis1.set_yscale('log')
        # axis1.set_xscale('log')

    if title != None:
        plt.title(title, fontsize=20, weight='bold')

    axis1.set_xlabel(x_label, fontsize=20, weight='bold')
    axis1.set_ylabel(y_label, fontsize=20, weight='bold')
    plt.grid(False)

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    if figname != None:
        plt.savefig(figname)
    else:
        plt.show()
    plt.clf()


def get_completetext(documents):
    all_tokens = []

    for doc in documents:
        for t in doc:
            all_tokens.append(t)

    complete_text = ' '.join(all_tokens)

    return complete_text


def plot_wordcloud(text=None, frequencies=None, figname=None, collocations=False):

    fig, (axis1) = plt.subplots(figsize=(10,8))
    # Create and generate a word cloud image:
    if text != None:
        wordcloud = WordCloud(width=800, height=800, background_color='white',
                            min_font_size=13, collocations=collocations,
                            normalize_plurals=False).generate(text)
    elif frequencies != None:
        wordcloud = WordCloud(width=800, height=800, background_color='white',
                            min_font_size=22, collocations=collocations,
                            normalize_plurals=False,
                            max_words=100).generate_from_frequencies(frequencies)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    if figname != None:
        plt.savefig(figname)
    else:
        plt.show()
    plt.clf()



def print_statistics(numbers):

    print('Mean:', np.mean(numbers))
    print('First quartile:', np.percentile(numbers, 25, interpolation='midpoint'))
    print('Median:', np.median(numbers))
    print('Third quartile:', np.percentile(numbers, 75, interpolation='midpoint'))
    print('Std:', np.std(numbers))
    print('Var:', np.var(numbers))
    print('Max:', np.max(numbers))
    print('Min:', float(np.min(numbers)))


def groups_frequency_sort(groups):

    groups_names_size = []

    for group, items in groups.items():
        size = len(items)
        groups_names_size.append((group, size))

    groups_names_size.sort(key=lambda x:x[1], reverse=True)
    return groups_names_size
