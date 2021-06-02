# imports

import collections
import json
from nlp.preprocessing_portuguese import TextPreProcessing as tpp
from nlp.preprocessing import PreprocessingText
from gensim.parsing.preprocessing import strip_multiple_whitespaces
import re


def add_first_token_column(data):
    '''
        Add "first_token" column to dataframe.
    '''

    data['first_token'] = data['cluster'].str.split('_').str[0]
    return data


def add_outlier_column(data):
    '''
        Add "outlier" column to dataframe.
    '''

    data['outlier'] = data['cluster'].str.split('_').str[1]
    data['outlier'].fillna(1, inplace=True)
    data['outlier'].replace({'-1': 1}, inplace=True)

    return data


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
    return item_dict['preço']


def isfloat(value):
    value_ = value.replace(',','.')
    try:
        float(value_)
        return True
    except ValueError:
        return False

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def remove_special_characters(text):
    lista = '-#@%?º°ª:/;~^`[{]}\\|!$"\'&*()=+><\t\r\n…'
    result = text
    for i in range(0, len(lista)):
        result = result.replace(lista[i], ' ')
    return result

def remove_dots(text):
    lista = '.'
    result = text
    for i in range(0, len(lista)):
        result = result.replace(lista[i], '')
    return result

def lemmatize_unit_metric(text, canonical_unit_metric):
    if text in canonical_unit_metric:
        return canonical_unit_metric[text]
    return text

def process_un_medida(un_medida, stopwords, preprocessing, canonical_unit_metric):

    un_medida_process = str(un_medida).lower()

    if len(un_medida_process) == 50:
        un_medida_process = un_medida_process.lstrip('0')
    un_medida_process = remove_prefix(un_medida_process, "1 ")
    un_medida_process = tpp.remove_accents(un_medida_process)

    un_medida_process = remove_special_characters(un_medida_process)
    un_medida_process = remove_dots(un_medida_process)

    un_medida_process = ' '+un_medida_process+' '
    un_medida_process = re.sub(r' (\d+)([a-z]+) ', r' \1 \2 ', un_medida_process, flags=re.I)
    un_medida_process = un_medida_process.strip()

    items = un_medida_process.split(' ')
    new_items = []
    for item in items:
        if item not in stopwords:
            new_items.append(item)
    items = new_items

    items = preprocessing.lemmatization_document(items)
    new_items = []
    for item in items:
        new_items.append(lemmatize_unit_metric(item, canonical_unit_metric))

    un_medida_process = ' '.join(new_items)

    un_medida_process = strip_multiple_whitespaces(un_medida_process)

    return un_medida_process


def group_dsc_unidade_medida(items_df):
    '''
        Group some of the unit metrics ('dsc_unidade_medida') in the dataframe
        (replacing some of the values):
        'cx' -> 'caixa', ['unitario', 'unid', 'und'] -> 'unidade'.
    '''

    with open('../data/palavras/unit_metric_canonical.json', 'r') as data:
        unit_metric_canonical = json.load(data)

    # for canonic, unit_list in unit_metric_canonical.items():
        # items_df.replace(unit_list, value=canonic, inplace=True)

    canonical_unit_metric = {}

    for canonic, unit_list in unit_metric_canonical.items():
        for unit in unit_list:
            canonical_unit_metric[unit] = canonic

    preprocessing = PreprocessingText(spellcheck=False)
    stopwords = preprocessing.stopwords

    items_df['dsc_unidade_medida'] = items_df['dsc_unidade_medida'].apply(lambda x: process_un_medida(x, stopwords, preprocessing, canonical_unit_metric))
