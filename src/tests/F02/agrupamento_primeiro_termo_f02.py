import re
import math
import numpy as np
import collections
import json
import gensim
import nltk
import zipfile
import multiprocessing
import pandas
from preprocessing import (
    get_stopwords,
    preprocess_document,
    tokenize_document,
    spellcheck_document,
    lemmatization_document,
    check_first_token
)


def get_ranges(num_items, n_threads):

    if(n_threads == 1):
        return 0, (num_items - 1)

    total_len = num_items
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

    upper[n_threads - 1] = num_items - 1

    return lower, upper


def preprocess_items_thread(items, stopwords_, right_word, canonical_form,
                            it_thread, lower, upper, lemmatize, spellcheck,
                            results_threads):

    items_descriptions = []

    for item in items[lower:upper]:
        # ALTERAR DE ACORDO COM A TABELA DE ENTRADA
        description = item[0]
        licitacao = item[1]
        doc = preprocess_document(description, remove_numbers=False,
                                  stopwords=stopwords_)
        doc = tokenize_document(doc)
        if spellcheck:
            doc = spellcheck_document(doc, right_word)
        if lemmatize:
            doc = lemmatization_document(doc, canonical_form)
        doc = check_first_token(doc, stopwords_)
        items_descriptions.append((doc, licitacao))

    results_threads[it_thread] = items_descriptions


'''
    Realiza o pré-processamento das descrições de objetos de licitação.
'''
def preprocess_items(items, n_threads=10, lemmatize=True, spellcheck=True):

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

    # TALVEZ SEJA NECESSARIO ALTERAR O DIRETORIO
    with open('right_words_nilc.json', "r") as jfile:
        right_word = json.load(jfile)
    jfile.close()

    # It defines the ranges (of the items) the threads will work on:
    thread_ranges = get_ranges(len(items), n_threads)
    print('Read ranges')
    print(thread_ranges)

    manager = multiprocessing.Manager()
    results_threads = manager.dict()
    jobs = []

    for i in range(n_threads):
        p = multiprocessing.Process(target=preprocess_items_thread,
        args = (items, stopwords_, right_word, canonical_form, i, \
                thread_ranges[0][i], thread_ranges[1][i], lemmatize, spellcheck, \
                results_threads))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    items_descriptions = []
    for i in range(n_threads):
        items_descriptions += results_threads[i]

    return items_descriptions


'''
    Recuperar conjunto de palavras.
'''
def get_tokens_set(file):

    tokens = open(file, 'r').readlines()
    tokens = set([token.replace('\n', '') for token in tokens])

    return tokens


'''
    Verifica o termo passado como parâmetro é um número.
'''
def is_number(token):
    return True if token.isnumeric() else False


'''
    Estrutura a descrição de um item em 7 categorias: 'números', 'unidades de medida',
    'cores', 'materiais', 'tamanho', 'quantidade' e 'palavras'.

    description (str): descrição do objeto.
    licitacao (int): id da licitação a qual objeto faz parte.
    set_unit_metrics (set): conjunto de unidades de medida.
    set_colors (set): conjunto de cores.
    set_materials (set): conjunto de materiais.
    set_sizes (set): conjunto de tamanhos.
    set_quantities (set): conjunto de quantidades.
    set_qualifiers (set): conjunto de qualificadores.
    set_numbers (set): conjunto de números.
'''
def extract_entities(description, licitacao, set_unit_metrics, set_colors,
                    set_materials, set_sizes, set_quantities, set_qualifiers,
                    set_numbers):

    numbers = []
    unit_metrics = []
    colors = []
    materials = []
    sizes = []
    quantities = []
    words = []

    for token in description:
        if token in set_qualifiers:
            continue

        if is_number(token) or token in set_numbers:
            numbers.append(token)
        elif token in set_unit_metrics:
            unit_metrics.append(token)
        elif token in set_colors:
            colors.append(token)
        elif token in set_materials:
            materials.append(token)
        elif token in set_sizes:
            sizes.append(token)
        elif token in set_quantities:
            quantities.append(token)
        elif len(token) == 1:
            continue
        else:
            words.append(token)

    return words


'''
    Seleciona o primeiro termo das descrições de objetos de licitação.

    Entrada -> items_descriptions (list of tuples): descrição de objetos e suas
    respectivas licitações.
'''
def first_token_grouping(items_descriptions):

    # TALVEZ SEJA NECESSARIO ALTERAR O DIRETORIO
    set_unit_metrics = get_tokens_set('./estruturacao/unit_metrics.txt')
    set_colors = get_tokens_set('./estruturacao/colors.txt')
    set_materials = get_tokens_set('./estruturacao/materials.txt')
    set_sizes = get_tokens_set('./estruturacao/sizes.txt')
    set_quantities = get_tokens_set('./estruturacao/quantities.txt')
    set_qualifiers = get_tokens_set('./estruturacao/qualifiers.txt')
    set_numbers = get_tokens_set('./estruturacao/numbers.txt')

    results = []

    # ALTERAR DE ACORDO COM A TABELA DE ENTRADA
    for description, licitacao in items_descriptions:
        words = extract_entities(description, licitacao, set_unit_metrics,
                                 set_colors, set_materials, set_sizes,
                                 set_quantities, set_qualifiers, set_numbers)
        if len(words) == 0:
            group = ""
        else:
            group = words[0]

        results.append((licitacao, group))

    return results


'''
    Gerar tabela de itens, sendo que cada item é representado pelo primeiro termo
    de sua descrição.

    Entrada -> input_df (DataFrame): tabela de itens. Cada linha da tabela é um
    item. A tabela deve possuir as colunas "nom_item" (descrição do item) e
    "seq_dim_licitacao" (licitação a qual o item faz parte).

    Saída -> dataframe (Dataframe): tabela de itens. Cada linha da tabela é um
    item. A tabela possui as colunas "seq_dim_licitacao" (licitação a qual o item
    faz parte) e "item" primeiro termo da descrição do item (de acordo com a
    abordagem utilizada no projeto F03).
'''
def group_items(input_df):

    items = input_df[['nom_item', 'seq_dim_licitacao']].values.tolist()
    items_descriptions = preprocess_items(items)
    results = first_token_grouping(items_descriptions)

    dataframe = pd.DataFrame(results, columns=['seq_dim_licitacao', 'item'])

    return dataframe
