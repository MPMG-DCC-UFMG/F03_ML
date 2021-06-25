# imports

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from .item import Item
from .utils import (
    get_tokens_set,
    count_tokens)
from utils.hive_access import (
    dataframe_to_hive_table,
    hive_table_to_dataframe
)
from spellcheck.spellcheckeropt import SpellcheckerOpt
from nlp.preprocess_units import group_dsc_unidade_medida
import ast
import zipfile
import collections
import numpy as np
import pandas as pd
import re
import random
import copy


class ItemList:

    def __init__(self):
        self.items_list = []
        self.items_df = None
        self.set_unit_metrics = get_tokens_set('../data/palavras/estruturacao/unit_metrics.txt')
        self.set_colors = get_tokens_set('../data/palavras/estruturacao/colors.txt')
        self.set_materials = get_tokens_set('../data/palavras/estruturacao/materials.txt')
        self.set_sizes = get_tokens_set('../data/palavras/estruturacao/sizes.txt')
        self.set_quantities = get_tokens_set('../data/palavras/estruturacao/quantities.txt')
        self.set_qualifiers = get_tokens_set('../data/palavras/estruturacao/qualifiers.txt')
        self.set_numbers = get_tokens_set('../data/palavras/estruturacao/numbers.txt')
        self.set_ambiguous = get_tokens_set('../data/palavras/estruturacao/ambiguous.txt')
        self.size = 0
        self.unique_words = None
        self.word_id = None
        self.id_word = None


    def structure_items(self, items_descriptions):
        '''
            Structure item descriptions.

            items_descriptions (list): list of items. Each item has the following
            informations: description, seq_licitacao_item, licitacao, price,
            dcs_unidade_medida, original, ano.
        '''

        for description, licitacao_item, licitacao, price, dsc_unidade, original, \
            ano in items_descriptions:
            if len(description) == 0:
                continue
            item = Item()
            item.extract_entities(description, licitacao_item, licitacao, price,
                                  dsc_unidade, original, ano, self.set_unit_metrics,
                                  self.set_colors, self.set_materials,
                                  self.set_sizes, self.set_quantities,
                                  self.set_qualifiers, self.set_numbers,
                                  self.set_ambiguous)
            self.items_list.append(item)

        self.size = len(self.items_list)
        self.set_unique_words()
        self.set_words_ids()


    def to_dataframe(self, items=None):
        '''
            Build a dataframe with the set of items.

            items (list): list of Item objects.
        '''

        if items == None:
            items = self.items_list

        tuples = []

        for i, item in enumerate(items):
            words = item.words
            unit_metrics = item.unit_metrics
            numbers = item.numbers
            colors = item.colors
            materials = item.materials
            sizes = item.sizes
            quantities = item.quantities
            price = item.price
            dsc_unidade = item.dsc_unidade
            original = item.original
            licitacao = item.licitacao
            licitacao_item = item.licitacao_item
            original_preprocessed = item.original_preprocessed
            ano = item.ano

            tuples.append(tuple([words, unit_metrics, numbers, colors, materials,
                        sizes, quantities, price, dsc_unidade, original,
                        original_preprocessed, ano, licitacao, licitacao_item]))

        columns = ['palavras', 'unidades_medida', 'numeros', 'cores', 'materiais', \
                    'tamanho', 'quantidade', 'preco', 'dsc_unidade_medida', 'original', \
                    'original_prep', 'ano', 'licitacao', 'licitacao_item']
        dataframe = pd.DataFrame(tuples, columns=columns)
        dataframe["item_id"] = list(range(len(dataframe)))

        return dataframe


    def load_items_from_file(self, file, original=False, type='dataframe'):
        '''
            It reads a file that contains the Items.

            file (str): 'csv.zip' or 'zip' file.
            origial (bool): if the original descriptions should be saved.
            type (str): it the items are stores in a dataframe or in a json.
        '''

        if type == 'dict':
            with zipfile.ZipFile(file, 'r') as zipped:
                with zipped.open(file[:-4], 'r') as data:
                    lines = data.readlines()
                    for line in lines:
                        line = line.decode('utf-8')
                        line = line.strip("\n")
                        item_json = ast.literal_eval(line)
                        item = Item(item_json, original)
                        self.items_list.append(item)
                data.close()
            self.size = len(self.items_list)
        elif type == 'dataframe':
            self.items_df = pd.read_csv(file, sep=';',
                                        low_memory=False)
            self.size = len(self.items_df)
        self.set_unique_words()
        self.set_words_ids()


    def load_items_from_hive_table(self, table, password):
        '''
            It reads a Hive table that contains the Items.

            table (str): Hive table.
            password (str): connection password.
        '''

        self.items_df = hive_table_to_dataframe(table, password)
        self.size = len(self.items_df)
        self.set_unique_words()
        self.set_words_ids()


    def save_items_in_dict(self, file, items=None):
        '''
            It saves the items as dictionaries.
        '''

        if items == None:
            items = self.items_list

        with open('../data/' + file[:-4], 'w') as json_file:
            for item in self.items_list:
                item_dict = item.get_item_dict()
                json_file.write(str(item_dict) + '\n')
        json_file.close()

        with zipfile.ZipFile('../data/' + file, 'w') as zip:
            zip.write(os.path.join('../data/', file[:-4]), arcname=file[:-4])

        os.remove('../data/' + file[:-4])


    def save_items_in_hive_table(self, table, dataframe, version, password):
        '''
            It saves the items as a dataframe in a Hive Table.
        '''
        dataframe_to_hive_table(dataframe, table, version, password)


    def save_items_in_dataframe(self, file, dataframe):
        '''
            It saves the items as a dataframe.
        '''
        dataframe.to_csv(file, sep=';', index=False, compression='zip')


    def set_unique_words(self):
        '''
            It gets the unique words from item descriptions.
        '''

        words = []
        if self.items_df is None:
            for item in self.items_list:
                if item.original_preprocessed != None:
                    for word in item.original_preprocessed:
                        words.append(word)
                else:
                    for word in item.words:
                        words.append(word)
        else:
            items = list(self.items_df['original_prep'])
            for item in items:
                for word in eval(item):
                    words.append(word)

        self.unique_words = list(set(words))


    def set_words_ids(self):
        '''
            It maps each unique word from the dataset to an id.
        '''

        self.word_id = {}
        self.id_word = {}
        id = 0

        for word in self.unique_words:
            if word not in self.word_id:
                self.word_id[word] = id
                self.id_word[id] = word
                id += 1


    def get_item(self, item):
        '''
            It gets an item from the list of items.

            item (int): item id (position in the list/dataframe).
        '''

        if self.items_df is None:
            return self.items_list[item].get_item_dict()
        else:
            return self.items_df.iloc[item].to_dict()


    def get_items_words(self, items=None):
        '''
            It gets the words of an item.

            items (list): list of item objects.
        '''

        if self.items_df is None:
            if items == None:
                return [item.words for item in self.items_list]
            else:
                return [item.words for item in items]
        else:
            return [eval(d) for d in list(self.items_df['palavras'])]


    def products_services_split(self, products_table, services_table, version='',
                                save_in_hive=False, password=''):
        '''
            It splits the set of items into products and services subsets.

            products_table (str): table where the products should be saved.
            services_table (str): table where the services should be saved.
            version (str): execution version.
        '''

        services = []
        products = []

        for item in self.items_list:
            item_dict = item.get_item_dict()
            flag = False
            for tok in item_dict['palavras']:
                if 'servico' in tok or 'prestacao' in tok or 'servicos' in tok or 'prestacoes' in tok:
                    flag = True
                    break
            if 'servico' in item_dict['dsc_unidade_medida'] or 'prestacao' in item_dict['dsc_unidade_medida'] or \
               'servicos' in item_dict['dsc_unidade_medida'] or 'prestacoes' in item_dict['dsc_unidade_medida']:
                flag = True
            if flag:
                services.append(item)
            else:
                products.append(item)

        services_df = self.to_dataframe(items=services)
        products_df = self.to_dataframe(items=products)

        services_df["item_id"] = list(range(len(services_df)))
        products_df["item_id"] = list(range(len(products_df)))

        group_dsc_unidade_medida(services_df)
        group_dsc_unidade_medida(products_df)

        self.save_items_in_dataframe(services_table, services_df)
        self.save_items_in_dataframe(products_table, products_df)

        if save_in_hive:
            self.save_items_in_hive_table(services_table, services_df, version,
                                          password)
            self.save_items_in_hive_table(products_table, products_df, version,
                                          password)

        return products, services


    def train_test_split(self, train_table, test_table, version='', train_size=0.8,
                         save_in_hive=False, password=''):
        '''
            It splits the set of items into random train and test subsets

            train_table (str): table where the train subset should be saved.
            test_table (str): table where the test subset should be saved.
            version (str): execution version.
            train_size (float): should be between 0.0 and 1.0 and represent the
                                proportion of the dataset (using the field "licitacao")
                                to include in the train split.
        '''

        random.seed(999)

        licitacoes_set = set()
        for item in self.items_list:
            item_dict = item.get_item_dict()
            licitacoes_set.add(item_dict['licitacao'])

        train_set = set(random.sample(list(licitacoes_set), int(train_size*len(licitacoes_set))))
        test_set = licitacoes_set - train_set

        train = []
        test = []

        for item in self.items_list:
            item_dict = item.get_item_dict()
            if item_dict['licitacao'] in train_set:
                train.append(item)
            else:
                test.append(item)

        train_df = self.to_dataframe(items=train)
        test_df = self.to_dataframe(items=test)

        train_df["item_id"] = list(range(len(train_df)))
        test_df["item_id"] = list(range(len(test_df)))

        group_dsc_unidade_medida(train_df)
        group_dsc_unidade_medida(test_df)

        self.save_items_in_dataframe(train_table, train_df)
        self.save_items_in_dataframe(test_table, test_df)

        if save_in_hive:
            self.save_items_in_hive_table(train_table, train_df, version, password)
            self.save_items_in_hive_table(test_table, test_df, version, password)

        return train, test


    def get_unigram_groups(self):
        '''
            Group  items using the most frequent word in each description.
        '''

        groups = collections.defaultdict(list)

        if self.items_df is None:
            documents = [item.words for item in self.items_list]
        else:
            documents = list(self.items_df['palavras'])
            documents = [eval(d) for d in documents]

        token_count = count_tokens(documents)
        id = 0

        for item in documents:
            maxi = 0
            for token in item:
                if token_count[token] > maxi:
                    maxi = token_count[token]
            tokens_sorted = copy.deepcopy(words)
            tokens_sorted.sort()
            unigram = None
            for token in tokens_sorted:
                if token_count[token] == maxi:
                    unigram = token
                    break

            if unigram != None:
                groups[unigram].append(id)
                id += 1

        return groups


    def get_first_token_groups(self, original_prep=False):
        '''
            Group items using the first token (word) in each description.
        '''

        groups = collections.defaultdict(list)

        if self.items_df is None:
            if original_prep:
                documents = [item.original_preprocessed for item in self.items_list]
            else:
                documents = [item.words for item in self.items_list]
        else:
            if original_prep:
                documents = list(self.items_df['original_prep'])
            else:
                documents = list(self.items_df['palavras'])
            documents = [eval(d) for d in documents]

        id = 0

        for item in documents:
            words = item
            if len(words) != 0:
                groups[words[0]].append(id)
            id += 1

        return groups


    def get_firsttwo_tokens_groups(self):
        '''
            Group items using the first and second tokens (word) in each description.
        '''

        groups = collections.defaultdict(list)

        if self.items_df is None:
            documents = [item.words for item in self.items_list]
        else:
            documents = list(self.items_df['palavras'])
            documents = [eval(d) for d in documents]

        id = 0

        for item in documents:
            words = item
            if len(words) == 1:
                groups[words[0]].append(id)
            elif len(words) > 1:
                groups[(words[0], words[1])].append(id)
            id += 1

        return groups


    def get_groups_size(self, groups):
        '''
            It gets the size for each group in the dictionary.

            groups (dict): dictionary of groups.
        '''

        groups_size = []

        for group, items in groups.items():
            groups_size.append(len(items))

        return groups_size


    def get_group_items(self, group):
        '''
            It gets the items from a group.

            group (list): list of items.
        '''

        items = []
        for item_id in group:
            if self.items_df is None:
                items.append(self.items_list[item_id].get_item_dict())
            else:
                items.append(self.items_df.iloc[item_id].to_dict())

        return items
