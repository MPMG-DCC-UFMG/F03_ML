# imports

from .item import Item
from .utils import (
    get_tokens_set,
    count_tokens)
from spellcheck.spellcheckeropt import SpellcheckerOpt
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
        self.set_unit_metrics = get_tokens_set('../dados/palavras/unit_metrics.txt')
        self.set_colors = get_tokens_set('../dados/palavras/colors.txt')
        self.set_materials = get_tokens_set('../dados/palavras/materials.txt')
        self.set_sizes = get_tokens_set('../dados/palavras/sizes.txt')
        self.set_quantities = get_tokens_set('../dados/palavras/quantities.txt')
        self.set_qualifiers = get_tokens_set('../dados/palavras/qualifiers.txt')
        self.set_numbers = get_tokens_set('../dados/palavras/numbers.txt')
        self.size = 0
        self.unique_words = None
        self.word_id = None
        self.id_word = None


    def load_items_from_list(self, items_descriptions):

        for description, item_id, licitacao, price, dsc_unidade, original, funcao, ano in items_descriptions:
            if len(description) == 0:
                continue
            item = Item()
            item.extract_entities(description, item_id, licitacao, price, dsc_unidade,
                                  original, funcao, ano, self.set_unit_metrics,
                                  self.set_colors, self.set_materials,
                                  self.set_sizes, self.set_quantities,
                                  self.set_qualifiers, self.set_numbers)
            self.items_list.append(item)

        self.size = len(self.items_list)
        self.set_unique_words()
        self.set_words_ids()


    def load_items_from_file(self, file, just_words=False, original=False):

        with zipfile.ZipFile(file, 'r') as zipped:
            with zipped.open(file[9:-4], 'r') as data:
                lines = data.readlines()
                for line in lines:
                    line = line.decode('utf-8')
                    line = line.strip("\n")
                    item_json = ast.literal_eval(line)
                    item = Item(item_json, original)
                    if just_words:
                        self.items_list.append(item.words)
                    else:
                        self.items_list.append(item)
            data.close()
        self.size = len(self.items_list)
        self.set_unique_words(just_words)
        self.set_words_ids()


    def save_items(self, file):

        with open(file, 'w') as json_file:
            for item in self.items_list:
                item_dict = item.get_item_dict()
                json_file.write(str(item_dict) + '\n')
        json_file.close()


    def set_unique_words(self, just_words=False):

        words = []
        for item in self.items_list:
            if just_words:
                for word in item:
                    words.append(word)
            else:
                if item.original_preprocessed != None:
                    for word in item.original_preprocessed:
                        words.append(word)
                else:
                    for word in item.words:
                        words.append(word)

        self.unique_words = list(set(words))


    def set_words_ids(self):

        self.word_id = {}
        self.id_word = {}
        id = 0

        for word in self.unique_words:
            if word not in self.word_id:
                self.word_id[word] = id
                self.id_word[id] = word
                id += 1


    def get_item(self, item):
        return self.items_list[item]


    def get_items_words(self, items=None):
        if items == None:
            return [item.words for item in self.items_list]
        else:
            return [item.words for item in items]


    def get_unigram_groups(self):

        groups = collections.defaultdict(list)

        documents = [item.words for item in self.items_list]
        token_count = count_tokens(documents)
        id = 0

        for item in self.items_list:
            words = item.words
            maxi = 0
            for token in words:
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


    def get_first_token_groups(self, original_prep=False, just_words=False):

        groups = collections.defaultdict(list)
        id = 0

        for item in self.items_list:
            if just_words:
                words = item
            elif original_prep:
                words = item.original_preprocessed
            else:
                words = item.words

            if len(words) != 0:
                groups[words[0]].append(id)
            id += 1

        return groups


    def get_firsttwo_tokens_groups(self):

        groups = collections.defaultdict(list)
        id = 0

        for item in self.items_list:
            words = item.words
            if len(words) == 1:
                groups[words[0]].append(id)
            elif len(words) > 1:
                groups[(words[0], words[1])].append(id)
            id += 1

        return groups


    def regroup_first_token_groups(self, groups, distance=2, verbose=False):

        words = get_tokens_set('../dados/palavras/words_cbras_preprocess.txt')

        groups_names = [group for group, items in groups.items()]

        spellchecker = SpellcheckerOpt()
        spellchecker.load_words(groups_names)

        words_checked = 0
        similar_words = {}

        for word in groups_names:
            words_list = spellchecker.search(word, distance)
            words_list.sort(key=lambda x:(x[1], x[0]))
            similar_words[word] = words_list
            words_checked += 1
            if verbose and words_checked%1000 == 0:
                print('%d words checked' % (words_checked))

        for group, items in groups.items():
            if group not in words:
                new_group = None
                for word, dist in similar_words[group]:
                    if word in words:
                        new_group = word
                        break
                if new_group != None:
                    groups[new_group] += items
                    groups[group] = []

        new_groups = {}
        for group, items in groups.items():
            if len(items) > 0:
                new_groups[group] = items

        return new_groups


    def get_groups_size(self, groups):

        groups_size = []

        for group, items in groups.items():
            groups_size.append(len(items))

        return groups_size


    def get_group_items(self, group):

        items = []
        for item_id in group:
            items.append(self.items_list[item_id])

        return items


    def get_similar_words(self, distance=2, verbose=False):

        spellchecker = SpellcheckerOpt()
        spellchecker.load_words(self.unique_words)

        words_checked = 0
        similar_words = {}

        for word in self.unique_words:
            words_list = spellchecker.search(word, distance)
            words_list.sort(key=lambda x:(x[1], x[0]))
            similar_words[word] = words_list
            words_checked += 1
            if verbose and words_checked%1000 == 0:
                print('%d words checked' % (words_checked))

        return similar_words


    def save_similar_words(self, similar_words, file):

        with open(file, "w") as words_file:
            for word, words_list in similar_words.items():
                words_file.write(word + ' ' + str(words_list) + '\n')
        words_file.close()


    def load_similar_words(self, file):

        similar_words = {}

        with zipfile.ZipFile(file, 'r') as zipped:
            with zipped.open('similar_words', 'r') as data:
                lines = data.readlines()
                for line in lines:
                    line = line.decode("utf-8")
                    similar_list = line.split(' ', maxsplit=1)
                    word = similar_list[0]
                    word_id = self.word_id[word]
                    words_list = []
                    for t in re.findall("\((.*?)\)", similar_list[1]):
                        t = t.split(',')
                        w = str(t[0].strip("'"))
                        d = int(t[1].strip("'"))
                        words_list.append((w, d))
                    similar_words[word_id] = words_list
            data.close()

        return similar_words


    def get_item_words_list(self, item, similar_words):

        words_list = []

        for word in item.words:
            word_id = self.word_id[word]
            for w in similar_words[word_id]:
                words_list.append(w[0])

        return words_list


    def get_items_group_words_list(self, group, similar_words):

        item_words_list = {}

        for item_id in group:
            words_list = self.get_item_words_list(self.items_list[item_id],
                                                  similar_words)
            words_id = [self.word_id[word] for word in words_list]
            item_words_list[item_id] = set(words_id)

        return item_words_list


    def get_items_distance(self, group, similar_words, rank_size=None, verbose=False):

        items_ids = list(group)
        items_words_list = self.get_items_group_words_list(group, similar_words)
        items_distance = {}
        i = 0

        for itemA in items_ids:
            items_list = []
            itemA_set = items_words_list[itemA]

            if len(items_ids) > 1000:
                items_ids_sample = random.sample(items_ids, k=1000)
            else:
                items_ids_sample = items_ids

            for itemB in items_ids_sample:
                if itemA != itemB:
                    items_in_common = itemA_set.intersection(items_words_list[itemB])
                    items_list.append((itemB, len(items_in_common)))
            items_list.sort(key=lambda x:(-x[1]))
            if rank_size != None:
                items_distance[itemA] = items_list[:rank_size]
            else:
                items_distance[itemA] = items_list

            i += 1
            if verbose and i%1000 == 0:
                print("%d items checked" % i)

        return items_distance


    def get_similar_items(self, item_id, items_distance, rank_size=10):

        items = items_distance[item_id][:rank_size]
        similar_items = []

        for id, distance in items:
            similar_item = self.items_list[id]
            similar_items.append(similar_item)

        return similar_items


    def get_item_reference_price(self, item_id, items_distance, rank_size=10):

        similar_items = self.get_similar_items(item_id, items_distance, rank_size)

        prices = []
        size = 0
        for item_similar in similar_items:
            prices.append(item_similar.price)
            size += 1

        mean = np.mean(prices)
        first_quartile = np.percentile(prices, 25, interpolation='midpoint')
        median = np.median(prices)
        third_quartile = np.percentile(prices, 75, interpolation='midpoint')
        std = np.std(prices)
        var = np.var(prices)
        maxi = np.max(prices)
        mini = np.min(prices)

        return (mean, first_quartile, median, third_quartile, std, var, maxi, mini)


    def get_items_referece_prices(self, grouping_type='first_token',rank_size=10,
                                  verbose=False):

        reference_price = {}
        similar_words = self.load_similar_words('../dados/similar_words.zip')

        if grouping_type == 'first_token':
            groups = self.get_first_token_groups()
            groups = self.regroup_first_token_groups(groups)
        elif grouping_type == 'first_two_tokens':
            groups = self.get_firsttwo_tokens_groups()

        groups_checked = 0
        for group_name, items_list in groups.items():
            print(group_name, ':', len(items_list))
            items_distance = self.get_items_distance(items_list, similar_words,
                                                     rank_size=rank_size)
            for item_id in items_list:
                item =  self.get_item(item_id)
                if len(items_distance[item_id]) != 0:
                    (mean, first_quartile, median, third_quartile, std, var, max, \
                    min) = self.get_item_reference_price(item_id, items_distance)
                else:
                    (mean, first_quartile, median, third_quartile, std, var, max, \
                    min) = (-1, -1, -1, -1, -1, -1, -1, -1)
                reference_price[item_id] = [group_name, item.price, mean,
                                        first_quartile, median, third_quartile,
                                        std, var, max, min]
            groups_checked += 1
            if verbose and groups_checked%100 == 0:
                print('%d groups checked' % (groups_checked))

        return reference_price


    def save_reference_prices(self, reference_price, file):

        tuples = []
        for item_id, stats in reference_price.items():
            item = self.get_item(item_id)
            item_information = [item_id, str(item.get_item_dict())] + stats
            tuples.append(tuple(item_information))

        data = pd.DataFrame(tuples, columns=['id', 'item', 'grupo', 'preco', \
                            'avg', 'quartil_1', 'median', 'quartil_3', \
                            'std', 'var', 'max', 'min'])
        data.to_csv(file, sep=';', index=False)


    def load_reference_prices(self, file):

        items_reference_prices = {}

        with zipfile.ZipFile('../dados/items_reference_prices.zip', 'r') as zipped:
            with zipped.open('items_reference_prices.csv', 'r') as data:
                data.readline()
                lines = data.readlines()
                for line in lines:
                    line = line.decode("utf-8")
                    line = line.strip("\n")
                    line = line.split(";")
                    item_id = int(line[0])
                    items_reference_prices[item_id] = line[2:]
            data.close()

        return items_reference_prices
