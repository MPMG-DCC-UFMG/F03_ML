from nlp.utils import (
    isfloat,
    get_scientific_notation
)


class Item:

    def __init__(self, item=None, original=False):
        self.licitacao_item = None
        self.words = []
        self.unit_metrics = []
        self.numbers = []
        self.colors = []
        self.materials = []
        self.sizes = []
        self.quantities = []
        self.price = None
        self.dsc_unidade = None
        self.licitacao = None
        self.original = None
        self.original_preprocessed = None
        self.ano = None
        if item != None:
            self.load_item(item, original)


    def is_number(self, token):
        '''
            Check if a token is a number. Return True if the token is a number and
            False otherwise.
        '''

        return True if token.isnumeric() else False


    def extract_entities(self, description, licitacao_item, licitacao, price,
                         dsc_unidade, original, ano, set_unit_metrics,
                         set_colors, set_materials, set_sizes, set_quantities,
                         set_qualifiers, set_numbers, set_ambiguous, stopwords):
        '''
            Structure item descriptions.

            description (str): item description.
            licitacao_item (int): item id in the table (sequence of numbers). This
                                  field is not used in the algorithm.
            licitacao (int): "Licitação" id.
            price (float): item price.
            dsc_unidade (str): item unit of measure.
            original (str): item original description.
            ano (int): year the item was traded.
        '''

        self.licitacao_item = licitacao_item
        self.licitacao = licitacao
        self.price = price
        self.dsc_unidade = dsc_unidade
        self.original = original
        self.original_preprocessed = description
        self.ano = ano

        for token in description:
            if token in set_qualifiers:
                continue

            if self.is_number(token) or isfloat(token):
                self.numbers.append(get_scientific_notation(token))
            elif token in set_numbers:
                self.numbers.append(token)
            elif token in set_unit_metrics:
                self.unit_metrics.append(token)
            elif token in set_colors:
                self.colors.append(token)
            elif token in set_materials:
                self.materials.append(token)
            elif token in set_sizes:
                self.sizes.append(token)
            elif token in set_quantities:
                self.quantities.append(token)
            elif len(token) == 1:
                continue
            else:
                self.words.append(token)

        if len(self.words) > 0 and self.words[0] in stopwords:
            self.words = [description[0]] + self.words
        else:
            if len(self.quantities) > 0 and self.quantities[0] in set_ambiguous:
                self.words.append(self.quantities[0])
            elif len(self.materials) > 0 and self.materials[0] in set_ambiguous:
                self.words.append(self.materials[0])
            elif len(self.colors) > 0 and self.colors[0] in set_ambiguous:
                self.words.append(self.colors[0])
            elif len(self.sizes) > 0 and self.sizes[0] in set_ambiguous:
                self.words.append(self.sizes[0])


    def load_item(self, item, original):
        '''
            It loads an item from a dictionary.

            item (dict): items informations.
            original (bool): if the original description should be saved.
        '''

        self.words = item['palavras']
        self.unit_metrics = item['unidades_medida']
        self.numbers = item['numeros']
        self.colors = item['cores']
        self.materials = item['materiais']
        self.sizes = item['tamanho']
        self.quantities = item['quantidade']
        self.price = item['preco']
        self.dsc_unidade = item['dsc_unidade_medida']

        if original:
            self.original = item['original']
        else:
            self.original = None

        self.licitacao = item['licitacao']
        self.original_preprocessed = item['original_prep']
        self.ano = item['ano']
        self.licitacao_item = item['licitacao_item']


    def print_item(self):

        item_dict = self.get_item_dict()
        print(item_dict)


    def get_item_dict(self):

        item_dict = {
            'palavras' : self.words,
            'unidades_medida' : self.unit_metrics,
            'numeros' : self.numbers,
            'cores' : self.colors,
            'materiais' : self.materials,
            'tamanho' : self.sizes,
            'quantidade' : self.quantities,
            'preco' : self.price,
            'dsc_unidade_medida' : self.dsc_unidade,
            'original' : self.original,
            'licitacao' : self.licitacao,
            'original_prep' : self.original_preprocessed,
            'ano' : self.ano,
            'licitacao_item' : self.licitacao_item
        }

        return item_dict
