# imports

import collections
from .utils import (
    get_tokens_set
)


def get_tokens_categories():
    '''
        Get the categories of labeled words.
    '''

    set_unit_metrics = get_tokens_set('../dados/palavras/estruturacao/unit_metrics.txt')
    set_colors = get_tokens_set('../dados/palavras/estruturacao/colors.txt')
    set_materials = get_tokens_set('../dados/palavras/estruturacao/materials.txt')
    set_sizes = get_tokens_set('../dados/palavras/estruturacao/sizes.txt')
    set_quantities = get_tokens_set('../dados/palavras/estruturacao/quantities.txt')
    set_qualifiers = get_tokens_set('../dados/palavras/estruturacao/qualifiers.txt')
    set_numbers = get_tokens_set('../dados/palavras/estruturacao/numbers.txt')

    word_category = {}

    for token in set_unit_metrics:
        word_category[token] = "UNIT_METRIC"

    for token in set_colors:
        word_category[token] = "COLOR"

    for token in set_materials:
        word_category[token] = "MATERIAL"

    for token in set_sizes:
        word_category[token] = "SIZE"

    for token in set_quantities:
        word_category[token] = "QUANTITY"

    for token in set_qualifiers:
        word_category[token] = "QUALIFIER"

    for token in set_numbers:
        word_category[token] = "NUMBER"

    return word_category
