# imports

import collections


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


def group_dsc_unidade_medida(items_df):
    '''
        Group some of the unit metrics ('dsc_unidade_medida') in the dataframe
        (replacing some of the values):
        'cx' -> 'caixa', ['unitario', 'unid', 'und'] -> 'unidade'.
    '''

    unidade = ['un', 'unid', 'und', 'uni', 'unidad', 'unid.', 'unitario', 'unidade.',
                'un.', 'uni.', 'und.', 'unida', 'unita', 'und,', 'un1', 'unid..', 'um',
                '0000000000000000000000000000000000000000000unidade', 'u']
    unidades = ['unidade(s)', 'unidade_s', 'unidade _s', 'unids']
    hora = ['hr', 'hrs', 'horas']
    jogo = ['jg', 'jgs']
    peca = ['pe', 'pc', 'pc.']
    saco = ['sc', 'sacos']
    caixa = ['cx', 'cx.', '000000000000000000000000000000000000000000000caixa', 'cxa']
    comprimido = ['cp', 'comp', 'comprimido 1 unidade', 'comp.', 'cmp', 'compr',
                  'comprimid', 'com', 'cpr', 'capsula', 'cap',
                  '0000000000000000000000000000000000000000comprimido']
    pacote = ['pct', 'pt', 'pcte', 'pacote.', 'pct.', 'pacot']
    kg = ['kilo', 'quilo', 'quilograma', 'kilograma', 'kilos', 'quilos', 'quilogramas',
          'kilogramas', 'quilo.', 'quilograma_s']
    frasco = ['fr', 'frasco.', 'fras', 'frasco 1 unidade', 'frasco/ampola', 'frasc']
    litro = ['litros', 'ltr', 'lit', 'litro.', 'li', 'l', 'lt', 'lts']

    items_df.replace(unidade, value='unidade', inplace=True)
    items_df.replace(unidades, value='unidades', inplace=True)
    items_df.replace(hora, value='hora', inplace=True)
    items_df.replace(jogo, value='jogo', inplace=True)
    items_df.replace(peca, value='peca', inplace=True)
    items_df.replace(saco, value='saco', inplace=True)
    items_df.replace(caixa, value='caixa', inplace=True)
    items_df.replace(comprimido, value='comprimido', inplace=True)
    items_df.replace(pacote, value='pacote', inplace=True)
    items_df.replace(kg, value='kg', inplace=True)
    items_df.replace(frasco, value='frasco', inplace=True)
    items_df.replace(litro, value='litro', inplace=True)
