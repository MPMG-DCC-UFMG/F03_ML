# imports

import pandas as pd


def get_items(file_items, area=None):

    data = pd.read_csv(file_items, sep=';', low_memory=False)

    if 'areas' not in data.columns:
        data = data.rename({'nom_funcao':'areas'}, axis=1)

    if 'seq_licitacao_item' not in data.columns:
        data = data.rename({'seq_dim_item':'seq_licitacao_item'}, axis=1)

    if area != None:
        data = data[data.areas == area]

    if 'areas' not in data.columns:
        items = data[['nom_item', 'seq_licitacao_item', 'seq_dim_licitacao',
                     'vlr_unitario_homologado', 'dsc_unidade_medida',
                     'ano_exercicio', 'mes_exercicio', 'data_cotacao',
                     'nome_municipio', 'nome_orgao']].values.tolist()
    else:
        items = data[['nom_item', 'seq_licitacao_item', 'seq_dim_licitacao',
                  'vlr_unitario_homologado', 'dsc_unidade_medida', 'areas', \
                   'ano_exercicio', 'mes_exercicio', 'data_cotacao',
                   'nome_municipio', 'nome_orgao']].values.tolist()

    return items
