# imports

import pandas as pd


def get_items(file_items, area=None):

    data = pd.read_csv(file_items, sep=';', low_memory=False)

    if 'seq_licitacao_item' not in data.columns:
        data = data.rename({'seq_dim_item':'seq_licitacao_item'}, axis=1)

    items = data[['nom_item', 'seq_licitacao_item', 'seq_dim_licitacao', \
                  'vlr_unitario_homologado', 'dsc_unidade_medida', \
                  'ano_exercicio']].values.tolist()

    return items
