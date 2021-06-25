# imports

import pandas as pd


def get_items(file_items):

    data = pd.read_csv(file_items, sep=';', low_memory=False)

    if 'seq_dim_licitcao' in data.columns:
        items = data[['nom_item', 'seq_dim_item', 'seq_dim_licitacao', \
                      'vlr_unitario_homologado', 'dsc_unidade_medida', \
                      'ano_exercicio']].values.tolist()
    else:
        items = data[['nom_item', 'seq_item_nota', 'seq_nota_fiscal', \
                      'vlr_unitario_homologado', 'dsc_unidade_medida', \
                      'ano_exercicio']].values.tolist()

    return items
