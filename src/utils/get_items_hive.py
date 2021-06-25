# imports

import pandas as pd
from .hive_access import (
    hive_table_to_dataframe
)

def get_items_hive(table_items):

    data = hive_table_to_dataframe(table_items)

    if 'seq_dim_licitacao' in data.columns:
        items = data[['nom_item', 'seq_dim_item', 'seq_dim_licitacao', \
                      'vlr_unitario_homologado', 'dsc_unidade_medida', \
                      'ano_exercicio']].values.tolist()
    else:
        items = data[['nom_item', 'seq_item_nota', 'seq_nota_fiscal', \
                      'vlr_unitario_homologado', 'dsc_unidade_medida', \
                      'ano_exercicio']].values.tolist()

    return items
