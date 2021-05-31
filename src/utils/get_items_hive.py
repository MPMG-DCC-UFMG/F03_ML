# imports

import pandas as pd
from hive_access import (
    hive_table_to_dataframe
)

def get_items_hive(table_items, recurso_limit=5.0, area=None):
    table_recurso = 'f03_licitacao_vlr_recurso_funcao'
    data_recurso = hive_table_to_dataframe(table_recurso)
    if area != None:
        licitacoes = data_recurso.loc[(data_recurso['nom_funcao'] == area) & \
                                  (data_recurso['proporcao_vlr'] >= recurso_limit)]
    else:
        licitacoes = data_recurso.loc[(data_recurso['proporcao_vlr'] >= recurso_limit)]
    seq_dim_licitacao_list = list(licitacoes['seq_dim_licitacao'])

    data = hive_table_to_dataframe(table_items)
    table_items
    if area != None and recurso_limit >= 0.0:
        data = data.loc[data['seq_dim_licitacao'].isin(seq_dim_licitacao_list)]

    if 'areas' not in data.columns:
        data = data.rename({'nom_funcao':'areas'}, axis=1)

    if 'seq_licitacao_item' not in data.columns:
        data = data.rename({'seq_dim_item':'seq_licitacao_item'}, axis=1)

    items = data[['nom_item', 'seq_licitacao_item', 'seq_dim_licitacao',
                  'vlr_unitario_homologado', 'dsc_unidade_medida', 'areas', \
                   'num_exercicio']].values.tolist()

    return items
