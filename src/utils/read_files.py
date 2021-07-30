# imports

import pandas as pd
import json
import pickle


def get_items(file_items):

    data = pd.read_csv(file_items, sep=';', low_memory=False)

    if 'seq_dim_licitacao' in data.columns:
        items = data[['nom_item', 'seq_dim_item', 'seq_dim_licitacao', \
                      'vlr_unitario_homologado', 'dsc_unidade_medida', \
                      'ano_exercicio']].values.tolist()
    else:
        items = data[['nom_item', 'seq_item_nota', 'seq_nota_fiscal', \
                      'vlr_unitario_homologado', 'dsc_unidade_medida', \
                      'ano_exercicio']].values.tolist()

    return items


def read_json_file(file):

    with open(file, "r") as JFile:
        data = json.load(JFfile)
    JFile.close()

    return data


def read_pickle_file(file):

    with open(file, "rb") as PFile:
        data = pickle.load(PFile)
    PFile.close()

    return data
